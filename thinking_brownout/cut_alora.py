#!/usr/bin/env python3
"""Train and evaluate the cut adapter: an Activated LoRA (aLoRA) that turns on at the [cut] marker.

  python cut_alora.py selftest                      # tiny random Llama, CPU, seconds: checks the PEFT behaviour we rely on
  python cut_alora.py selftest --model Qwen/Qwen3.8-27B   # same checks on the REAL architecture (config only,
                                                          # random weights, 4 layers); lists the Linear layers
  # SFT on the decisive targets (make_sft.py); validation = 10% of the PROBLEMS, best epoch kept
  python cut_alora.py train sft/Qwen__Qwen3.8-27B.decisive.sft.jsonl --model Qwen/Qwen3.8-27B --out adapters/cut-r16
  python cut_alora.py check traces/Qwen__Qwen3.8-27B.jsonl --model Qwen/Qwen3.8-27B --adapter adapters/cut-r16
  python cut_alora.py eval  runs/Qwen__Qwen3.8-27B.bench.jsonl traces/Qwen__Qwen3.8-27B.jsonl \\
      --model Qwen/Qwen3.8-27B --adapter adapters/cut-r16 --name alora-sft   # appends rows for cut_report.py
  # GRPO on NEW problems (cut states from cut_bench.py on the .grpo traces), anchored on the SFT targets
  python cut_alora.py grpo  sft/Qwen__Qwen3.8-27B.decisive.sft.jsonl traces/Qwen__Qwen3.8-27B.grpo.jsonl \\
      --runs runs/Qwen__Qwen3.8-27B.grpo.states.jsonl --model Qwen/Qwen3.8-27B \\
      --adapter adapters/cut-r16 --out adapters/cut-grpo

Token layout of one training example (weights change only from the marker on):

  [chat prompt][<think>][thinking prefix, written by the base model] | [cut][target thinking]</think>[answer][eos]
   \\________________ base weights, no gradient ___________________/   \\________ adapter on; loss on the target part ________/

The target was written by the base model under a longer data-generation instruction (decisive_sample.py);
here it follows the bare marker, so the adapter learns that behaviour without being told.
"""
import argparse
import copy
import json
import math
import os
import random
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

INVOKE = "\n\n[cut]\n"        # a short fixed marker, nothing more: the adapter switches on at these tokens.
                               # It carries no instruction and no budget; what to do after it is learned in training.
CLOSE = "</think>"


REST = ""                      # nothing follows the marker


def target_text(thinking, answer):
    thinking = (thinking or "").strip()
    return (thinking + "\n" if thinking else "") + CLOSE + "\n\n" + answer.strip()


# ---------------------------------------------------------------- model

def target_spec(targets):
    return "all-linear" if targets == "all-linear" else targets.split(",")


def load(model_name, adapter=None, r=32, alpha=32, dropout=0.05, targets="all-linear", trainable=False):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    base, info = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto",
                                                      output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if "lm_head" not in k]
    if missing:      # e.g. a multimodal checkpoint whose text weights live under another prefix
        raise SystemExit(f"{len(missing)} weights were NOT loaded (they would be random), e.g. {missing[:3]}. "
                         "Send me this message.")
    invoke_ids = tok.encode(INVOKE, add_special_tokens=False)
    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    else:
        cfg = LoraConfig(task_type="CAUSAL_LM", r=r, lora_alpha=alpha, lora_dropout=dropout,
                         target_modules=target_spec(targets), alora_invocation_tokens=invoke_ids)
        model = get_peft_model(base, cfg)
    on_cpu = {d for d in (getattr(base, "hf_device_map", None) or {}).values() if str(d) in ("cpu", "disk")}
    if on_cpu:
        raise SystemExit(f"part of the model was placed on {on_cpu}: not enough GPU memory. Stop vLLM and retry.")
    return tok, model, invoke_ids


def chat_ids(tok, user, system, think_open):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if think_open == "auto":
        think_open = "" if text.rstrip().endswith("<think>") else "<think>\n"
    return tok.encode(text + think_open, add_special_tokens=False)


def prefill(model, cache, ids, dev):
    """Extend the cache with base-model keys/values for `ids` (adapter off, no gradient)."""
    if ids:
        with torch.no_grad(), model.disable_adapter():
            model(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True)


def fork(cache):
    """A private copy of the base cache for one tail. The base cache itself is never changed, so this
    works for any cache type, including hybrid models whose linear-attention state cannot be cropped.

    Hybrid models (linear attention, e.g. Gated DeltaNet) read their recurrent / conv state from the
    cache and then overwrite it IN PLACE. With gradients on, backward still needs the old value, so the
    copy hands out clones on read: the in-place write then hits the cache's own tensor, not the one
    the graph saved."""
    c = copy.deepcopy(cache)
    for name in ("get_recurrent_state", "get_conv_state"):
        orig = getattr(c, name, None)
        if orig is not None:
            setattr(c, name, (lambda f: lambda *a, **k: (lambda t: None if t is None else t.clone())(f(*a, **k)))(orig))
    return c


def crop_to(cache, n):
    """Drop everything after the first n tokens (negative crop works on old and new transformers)."""
    extra = cache.get_seq_length() - n
    if extra > 0:
        cache.crop(-extra)
    # what is left must not point into the graph of the pass that was just cropped away
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if getattr(layer, "keys", None) is not None:
                layer.keys, layer.values = layer.keys.detach(), layer.values.detach()
    else:                                                   # older transformers
        cache.key_cache = [k.detach() for k in cache.key_cache]
        cache.value_cache = [v.detach() for v in cache.value_cache]


def tail_loss(model, cache, notice_ids, target_ids, dev, close_id=None, close_weight=2.0):
    """Adapter on. Feed notice + target on top of the base cache; cross-entropy on the target tokens only."""
    ids = torch.tensor([notice_ids + target_ids], device=dev)
    logits = model(input_ids=ids, past_key_values=cache, use_cache=True).logits[0]
    n = len(notice_ids)
    pred, gold = logits[n - 1:-1].float(), ids[0, n:]                 # position i predicts token i+1
    loss = F.cross_entropy(pred, gold, reduction="none")
    w = torch.ones_like(loss)
    if close_id is not None:
        w[gold == close_id] = close_weight                           # closing the thinking is the key decision
    return (loss * w).sum() / w.sum()


# ---------------------------------------------------------------- train

def group_examples(path, val_frac=0.05, seed=0):
    by_trace = defaultdict(list)
    for line in open(path):
        e = json.loads(line)
        by_trace[(e["id"], e["sample"])].append(e)
    ids = sorted({k[0] for k in by_trace})
    random.Random(seed).shuffle(ids)
    val = set(ids[: max(1, int(len(ids) * val_frac))])
    tr = [g for k, g in by_trace.items() if k[0] not in val]
    va = [g for k, g in by_trace.items() if k[0] in val]
    return tr, va


def run_group(model, tok, group, invoke_ids, args, dev, train=True):
    """All examples of one trace. The thinking prefix is read once and extended cut point by cut point."""
    from transformers import DynamicCache
    e0 = group[0]
    base = chat_ids(tok, e0["user"], args.system, args.think_open)
    close_id = tok.convert_tokens_to_ids(CLOSE)
    cache, done, total, n = DynamicCache(), 0, 0.0, 0
    cuts = defaultdict(list)
    for e in group:
        cuts[len(e["thinking_prefix"])].append(e)
    for chars in sorted(cuts):
        prefix = base + tok.encode(cuts[chars][0]["thinking_prefix"], add_special_tokens=False)
        if len(prefix) > args.max_prefix or len(prefix) < done:
            continue
        prefill(model, cache, prefix[done:], dev)
        done = len(prefix)
        for e in cuts[chars]:
            notice = invoke_ids + tok.encode(REST, add_special_tokens=False)
            target = tok.encode(target_text(e["target_thinking"], e["target_answer"]), add_special_tokens=False)
            target = target[: args.max_target] + [tok.eos_token_id]
            with torch.set_grad_enabled(train), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = tail_loss(model, fork(cache), notice, target, dev, close_id) * e.get("weight", 1.0)
            if train:
                (loss / args.accum).backward()
            total += loss.item()
            n += 1
    return total, n


def train(args):
    tok, model, invoke_ids = load(args.model, None, args.rank, args.alpha or args.rank, args.dropout, args.targets)
    dev = next(model.parameters()).device
    model.print_trainable_parameters()
    tr, va = group_examples(args.sft, args.val_frac)
    if args.limit:                                          # smoke run: a few traces, one epoch
        tr, va, args.epochs = tr[: args.limit], va[: max(1, args.limit // 5)], 1
        print(f"--limit {args.limit}: smoke run, 1 epoch")
    n_ex = sum(len(g) for g in tr)
    steps = math.ceil(n_ex * args.epochs / args.accum)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.03 * steps)))
                                              * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    print(f"{n_ex} training examples in {len(tr)} traces, {sum(len(g) for g in va)} validation examples "
          f"({len({g[0]['id'] for g in va})} held-out problems), {steps} optimizer steps")
    state = {"best": float("inf"), "bad": 0, "seen": 0}
    run = []

    def validate(tag):
        model.eval()
        vt = vn = 0
        for g in va:
            tot, n = run_group(model, tok, g, invoke_ids, args, dev, train=False)
            vt, vn = vt + tot, vn + n
        val = vt / max(vn, 1)
        model.train()
        if val < state["best"] - 1e-4:
            state["best"], state["bad"] = val, 0
            model.save_pretrained(args.out)
            json.dump({"invoke": INVOKE, "notice_rest": REST, "val_loss": val, "at": tag, "rank": args.rank},
                      open(os.path.join(args.out, "cut_meta.json"), "w"))
            print(f"== {tag}: validation loss {val:.4f}  (best, saved to {args.out})", flush=True)
        else:
            state["bad"] += 1
            print(f"== {tag}: validation loss {val:.4f}  (best {state['best']:.4f}; "
                  f"{state['bad']}/{args.patience} without improvement)", flush=True)
        return state["bad"] >= args.patience

    stop = False
    for ep in range(args.epochs):
        random.Random(ep).shuffle(tr)
        model.train()
        for gi, g in enumerate(tr):
            tot, n = run_group(model, tok, g, invoke_ids, args, dev, train=True)
            if not n:
                continue
            run.append(tot / n)
            seen = state["seen"] = state["seen"] + n
            if seen // args.accum > (seen - n) // args.accum:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if (gi + 1) % 20 == 0:
                print(f"epoch {ep + 1} trace {gi + 1}/{len(tr)} | loss {sum(run[-20:]) / len(run[-20:]):.3f} "
                      f"| lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if args.val_every and (gi + 1) % args.val_every == 0 and gi + 1 < len(tr):
                if validate(f"epoch {ep + 1}, trace {gi + 1}"):
                    stop = True
                    break
        if stop or validate(f"end of epoch {ep + 1}"):
            print("validation loss stopped improving: stopping early (the best adapter is saved)")
            break
    print(f"best validation loss {state['best']:.4f}; adapter in {args.out}")


# ---------------------------------------------------------------- GRPO (after SFT)

SAME_AS_UNCUT = 0.15


def reward(q, n_think, n_rest, closed, n_answer, n_answer_uncut, dither=0.0, a=0.3, b=0.3, c=0.1, d=0.1):
    """Must be correct; then the fewer tokens and the less hesitation the better. There is no budget.

    q        answer quality: 1 = correct, 0.15 = wrong but the same answer the uncut run gave, 0 = anything else
    used     thinking after the marker / thinking the uncut run still had ahead, capped at 1
    growth   how much longer the answer is than the uncut answer, capped at 1 (0 = not longer)
    dither   hesitation phrases per 1,000 thinking tokens after the marker / 10, capped at 1
             (re-checking, other methods, re-reading the problem); small weight: length does most of the work,
             this only breaks ties between equally short rollouts and cannot be gamed by just avoiding the words
    R = q * (1 - a*used - b*growth - c*[ran into the safety cap without closing its thinking] - d*dither)

    The three outcomes never overlap: correct 0.20..1.00, same-as-uncut 0.03..0.15, wrong 0.
    So the slowest, most hesitant correct rollout still beats the fastest wrong one; saving tokens can never
    pay for a wrong answer, and tokens moved from thinking into the answer are charged through `growth`."""
    used = min(1.0, n_think / max(n_rest, 1))
    growth = min(1.0, max(0.0, n_answer / max(n_answer_uncut, 1) - 1.0))
    hes = min(1.0, max(0.0, dither) / 10.0)
    return q * (1.0 - a * used - b * growth - (0.0 if closed else c) - d * hes)


@torch.no_grad()
def rollout(model, cache, done, ids0, cap, close_id, forced_close, answer_max, tau, dev):
    """Sample one wrap-up with the adapter. Returns the tail tokens, which of them the policy chose,
    how many thinking tokens it used, whether it closed by itself, and the answer tokens."""
    gen = dict(do_sample=True, temperature=tau, top_p=1.0, top_k=0, pad_token_id=close_id)
    c = fork(cache)
    g = model.generate(input_ids=torch.tensor([ids0], device=dev), past_key_values=c,
                       max_new_tokens=max(cap, 4), eos_token_id=close_id, **gen)
    think = g[0, len(ids0):].tolist()
    closed = close_id in think
    forced = [] if closed else forced_close
    ids1 = ids0 + think + forced
    g = model.generate(input_ids=torch.tensor([ids1], device=dev), past_key_values=c,
                       max_new_tokens=answer_max, **gen)
    answer = g[0, len(ids1):].tolist()
    n_think = len(think) - (1 if closed else 0)
    return (think + forced + answer, [1] * len(think) + [0] * len(forced) + [1] * len(answer), n_think, closed, answer,
            think)


def seq_logp(model, cache, done, notice, tail, mask, tau, dev):
    """Mean log-probability of the tokens the policy chose in one rollout (adapter on, gradient on)."""
    ids = torch.tensor([notice + tail], device=dev)
    logits = model(input_ids=ids, past_key_values=cache, use_cache=True).logits[0]
    n = len(notice)
    logp = F.log_softmax(logits[n - 1:-1].float() / tau, dim=-1).gather(1, ids[0, n:, None]).squeeze(1)
    m = torch.tensor(mask, device=dev, dtype=logp.dtype)
    return (logp * m).sum() / m.sum()                      # forced tokens are masked out


def grpo_group(model, cache, done, ids0, cap, close_id, forced_close, answer_max, tau, dev, score, group):
    """One GRPO step on one cut state: sample `group` wrap-ups, score them, and push the adapter toward
    the ones that scored above the group mean. Gradients are added to .grad; returns the rewards."""
    rolls = [rollout(model, cache, done, ids0, cap, close_id, forced_close, answer_max, tau, dev) for _ in range(group)]
    R = torch.tensor([score(*r[2:]) for r in rolls])
    if float(R.std()) < 1e-6:
        return R, False, rolls                            # all the same: nothing to learn from this state
    adv = (R - R.mean()) / (R.std() + 1e-6)
    for (tail, mask, *_), a in zip(rolls, adv):
        (-(float(a) / group) * seq_logp(model, fork(cache), done, ids0[done:], tail, mask, tau, dev)).backward()
    return R, True, rolls


def grpo(args):
    from transformers import DynamicCache
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cutlib import clean, dither_rate, final_box, graded, read_traces, same
    tok, model, invoke_ids = load(args.model, args.adapter, trainable=True)
    dev = next(model.parameters()).device
    close_id = tok.convert_tokens_to_ids(CLOSE)
    forced_close = tok.encode("\n" + CLOSE, add_special_tokens=False)
    traces = {(r["id"], r.get("sample", 0)): r for r in read_traces(args.traces)}
    by_state, anchors = defaultdict(list), defaultdict(list)
    for line in open(args.sft):
        e = json.loads(line)
        anchors[(e["id"], e["sample"], e["cut_tokens"])].append(e)
    if args.runs:                       # every cut point of the runs files, also those with no SFT target
        seen = set()
        for path in args.runs:
            for line in open(path):
                r = json.loads(line)
                key = (r["id"], r["sample"], r["u"])
                if key in seen or key[:2] not in traces:
                    continue
                seen.add(key)
                t = traces[key[:2]]
                state = ("has conclusion" if r["prefix_has_final_box"] else
                         "mid-way, solvable" if r["uncut_correct"] else "unsolved when uncut")
                by_state[state].append({"id": r["id"], "sample": r["sample"], "cut_tokens": r["u"], "rest": r["L"] - r["u"],
                                        "user": t["prompt"] + ("\n\nPut your final answer in \\boxed{}." if t.get("suffix") is None else t["suffix"]),
                                        "thinking_prefix": t["thinking"][: r["prefix_chars"]]})
    else:                               # only the cut points that have an SFT target
        for key, es in anchors.items():
            by_state[es[0]["state"]].append(dict(es[0], rest=es[0]["uncut_thinking_tokens"] - es[0]["cut_tokens"]))
    anchor_pool = [x for es in anchors.values() for x in es]
    n_own = sum((e["id"], e["sample"], e["cut_tokens"]) in anchors for v in by_state.values() for e in v)
    print("cut states to sample from: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(by_state.items()))
          + f" | {n_own} of them have their own SFT target; the others are anchored on random SFT examples")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    rng = random.Random(0)
    log = defaultdict(list)
    for step in range(1, args.steps + 1):
        e = rng.choice(by_state[rng.choice(sorted(by_state))])            # states are drawn evenly
        t = traces[(e["id"], e["sample"])]
        uncut_final, n_ans_uncut = final_box(t.get("answer")), t.get("answer_tokens") or 1
        prefix = chat_ids(tok, e["user"], args.system, args.think_open) + tok.encode(e["thinking_prefix"], add_special_tokens=False)
        if len(prefix) > args.max_prefix:
            continue
        cache = DynamicCache()
        prefill(model, cache, prefix, dev)
        ids0 = prefix + invoke_ids + tok.encode(REST, add_special_tokens=False)
        cap = min(args.cap, max(64, e["rest"]))             # never allow more than the uncut run itself used

        def score(n_think, closed, answer_ids, think_ids):
            ans = clean(tok.decode(answer_ids, skip_special_tokens=False).split(CLOSE)[-1])
            box = final_box(ans)
            q = 1.0 if graded(ans, t["gold"]) else SAME_AS_UNCUT if (box is not None and uncut_final is not None
                                                                    and same(box, uncut_final)) else 0.0
            dr = dither_rate(tok.decode(think_ids, skip_special_tokens=True), n_think) if n_think >= 20 else 0.0
            log["ok"].append(q > 0); log["think"].append(n_think); log["closed"].append(closed); log["dither"].append(dr)
            return reward(q, n_think, e["rest"], closed, len(answer_ids), n_ans_uncut, dr)

        model.eval()                                                       # no dropout while sampling
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            R, used, _ = grpo_group(model, cache, len(prefix), ids0, cap, close_id, forced_close, args.answer_max,
                                 args.temp, dev, score, args.group)
            if args.anchor > 0 and anchor_pool:          # stay close to the SFT targets
                sup = anchors.get((e["id"], e["sample"], e["cut_tokens"]))
                if sup:                                  # this state has its own SFT target: reuse its cache
                    x, acache = rng.choice(sup), cache
                else:                                    # a GRPO-only problem: anchor on a random SFT example
                    x = rng.choice(anchor_pool)
                    aprefix = chat_ids(tok, x["user"], args.system, args.think_open) + \
                        tok.encode(x["thinking_prefix"], add_special_tokens=False)
                    acache = None
                    if len(aprefix) <= args.max_prefix:
                        acache = DynamicCache()
                        prefill(model, acache, aprefix, dev)
                if acache is not None:
                    tgt = tok.encode(target_text(x["target_thinking"], x["target_answer"]), add_special_tokens=False)
                    tgt = tgt[: args.max_target] + [tok.eos_token_id]
                    nt = invoke_ids + tok.encode(REST, add_special_tokens=False)
                    (args.anchor * tail_loss(model, fork(acache), nt, tgt, dev, close_id)).backward()
        log["R"].append(float(R.mean())); log["used"].append(used)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            k = 10 * args.group
            print(f"step {step}/{args.steps} | reward {sum(log['R'][-10:]) / 10:.3f} | acceptable {sum(log['ok'][-k:]) / k:.0%} "
                  f"| thinking after [cut] {sum(log['think'][-k:]) / k:.0f} tok | closed by itself {sum(log['closed'][-k:]) / k:.0%} "
                  f"| hesitation {sum(log['dither'][-k:]) / k:.1f}/1k "
                  f"| states with a signal {sum(log['used'][-10:])}/10", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            model.save_pretrained(args.out)
    print(f"saved to {args.out}. Now run the eval and compare with the SFT adapter before using it.")


# ---------------------------------------------------------------- eval (exact, with Hugging Face generate)

@torch.no_grad()
def evaluate(args):
    from transformers import DynamicCache
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cutlib import clean, graded, read_traces
    tok, model, invoke_ids = load(args.model, args.adapter)
    model.eval()
    dev = next(model.parameters()).device
    traces = {(r["id"], r.get("sample", 0)): r for r in read_traces(args.traces)}
    cuts, done = {}, set()
    for line in open(args.runs):
        r = json.loads(line)
        cuts.setdefault((r["id"], r["sample"], r["u"]), r)
        done.add(r["key"])
    items = sorted(cuts.items())
    random.Random(0).shuffle(items)
    items = items[: args.max_cuts] if args.max_cuts else items
    out = open(args.runs, "a")
    for ci, (key, meta) in enumerate(items):
        t = traces[key[:2]]
        prefix = chat_ids(tok, t["prompt"] + ("\n\nPut your final answer in \\boxed{}." if t.get("suffix") is None else t["suffix"]),
                          args.system, args.think_open)
        prefix += tok.encode(t["thinking"][: meta["prefix_chars"]], add_special_tokens=False)
        base_cache = DynamicCache()
        prefill(model, base_cache, prefix, dev)
        for b in [args.cap]:
            rk = f"{key[0]}|{key[1]}|{key[2]}|{args.name}|0"
            if rk in done:
                continue
            ids = prefix + invoke_ids + tok.encode(REST, add_special_tokens=False)
            torch.manual_seed(1000 + ci)
            think, n_think, closed = "", 0, True
            c = fork(base_cache)
            if b:                                                    # phase 1: wrap-up thinking, at most b tokens
                g = model.generate(input_ids=torch.tensor([ids], device=dev), past_key_values=c,
                                   max_new_tokens=b, do_sample=True, stop_strings=[CLOSE], tokenizer=tok)
                new = g[0, len(ids):].tolist()
                think = tok.decode(new, skip_special_tokens=False)
                closed = CLOSE in think
                n_think = len(new)
                ids = ids + new + ([] if closed else tok.encode("\n" + CLOSE, add_special_tokens=False))
            g = model.generate(input_ids=torch.tensor([ids], device=dev), past_key_values=c,
                               max_new_tokens=args.answer_max, do_sample=True)
            ans_ids = g[0, len(ids):].tolist()
            answer = clean(tok.decode(ans_ids, skip_special_tokens=False).split(CLOSE)[-1])
            rec = dict({x: meta[x] for x in ("id", "sample", "source", "gold", "L", "u", "frac", "prefix_chars",
                                             "uncut_correct", "uncut_finished", "uncut_answer_tok", "prefix_has_final_box")},
                       key=rk, cond=args.name, kind="alora", grace=b, k=0, think_after=clean(think.split(CLOSE)[0]),
                       think_after_tok=n_think, self_term=closed if b else None, answer=answer,
                       answer_tok=len(ans_ids), answer_finish="", correct=graded(answer, t["gold"]))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
        if (ci + 1) % 10 == 0:
            print(f"{ci + 1}/{len(items)} cut points", flush=True)
    print(f"appended to {args.runs}. Next: python cut_report.py {args.runs}")


@torch.no_grad()
def check(args):
    """On the real model: with the adapter loaded, a request that has no notice must give the base model's logits."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cutlib import read_traces
    tok, model, _ = load(args.model, args.adapter)
    model.eval()
    dev = next(model.parameters()).device
    rows = [r for r in read_traces(args.traces) if r.get("thinking")]
    random.Random(0).shuffle(rows)
    worst = 0.0
    for r in rows[: args.n]:
        ids = chat_ids(tok, r["prompt"], args.system, args.think_open) + tok.encode(r["thinking"], add_special_tokens=False)[:2000]
        x = torch.tensor([ids], device=dev)
        with model.disable_adapter():
            a = model(input_ids=x).logits.float()
        b = model(input_ids=x).logits.float()
        worst = max(worst, float((a - b).abs().max()))
    print(f"{args.n} uncut requests: largest logit difference between base and adapter-loaded model = {worst:.2e}")
    print("PASS: identical" if worst == 0.0 else "Not exactly zero: send me this number.")


# ---------------------------------------------------------------- selftest (tiny random model, CPU)

def build_test_model(model_name, layers, dev):
    """A small model with random weights. With --model it has the REAL architecture of that model
    (its config, its layer classes and module names), only fewer layers, so the checks below run on
    the same code path the 27B will use. Without --model it is a tiny Llama."""
    from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    if not model_name:
        cfg = LlamaConfig(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
        return LlamaForCausalLM(cfg).to(dev).eval(), cfg, 300
    cfg = AutoConfig.from_pretrained(model_name)
    text = getattr(cfg, "text_config", None) or cfg
    n_full = text.num_hidden_layers
    full_types = list(text.layer_types) if getattr(text, "layer_types", None) else ["layer"] * n_full
    text.num_hidden_layers = layers
    if getattr(text, "layer_types", None):
        text.layer_types = list(text.layer_types)[:layers]
    cfg._full_layer_types = full_types
    print(f"architecture: {cfg.model_type} ({type(cfg).__name__}), {n_full} layers in the real model, testing {layers}")
    if getattr(text, "layer_types", None):
        print(f"layer types in the test model: {text.layer_types}")
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    return m.to(dev).eval(), cfg, text.vocab_size


def report_targets(model, targets):
    """Which Linear layers the adapter will be attached to, by name suffix."""
    import torch.nn as nn
    from collections import Counter
    names = [n for n, mod in model.named_modules() if isinstance(mod, nn.Linear)]
    suffix = Counter(n.split(".")[-1] for n in names)
    print("Linear layers by name:", ", ".join(f"{k} ×{v}" for k, v in sorted(suffix.items())))
    if targets != "all-linear":
        missing = [t for t in targets.split(",") if t not in suffix]
        if missing:
            print(f"!! these target names do not exist in this model: {missing}")


def selftest(args):
    from peft import LoraConfig, get_peft_model
    from transformers import DynamicCache
    torch.manual_seed(0)
    dev = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    base, cfg, V = build_test_model(args.model, args.layers, dev)
    report_targets(base, args.targets)
    lo_id = 20
    invoke = [7, 8, 9]
    rnd = lambda n: torch.randint(lo_id, min(V, 30000), (n,)).tolist()
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=args.rank, lora_alpha=args.rank, init_lora_weights=False,
                                            target_modules=target_spec(args.targets), alora_invocation_tokens=invoke))
    model.eval()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"adapter parameters in the test model: {n_tr:,} (rank {args.rank}, targets {args.targets})")
    full_types = getattr(cfg, "_full_layer_types", None)
    if full_types:      # scale the per-layer counts up to the real number of layers of each type
        import re as _re
        text = getattr(cfg, "text_config", None) or cfg
        test_types = list(getattr(text, "layer_types", None) or ["layer"] * text.num_hidden_layers)
        per = defaultdict(list)
        counts = defaultdict(int)
        for name, prm in model.named_parameters():
            m = _re.search(r"layers\.(\d+)\.", name)
            if prm.requires_grad and m:
                counts[int(m.group(1))] += prm.numel()
        for i, t in enumerate(test_types):
            per[t].append(counts.get(i, 0))
        est = sum(sum(per[t]) / len(per[t]) for t in full_types if per.get(t))
        from collections import Counter
        print("layers in the real model: " + ", ".join(f"{k} ×{v}" for k, v in Counter(full_types).items()))
        for r in (16, 32):
            print(f"  estimated adapter size on the real model at rank {r}: {est * r / args.rank / 1e6:.0f}M parameters "
                  f"({est * r / args.rank * 2 / 2**20:.0f} MiB in bf16)")
    close = lambda a, b: float((a - b).abs().max() / (b.abs().max() + 1e-6))     # relative difference
    tol = 1e-4 if dev.type == "cpu" else 2e-3
    prefix = rnd(40)
    tail = invoke + rnd(12)
    full = torch.tensor([prefix + tail], device=dev)
    with torch.no_grad():
        with model.disable_adapter():
            base_full = model(input_ids=full).logits[0].float()
            base_plain = model(input_ids=torch.tensor([prefix], device=dev)).logits[0].float()
        one_pass = model(input_ids=full).logits[0].float()
        cache = DynamicCache(config=cfg) if args.model else DynamicCache()
        prefill(model, cache, prefix, dev)
        two_pass = model(input_ids=torch.tensor([tail], device=dev), past_key_values=fork(cache), use_cache=True).logits[0].float()
        no_invoke = model(input_ids=torch.tensor([prefix], device=dev)).logits[0].float()
    n = len(prefix)
    d1, d2, d3, d4 = (close(one_pass[:n - 1], base_full[:n - 1]), close(one_pass[n:], base_full[n:]),
                      close(two_pass, one_pass[n:]), close(no_invoke, base_plain))
    checks = {
        f"1 before the notice, adapter-on logits equal the base model (rel. diff {d1:.1e})": d1 < tol,
        f"2 after the notice, the adapter changes the logits (rel. diff {d2:.1e})": d2 > 10 * tol,
        f"3 base cache + tail gives the same logits as one full pass (rel. diff {d3:.1e})": d3 < tol,
        f"4 a request with no notice is identical to the base model (rel. diff {d4:.1e})": d4 < tol,
    }
    g = model.generate(input_ids=full[:, : n + 3], past_key_values=fork(cache), max_new_tokens=4, do_sample=False)
    checks["5 generate() continues from the base cache"] = g.shape[1] == n + 3 + 4
    # 6: the training step learns (overfit four examples)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-2 if not args.model else 2e-3)
    ex = [(rnd(30), invoke + [11], rnd(10)) for _ in range(4)]
    hist = []
    model.train()
    for _ in range(60 if not args.model else 15):
        tot = 0.0
        for p, nt, tg in ex:
            c = DynamicCache(config=cfg) if args.model else DynamicCache()
            prefill(model, c, p, dev)
            loss = tail_loss(model, fork(c), nt, tg, dev)
            loss.backward()
            tot += loss.item()
        opt.step(); opt.zero_grad()
        hist.append(tot / 4)
    checks[f"6 training lowers the loss ({hist[0]:.2f} -> {hist[-1]:.2f})"] = hist[-1] < hist[0] - 0.5
    c = DynamicCache(config=cfg) if args.model else DynamicCache()     # as in training: several tails on one prefix
    prefill(model, c, ex[0][0], dev)
    same_twice = []
    for _ in range(2):
        loss = tail_loss(model, fork(c), ex[0][1], ex[0][2], dev)
        loss.backward()
        same_twice.append(loss.item())
    opt.zero_grad()
    checks["6b one prefix cache serves several tails (same loss twice)"] = abs(same_twice[0] - same_twice[1]) < 1e-3
    # 7: after one GRPO step, the best rollout must have become more likely relative to the worst one
    model.eval()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05 if not args.model else 0.005)
    p7 = rnd(20)
    toy = lambda n_think, closed, answer, think=None: sum(x < V // 2 for x in answer) / max(len(answer), 1)
    c = DynamicCache(config=cfg) if args.model else DynamicCache()
    prefill(model, c, p7, dev)
    R, used, rolls = grpo_group(model, c, len(p7), p7 + invoke, 4, 5, [5], 12, 1.0, dev, toy, 6)
    hi, lo = rolls[int(R.argmax())], rolls[int(R.argmin())]

    def gap():
        with torch.no_grad():
            a = seq_logp(model, fork(c), len(p7), invoke, hi[0], hi[1], 1.0, dev)
            b = seq_logp(model, fork(c), len(p7), invoke, lo[0], lo[1], 1.0, dev)
        return float(a - b)

    g0 = gap()
    opt.step(); opt.zero_grad()
    g1 = gap()
    checks[f"7 a GRPO step favours the best rollout over the worst (gap {g0:+.3f} -> {g1:+.3f})"] = used and g1 > g0
    worst_ok = reward(1.0, 9000, 3000, False, 4000, 500, 99)        # slowest, most hesitant correct rollout
    best_same = reward(SAME_AS_UNCUT, 0, 3000, True, 500, 500, 0)
    worst_same = reward(SAME_AS_UNCUT, 9000, 3000, False, 4000, 500, 99)
    best_bad = reward(0.0, 0, 3000, True, 100, 500, 0)
    checks[f"8 slowest correct ({worst_ok:.2f}) > fastest same-as-uncut ({best_same:.2f}) > "
           f"slowest same-as-uncut ({worst_same:.3f}) > any wrong ({best_bad:.2f})"] = worst_ok > best_same > worst_same > best_bad
    import peft
    import transformers
    print(f"peft {peft.__version__}, transformers {transformers.__version__}, torch {torch.__version__}, device {dev}")
    for k, v in checks.items():
        print(("PASS  " if v else "FAIL  ") + k)
    if not all(checks.values()):
        sys.exit("Some checks failed. Send me this output before training.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("selftest")
    st.add_argument("--model", default=None, help="test the real architecture of this model (random weights, few layers)")
    st.add_argument("--layers", type=int, default=4)
    st.add_argument("--targets", default="all-linear")
    st.add_argument("--rank", type=int, default=8)
    st.add_argument("--device", default="auto")
    for name in ("train", "eval", "check", "grpo"):
        p = sub.add_parser(name)
        if name == "train":
            p.add_argument("sft")
            p.add_argument("--out", required=True)
            p.add_argument("--rank", type=int, default=16,
                           help="16 is enough for this narrow behaviour and overfits less; compare with 32 on eval")
            p.add_argument("--alpha", type=int, default=0, help="0 = same as --rank")
            p.add_argument("--dropout", type=float, default=0.05)
            p.add_argument("--targets", default="all-linear",
                           help="all-linear (every Linear layer except lm_head) or a comma list such as q_proj,k_proj,v_proj")
            p.add_argument("--lr", type=float, default=1e-4)
            p.add_argument("--epochs", type=int, default=2)
            p.add_argument("--val-frac", type=float, default=0.1, help="share of PROBLEMS held out for validation")
            p.add_argument("--val-every", type=int, default=0,
                           help="also validate every N training traces (0 = only at the end of each epoch)")
            p.add_argument("--patience", type=int, default=2,
                           help="stop after this many validations in a row without a new best")
            p.add_argument("--accum", type=int, default=8, help="examples per optimizer step")
            p.add_argument("--max-prefix", type=int, default=20000, help="skip cut points with a longer prompt + prefix")
            p.add_argument("--max-target", type=int, default=3000)
            p.add_argument("--limit", type=int, default=0,
                           help="train on the first N traces only and 1 epoch (a smoke run on the real model)")
        elif name == "grpo":
            p.add_argument("sft")
            p.add_argument("traces")
            p.add_argument("--runs", nargs="*", default=[],
                           help="runs files from cut_bench.py: sample cut states from all their cut points "
                                "(also the ones the uncut model could not solve)")
            p.add_argument("--adapter", required=True, help="the SFT adapter to start from")
            p.add_argument("--out", required=True)
            p.add_argument("--steps", type=int, default=400)
            p.add_argument("--group", type=int, default=4, help="rollouts per cut state")
            p.add_argument("--cap", type=int, default=2048, help="safety cap on thinking after the notice (not told to the model)")
            p.add_argument("--lr", type=float, default=2e-5)
            p.add_argument("--temp", type=float, default=1.0)
            p.add_argument("--anchor", type=float, default=0.5, help="weight of the SFT loss mixed into every step")
            p.add_argument("--answer-max", type=int, default=1536)
            p.add_argument("--max-prefix", type=int, default=20000)
            p.add_argument("--max-target", type=int, default=3000)
            p.add_argument("--save-every", type=int, default=50)
        elif name == "check":
            p.add_argument("traces")
            p.add_argument("--adapter", required=True)
            p.add_argument("--n", type=int, default=20)
        else:
            p.add_argument("runs")
            p.add_argument("traces")
            p.add_argument("--adapter", required=True)
            p.add_argument("--cap", type=int, default=2048, help="safety cap on thinking after the notice (not told to the model)")
            p.add_argument("--name", default="alora", help="row name in cut_report.py, e.g. alora-sft or alora-grpo")
            p.add_argument("--max-cuts", type=int, default=0, help="0 = every cut point in the runs file")
            p.add_argument("--answer-max", type=int, default=2048)
        p.add_argument("--model", required=True)
        p.add_argument("--system", default=None)
        p.add_argument("--think-open", default="auto")
    args = ap.parse_args()
    {"selftest": selftest, "train": train, "eval": evaluate, "check": check, "grpo": grpo}[args.cmd](args)


if __name__ == "__main__":
    main()
