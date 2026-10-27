#!/usr/bin/env python3
"""GRPO on top of the SFT adapter: reward = right final answer after a mid-thought budget cut.

One step:
  1. take --groups x --oversample cut prompts (problem + thinking up to the cut + notice with W)
  2. sample G continuations each with vLLM (current adapter), at most W tokens; if the model does not close
     </think> itself it is closed for it, then "The final answer is \\boxed{" and up to --final-max tokens
  3. reward 1 if the answer is right, else 0 (optionally + --save-bonus * unused share of W when right)
  4. advantage = (reward - group mean) / group std; groups where all G agree carry no signal and are skipped
  5. one policy-gradient update on the LoRA (token-level mean, only sampled tokens get gradient), then the new
     adapter is handed to vLLM for the next step
No KL term and no value model. Starts from --init-adapter (the SFT model) and only trains the LoRA.

  python grpo.py --init-adapter init_adapter --prompts grpo_prompts.jsonl --out grpo_out
  python grpo.py ... --resume                        (continue from the last checkpoint)
  python grpo.py --mock --model tiny --init-adapter tiny_adapter --prompts p.jsonl --out m   (CPU test)
At the end it writes --out/merged, a normal model folder for vLLM / aime_drop_bench.py.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import time
from pathlib import Path

FINAL_PREFIX = "The final answer is \\boxed{"


class VEngine:
    def __init__(self, a):
        import os
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        self.llm = LLM(model=a.model, enable_lora=True, max_lora_rank=a.rank, max_loras=1,
                       gpu_memory_utilization=a.vllm_util, max_model_len=a.max_len, enable_prefix_caching=True,
                       max_num_seqs=a.max_num_seqs, seed=a.seed)
        self.n = 0

    def set_adapter(self, path):
        from vllm.lora.request import LoRARequest
        self.n += 1
        self.lora = LoRARequest(lora_name=f"policy{self.n}", lora_int_id=self.n, lora_path=str(path))

    def generate(self, prompts, max_tokens, seeds, temp, stop_ids=None):
        from vllm import SamplingParams
        sps = [SamplingParams(temperature=temp, top_p=1.0, max_tokens=int(m), seed=int(s), stop_token_ids=stop_ids)
               for m, s in zip(max_tokens, seeds)]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sps, lora_request=self.lora,
                                 use_tqdm=False)
        return [(list(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outs]


class MockEngine:
    """CPU test: random continuations, answer right with a probability that grows with length."""

    def __init__(self, tok, golds):
        self.tok, self.golds = tok, golds
        self.t_close = tok.convert_tokens_to_ids("</think>")
        self.vocab = [tok.encode(w, add_special_tokens=False)[0] for w in (" so", " we", " get", " x", "\n")]

    def set_adapter(self, path):
        pass

    def generate(self, prompts, max_tokens, seeds, temp, stop_ids=None):
        out = []
        for p, m, s in zip(prompts, max_tokens, seeds):
            h = random.Random(int(s))
            if stop_ids:
                n = h.randint(1, int(m))
                ids = [h.choice(self.vocab) for _ in range(n)]
                out.append((ids + [self.t_close], "stop") if n < m and h.random() < 0.5 else (ids[:int(m)], "length"))
            else:
                gold = self.golds.get(tuple(p[:40]), "0")
                ans = gold if h.random() < 0.5 else "7"
                out.append((self.tok.encode(ans + "}.", add_special_tokens=False), "stop"))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--init-adapter", type=Path, required=True, help="the SFT LoRA to start from")
    ap.add_argument("--prompts", type=Path, required=True, help="prep_grpo.py output")
    ap.add_argument("--out", type=Path, default=Path("grpo_out"))
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--groups", type=int, default=16, help="groups with a learning signal used per step")
    ap.add_argument("--oversample", type=float, default=2.0, help="prompts sampled per step = groups x this")
    ap.add_argument("--G", type=int, default=8, help="samples per prompt")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--final-max", type=int, default=48)
    ap.add_argument("--save-bonus", type=float, default=0.0, help="extra reward for unused budget when right")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank of the init adapter")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--vllm-util", type=float, default=0.45, help="share of the GPU for vLLM; the rest trains")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=2048, help="positions per vocab-projection chunk")
    ap.add_argument("--micro", type=int, default=2, help="sequences per forward pass (memory vs speed)")
    ap.add_argument("--w-scale", type=float, default=1.0,
                    help="shrink every budget W (and the number in the notice) to make the prompts harder")
    ap.add_argument("--no-seen", action="store_true", help="drop prompts whose prefix already has the answer")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mock", action="store_true", help="CPU test with a fake rollout engine")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import thinking_budget_bench as TB

    rows = [json.loads(l) for l in a.prompts.open()]
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.no_seen:
        rows = [r for r in rows if not r.get("answer_seen")]
    if a.w_scale != 1.0:                                  # rewrite the notice at the end of each prompt
        from budget_drop_bench import WARNS
        notice = lambda W: tok.encode(WARNS["warn_count"].format(W=W), add_special_tokens=False)
        kept = []
        for r in rows:
            old = notice(r["W"])
            if r["input_ids"][-len(old):] != old:
                continue
            W = max(32, int(r["W"] * a.w_scale))
            kept.append({**r, "input_ids": r["input_ids"][:-len(old)] + notice(W), "W": W})
        rows = kept
    rows = [r for r in rows if len(r["input_ids"]) + r["W"] + a.final_max + 32 <= a.max_len]
    t_close = tok.convert_tokens_to_ids("</think>")
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    fp = enc("\n\n" + FINAL_PREFIX)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # vLLM first, while the GPU is empty; it runs in its own process and takes --vllm-util of the GPU
    eng = MockEngine(tok, {tuple(r["input_ids"][:40]): r["gold"] for r in rows}) if a.mock else VEngine(a)
    dev = "cpu" if a.mock else "cuda"
    import transformers
    new_tf = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56)
    kw = {"dtype": torch.bfloat16} if new_tf else {"torch_dtype": torch.bfloat16}
    base = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation="sdpa", **kw).to(dev)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    ck = a.out / "checkpoint"
    resume = a.resume and (ck / "state.json").exists()
    model = PeftModel.from_pretrained(base, ck if resume else a.init_adapter, is_trainable=True)
    inner = model.get_base_model()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    state = {"step": 0, "cursor": 0}
    if resume:
        state = json.loads((ck / "state.json").read_text())
        if (ck / "optim.pt").exists():
            opt.load_state_dict(torch.load(ck / "optim.pt", map_location=dev))
        print(f"resuming at step {state['step']}", flush=True)
    order = list(range(len(rows)))
    random.Random(a.seed).shuffle(order)
    log_path = a.out / "log.jsonl"
    live = a.out / "live"

    def push_adapter(step):
        path = live / f"s{step}"
        model.save_pretrained(path)
        eng.set_adapter(path)
        for old in live.glob("s*"):                       # keep the last two
            if old != path and old.name != f"s{step - 1}":
                shutil.rmtree(old, ignore_errors=True)

    def save_ckpt():
        model.save_pretrained(ck)
        torch.save(opt.state_dict(), ck / "optim.pt")
        (ck / "state.json").write_text(json.dumps(state))

    push_adapter(state["step"])
    print(f"{len(rows)} prompts, {a.steps} steps x {a.groups} groups x {a.G} samples, lr {a.lr}", flush=True)

    model.train()
    t_all, first = time.time(), state["step"]
    while state["step"] < a.steps:
        t0 = time.time()
        step = state["step"]
        n_pick = int(a.groups * a.oversample)
        batch = []
        for _ in range(n_pick):
            batch.append(rows[order[state["cursor"] % len(order)]])
            state["cursor"] += 1
        rng = random.Random(a.seed * 1_000_003 + step)

        # ---------------- rollouts
        prompts = [r["input_ids"] for r in batch for _ in range(a.G)]
        Ws = [r["W"] for r in batch for _ in range(a.G)]
        conts = eng.generate(prompts, Ws, [rng.randrange(2 ** 31) for _ in prompts], a.temp, stop_ids=[t_close])
        cont_ids, closed = [], []
        for ids, why in conts:
            c = why == "stop" or (len(ids) > 0 and ids[-1] == t_close)
            cont_ids.append(ids[:-1] if ids and ids[-1] == t_close else ids)
            closed.append(c)
        ans_prompts = [p + c + [t_close] + fp for p, c in zip(prompts, cont_ids)]
        answers = eng.generate(ans_prompts, [a.final_max] * len(prompts),
                               [rng.randrange(2 ** 31) for _ in prompts], a.temp)
        rewards, correct = [], []
        for i, (ids, _) in enumerate(answers):
            r = batch[i // a.G]
            ok = TB.grade("math500", "\\boxed{" + tok.decode(ids, skip_special_tokens=True), r["gold"])
            correct.append(ok)
            used = (len(cont_ids[i]) + 1) / r["W"]
            rewards.append(float(ok) + (a.save_bonus * max(0.0, 1 - used) if ok else 0.0))
        t_roll = time.time() - t0

        # ---------------- groups with a learning signal
        groups = []
        for g in range(len(batch)):
            rs = rewards[g * a.G:(g + 1) * a.G]
            sd = statistics.pstdev(rs)
            if sd > 1e-6:
                mu = statistics.mean(rs)
                groups.append((g, [(x - mu) / (sd + 1e-4) for x in rs]))
        rng.shuffle(groups)
        groups = groups[:a.groups]

        # ---------------- policy-gradient update (token-level mean over all trained tokens)
        def build(i):
            """full sequence and a mask of the tokens the model actually sampled"""
            p = prompts[i]
            seq = p + cont_ids[i] + [t_close] + fp + answers[i][0]
            mask = [0] * len(p) + [1] * len(cont_ids[i]) + [1 if closed[i] else 0] + [0] * len(fp) + \
                   [1] * len(answers[i][0])
            return seq, mask

        built = {g: [build(g * a.G + k) for k in range(a.G)] for g, _ in groups}
        n_tok = sum(sum(m) for g in built for _, m in built[g])
        loss_sum = 0.0
        if groups and n_tok:
            head = inner.lm_head
            def piece(hc, yc, wc):
                return (F.cross_entropy(head(hc).float(), yc, reduction="none") * wc).sum()

            for g, adv in groups:
                for m0 in range(0, len(built[g]), a.micro):       # a few sequences per pass to save memory
                    seqs = built[g][m0:m0 + a.micro]
                    advs = adv[m0:m0 + a.micro]
                    T = max(len(s_) for s_, _ in seqs)
                    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
                    att = torch.zeros((len(seqs), T), dtype=torch.long)
                    for k, (s_, _) in enumerate(seqs):
                        ids[k, :len(s_)] = torch.tensor(s_)
                        att[k, :len(s_)] = 1
                    ids, att = ids.to(dev), att.to(dev)
                    rows_i, cols_i, wts = [], [], []
                    for k, (s_, m) in enumerate(seqs):
                        for t in range(1, len(s_)):
                            if m[t]:
                                rows_i.append(k)
                                cols_i.append(t)
                                wts.append(advs[k])
                    if not rows_i:
                        continue
                    h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
                    ri = torch.tensor(rows_i, device=dev)
                    ci = torch.tensor(cols_i, device=dev)
                    hs = h[ri, ci - 1]                          # hidden state that predicts token t
                    ys = ids[ri, ci]
                    ws = torch.tensor(wts, device=dev, dtype=torch.float32)
                    tot = 0.0
                    for c0 in range(0, len(ys), a.chunk):
                        sl = slice(c0, c0 + a.chunk)
                        tot = tot + checkpoint(piece, hs[sl], ys[sl], ws[sl], use_reentrant=False)
                    loss = tot / n_tok                              # minimising A * (-log p)
                    loss.backward()
                    loss_sum += float(loss.detach())
                    del h, hs, loss, tot
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        state["step"] = step + 1
        push_adapter(state["step"])
        rec = {"step": state["step"], "reward": round(statistics.mean(rewards), 4),
               "acc": round(statistics.mean(correct), 4), "groups_used": len(groups),
               "groups_with_signal_share": round(len(groups) / max(1, len(batch)), 3),
               "closed_itself": round(statistics.mean(closed), 3),
               "used_W": round(statistics.mean((len(c) + 1) / w for c, w in zip(cont_ids, Ws)), 3),
               "loss": round(loss_sum, 5), "trained_tokens": n_tok,
               "rollout_s": round(t_roll, 1), "step_s": round(time.time() - t0, 1)}
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        el = time.time() - t_all
        print(f"step {rec['step']}/{a.steps}  acc {rec['acc']:.3f}  closed {rec['closed_itself']:.2f}  "
              f"used W {rec['used_W']:.2f}  groups {rec['groups_used']}  loss {rec['loss']:+.4f}  "
              f"{rec['step_s']:.0f}s/step  about {el / (rec['step'] - first) * (a.steps - rec['step']) / 60:.0f} min left",
              flush=True)
        if state["step"] % a.save_every == 0 or state["step"] == a.steps:
            save_ckpt()

    model.save_pretrained(a.out / "adapter")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(a.out / "merged", safe_serialization=True)
    tok.save_pretrained(a.out / "merged")
    print(f"done: adapter in {a.out / 'adapter'}, merged model in {a.out / 'merged'}", flush=True)


if __name__ == "__main__":      # vLLM starts its engine with spawn, which re-imports this file
    main()
