#!/usr/bin/env python3
"""Step 3: LoRA fine-tune of Qwen3-4B on sft.jsonl (make_sft.py), loss only on the target tokens.

  rank 16, alpha 32, dropout 0.05, all linear projections, lr 1e-4, cosine with 3% warmup
  batch 1 x --accum 16, bf16, gradient checkpointing, 3% of examples held out for a loss check
Each example's loss is the mean over its own target tokens, so long uncut traces do not drown out the
short cut ones. The vocab projection is done in chunks of 1024 positions to keep memory down.

  pip install peft                                   (once; vllm already brought torch and transformers)
  python train_lora.py --data sft/sft.jsonl --out lora --merge
  python train_lora.py --data sft/sft.jsonl --out lora --resume       (continue from the last checkpoint)
  python train_lora.py --data sft/sft.jsonl --out lora_quick --limit 40 --epochs 1   (smoke test)

--merge writes lora/merged, a normal model folder that vLLM can load (adapter folded in = always on).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def example_loss(base, ids, n_in, chunk=1024):
    """Mean next-token loss over target positions only (tokens n_in .. end)."""
    h = base.model(input_ids=ids[None]).last_hidden_state[0]      # decoder only, no full-vocab logits
    hs, ys = h[n_in - 1:-1], ids[n_in:]
    head = base.lm_head

    def piece(hc, yc):
        return F.cross_entropy(head(hc).float(), yc, reduction="sum")

    tot = 0.0
    for i in range(0, len(ys), chunk):
        hc, yc = hs[i:i + chunk], ys[i:i + chunk]
        tot = tot + (checkpoint(piece, hc, yc, use_reentrant=False) if torch.is_grad_enabled() else piece(hc, yc))
    return tot / len(ys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("lora"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--accum", type=int, default=16, help="examples per optimizer step")
    ap.add_argument("--max-seq", type=int, default=10240)
    ap.add_argument("--holdout", type=float, default=0.03)
    ap.add_argument("--save-every", type=int, default=50, help="optimizer steps between checkpoints")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--merge", action="store_true", help="also write out/merged for vLLM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="tests only")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    dtype = torch.bfloat16

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    rows = [json.loads(l) for l in a.data.open()][:a.limit]
    rows = [r for r in rows if len(r["input_ids"]) + len(r["target_ids"]) <= a.max_seq and r["target_ids"]]
    random.Random(a.seed).shuffle(rows)
    n_hold = max(1, int(a.holdout * len(rows))) if len(rows) > 20 else 0
    hold, train = rows[:n_hold], rows[n_hold:]
    steps_per_epoch = math.ceil(len(train) / a.accum)
    total_steps = max(1, round(a.epochs * steps_per_epoch))
    warm = max(1, int(0.03 * total_steps))
    print(f"{len(train)} train / {len(hold)} held-out examples, {total_steps} optimizer steps "
          f"({a.epochs} epochs x {steps_per_epoch}), device {dev}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    import transformers
    new_tf = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56)
    kw = {"dtype": dtype} if new_tf else {"torch_dtype": dtype}     # the argument was renamed in 4.56
    model = AutoModelForCausalLM.from_pretrained(a.model, attn_implementation="sdpa", **kw).to(dev)
    assert next(model.parameters()).dtype == dtype, "model did not load in bf16"
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    ck = a.out / "checkpoint"
    if a.resume and (ck / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, ck, is_trainable=True)
        state = json.loads((ck / "state.json").read_text())
        print(f"resuming from step {state['step']}", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                                 target_modules=TARGETS, task_type="CAUSAL_LM"))
        state = {"step": 0, "log": []}
    model.print_trainable_parameters()
    base = model.get_base_model()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    if a.resume and (ck / "optim.pt").exists():
        opt.load_state_dict(torch.load(ck / "optim.pt", map_location=dev))
    lr_at = lambda s: a.lr * (s + 1) / warm if s < warm else \
        a.lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm)))

    def to_ids(r):
        return torch.tensor(r["input_ids"] + r["target_ids"], device=dev), len(r["input_ids"])

    def held_out_loss():
        if not hold:
            return None
        model.eval()
        with torch.no_grad():
            v = [float(example_loss(base, *to_ids(r))) for r in hold]
        model.train()
        return round(sum(v) / len(v), 4)

    def save(path, extra):
        model.save_pretrained(path)
        torch.save(opt.state_dict(), path / "optim.pt")
        (path / "state.json").write_text(json.dumps(extra))

    order = []
    ep = 0
    while len(order) < total_steps * a.accum:                       # the example order for the whole run
        idx = list(range(len(train)))
        random.Random(a.seed + ep).shuffle(idx)
        order += idx
        ep += 1
    if state["step"] == 0:
        state["log"].append({"step": 0, "held_out_loss": held_out_loss()})
        print(f"held-out loss before training: {state['log'][-1]['held_out_loss']}", flush=True)

    model.train()
    t0, toks, run_loss, first = time.time(), 0, [], state["step"]
    for step in range(state["step"], total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for i in order[step * a.accum:(step + 1) * a.accum]:
            ids, n_in = to_ids(train[i])
            loss = example_loss(base, ids, n_in) / a.accum
            loss.backward()
            run_loss.append(float(loss.detach()) * a.accum)
            toks += len(ids)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        state["step"] = step + 1
        if (step + 1) % 10 == 0 or step + 1 == total_steps:
            el = time.time() - t0
            mem = f", {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak" if dev == "cuda" else ""
            print(f"step {step + 1}/{total_steps}  loss {sum(run_loss) / len(run_loss):.4f}  lr {lr_at(step):.2e}  "
                  f"{toks / el:.0f} tok/s{mem}  about {el / (step + 1 - first) * (total_steps - step - 1) / 60:.0f} min left",
                  flush=True)
            state["log"].append({"step": step + 1, "train_loss": round(sum(run_loss) / len(run_loss), 4)})
            run_loss = []
        if (step + 1) % steps_per_epoch == 0 and step + 1 < total_steps:
            state["log"].append({"step": step + 1, "held_out_loss": held_out_loss()})
            print(f"held-out loss after epoch {(step + 1) // steps_per_epoch}: {state['log'][-1]['held_out_loss']}",
                  flush=True)
        if (step + 1) % a.save_every == 0:
            save(ck, state)
    state["log"].append({"step": total_steps, "held_out_loss": held_out_loss()})
    print(f"held-out loss after training: {state['log'][-1]['held_out_loss']}", flush=True)
    save(ck, state)
    model.save_pretrained(a.out / "adapter")
    (a.out / "train_log.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()},
                                                      "train": len(train), "held_out": len(hold),
                                                      "log": state["log"]}, indent=2))
    print(f"adapter saved to {a.out / 'adapter'}", flush=True)

    if a.merge:
        merged = model.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(a.out / "merged", safe_serialization=True)
        tok.save_pretrained(a.out / "merged")
        print(f"merged model saved to {a.out / 'merged'} (load this path in vLLM)", flush=True)


if __name__ == "__main__":
    main()
