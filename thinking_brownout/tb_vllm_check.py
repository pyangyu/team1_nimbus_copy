"""Correctness check for the in-engine cut (tb_vllm.py), in two steps.

1) vLLM running with tb_vllm loaded (see tb_vllm.py), greedy decoding, one request at a time:
     python tb_vllm_check.py vllm traces/Qwen__Qwen3.8-27B.eval.jsonl --n 8 --out runs/tb_vllm_check.json
   mask  : the adapter attached but not cut must give the base model's tokens   (aLoRA is off before the cut)
   tb    : a tb cut must write the marker ids exactly, then the adapter continues
   hard  : a hard cut must write "\\n</think>\\n\\n" exactly
2) vLLM stopped (the GPU is needed), the same adapter in HF + PEFT, exactly as training runs it
   (prefix with the base weights, the adapter on from the marker):
     python tb_vllm_check.py hf runs/tb_vllm_check.json --model Qwen/Qwen3.8-27B --adapter adapters/cut-grpo
   For vLLM's tokens after the marker, compares HF's log-probs and greedy choices. As a control it also scores
   them with the base model (no adapter): if vLLM really switched the adapter on at the marker, it agrees
   with the aLoRA run and clearly less with the base run. (Step 1's mask check covers "off before the marker".)
"""
import argparse
import asyncio
import json
import os
import random
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INVOKE = "\n\n[cut]\n"
CLOSE = "\n</think>\n\n"


async def vllm_step(args):
    import httpx
    from cutlib import Server, find_think_open, messages_for, read_traces
    srv = Server(args.base_url, args.model, 4)
    c = httpx.AsyncClient(base_url=args.base_url, timeout=None)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rows = [r for r in read_traces(args.traces) if r.get("finished") and r.get("thinking")]
    random.Random(0).shuffle(rows)
    rows = rows[: args.n]
    think_open = await find_think_open(srv, messages_for(rows[0]), {"enable_thinking": True}, verbose=True)
    inv, close = await srv.tok(INVOKE), await srv.tok(CLOSE)
    print("marker ids", inv, " close ids", close)
    tag = f"chk{random.randrange(10**6):06d}"

    async def gen(prompt, model, n, rid=None, cut=None):
        if cut:
            udp.sendto(f"{rid} {cut}".encode(), ("127.0.0.1", args.tb_port))
            await asyncio.sleep(0.05)
        body = {"model": model, "prompt": prompt, "max_tokens": n, "temperature": 0, "seed": 0,
                "logprobs": 1, "return_tokens_as_token_ids": True, "skip_special_tokens": False}
        if rid:
            body["vllm_xargs"] = {"tb_rid": rid}
        d = (await c.post("/v1/completions", json=body)).json()
        if "choices" not in d:
            raise SystemExit(f"vLLM answered: {json.dumps(d)[:800]}")
        lp = d["choices"][0]["logprobs"]
        ids = [int(t.split(":", 1)[1]) for t in lp["tokens"]]
        return ids, lp["token_logprobs"]

    out, ok = [], {"mask": 0, "tb": 0, "hard": 0}
    for k, r in enumerate(rows):
        base = await srv.tok_chat(messages_for(r), {"enable_thinking": True})
        base += await srv.tok(think_open) if think_open else []
        th = await srv.tok(r["thinking"])
        u = max(256, int(len(th) * args.frac))
        prefix = base + th[:u]
        a, _ = await gen(prefix, args.model, 48)
        b, _ = await gen(prefix, args.adapter, 48, rid=f"{tag}-m{k}")
        t, tlp = await gen(prefix, args.adapter, args.n_after, rid=f"{tag}-t{k}", cut="tb")
        h, _ = await gen(prefix, args.model, 8, rid=f"{tag}-h{k}", cut="hard")
        m_ok, t_ok, h_ok = a == b, t[: len(inv)] == inv, h[: len(close)] == close
        ok["mask"] += m_ok
        ok["tb"] += t_ok
        ok["hard"] += h_ok
        first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        print(f"{r['id']:>16s} u={u:5d}  mask {'ok' if m_ok else f'DIFF at {first_diff}'}  "
              f"tb marker {'ok' if t_ok else 'BAD ' + str(t[:8])}  hard {'ok' if h_ok else 'BAD ' + str(h[:6])}")
        print("    after the cut:", repr((await srv.detok(t[len(inv):]))[:160]))
        out.append({"id": r["id"], "prefix": prefix, "tb_ids": t, "tb_logprobs": tlp, "n_invoke": len(inv)})
    n = len(rows)
    print(f"\nmask {ok['mask']}/{n}   tb marker {ok['tb']}/{n}   hard {ok['hard']}/{n}")
    if ok["mask"] < n:
        print("A mask DIFF can also be bf16 batch noise if it starts late; run the base twice to compare. Send me the output.")
    json.dump(out, open(args.out, "w"))
    print(f"-> {args.out}  (now stop vLLM and run the hf step)")
    await c.aclose()
    await srv.close()


def hf_step(args):
    import torch
    from transformers import DynamicCache
    from cut_alora import load, prefill
    tok, model, invoke_ids = load(args.model, args.adapter)
    model.eval()
    dev = next(model.parameters()).device
    rows = json.load(open(args.saved))
    tot = {"alora_lp": [], "base_lp": [], "alora_agree": [], "base_agree": []}
    for r in rows:
        ids, vlp, ni = r["tb_ids"], r["tb_logprobs"], r["n_invoke"]
        if ids[:ni] != invoke_ids:
            print(f"{r['id']}: vLLM marker {ids[:ni]} != training marker {invoke_ids}; skipped")
            continue
        n = len(r["prefix"])
        with torch.no_grad():
            cache = DynamicCache()                                          # as in cut_alora training
            prefill(model, cache, r["prefix"], dev)                       # base weights, as in training
            lg = model(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True).logits[0].float()
            a_lp = torch.log_softmax(lg, -1)                               # adapter on from the marker
            with model.disable_adapter():                                  # control: no adapter at all
                full = torch.tensor([r["prefix"] + ids], device=dev)
                lg2 = model(input_ids=full).logits[0, n:].float()
            l_lp = torch.log_softmax(lg2, -1)
        # position j predicts ids[j+1]; score the tokens after the marker
        js = range(ni - 1, len(ids) - 1)
        tgt = torch.tensor([ids[j + 1] for j in js], device=dev)
        v = torch.tensor([vlp[j + 1] for j in js], device=dev)
        for name, lp in (("alora", a_lp), ("base", l_lp)):
            pick = lp[list(js)].gather(1, tgt[:, None])[:, 0]
            tot[f"{name}_lp"].append(float((pick - v).abs().mean()))
            tot[f"{name}_agree"].append(float((lp[list(js)].argmax(-1) == tgt).float().mean()))
        print(f"{r['id']:>16s}  |logp diff| aLoRA {tot['alora_lp'][-1]:.3f}  base {tot['base_lp'][-1]:.3f}   "
              f"greedy agree aLoRA {tot['alora_agree'][-1]:.1%}  base {tot['base_agree'][-1]:.1%}")
    m = lambda xs: sum(xs) / max(len(xs), 1)
    print(f"\nmean |logp diff| vs vLLM: aLoRA {m(tot['alora_lp']):.3f}   base {m(tot['base_lp']):.3f}")
    print(f"greedy agreement with vLLM: aLoRA {m(tot['alora_agree']):.1%}   base {m(tot['base_agree']):.1%}")
    print("PASS if aLoRA agreement is high (bf16: ~95%+) and clearly above the base model. Send me the output either way.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vllm")
    v.add_argument("traces")
    v.add_argument("--n", type=int, default=8)
    v.add_argument("--frac", type=float, default=0.4, help="cut after this share of the trace's thinking")
    v.add_argument("--n-after", type=int, default=160)
    v.add_argument("--out", default="runs/tb_vllm_check.json")
    v.add_argument("--base-url", default="http://localhost:8000")
    v.add_argument("--model", default="Qwen/Qwen3.8-27B")
    v.add_argument("--adapter", default="cut")
    v.add_argument("--tb-port", type=int, default=5599)
    h = sub.add_parser("hf")
    h.add_argument("saved")
    h.add_argument("--model", default="Qwen/Qwen3.8-27B")
    h.add_argument("--adapter", required=True)
    args = ap.parse_args()
    if args.cmd == "vllm":
        asyncio.run(vllm_step(args))
    else:
        hf_step(args)


if __name__ == "__main__":
    main()
