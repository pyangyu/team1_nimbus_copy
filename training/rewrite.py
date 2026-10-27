# Asks a bigger model to compress the rest of each cut trace so it fits the budget W (targets for make_sft.py).
#   python rewrite.py --model Qwen/Qwen3.6-27B --fp8 --inp rewrite_input.jsonl --out rewrite_output.jsonl
#   python rewrite.py --dry-run --inp rewrite_input.jsonl     (prints one prompt, no GPU)
# rewrite_input.jsonl is written by make_sft.py. A 27B model in fp8 fits one 80 GB GPU.
import argparse
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("rewrite_output.jsonl"))
    ap.add_argument("--prompt", type=Path, default=Path(__file__).with_name("rewrite_prompt.txt"))
    ap.add_argument("--n", type=int, default=2, help="samples per example")
    ap.add_argument("--temp", type=float, default=0.4)
    ap.add_argument("--prefix-chars", type=int, default=6000, help="how much of the reasoning so far to show")
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--fp8", action="store_true", help="fp8 weights on the fly (H100): half the memory, more room for KV")
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--words-per-token", type=float, default=0.5,
                    help="word limit in the prompt = W x this (math notation costs ~1.5 tokens per word)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tmpl = a.prompt.read_text()
    rows = [json.loads(l) for l in a.inp.open()][:a.limit]

    def make_prompt(r):
        s = tmpl
        for k, v in (("[[NOTICE]]", r["notice"].strip()), ("[[WORDS]]", str(int(r["budget_tokens"] * a.words_per_token))),
                     ("[[PROBLEM]]", r["problem"]), ("[[PREFIX]]", r["prefix"][-a.prefix_chars:]),
                     ("[[REST]]", r["original_rest"])):
            s = s.replace(k, v)
        return s

    if a.dry_run:
        print(make_prompt(rows[0]))
        return

    from vllm import LLM, SamplingParams

    kw = {"quantization": "fp8"} if a.fp8 else {}
    try:                                   # text only: skip the vision encoder of multimodal checkpoints
        llm = LLM(model=a.model, max_model_len=a.max_len, tensor_parallel_size=a.tp, gpu_memory_utilization=0.9,
                  enable_prefix_caching=True, max_num_seqs=a.max_num_seqs,
                  limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    except TypeError:
        llm = LLM(model=a.model, max_model_len=a.max_len, tensor_parallel_size=a.tp, gpu_memory_utilization=0.9,
                  enable_prefix_caching=True, max_num_seqs=a.max_num_seqs, **kw)
    tok = llm.get_tokenizer()

    def chat(p):
        msgs = [{"role": "user", "content": p}]
        try:  # we don't want it thinking before it rewrites
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompts = [chat(make_prompt(r)) for r in rows]
    # room for ~4x the budget, too-long ones get dropped later anyway
    params = [SamplingParams(n=a.n, temperature=a.temp, top_p=0.95,
                             max_tokens=min(4 * r["budget_tokens"] + 256, 8192)) for r in rows]
    outs = llm.generate(prompts, params)

    with a.out.open("w") as f:
        for r, o in zip(rows, outs):
            for k, c in enumerate(o.outputs):
                text = re.sub(r"<think>.*?</think>", "", c.text, flags=re.S).strip()
                f.write(json.dumps({"id": r["id"], "sample": k, "rewrite": text, "finish": c.finish_reason,
                                    "out_tokens": len(c.token_ids), "model": a.model}) + "\n")
    print(f"wrote {a.out} ({len(rows)} examples x {a.n})")


if __name__ == "__main__":      # vLLM may start its engine with spawn, which re-imports this file
    main()
