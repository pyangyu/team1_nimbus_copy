#!/usr/bin/env python3
"""Merge a LoRA adapter into Qwen3-4B and save a normal model folder that vLLM and aime_drop_bench.py can load.

  python merge_lora.py --adapter r1_adapter --out r1_merged
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-4B")
ap.add_argument("--adapter", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

new_tf = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56)
kw = {"dtype": torch.bfloat16} if new_tf else {"torch_dtype": torch.bfloat16}
m = AutoModelForCausalLM.from_pretrained(a.model, **kw)
m = PeftModel.from_pretrained(m, a.adapter).merge_and_unload()
m.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
print(f"merged {a.adapter} -> {a.out}")
