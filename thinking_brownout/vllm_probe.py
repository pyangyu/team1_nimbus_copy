"""Read-only look at the installed vLLM, to write the in-engine aLoRA patch against the exact version.
No GPU, loads no weights, changes nothing. Run inside the vLLM env and send me the whole output:

    python vllm_probe.py --model Qwen/Qwen3.8-27B --adapter adapters/cut-r16  > probe.txt 2>&1
"""
import argparse
import importlib
import inspect
import json
import os
import re


def show(title, fn):
    print(f"\n===== {title}")
    try:
        fn()
    except Exception as e:
        print(f"!! {type(e).__name__}: {e}")


def src(obj, max_lines=80):
    lines = inspect.getsource(obj).splitlines()
    print(f"# {inspect.getsourcefile(obj)}:{inspect.getsourcelines(obj)[1]}")
    print("\n".join(lines[:max_lines]) + ("\n# ... (cut)" if len(lines) > max_lines else ""))


def grep(module_name, pattern, ctx=2):
    m = importlib.import_module(module_name)
    path = inspect.getsourcefile(m)
    lines = open(path).read().splitlines()
    print(f"# {path}")
    for i, l in enumerate(lines):
        if re.search(pattern, l):
            print(f"{i + 1}: " + "\n    ".join(lines[max(0, i - ctx): i + ctx + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--adapter", default="adapters/cut-r16")
    a = ap.parse_args()
    import vllm
    print("vllm", vllm.__version__)
    import torch
    print("torch", torch.__version__)

    def model_class():
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(a.model)
        archs = getattr(cfg, "architectures", None)
        print("architectures:", archs)
        from vllm.model_executor.models import ModelRegistry
        for arch in archs or []:
            cls = ModelRegistry._try_load_model_cls(arch) if hasattr(ModelRegistry, "_try_load_model_cls") else None
            print(arch, "->", cls)
            if cls is None:
                continue
            from vllm.model_executor.models.interfaces import supports_lora
            print("supports_lora:", supports_lora(cls))
            for k in ("packed_modules_mapping", "embedding_modules", "supported_lora_modules"):
                print(k, "=", getattr(cls, k, None))
            print("module source:", inspect.getsourcefile(cls))
    show("model class and LoRA support", model_class)

    def adapter():
        cfg = json.load(open(os.path.join(a.adapter, "adapter_config.json")))
        print({k: cfg.get(k) for k in ("peft_type", "r", "lora_alpha", "target_modules", "alora_invocation_tokens",
                                       "modules_to_save", "base_model_name_or_path")})
        from safetensors import safe_open
        f = safe_open(os.path.join(a.adapter, "adapter_model.safetensors"), "pt")
        names = sorted({re.sub(r"\.\d+\.", ".N.", k) for k in f.keys()})
        print(len(list(f.keys())), "tensors; distinct names:")
        print("\n".join(names[:60]))
    show("adapter", adapter)

    show("v1 logits processor API", lambda: [
        src(importlib.import_module("vllm.v1.sample.logits_processor").LogitsProcessor, 60),
        src(importlib.import_module("vllm.v1.sample.logits_processor").BatchUpdate, 60)])
    show("InputBatch.make_lora_inputs", lambda: src(importlib.import_module("vllm.v1.worker.gpu_input_batch").InputBatch.make_lora_inputs))
    show("CachedRequestState", lambda: src(importlib.import_module("vllm.v1.worker.gpu_input_batch").CachedRequestState, 40))
    show("LoRA calls in the GPU model runner", lambda: grep("vllm.v1.worker.gpu_model_runner", r"make_lora_inputs|set_active_loras|_prepare_inputs\(|lora_mapping"))
    show("LoRA mixin", lambda: src(importlib.import_module("vllm.v1.worker.lora_model_runner_mixin").LoRAModelRunnerMixin, 120))
    show("LoRAMapping", lambda: src(importlib.import_module("vllm.lora.layers").LoRAMapping if hasattr(importlib.import_module("vllm.lora.layers"), "LoRAMapping") else importlib.import_module("vllm.lora.layers.utils").LoRAMapping, 30))
    show("lora extra hash keys (prefix cache)", lambda: grep("vllm.v1.core.kv_cache_utils", r"lora", 3))
    show("vllm_xargs / extra_args", lambda: grep("vllm.entrypoints.openai.protocol", r"vllm_xargs|extra_args", 1))
    show("hybrid / mamba prefix caching flags", lambda: grep("vllm.config.cache", r"mamba|prefix_caching", 1))


if __name__ == "__main__":
    main()
