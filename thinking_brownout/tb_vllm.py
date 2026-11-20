"""Thinking Brownout inside vLLM (tested against vLLM 0.28): an in-place cut with an activated LoRA.

Load it as a logits processor; importing the module also installs the aLoRA mask:

    PYTHONPATH=~/thinkingcut TB_ALORA=cut,cut-sft \\
    vllm serve Qwen/Qwen3.8-27B --reasoning-parser qwen3 --max-num-seqs 64 \\
        --enable-lora --max-lora-rank 16 --max-loras 2 \\
        --lora-modules cut=adapters/cut-grpo cut-sft=adapters/cut-sft \\
        --logits-processors tb_vllm:TBCut

How a request is cut (no abort, no second request, no prefill again):
  * A Thinking Brownout request is sent with model=<cut adapter> and vllm_xargs={"tb_rid": "<id>"}.
    The adapter is attached from the start but MASKED: until the request is cut, every one of its tokens runs
    with the base weights, exactly like a base request (prompt, thinking, KV cache, DeltaNet state).
  * The controller sends "<id> tb" (UDP, 127.0.0.1:$TB_PORT). On its next step this processor forces the
    marker tokens "\\n\\n[cut]\\n" one per step. From the first marker token on, the request's tokens use the
    adapter: the aLoRA rule the adapter was trained with (PEFT alora_invocation_tokens). Everything before
    the marker keeps its base-weight KV and recurrent state, in place.
  * If the adapter has not closed the thinking after $TB_KAPPA tokens, "\\n</think>\\n\\n" is forced.
  * "<id> hard" forces "\\n</think>\\n\\n" at once (the hard stop; also used by the budget and early-exit
    baselines), on a base or an adapter request alike, so every policy cuts the same way, in place.
A cut that arrives after the request has closed its thinking is ignored.

Env: TB_PORT (5599), TB_KAPPA (2048), TB_ALORA (comma-separated LoRA names that are activated LoRAs; every
other LoRA is served as a normal LoRA), TB_LOG (1 = log each cut).
"""
import os
import queue
import socket
import threading
import time

import numpy as np
import torch

INVOKE = "\n\n[cut]\n"            # must match cut_alora.INVOKE
CLOSE = "\n</think>\n\n"
PORT = int(os.environ.get("TB_PORT", "5599"))
KAPPA = int(os.environ.get("TB_KAPPA", "2048"))
ALORA_NAMES = {s.strip() for s in os.environ.get("TB_ALORA", "cut").split(",") if s.strip()}
LOG = os.environ.get("TB_LOG", "1") == "1"

STATS = {"cuts": 0}


def log(msg):
    if LOG:
        print(f"[tb] {msg}", flush=True)


# ------------------------------------------------------------------ the aLoRA mask
# The whole aLoRA is this: for a request on an activated LoRA, find the marker tokens in what it has
# written; tokens before the marker run with LoRA id 0 (base weights), tokens from the marker on with
# the adapter. Nothing else in vLLM changes.

_MARK = {"ids": None}       # marker token ids, set by the processor (it has the tokenizer)
_FOUND = {}                 # req_id -> (position of the first marker token or None, output tokens scanned)
# The marker is several tokens and decoding writes one per step, so the full marker is only visible after
# its first tokens have already gone through the model. Training runs the adapter on the marker tokens too,
# so the processor, which is the one writing the marker, flags "the marker starts at output k" right away:
_EARLY = {}                 # id(request's output token list) -> output index of the first marker token


def _marker_pos(req_id, out, n_prompt):
    mk = _MARK["ids"]
    pos, scanned = _FOUND.get(req_id, (None, 0))
    if pos is None and mk and len(out) >= len(mk):
        lo = max(0, scanned - len(mk) + 1)
        for k in range(lo, len(out) - len(mk) + 1):
            if out[k] == mk[0] and out[k: k + len(mk)] == mk:
                pos = n_prompt + k
                break
        _FOUND[req_id] = (pos, len(out))
    return pos


def _install_mask():
    from vllm.v1.worker.gpu_input_batch import InputBatch
    if getattr(InputBatch.make_lora_inputs, "_tb", False):
        return
    orig = InputBatch.make_lora_inputs

    def make_lora_inputs(self, num_scheduled_tokens, num_sampled_tokens):
        prompt_map, token_map, reqs = orig(self, num_scheduled_tokens, num_sampled_tokens)
        n = self.num_reqs
        lora = np.asarray(self.request_lora_mapping[:n])
        if n == 0 or not lora.any():
            return prompt_map, token_map, reqs
        alora_ids = {i for i, r in self.lora_id_to_lora_request.items() if r.lora_name in ALORA_NAMES}
        if not alora_ids:
            return prompt_map, token_map, reqs
        nsch, nsam = np.asarray(num_scheduled_tokens), np.asarray(num_sampled_tokens)
        tok, prm = np.array(token_map, dtype=np.int64), np.array(prompt_map, dtype=np.int64)
        tok_off, prm_off = np.cumsum(nsch) - nsch, np.cumsum(nsam) - nsam
        for i in range(n):
            lid = int(lora[i])
            if lid not in alora_ids:
                continue
            out = self.req_output_token_ids[i] or []
            n_prompt = int(self.num_prompt_tokens[i])
            p0 = _marker_pos(self.req_ids[i], out, n_prompt)
            if p0 is None and id(out) in _EARLY:
                p0 = n_prompt + _EARLY[id(out)]           # marker being written right now
            elif p0 is not None:
                _EARLY.pop(id(out), None)
            ns, ms = int(nsch[i]), int(nsam[i])
            pos = int(self.num_computed_tokens_cpu[i]) + np.arange(ns)
            act = np.zeros(ns, dtype=bool) if p0 is None else pos >= p0
            tok[tok_off[i]: tok_off[i] + ns] = np.where(act, lid, 0)
            if ms:
                prm[prm_off[i]: prm_off[i] + ms] = np.where(act[-ms:], lid, 0)
        if len(_FOUND) > 4096:                      # forget finished requests
            live = set(self.req_ids[:n])
            for k in [k for k in _FOUND if k not in live]:
                _FOUND.pop(k)
        return tuple(prm.tolist()), tuple(tok.tolist()), reqs

    make_lora_inputs._tb = True
    InputBatch.make_lora_inputs = make_lora_inputs
    log(f"aLoRA mask installed for LoRAs {sorted(ALORA_NAMES)}; kappa={KAPPA}; port={PORT}")


try:
    _install_mask()
except Exception as e:      # the API-server process may not have the worker modules; the worker will
    print(f"[tb] mask not installed in this process ({e!r})", flush=True)


# ------------------------------------------------------------------ the control channel

_CMDS = queue.Queue()
_LISTENER = {"started": False}


def _listen():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", PORT))
    log(f"listening for cuts on udp 127.0.0.1:{PORT}")
    while True:
        data, _ = s.recvfrom(4096)
        for line in data.decode(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] in ("tb", "hard"):
                _CMDS.put((parts[0], parts[1], time.monotonic()))


def _start_listener():
    if not _LISTENER["started"]:
        _LISTENER["started"] = True
        threading.Thread(target=_listen, daemon=True, name="tb-cut-listener").start()


# ------------------------------------------------------------------ the logits processor

try:
    from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
except Exception:            # keeps the module importable for unit tests without vLLM
    LogitsProcessor, BatchUpdate = object, None


def _tokenizer(vllm_config):
    from transformers import AutoTokenizer
    mc = vllm_config.model_config
    return AutoTokenizer.from_pretrained(getattr(mc, "tokenizer", None) or mc.model,
                                         trust_remote_code=getattr(mc, "trust_remote_code", False))


class TBCut(LogitsProcessor):
    """Forces the marker (or the close) one token per step. A forced token counts as written only once it
    shows up in the request's output list, so a step whose sample is thrown away (a chunk of prefill after a
    preemption) simply forces it again. Run vLLM with --no-async-scheduling so the output list is current."""

    def __init__(self, vllm_config, device, is_pin_memory, tokenizer=None):
        tok = tokenizer or _tokenizer(vllm_config)
        self.invoke = tok.encode(INVOKE, add_special_tokens=False)
        self.close = tok.encode(CLOSE, add_special_tokens=False)
        self.close_id = tok.convert_tokens_to_ids("</think>")
        _MARK["ids"] = list(self.invoke)
        self.req = {}          # batch index -> state
        self.by_rid = {}       # tb_rid -> batch index
        self.parked = {}       # tb_rid -> (state, time): removed from the batch, may come back (preemption)
        self.waiting = {}      # cut commands for rids not in the batch yet: rid -> (mode, time)
        log(f"processor ready: marker ids {self.invoke}, close ids {self.close}, </think> id {self.close_id}")
        _start_listener()

    def is_argmax_invariant(self):
        return False

    # ---- batch bookkeeping
    def _remove(self, idx, park):
        st = self.req.pop(idx, None)
        if st is None:
            return
        if self.by_rid.get(st["rid"]) == idx:
            self.by_rid.pop(st["rid"], None)
        if park:
            st["applied_len"] = None
            self.parked[st["rid"]] = (st, time.monotonic())

    def _place(self, idx, st):
        self.req[idx] = st
        self.by_rid[st["rid"]] = idx

    def update_state(self, batch_update):
        now = time.monotonic()
        if batch_update is not None:
            for idx in batch_update.removed:
                self._remove(idx, park=True)
            for item in batch_update.added:
                idx, params, out = item[0], item[1], item[3]
                self._remove(idx, park=True)
                rid = (getattr(params, "extra_args", None) or {}).get("tb_rid") if params is not None else None
                if rid is None:
                    continue
                rid = str(rid)
                st = self.parked.pop(rid, (None, 0))[0] or {"rid": rid, "mode": None, "force": [],
                                                             "applied_len": None, "act_len": None}
                st["out"] = out
                self._place(idx, st)
            for mv in batch_update.moved:
                a, b, direction = mv[0], mv[1], mv[2]
                sa, sb = self.req.pop(a, None), self.req.pop(b, None)
                if "SWAP" in str(direction).upper():
                    if sa is not None:
                        self._place(b, sa)
                    if sb is not None:
                        self._place(a, sb)
                else:                                   # a moves to b; whatever was at b is gone
                    if sb is not None:
                        if self.by_rid.get(sb["rid"]) == b:
                            self.by_rid.pop(sb["rid"], None)
                    if sa is not None:
                        self._place(b, sa)
            for rid in [r for r, (_, t) in self.parked.items() if now - t > 900]:
                self.parked.pop(rid)
        # was last step's forced token written?
        for st in self.req.values():
            n = st["applied_len"]
            if n is None:
                continue
            st["applied_len"] = None
            out = st["out"]
            if len(out) > n and st["force"] and out[n] == st["force"][0]:
                if st["mode"] == "tb" and len(st["force"]) == len(self.invoke):
                    st["act_len"] = n                   # outputs before the marker (for the kappa count)
                    _EARLY[id(out)] = n                 # adapter on from this token, at the next forward
                st["force"].pop(0)
        # new cut commands
        while True:
            try:
                rid, mode, t = _CMDS.get_nowait()
            except queue.Empty:
                break
            self.waiting[rid] = (mode, t)
        for rid in list(self.waiting):
            mode, t = self.waiting[rid]
            idx = self.by_rid.get(rid)
            if idx is None:
                if now - t > 30:
                    self.waiting.pop(rid)
                continue
            self.waiting.pop(rid)
            st = self.req[idx]
            if st["mode"] is not None or self.close_id in st["out"]:
                continue                                # already cut, or the thinking is over
            st["mode"] = mode
            st["force"] = list(self.invoke if mode == "tb" else self.close)
            STATS["cuts"] += 1
            log(f"cut {rid} mode={mode} after {len(st['out'])} output tokens")
        # backstop: an activated request still thinking after KAPPA tokens
        for st in self.req.values():
            if st["mode"] == "tb" and not st["force"] and st["act_len"] is not None:
                after = st["out"][st["act_len"] + len(self.invoke):]   # thinking written after the marker
                if len(after) >= KAPPA and self.close_id not in after:
                    st["force"] = list(self.close)
                    st["mode"] = "tb-capped"
                    log(f"kappa cap {st['rid']}")

    def apply(self, logits):
        for idx, st in self.req.items():
            if not st["force"] or idx >= logits.shape[0]:
                continue
            t = st["force"][0]
            row = logits[idx]
            row.fill_(float("-inf"))
            row[t] = 0.0
            st["applied_len"] = len(st["out"])
        return logits
