# Training: cut-aware thinking for Qwen3-4B

This folder is the training side of Thinking Brownout. The question we wanted to answer: when the serving
system has to cut a request's thinking budget in the middle of a thought, can we tell the model about it and
have it wrap up well, instead of just chopping the thinking off?

The notice we insert at the cut is:

> I only have about {W} tokens of thinking left, so I must finish my reasoning quickly and move to the final answer.

Short version of what we found. The untrained model handles this notice badly, worse than a silent cut. SFT
fixes that. SFT plus a short GRPO run is the best we got: a little above the silent cut with less thinking,
but with 30 AIME problems the gap is still inside the noise.

## How the test works

- AIME 2026, 30 problems, 8 samples each. It was held after Qwen3 came out, so it can't be in the training data.
- Qwen3-4B first thinks with no limit. 232 of the 240 traces finish thinking inside the context window, and
  those are the ones we test on.
- Each trace is cut at 50% of its natural length. From there the model gets W more thinking tokens, with
  W = 25% or 10% of the natural length. After that the thinking is closed and it has to answer.
- Arms:
  - `silent`: no notice, thinking is just closed after W tokens
  - `countdown notice` (`warn_count` in the code): the line above, with the real W
  - `soft notice` (`warn_soft`): "Wait, I'm running out of time..." with no number
  - `cut_now`: thinking is closed right at the cut
- Only the final answer may come after `</think>`: we prefill `The final answer is \boxed{` and allow a few
  tokens. If you let it write a full answer, the model simply keeps reasoning in the answer part (about 1,300
  tokens on average), so the cut doesn't really cut anything.
- Every model runs on the same 232 base traces with the same cut points, so all comparisons are paired.
  95% CIs are bootstrapped over problems, so all 8 samples of a problem move together.

## Results

Accuracy (%) on AIME 2026 after the cut. "Thinking used" is all thinking as a share of the natural trace.

| Model | Arm | W = 25% | W = 10% | Thinking used (25% / 10%) |
|---|---|---|---|---|
| Qwen3-4B | no cut | 64.7 | 64.7 | 1.00 |
| Qwen3-4B | cut_now | 31.5 | 31.5 | 0.50 |
| Qwen3-4B | silent | 54.3 | 42.7 | 0.75 / 0.60 |
| Qwen3-4B | countdown notice | 41.4 | 36.6 | 0.57 / 0.53 |
| Qwen3-4B | soft notice | 46.6 | 39.7 | 0.64 / 0.56 |
| SFT | countdown notice | 53.0 | 45.3 | 0.72 / 0.60 |
| SFT on 27B rewrites | countdown notice | 45.3 | 44.0 | 0.62 / 0.57 |
| SFT + GRPO | countdown notice | **56.0** | **46.1** | 0.67 / 0.58 |

Paired differences against the untrained model's silent cut, in percentage points with 95% CI:

| | W = 25% | W = 10% |
|---|---|---|
| SFT | -1.3 [-5.2, +2.6] | +2.6 [-0.9, +6.1] |
| SFT on 27B rewrites | -9.1 [-15.0, -3.5] | +1.3 [-5.5, +7.9] |
| SFT + GRPO | +1.7 [-2.2, +5.7] | +3.4 [-0.4, +8.1] |
| SFT + GRPO vs SFT, same H100 | +0.4 [-3.0, +3.9] | +2.2 [-1.7, +6.5] |

With the countdown notice on both sides, SFT beats the untrained model by +11.6 [+6.4, +17.2] / +8.6
[+4.6, +13.0], and SFT + GRPO by +14.7 [+9.6, +20.2] / +9.5 [+5.0, +14.5].

A caveat about hardware. Everything ran on an RTX 4090 except the GRPO evals, which ran on the H100 used for
training. The same SFT model scored 55.6 instead of 53.0 at W = 25% on the H100, just from different kernels
and sampling noise. So the "vs SFT, same H100" row is the fair one for GRPO, and the GRPO vs silent numbers
carry a couple of points of extra noise.

## What we learned

1. **The untrained model panics.** With the countdown notice it closes the thinking almost right away in 76% of
   traces (median use of the window: 1%), and ends up 13 points below saying nothing at all. On MATH-500
   levels 4-5 you barely see this (96.0 vs 97.1), since those problems are too easy. That's why we moved to AIME.
2. **Our first SFT try copied the panic.** In many examples the answer already shows up in the thinking before
   the cut, and there the shortest correct continuation is basically "ok, done". 76% of the cut examples in the first dataset were
   like that, with a median target of 15 tokens, and the model learned to stop right after any notice
   (-12.5 vs silent). The fix: every target must use at least half of W, and examples where the answer is
   already seen are capped at 30%. That's the SFT row above. The first try is in `results/sft_first_try`.
3. **SFT removes the panic but only ties the silent cut.** It now uses the window: it closes early in 34% / 11%
   of cases instead of 76% / 69%. With full answers allowed it also stopped moving reasoning into the answer,
   about 35 answer tokens instead of about 1,300. When it writes its own traces, the way it would run in
   serving, notice vs silent is +1.3 [-3.2, +6.0] / +0.4 [-3.4, +4.3], and its no-cut accuracy stays about the
   same (63.4 vs 64.7).
4. **More samples didn't help.** Six extra no-notice samples for every example that still had no target, to
   find shorter paths that fit in W, gave basically the same model: -2.2 / +0.4 vs silent (`results/sft_extra_samples`).
5. **Compressed targets hurt.** We had Qwen3.6-27B rewrite the rest of each trace so it fits in W. The 4B learned
   "be short" too well. It closes early in 76% of cases, uses about a third of the window, and lost 9 points
   at W = 25%.
6. **GRPO helped a bit.** The reward is just "final answer correct". We started from the SFT adapter, kept only
   prompts where the answer isn't in the prefix yet, and halved W, because with the normal W the model already
   got 99% of the training prompts right and there was nothing to learn. Training accuracy went up for about 20
   steps and then stayed flat while the model kept closing earlier, so we stopped at 40 steps. The result
   closes on its own in 62% of cases at W = 25%, uses 0.67 of the natural length instead of 0.75, and is a bit
   more accurate than silent. None of the gains vs silent are significant with 30 problems though.

## Running it

You need Linux (WSL2 works), Python 3.10+, one 24 GB GPU for most of it, and an 80 GB GPU for the 27B rewrites
and GRPO. We used vLLM 0.31/0.32.

```bash
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install vllm peft datasets math-verify
```

Outputs go to `$WORK` (default `~/brownout_train`). The other settings are at the top of `scripts/env.sh`.

```bash
bash scripts/baselines.sh all        # untrained model: MATH-500 L4-5, AIME with full answers, AIME final-only
bash scripts/sft.sh                  # MATH train traces -> SFT data -> LoRA -> AIME eval
bash scripts/eval_more.sh            # leakage and own-trace checks for the SFT model

# optional: SFT on 27B rewrites
bash scripts/rewrite_27b.sh rewrite_input.jsonl             # on the big GPU, input is $WORK/sft/rewrite_input.jsonl
REWRITES=/path/to/rewrite_output.jsonl bash scripts/sft_rewrite.sh

# GRPO
bash scripts/grpo_pack.sh            # makes $WORK/grpo_pack.zip, copy it to the big GPU
bash grpo_run.sh                     # on the big GPU, inside grpo_pack/
```

All of these pick up where they stopped if you run them again. The runs are long, so start them with
`nohup ... &` or inside tmux. To check the pipeline without a GPU, most of the Python scripts take `--mock`.

## Files

| File | What it does |
|---|---|
| `thinking_budget_bench.py` | MATH-500 natural traces and a fixed-budget sweep; shared prompt and grading code |
| `budget_drop_bench.py` | the budget-drop test on MATH-500 traces; the notice texts live here (`WARNS`) |
| `aime_drop_bench.py` | the same test on AIME 2026, 8 samples per problem; every eval above uses it |
| `compare_tuned.py` | paired comparison of two `aime_drop_bench.py` runs on the same traces |
| `gen_traces.py` | natural traces on MATH train, levels 3-5, the raw material for SFT |
| `make_sft.py` | picks cut points and finds a good continuation for each one by rejection sampling |
| `train_lora.py` | LoRA SFT, then merges the adapter into a normal model folder for vLLM |
| `rewrite.py`, `rewrite_prompt.txt` | the 27B rewrites |
| `prep_grpo.py` | GRPO prompts from the SFT cut examples |
| `grpo.py` | GRPO: vLLM with LoRA for rollouts, HF + PEFT for the updates |
| `scripts/` | the run scripts above |
| `results/` | summaries and comparisons for every row in the tables |

## Training details

SFT data:
- 2,000 MATH train problems (levels 3-5, minus anything in MATH-500). 1,854 traces finish thinking and 94% of
  those are correct.
- Two cuts per correct trace: the cut point is uniform in 30-70% of the thinking, moved to the next line break,
  and W is uniform in 20-60% of what's left.
- For each cut we sample continuations from the 4B itself, with no notice, with the soft notice, or with a
  "keep going, skip the double-checks" notice. One is kept only if it closes the thinking within W, the 4B then
  gets the right answer, and it uses at least half of W. We keep the longest one that passes.
- 20% of the final set are normal traces with no cut, so normal thinking doesn't drift.
- The final set is 746 examples. Most cuts have no passing continuation, mostly because the model simply
  doesn't finish within W.

LoRA: rank 16, alpha 32, all linear layers, lr 1e-4, 2 epochs, loss only on the part after the notice.

GRPO: 8 samples per prompt, advantage = (reward - group mean) / group std, groups where every sample gets the
same reward are skipped, no KL term, lr 2e-5. 12 groups per step, W halved, only prompts where the answer
isn't in the prefix yet. The training log is `results/grpo/train_log.jsonl`. It goes to step 43 because we
stopped it by hand; the model we tested is the step-40 checkpoint.

## Things that bit us

- If the GPU also drives your screen, vLLM may refuse to start because of free memory. Set `GPU_UTIL=0.85`.
- vLLM can keep the process alive after a script is done. `make_sft.py` shuts its engine down itself now. If
  something still sits there, check `nvidia-smi` and kill the leftover `EngineCore`.
- Don't start the same script twice on one GPU. Check `nvidia-smi` first.
- On fresh Ubuntu cloud images, Triton needs `python3-dev` and `build-essential`. `grpo_run.sh` installs them.
- The GRPO update ran out of memory on 80 GB until we added micro-batches (`--micro 2`) and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Qwen3.6-27B is a multimodal model. `rewrite.py` loads it text-only, and its main code has to sit under
  `if __name__ == "__main__":` because vLLM spawns worker processes.

## Not in the repo

Traces, SFT data, adapters and merged models are several GB, so they're not in git. Ask Matthew if you need the
adapters.
