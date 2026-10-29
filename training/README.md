# Training: cut-aware thinking for Qwen3-4B

This folder is the training side of Thinking Brownout. The question we wanted to answer: when the serving
system has to cut a request's thinking budget in the middle of a thought, can we tell the model about it and
have it wrap up well, instead of just chopping the thinking off?

The notice we insert at the cut is:

> I only have about {W} tokens of thinking left, so I must finish my reasoning quickly and move to the final answer.

Short version of what we found, over AIME 2026 plus four more contests (176 problems):
- The untrained model handles the notice badly: it stops thinking almost right away and loses 8.7 points
  against a silent cut. This happens on all five contests.
- After SFT + GRPO the notice helps instead: +2.0 against the same model's silent cut, and +6.6 against the
  untrained model's silent cut. Both CIs are above zero.
- A second GRPO round on harder problems kept the accuracy and made the model stop earlier: about 17% less
  thinking than a silent cut. For brownout that means the request finishes and frees its KV cache sooner.

The first sections below are the first round, measured on AIME only. "Round 2" further down has the
five-contest numbers.

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

## Round 2: harder GRPO and four more contests

Two things changed. GRPO ran again, starting from the first GRPO adapter, on competition problems
(DeepScaleR) instead of MATH train, with the normal W instead of half of it. And every model was tested on
four more contests next to AIME 2026: HMMT Feb 2026, HMMT Nov 2025, BRUMO 2025 and SMT 2025. That's 176
problems and 1,372 cut traces. Everything in this section ran on one H100, so all models share the same
hardware.

Accuracy (%) pooled over the five contests, 25% / 10% of the natural thinking left after the cut:

| Model | Arm | W = 25% | W = 10% | Thinking used (25% / 10%) | Closes on its own (25%) |
|---|---|---|---|---|---|
| Qwen3-4B | silent | 40.5 | 32.2 | 0.75 / 0.60 | |
| Qwen3-4B | countdown notice | 31.8 | 28.1 | 0.55 / 0.53 | 82% |
| GRPO round 1 | silent | 45.1 | 36.3 | 0.75 / 0.60 | |
| GRPO round 1 | countdown notice | 47.1 | 38.6 | 0.67 / 0.59 | 60% |
| GRPO round 2 | silent | 45.1 | 36.2 | 0.75 / 0.60 | |
| GRPO round 2 | countdown notice | 47.0 | 40.0 | 0.62 / 0.57 | 79% |

Paired differences in percentage points, 95% CI bootstrapped over problems (`results/round2/pooled.txt`, made
by `pool_evals.py`):

| | W = 25% | W = 10% |
|---|---|---|
| Untrained: notice vs its own silent cut | -8.7 [-11.2, -6.4] | -4.1 [-6.0, -2.2] |
| Round 1: notice vs its own silent cut | +2.0 [+0.1, +3.9] | +2.3 [+0.4, +4.2] |
| Round 2: notice vs its own silent cut | +1.9 [-0.2, +4.1] | +3.9 [+1.9, +5.9] |
| Round 1 + notice vs untrained silent cut | +6.6 [+3.8, +9.5] | +6.3 [+3.6, +9.5] |
| Round 2 + notice vs untrained silent cut | +6.5 [+3.6, +9.7] | +7.8 [+5.0, +10.9] |
| Round 1 silent vs untrained silent cut | +4.6 [+2.4, +7.1] | +4.1 [+2.0, +6.5] |
| Round 2 vs round 1, both with the notice | -0.1 [-1.7, +1.5] | +1.5 [+0.0, +3.0] |

What this says:
1. **The panic is real everywhere.** Before training, the notice hurts on all five contests.
2. **Training flips it.** After training the notice helps on all five contests, by about 2 points on its own.
3. **The 6-8 points against the untrained silent cut have two parts.** About 4-5 come from the trained model
   answering better after any cut, with or without the notice (the "round 1 silent" row). The other 2-4 come
   from the notice. A paper should report both rows, not just the big one.
4. **Round 2 bought speed, not accuracy.** Same accuracy as round 1, but at W = 25% it closes on its own 79%
   of the time instead of 60%, and the median trace uses 39% of the leftover budget instead of 83%. Total
   thinking drops to 0.62 of the natural length, against 0.75 for a silent cut. The GRPO log shows the same
   thing: rollout accuracy stayed at about 0.87 for all 100 steps while closing on its own rose from 0.49 to
   0.69. The reward is only "final answer correct", so it found that stopping early often costs nothing.
5. **Earlier AIME-only numbers were too pessimistic.** Round 1 is +2.6 on AIME but +6.6 pooled; with 30
   problems you can't tell those apart.

BRUMO 2025 and SMT 2025 were held in spring 2025, close to Qwen3's release, so they could be in its training
data. On the three contests held later (AIME 2026, HMMT Feb 2026, HMMT Nov 2025; 93 problems) the picture is
the same but smaller: untrained notice vs silent -9.3 [-12.3, -6.5], round 1 notice vs its own silent +2.2
[-0.1, +4.5] / +2.4 [+0.4, +4.4], round 1 notice vs untrained silent +3.7 [+0.8, +6.8] / +4.0 [+1.8, +6.4].
Most of the difference is in the "answers better after any cut" part, which is +1.5 there instead of +4.6.

The training data for round 2: 1,600 DeepScaleR problems, 4 traces each (88% finish thinking, 83% of those
right, median 5,471 thinking tokens). Problems the model never solved were dropped, since after a cut they
are wrong anyway and give no learning signal. That left 1,278 problems and 2,312 prompts after the length
and "answer already in the prefix" filters. One thing we saw on the way: the longer a trace, the more often it
is wrong (in the first 5,120 traces: 94% right under 3k tokens, 61% over 10k). The requests brownout most wants to cut are also the
hardest ones.

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

# round 2: harder GRPO + five-contest eval, on one 80 GB GPU (about 15 h on an H100)
# put round2_run.sh, the .py files, r1_adapter/ and aime_drop/phase_a.jsonl in one folder, then
nohup bash round2_run.sh > log.txt 2>&1 &
python pool_evals.py --evals evals --out pooled
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
| `pool_evals.py` | the same comparisons pooled over several contests |
| `gen_traces.py` | natural traces on MATH train or any problem/answer dataset, one or several per problem |
| `make_sft.py` | picks cut points and finds a good continuation for each one by rejection sampling |
| `train_lora.py` | LoRA SFT, then merges the adapter into a normal model folder for vLLM |
| `rewrite.py`, `rewrite_prompt.txt` | the 27B rewrites |
| `prep_grpo.py` | GRPO prompts, from the SFT cut examples or drawn from multi-sample traces |
| `grpo.py` | GRPO: vLLM with LoRA for rollouts, HF + PEFT for the updates |
| `merge_lora.py` | merges an adapter into Qwen3-4B so vLLM can load it as a normal model |
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

GRPO round 2: same settings, but from the round 1 adapter, on the DeepScaleR prompts with the normal W,
8 groups per step out of 16 sampled prompts, prompts up to 10,240 tokens, 100 steps (about 2 h on an H100).
Log: `results/round2/grpo_train_log.jsonl`.

## Things that bit us

- If the GPU also drives your screen, vLLM may refuse to start because of free memory. Set `GPU_UTIL=0.85`.
- vLLM can keep the process alive after a script is done. `make_sft.py` shuts its engine down itself now. If
  something still sits there, check `nvidia-smi` and kill the leftover `EngineCore`.
- Don't start the same script twice on one GPU. Check `nvidia-smi` first.
- On fresh Ubuntu cloud images, Triton needs `python3-dev` and `build-essential`. `grpo_run.sh` installs them.
- The GRPO update ran out of memory on 80 GB until we added micro-batches (`--micro 2`) and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Long, hard traces are slow even on an H100: about 3,000 tokens/s at 128 sequences, because every decode step
  reads a lot of KV cache. Plan on hours for a few thousand competition traces.
- Qwen3.6-27B is a multimodal model. `rewrite.py` loads it text-only, and its main code has to sit under
  `if __name__ == "__main__":` because vLLM spawns worker processes.

## Not in the repo

Traces, SFT data, adapters and merged models are several GB, so they're not in git. Ask Matthew if you need the
adapters.
