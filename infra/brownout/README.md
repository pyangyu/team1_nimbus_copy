# Brownout on AWS

Three boxes in the team account (826671498662, us-east-1). They talk over private IPs
inside `brownout-internal-sg`, which allows ports 8000 and 9000 only between its members.

| Box | Instance | Runs |
|---|---|---|
| `brownout-agent` | t3.small | the math agent: live tests and `loadgen` |
| `brownout-jev` | g5.xlarge (the old `lifeline-vllm-rt`) | Open-Jev-2B on `127.0.0.1:8791` and `:8792`, two batching servers (its own venv: Open-Jev pins transformers 5.10.2; see `../../jevserve`), and the Brownout gateway on `:9000` |
| `lifeline-vllm-batch` | g5.xlarge | vLLM Qwen3-8B on `:8000` (unchanged, see `../vllm-ec2`) |

The account's G quota (8 vCPU) fits exactly the two GPU boxes. That is why rt was
repurposed, and why `../vllm-ec2/up.sh rt` now refuses to run.

No box has an SSH key. Everything runs through SSM Run Command (`env.sh: ssm_script`),
which prepends `remote/common.sh` to each script. Code ships as a `git archive` of one
commit in `s3://capstone-ci-826671498662/ci/<sha>/src.tgz`. Results go to
`ci/<sha>/results/<tag>/`. The bucket expires `ci/` after 30 days.

## One-time setup (done 2026-10-21, admin profile)

```
AWS_PROFILE=rlwalkeradmin infra/brownout/setup_iam.sh    # role brownout-ec2 + CI policy brownout-e2e
AWS_PROFILE=rlwalkeradmin infra/brownout/setup_boxes.sh  # bucket, SG, agent box, rt -> brownout-jev
```

`remote/setup_jev.sh` is idempotent, and `e2e.sh` runs it every time. The first run on the
Jev box:
- disables the old vLLM service;
- installs torch 2.14 (cu130), transformers 5.10.2, peft, accelerate and flash-linear-attention;
- checks out Open-Jev at a pinned commit;
- downloads the `ZefanCai/Open-Jev-2B` checkpoint and the `Qwen/Qwen3.5-2B` base;
- installs the `open-jev@.service` template. `deploy.sh` (role `jev`) then starts one
  `jevserve/batch_server.py` per port in `JEV_PORTS` from the deployed commit, probes each with a
  10-bucket score question, and starts the gateway with all of their URLs.

For a GSM8K-only sweep: `gh workflow run CI -f suite=full -f hard_dataset=none`.

## Run it

```
infra/brownout/e2e.sh [git-ref]            # default HEAD; must be a commit
BROWNOUT_STOP=1 infra/brownout/e2e.sh      # also stop the boxes this run started
infra/brownout/up.sh / down.sh             # start/stop by hand
```

`e2e.sh` does six steps:
1. Bundles the commit to S3.
2. Starts the three boxes (`up.sh`), waiting for SSM and vLLM.
3. Runs `setup_jev.sh`.
4. Deploys the gateway and classifier on the Jev box, then the agent on the agent box.
5. Runs `agent/tests/test_live.py` and `agent/tests/test_brownout_live.py` on the agent box,
   against the gateway.
6. Collects the gateway stats and logs. Everything lands in `./brownout-results/`.

The policy sweep is controlled by environment variables on the agent box. They are set in
`remote/run_e2e.sh` (defaults shown):

| Variable | Default |
|---|---|
| `BROWNOUT_POLICIES` | `none,reject,uniform,brownout:silent:jev,brownout:announced:jev,brownout:announced:jev-serial,brownout:announced:online,brownout:announced:oracle,brownout:announced:jev-learn` (`policy:notice:predictor[:cut][:f<frac>]`, cut = `proportional` or `ranked`, `f0.5` caps that run's shed fraction (default `BROWNOUT_MAX_FRAC`, 0.9); `skip` runs no sweep on this dataset: `jev` = untrained Jev beside the LLM, `jev-serial` = untrained Jev before the LLM, `online` = the word model, `oracle` = the thinking length the same prompt used in an earlier uncut run (an upper bound; run `none` first), `jev-learn` = trained Jev, skipped until its learner is deployed) |
| `BROWNOUT_N` | `64` |
| `BROWNOUT_RATE` | `4.0` workflows/s (24 at 1/s never queued on one A10G) |
| `BROWNOUT_DATASET` | `gsm8k` (budget 1024, up to 2048) |
| `BROWNOUT_HARD_DATASET` | `math500` (second sweep, budget 4096, up to 6144; empty skips it) |
| `BROWNOUT_HARD_POLICIES` | `none,uniform,brownout:announced:jev,brownout:announced:jev-serial,brownout:announced:online,brownout:announced:oracle` |
| `BROWNOUT_HARD_N`, `BROWNOUT_HARD_RATE` | `32`, `2.0` workflows/s |
| `BROWNOUT_REJECT_WAITING` | `2` (`defer` only) |
| `BROWNOUT_ADMIT_RATE` | `0.25` new workflows/s (`reject`) |
| `BROWNOUT_ADMIT_BURST` | `8` (`reject`) |
| `BROWNOUT_SEEDS` | `0`; comma-separated loadgen seeds, one full sweep per seed (result dirs get `-s<seed>` for seeds other than 0) |
| `BROWNOUT_GATEWAY_ARGS` | extra gateway flags, e.g. `--labels needed` (needed-budget labels by probing; see `../../gateway/README.md`) |
| `BROWNOUT_SKIP_AGENT_LIVE` | set to skip `test_live.py` and run the sweep only |

Each run starts from a reset classifier (word model, Jev examples, stats), so no predictor
benefits from the runs before it. Accuracy and deadline numbers are reported in
`results.md`, not asserted. The test does assert these:
- every think request was observed by the classifier
- no upstream errors
- `none` never cuts or rejects
- the cutting policies cut once there was pressure
- only `reject` rejects, and it does reject when the arrivals exceed the bucket
- Jev answered at least once
- outside `reject`, Jev answered some requests before they finished, so the word model and Jev
  are scored on the same requests (`paired_n`, then exact, within-one and under-prediction
  rates for each in `results.md`)
- `jev-serial` waited for Jev on every think request, and Jev answered before the first token
- with `--labels needed`, the `none` run produced probe labels with no probe errors, and ranked
  runs that cut did so through the ranking (`ranked_cuts_with_probs` + `ranked_cuts_flat`)

The sweep waits for the gateway's background probes to land after each run, so `observed`
still equals the number of think requests.

## Spike suite (the standard test, one BurstGPT spike)

`BROWNOUT_SUITE=spike` is the realistic suite below cut down to the part that tests Brownout: each policy gets
one spike instead of a 30-minute replay. The spike is the sharpest in the default BurstGPT window: 16 minutes of
trace from hour 522.0, two minutes of normal traffic at about 0.9 C, then two minutes at about 3.2 C, replayed in
240 s at the full test's scale (`BROWNOUT_LOAD=2.04` thins the spike the way 1.3 x C thins the two-hour window).
With C = 0.2 that is 98 workflows per policy, the same problems at the same times in every run.

C is given, not measured: `BROWNOUT_CAPACITY` is required, and `e2e.sh` stops before starting any box without it.
For Qwen3-4B on the two A10G replicas it is 0.2 workflows/s: with a queue they generate about 1,750 tokens/s, and a
workflow of the default mix produces about 7,800. The closed-loop measurement only counts workflows that finish
inside its window, so it undercounts when they take longer than the warm-up: at 192 in flight none finished before
575 s, and it reported 0.10.

Before the policies, a warm-up run (policy `none`, `BROWNOUT_WARMUP_N` workflows at one per second) teaches the
word-model predictor thinking lengths, and the gateway snapshots it (`POST /admin/policy {"snapshot": true}`). Each
policy starts from that snapshot (`"reset": "warm"`). A cold predictor puts every request in the median bucket of a
uniform prior, 512-1K tokens, so a spike shorter than the first completions would measure over-cutting, not
Brownout. The warm-up and the policies use disjoint halves of MATH-500 (`loadgen --split b` and `--split a`, by a
hash of the problem id), so the predictor has not seen a measured problem.

| Variable | Default in this suite |
|---|---|
| `BROWNOUT_CAPACITY` | required (workflows/s) |
| `BROWNOUT_TRACE` | `burstgpt` |
| `BROWNOUT_TRACE_START_H`, `BROWNOUT_TRACE_WINDOW_H` | `522.0`, `0.26667` |
| `BROWNOUT_TRACE_S` | `240` |
| `BROWNOUT_LOAD` | `2.04` |
| `BROWNOUT_TAIL_S` | `900`: long enough for `none` to finish (at 300 s, half of a barely-cut run was cancelled) |
| `BROWNOUT_WARMUP_N` | `64` |

The other variables are the realistic suite's.

```
AWS_PROFILE=rlwalkeradmin BROWNOUT_SUITE=spike BROWNOUT_CAPACITY=0.2 BROWNOUT_MODEL=Qwen/Qwen3-4B BROWNOUT_STOP=1 \
  infra/brownout/e2e.sh <commit>
```

To compare predictors on the same LLM hardware, run `BROWNOUT_JEV=1 BROWNOUT_LLM_ON_JEV=0` with half the
capacity: every policy gets one vLLM replica on the batch box, and Jev has the Jev box's GPU (`BROWNOUT_MODEL` or
`BROWNOUT_LORA` sets that one replica up, as with two).

It takes about 10 minutes of setup, 6 for the warm-up, and 5-10 per policy (`none` is the slowest). Results are
the realistic suite's, plus `warmup/` and the warm-up's label counts in `results.json` and `results.md`.
`python3 infra/brownout/report.py <results folder> --open` charts a run into `report.html`: correct and on time per
policy, vLLM's running and waiting requests over the run, and latency and thinking tokens by arrival time.

## Realistic suite (MATH-500, rate profile)

`BROWNOUT_SUITE=realistic` replaces the fixed-rate sweeps above with a suite where the one
vLLM GPU stands for one replica of a fleet: below capacity most of the time, short bursts
above it, recovery after. It cuts thinking with the word-model predictor (`online`) only, and
Jev is not used: `e2e.sh` sets `BROWNOUT_JEV=0`, which skips `setup_jev.sh`, stops the Open-Jev
servers and starts the gateway without `--jev-url`, so no request is sent to Jev and priority
comes from the header heuristic.

The Jev box's GPU then serves a second vLLM replica (`BROWNOUT_LLM_ON_JEV=1`, the default in
this suite): `remote/setup_llm.sh` stops the Open-Jev servers and runs the same container as
`lifeline-vllm-batch` (`VLLM_IMAGE`, `MODEL`, `MAX_MODEL_LEN` from `../vllm-ec2/config.env`,
`--gpu-memory-utilization 0.92 --enable-prefix-caching`, as `../vllm-ec2/up.sh` writes it) as
`vllm.service`, and waits until it answers. `remote/show_vllm.sh` prints the batch box's vLLM
command line (key masked) and GPU memory into the log, to compare. The gateway, still on the Jev
box's CPU, gets `--upstream http://127.0.0.1:8000/v1,http://<batch private IP>:8000/v1`, so it
reaches the local replica on localhost and the other one inside `brownout-internal-sg` (port
8000). `deploy.sh` waits for every upstream to answer before it starts the gateway. Routing and
per-replica pressure are described in `../../gateway/README.md`. The next default-suite run's
`setup_jev.sh` disables that `vllm.service` and removes the image and weights again, so a
realistic run after a default run downloads them again (about 25 GB).

It runs only `agent/tests/test_brownout_live.py` (not `test_live.py`):

1. Capacity: policy `none`, closed loop. `BROWNOUT_CAPACITY_CONCURRENCY` workflows stay in flight,
   so vLLM always has a queue. C is for all replicas together. Workflows finishing between `BROWNOUT_CAPACITY_WARMUP_S` and
   warmup + `BROWNOUT_CAPACITY_S` give C in workflows/s (and thinking tokens/s). Workflows still
   running at the end are cancelled. `BROWNOUT_CAPACITY` skips this and uses the given C.
2. Each policy in `BROWNOUT_POLICIES` runs the same rate profile: segments of
   `duration_s:rate`, the rate a multiple of C, Poisson within each segment. Arrival times,
   problems and deadline slacks come from `BROWNOUT_SEED`, so every policy gets the same
   sequence. After the profile ends, workflows still running after `BROWNOUT_TAIL_S` are
   cancelled and counted as wrong and late. The classifier is reset before each run, and each
   run waits until vLLM has no running or waiting requests.

`BROWNOUT_TRACE=burstgpt` replaces the rate profile with real arrivals: a two-hour window of the
BurstGPT trace (Azure ChatGPT and GPT-4 traffic, KDD 25; human conversation log by default) is compressed into
`BROWNOUT_TRACE_S` and thinned at random to a mean rate of `BROWNOUT_LOAD` x C, so the trace's bursts keep their
shape. The agent box downloads `BurstGPT_1.csv` (50 MB) from the BurstGPT GitHub repo once. Each kept request
becomes a MATH-500 problem by rank: the trace's longest responses get the hardest levels, in MATH-500's level mix,
so a burst brings the problems its requests would have. Segments and the baseline/burst phases come from the whole
window in 120 s bins (a bin above C is a burst), and the derived profile is written to `summary.json` and
`results.md`. See `agent/math_agent/trace.py` for why the default window was chosen.

A policy can end in `@name` to send that run to model `name`, e.g. `brownout:announced:online@r2` for a LoRA
served with `BROWNOUT_LORA=r2=...`. Both replicas are set up by `remote/setup_llm.sh` at the start of each run,
so they always serve the same model.

The client thinking budget is set per problem (`loadgen --level-budgets`): MATH levels 1-2 ask
for 2048 (up to 4096 on a retry), level 3 for 4096 (up to 6144), levels 4-5 for 6144; GSM8K
1024 (up to 2048). The solo latency used for deadlines is 150 s, 300 s and 450 s for those
MATH bands and 90 s for GSM8K.

| Variable | Default |
|---|---|
| `BROWNOUT_SUITE` | `default` (the sweeps above); `realistic` runs this suite, `spike` the spike suite above |
| `BROWNOUT_WORKLOAD` | `math500:1-5`. Format `dataset[:levels][=weight],...`, e.g. `math500:4-5` or `gsm8k=0.5,math500:4-5=0.5` |
| `BROWNOUT_PROFILE` | `180:0.7,120:1.2,180:0.7,180:1.5,240:0.7,120:2.0,180:0.7,240:1.2,300:0.7` (29 min; rate > 1 is a burst) |
| `BROWNOUT_TRACE` | unset (use the profile); `burstgpt` replays BurstGPT arrivals |
| `BROWNOUT_TRACE_LOG` | `conversation` (`api` or `all` add the batch API traffic, whose spikes reach 4000x its median hour) |
| `BROWNOUT_TRACE_START_H`, `BROWNOUT_TRACE_WINDOW_H` | `520.5`, `2` (hours from the start of the trace) |
| `BROWNOUT_TRACE_S` | `1800` (the window is replayed in this many seconds) |
| `BROWNOUT_LOAD` | `0.9` (mean arrival rate in units of C; the default window peaks at 1.9 C) |
| `BROWNOUT_MODEL` | unset (`MODEL` in `../vllm-ec2/config.env`); both replicas are set up with it, and runs without `@name` ask for it |
| `BROWNOUT_LORA` | unset; `name=ci/key.tgz` serves a LoRA tarball (one top folder with `adapter_config.json`) from the CI bucket as model `name` on both replicas |
| `BROWNOUT_NOTICE` | unset (`<budget_cut left=N>`); `countdown` is the sentence the `training/` models learned, or any text with `{N}` |
| `BROWNOUT_TPOT_SLO_MS` | unset (pressure is the queue only); a value starts the gateway with `--tpot-slo-ms`, so a crowded batch that decodes slower than this many ms per token per request also counts as pressure. 35 fits the default workload: a level 1-2 workflow of about 6,000 tokens still meets a 1.5x deadline (225 s) at 28 tokens/s |
| `BROWNOUT_POLICIES` | `none,uniform,brownout:silent:online,brownout:announced:online` |
| `BROWNOUT_JEV` | `0` in this suite, `1` otherwise (`1` deploys Open-Jev and passes `--jev-url` to the gateway) |
| `BROWNOUT_LLM_ON_JEV` | `1` in this suite (second vLLM replica on the Jev box; needs `BROWNOUT_JEV=0`), `0` otherwise |
| `BROWNOUT_CAPACITY` | unset (measure C) |
| `BROWNOUT_CAPACITY_CONCURRENCY` | `32` workflows in flight per replica |
| `BROWNOUT_CAPACITY_WARMUP_S`, `BROWNOUT_CAPACITY_S` | `180`, `600` |
| `BROWNOUT_TAIL_S` | `1200` |
| `BROWNOUT_SEED` | `0` |
| `BROWNOUT_AGENT_TIMEOUT` | computed by `e2e.sh`: capacity + policies x (profile + tail + 1800 s) + 1800 s, 23340 s with the defaults |

`e2e.sh` forwards these variables to the agent box. Run it from a shell with admin credentials:

```
AWS_PROFILE=rlwalkeradmin BROWNOUT_SUITE=realistic BROWNOUT_STOP=1 infra/brownout/e2e.sh <commit>
```

With the defaults it takes about 3.5 hours: 15-30 min setup (more when the second replica has
to download the vLLM image and Qwen3-8B), 15 min capacity, and 30-40 min per policy. That is longer than the CI job (230 min) and the CI role session (4 h), so run it
locally.

Results (`brownout-results/`):
- `results.md`: C, then one row per policy: accuracy, deadline met, correct and on time
  (goodput) overall and for arrivals in baseline and burst segments, latency p50/p95,
  thinking tokens, cuts and notices, pressure (mean, fraction at the cap, fraction at 0,
  fraction of samples with a vLLM queue, mean in baseline and burst). Then goodput by MATH level and a pressure
  histogram per policy. Instead of the Jev columns, it reports the word model's accuracy over
  its last 200 uncensored labels of the run (`word_model_exact`, `_within_one`, `_under`) and
  `jev_asked` (0 when Jev is off), think requests per replica (`replica_think`) and mean
  pressure per replica (`p_mean_replicas`).
- `results.json`: the same plus breakdowns by segment, by level and by level x phase.
- `capacity/` and `realistic-<policy>/`: the loadgen output and `pressure.jsonl`, the
  gateway's samples of `[t, p, waiting, running, replica]` during the run
  (`GET /v1/pressure?since=<unix time>`). The pressure columns pool the samples of all replicas;
  `pressure.replicas` in `results.json` has each replica's.

The test asserts only that there were no upstream or client errors, that the classifier saw
every think request (less any cancelled at the tail), that `none` never cuts, and, with Jev off,
that Jev was never asked.

CI runs the `brownout-e2e` job after the `aws` job, only in the full suite
(`gh workflow run CI -f suite=full`); pushes to main run the smoke suite only. It uses the same OIDC role and stops only the boxes it started.
A run costs roughly $5: two A10Gs for about two and a half hours. The CI role's session lasts 4 hours (`setup_iam.sh`).
