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
| `BROWNOUT_POLICIES` | `none,reject,uniform,brownout:silent:jev,brownout:announced:jev,brownout:announced:jev-serial,brownout:announced:online,brownout:announced:oracle,brownout:announced:jev-learn` (`policy:notice:predictor[:cut]`, cut = `proportional` or `ranked`; `skip` runs no sweep on this dataset: `jev` = untrained Jev beside the LLM, `jev-serial` = untrained Jev before the LLM, `online` = the word model, `oracle` = the thinking length the same prompt used in an earlier uncut run (an upper bound; run `none` first), `jev-learn` = trained Jev, skipped until its learner is deployed) |
| `BROWNOUT_N` | `64` |
| `BROWNOUT_RATE` | `4.0` workflows/s (24 at 1/s never queued on one A10G) |
| `BROWNOUT_DATASET` | `gsm8k` (budget 1024, up to 2048) |
| `BROWNOUT_HARD_DATASET` | `math500` (second sweep, budget 4096, up to 6144; empty skips it) |
| `BROWNOUT_HARD_POLICIES` | `none,uniform,brownout:announced:jev,brownout:announced:jev-serial,brownout:announced:online,brownout:announced:oracle` |
| `BROWNOUT_HARD_N`, `BROWNOUT_HARD_RATE` | `32`, `2.0` workflows/s |
| `BROWNOUT_REJECT_WAITING` | `2` (`defer` only) |
| `BROWNOUT_ADMIT_RATE` | `0.25` new workflows/s (`reject`) |
| `BROWNOUT_ADMIT_BURST` | `8` (`reject`) |
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

CI runs the `brownout-e2e` job after the `aws` job, only in the full suite
(`gh workflow run CI -f suite=full`); pushes to main run the smoke suite only. It uses the same OIDC role and stops only the boxes it started.
A run costs roughly $5: two A10Gs for about two and a half hours. The CI role's session lasts 4 hours (`setup_iam.sh`).
