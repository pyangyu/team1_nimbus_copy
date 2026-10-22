# Brownout on AWS

Three boxes in the team account (826671498662, us-east-1). They talk over private IPs
inside `brownout-internal-sg`, which allows ports 8000 and 9000 only between its members.

| Box | Instance | Runs |
|---|---|---|
| `brownout-agent` | t3.small | the math agent: live tests and `loadgen` |
| `brownout-jev` | g5.xlarge (the old `lifeline-vllm-rt`) | Open-Jev-2B on `127.0.0.1:8791` (its own venv: Open-Jev pins transformers 5.10.2), and the Brownout gateway on `:9000` |
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
- installs `open-jev.service`, then probes one 10-bucket score question and prints the time.

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
| `BROWNOUT_POLICIES` | `none,reject,uniform,brownout,brownout:announced` |
| `BROWNOUT_N` | `24` |
| `BROWNOUT_RATE` | `1.0` workflows/s |
| `BROWNOUT_DATASET` | `gsm8k` |
| `BROWNOUT_REJECT_WAITING` | `2` |

Accuracy and deadline numbers are reported in `results.md`, not asserted. The test does
assert these:
- every think request was observed by the classifier
- no upstream errors
- `none` never cuts or rejects
- the cutting policies cut once there was pressure
- only `reject` rejects
- Jev answered at least once

CI runs the `brownout-e2e` job after the `aws` job, on push to main and on
`workflow_dispatch`. It uses the same OIDC role and stops only the boxes it started.
A run costs roughly $2: two A10Gs for about an hour.
