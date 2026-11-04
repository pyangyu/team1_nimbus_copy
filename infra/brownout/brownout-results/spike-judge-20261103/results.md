Workload `math500:1-5`, trace `--trace burstgpt --trace-start-h 522.0 --trace-window-h 0.26667 --trace-s 240 --load 2.04` (profile from the trace, rates x C: `120:0.887,120:3.193`), seed 0.

Capacity C = 0.1 workflows/s (BROWNOUT_CAPACITY; None thinking tokens/s over None vLLM replica(s), None workflows in the window, queue present in None of samples).

Pressure: the queue (waiting / running) and decode time past 35 ms per output token.

Spike suite: problems from split a; every arm starts from a length predictor warmed on 64 split-b workflows (75 labels, 93 censored) under policy none.

| policy | n | accuracy | deadline_met | correct_and_on_time | goodput_baseline | goodput_burst | accuracy_baseline | accuracy_burst | cancelled | latency_p50_s | latency_p95_s | thinking_tokens_mean | think | cut | announced | notices | p_mean | p_at_cap | p_zero | queue_frac | tpot_ms_mean | p_mean_baseline | p_mean_burst | word_model_window | word_model_exact | word_model_within_one | word_model_under | jev_asked | replica_think | p_mean_replicas | cut_mode | max_frac | exit_at | judge | settled_cuts | exits | judged | judge_mean_ms | judge_mean_settled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 49 | 0.8367 | 0.8776 | 0.7959 | 1 | 0.7619 | 1 | 0.8095 | 0 | 417.9027 | 701.174 | 6368.2245 | 124 | 0 | 0 | 0 | 0.3283 | 0.0373 | 0.2933 | 0.2408 | 44.6 | 0.0 | 0.1235 | 130 | 0.2692 | 0.6846 | 0.3077 | 124 | [124] | [] | proportional | 0.9 |  | jev | 0 | 0 | 0 |  |  |
| brownout:announced:online | 49 | 0.8367 | 0.9796 | 0.8367 | 1 | 0.8095 | 1 | 0.8095 | 0 | 161.0009 | 263.2682 | 3139.7551 | 124 | 36 | 104 | 419 | 0.0996 | 0.0 | 0.4964 | 0.0 | 33.5 | 0.0 | 0.1415 | 82 | 0.1829 | 0.5732 | 0.3293 | 124 | [124] | [] | proportional | 0.9 |  | jev | 0 | 0 | 0 |  |  |
| brownout:announced:jev | 49 | 0.8163 | 1 | 0.8163 | 0.8571 | 0.8095 | 0.8571 | 0.8095 | 0 | 150.9369 | 220.9278 | 2893.1429 | 123 | 61 | 97 | 244 | 0.0681 | 0.0 | 0.6191 | 0.0 | 31.9 | 0.0 | 0.1488 | 82 | 0.1707 | 0.561 | 0.3293 | 123 | [123] | [] | proportional | 0.9 |  | jev | 0 | 0 | 0 |  |  |
| brownout:announced:jev:settled:e0.7:jjev | 49 | 0.8776 | 0.9796 | 0.8776 | 0.8571 | 0.881 | 0.8571 | 0.881 | 0 | 157.9744 | 420.6588 | 4013.2449 | 121 | 4 | 62 | 120 | 0.1139 | 0.0 | 0.4979 | 0.0 | 34.6 | 0.0 | 0.1209 | 84 | 0.1786 | 0.5714 | 0.3214 | 121 | [121] | [] | settled | 0.9 | 0.7 | jev | 38 | 29 | 383 | 1301.7 | 0.548 |
| brownout:announced:online:settled:e0.98:jfeat | 49 | 0.7959 | 1 | 0.7959 | 1 | 0.7619 | 1 | 0.7619 | 0 | 122.738 | 250.2566 | 2515.3878 | 123 | 3 | 80 | 187 | 0.0626 | 0.0 | 0.6929 | 0.0 | 30.8 | 0.0 | 0.1489 | 85 | 0.1765 | 0.5647 | 0.3176 | 123 | [123] | [] | settled | 0.9 | 0.98 | feat | 64 | 11 | 244 | 0.9 | 0.859 |

Correct and on time (accuracy, n) by band:

| policy | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| none | 1 (1, 3) | 0.8 (0.8, 10) | 0.7273 (0.9091, 11) | 0.9231 (0.9231, 13) | 0.6667 (0.6667, 12) |
| brownout:announced:online | 1 (1, 3) | 0.8 (0.8, 10) | 0.9091 (0.9091, 11) | 0.9231 (0.9231, 13) | 0.6667 (0.6667, 12) |
| brownout:announced:jev | 1 (1, 3) | 0.7 (0.7, 10) | 1 (1, 11) | 0.8462 (0.8462, 13) | 0.6667 (0.6667, 12) |
| brownout:announced:jev:settled:e0.7:jjev | 1 (1, 3) | 0.8 (0.8, 10) | 1 (1, 11) | 0.9231 (0.9231, 13) | 0.75 (0.75, 12) |
| brownout:announced:online:settled:e0.98:jfeat | 1 (1, 3) | 0.7 (0.7, 10) | 0.9091 (0.9091, 11) | 0.7692 (0.7692, 13) | 0.75 (0.75, 12) |

Pressure histogram (fraction of samples per decile of p; the last one includes p = 1):

| policy | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.3417 | 0.0694 | 0.1254 | 0.063 | 0.0676 | 0.1294 | 0.0799 | 0.0426 | 0.0286 | 0.0525 |
| brownout:announced:online | 0.5349 | 0.2873 | 0.107 | 0.0337 | 0.0325 | 0.0048 | 0.0 | 0.0 | 0.0 | 0.0 |
| brownout:announced:jev | 0.7758 | 0.0611 | 0.1134 | 0.0293 | 0.0051 | 0.0038 | 0.0076 | 0.0038 | 0.0 | 0.0 |
| brownout:announced:jev:settled:e0.7:jjev | 0.5767 | 0.1641 | 0.1009 | 0.0943 | 0.0591 | 0.0049 | 0.0 | 0.0 | 0.0 | 0.0 |
| brownout:announced:online:settled:e0.98:jfeat | 0.7878 | 0.0474 | 0.0836 | 0.04 | 0.035 | 0.0062 | 0.0 | 0.0 | 0.0 | 0.0 |
