Workload `math500:1-5`, trace `--trace burstgpt --trace-start-h 522.0 --trace-window-h 0.26667 --trace-s 240 --load 2.04` (profile from the trace, rates x C: `120:0.887,120:3.193`), seed 0.

Capacity C = 0.1 workflows/s (BROWNOUT_CAPACITY; None thinking tokens/s over None vLLM replica(s), None workflows in the window, queue present in None of samples).

Pressure: the queue (waiting / running) and decode time past 35 ms per output token.

Spike suite: problems from split a; every arm starts from a length predictor warmed on 64 split-b workflows (75 labels, 87 censored) under policy none.

| policy | n | accuracy | deadline_met | correct_and_on_time | goodput_baseline | goodput_burst | accuracy_baseline | accuracy_burst | cancelled | latency_p50_s | latency_p95_s | thinking_tokens_mean | think | cut | announced | notices | p_mean | p_at_cap | p_zero | queue_frac | tpot_ms_mean | p_mean_baseline | p_mean_burst | word_model_window | word_model_exact | word_model_within_one | word_model_under | jev_asked | replica_think | p_mean_replicas |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 49 | 0.7959 | 0.9796 | 0.7959 | 0.7143 | 0.8095 | 0.7143 | 0.8095 | 0 | 333.5782 | 650.2173 | 5915.5918 | 121 | 0 | 0 | 0 | 0.2637 | 0.0137 | 0.313 | 0.1047 | 41.9 | 0.0 | 0.126 | 131 | 0.3435 | 0.7405 | 0.3282 | 121 | [121] | [] |
| brownout:announced:online | 49 | 0.8163 | 1 | 0.8163 | 1 | 0.7857 | 1 | 0.7857 | 0 | 93.8938 | 273.6999 | 2347.0204 | 126 | 1 | 91 | 110 | 0.0319 | 0.0 | 0.7428 | 0.0 | 30.6 | 0.0 | 0.0778 | 87 | 0.2989 | 0.6552 | 0.3218 | 114 | [126] | [] |
| brownout:announced:jev | 49 | 0.898 | 1 | 0.898 | 1 | 0.881 | 1 | 0.881 | 0 | 92.5382 | 220.6526 | 2111.0816 | 126 | 3 | 93 | 102 | 0.029 | 0.0 | 0.763 | 0.0 | 29.3 | 0.0 | 0.0697 | 88 | 0.2841 | 0.6705 | 0.3182 | 121 | [126] | [] |

Correct and on time (accuracy, n) by band:

| policy | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| none | 1 (1, 3) | 0.9 (0.9, 10) | 0.9091 (0.9091, 11) | 0.8462 (0.8462, 13) | 0.5 (0.5, 12) |
| brownout:announced:online | 1 (1, 3) | 0.8 (0.8, 10) | 1 (1, 11) | 0.7692 (0.7692, 13) | 0.6667 (0.6667, 12) |
| brownout:announced:jev | 1 (1, 3) | 0.9 (0.9, 10) | 0.9091 (0.9091, 11) | 0.9231 (0.9231, 13) | 0.8333 (0.8333, 12) |

Pressure histogram (fraction of samples per decile of p; the last one includes p = 1):

| policy | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.3641 | 0.1596 | 0.0979 | 0.0374 | 0.1191 | 0.116 | 0.0393 | 0.0343 | 0.0156 | 0.0168 |
| brownout:announced:online | 0.8486 | 0.095 | 0.0493 | 0.0072 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| brownout:announced:jev | 0.8695 | 0.0759 | 0.0533 | 0.0013 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
