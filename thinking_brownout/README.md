# Thinking Brownout: code (training + evaluation)

All files go in one folder (~/thinkingcut); they import each other by name.

Shared
  cutlib.py           vLLM client, trace reading, grading, cut points, hesitation lexicon
  analyze_boxed.py    \boxed{} extraction and answer matching (used by the grader)

Data and predictor
  collect_traces.py   uncut traces from vLLM (prompt, thinking, answer, correct)
  classifier.py       self-rating predictor (EffortScorer: 0-9, thinking off, expected level)

Cut behaviour (offline)
  sample_bench.py     pick cut points from traces
  cut_bench.py        hard / notice cuts at those points (baselines)
  cut_report.py       accuracy, lost / won, after-cut thinking, by state and cut position
  regrade.py          re-grade saved runs with the current grader
  prune_tail.py       helpers (mask_answer) used by decisive_sample

Training the cut adapter (aLoRA on [cut])
  decisive_sample.py  decisive sampling at cut points (+ --report)
  make_sft.py         SFT examples from the decisive picks (stop share <= 50%)
  cut_alora.py        selftest / train (SFT) / check / eval / grpo

Serving evaluation
  tb_vllm.py          inside vLLM 0.28: activated LoRA from the [cut] marker + in-place cuts (logits processor)
  tb_vllm_check.py    correctness: mask / marker / hard in vLLM, then vLLM vs HF+PEFT after the marker
  serve_eval.py       burst replay, one policy per run (nocut reject budget exit hard tb; --select)
  serve_report.py     goodput, SLO met, accuracy, thinking, throughput, TTFT p50/p99, cut, lost/won, reaction
  run_serve.sh        calibrate | main (Table II) | sweep | select (selection + ablation); vLLM command at the end
  vllm_probe.py       read-only look at the installed vLLM
