import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("MATH_AGENT_LIVE"),
                                reason="set MATH_AGENT_LIVE=1 and LIFELINE_BASE_URL/LIFELINE_API_KEY to run")


def test_one_gsm8k_workflow_against_the_real_server(tmp_path):
    p = subprocess.run([sys.executable, "-m", "math_agent.run", "--dataset", "gsm8k", "--index", "0",
                        "--budget", "1024", "--max-budget", "2048", "--out", str(tmp_path)],
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr
    w = json.loads((tmp_path / "workflows.jsonl").read_text())
    assert w["error"] is None and w["llm_calls"] >= 2 and w["thinking_tokens"] > 0
    assert w["pred"] is not None
