#!/usr/bin/env bash
set -euo pipefail
systemctl cat vllm 2>/dev/null | grep -A3 '^ExecStart=' | sed -E 's/--api-key [^ ]+/--api-key ***/'
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
