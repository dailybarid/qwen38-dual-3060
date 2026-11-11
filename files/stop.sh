#!/usr/bin/env bash
# Stop the Qwen3.8-27B unit and wait for the VRAM to actually come back.
set -euo pipefail
UNIT=qwen38tp.service
systemctl --user stop "$UNIT"
for i in $(seq 1 30); do
  if ! systemctl --user is-active --quiet "$UNIT"; then break; fi
  sleep 1
done
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
echo "stopped $UNIT"
