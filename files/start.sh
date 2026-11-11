#!/usr/bin/env bash
# Start the Qwen3.8-27B llama-server unit and wait for /health.
set -euo pipefail
UNIT=qwen38tp.service
URL=http://127.0.0.1:18435

systemctl --user start "$UNIT"
echo "started $UNIT — loading 16.5 GB takes 4-5 min on 2x RTX 3060"

for i in $(seq 1 90); do
  if curl -sf "$URL/health" >/dev/null 2>&1; then
    echo "ready: $(curl -s "$URL/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')"
    exit 0
  fi
  printf '.'
  sleep 5
done
echo
echo "still not healthy — check: journalctl --user -u $UNIT -n 50" >&2
exit 1
