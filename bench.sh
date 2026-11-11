#!/usr/bin/env bash
# Measure generation + MTP acceptance on a running server.
# Usage: ./bench.sh ["code prompt"] ["prose prompt"]
set -euo pipefail
URL=${URL:-http://127.0.0.1:18435}
MODEL=${MODEL:-Qwen3.8-27B}
CODE=${1:-Write a Python function that merges two sorted lists into one sorted list, with docstring and type hints.}
PROSE=${2:-Explain in three paragraphs why memory bandwidth limits token generation speed on consumer GPUs.}

run() {
  python3 - "$URL" "$MODEL" "$1" <<'PY'
import json, sys, urllib.request
url, model, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 400, "temperature": 0,
                   "chat_template_kwargs": {"enable_thinking": False}}).encode()
req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
t = json.load(urllib.request.urlopen(req, timeout=900))["timings"]
acc = t.get("draft_n_accepted") or 0
tot = t.get("draft_n") or 0
print(f"  gen {t['predicted_n']:>4} tok @ {t['predicted_per_second']:6.1f} t/s"
      f" | prompt {t.get('prompt_per_second', 0):6.1f} t/s"
      f" | MTP {acc}/{tot}" + (f" ({100*acc/tot:.0f}%)" if tot else ""))
PY
}

echo "code:";  run "$URL" "$MODEL" "$CODE"
echo "prose:"; run "$URL" "$MODEL" "$PROSE"
