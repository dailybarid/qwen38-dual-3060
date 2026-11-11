# Qwen3.8-27B on two RTX 3060s: 40–55 tok/s with plain llama.cpp

Everything here was measured on one desktop, no new engine, no patched fork, no
quantization tricks. Stock `llama.cpp` with **tensor split + the model's own MTP
head** turns a 27B Q4 model that "shouldn't fit" into something that generates
faster than a 24.5 ms/token memory-bandwidth floor would suggest.

| | |
|---|---|
| Model | `unsloth/Qwen3.8-27B-GGUF`, `UD-Q4_K_M`, 16.46 GB |
| Hardware | 2× RTX 3060 12 GB (no NVLink), Ryzen 7 5700X, 31 GB RAM, Linux |
| Backend | llama.cpp `b11671` (`ba3c05f71`), CUDA 13 build |
| Result | **44–55 tok/s** code, **38–42 tok/s** prose, 526 tok/s prompt processing |
| Context | 100,400 tokens at **~9.9 / 11.2 GiB** per card |

Write-up version of this page (with charts):
<https://www.eljazeir.com/archives/627>

## Why it works

A dense model reads **all of its weights once per generated token**. With 720 GB/s
of combined bandwidth across the two 3060s (360 GB/s each, no NVLink — traffic
crosses PCIe), 16.46 GB of weights means:

```
16.46 GB / 720 GB/s ≈ 24.5 ms per token  →  ≈ 40 tok/s hard ceiling
```

That ceiling is why plain decoding on this pair lands around 30 tok/s in
practice. Crossing 40 means generating **more than one token per pass over the
weights**, which is exactly what speculative decoding does — and Qwen3.8 ships a
native **MTP (multi-token prediction) head**, so no separate draft model is
needed. Two settings unlock it:

1. **`--split-mode tensor`** — splits every weight matrix across both cards, so
   both memory controllers feed each token. The alternative (`layer`, the
   default) puts whole layers on one card and the second card idles during any
   single layer's work.
2. **`--spec-type draft-mtp --spec-draft-n-max 2`** — the model drafts 2 tokens,
   one batched pass verifies both, and every accepted draft token is free.

Measured MTP acceptance on this box: **84–96%**. That is where the extra speed
comes from, and the output is token-for-token identical to plain decoding —
rejected drafts are discarded, never emitted.

## The command

```bash
llama-server \
  -m Qwen3.8-27B-UD-Q4_K_M.gguf \
  -ngl 99 \
  --split-mode tensor --tensor-split 1,1 \
  -c 102400 \
  -ctk q4_0 -ctv q4_0 \
  -np 1 -fa on \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --jinja --metrics \
  --alias Qwen3.8-27B \
  --host 127.0.0.1 --port 18435
```

```
GGML_CUDA_ALLREDUCE=internal
```

`GGML_CUDA_ALLREDUCE=internal` makes llama.cpp do the cross-GPU reduce inside its
own kernel instead of through NCCL. Consumer 3060s have **no P2P DMA**, so NCCL
would stage every all-reduce through host memory. In `tensor` mode this variable
is not optional — expect a large slowdown without it.

### What each flag buys you

| Flag | Why |
|---|---|
| `--split-mode tensor` | both cards' bandwidth per token (the whole point) |
| `--tensor-split 1,1` | even split; the cards are identical |
| `--spec-type draft-mtp --spec-draft-n-max 2` | 1.8–1.9 tokens per weight pass at 84–96% acceptance |
| `-ctk q4_0 -ctv q4_0` | 4-bit KV cache; frees ~3 GB, needed to pass 64k context |
| `-fa on` | flash attention; required for quantized KV on these cards |
| `-c 102400` | 100k context in ~9.9/11.2 GiB (see the KV section) |
| `--jinja` | use the model's own chat template (MTP + tool calling) |

`n-max` is a dial, not a constant: 2 measured best here (57 tok/s in the highest
published report on the same pair). Going to 3–7 raises acceptance only until
validation cost overtakes the gain; benchmark your own workload.

## Context: cheaper than the naive math says

Qwen3.8-27B is **not** a plain transformer. Its metadata declares 65 blocks but
only **17 of them carry `attn_k`/`attn_v` tensors** (blocks 3, 7, 11, … every
fourth). The other 48 are recurrent (delta-net / mamba2, 336 recurrent tensors)
and hold a **constant-size state** regardless of context length.

So KV cost is not `layers × heads × 2 × head_dim`; it is roughly **19–33 KiB per
token**, not the ~73 KiB a dense 65-layer model at this head count would imply.
That is the difference between "256k is impossible" and "256k fits in ~4.8 GiB".

Measured VRAM (identical build, `-fa on`, q4_0 KV, split 1,1):

| Context | GPU0 used | GPU1 used | Free (0 / 1) |
|---|---|---|---|
| 65,536  | 9,412 MiB  | 10,989 MiB | 2,501 / 922 MiB |
| 81,920  | 9,596 MiB  | 11,237 MiB | — |
| 102,400 | 9,936 MiB  | 11,226 MiB | 1,977 / 684 MiB |
| 131,072 | 10,412 MiB | 11,753 MiB | 158 MiB on GPU1 |

128k technically loads but leaves ~158 MiB — one long prompt away from an OOM.
**100k is the practical ceiling on 2×12 GB**; the model's native context is
262,144, so a third card, a smaller quant, or `-ub 256` are the levers if you
need more. Load time is 4.5–5 minutes (16.5 GB read, then a full double upload in
tensor mode) — budget for it.

## Two warnings that are not errors

```
common_fit_params: failed to fit params to free device memory:
  llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort
```

Harmless: llama.cpp's auto-fit never runs in tensor mode, so you size `-c`
yourself (the table above is that sizing).

```
backend offload failed for seq_id=0; using CPU sampler
```

Harmless: backend sampling is unsupported with tensor split, so sampling runs on
the CPU. At ~50 tok/s of generation the sampler is not the bottleneck.

Also: `--split-mode tensor` requires the model to fit **entirely** in the two
cards' combined VRAM. There is no CPU offload path here — if it doesn't fit, it
doesn't start. That is the trade for the bandwidth.

## Hardware notes

- **No NVLink, asymmetric links, and it still works.** In this box GPU0 sits at
  PCIe 4.0 **x4** (second slot) and GPU1 at **x16**. Tensor split does not
  require matched links; the x4 card simply limits the all-reduce. Matched x16/x16
  should be a little faster, not a different league.
- **Temperature is the real limit, not VRAM.** Sustained prefill of a 125k-token
  prompt pushed GPU1 to **85–86 °C** with fans at 100%. Generation sits around
  46–73 °C. Cap your context, and don't plan on hour-long bulk ingestion without
  airflow.
- **`--tensor-split` is not a memory-resizer.** It splits weights; the 2 GiB
  difference in the table above comes from the KV cache and the CUDA context, not
  from the ratio. If one card is hot, reduce work — don't rebalance the split.

## Run it as a service

`files/qwen38tp.service` is the systemd **user** unit used here: pre-flight VRAM
check, `GGML_CUDA_ALLREDUCE=internal`, restart on failure, log to a file. Edit
the model path (the snapshot directory, not the `blobs/…` target) and install:

```bash
cp files/qwen38tp.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now qwen38tp.service
journalctl --user -u qwen38tp -f        # or tail the log file
```

`start.sh` / `stop.sh` wrap those two commands. The unit's `ExecStartPre` refuses
to start with less than 8600 MiB free per card, which turns a confusing CUDA OOM
during load into one clear line.

Then point any OpenAI-compatible client at `http://127.0.0.1:18435/v1`
(model name `Qwen3.8-27B`). Sanity check:

```bash
curl -s http://127.0.0.1:18435/v1/models | jq -r '.data[].id'
curl -s http://127.0.0.1:18435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"Say OK."}],"max_tokens":16}' \
  | jq -r '.choices[0].message.content'
```

`GET /props` shows `n_ctx`, the active chat template and the build string when
you need to confirm what is actually running.

## Benchmark honestly

Generation speed depends on the tokens you generate — code is repetitive and
speculative decoding eats it, prose is not. Always report both, always report
token counts, and never quote a 2-token reply as throughput.

```bash
curl -s http://127.0.0.1:18435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":"<prompt>"}],
       "max_tokens":400,"temperature":0}' \
  | jq '.timings | {predicted_n, predicted_per_second, prompt_per_second,
                    draft_n, draft_n_accepted}'
```

`draft_n_accepted / draft_n` is your MTP acceptance rate, live.

## What did not work

- **Layer split** (`--split-mode layer`, the default): ~30 tok/s, and MTP drafts
  can't be validated across cards efficiently. This is the mode most people
  accidentally benchmark.
- **Bigger drafts** (`--spec-type draft-dflash`, `n-max 7`) on layer split:
  36 tok/s code but **16 tok/s prose** — worse than no speculation. DFLASH needs
  tensor split to be worth anything.
- **No `GGML_CUDA_ALLREDUCE=internal`**: all-reduce over PCIe via NCCL dominates
  the token time.
- **256k context** on 2×12 GB: needs ~22.5 GiB against ~22.8 GiB usable, i.e. no
  margin. It loads once, dies on a real request.

## Credits

Method assembled from llama.cpp's own tensor-split and speculative-decoding
support plus published dual-3060 reports; all numbers above were re-measured on
the hardware described. Model by Qwen, GGUF quants by unsloth. llama.cpp is MIT
licensed.

## License

MIT — see `LICENSE`.
