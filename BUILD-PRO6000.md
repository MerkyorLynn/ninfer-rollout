# Build and load on PRO 6000

This tree already contains the rollout scoring changes. Do not apply another patch.

Base: Neroued/ninfer `81c8ce093b2c1646a87566a8e59d807fcf0ec95c` (2026-10-07), plus the rollout scoring changes.

PRO 6000 is Blackwell. Build `120a`. Do not run an `sm_121a` binary on it.

```bash
cmake -S . -B build-sm120a -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build-sm120a -j --target ninfer-serve
```

Load an FP8 `.ninfer` pack with DFlash2 draft length 7. This is the measured configuration. Produce the pack with the existing `qwen3_8_27b_nvfp4` recipe; see [Weight conversion](docs/weight-conversion.md).

```bash
./build-sm120a/apps/ninfer-serve MODEL.ninfer \
  --spec dflash2 --draft-tokens 7 \
  --rollout-score --rollout-score-context 8192 \
  --host 127.0.0.1 --port 19421 --model-id ninfer-rollout
```

On one RTX PRO 6000 Blackwell 96GB, at C8, 100K context, `max_tokens` 94,208, 80 samples, and logprobs enabled: 917.3 tok/s, busy wall 2,153 s, 68 correct / 9 wrong / 3 truncated, acceptance length 4.676. Against an SGLang BF16 teacher-forced reference, k3 per-token mean KL is 0.001721, 1.66% of tokens have a probability ratio outside 0.8–1.25, and max|Δ| is 8.59.

Acceptance:

```bash
python3 probe.py --base http://127.0.0.1:19421 --model ninfer-rollout --context-length 8192
```

`probe.py` is in https://github.com/MerkyorLynn/ninfer-rollout

The same source was probed on an `sm_121a` build with an FP8 + DFlash2 pack. Chat stopped with content. `POST /generate` returned 4 aligned logprob rows for input ids `[1,2,3,4,5]` and `logprob_start_len=1`. That binary is not the PRO 6000 binary.
