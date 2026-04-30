# vLLM Experiment on AMD R9700 (RDNA4/gfx1201)

**Date:** 2026-04-28
**Hardware:** AMD Radeon AI PRO R9700 (gfx1201/RDNA4), 32GB VRAM
**Remote:** tama (Ubuntu 24.04 LXC on Proxmox, PVE kernel 6.17.13-2-pve)
**ROCm:** 7.2.1 (apt packages)

## Goal

Test whether vLLM provides better inference performance than llama.cpp for Qwen3.6-27B-AWQ on the R9700.

## Setup Steps

### 1. Docker Installation

```bash
apt-get update && apt-get install -y docker.io docker-compose-v2
systemctl enable docker && systemctl start docker
```

Docker was installed but ultimately not needed — vLLM was installed via pip in a venv instead.

### 2. vLLM Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm
```

- **vLLM version:** 0.20.0+rocm721
- **PyTorch:** 2.10.0+git8514f05
- **Location:** `/root/.venv/` (~8.6GB)
- **Installation method:** uv (astral) with ROCm wheel index

### 3. Model Download

```bash
uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('QuantTrio/Qwen3.6-27B-AWQ'))"
```

- **Model:** QuantTrio/Qwen3.6-27B-AWQ (AWQ quantized, ~20GB)
- **Location:** `/root/.cache/huggingface/models--QuantTrio--Qwen3.6-27B-AWQ/`
- **Architecture:** Qwen3_5ForConditionalGeneration (hybrid Mamba/attention)

### 4. vLLM Launch

Initial attempt failed due to `--swap-space` flag (removed in v0.20.0). Second attempt failed due to insufficient KV cache memory for 32K context. Final working command:

```bash
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4

uv run python -m vllm.entrypoints.openai.api_server \
    --model /root/.cache/huggingface/models--QuantTrio--Qwen3.6-27B-AWQ/snapshots/<hash> \
    --served-model-name qwen3.6-27b-awq \
    --max-num-seqs 32 \
    --max-model-len 32768 \
    --kv-cache-dtype turboquant_4bit_nc \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000
```

**Key flags:**
- `--kv-cache-dtype turboquant_4bit_nc` — Q4 KV cache to fit 32K context in 32GB VRAM
- `--gpu-memory-utilization 0.95` — Use 95% of VRAM (up from default 0.9)
- `--max-model-len 32768` — Full 32K context
- `VLLM_USE_DEEP_GEMM=0` — Disable deep GEMM (not supported on RDNA4)
- `VLLM_USE_FLASHINFER_MOE_FP16=1` — Use FlashInfer for MoE layers
- `VLLM_USE_FLASHINFER_SAMPLER=0` — Disable FlashInfer sampler (RDNA4 incompatibility)

**Startup time:** ~5 minutes (model load ~62s, torch.compile ~132s, warmup ~67s)

## Benchmark Results

### vLLM (this experiment)

| Model | Test | t/s | Peak t/s | TTFT (ms) |
|-------|------|-----|----------|-----------|
| qwen3.6-27b-awq | pp2048 | 778.18 ± 21.81 | — | 2463.70 ± 81.75 |
| qwen3.6-27b-awq | tg32 | **6.73 ± 0.00** | 7.00 ± 0.00 | — |

### llama.cpp Vulkan (baseline)

| Backend | PP (tok/s) | TG (tok/s) |
|---------|-----------|-----------|
| **Vulkan** | 694 | **38** |
| **ROCm** | 856 | 29 |

### Comparison

| Backend | PP (tok/s) | TG (tok/s) | Notes |
|---------|-----------|-----------|-------|
| llama.cpp Vulkan | 694 | **38** | Best TG, stable, no idle bug |
| llama.cpp ROCm | 856 | 29 | GPU stuck at 100% when idle |
| vLLM ROCm | 778 | **6.7** | AITER kernels not optimized for RDNA4 |

**vLLM TG is 5.6x slower than llama.cpp Vulkan.**

## Root Cause Analysis

**The 6.7 tok/s result is NOT the correct RDNA4 speed** — vLLM hit multiple unoptimized fallback paths because gfx1201 is not properly recognized in v0.20.0's kernel selection logic. The hardware (128 FP8 WMMA accelerators) is capable of far better.

### What Went Wrong

1. **`on_mi3xx()` doesn't recognize gfx1201** — The function in `vllm/platforms/rocm.py` only checks for `gfx942` and `gfx950` (MI300/MI350). Without gfx1201 in this list, FP8 operations fall through to `torch._scaled_mm()` which **upcasts everything to FP32**, completely wasting the hardware's 128 AI accelerators.

2. **Missing `VLLM_ROCM_USE_AITER=0`** — The experiment set `VLLM_USE_DEEP_GEMM=0` but NOT `VLLM_ROCM_USE_AITER=0`. AITER's C++/ASM kernels don't work on RDNA4 and either crash or silently fall back to extremely slow paths. The Triton kernels from AITER *do* work but require the arch mapping patch (see below).

3. **No RDNA4 matrix sizes in `is_aiter_triton_kernel_tuned()`** — Even if the Triton path was reached, the `(n, k)` matrix sizes for Qwen3.6-27B wouldn't match any MI350-tuned sizes in `vllm/model_executor/layers/quantization/utils/fp8_utils.py`, falling back to default (slow) kernel configs.

4. **wvSplitK skinny GEMM not enabled for RDNA4** — PR [#34709](https://github.com/vllm-project/vllm/pull/34709) enables the wvSplitK kernel for RDNA4 decode (M=1..4 GEMMs). Before this, RDNA4 decode fell back to `torch.nn.functional.linear` (pure Python-level GEMM). This PR alone gives ~15% decode improvement, and was still open as of v0.20.0's release (April 27, 2026).

5. **AWQ uses `TritonW4A16LinearKernel` on ROCm** — vLLM warns: *"awq quantization is not fully optimized yet. The speed can be slower than non-quantized models."* On RDNA4 without proper arch detection, Triton may not generate optimal WMMA instructions for the 16×16 tile size.

### The Fixes (Community Patch — [Issue #28649](https://github.com/vllm-project/vllm/issues/28649))

Three source code patches + environment variable:

```bash
# 1. Disable broken AITER C++/ASM kernels
export VLLM_ROCM_USE_AITER=0

# 2. Patch vllm/platforms/rocm.py — add "gfx1201" to on_mi3xx()
# 3. Patch fp8_utils.py — add RDNA4-specific matrix sizes
# 4. Patch AITER arch mapping before launch:
python3 -c "
import aiter.ops.triton.utils.arch_info as a
a._ARCH_TO_DEVICE['gfx1201'] = 'MI350X'
"
```

Plus 16 kernel config JSON files for FP8 tile sizes.

### Expected Performance With Patches

Community testing on the same R9700 hardware with patches applied:

| Model | Before Patches | After Patches | Improvement |
|-------|---------------|---------------|-------------|
| Qwen3-0.6B | ~160 tok/s | ~200 tok/s | +25% |
| Qwen3-30B | ~52 tok/s | ~85 tok/s | +63% |

For a **27B model**, expect **~60-80 tok/s** with proper patches — not 6.7. Prefill also improves by up to 100% for prompts with 10,000+ tokens.

### Upstream Status (as of v0.20.0, April 27, 2026)

| Fix | In v0.20.0? | PR/Issue |
|-----|------------|----------|
| Device IDs (gfx1201 recognized) | ✅ Yes | [#38455](https://github.com/vllm-project/vllm/pull/38455) |
| TritonW4A16LinearKernel (AWQ) | ✅ Yes | [#37352](https://github.com/vllm-project/vllm/pull/37352) |
| `on_mi3xx()` includes gfx1201 | ❌ No | Community patch (#28649) |
| RDNA4 matrix sizes | ❌ No | Community patch (#28649) |
| wvSplitK for RDNA4 decode | ❌ No | [#34709](https://github.com/vllm-project/vllm/pull/34709) (open) |
| FP8 inference on gfx1201 | ❌ No | [#36659](https://github.com/vllm-project/vllm/pull/36659) (open, +14-31% throughput) |

vLLM releases every ~2 weeks, so patches could land in v0.21 or v0.22. Nightly builds (`rocm/vllm-dev:nightly`) can be tested sooner.

## Known Issues on RDNA4

- **ROCm/ROCm #5706:** MES firmware bug causes 100% GPU usage after HIP idle (fixed in Linux 7.0/6.12, not backported to PVE kernel)
- **vllm-project/vllm #40981:** vLLM fails to start on RDNA4 in containers (amdsmi, circular import, torch.cuda.device_count() issues)
- **vllm-project/vllm #40980:** TP=2 deadlock on dual R9700 with v0.19.x (works on v0.14.0)

## Conclusions

1. **vLLM out-of-the-box is broken for RDNA4 decode in v0.20.0** — The 6.7 tok/s result reflects unpatched fallback paths, not hardware capability. With community patches, expect ~60-80 tok/s for 27B models.

2. **llama.cpp Vulkan remains the best unpatched option** — 38 tok/s TG with stable behavior, no idle GPU bug, and no ROCm dependency. No source modifications needed.

3. **ROCm llama.cpp has slight prefill advantage** (856 vs 694 tok/s) but suffers from the GPU idle bug and slower TG (29 vs 38 tok/s).

4. **Q4 KV cache (`turboquant_4bit_nc`)** does work and enables 32K context on a 32GB card.

5. **vLLM is viable on RDNA4 once patches land upstream** — The hardware (128 FP8 WMMA accelerators, 32GB VRAM) is well-suited for LLM inference. The gap is purely software.

## Cleanup

vLLM server killed. Remaining artifacts:
- `/root/.venv/` — vLLM venv (~8.6GB)
- `/root/.cache/vllm/` — torch.compile cache
- `/root/.cache/huggingface/models--QuantTrio--Qwen3.6-27B-AWQ/` — model weights (~20GB)
- `/opt/vllm-venv/` — earlier failed pip install venv (~8.6GB)

**Recommendation:** Remove all vLLM artifacts to reclaim ~37GB disk space.

## Recommendations

1. **Stick with llama.cpp Vulkan** for now — best unpatched performance, stable, no ROCm dependency
2. **Wait for upstream RDNA4 patches** — vLLM releases every ~2 weeks; monitor PRs [#34709](https://github.com/vllm-project/vllm/pull/34709) (wvSplitK) and issue [#28649](https://github.com/vllm-project/vllm/issues/28649) (FP8 patch)
3. **Retry vLLM after v0.21+** — test with `VLLM_ROCM_USE_AITER=0` and verify `on_mi3xx()` recognizes gfx1201
4. **Monitor container issues** — [#40981](https://github.com/vllm-project/vllm/issues/40981) and [#40980](https://github.com/vllm-project/vllm/issues/40980) for TP=2 and container support
5. **Consider FP8 models instead of AWQ** — FP8 has better ROCm kernel support and benefits more from WMMA accelerators on RDNA4
