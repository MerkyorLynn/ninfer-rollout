#pragma once

// Implements: include/ninfer/ops/sampled_logprobs.h
// Algorithm: one kSamplerBlock CTA per column. The combined kernel uses a two-pass logsumexp and
// rebuilds truncated support. The dedicated raw-only kernel reuses the target argmax's maximum
// and performs only the sum-exp pass, preserving the existing sum reduction order.

#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ float sampled_logprobs_block_reduce(float value, float* scratch,
                                                               bool take_max) {
    const int tid = threadIdx.x;
    scratch[tid]  = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = take_max ? fmaxf(scratch[tid], scratch[tid + stride])
                                    : scratch[tid] + scratch[tid + stride];
        }
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__launch_bounds__(kSamplerBlock) __global__ void sampled_token_logprobs_kernel(
    const __nv_bfloat16* logits, const std::int32_t* tokens, const std::int32_t* counts,
    const SamplingConfig* configs, const std::int32_t* overlay, std::int32_t width,
    std::int32_t token_domain, std::int32_t physical_rows, float* sampling_out, float* raw_out) {
    const int tid    = threadIdx.x;
    const int column = static_cast<int>(blockIdx.x);
    const int row    = column / width;
    const int col    = column - row * width;
    if (counts != nullptr && col >= counts[row]) {
        if (tid == 0) {
            sampling_out[column] = CUDART_NAN_F;
            raw_out[column]      = CUDART_NAN_F;
        }
        return;
    }

    __shared__ float red[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int n_support;

    const SamplingConfig cfg = configs[row];
    const int token          = tokens[column];
    const std::int64_t base  = static_cast<std::int64_t>(column) * physical_rows;
    const bool token_ok      = token >= 0 && token < token_domain;

    float local_max = -CUDART_INF_F;
    for (int v = tid; v < token_domain; v += blockDim.x) {
        local_max = fmaxf(local_max, __bfloat162float(logits[base + v]));
    }
    const float maximum = sampled_logprobs_block_reduce(local_max, red, true);
    float local_sum     = 0.0f;
    for (int v = tid; v < token_domain; v += blockDim.x) {
        local_sum += expf(__bfloat162float(logits[base + v]) - maximum);
    }
    const float sum = sampled_logprobs_block_reduce(local_sum, red, false);
    if (tid == 0) {
        raw_out[column] =
            token_ok ? __bfloat162float(logits[base + token]) - maximum - logf(sum) : CUDART_NAN_F;
    }

    if (!(cfg.temperature > 0.0f)) {
        if (tid == 0) { sampling_out[column] = token_ok ? 0.0f : CUDART_NAN_F; }
        return;
    }

    const std::int32_t* row_overlay =
        overlay != nullptr && width > 1 ? overlay + static_cast<std::int64_t>(row) * (width - 1)
                                        : nullptr;
    const int overlay_len = row_overlay != nullptr ? col : 0;
    if (token_domain <= kSamplerTileItems) {
        sampling_build_truncated_small(logits, base, token_domain, cfg, red, red_idx, cand_val,
                                       cand_idx, prob, &n_support, row_overlay, overlay_len);
    } else {
        sampling_build_truncated_block_fast(logits, base, token_domain, cfg, merge_val, merge_idx,
                                            cand_val, cand_idx, prob, &n_support, row_overlay,
                                            overlay_len);
    }
    if (tid == 0) {
        float lp = -CUDART_INF_F;
        for (int j = 0; j < n_support; ++j) {
            if (cand_idx[j] == token) {
                lp = logf(prob[j]);
                break;
            }
        }
        sampling_out[column] = token_ok ? lp : CUDART_NAN_F;
    }
}

__launch_bounds__(kSamplerBlock) __global__ void sampled_token_raw_logprobs_kernel(
    const __nv_bfloat16* logits, const std::int32_t* tokens, const std::int32_t* maxima,
    const std::int32_t* counts, std::int32_t width, std::int32_t token_domain,
    std::int32_t physical_rows, float* raw_out) {
    const int tid    = threadIdx.x;
    const int column = static_cast<int>(blockIdx.x);
    const int row    = column / width;
    const int col    = column - row * width;
    if (counts != nullptr && col >= counts[row]) {
        if (tid == 0) { raw_out[column] = CUDART_NAN_F; }
        return;
    }

    __shared__ float red[kSamplerBlock];
    const int token         = tokens[column];
    const std::int64_t base = static_cast<std::int64_t>(column) * physical_rows;
    const bool token_ok     = token >= 0 && token < token_domain;

    const float maximum = __bfloat162float(logits[base + maxima[column]]);
    float local_sum     = 0.0f;
    for (int v = tid; v < token_domain; v += blockDim.x) {
        local_sum += expf(__bfloat162float(logits[base + v]) - maximum);
    }
    const float sum = sampled_logprobs_block_reduce(local_sum, red, false);
    if (tid == 0) {
        raw_out[column] =
            token_ok ? __bfloat162float(logits[base + token]) - maximum - logf(sum) : CUDART_NAN_F;
    }
}

} // namespace ninfer::ops
