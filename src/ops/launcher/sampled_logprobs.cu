// Implements: include/ninfer/ops/sampled_logprobs.h
// Match: wrapper-validated contiguous tensors; one CTA per column.
#include "ops/launcher/sampled_logprobs.h"

#include "core/device.h"
#include "ops/kernel/sampled_logprobs.cuh"

namespace ninfer::ops::detail {

void sampled_token_logprobs_launch(const Tensor& logits, const Tensor& tokens,
                                   const std::int32_t* counts, const SamplingConfig* configs,
                                   const std::int32_t* overlay, std::int32_t width,
                                   std::int32_t token_domain, std::int32_t columns,
                                   Tensor& sampling_logprobs, Tensor& raw_logprobs,
                                   cudaStream_t stream) {
    sampled_token_logprobs_kernel<<<static_cast<unsigned int>(columns), kSamplerBlock, 0,
                                    stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(tokens.data), counts, configs, overlay, width,
        token_domain, logits.ne[0], static_cast<float*>(sampling_logprobs.data),
        static_cast<float*>(raw_logprobs.data));
    CUDA_CHECK(cudaGetLastError());
}

void sampled_token_raw_logprobs_launch(const Tensor& logits, const Tensor& tokens,
                                       const std::int32_t* counts, std::int32_t width,
                                       std::int32_t token_domain, std::int32_t columns,
                                       Tensor& raw_logprobs, cudaStream_t stream) {
    sampled_token_logprobs_kernel<<<static_cast<unsigned int>(columns), kSamplerBlock, 0,
                                    stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(tokens.data), counts, nullptr, nullptr, width,
        token_domain, logits.ne[0], nullptr, static_cast<float*>(raw_logprobs.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
