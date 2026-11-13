#pragma once

// ninfer::ops::detail - private launch prototype for sampled_token_logprobs.

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void sampled_token_logprobs_launch(const Tensor& logits, const Tensor& tokens,
                                   const std::int32_t* counts, const SamplingConfig* configs,
                                   const std::int32_t* overlay, std::int32_t width,
                                   std::int32_t token_domain, std::int32_t columns,
                                   Tensor& sampling_logprobs, Tensor& raw_logprobs,
                                   cudaStream_t stream);

} // namespace ninfer::ops::detail
