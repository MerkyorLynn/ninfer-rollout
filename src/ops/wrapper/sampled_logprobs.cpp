// ninfer::ops - sampled_token_logprobs wrapper: contract validation and launcher dispatch.
#include "ninfer/ops/sampled_logprobs.h"

#include "ops/launcher/sampled_logprobs.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::invalid_argument("sampled_token_logprobs: " + message);
}

std::int64_t trailing_elements(const Tensor& tensor) {
    return static_cast<std::int64_t>(tensor.ne[1]) * tensor.ne[2] * tensor.ne[3];
}

} // namespace

void sampled_token_logprobs(const Tensor& logits, const Tensor& tokens, const Tensor* counts,
                            const SamplingConfig* configs, const Tensor* overlay,
                            std::int32_t width, std::int32_t token_domain,
                            Tensor& sampling_logprobs, Tensor& raw_logprobs, cudaStream_t stream) {
    if (logits.dtype != DType::BF16 || !logits.is_contiguous() || logits.data == nullptr) {
        fail("logits must be contiguous non-null BF16");
    }
    if (tokens.dtype != DType::I32 || !tokens.is_contiguous() || tokens.data == nullptr) {
        fail("tokens must be contiguous non-null I32");
    }
    if (configs == nullptr) { fail("configs must be non-null"); }
    if (width <= 0) { fail("width must be positive"); }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        fail("token_domain must be in [1, physical_rows]");
    }
    const std::int64_t columns = trailing_elements(logits);
    if (columns <= 0 || columns % width != 0 || tokens.numel() != columns) {
        fail("logits columns and tokens must both equal width*B");
    }
    const std::int64_t batch = columns / width;
    for (const Tensor* out : {&sampling_logprobs, &raw_logprobs}) {
        if (out->dtype != DType::FP32 || !out->is_contiguous() || out->data == nullptr ||
            out->numel() != columns) {
            fail("outputs must be contiguous non-null FP32 with width*B elements");
        }
    }
    const std::int32_t* count_ptr = nullptr;
    if (counts != nullptr && counts->data != nullptr) {
        if (counts->dtype != DType::I32 || counts->numel() != batch) {
            fail("counts must be I32 [B]");
        }
        count_ptr = static_cast<const std::int32_t*>(counts->data);
    }
    const std::int32_t* overlay_ptr = nullptr;
    if (overlay != nullptr && overlay->data != nullptr && width > 1) {
        if (overlay->dtype != DType::I32 || overlay->numel() != (width - 1) * batch) {
            fail("overlay must be I32 [width-1, B]");
        }
        overlay_ptr = static_cast<const std::int32_t*>(overlay->data);
    }
    detail::sampled_token_logprobs_launch(logits, tokens, count_ptr, configs, overlay_ptr, width,
                                          token_domain, static_cast<std::int32_t>(columns),
                                          sampling_logprobs, raw_logprobs, stream);
}

void sampled_token_raw_logprobs(const Tensor& logits, const Tensor& tokens,
                                const Tensor& maxima, const Tensor* counts, std::int32_t width,
                                std::int32_t token_domain, Tensor& raw_logprobs,
                                cudaStream_t stream) {
    const auto fail = [](const std::string& message) -> void {
        throw std::invalid_argument("sampled_token_raw_logprobs: " + message);
    };
    if (logits.dtype != DType::BF16 || !logits.is_contiguous() || logits.data == nullptr) {
        fail("logits must be contiguous non-null BF16");
    }
    if (tokens.dtype != DType::I32 || !tokens.is_contiguous() || tokens.data == nullptr) {
        fail("tokens must be contiguous non-null I32");
    }
    if (width <= 0) { fail("width must be positive"); }
    if (token_domain <= 0 || token_domain > logits.ne[0]) {
        fail("token_domain must be in [1, physical_rows]");
    }
    const std::int64_t columns = trailing_elements(logits);
    if (columns <= 0 || columns % width != 0 || tokens.numel() != columns) {
        fail("logits columns and tokens must both equal width*B");
    }
    const std::int64_t batch = columns / width;
    if (maxima.dtype != DType::I32 || !maxima.is_contiguous() || maxima.data == nullptr ||
        maxima.numel() != columns) {
        fail("maxima must be contiguous non-null I32 with width*B elements");
    }
    if (raw_logprobs.dtype != DType::FP32 || !raw_logprobs.is_contiguous() ||
        raw_logprobs.data == nullptr || raw_logprobs.numel() != columns) {
        fail("raw output must be contiguous non-null FP32 with width*B elements");
    }
    const std::int32_t* count_ptr = nullptr;
    if (counts != nullptr && counts->data != nullptr) {
        if (counts->dtype != DType::I32 || counts->numel() != batch) {
            fail("counts must be I32 [B]");
        }
        count_ptr = static_cast<const std::int32_t*>(counts->data);
    }
    detail::sampled_token_raw_logprobs_launch(logits, tokens, maxima, count_ptr, width,
                                              token_domain, static_cast<std::int32_t>(columns),
                                              raw_logprobs, stream);
}

} // namespace ninfer::ops
