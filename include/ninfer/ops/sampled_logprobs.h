#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Op: Sampling-time log-probabilities of emitted tokens (RL rollout).
 *
 * Columns are request-major: column c belongs to request row r = c / width at verify column
 * j = c % width. logits is BF16 [physical_rows, width*B] (any rank whose first dimension is
 * physical_rows and whose remaining dimensions flatten to width*B), tokens is I32 [width*B].
 *
 * For every valid column (counts == null, or j < counts[r]) with token t:
 *
 *   raw_logprobs[c]      = l[t,c] - log(sum_{v<token_domain} exp(l[v,c]))
 *                          (full vocabulary, temperature 1, no truncation; the same formula as
 *                          target_logprobs and the teacher-forced rollout scorer)
 *   sampling_logprobs[c] = log p_cfg(t | column c), where p_cfg is the exact truncated target
 *                          distribution the sampler draws from: optional grammar mask, presence /
 *                          frequency penalties with the round-local draft overlay of the first j
 *                          drafts, top-k (capped at 20), temperature, top-p and min-p, then
 *                          renormalized. Greedy rows (temperature <= 0) report 0. A token outside
 *                          the truncated support reports -inf.
 *
 * Under speculative rejection sampling each emitted token (accepted draft, correction or bonus) is
 * marginally distributed as p_cfg at its position, so sampling_logprobs is the behaviour-policy
 * log-probability RL importance ratios need. Invalid columns are written as NaN.
 *
 * overlay, when non-null, is I32 [width-1, B] (the verified drafts) and is only read when
 * penalties are active. configs is a device SamplingConfig[B].
 */
void sampled_token_logprobs(const Tensor& logits, const Tensor& tokens, const Tensor* counts,
                            const SamplingConfig* configs, const Tensor* overlay,
                            std::int32_t width, std::int32_t token_domain,
                            Tensor& sampling_logprobs, Tensor& raw_logprobs, cudaStream_t stream);

/**
 * Op: raw log-probabilities of emitted tokens when the processed values already exist.
 *
 * Uses the same full-vocabulary temperature-1 formula and request-major [width,B] layout as
 * sampled_token_logprobs. maxima is I32 [width,B] containing each column's argmax over the same
 * current logits and token domain, with no intervening mutation; the caller guarantees every index
 * is in [0,token_domain). Its represented logit supplies the exact LSE maximum without a separate
 * scan. counts limits each row to its licensed prefix and invalid outputs are NaN. This entry does
 * not read sampling configs or reconstruct a truncated distribution.
 */
void sampled_token_raw_logprobs(const Tensor& logits, const Tensor& tokens,
                                const Tensor& maxima, const Tensor* counts, std::int32_t width,
                                std::int32_t token_domain, Tensor& raw_logprobs,
                                cudaStream_t stream);

} // namespace ninfer::ops
