#pragma once

#include "runtime/contract/request.h"
#include "runtime/contract/timing.h"
#include <compare>
#include <cstdint>
#include <span>

namespace ninfer::runtime {

// Borrowed for one synchronous Program call. Engine maps compact rows to request-owned matchers.
class TokenMaskProvider {
public:
    virtual ~TokenMaskProvider()                                           = default;
    [[nodiscard]] virtual bool constrained(std::size_t row) const noexcept = 0;
    // Returns dead-end position bits. Even dead/unreachable positions receive a safe nonempty
    // device mask. A dead end fails a row only if verification reaches that position.
    [[nodiscard]] virtual std::uint32_t fill(std::size_t row, std::span<const TokenId> drafts,
                                             std::span<std::uint32_t> words) = 0;
    virtual void uploaded(std::size_t row, std::size_t bytes) noexcept       = 0;
};

struct LaneId {
    std::uint32_t value = 0;

    [[nodiscard]] friend constexpr bool operator==(LaneId, LaneId) noexcept  = default;
    [[nodiscard]] friend constexpr auto operator<=>(LaneId, LaneId) noexcept = default;
};

enum class ConsumeStatus : std::uint8_t {
    Consumed,
    InvariantMismatch,
};

enum class CommitDisposition : std::uint8_t {
    Active,
    Finishable,
    CancelledReleased,
    FailedReleased,
};

// The product Engine only needs statistics for rows whose sequence is released by commit.
// Direct diagnostic callers may temporarily request cumulative snapshots for every row.
enum class CommitObservation : std::uint8_t {
    ReleasedRowsOnly,
    AllRows,
};

struct CommitDecision {
    std::uint32_t accepted_tokens = 0;
    bool terminal                 = false;
    bool cancelled                = false;
    bool failed                   = false;
    // Copied unchanged from the corresponding OutputDecision; still relative to this row's
    // accepted span.
    std::optional<std::uint32_t> prefix_execution_split_after;
};

struct BeginSummary {
    std::uint32_t prompt_tokens        = 0;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::Root;

    [[nodiscard]] friend constexpr bool operator==(BeginSummary, BeginSummary) noexcept = default;
};

// How a generated token was produced (rollout logprob provenance).
enum class SampledTokenKind : std::uint8_t {
    Sampled       = 0, // ordinary sample (Begin, ordinary decode, speculative round with no draft)
    AcceptedDraft = 1, // draft token accepted by speculative rejection sampling
    Correction    = 2, // resampled from the residual after a draft rejection
    Bonus         = 3, // sampled from the last verify column after every draft was accepted
    Forced        = 4, // appended by output control (not sampled; no logprob)
};

// Optional sampling-time logprob spans share the token layout. Empty spans mean "not recorded".
struct GeneratedRound {
    std::span<const TokenId> tokens;
    std::span<const float> sampling_logprobs;
    std::span<const float> raw_logprobs;
    std::span<const std::uint8_t> token_kinds;
};

struct BatchedGeneratedRound {
    std::span<const TokenId> tokens;
    std::span<const std::int32_t> row_counts;
    std::uint32_t row_stride = 1;
    ExecutionTiming timing;
    std::span<const float> sampling_logprobs;
    std::span<const float> raw_logprobs;
    std::span<const std::uint8_t> token_kinds;
};

struct PrefillStepResult {
    BeginSummary summary;
    GeneratedRound round;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    ExecutionTiming timing;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

} // namespace ninfer::runtime
