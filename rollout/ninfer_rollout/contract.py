"""SGLang rollout contract used by the existing trainer cache.

The scoring request is teacher-forced. It is not an OpenAI chat logprobs flag.
"""

from __future__ import annotations

TRAIN_MAX_LEN = 102400
DEFAULT_SAFETY_TOKENS = 16


def completion_budget(prompt_tokens: int, max_tokens: int, context_length: int, safety_tokens: int = DEFAULT_SAFETY_TOKENS) -> int:
    if prompt_tokens < 1:
        raise ValueError("prompt_tokens must be positive")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if prompt_tokens + max_tokens + safety_tokens > context_length:
        raise ValueError(
            f"prompt_plus_completion_exceeds_context:{prompt_tokens}+{max_tokens}+{safety_tokens}>{context_length}"
        )
    return max_tokens


def train_fits(prompt_tokens: int, completion_tokens: int, train_max_len: int = TRAIN_MAX_LEN) -> bool:
    return prompt_tokens + completion_tokens <= train_max_len


def score_request(input_ids: list[int], logprob_start_len: int) -> dict:
    if logprob_start_len < 0 or logprob_start_len >= len(input_ids):
        raise ValueError(f"logprob_start_len out of range: {logprob_start_len} for {len(input_ids)} tokens")
    return {
        "input_ids": list(input_ids),
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": logprob_start_len,
        "top_logprobs_num": 0,
        "return_text_in_logprobs": False,
    }


def parse_logprob_entry(value):
    token_id = None
    if isinstance(value, (list, tuple)):
        if len(value) < 2:
            raise ValueError("logprob entry list needs logprob and token_id")
        logp, token_id = value[0], value[1]
    elif isinstance(value, dict):
        if "logprob" not in value or "token_id" not in value:
            raise ValueError("logprob entry dict needs logprob and token_id")
        logp = value["logprob"]
        token_id = value["token_id"]
    else:
        raise ValueError(f"unsupported logprob entry: {type(value).__name__}")
    return float(logp), int(token_id)


def parse_input_token_logprobs(payload: dict, input_ids: list[int], logprob_start_len: int) -> list[float]:
    try:
        rows = payload["meta_info"]["input_token_logprobs"]
    except (KeyError, TypeError) as exc:
        raise ValueError("missing meta_info.input_token_logprobs") from exc
    expected = len(input_ids) - logprob_start_len
    if len(rows) != expected:
        raise ValueError(f"logprob length {len(rows)} != {expected}")
    logps = []
    for offset, row in enumerate(rows):
        logp, token_id = parse_logprob_entry(row)
        absolute = logprob_start_len + offset
        if token_id != input_ids[absolute]:
            raise ValueError(f"token_id mismatch at {absolute}: got {token_id}, expected {input_ids[absolute]}")
        logps.append(logp)
    return logps


def chat_request(
    model: str,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    reasoning_effort: str = "xhigh",
) -> dict:
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "stream": False,
        "reasoning_effort": reasoning_effort,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": reasoning_effort,
        },
    }


def temperature_for_sample(sample_index: int, group_size: int) -> float:
    half = (group_size + 1) // 2
    return 0.6 if sample_index < half else 0.8


def seed_for_sample(seed_base: int, prompt_index: int, sample_index: int) -> int:
    return seed_base + prompt_index * 100 + sample_index
