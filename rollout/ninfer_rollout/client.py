"""Rollout client for a server that speaks the trainer's two endpoints."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ninfer_rollout.contract import (
    TRAIN_MAX_LEN,
    chat_request,
    completion_budget,
    parse_input_token_logprobs,
    score_request,
    seed_for_sample,
    temperature_for_sample,
    train_fits,
)


class RolloutError(RuntimeError):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(kind if not detail else f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


def post_json(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        exc.close()
        raise RolloutError(f"HTTP_{exc.code}", body) from exc


class RolloutClient:
    def __init__(
        self,
        chat_url: str,
        generate_url: str,
        model: str,
        *,
        context_length: int,
        timeout: int = 3600,
        safety_tokens: int = 16,
        train_max_len: int = TRAIN_MAX_LEN,
    ):
        self.chat_url = chat_url
        self.generate_url = generate_url
        self.model = model
        self.context_length = context_length
        self.timeout = timeout
        self.safety_tokens = safety_tokens
        self.train_max_len = train_max_len

    def sample(
        self,
        messages: list[dict],
        *,
        prompt_tokens: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        reasoning_effort: str = "xhigh",
    ) -> dict:
        budget = completion_budget(prompt_tokens, max_tokens, self.context_length, self.safety_tokens)
        payload = chat_request(
            self.model,
            messages,
            max_tokens=budget,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            reasoning_effort=reasoning_effort,
        )
        body = post_json(self.chat_url, payload, self.timeout)
        choice = body["choices"][0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        finish = choice.get("finish_reason")
        row = {
            "ok": finish == "stop",
            "finish_reason": finish,
            "content": message.get("content") or "",
            "reasoning_content": message.get("reasoning_content") or "",
            "usage": usage,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "seed": seed,
            "temperature": temperature,
            "train_fit": train_fits(prompt_tokens, completion_tokens, self.train_max_len),
        }
        if not row["train_fit"]:
            row["ok"] = False
            row["censor_reason"] = "exceeds_train_max_len"
        elif finish != "stop":
            row["censor_reason"] = f"finish_{finish}"
        return row

    def score(self, input_ids: list[int], logprob_start_len: int) -> list[float]:
        payload = score_request(input_ids, logprob_start_len)
        body = post_json(self.generate_url, payload, self.timeout)
        return parse_input_token_logprobs(body, input_ids, logprob_start_len)

    def collect_group(
        self,
        messages: list[dict],
        *,
        prompt_index: int,
        prompt_tokens: int,
        max_tokens: int,
        group_size: int = 8,
        seed_base: int = 700000,
        top_p: float = 0.95,
        reasoning_effort: str = "xhigh",
    ) -> dict:
        members = []
        for sample_index in range(group_size):
            try:
                row = self.sample(
                    messages,
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature_for_sample(sample_index, group_size),
                    top_p=top_p,
                    seed=seed_for_sample(seed_base, prompt_index, sample_index),
                    reasoning_effort=reasoning_effort,
                )
            except RolloutError as exc:
                row = {
                    "ok": False,
                    "finish_reason": exc.kind,
                    "content": "",
                    "reasoning_content": "",
                    "censor_reason": exc.kind,
                    "error": exc.detail,
                    "seed": seed_for_sample(seed_base, prompt_index, sample_index),
                    "temperature": temperature_for_sample(sample_index, group_size),
                    "train_fit": False,
                }
            row["sample_index"] = sample_index
            members.append(row)
        censored = [row for row in members if not row.get("ok")]
        return {
            "group_size": group_size,
            "kept": group_size - len(censored),
            "censored": len(censored),
            "group_censored": len(censored) > 0,
            "members": members,
        }
