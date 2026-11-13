#!/usr/bin/env python3
"""Acceptance probe for a live rollout server.

Exit 0 only when chat fields and teacher-forced scoring both match the contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ninfer_rollout.client import RolloutClient, RolloutError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Server origin, for example http://127.0.0.1:19131")
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    client = RolloutClient(
        f"{base}/v1/chat/completions",
        f"{base}/generate",
        args.model,
        context_length=args.context_length,
        timeout=args.timeout,
    )
    report = {"chat": None, "score": None}
    try:
        sample = client.sample(
            [{"role": "user", "content": "Reply with the single letter B."}],
            prompt_tokens=32,
            max_tokens=64,
            temperature=0.0,
            top_p=1.0,
            seed=7,
        )
        report["chat"] = {
            "finish_reason": sample["finish_reason"],
            "has_content": bool(sample["content"] or sample["reasoning_content"]),
            "seed_echo_not_required": True,
        }
        if not report["chat"]["has_content"]:
            raise RolloutError("empty_chat_message")
    except Exception as exc:
        report["chat"] = {"error": str(exc)}
        print(json.dumps(report, ensure_ascii=False))
        return 1
    input_ids = [1, 2, 3, 4, 5]
    try:
        logps = client.score(input_ids, 1)
        report["score"] = {"count": len(logps), "ok": len(logps) == 4}
        if len(logps) != 4:
            raise RolloutError("score_length")
    except Exception as exc:
        report["score"] = {"error": str(exc)}
        print(json.dumps(report, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
