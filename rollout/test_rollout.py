#!/usr/bin/env python3
"""Local contract tests. No GPU and no live server."""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ninfer_rollout.client import RolloutClient, RolloutError
from ninfer_rollout.contract import (
    completion_budget,
    parse_input_token_logprobs,
    score_request,
    temperature_for_sample,
)


class Handler(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, fmt, *args):
        return

    def _read(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode())

    def _write(self, code, payload):
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        body = self._read()
        Handler.seen.append((self.path, body))
        if self.path == "/v1/chat/completions":
            self._write(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "B", "reasoning_content": "check"},
                        }
                    ],
                    "usage": {"completion_tokens": 4, "prompt_tokens": 8},
                },
            )
            return
        if self.path == "/generate":
            start = body["logprob_start_len"]
            rows = [[-0.1 * (i + 1), token] for i, token in enumerate(body["input_ids"][start:])]
            self._write(200, {"meta_info": {"input_token_logprobs": rows}})
            return
        self._write(404, {"error": "missing"})


class ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        Handler.seen = []
        self.client = RolloutClient(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            f"http://127.0.0.1:{self.port}/generate",
            "rollout",
            context_length=131072,
        )

    def test_context_gate_rejects_before_http(self):
        with self.assertRaises(ValueError):
            completion_budget(80000, 94208, 131072)
        self.assertEqual(Handler.seen, [])

    def test_chat_sends_seed_and_thinking_flags(self):
        row = self.client.sample(
            [{"role": "user", "content": "Say B"}],
            prompt_tokens=32,
            max_tokens=64,
            temperature=0.6,
            top_p=0.95,
            seed=700007,
        )
        self.assertTrue(row["ok"])
        self.assertEqual(row["content"], "B")
        self.assertEqual(row["reasoning_content"], "check")
        path, body = Handler.seen[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["seed"], 700007)
        self.assertEqual(body["chat_template_kwargs"]["enable_thinking"], True)
        self.assertEqual(body["chat_template_kwargs"]["reasoning_effort"], "xhigh")
        self.assertFalse(body["stream"])

    def test_score_aligns_token_ids(self):
        logps = self.client.score([10, 11, 12, 13], 1)
        self.assertEqual(len(logps), 3)
        path, body = Handler.seen[0]
        self.assertEqual(path, "/generate")
        self.assertEqual(body, score_request([10, 11, 12, 13], 1))
        self.assertEqual(logps[0], -0.1)

    def test_score_rejects_token_mismatch(self):
        payload = {"meta_info": {"input_token_logprobs": [[-0.2, 99]]}}
        with self.assertRaises(ValueError):
            parse_input_token_logprobs(payload, [1, 2], 1)

    def test_group_marks_http_error_censored(self):
        self.client.chat_url = f"http://127.0.0.1:{self.port}/missing"
        group = self.client.collect_group(
            [{"role": "user", "content": "Say B"}],
            prompt_index=3,
            prompt_tokens=16,
            max_tokens=32,
            group_size=2,
        )
        self.assertEqual(group["censored"], 2)
        self.assertTrue(group["group_censored"])
        self.assertEqual(group["members"][0]["finish_reason"], "HTTP_404")
        self.assertEqual(group["members"][1]["temperature"], 0.8)
        self.assertEqual(temperature_for_sample(0, 8), 0.6)
        self.assertEqual(temperature_for_sample(4, 8), 0.8)

    def test_overlong_completion_is_censored_not_truncated(self):
        self.client.train_max_len = 10
        row = self.client.sample(
            [{"role": "user", "content": "Say B"}],
            prompt_tokens=8,
            max_tokens=32,
            temperature=0.6,
            top_p=0.95,
            seed=1,
        )
        self.assertFalse(row["ok"])
        self.assertEqual(row["censor_reason"], "exceeds_train_max_len")
        self.assertEqual(row["content"], "B")
        self.assertIsInstance(RolloutError("x"), RuntimeError)


if __name__ == "__main__":
    unittest.main()
