#!/usr/bin/env python3
"""Quick GPU probe for NInfer sampling-time logprobs (short prompts, concurrent).

Checks per request: logprobs.content present; len == usage.completion_tokens == len(completion_token_ids);
token ids match; raw/sampling logprobs finite (except forced) and <= 0; greedy rows sampling_logprob == 0;
then teacher-forces the same ids through POST /generate (--rollout-score) and compares raw logprobs.
Exit 0 = PASS. Writes JSON summary to --out.
"""
import argparse, concurrent.futures as cf, json, math, statistics, sys, time, urllib.request

def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, then explain it in two sentences.",
    "List five prime numbers greater than 100 and explain how you checked each one.",
    "Explain what a hash map is to a beginner in one short paragraph.",
    "Translate to French: 'The weather is nice today, let us go for a walk in the park.' Then give the literal word order.",
    "What is 17 * 23? Show the steps.",
    "Give three tips for writing clear commit messages.",
]

def one(base, model, prompt, temp, seed, max_tokens, effort):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temp, "top_p": 0.95, "top_k": 20, "seed": seed, "logprobs": True,
            "return_token_ids": True}
    if effort:
        body["reasoning_effort"] = effort
    t0 = time.time()
    r = post(f"{base}/v1/chat/completions", body)
    ch = r["choices"][0]
    content = (ch.get("logprobs") or {}).get("content")
    ids = r.get("completion_token_ids")
    pids = r.get("prompt_token_ids")
    ct = r["usage"]["completion_tokens"]
    res = {"temp": temp, "seed": seed, "completion_tokens": ct, "seconds": round(time.time() - t0, 2),
           "finish_reason": ch.get("finish_reason"), "problems": []}
    P = res["problems"]
    if not content: P.append("no logprobs.content"); return res
    if len(content) != ct: P.append(f"len(content)={len(content)} != completion_tokens={ct}")
    if ids is None or len(ids) != ct: P.append("completion_token_ids missing/len mismatch")
    if ids is not None and [c["token_id"] for c in content] != ids: P.append("token_id sequence != completion_token_ids")
    raw = [c["logprob"] for c in content]; smp = [c["sampling_logprob"] for c in content]; kinds = [c["kind"] for c in content]
    for i, (a, b, k) in enumerate(zip(raw, smp, kinds)):
        if k == "forced": continue
        if a is None or not (a <= 1e-4): P.append(f"tok{i} raw={a} kind={k}"); break
        if b is None or not (b <= 1e-4): P.append(f"tok{i} sampling={b} kind={k} (emitted token outside sampled support?)"); break
        if temp == 0 and abs(b) > 1e-6: P.append(f"greedy tok{i} sampling={b} != 0"); break
    res["kinds"] = {k: kinds.count(k) for k in set(kinds)}
    # teacher-forced rescoring of the same ids
    if pids and ids:
        sc = post(f"{base}/generate", {"input_ids": pids + ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                                       "return_logprob": True, "logprob_start_len": len(pids), "top_logprobs_num": 0,
                                       "return_text_in_logprobs": False})
        rows = sc["meta_info"]["input_token_logprobs"]
        tf = [(x[0] if isinstance(x, list) else x["logprob"]) for x in rows]
        if len(tf) != len(ids): P.append(f"rescore len {len(tf)} != {len(ids)}")
        d = {}
        for a, t, k in zip(raw, tf, kinds):
            if a is None or t is None: continue
            d.setdefault(k, []).append(abs(a - t))
        allv = [x for v in d.values() for x in v]
        res["tf_absdiff"] = {"all": {"n": len(allv), "max": max(allv) if allv else None, "mean": statistics.fmean(allv) if allv else None}}
        for k, v in d.items():
            res["tf_absdiff"][k] = {"n": len(v), "max": max(v), "mean": statistics.fmean(v)}
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:19431")
    ap.add_argument("--model", default="ninfer-rollout")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--tol-max", type=float, default=1.0, help="max |raw - teacher_forced| allowed (BF16 logits; prefill vs verify kernels)")
    ap.add_argument("--tol-mean", type=float, default=0.03)
    ap.add_argument("--out", default="probe_logprobs.json")
    a = ap.parse_args()
    jobs = []
    for i, p in enumerate(PROMPTS):
        jobs.append((p, [0.6, 0.8, 0.0, 0.6, 0.8, 0.6][i], 900000 + i))
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(lambda j: one(a.base, a.model, j[0], j[1], j[2], a.max_tokens, a.effort), jobs))
    ok = True
    for r in results:
        td = (r.get("tf_absdiff") or {}).get("all") or {}
        if r["problems"] or not td.get("n") or td["max"] > a.tol_max or td["mean"] > a.tol_mean:
            ok = False
    summary = {"pass": ok, "tol_max": a.tol_max, "tol_mean": a.tol_mean, "results": results}
    json.dump(summary, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
