#!/usr/bin/env python3
"""Compare stored sampling-time logprobs (tokens.jsonl) with teacher-forced rescoring (logprobs_<label>.jsonl).

Reports per-token |raw_sample - teacher_forced| overall and per kind (accepted_draft / correction / bonus /
sampled), length/alignment checks, and whether every emitted (non-forced) token had a finite
sampling_logprob (i.e. lay inside the truncated distribution it was sampled from).
"""
import argparse, json, math, statistics, sys
from pathlib import Path

def rj(p):
    p = Path(p)
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []

def q(v, f):
    if not v: return None
    v = sorted(v); return v[min(len(v) - 1, int(f * len(v)))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--label", default="tf_fp8")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rd = Path(a.run_dir)
    toks = {(t["id"], t["sample_index"]): t for t in rj(rd / "tokens.jsonl")}
    ref = {(r["id"], r["sample_index"]): r for r in rj(rd / f"logprobs_{a.label}.jsonl") if not r.get("error")}
    per_kind, allv, signed = {}, [], []
    samples, problems = [], []
    nonfinite_sampling = 0; emitted = 0; len_ok = 0; id_ok = 0
    for key, t in sorted(toks.items()):
        raw, smp, kinds, out = t.get("sample_logprobs"), t.get("sample_sampling_logprobs"), t.get("sample_token_kinds"), t.get("output_ids")
        if raw is None:
            problems.append(f"{key}: no sample_logprobs"); continue
        if out is not None and len(raw) == len(out): len_ok += 1
        else: problems.append(f"{key}: len(sample_logprobs)={len(raw)} len(output_ids)={len(out or [])}")
        if t.get("sample_logprob_ids_match"): id_ok += 1
        kinds = kinds or ["?"] * len(raw); smp = smp or [None] * len(raw)
        for s, k in zip(smp, kinds):
            if k == "forced": continue
            emitted += 1
            if s is None or not math.isfinite(s) or s > 1e-4: nonfinite_sampling += 1
        r = ref.get(key)
        if r is None: problems.append(f"{key}: no teacher-forced row"); continue
        tf = r["logprobs"]
        if len(tf) != len(raw): problems.append(f"{key}: tf len {len(tf)} != {len(raw)}"); continue
        d = []
        for x, y, k in zip(raw, tf, kinds):
            if x is None or y is None: continue
            per_kind.setdefault(k, []).append(abs(x - y)); allv.append(abs(x - y)); signed.append(x - y); d.append(abs(x - y))
        samples.append({"id": key[0], "sample_index": key[1], "n": len(raw), "max": max(d) if d else None,
                        "mean": statistics.fmean(d) if d else None,
                        "kinds": {k: kinds.count(k) for k in set(kinds)}})
    stat = lambda v: {"n": len(v), "max": max(v) if v else None, "mean": statistics.fmean(v) if v else None,
                      "p50": q(v, .5), "p99": q(v, .99), "p999": q(v, .999)}
    s = {"run_dir": str(rd), "label": a.label, "samples_with_logprobs": len(toks) - sum(1 for p in problems if "no sample_logprobs" in p),
         "len_ok": len_ok, "ids_match": id_ok, "emitted_tokens": emitted,
         "emitted_with_nonfinite_or_positive_sampling_logprob": nonfinite_sampling,
         "raw_vs_teacher_forced": stat(allv), "mean_signed_diff": statistics.fmean(signed) if signed else None,
         "per_kind": {k: stat(v) for k, v in sorted(per_kind.items())},
         "per_sample": samples, "problems": problems[:50]}
    out = Path(a.out) if a.out else rd / f"logprob_check_{a.label}.json"
    out.write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k != "per_sample"}, indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
