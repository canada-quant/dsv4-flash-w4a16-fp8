#!/usr/bin/env python3
"""Throughput + TPOT benchmark for dual-DGX-Spark DSv4 W4A16 no-MTP.

Sends N requests at concurrency C, each with prompt_tokens + decode_tokens,
captures tok/s and TPOT percentiles. Writes JSON summary.

Usage:
    python throughput_bench.py http://spark-5:8888 \\
        --model DSV4-W4A16-FP8 --concurrency 1 --requests 10 \\
        --prompt-tokens 1024 --decode-tokens 1024 \\
        --out ~/dsv4-test-2gb10/results/sha-E/throughput_c1.json
"""
from __future__ import annotations
import argparse, json, statistics, sys, time
import concurrent.futures
from urllib.request import urlopen, Request


def _post(url: str, payload: dict, timeout: int = 600) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, method="POST", headers={"Content-Type": "application/json"}, data=body)
    t0 = time.time()
    with urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    return out, time.time() - t0


def one_request(base: str, model: str, prompt_tokens: int, decode_tokens: int) -> dict:
    # Build a long prompt deterministically — use /v1/completions to avoid chat-template overhead
    placeholder = "The quick brown fox jumps over the lazy dog. "
    # ~10 tokens per placeholder repetition (depends on tokenizer); over-build then truncate
    prompt = (placeholder * (max(1, prompt_tokens // 10) + 1))[:prompt_tokens * 5]
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": decode_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
    }
    try:
        resp, wall = _post(base + "/v1/completions", payload)
    except Exception as e:
        return {"error": str(e), "wall_s": None}
    usage = resp.get("usage") or {}
    prompt_t = usage.get("prompt_tokens", 0)
    completion_t = usage.get("completion_tokens", 0)
    return {
        "wall_s": wall,
        "prompt_tokens": prompt_t,
        "completion_tokens": completion_t,
        "throughput_tps": completion_t / wall if wall and completion_t else 0.0,
        "tpot_ms": (wall * 1000) / max(completion_t, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="e.g. http://spark-5:8888")
    ap.add_argument("--model", default="DSV4-W4A16-FP8")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--requests", type=int, default=10)
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--decode-tokens", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    print(f"[bench] base={args.base} model={args.model} "
          f"c={args.concurrency} n={args.requests} "
          f"prompt={args.prompt_tokens} decode={args.decode_tokens}", file=sys.stderr)

    if args.warmup:
        print(f"[bench] {args.warmup} warmup requests...", file=sys.stderr)
        for _ in range(args.warmup):
            one_request(args.base, args.model, args.prompt_tokens, args.decode_tokens)

    print(f"[bench] running {args.requests} measured requests at c={args.concurrency}...", file=sys.stderr)
    results = []
    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(one_request, args.base, args.model, args.prompt_tokens, args.decode_tokens)
            for _ in range(args.requests)
        ]
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            results.append(r)
            if "error" in r:
                print(f"  request error: {r['error']}", file=sys.stderr)
            else:
                print(f"  +{r['completion_tokens']:>5} tok in {r['wall_s']:.2f}s = "
                      f"{r['throughput_tps']:.1f} tok/s, TPOT {r['tpot_ms']:.1f}ms",
                      file=sys.stderr)
    t_end = time.time()

    ok = [r for r in results if "error" not in r]
    if not ok:
        print("[bench] ERROR: no successful requests", file=sys.stderr)
        sys.exit(2)

    total_tokens = sum(r["completion_tokens"] for r in ok)
    aggregate_tps = total_tokens / (t_end - t_start)
    summary = {
        "config": {
            "base": args.base,
            "model": args.model,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
        },
        "results": results,
        "summary": {
            "successful": len(ok),
            "failed": len(results) - len(ok),
            "total_wall_s": t_end - t_start,
            "total_output_tokens": total_tokens,
            "aggregate_tok_per_s": aggregate_tps,
            "per_request_tps": {
                "median": statistics.median([r["throughput_tps"] for r in ok]),
                "mean": statistics.mean([r["throughput_tps"] for r in ok]),
                "min": min(r["throughput_tps"] for r in ok),
                "max": max(r["throughput_tps"] for r in ok),
            },
            "tpot_ms": {
                "median": statistics.median([r["tpot_ms"] for r in ok]),
                "p95": sorted(r["tpot_ms"] for r in ok)[max(0, int(len(ok) * 0.95) - 1)] if len(ok) >= 2 else ok[0]["tpot_ms"],
                "mean": statistics.mean([r["tpot_ms"] for r in ok]),
            },
        },
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[bench] wrote {args.out}", file=sys.stderr)
    print(f"[bench] aggregate {aggregate_tps:.1f} tok/s | "
          f"per-req median {summary['summary']['per_request_tps']['median']:.1f} tok/s | "
          f"TPOT median {summary['summary']['tpot_ms']['median']:.1f}ms",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
