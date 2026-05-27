#!/usr/bin/env python3
"""Long-reasoning think-max benchmark for DSv4 no-MTP on dual spark.

Sends 5 AIME-2024 problems with chat_template_kwargs={thinking: true, reasoning_effort: high}
and max_tokens=65536. Captures: time-to-first-token, total wall, decode tok/s, reasoning vs
final-answer tokens, truncation rate.

Usage:
    python aime_think_max.py http://spark-5:8888 \\
        --model DSV4-W4A16-FP8 \\
        --out ~/dsv4-test-2gb10/results/sha-E/think_max.json
"""
from __future__ import annotations
import argparse, json, sys, time
from urllib.request import urlopen, Request

# AIME 2024 problems 1-5 (curated; full problem statements)
AIME_2024 = [
    {"id": "AIME-2024-I-1", "answer": 204, "problem": "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, including t minutes spent in the coffee shop. When she walks s+2 kilometers per hour, the walk takes her 2 hours and 24 minutes, including t minutes spent in the coffee shop. Suppose Aya walks at s + 1/2 kilometers per hour. Find the number of minutes the walk takes her, including the t minutes spent in the coffee shop."},
    {"id": "AIME-2024-I-2", "answer": 25, "problem": "There exist real numbers x and y, both greater than 1, such that log_x(y^x)=log_y(x^(4y))=10. Find xy."},
    {"id": "AIME-2024-I-3", "answer": 80, "problem": "Alice and Bob play the following game. A stack of n tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either 1 token or 4 tokens from the stack. Whoever removes the last token wins. Find the number of positive integers n less than or equal to 2024 for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice's play."},
    {"id": "AIME-2024-I-4", "answer": 116, "problem": "Jen enters a lottery by picking 4 distinct numbers from S={1,2,3,...,9,10}. Four numbers are randomly chosen from S. She wins a prize if at least two of her numbers were 2 of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is m/n where m and n are relatively prime positive integers. Find m+n."},
    {"id": "AIME-2024-I-5", "answer": 104, "problem": "Rectangles ABCD and EFGH are drawn such that D,E,C,F are collinear. Also, A,D,H,G all lie on a circle. If BC=16, AB=107, FG=17, and EF=184, what is the length of CE?"},
]


def post(url: str, payload: dict, timeout: int = 1800) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, method="POST", headers={"Content-Type": "application/json"}, data=body)
    t0 = time.time()
    with urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    return out, time.time() - t0


def extract_answer(text: str) -> str | None:
    """Try to extract a 3-digit integer at the end of the text (AIME answer format)."""
    import re
    # AIME answers are integers 0-999; pull last 1-3 digit standalone number
    matches = re.findall(r"(?:final answer|answer)\D*?(\d{1,3})\b", text, re.IGNORECASE)
    if matches:
        return matches[-1]
    # Fallback: last standalone integer in last 200 chars
    tail = text[-500:]
    nums = re.findall(r"\b(\d{1,3})\b", tail)
    return nums[-1] if nums else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--model", default="DSV4-W4A16-FP8")
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--out", required=True)
    ap.add_argument("--problems", type=int, default=5, help="how many problems (1-5)")
    args = ap.parse_args()

    results = []
    for prob in AIME_2024[: args.problems]:
        print(f"\n=== {prob['id']} (answer={prob['answer']}) ===", file=sys.stderr)
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "You are a careful mathematician. Solve the problem step by step. End your response with 'Final answer: <number>'."},
                {"role": "user", "content": prob["problem"]},
            ],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"},
        }
        try:
            resp, wall = post(args.base + "/v1/chat/completions", payload)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            results.append({"id": prob["id"], "error": str(e)})
            continue
        msg = resp["choices"][0]["message"]
        reasoning = msg.get("reasoning") or ""
        content = msg.get("content") or ""
        usage = resp.get("usage") or {}
        prompt_t = usage.get("prompt_tokens", 0)
        completion_t = usage.get("completion_tokens", 0)
        truncated = resp["choices"][0]["finish_reason"] == "length"
        extracted = extract_answer(content + " " + reasoning)
        correct = (str(extracted) == str(prob["answer"])) if extracted is not None else False
        result = {
            "id": prob["id"],
            "expected": prob["answer"],
            "extracted": extracted,
            "correct": correct,
            "truncated": truncated,
            "wall_s": wall,
            "prompt_tokens": prompt_t,
            "completion_tokens": completion_t,
            "reasoning_chars": len(reasoning),
            "content_chars": len(content),
            "decode_tok_per_s": completion_t / wall if wall else 0.0,
            "tpot_ms": (wall * 1000) / max(completion_t, 1),
        }
        results.append(result)
        print(f"  wall={wall:.1f}s tokens={completion_t} tps={result['decode_tok_per_s']:.1f}"
              f" truncated={truncated} extracted={extracted} correct={correct}",
              file=sys.stderr)

    summary = {
        "config": {
            "base": args.base,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "thinking": "high",
        },
        "results": results,
        "summary": {
            "n": len(results),
            "completed": sum(1 for r in results if "error" not in r),
            "correct": sum(1 for r in results if r.get("correct")),
            "truncated": sum(1 for r in results if r.get("truncated")),
            "avg_wall_s": sum(r.get("wall_s", 0) for r in results) / max(len(results), 1),
            "avg_decode_tps": sum(r.get("decode_tok_per_s", 0) for r in results) / max(len(results), 1),
            "avg_completion_tokens": sum(r.get("completion_tokens", 0) for r in results) / max(len(results), 1),
        },
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}", file=sys.stderr)
    print(json.dumps(summary["summary"], indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
