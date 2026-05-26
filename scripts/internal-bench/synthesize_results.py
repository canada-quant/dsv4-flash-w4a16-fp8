#!/usr/bin/env python3
"""Synthesize bench results across SHAs into a model-card-ready Markdown block.

Inputs:  ~/dsv4-test-2gb10/results/{pairA_baseline_428e08e,pairB_head_5d647,...}/
         each containing summary.md + throughput_*.json + aime_think_max.json
         and (optionally) an oom_sweep/sweep_summary.txt

Output: prints markdown that can be pasted into model-card-draft.md or HF README.md.
"""
from __future__ import annotations
import argparse, glob, json, os, sys


def load_throughput(d: dict) -> dict:
    s = d.get("summary") or {}
    return {
        "aggregate_tps": s.get("aggregate_tok_per_s", 0),
        "per_req_median": (s.get("per_request_tps") or {}).get("median", 0),
        "tpot_median": (s.get("tpot_ms") or {}).get("median", 0),
        "tpot_p95": (s.get("tpot_ms") or {}).get("p95", 0),
        "successful": s.get("successful", 0),
    }


def load_aime(d: dict) -> dict:
    s = d.get("summary") or {}
    return {
        "n": s.get("n", 0),
        "correct": s.get("correct", 0),
        "truncated": s.get("truncated", 0),
        "avg_wall": s.get("avg_wall_s", 0),
        "avg_decode_tps": s.get("avg_decode_tps", 0),
        "avg_completion_tokens": s.get("avg_completion_tokens", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/dsv4-test-2gb10/results"))
    ap.add_argument("--out", default="-", help="output markdown file, or - for stdout")
    args = ap.parse_args()

    runs = {}
    for d in sorted(glob.glob(os.path.join(args.root, "*/"))):
        tag = os.path.basename(d.rstrip("/"))
        runs[tag] = {}
        for f in ("throughput_c1_p1k_d1k.json", "throughput_c1_p64k_d2k.json", "throughput_c4_p1k_d512.json", "aime_think_max.json"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                try:
                    runs[tag][f.replace(".json", "")] = json.load(open(p))
                except Exception as e:
                    print(f"warn: failed to parse {p}: {e}", file=sys.stderr)
        for f in ("sweep_summary.txt",):
            p = os.path.join(d, "oom_sweep", f)
            if os.path.exists(p):
                runs[tag][f.replace(".txt", "")] = open(p).read().strip().split("\n")

    if not runs:
        print(f"no runs found under {args.root}", file=sys.stderr)
        return 2

    out = []
    out.append("## Dual DGX Spark TP=2 (GB10, SM 12.1a, 2 × 121 GiB UMA, QSFP 200 Gbps)")
    out.append("")
    out.append("**Test run:** 2026-05-26. Image built via `scripts/bootstrap_dsv4_spark.sh`, no custom flags. "
               "Both pairs (S1↔S2 over `enp1s0f1np1`/192.168.1.0/30, S5↔S6 over `enp1s0f0np0`/192.168.101.0/30) "
               "tested concurrently. See `findings/dual_spark_jasl_sha_regression_2026-05-26/` for raw JSONs + per-cell logs.")
    out.append("")

    # Throughput summary table
    out.append("### Throughput by jasl/vllm SHA")
    out.append("")
    out.append("| SHA tag | c=1 P=1K D=1K aggregate tok/s | TPOT median (ms) | c=1 P=64K D=2K tok/s | c=4 P=1K D=512 aggregate tok/s |")
    out.append("|---|---|---|---|---|")
    for tag, data in runs.items():
        a = load_throughput(data.get("throughput_c1_p1k_d1k", {}))
        b = load_throughput(data.get("throughput_c1_p64k_d2k", {}))
        c = load_throughput(data.get("throughput_c4_p1k_d512", {}))
        out.append(f"| `{tag}` | {a['aggregate_tps']:.1f} | {a['tpot_median']:.1f} | {b['aggregate_tps']:.1f} | {c['aggregate_tps']:.1f} |")
    out.append("")

    # OOM table (context wall)
    out.append("### OOM threshold sweep (--gpu-memory-utilization 0.90)")
    out.append("")
    for tag, data in runs.items():
        sweep = data.get("sweep_summary")
        if not sweep:
            continue
        out.append(f"**{tag}:**")
        out.append("")
        out.append("| (ctx, seqs) | status | boot_s |")
        out.append("|---|---|---|")
        for line in sweep:
            # format: "ctxNNNN_seqM STATUS Xs"
            parts = line.split()
            if len(parts) < 3:
                continue
            cell, status, boot = parts[0], parts[1], parts[2]
            # extract ctx + seqs
            cell_clean = cell.replace("ctx", "").replace("_seq", " / seq=")
            out.append(f"| {cell_clean} | {status} | {boot} |")
        out.append("")

    # AIME / think-max
    out.append("### Long-reasoning think-max (AIME-2024, max_tokens=65536, reasoning_effort=high)")
    out.append("")
    out.append("| SHA tag | problems | completed | truncated @ 65K | avg wall (s) | avg decode tok/s | avg completion tokens |")
    out.append("|---|---|---|---|---|---|---|")
    for tag, data in runs.items():
        a = load_aime(data.get("aime_think_max", {}))
        out.append(f"| `{tag}` | {a['n']} | {a['correct']} | {a['truncated']} | {a['avg_wall']:.0f} | {a['avg_decode_tps']:.1f} | {a['avg_completion_tokens']:.0f} |")
    out.append("")

    # Summary delta
    if len(runs) >= 2:
        out.append("### Cross-SHA regression delta")
        out.append("")
        tags = list(runs.keys())
        for i in range(1, len(tags)):
            base, new = tags[0], tags[i]
            a0 = load_throughput(runs[base].get("throughput_c1_p1k_d1k", {}))
            a1 = load_throughput(runs[new].get("throughput_c1_p1k_d1k", {}))
            if a0["aggregate_tps"] > 0:
                delta_tps = 100 * (a1["aggregate_tps"] - a0["aggregate_tps"]) / a0["aggregate_tps"]
                delta_tpot = a1["tpot_median"] - a0["tpot_median"]
                out.append(f"- `{new}` vs `{base}`: aggregate tok/s **{delta_tps:+.1f}%**, TPOT median **{delta_tpot:+.1f} ms**")
        out.append("")

    out.append("**Production canonical recommendation:** _TBD_ (after data lands).")
    out.append("")

    text = "\n".join(out)
    if args.out == "-":
        sys.stdout.write(text)
    else:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
