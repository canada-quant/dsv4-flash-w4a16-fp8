#!/usr/bin/env python3
"""
AWQ-W4A16 quantization of DeepSeek-V4-Flash for hak-uma Coder B deployment.

Source: BF16 model produced in Phase 2 (flagos converter from native FP4/FP8).
Calibration: 256 code samples (the-stack-smol Python). Code-heavy because the
target deployment is hak-uma's Coder B role.

CLI:
  --samples N        override calibration sample count (16 for dry-run, 256 full)
  --input PATH       BF16 input model dir
  --output PATH      AWQ output dir
  --max-seq-len N    sequence length cap for calibration
"""
import argparse
import os
import torch
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/workspace/model-bf16")
    p.add_argument("--output", default="/workspace/model-awq-w4a16")
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--max-seq-len", type=int, default=2048)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[quant] input={args.input} output={args.output} samples={args.samples}")

    print("[quant] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)

    print("[quant] Loading model (bfloat16, device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.input,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("[quant] Loading calibration data (the-stack-smol, python)...")
    code_ds = load_dataset(
        "bigcode/the-stack-smol",
        data_dir="data/python",
        split=f"train[:{args.samples}]",
    )

    def tokenize(ex):
        out = tokenizer(
            ex["content"],
            truncation=True,
            max_length=args.max_seq_len,
            return_tensors="pt",
        )
        return {"input_ids": out["input_ids"][0], "attention_mask": out["attention_mask"][0]}

    ds = code_ds.map(tokenize, remove_columns=code_ds.column_names)

    print("[quant] Configuring GPTQModifier W4A16...")
    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "lm_head",
            "re:.*router.*",
            "re:.*gate.*",
            "re:.*shared_expert.*",
            "re:.*indexer.*",
            "re:.*mla.*",
        ],
        dampening_frac=0.01,
        block_size=128,
    )

    print("[quant] Running oneshot...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.samples,
        output_dir=args.output,
    )

    print(f"[quant] DONE. Saved to {args.output}")


if __name__ == "__main__":
    main()
