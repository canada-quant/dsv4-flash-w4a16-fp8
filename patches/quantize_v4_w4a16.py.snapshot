#!/usr/bin/env python3
"""
DeepSeek-V4-Flash → W4A16 GPTQ quantization, adapted from kylesayrs's
deepseek_v4_example.py (vllm-project/llm-compressor PR #2647 branch
`kylesayrs/transformers-v5`).

Differences from upstream example:
- Recipe is W4A16 (not NVFP4 + FP8_BLOCK) — target is hak-uma Coder B on
  DGX Spark, where Marlin W4A16 deploys cleanly and NVFP4 needs a patch
  stack we don't want to pin operator to.
- MODEL_ID points at our flagos-converted BF16 dir.

Same as upstream:
- linearize_moe_model() converts V4's hash-routed FP4 expert modules to
  standard nn.Linear so GPTQ can calibrate them.
- Calibration: HuggingFaceH4/ultrachat_200k with V4's manual chat encoding.
- sequential_targets=["DeepseekV4DecoderLayer"] so blocks load to GPU one
  at a time during calibration.
"""
import argparse
import os

import torch
from compressed_tensors.quantization.quant_scheme import (
    W4A16,
    QuantizationScheme,
)
from compressed_tensors.offload import load_offloaded_model, init_dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modeling.moe.linearize import linearize_moe_model
from llmcompressor.modifiers.quantization import GPTQModifier


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/workspace/model-bf16",
                   help="BF16 input model dir")
    p.add_argument("--output", default=None,
                   help="output dir; default = <input basename>-W4A16-GPTQ")
    p.add_argument("--samples", type=int, default=1024,
                   help="number of calibration samples (use 16 for dry-run)")
    p.add_argument("--max-seq-len", type=int, default=512,
                   help="max calibration sequence length")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--offload-dir", default="/workspace/offload_folder")
    return p.parse_args()


def preprocess(example, tokenizer):
    """V4 has no Jinja chat template — encode manually per
    https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/encoding
    """
    BOS = "<｜begin▁of▁sentence｜>"
    EOS = "<｜end▁of▁sentence｜>"
    text = BOS
    for message in example["messages"]:
        role = message["role"]
        content = message["content"]
        if role == "system":
            text += content
        elif role == "user":
            text += f"<｜User｜>{content}"
        elif role == "assistant":
            text += f"<｜Assistant｜></think>{content}{EOS}"
    return {"text": text}


def main():
    args = parse_args()

    output = args.output
    if output is None:
        base = os.path.basename(args.input.rstrip("/"))
        output = f"/workspace/{base}-W4A16-GPTQ"

    print(f"[quant] input={args.input}")
    print(f"[quant] output={output}")
    print(f"[quant] samples={args.samples}, max_seq_len={args.max_seq_len}, batch_size={args.batch_size}")

    init_dist()

    print("[quant] loading model with offload...")
    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(
            args.input,
            torch_dtype="auto",
            device_map="auto_offload",
            offload_folder=args.offload_dir,
        )

    print("[quant] linearizing MoE experts...")
    linearize_moe_model(model)

    tokenizer = AutoTokenizer.from_pretrained(args.input)

    print("[quant] loading + preprocessing calibration dataset...")
    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"
    ds = load_dataset(
        DATASET_ID,
        split=get_rank_partition(DATASET_SPLIT, args.samples),
    )
    ds = ds.shuffle(seed=42)
    ds = ds.map(lambda ex: preprocess(ex, tokenizer))

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    print("[quant] configuring GPTQ W4A16 recipe...")
    # Ignore patterns adapted from V3-family resolution (issue #1482):
    #   - lm_head: never quantize the output head
    #   - self_attn.*: V4's CSA/HCA attention is structurally a separate quant target;
    #     leave it BF16 here (a follow-on FP8_BLOCK pass can target it later if needed).
    #   - shared_experts.*: shared MLP — historically benefits from staying BF16.
    # Routed experts are NOT ignored — those are the bulk of the model and the
    # main W4A16 target.
    recipe = GPTQModifier(
        config_groups={
            "default": QuantizationScheme(
                targets=["Linear"],
                **W4A16,
            ),
        },
        ignore=[
            "lm_head",
            "re:.*self_attn.*",
            "re:.*shared_experts.*",
        ],
        dampening_frac=0.1,
    )

    print("[quant] running oneshot calibration...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.samples,
        sequential_targets=["DeepseekV4DecoderLayer"],
        batch_size=args.batch_size,
    )

    print(f"[quant] saving to {output}...")
    model.save_pretrained(output, save_compressed=True)
    tokenizer.save_pretrained(output)
    print(f"[quant] DONE. Output at {output}")


if __name__ == "__main__":
    main()
