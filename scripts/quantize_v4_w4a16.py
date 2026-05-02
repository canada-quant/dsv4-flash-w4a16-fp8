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
    FP8_BLOCK,
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

    print("[quant] configuring GPTQ mixed-precision recipe (FP8_BLOCK attn + W4A16 experts)...")
    # REVISED 2026-05-02 (Path B): match RedHat's published reference structure.
    #
    # Earlier W4A16-everywhere recipe loaded into vLLM but broke at runtime:
    # vllm/model_executor/layers/deepseek_v4_attention.py:418 hardcodes
    # `wo_a_fp8 = self.wo_a.weight` then passes it to a custom FP8 BMM einsum
    # kernel (`torch.ops.vllm.deepseek_v4_fp8_einsum`). This kernel only accepts
    # FP8 tensors. With W4A16 wo_a, `.weight` doesn't exist (it's `.weight_packed`),
    # and even if we re-aliased, the einsum kernel would reject int4 packed input.
    #
    # Path B keeps attn quantized in FP8_BLOCK (matching RedHat's NVFP4-FP8
    # reference and the kernel path the kylesayrs PR was tested against), with
    # W4A16 only on the routed experts (the bulk of the params). shared_experts
    # stay BF16. This is the published-and-validated topology.
    #
    # Note: the model's transformers-v5 internal names use `self_attn.q_a_proj`
    # etc. (NOT the post-rename `attn.wq_a` form). Recipe regexes target the
    # transformers-v5 names since that's what llmcompressor sees during
    # calibration. rewrite_for_vllm.py renames at the end.
    recipe = GPTQModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=[
                    r"re:.*self_attn\.(q_a_proj|q_b_proj|kv_proj|o_a_proj|o_b_proj)$",
                    r"re:.*self_attn\.compressor\.(gate_proj|kv_proj)$",
                    r"re:.*self_attn\.compressor\.indexer\.(gate_proj|kv_proj|q_b_proj|weights_proj)$",
                ],
                **FP8_BLOCK,
            ),
            "experts": QuantizationScheme(
                targets=[
                    r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$",
                ],
                **W4A16,
            ),
        },
        ignore=["lm_head"],
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
