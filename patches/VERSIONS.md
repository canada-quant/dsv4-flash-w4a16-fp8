# DeepSeek-V4-Flash AWQ-W4A16 — patch provenance

Generated 2026-05-01T23:12:22Z

## Repos & branches

### llm-compressor (vllm-project)
- repo: https://github.com/vllm-project/llm-compressor.git
- branch: kylesayrs/transformers-v5  (PR #2647, draft)
- HEAD:  a308bc0e02181a46567a54fcfd082c9fb89e0337
- HEAD subject: add random scripts
- file patched: src/llmcompressor/pipelines/sequential/helpers.py

### transformers (huggingface)
- installed via: pip install git+https://github.com/huggingface/transformers.git@add-deepseek-v4
- branch: add-deepseek-v4 (PR #45643, open)
- installed package: 5.8.0.dev0
- installed pkg location: /workspace/.venv-quant/lib/python3.13/site-packages/transformers
- file patched: transformers/models/deepseek_v4/modeling_deepseek_v4.py

### compressed-tensors (vllm-project)
- pre-release pin: compressed-tensors>=0.15.1a2
- installed: 0.15.1.a20260428

### torch / cuda
- 

### vLLM (jasl/ds4-sm120, in DLAMI venv at /opt/pytorch)
- repo: https://github.com/jasl/vllm.git
- branch: ds4-sm120
- HEAD: 428e08ec2b1a828e2b8223c09b643559ab9ca9ee

## Patches

### helpers.py.diff
Adds Cache-class handling to SequentialTracer.create_arg, mirroring the
existing PretrainedConfig pattern. Without this fx fails with
`NotImplementedError: argument of type: <class transformers.cache_utils.DynamicCache>`
when tracing models whose forward passes Cache objects through autowrapped
helpers. Implementation: emit a fresh empty-constructor call so the traced
graph produces a real cache instance at runtime.

### modeling_deepseek_v4.py.diff
Skips DynamicCache auto-construction when caller passes past_key_values=None.
Reason: V4-Flash config has layer_types=None, so DynamicCache(config=...) falls
back to generic DynamicLayer (no store_compression_weights), and the DSV4
compressor at modeling_deepseek_v4.py:763 then crashes calling that method.
With past_key_values left None, compressor.forward takes its cache_layer-is-None
fallback path which works for fresh calibration (no decode-style accumulation).

### quantize_v4_w4a16.py.snapshot
Adapted from kylesayrs/transformers-v5 examples/quantizing_moe/deepseek_v4_example.py:
- recipe: GPTQModifier with W4A16 (was NVFP4 + FP8_BLOCK)
- dampening_frac=0.1 (per V3-family resolution issue #1482)
- ignore: lm_head, re:.*self_attn.*, re:.*shared_experts.* (routed experts NOT ignored)
- calibration: HuggingFaceH4/ultrachat_200k with V4 manual chat encoding
- launch: torchrun --nproc-per-node 8

## How to reproduce on a fresh box
1. Clone llm-compressor at the HEAD above, checkout kylesayrs/transformers-v5.
2. Apply helpers.py.diff to src/llmcompressor/pipelines/sequential/helpers.py.
3. pip install -e . into a fresh venv.
4. pip install --pre "compressed-tensors>=0.15.1a2".
5. pip install git+https://github.com/huggingface/transformers.git@add-deepseek-v4.
6. Apply modeling_deepseek_v4.py.diff to the installed transformers package.
7. Drop quantize_v4_w4a16.py.snapshot at /workspace/quantize_v4_w4a16.py.
8. mkdir -p /workspace/offload_folder
9. sudo mount -o remount,size=1800G /dev/shm  (default 1T not enough)
10. torchrun --nproc-per-node 8 /workspace/quantize_v4_w4a16.py --samples N --output ...
