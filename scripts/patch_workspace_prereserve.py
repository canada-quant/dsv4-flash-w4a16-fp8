#!/usr/bin/env python3
"""Pre-reserve DSV4 prefill workspace during warmup so lock_workspace() doesn't
crash on the first real request.

Root cause:
    vllm/v1/worker/gpu_model_runner.py:6151+ captures CUDA graphs (decode
    shapes only), then calls lock_workspace() at line 6185. After lock,
    workspace.py:_ensure_workspace_size raises AssertionError on growth.

    DSV4's attention_impl (deepseek_v4_attention.py:567) returns early on
    the dummy run (attn_metadata is not a dict), so _forward_prefill never
    runs during warmup, so its workspace allocations are never sized. First
    real prefill request -> tries to grow workspace -> assertion.

    Note: deepseek_v4_attention.py:170-172 carries this comment:

        # Prefill is processed in fixed-size chunks; this bounds the bf16
        # kv-gather workspace allocated at _forward_prefill (and the matching
        # profile-time reservation in attention_impl's dummy-run branch).

    "matching profile-time reservation in attention_impl's dummy-run branch"
    implies the pre-reservation was always intended; it just never landed.

Fix:
    Inject a _warmup_reserve_prefill_workspace() method on
    DeepseekV4MLAAttention that calls get_simultaneous() with worst-case
    shapes derived from max_num_batched_tokens / max_model_len / config.
    Call it from the wrapper's dummy-run early-return path so warmup sizes
    the workspace pessimistically before lock_workspace() fires.

Anchors target jasl/vllm@77bbc16 + neuralmagic/kylesayrs/deepseek-ct@f910a73a93
cherry-pick + the packed_modules_mapping patch from this same repo.

Validation: applied to vllm-w4a16-dsv4:warmup; en2zh_bus_001 (1304-tok prompt
that previously triggered the workspace lock) now passes at ~14 tok/s decode
with full CUDA graphs (vs ~3.9 tok/s under --enforce-eager workaround).
"""
import sys

F = sys.argv[1] if len(sys.argv) > 1 else (
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/deepseek_v4_attention.py"
)

with open(F) as f:
    src = f.read()

# ------------------------------------------------------------------------
# Insertion 1: helper method on DeepseekV4MLAAttention.
# Anchor: the existing forward() method definition at the start of this class.
# ------------------------------------------------------------------------
helper_anchor = """    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, ("""

helper_method = '''    def _warmup_reserve_prefill_workspace(self) -> None:
        """PATCH (paul/dsv4): pre-reserve _forward_prefill workspace at worst-case
        sizes during warmup, before lock_workspace() is called by gpu_model_runner.

        Without this, the first real prefill request fails with
        'Workspace is locked but allocation requires X MB, current size is Y MB'
        because warmup uses dummy attn_metadata that bails out before
        _forward_prefill, so the workspace is never sized for prefill geometry.
        """
        try:
            workspace_manager = current_workspace_manager()
        except AssertionError:
            # Workspace manager not initialized yet -- happens early in init.
            return
        # Worst-case bounds matching _forward_prefill at the get_simultaneous
        # call-site. M = compressed_region_size + max_gather_len, both bounded
        # by max_model_len. max_query_chunk_tokens bounded by max_num_batched_tokens.
        max_seq = int(self.max_model_len)
        compress_ratio = max(int(self.compress_ratio), 1)
        compressed_region_size = max_seq // compress_ratio
        M_worst = compressed_region_size + max_seq
        max_query_chunk_tokens = int(self.max_num_batched_tokens)
        # Triton sparse MLA path bounds query_chunk_size by a static config size.
        # Take the larger of the two so reservation is a safe upper bound.
        try:
            tsm_chunk = int(triton_sparse_mla_query_chunk_size())
        except Exception:
            tsm_chunk = max_query_chunk_tokens
        query_chunk_size = max(min(max_query_chunk_tokens, tsm_chunk), 1)
        # combined_topk: bounded by sparse_prefill_combined_topk_size with a
        # generous top_k upper bound. self.window_size is set on the impl.
        try:
            combined_topk = int(
                sparse_prefill_combined_topk_size(8192, int(self.window_size))
            )
        except Exception:
            combined_topk = 8192
        head_dim = int(self.head_dim)
        # Allocate using get_simultaneous so the workspace grows to fit. After
        # this call returns we discard the views; the underlying workspace
        # tensor stays at the new size, which is what we want.
        try:
            workspace_manager.get_simultaneous(
                ((PREFILL_CHUNK_SIZE, M_worst, head_dim), torch.bfloat16),
                ((max_query_chunk_tokens, combined_topk), torch.int32),
                ((max_query_chunk_tokens,), torch.int32),
                ((query_chunk_size, self.num_heads), torch.float32),
                ((query_chunk_size, self.num_heads), torch.float32),
                ((query_chunk_size, self.num_heads, head_dim), torch.float32),
            )
        except Exception as e:
            # If we can't pre-reserve (e.g., shapes don't match runtime path),
            # log and continue -- falling back to the current behaviour just
            # means the first real prefill may still hit the lock assertion.
            logger.warning(
                "DSV4 prefill workspace pre-reservation failed: %s. "
                "First prefill request may still hit workspace lock.",
                e,
            )

'''

assert helper_anchor in src, "helper anchor not found - vLLM may have moved forward()"
src = src.replace(helper_anchor, helper_method + helper_anchor, 1)

# ------------------------------------------------------------------------
# Insertion 2: modify wrapper's dummy-run early-return to call the helper.
# Anchor: the dummy-run guard in attention_impl.
# ------------------------------------------------------------------------
dummy_anchor = """        # Handle dummy run (no metadata).
        if not isinstance(attn_metadata, dict):
            out.zero_()
            return"""

dummy_replacement = """        # Handle dummy run (no metadata).
        if not isinstance(attn_metadata, dict):
            out.zero_()
            # PATCH (paul/dsv4): pre-reserve prefill workspace at worst-case
            # sizes so it's correctly sized before lock_workspace() runs.
            # Without this, _forward_prefill never runs during warmup, the
            # workspace is never sized for prefill geometry, and the first
            # real prefill request crashes with a workspace lock assertion.
            try:
                self.mla_attn._warmup_reserve_prefill_workspace()
            except Exception as _e:
                logger.warning(
                    "DSV4 prefill workspace pre-reservation hook failed: %s",
                    _e,
                )
            return"""

assert dummy_anchor in src, "dummy-run anchor not found - wrapper may have moved"
src = src.replace(dummy_anchor, dummy_replacement, 1)

with open(F, "w") as f:
    f.write(src)
print(f"patched {F}")
