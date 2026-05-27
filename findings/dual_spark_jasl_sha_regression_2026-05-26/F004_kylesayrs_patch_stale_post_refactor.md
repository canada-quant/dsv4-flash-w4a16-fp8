# F004 — kylesayrs-deepseek-ct.patch stale post 2026-05-19 DSV4 model refactor

**Severity:** blocking (every bootstrap invocation against jasl/vllm@ds4-sm120-preview-dev SHAs newer than 2026-05-19)
**Discovered:** 2026-05-26 during dual-spark bench run after F001-F003 unblocked
**Repo:** canada-quant/dsv4-flash-w4a16-fp8
**File:** `scripts/kylesayrs-deepseek-ct.patch`

## Repro
```
./bootstrap_dsv4_spark.sh \
  --head-host spark-A --worker-host spark-B \
  --vllm-ref ds4-sm120-preview-dev   # HEAD = 5d6479811 as of 2026-05-26
```
Build step 4-5 (eugr/spark-vllm-docker build) fails inside the Dockerfile at:
```
RUN git config --global user.email "..." && \
    git apply --check /tmp/kylesayrs-deepseek-ct.patch && \
    git am --keep-cr /tmp/kylesayrs-deepseek-ct.patch
```
with:
```
error: vllm/model_executor/layers/deepseek_compressor.py: No such file or directory
error: vllm/model_executor/layers/deepseek_v4_attention.py: No such file or directory
error: vllm/model_executor/models/deepseek_v4.py: No such file or directory
```

## Root cause
Commit `287471b99` on `jasl/vllm@ds4-sm120-preview-dev` (2026-05-19,
`[Model Refactoring] Migrate DeepSeek V4 to vllm/models/ [1/N]  (#43004)`)
relocated DSV4 code from `vllm/model_executor/{layers,models}/deepseek_*` to
`vllm/models/deepseek_v4/`. The kylesayrs-deepseek-ct.patch hardcodes the
PRE-refactor paths, so `git apply --check` fails on every SHA at or after that
commit.

## Affected SHA range
- ✅ jasl HEAD as of 2026-05-19 morning and earlier (e.g., `428e08e` 2026-05-05,
  `0a65d4662` 2026-05-14, `a8887c208` 2026-05-13): patch applies cleanly.
- ❌ 2026-05-19 afternoon onwards (e.g., `287471b99`, `a937d4b28` 2026-05-24
  "Stabilize SM12x sparse MLA long prefill", `5d6479811` 2026-05-25 HEAD):
  patch fails at git apply --check.

## Practical impact for this test run
We wanted to test 3 SHAs across the SM12x sparse-MLA work:
- BASELINE 428e08e (2026-05-05) — works
- INTERMEDIATE a937d4b28 (2026-05-24) — does NOT work post-refactor
- HEAD 5d6479811 (2026-05-25) — does NOT work post-refactor

Pivoted to:
- BASELINE 428e08e — works (Pair A)
- INTERMEDIATE 0a65d4662 (2026-05-14, "Fuse norm and router") — works (Pair B)

Loses the headline regression-vs-HEAD comparison data. Best surrogate available
on Pair B without patch redevelopment.

## Suggested upstream fix
Either:
1. **Refresh the patch**: cherry-pick the kylesayrs work onto the post-refactor
   `vllm/models/deepseek_v4/` layout, vendor as `kylesayrs-deepseek-ct-v2.patch`,
   and have the bootstrap select v1 or v2 based on `--vllm-ref` date.
2. **Verify jasl integrated kylesayrs natively**: if the post-refactor branch
   already includes the kylesayrs functionality (some sm120 fixes), the patch
   becomes unnecessary. Bootstrap should detect that and skip apply.
3. **Pin VLLM_REF default to a pre-refactor SHA** in bootstrap_dsv4_spark.sh
   (e.g., `0a65d4662`) and document that newer SHAs require manual patch
   updates.

## Workaround applied for this run
Use SHA `0a65d4662` (2026-05-14) for Pair B's intermediate test — gives us a
9-day-newer-than-baseline data point. Don't test post-2026-05-19 SHAs in this run.

Detection of refactor commit in jasl's commit log:
```
gh api 'repos/jasl/vllm/commits?sha=ds4-sm120-preview-dev&path=vllm/model_executor/models/deepseek_v4.py&per_page=10'
```
Last commit touching the OLD path: `287471b99` 2026-05-19. Any SHA newer doesn't
have that file.
