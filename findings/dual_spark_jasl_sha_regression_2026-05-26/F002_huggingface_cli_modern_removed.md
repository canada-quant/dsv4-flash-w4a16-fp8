# F002 — huggingface-cli removed from modern huggingface_hub breaks bootstrap

**Severity:** blocking (every OOB invocation with `huggingface_hub >= 0.35`)
**Discovered:** 2026-05-26 during dual-spark OOB run (after F001 workaround applied)
**Repo:** canada-quant/dsv4-flash-w4a16-fp8
**File:** `scripts/bootstrap_dsv4_spark.sh:126`

## Repro
After installing modern `huggingface_hub==0.36.2` system-wide (which F001's workaround
produced), the bootstrap re-runs step 2 and the now-existing `/usr/local/bin/huggingface-cli`
crashes with:
```
ModuleNotFoundError: No module named 'huggingface_hub.cli.deprecated_cli'
```

## Root cause
`huggingface_hub >= 0.35` removed the deprecated `huggingface_hub.cli.deprecated_cli` module
that backed the legacy `huggingface-cli` entry point. The modern CLI is `hf`.

Confirmed via Python: `import huggingface_hub.cli.deprecated_cli` fails on the live 0.36.2
install; `hf --help` works.

## Workaround applied for this run
Installed a 4-line shim at `/usr/local/bin/huggingface-cli` on all 4 head+worker sparks:
```bash
#!/usr/bin/env bash
# Compatibility shim — bootstrap_dsv4_spark.sh calls 'huggingface-cli download X >/dev/null'
# Modern huggingface_hub (>=0.35) removed the legacy CLI; redirect to 'hf'.
exec /usr/local/bin/hf "$@"
```
Verified: `huggingface-cli download canada-quant/DeepSeek-V4-Flash-W4A16-FP8 --include "*.json"`
returns the snapshot path normally.

## Suggested upstream fix
Combine with F001 fix:
```diff
- command -v huggingface-cli >/dev/null 2>&1 || pip install --quiet --user huggingface_hub
- huggingface-cli download canada-quant/DeepSeek-V4-Flash-W4A16-FP8 >/dev/null
+ # Modern huggingface_hub ships `hf`; legacy `huggingface-cli` was removed in 0.35+.
+ # Prefer `hf`; fall back to `huggingface-cli` only if `hf` is unavailable.
+ export PATH="$HOME/.local/bin:$PATH"
+ if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
+   pip install --quiet --user --break-system-packages --ignore-installed rich huggingface_hub hf-transfer
+ fi
+ HF_BIN=$(command -v hf || command -v huggingface-cli)
+ "$HF_BIN" download canada-quant/DeepSeek-V4-Flash-W4A16-FP8 >/dev/null
```

## Status
Workaround applied. Upstream PR will combine F001+F002 into one bootstrap_dsv4_spark.sh fix.
