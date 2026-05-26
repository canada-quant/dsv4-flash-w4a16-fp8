# F001 — bootstrap_dsv4_spark.sh step 2 fails on Ubuntu 24.04 (PEP 668)

**Severity:** blocking (every OOB invocation on a stock DGX Spark)
**Discovered:** 2026-05-26 during dual-spark OOB run
**Repo:** canada-quant/dsv4-flash-w4a16-fp8
**File:** `scripts/bootstrap_dsv4_spark.sh:125`

## Repro
```
./bootstrap_dsv4_spark.sh --head-host spark-N --worker-host spark-M ...
```

## Symptom
Step 2 ("Ensuring canada-quant/DeepSeek-V4-Flash-W4A16-FP8 is fully cached") fails immediately on a stock DGX Spark (Ubuntu 24.04, Python 3.12) with:
```
error: externally-managed-environment
× This environment is externally managed
```
The failing line is:
```bash
command -v huggingface-cli >/dev/null 2>&1 || pip install --quiet --user huggingface_hub
```
On Ubuntu 24.04, `pip install --user` is blocked by PEP 668 unless `--break-system-packages` is passed.
Also: the modern HF CLI binary is `hf`, not `huggingface-cli` (deprecated alias may be absent on
fresh installs).

## Workaround applied for this run
On each of the 4 head/worker sparks:
```
sudo pip install --break-system-packages -q huggingface_hub hf-transfer
```

## Suggested upstream fix
```diff
- command -v huggingface-cli >/dev/null 2>&1 || pip install --quiet --user huggingface_hub
- huggingface-cli download canada-quant/DeepSeek-V4-Flash-W4A16-FP8 >/dev/null
+ command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1 || \
+   pip install --quiet --user --break-system-packages huggingface_hub hf-transfer
+ HF_BIN=$(command -v hf || command -v huggingface-cli)
+ "$HF_BIN" download canada-quant/DeepSeek-V4-Flash-W4A16-FP8 >/dev/null
```

## Status
Workaround applied for this run. Upstream patch candidate for `dsv4-flash-w4a16-fp8` repo + same fix applies to the MTP / NVFP4 sibling repos if they have parallel bootstraps.

## Refined workaround (2026-05-26 retry)
The first attempt `sudo pip install --break-system-packages huggingface_hub` ALSO fails with:
```
ERROR: Cannot uninstall rich 13.7.1, RECORD file not found. Hint: The package was installed by debian.
```
because `huggingface_hub` pulls in a newer `rich` and tries to replace the Debian-managed one.
Final working command:
```
sudo pip install --break-system-packages --ignore-installed rich -q huggingface_hub hf-transfer
```
This places `huggingface-cli` and `hf` in `/usr/local/bin/` where the bootstrap's non-interactive
SSH PATH (`/usr/local/sbin:/usr/local/bin:/usr/sbin:...`) actually finds them.

Note: the binaries ALREADY existed in `~/.local/bin/` on the user account, but bootstrap's
`ssh ... 'command -v huggingface-cli'` doesn't see them because non-interactive SSH on Ubuntu
24.04 doesn't source `.profile` / `.bashrc` and so `~/.local/bin` isn't in PATH.

## Suggested upstream fix v2 (better)
```diff
- command -v huggingface-cli >/dev/null 2>&1 || pip install --quiet --user huggingface_hub
- huggingface-cli download canada-quant/DeepSeek-V4-Flash-W4A16-FP8 >/dev/null
+ # Add ~/.local/bin to PATH so user-local pip installs are visible to non-interactive ssh
+ export PATH="$HOME/.local/bin:$PATH"
+ if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
+   pip install --quiet --user --break-system-packages --ignore-installed rich huggingface_hub hf-transfer
+ fi
+ HF_BIN=$(command -v hf || command -v huggingface-cli)
+ "$HF_BIN" download canada-quant/DeepSeek-V4-Flash-W4A16-FP8 >/dev/null
```
