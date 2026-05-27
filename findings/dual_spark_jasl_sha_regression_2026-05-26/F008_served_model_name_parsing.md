# F008 — `--served-model-name A B C` only registered the LAST name on Pair B

**Severity:** low — but trips up any client that uses the canonical first name (e.g. `DSV4-W4A16-FP8`)

## Repro
Bootstrap launches vllm with:
```
vllm serve canada-quant/DeepSeek-V4-Flash-W4A16-FP8 \
  --served-model-name DSV4-W4A16-FP8 deepseek-ai/DeepSeek-V4-Flash deepseek-v4-flash
```
On Pair B, `/v1/models` returns ONLY `deepseek-v4-flash` (last name). On Pair A, returns all 3.
Both pairs run the same image. Same launch command. Different vllm flag-parsing behavior.

## Likely cause
vllm version difference in how `--served-model-name` (nargs='+') is parsed when the same
container image is launched with subtle env / arg-ordering differences. The bootstrap
passes the names as space-separated tokens in a heredoc-derived bash array; quoting may
collapse them on one path but preserve on another.

## Workaround
Pass model names individually with `--served-model-name=NAME` syntax, OR query `/v1/models`
to discover the served name(s) before sending requests.

## Test data impact
GSM8K v1+v2 on Pair B failed (100% HTTP 404) until we switched the model arg from
`DSV4-W4A16-FP8` to `deepseek-v4-flash`. Pair A's GSM8K (running concurrently) used
`DSV4-W4A16-FP8` and succeeded.
