# F005 — bootstrap image-copy step assumes WORKER_HOST resolvable from head spark

**Severity:** blocking when WORKER_HOST is a short alias only in the caller's ssh config (not in the head spark's hostname resolution)
**Discovered:** 2026-05-26 during Pair B baseline bootstrap; head was spark-5, worker passed as `spark-6`
**Repo:** canada-quant/dsv4-flash-w4a16-fp8
**File:** `scripts/bootstrap_dsv4_spark.sh:178-181` calling `eugr/spark-vllm-docker/build-and-copy.sh -c $WORKER_HOST`

## Repro
```
./bootstrap_dsv4_spark.sh --head-host spark-5 --worker-host spark-6 ...
```
Image builds successfully on spark-5; then `build-and-copy.sh -c spark-6` runs
INSIDE the ssh session on spark-5, where `spark-6` is not a resolvable hostname
(only resolvable on basementdocker via `~/.ssh/config`). Fails with:
```
ssh: Could not resolve hostname spark-6: Temporary failure in name resolution
Copy to spark-6 failed.
Cleaning up temporary image /tmp/vllm_image.bsZXoY
```
Bootstrap exits with `set -e`.

## Root cause
`build-and-copy.sh -c $WORKER_HOST` does `ssh $WORKER_HOST ...` from inside the
head's shell. The bootstrap passes the user's CLI-friendly short hostname
(e.g., `spark-6`) which lives only in the basementdocker `~/.ssh/config`. The
head spark's DNS (Tailscale MagicDNS or /etc/hosts) doesn't recognize it.

## Workaround applied for this run
Manually transferred image post-build:
```
ssh spark-5 'docker save vllm-w4a16-dsv4:baseline-428e08e | gzip -1 | \
  ssh -o StrictHostKeyChecking=no pcozz@192.168.101.2 "gunzip | docker load"'
```
Uses the QSFP IP directly (resolvable everywhere), 200 Gbps so fast. Then
re-runs bootstrap with `--skip-build --skip-network --skip-download` which
proceeds to step 6 (launch containers).

## Suggested upstream fix
The bootstrap should pass a worker identifier that's reachable from the head.
Options:
1. Pass the QSFP IP directly (`--worker-qsfp-ip 192.168.101.2` already exists;
   reuse for the image copy step).
2. Detect short-form aliases and translate to FQDN via Tailscale (`tailscale
   status --json`).
3. Document that `--worker-host` MUST be resolvable on the head spark (not just
   on the operator's machine), and recommend using Tailscale FQDN.

Cleanest is (1) — repurposing the already-passed `--worker-qsfp-ip` for image
distribution. Bonus: image transfer over 200 Gbps QSFP instead of Tailscale.
