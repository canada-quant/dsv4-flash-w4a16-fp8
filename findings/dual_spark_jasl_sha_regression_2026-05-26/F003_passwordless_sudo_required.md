# F003 — bootstrap step 3 requires passwordless sudo for `ip`

**Severity:** blocking (step 3 of bootstrap_dsv4_spark.sh after step 2 download succeeds)
**Discovered:** 2026-05-26 during dual-spark OOB run

## Repro
After F001+F002 workaround applied and downloads progressing, the bootstrap will hit
step 3:
```bash
ssh $SSH_OPTS "$H" "
  sudo ip addr replace ${HEAD_QSFP_IP}/30 dev ${QSFP_IFNAME}
  sudo ip link set dev ${QSFP_IFNAME} mtu 9000
  sudo ip link set dev ${QSFP_IFNAME} up
"
```
On a stock DGX Spark, `pcozz` does not have passwordless sudo. The non-interactive
ssh session can't supply a password → step 3 fails.

## Workaround applied for this run
Added `/etc/sudoers.d/pcozz-dsv4-bench` on all 4 head+worker sparks:
```
pcozz ALL=(root) NOPASSWD: /usr/bin/ip, /usr/sbin/ip, /usr/bin/netplan, /usr/sbin/netplan
```

## Suggested upstream fix
The bootstrap doc should note that the SSH user needs passwordless sudo for `ip`,
OR the bootstrap should support `--skip-network` flag (which already exists) and
prompt the user to pre-configure QSFP manually.

Better: provide a one-time `scripts/grant_qsfp_sudo.sh` helper that the user runs
ONCE per spark with their password to install the sudoers fragment.
