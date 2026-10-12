# Paired warm-up batch, tag 2026-10-12

`./run.sh warmup --tag 2026-10-12` on each machine; compare with
`./run.sh pair results/paired/2026-10-12/atom/runs results/paired/2026-10-12/<mac>/runs --labels atom,mac`.

- `mac-Apple-M3-Max-battery/`: the Mac ran on battery in Low Power Mode (machine.txt: 'Battery Power'; pmset
  powermode 1): steady latencies 1.3-2x above the plugged-in runs, so not a valid Mac side for the pair. The warm-up
  shapes and the lane's effect are still usable (one latency state per index, P0/P1 split ~50/50, LIPP's page-touch
  transient ~1e7 lookups). `scripts/mac_lane.sh` now refuses timed runs on battery or in Low Power Mode.
- The valid Mac side is to be rerun plugged in, into `mac-Apple-M3-Max/`; the Atom side goes into `atom/`.
