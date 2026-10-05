# Warm-up evidence at 200M keys (Mac, Sep 30 - Oct 4 2026)

Raw outputs behind the warm-up findings in `docs/METRICS.md` and `results/aidb_ba/PROTOCOL.md` section 3.2.
All runs used `build-fs/scaleli_bench` on fb and osm (full 200M keys) on the Apple M3 Max.

- `fullscale/cold/`, `fullscale/runs/`, `fullscale/final/`: `--prefault 0|1` x `--warmup 0|200000|2000000` x
  `--warmup-mode strided|workload`; `.time` files are `/usr/bin/time -l` output (major/minor faults, max RSS).
- `verify/A`, `verify/A2`, `verify/B`, `verify/M`: repeat runs checking the prefault and first-chunk findings, and
  memory per variant (`M`).
- Summarised by `../aidb_ba/noise.py`.

Findings: prefault is a DRAM read sweep, not a paging fix (1-2 major faults either way); with no warm-up the first
1/16 of a 5M replay runs ~19% slow; a 200k-lookup workload-mode warm-up hides it; `steady_state_ratio` cannot see a
slow first span. The host was busy (a VM among the top-3 CPU users in 98% of samples) and probably in Low Power
Mode, so the timings are provisional.
