# Warm-up traces on the Mac, fb, 2026-10-12 (previews; counts of chunks, not headline timings)

`trace_<index>` runs with `GRE_TRACE=geo:64:65536` (chunks of 64, 128, ... 65,536 lookups, then 65,536), no warm-up,
120M lookups, analysed by `scripts/warmup_detect.py` (`warmup.md`, `warmup.json`, `plots/` in each folder).

| Folder | Runs | Conditions |
|---|---|---|
| `fb_w0/` | 7 indexes x 3 rounds | open machine (Discord, ChatGPT, browsers busy), no CPU column |
| `fb_w0_cpu_open/` | nullindex, splice, lipp, pgm x 3 | open machine, `GRE_TRACE_CPU=1` |
| `fb_w0_cpu_lane/` | the same x 2 | inside `scripts/mac_lane.sh` (heavy apps on the efficiency cores); round 3 and the page-fault round were refused by the busy check while verification jobs ran |
| `open_vs_lane.*` | | `warmup_detect.py pair fb_w0_cpu_open fb_w0_cpu_lane` |

The power state was not recorded for these batches (later runs showed the Mac on battery in Low Power Mode, which
slows it 1.3-2x), so compare shapes and jitter, not absolute latencies.

- LIPP has a large cold transient: ~5-10M lookups to settle, 7-8 s of extra time in a 120M run, relaxation tau
  ~0.5-1.3M lookups, about the number of its 16 KiB pages (27 GB / 16 KiB = 1.6M) per page touched per lookup: the first
  touch of its pages (zero-fill or the macOS compressor). ALEX shows the same at its size (tau ~2-3e5). PGM, B+tree and
  the sorted array show no warm-up beyond noise; SPLICE (pre-touched arena) is flat or settles within ~1e5-1e6.
- The open machine has two latency states in the steady phase (about 1.6-1.8x apart, the same ratio for every index):
  machine-level, not the index. Inside the lane the robust CoV halves (lipp 37 -> 15%, pgm 25 -> 14%, splice 20 -> 11%,
  harness 17 -> 5%) and splice and the harness lose the second state. The benchmark stays on the performance cores in
  both, split about evenly between the two performance clusters (CPU 4-9 and 10-15, each with its own 16 MB L2).
- The 20M warm-up of the headline protocol covers every transient seen here.
