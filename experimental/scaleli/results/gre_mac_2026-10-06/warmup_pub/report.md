# GRE baselines: warmup_pub

## Provenance

| | |
|---|---|
| date (UTC) | 2026-10-06T02:33:44Z |
| host | Louiss-MacBook-Pro.local |
| kernel | Darwin Louiss-MacBook-Pro.local 24.6.0 Darwin Kernel Version 24.6.0: Fri Feb 27 19:34:51 PST 2026; root:xnu-11417.140.69.709.8~1/RELEASE_ARM64_T6031 arm64 |
| CPU | (not recorded) |
| cores (nproc) | (not recorded) |
| governors | (not recorded) |
| free -g | (not recorded) |
| load average at start | 4:33  up 10 days,  2:16, 5 users, load averages: 4.51 16.75 31.07 |
| SPLICE | (not recorded) |
| GRE build | GRE e807edcef51df6732f07f94d4c797fb3897519ba, trimmed by SPLICE gre_lite.sh: competitor.h and CMakeLists.txt replaced (originals *.full) |
| ALEX | ALEX_USE_LZCNT=0 (1 = as shipped) |
| compiler | Apple clang version 17.0.0 (clang-1700.0.13.5) |
| -march | (not recorded) |
| runtime libraries (ldd) | (not recorded) |
| transparent huge pages | (not recorded) |
| numa_balancing | (not recorded) |
| jemalloc options | MALLOC_CONF=(unset) |
| binary sha256 | 6ddeaf856bf5423b8fbc7caa105eb35683cfd1e1b8833da314ca60d9221030cc |
| resumed | 2026-10-06T03:19:21Z |
| condition | with warm-up (datasets=fb,osm,books,stack.sorted, indexes=sortedarray,pgm,lipp,btree,alex, repeats=3, ops=100000000, warmup=20000000, pin=-1, init_ratio=1, table_size=-1, read=1, insert=0, extra=) |

Command: `/private/tmp/claude-501/-Users-louisvasseur-Downloads-scaleli-sota/ac6d4251-6384-4db6-8aba-a17e1917a2ab/scratchpad/run200/gre_run.sh /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/results/gre_mac_2026-10-06/warmup_pub --gre /private/tmp/claude-501/-Users-louisvasseur-Downloads-scaleli-sota/ac6d4251-6384-4db6-8aba-a17e1917a2ab/scratchpad/gre_mac/GRE --data /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli/data/external/gre --datasets fb\,osm\,books\,stack.sorted --indexes sortedarray\,pgm\,lipp\,btree\,alex --repeats 3 --warmup 20000000 --pin -1`

## Correctness

60 runs of 60 planned; 6 fail a check. A run counts only if it exits 0, prints its throughput, finds every looked-up key when read-only (success_read == operations_num), finds every warm-up key (warmup_success_read == warmup_num) and, when pinning was requested, reports the core.

| dataset | index | runs | exit 0 | throughput | success_read | warm-up | pin | status |
|---|---|---|---|---|---|---|---|---|
| fb | sortedarray | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| fb | pgm | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| fb | lipp | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| fb | btree | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| fb | alex | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| osm | sortedarray | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| osm | pgm | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| osm | lipp | 3 | 0/3 | 0/3 | 0/3 | 0/3 | n/a | FAIL |
| osm | btree | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| osm | alex | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| books | sortedarray | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| books | pgm | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| books | lipp | 3 | 0/3 | 0/3 | 0/3 | 0/3 | n/a | FAIL |
| books | btree | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| books | alex | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| stack.sorted | sortedarray | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| stack.sorted | pgm | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| stack.sorted | lipp | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| stack.sorted | btree | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |
| stack.sorted | alex | 3 | 3/3 | 3/3 | 3/3 | 3/3 | n/a | PASS |

Failed runs:

- r1 osm lipp (`logs/osm__lipp__r1.log`): exit 137; no Throughput line; success_read None != operations_num 100000000; warmup_success_read None != warmup_num 20000000; no warmup_ns
- r1 books lipp (`logs/books__lipp__r1.log`): exit 137; no Throughput line; success_read None != operations_num 100000000; warmup_success_read None != warmup_num 20000000; no warmup_ns
- r2 osm lipp (`logs/osm__lipp__r2.log`): exit 137; no Throughput line; success_read None != operations_num 100000000; warmup_success_read None != warmup_num 20000000; no warmup_ns
- r2 books lipp (`logs/books__lipp__r2.log`): exit 137; no Throughput line; success_read None != operations_num 100000000; warmup_success_read None != warmup_num 20000000; no warmup_ns
- r3 osm lipp (`logs/osm__lipp__r3.log`): exit 137; no Throughput line; success_read None != operations_num 100000000; warmup_success_read None != warmup_num 20000000; no warmup_ns
- r3 books lipp (`logs/books__lipp__r3.log`): exit 137; no Throughput line; success_read None != operations_num 100000000; warmup_success_read None != warmup_num 20000000; no warmup_ns

Load average (1 min) at the start of the runs: min 2.94, median 4.67, max 15.33.
1 runs started while another benchmark was running: r1 fb alex.

## Results per dataset

Throughput in Mops/s over the runs that pass every check (GRE times one pass over the operations per run). "vs sortedarray" is the geometric-mean ratio with its 95% CI. Index RSS = RSS after bulk load minus RSS before it (both after an allocator purge), per bulk-loaded key; "self-reported" is GRE's `Memory:` per key. "run growth" is RSS after the timed run minus RSS after the build, per key. "unpurged" is RSS right after bulk load, before the purge, minus RSS before the build, per key (build temporaries the allocator still holds included). "peak GB" is the process's peak RSS (VmHWM), GRE's own arrays included. "SD ln" is the cell's own SD of ln throughput; * marks one above 2x the pooled SD, whose CI assumes equal variance and is then too narrow.

### fb

Bulk-loaded keys: 200000000; pooled SD of ln throughput 0.1456 (df 10, t 2.228).

| index | runs | mean Mops/s | min | max | SD ln | vs sortedarray [95% CI] | index RSS B/key | unpurged B/key | self-reported B/key | run growth B/key | peak GB | build s | warm-up Mops/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sortedarray | 3 | 0.901 | 0.783 | 0.998 | 0.1236 | 1 (reference) | 16.01 | 16.23 | 16.00 | 0.00 | 11.69 | 0.90 | 1.044 |
| pgm | 3 | 3.187 | 2.906 | 3.737 | 0.1440 | 3.530 [2.708, 4.601] | 16.18 | 16.37 | 16.18 | 0.00 | 11.40 | 4.72 | 3.214 |
| lipp | 3 | 6.296 | 6.002 | 6.527 | 0.0429 | 7.019 [5.385, 9.148] | 134.53 | 135.29 | 120.51 | 0.06 | 38.46 | 7.35 | 4.891 |
| btree | 3 | 1.423 | 1.210 | 1.840 | 0.2400 | 1.556 [1.194, 2.028] | 19.38 | 19.44 | 0 (not reported) | 0.00 | 12.04 | 1.15 | 1.423 |
| alex | 3 | 1.988 | 1.854 | 2.231 | 0.1030 | 2.210 [1.696, 2.880] | 24.01 | 24.08 | 23.40 | 0.00 | 13.00 | 41.04 | 2.244 |

Memory-method check: sortedarray index RSS 16.013 B/key against 16 B/key expected (+0.08%, tolerance 2%): OK

### osm

Bulk-loaded keys: 200000000; pooled SD of ln throughput 0.1486 (df 8, t 2.306).

| index | runs | mean Mops/s | min | max | SD ln | vs sortedarray [95% CI] | index RSS B/key | unpurged B/key | self-reported B/key | run growth B/key | peak GB | build s | warm-up Mops/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sortedarray | 3 | 0.797 | 0.775 | 0.833 | 0.0395 | 1 (reference) | 16.01 | 16.17 | 16.00 | 0.00 | 11.69 | 0.89 | 0.772 |
| pgm | 3 | 2.911 | 2.467 | 3.625 | 0.2054 | 3.603 [2.723, 4.766] | 16.12 | 16.18 | 16.11 | 0.00 | 11.39 | 4.67 | 3.031 |
| lipp | 0 | (every run failed) | | | | | | | | | | | |
| btree | 3 | 1.298 | 1.224 | 1.444 | 0.0951 | 1.624 [1.228, 2.149] | 19.38 | 19.44 | 0 (not reported) | 0.00 | 12.04 | 1.19 | 1.221 |
| alex | 3 | 2.487 | 2.194 | 3.055 | 0.1886 | 3.085 [2.332, 4.081] | 24.51 | 24.84 | 23.41 | 0.00 | 13.07 | 38.56 | 2.521 |

Memory-method check: sortedarray index RSS 16.010 B/key against 16 B/key expected (+0.06%, tolerance 2%): OK

### books

Bulk-loaded keys: 200000000; pooled SD of ln throughput 0.1390 (df 8, t 2.306).

| index | runs | mean Mops/s | min | max | SD ln | vs sortedarray [95% CI] | index RSS B/key | unpurged B/key | self-reported B/key | run growth B/key | peak GB | build s | warm-up Mops/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sortedarray | 3 | 0.904 | 0.779 | 1.149 | 0.2222 | 1 (reference) | 16.01 | 16.23 | 16.00 | 0.00 | 11.69 | 0.87 | 0.940 |
| pgm | 3 | 3.966 | 3.570 | 4.452 | 0.1116 | 4.442 [3.419, 5.771] | 16.07 | 16.07 | 16.06 | 0.00 | 11.38 | 3.60 | 3.671 |
| lipp | 0 | (every run failed) | | | | | | | | | | | |
| btree | 3 | 1.383 | 1.213 | 1.551 | 0.1231 | 1.548 [1.192, 2.011] | 19.38 | 19.44 | 0 (not reported) | 0.00 | 12.04 | 1.19 | 1.196 |
| alex | 3 | 6.340 | 6.219 | 6.440 | 0.0177 | 7.130 [5.489, 9.264] | 23.19 | 23.19 | 23.04 | 0.00 | 12.80 | 27.46 | 6.660 |

Memory-method check: sortedarray index RSS 16.012 B/key against 16 B/key expected (+0.07%, tolerance 2%): OK

### stack.sorted

Bulk-loaded keys: 200000000; pooled SD of ln throughput 0.1323 (df 10, t 2.228).

| index | runs | mean Mops/s | min | max | SD ln | vs sortedarray [95% CI] | index RSS B/key | unpurged B/key | self-reported B/key | run growth B/key | peak GB | build s | warm-up Mops/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sortedarray | 3 | 0.839 | 0.777 | 0.930 | 0.0949 | 1 (reference) | 16.01 | 16.20 | 16.00 | 0.00 | 11.69 | 0.83 | 0.825 |
| pgm | 3 | 4.774 | 4.674 | 4.973 | 0.0357 | 5.709 [4.488, 7.261] | 16.01 | 16.01 | 16.00 | 0.00 | 11.37 | 2.84 | 4.724 |
| lipp | 3 | 26.817 | 22.744 | 33.103 | 0.1979 | 31.652 [24.883, 40.262] | 44.82 | 45.08 | 32.00 | 0.00 | 20.22 | 1.92 | 21.913 |
| btree | 3 | 1.498 | 1.253 | 1.777 | 0.1749 | 1.774 [1.394, 2.256] | 19.38 | 19.44 | 0 (not reported) | 0.00 | 12.04 | 1.06 | 1.420 |
| alex | 3 | 8.975 | 8.216 | 9.761 | 0.0862 | 10.710 [8.420, 13.623] | 23.44 | 23.44 | 23.04 | 0.00 | 12.86 | 25.25 | 10.178 |

Memory-method check: sortedarray index RSS 16.012 B/key against 16 B/key expected (+0.08%, tolerance 2%): OK

## Notes

- Index RSS counts what the index holds after bulk load: nodes, its own copies of keys and payloads, and allocator slack. It needs the gre_lite.sh patch; unpatched logs leave it empty.
- GRE's self-reported `Memory:` is each wrapper's own accounting, taken after the timed run: the STX B+tree wrapper reports 0, and ART (artunsync) excludes the 16-B key/payload records it points to.
- A sorted array of (key, payload) pairs needs exactly 16 B/key, which is what the memory-method check compares against. Before building, sortedarray also allocates and frees 16 B/key in small blocks, so OK means both that the RSS delta tracks one resident allocation and that the purge released freed small blocks. It does not show that an index's RSS holds no other slack.
- Index RSS (purged, around the build) is not SCALE-LI's "real B/key" (aidb_ba/memory_report.py: unpurged RSS plateau during the run minus nominal buffers). The closer match is "unpurged"; compare like with like only.
- Throughput CIs pool the SD over the cells of a dataset (equal variance). With few repeats a cell flagged * has a CI that understates its spread; read its min and max.

## Warnings

- WARN 1 runs started beside another benchmark process; their timings may be disturbed
