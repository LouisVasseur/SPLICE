# GRE baselines

Only the GRE baselines: GRE ([gre4index/GRE](https://github.com/gre4index/GRE), Wongkham et al., VLDB 2022) at a
pinned commit, built with its own competitors and run with its own driver. Nothing else.

- GRE commit `e807edcef51df6732f07f94d4c797fb3897519ba` (master, 2022-11-09), cloned outside this repository (GRE
  ships no licence, so none of it is committed here).
- Indexes: ALEX, LIPP, dynamic PGM, STX B+tree, ART (unsynchronised): GRE's single-threaded indexes that build on the
  reference machine (an Atom, Goldmont: no AVX2) without Intel MKL. Dropped: HOT (AVX2/BMI2), XIndex and FINEdex
  (link against MKL), and the thread-safe indexes (alexol, lippol, btreeolc, artolc, hotrowex, wormhole_u64, masstree),
  which a single-thread benchmark does not need.
- Workload: GRE's own read-only, single-thread lookup benchmark: lookups of uniformly sampled bulk-loaded keys.
- Datasets: the ten GRE/SOSD 200M-key files, checked against `data/SHA256SUMS`.

## Run

```bash
git clone --branch gre-baselines --single-branch https://github.com/LouisVasseur/SPLICE.git GRE-baselines && cd GRE-baselines
data/download.sh                       # ~16 GB, resumable, then sha256 check
scripts/build_gre.sh                   # clones + builds GRE in $HOME/gre-baselines/GRE
scripts/run_baselines.sh results/run1  # 10 datasets x all indexes x 3 repeats, 100M lookups each
```

On Linux, run `scripts/build_gre.sh` inside an activated conda environment. It installs TBB 2020.2 and jemalloc 5.4.0
into that environment, the versions GRE's CMake finds. On the Atom, set `CORE=K` to pin with `taskset`.

Smoke test (first 10M keys of each file, 1M lookups, about 2 minutes on an M3 Max). osm, books and fb are stored
sorted, so for them the first 10M keys are the 5% smallest, not a sample: a functional check, not a measurement.

```bash
REPEATS=1 TABLE=10000000 OPS=1000000 scripts/run_baselines.sh results/smoke
```

`run_baselines.sh` writes the following, then prints the summary (`scripts/summarize.py`):

- `runs.tsv`: one row per successful run, with throughput, GRE's self-reported memory and success_read;
- `logs/`: GRE's stdout and CSV;
- `MANIFEST.txt`: host, binary, the build manifest, and the settings.

A cell counts only if `success_read` equals the number of lookups. Runs are resumable; `SETTINGS` refuses a rerun
into the same directory with other lookups, key count, ratio, core or binary.

## What is changed in GRE

Build-enabling edits only, made in the clone and recorded in `build/BUILD_MANIFEST.txt`. GRE's driver and its timed
loop (`src/benchmark/benchmark.h`, `microbench.cpp`) are untouched.

- `scripts/trim_gre.py` deletes lines from `competitor.h` and `CMakeLists.txt`, keeping the competitors listed
  above. Each anchor must occur exactly once. It also removes `${MKL_LIBRARIES}` from the link line, turns one
  `else if` into `if`, and adds `-include cstdint` (GCC 11+ no longer includes it transitively).
- `ALEX_USE_LZCNT 0` when the CPU has no LZCNT/BMI1. This is ALEX's own documented switch, and it only affects the
  insert-time gap search.

## macOS (preview only)

`scripts/build_gre.sh` also builds natively on Apple-silicon Macs with Apple clang. These numbers are a preview; the
reference numbers come from the Atom. Differences from the Linux build:

- ART is not built (it needs SSE2).
- The timer reads `CNTVCT_EL0` instead of `rdtsc`.
- x86 asm that no kept index uses at run time is compiled out: ALEX's CPUID, and GRE's `cmpxchg` (XIndex).
- Shims replace the missing pieces: `_mm_popcnt_u64` (x86intrin), single-thread OpenMP, `std::sort` for
  `tbb::parallel_sort`, no jemalloc.
- libc++ instead of libstdc++, so the same `--seed` samples different lookup keys.
- **LIPP fails on covid, osm and books**: its `isfinite(model.a)` assert, or memory growth until macOS kills it
  (11 GB within seconds on 1M covid keys). `run_baselines.sh` skips these three cells on macOS. Likely cause:
  `long double` is 64-bit `double` on arm64, and these keys lie above 2^53 (covid and osm all, books 99.76%),
  where adjacent keys can round to the same double; x86's 80-bit `long double` holds them exactly.
- Time only on AC power with Low Power Mode off; `run_baselines.sh` warns otherwise and records the power state.

## Licence

The scripts here are MIT (`LICENSE`). GRE has no licence; its competitors keep their own (the STX B+tree is GPL-3).
