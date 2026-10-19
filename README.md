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

This branch (`gre-transform`) adds one opt-in layer on top, GRE's indexes on transformed keys: see
[Transform builds](#transform-builds-branch-gre-transform). Without `EXTRA=transforms` everything below is unchanged.

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

## Transform builds (branch `gre-transform`)

GRE's own indexes built on *transformed* keys (NFL-style: Wu et al., PVLDB 2022), plus a learned per-dataset gate that
picks the transform. The spec is `DESIGN.md`. The baselines path is unchanged: without `EXTRA`, `scripts/build_gre.sh`
writes the same GRE sources as the `gre-baselines` branch, byte for byte (`tests/test_build_unchanged.sh` checks it).

```bash
EXTRA=transforms GRE_SEED=$HOME/gre-baselines/GRE scripts/build_gre.sh $HOME/gre-transform/GRE
/usr/bin/python3 data/sample.py SRC_DIR data/samples        # SRC_DIR: the ten GRE files; writes <name>_10M
tests/run_tests.sh $HOME/gre-transform/GRE                  # unit tests under ASan/UBSan, then tests/test_flow.sh
GRE_DIR=$HOME/gre-transform/GRE DATA_DIR=data/samples DATASETS="fb_10M books_10M" FLOW_DIR=flows \
  INDEXES="lipp pw512_lipp pw4096_lipp nf_lipp gate_lipp nflsw_lipp" REPEATS=1 OPS=10000000 \
  scripts/run_baselines.sh results/transforms_smoke         # a cell counts only if success_read == OPS
```

- **Build.** With `EXTRA=transforms`, `scripts/add_transforms.py` makes two anchored additions to GRE's `competitor.h`,
  after the trim: `#include "gret/registry.h"` after the last competitor include, and an `else if` asking
  `gret::make_index` right before GRE's "Could not find" branch, so GRE's own names still match first. On Linux it
  also adds `include_directories(REPO/include)` to `CMakeLists.txt`; on macOS the compile gets `-I REPO/include`.
  `BUILD_MANIFEST.txt` gets an `extra:` line (with a cksum of `include/gret`) and a `transform indexes:` line. The
  default DEST is `$HOME/gre-transform/GRE`, and building the transforms into the baselines clone is refused.
- **Index names.** On the bases `lipp`, `btree`, `alex`, `pgm`, plus `artunsync` on Linux:
  - `pw<K>_<base>`: the piecewise-linear CDF `PwlCdf` with K equi-rank knots, K from 2 to 65536 (`pw512_lipp`);
  - `pw_<base>`: the same with `K = $TRANSFORM_K` (default 512);
  - `nf_<base>`: the monotone flow `MolFlow` read from `$TRANSFORM_FLOW`;
  - `gate_<base>`: the learned gate, which picks raw, pw512, pw4096 or nf from features of the keys at bulk load;
  - `nflsw_<base>`: NFL's switch, the non-learned baseline (pw512 if its tail conflict degree is at most raw's).

  The index names stay out of the manifest's `indexes:` line, so `INDEXES` must name them. Each transform index
  prints one `gret: ...` line to stderr (it lands in the run's log) with its key, collision and build-time counts,
  or with the gate's choice and features.
- **Contract: read-only point lookups**, GRE's `--read=1 --insert=0` workload.
  - `get()` of a bulk-loaded key returns its own payload, always. Keys whose transformed values collide go to an
    exact overflow array, searched by raw key.
  - `put`, `update` and `remove` return false, and `scan` returns 0: a transform is not order-preserving in general.
  - A lookup of a key that was never loaded may return a false positive.
  - GRE gives every key the payload 123456789, so `success_read` cannot catch a lookup that returns another key's
    payload. The tests load distinct payloads and check each one.
  - Transformed keys lie in `[0, 2^52)`: they are exact doubles, so LIPP's macOS failure above 2^53 does not apply.
    `run_baselines.sh` skips only raw `lipp` on covid, osm and books (and their samples `covid_*`, `osm_*`, `books_*`).
- **Samples.** `data/sample.py SRC_DIR OUT_DIR [--n 10000000] [--datasets "..."]` writes `<name>_10M`: n evenly
  spaced ranks of the sorted distinct keys, minimum and maximum included, in GRE format. This keeps the CDF shape,
  where `--table_size` reads a prefix. It also writes `SAMPLES.txt` (`sha256sum -c` format). This is a dev tool that
  needs numpy (on the Mac, `/usr/bin/python3`). It holds one source file in memory, 1.6 GB for 200M keys.
- **Flows.** `/usr/bin/python3 transform/train_flow.py KEYS OUT.flow [--report OUT.json] [--m 32] [--steps 2000]
  [--sample 2000000] [--threads 4]` (numpy and torch). For `run_baselines.sh`, name each flow after the dataset as
  it is run: with `FLOW_DIR=flows`, `fb_10M` reads `flows/fb_10M.flow`. A missing flow is a warning: `nf_*` fails on
  that dataset, and `gate_*` does not consider nf there. `TRANSFORM_FLOW` from the caller's environment is ignored.
  `FLOW_DIR` and `TRANSFORM_K` go into `SETTINGS`, and each flow's sha256 goes into `MANIFEST.txt` and `OUT_DIR/FLOWS`.
  A rerun into the same OUT_DIR with another flow for a dataset is refused.
  ```bash
  for d in covid genome libio osm books fb history planet stack wise; do
      /usr/bin/python3 transform/train_flow.py data/samples/${d}_10M flows/${d}_10M.flow --report flows/${d}_10M.json
  done
  ```
- **Gate.** The features come from `tools/gate_features.cpp` (`c++ -std=c++17 -O2 -Iinclude tools/gate_features.cpp
  -o build/gate_features`, then `build/gate_features FILE...`, one JSON line per file), the same code as `gate.h`.
  - The labels are Mac timings on the 10M samples: raw, pw512, pw4096 and nf, on lipp and btree, `OPS=10000000`,
    3 repeats, under the lane described in `DESIGN.md`, on AC power with Low Power Mode off.
  - `gate/train_gate.py` (leave-one-dataset-out) regenerates `include/gret/gate_model.h`. Until it has run, that
    file is the documented placeholder: always raw, or pw512 where raw is vetoed: LIPP when more than 0.1% of the
    keys are above 2^53 on macOS, and ALEX and PGM when the keys include 2^64 - 1 (fb), their sentinel.
  - GRE's own raw ALEX and PGM mishandle the key 2^64 - 1, which fb contains: on `fb_10M` with distinct payloads,
    raw ALEX returns payload 0 for it, and raw PGM segfaults when looking it up. GRE's constant payload hides the
    first, and its uniform lookups reach that one key only by chance (about 1 run in 10 at `OPS=1000000` on a 10M
    sample), so the `alex` and `pgm` baseline cells on fb rest on luck for that key.
  - Rebuild with `EXTRA=transforms` after the model changes: the manifest's `include/gret` cksum records which
    model a binary has.
- **Tests.**
  - `tests/run_tests.sh GRE_DIR` compiles each `tests/test_*.cpp` and runs it, then runs `tests/test_flow.sh`. It exits
    non-zero on any failure. `TESTS="test_pwlcdf ..."` selects tests.
    - The compile uses `-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all` with `build_gre.sh`'s include
      paths, shims and defines: on macOS `GRE_DIR/build/mac/include`, so GRE_DIR must be a Mac build; on Linux the
      activated conda environment.
    - NDEBUG is not defined, so `assert` stays on (`TEST_NDEBUG=1` defines it).
    - Each test then runs again as `<test>-O3`, compiled like the GRE binary (`-O3 -DNDEBUG`, no sanitizer), so that
      exactness is also checked on the code that is timed (`TEST_RELEASE=0` skips this pass).
    - The tests see `GRE_DIR`, `REPO_ROOT`, `SAMPLE_DIR` (default `data/samples`), `PYTHON`, `GRET_CXX` (the
      compiler with all of these flags) and `GRET_BUILD` (`build/tests`, which also holds the logs).
  - `tests/test_build_unchanged.sh [REF]` runs `build_gre.sh` into a temporary GRE clone three times: REF's script
    (default `gre-baselines`), this one without EXTRA, and this one with `EXTRA=transforms`. The compile is stubbed
    out, or real with `FULL=1`. It checks that without EXTRA every GRE edit is byte-identical to REF's, and
    `competitor.h` to the one in `$HOME/gre-baselines/GRE`. It checks that EXTRA adds exactly the registry lines, and
    that the anchors also hold on the other platform's trim.
  - Both scripts refuse to run while a timed run is active (exit 2). `run_tests.sh` runs anyway with `TESTS_BG=1`,
    at background priority.

## Licence

The scripts here are MIT (`LICENSE`). GRE has no licence; its competitors keep their own (the STX B+tree is GPL-3).
