# GRE baselines, and transform builds on them (branch `gre-transform`)

The GRE baselines: GRE ([gre4index/GRE](https://github.com/gre4index/GRE), Wongkham et al., VLDB 2022) at a pinned
commit, built with its own competitors and run with its own driver. On the `gre-baselines` branch, nothing else; this
branch adds one opt-in layer, described under [Transform builds](#transform-builds-branch-gre-transform).

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
  (11 GB within seconds on 1M covid keys). `run_baselines.sh` skips these three cells on macOS. The cause:
  `long double` is 64-bit `double` on arm64, and two adjacent keys that convert to the same double land in the same
  slot at every level of LIPP's bulk load. At 200M keys covid has 61,553 such pairs, osm 2,571 and books 264; the
  other seven files have none (`scripts/double_ties.py`). Keys merely above 2^53 are fine (1 ulp apart is enough);
  x86's 80-bit `long double` holds every uint64 exactly.
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
  - `pw_<base>`: the same with `K = $TRANSFORM_K` (default 512), the only names that read `TRANSFORM_K`;
  - `nf_<base>`: the monotone flow `MolFlow` read from `$TRANSFORM_FLOW`;
  - `gate_<base>`: the gate, a model from `gate_model.h` choosing raw, pw512, pw4096 or nf from features of the keys
    at bulk load; the current Mac model is constant (raw, pw512 where raw is vetoed);
  - `nflsw_<base>`: an NFL-style switch on pw512, the non-learned baseline (pw512 if its tail conflict degree is at
    most raw's; NFL itself switches to its own flow).

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
  - On a LIPP base, an unloaded key can also end the process, as on raw LIPP: LIPP's lookup `RT_ASSERT`s (exit 0)
    when the slot it reaches holds another key. GRE's workload only looks up loaded keys.
  - Transformed keys lie in `[0, 2^52)`: they are exact doubles, so LIPP's macOS failure does not apply.
    `run_baselines.sh` skips only raw `lipp`, on macOS, where it fails: covid, osm and books by name (as the
    baselines), and any other file name where `scripts/double_ties.py` finds keys equal as doubles. Among the 10M
    samples that is books_10M (8 pairs); raw lipp runs on covid_10M and osm_10M, where it is exact.
  - `memory_consumption()` is the base's figure plus the transform and overflow, or 0 ("not reported", as GRE's
    B+tree itself) on btree, so the overhead alone never reads as a total; the `gret:` line gives it as `aux_bytes`.
- **Samples.** `data/sample.py SRC_DIR OUT_DIR [--n 10000000] [--datasets "..."]` writes `<name>_10M`: n evenly
  spaced ranks of the sorted distinct keys, minimum and maximum included, in GRE format. This keeps the CDF shape,
  where `--table_size` reads a prefix. It also writes `SAMPLES.txt` (`sha256sum -c` format). This is a dev tool that
  needs numpy (on the Mac, `/usr/bin/python3`). It holds one source file in memory, 1.6 GB for 200M keys.
- **Flows.** `/usr/bin/python3 transform/train_flow.py KEYS OUT.flow [--report OUT.json] [--m 32] [--steps 2000]
  [--sample 2000000] [--threads 4]` (numpy and torch). For `run_baselines.sh`, name each flow after the dataset as
  it is run: with `FLOW_DIR=flows`, `fb_10M` reads `flows/fb_10M.flow`. A missing flow is a warning: `nf_*` fails on
  that dataset, and `gate_*` does not consider nf there. `TRANSFORM_FLOW` from the caller's environment is ignored.
  `FLOW_DIR` and `TRANSFORM_K` go into `SETTINGS`, and each flow's sha256 goes into `MANIFEST.txt` and `OUT_DIR/FLOWS`.
  A rerun into the same OUT_DIR with another flow for a dataset is refused. The trainer needs numpy >= 1.17 and torch
  (1.x or 2.x, CPU). Where torch is missing (e.g. the Atom's conda env), train on the Mac and copy the `.flow` files,
  and run `TESTS="test_pwlcdf test_index" tests/run_tests.sh` there (test_flow needs torch).
  - MolFlow's T depends on whether the build fuses a * b + c (`__FP_FAST_FMA`: the Mac and FMA x86 builds do, an Atom
    without FMA does not): about 0.1% of the keys get a T that differs by 1. Lookups are exact either way; the nf
    `gret:` line prints `fma=0/1` and `BUILD_MANIFEST.txt` a `molflow madd:` line. Do not mix nf labels across them.
  ```bash
  for d in covid genome libio osm books fb history planet stack wise; do
      /usr/bin/python3 transform/train_flow.py data/samples/${d}_10M flows/${d}_10M.flow --report flows/${d}_10M.json
  done
  ```
- **Gate.** The features come from `tools/gate_features.cpp` (`c++ -std=c++17 -O2 -Iinclude tools/gate_features.cpp
  -o build/gate_features`, then `build/gate_features FILE...`, one JSON line per file), the same code as `gate.h`.
  - The labels are Mac timings on the 10M samples: raw, pw512, pw4096 and nf, on lipp and btree, `OPS=10000000`,
    3 repeats, under the lane described in `DESIGN.md`, on AC power with Low Power Mode off.
  - `gate/train_gate.py LABELS_DIR FEATURES.jsonl --out-dir results/gate_mac_<date> --header
    include/gret/gate_model.h` labels each dataset with its fastest variant (median over repeats), fits a constant,
    a one-feature threshold rule and a depth-2 tree per base, validates them leave-one-dataset-out, and writes the
    least-regret one into `gate_model.h` (plus labels, LODO and choice tables in the out dir). Raw is never a label
    where the gate vetoes it (`raw_veto()` in `gate.h`): LIPP on macOS when two adjacent keys are equal as doubles
    (the features' `double_ties`), ALEX and PGM when the keys include 2^64 - 1 (fb), their sentinel, and ALEX when
    n / central_share > 2^30 (a far outlier overflows its int slot prediction: UB). Bases without labels (alex, pgm)
    keep the placeholder: raw, pw512 where raw is vetoed. The model type is selected by the same LODO it reports
    (no nested LODO), and oracles whose margin is below the cells' spread are flagged as noise ties.
  - The current model (`results/gate_mac_2026-10-19/`: `tables.md`, and the sources it was made from, `labels/`,
    `gatecheck/`, `flows/`, `features_10M.jsonl`): on the Mac the fastest variant is raw
    on every dataset where raw was measured, for both lipp and btree, and pw512 (within noise of pw4096) where raw
    LIPP was not run. So on these labels the gate is the hand-written rule (raw, pw512 where raw is vetoed): the
    veto is applied outside the model, LODO cannot tell the constant, the rule and the tree apart, and the learned
    gate adds 0% over always-raw. The transforms do not pay on these bases for GRE's read-only uniform lookups at
    10M keys, with the caveat that these are labels from 10M-key stride samples on one M3 Max on AC power,
    ten datasets, validated by leave-one-dataset-out only.
  - **Invalidated, to rerun.** The lipp labels predate the exact raw LIPP veto: raw lipp was never timed on covid_10M
    and osm_10M, where it works (exact on all 10M keys of each), so the lipp oracle, regrets and model there are
    invalid, and the current `gate_lipp` builds raw on those two. The btree magnitudes (oracle margins 8.0% to 22.1%,
    the NFL-style switch's 12.04%) move between binaries by more than the repeat spread (the gate check's binary ran
    pw512 2.4% to 9.9% faster than the labels, with the same PwlCdf code): cite only the direction, raw first.
    Rerun all of `lipp pw512_lipp pw4096_lipp nf_lipp btree pw512_btree pw4096_btree nf_btree gate_lipp gate_btree
    nflsw_lipp nflsw_btree` on the ten samples on one binary, 3 repeats, under the lane, then `train_gate.py` with its
    default `--veto-rule ties` (on the 2026-10-19 labels it refuses: no raw lipp on covid_10M), rebuild, and
    `check_gate.py`. The tables there were regenerated with `--veto-rule count`, the rule the labels were made with.
    Their build_ms and feature_ms predate this branch's bulk_load and double_ties changes, and their nf cells come
    from an `fma=1` build.
  - `gate/check_gate.py RUN_DIR results/gate_mac_<date>` checks `gate_*` and `nflsw_*` runs of a rebuilt binary:
    each choice must match the model, and the features printed at bulk_load must match `gate_features`.
  - GRE's own raw ALEX and PGM mishandle the key 2^64 - 1, which fb contains: on `fb_10M` with distinct payloads,
    raw ALEX returns payload 0 for it, and raw PGM segfaults when looking it up. GRE's constant payload hides the
    first, and its uniform lookups reach that one key only by chance (about 1 run in 10 at `OPS=1000000` on a 10M
    sample), so the `alex` and `pgm` baseline cells on fb rest on luck for that key.
  - Rebuild with `EXTRA=transforms` after the model changes: the manifest's `include/gret` cksum records which
    model a binary has.
- **Tests.**
  - `tests/run_tests.sh GRE_DIR` compiles each `tests/test_*.cpp` and runs it, then runs `tests/test_flow.sh`. It exits
    non-zero on any failure. `TESTS="test_pwlcdf ..."` selects tests. Pass the samples with
    `SAMPLE_DIR=$HOME/gre-transform/data10M` (default `data/samples`) and their flows with `FLOW_DIR`.
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
    that the anchors also hold on the other platform's trim, and that this branch contains REF (a baselines fix
    that was never merged would otherwise go unseen).
  - Both scripts refuse to run while a timed run is active (exit 2). `run_tests.sh` runs anyway with `TESTS_BG=1`,
    at background priority.

## Licence

The scripts here are MIT (`LICENSE`). GRE has no licence; its competitors keep their own (the STX B+tree is GPL-3).
