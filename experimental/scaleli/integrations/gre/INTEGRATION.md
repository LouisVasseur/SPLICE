# Wiring SCALE-LI into the GRE build and runner

This directory holds the GRE-side code: [README.md](README.md) lists the files and defines the cells. Nothing here
edits `tools/gre_lite.sh`, `tools/gre_run.sh`, `tools/gre_report.py` or `SERVER.md`: another workflow owned them when
this was written. This file lists the edits they need. Every edit is anchored on text, never on a line number, and
`apply_integration.py` applies them mechanically.

```sh
cd experimental/scaleli
python3 integrations/gre/apply_integration.py gre_lite.sh tools/gre_lite.sh --check   # every anchor found once?
python3 integrations/gre/apply_integration.py gre_lite.sh tools/gre_lite.sh
python3 integrations/gre/apply_integration.py gre_run.sh  tools/gre_run.sh
bash -n tools/gre_lite.sh && bash -n tools/gre_run.sh
```

If an anchor moved or changed, nothing is written and the failing anchor is printed. Apply that edit by hand from
section 5, which shows the same edits as text (generated from `apply_integration.py`). The script refuses to run
twice on the same file.

Prerequisite in SPLICE: `include/scaleli/cli.hpp`, the flag parser and JSON blocks that scaleli_bench and the facade
share. `CMakeLists.txt` builds the facade as `scaleli_gre` and its checker as `scaleli_gre_check`, with ctest
`scaleli_gre_check`.

## 1. What changes where

**`tools/gre_lite.sh`** (11 edits, 5.1):

- `SCALELI_DIR`, this checkout's `experimental/scaleli`, is computed before `cd "$DEST"`.
- Step 3 copies `scaleli_interface.h`, `scaleli_gre.hpp` and `scaleli_gre.cpp` into GRE's
  `src/competitor/scaleli/`, rewritten whole on every run like `sortedarray.h`. They are SPLICE MIT code. The
  SCALE-LI headers are not copied: they are read in place through `-DSCALELI_INCLUDE`.
- The trimmed `competitor.h` includes `./scaleli/scaleli_interface.h` and registers six names: `scaleli_b`,
  `scaleli_n`, `scaleli_c`, `scaleli_nc`, `scaleli_j` and `scaleli_jg0`. The `gre_lite:` list in the error message,
  which `gre_run.sh` reads back from the binary, gains them too.
- The trimmed `CMakeLists.txt` adds the `scaleli_gre` static library:
  - `CXX_STANDARD 20`, overriding the file's global 17 for that target only;
  - GRE's global `-faligned-new -march=native -g -O3 -include cstdint` and Release `-DNDEBUG`, as for every other
    target;
  - `-ffp-contract=off`;
  - `Threads::Threads`.

  It is linked into `microbench`, whose TU stays C++17 and unchanged. No LTO: LTO would re-optimise GRE's TU and
  so change ALEX/LIPP/PGM code generation against their published numbers.
- The cmake configure line passes `-DSCALELI_INCLUDE="$SCALELI_DIR/include"`. CMake stops if
  `scaleli/index.hpp` is not under it.
- `gre_lite_build.txt` records the cksum of the three copied files, the cksum of
  `include/scaleli/*.hpp`, the SPLICE HEAD, and its count of uncommitted paths.
- The header comment and the final `built:` line name the new indexes.

**`tools/gre_run.sh`** (10 edits, 5.2):

- `--flows DIR`. The default is `<SPLICE>/experimental/scaleli/results/aidb_flowv2/flows_free`, tracked in git.
- `--build-threads N`, default 16 (run_ba.py's `SCALELI_BUILD_THREADS`).
- `period()` mirrors run_ba.py's `PERIOD`: `s512` for fb, osm and planet, `s64` otherwise. Before any run, every
  dataset must have `$FLOWS/<ds>_<period>_t2000.txt` when the index list holds `scaleli_n`, `_nc`, `_j` or `_jg0`.
- Each run gets an environment:
  - always `OMP_NUM_THREADS=1`;
  - for every `scaleli_*` index, `SCALELI_BUILD_THREADS`;
  - for the flow cells, `SCALELI_FLOW=<that file>`.

  The `cmd:` line of the log records this environment.
- `config.txt` gains `flows=`, `build_threads=` and `scaleli_args=` only when a `scaleli_*` index is in the list,
  so `--resume` on existing baseline OUTDIRs still matches.
- The index-name check reads the `gre_lite:` list with `[a-z0-9_ ]` instead of `[a-z ]`, so it sees the underscore
  names.
- A `scaleli_*` run fails if its log has no `scaleli_cell:` line or has a `scaleli_error`, on top of the existing
  checks: exit code, throughput, and `success_read == ops` when read-only.

**`tools/gre_report.py`** (optional): collect per run
`^scaleli_(root|accounted_bytes|bulk_load_ns|flow_regions|virtual_points|root_virtual): ` into a SCALE-LI table
next to `index_rss_bytes`. `^scaleli_learnability: ` is one JSON object per run;
`root_joint_status`/`root_probes_joint` say whether J's joint root was adopted and at what score.

**`SERVER.md`**: one line under the GRE runs:

> SCALE-LI cells (`scaleli_b` `_n` `_c` `_nc` `_j` `_jg0`): `gre_run.sh` passes `SCALELI_FLOW` (from `--flows`,
> per dataset) and `SCALELI_BUILD_THREADS` (`--build-threads`, default 16); a facade error prints `scaleli_error`
> and exits 2.

**`results/aidb_ba/run_ba.py`**: the J and Jg0 cells belong to the joint-root change and are not part of this
directory. Their GRE flags are in `scaleli_gre.cpp` `cell_flags()`. If run_ba.py gains them, the two lists must stay
equal.

## 2. Runner semantics and caveats

- **One writer.** Run `--thread_num=1`; `init()` refuses anything else with exit 2, because `find()` updates
  per-region write heat and splits refit the root. Run one index per process (GRE never frees an index).
- **Key set.** `--init_table_ratio=1` with `--table_size=N` loads the same keys as
  `scaleli_bench --limit N --load-ratio 1` on the same file (both read a file prefix, then sort and deduplicate).
  The payloads differ: GRE's are all 123456789, the bench's are `mix64(row)`. Neither the structure nor the memory
  depends on them.
- **The config is printed.** `scaleli_config:` gives the scaleli_bench flags of the run. Pasting them after
  `scaleli_bench --data <file> --limit N --load-ratio 1` rebuilds the same index.
  - The flow comes only from `SCALELI_FLOW`. B and C ignore it and print `scaleli_flow_file: none`.
  - `SCALELI_ARGS` may append Config flags only (workload and `--flow` flags are rejected). They come last, so
    they win. Examples: `--root-joint-kmax 1024` for a scaled k sweep, `--root-joint-order tgv`.
- **Memory.** GRE's `Memory:` for `scaleli_*` is `accounted_bytes`: the live bytes the index owns. It is the same
  number as scaleli_bench's `memory_before.accounted_bytes` and is printed again after the run. It excludes the flow
  weights (about 208 B), allocator overhead and the facade handle. The other indexes report their own scopes (ALEX:
  model + data; B+tree: 0), so cross-index claims use gre_lite's `index_rss_bytes`.
- **Build time.** `build_ns` (gre_lite) brackets the copy of GRE's pair array (16 B/key, as scaleli_bench also
  copies it), `Index::bulk_load` and the stats print. `scaleli_bulk_load_ns` is `Index::bulk_load` alone, and
  `scaleli_stats_ns` is the print.
- **Operations.**
  - `get` is `find`.
  - `put` is `upsert`, true iff the key was new. On an existing key SCALE-LI overwrites where ALEX does not; GRE
    never re-inserts a loaded key, so this is unreachable.
  - `update` is `find` then `upsert`: two traversals, false and no insert when absent. ALEX's GRE `update` is a
    no-op, so update mixes do not compare with ALEX.
  - `remove` is `erase`; `scan` is `Index::scan` copied into GRE's buffer.
- **Inserts at scale.** Every region split refits the root: NC's root smoothing at 48,829 fences, and for J a
  re-score of the carried joint root. Insert-heavy runs at 200M with C, NC or J would mostly time root refits. Keep
  mixed runs small, or use B and N.
- **Floating point.**
  - Goldmont has no FMA. `LinearModel::predict_x` calls `std::fma`, so every SCALE-LI cell pays a software fma
    (glibc) twice per lookup. This handicap predates the integration; state it when comparing with ALEX, LIPP
    and PGM.
  - `-ffp-contract=off` keeps the facade's arithmetic equal to the portable `build-fs/scaleli_bench` on any x86
    host. GCC would otherwise contract under `-march=native` on an FMA machine.
  - x86 results (80-bit long double) differ from arm64 results. Compare builds of the same architecture only.
- **J's build at 200M.** The joint fit runs inside `bulk_load` on up to `build_threads` threads, one per k. The
  joint-root spec estimates 2 to 5.5 extra minutes on the Atom on top of NC's build; this is not measured yet.
  `learnability.root_joint_build_ns` records it. Runner timeouts must allow it (gre_run.sh has none).
- **Rosetta.** Local x86 runs under Rosetta emulate x87 long double in software. C, NC and J at 1M keys took about
  9 minutes to build here, on a heavily loaded machine, against about 9 s for an arm64 scaleli_bench. Local x86 runs
  are for correctness only.

## 3. Local (Mac) build without the conda toolchain

`GRE_LITE_EDIT_ONLY=1 tools/gre_lite.sh DEST` with the edits applied, then:

```sh
F="-arch x86_64 -march=goldmont -O3 -DNDEBUG -g -faligned-new -include cstdint"
clang++ $F -std=c++20 -ffp-contract=off -I<SPLICE>/experimental/scaleli/include \
    -c src/competitor/scaleli/scaleli_gre.cpp -o build/scaleli_gre.o
clang++ $F -std=c++17 -w -D__float128='long double' -D_LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE -I<stubs> \
    src/benchmark/microbench.cpp build/scaleli_gre.o -lpthread -o build/microbench
OMP_NUM_THREADS=1 SCALELI_FLOW=<flows_free/fb_s512_t2000.txt> build/microbench --keys_file=<data>/fb \
    --keys_file_type=binary --read=1 --insert=0 --operations_num=1000000 --table_size=1000000 \
    --init_table_ratio=1 --thread_num=1 --memory --index=scaleli_n
```

`<stubs>` provides `omp.h`, `tbb/parallel_sort.h` (std::sort), a minimal `tbb/enumerable_thread_specific.h` for ART
and `jemalloc/jemalloc.h`. The x86 binary runs under Rosetta without `arch`.

To check the facade against scaleli_bench (same architecture for both):

```sh
cmake --build <build> --target scaleli_bench scaleli_gre_check
scaleli_bench --data <file> --limit 1000000 --load-ratio 1 --profile read_only --miss 0.1 --ops 200000 \
    --seed 1001 --instrument 1 --verify 1 --latency 0 --build-threads 16 --dump-trace t.csv <cell flags> > bench.json
SCALELI_FLOW=<flow> scaleli_gre_check <CELL> <file> 1000000 t.csv > facade.log
python3 integrations/gre/compare.py bench.json facade.log [GRE logs of the same cell and key set ...]
```

## 4. Validation done (2026-10-05, this Mac; GRE e807edc pristine clone plus the edited gre_lite.sh)

Facade against scaleli_bench, arm64 native, fb and osm 1M prefixes. Every cell B, N, C, NC, J and Jg0 passed
`compare.py` exactly:

- `memory_before` (all fields) and the learnability block, joint keys included;
- the read digest equal to `result_checksum` (200k reads, 10% misses, seed 1001);
- every work counter.

The bench runs used `--verify 1`. The `scaleli_config` line, run through scaleli_bench, rebuilt the same index.
Roots chosen:

| Dataset | B, N | C, NC | J | Jg0 |
|---|---|---|---|---|
| fb | raw | raw_vp | raw_vp (joint candidate lost: 2.321 against NC's 2.305) | joint_vp (k=1; 2.141 root probes per lookup against NC's 2.174) |
| osm | binary | binary | binary | joint_vp (k=64; 5.46 root probes per lookup against 7.96) |

The other checks:

- **GRE, x86_64 under Rosetta** (`-march=goldmont`, stubs, the edited gre_lite.sh copy), read-only, 1M ops at
  table_size 1M: `scaleli_b`, `_n`, `_c` and `_nc` on fb, and `scaleli_b` and `_n` on osm, gave exit 0,
  `success_read` = 1000000 and `Memory` = `scaleli_accounted_bytes`. For fb B, N and C, `compare.py` against an
  x86 scaleli_bench also passed (memory and learnability). The x86 facade_check passed fb B and N against the x86
  bench, checksum included.
- **gre_run.sh**, edited copy, runs end to end on fb and osm with sortedarray, scaleli_b and scaleli_n. It refuses
  a missing flow file before starting, and marks a run FAILED on `scaleli_error`.
- **Error paths**, each exit 2 with `scaleli_error`:
  - N without `SCALELI_FLOW`;
  - an unreadable flow file;
  - `--thread_num=2`;
  - a non-Config or unknown flag in `SCALELI_ARGS`;
  - an invalid Config.
- **Mixed GRE workload** (fb, table 200k, read 0.5 / insert 0.5, init ratio 0.5): `scaleli_b`, `_n`, `_c` and `_nc`
  match sortedarray's success counts exactly.
- **ctest** `scaleli_gre_check` passes.

Still running when this was written, results to be checked:

- x86 runs: GRE osm C and NC, GRE J and Jg0, and the x86 bench J and Jg0 references;
- the update/scan and delete mixed workloads.

The cross-TU overhead could not be measured: the machine's load average was 150 or more, so the timings were
noise. Measure it on the server with `scaleli_gre_check B <file> 1000000 --time`.

## 5. The edits

Generated from `apply_integration.py`. "Insert after/before the line" anchors on a whole line, leading spaces
included. "Replace the text" is a substring replacement. Each anchor occurs exactly once in the files as of
2026-10-05.

### 5.1 tools/gre_lite.sh

gre_lite.1 Insert after the line

```
# plus 'sortedarray', our own binary-search baseline.
```

these lines

```
# SCALE-LI (SPLICE): scaleli_b scaleli_n scaleli_c scaleli_nc scaleli_j scaleli_jg0, the protocol cells of
# results/aidb_ba/run_ba.py plus the joint G+T+V root, from integrations/gre/ (README.md 'GRE cells'). They
# read SCALELI_FLOW (per-dataset flow file, cells _n _nc _j _jg0) and SCALELI_BUILD_THREADS (default 16).
```

gre_lite.2 Insert before the line

```
cd "$DEST"
```

these lines

```
SCALELI_DIR=$(cd "$(dirname "$0")/.." && pwd)   # experimental/scaleli of this SPLICE checkout (SCALE-LI cells)
```

gre_lite.3 Replace the text

```
// Trimmed by SPLICE gre_lite.sh: ALEX, LIPP, PGM, STX B+tree, ART only (no AVX2 / MKL needed), plus SPLICE's sortedarray.
```

with

```
// Trimmed by SPLICE gre_lite.sh: ALEX, LIPP, PGM, STX B+tree, ART only (no AVX2 / MKL needed), plus SPLICE's sortedarray
// and SCALE-LI (scaleli_*).
```

gre_lite.4 Insert after the line

```
#include "./sortedarray/sortedarray.h"
```

these lines

```
#include "./scaleli/scaleli_interface.h"
```

gre_lite.5 Insert after the line

```
  else if (index_type == "sortedarray") index = new SortedArrayInterface<KEY_TYPE, PAYLOAD_TYPE>;
```

these lines

```
  else if (index_type == "scaleli_b") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("B");
  else if (index_type == "scaleli_n") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("N");
  else if (index_type == "scaleli_c") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("C");
  else if (index_type == "scaleli_nc") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("NC");
  else if (index_type == "scaleli_j") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("J");
  else if (index_type == "scaleli_jg0") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("Jg0");
```

gre_lite.6 Replace the text

```
(gre_lite: alex lipp pgm btree artunsync sortedarray)
```

with

```
(gre_lite: alex lipp pgm btree artunsync sortedarray scaleli_b scaleli_n scaleli_c scaleli_nc scaleli_j scaleli_jg0)
```

gre_lite.7 Insert after the line

```
target_link_libraries(microbench PUBLIC OpenMP::OpenMP_CXX ${JEMALLOC_LIBRARIES} ${TBB_LIBRARIES})
```

these lines

```
# SPLICE SCALE-LI: C++20 in its own TU (GRE's TU is C++17 and ALEX needs that), linked into microbench.
# -ffp-contract=off: with -march=native on an FMA host, GCC would fuse multiply-adds the portable scaleli_bench does not.
set(SCALELI_INCLUDE "" CACHE PATH "SPLICE experimental/scaleli/include")
if(NOT EXISTS "${SCALELI_INCLUDE}/scaleli/index.hpp")
  message(FATAL_ERROR "SCALELI_INCLUDE must point to SPLICE experimental/scaleli/include")
endif()
find_package(Threads REQUIRED)
add_library(scaleli_gre STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/competitor/scaleli/scaleli_gre.cpp)
set_target_properties(scaleli_gre PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_include_directories(scaleli_gre PRIVATE ${SCALELI_INCLUDE})
target_compile_options(scaleli_gre PRIVATE -ffp-contract=off)
target_link_libraries(scaleli_gre PUBLIC Threads::Threads)
target_link_libraries(microbench PUBLIC scaleli_gre)
```

gre_lite.8 Insert after the line

```
mkdir -p src/competitor/sortedarray
```

these lines

```
# SCALE-LI wrapper and facade (SPLICE, MIT), copied whole on every run; the core headers are read in place.
mkdir -p src/competitor/scaleli
cp "$SCALELI_DIR"/integrations/gre/scaleli_interface.h "$SCALELI_DIR"/integrations/gre/scaleli_gre.hpp \
   "$SCALELI_DIR"/integrations/gre/scaleli_gre.cpp src/competitor/scaleli/
```

gre_lite.9 Replace the text

```
-DTBB_ROOT_DIR="$CONDA_PREFIX" -DJEMALLOC_ROOT_DIR="$CONDA_PREFIX"
```

with

```
-DTBB_ROOT_DIR="$CONDA_PREFIX" -DJEMALLOC_ROOT_DIR="$CONDA_PREFIX" \
          -DSCALELI_INCLUDE="$SCALELI_DIR/include"
```

gre_lite.10 Insert after the line

```
  echo "  src/competitor/sortedarray/sortedarray.h: SPLICE, cksum $(cksum < src/competitor/sortedarray/sortedarray.h)"
```

these lines

```
  for f in src/competitor/scaleli/*; do echo "  $f: SPLICE, cksum $(cksum < "$f")"; done
  echo "  SCALE-LI headers $SCALELI_DIR/include/scaleli: cksum $(cat "$SCALELI_DIR"/include/scaleli/*.hpp | cksum)"
  echo "  SPLICE $(git -C "$SCALELI_DIR" rev-parse HEAD 2>/dev/null || echo unknown) ($(git -C "$SCALELI_DIR" status --porcelain -- . 2>/dev/null | wc -l | tr -d ' ') uncommitted paths in experimental/scaleli)"
```

gre_lite.11 Replace the text

```
(indexes: alex lipp pgm btree artunsync sortedarray)
```

with

```
(indexes: alex lipp pgm btree artunsync sortedarray scaleli_b scaleli_n scaleli_c scaleli_nc scaleli_j scaleli_jg0)
```

### 5.2 tools/gre_run.sh

gre_run.1 Insert after the line

```
#       [--table-size -1] [--read 1 --insert 0] [--resume] [-- extra microbench flags]
```

these lines

```
#       [--flows DIR] [--build-threads 16]   (SCALE-LI cells scaleli_b _n _c _nc _j _jg0, integrations/gre/)
```

gre_run.2 Insert after the line

```
DATASETS= INDEXES= REPEATS=3 OPS=100000000 WARMUP=0 INIT_RATIO=1 PIN=-1 TABLE_SIZE=-1 READ=1 INSERT=0 RESUME=0
```

these lines

```
# SCALE-LI: the per-dataset NFL flow files (tracked in SPLICE) and run_ba.py's 16 build threads
FLOWS=$(cd "$(dirname "$0")/.." && pwd)/results/aidb_flowv2/flows_free
BUILD_THREADS=16
```

gre_run.3 Insert after the line

```
    --insert) INSERT=$2; shift 2;;
```

these lines

```
    --flows) FLOWS=$2; shift 2;;
    --build-threads) BUILD_THREADS=$2; shift 2;;
```

gre_run.4 Insert after the line

```
range --repeats "$REPEATS" 1 1000
```

these lines

```
int --build-threads "$BUILD_THREADS"; range --build-threads "$BUILD_THREADS" 1 1024
```

gre_run.5 Insert after the line

```
extra=${EXTRA[*]+${EXTRA[*]}}"
```

these lines

```
# SCALE-LI cells: flow file per dataset with run_ba.py's PERIOD; recorded only when a scaleli_* index runs, so
# OUTDIRs of the baseline indexes resume as before.
period() { case $1 in fb|osm|planet) echo s512;; *) echo s64;; esac; }
flow_cell() { case $1 in scaleli_n|scaleli_nc|scaleli_j|scaleli_jg0) return 0;; *) return 1;; esac; }
case ",$INDEXES," in *,scaleli_*) CONFIG="$CONFIG
flows=$FLOWS
build_threads=$BUILD_THREADS
scaleli_args=${SCALELI_ARGS-}";; esac
for idx in "${IX[@]}"; do
  flow_cell "$idx" || continue
  for ds in "${DS[@]}"; do
    f=$FLOWS/${ds}_$(period "$ds")_t2000.txt
    [ -f "$f" ] && [ -r "$f" ] || die "$idx needs the flow file $f (--flows DIR)"
  done
done
```

gre_run.6 Replace the text

```
'gre_lite: [a-z ]*)'
```

with

```
'gre_lite: [a-z0-9_ ]*)'
```

gre_run.7 Insert after the line

```
  [ "$PIN" = -1 ] || [ "$pc" = "$PIN" ] || bad="${bad:+$bad, }not pinned to core $PIN"
```

these lines

```
  case $log in *__scaleli_*)  # the facade prints scaleli_cell after bulk_load, scaleli_error on any failure
    grep -q '^scaleli_cell: ' "$log" || bad="${bad:+$bad, }no scaleli_cell line"
    ! grep -q 'scaleli_error' "$log" || bad="${bad:+$bad, }scaleli_error";; esac
```

gre_run.8 Insert before the line

```
      echo "cmd: OMP_NUM_THREADS=1 ${cmd[*]}" > "$log"
```

these lines

```
      runenv=(OMP_NUM_THREADS=1)
      case $idx in scaleli_*) runenv+=(SCALELI_BUILD_THREADS="$BUILD_THREADS");; esac
      if flow_cell "$idx"; then runenv+=(SCALELI_FLOW="$FLOWS/${ds}_$(period "$ds")_t2000.txt"); fi
```

gre_run.9 Replace the text

```
      echo "cmd: OMP_NUM_THREADS=1 ${cmd[*]}" > "$log"
```

with

```
      echo "cmd: ${runenv[*]} ${cmd[*]}" > "$log"
```

gre_run.10 Replace the text

```
      OMP_NUM_THREADS=1 "${cmd[@]}" >> "$log" 2>&1 & child=$!
```

with

```
      env "${runenv[@]}" "${cmd[@]}" >> "$log" 2>&1 & child=$!
```

## Added at integration (2026-10-05)

- `scaleli_cr` (Cr: B + `--root-alpha 0.1`, CSV at the root only) and `scaleli_splice` (the same cell as `scaleli_jg0`: SPLICE as reported, compression offered free at selection) are registered too; `apply_integration.py` carries both.
