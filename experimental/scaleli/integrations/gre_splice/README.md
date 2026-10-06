# SPLICE-H in GRE

GRE (gre4index/GRE @ e807edc, Wongkham et al., VLDB 2022) is the harness the learned-index literature uses. These
files let GRE's `microbench`, as built by `tools/gre_lite.sh`, run **SPLICE-H**, the read-only index of
`include/splice/` (design: `docs/SPLICE_DESIGN.md`), next to ALEX, LIPP, PGM, the B+tree, ART, `sortedarray` and the
SCALE-LI cells.

**Not the SCALE-LI cells.** `scaleli_splice` (alias of `scaleli_jg0`) is SCALE-LI's joint-root cell Jg0 from
`integrations/gre/`, a different index; it keeps its name because earlier server runs use it. The SPLICE index is
`splice`.

| File | What it is |
|---|---|
| `splice_gre.hpp` | Facade header, C++17-clean: std types plus `splice/layout.hpp` for `splice::View`. An opaque handle with create / bulk_load / view / total_bytes. |
| `splice_gre.cpp` | The facade, C++20: parses `SPLICE_ARGS` (`splice/params.hpp`), builds `splice::Index` (`splice/build.hpp`) directly over GRE's pair array and prints the `splice_*` lines. No `get()` here. |
| `splice_interface.h` | GRE's `indexInterface<K,P>` for `splice` and `splice_thp`, C++17. Copies the View after bulk_load; `get()` is `splice::get(v_, key, val)`, inlined. |
| `trace_interface.h` | `TraceInterface<K,P,Inner>`, C++17: wraps an index (`trace_splice`, `trace_lipp`) and timestamps every 1,000,000th `get()` for the warm-up transient check (`tools/gre_transient.py`). Never a headline number. |
| `null_interface.h` | `NullInterface<K,P>` (`nullindex`): an O(1) `get()` that touches no memory and reports every key found (payload = key). The harness baseline subtracted from perf counters (SERVER.md 5b step 8) and the harness residual of the calibration kit. Never a result. |
| `splice_gre_check.cpp` | C++17 driver (CMake target `splice_gre_check`, test `splice_gre_check`): loads a SOSD file as GRE does (sort, unique), builds through the facade, checks every key and 1M absent keys with `splice::get()` on a View copy, prints the layout hash and wall times. |

## Why two translation units

GRE compiles its whole benchmark as one C++17 TU, and ALEX in that TU uses `std::allocator` members that C++20
removed. The SPLICE builder (params, cost model, residency simulator, fill) is C++20, so it sits behind a handle in its
own library, as SCALE-LI's facade does. Unlike SCALE-LI, the lookup does **not** cross the boundary:
`splice/layout.hpp` is C++17 and self-contained (POD structures and an integer-only `get()`), so GRE's TU includes it
and `SpliceInterface::get` inlines the whole lookup. Each GRE operation pays its usual virtual call and nothing more:
no cross-TU call, no try/catch, no allocation, and no store except the payload of a hit (`tools/splice_asm_check.sh`
checks the generated code of the same `get()`).

## GRE names

| GRE index | What it is |
|---|---|
| `splice` | SPLICE-H, read-only. Writes (`put`, `update`, `remove`) return false and `scan` returns 0. `--thread_num` must be 1. |
| `splice_thp` | The same layout (selected on the 4 KiB arm), plus `madvise(MADV_HUGEPAGE)` on the arena before it is written (Linux; recorded as `splice_arena_anon_huge_bytes`). Isolates the page-size effect. |
| `trace_splice` | `splice` inside the chunk timer. |
| `trace_lipp` | `lipp` inside the chunk timer: the paired reference for the transient check. |
| `nullindex` | GRE's harness alone (op sampling, the operations array, the virtual call, the timer). |

## Environment

- `SPLICE_ARGS`: whitespace-separated `key=value` tokens (`splice/params.hpp`, `docs/SPLICE_DESIGN.md` section 4), a
  later token overriding an earlier one. `gre_run.sh` passes the per-dataset plan file first
  (`results/splice_h/args/<ds>.args`, written by `splice_count` with `eps` explicit so the build runs one PLA pass)
  and the global `SPLICE_ARGS` last. Unknown keys or bad values are errors.
- `SPLICE_BUILD_THREADS`: build threads (decimal, 1..1024, default 16), used when `SPLICE_ARGS` leaves `threads=0`.
  The layout does not depend on it.
- The index name sets `thp` (`splice` 0, `splice_thp` 1); a conflicting `thp=` in `SPLICE_ARGS` is an error.

Every error prints `splice_error: <message>` to stderr and exits with status 2 (GRE itself exits 0 on a bad index).

## Log lines

Printed once after bulk_load, each on its own line and flushed (`gre_run.sh` and `gre_report.py` read them):

```
splice_index: splice|splice_thp
splice_args_effective: <every parameter, canonical k=v form>
splice_build_threads: <int>
splice_keys: <n>
splice_bulk_load_ns: <build only>
splice_hash_ns: <layout and full hash>
splice_layout_hash: 0x<16 hex>      payload-free: equal to splice_count's for the same keys and cell
splice_full_hash: 0x<16 hex>
splice_total_bytes: <arena + records + router + exceptions>
splice_arena_bytes: <int>
splice_arena_anon_huge_bytes: <AnonHugePages of the arena mapping, -1 off Linux>
splice_predicted_ed_milli: <model E[D] in milli serialized DRAM accesses, 4 KiB pages, 2 MB L2>
splice_stats: <one-line JSON, integers only>
```

After the run, from `memory_consumption()`: `splice_total_bytes_after_run: <int>`. The trace wrappers print
`trace_chunk_size: 1000000`, `trace_calls: <get calls>`, `trace_chunks_n: <N>` and N lines `trace_chunk: <k> <ns>`.

## Fairness of the timed region

`bulk_load` runs inside GRE's `build_ns` window and does everything GRE already allows a bulk load to do: it builds,
writes the whole arena (so every page is touched once, with `madvise` first for `splice_thp`), and reads the router
and record lines once at the end. No read-side state exists, so this is the same with and without GRE's warm-up and
moves no lookup work out of the timed region. GRE's own arrays and the lookup trace are untouched. The build is
multi-threaded (`SPLICE_BUILD_THREADS`, default 16) and includes the hash pass in GRE's `build_ns`, while the
competitors build on one thread under `OMP_NUM_THREADS=1`: compare build times only from a round with
`--build-threads 1` and `SPLICE_ARGS=hash=0` (`docs/SPLICE_DESIGN.md` section 4.5).

## Wiring

`tools/gre_lite.sh` copies `splice_interface.h`, `trace_interface.h`, `null_interface.h`, `splice_gre.hpp` and
`splice_gre.cpp` into GRE's `src/competitor/splice/`, registers the five names in `competitor.h`, compiles `splice_gre.cpp` as a C++20
library with `-ffp-contract=off`, adds `SCALELI_INCLUDE` to microbench's include path (only `splice/layout.hpp` is
included there) and records every copied file's cksum and the `include/splice` headers' cksum in
`build/gre_lite_build.txt`. Server commands: `SERVER.md` section 5b.
