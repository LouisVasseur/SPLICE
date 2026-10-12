# GRE integration: SCALE-LI cells and SPLICE-H

GRE (gre4index/GRE @ e807edc, Wongkham et al., VLDB 2022) is the benchmark harness the learned-index literature
uses, NFL's and CSV's papers included. GRE has no licence, so it is cloned and edited outside this repository:
`scripts/gre_lite.sh` (the Atom) and `scripts/gre_mac.sh` (the Mac, [MAC.md](MAC.md)) copy the files of `src/gre/`
into the clone's `src/competitor/scaleli/` and `src/competitor/splice/`, register the indexes and build `microbench`.
`./run.sh setup gre` runs the right one for the machine; `./run.sh smoke` checks every registered index.
Two families live here: SCALE-LI as eight cells (`scaleli_*`) and SPLICE-H, the read-only index of `src/splice/`
([DESIGN.md](DESIGN.md)), as `splice` and `splice_thp`, plus the `trace_<index>` chunk timer and `nullindex`.

## Files (`src/gre/`)

| File | What it is |
|---|---|
| `scaleli_gre.hpp` | Facade header, C++11-compatible, std types only: an opaque handle with bulk_load/get/put/update/remove/scan/memory. |
| `scaleli_gre.cpp` | The facade, C++20. Builds a cell's `Config` with scaleli_bench's own parser (`scaleli/cli.hpp`) and prints the `scaleli_*` stats lines. |
| `scaleli_interface.h` | GRE's `indexInterface<K,P>` for the cells, C++17. Copied into GRE's `src/competitor/scaleli/`; it includes GRE's `../indexInterface.h` at build time, and nothing of GRE is copied here. |
| `facade_check.cpp` | C++17 driver (target `scaleli_gre_check`): builds a cell through the facade from the records scaleli_bench loads and replays a `--dump-trace` read trace (scaleli_bench: main@ca8d80e). |
| `splice_gre.hpp` | Facade header, C++17-clean: std types plus `splice/layout.hpp` for `splice::View`. An opaque handle with create / bulk_load / view / total_bytes. |
| `splice_gre.cpp` | The facade, C++20: parses `SPLICE_ARGS` (`splice/params.hpp`), builds `splice::Index` (`splice/build.hpp`) directly over GRE's pair array and prints the `splice_*` lines. No `get()` here. |
| `splice_interface.h` | GRE's `indexInterface<K,P>` for `splice` and `splice_thp`, C++17. Copies the View after bulk_load; `get()` is `splice::get(v_, key, val)`, inlined. |
| `trace_interface.h` | `TraceInterface<K,P,Inner>`, C++17: wraps an index (`trace_<index>`) and timestamps chunk boundaries of its `get()` calls. `GRE_TRACE` unset: every 1,000,000th call, for the transient check (`scripts/gre_transient.py`); `GRE_TRACE=geo:F:C`: chunks of F, 2F, 4F, ... calls up to C, then C each, and `fixed:C`: chunks of C, for `scripts/warmup_detect.py`. `GRE_TRACE_FAULTS=1` adds per-boundary page faults (getrusage; diagnosis only, not timing) and `GRE_TRACE_CPU=1` the CPU the thread is on. Never a headline number. |
| `null_interface.h` | `NullInterface<K,P>` (`nullindex`): an O(1) `get()` that touches no memory and reports every key found (payload = key). The harness baseline subtracted from perf counters (PROTOCOL.md 5b step 8) and the harness residual of the calibration kit. Never a result. |
| `splice_gre_check.cpp` | C++17 driver (CMake target `splice_gre_check`, test `splice_gre_check`): loads a SOSD file as GRE does (sort, unique), builds through the facade, checks every key and 1M absent keys with `splice::get()` on a View copy, prints the layout hash and wall times. |

`compare.py` (checks that a facade or GRE log describes the index scaleli_bench builds) and scaleli_bench itself are
on main@ca8d80e.

## Why separate translation units

GRE compiles its whole benchmark as one C++17 TU, and ALEX in that TU uses
`std::allocator` members that C++20 removed. SCALE-LI is C++20, so it sits behind a handle with a std-types-only
header, compiled as its own C++20 library and linked into `microbench`. Each GRE operation already pays one virtual
call; the facade adds one direct call into the other TU, where `Index::find` is inlined.

The SPLICE builder (params, cost model, residency simulator, fill) is C++20, so it sits behind a handle in its
own library, as SCALE-LI's facade does. Unlike SCALE-LI, the lookup does **not** cross the boundary:
`splice/layout.hpp` is C++17 and self-contained (POD structures and an integer-only `get()`), so GRE's TU includes it
and `SpliceInterface::get` inlines the whole lookup. Each GRE operation pays its usual virtual call and nothing more:
no cross-TU call, no try/catch, no allocation, and no store except the payload of a hit (`scripts/splice_asm_check.sh`
checks the generated code of the same `get()`).

## SCALE-LI cells

The GRE index names are lowercase because `gre_run.sh` accepts only `[a-z0-9_]` index names. Each name is a cell of
`main@ca8d80e:experimental/scaleli/results/aidb_ba/run_ba.py`'s `cells(d)` plus the joint root, with every cell built with
`--build-threads $SCALELI_BUILD_THREADS` (default 16, the protocol's).

| GRE index | Cell | Flags (scaleli_bench) | What it is |
|---|---|---|---|
| `scaleli_b` | B | `--policy min_bytes --routing rank --root model` | SCALE-LI before NFL and CSV: compressed blocks, rank models, a learned root over the region fences (binary search if the root model does not save probes). |
| `scaleli_n` | N | B + `--flow $SCALELI_FLOW --flow-bypass 1 --flow-cost 0` | Our NFL: the dataset's NFL flow is the region-model feature where NFL's own 10% tail-conflict switch keeps it, and the root is offered the flow candidate for free. |
| `scaleli_c` | C | B + `--virtual-alpha 0.1 --root-alpha 0.1` | Our CSV: CSV Algorithm 1 virtual points in every region (alpha 0.1) and virtual fences at the root, with a uint32 slot-to-region table. |
| `scaleli_cr` | Cr | B + `--root-alpha 0.1` | CSV at the root only: virtual fences and the slot-to-region table, regions as B. The reference for SPLICE vs CSV alone at the root. |
| `scaleli_nc` | NC | C + N's flow flags | Both, applied in sequence. |
| `scaleli_j` | J | NC + `--root-joint-rounds 6 --root-joint-order gtv --root-joint-kmin 0 --root-joint-kmax 64 --root-joint-gap-charge 1` | Our simultaneous learned compression and virtual-points model, with NFL: NC's regions bit for bit, plus a joint G+T+V root candidate. |
| `scaleli_jg0` | Jg0 | J with `--root-joint-gap-charge 0` | The same joint root, with the gap table offered for free, so measured throughput decides whether learned compression pays. |
| `scaleli_splice` | Jg0 | the same as `scaleli_jg0` | SPLICE as reported: compression and warp offered free at selection (N's convention), GRE measures their real cost. At gap charge 1 (J) compression is never chosen on the samples, so J tests only joint T+V. |

**`scaleli_splice` is not the SPLICE index.** It is SCALE-LI's joint-root cell Jg0 under an older name, kept because
earlier server runs use it. The SPLICE-H read-only index is the GRE index `splice` (and `splice_thp`), from
`src/gre/` (below; design in `DESIGN.md`).

The joint root (J, Jg0) is one least-squares model of the root fences, fitted by alternating three blocks with a
guard on the root's probe score (main@ca8d80e:experimental/scaleli/results/aidb_threeblock, arm A):

- **G**, learned compression of the input axis: the k widest fence gaps, among those wider than the median gap, are
  shrunk to the median width through a table of k entries (32 B each).
- **T**, an order-preserving transform (NFL's two-unit tanh decoder): at most two units with non-negative weights,
  gains from a 64-point log grid on [0.01, 200]. It is fitted to the current targets and must stay strictly
  increasing on the fences and fence midpoints.
- **V**, CSV virtual fences at the root, with budget `root_alpha` (0.1, as in NC). The previous virtual fences are
  kept if they fit no worse.

Each round runs G, T, V in order on the current state. The joint candidate is scored like fit_root's other root
candidates: the probes the real root lookup spends on the fences and fence midpoints, plus `gap_charge` ×
ceil(log2(k+1)) for the gap-table search. Round 1 is always accepted; later rounds are accepted only if they lower
the score by more than 1e-6, for at most 6 rounds. The cycle runs for k in {0, 1, 4, 16, 64} and keeps the
lowest-scoring k, ties going to the smaller k. The winner is appended after NC's four root candidates and adopted
only if it is strictly cheaper than all of them and than binary search. J against NC therefore isolates joint
against sequential at the root: the regions are identical, and when the joint candidate loses, J's index is NC's
plus a few hundred bytes of fit diagnostics (`root_joint_bytes`).

Why these settings, from the prototypes:

- **6 rounds, tolerance 1e-6, keep-previous V guard.** These are arm A's guarded settings verbatim (armA.py
  MAX_ROUNDS, TOL). With k = 0, round 1 is exactly the sequential T then V pass, so J can never score worse than
  sequential.
- **Order GTV.** Order has no consistent effect: TGV ends better in 25 of 80 cells, with a mean difference of 0.044
  probes. Five of the six robust cycle gains are GTV (main@ca8d80e:experimental/scaleli/results/aidb_threeblock/RESULT.md section 3). `--root-joint-order tgv`
  stays available through `SCALELI_ARGS`.
- **k grid {0, 1, 4, 16, 64}, minimum total.** The prototypes' best is the minimum over k with k = 0 included. k is
  absolute because the table's search cost, ceil(log2(k+1)), does not grow with n.
- **V budget `root_alpha` 0.1, not the prototypes' 4.** At equal budgets J against NC isolates jointness. At 200M,
  alpha 4 costs about 2,000 s per root V run, and the cycle can need 30 runs.
- **T charge `flow_cost` 0.** This is N's convention: the transform is offered free and throughput decides. J's T
  is fitted from scratch and never reads the NFL weight file, because the free flow is not monotone and cannot feed
  a slot table.
- **Gap charge 1 in J, 0 in Jg0.** At charge 1 the score is the prototypes' `total_uncharged`, and on the 2M
  samples G is then never chosen (k = 0 on 20 of 20), so J measures the joint T and V fit against the sequential
  one. A 64-entry table lives in L1 while fence probes chase pointers, so a 1:1 charge is a modelling guess. Jg0
  removes it; on the samples it then picks G on 19 of 20, so Jg0 is the cell where learned compression is actually
  exercised.

Every scaleli_* run prints, once after bulk_load, `scaleli_cell`, `scaleli_config` (the exact scaleli_bench flags;
pasting them into scaleli_bench, on main@ca8d80e, rebuilds the same index), `scaleli_flow_file`, `scaleli_build_threads`,
`scaleli_bulk_load_ns`, `scaleli_accounted_bytes`, `scaleli_memory` (scaleli_bench's `memory_before`),
`scaleli_root` (binary, raw, raw_vp, flow, flow_vp, joint or joint_vp), `scaleli_flow_regions`,
`scaleli_virtual_points`, `scaleli_root_virtual` and `scaleli_learnability` (scaleli_bench's learnability block,
including the `root_joint_*` diagnostics for J and Jg0).

### Runner semantics and caveats

`scripts/gre_lite.sh` copies `src/gre/` into GRE and registers the cells; `scripts/gre_run.sh` passes `SCALELI_FLOW` (from
`--flows`, default `data/flows`, per dataset: `s512` for fb, osm and planet, `s64` otherwise) and
`SCALELI_BUILD_THREADS` (`--build-threads`, default 16), and fails a run whose log has no `scaleli_cell:` line or has
a `scaleli_error`. The Mac build is `scripts/gre_mac.sh` ([MAC.md](MAC.md)).

scaleli_bench (`src/benchmark.cpp`), `compare.py`, `apply_integration.py` and the text of the original edits are on
main@ca8d80e (`experimental/scaleli/`); the sections below cite scaleli_bench as the reference the facade was
checked against.

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
  - `-ffp-contract=off` keeps the facade's arithmetic equal to the portable scaleli_bench build on any x86
    host. GCC would otherwise contract under `-march=native` on an FMA machine.
  - x86 results (80-bit long double) differ from arm64 results. Compare builds of the same architecture only.
- **J's build at 200M.** The joint fit runs inside `bulk_load` on up to `build_threads` threads, one per k. The
  joint-root spec estimates 2 to 5.5 extra minutes on the Atom on top of NC's build; this is not measured yet.
  `learnability.root_joint_build_ns` records it. Runner timeouts must allow it (gre_run.sh has none).
- **Rosetta.** Local x86 runs under Rosetta emulate x87 long double in software. C, NC and J at 1M keys took about
  9 minutes to build here, on a heavily loaded machine, against about 9 s for an arm64 scaleli_bench. Local x86 runs
  are for correctness only.

### Validation record (2026-10-05, the Mac; GRE e807edc pristine clone plus the edited gre_lite.sh)

scaleli_bench and compare.py, needed to redo this, are on main@ca8d80e.

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

## SPLICE-H

**Not the SCALE-LI cells.** `scaleli_splice` (alias of `scaleli_jg0`) is SCALE-LI's joint-root cell Jg0 from
`src/gre/`, a different index; it keeps its name because earlier server runs use it. The SPLICE index is
`splice`.

### GRE names

| GRE index | What it is |
|---|---|
| `splice` | SPLICE-H, read-only. Writes (`put`, `update`, `remove`) return false and `scan` returns 0. `--thread_num` must be 1. |
| `splice_thp` | The same layout (selected on the 4 KiB arm), plus `madvise(MADV_HUGEPAGE)` on the arena before it is written (Linux; recorded as `splice_arena_anon_huge_bytes`). Isolates the page-size effect. |
| `trace_<index>` | The index inside the chunk timer, for `splice`, `splice_thp`, `lipp`, `pgm`, `alex`, `btree`, `sortedarray`, `nullindex` and `artunsync` (`trace_artunsync` is not built on arm64). `trace_lipp` is the paired reference of the transient check. |
| `nullindex` | GRE's harness alone (op sampling, the operations array, the virtual call, the timer). |

### Environment

- `SPLICE_ARGS`: whitespace-separated `key=value` tokens (`splice/params.hpp`, `DESIGN.md` section 4), a
  later token overriding an earlier one. `gre_run.sh` passes the per-dataset plan file first
  (`results/splice_h/args/<ds>.args`, written by `splice_count` with `eps` explicit so the build runs one PLA pass)
  and the global `SPLICE_ARGS` last. Unknown keys or bad values are errors.
- `SPLICE_BUILD_THREADS`: build threads (decimal, 1..1024, default 16), used when `SPLICE_ARGS` leaves `threads=0`.
  The layout does not depend on it.
- The index name sets `thp` (`splice` 0, `splice_thp` 1); a conflicting `thp=` in `SPLICE_ARGS` is an error.

Every error prints `splice_error: <message>` to stderr and exits with status 2 (GRE itself exits 0 on a bad index).

### Log lines

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

After the run, from `memory_consumption()`: `splice_total_bytes_after_run: <int>`. The trace wrappers print,
with `GRE_TRACE` unset, `trace_chunk_size: 1000000`, `trace_calls: <get calls>`, `trace_chunks_n: <N>` and N lines
`trace_chunk: <k> <ns>`; with a `geo:` or `fixed:` schedule, `trace_schedule`, `trace_calls`, `trace_chunks_n`, N lines
`trace_point: <k> <call> <ns since point 0> <faults> <cpu>` (-1 when not requested) and `trace_end: <calls> <ns>
<faults>` (`trace_interface.h` header).

### Fairness of the timed region

`bulk_load` runs inside GRE's `build_ns` window and does everything GRE already allows a bulk load to do: it builds,
writes the whole arena (so every page is touched once, with `madvise` first for `splice_thp`), and reads the router
and record lines once at the end. No read-side state exists, so this is the same with and without GRE's warm-up and
moves no lookup work out of the timed region. GRE's own arrays and the lookup trace are untouched. The build is
multi-threaded (`SPLICE_BUILD_THREADS`, default 16) and includes the hash pass in GRE's `build_ns`, while the
competitors build on one thread under `OMP_NUM_THREADS=1`: compare build times only from a round with
`--build-threads 1` and `SPLICE_ARGS=hash=0` (`DESIGN.md` section 4.5).

### Wiring

`scripts/gre_lite.sh` copies `splice_interface.h`, `trace_interface.h`, `null_interface.h`, `splice_gre.hpp` and
`splice_gre.cpp` into GRE's `src/competitor/splice/`, registers `splice`, `splice_thp`, `nullindex` and the nine `trace_<index>` names in
`competitor.h`, compiles `splice_gre.cpp` as a C++20 library with `-ffp-contract=off`, adds `SCALELI_INCLUDE` to microbench's include path (only `splice/layout.hpp` is
included there) and records every copied file's cksum and the `src/splice` headers' cksum in
`build/gre_lite_build.txt`. Server commands: `PROTOCOL.md` section 5b.
