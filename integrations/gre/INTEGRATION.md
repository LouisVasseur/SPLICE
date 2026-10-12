# SCALE-LI in GRE: runner semantics and validation record

The GRE-side code is in this directory ([README.md](README.md) lists the files and defines the cells).
`tools/gre_lite.sh` copies it into GRE and registers the cells; `tools/gre_run.sh` passes `SCALELI_FLOW` (from
`--flows`, default `data/flows`, per dataset: `s512` for fb, osm and planet, `s64` otherwise) and
`SCALELI_BUILD_THREADS` (`--build-threads`, default 16), and fails a run whose log has no `scaleli_cell:` line or has
a `scaleli_error`. The Mac build is `tools/gre_mac.sh` ([gre_mac.md](../../tools/gre_mac.md)).

scaleli_bench (`src/benchmark.cpp`), `compare.py`, `apply_integration.py` and the text of the original edits are on
main@ca8d80e (`experimental/scaleli/`); the sections below cite scaleli_bench as the reference the facade was
checked against.

## 1. Runner semantics and caveats

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

## 2. Validation done (2026-10-05, this Mac; GRE e807edc pristine clone plus the edited gre_lite.sh)

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
