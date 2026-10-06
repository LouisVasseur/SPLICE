# gre_mac.sh: GRE natively on an Apple-silicon Mac (PREVIEW only)

`tools/gre_mac.sh` builds the same trimmed GRE as `tools/gre_lite.sh`, with GRE's own driver
(`src/benchmark/microbench.cpp`, `benchmark.h`, TSCNS timing), natively on arm64 macOS with Apple clang. Use it for
quick preview numbers. **Numbers from it are not the reference numbers and must not be mixed with Atom results**:
the Atom build (`gre_lite.sh`, SERVER.md) is the reference.

## Usage

```
experimental/scaleli/tools/gre_mac.sh [DEST]                    # clones GRE from GitHub, like gre_lite.sh
GRE_MAC_SEED=<existing GRE clone> experimental/scaleli/tools/gre_mac.sh [DEST]   # offline: clone from a local copy
GRE_MAC_LITE=experimental/scaleli/tools/gre_lite.sh experimental/scaleli/tools/gre_mac.sh   # use the working-tree gre_lite.sh
```

Then, for example (a 2M-key prefix, read-only, 1M warm-up lookups):

```
OMP_NUM_THREADS=1 [SCALELI_FLOW=results/aidb_flowv2/flows_free/fb_s512_t2000.txt] DEST/build/microbench \
    --keys_file=data/external/gre/fb --keys_file_type=binary --read=1 --insert=0 --operations_num=2000000 \
    --table_size=2000000 --init_table_ratio=1 --thread_num=1 --memory --index=sortedarray \
    --output_path=out.csv --warmup_num=1000000
```

Datasets: fb, osm and books are sorted as shipped. For covid, genome, history, libio, planet, stack and wise, use
`<name>.sorted`.

## What it does

1. Clones GRE at the commit `gre_lite.sh` pins (`e807edc`), with the same five submodules. If `GRE_MAC_SEED` is set
   and DEST is new, the clone comes from a local copy instead.
2. Runs a copy of the **committed** `gre_lite.sh` (`git show HEAD:`) with `GRE_LITE_EDIT_ONLY=1`. In-progress edits
   to the working-tree file therefore cannot break it. This applies every gre_lite edit unchanged: trimmed
   registry, `sortedarray`, the `gre_lite_patch.h` probes (`--warmup_num`, `--pin_core`, RSS, `build_ns`) and the
   SCALE-LI cells. `$CXX` is a wrapper that turns `-march=native` into `-mcpu=native`, so gre_lite's CPU check sees
   the same target as the build (no LZCNT, so `ALEX_USE_LZCNT 0`, as on the Atom).
3. Applies the arm64 edits below. Each one starts from the pristine file and is anchored on whole lines; the script
   stops if an anchor is missing or appears twice.
4. Compiles without CMake, mirroring gre_lite's `CMakeLists.txt`. `microbench.cpp` is built as gnu++17. Every
   `add_library` TU (today the SCALE-LI facade `scaleli_gre.cpp`; later a SPLICE TU if gre_lite registers one) is built
   as gnu++20 with `-ffp-contract=off -I include`. All TUs share GRE's `-faligned-new -g -O3 -include cstdint` plus
   Release `-O3 -DNDEBUG`, with `-mcpu=native` (apple-m3). The script stops if the CMakeLists sets anything it does
   not mirror. A new TU with the same settings needs no change here.
5. Writes `DEST/build/gre_mac_build.txt`: the gre_lite source and checksum, each edit's diff checksum, the shim
   checksums, the compiler, per-TU flags, the host, a hash of the binary's code and data, linked libraries, and
   gre_lite's own record. The binary file changes on every link (UUID, debug-map timestamps), but its code and
   data hash does not.

Nothing from GRE, the STX B+tree (GPL-3) or the other submodules enters this repository. The shims are written by
the script.

## arm64 edits (in DEST only)

| File | Why | Edit |
|---|---|---|
| `src/tscns.h` (MIT) | `rdtsc` is `__builtin_ia32_rdtsc` | on aarch64, read `CNTVCT_EL0`; TSCNS still calibrates ticks against `CLOCK_REALTIME` in `init()` |
| ALEX `alex_base.h` (MIT) | `CPUID` class is x86 inline asm (a compile error on arm64 even if unused) | compiled out on aarch64; only read by `cpu_supports_bmi()` in asserts of the LZCNT paths |
| GRE `src/benchmark/utils.h` | `cmpxchg`/`cmpxchgb` x86 asm | compiled out on aarch64; only XIndex (not built) calls them |
| `competitor.h` (gre_lite's) | ART (`artunsync`) needs SSE2 (`N16.cpp`) | **ART removed** from the registry and the `gre_lite:` name list |
| `gre_lite_patch.h` (ours) | see "Memory" below | macOS RSS = `phys_footprint`, peak = `ledger_phys_footprint_peak` |
| shim `x86intrin.h` | ALEX includes it for `_mm_popcnt_u64` | `__builtin_popcountll` |
| shim `omp.h` | Apple clang has no OpenMP | single thread: `omp_get_thread_num() == 0`, pragmas ignored |
| shim `tbb/parallel_sort.h` | no TBB | `std::sort` (setup only) |
| shim `jemalloc/jemalloc.h` | `benchmark.h` includes it | empty: `gre_lite_purge()` takes only its `malloc_zone_pressure_relief` branch |
| flags | Apple libc++ / arm64 | `-D_LIBCPP_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE`; `-D__float128='long double'` (only `--dataset_statistic` uses it) |

Mac registry: `alex lipp pgm btree sortedarray scaleli_b scaleli_n scaleli_c scaleli_cr scaleli_nc scaleli_j
scaleli_jg0 scaleli_splice`.

## Differences from the Atom (why these are previews)

- **CPU and caches.** M3 Max P-core (4 GHz class, wide out-of-order): 128 KiB L1d and 16 MiB L2 per P cluster,
  plus an SLC. The Atom (Goldmont) has far smaller caches and narrower cores. A 2M-key index (32 MB of pairs)
  fits very differently in each, so index rankings can change, not only absolute throughput.
- **16 KiB pages** (4 KiB on the Atom): different TLB reach, and RSS deltas are 16 KiB-granular.
- **Allocator and memory probe.** The Mac uses the macOS system malloc, not jemalloc 5.4.0. macOS returns freed
  pages as `MADV_FREE_REUSABLE`: they leave `phys_footprint` at once but stay in `resident_size`. Measured with
  resident_size, sortedarray read about 22 B/key; with phys_footprint it reads 16.2 B/key. So on the Mac,
  `rss_*`/`index_rss_bytes` are phys_footprint deltas, the analogue of the Atom's VmRSS after the jemalloc purge,
  but from a different allocator, with different per-allocation overhead.
- **No thread pinning.** macOS has no `sched_setaffinity`: `--pin_core` prints
  `gre_lite: could not pin to core K, running unpinned` and continues. A foreground process at default QoS runs on
  P-cores (a busy loop took 1.08 s at default QoS and 2.68 s under `taskpolicy -c background`, which forces
  E-cores). Do not run it under `taskpolicy -c background` or `-c maintenance`.
- **Single thread only.** OpenMP and TBB are stubs, so `--thread_num` greater than 1 silently runs every operation
  on one thread while the output still reports that thread count: always pass `--thread_num=1`.
- **Other operation lists.** The driver source is byte-identical to the Atom's (checked against an edit-only
  gre_lite.sh run: `benchmark.h`, `microbench.cpp`, `sortedarray.h`, the SCALE-LI files and `CMakeLists.txt` are
  identical), and every index on the Mac gets the same list. But libc++ and libstdc++ implement
  `uniform_int_distribution` and `std::shuffle` differently (checked with Apple clang against Homebrew g++-15), so
  the same `--seed` samples other lookup keys than on the Atom, and with `--init_table_ratio` below 1 it also
  bulk-loads another subset of keys. The distributions are the same; the samples are not.
- **ART missing** (SSE2).
- **Timer.** On arm64, TSCNS reads `CNTVCT_EL0` instead of the TSC. It is a fixed-frequency, monotonic counter
  shared by all cores (CNTFRQ 1 GHz on this M3 Max), read in about 1 ns without a syscall, but its value advances in
  steps of 17 ticks or more. That is irrelevant for `Throughput` (one interval around the whole loop) but quantises
  `--latency_sample` latencies.
- **Floating point.** The M3 has fused multiply-add; the Atom (Goldmont) has none. SCALE-LI's models call
  `std::fma`, which is one `fmadd` here but a libm call on the Atom, and Apple clang contracts `a * x + b` into
  `fmadd` in GRE's TU (for example in LIPP's lookup), where GCC on the Atom emits a multiply and an add. arm64
  `long double` is `double`, while x86 uses 80-bit x87: LIPP's model intercept and ALEX's regression sums use it, so
  their models (and hence their structures) can differ from the Atom's, and the facade's results differ too
  (integrations/gre/INTEGRATION.md). Expect the gap between SCALE-LI and the learned competitors to differ from the
  Atom's for this reason alone. `random_shuffle` comes from libc++, so data_shift and insert key orders differ too.
- **Noise.** On a desktop with other work running (load average around 6), repeated runs of one cell varied by
  about ±20% (e.g. sortedarray on fb: 232 to 347 ns/op). Report the median of several runs and quit heavy apps.
- **LIPP's fixed memory pool** (not Mac-specific). LIPP's constructor pre-builds 10^7 two-key nodes into a pool
  (`pending_two`), about 2.7 GB whatever the key count, and `index_rss_bytes` counts it on both machines: about
  1350 B/key on a 2M-key prefix, against GRE's self-reported `Memory` of about 240 MB.

## Validation (2026-10-06, M3 Max, macOS 15.7.5, Apple clang 17.0.0)

- Every Mac-registry index, on 2M-key prefixes of fb and osm (`--init_table_ratio=1`, read-only, 2M ops,
  `--warmup_num=1000000`): `success_read` = 2000000, `warmup_success_read` = 1000000, and `Throughput`,
  `index_rss_bytes` and `build_ns` are printed for all 26 runs.
- sortedarray `index_rss_bytes`: 16.15 to 16.22 B/key on fb and osm, against 16 B/key expected (+0.9% to +1.4%).
- Timer: TSCNS against `steady_clock` and `CLOCK_REALTIME` over 1 to 3 s busy loops and a 1.5 s sleep agreed within
  about 5e-5. TSCNS calibrates for 10 ms against `CLOCK_REALTIME`, which macOS reports at 1 µs resolution, so its
  rate is good to about 1e-4 (calibrated 0.99995 to 1.00000 GHz). In GRE runs, ns/op from `Throughput` matched
  ns/op of the steady_clock-timed warm-up to within run-to-run noise.
- Idempotence: a second and third run reproduced `gre_mac_build.txt` byte for byte, the same GRE tree state, and the
  same code and data hash.
- Review (same day): the timed region contains only GRE's own code and two `CNTVCT_EL0` reads (no syscall).
  `x86intrin.h`, `omp.h`, `tbb/parallel_sort.h` and the CPUID and `cmpxchg` edits touch no lookup path: ALEX's
  popcount gives the same count either way and serves its iterators, cost model and checks, not `get`, and
  `ALEX_USE_LZCNT` is 0 on both machines. All GRE indexes share one TU and its flags, as on the Atom. Memory: after
  `malloc_zone_pressure_relief`, 256 MB of freed blocks from 32 B to 64 KiB left 0 MB in `phys_footprint` but up
  to 256 MB in `resident_size`. The 26 prefix runs above were repeated after the review edits, all passing.
