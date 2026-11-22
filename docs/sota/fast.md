# fast: FAST behind GRE's interface (a third-party implementation)

`include/gret/sota/fast.h`, `tests/sota/test_fast.cpp`. One name: `fast`.

## What it is

FAST (Kim, Chhugani, Satish, Sedlar, Nguyen, Kaldewey, Lee, Brandt, Dubey: "FAST: fast architecture sensitive tree search on modern
CPUs and GPUs", SIGMOD 2010) is a read-only binary search tree over sorted keys that is **not learned**. Its nodes are laid out in
blocks sized to the hardware: the keys of a few levels fit one SIMD register and are compared with one instruction, a few SIMD blocks
fit one cache line, and a few cache-line blocks fit one memory page, so that a lookup costs few cache misses and few TLB misses. It
is the strongest static, non-learned tree in the literature the learned indexes are compared with.

## Where the code comes from: third party, not the authors, not ours

**Intel never released FAST.** What runs here is a **third-party implementation**:

| | |
|---|---|
| Written by | Viktor Leis (TUM, 2012), for 32-bit keys with SSE; the file's own header says so |
| Generalized by | the TLI benchmark (Sun, Zhou, Li: "Learned index: a comprehensive experimental evaluation", PVLDB 2023): a template over the key type, 256-bit AVX2 registers, 64-bit keys |
| Repository | `https://github.com/curtis-sun/TLI` |
| Commit | `13224d31d2dcb8ad4481529154c4c286cf4c32fc` (main, 2023-06-10) |
| Licence | GPL-3.0 (`LICENSE` and `COPYING` at the top of TLI, identical). `fast.h` has no licence header of its own |
| File used | `competitors/fast/src/fast.h` (232 lines, sha256 `ac042e63...c9e8b`), **unmodified**, included by path |
| Local clone | `/Users/louisvasseur/Downloads/readonly_competitors/TLI` (a sparse checkout: the licence, the README, `util.h`, `competitors/fast.h`, `competitors/base.h`, `competitors/fast/`, `benchmarks/benchmark_fast.*`, `searches/`) |
| On the cluster | `/scratch/lvasseur/readonly_competitors/TLI` (the same files, without `.git`) |

Nothing of TLI is copied into this repository (GPL). SOSD's `competitors/fast.h` was not examined; TLI was checked to hold 64-bit
keys and was used. The index says what it is each time it is built, in two lines on stderr:

    gret: fast third-party implementation (Leis, via TLI): keys=... depth=... tree_mapped_bytes=... tree_resident_bytes=... tree_huge_page_bytes=... thp_disabled_for_process=... compares=gret_shim diag_ms=...
    gret: fast(third-party) n=... dir_bytes=... bytes_per_key=... build_ms=...

## How it was built

1. **The tree** is TLI's `FAST<uint64_t>`, as is: `buildFAST(keys, n)` and `lower_bound(key)`.
2. **The wrapper** (`gret::sota::FastBound<K>`) makes it a Model for `SortedBoundIndex` (`include/gret/readonly.h`), the footing of
   the other read-only baselines: one sorted array of (key, payload) pairs, a model that says where a key is. FAST's `lower_bound` is
   exact (for stored keys and for keys never stored), so the model returns the empty range at that position and `SortedBoundIndex`
   reads the pair there with no search of its own. That is what TLI's own wrapper does (`competitors/fast.h`: `fast_.lower_bound`,
   then `data_[it]`).
3. **One piece of the lookup path is ours.** TLI's `fast.h` does not compile alone: it calls `_mm256_cmpgt_epu32` and
   `_mm256_cmpgt_epu64`, which are not Intel intrinsics (AVX2 has only the signed compare). TLI defines them in its `util.h`, a
   449-line header that also defines a non-inline function (a link error with two translation units), a `namespace util`, and a
   constant named `OVERFLOW` (a macro of `<math.h>` on macOS). `gret/sota/fast.h` therefore supplies the two compares itself, six
   lines: flip the top bit of both sides, then the signed compare. Building with `-DGRET_FAST_TLI_UTIL` takes TLI's `util.h`
   instead; the test passes both ways on the cluster and the Mac, with the same sanity timings (552 and 550 ns at depth 28).
4. **Name isolation.** TLI's file defines global names (`FAST`, `SIMD_BYTE`, `CACHE_LINE_BYTE`, `PAGE_BYTE`). It is included inside
   `namespace gret::sota::tli`, after every header it includes has been included at global scope, and the macros `util.h` would set
   are saved and restored. Checked: it compiles and runs in one translation unit with GRE's `competitor.h` and the whole `GRET_SOTA`
   block of `registry.h`, DILI included (`/scratch/lvasseur/wf/fast/coexist.cpp`).
5. **Guards** added by the wrapper, each a throw, none a change of an answer:
   - no keys (`std::invalid_argument`): TLI takes `log2` of the count;
   - keys not sorted (`std::invalid_argument`): the tree would answer wrongly without saying so;
   - 2^30 keys or more (`std::length_error`): see "Defects" below;
   - the mapping cannot be made (`std::runtime_error`): TLI does not look at what `mmap` returns, so the wrapper asks for the same
     mapping first and releases it;
   - a second build (`std::logic_error`): TLI would leak the first mapping;
   - after every build, TLI's `size_in_byte()` is compared with the size the layout needs as worked out in the wrapper
     (`std::logic_error` if they differ), so a change in TLI's code cannot pass unnoticed.
6. **Diagnostics in the build line.** How much of the mapping is resident (`mincore`) and on transparent huge pages (`AnonHugePages`
   of `/proc/self/smaps`), and whether huge pages are disabled for the process (`prctl(PR_GET_THP_DISABLE)`). The mapping's address
   is a private field of TLI's class; it is read through an explicit template instantiation (the technique `plex.h` uses), for these
   numbers only. They cost `diag_ms` (53 ms at 134M keys) and that time is inside `build_ms`.
7. **Without AVX2** (arm64, x86-64 without `-mavx2`) the header declares nothing of TLI's; `make_fast("fast")` throws
   `std::runtime_error` with a message that says why, every other name gives `nullptr`. `GRET_FAST_AVAILABLE` is 1 or 0.

## What TLI's code does (64-bit keys), as read from it and pinned by the test

- **Tree.** A complete binary search tree of depth `d = floor(log2 n) + 1` over `2^d` positions: the n keys, then the largest value
  of the key type as padding (at least one position; n more when n is a power of two). A node holds the last key of its left half; a
  lookup goes right when `key > node`. The leaf reached is the first position whose key is `>= key`: the lower bound, n if the key is
  above every stored key. 2^64 - 1 is an ordinary key. Repeated keys give the first of them (GRE has none).
- **Blocks.** 2 levels (3 keys) per AVX2 compare; 3 levels (7 keys in 8 slots) per cache line, which is one SIMD compare and one
  scalar compare; 15 levels (5 cache lines deep) per page block. With depth 16 to 30 there are two page levels: the top block, and
  2^15 blocks of `d - 15` levels each.
- **At 200M keys:** depth 28, a lookup reads 5 cache lines of the top page block (300 KB in all, small enough to stay cached), then 4 cache lines and
  one last key of a second-level block, then the pair: about 5 reads outside the cache in the tree plus 1 for the pair.
- **Memory.** One `mmap` of `8 * (2^(d+1) + 2^18 if d > 15)` bytes. At 200M keys that is 4,297,064,448 bytes, **21.5 bytes a key for
  the tree**, 37.5 with the pairs. The tree holds every key once more, padded to 2^28, in slots laid out for 2^29.
- **Huge pages.** After the `mmap`, `madvise(MADV_HUGEPAGE)` on Linux. There is no `MAP_HUGETLB`: huge pages only come from the
  kernel's transparent huge pages. **The benchmark runs every index under `prctl(PR_SET_THP_DISABLE)`** (`/scratch/lvasseur/bin/nothp`),
  and there the `madvise` changes nothing: the tree lives on 4 KiB pages (measured: `tree_huge_page_bytes=0`,
  `thp_disabled_for_process=1`). Two consequences. The page blocks, FAST's answer to TLB misses, save none: a second-level block of
  128 KB spans 32 small pages and a lookup touches about 3 of them. And the slots the build never writes stay unbacked: at depth 28,
  **3,489,964,032 bytes are resident of 4,297,064,448 mapped** (measured at 134M keys, the same depth as 200M). Without the wrapper
  (Jed's login node has transparent huge pages on `always`) all of it is resident and on huge pages (4,294,967,296 of 4,297,064,448
  bytes).
- **What `memory_consumption` counts:** the pairs, this object, and the **whole mapping** (TLI's `size_in_byte()`, the number TLI's
  own benchmark reports). With huge pages off the resident tree is 19% smaller than that; the build line gives both numbers, and a
  table that wants the resident one should take `tree_resident_bytes` and say so.

## Defects found in the third-party code (none changed; each guarded or stated)

1. **The mapping is too small from depth 31 (2^30 keys) on.** The size formula counts `2^((d-1)/15) - 1` full page blocks, the build
   writes `1 + 2^15` of them at depth 31. Up to depth 30 the two agree (the build's own `assert(offset == n)` holds for every size in
   the test's sanitized build, where asserts are on). The wrapper refuses 2^30 keys or more. 200M keys is depth 28.
2. **`mmap`'s result is not checked**; a refusal would be a write through `MAP_FAILED`. Guarded as said above.
3. **No copy control**: the class unmaps in its destructor and can be copied. `FastBound` cannot be copied.
4. **`size_in_byte()` is the mapping, not what is touched**: see "Memory".
5. **The layout constants are per-object `const` fields**, not compile-time constants, so the three nested loops of `lower_bound`
   run with trip counts read at run time and are not unrolled. This is a cost of this implementation, not of FAST; nothing in the
   wrapper can remove it without editing TLI's file.

No sanitizer finding in TLI's code (ASan and UBSan, x86-64 under Rosetta). ASan does not watch the bounds of an anonymous `mmap`, so
for the tree itself the evidence is the build's asserts, the size check and the answers.

## Deviations from the paper

Known differences between this implementation and Kim et al., none of them introduced here:

1. **Not the authors' code**, and written for a later instruction set: AVX2 256-bit compares where the paper uses 128-bit SSE.
2. **Key width.** The paper's tree is over 32-bit keys (wider keys are handled by compressing them). Here
   the tree holds full 64-bit keys. That halves the fanout the paper works with: 3 keys per SIMD block and 7 per cache line, against
   the paper's 3 and 15, so a cache line resolves 3 levels where the paper's resolves 4.
3. **One SIMD compare and one scalar compare per cache line**, because 3 levels do not divide into SIMD blocks of 2 levels. The
   paper's cache-line block is two levels of SIMD blocks.
4. **Page blocks.** The paper picks the page-block depth from the page size (10 levels for 4 KiB, 19 for 2 MiB pages, 32-bit keys).
   Here a page block is 15 levels placed in its own 2 MiB of address space whether or not the memory is on huge pages, and under the
   benchmark it is not.
5. **No payloads in the tree.** The paper's leaves hold (key, rid). Here the tree gives a position in a separate sorted array of
   pairs, one more cache line per lookup, as TLI's own wrapper does.
6. **Padding.** The tree is padded to `2^(floor(log2 n) + 1)` positions with the largest key; the paper sizes the tree to the data.
7. **Build.** One thread, scalar, key by key. The paper builds in parallel with SIMD.
8. **Not implemented:** key compression, the GPU version, the paper's processing of several lookups at once (every lookup
   here is one dependent chain of reads), and range scans in the tree (a scan here is the lower bound, then a copy from the array).

## Knobs and sweep

**None.** The block sizes are constants of TLI's file (`SIMD_BYTE`, `CACHE_LINE_BYTE`, `PAGE_BYTE`) and TLI's own wrapper ignores its
parameter list; the paper has no tuning parameter either (the blocks follow from the hardware). The sweep is the one name **`fast`**.

## How it maps to GRE's interface

| GRE | here |
|---|---|
| `bulk_load` | copy the pairs, copy out the keys, the guards above, TLI's `buildFAST`. Feasible at 200M keys: 3.2 GB of pairs, 1.6 GB of keys for the time of the build, 4.3 GB mapped for the tree |
| `get` | reject keys outside [first, last]; TLI's `lower_bound`; read the pair there and compare its key |
| `scan` | TLI's `lower_bound`, then a copy of the next pairs. Exact for starts that are not stored |
| `put`, `update`, `remove` | `false`, nothing changes |
| `memory_consumption` | `16 n` for the pairs + `sizeof(FastBound)` (96 bytes) + the whole mapping |

Below 64 keys `SortedBoundIndex` does not build any model and searches the array directly (the test still builds and checks the tree
alone on such sets).

## Registry

`registry.h` is not edited by this work. To add, inside the `GRET_SOTA` blocks:

    #include "gret/sota/fast.h"
    if (indexInterface<K, P>* i = sota::make_fast<K, P>(name)) return i;  // fast: FAST, a third-party implementation (Leis, via TLI); x86-64 AVX2 only

Include path: the directory that holds the TLI clone. On the cluster that is `/scratch/lvasseur/readonly_competitors`, which the
benchmark's `CMakeLists.txt` already has (for PLEX). No define, no extra flag: `-march=native` gives AVX2 there. On the Mac (arm64)
the header compiles to the throwing `make_fast`, so put the line where a thrown `std::runtime_error` for the name `fast` is acceptable.

## Tests

`tests/sota/test_fast.cpp` checks every answer against the sorted array: the tree's lower bound for every stored key, both its
neighbours, 0, 1, 2^64 - 1, keys around 2^63 and 2^32, keys in gaps and random keys; `get` and `scan` through the interface; writes
returning `false`; `memory_consumption`; the throws; the name. Key sets as in `tests/test_scan.cpp` (28 sets, three of 2M keys) plus
63 key counts `2^k - 1`, `2^k`, `2^k + 1` for k = 1 to 21, which is every tree depth from 1 to 22 and both sides of the step from one
page level to two. `deep<k>` on the command line adds `2^k + 12345` keys, model only.

| where | build | result |
|---|---|---|
| Mac, arm64, sanitized (`build/tests/cxx`) | no AVX2 | `PASS test_fast: 2 checks; NO AVX2 IN THIS BUILD, so FAST itself was not built and not tested` |
| Mac, arm64, `-O3` (`build/tests/cxx-O3`) | no AVX2 | the same line |
| Mac, x86-64 under Rosetta, sanitized (ASan + UBSan, `-O1`, asserts on) | `-arch x86_64 -mavx2` | `PASS test_fast: 178 checks, compares from fast.h's shim`, no sanitizer output |
| Mac, x86-64 under Rosetta, `-O3` | `-arch x86_64 -mavx2` | `PASS test_fast: 179 checks` with `deep22`; also PASS with `-DGRET_FAST_TLI_UTIL` |
| Jed login node, g++ 11.4.1, `-O3 -march=native` (`wf_cxx`) | AVX2 | `PASS test_fast: 178 checks, compares from fast.h's shim` (14 s) |
| the same, under `nothp`, `./t deep24 deep27` | AVX2 | `PASS test_fast: 180 checks`; depth 28 (134,230,073 keys) every key and both neighbours right, `tree_huge_page_bytes=0` |
| the same, `-DGRET_FAST_TLI_UTIL`, under `nothp`, `deep24 deep27` | AVX2 | `PASS test_fast: 180 checks, compares from TLI's util.h` |
| Jed login node, with GRE's `competitor.h` and all of `registry.h` | AVX2 | `coexist: ok (0 bad)` |

The x86-64 builds on the Mac use the flags of `build/tests/cxx` and `build/tests/cxx-O3` with `-mcpu=native` replaced by
`-arch x86_64 -mavx2` (clang refuses `-mcpu=native` together with `-arch x86_64`). The Mac's native arm64 builds cannot test FAST.

Not run on the 200M key files (that is for the job). The test takes key files as arguments for that:
`./t /scratch/lvasseur/gre_data/<file>` checks the tree and the index on every key, about 14 GB.

Sanity timings from the login node (shared, noisy, not a result): through the interface on 2M random 64-bit keys `fast` 214 to 234
ns a lookup beside `spgm64` at 190 ns; the tree's lower bound alone at depth 28 with huge pages off 552 ns beside `std::lower_bound`
on the same keys at 514 ns.

## Limits and open points

- **x86-64 with AVX2 only.** Nothing to run on arm64.
- **Fewer than 2^30 keys** (defect 1).
- **This is not the paper's FAST and may be slower than it.** 64-bit keys halve the fanout, the loops are not specialized (defect
  5), a lookup is one dependent chain of about ten cache-line reads with no pipelining, and under the benchmark's 4 KiB pages the
  page blocking is lost. On the login node the tree's lower bound was no faster than `std::lower_bound` at depth 28. Whatever the
  job measures, report it as "FAST, third-party implementation (Leis, via TLI), 64-bit keys, 4 KiB pages", not as FAST.
- **Huge pages would help this baseline more than most.** If a comparison with transparent huge pages on is run, `fast` belongs in it.
- **Memory**: the number reported is the mapping; the resident tree is 19% smaller at 200M keys with huge pages off.
- The two unsigned compares are ours by default (point 3 of "How it was built").
