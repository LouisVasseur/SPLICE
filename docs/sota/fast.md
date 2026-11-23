# fast: FAST behind GRE's interface (a third-party implementation)

`include/gret/sota/fast.h`, `tests/sota/test_fast.cpp`. Two names: **`fast`** (the one to report) and `fast_asis` (the same tree,
the same code, called the way TLI's own benchmark calls it; about 1.8 times slower, see "The two names").

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
| File used | `competitors/fast/src/fast.h` (232 lines, sha256 `ac042e635125d4fe9a581b18ffd9000e902b5ad3ff8e8a95235d1c8fd67c9e8b`), **unmodified**, included by path |
| Local clone | `/Users/louisvasseur/Downloads/readonly_competitors/TLI` (a sparse checkout: the licence, the README, `util.h`, `competitors/fast.h`, `competitors/base.h`, `competitors/fast/`, `benchmarks/benchmark_fast.*`, `searches/`) |
| On the cluster | `/scratch/lvasseur/readonly_competitors/TLI` (the same files, without `.git`; `fast.h` and `util.h` have the same sha256 there) |

To make the clone again (git only, nothing is built or installed):

    git clone --filter=blob:none --sparse https://github.com/curtis-sun/TLI
    git -C TLI checkout 13224d31d2dcb8ad4481529154c4c286cf4c32fc
    git -C TLI sparse-checkout set --no-cone /LICENSE /COPYING /README.md /util.h /competitors/fast.h /competitors/base.h /competitors/fast/ /benchmarks/benchmark_fast.cc /benchmarks/benchmark_fast.h /searches/
    rsync -rlz --checksum --exclude .git TLI/ jed.hpc.epfl.ch:/scratch/lvasseur/readonly_competitors/TLI/

Only `competitors/fast/src/fast.h` is compiled (and `util.h` with `-DGRET_FAST_TLI_UTIL`); the other files are there to read.

Nothing of TLI is copied into this repository (GPL). **Why TLI and not SOSD:** SOSD (`learnedsystems/SOSD`, commit `f52f4cb`)
has two FASTs. Its `competitors/fast.h` is Leis's original for **32-bit keys only** (it takes `KeyValue<uint32_t>` and refuses
keys from `INT32_MAX` up). Its 64-bit one, `competitors/fast64_search.h`, calls `fast64` (`github.com/RyanMarcus/fast64`), which
SOSD builds with `cargo build --release`: Rust, and neither cargo nor rustc is installed here or on the cluster. TLI's file is the
one C++ FAST for 64-bit keys that an established learned-index benchmark ships. The index says what it is each time it is built,
in two lines on stderr:

    gret: fast third-party implementation (Leis, via TLI): keys=... depth=... tree_mapped_bytes=... tree_resident_bytes=... tree_huge_page_bytes=... thp_disabled_for_process=... compares=gret_shim lookup=folded diag_ms=...
    gret: fast(third-party) n=... dir_bytes=... bytes_per_key=... build_ms=...

(`gret: fast_asis ... lookup=as_is ...` and `gret: fast_asis(third-party) ...` for the other name.)

## The two names: how TLI's `lower_bound` is called

Both names build the same tree with TLI's `buildFAST` and answer with TLI's `lower_bound`, unmodified. They differ in one thing,
which is ours and was added by the independent verification of this baseline:

- **`fast_asis`** calls `lower_bound` on the built `FAST` object, as TLI's own benchmark does (`competitors/fast.h`: `fast_.lower_bound`).
  TLI declares the layout constants (`SIMD_DEPTH`, `CACHE_LINE_DEPTH`, `PAGE_DEPTH`, `PAGE_CACHE_DEPTH`, `SIMD_MASK`, ...) as
  per-object `const` fields, not as compile-time constants. The built object lives on the heap inside the index, so the compiler
  must load those fields at every lookup and cannot fix the trip counts of the three nested loops of `lower_bound` or unroll them.
- **`fast`** calls the same `lower_bound` on a `FAST` object constructed in the calling function's own frame
  (`fast_detail::folded_lower_bound` in `fast.h`): a default-constructed `tli::FAST<uint64_t>` placed in a local buffer, whose three
  run-time fields (`v`, the mapping; `depth`; `len`) are set from the built tree. The compiler sees the constructor store the
  layout constants and folds them. The function is marked `flatten`, so the constructor and `lower_bound` are inlined into it
  whatever the inliner's size limits say. The local object's destructor is never run (it would unmap the tree, which the built
  object owns). The private fields are reached through an explicit template instantiation, without touching TLI's file.

The answers are the same by construction (one code, one tree) and are checked against each other key by key in the test, and on up
to 4096 stored keys at every build (a mismatch throws). The cost of the as-is call, measured on Jed's login node (g++ 11.4.1,
`-O3 -march=native`, CPU time, min of 3 or 4 runs over 2M stored keys in random order; the node is shared, so these are sanity numbers,
not results):

| what | keys | pages | `fast` | `fast_asis` | binary search on the keys |
|---|---|---|---|---|---|
| tree lower bound alone | 2M (depth 21) | 4 KiB | 77 ns | 141 ns | 178 ns |
| tree lower bound alone | 134.3M (depth 28, the depth of 200M) | 4 KiB | 337 ns | 665 ns | 541 ns |
| tree lower bound alone | 134.3M (depth 28) | 2 MiB (THP) | 259 ns | 467 ns | 413 ns |
| `get` through GRE's interface | 2M | 4 KiB | 131 ns | 243 ns | |
| `get` through GRE's interface | 100M (depth 27) | 4 KiB | 316 ns | 598 to 619 ns | |
| `get` through `get_index` in a build with the whole registry (the benchmark's flags) | 2M | 4 KiB | 185 ns | 326 ns | |

So the as-is call costs FAST a factor 1.8 to 2 for a reason that is not FAST's (at depth 28 it made the tree slower than a plain
binary search). **Report `fast`.** `fast_asis` is there so that the number "FAST exactly as TLI ships and benchmarks it" can be shown
beside it. A copy of TLI's file in which the eight constants are `static constexpr` (made in a scratch folder on the cluster, never
in this repository) gives the same times as `fast` (in one run at 2M keys: 141 ns as is, 77 ns with `constexpr`, 76 ns on a local
object; at 150M keys: 575, 303 and 308 ns): the local-object call recovers all of it.

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
   instead; the test passes both ways on the cluster and the Mac, with the same sanity timings (552 and 550 ns at depth 28, as-is call).
   The other piece that is ours is the way `fast` calls `lower_bound` (the section above).
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
   - after every build, TLI's `size_in_byte()` is compared with the size the layout needs as worked out in the wrapper, and its
     `depth` and `len` with the wrapper's (`std::logic_error` if they differ), so a change in TLI's code cannot pass unnoticed;
   - after every build, up to 4096 stored keys spread over the array are looked up both ways (on the built object, and on a local
     object); each must come back at its own position (`std::logic_error` otherwise). A fraction of a millisecond.
6. **Diagnostics in the build line.** How much of the mapping is resident (`mincore`) and on transparent huge pages (`AnonHugePages`
   of `/proc/self/smaps`), and whether huge pages are disabled for the process (`prctl(PR_GET_THP_DISABLE)`). The mapping's address
   is a private field of TLI's class; it is read through an explicit template instantiation (the technique `plex.h` uses), for these
   numbers and for the local-object call of `fast`. They cost `diag_ms` (53 to 280 ms at 134M keys on 4 KiB pages) and that time is
   inside `build_ms`.
7. **Without AVX2** (arm64, x86-64 without `-mavx2`) the header declares nothing of TLI's; `make_fast("fast")` and
   `make_fast("fast_asis")` throw `std::runtime_error` with a message that says why, every other name gives `nullptr`.
   `GRET_FAST_AVAILABLE` is 1 or 0.

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
  **3,489,964,032 bytes are resident of 4,297,064,448 mapped** (measured at 134M keys, the same depth as 200M). Without `nothp`
  (Jed's login node has transparent huge pages on `always`) all of it is resident and on huge pages (4,294,967,296 of 4,297,064,448
  bytes).
- **What `memory_consumption` counts:** the pairs, this object, and the **whole mapping** (TLI's `size_in_byte()`; TLI's own
  benchmark reports exactly `size_in_byte() + 16 n`). With huge pages off the resident tree is smaller than that: 19% smaller at
  depth 28 (200M keys), 38% smaller at depth 27, because a last-level page block of `L` levels is given `2^(L+1)` slots and fills
  12872 of 16384 of them for `L = 13`, 4680 of 8192 for `L = 12`. Checked against the process's own RSS on 4 KiB pages through the
  interface at 100M keys (depth 27): `memory_consumption` 3,749,580,912 bytes, RSS taken by the index 2,942 MB, so the reported
  number is 807 MB **above** what is held (never below). At 200M keys the same gap is 807 MB of 7.50 GB. With huge pages on, the
  two agree. The build line gives both numbers; a table that wants the resident one takes `tree_resident_bytes` and says so.

## Defects found in the third-party code (none changed; each guarded or stated)

1. **The mapping is too small from depth 31 (2^30 keys) on.** The size formula counts `2^((d-1)/15) - 1` full page blocks, the build
   writes `1 + 2^15` of them at depth 31. Up to depth 30 the two agree (the build's own `assert(offset == n)` holds for every size in
   the test's sanitized build, where asserts are on). The wrapper refuses 2^30 keys or more. 200M keys is depth 28.
2. **`mmap`'s result is not checked**; a refusal would be a write through `MAP_FAILED`. Guarded as said above.
3. **No copy control**: the class unmaps in its destructor and can be copied. `FastBound` cannot be copied.
4. **`size_in_byte()` is the mapping, not what is touched**: see "Memory".
5. **The layout constants are per-object `const` fields**, not compile-time constants, so on a built object that lives on the
   heap the three nested loops of `lower_bound` run with trip counts read at run time and are not unrolled. This is a cost of this
   implementation, not of FAST, and it is large (a factor 1.8 to 2 per lookup). `fast` removes it without editing TLI's file, by
   running `lower_bound` on a local object ("The two names"); `fast_asis` keeps it.

No sanitizer finding in TLI's code or in the wrapper (ASan and UBSan, x86-64 under Rosetta, the repository's test and the
verifier's adversarial sets, both names). ASan does not watch the bounds of an anonymous `mmap`, so
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

Differences between `fast` and the third-party code as TLI runs it, introduced here:

9. **`fast` runs TLI's `lower_bound` on a local object** so that the layout constants fold ("The two names"). Same code, same tree,
   same answers; only the compiler's knowledge of eight constants changes. `fast_asis` does not do this.
10. **The two unsigned AVX2 compares** are supplied by `gret/sota/fast.h` instead of TLI's `util.h` (same answers and times;
    `-DGRET_FAST_TLI_UTIL` takes TLI's).

## Knobs and sweep

**FAST has no tuning knob.** The block sizes are constants of TLI's file (`SIMD_BYTE`, `CACHE_LINE_BYTE`, `PAGE_BYTE`; they cannot
be changed from outside it) and TLI's own wrapper ignores its parameter list; the paper has no tuning parameter either (the blocks
follow from the hardware). The one choice is the name: **`fast`** (recommended, the number to report as FAST) and `fast_asis` (FAST as
TLI calls it). Sweep: `fast`, and `fast_asis` once for the record. Both use the same memory.

## How it maps to GRE's interface

| GRE | here |
|---|---|
| `bulk_load` | copy the pairs, copy out the keys, the guards above, TLI's `buildFAST`. Feasible at 200M keys: 3.2 GB of pairs, 1.6 GB of keys for the time of the build, 4.3 GB mapped for the tree |
| `get` | reject keys outside [first, last]; TLI's `lower_bound`; read the pair there and compare its key |
| `scan` | TLI's `lower_bound`, then a copy of the next pairs. Exact for starts that are not stored |
| `put`, `update`, `remove` | `false`, nothing changes |
| `memory_consumption` | `16 n` for the pairs + `sizeof(FastBound)` (112 bytes) + the whole mapping |

Below 64 keys `SortedBoundIndex` does not build any model and searches the array directly (the test still builds and checks the tree
alone on such sets).

## Registry

`registry.h` is not edited by this work. The lines, inside the `GRET_SOTA` blocks (they are there since 2026-11-21 and need no
change for the second name: `make_fast` answers both):

    #include "gret/sota/fast.h"
    if (indexInterface<K, P>* i = sota::make_fast<K, P>(name)) return i;  // fast, fast_asis: FAST, a third-party implementation (Leis, via TLI); x86-64 AVX2 only

**The benchmark's copy of the header must be the current one.** Until 2026-11-22 the tree the benchmark is built from
(`/scratch/lvasseur/GRE-gap/include/gret/sota/fast.h`, sha256 `4b03f88f...`) held the first version of this wrapper, in which the
name `fast` was the as-is call. Any `fast` number measured from that build is what is now called `fast_asis`; copy the current
`fast.h` there, rebuild, and run `fast` again.

Include path: the directory that holds the TLI clone. On the cluster that is `/scratch/lvasseur/readonly_competitors`, which the
benchmark's `CMakeLists.txt` already has (for PLEX). No define, no extra flag: `-march=native` gives AVX2 there. On the Mac (arm64)
the header compiles to the throwing `make_fast`, so put the line where a thrown `std::runtime_error` for the name `fast` is acceptable.

## Tests

`tests/sota/test_fast.cpp` checks every answer against the sorted array: the tree's lower bound for every stored key, both its
neighbours, 0, 1, 2^64 - 1, keys around 2^63 and 2^32, keys in gaps and random keys, **each through both calls** (on a local object,
and on the built object); `get` and `scan` through the interface **under both names**; writes returning `false`;
`memory_consumption`; the throws; the names. Key sets as in `tests/test_scan.cpp` (28 sets, three of 2M keys) plus 63 key counts
`2^k - 1`, `2^k`, `2^k + 1` for k = 1 to 21, which is every tree depth from 1 to 22 and both sides of the step from one page level
to two. `deep<k>` on the command line adds `2^k + 12345` keys, model only (both calls).

Commands (the Mac's x86-64 builds use the flags of `build/tests/cxx` and `build/tests/cxx-O3` with `-mcpu=native` replaced by
`-arch x86_64 -mavx2`, since clang refuses `-mcpu=native` together with `-arch x86_64`; they run under Rosetta):

    # Mac, native arm64: compiles, checks the names and the throw only
    build/tests/cxx    -I ~/Downloads/readonly_competitors tests/sota/test_fast.cpp -o build/tests/sota/fast/t
    build/tests/cxx-O3 -I ~/Downloads/readonly_competitors tests/sota/test_fast.cpp -o build/tests/sota/fast/t-O3
    # Mac, x86-64 with AVX2: the real test. Take the clang++ line of build/tests/cxx (or cxx-O3), put -arch x86_64 -mavx2 in place of -mcpu=native
    # Jed, login node, g++ 11
    /scratch/lvasseur/wf/wf_cxx fast -isystem /scratch/lvasseur/readonly_competitors /scratch/lvasseur/wf/fast/test_fast.cpp -o /scratch/lvasseur/wf/fast/t
    nice -n 10 /scratch/lvasseur/bin/nothp ./t          # nothp: 4 KiB pages, as the benchmark

Results with the current header (2026-11-22, the independent verification):

| where | build | result |
|---|---|---|
| Mac, arm64, sanitized (`build/tests/cxx`) | no AVX2 | `PASS test_fast: 2 checks; NO AVX2 IN THIS BUILD, so FAST itself was not built and not tested` |
| Mac, arm64, `-O3` (`build/tests/cxx-O3`) | no AVX2 | the same line |
| Mac, x86-64 under Rosetta, sanitized (ASan + UBSan, `-O1`, asserts on) | `-arch x86_64 -mavx2` | `PASS test_fast: 261 checks, compares from fast.h's shim`, no sanitizer output (78 s) |
| Mac, x86-64 under Rosetta, `-O3`, `./t deep22` | `-arch x86_64 -mavx2` | `PASS test_fast: 262 checks`; also `PASS ... 261 checks, compares from TLI's util.h` with `-DGRET_FAST_TLI_UTIL` |
| Jed login node, g++ 11.4.1, `-O3 -march=native` (`wf_cxx`) | AVX2 | `PASS test_fast: 261 checks, compares from fast.h's shim` (17 s of CPU) |
| the same, `-DGRET_FAST_TLI_UTIL`, under `nothp` | AVX2 | `PASS test_fast: 261 checks, compares from TLI's util.h` |
| Jed login node, `competitor.h` and all of `registry.h` (`-DGRET_SOTA -DGRET_SOTA_DILI`), both names through `get_index` | AVX2 | `coexist: ok (0 bad)`; `fast` 185 ns, `fast_asis` 326 ns a `get` at 2M keys |

With the first version of the header (as-is call only, 2026-11-21, by the implementer): the same test, 178 checks, passed in the
same places, and on Jed under `nothp` with `deep24 deep27` (`PASS test_fast: 180 checks`: depth 28, 134,230,073 keys, every key and
both neighbours), with and without `-DGRET_FAST_TLI_UTIL`. `deep27` was not repeated with the current header: every key twice over
is about ten minutes of CPU, and the login node now gives a session a fraction of one core and 8 GB shared (see "Limits").

**The verifier's adversarial checks** (scratch files, not part of the repository's tests: `build/tests/sota/fast/verify2/adv2.cpp`,
`deep.cpp`, `ifbench.cpp`, `coexist2.cpp`; on the cluster in `/scratch/lvasseur/wf/fast_verify/`), all against the sorted array,
model through both calls and index under both names:

- one key (0, 1, 2^64 - 1, 2^63, 2^32, ...), two keys ({0, 2^64 - 1}, {2^32 - 1, 2^32}, {k, k + 2^32}, {k, k + 2^63}, ...);
- every key count from 1 to 400 and from `2^k - 2` to `2^k + 2` for k = 9 to 18, several kinds of keys each;
- keys equal mod 2^32, keys with equal high halves (0, `0x7fffffff`, `0x80000000`, `0xffffffff`), low half zero, bit 31 alternating;
- dense runs then a huge gap, dense at both ends, runs across 2^63, everything above 2^63, everything below; each with and without
  0 and 2^64 - 1 stored;
- probes: every stored key and both neighbours, each bit of a sample of keys flipped, `k +- 2^32`, same high half with another low
  half and the reverse, in-gap and random keys, 0, 1, 2^64 - 1 and the powers of two; scans of length 0, 1, 2, 7, 100, n and
  n + 3 with guard slots before and after the output;
- 3M, 4M and 5M random keys (wide, and skewed over every scale with 0 and 2^64 - 1), every key and both neighbours;
- 300 builds and destructions of one index; 4 threads reading one index at once (both names).

| where | result |
|---|---|
| Mac, x86-64 under Rosetta, `-O3` | `PASS adv2: 1790 sets, 870726708 probes, 0 failures` |
| Mac, x86-64 under Rosetta, ASan + UBSan (large sets: 3M only) | `PASS adv2: 1786 sets, 808808901 probes, 0 failures`, no sanitizer output |
| Jed login node, g++ 11.4.1, under `nothp`, in nine parts | every part `PASS`, 0 failures (1790 sets in all) |
| Jed, `deep.cpp`: 134,300,000 keys (depth 28), 3M random positions with both neighbours and 3M random keys, both calls | 0 wrong of 24M, with huge pages off and on |
| Jed, `ifbench.cpp`: 100M keys (depth 27) through the interface, both names, 1M positions with both neighbours | 0 wrong of 3M per name; `bulk_load` 2.1 to 3.0 s of CPU (21 to 30 ns a key) |

**Scale.** The build is linear: 15 to 20 ns of CPU a key for the tree at 134.3M keys (2.0 to 2.7 s), 21 to 30 ns a key for the
whole `bulk_load` at 100M. At 200M keys `bulk_load` holds 3.2 GB of pairs, 1.6 GB of keys until the tree is built and a 4.3 GB
mapping (3.5 GB of it touched on 4 KiB pages): about 8.3 GB at the peak beside the caller's own 3.2 GB.

Not run on the 200M key files (that is for the job). The test takes key files as arguments for that:
`./t /scratch/lvasseur/gre_data/<file>` checks the tree and the index on every key under both names, about 14 GB (a job, not the
login node).

Sanity timings are in the table of "The two names". With the first version of the header the implementer measured, on the same
login node, 214 to 234 ns a `get` at 2M keys (now the as-is call) beside `spgm64` at 190 ns, and 552 ns for the tree's lower bound
at depth 28 beside 514 ns for `std::lower_bound`.

## Limits and open points

- **x86-64 with AVX2 only.** Nothing to run on arm64; on the Mac FAST is tested under Rosetta only, and in a Mac registry build
  the names `fast` and `fast_asis` throw `std::runtime_error` instead of giving an index.
- **Fewer than 2^30 keys** (defect 1).
- **This is not the paper's FAST.** 64-bit keys halve the fanout, a lookup is one dependent chain of about ten cache-line reads
  with no batching, and under the benchmark's 4 KiB pages the page blocking saves no TLB misses. Report it as "FAST, third-party
  implementation (Leis, via TLI), 64-bit keys, 4 KiB pages", not as FAST.
- **`fast` is not TLI's FAST as TLI runs it either**: it calls TLI's code so that eight constants fold. That makes the baseline
  faster (1.8 to 2 times), never slower; `fast_asis` is the as-shipped call. A paper should say which one a number is.
- **The fold depends on the compiler.** It is guaranteed to be correct (same code), not guaranteed to be fast: it needs the
  optimizer to forward the constructor's stores, which g++ 11 at `-O3` does (measured, also inside the registry build) and Apple
  clang does in part (358 against 422 ns under Rosetta). After a change of compiler or flags, compare `fast` with `fast_asis` once.
- **Huge pages would help this baseline more than most** (259 against 337 ns for the tree's lower bound at depth 28 on the login
  node). The benchmark runs everything on 4 KiB pages. If a comparison with transparent huge pages on is run, `fast` belongs in it.
- **Memory**: the number reported is the mapping, as TLI reports it; with huge pages off the resident tree is 19% smaller at 200M
  keys (807 MB of 7.50 GB reported are never touched).
- The two unsigned compares are ours by default (point 3 of "How it was built").
- **The login node is not the place for big runs.** A user's sessions there share 8 GB of memory and one core (cgroup
  `user-<uid>.slice`: `memory.max` 8589934592, `cpu.max` 100000 100000); a 200M-key build is killed there, and wall-clock times
  measure the neighbours. The timings above are CPU times.
