# RadixSpline in GRE's harness (`rs<max_error>r<radix_bits>`)

A wrapper around the authors' code, not a reimplementation.

## What it is

RadixSpline (Kipf, Marcus, van Renen, Stoian, Kemper, Kraska, Neumann, "RadixSpline: a single-pass learned index",
aiDM@SIGMOD 2020, doi 10.1145/3401071.3401659) is a read-only learned index over a sorted array: a linear spline through
a subset of the CDF points, built in one pass with a bounded error (GreedySplineCorridor), and a radix table over the top
bits of `key - min` that narrows the search for the spline segment. A lookup reads the radix table, finds the segment,
interpolates a position, and searches the array within `max_error` of it.

## Source

| | |
|---|---|
| Repository | https://github.com/learnedsystems/RadixSpline (the authors' repository; its README cites the aiDM paper) |
| Commit | `ab96aa59d429e7423beba2350bdcdf88952df282` (2021-05-13, "Add large OSM configs") |
| Licence | MIT (Copyright (c) 2020 Andreas Kipf, Alexander van Renen, Mihail Stoian) |
| Local clone | `/Users/louisvasseur/Downloads/readonly_competitors/RadixSpline` (was already cloned; unmodified, `git status` clean) |
| Used | `include/rs/builder.h`, `radix_spline.h`, `common.h`, header-only, included by path |

Build with `-I <RadixSpline>/include`. Nothing of it is copied into this repository.
`PLEX/include/rs` (stoianmihail/PLEX, commit 6cdd57e) is a byte-identical copy of the same five headers (`diff -rq`
finds no difference), so a build that also has `-I <PLEX>/include` gets the same code whichever directory comes first;
checked in both orders.

## Files

- `include/gret/sota/radixspline.h`: `gret::sota::RadixSplineBound<K>` (a Model for `SortedBoundIndex`),
  `gret::sota::make_radixspline<K, P>(name)`, `parse_rs_name`.
- `tests/sota/test_radixspline.cpp`
- this file

## Parameters and names

`rs<max_error>r<radix_bits>`, both decimal without leading zeros, `max_error` in [1, 2^20], `radix_bits` in [1, 30].
Every other string is not a RadixSpline name (`make_radixspline` returns `nullptr`).

- `rs32r18` is the default of the authors' `rs::Builder` (18 radix bits, error 32). The paper names no default: it
  tunes both parameters per dataset "for minimum lookup latency" (Section 3) and uses r = 18 only as its example of a
  table size (1 MiB). The two configurations it states are for face (fb): error 2 with 25 radix bits (its fastest,
  about 650 MiB) and error 16 with 20 radix bits (11.5% slower, far smaller): `rs2r25` and `rs16r20`.
- The authors' `bench.cc` (`rs_manual_tuning::GetTuning`) holds their hand-tuned pairs for ten size scales per SOSD
  dataset. For the 200M uint64 sets, from the smallest index to the largest:
  - books: `rs380r12 rs170r16 rs110r16 rs50r20 rs30r20 rs20r22 rs10r22 rs3r24 rs3r26 rs2r28`
  - fb: `rs140r8 rs90r10 rs90r22 rs70r24 rs80r26 rs7r26 rs80r28`
  - osm: `rs160r20 rs80r20 rs40r24 rs20r24 rs8r26 rs3r26 rs2r28`
  - wiki_ts: `rs100r14 rs60r16 rs20r18 rs20r20 rs9r20 rs5r20 rs3r22 rs2r26 rs1r26`
- covid, genome, history, libio, planet, stack and wise have no tuning by the authors. A sweep in the range the authors
  use (error 1 to 380, radix bits 8 to 28), for instance `rs{2,8,32,128}r{12,18,24,28}` plus `rs32r18`, finds the best
  point the same way their table was made. A radix table of `r` bits is up to `4 * (2^r + 1)` bytes: 1 MB at 18,
  1 GB at 28 (5.4 bytes per key at 200M).

The parameters are run-time values, as they are in `rs::RadixSpline` itself (`max_error_`, `num_shift_bits_` are members),
so any pair in a name works without a new template instance. `SortedBoundIndex` default-constructs its Model and takes
only a name; `make_radixspline` therefore leaves the pair in a `thread_local` (`rs_pending()`) that the Model's default
constructor reads, and puts the defaults back before returning. A `RadixSplineBound` constructed directly has 32 / 18,
or what is passed to its `RsConfig` constructor.

## How it maps to GRE's interface

Through `gret::SortedBoundIndex<K, P, RadixSplineBound<K>>` (`include/gret/readonly.h`, unchanged), exactly like `spgm`:

| GRE | here |
|---|---|
| `bulk_load` | `SortedBoundIndex` copies the pairs into one sorted array and the keys into a vector; `build` runs `rs::Builder(min, max, radix_bits, max_error)`, `AddKey` for each key in one pass, `Finalize`, then drops the spare capacity of the spline vector |
| `get` | `rs::RadixSpline::GetSearchBound(key)` gives `[begin, end)`; `SortedBoundIndex` runs its branch-free binary search over the pairs in that range and compares the key |
| `scan` | `SortedBoundIndex`: lower bound through the same bound (with its fallback for keys that were never stored), then a copy |
| `put`, `update`, `remove` | `false` (read-only) |
| `memory_consumption` | 16 bytes per pair + the model: `sizeof(RadixSplineBound)` + 4 bytes per radix entry + 16 bytes per spline point, counted from the vectors' capacity |

Below 64 keys `SortedBoundIndex` builds no model at all and searches the whole array.

## What the authors' code does at the edges (read from the code, pinned by the test)

- **Smallest and largest key.** `min` is the first spline point; `Finalize` adds `max` as the last one if the corridor
  did not. `GetEstimatedPosition` returns 0 for any key `<= min` and `n - 1` for any key `>= max`. There is no sentinel
  value: 2^64 - 1 is an ordinary key, and unlike `PgmBound` the wrapper drops nothing from the model.
- **Search bound.** `[begin, end)`, end exclusive: `begin = max(0, est - max_error)`, `end = min(n, est + max_error + 2)`,
  `est` the interpolated position truncated to an integer; at most `2 * max_error + 2` wide. The `+ 2` covers the
  truncation and one position of floating-point slack. This is `KeyBound`'s contract ("the lower bound of a stored key
  is in [lo, hi)") as it stands; the wrapper passes it through unchanged.
- **Unique keys.** GRE passes unique keys, and then the bound holds the position of every stored key. Repeated keys
  would be accepted (a repeat adds no CDF point; the bound holds the first occurrence, and for the largest key the
  last). The test pins this, GRE does not use it.
- **`max == min`.** `rs::Builder` takes `__builtin_clzl(max - min)`, undefined at 0. Unreachable through
  `SortedBoundIndex` (no model below 64 keys, keys unique); `build` throws `std::invalid_argument` rather than reach it.
- **`radix_bits` larger than the bit length of `max - min`** (dense keys): the shift is 0 and the table has
  `max - min + 2` entries.
- **Size limit.** The radix table stores spline indexes as `uint32_t`, so fewer than 2^32 keys; `build` throws
  `std::length_error` above that.

## Deviations from the paper and from the authors' benchmark

1. **The last-mile search routine is the harness's, not the authors'.** The bound is searched with `SortedBoundIndex`'s
   branch-free binary search over an array of 16-byte (key, payload) pairs. The layout is the authors' own: the paper
   indexes "200 M key/value pairs (3.2 GiB)" in a sorted array and ends with "a binary search within the error bounds",
   and `bench.cc` (`NonOwningMultiMap`) and `rs::MultiMap` run `std::lower_bound` over a `vector<pair<Key, Value>>`
   inside `[begin, end)`. Only the routine differs (branch-free instead of `std::lower_bound`), over the same range.
   This is the same footing `spgm16/64/256` stand on, so the read-only indexes differ only in the model.
2. **Spare capacity trimmed.** `rs::Builder` grows the spline with `push_back`, which can leave up to twice the memory
   allocated; the authors' `GetSize()` counts sizes, not capacities. `rs::RadixSpline` declares every
   `rs::Serializer<T>` a friend, and the wrapper specializes `rs::Serializer<gret::sota::RsPeek>` to call
   `shrink_to_fit` on the two vectors and to read their capacity. No source file of the authors is changed, no value and
   no lookup is affected; the bytes reported are then the bytes held and equal `GetSize()` up to the wrapper's own
   `sizeof`. The authors' `rs::Serializer<uint64_t>` still compiles and round-trips next to it (checked).
3. **No per-dataset tuning is built in.** The authors' benchmark picks `(radix_bits, max_error)` per dataset from a
   table; here the pair is in the name and the sweep picks.
4. Nothing else: the spline, the radix table, the segment search (linear below 32 candidates, binary otherwise) and the
   interpolation are the authors' code as it is.

Checked against the paper's text (arXiv 2004.14541, Sections 2 and 3) by the independent verifier: the spline
(GreedySplineCorridor, error `e`), the radix table over `r` prefix bits with the common prefix removed, the lookup
(table, search of the spline points, interpolation, binary search within `p ± e`) and the data layout are what the
code above does. "The defaults" means `rs::Builder`'s default arguments; the paper gives none. Not checked: SOSD's own
RadixSpline wrapper was not read.

## Known limits

- Read-only; `put`, `update`, `remove` return `false`.
- The radix table is laid out over `max - min`. A few far keys (as at the top of fb) leave most prefixes to them and
  few to the bulk; this is the method's own behaviour (the authors' fb tunings use 8 to 10 bits at small sizes for that
  reason), not something the wrapper adds or removes.
- Exactness of `get` for stored keys rests on the spline's error bound, which the authors compute in `double`. Positions
  up to 2^53 are exact; key differences are rounded to 53 bits, which moves an estimate by about `n * 2^-52` positions
  (4e-8 at 200M), far inside the one position of slack in the bound. The test checks the bound for every key of every
  set, and takes GRE key files as arguments to do the same on the real sets (not run here: no dataset on this machine).
  Misses and scans are exact whatever the bound is (`SortedBoundIndex` compares the key and falls back for scans).
- `bulk_load` at 200M: one pass, memory beyond the pairs is `SortedBoundIndex`'s key copy (1.6 GB, freed after the
  build) plus the model; trimming copies the spline once.
- Keys: `uint64_t` (GRE) or `uint32_t`, as `rs::Builder` supports.

## Tests

`tests/sota/test_radixspline.cpp`, built with

    build/tests/cxx    -I /Users/louisvasseur/Downloads/readonly_competitors/RadixSpline/include tests/sota/test_radixspline.cpp -o build/tests/sota/radixspline/t
    build/tests/cxx-O3 -I /Users/louisvasseur/Downloads/readonly_competitors/RadixSpline/include tests/sota/test_radixspline.cpp -o build/tests/sota/radixspline/t-O3

26 key sets (1 to 5 keys, 17, 63, 64, 65, the top 64 keys of the domain, 0 and 2^64 - 1 as ends, uniform, wide, dense
runs, outliers, clusters, stride, huge gap, bit field, a CDF that bends everywhere, and three sets of 2M keys: wide,
multi-scale, runs) times 7 configurations (5 on the 2M sets: rs32r18, rs1r18, rs8r24, rs380r12, rs3r26; also rs1r1,
rs100000r8, rs2r10 on the others):

- model: every stored key's position inside its bound, the bound inside the array and no wider than
  `2 * max_error + 2`; valid ranges for keys never stored; the bounds at min, max, 0 and 2^64 - 1; bytes; radix table size;
- index: `get` on every stored key and both neighbours of every key, 0, 1, 2^64 - 1, below and above the range, in-gap
  and random keys; `scan` from stored, neighbouring, in-gap and extreme starts at lengths 1 to n + 5 with a guard slot;
  writes return `false` and change nothing; `memory_consumption` equals pairs + model;
- names accepted and rejected; what must throw; repeated keys.

Result on macOS arm64 (Apple clang): both builds `PASS test_radixspline: 334 checks`; ASan and UBSan report nothing, in
the wrapper or in the authors' headers, and the authors' `assert`s are live in the sanitized build (no `NDEBUG`).
Also compiled with Homebrew g++ 15 `-std=gnu++17 -O3 -Wall -Wextra` against libstdc++ (no warning from the wrapper or the
authors' headers) and passed. Linux x86-64 with g++ 11 was not available here.

Independent verification (a second pass, scratch program outside `tests/`, not kept in the repository): both builds
of the test rebuilt and rerun (`PASS`, 334 checks, sanitizers silent). 336 further cases at `-O3` and 306 under
ASan+UBSan, all exact: single keys and pairs (0, 2^64 - 1, either side of 2^63), sets either side of the 64-key
threshold, all keys equal mod 2^32, dense runs split by gaps above 2^53 (where distinct keys round to the same
`double` difference) at error 1, ranges whose width is 2^w - 1, 2^w, 2^w + 1, radix bits beyond the bit length of
`max - min` up to 30, keys above 2^63, a stride `double` cannot hold, and 3M to 5M random keys; `get` on every key, its
neighbours at distance 1 to 3 and the middle of every gap, scans, and every model bound. For the never-stored keys at
distance 1 and 2 from a stored key, the lower bound was also inside `[lo, hi]` in every case (0 exceptions; the spline
is monotone, so this is expected), so `SortedBoundIndex`'s scan fallback is a safety net here. `memory_consumption` was compared with the heap bytes really allocated (a counting `operator new`): it
differs only by the index object itself (under 256 bytes). The test and the scratch program were also compiled for
x86-64 (`clang++ -arch x86_64 -march=haswell`, run under Rosetta) and with g++ 15 (`-Wall -Wextra`, no warning from the
wrapper): same results. g++ 11 on Linux itself remains untested here.

Sanity timing only (2M uniform 64-bit keys, stored keys in random order, one thread, laptop, through the virtual
interface; not a benchmark):

| name | ns per get | bytes per key | model bytes | build ms |
|---|---|---|---|---|
| rs32r18 | 57 to 73 | 16.54 | 1,075,668 | 15 |
| rs8r24 | 73 to 94 | 49.74 | 67,478,116 | 23 |
| rs2r26 | 62 to 84 | 152.02 | 272,043,192 | 45 |
| spgm64 | 92 to 96 | 16.00 | 2,248 | 17 |

(ranges over two runs; at 2M keys a 2^24 or 2^26 table is larger than the data, which is not the regime the 200M runs
are in.)

## Registry

    #include "gret/sota/radixspline.h"   // after "gret/readonly.h"; the build needs -I <RadixSpline>/include

and, in `make_index`'s `uint64_t` branch, next to the `spgm` lines:

    if (indexInterface<K, P>* i = sota::make_radixspline<K, P>(name)) return i;  // rs<max_error>r<radix_bits>: rs32r18, rs2r28, ...

`trace_`, `w_` and `g_` then apply to it through `make_index`'s recursion, like any other name.
