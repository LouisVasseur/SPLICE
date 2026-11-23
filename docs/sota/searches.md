# searches: the plain searches of a SOSD-style table (binary, radix binary, interpolation, TIP)

`include/gret/sota/searches.h`, `tests/sota/test_searches.cpp`. Names: `bsearch`, `bsearchstd`, `rbs<bits>`, `rbsstd<bits>`,
`rbst<bits>`, `interp[<k>][e]`, `tip[<g>][a]`.

These are not indexes from a paper of their own. They are what every index over a sorted array has to beat, and one of them, radix
binary search, is the null hypothesis for a radix directory: a table over the top bits of the key and nothing learned.

## What each one is, and whose code runs

| name | what it does | whose code answers the lookup | what the build line says |
|---|---|---|---|
| `bsearch` | binary search over the whole array: the floor | **ours**: a Model that bounds nothing; `SortedBoundIndex`'s branch-free binary search (the last-mile search of every read-only baseline here) | `bsearch(reimplementation)` |
| `bsearchstd` | the same with `std::lower_bound`, which is what SOSD's "BS" runs | **ours** (`std::lower_bound`) | `bsearchstd(reimplementation)` |
| `rbs<bits>` | radix binary search: a table over the top `<bits>` bits of `key - min` gives a range, binary search inside | **SOSD's own class** `RadixBinarySearch<K, bits>`, unmodified, included by path. SOSD is the benchmark that introduced RBS (Kipf et al., "SOSD: a benchmark for learned indexes", 2019; Marcus et al., "Benchmarking learned indexes", PVLDB 2020), so this is the origin's code. The search inside the range is `SortedBoundIndex`'s | a line of its own: `gret: rbs20 is SOSD's own RadixBinarySearch (learnedsystems/SOSD competitors/radix_binary_search.h, unmodified, by path): table_entries=... table_bytes=...`, then `gret: rbs20 n=...` |
| `rbsstd<bits>` | the same class and the same bound, searched with `std::lower_bound`: radix binary search as SOSD's own benchmark runs it, end to end | **SOSD's own class** for the table and the bound; `std::lower_bound` inside `[start, stop)` is what SOSD's default last-mile search (`searches/branching_binary_search.h`) calls. Our wrapper makes the call, over our pair array | a line of its own: `gret: rbsstd20 is SOSD's own RadixBinarySearch (..., unmodified, by path), searched with std::lower_bound inside its bound as SOSD's benchmark does: ...`, then `gret: rbsstd20 n=...` |
| `rbst<bits>` | SOSD's table, entry for entry, with the bucket itself as the bound (SOSD widens it by one position on each side) | **a reimplementation (ours)** | `rbst20(reimplementation)` |
| `interp[<k>][e]` | interpolation search, at most `k` interpolations (default 8), then binary search of what is left; `e`: exponential search from the last probe, then binary search | **a reimplementation (ours)** of the textbook algorithm | `interp8(reimplementation)`, `interp8e(reimplementation)` |
| `tip[<g>][a]` | three-point interpolation search (Van Sandt, Chronis, Patel: "Efficiently searching in-memory sorted arrays: revenge of the interpolation search?", SIGMOD 2019), guard `g` (default 64); `a`: the first estimate as the authors' code computes it | **a reimplementation (ours)**, written next to the authors' `tip.h` and SOSD's `rm_tip.h` | `tip64(reimplementation)`, `tip64a(reimplementation)` |

Only `rbs<bits>` and `rbsstd<bits>` run borrowed code (the same class). Nothing here is the TIP authors' code and nothing is
presented as such.

## Where the borrowed code comes from

| | SOSD | the TIP authors' repository |
|---|---|---|
| URL | `https://github.com/learnedsystems/SOSD` | `https://github.com/UWHustle/Efficiently-Searching-In-Memory-Sorted-Arrays` |
| Commit | `f52f4cba01dfcd37f1574551fccef00198863b88` (2022-04-08) | `ec556de8dcb1badb46625da975cdf511f3f037c9` (2021-05-31) |
| Licence | GPL-3.0 (`LICENSE`, `COPYING`) | BSD 2-Clause (`LICENSE`) |
| Local clone | `/Users/louisvasseur/Downloads/readonly_competitors/SOSD` | `/Users/louisvasseur/Downloads/readonly_competitors/Efficiently-Searching-In-Memory-Sorted-Arrays` |
| On the cluster | `/scratch/lvasseur/readonly_competitors/SOSD` (same files by sha256, no `.git`) | not copied: nothing of it is compiled |
| Used by the header | `competitors/radix_binary_search.h` (87 lines, sha256 `a74746a5b0babbee...`), which includes `util.h` (`16c4ea231bf26e27...`) and `competitors/base.h` (`85909a286e3c153d...`); all unmodified, by path | nothing: read only (`src/algorithms/tip.h`, `interpolation_search.h`, `linear_search.h`, `src/padded_vector.h`, `src/util.h`, `src/benchmark.h`) |
| Used by the test only | `competitors/binary_search.h`, `competitors/rm_tip.h`, to pin what they compute | nothing |

Nothing of either repository is copied into this one. There is no shim header.

## How it was built

1. **The footing** is `SortedBoundIndex<K, P, Model>` (`include/gret/readonly.h`), as for RadixSpline, PLEX, CHT, RMI and FAST: one
   sorted array of (key, payload) pairs, a Model that says where a key can be, a branch-free binary search inside that range. `get`
   compares the key with the first and last key first; `scan` copies from the lower bound; writes return false.
2. **`bsearch`** is that index with a Model that returns `[0, n)` (`WholeArrayBound`). SOSD's `BinarySearch` returns the same bound
   and counts 0 bytes; the test checks both facts on SOSD's class. Ours counts the 8 bytes of the Model.
3. **`rbs<bits>`** (`RbsBound<K, Bits>`) holds SOSD's `RadixBinarySearch<K, Bits>` and calls its `Build` and `EqualityLookup`. The
   table, its layout, the bound and the byte count are SOSD's by construction. The wrapper adds:
   - a temporary `std::vector<KeyValue<K>>` (SOSD's 16-byte packed record), because that is what `Build` takes; it is freed after the
     build. At 200M keys it is 3.2 GB, and for a moment it stands beside the 1.6 GB key copy `SortedBoundIndex` makes (freed before
     `Build` runs): the peak is 24 bytes a key, 4.8 GB, above the index itself (measured: 40 bytes a key live at the peak);
   - two throws: fewer than two distinct keys (`__builtin_clzl(max - min)` is undefined for 0) and 2^32 keys or more (the table
     holds positions as `uint32_t`);
   - the build line quoted above.
   **`rbsstd<bits>`** (`RbsStdSearch<K, Bits>`) holds the same class, calls the same `Build` and `EqualityLookup`, and then calls
   `std::lower_bound` on the pair array inside `[start, stop)`. That is SOSD's own configuration: `benchmark.h` passes the bound to
   its searcher, and the default searcher is `std::lower_bound`. It cannot be a Model, because `SortedBoundIndex` owns the search
   inside the bound, so it runs in `SortedSearchIndex` (step 6). There the key copy does not exist: the peak is the 3.2 GB of records.
   The lower bound it returns is exact for stored and never-stored keys in `[min, max]` (every position before `start` holds a
   smaller prefix, the lower bound is at most `table[p + 1] <= stop`); the index answers keys outside `[min, max]` before the search.
   It was added by the verifying pass, after measuring that SOSD's widened bound costs more under the harness's branch-free search
   than under SOSD's own (section "Three ways to search SOSD's table").
4. **`rbst<bits>`** (`RbstBound<K, Bits>`) builds the same table in one pass over the keys, with no copy, and returns
   `[table[p], table[p + 1])`. The test proves it is the same table: for every probe, SOSD's bound is this bucket widened by one
   position on each side and cut at 0 and n. It exists because the null hypothesis should not lose on a detail: at 28 bits most
   buckets hold one key or none, and SOSD's widening turns a bucket of 1 into a range of 3.
5. **Name isolation.** SOSD's headers define `KeyValue`, `SearchBound`, `Competitor`, `Row`, `namespace util` and more at global
   scope, and DILI's `global.h` defines another `SearchBound`. They are included inside `namespace gret::sota::sosd`, after every
   standard header they include has been included at global scope (the technique `fast.h` and `bli.h` use). Checked: one
   translation unit with `registry.h` and its whole `GRET_SOTA` block compiles and runs, on the Mac and with g++ 11 on the cluster.
   A unit that also wants SOSD's `util.h` or `base.h` at global scope must include them before `searches.h` (they are `#pragma once`).
6. **`interp` and `tip` are searches, not bounds**, and this is how they were fitted. A Model is handed a copy of the keys only.
   Running the search there would mean a second array (8 more bytes a key) and one more cache miss per lookup to fetch the payload,
   neither of which the algorithms have. They therefore run in `SortedSearchIndex<K, P, Search>`, defined in the same header: the
   same sorted array of pairs, the same check against the first and last key, the same scan, the same writes that return false, the
   same memory count, with the search running over the pair array itself (the key and the payload of the record found are in one
   cache line). Fewer than 64 keys are binary searched, as in `SortedBoundIndex`. `bulk_load` throws unless the keys are strictly
   ascending, because both searches divide by key differences. `bsearchstd` and `rbsstd<bits>` run in the same class.
7. **A stored key that is not found where it is, is an exception.** After each build of `interp` or `tip`, 1000 evenly spaced stored
   keys are looked up and counted; a wrong position throws `std::logic_error`. The counts are printed:
   `gret: tip64(reimplementation) sample=1000 stored keys: three-point steps after the first estimate avg=..., records scanned avg=..., finished by binary search=...%`
   and `gret: interp8(reimplementation) sample=1000 stored keys: interpolations avg=..., hit the bound of 8 steps=...%, then ...`.
   These lines say, per dataset, whether the search is working as designed or living on its fallback.

## rbs: SOSD's table, as read from the class and pinned by the test

- **Entries.** `2^bits + 1` entries of `uint32_t`, whatever the keys are. Entry `j` is the first position whose prefix is `>= j`.
- **Prefix.** `(key - min) >> shift`, with `shift = bitlength(max - min) - bits` (0 when `max - min` has fewer bits). The prefix is
  the top `bits` bits of `max - min`, not of the key domain. The largest prefix is therefore between `2^(bits-1)` and `2^bits - 1`:
  depending on `max - min`, up to half of the table lies above the largest key and only repeats `n`. One far outlier at the top
  (fb) spends most of the table on empty space; that is RBS, and it is the behaviour the comparison is about.
- **Bound.** `[table[p] - 1, table[p + 1] + 1)`, cut at 0 and n. Below `min`: `[0, 1)`. Above `max`: `[n - 1, n)`.
- **Bytes.** SOSD's `size()` is `4 * (2^bits + 1)` and nothing else. `memory_consumption` is the pairs, that number, and the 56 bytes
  of the object. At 200M keys:

  | bits | table | bytes a key |
  |---|---|---|
  | 12 | 16 KB | 0.0001 |
  | 16 | 262 KB | 0.001 |
  | 20 | 4.2 MB | 0.021 |
  | 24 | 67 MB | 0.34 |
  | 28 | 1.07 GB | 5.37 |

- **Sweep.** SOSD's `benchmarks/benchmark_rbs.cc` runs 28 alone, or 8, 10, ..., 28 with `--pareto`. Here every value from 8 to 28 is
  a name.

## Three ways to search SOSD's table

The table is one table. What differs is the range handed to the last-mile search and the search itself:

| name | range | search inside | whose |
|---|---|---|---|
| `rbsstd<bits>` | SOSD's: the bucket widened by one position on each side | `std::lower_bound` | SOSD's configuration, end to end |
| `rbs<bits>` | SOSD's | `SortedBoundIndex`'s branch-free loop, as every other read-only baseline here | SOSD's bound, the harness's search |
| `rbst<bits>` | the bucket itself | the same branch-free loop | ours |

The widening matters when buckets are small: a bucket of one key becomes a range of three, often over two cache lines. The likely
reason the search then matters too: the branch-free loop reads such a range from its first position, each address depending on
the read before, where `std::lower_bound` reads the middle (the key itself) first and the processor runs ahead of the comparison.
Measured by the verifying pass on the Jed login node (Xeon, g++ 11.4.1 `-O3 -march=native`, 3M keys, 1M stored keys in random
order through `indexInterface::get`, CPU time, best of 5; a shared node, so read the order and not the digits):

| keys | bits | `rbsstd` | `rbs` | `rbst` |
|---|---|---|---|---|
| uniform over 64 bits | 20 | 48 ns | 51 ns | 45 ns |
| | 24 | 54 ns | 56 ns | 39 ns |
| | 28 | 54 ns | 65 ns | 38 ns |
| skewed (every scale) | 20 | 132 ns | 134 ns | 128 ns |
| | 24 | 102 ns | 105 ns | 101 ns |
| | 28 | 75 ns | 82 ns | 74 ns |
| dense runs with jumps | 20 | 117 ns | 124 ns | 121 ns |
| | 24 | 122 ns | 130 ns | 126 ns |
| | 28 | 121 ns | 128 ns | 121 ns |

On that machine `rbs` was never the fastest of the three and was up to 21% slower than SOSD's own configuration; `rbst` was the
fastest or within 4% of it. (On the Mac, with Apple clang, `rbs` is ahead of `rbsstd`, and `rbst` is again the fastest or within 6% of it.)
So radix binary search, as a null hypothesis, is **the best of the three at each table size**, and the paper should say which one
each number is: `rbsstd` is the one a reader can call "SOSD's RBS", `rbst` is ours and is labelled a reimplementation.

## Why interpolation search and TIP are reimplementations

**Interpolation search.** There is no usable code:
- SOSD `competitors/interpolation_search.h` does not compile at this commit (it calls `util::linear_search`, which does not
  exist; checked with `clang++ -fsyntax-only`), is not referenced by `benchmark.cc`, copies the data, ends in a linear scan of up to
  1000 records and exits the process on a key that is not stored.
- SOSD `searches/interpolation_search.h` is a last-mile search inside a bound, tied to SOSD's `std::vector<Row<K>>`, and reads the
  key at the exclusive end of its range.
- The TIP authors' `interpolation_search.h` has the limits of their `tip.h` listed next, and its loop has no bound on its steps.

**TIP, the authors' `src/algorithms/tip.h`.** It cannot answer GRE's interface without being edited:
- `Key` is `int64_t` (`src/util.h`): keys from 2^63 up would sort as negative. GRE's keys are `uint64_t` over the whole domain.
- `search(x)` returns the key it found, not its position: no payload, no way to say a key is absent other than by comparing, no scan.
- It searches a `PaddedVector<record_bytes>`: its own copy of the data with 32 sentinel records on each side (`INT64_MIN`,
  `INT64_MAX`); `tip.h` includes nothing itself and `linear_search.h` includes `<x86intrin.h>`.
- The sequential search at the end has no bound, and the estimates are converted from `double` to an index unchecked.
- The first estimate is not what the three-point formula gives (next section), and for keys far from zero that makes the
  sequential search start near the middle of the array for every key: about n/4 records a lookup.

**TIP, SOSD's `competitors/rm_tip.h`** (a third-party implementation by Ryan Marcus, 2019, "based on" the paper, with the constants
"lifted from the UW" constructor, as its comments say). Not referenced by SOSD's `benchmark.cc` at this commit, and it does not
include the `base.h` it needs. It copies the data with 8 sentinel records on each side, returns the sum of the values of the
matching keys (no position; a key that is not stored is told apart only by an `assert`), and has the same first estimate, the same
unbounded sequential search and the same unchecked conversions.

So: a reimplementation, and the build line says so.

## The first estimate of TIP: what the borrowed codes compute

TIP's loop uses the three-point formula of Jarratt and Nudds (Equation 2 of the paper; the same expression in both
codes), with `y_i = V[x_i] - key`:

    expected = x1 + y1 (x1 - x2)(x1 - x0)(y2 - y0) / ( y2 (x1 - x2)(y0 - y1) + y0 (x1 - x0)(y1 - y2) )

The first estimate takes the three points `x0 = 0, x1 = d = n/2, x2 = 2d` (the first, the middle and, in both codes, the last key), where everything but
the key is a constant. Substituting and dividing by `V[d] - V[last]`, with the authors' constants `a_0 = (V[0] - V[d]) / (V[d] - V[last])`
and `diff_scale = V[0] - a_0 V[last]`:

    expected = d + d (1 + a_0) (V[d] - key) / ( diff_scale - key (1 - a_0) )

Both borrowed codes compute `d + d (1 + a_0) (V[d] - key) / ( diff_scale - key (a_0 + 1) )`: `a_0 + 1` where the algebra gives
`1 - a_0`. As coded, the estimate passes through the middle key only. Measured by the test on 2001 keys, on SOSD's `interpolate_r`:

| keys | furthest estimate from the true position, borrowed expression | ours |
|---|---|---|
| 0, 10, ..., 20000 (a line from 0) | 667 (the last key is put at 1333 of 2000) | 0 |
| 1000000 + 10 i (the same line, away from 0) | 991 (every key is put near the middle) | 0 |
| i^2 | 400 | 178 |
| random 62-bit | 662 | 33 |
| 2^40 + i^3 | 997 | 301 |

`tip<g>` uses the form the algebra gives, which is also the paper's Equation 8 (below), written with integer differences (`TipSearch::first`); the test checks that it equals the
loop's three-point formula through positions 0, d and 2d on every key, and that it passes through the three keys. `tip<g>a`
computes the borrowed expression instead (the test checks it against SOSD's `interpolate_r` key by key), so the effect of that one
line can be measured; everything after the first estimate is the same code.

**Checked against the paper** by the verifying pass (PDF from `https://pages.cs.wisc.edu/~jignesh/publ/Revenge_of_the_Interpolation_Search.pdf`,
read on 2026-11-22; the link in the authors' README is dead):
- Equation 2 is the three-point formula quoted above, and Algorithm 3 has the 18 lines the comment above `TipSearch` refers to
  (lines 1-4 the three points and the first estimate "Eq. 8", 6-7 the guard around `mid`, 8-12 the bracket, 13-16 the guards at
  the ends of the bracket, 17-18 the next estimate "Eq. 2").
- Equation 8, as printed (Ridders' simplification for three equidistant points), is
  `expected = x1 + y1 (x1 - x0) (1 + (y0 - y1)/(y1 - y2)) / ( y0 - y2 (y0 - y1)/(y1 - y2) )`. With `a_0 = (y0 - y1)/(y1 - y2)` that is
  `d + y1 d (1 + a_0) / (y0 - a_0 y2)`, whose denominator is `diff_scale - key (1 - a_0)`: the paper's own equation has `1 - a_0`.
  `tip<g>` therefore computes the paper's Equation 8, and the `a_0 + 1` of the authors' `tip.h` (copied by `rm_tip.h`) departs
  from the paper it implements. `tip<g>a` keeps that departure measurable.
- Algorithm 3, line 6, tests `abs(expected - mid) < guard_size`; both codes test `<= guard`, and so does `tip<g>` (one position
  of difference in when the sequential search starts).
- The three points are the first, the middle (`n / 2`) and the last key. Equation 8 assumes them equidistant; for an even `n` the
  last key is at `2d - 1`. Both codes use the last key all the same, and so does `tip<g>`.
- Algorithm 1 of the paper (interpolation search) is the loop `interp` implements: `left = expected + 1` or `right = expected - 1`
  until the key is hit.

## Deviations

**rbs<bits>** (SOSD's class): none in the table, the bound or the bytes counted.
- The search inside the bound is `SortedBoundIndex`'s branch-free binary search, where SOSD's harness runs `std::lower_bound`
  (`searches/branching_binary_search.h`) by default. It is the last-mile search of every read-only baseline here.
- `memory_consumption` adds the 56 bytes of the object to SOSD's `size()`.
- `get` compares the key with the first and last key before the Model is asked; SOSD's lookup makes the same comparison again.
- SOSD sums the values of repeated keys; GRE's keys are unique.

**rbsstd<bits>** (SOSD's class, searched as SOSD searches it): none in the table, the bound, the bytes counted or the search.
- `std::lower_bound` is called on GRE's 16-byte (key, payload) pairs, where SOSD calls it on its 16-byte `Row` (key, data).
- No sum over repeated keys and no "key not found" message: GRE's keys are unique, and a key that is not stored is a `false`.
- `memory_consumption` adds the 56 bytes of the object; `get` compares with the first and last key first (as above).

**rbst<bits>** (ours): SOSD's table, with the bucket as the bound and one comparison that sends anything outside `[min, max]` to
its lower bound. Not SOSD's code.

**bsearch, bsearchstd** (ours): there is nothing to deviate from. `bsearch` uses the harness's branch-free loop, `bsearchstd`
`std::lower_bound`; which is faster depends on the compiler and on whether the array fits the cache, so both are names.

**interp[<k>][e]** (ours, the textbook algorithm: estimate `left + (key - V[left]) (right - left) / (V[right] - V[left])`, move the
side of the bracket the estimate fell on next to it):
- at most `k` interpolations (default 8), because the textbook loop is O(n) on skewed keys (with one far outlier at the top,
  every estimate is the left end of the bracket and the loop advances one position a step). What is left of the bracket is then binary searched, as SOSD's last-mile interpolation search ends. With `e`, the rest is
  an exponential search from the end of the bracket that moved last, then a binary search: fewer probes when the estimates were
  closing in from one side, up to twice as many when they were not;
- the quotient is taken in `double` (as both borrowed codes do); the estimate cannot leave the bracket because the quotient is
  at most 1;
- a key that is not stored ends the search as soon as the bracket shows it (two adjacent keys around it); the textbook
  algorithm assumes the key is there;
- keys must be unique.

**tip[<g>][a]** (ours, from the paper's method as the two borrowed codes implement it):
- keys are `uint64_t` over the whole domain; differences are taken as integers before they become `double`, and every estimate
  is cut to the bracket before it becomes an index. A NaN or an infinity from the division becomes an end of the bracket;
- the first estimate is the corrected form above (`tip<g>`), or the borrowed one (`tip<g>a`);
- the bracket closes on `mid` by comparing `V[mid]` with the key. Both borrowed codes, and lines 8-12 of the paper's Algorithm 3, compare the
  positions `mid` and `expected`, which is the same decision whenever the estimate is on the right side of `mid`, and skip the update when the two keys are equal;
- the sequential search compares in blocks of 8, as the authors' `LinearUnroll`, but stops after `8 * max(g, 8)` records; and
  after 16 three-point estimates the loop stops. In both cases what is left of the bracket is binary searched. The borrowed
  searches are unbounded there (O(n));
- no sentinel records and no copy of the data: the search runs over the index's own pair array, after the check against the first
  and last key, and then the sequential search cannot leave the bracket;
- records are GRE's 16-byte pairs (the authors' harness takes the record size as a parameter);
- keys must be unique; there is no sum over repeated keys;
- the guard defaults to 64: `tip<record_bytes, 64>` in the authors' `benchmark.h`, `GUARD = 64` in `rm_tip.h` (the paper says only
  that it is derived empirically); the guard tests are `<=`, as in both codes (the paper's line 6 has `<`);
- SIP, the paper's other search (fixed-point slope, for uniform keys), is not implemented.

## Knobs and recommended sweep

| name | knob | range | notes |
|---|---|---|---|
| `rbs<bits>`, `rbsstd<bits>`, `rbst<bits>` | prefix bits | 8 to 28 | SOSD's sweep is the even values; 5.4 bytes a key at 28 bits and 200M keys |
| `interp<k>`, `interp<k>e` | interpolations before the fallback | 1 to 64, default 8 | uniform keys need about log log n = 5 |
| `tip<g>`, `tip<g>a` | guard | 1 to 4096, default 64 | the authors' benchmark and SOSD's `rm_tip.h` use 64 |

Recommended, nine names a dataset: `bsearch`, `rbsstd20`, `rbsstd24`, `rbsstd28`, `rbst20`, `rbst24`, `rbst28`, `interp`, `tip`.
Report radix binary search as the faster of `rbsstd` (SOSD's configuration) and `rbst` (the tight bucket, ours) at each table
size, and say which. If there is room: `rbs20`, `rbs24`, `rbs28` (SOSD's bound under the harness's search), `rbsstd16`,
`bsearchstd`, `interpe`, `tipa`.

## Tests

`tests/sota/test_searches.cpp`, 1309 checks, every answer compared with a sorted array:
- **Key sets**: 1, 2, 3, 5, 17, 63, 64, 65 keys (both sides of the 64-key threshold), the top 64 keys of the domain, uniform,
  wide, dense runs (one from 0), far outliers, clusters, a regular stride, a huge gap, a bit field with 2^64 - 1, a CDF that bends
  everywhere, a cubic, an exponential, one low key under a block at the top, keys that are equal mod 2^32, keys that are all
  2^63 or more; and three sets of 2M keys (uniform over 64 bits; every scale at once with keys 0 and 2^64 - 1; dense runs with
  random gaps).
- **rbs, rbsstd and rbst models** at 8, 12, 16, 20, 24 bits (28 once, on 2M keys): every stored key inside SOSD's bound; our bucket
  holds exactly the keys of its prefix; SOSD's bound is our bucket widened by one on each side, for stored keys, their neighbours,
  random and in-gap keys; `std::lower_bound` inside SOSD's bound returns the lower bound of each of those probes; below and above
  the range; 4 bytes an entry, `2^bits + 1` entries, in all three.
- **interp and tip as searches**, on arrays from 3 keys up, at 1, 2, 3, 8, 64 steps (both finishes) and guards 1, 2, 8, 64, 1024
  (and the `a` variant): the lower bound of every stored key, of both neighbours of every key (of every fourth key in the 2M
  sets), of random and in-gap keys; the step and scan budgets are never exceeded; the test fails unless its counters show
  that the step bound, both finishes, the three-point loop, the sequential search and the binary-search fallback were all taken.
- **The first estimate of TIP**: the table above; `tip<g>a` equals SOSD's `interpolate_r`; `tip<g>` equals the three-point formula.
- **Every name as an index**: `get` on every stored key, both neighbours, 0, 1, 2^64 - 1, 2^63, below and above the range, random
  and in-gap keys (a miss must leave the value untouched); `scan` from stored, neighbouring, in-gap and extreme starts at lengths
  0 to n + 5 with a guard slot after the buffer and nothing written past what was returned; the caller's array is overwritten
  after `bulk_load` (the index must hold its own copy); `put`, `update`, `remove` return false and change nothing;
  `memory_consumption` equals the pairs plus the model, to the byte.
- **Names** accepted and rejected (60 rejected); what cannot be built throws (no keys, one distinct key, repeated or unsorted keys,
  knobs out of range).

`t --small` runs everything but the 2M sets, `t --big` the 2M sets and the sanity timing; `t` runs both. The halves exist because
the Jed login node gives one user one CPU and 8 GB in all (`cpu.max`, `memory.max` of the user's cgroup), shared by whatever else
that user runs there: the whole test is about a minute of CPU but took more than 5 minutes of wall time beside four other test
runs, and each half stays well under.

Results on the final files (verifying pass, 2026-11-22; same sha256 on the Mac and on the cluster):

| where | compiler | result |
|---|---|---|
| Mac arm64, sanitized (`-O1`, ASan + UBSan, no recovery) | Apple clang 17 | PASS, 1309 checks, 268 s; no sanitizer report, in our code or in SOSD's |
| Mac arm64, `-O3` | Apple clang 17 | PASS, 1309 checks, 51 s, 2.7 GB at the peak |
| Jed login node, `-O3 -march=native` | g++ 11.4.1 | `t --small`: PASS, 1236 checks, 31 s of CPU (117 s of wall time), 0.3 GB; `t --big`: PASS, 73 checks, 33 s of CPU (45 s of wall time), 2.2 GB |

Also built and run in one translation unit with `registry.h` under `-DGRET_SOTA`, on the Mac (RadixSpline, CHT, PLEX, RMI, the hash
tables) and on the cluster with the benchmark's own include path and defines (`-DGRET_SOTA -DGRET_SOTA_DILI`, so FAST and DILI
too): all searches answer, and none of their names is a name of the registry.

**Adversarial checks by the verifying pass** (a scratch program, not in `tests/`; Mac sanitized and `-O3`, and g++ 11 on the
cluster): 100 more small key sets (one key at 0, 1, 2^32, 2^63, 2^64 - 1; two keys; keys equal mod 2^32; all above 2^63; dense
then one key at the top; dense, a huge gap, dense; the top of the domain; powers of two and their neighbours; three clusters
2^63 apart; around 2^53, where a double stops telling neighbours apart), each under 33 names against `std::map`, probing every
key, its neighbours, the key with bit 32 or bit 63 flipped, and scans of every length from 0; 40 000 random small arrays
(3 to 3000 keys, eight shapes) given to `interp` and `tip` in heap blocks of exactly the array's size, so that a read outside the
array is an AddressSanitizer report; six sets of 4M keys (3M on the cluster) under 12 names; `memory_consumption` against the
bytes really allocated (a counting `operator new`: equal within 110 bytes, which is the index object and its name); and the
timings of the section "Three ways". No wrong answer and no sanitizer report anywhere.

The sanity timings at the end of the test (2M keys, stored keys in random order) are not measurements; the login node was shared.

## Build, run, and the lines for the registry

The clone (nothing else of SOSD is needed, no submodule, no build of SOSD):

    git clone https://github.com/learnedsystems/SOSD && git -C SOSD checkout f52f4cba01dfcd37f1574551fccef00198863b88

On the Mac, from the repository root (`build/tests/cxx` and `cxx-O3` are the compiler wrappers `tests/run_tests.sh` writes;
`RC` is the directory that holds the clone):

    RC=/Users/louisvasseur/Downloads/readonly_competitors
    mkdir -p build/tests/sota/searches
    build/tests/cxx    -I $RC tests/sota/test_searches.cpp -o build/tests/sota/searches/t    && build/tests/sota/searches/t
    build/tests/cxx-O3 -I $RC tests/sota/test_searches.cpp -o build/tests/sota/searches/t-O3 && build/tests/sota/searches/t-O3

On the cluster (login node only, g++ 11; `wf_cxx <dir>` compiles with the benchmark's flags and puts `/scratch/lvasseur/wf/<dir>/include` first):

    /scratch/lvasseur/wf/wf_cxx searches -isystem /scratch/lvasseur/readonly_competitors/SOSD /scratch/lvasseur/wf/searches/test_searches.cpp -o /scratch/lvasseur/wf/searches/t
    cd /scratch/lvasseur/wf/searches && nice -n 10 ./t --small && nice -n 10 ./t --big

In the benchmark, inside the `GRET_SOTA` block of `include/gret/registry.h` (the include path already has the directory that
holds `SOSD`, for PLEX; no new define):

    #include "gret/sota/searches.h"
    ...
    if (indexInterface<K, P>* i = sota::make_searches<K, P>(name)) return i;  // bsearch | bsearchstd | rbs<bits> | rbsstd<bits> | rbst<bits> | interp[<k>][e] | tip[<g>][a]

## Known limits

- Read-only: writes return false. Unique keys only for `bsearchstd`, `rbsstd`, `interp` and `tip` (a repeat throws at `bulk_load`).
- Fewer than 2^32 keys for `rbs`, `rbsstd` and `rbst` (throws above).
- Memory during `bulk_load` at 200M keys, above the index itself: `rbs<bits>` 4.8 GB at the peak (the 1.6 GB key copy of
  `SortedBoundIndex` and the 3.2 GB of 16-byte records SOSD's `Build` takes), `rbsstd<bits>` 3.2 GB (the records), `bsearch` and
  `rbst<bits>` 1.6 GB (the key copy), the others nothing. All freed before `bulk_load` returns.
- `interp` and `tip` are not guaranteed to beat binary search and on skewed keys they do not: on the test's "every scale at once"
  set, 99% of `interp8` lookups hit the step bound and 68% of `tip64` lookups end in the binary-search fallback. The fallback
  bounds the damage (O(log n) probes plus at most 512 records scanned for `tip64`); it is ours, not the paper's, and it flatters
  TIP where the paper's search would walk the array (two dense blocks around a huge gap: half the lookups scan 512 records and
  fall back, where the unbounded search would scan up to n/2).
- Not run on the 200M key files (by the rules of this pass): what `interp` and `tip` do on each dataset is in the `sample=1000`
  line of the build.
- The timings in "Three ways" come from a shared login node at 3M keys. They establish that the three variants differ and in
  which direction on that machine; the sizes of the differences at 200M keys are for the benchmark to measure.
