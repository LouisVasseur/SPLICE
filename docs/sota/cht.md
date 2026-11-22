# CHT: compact Hist-Tree (`cht<bins>e<max_error>`)

## What it is

Hist-Tree (Crotty, "Hist-Tree: those who ignore it are doomed to learn", CIDR 2021) is a read-only index over a sorted array that is
**not learned**: a radix tree whose nodes are histograms. Every node cuts its key range into `bins` equal-width bins; a bin with more
keys than a threshold gets a child node over its range, any other bin is terminal. The compact form (CHT, section 3.6 of the paper)
flattens the tree into one array of 32-bit entries: a non-terminal entry is the child's row, a terminal entry is the position of the
bin's first key with the top bit set. A lookup is one shift and one load per level, then a search over at most `max_error` keys of the
sorted array.

## Source

| | |
|---|---|
| Code | `https://github.com/stoianmihail/CHT`, header-only (`include/cht/{builder,cht,common}.h`) |
| Commit | `8b6f7b3641c375a34506e808db9753384c44ca26` (main, 2021-07-17) |
| Licence | MIT (Copyright 2021 Mihail Stoian) |
| Local clone | `/Users/louisvasseur/Downloads/readonly_competitors/CHT`, used unmodified, included by path: `-I <CHT>/include` |

This is **Mihail Stoian's implementation** of Crotty's compact Hist-Tree (the repository's README names the paper; the same code, with
auto-tuning added, is `ts_cht` inside PLEX). Crotty's own C implementation was not found published. Where Stoian's code differs from the paper's
text: nodes are laid out in breadth-first order (or the builder's cache-oblivious order) where the paper describes depth-first
pre-order, and a bin is terminal at `<= max_error` keys where the paper says "below the threshold". The header here is a wrapper, not a
reimplementation; nothing of the library is copied into this repository.

## Parameters and names

`cht<bins>e<max_error>`: `bins` per node (a power of two, 2 to 2^20) and `max_error`, the most keys a terminal bin may hold (1 to 2^20).
`ccht<bins>e<max_error>` is the same tree with the builder's `use_cache` flag (cache-oblivious node order; the authors: "makes sense
when the tree becomes deep"). Either name with **`std`** at the end (`cht64e32std`, `ccht16e8std`) is the same tree over the same
array, searched as the authors search it (see "Two searches" below). Any other name gives `nullptr`.

- Default: **`cht64e32`**, the README's example (64 bins, error 32).
- What the authors sweep: `grid_search.sh` runs both knobs over 2^0 .. 2^10 on books_200M (plain, cache-oblivious, single-pass); the
  paper varies the threshold "from eight up to several thousand" and reports that CHT is usually fastest at its largest size (small
  thresholds).
- A sweep around the default that always builds (`max_error >= bins / 2`, see defect 2): `cht16e8 cht16e16 cht16e32 cht64e32 cht64e64
  cht64e128 cht64e256 cht64e1024 cht256e128 cht256e256 cht1024e512`.
- Finer points that build or not depending on the data and its key range: `cht64e8 cht64e16 cht256e16 cht256e32 cht1024e32 cht1024e64`,
  and with 16 or 256 bins any threshold when the keys span all 64 bits (`cht16e2`, `cht256e8`: see defect 2). A setting that cannot be
  built throws from `bulk_load` and says the smallest `max_error` that builds, so run each name as its own process.
- Not exposed: the builder's `single_pass` mode. It materialises a node for every distinct key prefix at every level before pruning,
  which is far beyond memory at 200M keys.

## Two searches: `cht<bins>e<max_error>` and `cht<bins>e<max_error>std`

The authors' end-to-end benchmark (`bench_end_to_end.cc`) and README search an array of (key, payload) pairs with `std::lower_bound`
over `GetSearchBound`'s `[begin, end)`. `SortedBoundIndex` searches every read-only index here with a branch-free binary search. For
CHT the two are not equivalent in speed: its bound is one-sided (the key is within the first `count` positions of a window of
`max_error`, `count` being what the terminal bin really holds, often far fewer), so a search with branches, which is what g++ makes of
`std::lower_bound`, predicts the first steps, and a branch-free one pays every probe of the window. Apple clang compiles
`std::lower_bound` without branches, so on the Mac the effect is hidden. Both forms are therefore exposed:

- `cht64e32`: `SortedBoundIndex<ChtBound>`, the footing of the other read-only indexes (spgm, RadixSpline, PLEX).
- `cht64e32std`: `gret::sota::ChtStdIndex`, the same array, the same tree, `std::lower_bound` over the authors' own `[begin, end)`.

Measured through GRE's interface on the Mac (arm64), `-O3`, 1M stored-key lookups, best setting of a sweep over 16/64/256 bins and
`max_error` 8 to 1024 (`max_error >= bins / 2`); synthetic keys, order of magnitude only:

| keys | g++ 15: branch-free | g++ 15: `std` | Apple clang: branch-free | Apple clang: `std` |
|---|---|---|---|---|
| 4M uniform 64-bit | 48.0 ns (cht16e8) | 52.5 ns (cht16e8std) | 56.9 ns (cht64e32) | 69.9 ns (cht64e32std) |
| 5M, runs and gaps | 90.2 ns (cht64e64) | 80.5 ns (cht64e128std) | 86.7 ns (cht256e128) | 105.0 ns (cht256e128std) |
| 4M lognormal | 93.9 ns (cht64e32) | 81.0 ns (cht256e128std) | 79.1 ns (cht64e32) | 97.8 ns (cht16e16std) |

With g++ the authors' search is 11 and 14% faster on the two skewed sets, 9% slower on uniform keys, and up to 2x faster where the
window is much wider than the bins are full (cht64e256 on uniform keys: 121 ns against 67 ns). With Apple clang the branch-free search
is 17 to 19% faster on all three. The benchmark runs under g++ on x86-64, where neither was measured: **sweep both forms and report the
better one per dataset, saying which**. Reporting only the form without `std` could understate CHT by the margins above.

## How it maps to GRE's interface

`gret::sota::ChtBound<K>` is a Model for `SortedBoundIndex` (`include/gret/readonly.h`): one sorted array of (key, payload) pairs, the
CHT built over the keys, `GetSearchBound(key)` giving where the branch-free binary search looks. `gret::sota::make_cht<K, P>(name)`
returns the index.

| GRE | here |
|---|---|
| `bulk_load` | copy the pairs, `cht::Builder(min, max, bins, max_error, false, use_cache)`, `AddKey` for each key, `Finalize` |
| `get` | range check, `GetSearchBound`, binary search in the bound, key compare: exact for stored keys and for misses |
| `get`, names ending in `std` | range check, `GetSearchBound`, `std::lower_bound` over its `[begin, end)`, key compare: exact for stored keys and for misses |
| `scan` | lower bound of the start key the same way, then a copy from the array |
| `put`, `update`, `remove` | `false` (read-only) |
| `memory_consumption` | pairs (16 B a key) + the authors' `GetSize()` (their object and the table, 4 B x bins x nodes) + the wrapper's own fields |

Fewer than 64 keys: `SortedBoundIndex` builds no model (as for every model on that footing).

## Deviations from the authors' use of their code

1. **Search and bound, names without `std`.** The authors search with `std::lower_bound`; `SortedBoundIndex` uses its branch-free
   binary search (see "Two searches" above for what that costs or gains). `GetSearchBound` returns `[begin, begin + max_error + 1)`. A
   terminal bin holds at most `max_error` keys, so the wrapper hands `[begin, min(begin + max_error, built)]` to that search, which is
   still exact for hits and misses (the test checks it against the authors' bound for every probe) and saves the baseline one probe
   when `max_error` is a power of two. Names ending in `std` use the authors' search over the authors' bound, unchanged.
2. **Top keys left out when `max - min` is a power of two** (defect 1 below). The tree is built without them and anything above the last
   key built on goes to the tail of the array, as `PgmBound` does for PGM's sentinel. At most 64 keys.
3. **A setting the builder cannot build throws** `std::invalid_argument` instead of running into undefined behaviour (defect 2 below).
4. **A key range narrower than one node** (`ceil(log2(max - min)) < log2(bins)`, which the builder asserts against) gets no tree; the
   bound is the whole array, which then holds at most about `bins / 2` keys.
5. Keys must be strictly increasing (the builder itself notes that duplicates can leave a terminal bin over `max_error`); `build` checks
   and throws otherwise. GRE's keys are sorted and unique.

Nothing else is changed: the tree, its layout and its lookup are the library's.

## Defects found in the library (not hidden: both are guarded, neither can run through the wrapper)

1. **Heap overflow on the largest key**, `include/cht/builder.h:245` (`initNode`, write one past the root's bins; ASan:
   heap-buffer-overflow). Happens when `max - min` is an exact power of two, with any setting. Reproduced on the unwrapped library with
   keys 900 .. 4996. Reachable on real data only if a dataset's key range is exactly 2^k; not expected on the ten datasets, and not
   checked on them (the 200M files are not on this machine).
2. **Shift by a negative width on a partial last level**, `include/cht/builder.h:233` (UBSan: "shift exponent 4294967294 is too large"),
   then `:280` and `cht.h:60`. The root's bin width is `2^shift`, `shift = ceil(log2(max - min)) - log2(bins)`, and each level takes
   `log2(bins)` bits; unless `log2(bins)` divides `shift`, the deepest level has bins only `2^w` keys wide, `w = shift mod log2(bins)`. A
   dense run that still puts more than `max_error` keys in such a bin (`max_error < 2^w <= bins / 2`) makes the builder split it into a
   node of negative width. Reproduced with 50000 consecutive keys at 64 bins, error 8. Without a sanitizer, on machines that take shift
   counts mod 64, it builds a chain of about ten useless nodes per such bin (8.8 MB instead of 13 KB in that example) and still answers;
   for some widths it cannot terminate. **Reachable on real data**: any dataset with runs of consecutive or near-consecutive keys, at
   thresholds below `bins / 2`. Which of the ten datasets refuse which settings is not known yet (not checked here). On synthetic
   keys with runs of stride 1 to 3 (3M and 5M keys) `cht64e8` is refused while `cht16e8` and `cht64e32` build.
   `max_error >= bins / 2` never triggers it.

## Known limits

- Fewer than 2^31 keys (31-bit positions and node numbers in the table); `build` throws above that. At most 2^31 nodes: a node
  number with the top bit set would be read back as a terminal bin, silently; `build` throws after `Finalize` if the table has more
  (only reachable with a few bins and a tiny threshold on hundreds of millions of keys, and well over 100 GB of build memory).
- Build memory: the builder keeps its own copy of the keys (grown by `push_back`, so up to twice 8 B a key in capacity) and a pointer
  tree of about 8 B x bins + 40 B per node until `Finalize` returns, next to the final table. The wrapper frees `SortedBoundIndex`'s
  scratch key copy before `Finalize`. Build time is linear in keys x levels (2M uniform keys: 12 to 55 ms on the Mac). Measured peak
  heap during `bulk_load`, on top of the caller's pairs (3M to 5M keys, a counting `operator new`): 37 to 45 B a key at `cht64e32`
  (16 the pairs, 8 the scratch keys, up to 12.6 the builder's growing copy, the rest tree and table), 63 to 84 B a key where the tree
  is large (`cht16e8`, `ccht16e8` on keys with runs and gaps). At 200M keys that is about 8 to 17 GB beside the caller's 3.2 GB.
- The table can outgrow the data at small thresholds: `cht64e8` on 2M uniform 64-bit keys is 24.9 MB (12.5 B a key), `cht64e32` 1.07 MB.
- The search range is `max_error` wide whatever the terminal bin really holds (the table stores a bin's first position only), so a large
  `max_error` on a fine tree pays probes for nothing (`cht1024e512` below).
- Compilers: Apple clang (the two wrappers) and Homebrew g++ 15.2 with libstdc++ (`-std=gnu++17 -O3`, arm64): the test passes with
  both. Not compiled with g++ 11 and not on x86-64. The header is plain C++17 with `__builtin_clzll` and `__builtin_expect`, as the
  rest of the repository; the library uses `__builtin_clzl` on 64-bit values (a 64-bit `long`, true on both targets) and needs
  `<iostream>` and `<functional>` included first (the header does it).

## How CHT's structure differs from AdaptiveLineIndex (ab4)

Both refine a radix directory where the keys are dense. The differences, from `include/gret/adaptiveline.h` and the library:

| | CHT | AdaptiveLineIndex |
|---|---|---|
| Split rule | a bin holds more than `max_error` keys (a count) | a node's keys would read more than `AR_TAU` cache lines per lookup, and it holds more than `AR_LEAF` keys |
| Fan-out | the same `bins` at every level, root included | a root of up to 2^16 equal-width cells (about n / 8), then children of 2^k, k chosen per node (at most 8 bits) |
| Depth | until every bin is under the threshold | at most `AR_DEPTH` levels below the root (default 3) |
| Shared key bits | not skipped: a tight cluster in a wide bin costs a chain of nodes | a node first skips the bits all its keys share |
| What a leaf holds | the position of the bin's first key | a first slot and a straight line over the leaf's range (a slope) |
| Data layout | a dense sorted array, separate from the tree | buckets of 4 slots with spare slots (1.25 slots a key), laid out by the index |
| Last step | binary search over up to `max_error` pairs | read the predicted bucket, compare in it |
| Outliers | the key range starts at the smallest key and ends at the largest | end outliers peeled into an exception list first |
| Writes | none | `update` and `remove` in place, no inserts |

## Test

`tests/sota/test_cht.cpp`, compiled with `-I /Users/louisvasseur/Downloads/readonly_competitors/CHT/include`: 28 key sets (tiny,
uniform, wide, dense run, clusters, outliers, stride, huge gap, power-of-two ranges, ranges narrower than a node, 2M uniform, 1M mixed)
x up to 28 names (21 settings, 7 of them also with `std`; 14 names on the two large sets). Every stored key and several thousand misses per case against the sorted array, scans, the model's bound against
the authors' `GetSearchBound`, the node count against a separate recursion of the split rule, and that exactly the settings with a bin
over `max_error` below a node's width throw. The names ending in `std` get the same index checks. Sanitized (ASan + UBSan, asserts
on) and `-O3 -DNDEBUG`: both pass, 795 cases.

Sanity timing only (Mac, `-O3`, 2M uniform 64-bit keys, 1M stored-key lookups, best of 3; two runs of the test with other work on the
machine, so read the order of magnitude and nothing finer; not a benchmark):

| name | ns / lookup, run 1 | run 2 | model |
|---|---|---|---|
| cht64e32 | 50 | 59 | 1.07 MB |
| cht64e16 | 39 | 43 | 1.21 MB |
| cht64e8 | 51 | 56 | 24.9 MB |
| cht16e8 | 45 | 61 | 4.48 MB |
| cht256e128 | 74 | 93 | 0.26 MB |
| cht1024e64 | 81 | 87 | 4.20 MB |
| cht1024e512 | 149 | 180 | 4.20 MB |
| ccht64e8 | 45 | 59 | 24.9 MB |
| cht64e32std, cht16e8std, cht256e128std | see "Two searches" | | as without `std` |
| spgm64 (reference) | 77 | 119 | 0.002 MB |

## Registry

```cpp
#include "gret/sota/cht.h"   // after "gret/readonly.h"; the build needs -I <CHT>/include
...
    if (indexInterface<K, P>* c = sota::make_cht<K, P>(name)) return c;  // cht<bins>e<max_error>[std], ccht<bins>e<max_error>[std]
```
