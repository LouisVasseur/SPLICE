# plex&lt;maxerror&gt;: PLEX in GRE's harness

**What it is.** PLEX (Stoian, Kipf, Marcus, Kraska, "Towards Practical Learned Indexing", AIDB@VLDB 2021,
arXiv:2108.05117) is a read-only learned index: a greedy error-bounded spline over the sorted keys (as RadixSpline), and
on top of the spline points either a radix table or a compact hist-tree (CHT). Which of the two, and its size, is chosen
by PLEX's own self-tuning step from the spline alone, so that the user sets one parameter, the maximum error of the spline
in positions.

**This is the authors' code, not a reimplementation.** `include/gret/sota/plex.h` includes it by path and wraps it; no PLEX
source is copied into this repo.

| | |
|---|---|
| Repository | https://github.com/stoianmihail/PLEX (the first author's repository; its README names the paper) |
| Local clone | `/Users/louisvasseur/Downloads/readonly_competitors/PLEX` (unmodified, `git status` clean) |
| Commit | `6cdd57ea9b135bddcadadae2ce3631e30b8c7bae` (2022-06-30, "Update README.md") |
| Licence | MIT (Copyright (c) 2021 stoianmihail) |
| Used from it | `include/ts/builder.h`, `ts.h`, `common.h`, `ts_cht/builder.h`, `ts_cht/cht.h`, `ts_cht/common.h` (header-only) |
| Not used | `include/rs/` (the repo's copy of RadixSpline), `bench_end_to_end.cc`, the CMake build, `tune.sh`, `benchmark.sh` |

## Names and parameter

`plex<maxerror>`: the spline's maximum error, PLEX's only parameter. Registered values:

- powers of two: `plex1 plex2 plex4 plex8 plex16 plex32 plex64 plex128 plex256 plex512 plex1024` (32 is the value in the
  authors' `example.cc`);
- the other values of the authors' own sweep in `tune.sh`: `plex25 plex50 plex60 plex70 plex80 plex90 plex100 plex125
  plex150 plex175 plex200 plex225 plex250 plex275 plex300 plex350 plex400 plex450 plex500`.

Everything else (radix table or hist-tree, number of bins, the tree's error) is PLEX's tuner
(`ts::Builder::ComputeStatistics` and `InferTuning`), run as shipped, under the space limit the authors fixed (the layer
is no larger than the spline). `bench_end_to_end.cc` also holds hand-tuned per-dataset tables (`ts_manual_tuning`) for
four SOSD datasets; they are not used: the paper's point is the single parameter, and the tables do not cover GRE's
datasets. A sweep over the names above finds PLEX's best point per dataset, as for `spgm16/64/256`.

Environment (also as `plex32@PLEX_VERIFY=1` through `registry.h`'s `name@VAR=v`):

- `PLEX_VERIFY=1`: at build time, check that the bound of every stored key holds its position; throw if one does not.
  Off by default (one lookup a key; PLEX's build time should be PLEX's).
- `PLEX_GUARD=0`: call the authors' `ts::Builder::Finalize()` directly, without the check of deviation 2 below.

## How it maps to GRE's interface

`make_plex<K, P>(name)` returns `SortedBoundIndex<K, P, PlexBound<K, maxerror>>` (`include/gret/readonly.h`, the footing
`spgm` uses): one sorted array of `(key, payload)` pairs, a model that bounds where a key is, a branch-free binary search
inside the bound.

- `bulk_load`: copies the pairs, feeds the keys to `ts::Builder<uint64_t>(min, max, maxerror)` with `AddKey`, finalizes.
  Below 64 keys `SortedBoundIndex` uses no model.
- `get`: `ts::TrieSpline::GetSearchBound(key)` gives `[begin, end)` (at most `2 maxerror + 2` positions), searched by
  `SortedBoundIndex`. Keys below the smallest or above the largest stored key are answered before the model is asked.
  Exact for stored keys and for keys never stored.
- `scan`: lower bound through the same bound (with `SortedBoundIndex`'s fallback when the bound of a key that was never
  stored does not hold its lower bound), then a copy. Keys ascending, exact.
- `put`, `update`, `remove`: `false` (read-only).
- `memory_consumption`: 16 bytes a key plus `ts::TrieSpline::GetSize()` (spline points, radix table or CHT, both objects).

## Deviations from the authors' code path

The PLEX sources are untouched. Around them the wrapper does four things:

1. **A key range that is exactly a power of two.** PLEX at this commit cannot be built when `max_key - min_key == 2^k`:
   the tuner and the CHT builder take `ceil(log2(range))` bits, one too few for the largest key. `ts/builder.h:305,314`
   then indexes `counters[]` with `(unsigned)-1` (AddressSanitizer: BUS at `builder.h:314`; checked with the authors'
   code alone on 200k keys over a range of 2^40), and, by reading, `ts_cht/builder.h:117,129` would write one bin past a
   node. The
   wrapper builds the model on the keys below the largest one (repeatedly while the range stays a power of two) and
   sends a key above what the model was built on straight to the tail of the array, as `PgmBound` does for PGM's
   sentinel. If only power-of-two ranges are left (at most 65 keys), there is no model. On real data this needs a range
   of exactly 2^k, which none of the ten datasets is expected to have; it costs nothing when it does not apply.
2. **A hist-tree the builder cannot build.** When the tuner picks a tree whose `log2(numBins)` does not divide the bits
   of the key range, the bins of the last level are `2^r` keys wide with `0 < r < log2(numBins)`. If such a bin still
   holds more spline points than the tree's error, `ts_cht::Builder::BuildOffline` splits it into a node whose bin width
   is negative: `unsigned width = shift_ - level * log_num_bins_` wraps (`ts_cht/builder.h:111`) and the next line shifts
   by it (UBSan: `ts_cht/builder.h:117:71: runtime error: shift exponent 4294967294 is too large for 64-bit type`). The
   tuner's cost model does not exclude this choice. Seen with the authors' code alone on two synthetic sets (below); at
   -O3 on arm64 one of them builds and answers correctly by accident (shift counts taken mod 64) and the other does not
   finish (30 s limit; it keeps creating nodes).
   The tuner's choice cannot be seen from outside `Finalize()`, so `PlexBound::build` runs the statements of
   `ts::Builder::Finalize()` itself (`ts/builder.h:41-60`: last key onto the spline, `ComputeStatistics`, `InferTuning`,
   `ts_cht::Builder::Finalize`, the `TrieSpline` constructor), each one the authors' own function, with one check between
   the tuning and the tree build. The private members are reached through explicit template instantiation (legal C++,
   no `#define private`, no edit of PLEX). If the choice is unbuildable, `bulk_load` throws `std::runtime_error` naming
   the choice and the file and line; nothing is tuned differently and no other configuration is substituted. The test
   checks on every set that the model built this way has the same size and returns the same bounds as the authors'
   `Finalize()`.
3. **One copy of the model after the build**, so that the bytes held are the bytes `GetSize()` reports (the builder's
   spline vector grows by doubling; `GetSize()` counts its size, not its capacity).
4. **`PLEX_VERIFY`** (optional, off by default).

Not a deviation but worth knowing: the last-mile search is `SortedBoundIndex`'s branch-free binary search over
`[begin, end)` for every read-only index here, where the authors' benchmark uses `std::lower_bound` on the same range.

## Is defect 2 reachable on real data?

Unknown for the ten 200M datasets: none of them is on this machine and the rules of this task exclude the cluster. What is
known:

- It needs spline points packed closer than `2^r` keys (`r < log2(numBins) <= 20`), more than the tree's error of them in
  one aligned window, and the tuner picking a tree rather than a radix table. Dense keys with a small maximum error give
  that; sparse 64-bit keys with a maximum error of 16 or more are unlikely to.
- In a scratch run of the authors' tuner over 1344 synthetic combinations (7 shapes, 8 maximum errors, 3 sizes, 8 seeds)
  it occurred 5 times, all at maximum error 2: a dense set with two far keys (64-bit range, `2^11` bins, tree error 16,
  17 to 19 spline points in a `2^9` window) and 50 dense clusters (62-bit range, `2^8` bins, tree error 4, 5 points in a
  `2^6` window). Both are in the test as sets the wrapper must refuse.
- On the cluster a refusal is an exception from `bulk_load` whose message starts `PlexBound: plex<e>: PLEX's tuner
  chose ...`; another maximum error usually builds. Such a (dataset, error) pair has no PLEX point and should be reported
  as "PLEX's builder fails", not silently dropped.

## Known limits

- Read-only: no `put`, `update`, `remove`.
- `uint64_t` keys only (`make_plex` returns `nullptr` otherwise). Keys must be sorted and unique, as GRE's `bulk_load`
  gives them.
- PLEX stores spline indexes and CHT entries as `unsigned`, with the top bit as a leaf flag: fewer than 2^31 spline
  points (always so below 2^31 keys).
- Build memory beyond the pairs and the key copy `SortedBoundIndex` makes: PLEX's builder holds about 50 to 70 bytes per
  spline point while tuning (spline points, the CHT builder's copy of their keys, the LCP array and two histogram rows),
  and about three times the final layer while the CHT is built. Tuning time is about `64 * 3.6` passes over the spline
  points. Both are linear in the spline points; nothing is quadratic.
- The header is for one translation unit (as `registry.h`): it holds explicit template instantiations.
- Portability: plain C++17 plus `__builtin_clzl`/`__builtin_clzll`/`__builtin_expect` (GCC and clang); PLEX uses `double`
  and `std::fma` only, no `long double`, no intrinsics. Compiled and run here with Apple clang 17 (arm64) and Homebrew
  g++ 15 (`-std=gnu++17 -O3 -Wall -Wextra`, no warning from these files, same results). Not compiled on Linux x86-64
  with g++ 11 here.

## Build

Include path: the directory that holds the clone, `-I /Users/louisvasseur/Downloads/readonly_competitors`
(`#include "PLEX/include/ts/builder.h"`), which cannot clash with RadixSpline's own `rs/` headers; `-I .../PLEX/include`
also works (`#include "ts/builder.h"`). Before `gret/sota/plex.h`: GRE's `indexInterface.h` and `pgm/pgm.h`.

In `registry.h` (not edited by this work): `#include "gret/sota/plex.h"` after `gret/readonly.h`, and next to the `spgm`
lines `if (indexInterface<K, P>* x = make_plex<K, P>(name)) return x;`.

## Test

`tests/sota/test_plex.cpp` (`test_plex [KEYS...]`, GRE key files optional):

- the model alone at each maximum error: every stored key's bound holds its position, ends inside the array, is at most
  `2 e + 2` wide; same size and same bounds as the authors' `Finalize()` on the same keys;
- the index by name: `get` on every stored key and on keys never stored (neighbours of stored keys, between keys, 0,
  2^64 - 1, below and above the range, random keys), `scan` from each of those at lengths 1 to n + 5 against the sorted
  array, writes refused and without effect, `memory_consumption` = 16 n + the model's bytes;
- sets: 1 to 65 keys, uniform, wide (64-bit), runs (also from 0 and up to 2^64 - 1), clusters, far outliers, a stride, a
  huge gap, a bit field, power-of-two ranges (one, several in a row, down to two keys, down to no model), dense sets, ids
  with jumps, an exponential spread, dense clusters, and a mixed set of 2M keys;
- the two sets PLEX cannot build at maximum error 2 must be refused with the stated message (and build at 1, 4, 32);
  `unbuildable()` on hand-made tunings; `PLEX_VERIFY=1` and `PLEX_GUARD=0`; names that are not names.

Results on this machine (macOS arm64, Apple clang 17):

```
build/tests/cxx     (-O1, ASan + UBSan):  test_plex: ok (844 cases, 0 failed, 2 builds refused: PLEX cannot build its tuner's choice)
build/tests/cxx-O3  (-O3 -DNDEBUG):       test_plex: ok (844 cases, 0 failed, 2 builds refused: PLEX cannot build its tuner's choice)
```

The 2 refusals are the two sets above; no other set at any error was refused, and no sanitizer report came from PLEX's
code on anything the wrapper let through.

Sanity timing from the -O3 build (2M mixed keys, 1M lookups of stored keys in random order, best of three, a laptop
with other work running: for scale, not a result):

| index | model bytes | build ms | ns/get |
|---|---|---|---|
| plex4 | 2,293,832 | 25 | 55 |
| plex16 | 184,188 | 17 | 56 |
| plex32 | 78,444 | 16 | 68 |
| plex64 | 49,276 | 16 | 102 |
| plex256 | 48,652 | 18 | 139 |
| spgm16 | 51,632 | 30 | 77 |
| spgm64 | 32,880 | 38 | 115 |
| spgm256 | 16,352 | 36 | 160 |

## Independent verification (second pass, 2026-11-21)

A second agent rebuilt and reran the test both ways (same two lines as above) and added its own checks from scratch
files (not in `tests/`). Code and test were left as they are; this section is the only change.

- **Adversarial sets** (816 cases at -O3, 888 under ASan + UBSan at reduced size, 0 wrong answers): one key, two keys,
  63 to 129 keys around the 64-key threshold, 0 and 2^64 - 1 stored and not stored, all keys equal mod 2^32, dense then
  one key at 2^64 - 1, dense / gap / dense, keys above 2^63, keys around 2^53, runs with gaps of 2^0 to 2^50, 3M to 4.5M
  random keys. Every stored key, every neighbour (k - 1, k + 1, k + 2, mid-gap) of every stored key on sets up to 600k
  keys, 40k random probes, scans checked for content and for not writing past the requested length.
- **Memory**: live heap bytes (counted in a replaced `operator new`) after `bulk_load` against `memory_consumption()`:
  equal within 1 kB (the index object and its name) in all 816 cases. Peak heap during `bulk_load` on 4M keys: 24 to 33
  bytes a key (16 pairs + 8 key copy + PLEX's builder), 6 to 18 ns a key: no obstacle at 200M.
- **Guard, no false negative**: 9667 builds over every range width from 7 to 64 bits (four shapes, errors 1 to 64) under
  ASan + UBSan: no sanitizer report on anything the guard let through, no bound outside the array. **No false positive
  seen**: on a refused set the authors' `Finalize()` alone does report `ts_cht/builder.h:117:71: shift exponent
  4294967293 is too large` under UBSan (and, unsanitized, builds and bounds every stored key correctly on arm64 and on
  x86-64, by the shift count being taken mod 64).
- **How often it refuses**: 2 of 2700 tuner runs on ten shapes meant to resemble real key sets (200k, 1M, 3M keys, three
  seeds, all 30 errors), both at error 1 on a dense set with far outliers (the shape of SOSD's fb); the same shape at 4M
  keys is refused at error 2. In the fuzz above 77 of 9744, at errors 1 (57), 2 (19) and 4 (1). Nothing at error 8 or
  more was refused anywhere. Expect possible refusals at errors 1, 2, 4 on fb-like data at 200M; still unknown for the
  real datasets. If such a point is needed, `plex<e>@PLEX_GUARD=0:PLEX_VERIFY=1` runs the authors' path as shipped and
  checks every stored key's bound at build; a result obtained that way rests on undefined behaviour in PLEX and must be
  labelled so.
- **Fairness**: through `indexInterface`, `plex<e>` against a plain wrapper doing exactly what the authors' benchmark
  does (`GetSearchBound` + `std::lower_bound` over the pairs), 4M keys, errors 4 to 256, interleaved: `plex<e>` is equal
  or faster (the branch-free search is a little faster than `std::lower_bound` on this machine); model bytes identical
  to the authors' `GetSize()`.
- **x86-64**: both tests compiled as x86-64 (Apple clang, `-arch x86_64 -O3 -mavx2 -mfma`) and run under Rosetta: same
  verdicts, same 2 expected refusals, model sizes identical to arm64 on every set. Also Homebrew g++ 15
  (`-std=gnu++17 -O3 -Wall -Wextra -Wpedantic`): no warning from `plex.h` or the test, same results. g++ 11 on Linux
  itself is still not tried (no such compiler on this machine).
- **Still open before the 200M runs**: (1) the PLEX clone and its include path exist only on this Mac; the cluster needs
  the same commit and `-I`; (2) one `PLEX_VERIFY=1` build per (dataset, error), since exactness is tested to 4.5M keys
  only; (3) a refusal ends GRE's process with an uncaught exception: the run scripts must record it as "PLEX build
  failed", not as a missing row.
