# pgmpp: PGM++ in GRE's harness (`pgmpp<eps_l>[i<eps_i>][d<delta>][n]`)

**A hybrid, and the index says so in its build line**: `pgmpp64(authors' builder, reimplementation of the lookup)`.
The index structure is built by the PGM++ authors' own code, unmodified. The lookup is our reimplementation of theirs,
because theirs returns wrong ranges (below). The parameter tuner is ours and is not the paper's cost model.

## What it is

PGM++ (Liu, Han, Qi, Peng, Li, Lin, Chen, "Why Are Learned Indexes So Effective but Sometimes Ineffective?",
arXiv 2410.00846, Section 6) is the PGM-index of Ferragina and Vinciguerra with
its lookup and its two parameters reworked. A PGM-index is a stack of piecewise linear models: the last level maps a key
to its position in the sorted array within `eps_l`, each level above maps a key to a segment of the level below within
`eps_i`. PGM++ changes three things:

1. **Layer skipping** (6.1, step 1): the sparse levels at the top are not traversed. The lookup starts at the first
   level whose next level has more than `delta` segments.
2. **Hybrid search** (6.1, steps 2 and 3): inside a level, and for the last-mile search in the data, a linear scan when
   the range is at most `delta`, a branch-free binary search otherwise. `delta` is 8 on the paper's Intel (i7-13700K)
   and Apple M3 machines and 16 on its AMD (EPYC 7413) one.
3. **A tuner for the two error bounds** (6.2): `eps_l` from the space budget `B` (Eq. 22, from the mean and variance of
   the key gaps), then the `eps_i` in E = {4, 8, ..., 1024} with the lowest cost under a model of the lookup (Eq. 21,
   23).

The paper reports PGM++ 1.2x to 2.2x faster than the PGM-index at the same size (abstract: up to 2.31x) and up to 1.56x
faster than a tuned RMI, on fb, wiki, books, osm and three synthetic sets, with 5,000 lookups of stored keys.

## Where the code comes from

| | |
|---|---|
| Repository | https://github.com/qyliu-hkust/bench_search (the authors' repository; the paper cites it as the PGM++ code, the README says "Our PGM++ implementation is based on the original PGM-Index. The codes are all put in folder ./exp_pgm") |
| Commit | `a028f9dc8d14156d12e94f75f972a7d34588b195` (2025-02-02, "Update README.md") |
| Licence | **none stated for the repository** (no LICENSE file). `exp_pgm/pgm_index.h` and `exp_pgm/piecewise_linear_model.h` carry Vinciguerra's Apache-2.0 header (they are the authors' edited copy of https://github.com/gvinciguerra/PGM-index); `exp_pgm/search_algo.h` has no licence line |
| Local clone | `/Users/louisvasseur/Downloads/readonly_competitors/bench_search`, unmodified (`git status` clean); on the cluster `/scratch/lvasseur/readonly_competitors/bench_search` (same checksums) |
| Used | `exp_pgm/pgm_index.h` (`pgm::PGMIndex::build`, the `Segment` type), `exp_pgm/piecewise_linear_model.h` (the segmentation), `exp_pgm/search_algo.h` (included by `pgm_index.h`; the index calls nothing in it) |

Build with `-I <bench_search>` (the directory above `exp_pgm`). **Nothing of it is copied into this repository**, and
because the repository has no licence it must stay that way.

### Can the authors' implementation be used as it stands? No, for two reasons

`exp_pgm/pgm_index.h` is a usable class outside their benchmark (`pgm::PGMIndex<K, Epsilon, EpsilonRecursive,
BranchLessSearch, linear_search_threshold, Floating>`, used in `main.cpp` as `PGMIndex<uint64_t, eps_l, eps_i, true, 8,
float>`), but:

1. **Its branch-free search loses keys.** `search::upper_bound_branchless` (`search_algo.h:85-96`) ends its halving
   loop with one candidate left and returns it without comparing it. `segment_for_key` takes `std::prev` of the result
   (`pgm_index.h:185`), so whenever the right segment is the last one of the searched range the lookup lands one segment
   too early, and the position it then computes is the first position of the right segment instead of the key's. That
   is the case for every key under the last segment of a level, and whenever the prediction sits at the low end of its
   error. The path is taken for `eps_i` above their linear-scan threshold (8), i.e. for `eps_i` = 16 ... 1024 of their
   grid. The test builds their class as published and counts, on every run, the stored keys whose position is outside
   the range their `search()` returns: 1,703,378 of 36,065,484 key lookups in 130 of 428 builds on the Mac, 146 of 428
   builds on the cluster; 31% of the keys of one 2M-key set (dense runs, `eps_l` 64, `eps_i` 16; 25% on the cluster,
   where the segmentation runs in parallel chunks), and up to 90% of a 200-key set. Their benchmark times lookups and
   never checks a result. Their `std::upper_bound` configuration and their linear scan are correct.
   The verifier repeated the count with their header alone (no wrapper, no GRE header in the unit, 20M keys, Mac -O3):
   as instantiated in `main.cpp` their `search()` puts outside its range 0.04% of the stored keys of a uniform set at
   (`eps_l`, `eps_i`) = (64, 16), 0.47% at (8, 16), **45.7% at (8, 64)**; 1.7% of a lognormal set at (8, 16); 28.9% of a
   set with bending gaps at (8, 64); none at `eps_i` 4 (linear path); and none in any case with `BranchLessSearch = false`.
2. **Layer skipping, which the paper describes as step 1, is a commented-out block** in the published file
   (`pgm_index.h:163-169`), with a fixed threshold of 16.

Also, both error bounds are template arguments in their class, so one name per pair would need one instantiation per
pair (81 for the paper's grid).

## How it was built

`include/gret/sota/pgmpp.h` defines `gret::sota::PgmppBound<K>`, a Model for `gret::SortedBoundIndex`
(`include/gret/readonly.h`, unchanged: one sorted array of (key, payload) pairs, a model that bounds the position, a
branch-free binary search inside the bound), and `gret::sota::make_pgmpp<K, P>(name)`.

- **The authors' header is included inside `namespace gret::sota::pgmpp_authors`**, because it defines `namespace pgm`,
  `class PGMIndex` and `namespace search`, the names GRE's own PGM copy uses. Every standard header it includes is
  included before, outside the namespace. No shim header is needed.
- **The structure is theirs.** `PgmppBound` derives (privately) from one instantiation of their `PGMIndex` and calls
  their static `build(first, last, eps_l, eps_i, segments, levels_offsets, levels_segment_count, start_level)`, which
  takes both error bounds as run-time arguments. Segments are their 16-byte `Segment` (key, float slope, uint32
  intercept). The test compares our segments and level offsets with those of their constructor, entry for entry, for 16
  pairs of bounds on every key set.
- **The lookup is ours, after theirs line for line** (`segment_for_key` and `search`): clamp the key to the first key,
  find the segment at the start level by a scan, then per level compute `pos = min(segment(key), next.intercept)` and
  search `[pos - eps_i - 1, pos + eps_i + 2)`, by a forward linear scan when `eps_i <= delta` and by the branch-free
  halving loop otherwise, **completed by the comparison theirs lacks**. The result is `[pos - eps_l, pos + eps_l + 2)`,
  PGM's own range. Where their code is right (their `std::upper_bound` configuration; their linear path) the test
  requires our range to equal theirs exactly, for every stored key and every probe.
- **Layer skipping is implemented from the paper**: the start level is the lowest level with at most `delta` segments
  (not counting the sentinel entry), scanned linearly; `n` in the name starts at the root as their published code does.
- **The tuner**: without `i<eps_i>` in the name, the internal levels are built for each `eps_i` in the paper's set
  {4, ..., 1024} (the last level does not depend on `eps_i` and is built once; the levels above are built by their
  `build` on the last level's segment keys, and the test checks the result equals a single `build` over the data),
  whole lookups are timed on random stored keys (2^16 to warm up, then three blocks of 2^16, in two passes over the
  candidates, best block of each), and the smallest `eps_i` within 2% of the fastest is kept. The nine timings and the
  choice are printed in the build line.
- **Guards.** (a) 2^64 - 1 is PGM's sentinel and their builder throws on it: as `readonly.h`'s `PgmBound` does, the model
  is built without it and any key above the largest built key goes straight to the end of the array. (b) After the
  build, every stored key is looked up once with the code `get` runs; a key outside its range is counted
  (`outside_bound=` in the build line) and the range is widened for all lookups by the largest miss on each side. The
  count is 0 on every set tested except where one segment covers tens of millions of positions (next section but one,
  "Float slopes over long segments"): there the guard is what keeps the index exact. (c) Fewer than two keys below 2^64 - 1, 2^32 keys or more (segment intercepts are
  uint32), or parameters out of range: `build` throws. Keys not strictly increasing: their segmentation throws
  `std::logic_error`.

The build prints two lines to stderr, for example:

```
gret: pgmpp(authors' builder, reimplementation of the lookup) eps_l=64 eps_i=8(chosen by timing) delta=8 internal=linear skip=1 height=3 start_level=1 leaf_segments=4565 model_bytes=73512 outside_bound=0 widened=-0/+0 build_ms=2599.5 verify_ms=1721.2 candidates_ns=4:105 8:92 16:107 32:110 64:115 128:118 256:118 512:125 1024:128
gret: pgmpp64(authors' builder, reimplementation of the lookup) n=200000000 dir_bytes=73512 bytes_per_key=16.00 build_ms=4842.5
```

## Names, knobs, and the paper's setting rule

`pgmpp<eps_l>[i<eps_i>][d<delta>][n]`, decimal numbers without leading zeros; any other string is not a PGM++ name
(`make_pgmpp` returns `nullptr`).

| part | meaning | range | default |
|---|---|---|---|
| `<eps_l>` | error bound of the last level, in positions of the key array. The size knob | 1 .. 65536 | (required) |
| `i<eps_i>` | error bound of the internal levels | 1 .. 65536 | chosen at build among {4, ..., 1024} by timing |
| `d<delta>` | hybrid threshold: a level is scanned linearly when `eps_i <= delta`; layer skipping starts at the lowest level with at most `delta` segments | 0 .. 1024 | 8 |
| `n` | no layer skipping (start at the root, the authors' published code path) | | skipping on |

Examples: `pgmpp64`, `pgmpp64i16`, `pgmpp16i4`, `pgmpp64d16`, `pgmpp64i16d16n`, `pgmpp1024`.

**The paper's rule** is: pick `eps_l` from the space budget (Eq. 22: `eps_l = sqrt(sizeof(seg) / B * sum over chunks P
of N_P * var_P / mean_P^2)`, the chunks from a change-point detection on the gaps), then `eps_i = argmin over E of
Cost(eps_i, eps_l)` (Eq. 23), both restricted to E = {4, 8, ..., 1024}; `delta` = 8 on Intel and ARM, 16 on AMD. It
gives no fixed recommended pair; it shows the cost as U-shaped in `eps_i` and reports results per index size. Here:

- `eps_l` is swept, which reaches every point Eq. 22 can produce (one index size per `eps_l` in E). A space budget is
  honoured by taking the smallest `eps_l` whose printed `dir_bytes` fits.
- `eps_i` is left out of the name, so it is chosen per data set and per `eps_l` by timing.
- `delta` stays 8: the benchmark machine is an Intel Xeon.

**Recommended sweep** (six names): `pgmpp8 pgmpp16 pgmpp32 pgmpp64 pgmpp128 pgmpp256`. The whole grid of the paper adds
`pgmpp4`, `pgmpp512`, `pgmpp1024`. For a run that must be reproducible to the last digit, take the `eps_i` printed by
the first run and name it (`pgmpp64i8`): candidates within timing noise can swap between runs.

**So that the baseline's best point is found whatever the tuner did**, also run each size with `eps_i` pinned to 4 and
to 8 (`pgmpp8i4 pgmpp8i8 ... pgmpp256i4 pgmpp256i8`) and report the fastest of the three per size. Reason (verifier,
Mac -O3, 30M keys, three distributions, `eps_l` 8 and 64, 3M timed lookups through the index, 36 tuned builds): the
tuner kept 4 or 8 in 35 builds, which were also the fastest fixed values in all twelve cases, and once kept 64 where
its nine timings tied within 2% although the long measurement put 64 about 10% behind 8. The timing at build is short (2^16 lookups per
block) and a laptop is noisy; the same index timed twice differed by up to 10% there. The paper reports the same order
of loss for its own tuner (within 10% of the optimum in 91% of cases), and for the PGM-index it reports the fastest
`eps_i` of the nine for each `eps_l`, so reporting the best `eps_i` per size is, if anything, generous to PGM++.

## Deviations from the paper and from the authors' code

1. **Lookup reimplemented** (one more comparison per branch-free level than the authors' code, which is what makes it
   correct). See above.
2. **Position evaluation.** Their `Segment::operator()` computes `size_t(slope * double(key - segment.key)) +
   intercept`, undefined when the product is 2^64 or more. Ours returns the next segment's intercept when the product is
   2^63 or more, which is what their `min(..., next.intercept)` makes of any product that large. It differs from theirs
   only for keys never stored, deep in a gap behind a steep segment (the test counts them: 50,720 probes, each checked
   to hold the true lower bound).
3. **Linear or binary: the code's rule, not the paper's formula.** The paper states the switch on the size of the range
   (Eq. 21d: linear when `2 * eps_i + 1 <= delta`), which with its own numbers would never scan at `delta` 8 (the
   smallest `eps_i` is 4). The authors' code compares `eps_i` itself with the threshold (`EpsilonRecursive <=
   linear_search_threshold`, 8 in `main.cpp`), so `eps_i` 4 and 8 scan. The two disagree; this follows the code, which
   produced their numbers. `d3` (its `delta` 8) or `d7` (its `delta` 16) gives the paper's formula inside the levels;
   the same number is then also the layer-skipping threshold (3 or 7 segments), the two cannot be set apart.
4. **Last-mile search: branch-free binary at every `eps_l`**, `SortedBoundIndex`'s, the same code as for `spgm`, `rs`,
   `cht` and `plex`. That is what the authors' `search_data` does. The paper's step 3 applies the hybrid there too; by
   its formula that is a linear scan only for `eps_l <= 3` at `delta` 8 and `eps_l <= 7` at `delta` 16, so for none of
   its error bounds on its Intel and ARM machines. Not implemented.
5. **Layer skipping** is on by default although the published code has it commented out; the threshold is `delta`
   segments without the sentinel entry (their commented block: 16 entries, sentinel included, whatever the threshold).
6. **Tuner by timing, not by the cost model.** The paper's model needs four machine constants (`C_miss`, `C_hit`,
   `C_segment`, `C_linear`) "estimated by probe datasets for each platform"; neither the constants nor code for the
   model or for Eq. 22 are published. Timing each candidate is what the paper calls `T_opt` in its Figure 14, the
   reference its own tuner is measured against (it reports its tuner picks the optimum in 46% of cases and is within
   10% of it in 91%). So this can only favour the baseline. It costs build time (nine builds of the internal levels and
   about 4.7M lookups) and can pick differently between runs when candidates tie.
7. **Segments are 16 bytes** (float slope, uint32 intercept), as in the authors' code. The paper's space model says 24
   bytes (double slope and intercept). Sizes reported here are the 16-byte ones, which favours the baseline.
8. **Parameters are run-time values.** The paper notes the search strategy of each layer can be fixed at compile time;
   here `eps_i`, `delta` and the strategy are members read per lookup (one well-predicted branch per lookup).
9. **Payloads.** The paper searches a bare array of keys; here the last mile runs over 16-byte (key, payload) pairs,
   as for every read-only index on this footing.
10. **The check of every stored key after the build** and the widening it can apply are ours (build time: about 2 to 4 s
    for 200M keys on one thread; in parallel on up to 20 threads with OpenMP, from 2^20 keys). Where it widens, all
    lookups of that index search a few more positions than the authors' structure would (and find the keys it loses).
11. **2^64 - 1** is handled outside the model (their builder refuses it).
12. **Below 64 keys** `SortedBoundIndex` builds no model and searches the whole array.
13. With OpenMP (the benchmark build has it) the authors' segmentation runs on up to 20 threads from 2^15 keys, in
    chunks, exactly as in their code and in GRE's PGM: the segment count then exceeds the serial one by at most the
    number of chunks.

What to expect: the paper's gain over the PGM-index comes in good part from replacing `std::lower_bound` by a
branch-free search in the last mile. `spgm` on this footing already has that, so the gap between `pgmpp` and `spgm` here
isolates layer skipping, the internal hybrid and the choice of `eps_i`, and is small (laptop, 200M generated keys, not a
benchmark: `pgmpp64` 172 to 174 ns against `spgm64` 181 to 197 ns; `pgmpp8` 146 to 152 ns against the same index
without skipping and with `eps_i` 4, `pgmpp8i4n`, 161 to 164 ns).

## Float slopes over long segments: the authors' structure loses keys there, and the guard catches it

Found by the verifier; not a property of our lookup but of the structure the authors' builder makes (and of any
PGM-index with `float` slopes, GRE's own copy included, which has no such guard). A segment's slope is computed
exactly and then stored as a `float`. The rounding is up to 6e-8 relative (3e-8 for a slope of 1/3), so the predicted
position drifts from the true one by about one position per 17M to 35M positions of the segment. A segment that long
exists only where the keys lie on one line over tens of millions of positions; then, at a small `eps_l`, stored keys
fall outside `[pos - eps_l, pos + eps_l + 2)`:

| keys on one line (stride 3), one thread | `eps_l` | stored keys outside the range of the authors' own `search()` | our build |
|---|---|---|---|
| 72M (the test's `check_drift`, Mac and cluster) | 1 | 4,891,135 | counted 4,891,135, widened by -1, every key found |
| 200M (Mac -O3) | 1 | 132,891,135 | widened by -4, every key found |
| 200M | 2 | 99,336,702 | widened by -3, every key found |
| 200M | 4 | 32,227,836 | widened by -1, every key found |
| 200M | 64 | 0 | nothing to widen |
| 140M in two chunks (cluster, `OMP_NUM_THREADS=2`) | 1 | 2,891,134 | widened by -1, every key found |

So on real data the build line can show `outside_bound=` above 0 and `widened=-1/+0` or so, for a near-linear data set
at `eps_l` 4 or 8; that is the guard working, not an error, and the lookups are exact. Widening applies to every lookup
of that index (a few positions more in the last-mile search). With OpenMP the authors' segmentation cuts the keys into
up to 20 chunks, so no segment is longer than a twentieth of the data (10M positions at 200M keys) and the drift stays
below one position: in the benchmark build the count is expected to be 0.

## Undefined behaviour in the authors' code (UBSan, file:line, reachable on real data?)

| where | what | on real data |
|---|---|---|
| `exp_pgm/pgm_index.h:327:26` (`Segment::operator()`), from their `build()` at `:112` | `size_t` of a double of 2^64 or more: each level's last segment is evaluated at 2^64 - 2 to decide whether one more segment is appended | **yes**, whenever the last segment of a level has a slope of about 1 or more, i.e. the largest keys are a dense run (consecutive or nearly consecutive integers). Sparse tails (slope far below 1) do not reach it. On every dense set of the test |
| same line, from their `search()` | same conversion | only for keys never stored, in a gap of about 2^63 / slope behind a segment. The index does not call their `search()`; the test does, as the reference |

Consequence for the index: none that can be wrong. Whatever the conversion yields, the builder either appends the
extra segment `(last key + 1, slope 0, n)` or does not; that segment only serves keys above the largest key, which `get`
never sends to the model, and the check of every stored key after the build does not depend on it. The cost is at most
16 bytes per level. A compiler is nevertheless free to do anything with undefined behaviour; nothing of the kind was
seen in the three builds tested.

**How the sanitized test deals with it, plainly**: in a clang build with UBSan, `pgmpp.h` switches the
float-cast-overflow check off for the functions of `exp_pgm/pgm_index.h` and for nothing else (a `#pragma clang
attribute` around that one include; ASan and all other UBSan checks stay on the authors' code, everything stays on the
wrapper and the test). Without that the sanitized test stops at the first dense set, before testing anything of ours.
`-DPGMPP_STRICT_UBSAN` leaves the check on and shows it:

```
exp_pgm/pgm_index.h:327:26: runtime error: 2.76701e+19 is outside the range of representable values of type 'unsigned long'
```

No other sanitizer report, from their files or from ours.

Other things found in the authors' code, none changed: the missing comparison in `upper_bound_branchless` (above);
`PGMIndex`'s default constructor leaves `n`, `first_key` and `start_level` unset (`PgmppBound` sets them);
`segments.reserve(n / (eps_l * eps_l))` reserves 16 bytes per key at `eps_l` 1 (3.2 GB for 200M keys, released after
the build; 200 MB at `eps_l` 4); `search_algo.h` uses `std::cout` without including `<iostream>` and
`piecewise_linear_model.h` prints a `#pragma message` and defines `omp_get_num_procs()` and `omp_get_max_threads()` as
macros when OpenMP is off (as GRE's PGM copy does, with the same text).

## How it maps to GRE's interface

Through `gret::SortedBoundIndex<K, P, PgmppBound<K>>`, exactly like `spgm`:

| GRE | here |
|---|---|
| `bulk_load` | the pairs are copied into one sorted array, the keys into a vector; the authors' `build`; the tuner if `eps_i` is not named; the check of every stored key |
| `get` | `PgmppBound::get(key)` gives `[lo, hi)`; `SortedBoundIndex` searches the pairs in it and compares the key. Exact for stored keys (checked at build) and for keys never stored (a key that is not in the array is never reported) |
| `scan` | lower bound through the same range, with `SortedBoundIndex`'s fallback to the whole array for a start key that was never stored and whose range misses, then a copy |
| `put`, `update`, `remove` | `false` (read-only) |
| `memory_consumption` | 16 bytes per pair + the model: `sizeof(PgmppBound)` (304 bytes) + 16 bytes per segment entry of all levels + 8 bytes per level offset and per level count, from the vectors' capacity after `shrink_to_fit` |

`SortedBoundIndex` default-constructs its Model and takes only a name, so `make_pgmpp` leaves the parameters in a
`thread_local` (`pgmpp_pending()`) that the Model's default constructor reads, and puts the defaults back (as
`make_radixspline` does). `pgmpp_last_build()` holds a report of the last build on the thread (chosen `eps_i`, levels,
bytes, keys outside the bound), which is how the test reads what an index made by name has built.

## Registry

Not edited by this work. Lines to add to `include/gret/registry.h`, inside the `GRET_SOTA` block, and
`-I <readonly_competitors>/bench_search` on the include path:

```cpp
#include "gret/sota/pgmpp.h"  // PGM++: the authors' builder, a reimplementation of the lookup (docs/sota/pgmpp.md)
...
    if (indexInterface<K, P>* i = sota::make_pgmpp<K, P>(name)) return i;        // pgmpp<eps_l>[i<eps_i>][d<delta>][n]: authors' builder, reimplementation of the lookup
```

Checked: a translation unit with `GRET_SOTA`, `registry.h` (RadixSpline, CHT, PLEX, RMI, hash tables, FAST) and
`pgmpp.h` after it compiles and runs; `pgm`, `spgm64` and `rs32r18` are not PGM++ names, `pgmpp64` is not taken by
`make_index`.

## Tests

`tests/sota/test_pgmpp.cpp`, 1189 checks. To build and run it (`BS` = the clone of bench_search, `R` = this repository):

```
# Mac, sanitized and -O3 (build/tests/cxx and cxx-O3 are made by tests/run_tests.sh)
mkdir -p $R/build/tests/sota/pgmpp
$R/build/tests/cxx    -I $BS $R/tests/sota/test_pgmpp.cpp -o $R/build/tests/sota/pgmpp/t    && $R/build/tests/sota/pgmpp/t
$R/build/tests/cxx-O3 -I $BS $R/tests/sota/test_pgmpp.cpp -o $R/build/tests/sota/pgmpp/t-O3 && $R/build/tests/sota/pgmpp/t-O3
# cluster login node, g++ 11 (wf_cxx = g++ -std=gnu++17 -O3 -DNDEBUG -march=native -fopenmp and the GRE include paths)
/scratch/lvasseur/wf/wf_cxx pgmpp -isystem /scratch/lvasseur/readonly_competitors/bench_search test_pgmpp.cpp -o t
for p in 0 1 2; do OMP_NUM_THREADS=4 OMP_WAIT_POLICY=passive nice -n 10 ./t --part $p/3; done
```


- **key sets**: one, two, three and five keys, 17, 63, 64 and 65 keys (either side of `SortedBoundIndex`'s threshold),
  the top 64 keys of the domain, uniform, wide (the whole 64-bit range), dense runs (from 1000 and from 0), far
  outliers, clusters, a stride, two halves 2^64 apart, a bit field with 2^64 - 1, a CDF that bends everywhere, largest
  key 2^64 - 2, - 3 and - 4, a stride ending at 2^64 - 2, steep segments followed by a gap of nearly 2^64, 300,000 keys
  (three and four levels), and three sets of 2M keys (wide; every scale at once with 0 and 2^64 - 1; dense runs and
  gaps);
- **the model** (13 configurations per small set, 5 per 2M set): every stored key inside its range, the range inside
  the array and no wider than `2 * eps_l + 2`; valid ranges for 0, 1, 2^64 - 1, keys below and above the range, random
  keys and keys inside gaps; parameters as asked; the tuner's result identical to the index its `eps_i` builds when
  named; bytes as counted;
- **against the authors' classes** (16 pairs of bounds per small set, 4 per 2M set, 8 of our configurations each):
  segments and offsets equal entry for entry; ranges equal to their `std::upper_bound` configuration for every stored
  key, both neighbours of 20,000 keys and the probes; their published branch-free configuration run and its lost keys
  counted;
- **the index, through `indexInterface`**: `get` on every stored key (payload checked), on both neighbours of every
  stored key, on 0, 1, 2, 2^63 - 1, 2^63, 2^64 - 4 ... 2^64 - 1, below and above the range, random keys and keys inside
  gaps; `put`, `update`, `remove` return false and change nothing; `scan` from stored, neighbouring, in-gap and extreme
  start keys at lengths 1 to n + 5, every pair compared with the sorted array and one guard slot past the end;
  `memory_consumption` equal to pairs plus model;
- **names** (11 valid, 34 invalid) and **builds that must throw**;
- **drift** (`check_drift`, added by the verifier): 72M keys on one line, a single segment whose float slope pushes
  4.9M stored keys out of their range at `eps_l` 1. The authors' own class is built on the same keys and its lost keys
  counted; the build's check must count exactly as many, widen the bound, and leave every key inside it. This is the
  only built-in set on which the bound is widened (the summary line "model builds whose bound had to be widened"
  counts the other sets and stays 0). About 0.6 GB and 3 s at -O3.

| where | result |
|---|---|
| Mac, arm64, Apple clang, `build/tests/cxx` (-O1, ASan + UBSan, asserts on), the command as given | `PASS test_pgmpp: 1189 checks`, no sanitizer output, 4 min 40 s (writer: 1188 checks before `check_drift` was added, 4 min 52 s; verifier: both). One UBSan check is off for one file of the authors', as described above |
| same with `-DPGMPP_STRICT_UBSAN` | stops in the authors' `pgm_index.h:327:26` at the first dense set |
| Mac, arm64, `build/tests/cxx-O3` | `PASS test_pgmpp: 1189 checks`, about 55 s |
| cluster (Jed login node), g++ 11.4.1, `-std=gnu++17 -O3 -DNDEBUG -march=native -fopenmp` (`wf_cxx`) | `PASS` in three runs, `t --part 0/3`, `1/3`, `2/3`: 379 + 405 + 405 = 1189 checks, 2 min 30 s, 2 min 26 s and 4 min 06 s of wall time in the verifier's run (the login node gives the account one core, shared with other work), `check_drift` included with the same 4,891,135 keys as on the Mac. The writer's runs (1188 checks, before `check_drift`) passed too, in parts and once as a whole in 4 min 33 s |
| Mac `-O3`, generated sets of 200M keys through the index (`t --only --cfg ... --gen ... 200000000`) | random gaps at `pgmpp64`; dense runs and gaps at `pgmpp4` and `pgmpp16i4`: every stored key found, probes, writes and scans exact; `bulk_load` 4.8 to 6.3 s on one thread, 9.6 GB peak for the test process |
| Mac `-O3`, 200M keys whose CDF bends everywhere (a scratch program, not in the repository) | 1,124,425 last-level segments at `eps_l` 8: `bulk_load` 8.5 to 9.0 s, 0 keys outside their range, 16.09 bytes per key |

### Independent verification (a second agent, files and scratch programs of its own)

Everything above was rebuilt and rerun from the files (Mac sanitized, Mac -O3, cluster g++ 11.4.1 in its own directory),
then attacked with a separate program kept outside the repository (`build/tests/sota/pgmpp/verify/adv_pgmpp.cpp`):

- **61 further key sets x 16 configurations** (6 for the four sets of 3M to 5M keys), through `make_pgmpp` only: one
  key (0, 2^64 - 1, 2^63), two keys, 62 to 67 keys with and without 0, 2^64 - 2 and 2^64 - 1, the top 64 keys of the
  domain, keys equal modulo 2^32 (sequential and random high halves, with low half 0xFFFFFFFF), keys differing only in
  the low 32 bits at the top and at the bottom of the domain, a dense run then a gap of nearly 2^64, dense runs at both
  ends, 300k keys above 2^63 and around 2^63, powers of two and their neighbours, alternating gaps of 1 and 2^44, a
  cubic, an exponential, clusters of every width, 4M uniform keys with 0 and 2^64 - 1, 5M lognormal, 3M in dense runs
  above 2^63, 3.5M over every scale. Each: `get` of every stored key; of both neighbours, the key two below, the middle
  of the next gap and the three keys equal modulo 2^32 (every stored key, or every seventh on the large sets); 60,000
  random probes; scans from 3,000 starts; writes refused. Configurations include `pgmpp1i1`, `pgmpp8i9`, `pgmpp16d0`,
  `pgmpp16i16d0n`, `pgmpp2i3d1`, `pgmpp64i1024n`, `pgmpp65536i65536d1024`, `pgmpp3i2d1024`, `pgmpp5i1d0n`.
  938 checks: PASS on the Mac sanitized (4 min 53 s, no sanitizer output), Mac -O3 and the cluster (in three runs).
- **`memory_consumption` against a counting `operator new`**: from 64 keys up the index really holds at most 144 bytes
  more than it reports (the index object around the model, and its name), on every set, 200M keys included.
- **Strict UBSan with recovery** (`-DPGMPP_STRICT_UBSAN -fsanitize-recover=undefined`, so that every site reports):
  one site only, `exp_pgm/pgm_index.h:327:26` from the authors' `build()`, as stated above. Nothing in the wrapper.
- **The authors' class alone** (their header, no wrapper): the lost keys of the published configuration, reproduced at
  20M keys (figures in "Can the authors' implementation be used as it stands?").
- **Float slopes over long segments**: found here (section above), then added to the test as `check_drift`.
- **200M keys, gaps of every size, Mac -O3, one thread**: `pgmpp4` (3,651,790 last-level segments, 16.29 bytes per key)
  `bulk_load` 8.4 s (model and tuner 5.2 s, check of every key 2.8 s); `pgmpp16` 5.6 s; `pgmpp64` 5.5 s; `pgmpp4i4`
  7.5 s; every stored key found, no key reported that is not stored, peak 25.3 bytes per key.
- **The tuner against every fixed `eps_i`** (30M keys, three distributions): see "Recommended sweep". Same runs, sanity
  only: `pgmpp64` 136 to 187 ns per lookup against `spgm64` 146 to 196 ns, faster in each of the six comparisons (by
  4% to 18%), so the baseline is not handicapped against the PGM-index on this footing.
- **Registry lines**: a unit with `GRET_SOTA`, `registry.h` and `pgmpp.h` compiles and runs; `make_index` makes nothing
  of `pgmpp64`, `pgmpp8i4n`, `pgmpp16d16`; `make_pgmpp` makes nothing of `pgm`, `spgm64`, `rs32r18`, `plex64`,
  `cht64e32`, `ht_std`, `fast`, `ab4`, `alex`, `lipp`, `pgm64`, `rmi18`.

`--part I/N` runs a share of the built-in sets, `--cfg NAME` and `--gen KIND N` a generated set at the benchmark's
scale, and a GRE key file can be given as an argument (not done: the 200M key files were not to be touched, and the
login node's limit of 8 GB per account does not hold a 200M-key run; one attempt there was stopped by that limit).

## Known limits

- Read-only: `put`, `update` and `remove` return false.
- Fewer than 2^32 keys (uint32 intercepts in the authors' `Segment`).
- Build memory at 200M keys: the 3.2 GB of pairs, a 1.6 GB copy of the keys during `bulk_load` (`SortedBoundIndex`),
  and the authors' reserve of `n / eps_l^2` segments.
- The `eps_i` chosen by timing depends on the machine and its load at build time; it is printed, and can be pinned.
  On a noisy machine it was seen to keep a value about 10% slower than the best once in 36 builds: run the pinned
  `i4` and `i8` variants beside the tuned one (see "Recommended sweep").
- Below 64 keys `SortedBoundIndex` builds no model and `memory_consumption` is the pairs alone (the 304-byte model
  object inside the index object is not counted there). From 64 keys up the verifier's counting allocator found the
  bytes really held to exceed `memory_consumption` by at most 144 bytes (the index object around the model, and its
  name), on every set up to 200M keys.
- Peak memory during `bulk_load` at 200M keys: 24 to 25.3 bytes per key held by the index and its build (16 for the
  pairs, 8 for the key copy, the model, the tuner's copy of the last level's keys), measured with 3.65M last-level
  segments; the caller's own array of pairs comes on top.
- Timing at build assumes the build runs on the machine and core type that will serve the lookups.
- Where the largest key is 2^64 - 3 or above (not counting 2^64 - 1 itself), the tuner rebuilds the whole index per
  candidate instead of the internal levels only (nine builds, one timing pass).
- The linear last-mile scan of the paper for `eps_l <= 3` (or `<= 7` at `delta` 16) is not implemented (deviation 4).
- The authors' builder runs undefined behaviour on key sets with a dense tail (above); the result is a valid index
  either way, but it is their code and it is not fixed.
- The sanitized test passes only because one UBSan check is switched off for one of the authors' files, in the header,
  under UBSan only.
