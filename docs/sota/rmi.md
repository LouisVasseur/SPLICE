# rmi: the two-stage recursive model index, a reimplementation

`include/gret/sota/rmi.h`, `tests/sota/test_rmi.cpp`.

**This is a reimplementation, not the authors' code.** Every index it builds reports itself as
`<name>(reimplementation)`, and its build line starts with `gret: rmi(reimplementation)`.

## What it is

RMI: Kraska, Beutel, Chi, Dean, Polyzotis, "The Case for Learned Index Structures", SIGMOD 2018. Reference
implementation: Marcus, Zhang, Kraska, "CDFShop: Exploring and Optimizing Learned Index Structures", SIGMOD 2020,
the RMI used in SOSD (Marcus et al., VLDB 2020).

| | |
|---|---|
| Reference repository | https://github.com/learnedsystems/RMI |
| Commit read | `6f03e69cfa73979d52c589c86ece0b6d37c69330` (27 May 2026) |
| Licence of the reference | MIT (Copyright 2019 Ryan Marcus) |
| Local clone (read only, nothing built or run) | `/Users/louisvasseur/Downloads/readonly_competitors/RMI` |
| Used how | read; nothing copied into this repository, nothing included from it |

Why not the authors' code: the reference is a Rust program (`cargo run -- <data> <namespace> <layers> <branching>`)
that writes a C++ file and parameter files for one dataset. cargo and rustc are not installed and may not be, so it
was not run. `rmi.h` writes again, in C++, what its sources do (`rmi_lib/src/train/two_layer.rs`,
`train/lower_bound_correction.rs`, `models/{linear,linear_spline,cubic_spline,radix}.rs`, `codegen.rs`).

The structure is the two-stage RMI: a root model sends a key to one of N = 2^L second-stage models; that one is a
line fitted by least squares that predicts the key's position, stored with the largest error of that prediction
(two doubles and one 64-bit error: 24 bytes a leaf, the reference's layout); the position is searched inside
[prediction - error, prediction + error].

## Names and parameters

`gret::make_rmi<K, P>(name)`; `nullptr` for any other name and for keys other than `uint64_t`.

| Name | Root | Notes |
|---|---|---|
| `rmi<L>` | chosen at build time | L in 1..28, no leading zero. The root with the lowest average log2 error among linear, linear_spline, cubic, robust_linear and radix: what the reference's optimizer keeps among models of one size |
| `rmi<L>lin` | linear | least squares over all keys |
| `rmi<L>spl` | linear_spline | the line through the first and the last key |
| `rmi<L>cub` | cubic | the reference's cubic; it keeps the line instead where the line's L1 error is lower |
| `rmi<L>rob` | robust_linear | least squares without the lowest and highest 0.01% of the keys |
| `rmi<L>rdx` | radix | the bits below the keys' common prefix |
| `rmi<L>rt18`, `rmi<L>rt22` | radix18, radix22 | the reference's radix tables, 1 MiB and 16 MiB more; never chosen by `rmi<L>` |
| any of the above + `k` | same | leaves and errors of the 2018 paper: see below |

The size/accuracy knob is L: 24 * 2^L bytes of leaves. The reference's optimizer walks 2^6 to 2^24. SOSD's
configurations for books, fb and osm at 200M keys (`scripts/rmi_specs/*_200M_uint64.json`, read online) are ten
points each between 2^7 and 2^24 leaves, all with linear leaves, with roots robust_linear (fb, and osm's three
smallest), cubic (osm, books), linear_spline (books), radix22 at 2^20 leaves (books, osm) and radix18 at 2^15 (osm).

A sweep that covers what SOSD ran: `rmi<L>` for L = 7, 8, 10, 15, 17, 18, 19, 20, 22, 23, 24, plus `rmi20rt22` and
`rmi15rt18`, and the same with `k`. The build line gives the root in use, the average log2 error, the largest
error and the model's bytes.

### The two kinds of leaves

- **Default (no `k`): the reference.** A leaf is trained on the keys the root sends to it plus the last key before
  them and the first key after them, and its error covers every key between the smallest and the largest stored
  key, stored or not. This is what the reference does since it added its lower-bound correction.
- **`k`: the 2018 paper.** A leaf is trained on its own keys, and its error is the largest over the stored keys.
  Nothing is promised for a key that was never stored. On keys with wide gaps the errors are much smaller: on the
  test's clustered 2M set the average log2 error at 2^18 leaves is 9.98 without `k` and 6.29 with it; on uniform
  keys 2.88 and 2.26.

Both are exact in this harness (next section). Which one is "RMI" is a choice a paper has to state; both are here
so that the baseline is not held back by a guarantee the workload does not use. GRE's lookups are of stored keys.

## How it maps to GRE's interface

`RmiBound<K, Root>` is a Model for `SortedBoundIndex<K, P, Model>` (`include/gret/readonly.h`), like `PgmBound`
(spgm): one sorted array of (key, payload) pairs, the model bounds where a key is, a branch-free binary search
inside the bound.

| GRE | Here |
|---|---|
| `init` | nothing |
| `bulk_load` | copies the pairs; trains the model on the keys (fewer than 64 keys: no model, the whole array is searched) |
| `get` | root, leaf, bound, binary search; false for a key that is not stored |
| `scan` | from the lower bound of the start key, in key order |
| `put`, `update`, `remove` | return false (read-only) |
| `memory_consumption` | 16 n for the pairs + 24 * 2^L for the leaves + the radix table if any + 144 bytes for the model object (root parameters) |

The largest key (2^64 - 1) and key 0 need no special case: nothing is a sentinel.

### What is guaranteed

- **Stored keys, every name.** A leaf's error is measured after training with the functions the lookup calls (one
  `std::fma` per model, as in the reference's generated code, so no compiler contracts or reorders anything), over
  every key the root sends to it. A stored key's position is inside its bound whatever the models are.
- **Keys never stored, names without `k`, every root but a cubic that stayed a cubic.** The root is monotone over
  the keys' range, in floating point as well (a correctly rounded `fma(slope, x, intercept)`, a shift, a
  non-decreasing table), and is checked to be non-decreasing on the stored keys. A query sent to leaf j therefore
  lies strictly between the last key before the leaf and the first key after it. The leaf's line is monotone too,
  so its error is taken over the stored keys, over every gap of two or more inside the leaf, and over the two gaps
  at its ends at their far ends (first key of the next leaf minus one, last key of the previous leaf plus one, as
  the reference does). A leaf with no key is the constant "position of the next key".
- **Cubic root, keys never stored: not proved.** The reference evaluates the cubic on the raw key by Horner's rule
  and so does this. Its "keep it monotonic" step does not make it monotone on every key set (the test's
  `over64outliers` set gives one that starts downwards at 2^5 leaves), and floating point adds its own. It is
  checked on the stored keys; if it fails there the root becomes linear_spline and the build line says
  `asked=cubic:failed,linear_spline_instead` (the reference stops with an assertion).
- **`k` names, keys never stored: nothing promised.**
- **Answers are exact in all cases.** `SortedBoundIndex::get` returns false for an absent key whatever the bound,
  and `SortedBoundIndex::scan` checks both ends of the bound against the array and searches on if the bound was
  wrong. A scan that starts at a key never stored can then cost one more binary search over the array.

The build line ends with `never_stored_bound=proved`, `not_proved(cubic)` or `none`.

## Deviations from the paper and from the reference

1. **Not generated code.** One implementation for every dataset: the number of leaves and the root's parameters
   are data members, not constants compiled in. With an explicit root (`rmi<L>cub`) the lookup is straight-line
   code as the reference generates; with `rmi<L>` the root's kind is one predictable branch per lookup.
2. **Choosing the root.** The reference's optimizer trains many (root, leaf, branching factor) combinations and
   returns the Pareto front over size and average log2 error; SOSD then times ten of them. Here a name fixes the
   number of leaves, and `rmi<L>` takes the lowest average log2 error among the five roots of 16 to 32 bytes. The
   radix tables are a different size and are separate names. The metric is the reference's (mean over the keys
   of log2(2 err + 2)) computed with this file's errors (item 5).
3. **Model types.** Roots: linear, linear_spline, cubic, robust_linear, radix, radix18, radix22. Leaves: linear
   only. Not implemented: linear_spline and cubic leaves; loglinear, normal, lognormal, bradix, histogram, radix8,
   radix26, radix28 roots. More than two stages: the reference does not build them either.
4. **Leaves trained in one sequential pass.** The reference trains two halves of the keys in parallel; at the seam
   it leaves one key out of training and the two leaves there do not get each other's neighbouring key. Not
   reproduced.
5. **Errors are exact, not bounded.** The reference stores max(largest error over the leaf's keys, error at "next
   key minus one", error at "previous key plus one") + the longest run of equal keys (1 for unique keys), with
   predictions clamped to n in training and to n - 1 in the lookup. Here: the largest |prediction - lower bound|
   over the stored keys and over the ends of every gap a query sent to the leaf can fall in, with the lookup's own
   clamp. At most the reference's value. The window searched is [p - err, p + err] (2 err + 1 positions); SOSD
   searches [p - err, p + err) (2 err positions, with the reference's err, which is one more than its largest
   measured error for unique keys).
   Measured by the independent verifier against a line-by-line transcription of the reference's sequential
   training (`two_layer.rs`, `lower_bound_correction.rs`), 876 (root, leaves, key set) cases, 8.2 million leaves:
   leaf parameters bit-identical wherever the routing is (it differs on 2 of 186 million keys, cubic root only:
   products for `powf`); this file's error is never above the reference's (equal on 29% of the leaves, below on
   71%). On 4 million uniform keys and on 5 million keys with lognormal gaps, at 2^10 and 2^18 leaves, the
   average log2 error printed here is 0.00 to 0.15 below the reference's, and the window actually searched
   (2 err + 1 here, 2 err there) is within 0.05 of a binary-search step of the reference's, on either side:
   neither helped nor hurt. On 3 million lognormal keys with 0 and 2^64 - 1 added, where the spline, cubic and
   radix roots send most keys to leaf 0, it is 0.2 to 0.55 lower here: the next paragraph.
   One case where this file is clearly better than the reference: the reference's lower-bound term for leaf 0
   compares the prediction at key 1 with the index of the first key of the *next* non-empty leaf, so leaf 0's
   error is at least the number of keys in leaf 0. With 2^20 leaves or more that is one leaf of a few keys; with
   a root that sends a large share of the keys to leaf 0 (few leaves, or a linear root on keys with far outliers)
   the reference's window is up to twice this file's (average log2 error up to 1 lower here, and more where the
   whole key set is in leaf 0 and fits a line). Such configurations are not competitive for either.
6. **Leaves with no key.** All are constants; the reference leaves the last leaf as the zero model.
7. **The root's prediction is always clamped** to [0, N - 1], NaN to 0. The reference skips the clamp after a
   cubic or radix root (`(uint64_t) fpred`), which reads outside the leaf array for a prediction outside it.
8. **Where the reference stops or produces non-finite parameters, this falls back and says so.** A root that is
   not non-decreasing on the stored keys becomes linear_spline. Keys that are different integers and the same
   double (2^62 + i): linear_spline is a constant, the cubic is not built. robust_linear needs more than
   2 * max(1, n / 10000) + 1 keys.
9. **Radix table.** Entries past the last key's prefix hold N - 1; the reference writes the table's length there.
10. **Arithmetic.** The cubic's coefficients use products where the reference calls `powf`; the regression code
    may be contracted differently by different compilers. Parameters can differ in their last bits; validity does
    not depend on them (item 5).
11. **Size.** The reference reports 24 N + the root's parameters (16 or 32 bytes, or the table). Here
    `memory_consumption` adds the whole model object (144 bytes with this compiler) and the pairs.
12. **Search and layout.** `SortedBoundIndex`: branch-free binary search over 16-byte (key, payload) pairs, the
    same as spgm here. SOSD searches a keys-only array and has other last-mile searches.
13. **Keys.** `uint64_t`, unique (GRE's bulk load). The reference's handling of duplicates, of 32-bit and float
    keys is not here.
14. **The `k` names are not in the reference.** They follow the 2018 paper (errors over the stored keys), with one
    symmetric error per leaf where the paper stores a minimum and a maximum.
15. **Not here:** the reference's `cache_fix` (bounded spline in front), `--no-errors`, `train_for_size`.
16. **Build.** Single-threaded; `rmi<L>` trains five models and keeps one. Build time is not the reference's
    (about 10 ns per key for an explicit root and 40 to 50 ns per key for `rmi<L>` at 4 to 5 million keys on
    this Mac): do not report it as RMI's build time.
17. **Fewer than 64 keys.** `SortedBoundIndex` builds no model and searches the whole array (as for spgm).
18. **The name in GRE's result files.** GRE writes the `--index` name (`rmi20`) to its CSV, without the
    `(reimplementation)` that the build line on stderr carries. Tables and figures made from those files must
    add the label themselves.

## Known limits

- **Not checked against the reference's output**: it could not be run. A check that needs no cargo: SOSD publishes
  the reference's average log2 error per configuration, for example fb, robust_linear, 2^24 leaves: 4.516; osm,
  cubic, 2^24: 7.209; books, linear_spline, 2^23: 3.828. `rmi24rob` on fb, `rmi24cub` on osm and `rmi23spl` on
  books should print an `avg_log2_err` at or just below these (item 5). Not done: the 200M files are not on this
  machine.
- **Not compiled with g++ 11 on Linux x86-64.** Compiled with Apple clang 17 for arm64 (`-O1` with ASan and
  UBSan, and `-O3`) and for x86-64 with and without `-mfma -mavx2`, the x86-64 binaries run under Rosetta: all
  pass. The independent verifier also built the test with Homebrew GCC 15.2 (`g++-15`, arm64, libstdc++,
  `-O3 -std=gnu++17 -Wall -Wextra`, GCC's default `-ffp-contract=fast`): no diagnostic from `rmi.h` or the test
  (none either with `-std=c++17 -Wpedantic -Wconversion -Wshadow`), `test_rmi: ok (1074 cases, 0 failed)`. So
  GCC's front end and libstdc++ are covered, GCC 11 itself and Linux are not. The header uses no intrinsics, no
  `long double`, and `__builtin_clzll` only.
- **FMA.** `std::fma` is one instruction with `-march=native` on x86 with FMA3 and on arm64. Without it, it is a
  libm call per model evaluation: correct, slow.
- **No `-ffast-math`.** The equality of build-time and lookup-time predictions assumes IEEE arithmetic. `rmi.h`
  stops the compilation with an `#error` if `__FAST_MATH__` is defined (added by the verifier; GRE's flags and
  this project's do not set it).
- **Build cost at 200M keys, estimated, not measured.** About 8 ns per key per trained model at 2M keys on this
  Mac, so roughly 2 s for an explicit root and 10 s for `rmi<L>`, plus the passes over the leaves. Extra memory
  while building: one more set of leaves for `rmi<L>` (2 * 24 * 2^L in all), on top of `SortedBoundIndex`'s copy
  of the keys.
- **The number of leaves reaches the model through a thread-local** set by `make_rmi` and taken by the model's
  constructor, because `SortedBoundIndex` default-constructs its Model. A model constructed any other way without
  a number refuses to build.
- **Absent start keys in scans** with a cubic root or a `k` name can cost a second binary search.

## Tests

`tests/sota/test_rmi.cpp`, built with `build/tests/cxx` and `build/tests/cxx-O3` (no third-party include path):

- the index through GRE's interface, 33 names on 26 key sets and 13 names on two 2M-key sets: get on stored keys,
  on both neighbours of stored keys, 0, 1, 2^64 - 1, below and above the range, random; scans from stored keys,
  neighbours, gaps and the extremes at seven lengths; refused writes; `memory_consumption`;
- the model alone, every root at 2^1 to 2^20 leaves, with and without `k`: 149 million stored keys inside their
  bound; 130 million absent keys inside a proved bound; 0 of 4.4 million absent keys outside a cubic's bound (not
  proved, observed); with `k`, 0.78 million of 118 million absent keys outside their bound (none promised);
- `rmi<L>` has the lowest average log2 error of the five small roots; a dense run is predicted within one
  position; the names taken and refused.

`test_rmi: ok (1074 cases, 0 failed)` in both builds.

Independent verification (a second agent, its own scratch program, not in `tests/`): 34 names on 50 further key
sets through GRE's interface (one key, two keys, 63 to 66 keys, 0 and 2^64 - 1 stored and not stored, all keys
equal mod 2^32, dense runs then a huge gap, keys above 2^63, keys around 2^53 and 2^63, powers of two) and 13 names
on 3, 4 and 5 million random keys: every get, absent neighbour and scan against a sorted array;
`memory_consumption` within 512 bytes of the bytes actually live after the build (allocations counted); for the
roots whose bound is proved, every key of 20 small universes (65536 consecutive values at 0, at 2^40, around
2^63 and just below 2^64) inside its bound at seven leaf counts; the comparison with the reference's training of
deviation 5; the `k` leaves are the least-squares line of their own keys with exactly the largest error over
them. No failure with Apple clang (`-O3`; `-O1` with ASan and UBSan on a subset of the probes), with GCC 15.2,
or as x86-64 (`-march=haswell`) under Rosetta.

## Adding it to the registry

```cpp
#include "gret/sota/rmi.h"   // after "gret/readonly.h"

    // in make_index, next to the spgm lines (K is uint64_t there):
    if (indexInterface<K, P>* rmi = make_rmi<K, P>(name)) return rmi;  // rmi<L>[root][k]: RMI, a reimplementation (sota/rmi.h)
```
