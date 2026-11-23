# bli: BLI behind GRE's interface

`include/gret/sota/bli.h`, `tests/sota/test_bli.cpp`. Names: `bli[<L>][s<S>][f<PP>][e<E>][u]` (`bli`, `bli16`, `bli20s8`,
`bli17f70`, `bliu`).

## What it is

BLI (Dong, Wang, Liu, Du: "BLI: A High-performance Bucket-based Learned Index with Concurrency Support", arXiv 2502.10597,
February 2025) is an updatable learned index that is "globally sorted, locally unsorted". The pairs sit in **D-Buckets**:
fixed-size arrays that are not sorted inside; a pair goes to the first free slot at or after a hint (a hash of the key
modulo the bucket size), so a D-Bucket is a small open-addressing hash table with a bitmap of used slots. Above them are
levels of **Segments**: a linear model over an array of small **S-Buckets**, each entry the smallest key of a child and a
pointer to it. A full D-Bucket is split at its median; a Segment that has no room for the new pivot is cut again
(greedy error corridor) and its pieces go to the parent.

This is a **wrapper around the authors' own code, used unmodified**; it is not a reimplementation.

| | |
|---|---|
| repository | https://github.com/gdymind/Buck-Learned-Index (the single-threaded index: `include/buck_index.h`) |
| commit | `b229631d0038eb762c46678e4ed6b147f0ebd7aa` (2025-05-15, the head of the default branch) |
| local clone | `/Users/louisvasseur/Downloads/readonly_competitors/Buck-Learned-Index` (cluster: `/scratch/lvasseur/readonly_competitors/Buck-Learned-Index`) |
| licence | **none stated**: no LICENSE file, no licence header in the index's files (`tscns.h`, a timer by Meng Rao, is MIT). Included by path only, nothing copied into this repository |
| the authors' benchmark | https://github.com/wwl2755/GRE_buck_index, commit `fd5f6b6cc83ff8412e0ddb8a5e756ebf4462251f` (2025-09-13): a GRE fork with their wrapper (`src/competitor/buckindex/buckindex.h`) and the scripts of the paper's figures. Local clone `.../readonly_competitors/GRE_buck_index`. **Read for its settings and its wrapper; no file of it is compiled here** |

Why this is taken to be the authors' code: the paper's own artifact statement (arXiv HTML, front matter) names both
repositories, `github.com/gdymind/Buck-Learned-Index` and `github.com/wwl2755/GRE_buck_index`; the fork's README says it is
"to reproduce the results in our paper" and names the paper; and its `.gitmodules` takes the index from
`github.com:gdymind/BuckIndex`, at a commit that is in the history of the repository cloned here. The commits are by the
two accounts that own the repositories the paper names (gdymind, wwl2755).

To get the same sources (nothing else is needed: no build step, no library):

    git clone https://github.com/gdymind/Buck-Learned-Index && git -C Buck-Learned-Index checkout b229631d0038eb762c46678e4ed6b147f0ebd7aa
    git clone https://github.com/wwl2755/GRE_buck_index && git -C GRE_buck_index checkout fd5f6b6cc83ff8412e0ddb8a5e756ebf4462251f   # to read only

The fork pins the index as a submodule at `84caf59` (2024-10-31), an ancestor of the commit used here. The nine commits
between them change `scan` only (a top-k selection in each bucket, a parallel variant with worker threads); `lookup`,
`insert`, `bulk_load` and the Segment code are the same in both.

Files of theirs that are compiled: `include/buck_index.h`, `src/bucket.h`, `src/segment.h`, `src/segmentation.h`,
`src/greedy_error_corridor.h`, `src/linear_model.h`, `src/keyvalue.h`, `src/util.h`, `tscns.h`. Not used:
`include/bli_concurrent.h`, `include/bli_async.h` (the concurrent variants), `unittests/`, `util/`, their CMake.

## Parameters

`bli` is the setting of the authors' overall comparison (`script/run_overall.sh` in their fork, the paper's Figure 6):
D-Buckets of 131072 slots, S-Buckets of 4 entries, initial fill ratio 0.6, error bound 4, hint = murmur64, regression for a
new root (`HINT_MODE=murmur_hash` and `BUCKINDEX_USE_LINEAR_REGRESSION` are the defaults of their CMake).

Knobs in the name, in this order, each optional:

- `<L>`: log2 of the D-Bucket size, one of 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 22, 24 (`bli` = `bli17`). This is the
  knob the paper tunes (section 6.3, Figure 7; `script/run_s_dbuck_size.sh` sweeps 16384 to 33554432 slots). It is a
  template parameter of the authors' class, so every size is a separate instantiation.
- `s<S>`: the S-Bucket size, 4, 8 or 16 (default 4, which the paper finds best for inserts).
- `f<PP>`: the initial fill ratio in percent, 30 to 90 (default 60; section 6.4 and Figure 8 find 60 best;
  `script/run_fill_ratio.sh` sweeps 30 to 90).
- `e<E>`: the error bound of the bulk load's segmentation of the model levels, 1 to 1024 (default 4, the scripts' value).
  Leave it at 4: large values make defect 3 below likely.
- `u`: `put` calls the authors' insert alone, without looking for the key first (below).

**Recommended sweep** per data set: `bli16`, `bli17` (= `bli`, the authors' setting), `bli18`, `bli20`, and `bli17f70` for
the fill ratio; `bliu` once on an insert workload to see what the duplicate check costs. D-Buckets below 2^16 slots cannot
hold keys of every magnitude (defect 1) and are for tests.

Fixed at compile time by this header: `HINT_MURMUR_HASH`, `BUCKINDEX_USE_LINEAR_REGRESSION`; `BUCKINDEX_USE_SIMD` and
`BUCKINDEX_DEBUG` must be off (the header refuses to compile otherwise).

## The released code and the paper are not the same thing

Everything below is how the authors' code is; none of it is changed here.

- **No SIMD in the lookup.** `bucket.h` has an AVX2 lookup (`SIMD_lookup`, line 525), but `BuckIndex` stores its D-Buckets
  as `KeyValueList` (keys and payloads interleaved, `buck_index.h:31`), for which the SIMD key load is
  `assert(false); return __m256i();` (`bucket.h:484-486`): with `BUCKINDEX_USE_SIMD` on, every lookup would compare against
  a zero vector. The option is off in the authors' CMake. A D-Bucket lookup is a scalar scan from the hint
  (`bucket.h:406-413`), which is also what the paper's text describes (sections 4.2.2 and 5.1.2). So `bli` needs AVX2 only
  to compile (`bucket.h` includes `<immintrin.h>` and names `__m256i`).
- **A lookup of a key that is not stored reads the whole D-Bucket.** The scan does not stop at a free slot
  (`bucket.h:406`): 131072 slots, about 2 MB, per miss. GRE's read workloads look up stored keys, which end at the key.
- **A scan reads the whole D-Bucket too**, once for each bucket it touches: a top-k selection with a priority queue over
  all slots (`bucket.h:265`). The paper (5.1.3) describes a parallel sort by several threads; the code has that as
  `scan_parallel`, which the authors' GRE wrapper calls for scans of more than 1000 keys. The wrapper here calls the
  single-threaded `scan` always: the comparison is single-threaded.
- **No merging.** The paper lists node merging (5.3.2, `MergeNeighbors`) among the maintenance operations; the released
  single-threaded index has no merge, and no deletion at all.
- **Endpoint models at the bulk load.** The paper says Segments use a linear regression model (5.1.1). In the code the
  regression of the segmentation sits behind a misspelled macro (`BUICKINDEX_USE_LINEAR_REGRESSION`,
  `segmentation.h:70, 81, 96`), so bulk-loaded and re-cut Segments get the line through their first and last key
  (`Cut::get_model`, `segmentation.h:39`). Only a new root is fitted by least squares (`buck_index.h:446`).
- **Error bounds.** The paper (5.3.2) says one error bound serves the bulk load, splits and merges. The code uses the
  given bound (4) at the bulk load and `0.5 * SBUCKET_SIZE` (2 for S = 4) when a Segment is cut again (`segment.h:412`).
- **Hint.** The paper's figures show a MOD hash and compare it with a CL hash and a model; the code's default is murmur64.
- **A smallest key is required.** The structure routes a key through the entry with the largest key at or below it, and
  has nothing for keys below the first pivot. The authors' wrapper inserts the pair (0, 0) when the data does not start at
  0, and their insert turns a put of key 0 into an update (`buck_index.h:381`). With that alone, `get(0)` answers
  "found, 0" on every data set. The paper does not mention it.
- **Eleven idle threads.** The constructor starts 11 workers for `scan_parallel` (`buck_index.h:45`). They sleep on a condition
  variable for the whole run here.
- **`memory_consumption`** of the authors' wrapper returns the number of levels (a TODO). Here it returns bytes.

## How it maps to GRE's interface

| GRE | what runs |
|---|---|
| `bulk_load` | the authors' `BuckIndex::bulk_load` on a copy of the pairs (their type is a vector of `KeyValue`), then a check of every key and a count of the pairs held |
| `get` | the authors' `BuckIndex::lookup`. On a miss, a second walk that tells whether the first was misrouted (defect 2) |
| `put` | the authors' walk (`Segment::lb_lookup` per level), the duplicate check, then the authors' `Bucket::insert` on that D-Bucket; when it is full, the authors' `BuckIndex::insert`, which splits (and, if a Segment above is at risk of defect 2, a check of the Segments afterwards). With `u`: the authors' `BuckIndex::insert` and nothing else |
| `update` | the authors' walk, then the payload is overwritten in place (the authors' code has no update) |
| `remove` | not supported, returns false: the structure has no deletion |
| `scan` | the authors' `BuckIndex::scan`, then key 2^64 - 1 if it is stored |
| `memory_consumption` | `sizeof` of every D-Bucket (all slots, used or not) and of every Segment with its S-Buckets, the authors' index object, the wrapper object. Equal to the authors' `mem_size()` plus the wrapper (the test checks this). It is also where the structure is audited, because GRE gives no other hook after a run |

`get` adds one predictable branch (keys 0 and 2^64 - 1) to the authors' lookup; the index object is reached through one
pointer that stays in the first-level cache. The payload is stored in the authors' slot.

## What the wrapper adds

1. **Key 0.** The pair (0, 0) is always in the authors' structure; a flag says whether key 0 is stored. `scan` starts at 1
   when it is not.
2. **Key 2^64 - 1** is kept in the wrapper (a flag and a payload): it is the authors' "empty S-Bucket" pivot
   (`bucket.h:52`), and their lookup of it walks into empty S-Buckets and returns an uninitialised pointer.
3. **`put` looks for the key first.** The authors' insert does not, and stores a second copy of a stored key. The check
   scans from the hint to the first free slot. That is exact, and much shorter than the authors' lookup of an absent key,
   because their buckets fill the first free slot at or after the hint (`find_empty_slot`, `bucket.h:311`) and never free
   one. Names ending in `u` skip the check: for callers that never put a stored key (GRE's insert workloads do not). A
   stored key put again through `u` becomes a second pair, as in the authors' code; `memory_consumption` then reports more
   pairs than keys.
4. **`update`**, in place.
5. **Guards for three defects** of the authors' code (next section), each counted and reported.
6. **Checks.** `bulk_load` looks every key up afterwards and counts the pairs, and throws `std::runtime_error` if the
   structure does not answer exactly. `memory_consumption` counts the pairs again and looks up every stored key that
   defect 2 can touch (`BLI_AUDIT=1`: every key, and every key must lie in the key range of its D-Bucket); it throws
   `std::runtime_error` if pairs are missing, a key is misplaced, a Segment's S-Buckets are not in key order, or a key is
   not found. It prints one line to stderr:
   `gret: bli now keys=... pairs_in_buckets=... levels=... bucket_splits=... new_roots=... risk_segments=... risk_puts=...
   refused_puts=... risk_splits=... keys_the_authors_lookup_misses=... gets_served_past_it=... misplaced=...
   segments_out_of_order=... wrong=...`.
7. **Housekeeping.** The authors' progress lines on `std::cout` are silenced (`BLI_VERBOSE=1` keeps them); the destructor
   frees the nodes (the authors' frees none).

How the unmodified headers are brought in: inside `namespace gret::bli_authors` (they say `using namespace std;` and define
`Bucket`, `Segment`, `Cut` globally), `util.h` in an unnamed namespace (it defines `murmur64` without `inline`), every
standard header they include included before, and **`private` defined as `public` while their headers are read**: the
wrapper needs the root pointer, a Segment's model and a D-Bucket's slots, which have no accessors. No file of theirs is
edited.

## Three defects of the authors' code, and what guards them

**1. Keys the models cannot place** (`segment.h:91-109`). The Segment constructor computes an S-Bucket number from a
double model and indexes the array with it, without a bound. In exact arithmetic the last pivot lands a little more than
one slot before the end. The rounding error grows with key magnitude times slope, and the slope is at most
1 / (pivot distance x fill), the pivot distance being at least the number of keys of a D-Bucket when it is made. So for
D-Buckets of 2^16 slots and more (at fill 0.6) the error stays under a slot for every key below 2^64 - 1; smaller buckets
have a limit (`safe_key()`: 2^56.3 for 256 slots, 2^63.3 for 2^15). **Guard:** keys above the limit are refused with
`std::invalid_argument` (bulk load and put). Test set `refuse`.

**2. A truncated S-Bucket number** (`segment.h:300`). `predict_buck` converts the predicted S-Bucket number to
`unsigned int` before clamping it to the array. For a key far above the keys the Segment was fitted to (an outlier behind
a dense region: the factor is about 2^32 x S x fill / pivots times the Segment's key span) the number passes 2^32 and only
its low 32 bits survive. The search for the key's S-Bucket then starts somewhere below the right one, and going up it
stops at the first empty S-Bucket (`segment.h:315`; an empty S-Bucket has pivot 2^64 - 1). The walk ends in a wrong
D-Bucket. Consequences in the authors' code: a lookup misses a key that is stored where it belongs, and an insert stores a
key in an earlier D-Bucket, where lookups find it, scans return it out of order and a later split strands it.
**Guards:** (a) `get` and `update`: after a miss, the walk is repeated with a test for truncation; if it truncated, the
key's D-Bucket is found from the pivots alone and searched; (b) checked `put`: the same test on the walk it does anyway
(one multiply-add and compare per level); a truncating key is checked against the pivots and refused with
`std::runtime_error` if the authors' walk would misplace it; (c) `scan` needs nothing: a misrouted start is an earlier
bucket and the scan filters by key; (d) with `u` the authors' insert is not watched; `memory_consumption` reads every
D-Bucket below a Segment at risk and throws if a key is outside its bucket's range; (e) a misplaced pivot, next
paragraph.

**2 (e). A misplaced pivot.** The same truncating search places the pivots that a split makes: `batch_update` calls
`locate_buck` for the new pivot (`segment.h:231, 255`) and inserts it into the S-Bucket that search ends in (`segment.h:260`; if that S-Bucket is full
the insert fails and only an `assert` notices: the pivot and its subtree are dropped). A pivot whose S-Bucket number
truncates therefore lands in an earlier S-Bucket. Lookups of its keys still arrive, because they make the same wrong
search, so no key is missing and no key is outside its D-Bucket's range; but the Segment is no longer in key order, and
the authors' scan, which walks the S-Buckets in sequence, **returns keys out of order, silently**. It takes a D-Bucket
whose upper half consists of keys the walk misroutes (they arrived with a bulk load, or before a Segment above was cut
again: the checked put refuses a key that is misrouted at the time it is put), filled until it splits; or a Segment cut again whose new pieces start at keys that truncate in the parent.
**Guard:** before the authors' insert splits a full D-Bucket, the checked put asks whether any Segment on the path
computes an S-Bucket number of 2^32 or more for a key that can be routed to it (up to the next pivot of its parent, and
no more than the largest key stored: the test behind `risk_segments`); if so, after the insert every Segment is checked for
key order (each S-Bucket above the ones before it, its pivot its smallest key) and the reachable pairs are counted, and
`std::runtime_error` is thrown if either is off (`risk_splits` counts these checks). The structure is out of order by
then and the run must stop. With `u`, and for everything else, the audit in `memory_consumption` makes the same order
check on every Segment (`segments_out_of_order`) and throws. Test set `disorder` provokes the first way (a D-Bucket
split); the second (pieces of a Segment misplaced in the parent) is covered by the same check but no test provokes it.
This guard was added by the independent verification: the first version of the wrapper answered such scans out of order
without any report.

**How likely is defect 2.** It is **provoked** in the test sets `truncation` and `disorder` (256-slot buckets, a regular
stride with one hole 3.5 pivot distances wide, default error bound): the authors' lookup misses a loaded key, their
insert misplaces a key, and their split misplaces a pivot. At 131072 slots a truncating key is misrouted only if the low 32
bits of a large number fall below the index of an empty S-Bucket, a chance of the order of 10^-7 per outlier key: on the
200M sets expect `risk_segments` above zero on data with far outliers (fb) and the other counters at zero.

**3. A new root built past its array** (`buck_index.h:446-459`, `segment.h:98, 109`). When the root is cut, the new root
is a Segment over the pieces with a least-squares model, built by the same unbounded constructor. A least-squares line
overshoots the last pivot when the pivots lie dense first and sparse later; with nine or more pieces the overshoot can
reach a whole S-Bucket, and the constructor then reads and writes behind the array (**AddressSanitizer:
heap-buffer-overflow, `segment.h:98`, called from `buck_index.h:458`**). **Does real data reach it?** It was provoked only
with a bulk-load error bound of 1024 on keys with exponentially growing gaps (60,000 keys, 256-slot buckets: the 348th
put); with error bounds 4, 8, 16 and 64 the same keys took 2 million puts without it, and no default test set triggers
it. With the default bound the root is cut into few pieces at a time, so it is unlikely, not impossible. **Guard:** after
every new root the pairs reachable from it are counted (one pass over the nodes, no slot is read); a pivot written behind
the array is unreachable with its subtree, so a shortfall throws `std::runtime_error`. Memory is corrupted by then: the
run must stop, and may crash before the count. Test set `rootoverflow` (not in the default run: it corrupts the heap on
purpose).

The alternative the authors' build does not use has its own version of defect 3: the endpoint model for a new root
(`buck_index.h:449-456`) predicts `size` for the last of `size` pivots, which is the slot after the array whenever
`size / fill` is a multiple of S. That is why the authors' default, regression, is kept.

## Build

x86-64 with AVX2. Two include paths, no source file, no library:

    -I <path>/Buck-Learned-Index/include -I <path>/Buck-Learned-Index/src        (-isystem to keep their warnings out)

Without the authors' headers on the include path, or on a machine without AVX2 (arm64), the header compiles to
`make_bli` alone, which returns nullptr for other names and throws `std::invalid_argument` for a `bli` name. One
translation unit of a program should include the header. The header sets the authors' compile-time options itself and
restores the macros afterwards.

On arm64 the test builds with `-DGRET_BLI_SHIM` and a 14-line stand-in `immintrin.h` on the include path (types and
declarations only; none is called; the text is in `tests/sota/test_bli.cpp`). That is for testing the wrapper natively
under the sanitizers, not for measuring.

Compile time: the name selects one of 36 instantiations (12 D-Bucket sizes x 3 S-Bucket sizes) and all are compiled: about
a minute of g++ 11 -O3 for the test.

Registry (not added by this change; `GRET_SOTA_BLI` like `GRET_SOTA_DILI`):

    #ifdef GRET_SOTA_BLI  // needs Buck-Learned-Index/include and Buck-Learned-Index/src on the include path; x86-64 AVX2
    #include "gret/sota/bli.h"
    #endif
    ...
    #ifdef GRET_SOTA_BLI
        if (indexInterface<K, P>* i = make_bli<K, P>(name)) return i;  // bli[<L>][s<S>][f<PP>][e<E>][u]: the BLI authors' code, guarded
    #endif

## Cost at 200M keys

Bulk load: the authors' API takes a vector of their pair type, so the pairs are copied once (3.2 GB, freed after the
load); the D-Buckets are 16 bytes a slot at the fill ratio: 26.7 bytes a key at 0.6 plus bitmaps and Segments (27.5
measured on 2M to 4M keys, 5.5 GB at 200M). The peak during the load, measured through operator new, is 43.5 bytes a key
(8.7 GB at 200M, next to the caller's own 3.2 GB array). Time: on the Mac (-O3, arm64) the whole `bulk_load`, check
included, is 39 ns a key at 3M keys; on Jed's login node, shared with other users at about a seventh of a core, 2M keys
took 100 ns a key for the authors' load and 250 ns for the check, and 20M random keys (3 levels, 255 D-Buckets, 26.95
bytes a key, 42.95 at the peak) 9.4 s of wall time (`load_ms` 2205, `verify_ms` 6107), the whole program with 45M gets,
5M puts and 60 scans 7.1 s of CPU. At 200M that is of the order of a minute, single-threaded, at worst. Nothing was run
on the 200M files. The wrapper's check after the load is one lookup per key and one pass over the nodes;
the line `gret: bli (BLI: the authors' code, unmodified, behind guards) ...` gives `load_ms` (the authors' bulk load alone)
and `verify_ms` separately. A build time taken around `bulk_load` includes the check; use `load_ms` for the authors' part.

What the wrapper costs, on 3M uniform keys and 1.5M puts (`t --time`; Jed's login node under `nice`, shared with other
users, so **indicative only: differences of 30% between two lines are noise here**):

| | get | put | |
|---|---|---|---|
| the authors' `lookup` and `insert` called directly, 131072 slots | 109 ns | 247 ns | |
| `bli` | 100 ns | 200 ns | the duplicate check, then the authors' `Bucket::insert`; the authors' `BuckIndex::insert` only when the bucket is full |
| `bliu` | 148 ns | 321 ns | the authors' `BuckIndex::insert` alone |
| `bli20` / `bli20u` | 110 / 117 ns | 196 / 284 ns | |
| `bli14` / `bli14u` | 119 / 107 ns | 208 / 328 ns | |
| GRE's ALEX, LIPP | 114, 143 ns | 389, 227 ns | same binary, for scale |

The verification reran `t --time` on the same node under heavier load (about a seventh of a core): every line, ALEX and
LIPP included, came out three to four times slower in wall time (gets 300 to 450 ns), with no order among the BLI lines
that repeats. These numbers say only that the wrapper's cost is below what this node can resolve; the real comparison
is the benchmark job.

The checked put is not slower than the authors' insert, and seems faster: their insert allocates a vector for the path on
every call (`buck_index.h:358`), which the wrapper's own walk does not. So `bli`, not `bliu`, is the name to time; it is
the setting that favours the baseline and the one with the guards on.

## Tests

`tests/sota/test_bli.cpp`: every answer against a `std::map`. Key sets: none, 1, 2, 3, 5, 17, 300 keys; an index never
loaded; uniform; dense runs (from 1000 and from 0); clusters; a regular stride; lognormal gaps; keys over the whole 64-bit
range with 0, 2^63 and 2^64 - 1; a narrow bulk with far outliers; a run up from 0 with a run down from 2^64 - 1; small keys
with a few very large ones; 2M random keys. D-Buckets of 2^8 to 2^20 slots, S-Buckets of 4, 8, 16, fill 0.3 to 0.9, error
bounds 1 to 16, checked and unchecked put. Per set: get of every stored key and of absent keys (neighbours, gaps, 0, 1,
2^63, 2^64 - 1, random), scans from stored, neighbouring, in-between and extreme starts with lengths 1 to 70,000 and one
over everything, puts (neighbours, runs, below and above everything, duplicates), updates of stored and absent keys,
removes (must return false), then keys appended above the largest key and a dense run into one gap, so that buckets and
Segments split and roots are added; after each step the full audit (which includes the key order of every Segment). Then
the sets named above for the guards (`refuse`, `truncation`, `disorder`, `duplicate`), and the name parser.

Results of the final files (46 lines, each `ok`; `test_bli: all passed` everywhere; no sanitizer report in the default
run), rebuilt and rerun from the sources by the independent verification:

| where | build | result |
|---|---|---|
| Mac, arm64, Apple clang 17, the stand-in `immintrin.h` | `build/tests/cxx` (-O1, ASan + UBSan, asserts on) | all passed, 35 s |
| Mac, arm64, same | `build/tests/cxx-O3` | all passed, 21 s |
| Mac, x86-64 under Rosetta, the real `<immintrin.h>`, `-arch x86_64 -mavx2` | -O1, ASan + UBSan, asserts on | all passed, 46 s |
| Mac, x86-64 under Rosetta, same | -O3 | all passed, 27 s |
| Jed login node, Xeon 8360Y, g++ 11.4, `-std=gnu++17 -O3 -march=native` (`wf_cxx`) | -O3 | all passed, 25 s of CPU (1 min 46 s under `nice` on the shared node) |

The commands (B = `build/tests/sota/bli`, BLI = the clone; the Mac ones are in the head of `tests/sota/test_bli.cpp`):

    # Jed (the benchmark's compiler and flags; wf_cxx = g++ -std=gnu++17 -O3 -DNDEBUG -march=native -faligned-new -include cstdint
    #      -fopenmp -I<this repo>/include -I<GRE>/src/competitor -I<GRE>/src ... -ltbb)
    wf_cxx bli -isystem $BLI/include -isystem $BLI/src tests/sota/test_bli.cpp -o t && nice -n 10 ./t
    # Mac arm64, sanitized and -O3
    build/tests/cxx    -DGRET_BLI_SHIM -I $B/armshim -I $BLI/include -I $BLI/src tests/sota/test_bli.cpp -o $B/t    && $B/t
    build/tests/cxx-O3 -DGRET_BLI_SHIM -I $B/armshim -I $BLI/include -I $BLI/src tests/sota/test_bli.cpp -o $B/t-O3 && $B/t-O3

Also checked: without the authors' headers, and on arm64 without the stand-in, the test's other `main` runs (`make_bli`
throws for `bli`, returns nullptr for `alex`); `bli.h` included before `gret/registry.h` in one translation unit compiles
and both make their indexes (Mac -O3, and Jed with g++ 11).

`rootoverflow` (by name only): on the Mac under ASan the authors' constructor is stopped at `segment.h:98`
(heap-buffer-overflow, 168 bytes behind a 960-byte S-Bucket array); in the -O3 builds (Mac arm64, Rosetta, Jed) the
wrapper throws after the 348th put: "58183 pairs can be reached, 60350 were stored".

What the default run shows about the guards: `truncation` constructs a key the authors' walk misroutes (256-slot
buckets): the checked put throws, the unchecked put misplaces it and the audit reports `misplaced=1`, and loaded with the
data the authors' lookup misses it while `get`, `update`, `put` and `scan` answer. `disorder` loads 128 such keys as the
upper half of a D-Bucket and fills the bucket until it splits: the checked put throws at the split (`risk_splits=1`,
"1 Segment(s) are out of key order"); with the unchecked put the authors' scan is out of order afterwards (the test
checks that it is, so that the set keeps provoking the defect), the audit reports `segments_out_of_order=1` and
`memory_consumption` throws. In all other sets `refused_puts`, `misplaced`, `segments_out_of_order` and
`keys_the_authors_lookup_misses` are zero; `risk_splits` (splits below a Segment at risk, each followed by the check of
all Segments) is above zero in five of them (up to 161 a set), without a finding. `risk_puts` (puts whose walk truncated; each was checked against
the pivots and was routed right) is above zero wherever the test appends keys far above a dense region (`run`, `run0`,
`skew`, `clusters`, `hugegap`: up to 21,000 of them a set). Such a put costs a pass over the entries of the Segments on its
path; on real data it happens for the keys behind a large jump until the Segment there is cut again.

The earlier draft of this wrapper threw at the bulk load when a loaded key was misrouted, did not watch puts for
misrouting, and checked a new root by re-predicting the pivots it could still find, which cannot see a pivot written behind
the array. All three were replaced as described above.

### Independent verification

A second pass rebuilt and reran everything above from the files, read the authors' code against the wrapper, and ran its
own adversarial checks (a scratch program, `build/tests/sota/bli/verify/adv.cpp`, not part of the test suite; every answer
against a `std::map` or a sorted array, operator new instrumented to count the bytes really allocated):

- no key, one key, two keys, 0 and 2^64 - 1 stored and not stored, keys next to 2^64; keys equal modulo 2^32 and keys
  with few distinct high words; a dense run followed by a huge gap and single far keys; three dense islands; random and
  dense keys above 2^63; uniform sets with 1.5M interleaved put / update / remove / get operations per set (remove must
  return false and change nothing); 4M random 64-bit keys with 1.5M interleaved operations on `bli` and `bli16s8`; on
  D-Buckets of 2^8 to 2^18 slots, checked and unchecked. Mac arm64 sanitized and -O3, and Jed with g++ 11: no wrong
  answer, no sanitizer report.
- **Found and fixed:** defect 2 (e), the misplaced pivot. The first version answered every get but returned scans out of
  order after such a split, with a clean audit. The guard (the order check after a split at risk, and in every audit) and
  the test set `disorder` were added.
- `memory_consumption` against the bytes held through operator new: 1.5 KB (libstdc++) to 2.5 KB (libc++) lower, on every
  set and size (the authors' eleven thread objects and their queue, and the name string); 27.48 bytes a key at 3M keys
  and 26.95 at 20M, both ways.
- New roots (defect 3) with the default error bound: six key shapes (exponential, slowly exponential, stepped, quadratic,
  dense with jumps, sparse-then-dense gaps) times three insert patterns (into all gaps, appended above the largest key,
  one region densely) times five settings of 256- and 1024-slot buckets, x86-64 (the regression uses `long double`, which
  is 80 bits there and 64 on arm64) at -O3: 90 runs, up to 1.2M puts each, new roots built in at least 30 of them, no
  overflow and no lost key (and 12 more runs on 2^16 and 2^17 slots). That does not prove the default safe; it says the
  overflow needs more than these shapes. Keys within a narrow range at a huge offset (2^63 + 3i) do not upset the
  regression either, because the pair of key 0 is always the first pivot. `rootoverflow` was rerun: AddressSanitizer
  stops at `segment.h:98` as described, and the -O3 builds throw after the 348th put.
- The paper's section and figure numbers cited here were checked against the arXiv HTML. The words SIMD and AVX do not
  occur in the paper; its D-Bucket lookup (5.1.2) is a probe from the hint, which is what the released build runs. The
  AVX2 code in `bucket.h` is an unused path of the repository, not a feature the paper reports.
- Read for faithfulness: the settings against `script/run_overall.sh` and the fork's CMake (`-O3 -march=native`,
  murmur hint, regression on, SIMD off), the nine commits after the fork's pin (scan only: `bucket.h` gains `scan_kvs`,
  `keyvalue.h` an `operator>`, `buck_index.h` the scan functions and the workers), and the clone against its commit
  (clean, identical on the cluster).

## Known limits

- No deletion (`remove` returns false).
- Not thread-safe; the concurrent variants of the authors' repository are not wrapped.
- Lookups of absent keys and scans read whole D-Buckets (the authors' code): at 131072 slots a miss costs of the order of
  100 microseconds. A workload with misses or short scans measures that.
- D-Buckets below 2^16 slots refuse large keys (defect 1). The limit also falls with the fill ratio (it is
  2^50 x min(D x fill, D / 2) x fill): `bli16` holds every key from 50% up, `bli17` from 40% up, and `bli17f30` refuses keys
  above 2^63.5. For a fill-ratio sweep down to 30% on data with keys above 2^63, use `bli18f30` (2^18 slots hold every
  key from 30% up).
- A checked put that the authors' walk would misplace throws (defect 2); a split that misplaces a pivot throws after the
  fact, with the Segment out of key order (defect 2 e); a new root built past its array throws after the fact, with memory
  corrupted (defect 3). All are reported in the `gret: bli now ...` line and were zero on every default test set except
  the sets built to provoke them.
- `u` puts are not checked for duplicates, misrouting or misplaced pivots while they run; `memory_consumption` checks
  afterwards (a scan made in between can be out of order without a report: time `bli`, not `bliu`, on scan workloads).
- The check after the bulk load and the audit in `memory_consumption` are the wrapper's, and take time the authors' code
  does not: they are outside GRE's timed operations except for the bulk load (see `load_ms`).
- No licence: the authors' code cannot be redistributed with this repository.
