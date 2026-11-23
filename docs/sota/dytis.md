# dytis: DyTIS behind GRE's interface

`include/gret/sota/dytis.h`, `tests/sota/test_dytis.cpp`. Names: `dytis`, `dytiss`, `dytisu`, each optionally followed by
`a` (`dytisa`, `dytissa`, `dytisua`).

## What it is

DyTIS (Jin Yang, Heejin Yoon, Gyeongchan Yun, Sam H. Noh, Young-ri Choi: "DyTIS: A Dynamic Dataset Targeted Index Structure
Simultaneously Efficient for Search, Insert, and Scan", EuroSys 2023, doi 10.1145/3552326.3587434) is an updatable index
built on extendible hashing in which the "hash" keeps the key order. The top 9 bits of a key pick one of 512 tables; in a
table a directory of 2^(global depth) entries leads to a segment, and a segment is an array of 2 KB buckets (128 sorted
key/value pairs each). The bucket inside the segment is given by a piecewise-linear function of the key (the segment's
"local CDF"), which the index adjusts while keys arrive: a full bucket leads to a remapping (the function is re-tuned and
the segment may grow), an expansion (the segment doubles), or a split (extendible hashing's, with directory doubling). It
has no bulk load and no training phase: that is the paper's point.

This is a **wrapper around the authors' own code, used unmodified**. It is not a reimplementation and not a third-party
implementation.

## Where the code comes from

| | |
|---|---|
| repository | https://github.com/unist-ssl/DyTIS (linked from the paper; UNIST, the authors' group) |
| commit | `5cd954de7ec0dbadf698fcd79b7a3393b6eddc86` (2023-05-16, the head of `main`) |
| licence | Apache-2.0 (`LICENSE.md`, and a licence header in every source file) |
| local clone | `/Users/louisvasseur/Downloads/readonly_competitors/DyTIS` (`git status` clean) |
| on the cluster | `/scratch/lvasseur/readonly_competitors/DyTIS` (the same files, without `.git`) |
| files used | `util/pair.h`, `util/util.h`, `src/DyTIS.h`, `src/DyTIS_impl.h`, `src/ExtendibleHash.h`, `src/ExtendibleHash_impl.h`, `src/Directory.h`, `src/Directory_impl.h`: the whole index, 2,165 lines. Not used: `benchmark/` (their driver) |
| included | by path, inside `namespace gret::dytis_authors`; nothing of theirs is copied into this repository |

One dependency is bridged, and nothing else is ours in the index itself:

- **Boost.Pool.** `src/Directory.h:19` includes `<boost/pool/pool_alloc.hpp>` and uses `boost::pool<>` (20 pools for
  segments of 1 to 20 buckets, 10 for models of 2 to 1,024 pieces) and `boost::pool_allocator<Directory>` (segment
  headers). Boost is installed neither here nor on the cluster. The shim has two parts:
  - `/Users/louisvasseur/Downloads/readonly_competitors/DyTIS_shim/boost/pool/pool_alloc.hpp` (on the cluster
    `/scratch/lvasseur/readonly_competitors/DyTIS_shim/...`): a file of five comment lines and no code, so that the
    include line resolves. It is ours, it is not in the authors' clone, and it holds nothing of Boost's.
  - the two classes, **restated by us** in `dytis.h` (`gret::dytis_authors::boost::pool`, `pool_allocator`, about 130
    lines): a free list per chunk size, memory taken from the system in blocks of 32, 64, 128, ... chunks (Boost's
    `next_size` doubling, start 32, no upper limit), `free` pushes the chunk back on the list, nothing goes back to the
    system before the pool is purged; `pool_allocator::destroy` runs the destructor and returns no memory (the authors
    never call `deallocate`). Differences from Boost, all deliberate: 16 bytes of `0xff` in front of every chunk (below,
    defect 4), every block counted (for `memory_consumption` and for the optional limit `$DYTIS_MAX_GB`), and under
    AddressSanitizer every chunk is its own `malloc` block so that the sanitizer sees chunk bounds. If Boost is installed
    after all, its header is switched off through its include guard and the restated classes are still the ones used.
  The same goes for the `malloc` and `free` the authors call for segments above 20 buckets (`Directory.h:109, 204`): inside
  the namespace these resolve to two functions of ours that count the block and put the same pad in front.

RoBin (github.com/cds-ruc/RoBin, `src/competitor/dytis/dytis.h`, read on 2026-11-22, not used) has a GRE wrapper of 94
lines around the same code: `bulk_load` is one `Insert` per key in ascending order, `put` is `Insert` and returns true,
`get` is `Find`, `remove` is `Delete`, `scan` calls `Scan`, drops the result and returns the count asked for,
`memory_consumption` returns 0, and a SIGTERM handler prints "dytis timeout". None of the hazards below is handled
there. Our wrapper was written from the authors' sources, not from RoBin's.

## Parameters: the code is the paper's

Section 4.1 of the paper: 2^9 first-level tables (R = 9), bucket 2 KB, utilization threshold U_t = 0.6, remapping and
expansion from local depth L_start = 6, and, for data sets on which DyTIS itself observes many expansions, a limit on
the segment size that goes from 2 times to 128 times at local depth 8. The released code has
exactly these (`util/util.h`: `kDepth = 9`, `kBucSize = 128 * 16`, `BUC_THRE 0.6`, `RECLAIM_THRE 0.6`, `REMAP_THRE 6`,
`SKEWED_MAX_BITS 1`, `UNIFORM_MAX_BITS 7`; `Directory.h`: models of 2 to 2^17 pieces). The switch between the two limits
is **automatic in the released code**, not set by hand per data set: the first time a table reaches global depth 8,
`DyTIS::Insert` counts the segments that expanded without a model, and if they are more than a tenth of all segments it
sets the limit to 128 times for the rest of the run (`DyTIS_impl.h:69-93`). The build is the authors' default target
(`make DTS`: `-DSEP`, keys and values of a segment in two arrays, `-DNDEBUG`).

Names:

| name | mode | bulk_load inserts in |
|---|---|---|
| `dytis` | **the released code and the paper's default**: the code's own one-time choice | random order |
| `dytiss` | forced "skewed": limit 2 times from the start, the one-time test switched off | random order |
| `dytisu` | forced "uniform": limit 128 times from the start, the one-time test switched off | random order |
| `dytisa`, `dytissa`, `dytisua` | as above | ascending order |

The mode is the knob to sweep: **`dytis`, `dytiss`, `dytisu`**, best per data set, saying which. `dytisu` is not what the
released code does on data it would itself call uniform (there the larger limit starts at depth 8, here at depth 6); it
is "the larger limit throughout". The line `bulk_load` prints says which mode `dytis` ended in.

**The insertion order is a choice the paper does not make for us**, and it changes the index. DyTIS is built by inserts,
its structure depends on their order, and GRE hands `bulk_load` a sorted array. The paper inserts each data set in the
order of its file (time order, section 4.2) and also in shuffled order (its Group 2). The default here is a random
permutation of the keys (fixed seed), which is the order of GRE's own insert workload and of the paper's shuffled sets;
`a` gives ascending order, which is what RoBin's wrapper does. Ascending is cheap on uniform keys and ruinous on skewed
or dense ones, in the authors' code itself (measurements below): it is not usable at 200M on such data. Neither order is
"the paper's" for a SOSD-style file, which has no arrival order.

## How it maps to GRE's interface

| GRE | what runs |
|---|---|
| `bulk_load` | one insert per key (below), in the order of the name; then one walk over the whole structure comparing every key and payload with the array given, and one lookup of every key through the authors' `Find`, which must lead to the slot the walk is on (deviation 11); then the line on stderr |
| `get` | the authors' `DyTIS::Find`, whole (SEP layout). An edge key (below) is looked up in the side store instead |
| `put` | the wrapper finds the bucket the way the authors' `ExtendibleHash::Insert` does, looks the key up in it (false and no change if it is stored), then: room in the bucket: the authors' `Directory::Insert` on that bucket; bucket full, or no table yet: the authors' `DyTIS::Insert`, whole, with the checks of deviation 4 around it |
| `update` | the authors' `Update`, which is `Find` and an assignment |
| `remove` | the wrapper finds the bucket and the key (false if absent), then the authors' `Directory::Delete` on that bucket, or the wrapper's own shift in the two cases their Delete gets wrong (defects 2, 3) |
| `scan` | the wrapper's walk over the authors' segments: buckets in order, then the `sibling` pointer, then the next table; merged with the side store. Exact pairs, ascending |
| `memory_consumption` | every block of the pools (chunks in use and free ones, pads included), the large segments, the directories, the 512-entry table, models above 1,024 pieces, the side store's nodes, the object |

The payload is stored in the authors' value slot (a `uint64_t`), so a lookup has no indirection DyTIS's own would not
have. On `get` the wrapper adds one test of the key's low 24 bits in front of `Find`.

## Build

Header-only as the authors wrote it, but **their headers define non-inline globals** (the pools, `y_mask`,
`max_bucket_num`, the mode switch) and file-level statics, and the typedefs `Key_t`, `Value_t`, `Pair`, `Directory` at
global scope. `dytis.h` includes them inside `namespace gret::dytis_authors` (every standard header they include is
included before, outside the namespace) and removes the macros they leave (`SEP`, `REMAP_THRE`, `ADDR_BITS`,
`DO_NOTHING`, ...). Consequences:

- **one translation unit** of a program may include `dytis.h` (GRE's benchmark is one);
- **one `DytisIndex` alive at a time** in a process: the constructor throws `std::logic_error` for a second one. An index
  that was destroyed can be followed by another. **Under GRE that means one `dytis*` name per process**: GRE deletes its
  index through `indexInterface*`, which has no virtual destructor, so the wrapper is never told that the first index is
  gone, and a second `dytis*` name in the same `--index` list throws. The repository's drivers
  (`scripts/run_baselines.sh`, `scripts/gated_run.py`) run one index per process, so a sweep over `dytis`, `dytiss`,
  `dytisu` is three processes;
- single-threaded. The released code has no locks at all (the paper's section 3.4 describes a concurrent version; it is
  not in the repository).

Include paths: `-I <clones>/DyTIS_shim -I <clones>/DyTIS` (on the cluster `-isystem` with
`/scratch/lvasseur/readonly_competitors/DyTIS_shim` and `/scratch/lvasseur/readonly_competitors/DyTIS`). No define is
needed for the header itself. For the registry, guarded like DILI because of the one-translation-unit rule:

    #ifdef GRET_SOTA_DYTIS  // needs DyTIS_shim and DyTIS on the include path; one translation unit, one index alive at a time
    #include "gret/sota/dytis.h"
    #endif
    ...
    #ifdef GRET_SOTA_DYTIS
        if (indexInterface<K, P>* i = make_dytis<K, P>(name)) return i;  // dytis, dytiss, dytisu, each [a]: the DyTIS authors' code, guarded (sota/dytis.h)
    #endif

Checked with exactly these two blocks added to a scratch copy of `registry.h` (placed after the `GRET_SOTA_DILI` blocks,
inside `#ifdef GRET_SOTA`; the copy sits first on the include path, the repository's file is untouched): a translation
unit with GRE's ALEX, B+tree, LIPP and PGM headers and `gret/registry.h`, built with `-DGRET_SOTA -DGRET_SOTA_DYTIS` and
the other baselines' include paths, makes `dytis`, `dytisu` and `dytissa` through `gret::make_index`, answers `get` and
`scan`, and still makes `rs32r18`; with Apple clang here and with g++ 11.4.1 on the cluster. Not checked together with
`-DGRET_SOTA_DILI`.

**Layout and compiler.** The authors' default layout (`-DSEP`) does arithmetic on `void*` (`Directory.h:104` and nine
more places), which g++ accepts and clang rejects in C++. Under g++ the header builds SEP, the authors' default and
what the benchmark runs; under clang (or with `-DGRET_DYTIS_NOSEP`) it builds their other target, `DTS_noSEP` (an array of
pairs). The `bulk_load` line says `layout=SEP` or `layout=noSEP`. Their asserts are compiled out inside the header, as
in their Makefile (`-DNDEBUG`).

Environment variables (none is needed for the header to work): `DYTIS_MAX_GB` limits what the pools and the large SEP
segments may hold together (default: no limit; the test sets 10). **Set it in every benchmark run**, in the job's
environment or through the registry's name syntax (`dytis@DYTIS_MAX_GB=100`): on some key shapes the authors' code takes
kilobytes a key (Measurements, "Shapes on which the authors' code does not scale"), and without the limit that ends in
the kernel's out-of-memory killer, which in a batch of runs on one node takes the other runs with it; with the limit it
ends in an exception that names the limit; `DYTIS_MAX_DEPTH` (1 to 31, default 31) lowers the deepest local depth
the authors' insert may reach, for testing the depth guard.

## Deviations: everything the wrapper does that the authors' code does not

1. **A side store** (`std::map`) for keys the authors' code cannot hold, consulted only for such keys:
   - **edge keys**: the last 64 keys of every aligned block of 2^24 keys (low 24 bits >= 2^24 - 64). They never enter
     DyTIS. Reason: a segment covers an aligned range of 2^(55 - depth) keys, and at the top of that range the authors'
     model (`gradient * key + intercept` in doubles, `Directory_impl.h:1156-1162`), fitted to end exactly on a bucket
     boundary, rounds up and sends the last one or two keys to bucket number `seg_num`, one past the segment. Their
     insert then writes, and their search reads, outside the segment's array. Seen on the skewed test sets whenever the
     mode is uniform (run, clusters, lognormal, edges; and 1,285 of the models checked while 43M clustered keys were
     loaded), never by more than 2 keys. The margin of 64 is checked, not assumed: after every restructuring the
     wrapper finds, per model, where the overshoot starts, and throws if it reaches a key that is not an edge key.
     `2^64 - 1` (the authors' empty slot, `INVALID`) is an edge key; so is `2^64 - 2`, which `util/pair.h` names
     `SENTINEL` and no code uses. Edge keys are 3.8 in a million of keys whose low 24 bits are evenly spread: about 760
     of 200M;
   - a key whose insert would take a segment past **local depth 31**. The depth is kept in 5 bits of a tagged pointer
     (`LOCAL_DEPTH_BITS 5`); a split at depth 31 would store depth 32 as 0 (`ExtendibleHash_impl.h:94-129`) and nothing
     in the authors' code stops there. The wrapper lets the allocator of segment headers give out two headers per split
     still allowed; the first header beyond that throws out of the authors' `Insert` before the split changed anything,
     the wrapper writes the global depth the unwinding skipped (`DyTIS_impl.h:63-67`), and the key goes to the side
     store. Each such refusal leaks 16 bytes (two pointers the authors' `Split` allocates first). Not reached on any
     test set at the default depth; exercised with `DYTIS_MAX_DEPTH=12`.
   The line `bulk_load` prints gives both counts (`side=`, `edge_keys=`, `depth_limit=`), and `memory_consumption` prints
   the side store's size again when it changed. **A run where `side` is not a negligible share of n is not a measurement
   of DyTIS.**
2. **`get` goes through `Find`, not `Get`**: `Get` returns 0 for "not found" (`NONE`), so a payload of 0 could not be told
   from a miss. `Find` is the same search and returns the slot.
3. **`put` does not overwrite** a stored key (the authors' `Insert` does): it returns false and changes nothing, as GRE's
   ALEX and B+tree wrappers do. This costs one binary search of the bucket before the insert.
4. **Checks around a restructuring insert** (bucket full: the authors' code remaps, expands or splits, as often as it
   takes). Before: the segment's pairs are counted and summed (a 64-bit hash of each pair). After: the segments that
   now cover the same key range must hold the same count and sum plus the new pair, the keys ascending, every bucket's
   first and last key in the bucket the authors' model sends it to and in a segment the directory leads to, the
   directory entries aligned, the sibling chain intact, and every model monotone and inside its segment (but for edge
   keys). If not: `std::runtime_error`, never a silent loss. One pass over the segment before and one after, where the
   authors' restructuring makes three or more. No test set triggered the exception.
5. **`remove`** looks the key up first and returns false for an absent key (the authors' `Delete` returns true whatever
   happened). It calls their `Directory::Delete` except in the two cases that function gets wrong (defects 2 and 3).
6. **`scan`** is the wrapper's. The authors' `Scan` (`DyTIS_impl.h:116-144`, `Directory_impl.h:733-802`) returns values
   without keys in an array it allocates, does not say how many it wrote, takes a payload of 2^64 - 1 for an empty slot,
   and, when the start key is not stored, starts at the stored key below it.
7. **`memory_consumption`** is ours; the authors' `data_size()` functions are not called by anything and leave out the
   pools.
8. **The insert into a bucket with room skips the authors' second search.** The wrapper has already found the bucket
   (same directory walk, same `lcdf` call as `ExtendibleHash::Insert`, lines 22-37), so it calls the authors'
   `Directory::Insert` on it directly instead of `DyTIS::Insert`, which would find the same bucket again. The one-time
   uniform test of `DyTIS::Insert` still runs at the same moment as in the authors' code: it can only fire in a call
   that changed a global depth, and those calls go through `DyTIS::Insert`.
9. **Out of memory is final.** `std::bad_alloc` inside the authors' code (no memory, or `DYTIS_MAX_GB` reached) cannot
   be recovered from: `Expand` and `LocalRemap` change the segment before they allocate, and `LocalRemap` keeps scratch
   in two function statics (`Directory_impl.h:31, 50`) that only its return paths clear. The index throws
   `std::runtime_error`, and every `DytisIndex` constructed later in that process throws `std::logic_error`. (Found the
   hard way: a second index after such a failure crashed.)
10. The destructor frees every segment and table (the authors' `~DyTIS` frees one array). Deleted through
    `indexInterface*`, as GRE does (no virtual destructor), nothing is freed.
11. **`bulk_load` ends with one `Find` per key.** The walk that checks the loaded structure reads the segments directly;
    `get` runs the authors' `Find`, which evaluates the segment's model in its own inlined copy of `lcdf`
    (`gradient * key + intercept` in doubles). A compiler that fused the multiply and the add in one copy and not in
    another would send a key that lies on a bucket boundary to two different buckets, and `get` would miss a key the
    walk sees. So after the load every key is looked up through `Find`, and the pointer it returns must be the slot the
    walk is on; otherwise `std::runtime_error`. In key order this costs 7 ns a key on the Xeon (2M keys: the load line's
    `total_ms - build_ms` went from 9 to 24 ms), 1.5 s at 200M. Added by the verifier. For the record, in the g++ 11.4.1
    `-O3 -march=native` build on the cluster all 32 inlined copies of `Directory_impl.h:1161` are one `vfmadd132sd`
    (`objdump -dl`), so the copies agree there; the check makes that a tested fact in whatever binary the benchmark is.
    Keys `put` after the load are covered by the checks of deviation 4, which use the wrapper's copy of `lcdf`.

Not changed: every constant, the layout, the search inside a bucket, the models, the restructuring logic. That the wrapper
does not change what the authors' code builds was checked directly: the same keys inserted in the same order through
`bulk_load` and through the authors' `DyTIS::Insert` alone give the same number of bytes in the pools (held and in use),
the same bytes in large segments and the same number of large segments, after the load and again after a tenth more keys
were `put` (uniform, lognormal, dense and fb-like keys, 2.7M each, SEP layout).

## Defects in the authors' code (file:line, and whether real data reaches them)

| # | where | what | reached on real data | handled by |
|---|---|---|---|---|
| 1 | `DyTIS_impl.h:53`, `ExtendibleHash_impl.h:103-129` | the local depth is put in the top bits of a `Directory*` by pointer arithmetic that overflows (UBSan `pointer-overflow`, first report `ExtendibleHash_impl.h:128:35`). Works as an integer add on x86-64 and arm64, and only because `sizeof(Directory) == 64` (asserted) | yes, on every new table and every split | not handled: this one UBSan check is switched off for the authors' files in the sanitized test |
| 2 | `ExtendibleHash_impl.h:164-171` | `Delete` runs off the end of a `bool` function when key 0 heads the bucket (`Directory_impl.h:417`) | yes, if key 0 is stored | the wrapper does not call `ExtendibleHash::Delete`; it closes the gap itself when key 0 heads the bucket |
| 3 | `Directory_impl.h:412-415` (`429-432` noSEP) | `Delete` on a full bucket copies slot 128 into slot 127: the next bucket's first pair, or memory past the segment for its last bucket | yes: any removal from a full bucket | the wrapper removes from a full bucket itself |
| 4 | `Directory_impl.h:718-720`, `807-809` | `Get`/`Find` read the slot before the bucket's first when the key is below every key of the bucket; for bucket 0 that is 8 (SEP) or 16 bytes before the segment | yes: every miss below a segment's first key | pad of `0xff` in front of every pool chunk and every large SEP segment; a bound test in the noSEP build |
| 5 | `Directory_impl.h:1109-1153` | `tuning_local_cdf_by_range` has a path without a return (`seg_num >= PRACTICAL_MAX_SEG_NUM` at line 1146) | not reached on any test set (UBSan's return check is on in the sanitized builds) | not handled; the checks of deviation 4 would see the damage, not prevent it |
| 6 | `Directory_impl.h:1156-1162` | the model sends the top keys of a segment's range past its last bucket | yes | edge keys (deviation 1) |
| 7 | `ExtendibleHash_impl.h:94-100` | local depth 32 does not fit its 5 bits | only with extreme clustering | depth guard (deviation 1) |
| 8 | `Directory_impl.h:1055` | an `assert` casts a negative double to `uint64_t` | no: asserts are compiled out, as in the authors' build | - |
| 9 | `Directory_impl.h:31, 50` | function-static scratch vectors: not thread-safe, undefined after an exception | only on out of memory | deviation 9 |
| 10 | `ExtendibleHash_impl.h:140`, `Directory.h:198-218` | the header of a split segment is destroyed, never deallocated (64 bytes per split stay in the header pool); Boost pools never give memory back, so the chunks of every segment size a segment has outgrown stay allocated | yes | counted in `memory_consumption` (below) |

## Tests

`tests/sota/test_dytis.cpp`: against a sorted array and a `std::map`, every answer checked. `get` on every stored key and
on keys never stored (both neighbours of stored keys, keys in the gaps, 0, 1, 2^63 and its neighbours, 2^64 - 3 to
2^64 - 1, the ends of all 512 tables, the top of the 2^24 block of stored keys, random keys); `scan` from stored,
neighbouring, in-between and extreme start keys with lengths 1 to 70,000 and one scan over everything; then `put`
(neighbours, runs, keys far outside, duplicates, payloads 0 and 2^64 - 1, edge keys and the keys just below them),
`update`, `remove` (from full buckets and with key 0 stored), puts after removes, every key removed, the index filled
again; the same checks after each step. Key sets: empty, 1 to 129 keys, uniform, the whole key space with 0 and 2^64 - 1,
a dense run, a run from 0, clusters, far outliers, a stride, two runs at the ends of the key space, large keys with
pairs and triples, fb-like, lognormal, 96,000 edge keys, at 100,000 to 350,000 keys in the modes and orders that are
affordable; 2M random keys and 2M dense ones; a set under `DYTIS_MAX_DEPTH=12` where the depth guard must send keys to
the side store; the name parser; the one-index-at-a-time rule; a refused `bulk_load` (duplicate keys); out of memory
under a 1 MB limit. **`memory_consumption` is compared with a count of every byte the program asks `operator new` and
the header asks `malloc` for** between the index's construction and the end of the bulk load: 99.6% to 100.0% on every
set (the missing part: two scratch vectors of the authors' `LocalRemap`, and 16 bytes for every split the depth guard
stops: the two pointers the authors' `Split` had allocated before the guard threw).

| build | layout | result |
|---|---|---|
| Mac, Apple clang, `build/tests/cxx` (-O1, ASan + UBSan), **as given: fails** in the authors' code at `ExtendibleHash_impl.h:128:35` (defect 1) on the first insert | noSEP | - |
| the same with `-fsanitize-ignorelist` taking `pointer-overflow` off the authors' files only (ASan and every other UBSan check stay on them; the wrapper stays fully checked) | noSEP | all passed, 4 minutes |
| Mac, Apple clang, `build/tests/cxx-O3` | noSEP | all passed, about 1 minute |
| Mac, g++ 15.2 `-O3 -Wall -Wextra` (no warning from the header) | SEP | all passed |
| Mac, g++ 15.2 `-O1 -fsanitize=undefined -fno-sanitize=pointer-overflow,vptr -fsanitize-undefined-trap-on-error` | SEP | all passed (no trap) |
| **cluster login node, g++ 11.4.1, `wf_cxx` (-O3 -march=native)** | SEP | all passed: 40 lines, run as three groups of sets (`t tiny uniform uniforms uniformu wide`, `t run run0 clusters outliers stride hugegap`, `t highmag fblike lognormal edges depthlimit big oom`). About 1.5 minutes of CPU each; on a login node with a load of 20 they took 1.5, 5 to 9.5 and 2.5 minutes of wall time |

**Independent verification (2026-11-22, a second agent; scratch files in `build/tests/sota/dytis/verify/`, not part of
`tests/`).** The table above was rerun from the files, before and after the verifier's edit (deviation 11), with the
same results: Mac sanitized as given fails at `ExtendibleHash_impl.h:128:35`, with the ignore list all 40 lines pass;
Mac `-O3` 40 lines; Mac g++ 15.2 `-O3` (SEP) 40 lines; cluster login node, g++ 11.4.1 through `wf_cxx` (SEP), 40 lines
in five groups of at most 2 minutes of wall time each. On top of that a separate differential test
(`adv_dytis.cpp`: every `get`, `put`, `update`, `remove` and `scan` of 200,000 to 500,000 interleaved random operations
against `std::map`, and a full check of every stored key, its neighbours and scans before and after) on: no key, one
key (0, 2^64 - 1, 2^64 - 2, an ordinary one), two and three keys, keys equal modulo 2^32 (300,000 in sequence, 250,000
at random), 200,000 consecutive keys from 0 then 200,000 at 2^62 then 100,000 below 2^64, 400,000 keys above 2^63,
a stride of 3 down from 2^64 - 1, 150,000 keys that are all edge keys, 384,000 of which half are, 0 to 299 (key 0 at
the head of a full bucket), even keys from 0, an index filled only by `put` (ascending, descending) and then emptied
to a tenth and refilled four times, a second `bulk_load` on the same object, fb-like, clustered, lognormal and dense
keys at 600,000, and 3M to 5M random and lognormal keys in the three modes. No wrong answer on the Mac (clang `-O3` and
sanitized, g++ 15 SEP) or on the cluster (g++ 11.4.1 SEP; there without the put-only set). The one exception thrown was
the memory limit, on the clustered keys under `dytisu` (12 GB on the Mac; on the cluster that run was killed by the
login node's own cap of 8 GB a user, cgroup `memory.max`, so nothing larger can be tried there). On the cluster
`memory_consumption` was held against glibc's own count of the heap (`mallinfo2`: bytes in use plus mapped) before the
index was made and after the load: 0.98 to 1.00 on every set of 200,000 keys or more but the two with edge keys in bulk
(0.75 and 0.86: a `std::map` node is counted as its 48 bytes and costs 64 in glibc; irrelevant at 4 edge keys in a
million). The registry lines were compiled with
`-DGRET_SOTA -DGRET_SOTA_DILI -DGRET_SOTA_DYTIS` together (g++ 11.4.1): `dytis`, `dytisu`, `dytissa` and `dili` are made.
The clone is the head of the authors' repository (`git ls-remote`: `main` = `5cd954d`; the tag `v1.0` differs from it in
`README.md` only) and its eight source files have the same md5 on the cluster.

Not available: AddressSanitizer on the SEP layout (clang cannot compile SEP; Homebrew's g++ 15 has no libasan on this
Mac; on the cluster `libasan.so` points to a file that is not installed). The SEP layout, which the benchmark runs, is
therefore covered by the exact-answer tests, the allocation count and UBSan in trap mode, not by ASan.

Build lines are at the top of the test file. `t --time` is the timing mode used below.

## Measurements (not a benchmark: this laptop, one thread, single runs)

Apple M3 Max, g++ 15.2 `-O3`, SEP layout. A tenth of the keys is held back and `put` afterwards in random order.

**Build cost per key and memory, `dytis` (random insertion order).**

| keys | loaded | build, ns a key | get, ns | bytes a key held | of which in use | most held during the load | mode `dytis` ended in |
|---|---|---|---|---|---|---|---|
| uniform, 64 bits | 2.7M | 363 | 91 | 26.7 | 26.7 | 26.6 | not decided yet (global depth 6) |
| | 10.8M | 454 | 112 | 63.2 | 25.8 | 63.2 | not decided yet (depth 7) |
| | 43.2M | 553 | 120 | 65.6 | 25.1 | 65.6 | uniform |
| lognormal(0, 2) x 10^12 | 2.7M | 319 | 81 | 43.3 | 38.4 | 43.7 | skewed |
| | 10.8M | 398 | 93 | 35.6 | 34.4 | 37.0 | skewed |
| | 43.2M | 444 | 99 | 33.5 | 33.0 | 33.4 | skewed |
| dense (ids: runs of 1 to 200,000 consecutive keys, gaps between) | 2.7M | 400 | 88 | 57.2 | 56.7 | 75.0 | skewed |
| | 10.8M | 469 | 104 | 31.8 | 31.7 | 51.3 | skewed |
| | 43.2M | 993 | 89 | 56.2 | 56.1 | 74.0 | skewed |
| runs of 100 keys at random 64-bit places | 2.7M | 452 | 128 | 56.1 | 39.7 | 55.3 | skewed |
| | 10.8M | 566 | 153 | 96.0 | 72.3 | 96.0 | uniform |
| | 43.2M | 752 | 172 | 122.5 | 89.5 | 122.2 | uniform |

"Held" is what `memory_consumption` returns. "Most held during the load" counts the pools and the large segments (not the
directories), from the `bulk_load` line (`peak_bytes_per_key`). The build time includes the wrapper's checks (up to 10%
over the authors' `Insert` called in a loop, measured at 2.7M and 5.4M keys).

**Is 200M feasible?** It depends on the shape of the keys, and for some shapes the answer is no (next but one
paragraph). In random order, on the four shapes of the table above, yes by extrapolation: the cost per key grows by 10 to 35%
per fourfold n on three of the shapes and doubles from 10.8M to 43.2M on the dense one (most of its keys sit in one
segment, 2 GB at that size, and each restructuring passes over all of it). At that rate 200M keys take about 2 to 10
minutes of one core on this laptop, and more on the Xeon (the login node was too loaded to say how much). The index then
holds 30 to 125 bytes a key, 6 to 25 GB, and on dense keys up to 1.6 times its final size for a moment, while the
largest segment is restructured.
`bulk_load` also needs 4 bytes a key for the permutation and makes one verification pass over the structure. **Not
measured at 200M, and not on the ten data sets** (the rules of this task kept the 200M files out of reach): cost and
memory depend on the data, so the first run on each data set should be watched. The `bulk_load` line prints
`insert_ns_per_key`, the mode, the depths, the bytes and the peak. In **ascending order (`a`) 200M is not feasible** on
skewed or dense data: see the next table, and there the cost per key grows with n.

**Shapes on which the authors' code does not scale** (verifier's measurements; `dytis`, random order, SEP layout; Mac
g++ 15.2 unless the cell says cluster, which is the loaded login node; the cost is the authors' code's: `raw` is their
`Insert` alone on the same keys).

| keys | n | build, ns a key | bytes a key held | |
|---|---|---|---|---|
| clusters: 5,000 centres anywhere in 62 bits, each cluster uniform over a range of 2^20 to 2^39 | 300,000 (60 a cluster) | 199 | 30.4 | every cluster fits one bucket |
| | 600,000 (120 a cluster) | 1,206 (Mac), 2,155 (cluster) | **7,882** | 1.94M buckets, fill 0.0024 |
| | 1M (200 a cluster) | 2,519 | **15,272** (15.5 GB resident) | raw: 2,322 ns, 14,136 bytes a key in pools and segments |
| | 3M | - | - | stopped by a 12 GB limit |
| the same, `dytisu` | 600,000 | - | - | stopped by a 12 GB limit |
| fb-like: gaps below 4,096, one gap in 64 up to 2^30 | 600,000 | 4,034 (Mac), 5,040 (cluster) | 220.7 | |
| | 3M | 4,530 | 208.4 | raw at 2.7M: 3,267 ns |
| 200,000 consecutive keys at 0, 200,000 at 2^62, 100,000 below 2^64 | 500,000 | 1,703 (cluster) | 765 | |
| even keys from 0 | 200,000 | 1,770 (cluster) | 456 (`dytisu`: 2,712) | |
| the keys 0 to 299 | 300 | - | 2.7 GB in all | |

The pattern: a group of more than one bucket of keys (128) inside a key range far narrower than the distance to the next
group makes the authors' code grow the group's segment until one bucket is narrow enough to split the group, megabytes
for a few hundred keys. The author's own "run", "hugegap" and "run100k" test sets show the same (308, 711 and 10,801
bytes a key). **Whether one of the ten 200M data sets has such places is not known** (the 200M files were out of
reach for this task, and no smaller sample of them was at hand): data with a hierarchy of clusters (osm) or with far
outliers (fb) are the candidates. Hence the rule under Build: every benchmark run with `DYTIS_MAX_GB`, and a data set
on which the limit is hit is reported as "DyTIS: more than X GB", which is a result about DyTIS, not a failure of the
harness. At 4.5 microseconds a key (fb-like) the load of 200M keys is 15 minutes, if the cost per key stays flat.

**Insertion order** (2.7M keys loaded; build ns a key, bytes a key; `raw` is the authors' `Insert` called directly, no
wrapper):

| keys | `dytis` | `dytisa` | `dytisu` | `dytisua` | raw, random | raw, ascending |
|---|---|---|---|---|---|---|
| uniform, 64 bits | 333 ns, 26.7 B | 37 ns, 26.7 B | 332 ns, 26.7 B | 38 ns, 26.7 B | 326 ns | 36 ns |
| lognormal(0, 2) x 10^12 | 298 ns, 43.3 B | 5,504 ns, 56.8 B | 2,295 ns, 56.2 B | 15,438 ns, 70.3 B | 308 ns | 4,496 ns |
| runs of 100 at random places | 409 ns, 56.1 B | 43 ns, 47.1 B | 416 ns, 83.8 B | 47 ns, 82.5 B | 474 ns | 41 ns |
| dense (ids: runs with gaps) | 372 ns, 57.2 B | 19,713 ns, 498.4 B | 429 ns, 100.9 B | stopped at the 10 GB limit | - | 17,988 ns |

Ascending order is 9 times faster on uniform keys and 18 times (lognormal) to 53 times (dense) slower, with 8.7 times
the memory on dense keys; the slowness is the authors' code's (compare the two raw columns), the wrapper adds a tenth
to a fifth to it. `dytiss` was within 3% of `dytis` on all four (the released code chose "skewed" on each at this size).

**What the wrapper costs** (6M keys, three runs each; `raw` = the authors' `Insert` and `Find` called directly, edge keys
left out):

| keys | get, wrapper / raw (ns) | put, wrapper / raw (ns) | put, wrapper over raw |
|---|---|---|---|
| uniform, 64 bits | 102 / 104, 114 / 115, 111 / 110 | 381 / 340, 465 / 383, 418 / 374 | +12%, +21%, +12% |
| lognormal | 93 / 93, 98 / 98, 91 / 88 | 299 / 296, 328 / 321, 307 / 269 | +1%, +2%, +14% |
| dense | 81 / 84, 99 / 95, 82 / 83 | 416 / 375, 487 / 416, 420 / 375 | +11%, +17%, +12% |
| runs of 100 | 134 / 136, 153 / 153, 135 / 139 | 432 / 418, 516 / 501, 479 / 441 | +3%, +3%, +9% |

`get` is the authors' `Find` plus one test: no difference outside the run-to-run spread (identical runs differ by up to
10% here). **`put` is the authors' insert plus the guard: about a tenth slower than their unguarded insert**, between 1
and 21% in single runs. It carries the search for the key (deviation 3) and the checks of deviation 4, about half each
(with the checks compiled out, uniform keys showed +5%). A paper table of insert throughput should say so. A draft of
this wrapper that copied the segment before each restructuring and called `DyTIS::Insert` for every key cost 30% on
dense keys; the count-and-sum check and deviation 8 brought that down. On the cluster's login node (g++ 11.4.1, SEP,
a load of 20 to 35, so absolute times mean nothing) the same comparison at 2.7M uniform keys gave get 602 against 648
ns and put 2,647 against 2,695 ns: no difference that could be measured there. The benchmark machine was not available
for a quiet timing.

**Memory is what the pools hold, not what the keys need.** `memory_consumption` returns every byte allocated, as the rule
of this repository asks and as the paper measured (maximum process memory, section 4.3, where DyTIS used a third more
than ALEX). A large part of it can be free chunks: a segment that grows from 1 to 2 to 4 to 8 buckets leaves a chunk in
each smaller pool, and Boost pools keep them. On 43M uniform 64-bit keys that is 65.6 bytes a key held against 25.1 in
use; on the lognormal and dense sets, where most segments are above 20 buckets and come from `malloc`, held and in use
are within a few percent. The `bulk_load` line prints both (`pool blocks ..., of which in use ...`). A paper table
should quote the held figure and may add the other. The restated pool follows Boost's policy, so the authors' own build
holds the same chunks; it was not possible to confirm that against real Boost here (not installed).

## Known limits

- **Not run on the 200M data sets.** Feasibility above is an extrapolation from 3M to 48M synthetic keys, and it does
  not hold for every shape: clustered keys cost the authors' code up to 15 KB a key at 1M keys (Measurements). Without
  `DYTIS_MAX_GB` nothing stops the growth.
- **Scans on a sparse structure.** A scan walks bucket by bucket, as the authors' `Scan` does, and an empty bucket costs
  one read 2 KB further on. From a stored key that is cheap (2.6 microseconds for 100 pairs at fill 0.0024, 0.2 to 1.6
  elsewhere). A scan that starts in an emptied stretch of a large segment reads every empty bucket up to the next key:
  after removals had left a 5 GB segment almost empty, the verifier's test spent of the order of milliseconds per scan.
  The released code never shrinks a segment.
- `DYTIS_MAX_GB` does not see the large segments of the noSEP layout (clang builds: the test and nothing else), which
  come from `new[]`; there the limit covers the pools only.
- The paper has a journal extension (Yang, Yoon, Yun, Noh, Choi: "A Dynamic Characteristic Aware Index Structure
  Optimized for Real-world Datasets", ACM Transactions on Storage, online December 2024, doi 10.1145/3707642), which also studies
  sequential data sets and several threads. The repository was not updated for it (one branch, last commit 2023-05-16);
  what runs here is the EuroSys 2023 code. The cost of ascending inserts measured above is that code's.
- The share of keys in the side store on real data is not known beyond the rule: 3.8 edge keys in a million, plus keys
  stopped at depth 31, which takes a remapping that fails inside a range of 2^24 keys and was never seen. Read `side=`
  in the `bulk_load` line.
- **`bulk_load` can throw** `std::runtime_error` on data where the authors' model overshoots by more than 64 keys or
  stops being monotone, or where a restructuring loses a pair. No test set did; if a 200M set does, DyTIS cannot be run
  on it with this wrapper, and the message says which check failed.
- A directory has 2^(global depth) entries of 8 bytes: one tight cluster that drives a table to depth 31 costs 16 GB for
  that table's directory. This is extendible hashing as the authors implemented it, and it is counted.
- Insertion order: results are for random order unless the name ends in `a`. A reader comparing with RoBin's numbers
  should know RoBin loads in ascending order.
- The structure, and so size and speed, differ a little between targets: the model is evaluated in doubles, with a fused
  multiply-add where the compiler emits one (g++ `-march=native` on the Xeon, arm64), without elsewhere. Exactness is
  not affected: the wrapper's checks use the same inlined function as the authors' code.
- One translation unit, one index alive at a time, single-threaded, `Param` ignored. After any exception from
  `bulk_load` or `put` the index must not be used again.
- Segments never shrink: the released code has no merge (the paper's description of deletion mentions one), so memory
  after removals stays what it was.
- Defect 5 (the missing return) is undefined behaviour that no test reached; an optimizing compiler may do anything
  there. Defect 1 (pointer tagging) is relied upon on every insert.
- AddressSanitizer never saw the SEP layout (see Tests).
