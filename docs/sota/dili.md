# dili: DILI behind GRE's interface

`include/gret/sota/dili.h`, `tests/sota/test_dili.cpp`. Names: `dili`, `dili<B>`, each optionally followed by `r<PP>`
(`dilir20`, `dili32r20`).

## What it is

DILI (Li, Lu, Zhu, Ding, Yang, Pan: "DILI: A Distribution-Driven Learned Index", PVLDB 16(9), 2023) is an updatable learned
index: a bottom-up tree (BU-tree) over the keys decides the node layout, the index is then built top-down with a linear model
per node, and each leaf keeps its keys in a slot array twice their number, conflicts going to child nodes.

This is a **wrapper around the authors' own code, used unmodified**; it is not a reimplementation.

| | |
|---|---|
| repository | https://github.com/pfl-cs/DILI (the paper's PDF links to it; commits by an alibaba-inc.com address, the first author's affiliation) |
| commit | `babd1a2c53df841b55bc5867a6925ccc6fe88d87` (2023-05-18, the head of `main`) |
| local clone | `/Users/louisvasseur/Downloads/readonly_competitors/DILI` |
| licence | **none stated**: no LICENSE file, no licence header. Included by path only, nothing copied into this repository |

## Parameters

`dili` runs the code as released: `RHO = 0.1`, `buMinFan = 16`, `fanThreashold = 8192`, `LEAF_MAX_CAPACIY = 8192`, leaf
slots = 2 x keys (`diliNode.h:136`), leaf sizes searched between 16 and 200 keys (`interval_utils.cpp:1754`), compile flags
`ALLOW_FAN2_NODE` and `NOT_ADJUST` not set.

**The released code and the paper do not agree on everything.** Section 7.1 of the paper sets "the decaying rate rho,
enlarging ratio eta and maximum fanout omega" to 0.2, 2 and 4,096, and section 6.1 sets the adjustment threshold lambda to
2. In the code eta is 2 (`diliNode.h:136`), omega is 4,096 (`fanThreashold / 2`, `interval_utils.cpp:1726`) and lambda is 2
(the `<< 1` in `if_retrain`, `diliNode.h:406`), but **`RHO` is 0.1, not 0.2** (`global.cpp:46`). Two more differences are in
the adjustment: the paper enlarges a leaf by min(eta + 0.1 alpha, 4) at its alpha-th adjustment, the code has that line
commented out (`diliNode.h:139`); and every node starts with `n_adjust = 30` (`diliNode.h:146`) while the insert flags a
node internal from `n_adjust >= 4` (`:830`), so in the released code a node is examined for an adjustment on the first insert
that passes through it to a child and never again. None of this is changed here: the code is what the authors released.
Which of the two values of rho produced the paper's numbers cannot be told from the repository.

Knobs in the name:

- `dili<B>`, B in [2, 200]: `buMinFan`, the fewest keys the BU-tree gives a leaf (`dili` = `dili16`). It bounds the number
  of leaves at n / B and moves size and lookup cost a little (2.1M uniform keys: B = 8, 16, 64 give 51.8, 47.2, 44.1 bytes
  a key). The slot expansion, which sets most of the size, is compiled in.
- `r<PP>` after that, PP in [1, 99]: rho = PP / 100. `dili` = `dili16r10` (released); **`dilir20` is the paper's stated
  setting**. A comparison should run both and report the better one per data set, saying which.

## How it maps to GRE's interface

| GRE | what runs |
|---|---|
| `bulk_load` | the authors' `DILI::bulk_load(keys, ptrs, n)`, once per half of the key space (below), then a check of every key |
| `get` | the authors' `DILI::search` (the top bit of the key picks the instance); on a miss, the side store if it is not empty |
| `put` | a walk down the authors' nodes by the wrapper (checks below), then the authors' `DILI::insert` |
| `remove` | the authors' `DILI::erase` |
| `update` | the wrapper's walk finds the slot and overwrites the record pointer (the authors' code has no update) |
| `scan` | the wrapper's in-order walk over the authors' nodes (their `range_query` cannot serve: see below) |
| `memory_consumption` | every node with all its slots, the two-key leaves, the retrain buffers, the side store, the object. It also prints, when the number changed since the last call, how many keys are in the side store now and how many puts went there (GRE calls it once, after the run, and gives no other hook) |

The payload is stored as the authors' record pointer (a `long`), so a lookup has no indirection DILI's own would not have.

## Build

Not header-only. `dili.h` includes the authors' ten `.cpp` files itself, inside `namespace gret::dili_authors`, so the
translation unit that includes the header carries DILI and GRE's build needs **one include path and no new source file**:

    include_directories("<path>/DILI/src")                        # Linux, in GRE's CMakeLists.txt
    -I <path>/DILI/src -I <dir with empty linux/perf_event.h and asm/unistd.h>   # macOS (the two headers do not exist there)

Checked: GRE's `microbench.cpp` compiles (`-fsyntax-only`, macOS flags) with the header included right after
`gret/registry.h`. If two translation units of one program include the header, all but one must define
`GRET_DILI_DECL_ONLY`. Without the header's inclusion of the sources, the files to compile alongside would be
`src/global/{global,linearReg}.cpp`, `src/utils/{data_utils,file_utils}.cpp`,
`src/butree/{linearRegressor,interval,buInterval,interval_utils}.cpp`, `src/dili/{diliNode,DILI}.cpp`, with g++ and
`-fpermissive` (the authors' flag; clang has no equivalent and rejects `interval.cpp` and `buInterval.cpp`).

What the header does to make the unmodified sources compile there:

- `butree/interval.h` and `interval.cpp` are included with `const` defined away: `interval::data` is `const long*` and is
  passed to functions taking `long*`. The effect is what `-fpermissive` gives.
- Everything sits in a namespace (the sources say `using namespace std;` in headers and define `interval`, `partition`,
  `SearchBound`, ... globally); the macros they leave are removed, `KEY_TYPE` and `PAYLOAD_TYPE` among them, which GRE uses
  as template parameter names.
- `utils/linux_sys_utils.cpp` is left out (perf counters, Linux-only, unused).

## What the authors' code needs, plainly

- **Key type**: `long`, hard-wired (`global_typedef.h:19`), and keys must be >= 0: -1, -2, -3 in the key field mark a child, a
  two-key leaf, an empty slot (`global.h:64`). The largest key + 1 is computed (`DILI.cpp:160`, `DILI.h:83`).
- **Global state**: the retrain buffers `dili_auxiliary::retrain_keys/ptrs` (allocated by every DILI constructor, freed by
  every destructor, shared by all instances), `interval::data/probs`, the knobs, `num_adjust_stats`. Not thread-safe; one
  bulk load at a time. The wrapper re-allocates the buffers when another instance's destructor freed them.
- **Disk directory**: the bulk load writes `h0_borders` and `h0_rmses` (about 1 byte a key) to a "mirror" directory and, if
  files are already there, **restores another run's layout from them** (`interval_utils.cpp:1699-1737`). The wrapper gives
  each bulk load a new empty directory under `$DILI_MIRROR_DIR` (default: the system's temporary directory) and removes it.
- **Record pointer -1** is `search`'s "not found" (`DILI.h:267`): a payload of 2^64 - 1 cannot be stored.
- **At least about 2M keys**: below that the bulk loader reads out of bounds (`*cs.begin()` on an empty set,
  `interval_utils.cpp:1024, 1119`: ASan stack-buffer-overflow at 1, 2, 3, 5, 300, 1000, 3000 keys) or builds from a layout
  of zero nodes (`interval_utils.cpp:1418`: NaN cast). From 2,000,000 keys the first level has at least 10,000 nodes and
  these paths are not reached.
- **Asserts on**: the authors' CMake does not define NDEBUG. GRE's does.

## Deviations (everything the wrapper does that the authors' code does not)

1. **Two instances.** Keys below 2^63 in one DILI, keys from 2^63 up in another (minus 2^63). A half with fewer than
   2,000,000 keys the loader can take is kept in the side store instead; if neither half has that many, `bulk_load` throws
   `std::invalid_argument`. So `dili` does not run on small data at all.
2. **Side store** (`std::map`), consulted only when the authors' search misses and the store is not empty. It takes:
   - payload 2^64 - 1, keys 2^63 - 1 and 2^64 - 1 (`unrepresentable`);
   - a key closer to the key below it than `(key >> 50) + 1` (`close`): three keys in one slot get slope 2 / gap
     (`put_three_keys`, `diliNode.h:408`), and with slope x key past 2^51 the double product no longer separates them, so one
     overwrites another. Only keys from 2^50 up can be affected; at 2^62 the gap needed is 4097 to 8192;
   - a key whose predicted slot is 2^31 or more at some node on its path (`int_range`): `LR_PRED` casts the double to int
     (`global.h:8`), which is undefined there. arm64 saturates to the last slot, x86 gives slot 0, which puts a large key
     before small ones;
   - keys the bulk load lost (`lost`), and a half too small to load (`small_half`).
   The line `bulk_load` prints to stderr gives each count. **A run where `side` is not a negligible share of n is not a
   measurement of DILI.** The same rules decide at `put` whether a key goes to the authors' insert or to the side store;
   `memory_consumption` prints the side store's size and the number of puts that went there (`gret: dili now in_dili=...
   side=... `), so an insert run can be checked the same way.
3. **Check after the bulk load, and rebuilds.** Every key is looked up with the authors' search and with the wrapper's walk,
   then the in-order walk is compared with the keys. Keys past the int range are taken out. If keys are lost, the index is
   built again without the keys that caused it (up to four rebuilds, then `std::runtime_error`). On arm64 no test set needed
   a rebuild; **on x86-64 (under Rosetta) the two sets with far outliers needed one each** (two on another draw): there the
   authors' loader
   routes the outliers to slot 0 (`find_child`, `diliNode.h:401`), counts them into the wrong leaf, and every leaf after it
   is loaded with shifted keys.
4. **Checks before an insert.** (a) the walk refuses a key whose slot prediction leaves the int range (2, above);
   (b) a two-key leaf that would become a three-key node with a gap ratio of 2^29 or more goes to the side store instead:
   `put_three_keys` puts the largest key at 2.5 + 2 x upper gap / lower gap, past the int range from a ratio of 2^30, and
   x86 then writes it over the smallest; (c) a node that would retrain on this insert (`diliNode.h:406, 822`; in the
   released code a node can retrain once, because `n_adjust` starts at 30 and the insert flags a node internal from 4) is
   flagged internal first if three neighbours among its keys have such a ratio, so it does not retrain; otherwise its keys
   are checked after the retrain and `std::runtime_error` is thrown if one is gone; (d) a node that would retrain on this
   insert and holds more keys than the retrain buffers (2 x 8192, `diliNode.cpp:15`; `collect_and_clear`, `diliNode.h:823`,
   writes every key of the node into them) is flagged internal first, like (c), and then takes the key through the authors'
   insert. Such leaves are common: a leaf is an equal-width slice of its parent's range, so on skewed data the bulk load
   makes leaves of hundreds of thousands of keys (3M keys: 47% of the keys of a lognormal(0, 4) set and 30% of a set with a
   dense run sit in leaves above 16,382 keys), and nothing in the authors' insert keeps one from retraining. (Until the
   verification pass the wrapper sent every put under such a leaf to the side store instead: 400,000 of 400,000 puts into
   a dense region. That was a handicap, not a guard, and is gone.) Cost of the extra walk: at 3M uniform keys `put` takes
   about 190 ns against 150 to 180 ns for the authors' `insert` called directly (220 to 240 ns for keys from 2^50 up, where
   the margin of 2 is also checked): **insert throughput of `dili` is some 10 to 25% below the authors' unguarded insert**.
5. **scan** is the wrapper's. The authors' `range_query(k1, k2, ptrs)` returns record pointers without keys, for a key
   range and not a count, into a buffer whose size it does not know; `diliNode.h:356` writes `results[1]` and returns 1, and
   the function ends without a return when a single slot matches nothing (`:397`).
6. **memory_consumption** is not the authors' `total_size()`, which counts 16 bytes per stored key for a leaf and none of
   its empty slots or child nodes (21.0 against 47.2 bytes a key on uniform keys; 16.5 against 122.6 on clustered ones). Both are
   printed at the bulk load. The paper's memory figures (appendix A.7: 3.38 to 5.07 x 10^9 bytes on 200M keys, 17 to 25 bytes
   a key) are of the `total_size()` kind, so `dili` will show about twice the paper's number; that is the allocated memory,
   counted as GRE counts it for ALEX and LIPP. Checked against the heap: with `operator new` instrumented, the bytes live
   after a bulk load are 99.1% to 100.2% of what `memory_consumption` returns on nine key sets (the missing 0 to 0.9% is
   what the authors' loader leaks).
7. `std::cout` is silenced during the bulk load (`DILI_VERBOSE=1` keeps the authors' progress lines). The destructor frees the
   two-key leaves, which the authors' destructor leaves allocated.
8. A put after a remove: allowed, as in the authors' code with NDEBUG; with asserts on it can stop at
   `assert(num_nonempty > 1)` (`diliNode.h:877`), which does not hold once keys were erased.

## Undefined behaviour in the authors' code (UBSan, file:line, reachable on real data?)

The sanitized test stops at the first of these unless the two checks are switched off for the authors' files:

| where | what | on real data |
|---|---|---|
| `butree/linearRegressor.cpp:177:85`, `:177:80`, `:175:50` | signed overflow, `right_fan * dx * dx` and `dx * right_fan` in long | yes, on every set tried, 2.1M uniform keys below 2^50 included; at 200M whenever neighbouring BU intervals start more than about 2^29 apart. The merge losses of the BU-tree are then computed from wrapped numbers: the layout is affected, exactness is not |
| `dili/DILI.h:254:24` | `LR_PRED`: double outside int cast to int, in `search` | yes, for lookups of absent keys far from a node's keys; the result is still "not found" |
| `dili/diliNode.h:401:17` | same, in `find_child` during the bulk load | yes with far outliers (the `fblike` and `highmag` sets); on x86 this is what loses keys (3 above) |
| `dili/diliNode.h:413:20`, `:432:20`, `:434:20` | same, in `put_three_keys` | yes: three keys in one slot with a gap ratio from 2^30; `:434` (the largest key) is the one that overwrites on x86 |
| `dili/diliNode.h:800:20`, `:890:20` | same, in `insert` (below the int range: slot 0 on both targets) and `erase` (absent key) | yes; harmless |

Not UB, but wrong: `DILI.h:102` divides by `ubd - lbd`, which is 0 when the level under the root has one node (the root's
model is then NaN and every key goes to slot 0; not seen at 2.1M to 8M keys); `diliNode.h:639` (`simplify`) reads slots 0
and 1 of any node with two keys, leaf or not; `erase` leaks the two-key leaf and the emptied child it drops (`:904`, `:927`).

## Tests

`tests/sota/test_dili.cpp`: get on every stored key and on absent ones, scans, put / update / remove against a `std::map`,
then every key removed, on 12 sets of 2.1M to 4.3M keys (uniform at three values of `buMinFan` and with rho = 0.2, both
halves of the key space, a run, clusters, far outliers, a stride, a run down from 2^64 - 1, large keys with close pairs and
triples, fb-like), one set with a 300,000-key leaf that takes 300,000 puts (none may go to the side store), 10 sets that
must be refused, and the name parser. Build lines are at the top of the file.

- sanitized (`build/tests/cxx`), **as given, without an ignore list: fails** in the authors' code, first at
  `linearRegressor.cpp:177:85`;
- sanitized with `-fsanitize-ignorelist` taking signed-integer-overflow and float-cast-overflow off the authors' files
  only (ASan and every other check stay on them, everything stays on the wrapper): passes, asserts on (about 6 minutes);
  passes with `-DNDEBUG` as well, which adds the puts after removes;
- `build/tests/cxx-O3`: passes (about 2 minutes);
- also passing: g++ 15 `-O3 -Wall -Wextra` on arm64 (no warning from the header), and x86-64 builds of the test run under
  Rosetta, with `-march=haswell` (FMA) and without (SSE2).

Independent verification pass (a second set of key sets, kept out of the repository: every stored key and both its
neighbours looked up, full and partial scans, on: one and two keys and an index never loaded; all keys equal mod 2^32; a
dense run from 0 then a gap to 2^62; only keys from 2^63 up, with 0 and 2^64 - 1 put, updated to 2^64 - 1 and removed; 5M
random 64-bit keys with 1.5M puts, removes and puts again; keys at 2^62 exactly the margin apart and one below it; a
lognormal set and a uniform set loaded by half and doubled by puts; a dense region taking 400,000 puts and sequential
runs; 43 magnitudes; pairs and third keys at gap ratios to 2^33; two indexes alive at once, one destroyed, a reload):
all exact on arm64 (clang -O3, g++ 15 -O3, ASan + UBSan with the ignore list) and on x86-64 under Rosetta (haswell, SSE2).
It found the large-leaf rule of deviation 4(d) and the rho mismatch above; both are fixed in the header.

Sanity timing (3M uniform keys, this laptop, not a benchmark). Against the thinnest possible GRE wrapper around the authors'
code (a class whose `get` is `DILI::search`, `put` `DILI::insert`, `remove` `DILI::erase`), same keys, separate processes,
best of three, keys below 2^40 / 2^56 / 2^62: **get** 45.8 / 54.4 / 51.6 ns against 44.2 / 50.7 / 51.3 (+1 to +7%, inside
the run-to-run spread); **put** 147 / 235 / 200 ns against 112 / 183 / 166 (**+21 to +30%**: the walk and the checks of
deviation 4); **remove** 109 / 116 / 122 ns against 92 / 105 / 108. For scale (`t-O3 none --time`, keys below 2^50): ALEX
get 57 to 62 ns, LIPP 46 to 50, STX B+tree 130 to 150; put ALEX 181 to 199, LIPP 51 to 64, STX B+tree 235 to 290.

## Known limits

- Bulk load: 4 to 5 microseconds a key at 2M to 8M keys (single thread), so some 15 to 25 minutes at 200M, and again for
  every rebuild; peak memory about 180 bytes a key at 8M (the BU-tree starts from n / 2 intervals in a `std::set`,
  `interval_utils.cpp:952`), so roughly 35 GB at 200M on top of GRE's own arrays. Not measured at 200M.
- The share of keys the side store takes on the ten 200M data sets is not known (no data set on this machine). Keys above
  2^50 with neighbours closer than `(key >> 50) + 1` go there: possible on osm, covid, books, wise, genome. Read the
  `side=` count of the bulk-load line before using a number.
- x86-64 was exercised only under Rosetta, with Apple clang; g++ 11 on Linux was not available here.
- Lookups of absent keys still execute the authors' out-of-range cast (`DILI.h:254`): the answer is right on arm64 and
  x86-64, but it is undefined behaviour in C++.
- After a retrain the wrapper can detect a lost key, not restore it: it throws. No test set triggered this.
- If the process dies during a bulk load, the (small) scratch directory `gret_dili_<pid>_...` stays in the temporary
  directory.
- Single-threaded; one bulk load at a time in the process; `init`'s `Param` is ignored.
- **Insert throughput is that of the authors' insert plus the guard**: 20 to 30% slower than their unguarded insert in the
  timing above. A paper table should say so, or quote the unguarded number next to it for runs where the side-store line
  shows no put was diverted.
- **Sequential inserts degenerate, in the authors' code.** A key outside a node's range is clamped to its first or last
  slot, so keys inserted in ascending order above the largest key (or a run of consecutive keys into one gap) build a
  chain: after 20,000 such inserts into 2.5M keys the last key sits 3,292 nodes deep (1,257 for a run in one gap), the
  authors' `insert` called directly takes 24 microseconds per key (14), and `put` here 63 (28). The cost is quadratic, and
  the authors' `insert` and destructor recurse once per level, so a long enough run overflows the stack. `dili` is for
  workloads that insert keys in random order (as the paper's and GRE's do), not for sorted or append-only inserts.
- **The layout depends on the compiler and the target for some key sets.** `linearReg_with_max_b` (`diliNode.h:63`)
  subtracts sums of squared keys in double; for a leaf that is narrow relative to its keys' magnitude the result is
  rounding noise, and the fallback and nesting that follow differ from build to build: 2.2M keys 4,097 apart at 2^62 take
  64.9 bytes a key with Apple clang on arm64, 110.3 with g++ 15 on arm64 and 115.8 with clang for x86-64; a set with a
  400,000-key dense run near 2^47 takes 56.0 against 86.6. Exactness is not affected; size and speed measured on one
  build do not carry over to another.
- The authors' bulk loader leaks up to 0.9% of the index size (not counted by `memory_consumption`: unreachable).
- `bulk_load` needs 2,000,000 keys in one half of the key space; `dili` cannot be started empty or small (an index never
  loaded answers exactly, from the side store alone, which measures `std::map`).
- Deleted through `indexInterface*` (no virtual destructor, as GRE does), the authors' nodes are not freed.
