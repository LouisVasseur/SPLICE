# sali: SALI behind GRE's interface

`include/gret/sota/sali.h`, `tests/sota/test_sali.cpp`. Names: `sali`, `saliu`.

## What it is

SALI (Ge, Zhang, Shi, Luo, Guo, Chai, Chen, Pan: "SALI: A Scalable Adaptive Learned Index Framework based on Probability
Models", SIGMOD 2024) is the concurrent successor of LIPP. It keeps LIPP's structure: every node has a linear model that
sends a key to exactly one slot, a slot holds a key, a child node or nothing, and two keys that meet in a slot get a child
node, so a lookup never searches inside a node. What SALI changes is how the tree is maintained under many threads: locks are
per slot (a version word in each slot, read optimistically), the per-node counters that LIPP updates on every insert are
gone, and a node is rebuilt when a random draw says so (a probability computed from the node's size, age and insert rate)
instead of when a counter passes a threshold. The paper adds two more "evolving" strategies on top, one that flattens
subtrees that are read often and one that compresses cold nodes; both are compile-time options and both are off in the
released code.

This is a **wrapper around the authors' own code, used unmodified**; it is not a reimplementation.

| | |
|---|---|
| repository | https://github.com/cds-ruc/SALI (the authors' group; the README names the paper) |
| commit | `dba9157439e450af4a7d85d101696f62191f2554` (2023-11-30, the head of `main`). The authors' GRE fork, https://github.com/YunWorkshop/GRE_SALI, pins the same commit as a submodule (`src/competitor/sali/src @ dba9157`) |
| local clone | `/Users/louisvasseur/Downloads/readonly_competitors/SALI`; on the cluster `/scratch/lvasseur/readonly_competitors/SALI` (same files by checksum) |
| licence | MIT (`LICENSE`: Copyright 2023 Jiake Ge, 2022 eric-lo, 2021 Jiacheng-WU: the code descends from LIPP and from GRE's concurrent LIPP+) |
| files of theirs used | `src/core/sali.h` (the index), `src/core/sali_base.h` (the models), `src/core/concurrency.h` (the version lock), `src/core/piecewise_linear_model.h` (included by `sali.h`; used by the compression only). Included by path, nothing copied |
| files of theirs not used | `CMakeLists.txt` (it compiles with `-O0`), `src/examples/` |

## How it was built

`sali.h` of this repository includes the authors' `sali.h` (as `<sali.h>`, found through `-I <SALI>/src/core`) inside
`namespace gret::sali_authors`, after including every standard and TBB header their files include, so that only their own
code lands in the namespace. Around that include:

- `exit` and `fprintf` are defined as macros. The authors' sources use both names in one place only, their runtime check
  `RT_ASSERT` (`sali.h:44-51`), which prints a line and calls `exit(0)`: a failed check ends the process **with status 0**.
  With the two macros a failed check records file, line and expression and throws `gret::sali_detail::RtAssert` instead.
  Their files are not edited; this is the one place where their code behaves differently from a plain include.
- The macros their header leaves behind (`likely`, `unlikely`, `RT_ASSERT`, `READ_EVOLVE`, `COMPRESS`, `PRINT_DEBUG`,
  `RT_DEBUG`, `COLLECT_TIME`) are saved before and restored after, so GRE's LIPP, which defines some of the same names, can
  be included in the same translation unit (the test does that, in GRE's order).
- Two private members of their classes are read by `memory_consumption` (the pool of recycled two-key nodes and the lists of
  nodes waiting for an epoch) through explicit template instantiations, which may name private members. Nothing of theirs is
  written from outside.
- The authors' object lives inside the wrapper object (`std::aligned_storage`), as it lives by value in their own GRE
  wrapper: a lookup follows no pointer the authors' wrapper would not follow.

Needed on the include path and at link time: `<SALI>/src/core`; TBB's headers `tbb/combinable.h`,
`tbb/enumerable_thread_specific.h`, `tbb/spin_mutex.h` and `-ltbb`; `omp.h` (their code calls `omp_get_thread_num()` to pick
a pool; GRE builds with `-fopenmp`). GRE's own build has all of it.

**The Mac has no TBB and, on arm64, no `<immintrin.h>`.** There the build puts four *empty* files on the include path
(`tbb/combinable.h`, `tbb/enumerable_thread_specific.h`, `tbb/spin_mutex.h`, `immintrin.h`; the test's head has the
commands, they live in the build directory) and the wrapper header, seeing that TBB was not really included, supplies
stand-ins inside its own namespace: an `enumerable_thread_specific` that finds a thread's object by a search under a mutex,
a `spin_mutex` on `std::atomic_flag`, and `_mm_pause` as `sched_yield`. They exist so the test can run on the Mac; the line
`bulk_load` prints says `tbb=stand-in` or `tbb=real`, and **no timing from a stand-in build means anything** (its epoch
entry takes a mutex).

Their `concurrency.h` defines `sali::yield` without `inline`, and the wrapper instantiates two templates explicitly: include
`gret/sota/sali.h` in **one** translation unit of a program (GRE's benchmark is one unit).

Checked on the cluster: a file that includes GRE's whole `competitor/competitor.h` (ALEX, LIPP, PGM, B+tree, ART and
`gret/registry.h` with `GRET_SOTA` and `GRET_SOTA_DILI`, the benchmark's flags and include paths) and then
`gret/sota/sali.h` compiles (`g++ -fsyntax-only`), with both names made.

## Modes and knobs

SALI has no run-time knob, so the names carry none.

- `sali`: the code as released: `READ_EVOLVE 0`, `COMPRESS 0`, the constructor's arguments as in the authors' GRE wrapper
  (`BUILD_LR_REMAIN = 0`, no memory budget). Insert-evolving, the probability-triggered rebuild, is always on. `put` first
  asks the authors' `exists()` whether the key is stored.
- `saliu`: the same index, `put` without that question, which is what the authors' GRE wrapper does (`put` there is
  `sali.insert(key, value); return true;`). Use it to time inserts. A `put` of a key that is stored throws (for keys from 2^60
  up, where the neighbour test of deviation 4 sees the key first, it returns false).
- **No read-evolving variant.** The two modes are fixed by `#define READ_EVOLVE 0` and `#define COMPRESS 0` at
  `sali.h:35-36`, inside the file and without `#ifndef`; a `-D` on the command line is overwritten by them, and the
  authors' README says to edit the file to turn them on. Under the rule that their files stay untouched there is no way to
  select the mode. If a read-evolving run is wanted later, the honest route is a generated copy of `sali.h` with that one
  line changed, a second header name for it, and the name `salir` saying so; the wrapper refuses to compile against such a
  copy today (`#error` if either macro is set), because read-evolving changes `at()` (it rebuilds subtrees from inside a
  lookup, on a random draw) and was not tested here. Read-evolving targets skewed lookups; GRE's default lookups are
  uniform over the keys.

- **`-DGRET_SALI_SHIFT=n`** (a build flag, not a name; default 60 on x86-64) sets the shift of the side store's rule
  (deviation 4). It is not a knob of SALI but of the wrapper's guard, and exists to measure what the guard costs: a put
  of a key from 2^n up pays one extra range query, about half as much again as the authors' insert (below). 61 is still
  1.6 times the distance the derivation asks for and leaves keys below 2^61 (all of covid) alone; 63 is the narrowest
  rule the probe supports; 64 is no rule. The line `bulk_load` prints names the shift in use (`side store for gaps below
  (key >> 60) + 1`). Only the default has been through the whole test in all three places; 61 and 63 passed it on the
  cluster (Tests).

Recommended sweep: `sali` for read-only workloads (one point; there is nothing to tune), `saliu` in addition wherever
inserts are timed.

## How it maps to GRE's interface

| GRE | what runs | the authors' GRE wrapper, for comparison |
|---|---|---|
| `bulk_load` | the authors' `SALI::bulk_load(pairs, n)`, then every key looked up once with their `at()` | `bulk_load` alone |
| `get` | the authors' `SALI::at`; only when that does not find the key: the side store, if it is not empty | `at` |
| `put` | (`sali` only) their `exists()`; for keys from 2^60 up one range query of length 1 (below); their `SALI::insert`; if their rebuild inside it fails, a scan and a new bulk load (deviation 11; never in a GRE run) | `insert`, returns true whatever happened |
| `update` | their `exists()`, then their `SALI::update` | `update` |
| `remove` | their `exists()`, then their `SALI::remove` | `remove` |
| `scan` | their `SALI::range_query_len`, merged with the side store's keys if it holds any | `range_query_len` |
| `memory_consumption` | their `total_size()` plus what it leaves out (below) | `total_size()` |

## What the authors' code needs, plainly

Each line below was seen in a test or read off the source; the wrapper's answer is in the next section.

- **A lookup of a key that is not stored can end the process.** `at()` walks to the key's slot; if the slot holds another
  key it runs into `RT_ASSERT(node->items[pos].comp.data.key == key)` (`sali.h:610`): a line on stderr and `exit(0)`. It
  returns false only when the slot is empty. GRE looks up stored keys only, which is why their wrapper survives.
- **`update` and `remove` of a key that is not stored act on another key.** Neither compares the key in the slot
  (`sali.h:864`, `:796`): `update` overwrites the payload of whatever key is there and returns true, `remove` erases it.
- **`insert` of a stored key** ends in `RT_ASSERT(key1 < key2)` (`sali.h:1618`) with the slot's lock held.
- **Close keys at large magnitudes.** The models are `long double`. Two keys that share a slot get a node of 8 slots whose
  model aims them at slots 8/3 and 16/3; two keys 1 apart above 2^63.58 can come out in the same slot, and the check at
  `sali.h:1677` ends the process. A bulk load, or a rebuild, of three or more keys 1 apart above 2^63 can put all of them
  in one slot at every level and then never ends (memory grows until it is gone). On a machine where `long double` is a
  double (arm64) the same happens 2^11 times lower.
- **At most 2^30 slots in a node**: a predicted slot above `INT_MAX / 2` is clamped to the last one (`sali.h:121`), and the
  root has two slots a key from 1,000,000 keys on (`sali.h:93-99`, `:2361`), so a bulk load takes at most 2^29 keys. The
  key count is an `int`.
- **Removes and inserts mixed can end the process** (found by the independent verification, not by the first test).
  When an insert meets a stored key, the authors' code may rebuild a subtree on the path, and it sizes the new nodes with
  `speed / prev_speed`, where `speed` is (keys in the subtree now - keys when it was built) / time (`sali.h:3192`, `:3197`).
  Inserts alone make every subtree grow, so `speed` is positive. With removes in between, a subtree can hold exactly as many
  keys as when it was built: `speed` is 0, the rebuilt nodes inherit 0, and the next such rebuild below divides 0 by 0. The
  NaN goes through `args.ratio < 1 ? 0 : ...` (`sali.h:2301`, false for NaN) into the slot count, a cast of NaN to `int`
  (`sali.h:2361`, undefined behaviour). On x86-64 the cast gives INT_MIN and `new_items(INT_MIN)` throws
  `std::bad_array_new_length` out of `insert`: the authors' build ends in `std::terminate`. On arm64 it gives 0 and the
  check `node->model.a1 >= 0` (`sali.h:2435`) fails for a subtree of four keys or more: `exit(0)`. Either way the subtree
  that was being rebuilt is left with every slot locked for good, so a caller that caught the exception would spin for ever
  in the next lookup there. A remove and a put in turn on 200 keys reach this after about 1,250 steps on average (30 of 30
  runs on the cluster, 30 of 30 on the Mac). **GRE cannot reach it**: its benchmark refuses a run with both inserts and
  deletes (`benchmark.h:232`, `INVARIANT(insert_delete == insert_ratio || insert_delete == delete_ratio)`), removes alone
  never rebuild, and inserts alone never give a zero.
- **Memory that `total_size()` does not see**: the pool of recycled two-key nodes (1024 stacks, one per OpenMP thread
  number) and the nodes a rebuild replaced, which wait in per-thread lists until the epoch has advanced three times.
- It prints to `std::cout` from its constructor and destructor.
- `remove` empties a slot and nothing else: emptied nodes stay, and a scan walks every one of their slots.

## Deviations (everything the wrapper does that the authors' code does not)

1. **A failed runtime check throws** (above). When it happens anywhere but in a lookup that misses, the authors' object is
   in a state nobody can repair (a locked slot, or no root): the wrapper sets it aside, every later call throws
   `std::logic_error`, and a new `bulk_load` makes the index usable again. The message names the authors' file, line and
   expression. The same is done for any exception that is not a runtime check and comes out of the authors' `insert` or
   `bulk_load` (out of memory): the tree may then hold a locked slot or a child pointer that was never set. No test set
   that respects the side store's rule ends this way; the one failure that does occur in the tests is repaired (deviation 11).
2. **A lookup that misses.** The wrapper catches the one check at `sali.h:610` and answers false (after asking the side
   store). A hit runs the authors' `at()` and nothing else: no extra branch, no extra memory access. A miss that lands on
   another key costs a C++ throw, 4 to 16 microseconds on the cluster's login node. GRE's lookups are hits.
3. **`put`, `update`, `remove` ask the authors' `exists()` first** (a walk without locks or epoch), so a stored key is not
   inserted twice and an absent key is not updated or removed in another key's place. `saliu` leaves the question out of
   `put`, like the authors' wrapper; given a stored key it throws `std::runtime_error` (the check at `sali.h:1618`) and the
   index is unusable until the next `bulk_load`: loud, not wrong.
4. **Side store** (`std::map`) for keys the authors' models may not tell from a neighbour. A key that is closer to the key
   kept below it than `(key >> 60) + 1` stays out of SALI (`>> 49` where `long double` is a double). On x86-64 this can
   only concern keys from 2^60 up with a neighbour less than 2 (at 2^60) to 16 (at 2^64) away. The same rule decides at
   `put`: for a key from 2^60 up the wrapper asks the authors' range query for the first key at or above `key - margin + 1`
   and sends the new key to the side store if a stored key is within the margin on either side.
   Why this width: with p bits of mantissa a two-key node can only fail when its keys are closer than key / 2^(p - 0.42)
   (the product 8/3 x key / gap then has an ulp of 4, more than the 8/3 between the two targets), and a node of three keys
   or more puts two of its keys in different slots, so that the build makes progress, whenever neighbours are
   key / 2^(p - 2.3) apart or more (whichever of the three ways its model is fitted, two of its keys are at least 6 slots
   apart in exact arithmetic, and each computed position is off by at most one ulp of slope x key, which is then under
   2.5 slots). The rule asks for 3.2 times the second distance. Measured (`t --probe`, cluster, below): without any rule
   the authors' code failed on 268 of 33,600 runs of close keys, all of them keys 1 apart from 2^63 up; no run that a rule
   with shift 63, eight times narrower than the one used, would let through failed.
   **What the rule costs a put** (measured by the independent verification, cluster login node, 2M keys loaded and
   500,000 new keys put, the same keys with the rule and without, three passes each, no key in the side store): for keys
   below 2^60, nothing (about 380 ns with or without); for keys in [2^61, 2^62) and from 2^63 up, about 600 to 640 ns
   with the rule against 400 to 430 without it and about 400 for the authors' `insert` called directly, in `sali` and in
   `saliu` alike. So **on data sets whose keys are 2^60 or more, every insert costs SALI about 45 to 50% more than its
   authors' code needs** (at 2M keys; the share at 200M is not measured). `-DGRET_SALI_SHIFT` (Modes and knobs) moves or
   removes the rule for a measurement of that.
   **The line `bulk_load` prints gives the count (`side=`). A side-store key is answered by a miss of the authors' `at()`
   (the throw) and a map lookup, some 4 microseconds: at a side share f the mean lookup time grows by about f x 4 us, so
   0.4 ns at 0.01% and 40 ns at 1%. A run where the share is not negligible is not a measurement of SALI.**
5. **Check after the bulk load**: every key is looked up once with the authors' `at()`; `std::runtime_error` if one is not
   returned. This is inside `bulk_load` and therefore inside GRE's build time; the printed line gives `build_ms` (the
   authors' part) and `check_ms` separately (on the cluster, 1M and 2M keys: the check took 21 to 49% of the time of the
   authors' build).
6. **Refused input**: unsorted or repeated keys and more than 2^29 - 1 keys throw `std::invalid_argument` before the index
   is touched (the authors' code would end the process on the first, with status 0).
7. **memory_consumption** is the authors' `total_size()` (every reachable node with all its slots) plus the two-key pool,
   the nodes waiting for an epoch, the SALI object itself (148 KB with libstdc++: 1024 pool stacks and 1024 padded
   counters), the side store and the wrapper. After a bulk load the additions are a fixed 0.8 MB (0.4% at 1M keys). After
   inserts they can be large, and they are not the same from run to run: after the same 193,585 puts into 1M uniform keys,
   one run of the test held 436 MB of which 189 MB were replaced nodes waiting for an epoch, the next run 246 MB and
   none waiting (the authors' rebuild decisions use the wall clock and generators seeded with `time(0)`). `total_size()`,
   which is what the authors' wrapper reports, counts none of the waiting nodes. After writes `memory_consumption` prints
   one line with each part (`gret: sali now ... bytes: authors_total_size=... waiting_for_an_epoch=...`), so either number
   can be quoted. Against glibc's own count of bytes in use (`mallinfo2`) the returned number is 93.6 to 97.3% in the
   cases the test runs (after a load, after puts, an index grown from nothing by puts); the rest is allocator overhead and
   the epoch manager.
8. `scan(key, 0, ...)` returns 0 without calling the authors' code (theirs writes one result before looking at the length).
9. `std::cout` is silenced while the authors' constructor and destructor run (`GRET_SALI_VERBOSE=1` keeps their lines).
10. Deleted through `indexInterface*` (which has no virtual destructor, as GRE does it) the authors' nodes are not freed.
11. **A rebuild when the authors' rebuild fails** (the 0 / 0 above; added by the independent verification). `put` catches
    `std::bad_array_new_length` (x86-64) and the failed check `node->model.a1 >= 0` (arm64) coming out of the authors'
    `insert`, and nothing else this way. At that point the new key is in (the insert comes before the rebuild) and the
    subtree that was being rebuilt is still linked from its parent and complete, but locked, with its nodes on a free
    list: the authors' destructor would free it twice, and only their scan, which takes no lock, can still read it. The
    wrapper reads every key out with one `range_query_len`, checks that it got exactly the number it counts and in order,
    sets the authors' object aside as it is, bulk loads a new one from those keys and looks every key up. If any of that
    does not hold, the index is given up as in deviation 1. Costs: one scan and one bulk load of everything, at that put;
    **the bytes of the object set aside stay allocated** (their `total_size()` and pool are added to
    `memory_consumption` from then on, as `set_aside=`); a line on stderr each time, and the count in the line
    `memory_consumption` prints (`rebuilt N times after a failed rebuild of the authors'`). Before this fix the exception
    left the wrapper believing the index was usable, and the next `get` in that subtree never returned (seen on the
    cluster: killed after 20 s). In the test's cycles (132,000 remove+put steps at a constant 200, 3,000 and 500 keys) it
    happens 44 to 67 times per name; in every other test set, never. A run in which the count is not 0 measured the
    authors' code plus those rebuilds; GRE's runs cannot have one.

Nothing differs from the paper in the algorithm that runs, because the algorithm is the authors' code. What differs from
the paper's *evaluation*: one thread instead of many; read-evolving and compression off (as in the released code); and the
side store, the checks and the lookup after the bulk load listed above.

## What running a concurrent index on one thread costs

None of SALI's concurrency control is taken out, and none of it can be without editing the authors' files. On one thread
it buys nothing and costs:

- **per operation**, an epoch guard: two lookups of the thread's record in `tbb::enumerable_thread_specific` (enter and
  leave), a load of the global epoch and two atomic stores;
- **per node on the path**, the slot's version word read before the slot is used and again after;
- **per slot, 32 bytes instead of LIPP's 16** (plus LIPP's two bitmaps): the version word and the type byte. Measured, 3M
  uniform keys: 163 to 166 bytes a key for `sali` against 82 to 83 for GRE's LIPP;
- **per write**, a compare-and-swap to take the slot's lock and an atomic add to release it; replaced nodes are not freed but
  queued, and the queue is only emptied when the thread's list happens to hold a multiple of 64 nodes at the end of an
  insert (deviation 7).

Measured on the cluster's login node (3M uniform keys, shared machine, a sanity check and not a benchmark): the authors'
`exists()`, which is the same walk without epoch, version checks or payload, takes 36 to 57% of the time of their `at()`
on the same keys. About half of a `sali` lookup at this size is therefore concurrency control, and LIPP, which has none, is
the single-thread reference for what the structure itself costs. A reviewer comparing `sali` with single-threaded indexes
should read its numbers with that in mind; it is the price of running the code as released.

## Undefined behaviour and sanitizer findings in the authors' code

| where | what | on real data |
|---|---|---|
| `sali.h:739-740` (ASan, libc++: container-overflow, write of 8 bytes) | `bulk_load` reserves two vectors and fills them through `operator[]` without resizing them; the build then reads them through pointers (`sali.h:2332-2333`) | **yes: every key of every bulk load of 3 keys or more.** The memory is allocated and large enough, so nothing is read or written outside an allocation, but it is beyond `size()`. libstdc++ without `_GLIBCXX_ASSERTIONS` (the benchmark build) does not notice |
| `sali.h:2361:38` (UBSan: "nan is outside the range of representable values of type 'int'") | the slot count of a rebuilt node is `static_cast<int>` of size x (gaps + ratio), and the ratio is 0 / 0 after two rebuilds of a subtree that did not grow (`sali.h:3197`) | **only with removes and inserts mixed** (above); never in a bulk load, in inserts alone or in removes alone, so never in a GRE run. The cast is the authors' bug itself: INT_MIN on x86-64 (then `std::bad_array_new_length`), 0 on arm64 (then the check at `sali.h:2435`; for a rebuilt subtree of exactly three keys that check passes and a node of 0 slots is written to, see Known limits) |
| `sali.h:2367:53` (UBSan, x86-64 only: "signed integer overflow: -2147483648 - 2") | the next use of that slot count, `L - 2` | the same case and nothing else: it follows the cast above when the cast gave INT_MIN |
| `concurrency.h:62` | `sali::yield` defined in a header without `inline` | a link error if two translation units include it |
| `sali.h:44-51` | `RT_ASSERT` ends the process with status 0 | whenever a check fails; redirected (deviation 1) |
| `sali.h:117`, `:975` | `struct Node;` declared twice in the class | a clang warning, nothing else |

The sanitized test turns ASan's container-overflow check off for the whole test binary (a function at the end of the test
file; `-DTEST_SALI_STRICT_ASAN` leaves it on and the run then stops at `sali.h:739` on the first bulk load of three keys).
That switch also covers the wrapper's own vectors; every other ASan and UBSan check stays on for the authors' code and the
wrapper alike. The test also turns off one UBSan check, `float-cast-overflow`, for the functions of the authors' header
and of the wrapper (a `#pragma clang attribute` around the include in the test file; the wrapper converts no floating
point number to an integer): with it on, the sanitized run stops at the authors' cast at `sali.h:2361` in the `cycles`
part before the wrapper's repair can run, and could stop there in any part that mixes removes and puts. On x86-64
`signed-integer-overflow` is off in the same way, for `sali.h:2367` (the wrapper's own arithmetic is unsigned).
`-DTEST_SALI_STRICT_UBSAN` leaves both on and shows the report. With those switches, **no other sanitizer report**
comes from the authors' code or from the wrapper on any test set, on arm64 or on x86-64, the repair included.

## Tests

`tests/sota/test_sali.cpp` checks every answer against a sorted array and a `std::map`: get on every stored key and on keys
never stored (both neighbours of stored keys, 0, 1, 2^63 - 1, 2^63, 2^64 - 2, 2^64 - 1, below, above and inside the range),
scans from stored, neighbouring, in-between and extreme start keys at lengths 0 to 70,000 and over everything, then put,
update, remove (absent keys and a second remove must return false), put again, every key removed and the emptied index
refilled, with the same checks after each step; an index never loaded; the guards (refused input leaves the index as it
was, `saliu` given a stored key throws the authors' check and recovers at the next `bulk_load`, malformed names); that the
side store holds exactly the keys the rule sends there; that `memory_consumption` is the sum of its parts and, with glibc,
within 85 to 101% of what `mallinfo2` reports in use; and `cycles`: a remove and a put next to a stored key in turn, at a
constant 200, 3,000 and 500 keys (the last from 2^62 up), 132,000 steps per name, with a lookup of the key just put at
every step, of the key just removed at every 16th, and the full comparison at the end. That is the workload in which the
authors' rebuild divides 0 by 0 (deviation 11); the line it prints says how often the wrapper had to replace their object
(44 to 67 times per name in the runs below; the number depends on the clock).

Key sets: nothing, one key (5, 0, 2^64 - 1), 2, 3, 5 and 17 keys, uniform below 2^50, uniform over 64 bits, a dense run,
far outliers, outliers at the top, clusters at any magnitude and below 2^44, a regular stride, a huge gap, pairs, triples
and runs 1 apart next to 2^63 and 2^64, a mixed set at 2^61 to 2^62, `tight` (1,500 runs of 2 to 200 keys whose gaps are
exactly the smallest the rule lets into SALI, or up to three times that, at every magnitude from the rule's start to 2^64),
and two sets of 2M keys (uniform below 2^50, and over all 64 bits with 0, 2^63 - 1, 2^63 and 2^64 - 1).

| where | build | result |
|---|---|---|
| Mac, arm64, Apple clang 17, `build/tests/cxx` (-O1, ASan + UBSan), TBB stand-ins, long double = double (rule at `>> 49`) | with GRE's ALEX, LIPP, PGM and B+tree headers included first | **passes**, 68 s; container-overflow off, and float-cast-overflow off in the authors' functions (above); `cycles`: their object replaced 50 and 48 times |
| Mac, arm64, `build/tests/cxx-O3` | same | **passes**, 22 s; `cycles`: 44 and 49 times |
| Mac, x86-64 under Rosetta (`-arch x86_64 -mavx2`), ASan + UBSan and -O3, TBB stand-ins, 64-bit mantissa (rule at `>> 60`) | `-DTEST_SALI_ALONE` (GRE's Mac shim of `x86intrin.h` does not compile for x86) | **passes**, 96 s and 34 s; `cycles`: 49 and 48 times sanitized, 67 and 55 times at -O3 |
| cluster login node, g++ 11.4, `-std=gnu++17 -O3 -march=native -fopenmp`, real TBB 2020 (`wf_cxx`) | with GRE's headers included first | **passes**, 33 s of CPU (2 to 4 min of wall time on the shared node), including the `mallinfo2` checks; `cycles`: 44 and 48 times |
| same, with `-DGRET_SALI_SHIFT=61` and with `=63` | **both pass** the whole test (33 s of CPU each), with the key sets made for the rule (`pairs_top`, `triples63`, `run_top`, `highmag`, `tight`) following the shift in use; `cycles`: 61 and 50 times, 60 and 61 times |
| same, `t --probe` | | the rule's evidence, below; 3 min in the first pass. The verification's rerun, with the final header, was stopped by its own 290 s limit on a busier node while in the last of the five ranges; the four ranges it finished agree with the table below (no failure but runs 1 apart from 2^63 up) |

`t --probe` (Linux only: each case runs in a child process under 1 GB and 2 s of CPU) runs the authors' code **without** the
side store's rule on 33,600 small sets: runs of 2, 3, 4, 5, 7, 100 and 3000 keys with smallest gap 1, 2, 3, 4, 6, 8, 12,
16, 33 or 65, in [2^60, 2^61), [2^61, 2^62), [2^62, 2^63), [2^63, 2^63.58) and [2^63.58, 2^64), in three shapes (equal
gaps; gaps drawn from [gap, 3 gap); three runs 2^10 to 2^30 apart), alone or with a far key on each side, bulk loaded at
once or every other key loaded and the rest put. Result on the cluster:

| keys in | smallest gap | cases | not answered |
|---|---|---|---|
| [2^63.58, 2^64) | 1 | 672 | 267: the authors' check failed in 263, memory ran out in 4 |
| [2^63, 2^63.58) | 1 | 672 | 1: memory ran out (a run of 100) |
| every other combination | | 32,256 | 0 |

By rule: of the 32,461 cases a rule with shift 63 would let into SALI, 0 failed; shift 62: 0 of 30,256; shift 61: 0 of
26,460; **shift 60 (the header's): 0 of 21,412**. The test fails if a case the header's rule, or the rule half as wide, lets
through is not answered.

Sanity timing (`t --time`, cluster login node, 3M uniform keys, 2M lookups of stored keys, best of five passes; **not a
benchmark**: the node was shared with other tests and whole runs differed by up to a factor of ten, so only numbers from
one run compare). The quietest run (keys below 2^63): `sali` get 122 ns, the authors' `at()` called directly 123 ns (the
wrapper adds nothing measurable to a hit), their `exists()` 63 ns; a get that misses 4.4 us; put 391 ns (with the
`exists()` walk and, for these keys, the neighbour test); 166 bytes a key. In the same run GRE's LIPP: get 109 ns, put
189 ns, 83 bytes a key; ALEX: get 114 ns, put 434 ns, 26 bytes a key. Over the ten `sali`/`saliu` lines of five runs,
`exists()` took 36 to 57% of `at()`, a miss 4.1 to 16 us, and get through the wrapper was within the noise of `at()`
called directly. Timings that can be quoted have to come from the benchmark job.

## Independent verification

A second pass, by another agent that did not write the wrapper, rebuilt everything from the files, reran the test in the
three places, and ran checks of its own (a scratch program, not kept in `tests/`; what it found is now in the test as
`cycles`). Each of its key sets is bulk loaded and compared with a `std::map`, put through 6,000 to 1,500,000 interleaved
put / remove / update / get / scan operations on keys at and around the stored ones, compared again, bulk loaded a second
time and compared, for `sali` and `saliu`: nothing; one key (0, 1, 12345, 2^63 - 1, 2^63, 2^64 - 2, 2^64 - 1, the rule's
start and the key below it); two and three keys at the ends of the domain; keys equal modulo 2^32; dense runs with a
huge gap; keys above 2^63 and above 2^63.9; powers of two with their neighbours; every magnitude equally; runs 1 and 2
apart just below and across the rule's start (below it nothing is guarded); random walks with gaps of 1 to 3 margins and
of exactly the margin at every magnitude from the rule's start up; clusters with far outliers; a side store that holds
most keys (shifts 1 to 45), for the merge in `scan`; 200,000 to 300,000 puts in a row upwards and downwards at seven
magnitudes; 600,000 puts and removes of the same 4,000 keys; 4M random keys with 1.5M operations and 3M with 600,000;
and the authors' code under a rule at shift 63 on 200,000-key runs 1 apart in [2^60, 2^63) and 2 to 9 apart above 2^63.

| where | sets | answers compared | result |
|---|---|---|---|
| Mac, arm64, -O1 ASan + UBSan | 167 | 49.0 million | all right, no sanitizer report |
| Mac, arm64, -O3 (with the 4M and 3M sets) | 169 | 73.5 million | all right |
| Mac, x86-64 under Rosetta, -O3 (with the shift-63 sets) | 132 | 42.8 million | all right |
| cluster login node, g++ 11.4, real TBB (all of it, in six parts) | 134 | 67.3 million | all right, after the fix below |

What it found:

1. **The authors' rebuild divides 0 by 0 when removes and inserts are mixed**, and the wrapper did not treat the
   resulting `std::bad_array_new_length` as a failure of the index: one of the cluster runs ended there (`saliu` on the
   powers of two) and the next lookup would have spun for ever. Reproduced at will (a remove and a put in turn on 200
   keys: 30 of 30 runs), traced to `sali.h:3197` and `:2361` (UBSan names the cast), fixed in the wrapper (deviation 11)
   and added to the test (`cycles`). GRE has no run with both inserts and deletes, so no benchmark number is affected.
2. **The neighbour test of the side store's rule costs a put of a key from 2^60 up about 45 to 50%** (deviation 4). It
   was known and not measured. The rule is unchanged; `-DGRET_SALI_SHIFT` was added so the cost can be measured on a
   data set, and the bulk-load line now names the rule in use.
3. **The side store's share on the 200M files** can be bounded without running on them (Known limits): at most about
   0.03% on covid, far less on osm and books, nothing on the other seven.
4. Provenance is as stated: the clone is at `dba9157`, which is the head of the authors' `main` (`git ls-remote`) and
   the commit their GRE fork pins (`src/competitor/sali/src` in YunWorkshop/GRE_SALI), with a clean working tree and the
   same files on the cluster by checksum. The wrapper makes the calls of the authors' `SALIInterface` and constructs
   their object as it does (their member is default-constructed: `BUILD_LR_REMAIN` 0, no memory budget). A lookup hit
   runs their `at()` and nothing else.
5. `memory_consumption` against glibc's count of bytes in use, rerun: 96.2, 97.3 and 96.4% after a load, 95.6, 93.6 and
   95.8% after 200,000 puts. The authors' `total_size()` alone, which their wrapper reports, is 95.8% after a load and
   48.9 to 82.1% after puts (the rest is nodes their rebuilds replaced and have not freed).
6. Build at 200M was not run. On the login node the 2M sets took 0.45 to 0.7 us a key for the authors' build and 0.15 to
   0.2 us a key for the check (so minutes at 200M), and 164 to 168 bytes a key on uniform keys; clustered data make more
   child nodes, at 32 bytes a slot, and will need more.
7. One more fix: the copy of the pairs that `bulk_load` makes when the side store takes a key is allocated once
   (`reserve`), not grown by doubling, which at 200M keys held two copies for a while.

## Known limits

- **The side store's share on the ten 200M data sets was not measured** (the rule forbade running on them here), but it
  can be bounded from a count this repository already has. Only keys from 2^60 up with a neighbour closer than 2 (at
  2^60) to 16 (at 2^64) can go there. Such a pair is, at least 15 times in 16, also a pair of adjacent keys that are equal
  as IEEE doubles (doubles are 256 apart or more from 2^60 up), and `scripts/double_ties.py` counted those pairs on the
  200M files (`README.md`, "LIPP fails on covid, osm and books"): **covid 61,553, osm 2,571, books 264, the other seven
  files none**. So on the benchmark machine the side store can hold at most about 0.03% of covid (about 1.2 ns on the
  mean lookup at 4 us a side-store key, deviation 4), 0.0013% of osm, 0.0001% of books, and nothing of fb, genome,
  history, libio, planet, stack and wise; those counts include pairs up to 255 apart, so the real numbers are lower.
  Still read `side=` in the bulk-load line before using a number. The constructor takes another shift as its third
  argument if a set needs a narrower rule; 63 is the narrowest the probe supports. (On the Mac, where the rule starts at
  2^49 and is 2^11 times wider, the share on covid, osm and books would be far larger: no number from there means
  anything for this index.)
- **Build time includes the check** of every key (deviation 5); subtract `check_ms` for the authors' build alone. At 200M
  expect several minutes for the build and one to two for the check (not measured at 200M).
- **Memory at 200M**: 160 to 170 bytes a key on the 1M to 3M uniform sets here, so about 33 GB for the index, on top of
  GRE's own array, the authors' copy of keys and payloads during the bulk load (16 bytes a key) and, only if the side store
  takes a key, the wrapper's copy of the remaining pairs (16 bytes a key). Not measured at 200M.
- **Removes and puts mixed** are exact, but not a clean measurement of SALI: the authors' rebuild fails there now and
  then and the wrapper replaces their object (deviation 11), which costs a scan and a bulk load of everything at that put
  and leaves the old object's bytes allocated. GRE has no such run. One hole is left **on arm64 only** (where the index
  is tested, not timed): if the subtree being rebuilt at the 0 / 0 holds exactly three keys, the cast gives a node of 0
  slots, the authors' check at `sali.h:2435` passes (the model of three keys has a positive slope whatever the slot
  count) and their code writes a slot of an empty allocation. It needs a three-key subtree that was itself built by a
  rebuild that did not grow, grew past 64 keys under one of its children and shrank back to two; no test reached it
  (ASan would report it), and on x86-64 the same case throws like every other and is repaired.
- One thread only. The wrapper's counters and side store are not protected; the authors' index would be.
- `sali`'s `put` is slower than the authors' unguarded insert by one `exists()` walk; `saliu` is not. For keys from 2^60 up
  both also run one authors' range query of length 1 per put (the neighbour test of deviation 4), which on data sets that
  live up there (osm, planet, covid) is paid by every insert: about 45 to 50% on top of the authors' insert at 2M keys
  (deviation 4). **This is a handicap for SALI in any table of insert throughput on those sets**: the table should say
  so, or give the number of a build with `-DGRET_SALI_SHIFT=61` (covid is then free of it; the rule still exceeds the
  derived distance) or `=63` next to it, after checking that such a build loads the set (its `bulk_load` looks every key
  up and throws if one is missing).
- Lookups of keys that are not stored cost microseconds (the throw). GRE has none; a workload with negative lookups would
  measure the C++ runtime, not SALI.
- Memory after inserts depends on when the epoch last advanced (deviation 7), and SALI is not deterministic: its rebuild
  decisions use the wall clock and time-seeded generators, so two runs of one insert workload build different trees.
- After removes, scans walk the emptied nodes (the authors' `remove` frees nothing): a scan of an index emptied by removes
  takes as long as a scan of the full one.
- No read-evolving and no compression (above).
- Timings from a build with the TBB stand-ins (the Mac) are meaningless.
- x86-64 with real TBB was exercised on the cluster's login node only, without sanitizers there; the sanitized x86-64 run
  is the one under Rosetta, with the stand-ins.

## Registry lines (not applied: `registry.h` is not this baseline's file)

    #ifdef GRET_SOTA_SALI  // also needs SALI/src/core on the include path, TBB and OpenMP (GRE's build has both); one translation unit only
    #include "gret/sota/sali.h"
    #endif
    ...
    #ifdef GRET_SOTA_SALI
        if (indexInterface<K, P>* i = make_sali<K, P>(name)) return i;  // sali, saliu: the SALI authors' code, guarded (sota/sali.h)
    #endif

Build with `-DGRET_SOTA -DGRET_SOTA_SALI -I /scratch/lvasseur/readonly_competitors/SALI/src/core`.
