# afli, afli_cut, nfl_afli: AFLI (the index of NFL) behind GRE's interface, alone and behind our reimplementation of NFL's flow

`include/gret/sota/afli.h`, `tests/sota/test_afli.cpp`. Three families of names:

| name | what it is | provenance |
|---|---|---|
| `afli`, `afli<B>` | AFLI alone, on GRE's keys, in key order | **the NFL authors' own code, unmodified, behind a guarded wrapper** |
| `afli_cut`, `afli<B>_cut` | AFLI alone, keyed by key mod 10^15 as the authors' own data preparation cuts uint64 keys. **No key order: a point-lookup index, scan returns 0** | the same code and wrapper; only the coordinate differs |
| `nfl_afli[<B>][_cut][_on]` | NFL as far as it can be built without Intel MKL | **a HYBRID: this repository's reimplementation of NFL's flow in front of the authors' AFLI. It is not the authors' NFL** |

`<B>` is one digit in [1, 6]: AFLI's maximum bucket size (without it: the size the authors' code derives from the keys).
`nfl_afli` needs a weight file (`$TRANSFORM_NFL`) and the authors ship weights for **two** of our ten data sets: **books and
fb**. On the other eight it cannot run (see "The hybrid" below).

## What the method is

NFL (Wu, Cui, Yu, Sun, Kuo, Xue: "NFL: Robust Learned Index via Distribution Transformation", PVLDB 15(10), 2022) has two
stages. A small normalizing flow (two layers, two hidden units, trained per data set) maps every key to a new key so that the
mapped keys are close to uniform; a learned index, AFLI (After-Flow Learned Index), is then built on the mapped keys. AFLI is a
tree of nodes with one linear model each: the model gives a key its exact slot, a slot that several keys want holds a small
unsorted bucket (at most 6 keys), a slot that more keys want holds a child node, and keys too close for any model are kept in a
small sorted array (a "dense node"). There is no search inside a model node. NFL decides at the bulk load whether the flow pays
(it compares the "tail conflict degree", the 99th percentile of keys per slot, with and without it) and runs AFLI on the plain
keys if it does not.

## Where the code comes from

| | |
|---|---|
| Repository | `https://github.com/luffy06/NFL` (the paper's artifact; README contact is the first author's address) |
| Commit | `fe71172c326e2a0abb7e31d41b8383ff04672796` (2023-11-01, "update scripts", the head of the default branch) |
| Licence | **GPL-3.0** (`LICENSE`). Included by path only; nothing of it is copied into this repository |
| Local clone | `/Users/louisvasseur/Downloads/readonly_competitors/NFL` |
| On the cluster | `/scratch/lvasseur/readonly_competitors/NFL` (the same files, without `.git`) |
| Files used, unmodified | `src/afli/afli.h`, `afli_nodes.h`, `buckets.h`, `conflicts.h`, `iterator.h`; `src/models/linear_model.h`; `src/util/common.h` (helpers; everything the first six include) |
| Files read, not compiled | `src/nfl/nfl.h` (the switch and how NFL calls AFLI: reproduced in the wrapper), `src/benchmark/benchmark.h`, `src/benchmark.cc`, `src/util/format_data.cc`, `train/numerical_flow.py` |
| Files that cannot be used | `src/models/bnaf.h`, `src/models/numerical_flow.h`: the authors' flow inference, written on Intel MKL (`cblas_dgemm`, `vdTanh`). MKL is installed neither here nor on the cluster |
| Not run | `scripts/bootstrap.sh` (downloads and builds other projects) |
| Shim, **ours** | `/Users/louisvasseur/Downloads/readonly_competitors/NFL_shim/boost/optional.hpp` (cluster: `/scratch/lvasseur/readonly_competitors/NFL_shim`) |

**The shim.** `src/util/common.h:8` says `#include <boost/optional.hpp>`, and Boost is not installed. Nothing in NFL's `src/`
uses Boost (`grep -rn boost src` finds that one line). The shim directory holds one file, `boost/optional.hpp`, whose content
is a comment saying it is not Boost, `#pragma once` and `#include <optional>`. It is not part of NFL and not part of this
repository; it has to be on the include path of a build without Boost and must not be on the path of a program that uses the
real Boost.Optional. `afli.h` stops with an `#error` that says this if the header does not resolve.

The index says what it is every time it is built, in one line on stderr:

    gret: afli [the NFL authors' AFLI, wrapped] n=... in_afli=... side=... (same_coordinate=... far_slot=...) repaired=... coordinate=key slot_bytes=16|24 bucket=3(AFLI::compute_bucket_size) bytes_per_key=... build_ms=... of_which_check_ms=... lookups=the authors' find
    gret: nfl_afli_cut [HYBRID, not the authors' NFL: gret's reimplementation of NFL's flow (gret/nflflow.h) in front of the NFL authors' AFLI, wrapped] n=... ... flow=on|off (NFL's switch: tail conflicts 386 without, 4 with; its decision) weights=<file>

## How it was built

1. **The tree** is the authors' `nfl::AFLI<double, V>`. Their headers are included inside `namespace gret::afli_authors`, after
   every standard header they include has been included at global scope; the macros and include guards they leave are removed
   again (`likely`, `unlikely`, `BIT_TYPE`, ..., `COMMON_H`, ...). `common.h` defines three non-inline functions
   (`start_with`, `get_workload_name`, `path_join`): **include `afli.h` in one translation unit of a program**, which is what
   GRE's benchmark is.
2. **Key type.** The authors' benchmark has one key type, `float64` (`src/benchmark.cc:20`; the paper's setup says the same),
   and their data preparation turns every data set into doubles. AFLI with an integer key type is not usable: the fallback of
   the model builder is `model->slope_ = size / (max_key - min_key)` (`conflicts.h:82`), with integers a division that gives
   slope 0, every key in one slot and a child built from the same keys again (read in the code, not run). So the tree is keyed
   by the key **as a double**:
   - if every bulk-loaded key is below 2^53, each key is its own double and the tree is `AFLI<double, payload>`: 16-byte
     slots, exactly the authors' layout;
   - otherwise the tree is `AFLI<double, (key, payload)>`, 24-byte slots, the layout of NFL's own transformed index
     (`nfl.h:23`), and every hit is compared with the stored key. Two keys with the same double cannot both be in the tree:
     the second goes to the side store.
3. **Side store** (`std::map`, asked only when the tree misses and the store is not empty) for the keys the tree cannot hold
   exactly: a key whose coordinate (its double, or the flow's output) another key already has (`same_coordinate`), a key
   whose predicted slot leaves (-2^62, 2^62) at a node on its path (`far_slot`; the authors cast that double to `int64`
   without a check, `linear_model.h:17`), and, in a tree of 16-byte slots, a key from 2^53 up that arrives by `put`.
   **A run in which `side` is not a negligible share of `n` is not a measurement of AFLI.** At 200M keys with values up to
   2^64, doubles are 2048 apart at the top: keys spread over the range almost never collide (the test's `wide` set, 2.1M
   random 64-bit keys: none but the pair put there on purpose), but **runs of consecutive ids above 2^53 do** (the test's
   `clusters64`: 98% in the side store). Read the line on osm, wise, planet and the other sets with dense large keys before
   using a number. Where `side` is large, the AFLI number for that data set is `afli_cut`'s (next section), which has no
   such collisions, and it has to be reported as what it is.
4. **A check of every key after the bulk load and of every node the authors' insert rebuilds.** `build_linear_model` computes
   the first key's slot with the fitted model, then, if all keys fell into one slot, replaces the model and keeps that slot
   (`conflicts.h:75-86`): the first key of a node can sit where `find` does not look. This happens with keys one double apart
   (2^63 + 2048 i). The wrapper looks every key up with the authors' `find` and with its own walk; a key that is not found is
   taken out of its slot and put back through the authors' insert (`repaired`). Cost: about a third of the bulk load (2.1M
   keys: 52 of 149 ms), printed as `of_which_check_ms`. The walk also counts: the keys found must be the keys given (at the
   bulk load) or the size the authors' `build` recorded for the node (after a rebuild); otherwise the index throws. The
   count is there for the mirror image of the defect: were the first key's stale slot ever *above* the slot of the next
   key, the builder would emit that slot twice and the later run of keys would overwrite the first key, which would then be
   gone and not merely misplaced. The verifier's fuzz of `build_linear_model` (30.3M nodes of 2 to 10 keys, one to six
   doubles apart, at 2^53 .. 2^64, half on arm64 and half on x86-64) found the first key's slot wrong 280,351 times, always
   below, and a slot emitted twice never; the count turns that case, should data ever produce it, into a loud error.
5. **The bucket size** is the authors' rule (`AFLI::compute_bucket_size`, `afli.h:77-84`: the 99th percentile of keys per
   slot under one model over all keys, minus one, clamped to [1, 6]), computed by a copy of their `compute_tail_conflicts`
   with the null check theirs lacks (`conflicts.h:114` dereferences a null pointer when no model can be fitted: one key, or
   fewer keys than 2.2e-16 times their range), then passed to the authors' `bulk_load`. `aggregate_size` is the authors'
   default, 0.
6. **What each GRE call runs.**

   | GRE | what runs |
   |---|---|
   | `bulk_load` | the authors' `AFLI::bulk_load`, then the checks of 4 and 7 |
   | `get` | the authors' `TNode::find` from the root (all `AFLI::find` does, `afli.h:39-41`). Added in front: one test of the mode, one of the key against the range of the keys the tree has held; behind, with 24-byte slots, the comparison of the stored key. On a miss, the side store if it is not empty |
   | `put` | the wrapper walks the authors' nodes to the slot (duplicate? coordinate taken? slot in range?), then the authors' `TNode::insert` on that node; if the insert rebuilt a node, check 4 on it |
   | `update` | the authors' `find`, then the payload is overwritten in place. Their `update()` is not used: it clamps the slot with `size_` where `find` uses `capacity_` (`afli_nodes.h:135`), so it misses keys |
   | `remove` | the wrapper's walk, then what the paper describes (clear the slot's bit, or close the gap in the bucket or the sorted array). Their `remove()` is not used: same clamp (`afli_nodes.h:164`), and in a dense node it drops the last key as well (`afli_nodes.h:188-197`) |
   | `scan` | **the wrapper's**, an in-order walk over the authors' nodes merged with the side store. AFLI has no range scan. Only where the tree is in key order (`afli`; `nfl_afli` when the flow is off and nothing is cut; never `afli_cut` once a key reaches 10^15); otherwise 0. **Do not report scan numbers as the authors'** |
   | `memory_consumption` | every node with all its slots and both bitmaps, the models, every bucket at its allocated capacity, the flow's weights, the side store, the object. Not the authors' `index_size()`, which counts a child shared by several slots once per slot (`afli.h:130`) |

7. **A guard on the authors' cast for keys that are not stored.** `find` computes `static_cast<int64_t>(floor(slope * key +
   intercept))` with no range check. For a stored key the wrapper knows the value is in range (3). For a key that is not
   stored it can be out of range where a node is steep and the key is far from the node's keys: a few consecutive keys under
   a slot of the parent that is 2^62 wide. The earlier draft of this wrapper let that through; UBSan then reports

       NFL/src/models/linear_model.h:17:33: runtime error: 1.84467e+19 is outside the range of representable values of type 'long'

   after six calls on an empty index (`afli2`: put 2^20, put 2^64-1, remove 2^64-1, put 2^20+1, put 2^20+2, put 0, then
   get 2^64-100001). The answer was still "not found" (the cast's result is clamped into the node and the key compared), but
   it is undefined behaviour in the authors' code reached through ours. Now the wrapper shows, at the bulk load, after every
   rebuild and whenever the key range grows, that no key `get` can hand to `find` is cast out of range on its way (an interval
   walk over the models, `casts_in_range` in the header). Where it cannot show it, lookups walk the authors' nodes in the
   wrapper, which checks the range first, **instead of calling the authors' find**; a line on stderr says so at that moment and
   every later log line says `lookups=the wrapper's walk`. **Does real data reach it? No**: it needs a tree whose slots are
   about 2^60 keys wide, which means a tree built from a handful of keys, or a largest key that was removed and lay 2^62
   above the rest. The test demands `lookups=the authors' find` on every ordinary set, and three made-up trees check both
   directions. If a 200M run ever prints `lookups=the wrapper's walk`, its lookup time is not AFLI's.

## `afli_cut`: AFLI on the keys as the authors prepare them

**What it is.** The authors' benchmark never sees a uint64 key. `src/util/format_data.cc:60-66` keeps the **last 15 decimal
digits** of every key of a uint64 data set (`cut_key`: the last 15 characters of the decimal string, i.e. key mod 10^15),
sorts, removes duplicates and writes doubles; every index of their paper, AFLI and the baselines alike, runs on those.
`afli_cut` does the same to the coordinate: the authors' AFLI is keyed by `double(key mod 10^15)` (exact: below 2^53), the
slot holds (key, payload), and every hit is compared with the key. No flow, no weights, any data set.

**Why it exists.** `afli` keeps GRE's keys in order, as doubles. That is the honest order-preserving reading of the authors'
code, but where large keys sit closer than a double apart it cannot hold them, and they go to the side store (3 above).
Under the cut such keys differ (consecutive ids differ in their last digits): the test's `clusters64` set has 98.4% of its
keys in the side store under `afli` and none under `afli_cut`; `hugegap` 50% and none.

**What it is not.** It is **not an ordered index**: key mod 10^15 folds the key space (2^64 is 18,446 times 10^15), so the
tree is not in key order, `scan` returns 0, and a data set whose difficulty is its large-scale shape (fb's few huge keys,
osm's clusters) is folded into something much closer to uniform. That is how the paper's own numbers were obtained (its
AMZN and FB are the cut books and fb), and it flatters AFLI against indexes that keep order. Report `afli_cut` numbers as
"AFLI on keys cut to 15 digits as in its authors' benchmark, point lookups only", never as a range index.

**Details.** If no bulk-loaded key reaches 10^15 the cut changes nothing and is left out: the index is then exactly `afli`
(16-byte slots, scan works) and the log line says `_cut: no key reaches 10^15, nothing to cut`. Two keys with the same
last 15 digits cannot both be in the tree; the second goes to the side store (the authors drop such duplicates from the
data set instead). Against the authors' layout the slot is 24 bytes, not 16, because the key has to be kept for the
comparison. A lookup costs the modulo more than `afli`'s: 3M random 64-bit keys on the Mac, 76 to 78 ns against 65 to 68 ns
with the same tree size (50.4 bytes a key). An index never bulk-loaded (or loaded with no key) is built by `put` on the
keys themselves, whatever the name. The build line says what the index is:

    gret: afli_cut [the NFL authors' AFLI, wrapped, keyed by key mod 10^15 as the authors' data preparation cuts uint64 keys: no key order, no scan] n=... coordinate=key_mod_1e15 slot_bytes=24 ...

## The hybrid: `nfl_afli`

**What is ours and what is theirs.** Theirs: AFLI (as above), the weight files, the switch rule and the way NFL calls AFLI
(reproduced from `nfl.h`, not compiled from it, because `nfl.h` includes the MKL code). Ours: the flow inference,
`include/gret/nflflow.h`, written from the published model and the file format the authors' trainer writes
(`train/numerical_flow.py`, `save_weights`); none of their flow code is used. It computes, for a key k as a double:

    x = (k - mean) / var;  a = [x, x - floor(x)];  h = tanh(a W0);  o = h W1;  z = o0 + o1        (the shipped shape, "2D2H2L")

**How the weights are loaded.** From the file `$TRANSFORM_NFL` names, at every `bulk_load`: text, `in_dim hidden layers`, then
`mean var`, then each layer's matrix as `rows cols` and the values. A missing variable or file is a `std::runtime_error`,
not a silent fallback. In GRE the registry's `name@VAR=value` form sets it per index, for example

    nfl_afli_cut@TRANSFORM_NFL=/scratch/lvasseur/readonly_competitors/NFL/flow_weights/fb-200M-100R-zipf_2D2H2L_weights.txt

**Which data sets.** `flow_weights/` holds weights for books, fb, lognormal, longitudes, longlat, wiki-ts and ycsb, four
files each (`-0R-`, `-20R-`, `-80R-`, `-100R-`: the read share of the authors' workloads). For books the four files are
byte-identical, and so are the four for fb, so any of them does. Of our ten data sets that is **books and fb**. For covid,
genome, history, libio, osm, planet, stack and wise there are no weights: `nfl_afli` cannot run on them. Training new ones
needs the authors' PyTorch trainer (not installed, and no installs are allowed); it would also be our training, not theirs.

**What to expect, from the authors' own paper.** Its table of tail conflict degrees gives books (AMZN) 4 without the flow and
4 with it, and fb 386 without and 4 with; the text says NFL does not use the flow on YCSB, AMZN and WIKI. So **on books the
authors' own NFL is AFLI with the flow off**, and `nfl_afli` should print `flow=off` there; fb is the data set where the flow
matters. If fb prints `flow=off` or a large `side`, the weights do not fit GRE's fb keys (below).

**The keys the weights were trained on (`_cut`).** The authors do not index SOSD's uint64 keys. `format_data.cc:60-66` first
keeps the **last 15 decimal digits** of every key (key mod 10^15, so that it fits a double), removes duplicates, and writes
doubles; the flow is trained on those and the benchmark indexes those. The file headers confirm it (books: mean 0, var
999999984.77 with the trainer's 10^6 shifts, a range of 10^15). GRE keeps the raw keys, so:

- `nfl_afli_cut` feeds the flow (and NFL's switch) key mod 10^15, as the authors' pipeline would, stores (key, payload) and
  compares the key on every hit; two keys with the same last 15 digits cannot both be in the tree, the second goes to the
  side store. If no bulk-loaded key reaches 10^15 the cut changes nothing and is left out (the line says so). **Use `_cut`
  with the shipped weights.**
- `nfl_afli` feeds the flow the raw key: right only for weights trained on raw keys.
- `_on` keeps the flow whatever the switch says (to measure the flow where NFL would not use it).

**The switch** (`nfl.h:71-91`): tail conflicts of the keys as they are and behind the flow, both with amplification 1.5; the
flow is kept only if it lowers them by a tenth or more. The bucket size is then the tail conflict count of the choice
(`nfl.h:93-105`). Where a tail is undefined the authors' code stops on a null pointer; here the flow is then off.

**Not verified.** That GRE's books and fb files hold the same keys as SOSD's `books_200M_uint64` and `fb_200M_uint64`, which
the authors' scripts download. No test here ran on the 200M files. The bulk-load line prints the tail conflicts with and
without the flow; for fb compare them with the paper's 386 and 4.

## Deviations

From the paper, in the authors' released code (not ours; we run the code):

- The switch keeps the flow only for a reduction of 10% or more (`kConflictsDecay`, `nfl.h:26, 80-82`); the paper's text
  drops it only when the tail conflict degree gets larger.
- One kind of bucket, the unsorted one. The paper offers a second, ordered kind.
- A dense node is a sorted array with its free slots at the end, shifted on insert (`afli_nodes.h:250-259`); the paper
  describes a gapped array.
- `update` and `remove` are in the code but wrong (6 above); the paper's description of them is what the wrapper does.

From the authors' code and benchmark, in the wrapper:

1. Keys are uint64 held as doubles; 24-byte slots and a key comparison when a key reaches 2^53 (2 above).
2. A side store, a check after every build, repairs (3, 4).
3. `put` returns false for a key that is there (the authors' insert stores a second copy) and costs a walk more than theirs.
4. `update`, `remove`, `scan`, `memory_consumption` are the wrapper's (6).
5. The bucket size has a null check; the cast has a guard that can take lookups away from the authors' find (5, 7).
6. Asserts: the authors' CMake does not define `NDEBUG`, GRE's does. The headers used have no `assert` on a lookup or insert
   path (`assert_p` in `bulk_load` is a plain `if`), so nothing changes.
7. macOS only: the authors write `std::max(model_->predict(key), 0L)`, which does not compile where `int64_t` is
   `long long`. While their headers are read, and only on macOS, `int64_t` is defined to `long`. Linux is untouched.

In `afli_cut`, besides all of the above:

- The authors index the cut keys themselves, a different key set with duplicates removed and 16-byte slots. `afli_cut`
  indexes GRE's raw keys **by** their cut value: 24-byte slots, the key compared on every hit, keys with equal last 15
  digits in the side store. The bucket size is still the authors' rule, computed on the cut values.

In the hybrid, besides all of the above:

8. **The flow is our reimplementation, scalar, one key at a time**, with libm's `tanh` in place of MKL's `vdTanh`: the
   transformed keys can differ from the authors' in the last bits, and so can the slot of a key. Exactness is not affected
   (build and lookup use the same function, compiled without a*b+c contraction and out of line).
9. **The flow's output is rescaled** linearly to an integer in [0, 2^52) over the range the bulk-loaded keys produce, and
   that integer, as a double, is the tree's key. The authors feed the raw output. Reason: their key comparison is
   `fabs(a - b) < 2.2e-16` (`common.h:54-61`), which on raw outputs near zero would take two different keys for one; on
   integers it is exact. A linear rescale does not change what a linear model does with the keys, up to rounding. Keys with
   equal integers go to the side store; a key put later whose output is outside the bulk-loaded range clamps to an end and
   goes there too.
10. The authors index the cut keys themselves (a different key set, duplicates removed). The hybrid indexes GRE's raw keys
    by their cut value and keeps the key next to it.

## The cost of a per-key `get`, stated plainly

- **AFLI.** The authors' benchmark groups requests in batches of 256, but for AFLI the batch is only the unit of timing:
  inside it `afli.find` is called once per key (`benchmark.h:487`). GRE's per-key call is how AFLI is meant to be called;
  nothing is lost.
- **NFL.** Here the batch is real: the authors transform a whole batch with one MKL matrix product and one vector `tanh`
  (`nfl.transform`, `benchmark.h:552`), then look the keys up one by one. Their paper's table of transformation latency for
  this flow gives 169.5 ns a key at batch 1, 40.6 at 8, 8.4 at 256. The hybrid transforms **one key per `get`**, scalar:
  36 ns a key on an Apple M-series core (the test's `--time`), the price of two `tanh` calls. The machines differ (theirs is
  an i7-10700), so only the order of magnitude compares: per lookup the hybrid's flow costs several times what the authors'
  batched flow costs at batch 256, and a fraction of what theirs costs at batch 1. **`nfl_afli` lookup times are an upper
  bound on what NFL's design gives with batching, measured on a hybrid; they are not the authors' NFL.** A flow that is off
  costs nothing.
- **Where the hybrid's lookup time goes** (the verifier's breakdown; 3M skewed 15-digit keys, the books weights, flow on,
  Mac `-O3`, 2M random hits). The authors' `find` on flow outputs computed beforehand: **42 ns**. The flow alone, one key
  after the other: **30 ns**. The flow and then the authors' `find`, per key, called directly with no wrapper: **176 ns**.
  The same through GRE's virtual `get`: **235 ns**. The sum 42 + 30 is not reached because a per-key flow puts two `tanh`
  calls between consecutive lookups, and the processor can then no longer overlap their cache misses; a leaner `get` for
  this case was tried and changed nothing (236 ns). With the authors' batching the transformation is done before the
  lookups and costs 8.4 ns a key in their table: **their design would be near 50 ns here, the per-key hybrid is at 235.**
  Quote `nfl_afli` lookup times only with that sentence next to them.
- **The wrapper.** Same keys, same loop, both behind GRE's virtual `get`, 3M uniform keys on the Mac (not a benchmark): the
  authors' `find` with nothing around it 54 to 58 ns, `afli` 57 to 62 ns: **+2 to +7 ns, 5 ns in the middle (about 8%)** for
  the mode and range tests (24-byte slots: +3 to +8 ns with the key comparison). A lookup here is two or three cache misses that overlap with
  those of the next lookups, so each added instruction costs more than its own time; the fast path was cut to the tests
  that exactness needs. `put`: 180 to 190 ns against 135 to 150 ns for the authors' `insert` (**+25 to +35%**): the wrapper's walk,
  and the check of each node the insert rebuilds (the first-key defect of 4 does occur there: 413 repairs in one test set on
  x86-64). **Insert throughput of `afli` is below the authors' unguarded insert by that much.**
- The login node of the cluster is CPU-throttled (the same `--time` run shows 600 ns a lookup for every index, GRE's ALEX
  and LIPP included), so no timing was taken there.

## Knobs and the sweep

- `afli`: the authors' rule picks the bucket size per data set. It is the configuration of their benchmark and the one to
  report.
- `afli1` ... `afli6`: the bucket size forced. Smaller buckets mean more child nodes (deeper, larger: 3M uniform keys, 49.8
  bytes a key at 1 against 35.1 at the rule's 3) and faster hits in a slot; larger ones the reverse. The paper fixes the
  ceiling at 6 and has no other knob for AFLI.
- Recommended per data set: `afli`, `afli1`, `afli3`, `afli6`, report the best and say which; on **fb and books** also
  `nfl_afli_cut@TRANSFORM_NFL=<weights>` (`afli_cut` beside it is its flow-off reference: the same coordinate without
  the flow).
- `afli_cut` (and `afli1_cut`, `afli3_cut`, `afli6_cut`): on every data set whose keys reach 10^15, as a second row,
  labelled "keys cut to 15 digits as in the authors' benchmark, no key order". It is the only AFLI row to use where
  `afli` prints a `side` that is not negligible. On a data set whose keys all stay below 10^15 it is `afli`: leave it out.
- `$TRANSFORM_NFL` is read at `bulk_load`. `nfl_afli` on a data set without weights throws; leave it out of that sweep.

## Build and registry

`afli.h` is not in `registry.h` yet. It needs, like DILI, a define of its own because of its include paths and its licence:

    -DGRET_SOTA -DGRET_SOTA_AFLI -isystem <readonly_competitors>/NFL/src -isystem <readonly_competitors>/NFL_shim

(the shim only where Boost is missing). The include goes with the other `GRET_SOTA` headers, and the factory line goes into
the `GRET_SOTA` block of `make_index`, which is before the registry's own `<transform>_<base>` prefixes: `nfl_<base>` is
one of them and would otherwise claim `nfl_afli`.

    #ifdef GRET_SOTA_AFLI  // NFL/src (GPL-3.0) and, without Boost, NFL_shim on the include path; one translation unit only
    #include "gret/sota/afli.h"
    #endif
    ...
    #ifdef GRET_SOTA_AFLI
        if (indexInterface<K, P>* i = make_afli<K, P>(name)) return i;  // afli[<B>][_cut]: the NFL authors' AFLI, guarded (_cut: keyed by key mod 10^15, no scan); nfl_afli[<B>][_cut][_on]: HYBRID, our flow + their AFLI
    #endif

## Tests

`tests/sota/test_afli.cpp`; build lines at its top (`-I <NFL_shim> -I <NFL>/src`). Every answer is checked against a
`std::map`: get on every stored key and on keys never stored (both neighbours, 0, 1, 2^53 and its neighbours, 2^63,
2^64 - 1, keys sharing a stored key's double or last 15 digits, 20,000 random), scans from stored, neighbouring, in-between
and extreme starts with lengths 1 to 70,000 and one scan over everything (where the index keeps order; otherwise scan must
return 0 and write nothing), then put (neighbours, runs, anywhere, far outside, duplicates), update, remove, every key removed,
the emptied index filled again, with the same checks after each step. Key sets: test_scan's patterns (tiny sets, uniform,
wide, a dense run, clusters, far outliers, keys 0 and 2^64 - 1, a regular stride) from 0 to 2.1M keys, a 2.1M random set
at each key width, and sets made for what the authors' code cannot do: keys that share a double, keys one double apart at
2^63 (the first-key defect), sets too sparse for any model, an index never bulk-loaded, the three cast-guard trees. The
bucket knob at 1, 2, 3, 6. `afli_cut` on the sets where `afli` fills the side store (`clusters64`, `hugegap`), on wide random
keys up to 2.1M, on keys one double apart at 2^63, on pairs that share their last 15 digits (half must be in the side
store), on keys below 10^15 (it must then be `afli`: 16-byte slots, scan works) and never bulk-loaded. `nfl_afli` on weight files the test writes (a monotone flow, a folding one) and on the authors'
books and fb weights, with keys of the trained domain and raw 64-bit keys, with and without `_cut` and `_on`; a missing
weight file must throw. The -O3 build replaces `operator new` and compares `memory_consumption()` with the bytes really held:
it must agree within 0.2% plus 256 bytes (the name string and the flow's vector headers are not counted), and does on every
set. 74 combinations of name and key set and the three cast-guard trees; each must say `ok`.

| where | result |
|---|---|
| Mac arm64, Apple clang, `-O1` ASan + UBSan, no ignore list | **passes**, about 6 minutes; no report from either sanitizer, the authors' code included |
| Mac arm64, Apple clang, `-O3` | **passes**, about 100 s with `--time` |
| Jed login node, x86-64, g++ 11.4.1, `-O3 -march=native` (`/scratch/lvasseur/wf/wf_cxx`) | **passes**. The login node is throttled, so the sets are run in seven invocations, each under 5 minutes: `t tiny uniform buckets`, `t wide run clusters clusters64`, `t outliers over64outliers stride hugegap highmag`, `t highrun fblike castguard lognormal`, `t cut`, `t nfl`, `t shipped` (each with `--nfl <NFL>`; 42 to 147 s each) |

**Independent verification** (a second agent, from the files; its scratch sources are in `build/tests/sota/afli/verify/`
and `/scratch/lvasseur/wf/afli_verify/`, not in the repository). The table above is its rerun. On top of it, its own
adversarial program (`adv.cpp`) checks every `get`, `put`, `update`, `remove` and `scan` result against a `std::map` and
`memory_consumption()` against the bytes held **after writes as well as after the load** (within 0.5% plus 512 bytes), on:
28 tiny sets (no key, one key, two keys, 0 and 2^64 - 1 stored and not stored, keys around 2^53 and 2^63, keys equal mod
2^32, keys equal mod 10^15) at four bucket sizes, each loaded, written to, emptied and refilled; 300,000 and 400,000 keys
equal mod 2^32; a dense run and then a gap to the top of the key space; a run of 1M and the key 2^64 - 1, removed, then
appends; runs at the top where 2048 keys share a double; 1M random keys above 2^63; strides of 1024, 2048, 2049, 3000 and
4096 above 2^63; keys just below 2^53 with puts above, then the same object loaded again twice; 5M random 64-bit keys,
4M below 2^50, 3M lognormal; appends past the largest key; 400,000 puts into an index never loaded; `afli_cut` at three
bucket sizes on small sets, 1M and 5M wide keys, runs, strides and triples equal mod 10^15; `nfl_afli`, `_cut`, `_on` on
the authors' fb and books weights with 15-digit and raw keys. Each set is followed by 100,000 to 600,000 interleaved
puts, removes, updates and gets (3,000 to 4,000 on the tiny sets), a third of them next to recently written keys. After each phase: every stored key, the
neighbours (k +- 1, k +- 2, k xor 1024, k xor 2048, k +- 2^32, k +- 10^15, k mod 10^15, the midpoint to the next key) of
up to 400,000 stored keys, 60,000 random keys, and several thousand scans. Result: **no wrong answer and no memory
mismatch** on the Mac under ASan + UBSan (10 minutes, no sanitizer report), on the Mac at `-O3` (3.5 minutes), and with
g++ 11.4.1 on the Jed login node (20 invocations, each under two minutes). On the Mac it was run on the header as the first
agent left it and again on the header as it is now; on Jed in full on the header as it is now.

Also checked: the header in one translation unit with GRE's ALEX, LIPP, PGM and B+tree headers and `gret/registry.h` on the
Mac (`afli`, `afli3`, `ab4`, `alexu` built and queried through one factory; `nfl_afli_cut` reaches this header, not the
registry's `nfl_<base>` prefix, when its line comes first), and on the cluster with GRE's whole `competitor/competitor.h`
and every baseline of the `GRET_SOTA` block, DILI included (`/scratch/lvasseur/wf/afli/coexist.cpp`: compiles, and `afli`,
`afli3` and `rs32r18` answer 100,000 keys alike).

A second, independent check, kept out of the repository (`/scratch/lvasseur/wf/afli/stress.cpp`): 200,000 small trees
grown by random puts, removes and updates with keys of mixed magnitudes (dense runs next to four random anchors, keys one
double apart, the ends of the key space, neighbours of earlier keys), a third of them bulk-loaded first, at bucket sizes 1,
2, 3, 6 and the authors' rule; after every 16 operations every stored key, 40 absent keys and a scan are compared with a
`std::map`. 23.9M operations under ASan + UBSan on the Mac: no wrong answer, no sanitizer report, 36 trees ended on the
wrapper's walk. 100,000 trees with g++ 11 `-O3` on the cluster: no wrong answer.

No sanitizer report had to be hidden. The one defect a sanitizer can see in the authors' files, the cast of 7, is kept
from happening by the wrapper; the others (4, 5, 6) are wrong answers or a null pointer, found by reading and by the test, and
the wrapper does not call the code that has them or checks behind it.

## Known limits

- **Not run at 200M keys.** Memory at the bulk load: the (double, value) array (16 or 24 bytes a key), the authors' conflict
  arrays (8 bytes a key at the root) and the tree (35 to 80 bytes a key on the test sets, 16 on a run or a stride). The
  hybrid holds two such arrays while the switch decides. Measured at 10M and 20M keys on the Mac (`-O3`; both sizes give the
  same per-key figures, so the build is linear): `afli` builds in 65 to 135 ns a key and its peak memory above GRE's own key
  array is 58 to 107 bytes a key (the index itself 35 to 73); `afli_cut` 132 ns a key (it sorts the cut values);
  `nfl_afli_cut` 236 to 420 ns and 110 to 166 bytes a key. At 200M keys that is some 15 to 30 s and **12 to 22 GB** for
  `afli`, about a minute and **22 to 33 GB** for the hybrid, on top of GRE's 3.2 GB of pairs. Counts are `uint32` in the
  authors' code: at most 2^31 - 64 keys (`std::length_error` beyond).
- Keys that share a double, and runs of consecutive keys above 2^53, go to the side store (3). The log line is the check;
  `afli_cut` is the variant without that limit, and without key order.
- In a tree of 16-byte slots (no bulk-loaded key from 2^53 up), a later `put` of a key from 2^53 up goes to the side store.
  `memory_consumption` prints how many puts went there.
- One translation unit per program; not thread-safe (`scan`, `put`, `remove` and, once lookups are on the wrapper's walk,
  `get` use scratch vectors of the object); one index per object, a second `bulk_load` rebuilds it.
- **Keys appended past the largest key (or before the smallest) are AFLI's worst case, in the authors' code itself.** A
  key beyond a node's range goes to the node's last slot; when that slot's bucket is full it becomes a child node over
  just those few keys, and the next keys beyond go to *its* last slot. The tree grows by one level for every bucket of
  appended keys and nothing in the released code (or the paper) rebuilds it. Measured on the authors' AFLI called
  directly, 100,000 keys loaded and then keys appended in ascending order (bucket size 1 from their rule): after 40,000
  appends the last key is 40,001 levels deep, an insert costs 192 microseconds and a lookup of the newest keys 575
  microseconds; behind the wrapper the same inserts cost 465 microseconds (its walk and its check of each rebuilt node are
  also proportional to the depth), 71 with `afli6`. The authors' destructor and `find` recurse once per level. GRE's
  insert workloads shuffle the keys before choosing the bulk-loaded part (`GRE/src/benchmark/benchmark.h:135`), so inserts fall inside the loaded range and only
  the few keys beyond its ends chain; **do not run `afli` on a workload that appends in key order** (time series, a sorted
  insert stream).
- Two paths of the cast guard have no test that makes them fail: the range recomputation after a bulk load that sent keys
  to the side store as `far_slot` (no key set here produces one at a bulk load), and the check made when a new largest key
  arrives (it runs on every such put in the test and always passes). Both err towards the wrapper's walk. Nor can the
  count check after a rebuild (4) be made to fail: no key set found makes the authors' builder lose a key; if it ever
  fires, `put` throws and the run stops.
- `nfl_afli`: two data sets; our flow; per-key transformation; weights not checked against GRE's key files (above).
- GPL-3.0: a binary that includes this header links GPL code. Fine for running a benchmark, not for redistribution under
  another licence.
