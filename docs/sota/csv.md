# csv: CSV (distribution smoothing via virtual points) on GRE's LIPP and ALEX

`include/gret/sota/csv.h`, `tests/sota/test_csv.cpp`. Names: `csv_lipp`, `csv_alex` (alpha = 0.1) and `csv_lipp<PP>`,
`csv_alex<PP>` (alpha = PP / 100).

## What it is

CSV (Amarasinghe, Choudhury, Qi, Bailey: "Learned Indexes with Distribution Smoothing via Virtual Points",
arXiv 2408.06134; the text used here is version 3, 13 pages, stamped "15 Dec 2024" by arXiv and "Arvix, Oct, 2025" in its
own page headers) does not propose an index. It takes an
existing hierarchical learned index after it has been built and, for the keys of a subtree, inserts a few *virtual points*
(keys that hold no record) chosen so that a straight line fits the subtree's key-to-rank function better; if a cost condition
says so, the subtree is then rebuilt from the smoothed key set, so that keys which sat several levels down are held by a node
higher up. The paper applies it to ALEX, LIPP and SALI.

**This is our own reimplementation from the paper. The authors published no code** (none is linked from the paper or the
arXiv page; searched November 2026). It is not the original and must not be reported as such. The line `bulk_load` prints
starts with `gret: csv_lipp reimplementation of CSV (arXiv 2408.06134) on GRE's LIPP` (or `... ALEX`).

| | |
|---|---|
| CSV itself | written here from the paper: `csv_detail::smooth` (Algorithm 1), `CsvLipp::visit` and `CsvAlex::visit` (Algorithm 2) |
| base indexes | GRE's own copies of LIPP and ALEX, **their files unmodified, included by path** (`lipp/src/src/core/lipp.h`, `alex/src/src/core/alex.h` under GRE's `src/competitor`, which GRE's build already has on the include path). ALEX is the ALEX authors' repository. LIPP is **the GRE authors' fork** of the LIPP authors' repository: the bulk builder, the lookup and the insert are the LIPP authors'; `range_query_len` (January 2022), `remove` and `update` (May 2022) were added in the fork by the GRE authors. The virtual keys of a rebuilt LIPP subtree are erased with that fork's `remove` |
| GRE | https://github.com/gre4index/GRE, commit `e807edcef51df6732f07f94d4c797fb3897519ba` (2022-11-09); no licence file |
| LIPP | https://github.com/pohchaichon/lipp (GRE's fork of the LIPP authors' repository, a GRE submodule), commit `a1189c3956ebfef1335d252f8e13b91eb99fe8cf` (2022-05-19), MIT |
| ALEX | https://github.com/microsoft/ALEX (a GRE submodule), commit `f4365a73f910f43a71ee1da7924cd51219b42864` (2021-10-07), MIT |
| SALI | the paper's third base. **Not done**: GRE does not ship SALI |

Nothing new was cloned and nothing was copied into this repository. On the Mac, GRE's ALEX carries two portability edits
made earlier by this project's `build_gre.sh` (`ALEX_USE_LZCNT 0`, no CPUID on arm64); the cluster's copy is the upstream
file. The test passes against both.

## What the paper reports, so that a result can be compared

All of the paper's query numbers are **for the promoted keys only** (the keys CSV moved to a higher level), each queried
100 times, on four 200M-key sets (Facebook, Covid, OSM, Genome), single-threaded read-only unless said otherwise.

- Query time on the promoted keys: **up to 34% lower**, the larger gains on LIPP and SALI (Fig. 7a, 7b); on ALEX the
  axis of Fig. 7c ends at 15%. No number is given for a workload over all keys.
- Promoted keys as a share of the *promotable* keys (those at level 3 or below in the original index), LIPP and SALI:
  about 60% on Facebook, about 30% on Covid, up to 27% on OSM, up to 57% on Genome; for ALEX the paper only says the
  ranking of the data sets differs (Fig. 8g, axis to 75%).
- Storage: "in most cases" under 10% more than the original index, under 31% in the worst case; on ALEX under 0.5% in the
  insert experiment.
- Time of the CSV pass at 200M keys (Tables 3 and 4), alpha = 0.1: LIPP 337 s (Covid) to 2,329 s (OSM); ALEX 889 s (Facebook)
  to 2,902 s (Genome). At alpha = 0.8: up to 15,709 s on LIPP and 81,620 s on ALEX.
- Default alpha 0.1, swept over 0.05, 0.1, 0.2, 0.4, 0.8.

The line `bulk_load` prints gives what is needed for the same quantities: `promoted`, `keys_at_level3plus=before->after`
(the promotable keys), `virtual` (and its share of n), `nodes_removed` or `nodes=before->after`, `avg_level=before->after`,
`bytes_per_key`, `csv_ms`. Since the gain is on the promoted keys, the gain over all lookups is at most that times the
promoted share of the keys.

## How it was built

### Algorithm 1 (CDF_smoothing): `csv_detail::smooth`

Input: a sorted key set K and a budget lambda. Each round considers adding one integer k_v strictly between two adjacent
keys, refits the least-squares line rank ~ key over K and k_v (the keys after k_v move up one rank), and takes the k_v with
the least squared error L; it stops after lambda points or when L no longer falls. A point that is taken becomes part of K
(line 31). The candidates are filtered as in the paper's section 4.2: per run of free integers between two adjacent keys,
both ends if the run has one or two integers; otherwise the sign of dL/dk_v at the two ends decides: same sign, the two
ends; opposite signs, the minimum inside the run.

The paper's equations 5 to 21 are written here in centred sums, L = Cyy - Cxy^2 / Cxx, with Cxx a quadratic and Cxy a line
in k_v inside one run. The sums over K are computed once a round and the suffix sum of the keys gives the rank shift, as in
the paper's section 4.1.

**Checked against the paper's own example.** The ten keys of the paper's Fig. 2 (2, 4, 6, 7, 8, 10, 14, 20, 26, 32, read
off the figure's vector coordinates) with alpha = 0.5: this code gives loss 8.327 before, the points 23, 28, 18, 16, 24 in
that order (the paper's k_v1 to k_v5 in Fig. 2b; 23 is the "search target" of its Fig. 3), and loss 2.293 after, which are
the numbers of the paper's Table 2. The test asserts all of it.

### Algorithm 2 (CSV) on LIPP: `CsvLipp`

1. LIPP's own `bulk_load` builds the index on the real keys (the paper applies CSV "to a constructed learned index").
2. A post-order walk visits every node that has a child node, at the levels the paper names (section 6.1): from the second
   level from the bottom that has subtrees up to level 2; the root (level 1) is never rebuilt.
3. For such a node: its subtree's keys are collected, Algorithm 1 adds up to lambda = alpha x (keys in the subtree) virtual
   keys, LIPP's own bulk builder (`build_tree_bulk`, the FMCD one) makes a new subtree of the real and the virtual keys,
   and the virtual keys are erased from it with `LIPP::remove` (the one GRE's fork adds).
4. Cost condition: the new subtree replaces the old one if the real keys sit at a lower total level in it (see
   deviation 1).

### Algorithm 2 on ALEX: `CsvAlex`

1. ALEX's own `bulk_load` builds the index.
2. A post-order walk visits every model node below the root (the paper: on ALEX the pass starts at the bottom level and
   stops below the root).
3. For such a node: the keys of all data nodes below it are collected and smoothed. The paper's equation 22,
   cost = search_constant x expected searches + traversal_constant x level, is computed for the subtree as it is (the mean
   over its keys, with the expected exponential-search iterations ALEX stored in each data node when it built it) and for
   one data node holding the smoothed key set at the model node's level (ALEX's own `build_model` and
   `compute_expected_cost`: "the inbuilt function in ALEX that uses the log2 error").
4. If new cost - old cost < c, the model node and everything below it is replaced by that one data node ("creating a new
   leaf-node in place of the parent node"), built by ALEX's own data-node `bulk_load` with the smoothed key set, the virtual
   keys then erased with ALEX's own `erase_one`. The leaf chain, the parent's child pointers and ALEX's node counts are
   fixed up.

### What a virtual point is here

A key handed to the base's own node builder and then erased with the base's own erase. It ends as an **empty slot that
shaped the model**: `get` and `scan` cannot return one (the test probes every one of them), a lookup pays nothing for
them, and a later `put` can take the slot, which is what the paper says the points are for ("can be used to host data
insertions"; its section 6.3 speaks of "the gaps left by the virtual points"). The paper does not say how its
implementation stores them.

What the two erase functions leave behind, which a native implementation might not:

- LIPP: a two-key node that held one real and one virtual key keeps its real key alone in an 8-slot node, one level below
  the slot it could have had; a node all of whose keys were virtual is freed by `LIPP::remove` only if its last key sat in
  the node itself, otherwise it stays as an empty node (lookups pass through it and find nothing; its bytes are counted).
  The default cost condition (deviation 1) is computed after the erasing, so it sees these levels.
- ALEX: the wrapper sets the new node's `contraction_threshold_` to 0 while it erases and then restores it (60% of the
  slots). If the virtual keys were more than about 17% of the node's real keys (possible from alpha = 0.2 up), the node
  then holds fewer real keys than that threshold, and at the **first `remove`** that hits it ALEX's own code contracts it
  to 80% density: the model keeps its smoothed shape (it is rescaled, not refitted, from 50 keys up), but the room the
  virtual points left shrinks with it. Lookups and inserts do not trigger this.

## Where the public API of the bases was not enough

The paper's procedure rebuilds single subtrees inside the index. Neither base offers that. Their files are not edited;
this is what the wrapper reaches into instead, and each item is a place where our build is not "CSV through the public API":

- **LIPP** keeps its root, its node type and its builder private. Four private members are reached through explicit
  template instantiation, which C++ exempts from access checks: `LIPP::root`, `LIPP::build_tree_bulk`,
  `LIPP::destroy_tree`, and `LIPP::pending_two` (read only, for the size of the pool). The node type is deduced from
  `root`. LIPP's private `PREDICT_POS` is restated in the wrapper (`CsvLipp::slot`, the same expressions). To erase the
  virtual keys of a new subtree with `LIPP::remove`, which starts at the root, the root pointer is set to the new subtree
  for those calls and set back.
- **ALEX**: its nodes, `root_node_`, `stats_`, `params_` and `derived_params_` are public (its private `istats_`, the key
  domain and the two out-of-domain counters, is read through the same explicit instantiation, only by
  `CsvAlex::consistent()`, a check the tests call). The wrapper allocates a data node
  as ALEX does, sets `prev_leaf_` / `next_leaf_`, `duplication_factor_`, `cost_` and the two expected-cost fields, replaces
  the parent's child pointers, frees the old nodes and adjusts `stats_.num_data_nodes` / `num_model_nodes`. It sets the new
  node's `contraction_threshold_` to 0 while the virtual keys are erased (ALEX would otherwise repack the node and undo
  the layout) and restores it.

## Deviations from the paper, and places where the paper leaves a choice

1. **LIPP's cost condition.** The paper: "For indexes that does not contain any searching component such as LIPP and SALI,
   their loss function values can be taken as the cost conditions", because "if the new model could hold more keys than
   before" nothing else can make a lookup slower. Read to the letter, every subtree whose loss fell is rebuilt. But LIPP's
   builder fits its own model (FMCD), not the least-squares line whose loss fell, and a rebuilt subtree can put keys
   deeper. Default (`CSV_LIPP_RULE=levels`): the rebuilt subtree is kept only if its real keys sit at a lower total level
   than before. `CSV_LIPP_RULE=loss` is the letter of the paper. On the test sets the default reaches a lower average
   level with about half the virtual points (3M lognormal keys, alpha 0.1, `CSV_LAMBDA=floor`: 2.1169 with 4.5% virtual
   keys against 2.1233 with 7.7%), so the default favours the baseline.
2. **lambda as a count.** Definition 1 asks for |V| <= lambda = alpha x n, the integer part; Algorithm 1's loop, "while
   V.size < lambda", adds a point while the count is below the real number, which is rounding up. They differ for LIPP: with
   the integer part a subtree of fewer than 1 / alpha keys gets no point, and most LIPP subtrees are that small (3M uniform
   keys, alpha 0.1: no subtree is smoothed at all). Default `CSV_LAMBDA=ceil` (Algorithm 1 as printed; it also reproduces
   the paper's large promoted shares on easy data better); `CSV_LAMBDA=floor` is Definition 1. With `ceil` the virtual keys
   can exceed alpha x n by one per smoothed subtree.
3. **lambda across levels.** The paper does not say whether points accepted in a subtree count against its parent's budget.
   Here they do: a subtree of m real keys holds at most lambda(m) virtual keys in all, and the ones accepted below are part
   of the key set the parent is smoothed with (they are keys of K, as in line 31).
4. **The minimum inside a run.** The paper computes it "by using the two partial derivative values to find their
   intersection with the x-axis". Here it is the exact root of dL/dk_v = 0 (the numerator is linear in k_v), then the two
   integers around it; if the root is not inside the run, the two ends are taken. This can only find an equal or better
   point. On small sets a single point is checked against an exhaustive search over every free integer.
5. **Where the cost condition is tested without a new point.** Algorithm 2 tests it for every visited node (lines 10 to
   13). ALEX: so does the wrapper; a subtree can be merged although Algorithm 1 added nothing. LIPP: a rebuild from the
   same keys is the subtree LIPP already built, so the test is skipped unless virtual keys accepted below change the key
   set.
6. **Order of the pass.** Post-order instead of the paper's level-by-level loop. Each node is handled after everything
   below it, and what happens at a node depends only on its subtree, so the result is the same.
7. **LIPP's start level.** "CSV is run starting at the second level of the index structures" is read as the second level
   from the bottom that has subtrees (the next sentence contrasts it with ALEX's "bottom level"). If the authors meant
   level 2 from the top, their pass visits fewer nodes than ours.
8. **ALEX's cost constants.** The paper says to measure search_constant and traversal_constant by sampling queries and
   gives no values, and says c "should be set below 0" without a value. Here: ALEX's own cost weights, 20 per expected
   search iteration and 20 per level (`CSV_SEARCH_W`, `CSV_LEVEL_W`), and c = 0 (`CSV_COST_C`): any drop of the estimated
   cost merges. The expected search iterations of the merged node are ALEX's estimate over the smoothed key set, virtual
   keys included.
9. **ALEX's node size.** One ALEX data node holds at most 2^20 slots (16 MB) at density 0.7: a subtree with more than about
   734,000 keys and points cannot become one node and is left alone (`too_big`).
10. **Caps for feasibility**, which the paper does not have (its pass ran for up to 22 hours):
    - `CSV_MAX_SUBTREE` (default 131072): larger subtrees are not smoothed. Algorithm 1 costs lambda rounds over m gaps,
      alpha x m^2 in all (the paper's "O(lambda + n)" does not hold for it: every round must look at every run).
    - `CSV_BUDGET_S` (default 1200): before the pass, the cost of every subtree is estimated from a timing of Algorithm 1
      on this machine; if the sum is above the budget, the largest subtrees are left as the base built them
      (`over_budget`). A hard stop at 1.5 x the budget catches a low estimate (`late`). **When the budget binds, which
      subtrees are smoothed depends on the machine's speed**; `over_budget=0 late=0` in the line means it did not.
11. **Guards**, not part of CSV (next section).
12. **Keys**: unique 64-bit unsigned integers; payloads 64-bit.
13. **Candidates the base cannot hold** (LIPP only): Algorithm 1 is given a filter and never proposes a virtual key that
    LIPP's models could not separate from its neighbours (the same rule as for real keys, next section: only keys one
    apart from 2^63 up on x86). The paper has no such rule; below 2^63 the filter passes everything.
14. **Arithmetic of Algorithm 1**: keys are scaled to x = (k - k_0) / (k_last - k_0) in double and the sums of equations 10
    to 21 are kept in double. On a subtree whose keys span more than 2^53, neighbouring candidates can get the same x, and on
    subtrees of 10^5 keys the loss differences between the best candidates are near the rounding error, so the point taken
    can differ from the one exact arithmetic would take. It is always a free integer strictly between two keys, so answers
    are not affected; the paper does not say what arithmetic its implementation uses.
15. **A rebuilt LIPP subtree may again have several levels** (it is whatever LIPP's bulk builder makes of the smoothed
    keys), where the paper's text for the general case speaks of "creating a new leaf-node in place of the parent node".
    On ALEX the subtree does become one data node.

## What the wrappers guard in the bases

A side `std::map` holds the keys a base cannot hold. `get` looks there only on a miss and only when it is not empty. The
line `bulk_load` prints has `side=`; `memory_consumption` prints `in_lipp= / in_alex= side=` again when puts went there.
**A run where `side` is not a negligible share of n is not a measurement of the base.**

- **LIPP**: `LIPP::at` on a key that is not stored can hit `RT_ASSERT(key == key)` and `exit(0)`; `update` and `remove`
  act on whatever key sits in the slot; `insert` of a stored key exits. So `get`, `update`, `remove` and `put` first find
  the key with the wrapper's own walk (`CsvLipp::find`: LIPP::at's loop with a comparison in place of the assert; no
  extra indirection). Two adjacent keys too close for LIPP's long double models cannot share a two-key node: the later one
  goes to the side store. With an 80-bit long double (the cluster) that is keys 1 apart from 2^63 up; where long double is
  a double (arm64) it starts at 2^52. `LIPP::remove` of the last key dereferences a null parent: the wrapper reloads an
  empty index instead.
- **ALEX**: 2^64 - 1 is ALEX's gap sentinel (side store). A key outside the range of the bulk-loaded keys makes ALEX expand
  its root, with int casts that overflow for a far key and an assert that fails for a near one: such keys go to the side
  store, so ALEX's key domain never moves. More than four keys inside a window of a few double steps (only from 2^53 up)
  make `Alex::insert` split one data node without end: the fifth goes to the side store. `scan` does not use
  `Alex::lower_bound` (it can return a slot past a data node, `alex_nodes.h:1525`) but a seek that moves to the next data
  node. While ALEX's root is still a data node (small bulk loads: up to 100,000 keys and more, depending on the keys) the wrapper gives it only keys
  between the first and the last key that node holds, and takes ALEX's key domain when the node splits. That hand-over had
  a defect, **found and fixed by the verifier**: ALEX splits a full root data node before it looks whether the key is
  already there, so the put of a *stored* key can be the call that splits it; the wrapper then returned early without
  taking the new domain, later puts outside the domain reached ALEX, and ALEX's root expansion ended in
  `std::bad_array_new_length` (loud, no wrong answer was seen). The wrapper now takes the domain whatever the call
  returns; `CsvAlex::consistent()` checks ALEX's domain and its two out-of-domain counters against the wrapper's range,
  and the test has the case (`rootsplit`). A 200M-key bulk load never has a data-node root, so benchmark runs were not
  exposed.

`update`, `remove` and `scan` are supported by both (ALEX's interface in GRE has no update; the wrapper overwrites the
payload in place).

## memory_consumption

- `csv_lipp`: every LIPP node with all its slots and both bitmaps in full, the side store, the object. `LIPP::total_size()`,
  which GRE reports for `lipp`, counts one byte per bitmap; both are printed (they differ by under 2%).
  **Not counted: LIPP's pool.** Every LIPP object allocates 10^7 two-key nodes in its constructor and keeps the unused
  ones. GRE's `lipp` number leaves it out too, so the two rows are comparable; the line prints the pool's size so that it
  can be added. The printed size is the requested bytes (210 a node: 2.0 GB); the allocator rounds each node's four small
  blocks up, and what a fresh object really holds was measured at 2.48 GB on macOS (248 bytes a node).
- `csv_alex`: ALEX's `model_size() + data_size()` as GRE counts `alex`, plus the side store and the objects.
- The work arrays of the CSV pass (up to `CSV_MAX_SUBTREE` entries each) are freed when the pass ends. Before the
  verifier's fix they stayed allocated and uncounted (9 MB on a 2M-key `csv_alex`, 77% counted instead of 92%).

Checked against the bytes really held (a replacement `operator new` that sums the allocator's block sizes; 2M keys;
for LIPP the pool is taken out and the two-key nodes the tree took from it are put back in):

| index | set | `memory_consumption` | held by the index | counted |
|---|---|---|---|---|
| `csv_lipp` (Mac) | uniform | 80.8 B/key | 90.1 B/key | 89.6% |
| `csv_lipp` (Mac) | lognormal | 149.7 | 168.0 | 89.1% |
| `csv_lipp0` = plain LIPP (Mac) | uniform / lognormal | 79.8 / 146.7 | 89.3 / 164.0 | 89.5% / 89.4% |
| `csv_lipp` (cluster, glibc) | uniform / lognormal | 85.7 / 152.6 | 98.7 / 160.0 | 86.9% / 95.4% |
| `csv_alex` (Mac) | uniform / lognormal | 23.0 / 24.7 | 23.1 / 26.7 | 99.6% / 92.3% |
| `csv_alex` (cluster, glibc) | uniform / lognormal | 23.0 / 24.7 | 23.0 / 24.7 | 100.0% / 99.9% |
| `csv_alex0` = plain ALEX (Mac) | uniform / lognormal | 23.0 / 24.3 | 23.1 / 27.0 | 99.6% / 89.9% |

What is missing is the allocator's rounding of the bases' small blocks (LIPP allocates each one-byte bitmap of a two-key
node by itself), which GRE's own `lipp` and `alex` numbers leave out in the same proportion: the CSV rows are neither
flattered nor handicapped against them.

## How `include/gret/vpoints.h` differs from the paper (and from this file)

`vpoints.h` (`VirtualIndex`, the V of SPLICE) runs the same Algorithm 1, but nothing else of CSV:

| | the paper (and `csv.h`) | `vpoints.h` |
|---|---|---|
| key sets smoothed | the keys of each subtree of the built index | fixed runs of R = 256 consecutive sorted keys (`TRANSFORM_VP_R`), cut before any index exists |
| hierarchy | bottom-up over the levels, points of a subtree seen by its parent (Algorithm 2) | none: one flat pass |
| cost condition | per subtree: rebuilt only if it holds | none: every point Algorithm 1 adds is kept |
| budget | lambda = alpha x keys of the subtree | floor(alpha x R) per run |
| what a virtual point is | (here) an erased key: an empty slot | a stored key with the reserved payload 2^64 - 2; `get` runs the base's lookup and then refuses that payload; a real payload with that value is refused at the bulk load |
| the base index | rebuilt subtree by subtree, from inside | bulk-loaded once on real and virtual keys through its public API, unchanged; any base works (B-tree and PGM too) |
| effect sought | keys move to higher levels | the base's own models fit better |
| writes, scan | as the base | read-only; `scan` returns 0 |

Also, in `vpoints.h` the interior minimum is computed as k_0 + offset in long double: where long double is a double (arm64)
and keys are above 2^53 the candidate is rounded to the double grid (still a free integer inside the run, so answers stay
exact; only the choice of point is worse). `csv.h` computes the offset from k_0 and does not have this. `vpoints.h` was not
changed.

## Knobs and the recommended sweep

- **In the name: alpha**, the paper's smoothing threshold. `csv_lipp` = `csv_lipp10`, `csv_alex` = `csv_alex10`. PP in
  [0, 99], no leading zero. `csv_lipp0` / `csv_alex0`: no pass at all, the base behind the same guards (the control for
  what the wrapper costs).
- Paper's sweep: 5, 10, 20, 40, 80. The pass costs about alpha times as much, so 80 takes eight times as long as 10.
- Recommended per data set: `csv_lipp10`, `csv_lipp40`, `csv_lipp0`, `csv_alex10`, `csv_alex40`, `csv_alex0`, next to
  GRE's own `lipp` and `alex`.
- **Read CSV's effect against the control (`csv_lipp0`, `csv_alex0`), not against GRE's `lipp` and `alex` rows.** The
  wrapper's own lookup loop for LIPP (`CsvLipp::find`) is a little tighter than `LIPP::at`: in separate processes on the
  Mac, 3M keys, 4M lookups, `csv_lipp0` took 53.0 to 55.0 ns against 55.5 to 58.2 ns for `lipp` (4 to 8% less), with no CSV
  at all. `csv_alex0` and `alex` were equal within the spread (69 to 76 ns against 67 to 77 ns). Not measured on the
  benchmark machine (its login node shares one CPU).
- Before believing a `csv_alex` row, look at its bulk-load line: `candidates`, `too_big`, `accepted` and
  `avg_level=before->after` say whether the pass did anything (see "Known limits"), and `side=` whether the guards moved a
  share of the keys out of ALEX.
- Environment, read at `bulk_load` (GRE's `name@VAR=value` syntax in `registry.h` sets them per index): `CSV_LAMBDA`
  (ceil | floor), `CSV_LIPP_RULE` (levels | loss), `CSV_COST_C` (<= 0), `CSV_SEARCH_W`, `CSV_LEVEL_W`, `CSV_MAX_SUBTREE`,
  `CSV_BUDGET_S`. A malformed value throws `std::invalid_argument` before anything is loaded.

Registry lines (for the `GRET_SOTA` block of `registry.h`; no new include path, no new define):

    #include "gret/sota/csv.h"  // CSV on GRE's LIPP and ALEX: a reimplementation from the paper (arXiv 2408.06134), no authors' code exists
    if (indexInterface<K, P>* i = make_csv<K, P>(name)) return i;  // csv_lipp[<PP>], csv_alex[<PP>]: reimplementation of CSV, alpha = PP / 100

They must come before the `<kind>_<base>` parsing further down (the `GRET_SOTA` block does). Checked: a file that includes
GRE's `competitor.h` (which includes `registry.h`) and then this header compiles.

## What the sanitizers report in the bases (file:line, and whether real data reaches it)

The sanitized test, built as given (no ignore list), **passes for the `smooth`, `lipp` and `lipp2m` groups and stops in
ALEX's own code for every ALEX group** (the `alex` group now stops in its first case, `rootsplit`, at `alex_base.h:79:29`;
before that case existed, at `alex_base.h:164:18`). Nothing is reported in `csv.h` or in LIPP. In ALEX (paths under
`src/competitor/alex/src/src/core/`):

| check | where | what | on real data |
|---|---|---|---|
| float-cast-overflow | `alex_base.h:79:29` | `LinearModel::predict` casts a double outside int to int; the callers clamp the result | yes: any lookup of a key far from a data node's keys (a narrow node next to a far key) |
| signed-integer-overflow | `alex_base.h:164:18` | `extract_rightmost_one`: `-static_cast<int64_t>(value)` when the bitmap word is 2^63; the result is the intended one on two's complement | yes: any data node whose bitmap word has only its top bit set (the path of `ALEX_USE_LZCNT 0`, which the Mac build uses) |
| vptr | `alex_fanout_tree.h:353:15` and on (40 places in `alex_fanout_tree.h` and `alex_nodes.h`) | ALEX casts a data node to `AlexDataNode<T, P>` with the default template arguments, which is another type than the `allow_duplicates = false` node GRE instantiates; same layout, so it works | yes: every node split |
| address (heap-buffer-overflow, read of 8 bytes) | `alex_nodes.h:622`, `Iterator::initialize`, from `bulk_load_from_existing` / `split_sideways` / `insert` | reads the bitmap word after the last one when an iterator starts at the node's end | yes: inserts that split a node whose capacity is a multiple of 64 |

With an ignore list that takes these four off the files under `src/competitor/alex` (the fourth off that one function)
the whole test passes; ASan and every other check stay on ALEX, and everything stays on the wrapper, on LIPP and on the
test. The `printf` that writes the list is at the top of the test. GRE's plain `alex` has the same four.

## Tests

### How to build and run

    B=build/tests/sota/csv; mkdir -p $B
    printf '[float-cast-overflow]\nsrc:*/competitor/alex/*\n[signed-integer-overflow]\nsrc:*/competitor/alex/*\n[vptr]\nsrc:*/competitor/alex/*\n[address]\nfun:*AlexDataNode*Iterator*initialize*\n' > $B/san_alex.ignore
    build/tests/cxx    -fsanitize-ignorelist=$B/san_alex.ignore tests/sota/test_csv.cpp -o $B/t     # ASan + UBSan, ALEX's four known reports off
    build/tests/cxx    tests/sota/test_csv.cpp -o $B/t-plain                                        # as given: run "t-plain smooth lipp lipp2m"
    build/tests/cxx-O3 tests/sota/test_csv.cpp -o $B/t-O3
    $B/t && $B/t-plain smooth lipp lipp2m && $B/t-O3

No extra include directory: `build/tests/cxx` already has GRE's `src/competitor` (where `lipp/` and `alex/` are). On the
cluster login node (after the `mkdir` and the two `rsync` lines of the workflow's rule 5, with `<name>` = `csv`):

    /scratch/lvasseur/wf/wf_cxx csv /scratch/lvasseur/wf/csv/test_csv.cpp -o /scratch/lvasseur/wf/csv/t
    cd /scratch/lvasseur/wf/csv && nice -n 10 ./t smooth alex alex2m modes && nice -n 10 ./t lipp lipp2m

Two processes one after the other, because the login node gives this user 8 GB and one CPU in all (cgroup `user.slice`)
and each half peaks at 3.3 and 3.8 GB; `./t` alone also passes when nothing else is running there.
`t [GROUP...] [set:NAME] [--time]` runs only the named groups (`smooth alex alex2m lipp lipp2m modes`) or one key set;
`t-O3 none --time` prints the small lookup timing.

### What the test checks

`tests/sota/test_csv.cpp`, 70 result lines (68 key-set cases, each against a `std::map`, the Algorithm 1 group and the
root-split case):

- `smooth`: Algorithm 1 alone. The paper's Fig. 2 / Table 2 example (above); lambda as a count; on 400 random key sets (up
  to 3,000 keys, magnitudes to 2^64, dense runs) the result is sorted and unique, keeps every key, flags exactly the
  others, respects lambda and the base's filter, never raises the loss; on 66 small sets a single point is as good as the
  best of all free integers.
- `alex`, `lipp`: the 18 key sets (empty, 1, 2, 3, 5, 17 keys with 0 and 2^64 - 1; uniform; wide over all 64 bits with 0,
  2^63 - 1, 2^63, 2^64 - 1; a dense run; clusters; far outliers; a regular stride; a run up from 0 with a run down from
  2^64 - 1; nested clusters; a staircase of densities; a lognormal) at alpha 0.1, and the sets where the pass acts at 0.4
  and (ALEX) 0. Each: `get` on every stored key and on absent ones (both neighbours, 0, 1, 2^63 +- 1, 2^64 - 1, 2^64 - 2,
  below and above the range, random, inside gaps, **every virtual key**), scans of 1 to 70,000 pairs from stored,
  neighbouring, in-between and extreme starts and one over everything; then some 27,000 puts (neighbours, runs, in range,
  anywhere in 64 bits, far outside, duplicates, the virtual keys themselves), 20,000 updates, 22,000 removes and puts
  again, the same checks after each step; then every key removed and the index filled again. Counts, the side store's
  share, the base's own consistency check and the bound on the number of virtual keys are asserted.
- `alex`, first case `rootsplit` (added by the verifier): the put of a stored key splits ALEX's root data node, in four
  variants; then keys on both sides of the key domain ALEX fixed, then 20,000 more writes. With the fix taken out of
  `CsvAlex::put` it fails in all four variants (16 failures).
- `alex2m`, `lipp2m`: 2M uniform keys, 2M lognormal, 2M staircase, same checks with fewer probes.
- `modes`: `CSV_LAMBDA=floor`, `CSV_LIPP_RULE=loss`, a `CSV_BUDGET_S` too small for the pass and one of 0, `CSV_COST_C`,
  `CSV_SEARCH_W=0` (every subtree below ALEX's root merged), and seven malformed settings that must throw.

Results on the final files (`csv.h` md5 `3cb8b4b7f3ebc591a0deaae3070802e8`, test md5 `15b4f1b738b1f7cf3f246509708661da`; the same sums on the cluster):

| where | build | result |
|---|---|---|
| Mac (arm64, Apple clang) | `build/tests/cxx-O3` | passes, 70 of 70 lines, 38 s |
| Mac | `build/tests/cxx` (ASan + UBSan) with the ALEX ignore list | passes, 70 of 70, 3 min 35 s |
| Mac | `build/tests/cxx` as given, no ignore list | `smooth lipp lipp2m` pass (27 lines, 78 s); the `alex` group stops in ALEX at `alex_base.h:79:29` (above) |
| cluster login node (x86-64, g++ 11.4.1, `-O3 -march=native`, `wf_cxx`) | `nice ./t smooth alex alex2m modes` | passes, 44 lines, 36 s of CPU, 3.3 GB at most, 75 to 213 s of wall time depending on what else ran there |
| cluster login node | `nice ./t lipp lipp2m` | passes, 26 lines, 28 s of CPU, 3.8 GB at most, 37 to 184 s |

The first author's one-process run on the cluster took 5 min 28 s of wall time for 65 s of CPU, and two of its attempts
were killed by the 8 GB limit while other processes held memory (exit 137, no test failure); two of the verifier's
adversarial runs were likewise ended from outside (one SIGKILL while other processes held 6 GB, one SIGTERM) and passed
when run again. With `-Wall -Wextra` g++ 11 gives no warning from
`csv.h` (two from the test's timing helper, which deletes GRE's interface objects, whose classes have no virtual
destructor).

### The independent verification (22 November 2026)

A second agent rebuilt and reran all of the above, then wrote its own adversarial program (a scratch file,
`build/tests/sota/csv/verify/adv.cpp`, not part of `tests/`; `run_adv.sh` next to it has the runs): one key (0, 1, 12345,
2^63 - 1, 2^63, 2^64 - 2, 2^64 - 1), two keys (10 pairs), 3 to 6 keys at the extremes, keys equal modulo 2^32 (three
residues, in sequence and over all 64 bits, with 0 and 2^64 - 1), a dense run followed by a huge gap (to 2^63, 2^64 - 2,
2^64 - 1, to a second dense run), keys above 2^63 (uniform, runs with steps 1, 2 and 3, above two thirds of 2^64,
clusters, nested clusters, the keys around 2^63), 4M uniform keys over all 64 bits, 3M lognormal, 2.4M nested clusters;
after each bulk load `get` on every stored key and on never-stored ones (both neighbours at distance 1 and 2, the key
with bit 32 or bit 63 flipped, gap midpoints, 0, 2^64 - 1, 64 distances below and above the range, every virtual key),
scans, then 3,000 to 500,000 interleaved put / remove / get / update / scan operations against a `std::map`, and for
some sets every key removed and the index refilled. Names: `csv_lipp`, `csv_lipp40`, `csv_lipp80`, `csv_lipp0`,
`csv_alex`, `csv_alex40`, `csv_alex80`, `csv_alex0`, also with `CSV_LIPP_RULE=loss` and `CSV_SEARCH_W=0` (every LIPP
rebuild kept, every ALEX subtree merged) and `CSV_LAMBDA=floor`. Run on the Mac at `-O3` (510 cases) and under
ASan + UBSan with the ALEX ignore list (203 cases), and on the cluster with g++ 11 (262 cases). What it found:

1. **A defect, fixed**: the put of a stored key that splits ALEX's root data node (described under "What the wrappers
   guard"). Seen as `std::bad_array_new_length` out of `Alex::expand_root` on a dense run plus one far key; no wrong
   answer was observed before the exception.
2. **Fixed**: `memory_consumption` did not count the pass's work arrays, which stayed allocated (above).
3. **Fixed**: `csv_lipp` accepted exactly 2^30 keys, for which LIPP's root would need 2^31 slots (an int); the limit is
   now 2^30 - 1.
4. Nothing else: no wrong `get`, `scan`, `put`, `update` or `remove` answer in any case, on either machine, once 1 was
   fixed; nothing reported by the sanitizers in `csv.h` or LIPP.

What the pass did on three made-up sets (Mac, `-O3`, `t-O3 none --time`; **not a benchmark**):

| set | index | smoothed / kept subtrees | virtual keys | promoted | keys at level 3+ | average level | pass |
|---|---|---|---|---|---|---|---|
| 3M uniform | `csv_lipp` | 68,557 / 33,197 | 1.1% of n | 51,156 | 171,317 -> 135,702 | 1.4504 -> 1.4385 | 0.16 s |
| 3M lognormal | `csv_lipp` | 45,033 / 18,551 | 5.2% | 96,462 | 535,276 -> 474,981 | 2.1341 -> 2.1140 | 1.0 s |
| 3M lognormal | `csv_lipp40` | 45,033 / 27,247 | 17.1% | 146,219 | 535,276 -> 431,688 | 2.1341 -> 2.0995 | 2.0 s |
| 5.6M nested clusters | `csv_lipp` | 573,776 / 220,646 | 5.0% | 465,174 | 5,450,973 -> 5,449,449 | 3.3558 -> 3.2971 | 6.2 s |
| 5.6M nested clusters | `csv_lipp40` | 588,525 / 271,141 | 21.7% | 734,186 | 5,450,973 -> 5,429,484 | 3.3558 -> 3.2673 | 23.8 s |
| 3M lognormal | `csv_alex` | 3 / 3 (4 more too big) | 0.6% | 224,350 | - | 1.3836 -> 1.3088 | 19.5 s |
| 3M uniform, 5.6M nested | `csv_alex` | no model node below the root | 0 | 0 | - | 1.0 | 0 |

Promoted shares of the promotable keys here are 9 to 30% on LIPP against the paper's 27 to 60% on its data; memory grows
by 1 to 9%. Lookup times of `csv_lipp0`, `csv_lipp`, `csv_lipp40` (and of the three `csv_alex`) were within the run-to-run
spread of each other on the laptop, as an average-level change of 1 to 3% would suggest.

## Known limits

- Not the authors' code; every choice listed under "Deviations" is ours. Without their code or their constants the ALEX
  results in particular cannot be expected to match: with ALEX's own cost weights the condition rarely holds (2M staircase
  keys at alpha 0.1: 200 of 3,693 subtrees merged on the Mac and 0 of 3,275 on the cluster, where ALEX builds a slightly
  different tree from the same keys). `CSV_SEARCH_W` and `CSV_LEVEL_W` are there so that measured constants can be put
  in, as the paper says to.
- **`csv_alex` may do nothing at all.** ALEX's model nodes below the root hold very many keys on smooth data, and a
  subtree is skipped above `CSV_MAX_SUBTREE` (131,072 keys) and can never be merged above about 734,000 (one ALEX data
  node). Made-up sets on the Mac: 4M uniform keys, 4 candidates, all 4 too big; 20M uniform, no model node below the root;
  20M lognormal, 11 candidates, 10 too big, 1 merged, average level unchanged (1.3495). The bulk-load line says it
  (`candidates`, `too_big`, `accepted`, `avg_level`); a `csv_alex` row with `accepted=0` is ALEX, and should be reported as
  "CSV found nothing to merge", not as a CSV result. Raising `CSV_MAX_SUBTREE` to 734000 lets the pass try the rest, at
  up to 7 minutes for one subtree of 667,000 keys at alpha 0.1 (and only as far as `CSV_BUDGET_S` allows).
- **Not run at 200M keys.** Algorithm 1 costs about 10 ns per gap and round, alpha x m^2 per subtree of m keys. On made-up
  sets the LIPP pass grew in step with n (lognormal: 0.5 s at 2M keys, 8.2 s at 20M, 6.7% virtual keys; nested clusters:
  0.35 s at 1.2M, 4.1 s at 12M; 320 to 470 ns a key for the whole bulk load, of which the wrapper's check of every key is
  at most a tenth), which would put the pass at one to two minutes for 200M keys; real data with larger subtrees will cost more, and
  whether the pass then fits `CSV_BUDGET_S` is not known. The line says (`over_budget`, `late`, `plan_s`, `csv_ms`). The
  paper's own pass took 5 to 50 minutes at alpha 0.1. Peak memory is the base's own bulk load (LIPP's builder copies keys
  and payloads once) plus, only if a key had to go to the side store at the bulk load, one more copy of the input.
- **The side store on real data is not known.** `csv_alex` keeps at most four keys inside any window of 4 x 2^(e-52)
  around a key of magnitude 2^e >= 2^53 in ALEX, at the bulk load too (ALEX itself bulk-loads and looks up such runs
  correctly; it is an insert next to them that does not return, which the verifier reproduced on GRE's plain ALEX: 1,500
  keys 2 apart from 2^63, an insert between them still running after 20 s). Data
  with tight clusters of large keys could send a real share of its keys to the `std::map`. `side=` in the bulk-load line
  is the number to look at before the row is used. `csv_lipp` on x86 moves only keys exactly 1 apart from 2^63 up.
- A key outside the range of the bulk-loaded keys always goes to `csv_alex`'s side store (ALEX's key domain is never
  expanded): an insert workload that appends beyond the largest loaded key runs on the `std::map`, not on ALEX.
- LIPP and ALEX count keys in int: at most 2^30 - 1 keys for `csv_lipp`, 2^30 for `csv_alex` (`bulk_load` throws above).
- Each `csv_lipp` object holds LIPP's pool (2.0 GB requested, about 2.5 GB held), as each `lipp` does.
- Single-threaded, like the bases used here.
- SALI is not covered.
