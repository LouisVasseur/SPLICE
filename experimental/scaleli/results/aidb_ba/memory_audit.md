# Memory-footprint audit of SCALE-LI at 200M keys

Scope: does `accounted_bytes` match what the operating system reports?  Binary: `build-final/experimental/scaleli/scaleli_bench` (sha256 c63ceab0…, built 2026-09-21 20:45, verified not to contain the uncommitted cache-line counter, i.e. built from the `MANIFEST.sha256` sources).  Host: Apple M3 Max, 16 cores, 64 GiB, macOS 15.7.5, 16 KiB pages.  Datasets: `data/external/gre/fb` and `data/external/gre/planet.sorted`, 200,000,000 keys each, `--load-ratio 1`.

## Answer in three lines

1. `accounted_bytes` is **exact at the level it claims** (live `size()`/`sizeof()` bytes) — the structural model reproduces `metadata_bytes` to the byte — but it is **not** what the OS holds. Measured against steady-state RSS: compressed layouts are under-reported by **2.0-2.7%**, raw-key layouts by **23.6% (+790 MB)**.
2. The 23.6% is one macOS `libmalloc` size-class cliff: a raw region arena is 33,280 B, one step over the 32,768 B boundary, so each of the 48,829 regions costs an extra 16 KiB page. `reserved_slack_bytes` reports 0 for every run and cannot see it.
3. **Peak RSS from `/usr/bin/time -l` is useless here** (12 of 16 runs report the same 9.6 GB regardless of index size) because the harness holds three 3.2 GB copies of the dataset during workload construction. Use the replay plateau instead; the recipe is in §2b.

## 1. What the five fields cover, and what they do not

Accounting lives in two functions (`include/scaleli/index.hpp`): `Region::memory()` at line 325 and `Index::memory()` at line 511. Exactly:

```
key_bytes       = sum over regions of arena.size()
value_bytes     = sum over regions of values.size() * 8
metadata_bytes  = sizeof(Index)                                 (336 B)
                + regions_.capacity() * sizeof(unique_ptr)      (8 B per region)
                + root_slot_to_region_.size() * 4               (0 unless --root model with virtual fences)
                + sum over regions of [ sizeof(Region)          (280 B)
                                      + blocks.size() * sizeof(BlockDescriptor)   (64 B per 128-key block)
                                      + virtual_features.size() * 8 ]
delta_bytes     = sum over regions of delta.size() * sizeof(DeltaEntry)   (24 B, padding included)
reserved_slack  = capacity-minus-size of arena, values, blocks, delta only
accounted_bytes = the sum of those five
```

The structural model is exact, not approximate: with 200,000,000 keys, `region_keys=4096`, `block_keys=128` the formula predicts 48,829 regions, 1,562,500 block descriptors and

    336 + 8*48,829 + 280*48,829 + 64*1,562,500 = 114,063,088 bytes

and every one of the fourteen SCALE-LI runs reported `metadata_bytes = 114063088` (0.570 B/key). Block descriptors are 100.0 MB of that 114.1 MB, i.e. 0.5 B/key, and 7 of every 64 descriptor bytes are alignment padding around the 1-byte `codec` field (10.9 MB at this scale). The padding **is** counted.

What is **not** counted, in order of size at 200M keys:

| # | Not covered | Size here |
|---|---|---|
| 1 | **Allocator size-class rounding and page granularity.** Every field is a `size()`/`sizeof()`, never `malloc_size()`. | **+790 MB (raw), +42-53 MB (packed)** — measured in §2 |
| 2 | The `std::vector<Record>` copy `bulk_load` takes **by value** (`Index::bulk_load(std::vector<Record> rows)`), plus the `std::stable_sort` scratch buffer inside `canonicalize` | 3.2 GB + 3.2 GB, transient, during every build |
| 3 | The harness `Workload::initial` vector held for the whole process lifetime, and `Workload::trace` | 3.2 GB + 32 B/op |
| 4 | Per-region transient build buffers (`keys`, `ranks`, `positions`, `x`, `slots`, `EncodedBlock::bytes`, `EncodedBlock::key_positions`) | ~200 KB live at a time, single-threaded |
| 5 | `virtual_features` **capacity** slack — its `size()` is in `metadata_bytes` but its spare capacity is in no field | 0 here (`--virtual-alpha 0`); 8 B/point otherwise |
| 6 | `root_slot_to_region_` counted by `size()`, not `capacity()` | 0 here (`--root binary`) |
| 7 | The `FlowTransform` weights (`Config::flow` is a bare pointer; the pointee is reported only as `learnability.flow_bytes`) | 0 here (no `--flow`) |
| 8 | Stack, binary, C++ runtime, dyld shared cache, page tables | ~60-330 MB of process RSS |

`docs/METRICS.md` already disclaims 1, 2, 3, 4 and 8 in one sentence ("Core accounting excludes malloc headers, allocator fragmentation, stack, temporary compaction buffers, workload vectors, source dataset copies, query output vectors, and process runtime"). Items 5, 6 and 7 are not disclaimed anywhere. What the doc does not do is say **how big** item 1 is, and the answer turns out to depend on the codec.

## 2. Reported vs. what the OS says

### 2a. Peak RSS (`/usr/bin/time -l`) does not answer the question

Peak RSS of this benchmark process is set by workload construction, not by the index. The trajectory (sampled every 0.25 s with `ps -o rss=`) is: read 1.6 GB file into 3.2 GB of `Record`s -> `canonicalize` stable-sort scratch -> `w.initial` copy + its own stable-sort scratch (**9.6 GB peak, before the index exists**) -> free -> `bulk_load` copies `w.initial` again and sorts it again (9.6 GB again) -> build the index.

| dataset | variant | accounted | peak RSS `time -l` | peak/accounted |
|---|---|---|---|---|
| fb | sorted_vector (no index) | 3.200 GB | 9.607 GB | 3.00 |
| fb | raw_rank | 3.339 GB | 10.534 GB | 3.15 |
| fb | forced raw | 3.339 GB | 10.353 GB | 3.10 |
| fb | forced for | 2.113 GB | 9.601 GB | 4.54 |
| fb | forced delta | 2.129 GB | 9.607 GB | 4.51 |
| fb | forced linear | 2.105 GB | 9.607 GB | 4.56 |
| fb | packed_rank (min_bytes) | 2.101 GB | 9.601 GB | 4.57 |
| fb | packed_byte (min_bytes) | 2.101 GB | 9.607 GB | 4.57 |
| planet | sorted_vector (no index) | 3.200 GB | 9.608 GB | 3.00 |
| planet | raw_rank | 3.339 GB | 10.534 GB | 3.15 |
| planet | forced raw | 3.339 GB | 9.937 GB | 2.98 |
| planet | forced for | 2.018 GB | 9.608 GB | 4.76 |
| planet | forced delta | 2.086 GB | 9.608 GB | 4.61 |
| planet | forced linear | 1.992 GB | 9.607 GB | 4.82 |
| planet | packed_rank (min_bytes) | 1.991 GB | 9.608 GB | 4.83 |
| planet | packed_byte (min_bytes) | 1.991 GB | 9.608 GB | 4.83 |

Read that table as a **negative result**: the ratio column is meaningless. Twelve of the sixteen runs report 9.60-9.61 GB whatever the index costs, because 9.6 GB is the workload-construction plateau and every index smaller than ~3.2 GB hides underneath it. Only the four raw-key rows poke above it, and two of them that build a **byte-identical** index disagree by 0.6 GB (`planet forced_raw` 9.94 GB vs `planet raw_rank` 10.53 GB). Do not put peak RSS on a slide next to `accounted_bytes`.

### 2b. Steady-state RSS during the replay does answer it

Second pass, 10M read-only operations so the serving phase lasts long enough to sample: take the resident set on the flat stretch after the build (the last plateau before teardown), and subtract the two harness terms that are known exactly — `w.initial` = 16 B x 200M = 3.200 GB and `w.trace` = 32 B x 10M = 0.320 GB.

The subtraction is calibrated by `--index sorted_vector`, whose footprint is exactly one `std::vector<Record>`, i.e. 3,200,000,024 accounted bytes:

| dataset | variant | accounted | plateau RSS | plateau − 3.52 GB = real index | unaccounted | real/accounted |
|---|---|---|---|---|---|---|
| fb | sorted_vector (no index) | 3.200 GB | 6.722 GB | 3.202 GB | +2 MB | **1.000** |
| fb | raw_rank | 3.339 GB | 7.649 GB | 4.129 GB | +790 MB | **1.236** |
| fb | forced raw | 3.339 GB | 7.649 GB | 4.129 GB | +790 MB | **1.236** |
| fb | forced for | 2.113 GB | 5.680 GB | 2.160 GB | +47 MB | **1.022** |
| fb | forced delta | 2.129 GB | 5.699 GB | 2.179 GB | +50 MB | **1.024** |
| fb | forced linear | 2.105 GB | 5.670 GB | 2.150 GB | +45 MB | **1.021** |
| fb | packed_rank (min_bytes) | 2.101 GB | 5.663 GB | 2.143 GB | +42 MB | **1.020** |
| fb | packed_byte (min_bytes) | 2.101 GB | 5.665 GB | 2.145 GB | +45 MB | **1.021** |
| planet | sorted_vector (no index) | 3.200 GB | 6.722 GB | 3.202 GB | +2 MB | **1.000** |
| planet | raw_rank | 3.339 GB | 7.648 GB | 4.128 GB | +789 MB | **1.236** |
| planet | forced raw | 3.339 GB | 7.648 GB | 4.128 GB | +789 MB | **1.236** |
| planet | forced for | 2.018 GB | 5.584 GB | 2.064 GB | +47 MB | **1.023** |
| planet | forced delta | 2.086 GB | 5.652 GB | 2.132 GB | +46 MB | **1.022** |
| planet | forced linear | 1.992 GB | 5.565 GB | 2.045 GB | +53 MB | **1.027** |
| planet | packed_rank (min_bytes) | 1.991 GB | 5.564 GB | 2.044 GB | +53 MB | **1.027** |
| planet | packed_byte (min_bytes) | 1.991 GB | 5.559 GB | 2.039 GB | +48 MB | **1.024** |

The two `sorted_vector` rows land on 3.202 GB against 3.200 GB accounted (+1.5 MB, 0.05%), which is the control: the method resolves the index footprint to about 0.05%.

**Result.** Two clean regimes, reproduced on both datasets:

* every **compressed** layout is under-reported by **2.0-2.7%** (+42 to +53 MB on ~2.1 GB);
* every **raw-key** layout is under-reported by **23.6%** (+789 MB on 3.339 GB) — four runs, two datasets, two different ways of asking for raw (`--policy raw` and `--policy forced --codec raw`), all four within 0.6 MB of each other.

### 2c. Why 23.6%, exactly

Not fragmentation — `vmmap` reports 0-2% fragmentation in the malloc zone for every run. It is one size-class boundary. With `region_keys=4096`, `block_keys=128` and the raw codec a region arena is

    32 blocks x (16 B block header + 128 x 8 B) = 33,280 B

and macOS `libmalloc` has no class between 32,768 and 65,536. Measured directly with `malloc_size()` on this host:

```
  request   malloc_size   waste
    7,921         8,192     271      <- typical packed arena (512 B quantum)
   32,384        32,768     384
   32,768        32,768       0      <- the value column, 4096 x 8 B, exactly at the boundary
   32,769        65,536  32,767      <- the cliff
   33,280        65,536  32,256      <- the raw arena
```

Only the pages actually written become resident: 33,280 B spans 3 of the 16 KiB pages inside that 64 KiB block, so the resident waste is 49,152 - 33,280 = **15,872 B per region**, and

    15,872 B x 48,829 regions = 775.0 MB

against the 789.4-789.7 MB measured. The remaining ~14 MB is the 512 B quantum on the other per-region allocations plus magazine fragmentation. The value column escapes entirely because 4096 x 8 = 32,768 B sits exactly on the boundary.

Consequences worth stating to a supervisor:

1. This is a **macOS allocator artifact**, not a property of the data structure. glibc would round a 33,280 B request to a 16 B multiple. Any absolute RSS claim in this project is platform-specific.
2. It is nonetheless **real memory on the machine we measure on**, and it is invisible in every field we currently report — `reserved_slack_bytes` is 0 for all sixteen runs, because `shrink_to_fit()` does succeed at the `std::vector` level. Vector capacity equalling vector size says nothing about the block malloc handed out.
3. It falls on the **raw baseline**, which is the thing compression is measured against, so it makes compression look worse on paper than it is on the machine (§3).
4. It is avoidable: any region/block geometry whose raw arena is <= 32,768 B (e.g. `--region-keys 2048`, arena 16,640 B, malloc_size 16,896, waste 256 B) removes it. We have not tested whether that geometry is otherwise a good idea.

## 3. Compression survey, full 200M keys

Bytes per key, split by component, both datasets. `ratio vs 8 B` is `8 / key_bytes_per_key` — a pure key-arena ratio, as `docs/METRICS.md` defines `key_arena_ratio`. `key share` is the key arena as a fraction of the whole accounted index.

(`sorted_vector` stores interleaved `pair<Key,Value>`; its key/value split is notional, its 16.000 B/key is not.)

| dataset | variant | key B/key | value B/key | meta B/key | total B/key | ratio vs 8 B | key share | real B/key (§2b) |
|---|---|---|---|---|---|---|---|---|
| fb | sorted_vector | 8.000 | 8.000 | 0.000 | 16.000 | 1.00x | 50.0% | 16.008 |
| fb | raw_rank | 8.125 | 8.000 | 0.570 | 16.695 | 0.98x | 48.7% | 20.644 |
| fb | forced raw | 8.125 | 8.000 | 0.570 | 16.695 | 0.98x | 48.7% | 20.643 |
| fb | forced for | 1.993 | 8.000 | 0.570 | 10.564 | 4.01x | 18.9% | 10.800 |
| fb | forced delta | 2.075 | 8.000 | 0.570 | 10.646 | 3.85x | 19.5% | 10.896 |
| fb | forced linear | 1.956 | 8.000 | 0.570 | 10.526 | 4.09x | 18.6% | 10.752 |
| fb | packed_rank | 1.934 | 8.000 | 0.570 | 10.504 | 4.14x | 18.4% | 10.714 |
| fb | packed_byte | 1.934 | 8.000 | 0.570 | 10.504 | 4.14x | 18.4% | 10.727 |
| planet | sorted_vector | 8.000 | 8.000 | 0.000 | 16.000 | 1.00x | 50.0% | 16.008 |
| planet | raw_rank | 8.125 | 8.000 | 0.570 | 16.695 | 0.98x | 48.7% | 20.642 |
| planet | forced raw | 8.125 | 8.000 | 0.570 | 16.695 | 0.98x | 48.7% | 20.641 |
| planet | forced for | 1.517 | 8.000 | 0.570 | 10.088 | 5.27x | 15.0% | 10.320 |
| planet | forced delta | 1.860 | 8.000 | 0.570 | 10.430 | 4.30x | 17.8% | 10.661 |
| planet | forced linear | 1.390 | 8.000 | 0.570 | 9.961 | 5.75x | 14.0% | 10.227 |
| planet | packed_rank | 1.384 | 8.000 | 0.570 | 9.955 | 5.78x | 13.9% | 10.220 |
| planet | packed_byte | 1.384 | 8.000 | 0.570 | 9.955 | 5.78x | 13.9% | 10.196 |

### The value column dominates, and no codec touches it

`value_bytes` is `values.size() * sizeof(Value)` over a plain `std::vector<Value>` — 8 B/key, never encoded, on every variant. So:

* In the **uncompressed** index the key arena is 8.125 of 16.695 B/key, i.e. **48.7%** of the footprint. That is the ceiling on what key compression can ever address; the other 51.3% (values 8.000 + metadata 0.570) is untouchable.
* After the best compression it is **18.4% (fb) / 13.9% (planet)** of the index. The incompressible floor — values + block descriptors + region objects — is **8.570 B/key**, which is **81.6% (fb) / 86.1% (planet)** of the compressed total.
* Net: a 4.1x / 5.8x key-arena ratio buys a **37.1% (fb) / 40.4% (planet)** reduction in accounted index bytes (16.695 -> 10.504 / 9.955 B/key).
* Measured in real resident bytes (§2b), where the raw baseline also pays its 790 MB allocator penalty, the same change is **-48.1% (fb) / -50.5% (planet)** (20.64 -> 10.71 / 10.22 B/key). The accounted number understates the benefit of compression on this host by 10-11 points.
* Against the no-index baseline: `sorted_vector` is 16.000 B/key, `packed_rank` is 10.504 (fb) / 9.955 (planet), so the whole learned index — models, fences, descriptors and all — is **34-38% smaller than a plain sorted array** of the same pairs.

The obvious lever nobody has pulled: **halving the value column** (or compressing it at all) is worth more than any remaining key-codec work. Going from 8 B to 4 B values would cut the packed index by 38-40%; the best remaining key-codec gain is a few percent of 1.4-1.9 B/key.

### Build time and throughput

Throughput is a single closed-loop read-only replay of 10,000,000 operations (`--read 1 --miss 0.1`, uniform query distribution, `--warmup 4096`, `--qos 1`, `--latency 0 --instrument 0 --verify 0`), one seed, one process per row. `others` is the maximum number of **other** `scaleli_bench` processes seen on the host during that run (a second agent was benchmarking 200M datasets on the same machine with `--build-threads 16` for part of this session). Rows with `others > 0` are contended and their times are not usable.

| dataset | variant | build s | throughput ops/s | others | build s (200k-op batch) | throughput (200k-op batch) |
|---|---|---|---|---|---|---|
| fb | sorted_vector | 5.6 | 1 155 844 | 1 | 5.1 | 989 415 |
| fb | raw_rank | 14.4 | 957 629 | 1 | 13.8 | 564 094 |
| fb | forced raw | 14.8 | 910 698 | 0 | 14.7 | 434 076 |
| fb | forced for | 15.2 | 826 035 | 0 | 16.4 | 590 480 |
| fb | forced delta | 15.4 | 720 126 | 0 | 15.4 | 435 553 |
| fb | forced linear | 16.8 | 734 776 | 0 | 16.4 | 621 057 |
| fb | packed_rank | 19.3 | 725 607 | 0 | 19.7 | 512 935 |
| fb | packed_byte | 19.5 | 738 341 | 0 | 19.6 | 424 937 |
| planet | sorted_vector | 5.2 | 1 139 332 | 0 | 5.0 | 1 329 126 |
| planet | raw_rank | 14.4 | 955 924 | 0 | 14.0 | 475 844 |
| planet | forced raw | 14.7 | 1 046 974 | 0 | 15.3 | 263 098 |
| planet | forced for | 14.9 | 885 459 | 0 | 15.1 | 606 626 |
| planet | forced delta | 15.1 | 796 867 | 0 | 15.2 | 611 878 |
| planet | forced linear | 16.5 | 779 083 | 0 | 16.3 | 637 691 |
| planet | packed_rank | 18.8 | 763 983 | 0 | 19.2 | 679 079 |
| planet | packed_byte | 18.8 | 907 257 | 0 | 19.0 | 646 579 |

The 200k-op column is an earlier batch kept only as a cross-check; at 200,000 operations the replay lasts
~0.3 s and the numbers are worthless (`planet forced raw` 263 kops/s vs `planet raw_rank` 476 kops/s for a
**byte-identical** index), and most of that batch ran under contention. Use the 10M-op column.

## 4. Which codec wins

### On size: `min_bytes`, by about one percent

| | fb key B/key | planet key B/key |
|---|---|---|
| forced raw | 8.125 | 8.125 |
| forced delta | 2.075 | 1.860 |
| forced for | 1.993 | 1.517 |
| forced linear | 1.956 | 1.390 |
| **min_bytes (packed)** | **1.934** | **1.384** |

`min_bytes` is the smallest on both datasets, so **yes, the per-block policy does pick at least as well as the
best fixed codec** — it encodes each block four times and keeps the smallest, and mixing codecs across blocks
beats any single codec. But the margin over the best fixed codec (`linear` on both) is **1.1% of the key arena
on fb and 0.4% on planet**, which is **0.2% and 0.06% of the whole index**. The ranking of the fixed codecs is
the same on both datasets (`linear` < `for` < `delta`) but the gaps are not: on fb `for` is within 1.9% of
`linear`, on planet it is 9.1% behind. So the policy is doing real work; it is just that the work is worth
almost nothing once the 8 B value column is in the denominator.

`raw` is worth stating explicitly: at 8.125 B/key it is **1.6% larger than simply storing the keys**, because
of the 16-byte header on every 128-key block.

`packed_rank` and `packed_byte` have byte-identical arenas — `--routing` selects which model is consulted at
query time, not how keys are stored. It is a speed knob, not a size knob.

### On speed: raw wins; among compressed codecs, `for` (and byte routing)

From the uncontended 10M-op rows, microseconds per operation:

| variant | fb | planet |
|---|---|---|
| sorted_vector | 0.865* | 0.878 |
| raw_rank / forced raw | 1.044* / 1.098 | 1.046 / 0.955 |
| forced for | 1.211 | 1.129 |
| forced delta | 1.389 | 1.255 |
| forced linear | 1.361 | 1.284 |
| packed_rank (min_bytes) | 1.378 | 1.309 |
| packed_byte (min_bytes) | 1.354 | 1.102 |

(*contended row.)

* **Uncompressed is fastest.** Raw keys serve 0.91-1.05 Mops/s against 0.72-0.91 Mops/s compressed: decoding
  costs roughly 15-25%. That is the price of the 37-40% footprint reduction.
* **Among the rank-routed variants, `for` is the fastest compressed codec on both datasets** (+14% over
  `min_bytes` on fb, +16% on planet). Direction is consistent across two datasets, but this is one seed per
  cell and repeated measurements of an identical structure on this host have differed by 8-25% before, so
  treat it as a hypothesis to confirm with five seeds, not a result.
* **Routing is a comparable lever and is confounded here.** Every `forced` row uses `--routing rank`; the only
  byte-routed row, `packed_byte`, is the fastest compressed variant on planet (907 kops/s, ahead of
  `forced for`'s 885) while being indistinguishable from `packed_rank` on fb (738 vs 726). Forced codecs were
  not run with byte routing, so "`for` is fastest" is a statement about rank routing only.
* **`min_bytes` optimises the wrong objective.** Against `forced for` it buys **0.6% (fb) / 1.3% (planet)** off
  the total index and gives back **12% / 14%** of throughput, plus 27% more build time (19.3 s vs 15.2 s,
  because it encodes every block four times). It minimises key bytes, and key bytes are 14-18% of the index.
  If the meeting wants one recommendation from this table, it is: **default to `--policy forced --codec for`,
  not `min_bytes`, until `min_bytes` is taught to score decode cost as well as bytes.**
* **A plain sorted array still wins on latency** at 200M keys (0.87 us/op vs 1.10-1.38). The learned index's
  case here is footprint (10.0-10.5 vs 16.0 B/key) and ordered-map functionality, not lookup speed.

### Caveats on these numbers

* One seed, one process per cell; `--warmup 4096` only, i.e. **no meaningful cache warm-up** — the supervisor's
  point stands and is not addressed here. `warmup()` in `src/benchmark.cpp:44` touches 4,096 of 200,000,000 keys
  (0.002%) and does nothing about page residency.
* Throughput was measured on a shared machine; the `others` column marks the two rows where a second 200M
  benchmark was running.
* Read-only workload. `delta_bytes` is 0 and no compaction ran in any row, so this is a bulk-load footprint
  audit; write-path memory (delta buffers at 24 B/entry, compaction transients) is untested.
* All absolute overheads in §2 are macOS/`libmalloc` numbers. The method (plateau RSS minus known harness terms,
  calibrated on `sorted_vector`) transfers to Linux; the 23.6% does not.

## Reproduce

```
B=build-final/experimental/scaleli/scaleli_bench
D=experimental/scaleli/data/external/gre
$B --data $D/fb --format sosd --dtype uint64 --load-ratio 1 --ops 10000000 \
   --verify 0 --latency 0 --instrument 0 --qos 1 --policy raw --routing rank
# sample `ps -o rss= -p PID` every 0.25 s; take the last plateau; subtract 3.20 GB + 32 B x ops
```

Scripts and raw output: `run_all.sh` / `runs/` (peak RSS), `run_rss.sh` / `rss/` (trajectories + vmmap),
`run_thr.sh` / `thr/` (timings), `plateau.py`, `msize.cpp` (allocator size classes).
