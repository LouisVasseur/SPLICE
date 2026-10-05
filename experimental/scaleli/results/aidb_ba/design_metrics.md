# NFL and CSV, before and after, at 200M keys: what each of the supervisor's metrics can show

Designer 2 (the supervisor's metrics). This is a design written from code and existing result files only.
Nothing was run (a 200M timing pilot owns the machine). S = /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli.
Every "measured" number below comes from S/results/aidb_fullscale/sweep/results.jsonl (fb and planet, 200M keys,
seed 11, 2M lookups, `--instrument 1`, build-root binary). The scratchpad memory audit is
.../scratchpad/fullscale/memory_audit.md. Every "predicted" number gives its derivation.

---------------------------------------------------------------------------------------------------------

## 0. Bottom line

1. **Compression cannot move, by construction.** Blocks are encoded from the keys in key order *before*
   any feature or target exists (index.hpp:213-235; the feature and target code starts at :238). Neither the
   NFL feature nor the CSV slots reach `encode_block`. Measured: key bytes per key are identical to three
   decimals in all 16 runs at 200M (fb 1.934, planet 1.384; key_arena_ratio 0.2417 / 0.1730). Show this
   once as a deliberate null row, or drop it from the 2x2. Then show the compression work in its own panel
   (the codec survey), because the supervisor asked for it.
2. **Memory: NFL changes nothing measurable.** The flow weights are 144 B in total and are not even in
   `metadata_bytes` (memory_audit item 7). **CSV costs +0.80 B/key** (+7.6% fb, +8.0% planet). That cost
   is a storage-policy choice, not a lookup requirement. The 8 B per virtual point sits in
   `Region::virtual_features`, which only the compaction re-placement path reads (index.hpp:296-307).
   Lookups use `rank_model` and `BlockDescriptor::slot_begin`, and that field exists in every variant.
   A read-only index could drop the vector, which would bring CSV's memory cost to about 0.001-0.005 B/key
   (the root table only).
3. **The probes that matter are comparisons per lookup = root + block-coordinate + fence + in-block key_at.**
   Two stages are fixed by construction: coordinate (5.03, a binary search over the 32 `slot_begin` values)
   and key_at (8.02, a binary search over 128 keys). That is a floor of 13.05 that neither method can touch.
   Only the root and fence terms can move. At 200M the host makes 29.0-33.9 comparisons per lookup.
   Binary search over the sorted array makes 27.6. CSV at the root brings fb to 21.2. So at 200M, unlike
   2M, the learned variants make *fewer* comparisons than binary search, and still lose on time
   (memory_audit: sorted_vector 1.14-1.16 Mops vs packed 0.73-0.76 Mops).
4. **The faithful NFL + CSV combination equals CSV alone by construction:**
   - At the root, the virtual-fence candidate is skipped whenever the feature is not sorted
     (index.hpp:406, `if(!std::is_sorted(...))continue;`). The faithful flow is non-monotone.
   - In the regions, the selector picks the flow in 0 of 6,740 regions.

   So in the 2x2 the "NFL+CSV" cell is a layout-identical repeat of the "CSV" cell. Use it as an A/A test.
   "NFL alone" under `--fusion auto` is likewise byte-identical to "Before".
5. **A new finding in the existing 200M files, not in the established list.** On planet at 200M, the root
   selector *chose a flow*: `packed_rank_root_fusion` has `root_flow: true`, 111,688 virtual fences and
   3.32 root probes. CSV alone reaches 13.69 with 195,316 fences, the flow alone falls back to binary
   (15.66), and Before is 15.66. The interaction is **-10.4 root probes**, against the -0.36 interaction
   measured at 2M. This is synergy at full scale. **But the "flow" in that run is the old inert monotone
   flow** (results/aidb/flows/planet_2D2H2L.txt, used by every 200M run so far). That flow is a global
   monotone bend, not NFL's sawtooth mechanism. It also cost 2,973 s to build, and its single-seed
   throughput of 1.175x sits inside a 25% same-structure spread. It is the 200M version of the "monotone
   bend is a lever on root probes" result in results/aidb_joint, and it must be labelled that way, not as
   "NFL works".

---------------------------------------------------------------------------------------------------------

## 1. The four cells, as flags (host fixed: `--policy min_bytes --routing rank --region-keys 4096 --block-keys 128`)

| cell | flags added to the host | what it is |
|---|---|---|
| **B: Before** | `--root model` (root-alpha 0, no flow) | The best host without either method. On fb the raw learned root wins (10.81 root probes). On planet it falls back to binary (15.66). I recommend this rather than `--root binary` as Before, so that "has a learned root" is not credited to CSV. Show binary-root as a thin reference tick. |
| **N: after NFL** (primary) | `--flow flows_free/<d>_s512\|s64_t2000.txt --flow-bypass 1` (`--fusion manual`) | Faithful non-monotone flow, deployed with NFL's own switch, i.e. the conflict-degree rule. This is what "applying NFL" means. |
| N-auto (sanity) | same flow, `--fusion auto` | The probe-cost selector. It picks the flow in 0 of 6,740 regions, so the layout is byte-identical to B. This is a free **A/A timing control**. |
| N-forced (stress) | same flow, `--flow-bypass 0` | Every region uses the flow. This is the upper bound of the damage. |
| **C: after CSV** | `--virtual-alpha 0.1 --root-alpha 4` | Region virtual points plus root virtual fences. On planet the root at alpha 4 costs 2,068 s of single-threaded smoothing. |
| **NC: after both** | C flags + faithful flow + `--fusion auto` | Predicted layout-identical to C (index.hpp:406 plus the 0/6,740 selector result). Confirm with `--dump-layout` hashes, then reuse C's timing as a second A/A. |
| NC-bend (labelled side cell) | C flags + `--flow aidb_joint/flows_root/<d>_joint.txt --flow-cost 0` (or the inert results/aidb/flows/<d>_2D2H2L.txt that produced planet's 3.32) | A monotone bend plus CSV, i.e. the only combination that differs from C. Label it "monotone bend (not NFL's mechanism) + CSV". On fb the fitted warp is affine to 1e-5, so this cell equals C. |

Byte routing is excluded on purpose. Under `--routing byte` the model targets are byte positions
(`fresh.byte_model.fit_xy(keys,x,positions)`, index.hpp:319), so **CSV is a no-op under byte routing by
construction**. The memory audit found byte routing to be the fastest compressed configuration on planet
(907 vs 764 kops), so state this explicitly: the 2x2 lives on rank routing.

---------------------------------------------------------------------------------------------------------

## 2. Metric by metric: what can move, what cannot, and the prediction

### 2.1 Compression (key arena ratio)

- **NFL cannot move it.** z is computed after encoding and is never stored (index.hpp:238-290; z feeds
  only `rank_model`/`byte_model`). NFL's own design would move it only if records were stored in z order
  (AFLI). Our host never does that, and a z-ordered arena would wreck FOR/delta (non-monotone keys).
- **CSV cannot move it.** Virtual points are slot offsets and hold no record (smoothing.hpp:20-21). The
  arena is unchanged. Note that the CSV paper's "<10% typical, <31% worst" index growth comes from empty
  slots in gapped node arrays, a mechanism this host does not have.
- **Combination:** cannot.
- **Measured:** fb 1.934 B/key (4.14x vs 8 B), planet 1.384 (5.78x), identical in every one of the 16
  runs.
- **Recommendation:** in the 2x2, one table row, "key arena ratio: 0.2417 / 0.1730 in all four cells, by
  construction (encode before fit)". The compression *panel* is the codec survey, which is independent of
  NFL/CSV:
  - raw 8.125, for 1.993/1.517, delta 2.075/1.860, linear 1.956/1.390, min_bytes 1.934/1.384 B/key
    (fb/planet);
  - total accounted index -37.1% / -40.4%, real RSS -48.1% / -50.5%, because the raw baseline pays the
    790 MB allocator size-class cliff;
  - the decode cost of that compression: raw 0.91-1.05 Mops vs compressed 0.72-0.91 Mops;
  - the work counter `decoded_keys` per lookup (fb 14.62, planet 9.33) is also identical across cells,
    because it is a function of the codec and the block only.

### 2.2 Memory footprint (B/key, accounted and real)

**NFL.** It adds the `FlowTransform` (144 B, outside `metadata_bytes`) and nothing per key. The Region
struct already carries `flow_used` (index.hpp:98). Δ = 7e-7 B/key, i.e. **a null by construction**.

**CSV.**
- Region: 8 B per virtual point (index.hpp:340). At alpha 0.1 the greedy spends 99.9% (fb: 19,970,703
  points) / 99.2% (planet: 19,847,006) of the budget, so metadata goes 0.570 -> 1.370 (fb) / 1.364
  (planet).
- Root: 4 B per slot of `root_slot_to_region_` (index.hpp:535). That is (48,829 + 973) x 4 = 0.2 MB on fb
  (+0.001 B/key) and (48,829 + 195,316) x 4 = 0.98 MB on planet (+0.0049 B/key; measured 0.570 -> 0.575).
- Only the root table is needed by lookups. The region `virtual_features` exist only for compaction
  re-placement.

**Combination.** CSV's cost again (faithful), or 111,688 root fences on planet with the bend
(+0.0032 B/key).

| B/key (accounted) | fb B | fb N | fb C | fb NC | planet B | planet N | planet C | planet NC-bend |
|---|---|---|---|---|---|---|---|---|
| keys | 1.934 | 1.934 | 1.934 | 1.934 | 1.384 | 1.384 | 1.384 | 1.384 |
| values | 8.000 | 8.000 | 8.000 | 8.000 | 8.000 | 8.000 | 8.000 | 8.000 |
| metadata | 0.570 | 0.570 | 1.371 | 1.371 | 0.570 | 0.570 | 1.369 (pred.) | 1.367 (meas.) |
| total | 10.504 | 10.504 | 11.305 (+7.6%) | 11.305 | 9.955 | 9.955 | 10.753 (+8.0%) | 10.751 |
| read-only trim (drop region vf) | 10.504 | 10.504 | 10.505 | 10.505 | 9.955 | 9.955 | 9.960 | 9.958 |

Two predictions to check against the steady-state RSS plateau (the memory audit's §2b recipe; peak RSS
is useless here):

- (a) The packed cells' real/accounted ratio stays at 1.020-1.027.
- (b) The CSV cells show an *extra* unaccounted ~0.1-0.2 B/key (20-40 MB). Reason: `smooth_cdf` grows
  `virtual_features` by push_back with no reserve (smoothing.hpp:92), and `rebuild` shrinks only arena,
  values and blocks (index.hpp:322). A 409-point vector therefore sits in a 512-slot capacity, about 25%
  slack, which `reserved_slack_bytes` does not see (memory_audit item 5).

**Figure M (memory).**
- Form: horizontal stacked bars, one per cell; 8 bars (4 cells x fb, planet), or 40 for all ten datasets.
- Segments: keys | values | base metadata | region virtual points | root table.
- x axis: B/key from 0 to 12, linear.
- Overlay: a black dot for the steady-state RSS per key, and a dashed line at 16.0 for the plain sorted
  array.
- Intervals: none. Accounting is deterministic, and the RSS method resolves 0.05% (sorted_vector control,
  +2 MB on 3.2 GB). Say so in the caption.
- Annotate the read-only-trim total as a hollow tick on the C/NC bars.
- Message: the value column is 76-80% of every bar; NFL is invisible; CSV is +0.8 B/key, and that cost is
  optional.

### 2.3 Comparisons per lookup (probes counted correctly)

Definition: comparisons/lookup = `root_probes + coordinate_probes + fence_probes + key_at_calls`, all
divided by operations (delta_probes are 0 read-only). Report `transform_calls`/lookup next to it, because
a flow evaluation is not a comparison but costs 28-66 ns (6.9-16 probe-equivalents at the 2M exchange
rate).

Construction:

| stage | NFL can move? | CSV can move? | 200M value |
|---|---|---|---|
| root | yes, as a root feature (but monotone flows only can take virtual fences; :406) | yes, root virtual fences | 15.66 binary, fb 10.81 raw model |
| coordinate (binary search over 32 `slot_begin`, :127-133) | no, about log2(32) for any model | no | 5.03 all runs |
| fence correction (:139-157) | yes, via model quality | yes | fb 5.14, planet 3.99 |
| in-block key_at (:167-168) | no, the model is not consulted | no | 8.02 all runs |

| comparisons/lookup | fb | planet | how obtained |
|---|---|---|---|
| binary search over sorted array | 27.6 | 27.6 | log2(2e8) |
| host with binary root | 33.85 | 32.70 | measured |
| **B** (learned root, alpha 0) | **29.00** | **32.70** (fallback) | measured |
| **N** faithful, bypass 1 | 30.6 (+5.6%) | 34.4 (+5.1%) | predicted: fence x 1.316 (fb) / 1.417 (planet), the 2M bypass ratios in flowcheck_summary.json (3.423/2.601, 4.574/3.229); root unchanged; transform calls ~0.16 / ~0.32 per lookup (accepted-region fraction at 2M: 80/489, 157/489) |
| N-auto | 29.00 | 32.70 | identical by construction |
| N-forced | 38.9 (+34%) | 35.3 (+8%) | predicted: fence x 2.92 / x 1.654 (2M forced ratios); 1.0 transform call per lookup |
| **C** (vp 0.1 + root alpha 4) | **21.21 (-27%)** | **30.16 (-8%)** | fb measured; planet predicted as root 13.69 (vf4 run) + fence 3.42 (vp run) + 13.05; region and root savings add to within 0.003 probes |
| **NC** faithful | 21.21 | 30.16 | = C by construction |
| NC-bend | = C (warp affine on fb) | **19.79 (-39%)** | planet measured (`packed_rank_vp10_root_fusion`), plus 1.01 transform calls per lookup |

Interval: the counts are deterministic given the trace. Two things go next to them:

- The query-seed spread: three seeds, expected < 0.5%.
- The **CSV greedy-instability band**: rerun C on fb with feature perturbations x ± 1e-6·x², which is cheap
  because the fb root build takes 10 s. At 2M this moved the root by up to 0.36 probes, so any C vs NC-bend
  difference below the band is not a result.

**Figure P (probes).**
- Form: vertical stacked bars, from the bottom: key_at | coordinate | fence | root.
- Groups: the 4 cells within each dataset.
- y axis: comparisons per lookup, 0-40.
- Reference lines: a horizontal line at 27.6 for binary search, and a dotted line at 13.05 for the
  untouchable floor.
- Small text above each bar: transform calls per lookup when nonzero.

### 2.4 Throughput (ns/lookup and ratio to B)

**What NFL can change.**
- It adds a flow evaluation on every lookup routed through a flow region or a flow root. The cost is
  66 ns [56, 77] measured in the index at 2M, or 28.3 ns for a libm tanh pair in a dependency chain.
- It changes fence probes, upward in every measured case.
- It cannot use NFL's batching (8.4 ns/key at batch 256). The benchmark issues dependent single lookups.
  Say this in the caption, because it is the main reason NFL's own throughput claims do not transfer.

**What CSV can change.** Fewer root probes and fewer fence probes, at zero extra work per lookup: the
slot table lookup replaces a clamp.

**The scale argument (a prediction, not established).**
- At 2M the index is instruction-bound and root probes were nearly free in time (CSV root 0.99x).
- At 200M each root probe is two dependent loads: `regions_[m]`, then `->low_fence` in a separately
  allocated 280 B Region (index.hpp:372-378). The bottom ~8 levels of the 16-level search land on distinct
  Regions in a 13.7 MB object set.
- So a removed root probe should be worth more ns at 200M than the ~4 ns it was worth at 2M. The
  probe-equivalent exchange rate (and `--flow-cost 4`) is therefore scale-dependent and must be
  re-calibrated at 200M.

Predicted ratios to B (uninstrumented, 200M):

| cell | fb | planet | resolvable at sigma 6.5%? |
|---|---|---|---|
| N-auto | 1.000 by construction | 1.000 | A/A, use it to measure the noise floor |
| N faithful, bypass 1 | 0.97-0.99 (16% of lookups pay ~66 ns, +1.6 fence probes) | 0.95-0.98 (32% pay) | no; needs about 40-170 repeats per cell |
| N-forced | 0.85-0.93 (66 ns + 9.9 fence probes on ~1,450 ns) | 0.90-0.96 | yes with 4-13 repeats |
| C | 1.05-1.20 (-7.4 root probes, -0.4 fence) | 1.00-1.03 (-2.0 root, -0.57 fence) | fb yes (~4-13 repeats); planet no |
| C, region vp alone | ~1.002 (0.4 fence probes on cache lines already touched) | ~1.003 | **no, by resolution: about 8,000 repeats.** Show it as a null, not a bar |
| NC faithful | = C | = C | layout-identical; second A/A |
| NC-bend | = C | 1.05-1.15 (-12.3 root probes minus one 28-66 ns flow eval) | marginal (~4-13 repeats) |

Existing single-seed evidence (instrumented, 2M lookups; for orientation only):
- fb: the identical root-fences structure measured 0.812 / 0.651 / 0.723 Mops against B (raw root)
  0.611, i.e. 1.07-1.33x.
- planet: CSV-root 1.110x and bend+CSV 1.175x vs the binary root.
- None of this is a claim: the same structure spreads 25%.

**Protocol** (build-fs binary, serial, idle host):
- Flags: `--instrument 0` (the counters add `s->note` on every probe; take probes from a separate
  instrumented pass), `--qos 1`, `--warmup-mode workload --warmup 200000`, `--prefault 0`, `--chunks 16`,
  `--ops 10000000`. The memory audit shows that 200k-op replays are worthless.
- Statistic: the median of chunks 2-16 per process, with chunk 1 vs that median as the warm-up check
  (ignore `steady_state_ratio`).
- Repeats: randomised cell order within each repeat block. The ratio is a paired geometric mean over
  repeats with a 95% percentile bootstrap (10,000 resamples over repeats).
- Repeat budget per cell, from the noise study: 13 for a 5% effect, 4 for 10%. Spend the repeats where the
  prediction is resolvable: fb B/N-forced/C and planet B/NC-bend. Reuse the A/A cells instead of extra B
  runs.
- Cost: planet C and NC-bend rebuild for 2,068-2,973 s each. That is about 8 h at 13 repeats. Either
  accept 3 process repeats x 16 chunks and state ~15% resolution, or time planet only for B, N-forced and
  NC-bend.

**Figure T (throughput).**
- Form: forest plot. One row per cell x dataset; the point is the ratio to B, with a 95% bootstrap CI.
- x axis: log scale, 0.8-1.3, with a vertical line at 1.0.
- A grey vertical band shows the A/A interval (N-auto vs B and NC vs C).
- An open diamond marks the predicted range from the table above (pre-registered).
- A separate reference row shows the plain sorted array: memory_audit has 1.14-1.16 Mops vs packed
  0.73-0.76, about 1.5x.

### 2.5 Build time (s, 16 build threads; root smoothing is single-threaded)

| cell | fb | planet | source |
|---|---|---|---|
| B | 6.4 | 6.3 | measured |
| N | 7.0 (+0.6: 6.4 s of transform thread-time / 16) + offline flow training | 7.0 + training | measured; record train_flow.py wall time per dataset (NFL: ~38 s on an RTX 3080) |
| C | ~165 (region vp 161 + root 3.4) | ~2,250 (vp 190 + root 2,068) | fb measured (161.0 / 183.1); planet sum of measured parts |
| NC faithful (`--fusion auto`) | ~183 (smooths both features) | ~2,400 + flow | fb measured 183.1; planet predicted |
| NC-bend | ~183 | 5,758 (measured, root smoothing on both features) | measured |
| CSV paper, alpha 0.1, 200M | 889-2,902 (ALEX), 337-2,329 (LIPP) | | CSV Tables 3-4 |

**Figure BT (build time).**
- Form: horizontal stacked bars, one per cell x dataset.
- Segments: base build | flow transform (+ hatched offline training) | region smoothing | root smoothing.
- x axis: **log scale**, 1-10,000 s.
- A shaded band shows the CSV paper's 889-2,902 s.
- Intervals: none (single build; builds are compute-bound and their spread is small, ~5%). Report thread
  time next to wall time.
- Message: the root smoothing on planet is O(budget x fences) and dominates everything.

### 2.6 AIDB hardness (fourth panel)

What the metrics are: RMSE, ME (one least-squares line), CD (FMCD max conflicts), PLA-32, PLA-4096.
They are **properties of a sorted (feature, target) sequence**, and AIDB uses them to predict the
*ordering of throughput across datasets*, not the effect of a method within one dataset. Each method
moves them only in its own space:

- **NFL.** The metrics are defined only on z-sorted data, i.e. the dataset AFLI would see. That is NFL's
  space, and our host never uses it. In our host (z in key order, rank) PLA is undefined because x is not
  monotone, and `smooth_regions` throws on unsorted features (hardness.hpp:349). Prediction:
  - CD falls on the conflict-heavy sets, because the tail conflict degree at 2M fell osm 99->6,
    planet 18->3, fb 8->3. Full-file CD is a max, not a 99th percentile, so the size of the fall is not
    predictable.
  - RMSE, ME and PLA have no predicted sign. The inert flow gave -98% to +111% on RMSE, ±5 on PLA.
  - The existing `full_flow` scope used the INERT flows and must be recomputed with the faithful ones:
    about 14 s per dataset.
- **CSV.** It moves the targets (slots), region-wise. Predictions:
  - PLA-32 falls 40-60% on the locally easy sets (history, wise, stack, books, libio, covid) and 0-15% on
    the locally hard ones (fb, planet, osm, genome). The full files keep local structure like the window
    samples, where hard windows showed no PLA-32 cut.
  - PLA-4096 stays within ±5%: smoothing is region-local, and eps = 4096 is region-sized.
  - RMSE and ME rise 6-10%, **purely from slot units** (n becomes about 1.1n slots). Plot RMSE/N_slots
    and ME/N_slots, not raw values, or the panel shows a spurious "CSV makes data harder".
  - Cost: `--virtual-alpha 0.1` at full scope is 3-30 min per dataset (the books 2M sample took 16.6 s
    wall on 489 regions).
- **Combination.** CSV on z-sorted features (the `sample_flow_csv` scope generalised to full), i.e.
  NFL-space only.
- **Host-relevant addition.** The five metrics on the **48,829 root fences** (every 4,096th key; tiny,
  seconds), before and after root virtual fences and the bend. This is the one "dataset" in our host
  where CSV and the bend demonstrably act: fb 15.66 -> 3.42, planet 13.69 vs 3.32.

**Figure H (hardness).**
- Form: five small multiples, one per metric. In each, ten dataset rows, log x axis, dot-and-arrow from B.
- Arrow styles: filled for CSV (host space); hollow and dashed for NFL and NFL+CSV, captioned "z-sorted
  space, not usable by a key-ordered index".
- RMSE and ME normalised by sequence length.
- An inset: Spearman rank correlation, over ten datasets, between Δmetric and Δcomparisons/lookup (C vs B).
  This is the only honest way to say whether AIDB hardness predicts the methods' effect.
- Intervals: none (deterministic). Mark the paper's two printed values (history and libio PLA) as
  reproduced.

---------------------------------------------------------------------------------------------------------

## 3. The single opening figure

**Figure P, extended to one composite: "Where a 200M-key lookup spends its comparisons, before and after
NFL and CSV."**
- Layout: stacked bars of comparisons per lookup (key_at | coordinate | fence | root), the four cells
  B / N / C / NC for fb and planet (all ten once instrumented runs exist).
- Reference lines: binary search at 27.6, and the untouchable floor at 13.05.
- Text above each bar: throughput ratio [95% CI] and total B/key.

It opens the meeting because one picture carries every answer:

- 13 of the ~30 comparisons can never move.
- NFL makes the bar taller or leaves it identical.
- CSV removes up to 12 root comparisons, but only on fb, where the fence set is smoothable; planet needs
  2,000+ s for 2.
- NFL+CSV is the CSV bar again.
- The learned bars dip below binary search in comparisons, while the annotations show they still do not
  win on time and cost +0.8 B/key.

Slide two is the memory bar (values dominate). The compression and hardness panels come after.

---------------------------------------------------------------------------------------------------------

## 4. Show, null, or drop

| metric | NFL | CSV | NFL+CSV | recommendation |
|---|---|---|---|---|
| key arena ratio | null by construction | null by construction | null | one table row; compression shown via the codec survey |
| B/key accounted | null (144 B total) | +0.80 (0.001-0.005 if read-only) | = CSV | show (Figure M) |
| real RSS | null | +0.80 + ~0.1-0.2 capacity slack (pred.) | = CSV | show as dots; drop peak RSS |
| comparisons/lookup | +5% (bypass), +8-34% (forced), 0 (auto) | -27% fb, -8% planet | = CSV; bend -39% planet | show; open with it |
| throughput | 0 (auto, A/A), ~-2% (bypass, unresolvable), -5 to -15% (forced) | fb +5-20%, planet ~0 | = CSV; bend planet +5-15% | forest plot; region-vp-only as an explicit null |
| build time | +0.6 s + training | +160 s (regions), +2,068 s (planet root) | +180 s to 5,758 s | show, log axis |
| AIDB metrics | only in z-space | PLA-32 down, RMSE up in units | z-space | panel 4, normalised, plus the root-fence set |
| `steady_state_ratio` | | | | drop (broken) |

---------------------------------------------------------------------------------------------------------

## 5. Problems found in the existing 200M material

1. **Every existing 200M "flow" run used the inert monotone flows** (results/aidb/flows/<d>_2D2H2L.txt;
   config.json, config_c*.json). No faithful flow has been run at 200M. The N, N-forced and NC rows above
   are therefore predictions extrapolated from 2M ratios.
2. **planet root fusion chose the flow at 200M** (root_flow=true, 111,688 fences, root 3.32). This
   contradicts the "0 of 60 root cells" claim if that claim is quoted without "at 2M". The selector charged
   only `--flow-cost 4`:
   - At the 2M exchange rate the real cost is 6.9-16 probe-equivalents.
   - At 6.9 the choice still holds (3.32 + 6.9 = 10.2 < 13.69).
   - At 16 it flips (19.3 > 15.66).
3. **root_analysis.json is stale.** The deep guide says it lacks planet vf4. root_analysis.txt has all
   16 rows.
4. **The existing timings are instrumented** (`instrument: 1`) with 2M lookups. The memory audit's
   uninstrumented 10M-op runs give 0.73-0.76 Mops vs 0.58-0.65 here (different binaries too). Do not mix
   the two.
5. **CSV is inert under `--routing byte`** (index.hpp:319). Any host tuning toward byte routing silently
   removes CSV.
6. **Region `virtual_features` are kept for compaction only, and their capacity slack is unaccounted.**
   This is a fair one-line rebuttal if the supervisor asks why CSV costs 8% of memory.
