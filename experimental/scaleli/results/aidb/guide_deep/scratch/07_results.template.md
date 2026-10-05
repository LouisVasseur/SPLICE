# 07 — Results per dataset, with interpretation

**What this section gives you.** Every number the study produced, per dataset and per variant, recomputed from the raw result files with the stdlib script `results/aidb/guide_deep/scratch/compute_tables.py` (tables are pasted from its output, not from prose), and for each experiment an interpretation of the *mechanism* that produced the number. The experiments are E1 (hardness metrics on the ten AIDB datasets, raw and transformed, with and without CSV-style virtual points), E2 (region-level sweeps on 2M-key uniform and window samples), E3 (the final sweep with the root ablation, including the sentinel-fence artefact and the withdrawn "synergy"), E4 (region-size ablation), E5 (200M-key runs on fb and planet), E6 (conformance/coverage of the 25 metric compositions against our variants versus the paper's Table 2), a cost decomposition (ns per probe, ns per flow evaluation, break-even batch size), the throughput picture (paired speedups, noise band, the decoding-vs-routing ablation), a one-page "settled / noisy / open" summary with the exact sentences to say, and a set of hostile questions with answers. Everything here is about clean-room CONTROLS inside the SCALE-LI experimental map: an 8-parameter monotone tanh network in NFL's weight format trained by a stand-in trainer (`tools/train_flow.py`), CSV Algorithm 1 written from the paper (`include/scaleli/smoothing.hpp`), and the paper's five scalar hardness metrics (`include/scaleli/hardness.hpp`). Nothing is a reproduction of NFL, AFLI, CSV or the AIDB benchmark, whose six indexes were not run.

Notation (shared with the other sections): keys k₁ < … < kₙ (uint64), rank r(kᵢ) = i − 1; a linear model f(k) = w·φ(k) + b with φ the normalized raw key or the flow output z(k); slot s(kᵢ) = rank plus the number of virtual points before kᵢ; virtual point set V with budget λ = α·n; tail conflict degree D₉₉; region = 4,096 keys, block = 128 keys [index.hpp:44]; root = the whole-range structure over the 489 region fences of a 2M-key sample (⌈2,000,000/4,096⌉ = 489) or the 48,829 fences of a 200M-key file; probes = key comparisons counted by the software counters `root_probes`, `fence_probes`, `coordinate_probes` [index.hpp:124, :135, :350-353], `correction_distance` [index.hpp:151], `transform_calls`.

---

## 0. How to read a number in this section

### 0.1 The three probe counters and "total probes"

A lookup of key k in the packed control does, in order [index.hpp:350-356, :118-151]:

```
root:        binary search over the region fences (low_fence of each region)     -> root_probes
             (489 fences: ⌈log₂ 489⌉ = 9, measured mean 8.956; 48,829 fences: 15.66)
region:      p = predicted_block(k): evaluate f(k) = w·φ(k) + b, then binary-search the
             block whose slot range contains f(k) among the region's 32 blocks       -> coordinate_probes
             (32 blocks: log₂ 32 = 5, measured mean 5.03)
correction:  locate_block(k): if k is not inside block p, walk/binary-search the
             block fences (block.last) until it is                                   -> fence_probes,
             and |true block − p|                                                    -> correction_distance
block:       decode the block's packed codec and find k                              (not a probe counter)
```

`total probes per lookup = (root_probes + fence_probes + coordinate_probes) / operations` [root_analysis.py, `total_probes`]. Counters are deterministic for a given (dataset, seed, structure): the three query seeds of the final sweep never differ in root probes by more than 0.0045 per lookup [results/aidb_final/sweep/results.jsonl, max over 20 samples × 7 variants; key_numbers.json `e3_seed_root_spread`]. That is why every *mechanism* claim below is made on counters and every *time* claim is hedged.

### 0.2 Paired speedup and the "seed spread" bracket

For variant v on sample d, each query seed σ ∈ {11, 29, 47} replays the identical lookup trace (`trace_fingerprint` equal across variants; verified in [results/aidb/verification/recompute/recompute.json `pairing_ok: true`]) against the control:

```
ratio_σ  = throughput_v(d, σ) / throughput_control(d, σ)
speedup  = exp( mean_σ log ratio_σ )                      (geometric mean over the 3 seeds)
bracket  = 2.5 % and 97.5 % quantiles of 2,000 bootstrap resamples of the 3 log-ratios
           (resampling seeds, RNG 42)                     [root_analysis.py paired(); tools/summarize.py:9]
```

With three seeds the bracket is the *seed spread*, not a 95 % confidence interval; with one seed (E5) there is no bracket at all.

### 0.3 The samples the region-level experiments run on

Two 2M-key samples per dataset from the sorted 200M-key GRE file [results/aidb/provenance.json; data/samples/*.manifest.json]: `uniform` = 2,000,000 keys drawn uniformly at random (seed 42), which keeps the global shape and flattens local structure; `window` = one contiguous 2,000,000-key range at a seeded offset, which keeps local structure and loses the global shape. The hardness of the two differs a lot: fb PLA-32 is 3,034 on the uniform sample and 10,599 on the window [results/aidb/hardness.json fb.sample.pla_32; results/aidb_window/hardness.json fb.sample.pla_32].

---

## 1. E1 — Hardness moves: what the transform and the virtual points do to the five metrics

Sources: [results/aidb/hardness.json] (uniform run; `full` and `full_flow` are computed on the whole 200M-key file, `sample*` on the uniform 2M sample), [results/aidb_window/hardness.json] (window run; its `full` scope is byte-identical to the uniform run's, its `sample*` scopes are the window sample). Metric definitions: RMSE and ME of one least-squares fit key → rank [hardness.hpp:84]; CD via the FMCD fit [hardness.hpp:168, :218]; PLA-ε = optimal ε-bounded segment count [hardness.hpp:295]; all as in [AIDB §4.1 "Metrics under Test"].

### 1.1 Full 200M-key files, raw keys

{{T:e1_full_raw}}

### 1.2 Full 200M-key files, transformed keys z(k) (flow trained on the uniform sample), with % change vs raw

{{T:e1_full_flow}}

Worked example (fb RMSE): raw 57,735,023.7 → flow 1,161,273.7; Δ = (1,161,273.7 − 57,735,023.7)/57,735,023.7 = −0.9799 = −97.99 %. fb's raw keys are upsampled 64-bit IDs whose CDF is a step at the low end (min key 1, max 2⁶⁴ − 1 [results/aidb/hardness_details.json fb.full.min/max]); one line through that CDF has an RMSE of 29 % of n; the monotone tanh straightens the global shape, and one line through z(k) has an RMSE of 0.58 % of n. Its ME rises 253 % (99,999,995.5 → 352,904,667.5) because the tanh saturates at the extreme keys: a few keys at the ends are pushed far from the line.

Reading (the "orthogonal axes"):

- The transform moves the *global* metrics. RMSE changes by −97.99 % (fb) to +110.69 % (stack); it falls on 7 of 10 datasets and rises on genome (+11.39 %), libio (+18.19 %) and stack (+110.69 %). ME follows RMSE except on fb. On those three datasets the raw CDF is already close to a line (libio CD = 2, stack CD = 1) and an 8-parameter tanh fitted by maximum likelihood on 4,096 sampled keys [READING_GUIDE §4] bends what did not need bending.
- The transform does not move the *local* metrics: PLA-32 changes by at most 5 segments (planet 613,597 → 613,602), PLA-4096 by at most 4 (stack 133 → 129). CD changes by 164 on osm (4,107 → 3,943), by 11 on planet (21 → 32), and by ≤ 2 elsewhere. A PLA segment is a local object (a run of keys within ε of one line); a global monotone warp of the key axis leaves the number of such runs essentially unchanged.

### 1.3 The 2M samples: sample → sample_flow → sample_csv → sample_flow_csv (α = 0.1)

`sample_csv` = the sample with CSV-style virtual points inserted per 4,096-key region at budget λ = 0.1·n (185,327-199,707 virtual points per 2M-key sample [results/aidb/hardness_details.json <d>.sample_csv.virtual_points]); the metrics are then computed on the augmented key set with the virtual keys included. `sample_flow_csv` = the same on the transformed features.

Uniform samples, PLA-32:

{{T:e1_uniform_pla_32}}

Uniform samples, RMSE:

{{T:e1_uniform_rmse}}

Uniform samples, CD / ME / PLA-4096:

{{T:e1_uniform_conflict_degree}}

{{T:e1_uniform_max_error}}

{{T:e1_uniform_pla_4096}}

Window samples, PLA-32:

{{T:e1_window_pla_32}}

Window samples, RMSE:

{{T:e1_window_rmse}}

Window samples, CD / ME / PLA-4096:

{{T:e1_window_conflict_degree}}

{{T:e1_window_max_error}}

{{T:e1_window_pla_4096}}

Readings:

- Virtual points move the *local* axis. On the uniform samples PLA-32 falls by 43.3-61.5 % on six datasets (books 602 → 236, history 1,067 → 411, fb 3,034 → 1,718, libio 1,256 → 614, stack 593 → 319, wise 787 → 436), by 19.9-25.2 % on covid, genome and planet, and *rises* 5.5 % on osm (7,099 → 7,491). Mechanism: Algorithm 1 inserts points where the slot CDF deviates from the region's line, which makes the augmented CDF straighter *at the scale of a region* (4,096 keys), and a straighter CDF needs fewer ε = 32 segments. On osm the per-region CDF is so rough (mean D₉₉ = 36 per region, Table 3.3) that 410 extra points per region add new small kinks faster than they remove old ones.
- Virtual points raise RMSE by 6.5-10.0 % everywhere on the uniform samples (books +9.0, fb +9.9, osm +10.0, covid +9.7, genome +10.0, history +6.5, libio +8.6, planet +10.0, stack +7.6, wise +9.5). This is arithmetic, not a defect: the augmented set has 1.1·n points and the slots run to 1.1·n, so the global line's residuals scale by ≈ 1.1 when the virtual points do nothing globally; the datasets where the rise is below 10 % (history, stack, libio, books) are those where the per-region insertions happened to straighten the global CDF a little too.
- The two components are close to commuting: `sample_flow_csv` ≈ `sample_csv` on PLA-32 (e.g. books 239 vs 236, fb 1,721 vs 1,718) and ≈ `sample_flow` on RMSE (e.g. fb 12,742 vs 11,585). Each moves its own axis and barely touches the other's.
- Exceptions to keep in mind: (i) on the uniform samples the flow *raises* RMSE on fb (1,448 → 11,585, +700 %), genome, libio and stack, i.e. the sample-trained flow is worse than identity on the fb sample although it is far better than identity on the fb full file; the 2M uniform sample of fb is nearly linear already (RMSE 1,448 = 0.07 % of n) and the tanh's saturation hurts; (ii) on the windows the flow changes PLA-32 by at most 2 segments and CD by at most 12 (osm 152 → 164, genome 171 → 160, fb 96 → 89), and virtual points do *not* reduce PLA-32 on the locally hard windows (fb 10,599 → 11,034, +4.1 %; planet 8,498 → 8,628, +1.5 %; osm −1.3 %; genome −3.0 %) while still cutting it 40-61 % on the easy ones (covid, history, stack, wise). CD on the windows *rises* under virtual points on the easy datasets (libio 2 → 31, stack 1 → 6, history 8 → 21, planet 49 → 124): the FMCD model is fitted to the augmented set, and virtual keys placed in gaps create collisions that did not exist. CD is defined on the key set, and the augmented set is a different key set.

One anomaly to know about, in case the supervisor opens the window file: [results/aidb_window/hardness.json <d>.full_flow] was computed with the *window-trained* flow applied to the *full* 200M-key file, and that flow saturates outside its window: the transformed full files contain 39,473,722 (books) to 168,160,900 (osm) duplicate feature values (`full_flow.duplicates`; planet 0), CD explodes (osm 200,000,000, fb 83,901,921) and RMSE is ≈ 44-45 M on every dataset (the RMSE of a constant model, n/√12 ≈ 57.7 M, is what a fully saturated tanh approaches). Those `full_flow` rows are meaningless and are not used anywhere; the uniform-run `full_flow` values (Table 1.2) are the ones quoted. [results/aidb_window/hardness.json full_flow.duplicates]

---

## 2. E2 — Region-level sweeps (uniform and window 2M samples)

Protocol [results/aidb/sweep/config.json; results/aidb_window/sweep/config.json]: 10 datasets × 10 variants × 3 query seeds (11, 29, 47) = 300 runs per sample mode; read-only; 2M keys bulk-loaded with 16 build threads; 200,000 warm-up lookups then 1,000,000 measured lookups per run; `--verify 0 --instrument 1 --latency 0` (throughput pass and counter pass only); region 4,096 keys, block 128 keys. Binary sha256 3cc4fa88… [results/aidb/sweep/environment.json]. Variants: `packed_rank` (control: packed codecs, per-region rank model, fence binary-search root), `packed_rank_flow` (`--flow <weights> --flow-bypass 1`: the flow is kept in a region iff D₉₉(z) ≤ 0.9·D₉₉(raw) [index.hpp:281]), `packed_rank_flow_forced` (`--flow-bypass 0`), `packed_rank_vp10` (`--virtual-alpha 0.1`), `packed_rank_flow_vp10` (both, bypass on), `packed_rank_fusion_auto` (`--fusion auto --flow … --virtual-alpha 0.1`: the per-region selector over {raw, flow} × {ranks, slots} [index.hpp:239-264]), `packed_rank_flow_costsel` (`--fusion auto` with the flow only), plus `packed_byte`, `raw_rank`, `sorted_vector`.

### 2.1 Fence probes per lookup (the correction work after the region model's prediction)

Uniform samples:

{{T:e2_uniform_fence}}

Window samples:

{{T:e2_window_fence}}

Worked example (fb, uniform): control fence probes = 2,600,293 fence comparisons / 1,000,000 lookups = 2.600; vp10 = 2.181; Δ = (2.181 − 2.600)/2.600 = −16.1 %. In the vp10 structure each block records `slot_begin`, the rank model predicts a slot, and `predicted_block` [index.hpp:118] compares the prediction against slot boundaries; because the slots were chosen to make the region's CDF straighter (Algorithm 1 minimises Σ(w·xᵢ + b − sᵢ)² over K ∪ V), the prediction lands in the right block more often and the correction walk is shorter.

### 2.2 Coordinate probes, correction distance, transform calls

Uniform:

{{T:e2_uniform_coord}}

Window:

{{T:e2_window_coord}}

Readings:

- Coordinate probes are 5.03 in every row: the region always binary-searches its 32 block boundaries (log₂ 32 = 5; the residual 0.03 comes from the search's boundary handling and the shorter last region and was not analysed further). Neither component changes this: it is the cost of finding the predicted block, not of correcting the prediction.
- Correction distance (blocks between predicted and true block) falls 48-83 % on nine uniform samples (books −83.1 %, stack −78.2 %, wise −78.0 %, history −77.9 %, libio −67.7 %, covid −66.9 %, fb −66.0 %, genome −48.9 %, planet −48.3 %) and 14.7 % on osm; on the windows it falls 47-86 % on seven (covid −85.5, wise −85.2, stack −84.0, history −69.2, books −52.9, libio −47.7, genome −47.4) and 15.4 % (fb), 19.7 % (planet), 25.6 % (osm) on the locally hard three. Correction distance is the per-lookup *residual* of the region model measured in blocks; it moves more than fence probes because a fence-probe walk costs ⌈log₂(distance + 1)⌉-ish comparisons, not `distance` comparisons.
- `transform_calls` per lookup is exactly the fraction of lookups that land in a flow region: 0.053 on books_uniform under the bypass (26 of 489 regions accepted, Table 2.3), 1.000 when forced.

### 2.3 Flow acceptance under the bypass rule, selector choices, tail conflict degrees

Uniform:

{{T:e2_uniform_choices}}

Window:

{{T:e2_window_choices}}

Readings:

- The NFL-style bypass [index.hpp:281: keep the flow iff D₉₉(raw) − D₉₉(z) ≥ 0.10·D₉₉(raw)] accepts the flow in 0.20 % (stack) to 8.79 % (planet) of the 489 regions on the uniform samples and 0.0 % (stack) to 4.29 % (osm) on the windows. The mean per-region D₉₉ is 3.0-4.2 on eight uniform samples, 7.66 on planet, 8.45 on fb and 36.2 on osm, and the mean D₉₉ of the transformed features is within ±0.02 of the raw one on every dataset except osm (36.16 → 36.98, worse). A region of 4,096 keys spans ≈ 0.2 % of the key range; over that range a monotone global warp is nearly linear, so z(k) ≈ a·k + c inside the region and the per-region linear model absorbs it. That is the mechanism behind "the transform does nothing at region scale", stated as a fact about the counters (§2.1: forced flow leaves fence probes within 0.0006 of the control on every uniform sample and within 0.0002 on every window; key_numbers.json `e2_forced_maxdiff`).
- The cost-based selector [index.hpp:239-264: cost = mean over the region's own keys of (fence + coordinate probes of the real `locate_block`) + 4 probe-equivalents if the candidate uses the flow] never picks the flow: `fusion_auto` chooses `vp` in 470-489 of 489 regions (osm_uniform 470 = 96.1 %, fb_window 477 = 97.5 %, all others 486-489) and `flow` or `both` in 0; `flow_costsel` chooses `none` in 489 of 489 regions on all 20 samples. The estimated cost of the chosen candidate is 0.22-0.74 probes below the raw-rank cost (e.g. covid 8.088 → 7.351), which is the same saving the measured fence probes show (3.064 → 2.328).
- `flow_vp10` equals `vp10` to three decimals on every sample because the flow is bypassed in 91-100 % of regions and, where accepted, changes nothing measurable.

### 2.4 Rank SSE, memory, build time, preprocessing

Uniform:

{{T:e2_uniform_sse}}

Window:

{{T:e2_window_sse}}

Readings:

- `rank SSE` is the OLS squared error of the region models summed over regions, on ranks (`rank_sse_before`) and on slots after smoothing (`rank_sse_after`) [benchmark.cpp learnability block]. The ratio after/before is the CSV objective realised: 0.046 on books_uniform (a 22× reduction), 0.15-0.30 on fb, history, stack, wise, covid, 0.43-0.60 on libio, planet, genome, and 0.84 on osm. It is 1.000 for the forced flow: swapping φ = raw key for φ = z(k) leaves the region's SSE unchanged to the printed precision, again because z is linear inside a region.
- Virtual points cost 0.092-0.100 doubles per key of metadata (`virtual_points_per_key`), i.e. 0.74-0.80 B/key on top of 10.2-14.5 B/key for the packed control (18 of 20 samples; libio_window 0.68, stack_window 0.25 [results/aidb_window/sweep/summary.csv]). No key bytes are added: the virtual points only move the rank targets and each block records one `slot_begin`.
- Build time (uniform samples; window in the second table): 54-62 ms for the control at 2M keys; 1.95-4.42 s with virtual points (1.65-4.29 s on the windows; smoothing is 28-63 s of summed thread time at 16 threads, i.e. 14-32 µs per key; Algorithm 1 as written is O(λ·n) per region because every greedy round rescans every gap after the refit); 6.9-10.3 s for the fused selector, which smooths both features (103-159 s of thread time). Batched flow inference at build time costs 27.4-32.1 ns per key (`transform_ns / keys`, forced-flow rows).

### 2.5 Throughput and paired speedups (region level)

Uniform (1M lookups × 3 seeds, no QoS):

{{T:e2_uniform_speed}}

Window:

{{T:e2_window_speed}}

Readings (numbers in the tables):

- `vp10` vs control: 0.958-1.072 on the uniform samples (only libio's 1.072 [1.02, 1.13] excludes 1.0) and 0.913-1.049 on the windows (covid_window 0.913 [0.86, 0.94] below 1.0, stack_window 1.049 [1.01, 1.12] above). `flow_vp10`: 1.001-1.045 uniform. `fusion_auto`: 0.951-1.047 uniform. The flow-bearing variants sit at 0.908-1.037. All of this is inside the run-to-run band established in §8.
- The uncompressed `raw_rank` (same routing, no codec decoding) is 1.04-1.53× the control and the plain `sorted_vector` binary search is 1.04-1.59× on the uniform samples: at 2M keys (16 MB of keys) the whole array is cache-resident and decoding dominates; see §8.3.

### 2.6 The three region-level readings in one sentence each

1. Virtual points at α = 0.1 cut fence probes per lookup on all 20 samples, by 4.7 % (stack_window) to 24.0 % (covid_uniform); the saving is ≥ 10 % on 16 samples strictly and 9.98 % on a 17th (books_uniform); the two weakest are stack_window (4.7 %) and fb_window (7.7 %) [Tables 2.1].
2. Forcing the transform into every region leaves fence probes equal to the control's to three decimals on all 20 samples (max |Δ| = 0.0006 uniform, 0.0002 window), while adding exactly one transform evaluation per lookup [Tables 2.1, 2.2].
3. The selector picks the flow in 0 of 9,780 regions (20 samples × 489), and virtual points in 96.1-100 % [Tables 2.3].

---

## 3. E3 — Final sweep and the root ablation

Protocol [results/aidb_final/config.json]: 20 samples (10 datasets × {uniform, window}) × 9 variants × 3 query seeds (11, 29, 47) = 540 paired runs; 500,000 warm-up + 5,000,000 measured lookups per run; performance-core QoS (`--qos 1`); binary `build-final`. Variants add the root ablation to the E2 controls: `packed_rank_root_raw` (`--root model`: one global linear model on the regions' first keys, kept only if its estimated probes beat the fence binary search), `packed_rank_root_flow` (`--root model --flow …`: the same with the flow feature as a candidate), `packed_rank_root_vf4` (`--root model --root-alpha 4`: CSV-style virtual fences on the raw feature, budget 4 × 489 = 1,956), `packed_rank_root_fusion` (all four candidates {raw, flow} × {ranks, fences}), `packed_rank_vp10_root_fusion` (region virtual points + that root). The root selector [index.hpp:362-392] scores each candidate by the probes the real root `locate` spends on the fences and on the fence midpoints, adds `flow_cost` = 4 for a flow candidate [index.hpp:392], and falls back to the binary search if no candidate beats it. The sweep ran twice: the first pass 19:16-20:11, then the five root variants again 20:46-21:17 after the sentinel-fence fix with the binary rebuilt at 20:45 (sha256 c63ceab0…; `sweep/environment.json` still records the pre-fix binary 837daca7… because `--resume` keeps the first record) [READING_GUIDE §5 E3; results/aidb_final/sweep/environment.json]. The 307 pre-fix rows (300 root-variant rows + 7 strays from seed-61 and `fusion_auto` attempts [results/aidb/verification/recompute/recompute.json `extra_runs_outside_grid`]) are archived in `results.stale-root-before-fix.jsonl`.

### 3.1 Per-sample summary (post-fix)

{{T:e3_summary}}

### 3.2 The full ablation table (post-fix; medians over the 3 seeds; `est` = the selector's own estimates for binary / raw+ranks / flow+ranks / raw+fences / flow+fences, flow candidates include the +4 charge)

{{T:e3_root}}

### 3.3 Readings

- **Raw + virtual fences is chosen on 20 of 20 samples** (`chosen` column of `root fences` and `root fusion`) and is the same structure whether the flow is offered or not: `root fusion` and `root fences` have identical root probes on every sample. The flow feature is chosen in **0 of 60 candidate cells** (20 samples × {root_flow, root_fusion, vp10_root_fusion}; every run's `learnability.root_flow` is false [results/aidb_final/sweep/results.jsonl; key_numbers.json `e3_flow_cells` = (180 runs, 0)]).
- **Root probes fall from 8.956 to 2.08-2.84 on 16 samples** (fb_uniform 2.08, stack_window 2.08, books_window 2.10, history_window 2.10, stack_uniform 2.17, history_uniform 2.18, wise_window 2.18, libio_window 2.25, fb_window 2.27, covid_window 2.28, libio_uniform 2.32, genome_window 2.38, covid_uniform 2.44, books_uniform 2.60, wise_uniform 2.65, genome_uniform 2.84) and to 3.17 (planet_window), 3.43 (osm_window), 5.14 (planet_uniform), 8.17 (osm_uniform).
- **Total probes per lookup fall from 16.07-19.15 to 9.19-13.40 on 19 samples**: −34.9 % to −42.8 % on the 16 samples above (fb_window −34.9 %, genome_uniform −35.4 %, …, stack_window −42.8 %), −31.1 % on planet_window, −30.1 % on osm_window, −22.2 % on planet_uniform; osm_uniform goes 19.06 → 18.28 (−4.1 %), essentially unchanged. With region virtual points on top (`vp10 + root fusion`) the totals are 9.09-12.77 on the same 19 (−37.0 % to −44.0 %) and 17.84 on osm_uniform.
- **Virtual fences used**: 1 (fb_uniform, books_window), 13-130 on eleven samples, 206-294 on planet_window and genome_uniform, 662 on books_uniform, 984 on osm_window and 1,956 (the whole budget) on osm_uniform and planet_uniform. The greedy stops early when no insertion lowers the SSE [smoothing.hpp:47], so the count is a measure of how far the fence CDF is from a line.
- **The raw-key root alone** (`root raw`) beats the binary search on 16 samples (estimate 2.27-8.63 vs 8.956) and falls back on books_uniform (11.08), osm_uniform (11.67), planet_uniform (12.59) and osm_window (10.55). The flow-feature root (`root flow`, estimate including the +4 charge) is 6.71-15.84 and never below the raw candidate on the same sample. With the +4 charge removed, flow+fences would edge out raw+fences on 10 of 20 samples, by 0.02-0.18 probes on nine of them (e.g. covid_uniform 2.43 vs 2.56, libio_uniform 2.27 vs 2.46) and by 2.13 on planet_uniform (3.11 vs 5.24) [results/aidb_final/root_analysis.json <sample>.packed_rank_root_fusion.est, fifth entry minus 4 vs fourth entry]; a charge of 4 probe-equivalents ≈ 16 ns is already far below the measured 66 ns per evaluation (§7), so even the planet_uniform case (2.13 probes ≈ 9 ns) does not pay.
- **Build**: 52-60 ms for the control, 54-642 ms with the virtual-fence root (the 0.25-0.64 s cases are the three samples that used 984-1,956 fences: Algorithm 1 over 489 fences with budget 1,956 is ≈ 1,956 × 489 gap rescans), 1.67-4.29 s with region virtual points. **Metadata**: 0.571 B/key (control) → 0.571-0.576 with root fences (4 bytes per fence + per slot: 9,780 bytes for 1,956 fences on a 2M-key sample, i.e. 0.005 B/key), → 0.82-1.37 B/key with region virtual points.
- **Throughput** (last column; also §8): root fences 0.894-1.078, root fusion 0.932-1.076, vp10 + root fusion 0.892-1.093; pooled geometric means over all 60 pairs are 0.992 (root fences), 1.010 (root fusion), 0.993 (vp10 + root fusion). The time effect of removing ≈ 6.9 root probes is not resolvable on this host.

### 3.4 The sentinel-fence artefact and the withdrawn synergy (what the stale rows said and why it was wrong)

What the first pass stored [results/aidb_final/sweep/results.stale-root-before-fix.jsonl]:

{{T:e3_stale}}

(All 20 samples are listed; the pattern: `raw+fences` on 9 uniform samples, `flow+fences` on covid_uniform and on all windows except osm_window, which fell back to the binary search.)

What happened, concretely:

1. Region 0's `low_fence` is the sentinel 0, not its first key [index.hpp:432: `r->low_fence = i ? rows[i].first : 0`]. The first version of `fit_root` fitted the global line through *all* fences including that sentinel. On a sample whose keys start far from 0 (all of covid's keys lie in [1.34 × 10¹⁸, 1.45 × 10¹⁸] [results/aidb/hardness_details.json covid.full.min/max]), the fitted line has to pass near (0, 0) and near (kₘᵢₙ, 1): 93 % of the fitted range is the empty gap before the first real key on covid_uniform (`sentinel_gap_fraction` = 0.930 [results/aidb/verification/synergy/sentinel.json covid_uniform]) and the line is useless on the real fences: raw estimate 13.78 probes, whereas the same fit *without* the sentinel gives 4.63 [sentinel.json covid_uniform.raw_fit_without_sentinel]. On the windows the raw estimates were 14.1-14.4, i.e. worse than the 8.96 binary search, so `raw+ranks` and `raw+fences` (estimates 13.6-14.4) were both rejected.
2. The flow feature z(k) maps the sentinel 0 and the first real key to nearby values (the tanh is flat far below the data), so the flow line was *not* wrecked by the sentinel: the `flow+fences` estimate was 6.7-8.8 including the +4 charge, i.e. 2.7-4.8 uncharged. The selector therefore chose `flow+fences` on 10 of 20 samples (all windows except osm_window, plus covid_uniform), with measured root probes 2.56-4.77 and `transform_calls` ≈ 1.0 per lookup. That looked like a synergy: "the flow feature lets the virtual fences work where the raw feature cannot".
3. It was an artefact of a bug in the *raw* candidate, not a property of the flow. Verification [results/aidb/verification/synergy/sentinel.json: the stored estimates were re-derived from the fences to 1e-12, and re-fitted without the sentinel]: once the root is fitted on the regions' first real keys [index.hpp:366-370], the raw+fences candidate estimates 2.22-2.97 on the 16 samples above, 3.26 and 3.56 on planet_window and osm_window, 5.24 / 8.18 on planet_uniform / osm_uniform, the flow+fences candidate 6.27-7.65 charged (12.20 on osm_uniform), and the flow is chosen 0 of 60 times. The stale `speedup vs (post-fix) control` column also shows what the flow on the lookup path costs: 0.855-0.891 on the nine windows where it was chosen and 0.848 on covid_uniform, i.e. the ≈ 1 transform call per lookup made those runs 11-15 % *slower* than the control despite 4.2-6.4 fewer root probes; that is the raw material of the 66 ns figure in §7.
4. Why it matters for the story: the earlier notes' "+7 to +16 % throughput from flow + fences at the root" and the "root-level synergy" were derived from these rows and are withdrawn [MEETING_NOTES §3]. The corrected result is stronger and simpler: a correctly fitted raw root plus CSV-style fences does everything the flow appeared to do, at zero lookup-path cost.

A worked check you can do at the whiteboard, covid_uniform: binary = 8.956; pre-fix raw = 13.78 (rejected), pre-fix raw+fences = 13.04 (rejected), pre-fix flow+fences = 8.33 = 4.33 + 4 (accepted); post-fix raw = 4.64, raw+fences = 2.56 (accepted), flow+fences = 6.43 = 2.43 + 4 (rejected because 6.43 > 2.56, and would still be rejected at charge 0 because 2.43 < 2.56 is false). [sentinel.json covid_uniform; results/aidb_final/root_analysis.json covid_uniform.packed_rank_root_fusion.est]

---

## 4. E4 — Granularity: does the transform start to matter when regions grow?

Protocol [results/aidb_granularity/config.json]: fb, osm, planet, covid, genome (uniform 2M samples) × regions of 4,096, 16,384 and 32,768 keys × 5 variants (`packed_rank_r*`, `packed_rank_flow_forced_r*`, `packed_rank_vp10_r*`, `packed_rank_flow_vp10_r*` with the flow *forced*, `packed_rank_fusion_auto_r*`) × 2 seeds (11, 29) = 150 runs, 1M lookups after 200k warm-up, no QoS. Baseline of each region size is its own `packed_rank_r*`.

{{T:e4}}

Readings:

- **Flow chosen in 0 of 6,740 regions** = 2 seeds × (5 × 489 + 5 × 123 + 5 × 62) = 2 × 3,370; virtual points chosen in 3,345 of 3,370 per seed (99.3 %); the 25 `none` regions are all osm (19 + 2 + 3) and one genome region at 4,096 [`choices` column].
- **Forced flow stays within 0.051 fence probes of the control at every size** (max = planet at 32,768: 6.682 vs 6.631; everything else within 0.016). So the "region absorbs the curvature" explanation survives an 8× larger region: at 32,768 keys a region spans ≈ 1.6 % of the key range and the tanh is still linear enough over it for the region's own line to absorb it.
- **The virtual-point saving grows with region size** because a single line fits a larger region worse: fb 2.601 → 2.182 (−16.1 %) at 4,096, 3.209 → 2.237 (−30.3 %) at 16,384, 3.692 → 2.251 (−39.0 %) at 32,768; covid −24.0 / −35.9 / −39.1 %; planet −19.4 / −32.0 / −37.5 %; genome −20.8 / −29.6 / −29.9 %; osm −8.6 / −8.5 / −7.3 % (always the smallest). In words: virtual points repair what the line cannot fit; a longer line has more to repair.
- Note the trade the larger region makes: root probes fall 8.96 → 6.97 → 6.00 (fewer fences) while control fence probes rise 2.6 → 3.2 → 3.7 (fb). Virtual points at 32,768 bring fence probes back to 2.25, i.e. to the 4,096-key level, with 2.96 fewer root probes.
- Throughput: vp10 0.937-1.131, forced flow 0.863-1.013 (one transform per lookup), fusion 0.935-1.095; two seeds per cell, so the brackets are wide.

---

## 5. E5 — Full scale: 200M keys, fb and planet

Protocol [results/aidb_fullscale/config*.json]: the full sorted GRE files (200,000,000 keys), single seed 11, 200,000 warm-up + 2,000,000 measured lookups, QoS on, 16 build threads, binary `build-root` (sha256 c63ceab0…, post-fix). Variants: the packed control, `root_raw`, `root_flow`, virtual-fence roots at `--root-alpha 0.04` (≈ 1,950 fences, the absolute count the 2M runs spent at α = 4) and at α = 4 (195,316 fences), `root_fusion` (α = 4), and region vp10 combined with each root. At the time of writing (2026-09-22) **14 of 16 planned runs are done**; `config_c2` (planet `root_fusion` at α = 4) is running (47 minutes elapsed when checked) and `config_c3` (planet `vp10_root_fusion` at α = 4) is queued, each under a 4-hour cap [results/aidb/run_chain3.log; `ps`]. The table below was recomputed with `python3 results/aidb/root_analysis.py results/aidb_fullscale/sweep/results.jsonl --json results/aidb/guide_deep/scratch/fullscale_root_analysis.json` and the per-run script; `results/aidb_fullscale/root_analysis.txt` predates the planet α = 4 run.

{{T:e5}}

Readings:

- **Both files load into 48,829 regions** (⌈200,000,000/4,096⌉), so the fence binary search costs 15.66 root probes (⌈log₂ 48,829⌉ = 16). Coordinate probes stay at 5.03. Control fence probes are 5.14 on fb and 3.99 on planet, i.e. fb at 200M keys is harder per region than its 2M uniform sample (2.60) and close to its window (5.17), as the PLA-32 numbers in §1.3 predicted.
- **fb**: the raw-key root alone estimates 10.80 and measures 10.81 (−31 % root probes); raw key + virtual fences estimates 3.50, measures 3.42, with **973 fences**, and the greedy stops at 973 whether the budget is 1,953 (α = 0.04) or 195,316 (α = 4), so the two budgets coincide on fb; total probes 25.82 → 13.59 (−47.4 %). The flow feature estimates 21.02 (charged) against 10.80 raw and is never chosen (`root_fusion` picks raw+fences, 973 fences, identical counters). Region virtual points cut fence probes 5.14 → 4.74 (−7.8 %) and, with the root, total probes to 13.19 (−48.9 %).
- **planet**: the raw-key root estimates 25.80 probes and the ≈ 1,950-fence root (α = 0.04) 25.18, both worse than the binary search's 15.66, so both fall back and the counters equal the control's. **At α = 4 (finished 2026-09-22 morning) the greedy used the full budget of 195,316 fences and the raw+fences root estimates 13.75, measures 13.69** (−12.6 % root probes; total 24.68 → 22.70, −8.0 %) at a build cost of 2,067.8 s (34 min; Algorithm 1 over 48,828 fences with 195k greedy rounds, single-threaded at the root) and 0.575 B/key. planet's fence CDF is not "one line plus a few kinks": its keys are OSM node IDs with large empty ID ranges, so it needs fences at the density of the regions themselves before a line works, which is the failure mode CSV's own preprocessing cost (O(λ·n) per model) makes expensive. Region virtual points cut planet's fence probes 3.99 → 3.42 (−14.3 %).
- **Build**: 6.3-7.0 s for the control and the rank/flow roots, 10.0 s with fb's 973 root fences, 26.7 s for fb `root_fusion` (it smooths both features and evaluates the flow on 48,829 fences), 161-190 s with region virtual points (2,407-2,903 s of summed smoothing thread time at 16 threads, i.e. 12-15 µs per key), 2,067.8 s for planet's 195k-fence root. **Metadata** 0.570 B/key → 0.571 (973 fences) / 0.575 (195,316 fences) / 1.364-1.370 (region virtual points: 19,847,006-19,970,703 virtual points ≈ 0.1 per key).
- **No time claim.** Single seed, and the same fb `raw+fences, 973` structure was measured three times as `root_vf004` 0.651, `root_fusion` 0.723 and `root_vf4` 0.812 Mops against a 0.648 Mops control: a 25 % spread between identical structures at one seed (0.812/0.651 = 1.247). The 1.00-1.25 "ratios" in the table are therefore not evidence of anything. The same applies to planet's 0.53-0.64.

---

## 6. E6 — Conformance and coverage of the 25 metric compositions

Definitions [AIDB §3.2 eq. (1)-(3); tools/aidb_scores.py:82 `harder`, :87 `classify_pairs`, :100 `coverage`, :105 `conformance_of_variant`]: for a d-dimensional metric h, Sᵢ is harder than Sⱼ iff hₖ(Sᵢ) ≥ hₖ(Sⱼ) for all k with strict inequality for at least one k; C = the comparable ordered pairs, U = the incomparable unordered pairs, N = C(10, 2) = 45. Cov = (|C| − |U|)/N. For index I with throughput pᴵ normalised by its standard deviation over the ten datasets, p̂ᴵ(S) = pᴵ(S)/σᴵ, each comparable pair (i, j) gets an importance wᵢⱼ = sigmoid(p̂ᴵ(Sᵢ) − p̂ᴵ(Sⱼ)); a pair is conforming when the harder dataset has the lower throughput; Confᴵ = (R − P)/(R + P) with R = Σ w over conforming pairs and P over violating pairs, and Conf = mean over indexes. Here "indexes" are our ten E2 variants on the uniform samples (their `throughput_ops_s` medians [results/aidb/throughput.json]) and the hardness values are the scope's metrics; the paper's six indexes were not run.

Uniform run, scope `full` (hardness of the raw 200M-key files; the 25 compositions in the paper's Table 2 order; paper values from [AIDB §4.2 Table 2, columns Conf and Cov]):

{{T:e6_uniform_full}}

Window run, scope `full` (same hardness, throughputs of the window sweep):

{{T:e6_window_full}}

Conformance of each variant on the five scalar metrics, uniform run, scope `full`:

{{T:e6_variants_full}}

Orderings of the ten datasets by each raw scalar metric on the full files (the coverage of a scalar metric is 1.0 by construction; the *orderings* are what the compositions' coverage depends on):

{{T:e6_orderings}}

Readings:

- **Coverage matches the paper's Table 2 (to its two printed decimals) for all 15 CD-free compositions** in both runs: the 5 scalars (1.00), PLA-32·PLA-4096 0.467 vs 0.47, RMSE·ME 0.822 vs 0.82, RMSE·PLA-32 0.600 vs 0.60, RMSE·PLA-4096 0.422 vs 0.42, ME·PLA-32 0.600 vs 0.60, ME·PLA-4096 0.422 vs 0.42, RMSE·ME·PLA-32 0.511 vs 0.51, RMSE·ME·PLA-4096 0.333 vs 0.33, RMSE·PLA-32·PLA-4096 0.244 vs 0.24, ME·PLA-32·PLA-4096 0.244 vs 0.24. Coverage is index-independent (it depends only on the ten hardness vectors), so this is evidence that our RMSE, ME, PLA-32 and PLA-4096 *orderings* of the ten datasets equal the paper's (a match of ten coverages is consistent with, not a proof of, identical orderings). Our PLA-32/PLA-4096 for history (105,468 / 468) and libio (145,808 / 639) are the values the meeting notes report as matching the two the paper prints [MEETING_NOTES §1] [unverified: the paper's printed values were not located in the text extraction used here].
- **Every CD-containing composition has coverage lower than the paper's by exactly 2/45 = 0.044**: RMSE·CD 0.422 (|C| = 32, |U| = 13) vs 0.47 (= 0.467 = (33 − 12)/45), ME·CD 0.511 (34/11) vs 0.56 (35/10), CD·PLA-32 0.556 (35/10) vs 0.60 (36/9), CD·PLA-4096 0.378 (31/14) vs 0.42 (32/13), and likewise for the six 3-D compositions (−0.040 to −0.049 after the paper's rounding). Moving one pair from C to U changes Cov by (−1 − 1)/45 = −2/45: **one dataset pair that the paper's CD orders is tied or reversed in ours**, in every composition that contains CD. The paper prints no CD values, so the pair cannot be identified; the FMCD port [hardness.hpp:168] was read against LIPP's source without finding a discrepancy, and it keeps 64-bit key differences exact where LIPP uses doubles, so a ±1 difference in one CD is the likeliest cause (e.g. a tie such as libio 2 vs stack 1 becoming an order, or vice versa; the raw CDs are stack 1, libio 2, history 8, wise 10, planet 21, covid 27, fb 110, books 246, genome 585, osm 4,107 [Table 1.1]). This is the one open implementation question.
- **Conformance is not comparable to the paper's** and is not claimed to be: our "indexes" are ten variants of one structure whose throughputs differ by a few percent between variants and by ≈ 1.5× across datasets, so p̂ gaps are small and the sigmoid weights sit near 0.5 for every pair; the mean Conf of the scalar metrics is −0.41 to +0.18 (uniform, full) against the paper's 0.08-0.50. What *is* readable: CD is the only scalar with positive conformance for the packed control (0.472 uniform, 0.512 window) and for every packed variant, and adding CD to any composition raises Conf (e.g. RMSE −0.239 → RMSE·CD 0.431 for the control); PLA-4096 and RMSE are negatively conforming for our structure because our per-region model already removes global curvature, so the datasets the global metrics call hard (fb, planet, books by RMSE) are not slow for us (fb's control throughput, 2.64 Mops, is fifth of ten, above books, covid, genome, osm and wise [results/aidb/throughput.json packed_rank]). The verifier's independent implementation of eq. (1)-(3) agrees with `aidb_scores.py` to 4 × 10⁻¹⁶ [results/aidb/verification/scores/verify_scores.py; MEETING_NOTES §1].
- The other scopes (`full_flow`, `sample`, `sample_flow`, `sample_csv`, `sample_flow_csv`; files e6_*_<scope>.md in the scratch tables) score the same throughputs against the transformed or augmented hardness; they are diagnostics of how much each component changes the *orderings* (e.g. `full_flow` matches the paper's coverage on 9 of 25 compositions only, because the transform reorders RMSE and ME) and are not compared to Table 2.

---

## 7. Cost decomposition: what a probe costs, what the flow costs, when batching would pay

Sources: [results/aidb/chart_data.json "cost"], [results/aidb/verification/coststory/coststory.txt] (computed on the 540 first-pass rows, seeds 11/29/47, which is where the flow was actually on the lookup path), [MEETING_NOTES §3].

Method [coststory.txt "GROUP A" / "GROUP B"]: with ns/lookup = 10⁹/throughput (verified equal to elapsed/ops to 0.0000 ns), take every (sample, seed) pair of a *flow-free* model root against the control; S = root probes saved, Δns = ns(control) − ns(variant). Regress Δns on S:

```
through the origin (45 raw/vf4 pairs, dataset-cluster bootstrap):  ns per probe = 4.07  [2.61, 5.53]
with intercept (117 no-flow pairs):                                 Δns = +3.4 − 4.45 · S   (slope [2.56, 7.27])
pooled  Σ(−Δns)/ΣS = 4.02
```

Then take the 60 pairs where the flow *was* on the path (pre-fix `root_fusion` vs `packed_rank`, `vp10_root_fusion` vs `vp10`; T ≈ 1.014 transform calls per lookup, mean S = 5.15 probes saved): mean Δns = +46.7 ns [+40.0, +53.2] *slower*. The implied cost of one flow evaluation F = (Δns + b·S)/T:

```
b = 0 (probes are free):    F = 46.0 ns  [39.4, 52.4]
b = 4 ns/probe:             F = 66.4 ns  [59.8, 72.6]   = 16.6 probe-equivalents
b = 8 ns/probe:             F = 86.7 ns  [79.8, 93.4]
joint bootstrap (A and B resampled): F = 66 ns [56, 77]    <- the figure quoted everywhere
```

The numbers, with what they mean:

| quantity | value | source |
|---|---|---|
| ns per root probe (what one saved probe buys) | 4.1 [2.6, 5.5] | chart_data.json cost.ns_per_probe |
| one flow evaluation on the lookup path (2H2L, unbatched, our host) | 66 ns [56, 77] | cost.flow_eval_ns |
| root probes the transform can save at most (8.96 → 2.2-2.6 at 489 fences) | 4.2-6.4 | cost.probes_saved; coststory §C5 "max possible root saving ≈ 6.5" |
| net loss per lookup when the flow is on the path | 47 ns [40, 53] | cost.net_loss_ns |
| batched flow inference at build time, our host | 27-34 ns/key | cost.build_transform_ns_per_key; forced-flow rows 27.4-32.1 (Table 2.4) |
| NFL's Table 2, 2H2L (8 parameters), ns per key by batch size | 1: 169.53, 8: 40.60, 32: 15.28, 128: 9.52, 256: 8.38, 1024: 7.40, 2048: 7.29 | [NFL §3.2.2 Table 2] |

Break-even argument. The most the transform can buy at the root here is S × (ns per probe) = 6.4 × 4.1 = 26.2 ns per lookup (bounds 4.2 × 2.6 = 10.9 to 6.4 × 5.5 = 35.2 ns). Unbatched it costs 66 ns [56, 77], so it loses 30-66 ns per lookup, and the calibrated selector (charge 4 probe-equivalents ≈ 16 ns, already generous) is right never to pick it. On NFL's own cost curve (their hardware, MKL, batched inference) the 2H2L flow costs 169.5 ns at batch 1, 40.6 at batch 8 and 15.28 at batch 32: it crosses the 26.2 ns budget between batch 8 and batch 32 and reaches 8.38 ns at NFL's default batch size of 256 [NFL §4.1.3]. So the only regime in which the transform could pay *at the root of this structure* is batched lookups of ≥ 32 keys, and even then it competes with virtual fences that reach the same 2.1-2.8 root probes for free at lookup time. At region level there is nothing to buy (§2): S = 0.

Two qualifications: (i) the 66 ns figure is our unbatched C++ evaluation of a 2-input, 2-hidden tanh network with the [x, x − ⌊x⌋] encoding and the sum decoder [transform.hpp:64], not NFL's MKL kernel; (ii) the regression pools all datasets and uses the pre-fix rows, but the flow's cost is a property of the evaluation, not of which root was chosen, and the post-fix rows contain no on-path flow to re-measure it with.

---

## 8. Throughput overall

### 8.1 The strip of paired speedups per variant (final sweep, 5M lookups × 3 seeds, QoS on)

{{T:e8_speed}}

Pooled geometric means over all 60 (sample, seed) pairs, with the extreme single pair: region vp10 0.992 (0.88-1.14), raw root 0.977 (0.82-1.19), flow root 0.964 (0.76-1.14), root fences 0.992 (0.77-1.14), root fusion 1.010 (0.86-1.16), vp10 + root fusion 0.993 (0.79-1.24), raw_rank 1.208 (0.83-1.86), sorted_vector 1.678 (0.91-2.66) [key_numbers.json `e8_pooled`].

Per sample: region vp10 0.918 (osm_uniform) to 1.062 (planet_uniform); 15 of the 20 seed-spread brackets include 1.0 (those that do not: genome_uniform 1.053 [1.01, 1.09], genome_window 1.041 [1.01, 1.10], planet_uniform 1.062 [1.06, 1.07] above; osm_uniform 0.918 [0.90, 0.95], wise_uniform 0.937 [0.90, 0.99] below) [MEETING_NOTES §1 says 16 of 20] [unverified: the count depends on the bootstrap's resampling, my recount of root_analysis.json's own lo/hi gives 15]. Root fences 0.894 (osm_uniform) to 1.078 (history_uniform); root fusion 0.932 to 1.076.

### 8.2 The noise band: how far apart two measurements of the same structure land

Two ways to measure it from the final sweep itself:

- `root fusion` vs `root fences` when both chose raw+fences (all 20 samples): the root and fence counters are identical; the only difference is that `root fusion` has the flow file loaded and evaluates it in the 0.0-8.8 % of regions the region-level bypass accepted (`transform_calls` per lookup 0.0006-0.088). Per-seed ratios span **0.842 to 1.245** (60 pairs); per-sample geometric means span **0.942 (history_uniform) to 1.092 (books_uniform)** [key_numbers.json `e8_noise_same_structure`, `e8_noise_same_per_sample`].
- `raw root` that fell back to the binary search vs the control (identical lookup path, 4 samples × 3 seeds): per-seed ratios 0.824-1.029, per-sample 0.903 (planet_uniform), 0.904 (osm_window), 0.913 (osm_uniform), 0.948 (books_uniform) [`e8_noise_fallback`]. All four are below 1.0, which is consistent with a session effect: the root variants were measured in the second pass (20:46-21:17) and the control in the first (19:16-20:11) [READING_GUIDE §5 E3], so every root-variant speedup in §8.1 is a cross-session pair.
- The pre-fix verifier had 75 such sham pairs: ratio mean +0.3 %, sd 5.8 %, range −10.2 to +18.9 % [coststory.txt "NOISE FLOOR"].

Conclusion: per-dataset time effects below ≈ 6-10 % are not resolvable here, and single pairs can differ by 25 %. The meeting notes' "up to 8 % (10 % for the binary-fallback root)" [MEETING_NOTES §3] is the per-sample statement; the per-seed extremes are wider.

### 8.3 Where the time goes: the uncompressed-rank ablation

{{T:e8_decode}}

`sorted_vector` (plain binary search over the uncompressed 2M-key array, no index at all) is **1.147× (stack_window) to 2.294× (books_window) faster than the packed control** on every sample; `raw_rank` (the same routing as the control, i.e. the same root, region models and fences, but uncompressed blocks) is 0.988-1.560×. Splitting the gap ns(packed) − ns(sorted_vector) into the part that removing the codec recovers (ns(packed) − ns(raw_rank)) and the rest (ns(raw_rank) − ns(sorted_vector), the routing structure itself): by per-seed pairing and median over seeds the decoding share is 0 % (clipped; history/libio/stack, where raw_rank is not faster than packed) to 86 % (wise_uniform), and the routing share is the larger of the two on 10 of 20 samples [computed here; MEETING_NOTES §4 states "0-56 %" and "larger on 13 of 20" with a different aggregation] [unverified: the split is inside the noise band of §8.2 on most samples]. Two things are solid: (i) at 2M keys the array is cache-resident and a 21-probe binary search (log₂ 2,000,000) at 150-270 ns beats every packed variant, which spends 290-440 ns; (ii) the packed control's 16-19 probes are not its cost; the codec decoding and the block work are, which is why removing 6.9 root probes (≈ 28 ns at 4.1 ns/probe, 7-9 % of a 330-440 ns lookup) is invisible inside a ±10 % band. This ordering need not hold at 200M keys (1.6 GB of keys, not cache-resident), and the E5 runs are single-seed, so nothing is claimed there.

---

## 9. What is settled, what is noisy, what is open — and the sentence to say

**Settled (deterministic counters, reproduced by independent verifiers from the raw files):**

1. "On the full 200M-key files the transform changes RMSE by −98 % (fb) to +111 % (stack) and PLA-32 by at most 5 segments; on the 2M samples virtual points at α = 0.1 cut PLA-32 by 43-61 % on six datasets and raise RMSE by 6-10 %: the two components act on different hardness axes." [results/aidb/hardness.json; Tables 1.2, 1.3]
2. "Virtual points cut fence probes per lookup on all 20 samples, by 4.7 % to 24.0 %, ≥ 10 % on 16 of them, and correction distance by 15-86 %." [results/aidb/sweep/summary.csv, results/aidb_window/sweep/summary.csv; Tables 2.1, 2.2]
3. "Forcing the transform into every region leaves fence probes equal to the control's to three decimals on all 20 samples and at region sizes up to 32,768 keys; the cost-based selector chooses the flow in 0 of 9,780 regions at 4,096 keys and 0 of 6,740 in the granularity ablation." [Tables 2.1, 2.3, 4]
4. "With the root fitted on real keys, the raw key plus CSV-style virtual fences is chosen on 20 of 20 samples, cuts root probes from 8.96 to 2.1-2.8 on 16 of them, and total probes per lookup by 35-43 % on those 16 (22-31 % on three, −4 % on osm_uniform); the flow feature is chosen in 0 of 60 cells." [results/aidb_final/root_analysis.json; Table 3.1]
5. "At 200M keys fb's root goes from 15.66 to 3.42 probes with 973 virtual fences (total −47 %); planet's root needs 195,316 fences and 34 minutes of preprocessing to go from 15.66 to 13.69; region virtual points cut fence probes 8 % (fb) and 14 % (planet)." [results/aidb_fullscale/sweep/results.jsonl; Table 5]
6. "Coverage equals the paper's Table 2 for all 15 CD-free metric compositions; every CD-containing composition is lower by exactly one pair of 45." [results/aidb/scores.json; Table 6]
7. "The first-pass root-level 'synergy' was an artefact of fitting the raw root through region 0's sentinel fence; it is withdrawn and the pre-fix rows are archived." [results/aidb_final/sweep/results.stale-root-before-fix.jsonl; results/aidb/verification/synergy/sentinel.json; §3.4]

**Noisy (time; say only with the band):**

8. "Throughput of every region-level and root-level variant is within the host's noise band: paired speedups 0.89-1.09 while two measurements of the same structure differ by up to 9 % per sample and 25 % per seed pair; no per-dataset time claim below ≈ 10 % is made." [results/aidb_final/root_analysis.json; §8.1-8.2]
9. "One unbatched flow evaluation costs 66 ns [56, 77] on the lookup path, a saved root probe is worth 4.1 ns [2.6, 5.5], and the transform can save at most 6.4 root probes: it loses 30-66 ns per lookup unbatched and would break even only at NFL's batch sizes ≥ 32." [chart_data.json cost; coststory.txt; NFL Table 2; §7]
10. "Binary search over the uncompressed array is 1.15-2.29× faster than the packed control at 2M keys; decoding and block work, not routing probes, dominate lookup time there." [Table 8.3]

**Open:**

11. "One CD ordering of one dataset pair differs from the paper's; the paper prints no CD values, so it cannot be identified; the FMCD port was checked against LIPP's source without finding a discrepancy." [§6]
12. "planet at α = 4: root_fusion and vp10 + root_fusion are still running (4-hour cap each); the finished root_vf4 run says what they will show for the counters (raw+fences, 195,316 fences, 13.69 root probes) because the flow candidate estimates 27.87 against 25.80 raw+ranks and cannot win." [results/aidb_fullscale/sweep/results.jsonl planet root_flow est; run_chain3.log]
13. "Whether the probe savings turn into time at 200M keys (non-cache-resident) is untested: single seed, 25 % spread on identical structures." [§5]
14. "Whether a higher-capacity or per-node transform (NFL's 2H4L-4H4L, or one flow per region) would be chosen is untested; our 8-parameter monotone flow trained on 4,096 keys by a stand-in trainer is the weakest reasonable instance of NFL's idea." [READING_GUIDE §3.1]

---

## 10. Questions the supervisor may ask, with answers

1. **"Isn't this just showing your transform is badly trained?"** Partly, and the study is designed so that it does not matter for the conclusion. The flow *does* what NFL says a flow should do on the global axis: fb's full-file RMSE falls 98 % (57.7 M → 1.16 M), planet's 48 %, history's 47 % [Table 1.2]. It fails at region scale for a structural reason, not a training reason: a 4,096-key region spans 0.2 % of the key range, over which any smooth monotone warp is linear, and the region's own line absorbs it, which is why the *forced* flow changes fence probes by < 0.001 [Table 2.1] and the SSE ratio is exactly 1.000 [Table 2.4]. Growing the region to 32,768 keys (1.6 % of the range) does not change that [Table 4]. At the root, where scale is global, the flow is beaten not because it is bad but because the raw key plus virtual fences is already at 2.1-2.8 probes; a flow cannot do better than ≈ 2 probes on 489 fences, and it costs 66 ns to evaluate. A better-trained flow would move the root estimates from 6.7-15.8 towards 2-3, i.e. to a tie at a 66 ns cost.

2. **"Would NFL's real flow do better?"** NFL trains a BNAF by maximum likelihood on a normal target with variance 10¹⁶ and evaluates it batched with MKL at 8.38 ns per key at batch 256 [NFL §4.1.3; Table 2]. Better trained: yes, probably on the global axis; but the region-scale argument (question 1) is independent of training quality, and at the root the best a flow can buy is 6.4 probes × 4.1 ns = 26 ns per lookup, so it only pays if it costs less than that, i.e. batched at ≥ 32 keys on NFL's curve [§7]. NFL's AFLI is a different index (model nodes, buckets and dense nodes sized by the conflict degree), where the flow's effect on the node fan-out is the point; our structure has no such node to shrink.

3. **"Why is the time flat if probes drop 40 %?"** Because probes are not where the time goes. A root probe is worth 4.1 ns [2.6, 5.5] [§7]; 6.9 saved probes are ≈ 28 ns of a 290-440 ns lookup (7-9 %), inside a noise band of ±6-10 % per sample [§8.2]. The uncompressed binary search with 21 probes runs in 150-270 ns [Table 8.3]: decoding the packed blocks and the block work cost more than all the routing probes together at 2M keys.

4. **"Does any of this hold at 200M keys?"** The counters do: 48,829 regions, 15.66 root probes for the binary search, fb's root to 3.42 with 973 fences (−47 % total probes), planet's to 13.69 only with 195k fences [Table 5]. The time does not (single seed, 25 % spread on identical structures). Two things change at 200M: the array is 1.6 GB (not cache-resident, so probes that miss cache may cost more than 4 ns), and planet-like fence CDFs need fences at the density of the regions before a line helps.

5. **"What would change your conclusion?"** (a) Batched lookups: at batch ≥ 32 on NFL's cost curve the flow's 26 ns budget is met [§7]; (b) a host where routing rather than decoding dominates, e.g. non-cache-resident data, where a saved probe might be worth a cache miss (≈ 100 ns) rather than 4 ns; (c) a per-region or higher-capacity flow that changes the region SSE, which ours never does; (d) an index whose node size depends on the conflict degree (AFLI, LIPP), where the transform's CD effect (osm 4,107 → 3,943, planet 21 → 32) matters structurally.

6. **"How do I know the counters are right?"** They are exact integer counters incremented in the locate routines [index.hpp:124, :135, :350-353]; across the three query seeds root probes agree to 0.0045 per lookup [§0.1]; the root selector's estimate (computed at build time on fences and midpoints) and the measured root probes agree to ≈ 0.1 on every sample (e.g. fb_uniform 2.22 vs 2.08, covid_uniform 2.56 vs 2.44; [Table 3.2 `est` vs `root/op`]); differential tests exercise every option [tests/test_main.cpp]; five verifier agents recomputed the ablation, the cost story and the synergy from the raw rows [results/aidb/verification/].

7. **"The synergy you announced then withdrew: what exactly was wrong?"** The raw root was fitted through region 0's sentinel fence at key 0. On windows whose first key is ≈ 10¹⁸ the fit is dominated by an empty gap (93 % of the fitted range on covid), so the raw candidates were rejected at 13-14 probes while the flow candidate, whose tanh is flat below the data, was not hurt and won at 6.7-8.8. Fitting on the regions' first real keys [index.hpp:366-370] gives raw+fences 2.2-3.6 and the flow loses 60 of 60 cells [§3.4; sentinel.json]. The pre-fix rows are archived and the pre-fix timing claims (+7 to +16 %) are withdrawn.

8. **"Why 2M-key samples, and why two kinds?"** The index runs at 2M for time (300-540 runs per sweep); hardness is exact on the full 200M keys; E5 checks the root story at 200M on two datasets. Uniform samples keep the global shape (fb RMSE 0.07 % of n, PLA-32 3,034) and flatten local structure; windows keep local structure (fb PLA-32 10,599, fence probes 5.17 vs 2.60) and lose the global shape. The virtual-point saving appears on both; the root story appears on both; the exceptions (osm, planet at the root; fb, planet PLA-32 on windows) are the locally hard cases.

9. **"How reliable is the hardness computation?"** PLA counts are identical to PGM-index's `make_segmentation` for ε ∈ {0, 1, 8, 32, 128, 1024, 4096} on the fixture, synthetic sets, 5M-key prefixes of all ten files and the full books and wise files [results/aidb/verification/pla/]; RMSE/ME/CD agree with independent Python oracles [verification/metrics/vlib.py]; the scorer agrees with a second implementation of eq. (1)-(3) to 4 × 10⁻¹⁶ [verification/scores/]; and the coverage column reproduces the paper's Table 2 for every CD-free composition [Table 6].

10. **"Your CD differs from the paper's on one pair. Which?"** Unknown: the paper prints no CD values. The port keeps 64-bit differences exact where LIPP's FMCD uses doubles, so a ±1 difference on a small CD is plausible (stack 1, libio 2, history 8, wise 10, planet 21, covid 27 are the small ones). It affects only the CD-containing compositions, by exactly 2/45 of coverage.

11. **"What does CSV's preprocessing cost, and is it practical?"** Algorithm 1 as written is O(λ·n) per model because every greedy round rescans every gap after the refit. Per 4,096-key region at α = 0.1 that is 14-32 µs of thread time per key at 2M keys and 12-15 µs at 200M (161-190 s wall with 16 threads) [Tables 2.4, 5]. At the root it is 1,956 rounds over 489 fences (≤ 0.64 s) at 2M keys but 195,316 rounds over 48,828 fences at 200M (2,067.8 s on planet). The CSV paper's own preprocessing times at α = 0.1 on 200M keys are quoted in the meeting notes as 889-2,902 s [MEETING_NOTES §4] [unverified: not located in the CSV text extraction used here]. Inserts reuse the bulk-load decisions (`relearn_on_compaction = false`); re-running the search on every compaction made the 10 %-insert runs of the learnability sweep 116-232× slower [results/learnability/summary.csv, `packed_rank_vp10_relearn`].

12. **"Why does osm resist everything?"** osm's per-region CDF is the roughest of the ten: mean D₉₉ 36 per region (others 3-8), full-file CD 4,107, PLA-4096 5,495 [Tables 1.1, 2.3]. Virtual points at 10 % cannot straighten it (fence probes −8.6 %, SSE ratio 0.84, PLA-32 +5.5 %), 19 regions keep the plain rank model, and at the root the fence CDF needs all 1,956 fences to get from 8.96 to 8.17 probes on the uniform sample. It is the dataset where "harder" means "rough at every scale", and no linear-plus-kinks model helps.

13. **"Is any throughput difference real?"** Only the ones far outside the band: `sorted_vector` 1.15-2.29× and `raw_rank` up to 1.56× vs the packed control [Table 8.3]. Everything the two components do sits at 0.89-1.09, and identical structures measured twice sit at 0.94-1.09 per sample [§8.2]. The honest statement is "not resolvable on this host", not "no effect".

14. **"What would you run next, and what would it cost?"** (a) planet's two remaining α = 4 runs (queued, ≤ 4 h each); (b) a pinned-core, multi-seed rerun of E5 to put a bracket on the 200M-key time; (c) a batched-lookup benchmark mode to test the break-even at batch 32-256; (d) the 2H4L/4H3L/4H4L flow sizes from NFL's Table 2 as candidates, which changes nothing at region scale by the argument in question 1 but would settle the root-level tie.

15. **"Are you reproducing NFL, CSV or the AIDB benchmark?"** No. The transform is an 8-parameter monotone network in NFL's weight format trained by a stdlib stand-in, not the BNAF trainer (GPL-3, MKL), and AFLI is not run; CSV has no public code and Algorithm 1 was written from the paper and checked against an exhaustive toy oracle; the AIDB paper's six indexes are not run and only its protocol (metrics and eq. (1)-(3)) is used. Every result is a clean-room control inside the SCALE-LI experimental map [README.md; docs/APPROACHES.md].
