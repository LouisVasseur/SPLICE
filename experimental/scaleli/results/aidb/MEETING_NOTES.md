# Fusion evidence: NFL-style transform + CSV-style virtual points (2026-09-21, verified draft)

Everything below is a reduced-scale, single-host, clean-room control study inside the SCALE-LI
experimental map. The transform is an 8-parameter monotone tanh network in the NFL weight format,
trained here by a stand-in trainer; virtual points follow CSV Algorithm 1 per region; nothing is a
reproduction of NFL, AFLI, CSV or the AIDB paper. Datasets: the ten 200M-key sets of the AIDB 2026
hardness paper (GRE mirror; seven of the ten are served unsorted and were sorted once; provenance and
sha256 of every input are in results/aidb/provenance.json). Every number in this document was
recomputed by independent verifier agents from the raw result files (verdicts and their scripts are
archived under results/aidb/verification/); their qualifications are incorporated.

## 1. Settled (deterministic) findings

**Hardness metrics agree with the paper.** PLA-32 / PLA-4096 on the full files: history 105,468 / 468
and libio 145,808 / 639, matching the two values the paper prints (PLA-4096 exactly; PLA-32 to the three
digits the paper gives, 105k and 146k). The coverage column of our conformance/coverage table equals the
paper's Table 2 for all ten CD-free metric combinations, which is consistent with (not a proof of)
identical RMSE, ME, PLA-32 and PLA-4096 orderings of the ten datasets. Every CD-containing metric has
coverage lower than the paper's by exactly 2/45, i.e. one dataset pair that the paper's CD orders is tied
or reversed in ours; the paper does not print CD values, so the pair cannot be identified. The FMCD port
was read against LIPP's source without finding a discrepancy; this remains an open implementation
question. PLA counts were compared against PGM-index's make_segmentation (identical for eps 0, 1, 8, 32,
128, 1024, 4096 on the fixture, four synthetic sets and a 5M-key real prefix) and the conformance/
coverage scorer against a second implementation of eq. (1)-(3) (agreement 4e-16); both checks are
archived in results/aidb/verification/.

**The two components act on different hardness axes.** On full data the transform changes RMSE by -98%
(fb 57.7M -> 1.16M) to +111% (stack 0.84M -> 1.77M): it cuts RMSE on 7/10 datasets and raises it on
genome, libio and stack. PLA-32 moves by at most 5 segments (planet) and PLA-4096 by at most 4; CD moves
by up to 164 (osm 4107 -> 3943) and 11 (planet 21 -> 32), and by at most 2 elsewhere. On the 2M uniform
samples virtual points at a 10% budget cut PLA-32 by 43-61% on six datasets (books 602 -> 236, history
1067 -> 411, fb 3034 -> 1718, libio, stack, wise), by 20-25% on covid, genome and planet, and raise it 6%
on osm; RMSE rises 6-10% everywhere. Figure: results/aidb/fusion_report.html, "orthogonal moves" panel.

**Inside the index, virtual points reduce the work; the transform does not.** Exact fence probes per
lookup (uniform 2M samples, identical traces): fb 2.60 -> 2.18, covid 3.06 -> 2.33, osm 5.08 -> 4.64,
genome 3.30 -> 2.61, planet 3.23 -> 2.60; correction distance falls 48-83% on nine datasets and 15% on
osm. Forcing the transform into every region leaves the probes identical to the control on all ten
datasets; the cost-based selector chooses the flow in 0% of regions and virtual points in 96-100%
(osm_uniform 96%, fb_window 97.5%, the rest 99-100%). A plausible reason is scale: a region is 4096 keys
(0.2% of the key range) and a per-region linear model can absorb the curvature an 8-parameter monotone
tanh expresses; the granularity ablation (fb, osm, planet, covid, genome; regions of 4,096, 16,384 and 32,768
keys; 150 runs, two seeds per cell) still selects the flow in 0 of 6,740 regions, forced flow leaves fence
probes equal to the control's within 0.05, and the virtual-point saving grows with region size (fence
probes 9-24% lower at 4,096 keys, 9-36% at 16,384, 7-39% at 32,768; osm always the smallest).

**Contiguous 2M-key windows (local structure preserved; 300 paired runs).** The transform moves PLA-32
by at most 2 segments and CD by at most 12 (osm 152 -> 164, genome 171 -> 160, fb 96 -> 89). Virtual
points at a 10% budget do not reduce PLA-32 on the locally hard windows (fb 10,599 -> 11,034, planet
8,498 -> 8,628, osm/genome -1 to -3%) while still cutting it 40-60% on the easy ones (history, wise,
stack, covid). Fence probes per lookup, control -> virtual points: books 2.95 -> 2.44, fb 5.17 -> 4.77,
osm 4.37 -> 3.77, covid 2.60 -> 2.05, genome 3.05 -> 2.55, history 2.45 -> 2.10, libio 3.07 -> 2.55,
planet 4.62 -> 4.09, stack 2.08 -> 1.98, wise 2.32 -> 2.02. Flow accepted in 0-4% of regions, forced
flow changes no probe count, selector: 0% flow.

**Throughput of the region-level controls is within noise.** Uniform samples, clean rerun (300 paired
runs, 1M lookups x 3 seeds): virtual points 0.96-1.07 vs the packed control (only libio's 1.07 [1.02, 1.13]
excludes 1.0); flow + virtual points 1.00-1.05; fused selector 0.95-1.05. Final sweep (5M lookups x 3
seeds, performance-core QoS): virtual points alone 0.92-1.06, 16/20 intervals include 1.0. Run-to-run
variation between variants with identical probe counts was +-5-15% in the 1M-lookup runs and remains
+-6% (per seed up to +-13%) with QoS; the cause is not established (the host is shared and uncontrolled),
so per-dataset throughput effects below ~6% are not resolvable here. Memory: packed 10-15 B/key vs 16.7
B/key raw; virtual points add 0.8 B/key of metadata.

**Root level (the whole-range model).** With the root fitted on real keys, CSV-style virtual fences on the
raw key feature cut root probes from 8.96 to 2.1-2.8 on 16/20 samples (total probes per lookup -35 to
-43% on those 16; -22 to -31% on planet_uniform, osm_window and planet_window; osm_uniform unchanged); the
NFL-style flow feature is never chosen; the time effect is not resolvable within the host noise (section 3).

## 1b. The transform, retrained (2026-09-22): our first flows were inert

The ten flows used in sections 1-3 were trained on 4,096 keys for 200 steps with `--monotone`, and they are
inert by NFL's own criterion: measured with the repo's evaluator, the tail conflict degree of the whole 2M
sample is unchanged on nine of ten datasets (osm stays at 99) and only planet improves (18 -> 9). Nine of the
ten weight files carry almost identical weights regardless of dataset, with tanh arguments confined to about
[0, 0.37], i.e. the near-linear regime. Any earlier "the transform does nothing" claim was therefore a
statement about our trainer, not about NFL.

Retrained on 16,384 keys for 2,000 steps (`--shifts 512`, lr 0.05), with and without `--monotone`
(results/aidb_flowv2, 140 paired runs, ten uniform samples, two seeds):

| dataset | raw D99 | order-preserving flow | unconstrained flow (NFL-faithful) | inverted pairs | control fence probes | forced-flow fence probes |
|---|---|---|---|---|---|---|
| osm | 99 | 99 | 6 | 321 | 5.08 | 5.54 (+9%) |
| planet | 18 | 9 | 3 | 513 | 3.23 | 5.34 (+65%) |
| fb | 8 | 8 | 3 | 513 | 2.60 | 7.60 (+192%) |
| genome | 5 | 5 | 4 | 65 | 3.29 | 3.78 (+15%) |
| books | 4 | 4 | 3 | 65 | 2.23 | 2.86 (+28%) |
| covid | 3 | 3 | 3 | 65 | 3.06 | 3.63 (+19%) |
| history | 3 | 3 | 3 | 65 | 2.46 | 3.10 (+26%) |
| libio | 3 | 3 | 3 | 65 | 2.55 | 3.17 (+24%) |
| stack | 3 | 3 | 3 | 65 | 2.29 | 2.96 (+29%) |
| wise | 3 | 3 | 3 | 65 | 2.42 | 3.01 (+25%) |

Four findings, and they replace the earlier region-level transform claim:

1. **NFL's claim reproduces.** The unconstrained flow brings every dataset to a tail conflict degree of 3-6,
   which is NFL's stated "around 4" (§3.3), including osm 99 -> 6 and planet 18 -> 3.
2. **Under an order-preserving constraint the shape cannot help, at any training budget.** A 100-configuration
   sweep on five datasets (4,096 to 200,000 training keys, 200 to 10,000 steps, shifts 8 to 65,536, two
   learning rates) moved the held-out tail conflict degree by zero on fb, books and covid, by 2% on osm
   (83 -> 81) and not at all beyond what the 2.4-second default already gave on planet. The negative log
   likelihood moves over a factor of 30 across those configurations, so the optimizer works; the tail conflict
   degree simply does not follow it. The structural reason: with the fractional weights zeroed, z is a sum of
   two tanh of the same scalar, i.e. one monotone global bend, and the tail conflict degree is defined against
   a refitted least-squares line and is therefore invariant to any affine map of z. Local key clustering lives
   at a scale thousands of times finer than one bend. This is not an undertrained model; it is the shape.
3. **The reduction comes entirely from the fractional feature, which reorders keys.** The feature
   `x - floor(x)` is a sawtooth, so the composed map is not monotone: the number of inverted pairs tracks
   `--shifts` almost exactly (shifts 8 -> 9 inversions per 250k keys, 512 -> 513, 65,536 -> 9,347 on osm).
   NFL has the same property by construction and accepts it: Algorithm 3.1 claims only a 1-to-1 mapping,
   AFLI is built over the transformed keys sorted by z, and the paper evaluates lookups and insertions only,
   never a range scan over original keys (guide_deep/01_nfl.md §3.7).
4. **In a key-ordered map the faithful, non-monotone flow is a loss.** Our index keeps records in original key
   order, so exact search, fences and scans survive a non-monotone feature, but the model gets worse: forcing
   the unconstrained flow raises exact fence probes on all ten datasets, by 9% (osm) to 192% (fb). The
   cost-based selector chooses the flow in 0 regions, while NFL's own bypass rule, which looks at conflict
   degree rather than probes, would accept it in up to 157 of 489 regions (planet) and make those regions
   worse.

Two objections are closed by the paper itself. Our flow is not too small: NFL's deployed flow is "two layers,
two input dimensions, two hidden dimensions" (§4.1.3), the same shape we use, though its B-NAF carries biases
our eight-weight stand-in does not. And our flow is not undertrained: NFL samples 10% of the bulk-loaded keys
three times, and our sweep shows 200,000 training keys and 10,000 steps change nothing under the constraint
that matters.

So the correct statement is not "the transform does nothing". It is: the transform's benefit is real and
reproducible on NFL's own metric, it exists only in a map that reorders keys, and an index that stores records
in key order therefore cannot collect it. That is a composability limit between the two designs, and it is why
the cost-based selector is the right harness: it declines the flow on evidence rather than on a heuristic.
Figure: results/aidb/figures.html, "Retraining the transform", and the deck slide "b-r0".

Correction to section 1: the "transform changes RMSE by -98% (fb 57.7M -> 1.16M)" reading is outlier clipping
at full-file scope, not useful curvature. fb's full file spans 1 to 2^64-1, the tanh is saturated there, and
the maximum error gets 3.5x worse (9.99e7 -> 3.53e8) while the conflict degree is unchanged at 110. On the 2M
samples the flow raises fb's RMSE 8x (1,448 -> 11,585).

## 1c. The joint objective (2026-09-22): tested and refuted

Proposed: optimise the transform and the virtual points TOGETHER rather than in sequence, minimising
sum_i (a f(x_i) + b - s_i)^2 over a monotone f and a budgeted virtual point set V, by block coordinate
descent. Tested on the 489 root fences of all ten uniform samples, five methods at three budgets, scored
with a faithful Python replica of Index::fit_root's candidate score (it reproduces osm's measured root
probes to three decimals, 8.181 vs 8.18). Everything is in results/aidb_joint/.

**1. Alternating buys nothing.** Largest joint-minus-sequential gain over all 30 cells: 0.072 probes
(books). Five datasets give exactly 0.000, three are negative. The pre-registered prediction was a gain
above 0.5 probes on planet; measured -0.001. The optimiser is working (on books the objective falls 56%
over rounds 2-3) but loss and probe count are decoupled at this scale: rounds >= 2 buy 0.0% of planet's
loss reduction and 0.2% of osm's.

**2. The formulation is not novel and we must concede it.** Li, Chen, Ding, Zeng and Zhou, "A Pluggable
Learned Index Method via Sampling and Gap Insertion" (arXiv:2101.00808, 2021), which is CSV's own
reference [16], states this objective in its equation 2, solves it by block coordinate descent over
(model, positions), and has a paragraph "Gap Insertion for Non-Linear Models" covering any monotonically
increasing model. Only the model class, the measurement and the root setting would be new. Outside
indexing the pattern is older still: ACE (Breiman and Friedman 1985) alternates monotone transforms of
predictor and response, and companding quantisation carries the warning that input warp and output level
placement can be the same degree of freedom.

**3. One real positive, and it settles a question the tail-conflict-degree null did not.** A single
order-preserving tanh pair fitted to the least-squares objective, with ZERO virtual fences, cuts root
probes: books 11.08 -> 3.73, planet 12.59 -> 6.23, history 3.78 -> 2.31, wise 5.75 -> 4.19. Three of those
go from losing to binary search (8.96) to beating it. So a single global monotone bend IS a lever on the
root probe count, even though 100 training configurations proved it powerless on the tail conflict degree.
Conflict degree is local and affine-invariant in z; root probes are global and are not.

**4. It still does not pay, once charged.** No flow_cost was charged in the table above; the deployed
fit_root charges 4.0 probes and would select no flow candidate at all. The verifier benchmarked the real
cost in a serial dependency chain, which is the right measurement because the root prediction is on the
critical path: two tanh via libm cost 28.3 ns = 6.9 probe-equivalents, an algebraic sigmoid 12.5 ns = 3.05,
a 64-segment monotone piecewise-linear table about 1.8 ns marginal = 0.45. The budget it must come in
under is sequential minus CSV-alone per dataset: planet 2.41 probes, books 0.31, then 0.19 and below. The
tanh pair misses by 20x on books and 3x on planet. Only the piecewise-linear table is cheap enough, and
only on planet, and a monotone piecewise-linear warp with a segment table IS a segmented spline root, i.e.
the method collapses into the ordinary alternative.

**5. A caveat that lands on our own CSV numbers.** The adversarial verifier found the greedy is
chaotically sensitive at the scale of these differences: a physically meaningless monotone perturbation of
the feature, x +/- eps*x^2 with eps = 1e-6, moves CSV-alone at budget 4x by up to 0.361 probes (books),
0.263 (wise), 0.197 (genome). Every sequential-minus-CSV delta except planet's lies inside its own
dataset's instability band, and the sign flips with the sign of the perturbation. Our headline CSV results
are far outside this (total probes -39 to -44%, root 8.96 -> 2.1-2.8), so they stand; differences of a few
tenths of a probe between CSV variants do not. On four datasets (genome, libio, stack, fb) the fitted warp
is affine to within 1e-5 of its range, so there "sequential" is CSV with extra steps.

**Verdicts.** Adversarial verifier PARTIAL: metric identity, monotonicity and budget accounting all pass
and it independently reproduced 19 of 20 cells to four decimals from the C++ headers; stability and
affine-collapse fail. Practicality verifier REFUTED: each alternation round costs about 2,000 s at the
200M root to buy at most 0.072 probes, a 30,000-to-1 ratio. Files: results/aidb_joint/ (prototype, the
independent reimplementation under verify/, derivation.md, related_work.md, workflow_result.txt).


## 2. Equations actually used

- Region model: rank ~ w*phi(k) + b, phi = normalized key, or z(k) = sum_h tanh(((k-mu)/sigma) W0)_h * v_h.
- Virtual points (CSV eq. 4 / Alg. 1): minimize sum_i (w x_i + b - slot_i)^2 over K u V, |V| <= alpha*n,
  greedy insertion into gaps with refit; O(1) candidate scoring from running sums, SSE = S_yy - S_xy^2 / S_xx.
- NFL-style bypass: positions p_i = floor(s x_i + c) under one linear model with capacity 1.5n; tail conflict
  degree D = 99th percentile of collisions per position minus one; keep the flow iff D_flow <= 0.9 D_raw
  (our margin; NFL's rule is a plain comparison).
- Fusion selector (per region): C(F,T) = mean over the region's own keys of (fence probes + coordinate
  probes of the real locate_block) + c_flow [F = flow]; argmin over {raw, flow} x {rank, smoothed}.
- Learned root: candidates {raw, flow} x {ranks, virtual-fence slots (CSV smoothing on the fence features,
  budget alpha_root * regions, slot -> region table)}; fitted on the regions' first real keys; C = mean
  probes of the real root locate on fences and fence midpoints (+ c_flow for the flow feature); the
  candidate with the smallest C is kept only if C < probes of the fence binary search.
- Hardness: RMSE/ME from one least-squares fit key->rank; CD via LIPP's FMCD fit; PLA-eps = optimal
  eps-bounded segment count (O'Rourke / PGM). Conformance/coverage: paper eq. (1)-(3).
- Speedup: geometric mean over the 3 query seeds of per-seed throughput ratios on identical traces; the
  bracketed ranges are a 3-seed bootstrap and should be read as the seed spread, not as 95% confidence
  intervals (identical index structures measured twice differ by up to 8%, 10% for the binary-fallback root).

## 3. Root ablation (final sweep; root fitted on real first keys)

540 paired runs (20 dataset samples x 9 variants x 3 query seeds, 5M lookups each, performance-core QoS;
results/aidb_final/root_analysis.json). Region routing baseline: fence binary search, 8.96 probes per
lookup. Candidates at the root: {raw key, flow feature} x {ranks, virtual fences (budget 4x regions)},
scored by the probes the real root locate spends on fences and midpoints, binary fallback.

- Deterministic outcome (identical across seeds): a raw-feature root with virtual fences is chosen on all
  20 samples and lowers root probes to 2.1-2.8 on 16 of them (planet_window 3.2, osm_window 3.4,
  planet_uniform 5.1, osm_uniform 8.2). Total probes per lookup (root + fence + coordinate) fall from
  16-19 to 9.2-13.4 on 19 samples (-35 to -43% on the 16 samples above; -31%, -30% and -22% on
  planet_window, osm_window and planet_uniform); osm_uniform stays at 18.3. Virtual fences used: 1-294
  on most samples, 984-1956 on osm/planet. The flow feature is chosen in 0 of 60 candidate cells: with a
  correctly fitted raw feature it is never competitive (estimates 6.7-15.8 against 2.2-3.6 for fences).
- The earlier "synergy" (first pass) was an artifact of fitting the raw root through region 0's sentinel
  fence at key 0; verified, corrected, and the pre-fix rows are archived as
  results.stale-root-before-fix.jsonl.
- Throughput (paired vs binary root, geometric mean over seeds): root fences 0.99 (0.89-1.08),
  root fusion (same structure, second measurement) 1.01 (0.93-1.08), region vp + root 0.99 (0.89-1.09).
  Two measurements of the identical structure differ by up to 8% on the same sample (10% for the
  binary-fallback root), so the time effect
  of removing ~7 root probes is not resolvable on this host; the earlier +7 to +16% readings from the
  pre-fix sweep were within that noise and are withdrawn. Binary search over the uncompressed array stays
  1.15-2.29x faster than every packed variant: decoding and block work dominate, root probes do not.
- The transform on the lookup path costs 66 ns [56, 77] per evaluation (pooled, pre-fix data, unchanged
  by the fit); it can save at most 6.9 root probes; a calibrated selector never picks it at this scale.

- Full scale (200M keys, single seed 11, 2M lookups; results/aidb_fullscale/root_analysis.txt; 13 of 16
  runs done): both files load into 48,829 regions, so the fence binary search costs 15.66 root probes. fb:
  raw-key root 10.81; raw key + virtual fences 3.42 with 973 fences (the greedy stops at 973 under a 1,953
  or a 195k budget); total probes 25.8 -> 13.6 (-47%); the flow feature is never chosen (estimate 21.0 vs
  10.8 raw); region virtual points cut fence probes 5.14 -> 4.74. planet: raw-key root (estimate 25.8) and
  a 1,950-fence root (25.2) are both worse than binary and fall back; region virtual points cut fence probes
  3.99 -> 3.42. At budget 4 x regions the planet root spends the entire budget (195,316 virtual fences,
  2,068 s of single-threaded smoothing) for root probes 15.66 -> 13.69 (-13%; total 24.7 -> 22.7): one global
  line over 48,829 planet fences is not smoothable at this budget, unlike fb (973 fences, 15.66 -> 3.42).
  The remaining budget-4 planet runs (root fusion, vp10 + root fusion) are still running. Build 6-7 s control,
  10 s with root fences, 161-190 s with region virtual points; metadata 0.57 -> 1.37 B/key. The identical fb
  fences structure measured three times gave 0.65, 0.72 and 0.81 Mops against a 0.65 Mops control (25%
  spread at one seed): no 200M-key time claim.

## 4. Caveats to state

- Uniform 1% samples flatten local structure (tail conflict degree 3-8 on most datasets); windows keep it
  but lose the global shape. Full-scale hardness is exact; full-scale index runs (fb, planet, single seed;
  section 3) are done for 14 of 16 planned runs and no 200M-key throughput claim is made. At 200M keys the root smoothing at budget
  4 x regions (195k greedy rounds over 48,828 fences; Algorithm 1 is O(budget x fences)) did not finish in
  2 hours on planet, so the full-scale root runs use budget 0.04 x regions (about 1,950 fences, the absolute
  count the 2M runs spent) on both datasets plus budget 4 on fb; planet at budget 4 is queued with a 4-hour
  cap. The CSV paper itself reports 889-2,902 s of preprocessing at alpha 0.1 on 200M keys.
- A plain binary search over the sorted array is 1.15-2.3x faster than the packed control (1.14-2.2x faster
  than the best packed variant) at 2M keys, where the whole array is cache-resident. Removing decoding alone
  (uncompressed rank routing) recovers 0-56%; the remaining 5-90% is the routing structure, which is the
  larger part on 13 of 20 samples. Decoding is a major but not the dominant cost, and this ordering need
  not hold at 200M keys.
- Preprocessing: smoothing is O(alpha n^2 / regions) per region (12-32 us/key of summed thread time at
  alpha 0.1; the fused selector, which smooths both features, costs 2-4x that); batched flow inference at
  build time is 27-34 ns/key; compactions reuse bulk-load decisions.
- Files: uniform 2M: results/aidb/sweep/summary.csv, results/aidb/report.html; windows: results/aidb_window/;
  final sweep: results/aidb_final/sweep/ (results.jsonl; the pre-fix root rows are archived as
  results.stale-root-before-fix.jsonl); granularity: results/aidb_granularity/ (complete, summary_r*.csv); full scale:
  results/aidb_fullscale/ (in progress); fusion verdicts: results/aidb/fusion_report.html; verification:
  results/aidb/verification/verdicts.json and the verifiers' scripts.
