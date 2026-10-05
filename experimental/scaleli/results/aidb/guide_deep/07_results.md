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

| dataset | RMSE raw | ME raw | CD raw | PLA-32 raw | PLA-4096 raw |
|---|---|---|---|---|---|
| books | 18,053,637 | 96,380,338 | 246 | 262,604 | 97 |
| fb | 57,735,024 | 99,999,996 | 110 | 1,055,308 | 1,687 |
| osm | 24,177,498 | 67,384,215 | 4,107 | 661,115 | 5,495 |
| covid | 1,795,722 | 8,133,078 | 27 | 81,908 | 850 |
| genome | 7,531,940 | 19,622,982 | 585 | 1,290,208 | 1,426 |
| history | 815,542.7 | 2,303,085 | 8 | 105,468 | 468 |
| libio | 3,445,773 | 11,828,007 | 2 | 145,808 | 639 |
| planet | 29,771,622 | 60,462,050 | 21 | 613,597 | 2,314 |
| stack | 839,727.0 | 2,444,700 | 1 | 17,833 | 133 |
| wise | 2,200,402 | 4,988,461 | 10 | 79,035 | 382 |

### 1.2 Full 200M-key files, transformed keys z(k) (flow trained on the uniform sample), with % change vs raw

| dataset | RMSE flow (Δ%) | ME flow (Δ%) | CD flow (Δ%) | PLA-32 flow (Δ%) | PLA-4096 flow (Δ%) |
|---|---|---|---|---|---|
| books | 15,985,128 (-11.46%) | 78,944,811 (-18.09%) | 245 (-0.41%) | 262,604 (+0.00%) | 94 (-3.09%) |
| fb | 1,161,274 (-97.99%) | 352,904,667 (+252.90%) | 110 (+0.00%) | 1,055,308 (+0.00%) | 1,687 (+0.00%) |
| osm | 23,977,602 (-0.83%) | 59,074,284 (-12.33%) | 3,943 (-3.99%) | 661,115 (+0.00%) | 5,495 (+0.00%) |
| covid | 1,272,461 (-29.14%) | 4,762,463 (-41.44%) | 27 (+0.00%) | 81,909 (+0.00%) | 850 (+0.00%) |
| genome | 8,390,196 (+11.39%) | 21,430,457 (+9.21%) | 587 (+0.34%) | 1,290,208 (+0.00%) | 1,425 (-0.07%) |
| history | 432,676.2 (-46.95%) | 1,062,899 (-53.85%) | 8 (+0.00%) | 105,469 (+0.00%) | 467 (-0.21%) |
| libio | 4,072,387 (+18.19%) | 13,901,622 (+17.53%) | 2 (+0.00%) | 145,811 (+0.00%) | 638 (-0.16%) |
| planet | 15,545,176 (-47.79%) | 33,806,026 (-44.09%) | 32 (+52.38%) | 613,602 (+0.00%) | 2,313 (-0.04%) |
| stack | 1,769,181 (+110.69%) | 4,286,206 (+75.33%) | 1 (+0.00%) | 17,833 (+0.00%) | 129 (-3.01%) |
| wise | 1,511,721 (-31.30%) | 3,874,673 (-22.33%) | 9 (-10.00%) | 79,036 (+0.00%) | 382 (+0.00%) |

Worked example (fb RMSE): raw 57,735,023.7 → flow 1,161,273.7; Δ = (1,161,273.7 − 57,735,023.7)/57,735,023.7 = −0.9799 = −97.99 %. fb's raw keys are upsampled 64-bit IDs whose CDF is a step at the low end (min key 1, max 2⁶⁴ − 1 [results/aidb/hardness_details.json fb.full.min/max]); one line through that CDF has an RMSE of 29 % of n; the monotone tanh straightens the global shape, and one line through z(k) has an RMSE of 0.58 % of n. Its ME rises 253 % (99,999,995.5 → 352,904,667.5) because the tanh saturates at the extreme keys: a few keys at the ends are pushed far from the line.

Reading (the "orthogonal axes"):

- The transform moves the *global* metrics. RMSE changes by −97.99 % (fb) to +110.69 % (stack); it falls on 7 of 10 datasets and rises on genome (+11.39 %), libio (+18.19 %) and stack (+110.69 %). ME follows RMSE except on fb. On those three datasets the raw CDF is already close to a line (libio CD = 2, stack CD = 1) and an 8-parameter tanh fitted by maximum likelihood on 4,096 sampled keys [READING_GUIDE §4] bends what did not need bending.
- The transform does not move the *local* metrics: PLA-32 changes by at most 5 segments (planet 613,597 → 613,602), PLA-4096 by at most 4 (stack 133 → 129). CD changes by 164 on osm (4,107 → 3,943), by 11 on planet (21 → 32), and by ≤ 2 elsewhere. A PLA segment is a local object (a run of keys within ε of one line); a global monotone warp of the key axis leaves the number of such runs essentially unchanged.

### 1.3 The 2M samples: sample → sample_flow → sample_csv → sample_flow_csv (α = 0.1)

`sample_csv` = the sample with CSV-style virtual points inserted per 4,096-key region at budget λ = 0.1·n (185,327-199,707 virtual points per 2M-key sample [results/aidb/hardness_details.json <d>.sample_csv.virtual_points]); the metrics are then computed on the augmented key set with the virtual keys included. `sample_flow_csv` = the same on the transformed features.

Uniform samples, PLA-32:

| dataset | sample | sample_flow (Δ%) | sample_csv (Δ%) | sample_flow_csv (Δ%) |
|---|---|---|---|---|
| books | 602 | 602 (+0.0%) | 236 (-60.8%) | 239 (-60.3%) |
| fb | 3,034 | 3,034 (+0.0%) | 1,718 (-43.4%) | 1,721 (-43.3%) |
| osm | 7,099 | 7,099 (+0.0%) | 7,491 (+5.5%) | 7,468 (+5.2%) |
| covid | 1,192 | 1,193 (+0.1%) | 892 (-25.2%) | 892 (-25.2%) |
| genome | 2,045 | 2,045 (+0.0%) | 1,627 (-20.4%) | 1,629 (-20.3%) |
| history | 1,067 | 1,068 (+0.1%) | 411 (-61.5%) | 412 (-61.4%) |
| libio | 1,256 | 1,255 (-0.1%) | 614 (-51.1%) | 614 (-51.1%) |
| planet | 3,456 | 3,455 (-0.0%) | 2,767 (-19.9%) | 2,771 (-19.8%) |
| stack | 593 | 594 (+0.2%) | 319 (-46.2%) | 321 (-45.9%) |
| wise | 787 | 787 (+0.0%) | 436 (-44.6%) | 430 (-45.4%) |

Uniform samples, RMSE:

| dataset | sample | sample_flow (Δ%) | sample_csv (Δ%) | sample_flow_csv (Δ%) |
|---|---|---|---|---|
| books | 180,591.3 | 159,882.7 (-11.5%) | 196,846.2 (+9.0%) | 174,288.9 (-3.5%) |
| fb | 1,448.2 | 11,584.6 (+699.9%) | 1,591.2 (+9.9%) | 12,742.1 (+779.8%) |
| osm | 241,476.1 | 239,472.1 (-0.8%) | 265,521.5 (+10.0%) | 263,263.9 (+9.0%) |
| covid | 18,073.1 | 12,906.4 (-28.6%) | 19,819.2 (+9.7%) | 14,147.7 (-21.7%) |
| genome | 75,352.0 | 83,934.3 (+11.4%) | 82,874.1 (+10.0%) | 92,313.6 (+22.5%) |
| history | 8,104.9 | 4,290.7 (-47.1%) | 8,628.3 (+6.5%) | 4,846.6 (-40.2%) |
| libio | 34,472.4 | 40,748.0 (+18.2%) | 37,442.8 (+8.6%) | 44,356.9 (+28.7%) |
| planet | 297,556.5 | 155,367.6 (-47.8%) | 327,255.8 (+10.0%) | 170,882.5 (-42.6%) |
| stack | 8,395.8 | 17,705.8 (+110.9%) | 9,036.9 (+7.6%) | 19,180.3 (+128.5%) |
| wise | 21,926.3 | 15,026.4 (-31.5%) | 24,008.5 (+9.5%) | 16,401.8 (-25.2%) |

Uniform samples, CD / ME / PLA-4096:

| dataset | sample | sample_flow (Δ%) | sample_csv (Δ%) | sample_flow_csv (Δ%) |
|---|---|---|---|---|
| books | 11 | 9 (-18.2%) | 10 (-9.1%) | 9 (-18.2%) |
| fb | 24 | 24 (+0.0%) | 23 (-4.2%) | 23 (-4.2%) |
| osm | 1,102 | 1,100 (-0.2%) | 1,099 (-0.3%) | 1,099 (-0.3%) |
| covid | 8 | 8 (+0.0%) | 10 (+25.0%) | 11 (+37.5%) |
| genome | 77 | 82 (+6.5%) | 71 (-7.8%) | 73 (-5.2%) |
| history | 7 | 7 (+0.0%) | 21 (+200.0%) | 21 (+200.0%) |
| libio | 8 | 7 (-12.5%) | 21 (+162.5%) | 16 (+100.0%) |
| planet | 26 | 33 (+26.9%) | 37 (+42.3%) | 33 (+26.9%) |
| stack | 8 | 7 (-12.5%) | 7 (-12.5%) | 8 (+0.0%) |
| wise | 8 | 7 (-12.5%) | 9 (+12.5%) | 8 (+0.0%) |

| dataset | sample | sample_flow (Δ%) | sample_csv (Δ%) | sample_flow_csv (Δ%) |
|---|---|---|---|---|
| books | 961,973 | 787,945 (-18.1%) | 1,048,403 (+9.0%) | 858,950 (-10.7%) |
| fb | 3,631 | 30,693 (+745.2%) | 3,959 (+9.0%) | 33,743 (+829.2%) |
| osm | 672,433 | 589,443 (-12.3%) | 739,686 (+10.0%) | 648,294 (-3.6%) |
| covid | 80,794 | 47,121 (-41.7%) | 88,707 (+9.8%) | 51,691 (-36.0%) |
| genome | 196,544 | 214,605 (+9.2%) | 216,174 (+10.0%) | 236,038 (+20.1%) |
| history | 23,376 | 11,085 (-52.6%) | 24,889 (+6.5%) | 12,761 (-45.4%) |
| libio | 117,780 | 138,497 (+17.6%) | 128,490 (+9.1%) | 151,184 (+28.4%) |
| planet | 604,479 | 337,958 (-44.1%) | 664,813 (+10.0%) | 371,498 (-38.5%) |
| stack | 24,242 | 42,460 (+75.2%) | 26,127 (+7.8%) | 45,948 (+89.5%) |
| wise | 49,780 | 38,641 (-22.4%) | 54,783 (+10.1%) | 42,371 (-14.9%) |

| dataset | sample | sample_flow (Δ%) | sample_csv (Δ%) | sample_flow_csv (Δ%) |
|---|---|---|---|---|
| books | 8 | 8 (+0.0%) | 8 (+0.0%) | 8 (+0.0%) |
| fb | 1 | 3 (+200.0%) | 1 (+0.0%) | 3 (+200.0%) |
| osm | 68 | 68 (+0.0%) | 76 (+11.8%) | 76 (+11.8%) |
| covid | 9 | 9 (+0.0%) | 11 (+22.2%) | 9 (+0.0%) |
| genome | 18 | 18 (+0.0%) | 19 (+5.6%) | 19 (+5.6%) |
| history | 3 | 3 (+0.0%) | 3 (+0.0%) | 3 (+0.0%) |
| libio | 10 | 10 (+0.0%) | 11 (+10.0%) | 11 (+10.0%) |
| planet | 13 | 15 (+15.4%) | 15 (+15.4%) | 15 (+15.4%) |
| stack | 4 | 4 (+0.0%) | 4 (+0.0%) | 4 (+0.0%) |
| wise | 21 | 21 (+0.0%) | 22 (+4.8%) | 22 (+4.8%) |

Window samples, PLA-32:

| dataset | sample | sample_flow (Δ%) | sample_csv (Δ%) | sample_flow_csv (Δ%) |
|---|---|---|---|---|
| books | 5,789 | 5,789 (+0.0%) | 4,730 (-18.3%) | 4,730 (-18.3%) |
| fb | 10,599 | 10,599 (+0.0%) | 11,034 (+4.1%) | 11,033 (+4.1%) |
| osm | 4,919 | 4,918 (-0.0%) | 4,856 (-1.3%) | 4,859 (-1.2%) |
| covid | 827 | 827 (+0.0%) | 490 (-40.7%) | 489 (-40.9%) |
| genome | 12,719 | 12,719 (+0.0%) | 12,337 (-3.0%) | 12,330 (-3.1%) |
| history | 943 | 943 (+0.0%) | 422 (-55.2%) | 418 (-55.7%) |
| libio | 1,057 | 1,058 (+0.1%) | 941 (-11.0%) | 944 (-10.7%) |
| planet | 8,498 | 8,498 (+0.0%) | 8,628 (+1.5%) | 8,636 (+1.6%) |
| stack | 144 | 146 (+1.4%) | 86 (-40.3%) | 88 (-38.9%) |
| wise | 805 | 807 (+0.2%) | 315 (-60.9%) | 317 (-60.6%) |

Window samples, RMSE:

| dataset | sample | sample_flow (Δ%) | sample_csv (Δ%) | sample_flow_csv (Δ%) |
|---|---|---|---|---|
| books | 1,709.1 | 11,384.6 (+566.1%) | 1,875.1 (+9.7%) | 12,517.3 (+632.4%) |
| fb | 8,202.5 | 11,037.8 (+34.6%) | 9,015.3 (+9.9%) | 12,139.2 (+48.0%) |
| osm | 118,136.5 | 111,995.7 (-5.2%) | 129,938.9 (+10.0%) | 123,184.6 (+4.3%) |
| covid | 24,143.5 | 17,664.6 (-26.8%) | 26,482.5 (+9.7%) | 19,341.2 (-19.9%) |
| genome | 48,827.0 | 45,938.1 (-5.9%) | 53,700.7 (+10.0%) | 50,523.0 (+3.5%) |
| history | 9,651.7 | 18,579.0 (+92.5%) | 10,512.4 (+8.9%) | 20,400.1 (+111.4%) |
| libio | 8,854.0 | 13,617.6 (+53.8%) | 9,257.0 (+4.6%) | 15,157.8 (+71.2%) |
| planet | 42,645.1 | 47,239.9 (+10.8%) | 46,896.6 (+10.0%) | 51,950.2 (+21.8%) |
| stack | 25,321.9 | 34,350.2 (+35.7%) | 23,747.7 (-6.2%) | 33,100.3 (+30.7%) |
| wise | 5,551.0 | 11,518.2 (+107.5%) | 6,107.5 (+10.0%) | 12,666.0 (+128.2%) |

Window samples, CD / ME / PLA-4096:

| dataset | sample | sample_flow (Δ%) | sample_csv (Δ%) | sample_flow_csv (Δ%) |
|---|---|---|---|---|
| books | 27 | 26 (-3.7%) | 26 (-3.7%) | 26 (-3.7%) |
| fb | 96 | 89 (-7.3%) | 85 (-11.5%) | 86 (-10.4%) |
| osm | 152 | 164 (+7.9%) | 141 (-7.2%) | 149 (-2.0%) |
| covid | 24 | 23 (-4.2%) | 22 (-8.3%) | 22 (-8.3%) |
| genome | 171 | 160 (-6.4%) | 142 (-17.0%) | 167 (-2.3%) |
| history | 8 | 8 (+0.0%) | 21 (+162.5%) | 20 (+150.0%) |
| libio | 2 | 2 (+0.0%) | 31 (+1450.0%) | 20 (+900.0%) |
| planet | 49 | 47 (-4.1%) | 124 (+153.1%) | 190 (+287.8%) |
| stack | 1 | 1 (+0.0%) | 6 (+500.0%) | 10 (+900.0%) |
| wise | 8 | 7 (-12.5%) | 16 (+100.0%) | 10 (+25.0%) |

| dataset | sample | sample_flow (Δ%) | sample_csv (Δ%) | sample_flow_csv (Δ%) |
|---|---|---|---|---|
| books | 4,262 | 30,804 (+622.8%) | 4,565 (+7.1%) | 33,812 (+693.4%) |
| fb | 24,281 | 33,896 (+39.6%) | 26,484 (+9.1%) | 37,214 (+53.3%) |
| osm | 219,858 | 213,509 (-2.9%) | 241,721 (+9.9%) | 234,738 (+6.8%) |
| covid | 49,835 | 34,341 (-31.1%) | 54,691 (+9.7%) | 37,598 (-24.6%) |
| genome | 98,853 | 89,887 (-9.1%) | 108,703 (+10.0%) | 98,766 (-0.1%) |
| history | 21,774 | 48,045 (+120.7%) | 23,859 (+9.6%) | 52,813 (+142.6%) |
| libio | 21,656 | 44,065 (+103.5%) | 23,767 (+9.7%) | 48,428 (+123.6%) |
| planet | 81,355 | 100,454 (+23.5%) | 89,151 (+9.6%) | 110,461 (+35.8%) |
| stack | 48,380 | 73,343 (+51.6%) | 45,365 (-6.2%) | 71,409 (+47.6%) |
| wise | 11,283 | 29,992 (+165.8%) | 12,505 (+10.8%) | 32,988 (+192.4%) |

| dataset | sample | sample_flow (Δ%) | sample_csv (Δ%) | sample_flow_csv (Δ%) |
|---|---|---|---|---|
| books | 1 | 3 (+200.0%) | 1 (+0.0%) | 3 (+200.0%) |
| fb | 18 | 18 (+0.0%) | 20 (+11.1%) | 20 (+11.1%) |
| osm | 44 | 44 (+0.0%) | 49 (+11.4%) | 49 (+11.4%) |
| covid | 9 | 9 (+0.0%) | 9 (+0.0%) | 9 (+0.0%) |
| genome | 14 | 14 (+0.0%) | 14 (+0.0%) | 14 (+0.0%) |
| history | 6 | 6 (+0.0%) | 6 (+0.0%) | 6 (+0.0%) |
| libio | 9 | 9 (+0.0%) | 9 (+0.0%) | 9 (+0.0%) |
| planet | 35 | 35 (+0.0%) | 38 (+8.6%) | 37 (+5.7%) |
| stack | 3 | 4 (+33.3%) | 3 (+0.0%) | 4 (+33.3%) |
| wise | 4 | 3 (-25.0%) | 4 (+0.0%) | 4 (+0.0%) |

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

| dataset | control | flow (bypass) | flow forced | vp10 | flow_vp10 | fusion_auto | flow_costsel | vp10 Δ | fusion Δ |
|---|---|---|---|---|---|---|---|---|---|
| books | 2.231 | 2.231 | 2.231 | 2.008 | 2.008 | 2.008 | 2.231 | -10.0% | -10.0% |
| fb | 2.600 | 2.600 | 2.600 | 2.181 | 2.181 | 2.181 | 2.600 | -16.1% | -16.1% |
| osm | 5.077 | 5.076 | 5.076 | 4.642 | 4.642 | 4.640 | 5.077 | -8.6% | -8.6% |
| covid | 3.064 | 3.064 | 3.064 | 2.328 | 2.328 | 2.328 | 3.064 | -24.0% | -24.0% |
| genome | 3.298 | 3.298 | 3.298 | 2.610 | 2.610 | 2.610 | 3.298 | -20.9% | -20.9% |
| history | 2.459 | 2.459 | 2.459 | 2.073 | 2.073 | 2.073 | 2.459 | -15.7% | -15.7% |
| libio | 2.555 | 2.555 | 2.555 | 2.145 | 2.145 | 2.145 | 2.555 | -16.0% | -16.0% |
| planet | 3.229 | 3.229 | 3.230 | 2.603 | 2.603 | 2.603 | 3.229 | -19.4% | -19.4% |
| stack | 2.291 | 2.291 | 2.291 | 2.032 | 2.032 | 2.032 | 2.291 | -11.3% | -11.3% |
| wise | 2.417 | 2.417 | 2.417 | 2.059 | 2.059 | 2.059 | 2.417 | -14.8% | -14.8% |

Window samples:

| dataset | control | flow (bypass) | flow forced | vp10 | flow_vp10 | fusion_auto | flow_costsel | vp10 Δ | fusion Δ |
|---|---|---|---|---|---|---|---|---|---|
| books | 2.952 | 2.952 | 2.952 | 2.436 | 2.436 | 2.436 | 2.952 | -17.5% | -17.5% |
| fb | 5.166 | 5.166 | 5.166 | 4.770 | 4.770 | 4.768 | 5.166 | -7.7% | -7.7% |
| osm | 4.365 | 4.365 | 4.365 | 3.772 | 3.772 | 3.772 | 4.365 | -13.6% | -13.6% |
| covid | 2.595 | 2.595 | 2.595 | 2.055 | 2.055 | 2.055 | 2.595 | -20.8% | -20.8% |
| genome | 3.053 | 3.053 | 3.053 | 2.550 | 2.550 | 2.549 | 3.053 | -16.5% | -16.5% |
| history | 2.452 | 2.452 | 2.452 | 2.103 | 2.103 | 2.103 | 2.452 | -14.2% | -14.2% |
| libio | 3.070 | 3.070 | 3.070 | 2.553 | 2.553 | 2.553 | 3.070 | -16.8% | -16.8% |
| planet | 4.622 | 4.622 | 4.622 | 4.095 | 4.095 | 4.094 | 4.622 | -11.4% | -11.4% |
| stack | 2.079 | 2.079 | 2.079 | 1.981 | 1.981 | 1.981 | 2.079 | -4.7% | -4.7% |
| wise | 2.319 | 2.319 | 2.319 | 2.017 | 2.017 | 2.017 | 2.319 | -13.0% | -13.0% |

Worked example (fb, uniform): control fence probes = 2,600,293 fence comparisons / 1,000,000 lookups = 2.600; vp10 = 2.181; Δ = (2.181 − 2.600)/2.600 = −16.1 %. In the vp10 structure each block records `slot_begin`, the rank model predicts a slot, and `predicted_block` [index.hpp:118] compares the prediction against slot boundaries; because the slots were chosen to make the region's CDF straighter (Algorithm 1 minimises Σ(w·xᵢ + b − sᵢ)² over K ∪ V), the prediction lands in the right block more often and the correction walk is shorter.

### 2.2 Coordinate probes, correction distance, transform calls

Uniform:

| dataset | coord control | coord vp10 | corr.dist control | corr.dist vp10 | Δ | transform_calls flow | forced | flow_vp10 | fusion_auto |
|---|---|---|---|---|---|---|---|---|---|
| books | 5.030 | 5.030 | 0.108 | 0.018 | -83.1% | 0.0533 | 1.0000 | 0.0533 | 0.0000 |
| fb | 5.029 | 5.030 | 0.257 | 0.087 | -66.0% | 0.0840 | 1.0000 | 0.0840 | 0.0000 |
| osm | 5.026 | 5.023 | 1.971 | 1.681 | -14.7% | 0.0144 | 1.0000 | 0.0144 | 0.0000 |
| covid | 5.028 | 5.027 | 0.470 | 0.155 | -66.9% | 0.0346 | 1.0000 | 0.0346 | 0.0000 |
| genome | 5.029 | 5.029 | 0.625 | 0.319 | -48.9% | 0.0163 | 1.0000 | 0.0163 | 0.0000 |
| history | 5.031 | 5.031 | 0.204 | 0.045 | -77.9% | 0.0267 | 1.0000 | 0.0267 | 0.0000 |
| libio | 5.029 | 5.030 | 0.253 | 0.082 | -67.7% | 0.0429 | 1.0000 | 0.0429 | 0.0000 |
| planet | 5.029 | 5.029 | 0.561 | 0.290 | -48.3% | 0.0881 | 1.0000 | 0.0881 | 0.0000 |
| stack | 5.030 | 5.031 | 0.135 | 0.029 | -78.2% | 0.0006 | 1.0000 | 0.0006 | 0.0000 |
| wise | 5.030 | 5.030 | 0.188 | 0.041 | -78.0% | 0.0082 | 1.0000 | 0.0082 | 0.0000 |

Window:

| dataset | coord control | coord vp10 | corr.dist control | corr.dist vp10 | Δ | transform_calls flow | forced | flow_vp10 | fusion_auto |
|---|---|---|---|---|---|---|---|---|---|
| books | 5.029 | 5.029 | 0.405 | 0.191 | -52.9% | 0.0082 | 1.0000 | 0.0082 | 0.0000 |
| fb | 5.032 | 5.029 | 1.888 | 1.597 | -15.4% | 0.0245 | 1.0000 | 0.0245 | 0.0000 |
| osm | 5.034 | 5.031 | 1.297 | 0.965 | -25.6% | 0.0432 | 1.0000 | 0.0432 | 0.0000 |
| covid | 5.029 | 5.030 | 0.256 | 0.037 | -85.5% | 0.0021 | 1.0000 | 0.0021 | 0.0000 |
| genome | 5.028 | 5.029 | 0.451 | 0.237 | -47.4% | 0.0185 | 1.0000 | 0.0185 | 0.0000 |
| history | 5.029 | 5.030 | 0.212 | 0.065 | -69.2% | 0.0041 | 1.0000 | 0.0041 | 0.0000 |
| libio | 5.029 | 5.029 | 0.523 | 0.274 | -47.7% | 0.0103 | 1.0000 | 0.0103 | 0.0000 |
| planet | 5.030 | 5.027 | 1.527 | 1.226 | -19.7% | 0.0349 | 1.0000 | 0.0349 | 0.0000 |
| stack | 5.031 | 5.030 | 0.046 | 0.007 | -84.0% | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| wise | 5.030 | 5.030 | 0.143 | 0.021 | -85.2% | 0.0020 | 1.0000 | 0.0020 | 0.0000 |

Readings:

- Coordinate probes are 5.03 in every row: the region always binary-searches its 32 block boundaries (log₂ 32 = 5; the residual 0.03 comes from the search's boundary handling and the shorter last region and was not analysed further). Neither component changes this: it is the cost of finding the predicted block, not of correcting the prediction.
- Correction distance (blocks between predicted and true block) falls 48-83 % on nine uniform samples (books −83.1 %, stack −78.2 %, wise −78.0 %, history −77.9 %, libio −67.7 %, covid −66.9 %, fb −66.0 %, genome −48.9 %, planet −48.3 %) and 14.7 % on osm; on the windows it falls 47-86 % on seven (covid −85.5, wise −85.2, stack −84.0, history −69.2, books −52.9, libio −47.7, genome −47.4) and 15.4 % (fb), 19.7 % (planet), 25.6 % (osm) on the locally hard three. Correction distance is the per-lookup *residual* of the region model measured in blocks; it moves more than fence probes because a fence-probe walk costs ⌈log₂(distance + 1)⌉-ish comparisons, not `distance` comparisons.
- `transform_calls` per lookup is exactly the fraction of lookups that land in a flow region: 0.053 on books_uniform under the bypass (26 of 489 regions accepted, Table 2.3), 1.000 when forced.

### 2.3 Flow acceptance under the bypass rule, selector choices, tail conflict degrees

Uniform:

| dataset | regions | D99 raw mean | D99 flow mean | flow accepted (bypass) | fusion_auto none/flow/vp/both | costsel none/flow | fusion cost none | fusion cost selected | costsel cost none | costsel cost selected |
|---|---|---|---|---|---|---|---|---|---|---|
| books | 489 | 3.28 | 3.28 | 5.32% (26) | 0/0/489/0 | 489/0 | 7.259 | 7.037 | 7.259 | 7.259 |
| fb | 489 | 8.45 | 8.46 | 8.38% (41) | 0/0/489/0 | 489/0 | 7.627 | 7.208 | 7.627 | 7.627 |
| osm | 489 | 36.16 | 36.98 | 1.43% (7) | 19/0/470/0 | 489/0 | 10.097 | 9.657 | 10.097 | 10.097 |
| covid | 489 | 3.16 | 3.14 | 3.48% (17) | 0/0/489/0 | 489/0 | 8.088 | 7.351 | 8.088 | 8.088 |
| genome | 489 | 4.23 | 4.23 | 1.64% (8) | 1/0/488/0 | 489/0 | 8.321 | 7.635 | 8.321 | 8.321 |
| history | 489 | 3.10 | 3.10 | 2.66% (13) | 0/0/489/0 | 489/0 | 7.486 | 7.100 | 7.486 | 7.486 |
| libio | 489 | 3.37 | 3.36 | 4.29% (21) | 0/0/489/0 | 489/0 | 7.582 | 7.172 | 7.582 | 7.582 |
| planet | 489 | 7.66 | 7.65 | 8.79% (43) | 0/0/489/0 | 489/0 | 8.256 | 7.631 | 8.256 | 8.256 |
| stack | 489 | 3.01 | 3.01 | 0.20% (1) | 0/0/489/0 | 489/0 | 7.318 | 7.060 | 7.318 | 7.318 |
| wise | 489 | 3.03 | 3.03 | 0.82% (4) | 0/0/489/0 | 489/0 | 7.443 | 7.085 | 7.443 | 7.443 |

Window:

| dataset | regions | D99 raw mean | D99 flow mean | flow accepted (bypass) | fusion_auto none/flow/vp/both | costsel none/flow | fusion cost none | fusion cost selected | costsel cost none | costsel cost selected |
|---|---|---|---|---|---|---|---|---|---|---|
| books | 489 | 10.76 | 10.80 | 0.82% (4) | 0/0/489/0 | 489/0 | 7.977 | 7.463 | 7.977 | 7.977 |
| fb | 489 | 31.41 | 31.43 | 2.45% (12) | 12/0/477/0 | 489/0 | 10.190 | 9.789 | 10.190 | 10.190 |
| osm | 489 | 9.66 | 9.65 | 4.29% (21) | 2/0/487/0 | 489/0 | 9.396 | 8.800 | 9.396 | 9.396 |
| covid | 489 | 3.02 | 3.03 | 0.20% (1) | 0/0/489/0 | 489/0 | 7.621 | 7.081 | 7.621 | 7.621 |
| genome | 489 | 60.41 | 60.38 | 1.84% (9) | 2/0/487/0 | 489/0 | 8.075 | 7.575 | 8.075 | 8.075 |
| history | 489 | 3.10 | 3.11 | 0.41% (2) | 0/0/489/0 | 489/0 | 7.478 | 7.131 | 7.478 | 7.478 |
| libio | 489 | 1.61 | 1.60 | 1.02% (5) | 0/0/489/0 | 489/0 | 8.093 | 7.576 | 8.093 | 8.093 |
| planet | 489 | 35.78 | 35.67 | 3.48% (17) | 3/0/486/0 | 489/0 | 9.645 | 9.118 | 9.645 | 9.645 |
| stack | 489 | 1.00 | 1.00 | 0.00% (0) | 0/0/489/0 | 489/0 | 7.107 | 7.009 | 7.107 | 7.107 |
| wise | 489 | 3.01 | 3.00 | 0.20% (1) | 0/0/489/0 | 489/0 | 7.345 | 7.045 | 7.345 | 7.345 |

Readings:

- The NFL-style bypass [index.hpp:281: keep the flow iff D₉₉(raw) − D₉₉(z) ≥ 0.10·D₉₉(raw)] accepts the flow in 0.20 % (stack) to 8.79 % (planet) of the 489 regions on the uniform samples and 0.0 % (stack) to 4.29 % (osm) on the windows. The mean per-region D₉₉ is 3.0-4.2 on eight uniform samples, 7.66 on planet, 8.45 on fb and 36.2 on osm, and the mean D₉₉ of the transformed features is within ±0.02 of the raw one on every dataset except osm (36.16 → 36.98, worse). A region of 4,096 keys spans ≈ 0.2 % of the key range; over that range a monotone global warp is nearly linear, so z(k) ≈ a·k + c inside the region and the per-region linear model absorbs it. That is the mechanism behind "the transform does nothing at region scale", stated as a fact about the counters (§2.1: forced flow leaves fence probes within 0.0006 of the control on every uniform sample and within 0.0002 on every window; key_numbers.json `e2_forced_maxdiff`).
- The cost-based selector [index.hpp:239-264: cost = mean over the region's own keys of (fence + coordinate probes of the real `locate_block`) + 4 probe-equivalents if the candidate uses the flow] never picks the flow: `fusion_auto` chooses `vp` in 470-489 of 489 regions (osm_uniform 470 = 96.1 %, fb_window 477 = 97.5 %, all others 486-489) and `flow` or `both` in 0; `flow_costsel` chooses `none` in 489 of 489 regions on all 20 samples. The estimated cost of the chosen candidate is 0.22-0.74 probes below the raw-rank cost (e.g. covid 8.088 → 7.351), which is the same saving the measured fence probes show (3.064 → 2.328).
- `flow_vp10` equals `vp10` to three decimals on every sample because the flow is bypassed in 91-100 % of regions and, where accepted, changes nothing measurable.

### 2.4 Rank SSE, memory, build time, preprocessing

Uniform:

| dataset | rank SSE control | slot SSE vp10 | ratio vp10 | ratio flow forced | vp/key | B/key control | B/key vp10 | meta Δ B/key | build ms control | build ms vp10 | build ms fusion | smoothing s vp10 (thread) | smoothing s fusion | transform ns/key forced |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| books | 7.171e+08 | 3.27e+07 | 0.046 | 1.000 | 0.0925 | 14.53 | 15.27 | 0.740 | 59.8 | 4,419.5 | 10,278.0 | 62.76 | 154.58 | 31.1 |
| fb | 3.798e+09 | 5.866e+08 | 0.154 | 1.000 | 0.0999 | 11.39 | 12.19 | 0.799 | 57.5 | 2,504.4 | 8,136.4 | 36.94 | 124.15 | 29.6 |
| osm | 2.72e+11 | 2.285e+11 | 0.840 | 1.000 | 0.0999 | 14.25 | 15.05 | 0.799 | 62.0 | 1,954.6 | 6,890.1 | 27.75 | 103.18 | 30.8 |
| covid | 1.626e+10 | 4.704e+09 | 0.289 | 1.000 | 0.0995 | 13.78 | 14.58 | 0.796 | 61.2 | 3,509.1 | 9,165.1 | 51.97 | 135.54 | 28.3 |
| genome | 3.898e+10 | 2.333e+10 | 0.598 | 1.000 | 0.0999 | 11.56 | 12.36 | 0.799 | 61.6 | 3,091.0 | 8,404.8 | 45.38 | 125.77 | 27.7 |
| history | 4.249e+09 | 7.529e+08 | 0.177 | 1.000 | 0.0977 | 10.87 | 11.65 | 0.781 | 54.8 | 3,816.6 | 9,731.6 | 55.82 | 146.45 | 29.0 |
| libio | 8.162e+09 | 3.544e+09 | 0.434 | 1.000 | 0.0978 | 10.36 | 11.15 | 0.782 | 54.2 | 3,566.7 | 9,541.0 | 52.94 | 138.66 | 28.4 |
| planet | 2.694e+10 | 1.469e+10 | 0.545 | 1.000 | 0.0998 | 10.79 | 11.58 | 0.799 | 56.5 | 2,524.0 | 8,077.5 | 37.55 | 119.10 | 27.4 |
| stack | 2.029e+09 | 5.349e+08 | 0.264 | 1.000 | 0.0922 | 10.20 | 10.94 | 0.737 | 57.2 | 4,273.0 | 10,030.9 | 63.09 | 154.18 | 28.4 |
| wise | 3.624e+09 | 1.027e+09 | 0.283 | 1.000 | 0.0975 | 12.09 | 12.87 | 0.780 | 61.3 | 4,239.4 | 10,301.7 | 63.16 | 159.47 | 32.1 |

Window:

| dataset | rank SSE control | slot SSE vp10 | ratio vp10 | ratio flow forced | vp/key | B/key control | B/key vp10 | meta Δ B/key | build ms control | build ms vp10 | build ms fusion | smoothing s vp10 (thread) | smoothing s fusion | transform ns/key forced |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| books | 9.484e+09 | 2.583e+09 | 0.272 | 1.000 | 0.0999 | 14.03 | 14.83 | 0.799 | 62.0 | 2,149.6 | 7,534.0 | 30.74 | 115.77 | 30.8 |
| fb | 2.074e+11 | 1.594e+11 | 0.769 | 1.000 | 0.0999 | 10.50 | 11.30 | 0.799 | 56.3 | 1,653.6 | 6,658.3 | 24.84 | 102.86 | 30.5 |
| osm | 1.226e+11 | 8.794e+10 | 0.717 | 1.000 | 0.0999 | 13.48 | 14.28 | 0.799 | 63.6 | 2,044.8 | 7,306.3 | 30.33 | 111.80 | 31.1 |
| covid | 4.208e+09 | 2.156e+08 | 0.051 | 1.000 | 0.0987 | 12.94 | 13.73 | 0.790 | 60.9 | 3,913.1 | 10,278.7 | 60.87 | 158.15 | 26.8 |
| genome | 1.234e+10 | 3.87e+09 | 0.314 | 1.000 | 0.0999 | 10.68 | 11.48 | 0.799 | 58.7 | 1,866.2 | 7,324.7 | 27.27 | 112.43 | 31.7 |
| history | 7.051e+09 | 3.279e+09 | 0.465 | 1.000 | 0.0974 | 10.04 | 10.81 | 0.779 | 56.4 | 3,745.7 | 9,474.3 | 54.60 | 146.43 | 28.4 |
| libio | 3.262e+10 | 1.552e+10 | 0.476 | 1.000 | 0.0850 | 9.43 | 10.11 | 0.680 | 56.1 | 2,908.6 | 7,498.2 | 41.82 | 111.98 | 28.8 |
| planet | 1.731e+11 | 1.379e+11 | 0.796 | 1.000 | 0.0999 | 10.27 | 11.07 | 0.799 | 56.0 | 1,809.9 | 6,563.2 | 26.19 | 106.70 | 31.5 |
| stack | 1.379e+08 | 3.333e+06 | 0.024 | 1.000 | 0.0310 | 9.19 | 9.43 | 0.248 | 60.6 | 2,117.7 | 4,110.7 | 28.93 | 58.42 | 31.9 |
| wise | 1.27e+09 | 6.172e+07 | 0.049 | 1.000 | 0.0978 | 11.26 | 12.04 | 0.782 | 59.9 | 4,291.2 | 10,088.4 | 63.07 | 149.98 | 27.7 |

Readings:

- `rank SSE` is the OLS squared error of the region models summed over regions, on ranks (`rank_sse_before`) and on slots after smoothing (`rank_sse_after`) [benchmark.cpp learnability block]. The ratio after/before is the CSV objective realised: 0.046 on books_uniform (a 22× reduction), 0.15-0.30 on fb, history, stack, wise, covid, 0.43-0.60 on libio, planet, genome, and 0.84 on osm. It is 1.000 for the forced flow: swapping φ = raw key for φ = z(k) leaves the region's SSE unchanged to the printed precision, again because z is linear inside a region.
- Virtual points cost 0.092-0.100 doubles per key of metadata (`virtual_points_per_key`), i.e. 0.74-0.80 B/key on top of 10.2-14.5 B/key for the packed control (18 of 20 samples; libio_window 0.68, stack_window 0.25 [results/aidb_window/sweep/summary.csv]). No key bytes are added: the virtual points only move the rank targets and each block records one `slot_begin`.
- Build time (uniform samples; window in the second table): 54-62 ms for the control at 2M keys; 1.95-4.42 s with virtual points (1.65-4.29 s on the windows; smoothing is 28-63 s of summed thread time at 16 threads, i.e. 14-32 µs per key; Algorithm 1 as written is O(λ·n) per region because every greedy round rescans every gap after the refit); 6.9-10.3 s for the fused selector, which smooths both features (103-159 s of thread time). Batched flow inference at build time costs 27.4-32.1 ns per key (`transform_ns / keys`, forced-flow rows).

### 2.5 Throughput and paired speedups (region level)

Uniform (1M lookups × 3 seeds, no QoS):

| dataset | control Mops | flow (bypass) | flow forced | vp10 | flow_vp10 | fusion_auto | flow_costsel | packed_byte | raw_rank | sorted_vector |
|---|---|---|---|---|---|---|---|---|---|---|
| books | 2.144 | 0.922 [0.90, 0.96] | 0.908 [0.85, 0.96] | 0.989 [0.86, 1.09] | 1.005 [0.89, 1.08] | 0.961 [0.93, 0.99] | 0.932 [0.90, 0.96] | 0.969 [0.91, 1.01] | 1.427 [1.28, 1.60] | 1.593 [1.35, 2.06] |
| fb | 2.638 | 0.975 [0.94, 1.04] | 1.019 [0.92, 1.13] | 0.968 [0.93, 1.00] | 1.001 [0.98, 1.04] | 1.000 [0.96, 1.05] | 1.034 [0.95, 1.08] | 0.965 [0.91, 1.04] | 1.127 [1.10, 1.17] | 1.174 [0.95, 1.34] |
| osm | 1.932 | 1.018 [0.91, 1.12] | 0.957 [0.83, 1.05] | 0.958 [0.85, 1.02] | 1.028 [0.92, 1.09] | 1.002 [0.88, 1.09] | 1.015 [0.89, 1.17] | 0.978 [0.84, 1.09] | 1.454 [1.20, 1.62] | 1.436 [1.20, 1.96] |
| covid | 2.217 | 0.970 [0.95, 0.99] | 0.939 [0.92, 0.96] | 1.023 [0.94, 1.08] | 1.010 [0.97, 1.05] | 0.984 [0.91, 1.05] | 1.017 [0.98, 1.07] | 0.942 [0.90, 1.01] | 1.528 [1.42, 1.59] | 1.291 [1.19, 1.39] |
| genome | 2.476 | 1.028 [0.90, 1.10] | 1.018 [0.93, 1.13] | 0.992 [0.87, 1.07] | 1.045 [0.93, 1.13] | 1.031 [0.92, 1.15] | 0.973 [0.93, 1.01] | 1.001 [0.92, 1.08] | 1.238 [1.04, 1.44] | 1.476 [1.16, 2.36] |
| history | 2.816 | 0.995 [0.97, 1.04] | 0.961 [0.91, 1.01] | 1.009 [0.96, 1.07] | 1.003 [0.95, 1.05] | 0.980 [0.97, 0.99] | 0.980 [0.92, 1.06] | 1.008 [0.99, 1.04] | 1.042 [0.85, 1.15] | 1.117 [1.00, 1.39] |
| libio | 2.824 | 1.037 [0.97, 1.09] | 0.965 [0.90, 1.01] | 1.072 [1.02, 1.13] | 1.033 [0.96, 1.10] | 1.033 [0.94, 1.14] | 0.987 [0.92, 1.03] | 1.063 [1.01, 1.10] | 1.088 [0.97, 1.26] | 1.036 [0.87, 1.29] |
| planet | 2.789 | 0.947 [0.89, 0.99] | 1.001 [0.94, 1.04] | 0.960 [0.87, 1.12] | 1.012 [0.90, 1.12] | 0.951 [0.89, 1.04] | 0.963 [0.93, 1.03] | 0.954 [0.89, 1.03] | 1.183 [1.10, 1.30] | 1.092 [0.84, 1.37] |
| stack | 2.942 | 1.037 [0.99, 1.11] | 1.016 [0.96, 1.05] | 1.011 [0.99, 1.03] | 1.007 [0.99, 1.02] | 1.047 [1.01, 1.12] | 1.025 [0.98, 1.06] | 1.083 [1.06, 1.12] | 1.048 [0.97, 1.19] | 1.219 [0.94, 1.42] |
| wise | 2.482 | 0.986 [0.91, 1.09] | 0.973 [0.93, 1.06] | 0.958 [0.86, 1.05] | 1.005 [0.94, 1.09] | 1.005 [0.96, 1.07] | 1.003 [0.96, 1.06] | 1.038 [1.00, 1.07] | 1.222 [1.15, 1.29] | 1.420 [1.26, 1.51] |

Window:

| dataset | control Mops | flow (bypass) | flow forced | vp10 | flow_vp10 | fusion_auto | flow_costsel | packed_byte | raw_rank | sorted_vector |
|---|---|---|---|---|---|---|---|---|---|---|
| books | 2.420 | 1.049 [1.01, 1.12] | 0.865 [0.70, 0.99] | 1.035 [0.87, 1.28] | 0.992 [0.76, 1.24] | 0.949 [0.85, 1.18] | 1.075 [0.94, 1.29] | 0.958 [0.94, 0.97] | 1.277 [0.90, 1.54] | 1.777 [1.00, 2.56] |
| fb | 2.811 | 0.992 [0.94, 1.05] | 0.979 [0.94, 1.07] | 0.996 [0.92, 1.11] | 1.029 [0.98, 1.05] | 0.945 [0.82, 1.02] | 0.958 [0.94, 1.00] | 0.979 [0.94, 1.05] | 1.014 [0.97, 1.05] | 1.707 [1.16, 2.40] |
| osm | 2.448 | 0.922 [0.88, 1.01] | 0.815 [0.74, 0.97] | 1.000 [0.85, 1.10] | 0.967 [0.91, 1.00] | 0.914 [0.86, 0.96] | 0.924 [0.78, 1.07] | 0.966 [0.95, 0.98] | 1.220 [0.96, 1.46] | 1.731 [1.27, 2.28] |
| covid | 2.580 | 0.967 [0.86, 1.11] | 0.972 [0.87, 1.10] | 0.913 [0.86, 0.94] | 0.918 [0.86, 0.95] | 0.872 [0.82, 0.91] | 1.007 [0.89, 1.08] | 0.939 [0.84, 1.00] | 1.320 [1.24, 1.43] | 1.095 [0.95, 1.23] |
| genome | 2.548 | 0.997 [0.97, 1.04] | 0.893 [0.82, 1.00] | 0.964 [0.86, 1.06] | 0.966 [0.94, 1.02] | 0.946 [0.90, 1.01] | 0.973 [0.89, 1.02] | 1.018 [0.99, 1.04] | 1.146 [1.00, 1.23] | 1.300 [0.99, 1.68] |
| history | 3.178 | 0.915 [0.84, 0.96] | 0.896 [0.83, 0.99] | 1.025 [0.94, 1.11] | 0.940 [0.84, 1.04] | 0.984 [0.87, 1.13] | 0.962 [0.83, 1.04] | 0.957 [0.92, 1.00] | 1.046 [0.96, 1.23] | 1.414 [1.21, 1.63] |
| libio | 3.116 | 1.042 [0.87, 1.17] | 1.004 [0.94, 1.12] | 1.022 [0.95, 1.07] | 1.040 [0.98, 1.09] | 0.992 [0.97, 1.02] | 1.021 [0.97, 1.11] | 1.045 [1.00, 1.13] | 1.022 [0.70, 1.41] | 1.060 [0.91, 1.19] |
| planet | 2.738 | 1.034 [0.87, 1.14] | 1.025 [0.95, 1.07] | 1.011 [0.94, 1.09] | 0.987 [0.87, 1.05] | 1.105 [1.00, 1.18] | 1.031 [0.95, 1.16] | 1.044 [0.87, 1.15] | 1.098 [0.84, 1.42] | 1.071 [0.89, 1.51] |
| stack | 3.252 | 1.062 [1.04, 1.10] | 1.030 [0.99, 1.05] | 1.049 [1.01, 1.12] | 1.035 [0.96, 1.08] | 0.974 [0.92, 1.03] | 1.058 [1.04, 1.10] | 0.999 [0.91, 1.08] | 0.910 [0.76, 1.00] | 1.414 [1.10, 1.61] |
| wise | 2.901 | 0.980 [0.94, 1.03] | 0.950 [0.88, 1.03] | 0.957 [0.88, 1.06] | 1.019 [0.96, 1.12] | 0.946 [0.88, 1.00] | 1.108 [1.08, 1.13] | 0.951 [0.91, 0.98] | 1.296 [1.13, 1.49] | 1.873 [1.63, 2.06] |

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

| sample | binary root/op | raw root/op | raw chosen | fences root/op | virt | fences chosen | fusion root/op | fusion chosen | total control | total fusion | Δ | total vp10+fusion | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| books_uniform | 8.96 | 8.96 | binary(fallback) | 2.60 | 662 | raw+fences | 2.60 | raw+fences | 16.22 | 9.86 | -39.2% | 9.64 | -40.6% |
| books_window | 8.96 | 2.17 | raw+ranks | 2.10 | 1 | raw+fences | 2.10 | raw+fences | 16.94 | 10.08 | -40.5% | 9.57 | -43.5% |
| covid_uniform | 8.96 | 4.66 | raw+ranks | 2.44 | 58 | raw+fences | 2.44 | raw+fences | 17.05 | 10.53 | -38.2% | 9.79 | -42.6% |
| covid_window | 8.96 | 6.10 | raw+ranks | 2.28 | 86 | raw+fences | 2.28 | raw+fences | 16.58 | 9.90 | -40.3% | 9.36 | -43.5% |
| fb_uniform | 8.96 | 2.14 | raw+ranks | 2.08 | 1 | raw+fences | 2.08 | raw+fences | 16.59 | 9.71 | -41.5% | 9.29 | -44.0% |
| fb_window | 8.96 | 3.53 | raw+ranks | 2.27 | 34 | raw+fences | 2.27 | raw+fences | 19.15 | 12.47 | -34.9% | 12.07 | -37.0% |
| genome_uniform | 8.96 | 8.67 | raw+ranks | 2.84 | 294 | raw+fences | 2.84 | raw+fences | 17.28 | 11.16 | -35.4% | 10.48 | -39.4% |
| genome_window | 8.96 | 7.27 | raw+ranks | 2.38 | 130 | raw+fences | 2.38 | raw+fences | 17.03 | 10.45 | -38.6% | 9.95 | -41.6% |
| history_uniform | 8.96 | 3.64 | raw+ranks | 2.18 | 20 | raw+fences | 2.18 | raw+fences | 16.45 | 9.67 | -41.2% | 9.28 | -43.6% |
| history_window | 8.96 | 3.76 | raw+ranks | 2.10 | 20 | raw+fences | 2.10 | raw+fences | 16.44 | 9.58 | -41.7% | 9.23 | -43.8% |
| libio_uniform | 8.96 | 6.53 | raw+ranks | 2.32 | 117 | raw+fences | 2.32 | raw+fences | 16.54 | 9.90 | -40.2% | 9.49 | -42.6% |
| libio_window | 8.96 | 3.60 | raw+ranks | 2.25 | 25 | raw+fences | 2.25 | raw+fences | 17.05 | 10.35 | -39.3% | 9.83 | -42.3% |
| osm_uniform | 8.96 | 8.96 | binary(fallback) | 8.17 | 1956 | raw+fences | 8.17 | raw+fences | 19.06 | 18.28 | -4.1% | 17.84 | -6.4% |
| osm_window | 8.96 | 8.96 | binary(fallback) | 3.43 | 984 | raw+fences | 3.43 | raw+fences | 18.36 | 12.83 | -30.1% | 12.24 | -33.3% |
| planet_uniform | 8.96 | 8.96 | binary(fallback) | 5.14 | 1956 | raw+fences | 5.14 | raw+fences | 17.21 | 13.40 | -22.2% | 12.77 | -25.8% |
| planet_window | 8.96 | 7.60 | raw+ranks | 3.17 | 206 | raw+fences | 3.17 | raw+fences | 18.61 | 12.82 | -31.1% | 12.29 | -34.0% |
| stack_uniform | 8.96 | 3.34 | raw+ranks | 2.17 | 13 | raw+fences | 2.17 | raw+fences | 16.28 | 9.49 | -41.7% | 9.23 | -43.3% |
| stack_window | 8.96 | 6.27 | raw+ranks | 2.08 | 41 | raw+fences | 2.08 | raw+fences | 16.07 | 9.19 | -42.8% | 9.09 | -43.4% |
| wise_uniform | 8.96 | 5.73 | raw+ranks | 2.65 | 89 | raw+fences | 2.65 | raw+fences | 16.40 | 10.10 | -38.4% | 9.74 | -40.6% |
| wise_window | 8.96 | 2.94 | raw+ranks | 2.18 | 13 | raw+fences | 2.18 | raw+fences | 16.30 | 9.53 | -41.6% | 9.23 | -43.4% |

### 3.2 The full ablation table (post-fix; medians over the 3 seeds; `est` = the selector's own estimates for binary / raw+ranks / flow+ranks / raw+fences / flow+fences, flow candidates include the +4 charge)

| sample | variant | root/op | fence/op | total/op | total Δ | chosen | virt. fences | est bin/raw/flow/vf/fus | build s | meta B/key | Mops | speedup [seed spread] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| books_uniform | binary root (control) | 8.96 | 2.23 | 16.22 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.059 | 0.5706 | 2.259 | 1.000 [1.00, 1.00] |
|  | raw root | 8.96 | 2.23 | 16.22 | +0.0% | binary(fallback) | 0 | 8.96/11.08/0.00/0.00/0.00 | 0.057 | 0.5706 | 2.181 | 0.948 [0.86, 1.03] |
|  | flow root | 8.96 | 2.23 | 16.22 | -0.0% | binary(fallback) | 0 | 8.96/11.08/14.91/0.00/0.00 | 0.064 | 0.5706 | 2.151 | 0.924 [0.76, 1.02] |
|  | root fences | 2.60 | 2.23 | 9.86 | -39.2% | raw+fences | 662 | 8.96/11.08/0.00/2.73/0.00 | 0.151 | 0.5729 | 2.265 | 0.956 [0.89, 1.08] |
|  | root fusion | 2.60 | 2.23 | 9.86 | -39.2% | raw+fences | 662 | 8.96/11.08/14.91/2.73/6.61 | 0.221 | 0.5729 | 2.426 | 1.045 [0.89, 1.16] |
|  | region vp10 | 8.96 | 2.01 | 15.99 | -1.4% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 4.073 | 1.3108 | 2.298 | 1.012 [0.89, 1.14] |
|  | vp10 + root fusion | 2.60 | 2.01 | 9.64 | -40.6% | raw+fences | 662 | 8.96/11.08/14.91/2.73/6.61 | 4.454 | 1.3132 | 2.246 | 0.977 [0.83, 1.07] |
| books_window | binary root (control) | 8.96 | 2.95 | 16.94 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.058 | 0.5706 | 2.496 | 1.000 [1.00, 1.00] |
|  | raw root | 2.17 | 2.95 | 10.15 | -40.1% | raw+ranks | 0 | 8.96/2.32/0.00/0.00/0.00 | 0.058 | 0.5706 | 2.422 | 1.004 [0.88, 1.19] |
|  | flow root | 2.17 | 2.95 | 10.15 | -40.1% | raw+ranks | 0 | 8.96/2.32/8.11/0.00/0.00 | 0.065 | 0.5706 | 2.711 | 1.076 [1.04, 1.11] |
|  | root fences | 2.10 | 2.95 | 10.08 | -40.5% | raw+fences | 1 | 8.96/2.32/0.00/2.30/0.00 | 0.059 | 0.5716 | 2.466 | 0.941 [0.82, 1.03] |
|  | root fusion | 2.10 | 2.95 | 10.08 | -40.5% | raw+fences | 1 | 8.96/2.32/8.11/2.30/6.33 | 0.067 | 0.5716 | 2.532 | 0.977 [0.92, 1.03] |
|  | region vp10 | 8.96 | 2.44 | 16.42 | -3.0% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 2.053 | 1.3694 | 2.582 | 1.014 [0.89, 1.11] |
|  | vp10 + root fusion | 2.10 | 2.44 | 9.57 | -43.5% | raw+fences | 1 | 8.96/2.32/8.11/2.30/6.33 | 2.212 | 1.3704 | 2.725 | 1.093 [0.99, 1.24] |
| covid_uniform | binary root (control) | 8.96 | 3.06 | 17.05 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.057 | 0.5706 | 2.475 | 1.000 [1.00, 1.00] |
|  | raw root | 4.66 | 3.06 | 12.75 | -25.2% | raw+ranks | 0 | 8.96/4.64/0.00/0.00/0.00 | 0.058 | 0.5706 | 2.527 | 1.019 [1.01, 1.02] |
|  | flow root | 4.66 | 3.06 | 12.75 | -25.2% | raw+ranks | 0 | 8.96/4.64/8.49/0.00/0.00 | 0.064 | 0.5706 | 2.424 | 0.948 [0.92, 0.98] |
|  | root fences | 2.44 | 3.06 | 10.53 | -38.2% | raw+fences | 58 | 8.96/4.64/0.00/2.56/0.00 | 0.063 | 0.5717 | 2.388 | 0.928 [0.91, 0.96] |
|  | root fusion | 2.44 | 3.06 | 10.53 | -38.2% | raw+fences | 58 | 8.96/4.64/8.49/2.56/6.43 | 0.071 | 0.5717 | 2.466 | 0.958 [0.92, 1.00] |
|  | region vp10 | 8.96 | 2.33 | 16.31 | -4.3% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 3.387 | 1.3669 | 2.459 | 0.963 [0.92, 1.03] |
|  | vp10 + root fusion | 2.44 | 2.33 | 9.79 | -42.6% | raw+fences | 58 | 8.96/4.64/8.49/2.56/6.43 | 3.797 | 1.3680 | 2.382 | 0.931 [0.89, 0.98] |
| covid_window | binary root (control) | 8.96 | 2.60 | 16.58 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.059 | 0.5706 | 2.682 | 1.000 [1.00, 1.00] |
|  | raw root | 6.10 | 2.60 | 13.73 | -17.2% | raw+ranks | 0 | 8.96/6.15/0.00/0.00/0.00 | 0.058 | 0.5706 | 2.693 | 0.983 [0.97, 1.00] |
|  | flow root | 6.10 | 2.60 | 13.73 | -17.2% | raw+ranks | 0 | 8.96/6.15/9.23/0.00/0.00 | 0.064 | 0.5706 | 2.571 | 0.920 [0.84, 0.97] |
|  | root fences | 2.28 | 2.60 | 9.90 | -40.3% | raw+fences | 86 | 8.96/6.15/0.00/2.43/0.00 | 0.064 | 0.5717 | 2.674 | 0.984 [0.97, 1.00] |
|  | root fusion | 2.28 | 2.60 | 9.90 | -40.3% | raw+fences | 86 | 8.96/6.15/9.23/2.43/6.54 | 0.075 | 0.5717 | 2.771 | 1.030 [0.99, 1.07] |
|  | region vp10 | 8.96 | 2.05 | 16.04 | -3.3% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 3.982 | 1.3604 | 2.725 | 1.001 [0.96, 1.03] |
|  | vp10 + root fusion | 2.28 | 2.05 | 9.36 | -43.5% | raw+fences | 86 | 8.96/6.15/9.23/2.43/6.54 | 4.112 | 1.3615 | 2.613 | 0.949 [0.83, 1.05] |
| fb_uniform | binary root (control) | 8.96 | 2.60 | 16.59 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.054 | 0.5706 | 2.935 | 1.000 [1.00, 1.00] |
|  | raw root | 2.14 | 2.60 | 9.77 | -41.1% | raw+ranks | 0 | 8.96/2.27/0.00/0.00/0.00 | 0.054 | 0.5706 | 3.092 | 1.061 [0.97, 1.18] |
|  | flow root | 2.14 | 2.60 | 9.77 | -41.1% | raw+ranks | 0 | 8.96/2.27/8.14/0.00/0.00 | 0.060 | 0.5706 | 3.036 | 1.008 [0.89, 1.11] |
|  | root fences | 2.08 | 2.60 | 9.71 | -41.5% | raw+fences | 1 | 8.96/2.27/0.00/2.22/0.00 | 0.056 | 0.5716 | 3.106 | 1.055 [1.02, 1.11] |
|  | root fusion | 2.08 | 2.60 | 9.71 | -41.5% | raw+fences | 1 | 8.96/2.27/8.14/2.22/6.35 | 0.064 | 0.5716 | 3.206 | 1.076 [1.03, 1.10] |
|  | region vp10 | 8.96 | 2.18 | 16.17 | -2.5% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 2.401 | 1.3694 | 3.010 | 1.006 [0.94, 1.10] |
|  | vp10 + root fusion | 2.08 | 2.18 | 9.29 | -44.0% | raw+fences | 1 | 8.96/2.27/8.14/2.22/6.35 | 2.851 | 1.3704 | 3.014 | 1.016 [0.94, 1.08] |
| fb_window | binary root (control) | 8.96 | 5.17 | 19.15 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.055 | 0.5706 | 3.011 | 1.000 [1.00, 1.00] |
|  | raw root | 3.53 | 5.17 | 13.73 | -28.3% | raw+ranks | 0 | 8.96/3.58/0.00/0.00/0.00 | 0.051 | 0.5706 | 3.018 | 0.998 [0.95, 1.05] |
|  | flow root | 3.53 | 5.17 | 13.73 | -28.3% | raw+ranks | 0 | 8.96/3.58/7.86/0.00/0.00 | 0.057 | 0.5706 | 3.031 | 1.017 [0.98, 1.05] |
|  | root fences | 2.27 | 5.17 | 12.47 | -34.9% | raw+fences | 34 | 8.96/3.58/0.00/2.41/0.00 | 0.056 | 0.5716 | 3.059 | 1.021 [0.97, 1.06] |
|  | root fusion | 2.27 | 5.17 | 12.47 | -34.9% | raw+fences | 34 | 8.96/3.58/7.86/2.41/6.49 | 0.065 | 0.5716 | 3.088 | 1.029 [1.00, 1.07] |
|  | region vp10 | 8.96 | 4.77 | 18.75 | -2.1% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 1.674 | 1.3694 | 2.862 | 0.968 [0.93, 1.01] |
|  | vp10 + root fusion | 2.27 | 4.77 | 12.07 | -37.0% | raw+fences | 34 | 8.96/3.58/7.86/2.41/6.49 | 1.913 | 1.3705 | 2.926 | 0.993 [0.96, 1.01] |
| genome_uniform | binary root (control) | 8.96 | 3.30 | 17.28 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.058 | 0.5706 | 2.781 | 1.000 [1.00, 1.00] |
|  | raw root | 8.67 | 3.30 | 17.00 | -1.7% | raw+ranks | 0 | 8.96/8.63/0.00/0.00/0.00 | 0.058 | 0.5706 | 2.689 | 0.957 [0.87, 1.02] |
|  | flow root | 8.67 | 3.30 | 17.00 | -1.7% | raw+ranks | 0 | 8.96/8.63/12.98/0.00/0.00 | 0.065 | 0.5706 | 2.678 | 0.975 [0.96, 1.01] |
|  | root fences | 2.84 | 3.30 | 11.16 | -35.4% | raw+fences | 294 | 8.96/8.63/0.00/2.97/0.00 | 0.091 | 0.5722 | 2.783 | 1.020 [0.95, 1.13] |
|  | root fusion | 2.84 | 3.30 | 11.16 | -35.4% | raw+fences | 294 | 8.96/8.63/12.98/2.97/6.83 | 0.149 | 0.5722 | 2.693 | 0.989 [0.95, 1.07] |
|  | region vp10 | 8.96 | 2.61 | 16.60 | -4.0% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 3.013 | 1.3694 | 2.859 | 1.053 [1.01, 1.09] |
|  | vp10 + root fusion | 2.84 | 2.61 | 10.48 | -39.4% | raw+fences | 294 | 8.96/8.63/12.98/2.97/6.83 | 3.126 | 1.3710 | 2.891 | 1.048 [1.03, 1.09] |
| genome_window | binary root (control) | 8.96 | 3.05 | 17.03 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.055 | 0.5706 | 2.533 | 1.000 [1.00, 1.00] |
|  | raw root | 7.27 | 3.05 | 15.35 | -9.9% | raw+ranks | 0 | 8.96/7.29/0.00/0.00/0.00 | 0.056 | 0.5706 | 2.452 | 0.968 [0.93, 0.99] |
|  | flow root | 7.27 | 3.05 | 15.35 | -9.9% | raw+ranks | 0 | 8.96/7.29/11.41/0.00/0.00 | 0.065 | 0.5706 | 2.347 | 0.916 [0.87, 0.95] |
|  | root fences | 2.38 | 3.05 | 10.45 | -38.6% | raw+fences | 130 | 8.96/7.29/0.00/2.49/0.00 | 0.062 | 0.5718 | 2.424 | 0.954 [0.90, 0.99] |
|  | root fusion | 2.38 | 3.05 | 10.45 | -38.6% | raw+fences | 130 | 8.96/7.29/11.41/2.49/6.58 | 0.078 | 0.5718 | 2.534 | 1.010 [0.96, 1.05] |
|  | region vp10 | 8.96 | 2.55 | 16.53 | -2.9% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 1.792 | 1.3694 | 2.673 | 1.041 [1.00, 1.10] |
|  | vp10 + root fusion | 2.38 | 2.55 | 9.95 | -41.6% | raw+fences | 130 | 8.96/7.29/11.41/2.49/6.58 | 2.039 | 1.3706 | 2.566 | 0.990 [0.92, 1.03] |
| history_uniform | binary root (control) | 8.96 | 2.46 | 16.45 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.054 | 0.5706 | 2.989 | 1.000 [1.00, 1.00] |
|  | raw root | 3.64 | 2.46 | 11.13 | -32.3% | raw+ranks | 0 | 8.96/3.78/0.00/0.00/0.00 | 0.054 | 0.5706 | 2.965 | 0.976 [0.96, 1.00] |
|  | flow root | 3.64 | 2.46 | 11.13 | -32.3% | raw+ranks | 0 | 8.96/3.78/6.71/0.00/0.00 | 0.060 | 0.5706 | 3.169 | 1.001 [0.92, 1.14] |
|  | root fences | 2.18 | 2.46 | 9.67 | -41.2% | raw+fences | 20 | 8.96/3.78/0.00/2.32/0.00 | 0.055 | 0.5716 | 3.369 | 1.078 [1.04, 1.13] |
|  | root fusion | 2.18 | 2.46 | 9.67 | -41.2% | raw+fences | 20 | 8.96/3.78/6.71/2.32/6.29 | 0.061 | 0.5716 | 3.208 | 1.015 [0.95, 1.14] |
|  | region vp10 | 8.96 | 2.07 | 16.06 | -2.3% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 3.666 | 1.3521 | 3.062 | 0.998 [0.88, 1.11] |
|  | vp10 + root fusion | 2.18 | 2.07 | 9.28 | -43.6% | raw+fences | 20 | 8.96/3.78/6.71/2.32/6.29 | 3.769 | 1.3530 | 3.219 | 1.043 [0.97, 1.11] |
| history_window | binary root (control) | 8.96 | 2.45 | 16.44 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.055 | 0.5706 | 3.271 | 1.000 [1.00, 1.00] |
|  | raw root | 3.76 | 2.45 | 11.25 | -31.6% | raw+ranks | 0 | 8.96/3.85/0.00/0.00/0.00 | 0.053 | 0.5706 | 3.500 | 1.037 [1.02, 1.07] |
|  | flow root | 3.76 | 2.45 | 11.25 | -31.6% | raw+ranks | 0 | 8.96/3.85/9.15/0.00/0.00 | 0.060 | 0.5706 | 3.302 | 1.008 [1.00, 1.02] |
|  | root fences | 2.10 | 2.45 | 9.58 | -41.7% | raw+fences | 20 | 8.96/3.85/0.00/2.25/0.00 | 0.054 | 0.5716 | 3.656 | 1.061 [0.99, 1.14] |
|  | root fusion | 2.10 | 2.45 | 9.58 | -41.7% | raw+fences | 20 | 8.96/3.85/9.15/2.25/6.35 | 0.065 | 0.5716 | 3.468 | 1.067 [1.00, 1.15] |
|  | region vp10 | 8.96 | 2.10 | 16.09 | -2.1% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 3.772 | 1.3498 | 3.353 | 1.013 [0.96, 1.06] |
|  | vp10 + root fusion | 2.10 | 2.10 | 9.23 | -43.8% | raw+fences | 20 | 8.96/3.85/9.15/2.25/6.35 | 3.782 | 1.3509 | 3.438 | 1.029 [0.99, 1.06] |
| libio_uniform | binary root (control) | 8.96 | 2.56 | 16.54 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.056 | 0.5706 | 3.325 | 1.000 [1.00, 1.00] |
|  | raw root | 6.53 | 2.56 | 14.11 | -14.7% | raw+ranks | 0 | 8.96/6.52/0.00/0.00/0.00 | 0.053 | 0.5706 | 3.033 | 0.937 [0.88, 1.04] |
|  | flow root | 6.53 | 2.56 | 14.11 | -14.7% | raw+ranks | 0 | 8.96/6.52/10.89/0.00/0.00 | 0.060 | 0.5706 | 3.223 | 0.977 [0.96, 1.00] |
|  | root fences | 2.32 | 2.56 | 9.90 | -40.2% | raw+fences | 117 | 8.96/6.52/0.00/2.46/0.00 | 0.067 | 0.5718 | 3.254 | 0.995 [0.96, 1.05] |
|  | root fusion | 2.32 | 2.56 | 9.90 | -40.2% | raw+fences | 117 | 8.96/6.52/10.89/2.46/6.27 | 0.081 | 0.5718 | 3.463 | 1.033 [1.02, 1.05] |
|  | region vp10 | 8.96 | 2.15 | 16.13 | -2.5% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 3.434 | 1.3527 | 3.158 | 0.963 [0.92, 1.01] |
|  | vp10 + root fusion | 2.32 | 2.15 | 9.49 | -42.6% | raw+fences | 117 | 8.96/6.52/10.89/2.46/6.27 | 3.681 | 1.3540 | 3.059 | 0.935 [0.90, 0.98] |
| libio_window | binary root (control) | 8.96 | 3.07 | 17.05 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.052 | 0.5706 | 3.403 | 1.000 [1.00, 1.00] |
|  | raw root | 3.60 | 3.07 | 11.70 | -31.4% | raw+ranks | 0 | 8.96/3.62/0.00/0.00/0.00 | 0.053 | 0.5706 | 3.380 | 1.006 [0.99, 1.03] |
|  | flow root | 3.60 | 3.07 | 11.70 | -31.4% | raw+ranks | 0 | 8.96/3.62/8.03/0.00/0.00 | 0.059 | 0.5706 | 3.235 | 0.950 [0.94, 0.96] |
|  | root fences | 2.25 | 3.07 | 10.35 | -39.3% | raw+fences | 25 | 8.96/3.62/0.00/2.41/0.00 | 0.055 | 0.5716 | 3.413 | 1.010 [0.98, 1.04] |
|  | root fusion | 2.25 | 3.07 | 10.35 | -39.3% | raw+fences | 25 | 8.96/3.62/8.03/2.41/6.37 | 0.066 | 0.5716 | 3.529 | 1.011 [0.97, 1.04] |
|  | region vp10 | 8.96 | 2.55 | 16.54 | -3.0% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 2.941 | 1.2503 | 3.359 | 0.998 [0.96, 1.02] |
|  | vp10 + root fusion | 2.25 | 2.55 | 9.83 | -42.3% | raw+fences | 25 | 8.96/3.62/8.03/2.41/6.37 | 2.916 | 1.2513 | 3.530 | 1.044 [1.00, 1.07] |
| osm_uniform | binary root (control) | 8.96 | 5.08 | 19.06 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.060 | 0.5706 | 2.456 | 1.000 [1.00, 1.00] |
|  | raw root | 8.96 | 5.08 | 19.06 | +0.0% | binary(fallback) | 0 | 8.96/11.67/0.00/0.00/0.00 | 0.057 | 0.5706 | 2.282 | 0.913 [0.87, 0.98] |
|  | flow root | 8.96 | 5.08 | 19.06 | -0.0% | binary(fallback) | 0 | 8.96/11.67/15.84/0.00/0.00 | 0.065 | 0.5706 | 2.110 | 0.851 [0.84, 0.86] |
|  | root fences | 8.17 | 5.08 | 18.28 | -4.1% | raw+fences | 1956 | 8.96/11.67/0.00/8.18/0.00 | 0.613 | 0.5755 | 2.164 | 0.894 [0.77, 0.96] |
|  | root fusion | 8.17 | 5.08 | 18.28 | -4.1% | raw+fences | 1956 | 8.96/11.67/15.84/8.18/12.20 | 1.171 | 0.5755 | 2.188 | 0.932 [0.86, 1.05] |
|  | region vp10 | 8.96 | 4.64 | 18.62 | -2.3% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 1.804 | 1.3694 | 2.227 | 0.918 [0.90, 0.95] |
|  | vp10 + root fusion | 8.17 | 4.64 | 17.84 | -6.4% | raw+fences | 1956 | 8.96/11.67/15.84/8.18/12.20 | 3.107 | 1.3743 | 2.117 | 0.892 [0.79, 0.96] |
| osm_window | binary root (control) | 8.96 | 4.37 | 18.36 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.059 | 0.5706 | 2.546 | 1.000 [1.00, 1.00] |
|  | raw root | 8.96 | 4.37 | 18.36 | +0.0% | binary(fallback) | 0 | 8.96/10.55/0.00/0.00/0.00 | 0.058 | 0.5706 | 2.283 | 0.904 [0.82, 1.00] |
|  | flow root | 8.96 | 4.37 | 18.36 | +0.0% | binary(fallback) | 0 | 8.96/10.55/14.38/0.00/0.00 | 0.064 | 0.5706 | 2.384 | 0.947 [0.87, 1.05] |
|  | root fences | 3.43 | 4.37 | 12.83 | -30.1% | raw+fences | 984 | 8.96/10.55/0.00/3.56/0.00 | 0.250 | 0.5735 | 2.361 | 0.960 [0.93, 0.98] |
|  | root fusion | 3.43 | 4.37 | 12.83 | -30.1% | raw+fences | 984 | 8.96/10.55/14.38/3.56/7.65 | 0.420 | 0.5735 | 2.412 | 0.975 [0.93, 1.05] |
|  | region vp10 | 8.96 | 3.77 | 17.76 | -3.2% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 2.057 | 1.3694 | 2.487 | 1.005 [0.97, 1.04] |
|  | vp10 + root fusion | 3.43 | 3.77 | 12.24 | -33.3% | raw+fences | 984 | 8.96/10.55/14.38/3.56/7.65 | 2.607 | 1.3724 | 2.436 | 0.979 [0.88, 1.13] |
| planet_uniform | binary root (control) | 8.96 | 3.23 | 17.21 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.057 | 0.5706 | 3.002 | 1.000 [1.00, 1.00] |
|  | raw root | 8.96 | 3.23 | 17.21 | +0.0% | binary(fallback) | 0 | 8.96/12.59/0.00/0.00/0.00 | 0.054 | 0.5706 | 2.717 | 0.903 [0.87, 0.94] |
|  | flow root | 8.96 | 3.23 | 17.21 | -0.0% | binary(fallback) | 0 | 8.96/12.59/14.58/0.00/0.00 | 0.061 | 0.5706 | 2.806 | 0.928 [0.89, 0.96] |
|  | root fences | 5.14 | 3.23 | 13.40 | -22.2% | raw+fences | 1956 | 8.96/12.59/0.00/5.24/0.00 | 0.642 | 0.5755 | 3.062 | 1.020 [0.99, 1.05] |
|  | root fusion | 5.14 | 3.23 | 13.40 | -22.2% | raw+fences | 1956 | 8.96/12.59/14.58/5.24/7.11 | 0.712 | 0.5755 | 2.969 | 0.976 [0.93, 1.01] |
|  | region vp10 | 8.96 | 2.60 | 16.59 | -3.6% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 2.427 | 1.3693 | 3.175 | 1.062 [1.06, 1.07] |
|  | vp10 + root fusion | 5.14 | 2.60 | 12.77 | -25.8% | raw+fences | 1956 | 8.96/12.59/14.58/5.24/7.11 | 3.579 | 1.3742 | 2.899 | 0.942 [0.89, 0.98] |
| planet_window | binary root (control) | 8.96 | 4.62 | 18.61 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.053 | 0.5706 | 3.088 | 1.000 [1.00, 1.00] |
|  | raw root | 7.60 | 4.62 | 17.25 | -7.3% | raw+ranks | 0 | 8.96/7.59/0.00/0.00/0.00 | 0.052 | 0.5706 | 2.943 | 0.934 [0.90, 0.97] |
|  | flow root | 7.60 | 4.62 | 17.25 | -7.3% | raw+ranks | 0 | 8.96/7.59/11.89/0.00/0.00 | 0.058 | 0.5706 | 2.966 | 0.962 [0.95, 0.98] |
|  | root fences | 3.17 | 4.62 | 12.82 | -31.1% | raw+fences | 206 | 8.96/7.59/0.00/3.26/0.00 | 0.069 | 0.5720 | 3.016 | 0.987 [0.95, 1.04] |
|  | root fusion | 3.17 | 4.62 | 12.82 | -31.1% | raw+fences | 206 | 8.96/7.59/11.89/3.26/7.21 | 0.096 | 0.5720 | 3.106 | 0.998 [0.96, 1.02] |
|  | region vp10 | 8.96 | 4.09 | 18.08 | -2.8% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 1.724 | 1.3694 | 2.921 | 0.961 [0.93, 1.01] |
|  | vp10 + root fusion | 3.17 | 4.09 | 12.29 | -34.0% | raw+fences | 206 | 8.96/7.59/11.89/3.26/7.21 | 2.104 | 1.3708 | 3.135 | 1.000 [0.94, 1.06] |
| stack_uniform | binary root (control) | 8.96 | 2.29 | 16.28 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.054 | 0.5706 | 3.416 | 1.000 [1.00, 1.00] |
|  | raw root | 3.34 | 2.29 | 10.66 | -34.5% | raw+ranks | 0 | 8.96/3.32/0.00/0.00/0.00 | 0.054 | 0.5706 | 3.318 | 1.006 [0.97, 1.05] |
|  | flow root | 3.34 | 2.29 | 10.66 | -34.5% | raw+ranks | 0 | 8.96/3.32/9.05/0.00/0.00 | 0.059 | 0.5706 | 3.370 | 1.014 [0.98, 1.08] |
|  | root fences | 2.17 | 2.29 | 9.49 | -41.7% | raw+fences | 13 | 8.96/3.32/0.00/2.31/0.00 | 0.055 | 0.5716 | 3.423 | 1.021 [0.94, 1.10] |
|  | root fusion | 2.17 | 2.29 | 9.49 | -41.7% | raw+fences | 13 | 8.96/3.32/9.05/2.31/6.30 | 0.063 | 0.5716 | 3.387 | 1.025 [0.97, 1.08] |
|  | region vp10 | 8.96 | 2.03 | 16.02 | -1.6% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 4.038 | 1.3078 | 3.370 | 1.006 [0.98, 1.02] |
|  | vp10 + root fusion | 2.17 | 2.03 | 9.23 | -43.3% | raw+fences | 13 | 8.96/3.32/9.05/2.31/6.30 | 4.064 | 1.3088 | 3.625 | 1.092 [1.06, 1.14] |
| stack_window | binary root (control) | 8.96 | 2.08 | 16.07 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.053 | 0.5706 | 3.524 | 1.000 [1.00, 1.00] |
|  | raw root | 6.27 | 2.08 | 13.38 | -16.7% | raw+ranks | 0 | 8.96/6.29/0.00/0.00/0.00 | 0.053 | 0.5706 | 3.602 | 1.023 [1.00, 1.04] |
|  | flow root | 6.27 | 2.08 | 13.38 | -16.7% | raw+ranks | 0 | 8.96/6.29/11.05/0.00/0.00 | 0.064 | 0.5706 | 3.419 | 0.967 [0.91, 1.02] |
|  | root fences | 2.08 | 2.08 | 9.19 | -42.8% | raw+fences | 41 | 8.96/6.29/0.00/2.25/0.00 | 0.055 | 0.5716 | 3.701 | 1.040 [1.03, 1.05] |
|  | root fusion | 2.08 | 2.08 | 9.19 | -42.8% | raw+fences | 41 | 8.96/6.29/11.05/2.25/6.27 | 0.069 | 0.5716 | 3.676 | 1.034 [0.92, 1.11] |
|  | region vp10 | 8.96 | 1.98 | 15.97 | -0.6% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 2.093 | 0.8187 | 3.399 | 0.967 [0.93, 1.00] |
|  | vp10 + root fusion | 2.08 | 1.98 | 9.09 | -43.4% | raw+fences | 41 | 8.96/6.29/11.05/2.25/6.27 | 2.147 | 0.8198 | 3.597 | 1.052 [1.00, 1.14] |
| wise_uniform | binary root (control) | 8.96 | 2.42 | 16.40 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.058 | 0.5706 | 2.806 | 1.000 [1.00, 1.00] |
|  | raw root | 5.73 | 2.42 | 13.18 | -19.6% | raw+ranks | 0 | 8.96/5.75/0.00/0.00/0.00 | 0.057 | 0.5706 | 2.748 | 0.990 [0.98, 1.00] |
|  | flow root | 5.73 | 2.42 | 13.18 | -19.6% | raw+ranks | 0 | 8.96/5.75/8.49/0.00/0.00 | 0.067 | 0.5706 | 2.555 | 0.904 [0.85, 0.96] |
|  | root fences | 2.65 | 2.42 | 10.10 | -38.4% | raw+fences | 89 | 8.96/5.75/0.00/2.73/0.00 | 0.065 | 0.5717 | 2.684 | 0.961 [0.94, 0.99] |
|  | root fusion | 2.65 | 2.42 | 10.10 | -38.4% | raw+fences | 89 | 8.96/5.75/8.49/2.73/6.74 | 0.077 | 0.5717 | 2.766 | 0.998 [0.95, 1.07] |
|  | region vp10 | 8.96 | 2.06 | 16.04 | -2.2% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 4.201 | 1.3504 | 2.575 | 0.937 [0.90, 0.99] |
|  | vp10 + root fusion | 2.65 | 2.06 | 9.74 | -40.6% | raw+fences | 89 | 8.96/5.75/8.49/2.73/6.74 | 4.208 | 1.3516 | 2.727 | 0.962 [0.87, 1.07] |
| wise_window | binary root (control) | 8.96 | 2.32 | 16.30 | +0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.057 | 0.5706 | 3.085 | 1.000 [1.00, 1.00] |
|  | raw root | 2.94 | 2.32 | 10.29 | -36.9% | raw+ranks | 0 | 8.96/2.99/0.00/0.00/0.00 | 0.059 | 0.5706 | 2.962 | 0.986 [0.93, 1.06] |
|  | flow root | 2.94 | 2.32 | 10.29 | -36.9% | raw+ranks | 0 | 8.96/2.99/8.24/0.00/0.00 | 0.061 | 0.5706 | 3.091 | 1.006 [0.94, 1.08] |
|  | root fences | 2.18 | 2.32 | 9.53 | -41.6% | raw+fences | 13 | 8.96/2.99/0.00/2.34/0.00 | 0.057 | 0.5716 | 3.009 | 0.985 [0.96, 1.01] |
|  | root fusion | 2.18 | 2.32 | 9.53 | -41.6% | raw+fences | 13 | 8.96/2.99/8.24/2.34/6.32 | 0.067 | 0.5716 | 3.178 | 1.027 [0.96, 1.09] |
|  | region vp10 | 8.96 | 2.02 | 16.00 | -1.8% | - | 0 | 0.00/0.00/0.00/0.00/0.00 | 4.286 | 1.3529 | 2.985 | 0.968 [0.93, 1.01] |
|  | vp10 + root fusion | 2.18 | 2.02 | 9.23 | -43.4% | raw+fences | 13 | 8.96/2.99/8.24/2.34/6.32 | 4.362 | 1.3539 | 2.813 | 0.930 [0.90, 0.95] |

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

| sample | variant (PRE-FIX rows) | chosen | root/op | transform_calls/op | virt | est bin/raw/flow/vf/fus | speedup vs (post-fix) control |
|---|---|---|---|---|---|---|---|
| books_uniform | root fusion | raw+fences | 2.60 | 0.053 | 662 | 8.96/11.08/14.91/2.73/6.61 | 1.073 [0.92, 1.18] |
| books_uniform | vp10 + root fusion | raw+fences | 2.60 | 0.053 | 662 | 8.96/11.08/14.91/2.73/6.61 | 1.132 [1.06, 1.21] |
| books_window | root fusion | flow+fences | 2.93 | 1.008 | 1374 | 8.96/14.24/13.95/13.90/6.99 | 0.881 [0.71, 0.99] |
| books_window | vp10 + root fusion | flow+fences | 2.93 | 1.008 | 1374 | 8.96/14.24/13.95/13.90/6.99 | 0.900 [0.82, 0.99] |
| covid_uniform | root fusion | flow+fences | 4.34 | 1.035 | 1339 | 8.96/13.78/13.99/13.04/8.33 | 0.848 [0.72, 0.97] |
| covid_uniform | vp10 + root fusion | flow+fences | 4.34 | 1.035 | 1339 | 8.96/13.78/13.99/13.04/8.33 | 0.878 [0.82, 1.00] |
| covid_window | root fusion | flow+fences | 4.73 | 1.002 | 1340 | 8.96/14.38/14.02/14.37/8.80 | 0.881 [0.84, 0.94] |
| covid_window | vp10 + root fusion | flow+fences | 4.73 | 1.002 | 1340 | 8.96/14.38/14.02/14.37/8.80 | 0.893 [0.87, 0.91] |
| fb_uniform | root fusion | raw+fences | 2.08 | 0.084 | 1 | 8.96/2.27/8.14/2.22/6.35 | 1.117 [1.09, 1.15] |
| fb_uniform | vp10 + root fusion | raw+fences | 2.08 | 0.084 | 1 | 8.96/2.27/8.14/2.22/6.35 | 1.031 [0.92, 1.12] |
| fb_window | root fusion | flow+fences | 2.88 | 1.025 | 1386 | 8.96/14.34/14.07/14.23/7.00 | 0.881 [0.81, 0.92] |
| fb_window | vp10 + root fusion | flow+fences | 2.88 | 1.025 | 1386 | 8.96/14.34/14.07/14.23/7.00 | 0.894 [0.85, 0.96] |
| genome_uniform | root fusion | raw+fences | 2.79 | 0.016 | 309 | 8.96/8.63/12.98/2.91/6.97 | 1.083 [1.02, 1.12] |
| genome_uniform | vp10 + root fusion | raw+fences | 2.79 | 0.016 | 309 | 8.96/8.63/12.98/2.91/6.97 | 1.096 [1.02, 1.19] |
| genome_window | root fusion | flow+fences | 4.67 | 1.018 | 1474 | 8.96/14.37/13.33/14.28/8.73 | 0.861 [0.84, 0.88] |
| genome_window | vp10 + root fusion | flow+fences | 4.67 | 1.018 | 1474 | 8.96/14.37/13.33/14.28/8.73 | 0.895 [0.86, 0.93] |
| history_uniform | root fusion | raw+fences | 2.18 | 0.027 | 20 | 8.96/3.78/6.71/2.32/6.29 | 1.071 [0.98, 1.16] |
| history_uniform | vp10 + root fusion | raw+fences | 2.18 | 0.027 | 20 | 8.96/3.78/6.71/2.32/6.29 | 1.143 [1.10, 1.24] |
| history_window | root fusion | flow+fences | 3.91 | 1.004 | 1399 | 8.96/14.33/13.79/14.21/8.05 | 0.855 [0.82, 0.90] |
| history_window | vp10 + root fusion | flow+fences | 3.91 | 1.004 | 1399 | 8.96/14.33/13.79/14.21/8.05 | 0.824 [0.78, 0.86] |
| libio_uniform | root fusion | raw+fences | 2.16 | 0.043 | 126 | 8.96/6.52/10.89/2.31/6.27 | 1.073 [1.05, 1.10] |
| libio_uniform | vp10 + root fusion | raw+fences | 2.16 | 0.043 | 126 | 8.96/6.52/10.89/2.31/6.27 | 1.086 [1.01, 1.17] |
| libio_window | root fusion | flow+fences | 3.48 | 1.010 | 1361 | 8.96/14.35/13.59/14.25/7.55 | 0.871 [0.84, 0.91] |
| libio_window | vp10 + root fusion | flow+fences | 3.48 | 1.010 | 1361 | 8.96/14.35/13.59/14.25/7.55 | 0.907 [0.85, 0.97] |
| osm_uniform | root fusion | raw+fences | 8.17 | 0.014 | 1956 | 8.96/11.66/15.83/8.19/12.19 | 0.919 [0.87, 1.00] |
| osm_uniform | vp10 + root fusion | raw+fences | 8.17 | 0.014 | 1956 | 8.96/11.66/15.83/8.19/12.19 | 0.985 [0.93, 1.04] |
| osm_window | root fusion | binary(fallback) | 8.96 | 0.043 | 0 | 8.96/14.38/15.63/14.36/10.51 | 1.019 [0.98, 1.05] |
| osm_window | vp10 + root fusion | binary(fallback) | 8.96 | 0.043 | 0 | 8.96/14.38/15.63/14.36/10.51 | 0.930 [0.87, 0.98] |
| planet_uniform | root fusion | raw+fences | 5.14 | 0.088 | 1956 | 8.96/12.59/14.58/5.24/7.11 | 1.023 [1.02, 1.03] |
| planet_uniform | vp10 + root fusion | raw+fences | 5.14 | 0.088 | 1956 | 8.96/12.59/14.58/5.24/7.11 | 1.012 [0.97, 1.06] |
| planet_window | root fusion | flow+fences | 4.77 | 1.035 | 1706 | 8.96/14.14/14.80/13.64/8.83 | 0.891 [0.87, 0.92] |
| planet_window | vp10 + root fusion | flow+fences | 4.77 | 1.035 | 1706 | 8.96/14.14/14.80/13.64/8.83 | 0.865 [0.84, 0.89] |
| stack_uniform | root fusion | raw+fences | 2.17 | 0.001 | 13 | 8.96/3.32/9.05/2.31/6.30 | 1.066 [0.99, 1.20] |
| stack_uniform | vp10 + root fusion | raw+fences | 2.17 | 0.001 | 13 | 8.96/3.32/9.05/2.31/6.30 | 1.061 [0.99, 1.19] |
| stack_window | root fusion | flow+fences | 3.76 | 1.000 | 1519 | 8.96/14.34/14.29/14.22/7.86 | 0.884 [0.85, 0.95] |
| stack_window | vp10 + root fusion | flow+fences | 3.76 | 1.000 | 1519 | 8.96/14.34/14.29/14.22/7.86 | 0.909 [0.85, 0.97] |
| wise_uniform | root fusion | raw+fences | 3.29 | 0.008 | 575 | 8.96/6.54/10.39/3.35/7.46 | 1.060 [1.02, 1.08] |
| wise_uniform | vp10 + root fusion | raw+fences | 3.29 | 0.008 | 575 | 8.96/6.54/10.39/3.35/7.46 | 1.071 [1.05, 1.09] |
| wise_window | root fusion | flow+fences | 2.56 | 1.002 | 1398 | 8.96/14.37/14.13/14.31/6.71 | 0.868 [0.86, 0.89] |
| wise_window | vp10 + root fusion | flow+fences | 2.56 | 1.002 | 1398 | 8.96/14.37/14.13/14.31/6.71 | 0.855 [0.79, 0.95] |

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

| region keys | dataset | regions | root/op (binary) | fence control | fence vp10 | Δ | fence flow forced | forced − control | fence flow_vp10 (forced) | fence fusion_auto | choices none/flow/vp/both | vp10 speedup | forced speedup | fusion speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | fb | 489 | 8.96 | 2.601 | 2.182 | -16.1% | 2.601 | +0.000 | 2.182 | 2.182 | 0/0/489/0 | 1.007 [0.95, 1.07] | 0.939 | 0.989 |
| 4096 | osm | 489 | 8.96 | 5.078 | 4.643 | -8.6% | 5.078 | -0.000 | 4.646 | 4.641 | 19/0/470/0 | 1.071 [0.96, 1.19] | 0.863 | 1.051 |
| 4096 | planet | 489 | 8.96 | 3.229 | 2.603 | -19.4% | 3.229 | +0.001 | 2.604 | 2.603 | 0/0/489/0 | 0.993 [0.96, 1.03] | 0.933 | 0.935 |
| 4096 | covid | 489 | 8.96 | 3.064 | 2.328 | -24.0% | 3.064 | -0.000 | 2.328 | 2.328 | 0/0/489/0 | 1.039 [0.97, 1.12] | 0.929 | 1.041 |
| 4096 | genome | 489 | 8.96 | 3.295 | 2.610 | -20.8% | 3.295 | -0.000 | 2.610 | 2.610 | 1/0/488/0 | 1.012 [1.01, 1.02] | 1.005 | 0.968 |
| 16384 | fb | 123 | 6.97 | 3.209 | 2.237 | -30.3% | 3.210 | +0.001 | 2.238 | 2.237 | 0/0/123/0 | 1.036 [0.93, 1.15] | 0.974 | 1.028 |
| 16384 | osm | 123 | 6.97 | 8.336 | 7.625 | -8.5% | 8.336 | -0.001 | 7.627 | 7.625 | 2/0/121/0 | 0.937 [0.88, 1.00] | 0.932 | 0.986 |
| 16384 | planet | 123 | 6.97 | 4.996 | 3.399 | -32.0% | 5.012 | +0.016 | 3.400 | 3.399 | 0/0/123/0 | 0.990 [0.96, 1.03] | 0.943 | 0.962 |
| 16384 | covid | 123 | 6.97 | 4.946 | 3.170 | -35.9% | 4.946 | +0.000 | 3.170 | 3.170 | 0/0/123/0 | 1.111 [1.09, 1.13] | 0.930 | 1.026 |
| 16384 | genome | 123 | 6.97 | 5.746 | 4.043 | -29.6% | 5.746 | -0.000 | 4.043 | 4.043 | 0/0/123/0 | 1.070 [1.01, 1.14] | 0.983 | 1.059 |
| 32768 | fb | 62 | 6.00 | 3.692 | 2.251 | -39.0% | 3.689 | -0.003 | 2.251 | 2.251 | 0/0/62/0 | 1.131 [1.05, 1.22] | 1.013 | 1.079 |
| 32768 | osm | 62 | 6.00 | 10.120 | 9.384 | -7.3% | 10.119 | -0.001 | 9.386 | 9.376 | 3/0/59/0 | 0.965 [0.89, 1.05] | 0.913 | 1.074 |
| 32768 | planet | 62 | 6.00 | 6.631 | 4.145 | -37.5% | 6.682 | +0.051 | 4.173 | 4.145 | 0/0/62/0 | 0.996 [0.89, 1.12] | 0.927 | 1.095 |
| 32768 | covid | 62 | 6.00 | 5.795 | 3.531 | -39.1% | 5.792 | -0.003 | 3.530 | 3.531 | 0/0/62/0 | 1.064 [1.01, 1.12] | 0.924 | 0.987 |
| 32768 | genome | 62 | 6.00 | 7.379 | 5.171 | -29.9% | 7.379 | -0.001 | 5.170 | 5.171 | 0/0/62/0 | 1.013 [0.95, 1.08] | 0.969 | 1.034 |

Readings:

- **Flow chosen in 0 of 6,740 regions** = 2 seeds × (5 × 489 + 5 × 123 + 5 × 62) = 2 × 3,370; virtual points chosen in 3,345 of 3,370 per seed (99.3 %); the 25 `none` regions are all osm (19 + 2 + 3) and one genome region at 4,096 [`choices` column].
- **Forced flow stays within 0.051 fence probes of the control at every size** (max = planet at 32,768: 6.682 vs 6.631; everything else within 0.016). So the "region absorbs the curvature" explanation survives an 8× larger region: at 32,768 keys a region spans ≈ 1.6 % of the key range and the tanh is still linear enough over it for the region's own line to absorb it.
- **The virtual-point saving grows with region size** because a single line fits a larger region worse: fb 2.601 → 2.182 (−16.1 %) at 4,096, 3.209 → 2.237 (−30.3 %) at 16,384, 3.692 → 2.251 (−39.0 %) at 32,768; covid −24.0 / −35.9 / −39.1 %; planet −19.4 / −32.0 / −37.5 %; genome −20.8 / −29.6 / −29.9 %; osm −8.6 / −8.5 / −7.3 % (always the smallest). In words: virtual points repair what the line cannot fit; a longer line has more to repair.
- Note the trade the larger region makes: root probes fall 8.96 → 6.97 → 6.00 (fewer fences) while control fence probes rise 2.6 → 3.2 → 3.7 (fb). Virtual points at 32,768 bring fence probes back to 2.25, i.e. to the 4,096-key level, with 2.96 fewer root probes.
- Throughput: vp10 0.937-1.131, forced flow 0.863-1.013 (one transform per lookup), fusion 0.935-1.095; two seeds per cell, so the brackets are wide.

---

## 5. E5 — Full scale: 200M keys, fb and planet

Protocol [results/aidb_fullscale/config*.json]: the full sorted GRE files (200,000,000 keys), single seed 11, 200,000 warm-up + 2,000,000 measured lookups, QoS on, 16 build threads, binary `build-root` (sha256 c63ceab0…, post-fix). Variants: the packed control, `root_raw`, `root_flow`, virtual-fence roots at `--root-alpha 0.04` (≈ 1,950 fences, the absolute count the 2M runs spent at α = 4) and at α = 4 (195,316 fences), `root_fusion` (α = 4), and region vp10 combined with each root. At the time of writing (2026-09-22) **14 of 16 planned runs are done**; `config_c2` (planet `root_fusion` at α = 4) is running (47 minutes elapsed when checked) and `config_c3` (planet `vp10_root_fusion` at α = 4) is queued, each under a 4-hour cap [results/aidb/run_chain3.log; `ps`]. The table below was recomputed with `python3 results/aidb/root_analysis.py results/aidb_fullscale/sweep/results.jsonl --json results/aidb/guide_deep/scratch/fullscale_root_analysis.json` and the per-run script; `results/aidb_fullscale/root_analysis.txt` predates the planet α = 4 run.

| dataset | variant | Mops | ratio vs control (1 seed) | build s | root/op | fence/op | coord/op | total/op | total Δ | chosen | virt. fences | est bin/raw/flow/vf/fus | meta B/key | smoothing s (thread) | transform_calls/op |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fb | packed_rank | 0.648 | 1.000 | 6.6 | 15.66 | 5.14 | 5.03 | 25.82 | -0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.570 | 0 | 0.0000 |
| fb | packed_rank_root_raw | 0.611 | 0.944 | 6.4 | 10.81 | 5.14 | 5.03 | 20.97 | -18.8% | raw+ranks | 0 | 15.66/10.80/0.00/0.00/0.00 | 0.570 | 0 | 0.0000 |
| fb | packed_rank_root_flow | 0.597 | 0.922 | 7.0 | 10.81 | 5.14 | 5.03 | 20.97 | -18.8% | raw+ranks | 0 | 15.66/10.80/21.02/0.00/0.00 | 0.570 | 0 | 0.0001 |
| fb | packed_rank_root_vf004 | 0.651 | 1.005 | 10.0 | 3.42 | 5.14 | 5.03 | 13.59 | -47.4% | raw+fences | 973 | 15.66/10.80/0.00/3.50/0.00 | 0.571 | 0 | 0.0000 |
| fb | packed_rank_root_vf4 | 0.812 | 1.253 | 10.0 | 3.42 | 5.14 | 5.03 | 13.59 | -47.4% | raw+fences | 973 | 15.66/10.80/0.00/3.50/0.00 | 0.571 | 0 | 0.0000 |
| fb | packed_rank_root_fusion | 0.723 | 1.117 | 26.7 | 3.42 | 5.14 | 5.03 | 13.59 | -47.4% | raw+fences | 973 | 15.66/10.80/21.02/3.50/6.93 | 0.571 | 0 | 0.0001 |
| fb | packed_rank_vp10_root_vf004 | 0.721 | 1.113 | 161.0 | 3.42 | 4.74 | 5.03 | 13.19 | -48.9% | raw+fences | 973 | 15.66/10.80/0.00/3.50/0.00 | 1.370 | 2,407 | 0.0000 |
| fb | packed_rank_vp10_root_fusion | 0.727 | 1.123 | 183.1 | 3.42 | 4.74 | 5.03 | 13.19 | -48.9% | raw+fences | 973 | 15.66/10.80/21.02/3.50/6.93 | 1.370 | 2,498 | 0.0001 |
| planet | packed_rank | 0.580 | 1.000 | 6.3 | 15.66 | 3.99 | 5.03 | 24.68 | -0.0% | binary | 0 | 0.00/0.00/0.00/0.00/0.00 | 0.570 | 0 | 0.0000 |
| planet | packed_rank_root_raw | 0.533 | 0.919 | 6.4 | 15.66 | 3.99 | 5.03 | 24.68 | -0.0% | binary(fallback) | 0 | 15.66/25.80/0.00/0.00/0.00 | 0.570 | 0 | 0.0000 |
| planet | packed_rank_root_flow | 0.571 | 0.985 | 7.0 | 15.66 | 3.99 | 5.03 | 24.68 | -0.0% | binary(fallback) | 0 | 15.66/25.80/27.87/0.00/0.00 | 0.570 | 0 | 0.0105 |
| planet | packed_rank_root_vf004 | 0.556 | 0.959 | 7.9 | 15.66 | 3.99 | 5.03 | 24.68 | -0.0% | binary(fallback) | 0 | 15.66/25.80/0.00/25.18/0.00 | 0.570 | 0 | 0.0000 |
| planet | packed_rank_root_vf4 | 0.643 | 1.110 | 2,067.8 | 13.69 | 3.99 | 5.03 | 22.70 | -8.0% | raw+fences | 195316 | 15.66/25.80/0.00/13.75/0.00 | 0.575 | 0 | 0.0000 |
| planet | packed_rank_vp10_root_vf004 | 0.530 | 0.914 | 190.3 | 15.66 | 3.42 | 5.03 | 24.10 | -2.3% | binary(fallback) | 0 | 15.66/25.80/0.00/25.18/0.00 | 1.364 | 2,903 | 0.0000 |

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

| metric | paper Conf | paper Cov | our Conf (mean over variants) | our Cov | |C| | |U| | Cov matches paper | Conf_I packed_rank | Conf_I vp10 | Conf_I flow forced | Conf_I sorted_vector |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RMSE | 0.19 | 1.00 | -0.317 | 1.000 | 45 | 0 | yes | -0.239 | -0.267 | -0.343 | -0.565 |
| ME | 0.15 | 1.00 | -0.217 | 1.000 | 45 | 0 | yes | -0.149 | -0.167 | -0.151 | -0.498 |
| CD | 0.28 | 1.00 | 0.179 | 1.000 | 45 | 0 | yes | 0.472 | 0.165 | 0.317 | -0.291 |
| PLA-32 | 0.50 | 1.00 | -0.306 | 1.000 | 45 | 0 | yes | -0.185 | -0.402 | -0.279 | -0.288 |
| PLA-4096 | 0.08 | 1.00 | -0.407 | 1.000 | 45 | 0 | yes | -0.371 | -0.396 | -0.457 | -0.294 |
| PLA-32·PLA-4096 | 0.71 | 0.47 | -0.281 | 0.467 | 33 | 12 | yes | -0.155 | -0.343 | -0.288 | -0.250 |
| RMSE·ME | 0.25 | 0.82 | -0.242 | 0.822 | 41 | 4 | yes | -0.149 | -0.192 | -0.221 | -0.533 |
| RMSE·CD | 0.60 | 0.47 | 0.151 | 0.422 | 32 | 13 | no (-0.048) | 0.431 | 0.194 | 0.227 | -0.448 |
| RMSE·PLA-32 | 0.64 | 0.60 | -0.264 | 0.600 | 36 | 9 | yes | -0.124 | -0.311 | -0.261 | -0.429 |
| RMSE·PLA-4096 | 0.52 | 0.42 | -0.272 | 0.422 | 32 | 13 | yes | -0.188 | -0.223 | -0.311 | -0.407 |
| ME·CD | 0.49 | 0.56 | 0.184 | 0.511 | 34 | 11 | no (-0.049) | 0.439 | 0.224 | 0.328 | -0.402 |
| ME·PLA-32 | 0.61 | 0.60 | -0.211 | 0.600 | 36 | 9 | yes | -0.067 | -0.257 | -0.152 | -0.405 |
| ME·PLA-4096 | 0.50 | 0.42 | -0.212 | 0.422 | 32 | 13 | yes | -0.128 | -0.155 | -0.193 | -0.378 |
| CD·PLA-32 | 0.69 | 0.60 | 0.092 | 0.556 | 35 | 10 | no (-0.044) | 0.382 | 0.017 | 0.208 | -0.247 |
| CD·PLA-4096 | 0.55 | 0.42 | 0.054 | 0.378 | 31 | 14 | no (-0.042) | 0.289 | 0.054 | 0.093 | -0.216 |
| RMSE·ME·CD | 0.59 | 0.42 | 0.140 | 0.378 | 31 | 14 | no (-0.042) | 0.411 | 0.157 | 0.293 | -0.474 |
| RMSE·ME·PLA-32 | 0.63 | 0.51 | -0.243 | 0.511 | 34 | 11 | yes | -0.096 | -0.310 | -0.185 | -0.431 |
| RMSE·ME·PLA-4096 | 0.50 | 0.33 | -0.249 | 0.333 | 30 | 15 | yes | -0.163 | -0.212 | -0.231 | -0.408 |
| RMSE·CD·PLA-32 | 0.83 | 0.33 | 0.117 | 0.289 | 29 | 16 | no (-0.041) | 0.398 | 0.137 | 0.180 | -0.412 |
| RMSE·CD·PLA-4096 | 0.70 | 0.16 | 0.064 | 0.111 | 25 | 20 | no (-0.049) | 0.318 | 0.022 | 0.095 | -0.292 |
| RMSE·PLA-32·PLA-4096 | 0.73 | 0.24 | -0.273 | 0.244 | 28 | 17 | yes | -0.124 | -0.361 | -0.277 | -0.348 |
| ME·CD·PLA-32 | 0.78 | 0.38 | 0.135 | 0.333 | 30 | 15 | no (-0.047) | 0.399 | 0.152 | 0.279 | -0.409 |
| ME·CD·PLA-4096 | 0.66 | 0.20 | 0.084 | 0.156 | 26 | 19 | no (-0.044) | 0.321 | 0.040 | 0.203 | -0.289 |
| ME·PLA-32·PLA-4096 | 0.69 | 0.24 | -0.206 | 0.244 | 28 | 17 | yes | -0.051 | -0.294 | -0.142 | -0.311 |
| CD·PLA-32·PLA-4096 | 0.78 | 0.24 | 0.023 | 0.200 | 27 | 18 | no (-0.040) | 0.251 | -0.008 | 0.044 | -0.180 |

Window run, scope `full` (same hardness, throughputs of the window sweep):

| metric | paper Conf | paper Cov | our Conf (mean over variants) | our Cov | |C| | |U| | Cov matches paper | Conf_I packed_rank | Conf_I vp10 | Conf_I flow forced | Conf_I sorted_vector |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RMSE | 0.19 | 1.00 | -0.287 | 1.000 | 45 | 0 | yes | -0.171 | -0.354 | -0.353 | -0.447 |
| ME | 0.15 | 1.00 | -0.210 | 1.000 | 45 | 0 | yes | 0.078 | -0.268 | -0.273 | -0.518 |
| CD | 0.28 | 1.00 | 0.277 | 1.000 | 45 | 0 | yes | 0.512 | 0.316 | 0.348 | -0.401 |
| PLA-32 | 0.50 | 1.00 | -0.197 | 1.000 | 45 | 0 | yes | -0.112 | -0.100 | -0.221 | -0.346 |
| PLA-4096 | 0.08 | 1.00 | -0.260 | 1.000 | 45 | 0 | yes | -0.210 | -0.295 | -0.394 | -0.036 |
| PLA-32·PLA-4096 | 0.71 | 0.47 | -0.085 | 0.467 | 33 | 12 | yes | 0.043 | -0.043 | -0.188 | -0.078 |
| RMSE·ME | 0.25 | 0.82 | -0.219 | 0.822 | 41 | 4 | yes | -0.005 | -0.307 | -0.290 | -0.465 |
| RMSE·CD | 0.60 | 0.47 | 0.242 | 0.422 | 32 | 13 | no (-0.048) | 0.544 | 0.173 | 0.261 | -0.401 |
| RMSE·PLA-32 | 0.64 | 0.60 | -0.175 | 0.600 | 36 | 9 | yes | -0.051 | -0.142 | -0.248 | -0.348 |
| RMSE·PLA-4096 | 0.52 | 0.42 | -0.154 | 0.422 | 32 | 13 | yes | 0.002 | -0.250 | -0.295 | -0.148 |
| ME·CD | 0.49 | 0.56 | 0.254 | 0.511 | 34 | 11 | no (-0.049) | 0.688 | 0.200 | 0.273 | -0.486 |
| ME·PLA-32 | 0.61 | 0.60 | -0.131 | 0.600 | 36 | 9 | yes | 0.106 | -0.089 | -0.195 | -0.412 |
| ME·PLA-4096 | 0.50 | 0.42 | -0.111 | 0.422 | 32 | 13 | yes | 0.178 | -0.198 | -0.242 | -0.233 |
| CD·PLA-32 | 0.69 | 0.60 | 0.246 | 0.556 | 35 | 10 | no (-0.044) | 0.480 | 0.360 | 0.279 | -0.348 |
| CD·PLA-4096 | 0.55 | 0.42 | 0.285 | 0.378 | 31 | 14 | no (-0.042) | 0.553 | 0.289 | 0.226 | -0.106 |
| RMSE·ME·CD | 0.59 | 0.42 | 0.236 | 0.378 | 31 | 14 | no (-0.042) | 0.656 | 0.130 | 0.226 | -0.432 |
| RMSE·ME·PLA-32 | 0.63 | 0.51 | -0.155 | 0.511 | 34 | 11 | yes | 0.062 | -0.136 | -0.237 | -0.385 |
| RMSE·ME·PLA-4096 | 0.50 | 0.33 | -0.134 | 0.333 | 30 | 15 | yes | 0.134 | -0.254 | -0.288 | -0.187 |
| RMSE·CD·PLA-32 | 0.83 | 0.33 | 0.230 | 0.289 | 29 | 16 | no (-0.041) | 0.492 | 0.240 | 0.279 | -0.348 |
| RMSE·CD·PLA-4096 | 0.70 | 0.16 | 0.262 | 0.111 | 25 | 20 | no (-0.049) | 0.610 | 0.128 | 0.226 | -0.177 |
| RMSE·PLA-32·PLA-4096 | 0.73 | 0.24 | -0.094 | 0.244 | 28 | 17 | yes | 0.096 | -0.094 | -0.270 | -0.082 |
| ME·CD·PLA-32 | 0.78 | 0.38 | 0.241 | 0.333 | 30 | 15 | no (-0.047) | 0.641 | 0.246 | 0.281 | -0.413 |
| ME·CD·PLA-4096 | 0.66 | 0.20 | 0.271 | 0.156 | 26 | 19 | no (-0.044) | 0.778 | 0.135 | 0.228 | -0.271 |
| ME·PLA-32·PLA-4096 | 0.69 | 0.24 | -0.037 | 0.244 | 28 | 17 | yes | 0.322 | -0.018 | -0.203 | -0.188 |
| CD·PLA-32·PLA-4096 | 0.78 | 0.24 | 0.249 | 0.200 | 27 | 18 | no (-0.040) | 0.498 | 0.347 | 0.134 | -0.141 |

Conformance of each variant on the five scalar metrics, uniform run, scope `full`:

| variant | RMSE | ME | CD | PLA-32 | PLA-4096 |
|---|---|---|---|---|---|
| sorted_vector | -0.565 | -0.498 | -0.291 | -0.288 | -0.294 |
| raw_rank | -0.297 | -0.289 | -0.470 | -0.467 | -0.561 |
| packed_rank | -0.239 | -0.149 | 0.472 | -0.185 | -0.371 |
| packed_rank_flow | -0.316 | -0.111 | 0.079 | -0.436 | -0.430 |
| packed_rank_flow_forced | -0.343 | -0.151 | 0.317 | -0.279 | -0.457 |
| packed_rank_vp10 | -0.267 | -0.167 | 0.165 | -0.402 | -0.396 |
| packed_rank_flow_vp10 | -0.294 | -0.213 | 0.226 | -0.323 | -0.406 |
| packed_byte | -0.147 | -0.040 | 0.647 | -0.089 | -0.293 |
| packed_rank_fusion_auto | -0.314 | -0.225 | 0.402 | -0.249 | -0.442 |
| packed_rank_flow_costsel | -0.393 | -0.321 | 0.245 | -0.341 | -0.416 |

Orderings of the ten datasets by each raw scalar metric on the full files (the coverage of a scalar metric is 1.0 by construction; the *orderings* are what the compositions' coverage depends on):

| metric (full 200M keys, raw) | easiest → hardest |
|---|---|
| RMSE | history < stack < covid < wise < libio < genome < books < osm < planet < fb |
| ME | history < stack < wise < covid < libio < genome < planet < osm < books < fb |
| CD | stack < libio < history < wise < planet < covid < fb < books < genome < osm |
| PLA-32 | stack < wise < covid < history < libio < books < planet < osm < fb < genome |
| PLA-4096 | books < stack < wise < history < libio < covid < genome < fb < planet < osm |

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

| sample | region vp10 | raw root | flow root | root fences | root fusion | vp10 + root fusion | raw_rank | sorted_vector |
|---|---|---|---|---|---|---|---|---|
| books_uniform | 1.012 [0.89, 1.14] | 0.948 [0.86, 1.03] | 0.924 [0.76, 1.02] | 0.956 [0.89, 1.08] | 1.045 [0.89, 1.16] | 0.977 [0.83, 1.07] | 1.508 [1.31, 1.86] | 2.081 [1.90, 2.35] |
| books_window | 1.014 [0.89, 1.11] | 1.004 [0.88, 1.19] | 1.076 [1.04, 1.11] | 0.941 [0.82, 1.03] | 0.977 [0.92, 1.03] | 1.093 [0.99, 1.24] | 1.519 [1.36, 1.74] | 2.294 [2.09, 2.47] |
| covid_uniform | 0.963 [0.92, 1.03] | 1.019 [1.01, 1.02] | 0.948 [0.92, 0.98] | 0.928 [0.91, 0.96] | 0.958 [0.92, 1.00] | 0.931 [0.89, 0.98] | 1.560 [1.50, 1.67] | 2.069 [1.47, 2.66] |
| covid_window | 1.001 [0.96, 1.03] | 0.983 [0.97, 1.00] | 0.920 [0.84, 0.97] | 0.984 [0.97, 1.00] | 1.030 [0.99, 1.07] | 0.949 [0.83, 1.05] | 1.107 [1.07, 1.16] | 2.101 [1.52, 2.50] |
| fb_uniform | 1.006 [0.94, 1.10] | 1.061 [0.97, 1.18] | 1.008 [0.89, 1.11] | 1.055 [1.02, 1.11] | 1.076 [1.03, 1.10] | 1.016 [0.94, 1.08] | 1.355 [1.23, 1.46] | 1.823 [1.26, 2.53] |
| fb_window | 0.968 [0.93, 1.01] | 0.998 [0.95, 1.05] | 1.017 [0.98, 1.05] | 1.021 [0.97, 1.06] | 1.029 [1.00, 1.07] | 0.993 [0.96, 1.01] | 1.085 [0.99, 1.23] | 1.822 [1.72, 1.93] |
| genome_uniform | 1.053 [1.01, 1.09] | 0.957 [0.87, 1.02] | 0.975 [0.96, 1.01] | 1.020 [0.95, 1.13] | 0.989 [0.95, 1.07] | 1.048 [1.03, 1.09] | 1.157 [1.08, 1.24] | 1.662 [1.33, 2.34] |
| genome_window | 1.041 [1.00, 1.10] | 0.968 [0.93, 0.99] | 0.916 [0.87, 0.95] | 0.954 [0.90, 0.99] | 1.010 [0.96, 1.05] | 0.990 [0.92, 1.03] | 1.435 [1.34, 1.59] | 1.938 [1.64, 2.61] |
| history_uniform | 0.998 [0.88, 1.11] | 0.976 [0.96, 1.00] | 1.001 [0.92, 1.14] | 1.078 [1.04, 1.13] | 1.015 [0.95, 1.14] | 1.043 [0.97, 1.11] | 0.988 [0.97, 1.02] | 1.597 [1.20, 1.87] |
| history_window | 1.013 [0.96, 1.06] | 1.037 [1.02, 1.07] | 1.008 [1.00, 1.02] | 1.061 [0.99, 1.14] | 1.067 [1.00, 1.15] | 1.029 [0.99, 1.06] | 1.050 [0.86, 1.28] | 1.164 [1.06, 1.29] |
| libio_uniform | 0.963 [0.92, 1.01] | 0.937 [0.88, 1.04] | 0.977 [0.96, 1.00] | 0.995 [0.96, 1.05] | 1.033 [1.02, 1.05] | 0.935 [0.90, 0.98] | 1.023 [0.90, 1.15] | 1.580 [1.23, 1.82] |
| libio_window | 0.998 [0.96, 1.02] | 1.006 [0.99, 1.03] | 0.950 [0.94, 0.96] | 1.010 [0.98, 1.04] | 1.011 [0.97, 1.04] | 1.044 [1.00, 1.07] | 1.003 [0.90, 1.12] | 1.357 [1.11, 1.83] |
| osm_uniform | 0.918 [0.90, 0.95] | 0.913 [0.87, 0.98] | 0.851 [0.84, 0.86] | 0.894 [0.77, 0.96] | 0.932 [0.86, 1.05] | 0.892 [0.79, 0.96] | 1.458 [1.35, 1.54] | 1.740 [1.45, 2.17] |
| osm_window | 1.005 [0.97, 1.04] | 0.904 [0.82, 1.00] | 0.947 [0.87, 1.05] | 0.960 [0.93, 0.98] | 0.975 [0.93, 1.05] | 0.979 [0.88, 1.13] | 1.322 [1.28, 1.41] | 1.909 [1.49, 2.56] |
| planet_uniform | 1.062 [1.06, 1.07] | 0.903 [0.87, 0.94] | 0.928 [0.89, 0.96] | 1.020 [0.99, 1.05] | 0.976 [0.93, 1.01] | 0.942 [0.89, 0.98] | 1.279 [1.10, 1.43] | 1.691 [1.38, 2.16] |
| planet_window | 0.961 [0.93, 1.01] | 0.934 [0.90, 0.97] | 0.962 [0.95, 0.98] | 0.987 [0.95, 1.04] | 0.998 [0.96, 1.02] | 1.000 [0.94, 1.06] | 1.134 [1.02, 1.27] | 1.911 [1.67, 2.20] |
| stack_uniform | 1.006 [0.98, 1.02] | 1.006 [0.97, 1.05] | 1.014 [0.98, 1.08] | 1.021 [0.94, 1.10] | 1.025 [0.97, 1.08] | 1.092 [1.06, 1.14] | 1.002 [0.89, 1.16] | 1.304 [1.30, 1.31] |
| stack_window | 0.967 [0.93, 1.00] | 1.023 [1.00, 1.04] | 0.967 [0.91, 1.02] | 1.040 [1.03, 1.05] | 1.034 [0.92, 1.11] | 1.052 [1.00, 1.14] | 1.004 [0.83, 1.20] | 1.147 [0.91, 1.37] |
| wise_uniform | 0.937 [0.90, 0.99] | 0.990 [0.98, 1.00] | 0.904 [0.85, 0.96] | 0.961 [0.94, 0.99] | 0.998 [0.95, 1.07] | 0.962 [0.87, 1.07] | 1.304 [1.04, 1.47] | 1.372 [1.21, 1.58] |
| wise_window | 0.968 [0.93, 1.01] | 0.986 [0.93, 1.06] | 1.006 [0.94, 1.08] | 0.985 [0.96, 1.01] | 1.027 [0.96, 1.09] | 0.930 [0.90, 0.95] | 1.152 [0.96, 1.29] | 1.607 [1.15, 1.91] |

Pooled geometric means over all 60 (sample, seed) pairs, with the extreme single pair: region vp10 0.992 (0.88-1.14), raw root 0.977 (0.82-1.19), flow root 0.964 (0.76-1.14), root fences 0.992 (0.77-1.14), root fusion 1.010 (0.86-1.16), vp10 + root fusion 0.993 (0.79-1.24), raw_rank 1.208 (0.83-1.86), sorted_vector 1.678 (0.91-2.66) [key_numbers.json `e8_pooled`].

Per sample: region vp10 0.918 (osm_uniform) to 1.062 (planet_uniform); 15 of the 20 seed-spread brackets include 1.0 (those that do not: genome_uniform 1.053 [1.01, 1.09], genome_window 1.041 [1.01, 1.10], planet_uniform 1.062 [1.06, 1.07] above; osm_uniform 0.918 [0.90, 0.95], wise_uniform 0.937 [0.90, 0.99] below) [MEETING_NOTES §1 says 16 of 20] [unverified: the count depends on the bootstrap's resampling, my recount of root_analysis.json's own lo/hi gives 15]. Root fences 0.894 (osm_uniform) to 1.078 (history_uniform); root fusion 0.932 to 1.076.

### 8.2 The noise band: how far apart two measurements of the same structure land

Two ways to measure it from the final sweep itself:

- `root fusion` vs `root fences` when both chose raw+fences (all 20 samples): the root and fence counters are identical; the only difference is that `root fusion` has the flow file loaded and evaluates it in the 0.0-8.8 % of regions the region-level bypass accepted (`transform_calls` per lookup 0.0006-0.088). Per-seed ratios span **0.842 to 1.245** (60 pairs); per-sample geometric means span **0.942 (history_uniform) to 1.092 (books_uniform)** [key_numbers.json `e8_noise_same_structure`, `e8_noise_same_per_sample`].
- `raw root` that fell back to the binary search vs the control (identical lookup path, 4 samples × 3 seeds): per-seed ratios 0.824-1.029, per-sample 0.903 (planet_uniform), 0.904 (osm_window), 0.913 (osm_uniform), 0.948 (books_uniform) [`e8_noise_fallback`]. All four are below 1.0, which is consistent with a session effect: the root variants were measured in the second pass (20:46-21:17) and the control in the first (19:16-20:11) [READING_GUIDE §5 E3], so every root-variant speedup in §8.1 is a cross-session pair.
- The pre-fix verifier had 75 such sham pairs: ratio mean +0.3 %, sd 5.8 %, range −10.2 to +18.9 % [coststory.txt "NOISE FLOOR"].

Conclusion: per-dataset time effects below ≈ 6-10 % are not resolvable here, and single pairs can differ by 25 %. The meeting notes' "up to 8 % (10 % for the binary-fallback root)" [MEETING_NOTES §3] is the per-sample statement; the per-seed extremes are wider.

### 8.3 Where the time goes: the uncompressed-rank ablation

| sample | ns/lookup packed_rank | ns/lookup raw_rank | ns/lookup sorted_vector | sorted_vector × | raw_rank × | gap closed by removing decoding | gap remaining (routing) | meta B/key packed | B/key packed | B/key raw |
|---|---|---|---|---|---|---|---|---|---|---|
| books_uniform | 442.7 | 277.5 | 233.5 | 2.081 | 1.508 | 79% | 21% | 0.57 | 14.53 | 16.70 |
| books_window | 400.6 | 266.0 | 162.0 | 2.294 | 1.519 | 56% | 44% | 0.57 | 14.03 | 16.70 |
| covid_uniform | 404.1 | 246.4 | 164.6 | 2.069 | 1.560 | 66% | 34% | 0.57 | 13.78 | 16.70 |
| covid_window | 372.8 | 329.1 | 152.9 | 2.101 | 1.107 | 20% | 80% | 0.57 | 12.94 | 16.70 |
| fb_uniform | 340.7 | 245.1 | 179.2 | 1.823 | 1.355 | 59% | 41% | 0.57 | 11.39 | 16.70 |
| fb_window | 332.1 | 318.0 | 187.7 | 1.822 | 1.085 | 10% | 90% | 0.57 | 10.50 | 16.70 |
| genome_uniform | 359.5 | 310.9 | 256.7 | 1.662 | 1.157 | 47% | 53% | 0.57 | 11.56 | 16.70 |
| genome_window | 394.8 | 282.9 | 236.9 | 1.938 | 1.435 | 71% | 29% | 0.57 | 10.68 | 16.70 |
| history_uniform | 334.5 | 346.0 | 183.8 | 1.597 | 0.988 | -8% | 108% | 0.57 | 10.87 | 16.70 |
| history_window | 305.7 | 275.5 | 267.7 | 1.164 | 1.050 | 80% | 20% | 0.57 | 10.04 | 16.70 |
| libio_uniform | 300.8 | 308.6 | 181.1 | 1.580 | 1.023 | -7% | 107% | 0.57 | 10.36 | 16.70 |
| libio_window | 293.9 | 306.8 | 233.8 | 1.357 | 1.003 | -22% | 122% | 0.57 | 9.43 | 16.70 |
| osm_uniform | 407.2 | 288.9 | 235.8 | 1.740 | 1.458 | 69% | 31% | 0.57 | 14.25 | 16.70 |
| osm_window | 392.8 | 305.5 | 239.6 | 1.909 | 1.322 | 57% | 43% | 0.57 | 13.48 | 16.70 |
| planet_uniform | 333.1 | 250.4 | 205.1 | 1.691 | 1.279 | 65% | 35% | 0.57 | 10.79 | 16.70 |
| planet_window | 323.8 | 286.4 | 172.0 | 1.911 | 1.134 | 25% | 75% | 0.57 | 10.27 | 16.70 |
| stack_uniform | 292.7 | 299.1 | 224.7 | 1.304 | 1.002 | -9% | 109% | 0.57 | 10.20 | 16.70 |
| stack_window | 283.8 | 289.5 | 229.2 | 1.147 | 1.004 | -10% | 110% | 0.57 | 9.19 | 16.70 |
| wise_uniform | 356.4 | 245.9 | 266.0 | 1.372 | 1.304 | 122% | -22% | 0.57 | 12.09 | 16.70 |
| wise_window | 324.2 | 262.3 | 178.1 | 1.607 | 1.152 | 42% | 58% | 0.57 | 11.26 | 16.70 |

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
