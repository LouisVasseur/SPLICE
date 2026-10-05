# 03. How Hard Can Indexing Be? (Zhang, Tang, Ailamaki, AIDB@VLDB 2026), deep

What this section gives you: a self-contained account of the AIDB hardness paper at the level of the equations and of the numbers, so that you can define every one of its five scalar metrics on a napkin (with an eight-key example computed here), state the harder-than partial order, coverage and conformance exactly (equations 1-3, with worked examples), reproduce Table 2 and read it the way the authors do (Findings 1-7), describe the six indexes, the ten datasets and the measurement protocol precisely, and then say exactly which parts of that paper our study reuses (the ten datasets, a clean-room reimplementation of the five metrics, and a clean-room reimplementation of the conformance/coverage scorer applied to our own SCALE-LI control variants) and what matched. Every number is cited to the PDF, to the code, or to a result file; anything read off a figure or not stated in a source is tagged. Worked numbers were computed with `scratch/worked_examples_03.py` and `scratch/table2_gains_03.py` (standard library only; outputs in `scratch/worked_examples_03.out.txt`). Notation: keys k₁ < … < kₙ (uint64), rank r(kᵢ) = i − 1; the paper writes S = ⟨x₁ < … < x_N⟩ and R(x) = |{xᵢ ≤ x}| so that R(xᵢ) = i; the offset of one is absorbed by the intercept of every linear fit and changes none of the metrics.

---

## 1. Bibliographic facts and the question the paper asks

**Reference.** Siyuan Zhang, Chuzhe Tang, Anastasia Ailamaki (EPFL). "How Hard Can Indexing Be? Principled Dataset Hardness Measurement for Learned Indexes." VLDB 2026 Workshop: Applied AI for Database Systems and Applications (AIDB 2026); CC BY-NC-ND 4.0 [AIDB p. 1, header and footer]. Local copy: `/Users/louisvasseur/Downloads/aidb26_11.pdf` (9 pages; text extraction `build-verify-guide/scratch/aidb_fitz.txt`). Sections: 1 Introduction, 2 Background and Motivation (2.1 preliminaries, 2.2 measuring hardness and the GRE metric), 3 Metric Conformance and Coverage (3.1 strawman, 3.2 proposed measurement), 4 Experimental Results (4.1 setup, 4.2 results and analysis with Findings 1-7), 5 Discussion (open problems), 6 Related Work, 7 Conclusion.

**The question.** "How can we quantitatively measure dataset hardness, i.e., the difficulty of fitting a dataset with a learned index?" [AIDB §1]. The motivation is that learned-index performance depends heavily on the data distribution (ALEX varies from 2.8 MOPS to 6.0 MOPS across datasets [AIDB §1, citing [34] Fig. 8]), yet hardness only appears anecdotally: "a synthetic dataset [Linear] and three real-world datasets [Covid, Face, OSM] with increasing fitting difficulty" [18], or bare labels "Libio (easy)" / "Genome (hard)" [25] [AIDB §1]. Only Wongkham et al. [27] (the GRE benchmark paper, "Are updatable learned indexes ready?", PVLDB 15(11), 2022) had proposed a metric, without validating it [AIDB §1, §2.2].

**What a hardness metric is, formally** [AIDB §2.2]. A hardness metric h(S) takes a dataset S and outputs a value, possibly in a multidimensional space. Its contract: if h(S₁) < h(S₂) then S₁ is easier for learned indexes to fit and "thus yields better performance than S₂". Performance is fixed for the whole paper as "the index throughput under a uniform read-only workload over the same dataset used to populate the index". So a hardness metric is a *predictor of the ordering of lookup throughput across datasets*, not a predictor of absolute throughput and not a property of one index.

**The two quality properties the paper proposes** [AIDB §1, §3]:
- *Conformance*: how well the metric's ordering agrees with actual index performance on the pairs of datasets the metric can order (index-dependent; computed per index and averaged).
- *Coverage*: the fraction of dataset pairs the metric can order at all (index-agnostic; an intrinsic property of the metric).
Together they "capture the tradeoff between accuracy and applicability" [AIDB §1]; a multidimensional metric can be very accurate on the few pairs it orders and useless on the rest. The scores deliberately "place more emphasis on the challenging datasets that better differentiate candidate metrics, and less on the easy datasets that any reasonable metric should resolve correctly" [AIDB §1]; that emphasis is the sigmoid importance weight of section 4.4 below.

**Preliminaries you need** [AIDB §2.1]. Sorted keys S = ⟨x₁ < … < x_N⟩; empirical CDF F(x) = |{xᵢ ≤ x}|/N ∈ [0, 1]; rank function R(x) = |{xᵢ ≤ x}| ∈ {0, …, N} = N·F(x), which "directly gives the position of x in S". A learned index approximates R(·) with a hierarchy of models trained on the pairs {(xᵢ, R(xᵢ))}; at query time it predicts p̂ and recovers the exact position by "a small verification search around p̂, typically within a model-specific error window [p̂ − ε, p̂ + ε] clipped to [1, N]"; linear models are the norm. The paper's taxonomy of how indexes build their trees is what later explains which metric fits which index:

| partitioning strategy | indexes | tree shape | last-mile handling |
|---|---|---|---|
| error-driven greedy splits bounding each segment's error by ε | PGM-index [2], FITing-Tree [3] | balanced | local correction (bounded binary search) |
| even splits with error checking | XIndex [24] | (not stated) | local correction |
| cost-based splits minimising an estimated lookup cost | ALEX [1] | unbalanced, adaptive | local correction (exponential search) |
| conflict-based splits recursing where keys collide under the parent model | LIPP [29] | unbalanced, adaptive | none: each key sits in its predicted slot, a new node is chained on collision |

[AIDB §2.1: "Most perform a local correction around p̂ via either a bounded binary search or an exponential search, while LIPP eliminates correction entirely"]. The pairing of a strategy with a specific search type in the last column is my reading of §2.1 plus the original papers, not a statement the AIDB paper makes per index [unverified: per-index search type not stated in AIDB]. "Because each of these choices interacts with the shape of R(·), the same dataset can be easy for one learned index and hard for another" [AIDB §2.1]; that sentence is the whole reason conformance has to be measured per index.

---

## 2. The GRE metric (§2.2): PLA-32, PLA-4096, and why it fails

**Definition** [AIDB §2.2]. Wongkham et al. segment the dataset with piecewise linear approximation (PLA, O'Rourke 1981 [19]) and use the number of segments as hardness. Two error thresholds are used, 32 and 4096, giving a *local* dimension h_l = PLA-32 and a *global* dimension h_g = PLA-4096. The justification: "PLA produces the theoretical minimum number of linear models required to fit a dataset under error constraints. Therefore, intuitively, the number of segments reflects overall learning difficulty." Section 3.4 below defines PLA-ε precisely.

**Why the paper is unconvinced** [AIDB §2.2]. Wongkham et al. only reported how h_l and h_g *individually* correlate with performance on two indexes; nobody validated the two-dimensional metric. The paper's counterexample uses two of its own datasets:

| dataset | h_l = PLA-32 (paper) | h_g = PLA-4096 (paper) | our full-file value [results/aidb/hardness.json] | LIPP throughput (paper) |
|---|---|---|---|---|
| history | 105k | 468 | 105,468 / 468 | 5.6 MOPS |
| libio | 146k | 639 | 145,808 / 639 | 7.1 MOPS |

Under GRE, libio is harder than history on both dimensions (146k > 105k and 639 > 468), yet LIPP is *faster* on libio (7.1 vs 5.6 MOPS): in the vocabulary of section 4, (libio, history) is a comparable pair that LIPP *violates*. Our recomputation of PLA-32/PLA-4096 on the 200M-key files reproduces the two values the paper prints (PLA-4096 exactly, PLA-32 to the three digits the paper gives) [results/aidb/hardness.json history.full, libio.full; AIDB §2.2].

**Figure 2** [AIDB p. 3] plots the ten datasets in the (PLA-32, PLA-4096) plane; "27% (12/45) of the possible pairs are incomparable, spanning all ten datasets" [AIDB §3.1]. Recomputed from our full-file table, the twelve incomparable pairs under GRE are (books, covid), (books, history), (books, libio), (books, stack), (books, wise), (covid, history), (covid, libio), (fb, genome), (fb, osm), (fb, planet), (genome, osm), (genome, planet) [scratch/worked_examples_03.out.txt, section C; results/aidb/scores.json full.metrics."PLA-32·PLA-4096".pairs.incomparable]. Each dataset has between 4 (books) and 8 (stack, wise) comparable partners, which is the range of the colour scale printed on Figure 2 [unverified: colour-scale endpoints 4 and 8 read from the figure]. Example of an incomparable pair: books (262,604; 97) versus covid (81,908; 850): books is locally harder, covid globally harder, so GRE cannot say which is harder.

**Where the failure shows in Table 2.** GRE = PLA-32·PLA-4096 reaches Conf = 0.71 with Cov = 0.47, but its per-index conformance is RMI 1.00, PGM 0.71, ALEX 1.00, LIPP 0.42, XIndex 0.27, FINEdex 0.86 [AIDB Table 2]: it is a good predictor for RMI and ALEX and a poor one for LIPP and XIndex, on top of leaving 12 of 45 pairs unordered. So "does GRE fail?" has a two-part answer: it fails on coverage by construction (any 2-D metric does) and it fails on conformance for the conflict-based index (LIPP) because neither of its dimensions measures conflicts (section 6, Finding 3).

---

## 3. The five scalar metrics, with an eight-key worked example

Setting common to all five [AIDB §4.1]: the input is the sorted key set; every metric is O(N) on sorted keys; "they typically take 10-20 seconds to compute for each dataset". Our tool computes all five on the 200M-key files in 12.9 s (history) to 27.4 s (wise) wall clock with 16 threads, of which 2.7 s (history) is the scalar-metric pass proper [results/aidb/hardness_details.json <name>.full.elapsed_ns; results/aidb/hardness/history_full.json original_metrics_ns].

The toy set used below (8 keys, so LIPP's small-node parameters apply; nothing about it is representative of 200M keys, it exists to make every formula concrete):

```
keys  k = [ 1,  2,  3,  4, 10, 20, 30, 100]
ranks r = [ 0,  1,  2,  3,  4,  5,  6,   7]     (n = 8)
```

### 3.1 RMSE and ME: one least-squares line key → rank

Definition [AIDB §4.1]: "RMSE is the root mean square error of a linear model fitted on the dataset using least squares regression"; "ME is the maximum error of a linear model fitted on the dataset using least squares regression". One line over the *whole* dataset, so both measure global non-linearity.

```
fit   f(k) = w·k + b   minimising  Σᵢ (rᵢ − f(kᵢ))²
      w = Sₓᵧ / Sₓₓ,  b = r̄ − w·k̄,   Sₓₓ = Σ (kᵢ − k̄)²,  Sₓᵧ = Σ (kᵢ − k̄)(rᵢ − r̄)
      eᵢ = rᵢ − f(kᵢ)
RMSE  = sqrt( (1/n) Σᵢ eᵢ² )
ME    = maxᵢ |eᵢ|
```

Worked example [scratch/worked_examples_03.out.txt, section A]:

```
k̄ = 21.25,  r̄ = 3.5,  Sₓₓ = 7817.5,  Sₓᵧ = 445.0
w = 445.0 / 7817.5 = 0.056924,   b = 3.5 − 0.056924·21.25 = 2.290374
residuals e = [−2.3473, −1.4042, −0.4611, 0.4819, 1.1404, 1.5712, 2.0019, −0.9827]
RMSE = sqrt(Σe²/8) = 1.443477      ME = |e₁| = 2.347298
```

Reading: the line is pulled by the outlier 100; the first key is predicted at rank 2.35 instead of 0. On the real files the values are large because n = 2·10⁸: history RMSE = 815,542.67 and ME = 2,303,085.18 (0.41 % and 1.15 % of n), fb RMSE = 57,735,023.67 and ME = 99,999,995.54 (28.9 % and 50.0 % of n: one line over fb is useless), stack RMSE = 839,727.03 (0.42 %) [results/aidb/hardness.json <name>.full.rmse, .max_error; percentages from scratch/table2_gains_03.py]. The fitted history line has slope 0.02173 ranks per key unit and intercept 2,300,986 at the origin key 83 [results/aidb/hardness/history_full.json original.least_squares].

Implementation note (ours, not the paper's): `least_squares` [include/scaleli/hardness.hpp:84-106] uses centred (Welford) accumulators on the feature kᵢ − k₁ computed exactly in unsigned __int128 before conversion to long double, then a second pass for the residuals; verified against an exact rational least-squares oracle with worst relative disagreement 2.7e-14 on RMSE and 9.4e-12 on ME over 13 inputs [results/aidb/verification/verdicts.json, metrics verdict; results/aidb/verification/metrics/vlib.py]. RMSE and ME are invariant to shifting the feature or the rank by a constant (the intercept absorbs it), so the paper's R(xᵢ) = i and our r = i − 1 give identical values.

### 3.2 CD, the conflict degree, as LIPP's FMCD defines it

Definition [AIDB §4.1]: "CD, short for conflict degree, is the maximum number of keys mapped to the same rank by a linear model fitted on the dataset using Fastest Minimum Conflict Degree (FMCD) by Wu et al. [29]. It was originally proposed to quantify the model quality of LIPP, and thus has good correlation with LIPP's performance [10]."

What FMCD is (LIPP, Wu et al., PVLDB 14(8), 2021; the bulk-load root model of `LIPP::build_tree_bulk_fmcd` in `src/core/lipp.h`, ported in `fmcd_fit` [include/scaleli/hardness.hpp:168-215] with the provenance block at [hardness.hpp:15-22]). LIPP stores keys *in the slot their model predicts*; a node has L slots for `size` keys, L = size × (BUILD_GAP_CNT + 1) with BUILD_GAP_CNT = 5 for size < 100,000, 2 for size < 1,000,000, 1 otherwise [hardness.hpp:109-113, `lipp_gap_count`]; so a 200M-key root has L = 400,000,000 slots [results/aidb/hardness_details.json history.full_original.fmcd.capacity]. FMCD searches the *smallest D* such that a line with slope 1/U_T puts at most D keys into any slot, where U_T is the key span per slot:

```
FMCD(k[0..size-1]):
  L  = size · (gap + 1)
  D  = 1;  U_T = (k[size−1−D] − k[D]) / (L − 2) + 1e-6        # span per slot if the D-th and (size−1−D)-th keys land at slots 1 and L−2
  i  = 0
  while i < size − 1 − D:
      while i + D < size and k[i+D] − k[i] ≥ U_T:  i += 1        # every window of D+1 consecutive keys spans at least one slot: fine
      if i + D ≥ size: break                                    # all windows checked: D is feasible
      D += 1                                                    # a window of D+1 keys fits inside one slot: need a larger D
      if 3·D > size: break                                      # give up: fallback fit through the two tertile midpoints
      U_T = (k[size−1−D] − k[D]) / (L − 2) + 1e-6
  model:  a = 1 / U_T,  b = L/2 − a · (k[size−1−D] + k[D]) / 2      # i.e. pos(k) = (k − anchor)/U_T + L/2
  PREDICT_POS(k) = clamp( floor(a·k + b), 0, L − 1 )
CD = max over slots s of |{ i : PREDICT_POS(kᵢ) = s }|             # equal slots are contiguous runs on sorted keys
```

[hardness.hpp:175-214 for the search and the two fits, :138-147 for PREDICT_POS, :218-227 for the run count]. Two remarks. First, D is LIPP's own *bound* on the conflict degree; CD is the *measured* maximum after flooring, so the two can differ by one or two: history D = 9, CD = 8; books D = 245, CD = 246; fb D = 114, CD = 110; osm D = 4106, CD = 4107; libio D = 2, CD = 2; stack D = 1, CD = 1 [results/aidb/hardness_details.json <name>.full_original.fmcd.d; results/aidb/hardness.json <name>.full.conflict_degree]. Second, the slot is the floor of a floating-point product, exactly as LIPP does it, so a key within one double ulp of a slot boundary is assigned by rounding; the header documents a ±1 uncertainty (books 2M sample: CD 11 with LIPP's double arithmetic, 10 with exact rational arithmetic) and recommends treating CD differences of one as ties [hardness.hpp:119-127].

Worked example. The first toy gives CD = 1 (the 100 outlier makes U_T wide), so here is a second toy with a dense cluster [scratch/worked_examples_03.out.txt, section A2]:

```
k = [10, 11, 12, 13, 14, 60, 61, 100],  size = 8  ->  gap = 5, L = 48
D = 1:  U_T = (k[6] − k[1])/(48 − 2) + 1e-6 = (61 − 11)/46 = 1.086958
        window k[1] − k[0] = 1 < U_T  -> two keys would share a slot -> D = 2
D = 2:  U_T = (k[5] − k[2])/46 + 1e-6 = (60 − 12)/46 = 1.043479
        all windows k[i+2] − k[i] ≥ 1.043479 (2, 2, 2, 47, 47, 40) -> feasible
model:  a = 1/1.043479 = 0.958332,  anchor = (k[5] + k[2])/2 = 36,  base = L/2 = 24
        pos(k) = 0.958332·(k − 36) + 24
slots:  PREDICT_POS = [0, 0, 1, 1, 2, 46, 47, 47]      (k=10 -> −0.9 clamped to 0; k=11 -> 0.04 -> 0; k=100 -> 85.3 clamped to 47)
CD = 2  (LIPP's bound D = 2)
```

For comparison the OLS line on the same keys gives RMSE = 1.112085 and ME = 1.926016, and PLA-1 needs 2 segments: the three metric families see the same cluster through different lenses (a slot collision count, a squared error, a segment count).

Why CD is "aligned with LIPP" (the phrase you will be asked about): the number CD is *literally* the worst slot occupancy of LIPP's own root model on the whole key set, and LIPP's cost is driven by conflicts: "LIPP constructs its tree structure based on the prediction conflicts and only incurs a performance penalty when accessing conflicting keys. CD directly reflects the amount of conflicts seen by the model at the root node, where the majority of conflicts happen" [AIDB §4.2]. It says nothing about local curvature inside a slot's neighbourhood, which is why it does not predict indexes that correct locally.

Our deviation, disclosed: for real-valued features (the transformed and smoothed scopes only, never the raw keys) the additive 1e-6 on U_T is scaled by the mean gap, because LIPP's absolute constant would dominate U_T on O(1)-range flow outputs; the literal-constant value is emitted alongside as `fmcd.conflict_degree_lipp_epsilon` [hardness.hpp:149-167; src/hardness.cpp:159-160]. Raw-key CD (the `full` scope compared with the paper) uses LIPP's constant verbatim.

### 3.3 PLA-ε: segments of the optimal ε-bounded piecewise linear approximation

Definition [AIDB §4.1, §2.2]: "PLA-ε is a scalar metric. It is the number of segments produced by PLA given a fixed error bound ε." PLA here is the optimal algorithm of O'Rourke [19] as PGM-index uses it: partition the sorted points (kᵢ, rᵢ) into the minimum number of *contiguous* segments such that each segment admits a line g(k) = a·k + b with |rᵢ − g(kᵢ)| ≤ ε for every point of the segment. "Optimal" means minimum number of segments; a greedy scan that extends the current segment as long as a feasible line exists and starts a new one otherwise is optimal for this counting problem (any partition must break no later than the greedy one does, by induction over segment ends), which is why an online algorithm gives the optimum.

How feasibility is maintained in O(1) amortised per point (O'Rourke 1981; PGM's `OptimalPiecewiseLinearModel`, ported as `OptimalPLA` [hardness.hpp:230-286]): a segment is feasible iff some line passes *between* the upper points (kᵢ, rᵢ + ε) and the lower points (kᵢ, rᵢ − ε) of all its points. The algorithm keeps the upper convex hull of the lower points and the lower convex hull of the upper points, plus a "rectangle" of the two extreme feasible slopes; a new point is rejected (`add_point` returns false) when its upper point falls below the steepest feasible line or its lower point above the shallowest one, otherwise the hulls are trimmed and the extreme slopes updated [hardness.hpp:252-284]. Counting follows PGM's `make_segmentation` exactly: every rejection closes a segment, the rejected point starts the next one, a sentinel point (k_last + 1, n) is appended at the end, and the count is rejections + 1 [hardness.hpp:287-311]. The sentinel is why a perfectly linear set gives 1 segment at ε ≥ 1 but 2 at ε = 0 [results/aidb/verification/verdicts.json, C1 self-verification].

Worked example, ε = 0, 1, 2 on the first toy [scratch/worked_examples_03.out.txt, section A], where "segment starts at rank 8" is the sentinel:

```
keys  [1, 2, 3, 4, 10, 20, 30, 100]
PLA-0: 3 segments, starting at ranks [0, 4, 7]:  {1,2,3,4} exact line r = k − 1;  {10,20,30} exact line r = k/10 + 3;  {100}
PLA-1: 2 segments, starting at ranks [0, 5]:     {1,2,3,4,10} fits within ±1;  {20,30,100} fits within ±1
PLA-2: 1 segment:                                the whole set fits within ±2 (the OLS ME was 2.35, but PLA may use any line, not the OLS one)
```

The last line is the point to remember about PLA versus ME: ME is the worst error of the *least-squares* line, PLA asks whether *any* line fits within ε (a Chebyshev criterion), so a set with ME > ε may still be one PLA-ε segment.

Check of the port used for the example: on the repository fixture (16,384 keys) the Python port gives 107 segments at ε = 32 and 1 at ε = 4096, the values the independent PGM-index comparison recorded for `scaleli_hardness` [scratch/worked_examples_03.out.txt, section B; results/aidb/verification/verdicts.json, PLA verdict].

What ε = 32 versus ε = 4096 mean. In a PGM-index built with parameter ε, the true position of a key is guaranteed to lie in [p̂ − ε, p̂ + ε], so the last-mile search scans at most 2ε + 1 positions: 65 positions (520 bytes of 8-byte keys, about eight cache lines) at ε = 32, versus 8,193 positions (64 KiB) at ε = 4096. PLA-32 therefore counts how many *leaf* models a tight index needs ("local hardness"): the more segments, "the more locally non-linear the dataset is" [AIDB §4.2]. PLA-4096 counts the models needed when each is allowed to be coarse ("global hardness"). On the full files [results/aidb/hardness.json; keys per segment from scratch/table2_gains_03.py]:

| dataset | PLA-32 | keys per PLA-32 segment | PLA-4096 | keys per PLA-4096 segment |
|---|---|---|---|---|
| books | 262,604 | 762 | 97 | 2,061,856 |
| fb | 1,055,308 | 190 | 1,687 | 118,554 |
| osm | 661,115 | 303 | 5,495 | 36,397 |
| covid | 81,908 | 2,442 | 850 | 235,294 |
| genome | 1,290,208 | 155 | 1,426 | 140,252 |
| history | 105,468 | 1,896 | 468 | 427,350 |
| libio | 145,808 | 1,372 | 639 | 312,989 |
| planet | 613,597 | 326 | 2,314 | 86,430 |
| stack | 17,833 | 11,215 | 133 | 1,503,759 |
| wise | 79,035 | 2,531 | 382 | 523,560 |

genome is the locally hardest set (one ε = 32 line per 155 keys) while osm is the globally hardest (one ε = 4096 line per 36,397 keys); books is globally easy (97 coarse segments) but locally middling. These are the positions of the points in the paper's Figure 2 [AIDB p. 3; visual agreement only, the paper prints no table of values: unverified beyond history and libio].

Verification of our PLA counts [results/aidb/verification/verdicts.json, PLA verdict]: identical to PGM-index's `make_segmentation` (verbatim header, commit c6fcf3d) for ε ∈ {0, 1, 8, 32, 128, 1024, 4096} on the fixture, six synthetic sets (including one ending at 2⁶⁴ − 1, handled by widening to unsigned __int128 so PGM's sentinel does not wrap [hardness.hpp:287-294]), 5M-key prefixes of all ten GRE files, and the full 200M-key books (262,604 / 97) and wise (79,035 / 382) files.

---

## 4. Metric compositions, the harder-than order, coverage and conformance

### 4.1 The 25 metrics of Table 2

[AIDB §4.1]: the five scalars; "A·B is a two-dimensional metric. It composes two scalar metrics, A and B, chosen from the list above"; "A·B·C composes three scalar metrics". So the corpus is 5 + C(5,2) + C(5,3) = 5 + 10 + 10 = 25 metrics, and GRE is the special name of PLA-32·PLA-4096 [AIDB Table 2]. Enumerated in the paper's order [AIDB Table 2; tools/aidb_scores.py:52-57 generates the same list]:

```
scalar (5):   RMSE, ME, CD, PLA-32, PLA-4096
2-dim  (10):  GRE = PLA-32·PLA-4096, RMSE·ME, RMSE·CD, RMSE·PLA-32, RMSE·PLA-4096,
              ME·CD, ME·PLA-32, ME·PLA-4096, CD·PLA-32, CD·PLA-4096
3-dim  (10):  RMSE·ME·CD, RMSE·ME·PLA-32, RMSE·ME·PLA-4096, RMSE·CD·PLA-32, RMSE·CD·PLA-4096,
              RMSE·PLA-32·PLA-4096, ME·CD·PLA-32, ME·CD·PLA-4096, ME·PLA-32·PLA-4096, CD·PLA-32·PLA-4096
```

Composition does not combine the numbers (no weighted sum): a composed metric is a *vector*, and vectors are compared by the partial order below. Four- and five-dimensional compositions are not evaluated; the paper gives no reason [not reported].

### 4.2 The harder-than partial order, C and U

Notation [AIDB §3.2]: dataset corpus 𝒮 = {S₁, …, Sₙ} (n = 10), index corpus ℐ = {I₁, …, I_m} (m = 6), a d-dimensional metric h with components h_k. Definition (exact): **Sᵢ is harder than Sⱼ when h_k(Sᵢ) ≥ h_k(Sⱼ) for all k and h_k(Sᵢ) > h_k(Sⱼ) for at least one k.** Two datasets are *comparable* when one is harder than the other. C = {(i, j) | Sᵢ is harder than Sⱼ} is the set of comparable pairs *as ordered pairs* (harder first); U = {(i, j) | i < j ∧ Sᵢ, Sⱼ incomparable} is the set of incomparable *unordered* pairs. Every unordered pair of distinct datasets is in exactly one of C (once, with its orientation) or U, so |C| + |U| = C(n, 2) = 45 for ten datasets. Two consequences that are easy to miss: (i) a pair equal on *every* dimension is incomparable even for a scalar metric (neither is strictly harder), so a scalar metric does not automatically have coverage 1: on our 2M uniform samples covid, libio, stack and wise all have CD = 8, which makes the six pairs among them incomparable and gives scalar CD a coverage of (39 − 6)/45 = 0.733 [results/aidb/scores.json sample.metrics.CD]; (ii) a 2-D pair is incomparable exactly when the two dimensions disagree (or one ties and the other does not decide the other way), which is why adding a dimension "can only make some previously comparable pairs incomparable, but not the opposite" [AIDB §4.2, before Finding 7].

Implementation: `harder` and `classify_pairs` [tools/aidb_scores.py:82-97] are a literal transcription (`all(>=) and any(>)`).

### 4.3 Coverage (equation 3) with a worked example

```
Cov = (|C| − |U|) / (|C| + |U|) = (|C| − |U|) / N,   N = C(n, 2)        [AIDB eq. (3)]
```

Range [−1, 1]: 1 when every pair is comparable, −1 when none is; "we introduce the −|U| term for consistency with the range of the conformance score" [AIDB §3.2]. Coverage is independent of any index [AIDB §3.2].

Worked example on four datasets with a 2-D metric (x, y) [scratch/worked_examples_03.out.txt, section C]:

```
A = (1, 1),  B = (2, 3),  C = (3, 2),  D = (3, 4)
dim x alone:  B>A, C>A, D>A, C>B, D>B;  C vs D tie on x -> incomparable      |C| = 5, |U| = 1, Cov = (5−1)/6 = 0.6667
dim y alone:  B>A, C>A, D>A, B>C, D>B, D>C                                    |C| = 6, |U| = 0, Cov = 1.0000
(x, y):       B>A, C>A, D>A, D>B, D>C;  B vs C: x says C harder, y says B harder -> incomparable
                                                                              |C| = 5, |U| = 1, Cov = 0.6667
```

On the real full-file table, GRE gives |C| = 33, |U| = 12, Cov = 21/45 = 0.4667, printed 0.47 in Table 2 [scratch/worked_examples_03.out.txt section C; AIDB Table 2]. Because n = 10 fixes the denominator at 45, every coverage value in Table 2 decodes to a unique (|C|, |U|) [scratch/worked_examples_03.out.txt, section E]: 1.00 = (45, 0); 0.82 = (41, 4); 0.60 = (36, 9); 0.56 = (35, 10); 0.51 = (34, 11); 0.47 = (33, 12); 0.42 = (32, 13); 0.38 = (31, 14); 0.33 = (30, 15); 0.24 = (28, 17); 0.20 = (27, 18); 0.16 = (26, 19). This decoding is what lets section 7 say "our CD compositions differ from the paper's by exactly one pair".

### 4.4 Conformance (equations 1-2) with a worked example

The strawman [AIDB §3.1]: Conf = #conforming pairs / #total pairs, where (S₁, S₂) conforms when h(S₁) < h(S₂) and the index performs better on S₁. Two defects: it mixes incomparable pairs into the denominator (coverage leaks into conformance), and it weighs all pairs equally although "for conforming pairs, the ones with larger performance gaps are generally easier to order correctly … for violating pairs, those with larger performance gaps should weigh more heavily" [AIDB §3.1]; Figure 3 shows LIPP's throughput gaps over the comparable pairs spanning about −2 to +10 MOPS [AIDB Fig. 3; range read from the axis: unverified].

The proposed score [AIDB §3.2], per index I:

```
p_I(S)        throughput of index I on dataset S
σ_I           standard deviation of p_I over the dataset corpus 𝒮
p̂_I(S)        = p_I(S) / σ_I                                   (unit standard deviation; makes gaps comparable across indexes)
for each ordered comparable pair (i, j) ∈ C  (Sᵢ harder than Sⱼ):
    δp_ij     = p̂_I(Sᵢ) − p̂_I(Sⱼ)                                (signed gap: harder minus easier)
    w_ij      = sigmoid(δp_ij) = 1 / (1 + e^(−δp_ij))  ∈ (0, 1)   (importance)
    conforming if δp_ij ≤ 0  (harder is not faster)  -> C⁺_I ;  violating if δp_ij > 0  -> C⁻_I
R_I = Σ_{(i,j) ∈ C⁺_I} w_ij        (reward)
P_I = Σ_{(i,j) ∈ C⁻_I} w_ij        (penalty)
Conf_I = (R_I − P_I) / (R_I + P_I)                              [AIDB eq. (1)]
Conf   = mean_{I ∈ ℐ} Conf_I                                    [AIDB eq. (2)]
```

Why the sigmoid does what §3.1 asked for: for a conforming pair δ ≤ 0 and w = sigmoid(δ) ≤ 0.5, tending to 0 as the gap grows (a clearly separated pair "is easy to order correctly and should contribute little"); for a violating pair δ > 0 and w > 0.5, tending to 1 (a large reversed gap is "a more egregious mistake"). A pair with zero gap contributes 0.5 to R. Conf_I = 1 iff P_I = 0 (no violating pair), −1 iff R_I = 0, and the normalised difference keeps every index on the same [−1, 1] scale [AIDB §3.2]. The paper does not say whether σ_I is the population or the sample standard deviation [not reported]; our scorer defaults to the population value with `--ddof 1` available [tools/aidb_scores.py:19-23, 67-79]. Equation (1) is undefined when σ_I = 0 (a variant with identical throughput everywhere) or when R_I + P_I = 0 (no comparable pair); our scorer skips such variants rather than counting them as conforming [tools/aidb_scores.py:23-26, 181-185].

Worked example, same four datasets and metric (x, y) as in 4.3 (C = {B>A, C>A, D>A, D>B, D>C}), two fictitious indexes [scratch/worked_examples_03.out.txt, section D]:

```
index I1: p = {A: 10, B: 8, C: 9, D: 5}    σ = 1.870829 (population)   p̂ = {A: 5.345, B: 4.276, C: 4.811, D: 2.673}
   (B>A) δ = −1.069  w = 0.2556 conforming     (C>A) δ = −0.535  w = 0.3695 conforming
   (D>A) δ = −2.673  w = 0.0646 conforming     (D>B) δ = −1.604  w = 0.1675 conforming
   (D>C) δ = −2.138  w = 0.1054 conforming
   R = 0.962589, P = 0,  Conf_I1 = 1.000000
index I2: p = {A: 6, B: 7, C: 5, D: 4}     σ = 1.118034                 p̂ = {A: 5.367, B: 6.261, C: 4.472, D: 3.578}
   (B>A) δ = +0.894  w = 0.7098 VIOLATING      (C>A) δ = −0.894  w = 0.2902 conforming
   (D>A) δ = −1.789  w = 0.1432 conforming     (D>B) δ = −2.683  w = 0.0640 conforming
   (D>C) δ = −0.894  w = 0.2902 conforming
   R = 0.787574, P = 0.709803,  Conf_I2 = (0.787574 − 0.709803)/(0.787574 + 0.709803) = 0.051938
Conf = mean(1.000000, 0.051938) = 0.525969;   Cov(x·y) = 0.6667
```

Note how one violating pair with a moderate gap (w = 0.71) almost cancels four conforming pairs (their weights sum to only 0.79 because three of them are "easy"): this is the emphasis on challenging pairs at work, and it is why Table 2 contains many values near 0 and several negative ones.

A real instance from our study (control variant `packed_rank`, scalar CD, full-file hardness) [scratch/worked_examples_03.out.txt, section D; results/aidb/throughput.json packed_rank; results/aidb/scores.json full.metrics.CD.per_variant_detail.packed_rank]:

```
throughput (median of 3 seeds, MOPS): books 2.144, covid 2.217, fb 2.638, genome 2.476, history 2.816,
                                      libio 2.824, osm 1.932, planet 2.789, stack 2.942, wise 2.482
σ = 0.320381 MOPS;  p̂ = books 6.692, covid 6.921, fb 8.234, genome 7.728, history 8.789, libio 8.816,
                         osm 6.029, planet 8.705, stack 9.182, wise 7.747
scalar CD orders all 45 pairs (no ties): 40 conforming, 5 violating
violating: genome (CD 585) > books (246) but faster, δ = +1.036, w = 0.738
           fb (110) > covid (27), δ = +1.313, w = 0.788;   genome (585) > covid (27), δ = +0.807, w = 0.691
           fb (110) > wise (10), δ = +0.487, w = 0.619;    planet (21) > wise (10), δ = +0.958, w = 0.723
easiest conforming pairs: osm > stack (δ = −3.15, w = 0.041), osm > libio (w = 0.058), osm > history (w = 0.060)
R = 9.934451,  P = 3.559665,  Conf_packed_rank(CD) = 0.472412     (scores.json: 0.4724122609070333)
```

The paper's own anecdote in these terms: under GRE, (libio, history) ∈ C with libio harder; LIPP's gap is +1.5 MOPS/σ_LIPP; σ_LIPP is not printed, so the weight is between 0.68 (if σ were 2.0) and 0.95 (if σ were 0.5) [illustrative values from scratch/worked_examples_03.out.txt; σ_LIPP not reported].

---

## 5. Setup (§4.1): indexes, datasets, hardware, workload

### 5.1 The six learned indexes

[AIDB §4.1, one bullet each; family and correction column from §2.1 and §4.2]

| index | reference | one-line description (paper's words, abridged) | tree construction family [§2.1, §4.2] | local correction after the prediction |
|---|---|---|---|---|
| RMI | Kraska et al. 2018 [9] | "the initial learned index proposal that uses a recursive model architecture" | fixed architecture; "a single root linear model to dispatch keys to lower-level models, without considering the cost of mispredictions" [§4.2] | yes (verification search in the error window, §2.1) |
| PGM-index | Ferragina & Vinciguerra 2020 [2] | "a recursively PLA-constructed, optimal ε-bounded piecewise linear model to learn the rank function" | error-driven greedy splits, balanced | yes, bounded by ε |
| ALEX | Ding et al. 2020 [1] | "extends RMI with model-based inserts into gapped arrays and supports adaptive node expansion and splitting" | cost-based splits, unbalanced/adaptive | yes (exponential search in ALEX's paper [unverified in AIDB]) |
| LIPP | Wu et al. 2021 [29] | "a model-guided tree index that ensures precise key-to-position mappings without in-node search and achieves O(log N) lookup with amortized O(log² N) insert complexity" | conflict-based splits, unbalanced/adaptive | none |
| XIndex | Tang et al. 2020 [24] | "range-partitioned groups indexed by linear models and a delta index, combined with fine-grained synchronization and compaction, to support dynamic workloads" | even splits with error checking; "simply reuses RMI to index its range-partitioned groups" [§4.2] | yes (RMI-style) |
| FINEdex | Li et al. 2021 [11] | "organizes independently trained linear models as per-key two-level bins and supports in-place inserts and fine-grained concurrent retraining without a shared delta buffer" | (not classified in §2.1); grouped with PGM and ALEX as "cost-based partitioning … depend on local correction" [§4.2, Finding 3] | yes |

The grouping that matters for the findings is: {PGM, ALEX, FINEdex} correct locally and are predicted by PLA-32; LIPP has no correction and is predicted by CD; RMI and XIndex are predicted by no scalar metric (section 6, Findings 3-4).

### 5.2 The ten datasets (Table 1) and what they are like

Each dataset "contains 200 million 64-bit unsigned integer keys without duplicates"; together they are "a superset of the datasets used in recent work on learned indexes" [AIDB §4.1]. Table 1 [AIDB p. 4] with our observed characteristics appended [results/aidb/hardness_details.json <name>.full.min/.max; results/aidb/provenance.json sort_audit; results/aidb/hardness.json]:

| dataset | description (Table 1) | source (Table 1) | key range observed (min .. max) | stored sorted on the GRE mirror? | RMSE / n | CD | PLA-32 | PLA-4096 |
|---|---|---|---|---|---|---|---|---|
| books | Amazon book sales popularity | SOSD [7] | 0 .. 9,223,372,036,854,784,874 (just above 2⁶³) | yes | 9.03 % | 246 | 262,604 | 97 |
| fb | Upsampled Facebook user ID | SOSD [7] | 1 .. 18,446,744,073,709,551,615 (= 2⁶⁴ − 1) | yes | 28.87 % | 110 | 1,055,308 | 1,687 |
| osm | Uniformly sampled OpenStreetMap locations | SOSD [7] | 33,246,697,004,540,789 .. 13,748,550,930,623,082,253 | yes | 12.09 % | 4,107 | 661,115 | 5,495 |
| covid | Uniformly sampled Tweet ID with tag COVID-19 | Lopez & Gallemore 2021 [16] | 1,344,795,470,900,715,522 .. 1,446,626,459,003,486,210 | no | 0.90 % | 27 | 81,908 | 850 |
| genome | Loci pairs in human chromosomes | Rao et al. 2014 [21] | 2,489,750 .. 446,640,429,470 | no | 3.77 % | 585 | 1,290,208 | 1,426 |
| history | History node ID in OpenStreetMap | Google Cloud OSM [5] | 83 .. 9,178,997,263 | no | 0.41 % | 8 | 105,468 | 468 |
| libio | Repository ID from libraries.io | Libraries.io [14] | 21,335 .. 527,206,042 | no | 1.72 % | 2 | 145,808 | 639 |
| planet | Planet ID in OpenStreetMap | Google Cloud OSM [5] | 1 .. 9,178,997,250 | no | 14.89 % | 21 | 613,597 | 2,314 |
| stack | Vote ID from StackOverflow | StackExchange archive [22] | 1 .. 238,071,030 | no | 0.42 % | 1 | 17,833 | 133 |
| wise | Partition key from the data returned by the Wide-field Infrared Survey Explorer (WISE) | Wright et al. 2010 [28] | 8,796,093,034,805 .. 17,592,186,032,162 | no | 1.10 % | 10 | 79,035 | 382 |

Reading the shapes from the numbers (our characterisation, not the paper's): stack, history and covid are close to one global line (RMSE below 1 % of n) and have tiny conflict degrees: they are the "easy" end. fb spans the entire 64-bit range with RMSE 29 % of n and one ε = 32 segment per 190 keys: globally and locally hard, but its CD is only 110 because its density is spread rather than clustered. osm has by far the largest CD (4,107): heavy local clustering, which is exactly what LIPP suffers from. genome has the most PLA-32 segments (1,290,208) with a moderate global fit: locally jagged. books is globally the easiest of the SOSD three (97 coarse segments) and has a mid-range CD of 246. history and planet share the same ID space (both end at 9,178,997,2xx) but planet is far less linear (RMSE 14.9 % versus 0.4 %). Any statement in the paper about a dataset beyond Table 1's one-liners: not reported.

Provenance of our copies [results/aidb/provenance.json]: all ten files come from the GRE benchmark mirror `https://www.cse.cuhk.edu.hk/mlsys/gre/<name>` (from `github.com/gre4index/GRE/datasets/download.sh`), 1,600,000,008 bytes each (an 8-byte count header plus 200,000,000 × 8 bytes, SOSD format), sha256 recorded per file, retrieved 2026-09-21; the sort audit found seven files stored unsorted and no duplicates in any file; the sorted copies `<name>.sorted` are what every later step reads [src/hardness.cpp:122-135; results/aidb/provenance.json <name>.sort_audit].

### 5.3 Hardware and software

[AIDB §4.1, "Testbed and Configuration"]: a server with two Intel Xeon Gold 5118 CPUs (12 cores per socket, 2.3 GHz) and 384 GiB of RAM; a container running Ubuntu 20.04.6 on host Linux kernel 6.8.0-57-generic; hugepages disabled; "All indexes are ported in a common codebase and compiled with GCC 9.4.0 using the -O3 flag"; default configuration for each index "when applicable", with bug fixes "applied when necessary"; "The worker thread is pinned to a fixed CPU core for stability". Not reported: which core, whether the other socket was idle, memory allocation policy, index-specific parameters, the version or commit of each index.

### 5.4 Workload and how throughput is measured

[AIDB §4.1, "Workloads and Measurements"]: "We use index lookup throughput as the primary performance metric, as write operations introduce performance overhead that is not directly related to dataset hardness. For each index and dataset, we first bulk load the index and then perform random lookups for all keys in the dataset. For each run, we start with a warm-up phase of 20 million lookups and then a measurement phase of 100 million lookups, from which we calculate the throughput." Combined with §2.2 ("uniform read-only workload over the same dataset used to populate the index"):

- bulk load of all 200M keys, then lookups only (no inserts, no range scans);
- lookup keys drawn from the dataset's own keys (every lookup hits; "random lookups for all keys" reads as uniform random over the key set, sampling with or without replacement not stated [not reported]);
- 20M warm-up lookups discarded, 100M measured lookups, throughput = 10⁸ / measured time (in MOPS);
- one worker thread pinned to a core (the singular "worker thread" implies single-threaded measurement [inference]);
- number of repetitions per (index, dataset), variance, and whether the reported value is a mean or a median: not reported.

The only absolute throughput numbers printed are the LIPP anecdote (7.1 MOPS on libio, 5.6 on history) [AIDB §2.2]; everything else enters the paper only through p̂_I(S) = p_I(S)/σ_I in the conformance score.

---

## 6. Results (§4.2): Table 2, Findings 1-7, and the open problems of §5

### 6.1 Table 2, reproduced

Columns: Conf_I per index, then Conf (their mean) and Cov [AIDB Table 2, p. 5; transcription checked against the rendered page]. The printed Conf equals the mean of the six Conf_I to within 0.005 on every row (rounding of the printed values) [scratch/table2_gains_03.py].

| metric | RMI | PGM | ALEX | LIPP | XIndex | FINEdex | Conf | Cov |
|---|---|---|---|---|---|---|---|---|
| RMSE | 0.34 | 0.19 | 0.41 | 0.09 | −0.14 | 0.26 | 0.19 | 1.00 |
| ME | 0.32 | 0.04 | 0.39 | 0.22 | −0.15 | 0.11 | 0.15 | 1.00 |
| CD | 0.18 | 0.36 | 0.10 | 0.84 | 0.09 | 0.11 | 0.28 | 1.00 |
| PLA-32 | 0.41 | 0.64 | 0.71 | 0.29 | 0.02 | 0.91 | 0.50 | 1.00 |
| PLA-4096 | 0.38 | −0.03 | 0.17 | 0.03 | 0.03 | −0.10 | 0.08 | 1.00 |
| GRE (PLA-32·PLA-4096) | 1.00 | 0.71 | 1.00 | 0.42 | 0.27 | 0.86 | 0.71 | 0.47 |
| RMSE·ME | 0.44 | 0.18 | 0.53 | 0.20 | −0.09 | 0.24 | 0.25 | 0.82 |
| RMSE·CD | 0.68 | 0.69 | 0.69 | 0.89 | 0.14 | 0.53 | 0.60 | 0.47 |
| RMSE·PLA-32 | 0.69 | 0.73 | 1.00 | 0.34 | 0.05 | 1.00 | 0.64 | 0.60 |
| RMSE·PLA-4096 | 1.00 | 0.40 | 0.84 | 0.32 | 0.16 | 0.39 | 0.52 | 0.42 |
| ME·CD | 0.57 | 0.45 | 0.57 | 0.90 | 0.09 | 0.34 | 0.49 | 0.56 |
| ME·PLA-32 | 0.70 | 0.62 | 1.00 | 0.43 | 0.04 | 0.88 | 0.61 | 0.60 |
| ME·PLA-4096 | 1.00 | 0.30 | 0.85 | 0.42 | 0.14 | 0.29 | 0.50 | 0.42 |
| CD·PLA-32 | 0.58 | 0.87 | 0.73 | 0.89 | 0.21 | 0.87 | 0.69 | 0.60 |
| CD·PLA-4096 | 0.83 | 0.55 | 0.53 | 0.88 | 0.28 | 0.25 | 0.55 | 0.42 |
| RMSE·ME·CD | 0.67 | 0.68 | 0.68 | 0.88 | 0.13 | 0.51 | 0.59 | 0.42 |
| RMSE·ME·PLA-32 | 0.67 | 0.71 | 1.00 | 0.39 | 0.01 | 1.00 | 0.63 | 0.51 |
| RMSE·ME·PLA-4096 | 1.00 | 0.35 | 0.83 | 0.37 | 0.11 | 0.33 | 0.50 | 0.33 |
| RMSE·CD·PLA-32 | 0.79 | 1.00 | 1.00 | 1.00 | 0.17 | 1.00 | 0.83 | 0.33 |
| RMSE·CD·PLA-4096 | 1.00 | 0.76 | 0.77 | 1.00 | 0.14 | 0.53 | 0.70 | 0.16 |
| RMSE·PLA-32·PLA-4096 | 1.00 | 0.80 | 1.00 | 0.42 | 0.15 | 1.00 | 0.73 | 0.24 |
| ME·CD·PLA-32 | 0.81 | 0.83 | 1.00 | 1.00 | 0.18 | 0.84 | 0.78 | 0.38 |
| ME·CD·PLA-4096 | 1.00 | 0.60 | 0.80 | 1.00 | 0.16 | 0.41 | 0.66 | 0.20 |
| ME·PLA-32·PLA-4096 | 1.00 | 0.63 | 1.00 | 0.55 | 0.13 | 0.82 | 0.69 | 0.24 |
| CD·PLA-32·PLA-4096 | 1.00 | 0.80 | 1.00 | 0.85 | 0.20 | 0.81 | 0.78 | 0.24 |

Best metric per index [from the table; scratch/table2_gains_03.py]: RMI: best scalar PLA-32 (0.41), 1.00 with RMSE·PLA-4096 and several 3-D metrics; PGM: PLA-32 (0.64), 1.00 with RMSE·CD·PLA-32; ALEX: PLA-32 (0.71), 1.00 with RMSE·PLA-32; LIPP: CD (0.84), 1.00 with RMSE·CD·PLA-32, RMSE·CD·PLA-4096, ME·CD·PLA-32, ME·CD·PLA-4096; XIndex: CD (0.09), best overall CD·PLA-4096 at only 0.28; FINEdex: PLA-32 (0.91), 1.00 with RMSE·PLA-32.

### 6.2 The conformance-coverage plane and the approximate Pareto frontier (Figure 4)

Figure 4 [AIDB p. 5] plots (Conf, Cov) for the 25 metrics: the five scalars sit on the Cov = 1 line at Conf 0.08-0.50, the 2-D metrics at Cov 0.42-0.82 and Conf 0.25-0.71, the 3-D metrics at Cov 0.16-0.51 and Conf 0.50-0.83 [values from Table 2]. The "approximate Pareto frontier" is computed by a greedy, beam-search-like procedure [AIDB §4.2]: a scalar metric is nothing but an ordering of the ten datasets, so all 10! = 3,628,800 orderings are enumerated and scored; the top-1k by conformance become bases to which a second dimension (again any of the 3,628,800 orderings) is added; the 2-D Pareto-optimal metrics become bases for a third dimension, and so on up to six dimensions, "as it is guaranteed to achieve a conformance of one with six dimensions, each corresponding to the ordering given by the performance of one index" (six dimensions = six indexes: each dimension copies one index's throughput order, so no pair can violate). The frontier is the Pareto front of everything explored. It is an upper bound achievable by *any* ordering, including orderings no computable feature of the data could ever produce [AIDB §5].

### 6.3 Findings 1-7 with their evidence

**Finding 1** [AIDB §4.2]: "There is a clear tradeoff between conformance and coverage in hardness metric design. Metric conformance generally increases with dimensionality, while coverage decreases." Evidence: the ranges above; the mechanism is that "a high-conformance metric can be achieved by tailoring a multi-dimensional metric where each dimension targets a specific learned index design", but "with increased dimensionality, it becomes less likely two datasets have a harder-than relationship", whereas "a scalar metric achieves full coverage as it always totally orders all datasets" [AIDB §4.2] (with the tie caveat of section 4.2 above).

**Finding 2**: "None of the evaluated metrics sits close to the approximate conformance-coverage Pareto frontier. Among the closest metrics, PLA-32 and CD are the common hardness dimensions and thus can serve as valuable starting points for designing better metrics." Evidence: closest scalar PLA-32; closest 2-D CD·PLA-32 and GRE; closest 3-D RMSE·CD·PLA-32 and ME·CD·PLA-32; "PLA-32 appears in all of them as a hardness dimension, and CD appears in three" [AIDB §4.2]. How far "not close" is: not quantified in the text [not reported; Figure 4 only].

**Finding 3**: "Different learned indexes require different hardness indicators that reflect their design principles. CD is a good fit for indexes with conflict-based designs, while PLA-32 suits indexes relying on local error correction." Evidence [AIDB Table 2]: CD on LIPP 0.84 (versus 0.09-0.36 on the other five); PLA-32 on PGM 0.64, ALEX 0.71, FINEdex 0.91 (versus 0.29 on LIPP). Mechanism: LIPP "only incurs a performance penalty when accessing conflicting keys", and CD is the root model's conflict count; PGM, FINEdex and ALEX "use cost-based partitioning of the key range and depend on local correction to handle model mispredictions. As a result, their performance is more influenced by the local linearity of the dataset, which is well captured by PLA-32" [AIDB §4.2].

**Finding 4**: "Metrics that do not align with the design principles of any learned index do not achieve good conformance. None of the straightforward scalar metrics fits well with RMI and XIndex." Evidence: RMSE, ME and PLA-4096 never exceed 0.41 on any index (their best cells are 0.41, 0.39 and 0.38, all on RMI or ALEX); the best scalar for RMI is 0.41 and for XIndex 0.09 [AIDB Table 2]. Mechanism: RMSE, ME and PLA-4096 "primarily capture the non-linearity of the dataset as a whole, which corresponds to neither the conflict-based design of LIPP nor the local correction-based design"; RMI "uses a fixed model architecture and relies on a single root linear model to dispatch keys to lower-level models, without considering the cost of mispredictions"; XIndex "simply reuses RMI to index its range-partitioned groups" [AIDB §4.2].

**Finding 5**: "Introducing a new dimension to a hardness metric has a diminishing impact on both conformance improvement and coverage degradation as dimensionality increases. In some cases, going from two dimensions to three dimensions even reduces conformance." Evidence: Figure 5 [AIDB p. 6] gives the mean, min and max change in Conf (gain) and Cov (drop) when each scalar is added to a base that does not contain it. Recomputed from the printed Table 2 [scratch/table2_gains_03.py]:

| added dimension | onto scalar bases: mean Conf gain [min, max] | mean Cov drop [min, max] | onto 2-D bases: mean Conf gain [min, max] | mean Cov drop [min, max] |
|---|---|---|---|---|
| +RMSE | +0.250 [+0.10, +0.44] | 0.423 [0.18, 0.58] | +0.072 [0.00, +0.15] | 0.180 [0.09, 0.27] |
| +ME | +0.200 [+0.06, +0.42] | 0.400 [0.18, 0.58] | +0.023 [−0.02, +0.11] | 0.150 [0.05, 0.23] |
| +CD | +0.352 [+0.19, +0.47] | 0.488 [0.40, 0.58] | +0.185 [+0.07, +0.34] | 0.267 [0.22, 0.40] |
| +PLA-32 | +0.487 [+0.41, +0.63] | 0.432 [0.40, 0.53] | +0.255 [+0.19, +0.38] | 0.195 [0.14, 0.31] |
| +PLA-4096 | +0.290 [+0.21, +0.35] | 0.568 [0.53, 0.58] | +0.130 [+0.08, +0.25] | 0.373 [0.31, 0.49] |

Both the gain and the drop are smaller in the second step than in the first for every dimension, and +ME onto a 2-D base has a negative minimum (GRE 0.71 → ME·PLA-32·PLA-4096 0.69) [AIDB Table 2]. "These results suggest that two-dimensional metrics generally strike a good balance between conformance and coverage" [AIDB §4.2].

**Finding 6**: "Scalar metrics that align with the design principles of learned indexes are the most effective dimensions to add for conformance improvement. Combining misaligned scalar metrics, however, does not lead to significant conformance improvement." Evidence: "CD and PLA-32 … achieve at least 2.4× and up to 11.0× conformance improvement compared to other scalar metrics as the added dimension. Meanwhile, other metrics, such as ME, can bring an average increase as low as 0.02" [AIDB §4.2]; in the recomputed table, +ME onto 2-D bases averages +0.023 and +PLA-32 +0.255, a ratio of 11.1, while +CD/+RMSE = 0.185/0.072 = 2.6 (the paper's 2.4 presumably from unrounded values [unverified]). For XIndex, "since none of the scalar metrics fits well with it … combining them does not lead to significant conformance improvement, even for the most promising combination of CD and PLA-32" (XIndex: 0.21 on CD·PLA-32, at most 0.28 on any metric) [AIDB §4.2, Table 2].

**Finding 7**: "Coverage degradation does not pose a major concern in choosing hardness dimensions, as it does not necessarily correlate with the conformance improvement brought by the dimension." Evidence: "no dimension degrades coverage notably more than the others, as the drop is similar in magnitude across all of them" (the mean drops above range 0.40-0.57 onto scalars and 0.15-0.37 onto 2-D bases), and "coverage degradation is always positive, as expected, since adding a new dimension can only make some previously comparable pairs incomparable, but not the opposite" [AIDB §4.2].

### 6.4 Open problems (§5)

1. *Exact and tight Pareto frontier.* The greedy frontier is an approximation in two senses: being close to it "does not necessarily mean a metric is close to a true optimal metric", and even the exact frontier is "the upper bound achievable by any ordering of the datasets, including orderings that no computable feature of the data could ever reproduce". A "tight frontier that is realizable by some computable metric" is undefined so far [AIDB §5].
2. *Workload adaptation.* The tradeoff is inherent only for a workload-agnostic metric; since indexes favour different indicators (Findings 3-4), a metric specialised to the user's target indexes "pushes the frontier outward, rather than merely selecting a point on it". Proposal: a workload-aware metric selection framework taking target indexes as input and weighting the conformance objective toward them [AIDB §5].
3. *Filling gaps in the hardness space.* Under GRE and CD·PLA-32 (Figures 2 and 6) the ten datasets cluster, leaving regions with no dataset. Wongkham et al.'s generator for GRE targets was used to generate ten datasets matching the ten real GRE values; Table 3 [AIDB p. 7] reports that they neither match the target local hardness (differences from 0.00 % on stack to −12.23 % on genome; global hardness matched exactly) nor reproduce index behaviour: ALEX throughput differs by −34.23 % (books) to +63.84 % (fb), LIPP by −59.30 % (stack) to +64.32 % (osm). The generated sets were excluded from the main evaluation to avoid bias [AIDB §5, Table 3, footnote 1]. Conclusion: "dataset generation algorithms that reliably realize target hardness values for a wide range of metrics are needed" [AIDB §5].

---

## 7. How our study uses this paper

Everything in this section is a clean-room CONTROL study inside the SCALE-LI experimental map. We did not run RMI, PGM-index, ALEX, LIPP, XIndex or FINEdex, we did not run the paper's code (none is published with the paper), and nothing below is a reproduction of Table 2 or of NFL, AFLI or CSV. What we reuse from the paper is (a) its ten datasets, (b) the definitions of its five scalar metrics and (c) the definitions of its two scores; what we apply them to is our own index and our own variants.

### 7.1 The ten datasets, at two scales

- *Full files* (200M keys) are used for the hardness metrics only [results/aidb/run.json scale_note]. Seven were sorted once with `scaleli_hardness --sort-only 1 --write-sorted` (parallel chunk sort and merge, identical to `std::sort` [hardness.hpp:50-76]); no duplicates anywhere [results/aidb/provenance.json].
- *2M-key samples* are what the index runs load, in two modes [tools/datasets.py:73-83; run.json settings]: `uniform` = 2,000,000 sorted random indices drawn by `random.Random(42).sample(range(200_000_000), 2_000_000)` (keeps the global shape, flattens local structure); `window` = one contiguous run of 2,000,000 keys starting at a seeded random offset (keeps local structure, loses the global shape). Every sample carries a manifest with the source sha256 and its own sha256 [data/samples/<name>_2M_{uniform,window}_s42.manifest.json]. The two modes disagree on local hardness: fb's uniform sample has PLA-32 = 3,034, its window 10,599 [results/aidb/hardness.json fb.sample.pla_32; results/aidb_window/hardness.json fb.sample.pla_32].

### 7.2 The hardness tool: the five metrics reimplemented

`scaleli_hardness` [src/hardness.cpp] reads a SOSD file, checks sortedness and duplicates, and calls `compute_metrics` [hardness.hpp:323-333], which runs the four scans in parallel: least squares (RMSE, ME), FMCD fit plus CD, and one PLA pass per ε in `--pla-eps 32,4096`. The exact command per dataset, as recorded next to each output [results/aidb/hardness/history_full.json.command.json]:

```
scaleli_hardness --data data/external/gre/history.sorted --flow results/aidb/flows/history_2D2H2L.txt \
                 --dtype uint64 --region-keys 4096 --pla-eps 32,4096 --check-sorted 1
scaleli_hardness --data data/samples/history_2M_uniform_s42 --virtual-alpha 0.1 --dtype uint64 \
                 --region-keys 4096 --pla-eps 32,4096 --check-sorted 1          # and a third run: sample + --flow + --virtual-alpha 0.1
```

[tools/aidb_pipeline.py:449-455 `hardness_jobs`]. This yields six *scopes* per dataset in `results/aidb/hardness.json`: `full` (raw 200M keys: the block compared with the paper), `full_flow` (the same keys after our NFL-style transform, sorted), `sample`, `sample_flow`, `sample_csv` (the sample after CSV-style virtual points at α = 0.1 per 4,096-key region; the metrics are computed on the augmented sequence of real plus virtual features [hardness.hpp:335-377]), `sample_flow_csv` (both). Sections 04-06 use the non-`full` scopes to show which hardness axis each component moves; this section only needs `full`.

Cost on this machine: 12.9-27.4 s per full file (16 threads, including 0.6-0.8 s of reading and, for the `--flow` runs, 0.65 s of transform) [results/aidb/hardness_details.json <name>.full.elapsed_ns; results/aidb/hardness_timing/fb_full.json]; the paper's 10-20 s per dataset is in the same range [AIDB §4.1].

### 7.3 The scorer: equations 1-3 reimplemented and applied to our variants

`tools/aidb_scores.py` implements the protocol literally: `harder` [:82], `classify_pairs` [:87], `coverage` [:100], `conformance_of_variant` [:105] (sigmoid weights, R, P, (R − P)/(R + P)), `score_metric` [:123] (per-variant Conf and their mean), `score_scope` [:153] (population std normalisation, corpus = the datasets present in the hardness scope and in every variant's throughput). It was checked against an independent re-implementation written from the paper text on random and adversarial inputs, with agreement within the verifier's 1e-9 tolerance on every score [results/aidb/verification/verdicts.json, scores verdict; results/aidb/verification/scores/ref_scores.py, verify_scores.py]; MEETING_NOTES quotes 4e-16 [unverified: the archived verdict text I read states the 1e-9 tolerance, not the observed maximum].

The "index corpus" ℐ in our application is not the paper's six indexes but our ten control variants, all instances of the same SCALE-LI map with different options [tools/aidb_pipeline.py:52-66; results/aidb/scores.json full.variants]: `sorted_vector` (binary search, no index), `raw_rank`, `packed_rank` (the control), `packed_rank_flow`, `packed_rank_flow_forced`, `packed_rank_vp10`, `packed_rank_flow_vp10`, `packed_byte`, `packed_rank_fusion_auto`, `packed_rank_flow_costsel`. Their throughput p_I(S) is the median over three query seeds (11, 29, 47) of 1,000,000 uniform lookups (all hits) after 200,000 warm-up lookups on the 2M-key sample, single-threaded, from `throughput_ops_s` of `scaleli_bench` [tools/aidb_pipeline.py:159-171 `median_throughput`, :497-504 `sweep_config`; src/benchmark.cpp:140; results/aidb/throughput.json]. Scale reminder: the paper measures 100M lookups over 200M keys; we measure 1M lookups over 2M keys [results/aidb/run.json scale_note].

Consequence for reading our scores: the *coverage* column depends only on the hardness table, so for the `full` scope it is directly comparable with Table 2; the *conformance* columns are scores of our own variants at reduced scale and say nothing about the six indexes of the paper.

### 7.4 What matched, and what did not

**PLA values printed in the paper.** history 105,468 / 468 and libio 145,808 / 639 [results/aidb/hardness.json] versus "h_l = 105k, h_g = 468" and "h_l = 146k, h_g = 639" [AIDB §2.2]: PLA-4096 identical, PLA-32 identical to the three printed digits. Together with the PGM-index comparison on the full books and wise files, this is the evidence that our PLA-ε is the paper's PLA-ε.

**The coverage column** [results/aidb/scores.json full.metrics.*.coverage; AIDB Table 2 Cov]:

| composition | paper Cov (|C|, |U|) | ours Cov (|C|, |U|) |
|---|---|---|
| RMSE, ME, PLA-32, PLA-4096 | 1.00 (45, 0) | 1.000 (45, 0) |
| CD | 1.00 (45, 0) | 1.000 (45, 0) |
| GRE = PLA-32·PLA-4096 | 0.47 (33, 12) | 0.467 (33, 12) |
| RMSE·ME | 0.82 (41, 4) | 0.822 (41, 4) |
| RMSE·PLA-32, ME·PLA-32 | 0.60 (36, 9) | 0.600 (36, 9) |
| RMSE·PLA-4096, ME·PLA-4096 | 0.42 (32, 13) | 0.422 (32, 13) |
| RMSE·ME·PLA-32 | 0.51 (34, 11) | 0.511 (34, 11) |
| RMSE·ME·PLA-4096 | 0.33 (30, 15) | 0.333 (30, 15) |
| RMSE·PLA-32·PLA-4096, ME·PLA-32·PLA-4096 | 0.24 (28, 17) | 0.244 (28, 17) |
| RMSE·CD | 0.47 (33, 12) | 0.422 (32, 13) |
| ME·CD | 0.56 (35, 10) | 0.511 (34, 11) |
| CD·PLA-32 | 0.60 (36, 9) | 0.556 (35, 10) |
| CD·PLA-4096 | 0.42 (32, 13) | 0.378 (31, 14) |
| RMSE·ME·CD | 0.42 (32, 13) | 0.378 (31, 14) |
| RMSE·CD·PLA-32 | 0.33 (30, 15) | 0.289 (29, 16) |
| RMSE·CD·PLA-4096 | 0.16 (26, 19) | 0.111 (25, 20) |
| ME·CD·PLA-32 | 0.38 (31, 14) | 0.333 (30, 15) |
| ME·CD·PLA-4096 | 0.20 (27, 18) | 0.156 (26, 19) |
| CD·PLA-32·PLA-4096 | 0.24 (28, 17) | 0.200 (27, 18) |

All fifteen CD-free compositions agree exactly, which is consistent with (though not a proof of) our RMSE, ME, PLA-32 and PLA-4096 inducing the same ten-dataset orderings as the paper's. Every one of the ten CD-containing compositions has exactly one more incomparable pair than in the paper (2/45 = 0.044 lower coverage). The simplest explanation is a single pair that the paper's CD orders in the same direction as the other four scalars and that our CD orders the other way; the pairs with that property in our table are (books, fb) (our CD 246 vs 110), (covid, planet) (27 vs 21), (history, libio) (8 vs 2) and (libio, wise) (2 vs 10) [scratch/worked_examples_03.out.txt, section C]. None of them is within the ±1 rounding band, so this is not the floor-rounding caveat; the paper prints no CD values, so the pair cannot be identified. Reading the paper's Figure 6 (CD·PLA-32 plane) by eye, osm sits near CD ≈ 7,700 and genome near ≈ 1,000 while our values are 4,107 and 585, whereas books (≈ 250 vs 246) and fb (≈ 100 vs 110) look identical [unverified: positions read off the figure; AIDB Fig. 6, p. 7]. A larger CD on the same keys would come from a different FMCD configuration or arithmetic; our port was read against LIPP's source (commit fe6ca49) without finding a discrepancy [hardness.hpp:15-22; results/aidb/MEETING_NOTES.md §1]. This remains the open implementation question of the study. The per-dataset comparable-partner counts under GRE (4 to 8) and CD·PLA-32 (6 to 9) match the colour-scale ranges of Figures 2 and 6 [scratch/worked_examples_03.out.txt; unverified: scale endpoints read from the figures].

**Conformance of our variants** (full-file hardness, uniform samples, ten variants, ddof 0) [results/aidb/scores.json full]: scalar Conf = RMSE −0.317, ME −0.217, CD +0.179, PLA-32 −0.306, PLA-4096 −0.407; GRE −0.281; CD·PLA-32 +0.092; for the control `packed_rank` alone: CD +0.472, PLA-32 −0.185, GRE −0.155. On the window samples [results/aidb_window/scores.json full]: CD +0.277, PLA-32 −0.197, GRE −0.085; `packed_rank` on CD +0.512. Two honest readings. First, CD is the only scalar with a positive mean conformance for our variants on both sample modes, although our index corrects locally (which under Finding 3 would predict PLA-32): at 2M keys the throughput spread between datasets is small (σ = 0.32 MOPS on a 1.9-2.9 MOPS range for `packed_rank`) and dominated by block decoding rather than by model error, and the hardness was measured on the full 200M-key files while the index ran on samples whose local structure differs (section 7.1). Second, with ten datasets and a measurement noise of up to ±8 % between two runs of an identical structure on this host [results/aidb/MEETING_NOTES.md §3-4], per-pair gaps are often within noise, and the sigmoid weighting gives such pairs weights near 0.5 in either direction. These conformance numbers are therefore reported as a protocol exercise, not as evidence about which metric is right; the mechanism claims of sections 04-07 rest on the deterministic probe counters, not on throughput.

### 7.5 Where the rest of the story is

Section 04 (host index and controls) describes the SCALE-LI map, `Region`, `Index`, the probe counters and the variants; section 05 the NFL-style transform and its bypass; section 06 the CSV-style virtual points and the region/root selectors; section 07 the experiments E1-E7 and their numbers, including how each component moves the five metrics across the six scopes (transform: RMSE −98 % on fb to +111 % on stack, PLA-32 by at most 5 segments; virtual points at α = 0.1: PLA-32 −43 to −61 % on six uniform samples, RMSE +6 to +10 %) [results/aidb/MEETING_NOTES.md §1; results/aidb/hardness.json].

---

## 8. Questions the supervisor may ask, with answers

1. **"What exactly is a hardness metric in this paper, and what is it supposed to predict?"** A function h(S) of the dataset alone (possibly vector-valued) such that h(S₁) < h(S₂) implies S₁ gives higher lookup throughput than S₂ under a uniform read-only workload over the loaded keys [AIDB §2.2]. It predicts an *ordering* of datasets, per index, not an absolute throughput; its quality is scored by how often the ordering is right (conformance, per index, averaged) and how many pairs it orders at all (coverage).

2. **"Why do RMSE and ME not conform?"** Because they measure global non-linearity (one least-squares line over 200M keys) and no evaluated index pays for global error: LIPP pays for slot collisions (CD), PGM/ALEX/FINEdex pay for local misprediction inside their leaves (PLA-32), and RMI/XIndex have a fixed root that ignores misprediction cost altogether. Their best per-index cells are 0.41 (RMSE on ALEX) and 0.39 (ME on ALEX), and both are negative on XIndex [AIDB Table 2, Finding 4]. A concrete example of the mismatch: fb has RMSE = 29 % of n, 2.4× osm's, but osm has 37× fb's conflict degree [results/aidb/hardness.json]; which of the two is "harder" depends on the index.

3. **"Why is CD aligned with LIPP?"** CD is the maximum slot occupancy of LIPP's own bulk-load root model (FMCD): the same fit, the same slot mapping ⌊a·k + b⌋ clamped to [0, L − 1], the same count. LIPP stores keys at their predicted slot and chains a child node on every collision, so its lookup cost grows with conflicts and only with conflicts; the root sees most of them [AIDB §4.1, §4.2; hardness.hpp:108-227]. Hence Conf_LIPP(CD) = 0.84, the highest scalar cell of Table 2.

4. **"What does coverage trade against conformance, and why is the tradeoff inherent?"** Adding a dimension can only turn comparable pairs into incomparable ones (a pair stays comparable only if the new dimension agrees with the old ones), so coverage never rises with dimensionality; conformance, which is computed only on the pairs still ordered, can rise because the metric commits only where its dimensions agree. At six dimensions (one per index's throughput order) conformance is 1 by construction and coverage is at its lowest [AIDB §4.2, Finding 1, frontier construction]. Coverage guards against a metric "declaring most pairs incomparable and committing only to the safest few" [AIDB §3.2].

5. **"Is the paper's history/libio example a counterexample to GRE or just noise?"** It is a violating pair under GRE (libio harder on both dimensions, LIPP 27 % faster on libio), and Table 2 shows it is not isolated: GRE's conformance on LIPP is only 0.42 [AIDB §2.2, Table 2]. Whether 7.1 vs 5.6 MOPS is outside LIPP's run-to-run noise is not reported (no variance is printed).

6. **"Why the sigmoid? Why not count pairs?"** Counting is the strawman of §3.1. The sigmoid of the σ-normalised gap gives conforming pairs with a large gap a weight near 0 (any metric orders them) and violating pairs with a large gap a weight near 1 (egregious), so the score concentrates on the pairs that discriminate between metrics [AIDB §3.1-3.2]. Worked consequence in section 4.4: one violation with w = 0.71 against four conforming pairs summing to 0.79 gives Conf ≈ 0.05.

7. **"What is PLA-32 exactly, and how is it different from ME ≤ 32?"** PLA-32 is the minimum number of contiguous segments such that each has *some* line within ±32 ranks of all its points (O'Rourke's optimal online algorithm, as PGM-index implements it, with PGM's sentinel and duplicate conventions) [AIDB §2.2, §4.1; hardness.hpp:230-311]. ME is the worst error of the *least-squares* line over the whole set; a set with ME > ε can still be one PLA-ε segment (the toy set has ME = 2.35 and PLA-2 = 1 segment). In a PGM-index with ε = 32 the last-mile search covers 65 positions; PLA-32 counts the leaves such an index needs.

8. **"How did you make sure your metrics are the paper's metrics if there is no code?"** By reimplementing them from the sources the paper names (LIPP's FMCD for CD, PGM-index's optimal PLA for PLA-ε) and verifying against those sources: PLA counts identical to PGM's `make_segmentation` on the fixture, six synthetic sets, 5M-key prefixes of all ten files and the full books and wise files; RMSE/ME within 1e-11 relative of an exact rational oracle; FMCD read against LIPP's source; and, at the level of the paper's own numbers, history and libio's printed PLA values and the whole CD-free coverage column of Table 2 reproduced exactly [results/aidb/verification/verdicts.json; results/aidb/scores.json].

9. **"Then why do the CD compositions not match?"** Each of the ten CD-containing coverages is lower than the paper's by exactly 2/45, i.e. one pair more incomparable. Under our CD, four pairs are ordered against all four other scalars; the paper's CD must order at least one of them the other way. The paper prints no CD values; read off its Figure 6, osm and genome appear to have roughly twice our CD [unverified]. Our port follows LIPP's `build_tree_bulk_fmcd` line by line (gap count 1 at 200M keys, L = 400M slots, U_T + 1e-6, floor and clamp); a different gap count, a different node size or double instead of long double arithmetic in the paper's run would change CD. It is an open implementation question, disclosed as such [results/aidb/MEETING_NOTES.md §1; hardness.hpp:119-127].

10. **"Your conformance numbers are negative for PLA-32. Does that contradict Finding 3?"** No, because they are not measured in the paper's setting: our "indexes" are variants of one compressed map at 2M keys, where decoding dominates and throughput differences between datasets are within the ±8 % host noise, and the hardness was computed on the 200M-key files while the index ran on samples with different local structure [results/aidb/MEETING_NOTES.md §3-4; section 7.4]. The coverage column is the comparable part; our conformance numbers are a protocol exercise. The claims of our study rest on deterministic probe counters.

11. **"What does 'the worker thread is pinned' tell us about the measurement?"** Single-threaded lookups on one core of a two-socket Xeon Gold 5118 (12 cores/socket, 2.3 GHz, 384 GiB), 100M lookups measured after 20M warm-up, all keys resident [AIDB §4.1]. Not reported: repetitions, variance, NUMA placement, index parameters, whether lookups are sampled with replacement.

12. **"Why does the paper exclude writes?"** "Write operations introduce performance overhead that is not directly related to dataset hardness" [AIDB §4.1]: update cost depends on buffering, splitting and retraining policies, not on how well R(·) can be fitted.

13. **"What would a better metric look like, according to the paper?"** Two-dimensional (Finding 5), built from CD and PLA-32 (Findings 2 and 6), or adapted to the target index family (§5, workload adaptation); and the community needs generators that hit target hardness values, because the GRE generator produces datasets whose ALEX/LIPP behaviour differs from the reference by up to 64 % [AIDB Table 3].

14. **"Is the Pareto frontier of Figure 4 real?"** It is an upper bound: a greedy beam search over dataset orderings (10! candidates per dimension, top-1k scalar bases, Pareto-optimal 2-D bases, up to six dimensions) whose points may not be realisable by any computable feature of the keys; the paper lists defining a *tight*, realisable frontier as open [AIDB §4.2, §5].

15. **"How much did the ten full files cost you to score, and is it exact?"** 12.9-27.4 s per file on 16 threads for all five metrics, plus a one-off sort for the seven files served unsorted; RMSE/ME/PLA-ε are exact up to floating-point summation (keys are handled in 128-bit integers, never rounded to double), CD is LIPP-faithful but carries the ±1 floor-rounding caveat of LIPP's own arithmetic [results/aidb/hardness_details.json; hardness.hpp:30-33, 119-127].
