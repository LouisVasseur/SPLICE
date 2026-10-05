# Literature check: jointly optimising an input warp and an output slot assignment for learned indexes

Date: 2026-09-22. Scope: the proposed objective

```
min over (f, V, a, b):  sum_i ( a*f(x_i) + b - s_i )^2
s.t. f order-preserving on the data; s = slot ranks induced by V; |V| <= lambda; (a,b) refit by LS
solved by block coordinate descent: (1) fix V, fit f; (2) fix f, greedy virtual-point insertion in z-space; repeat.
```

## Bottom line (read this first)

**The joint objective and the block-coordinate-descent solution are already published.** Li, Chen, Ding, Zeng and Zhou, *A Pluggable Learned Index Method via Sampling and Gap Insertion* (arXiv:2101.00808, 2021), §5.1, states the gap-insertion problem as exactly this optimisation — a budgeted, monotonicity-constrained reassignment of positions that maximises the improvement of the *refitted* model — says in so many words that it is solved by block coordinate descent, and then adds a paragraph ("Gap Insertion for Non-Linear Models") explicitly extending the scheme to **any monotonically increasing non-linear model**. A learned monotone flow `z = f(x)` followed by a refitted line is exactly such a model. So the formulation Louis wrote down is, as a formulation, prior art from 2021.

What is *not* in the literature, as far as this search could establish:

1. Nobody has actually instantiated that non-linear branch with a learned key-space transform. Li et al. only ever run linear segment models; their "non-linear" paragraph is a one-paragraph remark with no algorithm and no experiment.
2. Nobody has combined NFL with CSV or with gap insertion. CSV (the EDBT'25 paper) explicitly frames NFL and gap insertion as **competing alternatives** to itself (its Table 1 is a three-way comparison CSV / NFL / GI), not as composable components. No paper citing CSV (4 citations) and no paper citing NFL (42 citations, scanned) combines a key transform with a gap/virtual-point budget.
3. Nobody has posed the allocation question — *given a fixed linearisation budget, how much should go to global warp capacity versus local slack* — as a measured trade-off curve. The closest is black-box RL tuning of ALEX's gap ratio alongside node size (LITune, SIGMOD'25), which tunes both but characterises neither.
4. The root-of-the-index / fence-array setting is essentially untouched: CSV and GI both operate on leaf-level key positions, NFL transforms all keys.

Honest summary: **the novelty available here is an empirical/mechanistic contribution, not a formulation contribution.** A reviewer who knows Li et al. will say "you changed the model class inside a known alternating scheme." That is a fair criticism and the write-up must concede it up front.

---

## Q1. Learned-index work that jointly optimises an input transform AND an output slot assignment

### 1.1 The closest prior work — same objective, same algorithm, different model class

**Yaliang Li, Daoyuan Chen, Bolin Ding, Kai Zeng, Jingren Zhou. "A Pluggable Learned Index Method via Sampling and Gap Insertion." arXiv:2101.00808 (2021).** https://arxiv.org/abs/2101.00808 — arXiv preprint only, no venue; it is reference [16] of the CSV paper.

What it does. Two pluggable techniques on top of any learned index: (a) sampling for cheap construction, (b) *result-driven gap insertion*, a budgeted reassignment of record positions. It first sets up an MDL objective for index learning, `M* = argmin_M L(M) + alpha * L(D|M)`, where `L(M)` is model description length (parameters, ops) and `L(D|M)` is the last-mile correction cost, and observes that RMI depth/width, FITing-Tree/PGM `eps` and B+-tree page size all act as the knob `alpha`.

The gap-insertion problem, verbatim in structure (their Eq. 2):

```
max over {y^g_i}   sum_{(x_i,y_i) in D}  L(y_i, M*_D(x_i)) - L(y^g_i, M*_{D^g}(x_i))
s.t.  u_i in Z>=0,  sum_i u_i <= rho * n,
      y^g_i = y_i + sum_{j<=i} u_j,   y^g_i < y^g_j iff x_i < x_j
```

i.e. minimise the loss of the *refitted-optimal* model over a budgeted (`rho*n`), monotonicity-preserving reassignment of ranks. That is the `V` half of Louis's objective, with `lambda = rho*n`.

Their solver, quoting the paper: *"Inspired by block coordinate gradient descent, here we describe a two-step result-driven solution … The solution first proposes a new index mechanism M' that can achieve better preciseness … and then move records to y^g that are as close as possible to the positions predicted by M'."* That is exactly the two blocks of the proposed BCD, with the roles ordered (model first, then slots) rather than (slots first, then model).

And the crucial paragraph, *Gap Insertion for Non-Linear Models*: *"the idea of results-driven gap insertion is general and easy to be extended to other non-linear models. Specifically, as long as the non-linear models to be learned are monotonically increasing functions, we can also introduce the non-linear hypothetical lines by anchoring a few points…"*

**How the proposed work differs.**
- Li et al. never build an input-side warp. Their `M` is a piecewise-linear index model over the raw keys; there is no `f` with its own parameters, no order-preserving flow, no separate warp capacity to budget against `lambda`. The non-linear extension is a remark, not a method.
- Their gaps *chase* the model: `y^g_i` is backward-inferred from `M'(x_i)` and monotonised. CSV's insertion is instead a greedy search over insertion sites scored by a linearity criterion. These are different step-(2) operators inside the same outer loop.
- They do not iterate to convergence; it is a two-step, not an alternating, procedure. The convergence argument ("each step cannot increase the loss") is trivially true for both and is not a contribution.
- They report a heavy space cost — CSV reports GI's storage overhead as up to **87%**, and notes GI can map several keys to the same position, needing an overflow array.
- Level: leaf positions, not root fences.

Residual novelty vs this paper: instantiating step (1) with a learned monotone transform and measuring whether it helps; the fence-level setting; the capacity-vs-budget allocation study. Nothing about the objective itself.

### 1.2 The two methods in play

**Shangyu Wu, Yufei Cui, Jinghuan Yu, Xuan Sun, Tei-Wei Kuo, Chun Jason Xue. "NFL: Robust Learned Index via Distribution Transformation." PVLDB 15(10): 2188–2200, 2022.** https://www.vldb.org/pvldb/vol15/p2188-wu.pdf · https://arxiv.org/abs/2205.11807 · code https://github.com/luffy06/NFL

Input-side only. A numerical normalizing flow maps keys to a near-uniform latent space; a downstream index (AFLI) is built there. Introduces *conflict degree* and *tail conflict degree* as the transform-quality metric. AFLI's data nodes are themselves gapped arrays, with the number of gaps set from the tail conflict degree — but those gaps are a **consequence** of the already-fixed transform, allocated by a formula, never optimised against it. NFL has no budgeted slot-assignment problem and no feedback from the layout back to the flow.

**Kasun Amarasinghe, Farhana Choudhury, Jianzhong Qi, James Bailey. "Learned Indexes with Distribution Smoothing via Virtual Points." EDBT 2025.** https://openproceedings.org/2025/conf/edbt/paper-204.pdf · https://arxiv.org/abs/2408.06134

Output-side only. Inserts virtual points (empty slots) under a smoothing budget so the CDF over slot ranks is straighter; CSV integrates this into existing learned indexes. Its related-work section is the single most important passage for this project, because it treats the two ideas as **mutually exclusive**:

- on NFL: *"The distribution transformation introduces overheads, while queries also need to be transformed to use the index. Further, the transformation may increase the tail conflict degree for certain distributions, making it unsuitable in those instances."*
- on GI [16]: rank manipulation causes position collisions, needs an overflow array, up to 87% storage increase.
- on gapped arrays generally (ALEX, APEX, FINEdex): *"they do not consider minimizing the indexes' model prediction errors when adding gaps, in contrast to our approach which does."*
- Table 1 is CSV vs NFL vs GI on four binary criteria.

So the *combination* is not proposed anywhere in CSV — and CSV's stated objection to NFL (that the transform can hurt on some distributions) is precisely the objection the joint method would have to answer. Note this also gives Louis an explicit, citable prior claim to contradict or confirm.

Methodological ancestry worth citing: CSV derives its insertion machinery from **Evgenios M. Kornaropoulos, Silei Ren, Roberto Tamassia, "The Price of Tailoring the Index to Your Data: Poisoning Attacks on Learned Index Structures," SIGMOD 2022, 1331–1344** (https://arxiv.org/abs/2008.00297) — finding points to *insert* into a key set that maximally change a CDF-fitted regression. CSV runs the same machine with the sign flipped. If the joint method is framed as "optimal anti-poisoning under a budget", that is the right lineage.

### 1.3 The rest of the named systems — none do this

| System | Citation | Input transform? | Output slot assignment? | Jointly? |
|---|---|---|---|---|
| RMI | Kraska, Beutel, Chi, Dean, Polyzotis, "The Case for Learned Index Structures," SIGMOD 2018, 489–504. https://arxiv.org/abs/1712.01208 | no | no | no |
| FITing-Tree | Galakatos, Markovitch, Binnig, Fonseca, Kraska, SIGMOD 2019, 1189–1206. https://dl.acm.org/doi/10.1145/3299869.3319860 | no | no (eps-segments over raw ranks) | no |
| PGM-index | Ferragina, Vinciguerra, PVLDB 13(8):1162–1175, 2020. https://arxiv.org/abs/1910.06169 | no | no | no |
| RadixSpline | Kipf, Marcus, van Renen, Stoian, Kemper, Kraska, Neumann, aiDM@SIGMOD 2020. https://dl.acm.org/doi/10.1145/3401071.3401659 | radix *bucketing* of the key prefix — a fixed, unlearned, piecewise-constant routing, not a fitted monotone warp | no | no |
| ALEX | Ding, Minhas, Yu, Wang, Do, Li, Zhang, Chandramouli, Gehrke, Kossmann, Lomet, Kraska, SIGMOD 2020, 969–984. https://arxiv.org/abs/1905.08898 | no | **yes** — gapped arrays with model-based inserts, density bounds, and a cost model over expected search+shift cost | gaps exist for *inserts*, sized by density bounds; not optimised against a transform |
| LIPP | Wu, Zhang, Yu, Cai, Zhang, Zhang, "Updatable Learned Index with Precise Positions," PVLDB 14(8):1276–1288, 2021. https://arxiv.org/abs/2104.05520 | no | **yes, degenerately** — every key is placed exactly at its predicted position, conflicts pushed into child nodes | no; this is the `lambda -> infinity` limit of the joint objective with no budget, which is worth saying explicitly |
| SALI | Ge, Zhang, Shi, Luo, Guo, Chai, Chen, Pan, PACMMOD 1(4):258, 2023. https://dl.acm.org/doi/10.1145/3626752 | no | inherits LIPP-style placement + probability-model-driven adaptation | no |
| DILI | Li, Lu, Zeng, Wang, Zheng et al., "DILI: A Distribution-Driven Learned Index," PVLDB 2023. https://arxiv.org/abs/2304.08817 | builds the tree from the data distribution | no budgeted slot assignment | no |
| LMG Index | Chen, Yao, arXiv:2512.24824, 2025/2026. https://arxiv.org/abs/2512.24824 | no | **yes** — "optimizing gap allocation" over gapped-array segments, plus an optimal error-threshold training algorithm | gap allocation and per-segment error threshold are co-designed; **no key transform**. This is the closest thing to a capacity-vs-slack co-design, one side of it |
| B^S-Tree | ICDE 2025, "B^S-Tree: A Gapped Data-Parallel B-Tree" | no | yes (gaps for SIMD-friendly parallel search) | no |
| SELIX | "On Self-Designing Learned Indexes," PACMMOD 2026. https://dl.acm.org/doi/10.1145/3802096 | no | node layout is one of the searched design dimensions | searches a structural design space; no key-space warp, no budgeted virtual points |

### 1.4 Adjacent work that names both ideas but keeps them separate

**Ali Hadian, Thomas Heinis. "Interpolation-friendly B-trees: Bridging the Gap Between Algorithmic and Learned Indexes." EDBT 2019, 710–713.** https://openproceedings.org/2019/conf/edbt/EDBT19_paper_355.pdf · DOI 10.5441/002/edbt.2019.93

Worth citing for the framing, not the method. Its Figure 1(c) lists the roles a "helping model" can play beside a classical index: **Transformation**, **Acceleration**, and **Layout optimization** — i.e. the taxonomy that separates input warping from layout already exists, and nobody filled in the diagonal. Its related work also anticipates the mechanism hypothesis: *"a data transformation that makes the key distribution closer to uniform can benefit IFB-tree and further decreases the interpolation error."* IFB-tree itself only labels interpolation-friendly nodes; it inserts no artificial keys and learns no transform.

**Vikram Nathan, Jialin Ding, Mohammad Alizadeh, Tim Kraska. "Learning Multi-Dimensional Indexes" (Flood), SIGMOD 2020.** https://arxiv.org/abs/1912.01668 — and **Ding, Nathan, Alizadeh, Kraska, "Tsunami," PVLDB 14(2):74–86, 2020**, https://arxiv.org/abs/2006.13282. Flood is the one learned index that genuinely *"jointly optimiz[es] the index structure and data storage layout"* for a workload. But: multi-dimensional, workload-driven, the "layout" is a grid partitioning, there is no monotone warp and no budget on inserted empty space. Correct citation for "joint model+layout optimisation exists in indexing", wrong shape for this objective.

**Ani Kristo, Kapil Vaidya, Ugur Çetintemel, Sanchit Misra, Tim Kraska. "The Case for a Learned Sorting Algorithm." SIGMOD 2020.** https://dl.acm.org/doi/10.1145/3318464.3389752 — uses a learned CDF model to scatter keys into an over-allocated array with gaps. Model drives slot assignment, but the gap budget is a uniform over-allocation factor, never optimised.

**Ali Hadian, Thomas Heinis. "Shift-Table: A Low-latency Learned Index for Range Queries using Model Correction."** https://arxiv.org/abs/2101.10457 — an output-side *correction layer* bolted onto a fitted model, fixing local error without changing the model. Conceptually "local slack on the output side", but as a lookup table rather than as inserted slots, and not co-optimised with the model.

**Descendants of NFL that stay transform-only** (proving the transform branch is being actively worked, without anyone adding the layout branch): **TPLI: Transformation-partitioned learned index for multi-domain key-value stores,** J. Intell. Inf. Syst., 2026, DOI 10.1007/s10844-026-01059-2 (NFL's numerical NF for numeric keys + k-means partitioning for strings, on XIndex-R); and **DTLI: Distribution transformation-based lightweight learned indexing,** Science Progress, 2026.

### 1.5 Monotone neural networks / learned CDF models for indexing

NFL's flow is the main one. Its own text notes a monotonic-neural-network line of work (`nfl.txt` line 444 refers to achieving state of the art "using a monotonic neural network"). Outside indexing, the relevant constructions are monotone networks with non-negative weights and saturating activations (Sill, "Monotonic Networks," NIPS 1997), Unconstrained Monotonic Neural Networks (Wehenkel & Louppe, NeurIPS 2019, https://arxiv.org/abs/1908.05164), and monotone/flow-based warpings such as **"Monotonic warpings for additive and deep Gaussian processes"** (arXiv:2408.01540). None of these are paired with an output-side slot budget. Louis's 2x2x2 order-preserving flow (with fractional-feature weights zeroed → `z = sum of two tanh` of a scalar = one global bend) sits at the very low-capacity end of this family; that low capacity is *the* thing being budgeted, and no indexing paper discusses capacity of a monotone warp as a resource at all.

---

## Q2. Is the general pattern known outside indexing? Yes — and it is old.

### 2.1 Closest formal analogue: alternating estimation of monotone transforms on both sides of a regression

**Leo Breiman, Jerome H. Friedman. "Estimating Optimal Transformations for Multiple Regression and Correlation." Journal of the American Statistical Association 80(391): 580–598, 1985.** (ACE — Alternating Conditional Expectations.) https://en.wikipedia.org/wiki/Alternating_conditional_expectations · R implementation `acepack::ace` (which exposes a *monotone* option per variable).

ACE finds transformations `theta(Y)` and `phi(X)` maximising the correlation between `theta(Y)` and `phi(X)` — i.e. it warps the *response* and the *predictor* so that the relation between them becomes linear — by alternating: fix one, optimise the other, repeat. That is, structurally, the proposed objective: transform the input, transform the output, refit a line, iterate. This is the single closest formal analogue and it predates learned indexes by 33 years.

Differences that matter: ACE's output transform is an arbitrary (optionally monotone) smooth function of `Y`, not a *budgeted integer* reassignment of ranks; there is no `|V| <= lambda` constraint, so no allocation problem; and ACE maximises correlation, which is invariant to affine rescaling of either side, whereas the root-probe objective is not.

Siblings in the same family, all alternating-least-squares over monotone transforms:
- **Forrest W. Young, Jan de Leeuw, Yoshio Takane. "Regression with qualitative and quantitative variables: An alternating least squares method with optimal scaling features." Psychometrika 41(4): 505–529, 1976.** The ALSOS / "optimal scaling" framework; MORALS is the monotone-regression instance. Monotone transforms of predictors and response, alternating LS, exactly this loop.
- **Robert Tibshirani. "Estimating Transformations for Regression via Additivity and Variance Stabilization (AVAS)." JASA 83: 394–405, 1988.** ACE with a variance-stabilising, monotone response transform.
- **G. E. P. Box, D. R. Cox. "An Analysis of Transformations." JRSS-B 26(2): 211–252, 1964.** One-parameter monotone response warp; the ancestor of "few-parameter monotone transform chosen to make a linear model fit".

### 2.2 Closest engineering analogue: companding quantisation

The compander — monotone compressor `c(x)`, then a **uniform** quantiser, then the inverse expander — is exactly "warp the input so a trivial output rule becomes optimal". The theory is complete and quantitative:
- **W. R. Bennett. "Spectra of Quantized Signals." Bell System Technical Journal 27(3): 446–472, 1948** (Bennett's integral; point density `lambda(x) = c'(x)`).
- **P. F. Panter, W. Dite. "Quantization distortion in pulse-count modulation with nonuniform spacing of levels." Proc. IRE 39(1): 44–48, 1951** (optimal point density proportional to `p(x)^{1/3}`).
- **S. P. Lloyd (1957/1982) and J. Max (1960)**: the optimal *non-uniform* quantiser via **alternating** optimisation of levels and boundaries — the Lloyd–Max iteration is a block coordinate descent of the same shape as the one proposed.
- **A. Gersho, R. M. Gray. *Vector Quantization and Signal Compression*, Kluwer, 1992** — the textbook statement that a compander with the right point density asymptotically matches the optimal non-uniform quantiser at high rate.
- Modern write-up: Stanford EE269, *Non-uniform quantization*, http://web.stanford.edu/class/ee269/Lecture_nonuniform_quantization.pdf

The sharp lesson for this project, and the one most likely to be raised against it: **in companding theory the input warp and the output level placement are the same degree of freedom** — a compressor *is* a level placement. If, in the fence setting, a high-capacity monotone warp and an unbounded virtual-point budget are also interchangeable, then the joint method has one effective knob, not two, and the interesting regime is exactly the one Louis identified: *low* warp capacity and *bounded* budget, where the two are no longer equivalent. That restriction is the defensible core of the idea and should be stated as such.

### 2.3 Monotone input warping and monotone output warping in ML — separately

- **Edward Snelson, Carl E. Rasmussen, Zoubin Ghahramani. "Warped Gaussian Processes." NIPS 16, 2004.** https://mlg.eng.cam.ac.uk/pub/pdf/SneRasGha04.pdf — learns a monotone warp of the *targets* jointly with GP hyperparameters, so a simple model fits in the warped space. Output-side. (Extensions: Lázaro-Gredilla, "Bayesian Warped GPs," NIPS 2012; "Compositionally-warped GPs," Neural Networks 2019.)
- **Jasper Snoek, Kevin Swersky, Richard Zemel, Ryan P. Adams. "Input Warping for Bayesian Optimization of Non-Stationary Functions." ICML 2014, PMLR 32:1674–1682.** https://arxiv.org/abs/1402.0929 — learns a few-parameter bijective monotone warp (Beta CDF) of the *inputs* so a stationary GP fits. Input-side, and notably a *low-capacity* monotone warp chosen for exactly the reason Louis wants one.

Searching specifically for work that warps both sides jointly in the GP setting returned nothing; the two literatures cite each other but were not, as far as this search showed, combined under one alternating objective. (This is a mildly encouraging gap, but it is a gap in a different field.)

### 2.4 Calibration / isotonic regression

Output-side monotone warping of a fixed model's scores is standard: **Zadrozny & Elkan, "Transforming classifier scores into accurate multiclass probability estimates," KDD 2002** (isotonic regression / PAV); **Platt, 1999** (parametric sigmoid). Recent monotone-constrained variants: **MCNet: Monotonic Calibration Networks** (arXiv:2503.00334), **Instance-Wise Monotonic Calibration by Constrained Transformation** (arXiv:2507.06516). In all of these the *model is frozen* and only the output map is learned — the opposite of the flow, and never alternating with it.

### 2.5 Optimal transport / histogram equalisation

Histogram equalisation is the 1-D monotone optimal-transport map `T = F_target^{-1} o F_source`, the unique monotone rearrangement pushing one density onto another (Villani, *Optimal Transport: Old and New*, 2009; Santambrogio, *OT for Applied Mathematicians*, 2015). Both NFL and CSV are, viewed this way, approximations of the same 1-D OT map: NFL approximates `T` on the *key* axis with a parametric monotone map; CSV approximates the inverse push-forward on the *rank* axis with a budgeted set of atoms. **Stating that explicitly — "two budget-constrained approximations of the same 1-D monotone transport map, one on each axis" — is, as far as this search found, not said anywhere in the indexing literature, and is the cleanest theoretical framing available for this project.** It is a framing contribution, not a result.

---

## Q3. Allocating a fixed "linearisation budget" between model capacity and local slack

This is the weakest-covered of the three questions, which is good news.

**What exists.**

1. **The MDL framing, Li et al. 2021 §3.2**: `L(M) + alpha * L(D|M)`, with the observation that RMI depth/width, PGM/FITing-Tree `eps`, and page size all act as `alpha`. This is "model capacity vs residual error", and it is the right vocabulary — but the *slack* there is search cost, not an allocatable resource. Their experiments sweep gap ratio `rho` against sample rate `s` (their Figure 10) and index size against latency (Figure 4), but never `rho` against model capacity on one axis pair.
2. **Model-size / latency Pareto exploration**: **Ryan Marcus, Emily Zhang, Tim Kraska, "CDFShop: Exploring and Optimizing Learned Index Structures," SIGMOD 2020 (demo)**, https://dl.acm.org/doi/10.1145/3318464.3384706, and **Marcus et al., "Benchmarking Learned Indexes," PVLDB 14(1):1–13, 2020** (SOSD), https://arxiv.org/abs/2006.12804; **Maltry & Dittrich, "A Critical Analysis of Recursive Model Indexes," PVLDB 15(5), 2022**, https://arxiv.org/abs/2106.16166. All of these trace size-vs-latency frontiers for *model* structure only. No gap budget on the axis.
3. **ALEX's cost model** (SIGMOD 2020) trades expected search cost against expected shift cost and picks node density/fanout accordingly — the only principled, published allocation involving gaps, but the gaps are for *inserts*, and the "model capacity" side is fanout, not warp capacity.
4. **LMG Index** (arXiv:2512.24824) co-designs per-segment error thresholds with gap allocation — capacity and slack in the same optimisation, but with a linear model class and no transform.
5. **LITune — Tao Wang, Ling Liang, Guangyu Yang, Thomas Heinis, Eiko Yoneki, "A New Paradigm in Tuning Learned Indexes: A Reinforcement Learning Enhanced Approach," SIGMOD 2025**, https://arxiv.org/abs/2502.05001 — RL over 14 ALEX knobs including *gap ratio / split policy* and *max node size* simultaneously. Joint tuning, black box, no trade-off curve, no interpretation.
6. **Rate-distortion bit allocation** (Gersho & Gray, ch. 8; reverse water-filling) is the textbook general theory of splitting a fixed budget across components to minimise total distortion, and is the right citation if the allocation is formalised.
7. A speculative recent preprint, **Faruk Alpay, Levent Sarioglu, "Residual-Entropy Accounting for Routed Atom-Budgeted Learned Indexes," arXiv:2605.29061 (May 2026)**, https://arxiv.org/abs/2605.29061, does define a "predictor-atom budget" and a shadow-price allocation rule across directory space, array storage and repair-program space. It is an unreviewed arXiv preprint by authors without a track record in this area and I would cite it only defensively, if at all.

**What does not exist.** No paper plots root/leaf probe cost against *(warp parameters, virtual-point budget)* as a two-axis allocation, and nobody has asked whether a bend in the key space and a gap in the rank space are substitutes or complements at a fixed total cost. That question is genuinely open.

---

## What would remain novel, stated honestly

Novel, if the experiments come out:
- **The empirical answer.** Nobody has measured whether a low-capacity monotone warp plus a budgeted virtual-point set beats either alone. CSV asserts NFL's transform can *hurt* ("may increase the tail conflict degree for certain distributions"); measuring the interaction on osm and planet either confirms a published claim or contradicts it, and both are reportable.
- **The setting.** Fitting at the root over region fences (489 points) rather than over 2M keys is a different regime — tiny `n`, tiny model, and the probe count is *not* affine-invariant, so it is not covered by the tail-conflict-degree invariance argument. Keeping that distinction explicit (as the task brief already does) is what makes the experiment non-vacuous given the 100-config sweep result.
- **The allocation curve.** Probes as a function of (warp capacity, `lambda`) at fixed total cost, and whether the two resources substitute or compose. This is the piece with the least prior art.
- **The OT framing** of NFL and CSV as two budget-constrained approximations of the same monotone 1-D transport map, one per axis.

Not novel, and must be conceded:
- **The objective.** Li et al. 2021 Eq. 2 is the same budgeted, monotone, refit-optimal position-assignment problem.
- **The algorithm.** Li et al. call theirs block coordinate descent, and explicitly allow any monotone non-linear model in step (1).
- **The convergence argument.** Monotone decrease under exact block minimisation is standard and is not a contribution.
- **"Combining two published systems"** is not by itself a contribution; the contribution has to be the mechanism claim (global bend vs local irregularity) plus the measurement.

Risks a reviewer will raise:
- *Companding collapse*: if warp capacity and gap budget are substitutes, the joint method has one knob and the paper has no result. Pre-empt by fixing the low-capacity regime and showing the substitution rate.
- *Query-time cost*: NFL's transform must be applied to every lookup key (CSV's stated objection). At the root a bend is cheap, but the probe saving must beat the transform cost in wall-clock, not just in probes.
- *Li et al.'s 87% storage blow-up* is the cautionary precedent for output-side methods without a tight budget.

---

## Source list

Learned indexes
- Kraska, Beutel, Chi, Dean, Polyzotis. The Case for Learned Index Structures. SIGMOD 2018, 489–504. https://arxiv.org/abs/1712.01208
- Galakatos, Markovitch, Binnig, Fonseca, Kraska. FITing-Tree. SIGMOD 2019, 1189–1206. https://dl.acm.org/doi/10.1145/3299869.3319860
- Ferragina, Vinciguerra. The PGM-index. PVLDB 13(8):1162–1175, 2020. https://arxiv.org/abs/1910.06169
- Kipf et al. RadixSpline. aiDM@SIGMOD 2020. https://dl.acm.org/doi/10.1145/3401071.3401659
- Ding et al. ALEX. SIGMOD 2020, 969–984. https://arxiv.org/abs/1905.08898
- Wu et al. LIPP: Updatable Learned Index with Precise Positions. PVLDB 14(8):1276–1288, 2021. https://arxiv.org/abs/2104.05520
- Ge et al. SALI. PACMMOD 1(4):258, 2023. https://dl.acm.org/doi/10.1145/3626752
- Li, Chen, Ding, Zeng, Zhou. A Pluggable Learned Index Method via Sampling and Gap Insertion. arXiv:2101.00808, 2021. https://arxiv.org/abs/2101.00808
- Wu, Cui, Yu, Sun, Kuo, Xue. NFL: Robust Learned Index via Distribution Transformation. PVLDB 15(10):2188–2200, 2022. https://www.vldb.org/pvldb/vol15/p2188-wu.pdf
- Amarasinghe, Choudhury, Qi, Bailey. Learned Indexes with Distribution Smoothing via Virtual Points. EDBT 2025. https://openproceedings.org/2025/conf/edbt/paper-204.pdf · https://arxiv.org/abs/2408.06134
- Kornaropoulos, Ren, Tamassia. The Price of Tailoring the Index to Your Data. SIGMOD 2022, 1331–1344. https://arxiv.org/abs/2008.00297
- Hadian, Heinis. Interpolation-friendly B-trees. EDBT 2019, 710–713. https://openproceedings.org/2019/conf/edbt/EDBT19_paper_355.pdf
- Hadian, Heinis. Shift-Table. https://arxiv.org/abs/2101.10457
- Nathan, Ding, Alizadeh, Kraska. Learning Multi-Dimensional Indexes (Flood). SIGMOD 2020. https://arxiv.org/abs/1912.01668
- Ding, Nathan, Alizadeh, Kraska. Tsunami. PVLDB 14(2):74–86, 2020. https://arxiv.org/abs/2006.13282
- Kristo, Vaidya, Çetintemel, Misra, Kraska. The Case for a Learned Sorting Algorithm. SIGMOD 2020. https://dl.acm.org/doi/10.1145/3318464.3389752
- Li et al. DILI. PVLDB 2023. https://arxiv.org/abs/2304.08817
- Chen, Yao. LMG Index. arXiv:2512.24824. https://arxiv.org/abs/2512.24824
- On Self-Designing Learned Indexes (SELIX). PACMMOD 2026. https://dl.acm.org/doi/10.1145/3802096
- Wang, Liang, Yang, Heinis, Yoneki. LITune. SIGMOD 2025. https://arxiv.org/abs/2502.05001
- Marcus, Zhang, Kraska. CDFShop. SIGMOD 2020 demo. https://dl.acm.org/doi/10.1145/3318464.3384706
- Marcus et al. Benchmarking Learned Indexes (SOSD). PVLDB 14(1):1–13, 2020. https://arxiv.org/abs/2006.12804
- Maltry, Dittrich. A Critical Analysis of Recursive Model Indexes. PVLDB 15(5), 2022. https://arxiv.org/abs/2106.16166
- Liu et al. Why Are Learned Indexes So Effective but Sometimes Ineffective? PVLDB 18, 2025. https://www.vldb.org/pvldb/vol18/p2886-liu.pdf · https://arxiv.org/abs/2410.00846
- TPLI. J. Intell. Inf. Syst., 2026. https://link.springer.com/article/10.1007/s10844-026-01059-2
- Alpay, Sarioglu. Residual-Entropy Accounting for Routed Atom-Budgeted Learned Indexes. arXiv:2605.29061, 2026. https://arxiv.org/abs/2605.29061 (unreviewed; cite defensively)

Statistics / ML analogues
- Breiman, Friedman. Estimating Optimal Transformations for Multiple Regression and Correlation (ACE). JASA 80(391):580–598, 1985.
- Young, de Leeuw, Takane. Regression with qualitative and quantitative variables: an ALS method with optimal scaling features. Psychometrika 41(4):505–529, 1976.
- Tibshirani. Estimating transformations for regression via additivity and variance stabilization (AVAS). JASA 83:394–405, 1988.
- Box, Cox. An Analysis of Transformations. JRSS-B 26(2):211–252, 1964.
- Snelson, Rasmussen, Ghahramani. Warped Gaussian Processes. NIPS 16, 2004. https://mlg.eng.cam.ac.uk/pub/pdf/SneRasGha04.pdf
- Snoek, Swersky, Zemel, Adams. Input Warping for Bayesian Optimization of Non-Stationary Functions. ICML 2014. https://arxiv.org/abs/1402.0929
- Zadrozny, Elkan. Transforming classifier scores into accurate multiclass probability estimates. KDD 2002.
- Sill. Monotonic Networks. NIPS 1997. Wehenkel, Louppe. Unconstrained Monotonic Neural Networks. NeurIPS 2019. https://arxiv.org/abs/1908.05164
- Monotonic warpings for additive and deep Gaussian processes. arXiv:2408.01540. https://arxiv.org/abs/2408.01540

Quantisation / transport
- Bennett. Spectra of Quantized Signals. BSTJ 27(3):446–472, 1948.
- Panter, Dite. Quantization distortion in PCM with nonuniform spacing of levels. Proc. IRE 39(1):44–48, 1951.
- Lloyd. Least Squares Quantization in PCM. IEEE Trans. IT 28(2):129–137, 1982 (1957 tech report). Max. Quantizing for minimum distortion. IRE Trans. IT 6(1):7–12, 1960.
- Gersho, Gray. Vector Quantization and Signal Compression. Kluwer, 1992.
- Stanford EE269, Non-uniform quantization. http://web.stanford.edu/class/ee269/Lecture_nonuniform_quantization.pdf
- Villani. Optimal Transport: Old and New. Springer, 2009. Santambrogio. Optimal Transport for Applied Mathematicians. Birkhäuser, 2015.
