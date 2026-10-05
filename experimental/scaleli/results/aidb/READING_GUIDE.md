# Reading guide (deep): NFL, CSV, the AIDB hardness protocol, our implementations, the exact protocol and the results

Assembled 2026-09-22 from the section files in results/aidb/guide_deep/ (each section was written from the papers, the code and the result files by a separate agent; the verification pass was stopped early at the user's request, so sections 01, 02 and 05 received part of their corrections and 03, 04, 06 and 07 are unverified drafts: treat numbers here as pointers and confirm any you plan to quote against results/aidb/MEETING_NOTES.md and READING_GUIDE_SHORT.md, both of which were fully verified). The previous short guide is results/aidb/READING_GUIDE_SHORT.md; the verified meeting notes are results/aidb/MEETING_NOTES.md. Paths are relative to experimental/scaleli/ unless absolute.

## Contents

0. [01. NFL: Robust Learned Index via Distribution Transformation (deep)](#01-nfl-robust-learned-index-via-distribution-transformation-deep)
1. [02. Learned Indexes with Distribution Smoothing via Virtual Points (CSV), deep account](#02-learned-indexes-with-distribution-smoothing-via-virtual-points-csv-deep-account)
2. [03. How Hard Can Indexing Be? (Zhang, Tang, Ailamaki, AIDB@VLDB 2026), deep](#03-how-hard-can-indexing-be-zhang-tang-ailamaki-aidb-vldb-2026-deep)
3. [04. The ten datasets and our samples](#04-the-ten-datasets-and-our-samples)
4. [05 — The host index and the clean-room implementations, with worked examples](#05-the-host-index-and-the-clean-room-implementations-with-worked-examples)
5. [06 — The exact experimental protocol](#06-the-exact-experimental-protocol)
6. [07 — Results per dataset, with interpretation](#07-results-per-dataset-with-interpretation)
7. [Caveats and likely questions (from the short guide)](#caveats-and-likely-questions-from-the-short-guide)


---

## 01. NFL: Robust Learned Index via Distribution Transformation (deep)

**What this section gives you.** A self-contained, equation-level account of the NFL paper (Wu et al., 2022):
what problem it attacks (learned-index error grows with the curvature of the key CDF), the thesis (learn a
transform z = G⁻¹(x) that makes the keys near-uniform, then index z with a very simple structure), the
mathematics of normalizing flows from the change-of-variables formula up to the exact tiny network NFL
deploys (input 2, hidden 2, two layers, 8 weights, tanh, sum decoder), the feature expansion of Algorithm 3.1
line by line, the conflict-degree metrics (Definitions 3.1 and 3.2) with a 10-key worked example computed in
Python, the switching rule and the full Table 2 (169.53 ns/key at batch 1 versus 8.38 ns at batch 256), the
After-Flow Learned Index (AFLI) with Algorithm 3.2, the experimental protocol and every number the paper
prints, and finally which pieces our clean-room CONTROLS borrow (the 2D2H2L weight format, the two-feature
encoder, the tail conflict degree, the bypass) and which they do not (AFLI, the B-NAF trainer, batching). All
numbers are cited to the PDF page/section, the NFL README, our code or our result files; everything that could
not be checked carries an [unverified: ...] tag. Worked numbers were produced by
`results/aidb/guide_deep/scratch/nfl_worked.py` and `nfl_worked2.py` (stdlib only); their outputs are saved next to
them as `nfl_worked.out` and `nfl_worked2.out`.

Notation (shared with the other sections): keys k₁ < … < kₙ, rank r(kᵢ) = i − 1; linear model
f(k) = w·φ(k) + b with feature φ either the normalized key or the flow output z(k); conflict degree D and
tail conflict degree D₉₉; in the paper's own symbols the original key is x, the transformed key is z, the
generator is G_θ (z → x) and its inverse G_θ⁻¹ (x → z), the linear model is M, the tail percent is γ.
One warning before you open the PDF: the paper itself flips the symbol. §2.3 defines the generator as z → x
("x_i = G_θ(z_i)", eq. 4), but §3.1 writes "the normalizing flow first transform it to a target distribution
p(z), i.e., z = G_θ(x)" [NFL §2.3 p.4; NFL §3.1 p.4]. This guide keeps §2.3's convention throughout (G_θ: z → x,
G_θ⁻¹: x → z, the direction executed online), so where §3.1 of the PDF says G_θ(x) read G_θ⁻¹(x).

---

### 1. Bibliographic facts

| Item | Value | Source |
|---|---|---|
| Title | NFL: Robust Learned Index via Distribution Transformation | [NFL p.1] |
| Authors | Shangyu Wu, Yufei Cui (corresponding author), Jinghuan Yu, Xuan Sun, Tei-Wei Kuo, Chun Jason Xue; all City University of Hong Kong | [NFL p.1] |
| Venue | "Accepted as a conference paper in VLDB 2022"; the PDF's "PVLDB Reference Format" block is the arXiv v1 placeholder, so volume and pages are **not reported** in our copy | [NFL p.1] |
| Volume/pages | PVLDB 15(10), 2022 is what the previous guide and the workspace registry state; the arXiv v1 PDF does not print them [unverified: not in the PDF we hold; taken from READING_GUIDE.md §1.1] | [results/aidb/READING_GUIDE.md:14] |
| Local copy | `/Users/louisvasseur/Downloads/scaleli_sota/sources/2205.11807v1.pdf`, 13 pages, arXiv 2205.11807 v1 | [fitz page count] |
| Artifact | https://github.com/luffy06/NFL | [NFL p.1, "PVLDB Artifact Availability"] |
| Licence | GPL-3 (the repository licence; this is why the workspace never copies its code and calls its own transform a clean-room control). Checked: the README we hold (`nfl_readme.md`, 64 lines: Introduction, Requirements, Getting Started, Training, Results, Contact) contains **no licence statement at all**; the only sources for "GPL-3" are our own code comment, the workspace approach table and the previous guide | [transform.hpp:5], [docs/APPROACHES.md:18], [results/aidb/READING_GUIDE.md:15] [unverified: licence file not among the sources; nfl_readme.md has no licence text; stated only by our code comment and the workspace docs] |
| Build dependencies (README) | Intel MKL 11.3, CMake 3.12, GNU C++ 17, OpenMP; `source ~/intel/oneapi/setvars.sh --force intel64` then `scripts/bootstrap.sh` | [nfl_readme.md, Requirements / Getting Started] |
| Training dependency | the flow is trained in PyTorch ("we train a modified B-NAF in PyTorch") via `scripts/prepare_keys_for_flows.sh` then `train/train_flow.sh` | [NFL §4.1.3], [nfl_readme.md, Training] |
| Result line format | `(dataset) (index) (batch size) (bulk loading time) (transformation time in bulk loading) (model size) (index size) (overall throughput) (avg-T) (avg-I) (50-T) (50-I) (75-T) (75-I) (99-T) (99-I) (995-T) (995-I) (9999-T) (9999-I) (max-T) (max-I)` where T = transformation time, I = indexing latency | [nfl_readme.md, Results] |
| Contact | shangyuwu2-c@my.cityu.edu.hk | [nfl_readme.md] |

Two facts from the README matter later: (a) the benchmark records the transformation time **separately** from
the indexing latency at every percentile (T and I columns), so the paper's reported latencies can be read
as index-only or transform+index; (b) `batch size` is a first-class column of every result line, which
confirms that all measurements are batched (§3.1 of the paper says why; §5 below quantifies it).

---

### 2. The problem and the thesis

#### 2.1 Learned indexes approximate the CDF

A learned index predicts the position of a key with a model of the cumulative distribution function:

```
pos = F(key) · N                                                     [NFL eq. (1), §2.1]
```

where F(·) approximates P(x ≤ key) and N is the number of keys. In our notation, the rank r(kᵢ) = i − 1 is exactly
N·F_empirical(kᵢ) − 1, so a learned index is a regression from key to rank. A single model cannot reach
last-mile precision ("from thousands to hundreds" of positions), hence the Recursive Model Index (RMI) hierarchy
[NFL §2.1, Fig. 1]. Follow-ups replace the hierarchy by explicit **segmentation** of the key space into
pieces on which one linear model is accurate: PGM-Index's convex-hull segmentation with error bound ε,
LIPP's conflict-driven segmentation (all keys colliding at one predicted position go to one child), ALEX's
cost-enumerated fanout [NFL §2.1, Fig. 2].

The paper's diagnosis: every one of those heuristics costs build time, pre-allocated space and engineering
effort ("requires a long period of time to process and large pre-allocated space to maintain", and each index
"requires designing a set of supportive operations and optimizing their hyper-parameters") [NFL §2.1, last
paragraph].

#### 2.2 Why the error depends on the key distribution

Segmentation looks for sub-curves of the CDF that are "roughly linear", i.e. whose local probability density
is near-uniform [NFL §2.2, Fig. 3]. A linear model f(k) = w·k + b fits the rank exactly iff the keys are
equally spaced, i.e. iff the density is uniform on the segment. Curvature of the CDF (density that varies over
the segment) is what produces residuals r(kᵢ) − f(kᵢ). Therefore: **the harder the distribution, the more
segments (or the deeper the tree) a learned index needs**, and depth is paid on every lookup as extra
predictions and cache misses [NFL §2.1-2.2].

#### 2.3 Distribution transformation

Instead of cutting the CDF into pieces, map the keys first: find an invertible function z = G⁻¹(x) such that the
transformed keys z are (near-)uniformly distributed. Then the whole transformed CDF is close to one straight
line, and "a single linear model is sufficient to fit such curve" [NFL §2.2]. "Distribution transformation"
means exactly this: a learned, data-adaptive bijection of the key space chosen so that the pushed-forward
density is as flat as possible. The paper's evidence that this works even for an existing index is Table 1
(ALEX with and without the flow) [NFL §2.2, Table 1, p.3]:

| ALEX statistic | LLT without NF | LLT with NF | FB without NF | FB with NF |
|---|---|---|---|---|
| Max tree height | 4 | 3 | 11 | 3 |
| Average tree height | 2.30 | 2.01 | 5.91 | 2.02 |
| # Prediction errors | 925,487,063 | 118,833,075 | 928,113,206 | 453,003,864 |
| # Predictions | 124,831,692 | 101,108,381 | 492,112,591 | 100,462,387 |
| Throughput (Mops/s) | 8.21 | 11.57 | 4.13 | 9.16 |

Arithmetic (script section F): LLT throughput ×1.41, predictions ×1.23 fewer, error sum ×7.79 fewer
(errors per prediction 7.41 → 1.18); FB throughput ×2.22, predictions ×4.90 fewer, error sum ×2.05 fewer
(errors per prediction 1.89 → 4.51; the paper attributes the FB gain to the height reduction 11 → 3 and the
LLT gain to fewer out-of-boundary keys per node [NFL §2.2]). What "# Prediction Errors" measures (sum of
absolute errors? count of non-zero errors?) is **not defined** in the paper.

Two things to be ready to defend about this table. (i) The apparent FB paradox: the flow cuts the number of
predictions ×4.90 and the total "errors" ×2.05, yet errors *per prediction* rise 1.89 → 4.51. The paper's only
explanation is structural: "the tree height of ALEX can be reduced from 11 to 3 levels", so each lookup makes
fewer predictions, and "the numbers of predictions are also reduced" [NFL §2.2]; because the unit of "#
Prediction Errors" is undefined (a count of mispredictions and a sum of error magnitudes behave differently
when the tree gets shallower), the per-prediction ratio cannot be interpreted further [unverified: metric
definition not given in the paper]. (ii) Table 1 is measured with the flow in front of **ALEX**, not AFLI: it is
the paper's evidence that the transformation is index-agnostic, which is what §3.1 later calls "flexible"
[NFL §2.2, Table 1; §3.1].

#### 2.4 The three challenges the paper states (§2.4)

The paper's own problem statement is a list of three challenges, each answered by one section of the design
[NFL §2.4 "Challenges", p.4]:

1. **Efficacy of normalizing flow.** "Naively using NF is limited in a few ways: 1) the NF perform poorly due to
   limited features from the numerical data of keys; 2) the uniform distribution is hard to function directly as
   an training objective." Answered by the feature-space expansion (§3.2.1, our §3.5) and by the wide-Gaussian
   training target instead of the uniform (§3.2, our §3.4).
2. **Efficiency of normalizing flow.** "The transformation must be an efficient online step. Such requirement
   also limits the complexity of normalizing flows. Directly reducing the number of parameters in normalizing
   flows might degrade the transformation quality so that learned indexes require deeper hierarchy and more
   models to approximate the CDF." Answered by the inference optimizations of §3.2.2 (tiny flow, MKL, batching;
   Table 2, our §5.3).
3. **Lack of proper indexes for transformed keys.** "With the transformation of Numerical NF that fundamentally
   makes linear models approximate better, the design of learned indexes should be reconsidered in a new
   perspective ... the locality of the transformed data distribution should be considered in the design of the
   learned index." Answered by AFLI (§3.3, our §6).

So the thesis has exactly three moving parts, flow quality, flow cost, and an index shaped for the flow's output,
and the experiments in §4 are organized to test each (Table 3 for quality, Table 2 and Figure 10 for cost,
Figures 7-9 for the index).

#### 2.5 The two-stage framework (§3.1) and the batching decision

Stage 1: a normalizing flow transforms input keys to a near-uniform distribution; stage 2: a learned index is
built on the transformed keys, pos = F(z)·N [NFL §3.1, Fig. 4]. The framework is presented as "measurable"
(the transformation quality is a number, the tail conflict degree of §3.1.1) and "flexible" (any index can
consume z) [NFL §3.1].

Batching is a stated design decision: "Since batching requests (e.g., batching queries, batching insertions) is
a common case in modern database [14, 15, 28, 29, 33], our NFL also processes requests in batches" [NFL §3.1;
references 14-15 are SharedDB / Shared Workload Optimization, 28-29 Many-query join / BatchDB, 33 OLTPShare].
Why it is *needed* rather than merely allowed becomes clear in Table 2: the flow is a small dense network
evaluated with MKL matrix kernels, and the per-key cost falls from 169.53 ns at batch 1 to 8.38 ns at batch
256 [NFL Table 2] because a batch of 256 keys turns 256 tiny matrix-vector products into a few matrix-matrix
products. All throughput and latency numbers in §4 are measured at batch size 256 [NFL §4.1.3]; P99 is a
per-batch quantity divided by 256 [NFL §4.3] (details in §7).

---

### 3. Normalizing flows from first principles, then NFL's numerical flow

#### 3.1 Change of variables in one dimension

Let X be a real random variable with density p_X, and let z = f(x) be a differentiable bijection. Probability
mass is conserved under the map: the mass of a small interval [x, x + dx] equals the mass of its image
[f(x), f(x) + f′(x)dx], so

```
p_X(x) · dx = p_Z(f(x)) · |f′(x)| · dx     ⇒     p_X(x) = p_Z(f(x)) · |df/dx|
```

The factor |df/dx| (in d dimensions: |det ∂f/∂x|, the absolute Jacobian determinant) is the local stretch of
the map. NFL writes the same identity from the generator's side, with G_θ: z → x and x = G_θ(z):

```
p_G(x) = | ∂G_θ(z)/∂z |⁻¹ · p(z)                                    [NFL eq. (4), §2.3]
```

i.e. |df/dx| = |dG_θ/dz|⁻¹ because f = G_θ⁻¹. The paper notes that this determinant "must be enough cheap to
compute, otherwise the NF might introduce non-negligible overhead" [NFL §2.3].

Worked example (script section D). Take X ~ Exp(1), p_X(x) = e⁻ˣ, and the map z = f(x) = 1 − e⁻ˣ (the CDF
itself). Then f′(x) = e⁻ˣ and p_Z ≡ 1 on [0, 1]:

| x | z = 1 − e⁻ˣ | dz/dx = e⁻ˣ | p_Z(z)·dz/dx | p_X(x) = e⁻ˣ |
|---|---|---|---|---|
| 0.1 | 0.09516 | 0.90484 | 0.90484 | 0.90484 |
| 0.5 | 0.39347 | 0.60653 | 0.60653 | 0.60653 |
| 2.0 | 0.86466 | 0.13534 | 0.13534 | 0.13534 |

This is the whole idea in miniature: **the CDF of the keys is the transform that makes them uniform**, and a
learned index is a CDF approximator, so "learn a flow that flattens the keys" and "learn the CDF" are the same
task seen from two sides. The flow is a smooth, parametric, invertible stand-in for the empirical CDF, and the
metric that matters is not likelihood but how few keys collide under one linear model afterwards (§4).

#### 3.2 Maximum-likelihood training

Because the density of the model is available in closed form through the change of variables, a flow is trained
by maximizing the log-likelihood of the observed keys, equivalently minimizing a KL divergence:

```
G* = argmax_G  E_{x ~ p_data} [ log p_G(x) ]                        [NFL eq. (2)]
G* = argmin_G  KL( p_G ‖ p_data )                                   [NFL eq. (3)]
```

Substituting eq. (4) in 1-D gives the per-key training loss actually optimized (with a Gaussian base density
p(z) = N(z; 0, σ²)):

```
−log p_G(x) = −log p(z) − log |dz/dx|
            = z²/(2σ²) + ½ log(2πσ²) − log |dz/dx|        with z = f_θ(x)
```

The first term pulls transformed keys toward the centre of the base density; the second (the log-Jacobian)
rewards stretching regions where the keys are dense and penalizes squeezing them, which is what flattens the
distribution. Example: z = 0.3 and dz/dx = 2 under N(0,1) gives 0.5·0.09 − ln 2 + ½ ln 2π = 0.27079 (script
section D). The paper's trainer is the PyTorch B-NAF code, with the base variance σ² = 10¹⁶ [NFL §4.1.3].

Our stand-in trainer optimizes a *related but not identical* 1-D objective with Adam and analytic gradients
[train_flow.py:48-68, `loss_and_grad`]. Per key, with d = dz/dx in normalized-x units:

```
if d > 0:   L = 0.5·z² − log(d) + barrier/d          (base density N(0, 1): σ = 1, constant ½·log 2π dropped,
                                                     the constant log var from dz/dkey = d/var dropped,
                                                     positivity barrier added; barrier default 1e-3)
else:       L = 0.5·z² + 50 − 10·d                   (linear push-back when the sign of dz/dx flips)
```

[train_flow.py:54-55; defaults lr 0.05, barrier 1e-3, Adam β = (0.9, 0.999), ε = 1e-8 at :98, :121-122]. The
two differences that matter: the paper's base variance is 10¹⁶ and ours is 1 (so our z² term is of order 1 and
pulls z toward 0, whereas the paper's is negligible, §3.4), and ours adds a barrier that enforces dz/dx > 0.
The best per-key NLL values reached are 3.38 (planet) to 4.32 (genome) [results/aidb/flows/training_report.json,
`best_nll`]; they are not comparable with any number in the paper (which prints no likelihoods).

#### 3.3 Why invertible and monotone matter

* **Invertibility** is what makes the density computable (eq. 4 needs G⁻¹ and the Jacobian) and what lets the
  same trained object be used in both directions: "Once we have a well-trained G, G⁻¹ could be obtained easily
  by taking the inverse. Then, G⁻¹ could be used to encode the original key xᵢ to the ideal key zᵢ" [NFL §2.3].
  For an index, only the encoding direction x → z is ever executed online.
* **Monotonicity** is the 1-D form of invertibility: a strictly increasing f is a bijection with f′ > 0, so the
  Jacobian is simply f′(x) and its log is well defined. Monotone neural networks (NAF [16], B-NAF [2]) are the
  flow family NFL picks precisely because "such a coupling method [RealNVP-style] requires the dimension of
  input keys to be at least greater than 2, which is not suitable in our problem" [NFL §5.2].
* For the *index* monotonicity has a second, independent meaning: if z(x) is increasing then the transformed
  keys keep the original order, range scans on x are range scans on z, and the record array can stay in key
  order. §3.7 explains that NFL's full pipeline (feature expansion + sum decoder) does **not** guarantee this.

#### 3.4 The flow family and the exact deployed architecture

| Aspect | What the paper states | Source |
|---|---|---|
| Family | "a modified B-NAF", i.e. Block Neural Autoregressive Flow (De Cao, Aziz, Titov, UAI 2019, ref. [2]); B-NAF "improves the structure of NAF by using a single feed-forward network to model the bijections" | [NFL §4.1.3], [NFL §5.2] |
| What "modified" means | not reported | — |
| Input dimension | 2 ("two input dimensions") | [NFL §4.1.3] |
| Hidden units | 2 ("two hidden dimensions") | [NFL §4.1.3] |
| Layers | 2 ("two layers"); Table 2 labels this shape "2H2L (8)" = 8 parameters | [NFL §4.1.3], [NFL Table 2] |
| Parameter count | 8 = 2×2 (input→hidden) + 2×2 (hidden→output), no biases; check: 2H4L = 4·(2×2) = 16, 4H3L = 2·4 + 4·4 + 4·2 = 32, 4H4L = 2·4 + 4·4 + 4·4 + 4·2 = 48, all matching the bracketed counts in Table 2 (script section E). The bracketed 8 is the only paper-side evidence for "no bias": with biases 2H2L would have 2·2+2 + 2·2+2 = 12 parameters (script 2, section O). The Introduction's complaint that "existing normalizing flows are of too high complexity (e.g., 4 layers, 16 parameters)" is exactly Table 2's 2H4L (16) column, so the 8-parameter net is the paper's deliberate minimum | [NFL Table 2], [NFL §1 p.1], [transform.hpp:8-9] |
| Activation | not named in the paper; its only hint is §3.2.2: "The computations in inference can be simplified as several matrix computations and nonlinear function computations" (so a nonlinearity exists, its type is not stated). The shipped weight file is evaluated as "a small tanh network" by the official C++ side, which is what our reader replicates (tanh after every layer except the last, then sum) | [NFL §3.2.2 p.6], [transform.hpp:6-10, 64-79] [unverified: tanh read from our code comment describing the official repository, not from the PDF] |
| How positivity/monotonicity is enforced | not stated in the paper. In B-NAF as published, the weight matrices are block lower-triangular, the diagonal blocks are made strictly positive by an element-wise exponential, and tanh is used between layers, which makes each output a strictly increasing function of its own input dimension and gives a closed-form log-Jacobian [unverified: from the B-NAF paper (ref. [2]), which is not among our sources]. Two facts about the *deployed* file: (a) it holds two full, unmasked 2×2 matrices; our reader checks only their shapes [transform.hpp:43-51] and evaluates them as a plain dense tanh net [transform.hpp:71-78], and `fb_2D2H2L.txt` shows a dense W0 with a non-zero first row and a zero second row, so whatever block-triangular/positive-diagonal structure B-NAF has during training is not visible in the exported format [unverified: whether the official exporter bakes masks and the exponential into the stored values]; (b) the general Jacobian of the deployed net with respect to the key is given in §3.8 and holds whether or not the fractional row is zero | [transform.hpp:43-51, 71-78], [results/aidb/flows/fb_2D2H2L.txt] |
| Base ("latent"/target) distribution | "a normal distribution with the variance of 10¹⁶" — i.e. the training objective is not the uniform (which "might encounter the Nan-loss issue or the INF-loss issue") but a Gaussian so wide that it is flat over the range of the keys | [NFL §3.2], [NFL §4.1.3] |
| Training data | "We only sample 10% bulk-loaded keys for three times to train the NF" (10 % of the 100M bulk-loaded keys; "three times" is not explained further) | [NFL §4.1.3] |
| Training hardware/time | NVIDIA GeForce RTX 3080 (10 GB), 64 GB main memory; "about 38 seconds"; done offline, once, "when the index bulk loads for the first time or when the distribution significantly shifts" | [NFL §4.1.3], [NFL §3.2.2] |
| Inference platform | C++ with Intel Math Kernel Library; "several matrix computations and nonlinear function computations" | [NFL §3.2.2], [NFL §4.1.3] |
| Weight file | plain-text "2D2H2L" export: header `in_dim hidden layers`, then `mean var`, then each matrix as `rows cols` followed by its rows | [transform.hpp:35-51] (our reader of the official format) |

Why the variance 10¹⁶ is a "near-uniform" target: a Gaussian N(0, σ²) with σ = 10⁸ has a density that changes by
a factor exp(−(z²)/(2σ²)); over any window of transformed keys whose width is small against 10⁸ it is flat to
first order, so maximizing likelihood under it behaves like maximizing likelihood under a uniform while keeping
log p(z) finite everywhere (the uniform has −∞ log-density outside its support, which is the "INF-loss" the
paper mentions) [NFL §3.2].

The number that makes "flat" concrete (script 2, section K): with σ² = 10¹⁶ the data term of the loss in §3.2 is
z²/(2σ²) ≤ 10¹²/(2·10¹⁶) = 5×10⁻⁵ for every |z| ≤ 10⁶, and 5×10⁻¹¹ for |z| ≤ 10³, i.e. negligible against the
log-Jacobian term, whose magnitude is of order 1. The objective therefore reduces to maximizing Σᵢ log|dz/dx|(xᵢ),
"stretch the key axis where the keys are dense", which is the CDF-learning reading of §3.1. Contrast with our
stand-in trainer, where σ = 1: for z of order 1 the data term is ≈0.5, of the same order as −log|dz/dx|, so our
objective *also* pulls z toward 0, and the best per-key NLL it reaches is 3.38 (planet) … 4.32 (genome)
[results/aidb/flows/training_report.json, `best_nll`; train_flow.py:48-55] (see §3.2 and §8).

#### 3.5 Feature space expansion (§3.2.1) and Algorithm 3.1 line by line

The paper's argument for expanding a scalar key into a vector: image/text flows learn from rich features, but
"the data of keys are numerical data with hardly any useful knowledge", and because keys are unique, feeding more
keys "would increase the burden of an NF rather than help training"; so the design "enrich[es] the features that
an NF could learn from, while maintaining high efficiency" [NFL §3.2.1]. The expansion is a brute-force,
1-to-1 map from a 1-D key to a d-dimensional vector in O(n·d) [NFL §3.2.1].

Algorithm 3.1 "Key Distribution Transformation" [NFL p.6], input: sorted keys X = {x₁..xₙ}, target dimension d,
scale factor θ, flow F; output: transformed keys Z:

```
 1: X_d = ∅
 2: μ = min(X);  σ = (max(X) − min(X)) / θ            ← scaled min-max normalization
 3: for xᵢ ∈ X do
 4:    x_norm = (xᵢ − μ) / σ                            ← x_norm ∈ [0, θ]
        /* Encoder, expand features for NF */
 5:    x_vec = []
 6:    x_int   = INT(xᵢ)                                ← integral part
 7:    x_float = xᵢ − INT(xᵢ)                           ← fractional part
 8:    add x_int to x_vec
 9:    for k from 1 to d − 2 do
10:        add x_int to x_vec
11:        x_float = x_float · θ                        ← shift one "digit" (base θ) of the fraction up
12:        x_int   = INT(x_float)
13:        x_float = x_float − x_int
14:    end for
15:    add x_float to x_vec                             ← the last fractional remainder
16:    append x_vec to X_d
17: end for
18: Z_d = F(X_d)                                        ← the flow, applied to the whole batch
        /* Decoder, merge features for index */
19: for z_vec ∈ Z_d do
20:    zᵢ = sum of z_vec                                ← decoder = sum of the d outputs
21:    add zᵢ to Z
22: end for
23: return Z for the index
```

Reading notes (all mine, flagged where the print is ambiguous):

* Line 2 divides the min-max range by θ so that x_norm spans [0, θ]; the paper says this is "to avoid keys
  without integral or floating part" [NFL §3.2.1]: an integer key has no fractional part and a key in [0,1) has
  no integral part, and the expansion needs both. The only value of θ the paper gives is the symbolic "scale
  factor"; the official trainer's default for 200M keys is 10⁶ according to our trainer's help text
  [train_flow.py, `--shifts` help: "author default 1e6 for 200M keys"] [unverified: not in the PDF].
* **Keys outside [min(X), max(X)] of the bulk load.** μ and σ are fixed at bulk load (line 2), so a queried or
  inserted key below min(X) gives x_norm < 0 and one above max(X) gives x_norm > θ; the integral part is then
  negative or larger than θ. The paper never discusses this case and avoids it experimentally: "all inserted
  keys are in the key space consisting of all bulk-loaded keys, which means all insertions are known-key-space
  insertions" [NFL §4.1.1]; behaviour for out-of-range keys: **not reported**. Our code: `transform.hpp:66`
  normalizes with the stored mean/var without clamping, tanh saturates gracefully, and a non-finite z is
  replaced by 0 [transform.hpp:79]; the trainer sets mean = first training key and var = (last − first)/shifts
  [train_flow.py:89], so keys of the full sample outside the *training subsample's* range already exercise
  this path (fb: 408 keys give x < 0 and 102 give x > 64, script 2, section N).
* Lines 6-7 as printed use xᵢ, not x_norm. Line 4 computes x_norm and never uses it otherwise, so the intended
  input to lines 6-7 is almost certainly x_norm [unverified: reading of an apparent typo].
* Line 10 as printed adds the *same* x_int again before computing the next digit; the natural intent is to add
  the *new* integer digit produced by line 12 each round (so the vector is [integer part, digit₁, digit₂, …,
  remainder]). The script shows both readings (section B); with d = 2 the loop is empty and both agree.
* **Why the fractional part is added**: the integral part alone is a step function of the key (all keys in one
  unit interval share it) and would destroy uniqueness; the fractional part restores the 1-to-1 property (line
  15 always appends the remaining fraction, so [x_int, digits…, x_float] determines xᵢ exactly). At the same
  time each fractional feature is a *periodic* (sawtooth) function of the key with period σ/θᵏ, which gives the
  network a feature that "wraps" the key space and lets a 2-unit tanh layer express local, repeating structure
  it could not express from a single monotone input. The cost is that the map key → features is not monotone
  (§3.7).

Worked example (script section B). X = {1000, 1500, 2500, 4000}, θ = 10: μ = 1000, σ = 300, x_norm = 0,
1.6667, 5, 10.

| key | x_norm | d = 2: [INT, frac] | d = 4, digit reading: [INT, digit₁, digit₂, remainder] |
|---|---|---|---|
| 1000 | 0.0000 | [0, 0.0] | [0, 0, 0, 0.0] |
| 1500 | 1.6667 | [1, 0.6667] | [1, 6, 6, 0.6667] |
| 2500 | 5.0000 | [5, 0.0] | [5, 0, 0, 0.0] |
| 4000 | 10.0000 | [10, 0.0] | [10, 0, 0, 0.0] |

and for x = 2.71828 with θ = 10, d = 4: [2, 7, 1, 0.828] — the decimal digits of the normalized key.

The deployed shape uses d = 2 ("two input dimensions" [NFL §4.1.3]), so at inference the vector is two numbers.
Our reader encodes [x, x − ⌊x⌋] (the *whole* normalized key plus its fraction) rather than [⌊x⌋, x − ⌊x⌋]
[transform.hpp:68-69]; the header comment says this is "identical to the original 'partition' encoder"
[transform.hpp:68] [unverified: the official encoder was not read for this guide; the difference from the
printed Algorithm 3.1 is that the first feature is x, not INT(x)].

#### 3.6 The decoder

Line 20 sums the d outputs of the flow into one scalar zᵢ. Nothing in the paper motivates the sum beyond
"merge the high-dimensional features into 1D keys" [NFL §3.2.1]. Two consequences: (i) the sum is not itself a
bijection of ℝᵈ → ℝ, so uniqueness of z is not guaranteed by construction, only empirically; (ii) the
log-Jacobian used in training is that of the d-dimensional flow F, not of the composed scalar map key → z. For
d = 2 with a final linear layer, the sum decoder folds the two output columns into a single effective output
weight per hidden unit, v_h = w1[h,0] + w1[h,1] [train_flow.py, `forward`], which is how our trainer and reader
treat it.

#### 3.7 What happens if the composed transform is not monotone

Each coordinate of a B-NAF is increasing in its own input, but the *composition* key → [⌊x⌋, frac(x)] → F → sum
is not monotone in the key: the fractional feature jumps from ≈1 back to 0 at every integer of x_norm, so z can
decrease there. NFL's answer is to treat z as *the* key: Algorithm 3.1 returns "Z for the index"; AFLI's
Modelling takes "an array of sorted key-payload pairs" [NFL Alg. 3.2 input], and its lookups compare "the
stored key with the queried key" after feeding "the queried key into the linear model" [NFL §3.3.2]. Reading
those three statements together, the index is built on z-values sorted by z, a lookup transforms the query and
searches for its z, and correctness of point lookups needs only that z be injective on the data, not monotone
[unverified: the paper never says explicitly whether stored keys are x or z; this is the only consistent
reading]. Range scans over original keys would be broken by a non-monotone z, and the paper evaluates none:
the workloads are lookups and insertions only [NFL §4.1.1]. Our control avoids the question by keeping records
in original key order and using z only as the model's feature: "Records stay in original key order, so exact
search, fences and scans are unaffected even when the network is not monotone. NFL instead sorts by transformed
key" [transform.hpp:14-16]; §8.

#### 3.8 Worked forward pass of a real 2D2H2L weight file

The file `results/aidb/flows/fb_2D2H2L.txt` (trained by our stand-in trainer on 4,096 keys sampled from the 2M
uniform fb sample [results/aidb/flows/fb_training.json]) is:

```
2   2   2                       ← in_dim=2, hidden=2, layers=2
15162980.0  1207648189.703125   ← mean = min, var = (max − min)/64 of the 4,096-key TRAINING SUBSAMPLE [train_flow.py:89, :128]
2 2
0.0058556031582764  0.0058592806847908     ← W0 row 0: weights of the x feature to hidden units 1, 2
0.0000000000000000  0.0000000000000000     ← W0 row 1: weights of the fractional feature (zero: --monotone)
2 2
1.2798795760698392  0.8812895293543010     ← W1 row 0: hidden unit 1 to the two outputs
1.4785700705830542  1.0331745748154413     ← W1 row 1: hidden unit 2 to the two outputs
```

Precision note on the header line: `mean` and `var` are the minimum and (max − min)/64 of the 4,096 keys drawn
with seed 1000000007 from the 2M sample [train_flow.py:87-89, :128], **not** of the 2M sample itself. Recomputed
(script 2, section N): the full `fb_2M_uniform_s42` file has min 97,995 and max 77,308,811,965; the subsample
has min 15,162,980 and max 77,304,647,121, which is exactly what the file stores (var = 1,207,648,189.703125).
Consequently the rows labelled "subsample min/max" below are the training-subsample extremes; 408 real keys
lie below 15,162,980 (x < 0, down to x = −0.012475) and 102 above 77,304,647,121 (x > 64, up to 64.003449),
harmless here only because the fractional-feature row is zero and tanh is defined everywhere.

Evaluation [transform.hpp:64-79]: x = (key − mean)/var; a = [x, x − ⌊x⌋]; u = a·W0; h = tanh(u); out = h·W1;
z = out₀ + out₁. The general Jacobian with respect to the key, valid also when the fractional row is non-zero
(this is the formula the trainer differentiates [train_flow.py:36-39]): with u_h = x·W0[0,h] + frac(x)·W0[1,h],
h_h = tanh u_h and v_h = W1[h,0] + W1[h,1] (the sum decoder folds the two output columns),

```
dz/dx = Σ_h (1 − h_h²) · (W0[0,h] + W0[1,h]) · v_h        almost everywhere (d frac(x)/dx = 1 between integers)
dz/dkey = (dz/dx) / var
jump at each integer x = m:  z(m⁺) − z(m⁻) = − Σ_h v_h · ( tanh(m·W0[0,h] + W0[1,h]) − tanh(m·W0[0,h]) )
```

(checked against a central finite difference to 1.5×10⁻⁹, script 2, section J). The jump term is what makes
the composed map non-monotone when W0[1,:] ≠ 0 (§3.7). Because the fractional row of *this* file is zero, z
depends on x only through x·W0[0,:], the jump vanishes, and dz/dx = Σ_h (1 − h_h²)·W0[0,h]·v_h > 0: this
particular file is monotone (script section C):

| key | x | u = (0.005856x, 0.005859x) | h = tanh u | out | z | dz/dx |
|---|---|---|---|---|---|---|
| 15,162,980 (subsample min) | 0 | (0, 0) | (0, 0) | (0, 0) | 0.00000 | 0.02737 |
| 19,337,534,015 | 16 | (0.09369, 0.09375) | (0.09342, 0.09347) | (0.25777, 0.17890) | 0.43667 | 0.02713 |
| 38,659,905,050 | 32 | (0.18738, 0.18750) | (0.18522, 0.18533) | (0.51108, 0.35471) | 0.86579 | 0.02643 |
| 57,982,276,086 | 48 | (0.28107, 0.28125) | (0.27389, 0.27406) | (0.75576, 0.52453) | 1.28029 | 0.02532 |
| 77,304,647,121 (subsample max) | 64 | (0.37476, 0.37499) | (0.35815, 0.35835) | (0.98823, 0.68587) | 1.67411 | 0.02386 |

Over 65 grid points the largest deviation of z from the straight chord between its endpoints is 0.02936, i.e.
1.75 % of the z-range: with 8 weights and an unsaturated tanh (|u| ≤ 0.375) this transform is very nearly
affine in the key. That is the capacity limit behind the paper's own remark that "the transformation quality of
the NF has almost reached to its upper limit" in their setting [NFL §4.4.1] and behind our observation that the
flow feature never wins inside a 4,096-key region (§8).

The table above only uses integer x (16, 32, 48, 64), where frac(x) = 0, and the fractional weights are zero
anyway, so it never shows the non-monotone case of §3.7 and Q2. Here is that case, computed (script 2, section
I) with the same file except a **hypothetical** fractional row W0[1,:] = [0.5, 0.5] (what `--monotone` would
have left free):

| x | key | frac(x) | u = (u₁, u₂) | z, real file (row 1 = 0) | z, hypothetical row 1 = [0.5, 0.5] | dz/dx (hyp., a.e.) |
|---|---|---|---|---|---|---|
| 15.90 | 19,216,769,196 | 0.90 | (0.54310, 0.54316) | 0.43396 | 2.31476 | 1.78379 |
| 15.99 | 19,325,457,533 | 0.99 | (0.58863, 0.58869) | 0.43640 | 2.47166 | 1.70250 |
| 16.00 | 19,337,534,015 | 0.00 | (0.09369, 0.09375) | 0.43667 | **0.43667** | 2.34319 |
| 16.01 | 19,349,610,497 | 0.01 | (0.09875, 0.09881) | 0.43694 | 0.46009 | 2.34091 |
| 16.50 | 19,941,358,110 | 0.50 | (0.34662, 0.34668) | 0.45024 | 1.55796 | 2.10107 |

Between integers z rises steeply (dz/dx ≈ 1.7-2.3 instead of 0.027), and at x = 16 it drops from 2.47166 back
to 0.43667: a jump of −2.05196, equal to the closed-form jump term above evaluated at m = 16. The map key → z
is a sawtooth with period var = 1.2×10⁹ key units, so ≈2.5 % of the fb key pairs would be re-ordered on every
tooth. That is exactly why `train_flow.py` has `--monotone` (it zeroes W0[1,:] at initialization and keeps the
gradient of those two weights at zero [train_flow.py:67, :96]) and why every `*_training.json` reports
`unordered_transformed_pairs = 0` with `monotone: true` [results/aidb/flows/fb_training.json; train_flow.py:133].
NFL itself keeps those weights ("deviation from NFL, which keeps them" [train_flow.py:124]) and lives with the
sawtooth by sorting on z (§3.7).

---

### 4. Conflict degree and tail conflict degree (Definitions 3.1 and 3.2)

Motivation: "the log probability of the transformed distribution in the NF cannot accurately evaluate how nearly
uniform the distribution should be for the learned index"; and because AFLI places keys at their predicted
positions ("placing data in the predicted positions can eliminate prediction errors"), what matters is how many
keys land on the same position [NFL §3.1.1].

**Definition 3.1 (conflict degree of a position).** For a key set X = {x₁..xₙ} and a fitted linear model M,

```
D_j^M = | { xᵢ ∈ X : M(xᵢ) == j } |        for j from MIN({M(xᵢ)}) to MAX({M(xᵢ)})      [NFL eq. (5)]
```

M(xᵢ) is the integer predicted position (Algorithm 3.2 says the "rounding operation rounds them to the same
integer" [NFL §3.3.2, after Alg. 3.2]); D_j counts the keys sharing position j.

**Definition 3.2 (tail conflict degree).** Let m be the number of positions with D_j > 0, γ the tail percent,
t = INT(m × γ) (floor). The tail conflict degree D_γ^M is "the t-th larger conflict degree among {D_j}". The paper
sets γ = 0.99 and gives the example m = 1000 ⇒ t = 990, "the 990-th larger conflict degree" [NFL §3.1.1].
"t-th larger" is ambiguous in English; the stated purpose ("the upper bound of the conflicts for most positions")
fixes it as the 99th percentile of the occupied-position conflict counts, i.e. the 990th value in *ascending*
order (the 990th *largest* would be the 11th smallest, which is no upper bound for anything). Uses: (1) the
flow on/off switch (§5); (2) the capacity threshold of buckets and dense nodes in AFLI (§6) [NFL §3.1.1].

**Three conflict degrees, not one.** A DB professor will ask how this relates to the other two "conflict
degrees" in this guide. (a) LIPP's conflict degree: "LIPP is more concerned with conflict degrees, i.e., the
number of keys predicted to the same position" [NFL §2.1 p.3]; LIPP's bulk load (FMCD) searches the smallest
bound D such that its line puts at most D keys into any slot. (b) NFL's tail conflict degree: the same count
per position under one least-squares line fitted on keys and *scaled positions*, but the statistic is the 99th
percentile over occupied positions, not the maximum [NFL Definitions 3.1-3.2]. (c) The AIDB paper's CD, which
our `hardness.json` reports: "the maximum number of keys mapped to the same rank by a linear model fitted on the
dataset using Fastest Minimum Conflict Degree (FMCD)", i.e. LIPP's fit and a *maximum* statistic, ported in
`fmcd_fit` [include/scaleli/hardness.hpp:7-8, :168; results/aidb/MEETING_NOTES.md §1 "CD via LIPP's FMCD fit"].
All three count keys per predicted position under one linear model; they differ in the fit (plain least squares
with scaled positions in NFL; FMCD's search over the slope with capacity L = size·(gap+1) in LIPP/AIDB) and in
the statistic (99th percentile vs maximum). Section 03 §3.2 gives the FMCD definition and its capacity rule.

#### 4.1 Worked example on 10 keys (script section A)

Keys X = {10, 11, 12, 13, 30, 31, 55, 80, 81, 100}, ranks 0..9. Least-squares fit of rank on key (the paper's
Modelling fits "using the input keys and scaled positions" [NFL Alg. 3.2 line 1]; we use the plain ranks here):

```
a = Sxy/Sxx = 0.084238,   b = ȳ − a·x̄ = 0.936719,   M(x) = round(a·x + b)
```

| key | rank | a·k + b | M(x) = round | 
|---|---|---|---|
| 10 | 0 | 1.779 | 2 |
| 11 | 1 | 1.863 | 2 |
| 12 | 2 | 1.948 | 2 |
| 13 | 3 | 2.032 | 2 |
| 30 | 4 | 3.464 | 3 |
| 31 | 5 | 3.548 | 4 |
| 55 | 6 | 5.570 | 6 |
| 80 | 7 | 7.676 | 8 |
| 81 | 8 | 7.760 | 8 |
| 100 | 9 | 9.361 | 9 |

Conflict degrees (Definition 3.1): D₂ = 4, D₃ = 1, D₄ = 1, D₆ = 1, D₈ = 2, D₉ = 1; positions 5 and 7 are empty
(D = 0) and are not counted in m. So m = 6 occupied positions, sorted ascending {1, 1, 1, 1, 2, 4}.

Tail conflict degree (Definition 3.2): t = INT(6 × 0.99) = INT(5.94) = 5 ⇒ D₉₉ = 5th value ascending = **2**.
The maximum conflict is 4 (the cluster 10-13), but the tail metric deliberately ignores the single worst
position: at m = 6 it discards the top 1/6 of positions, at m = 1000 the top 1 %.

The same keys under our control's convention [transform.hpp:90-106]: intercept rescaled so the first key maps to
position 0 (`intercept = −slope·x₀ + 0.5`, then floor [transform.hpp:96]), capacity clamped to
min(⌊1.5n⌋, last position + 1) = 9, positions {0,0,0,0,2,2,4,6,6,8}, counts ascending {1, 1, 2, 2, 4}, m = 5,
index ⌈0.99·5⌉ − 1 = 4 ⇒ counts[4] − 1 = **3**. The two conventions differ in three deliberate ways: our
value is *collisions* (keys minus one, so a perfectly spread set scores 0, not 1), our percentile index uses a
ceiling instead of a floor (which at m = 1000 gives the same 990th element but at tiny m picks the maximum), and
the amplification factor is fixed at 1.5 (the paper's α in Algorithm 3.2 line 7 has no stated value; **not
reported**). Section 05 uses our convention throughout; when comparing with the paper's Table 3 add one and
remember that the paper's positions are scaled ("we also scale the positions used to train linear models
according to the scaling relationship between keys" [NFL §3.3.1]), which is also unspecified.

#### 4.2 Why a "soft" 99th percentile and not the maximum

The paper calls the tail conflict degree "a soft measurement" [NFL §3.1.1]. A maximum is dominated by one
pathological cluster (e.g. many duplicates or a dense burst of timestamps), which AFLI handles with a child node
anyway (Algorithm 3.2 lines 18-22); sizing every bucket for that one cluster would waste space everywhere else.
The 99th percentile is what "most" positions need, so buckets of that size absorb almost all conflicts with a
linear scan, while the rare larger clusters recurse. Empirically the transformed data settle at D₉₉ ≈ 4 on every
dataset [NFL Table 3, §3.3], and the bucket cap is set to 6 [NFL §4.1.3].

---

### 5. The switching mechanism (§3.2.2) and Table 2

#### 5.1 The exact rule

"The NFL first tries to transform the input keys, and computes the tail conflict degree based on the input keys
and the transformed keys, respectively. If the latter tail conflict is larger, the NFL determines not to use the
Numerical NF" [NFL §3.2.2]. In symbols, with one linear model fitted on each feature at bulk load:

```
use_flow = NOT ( D₉₉(z) > D₉₉(x) )        i.e. keep the flow unless it makes the tail conflict degree worse
```

What the paper states about the granularity: the switch compares the tail conflict degrees of "the input keys"
and "the transformed keys" [NFL §3.2.2], and §4.2 reports **one on/off outcome per dataset** (YCSB, AMZN, WIKI
off; the other four on). That the decision is taken once per bulk load for the whole dataset, and that it is
all-or-nothing, is this guide's inference from those two passages together with the separate sentence that
"The transformed keys cannot be stored, as it would cause another indexing problem" (a statement about online
inference, not about the switch) [unverified: inferred from §3.2.2 + §4.2; the paper does not say "once per bulk
load"]. Motivation: "keys in some datasets are already near-uniform distributed, it is unnecessary to spend
extra time and memory to transform them" [NFL §3.2.2].

What the paper does **not** say about the switch, each **not reported**: (i) on which keys the two tail conflict
degrees are computed, all ≈100M bulk-loaded keys or the 10 % training sample; (ii) which linear model is used,
presumably one global fit over all keys, i.e. the root model of Algorithm 3.2 line 1 with "scaled positions",
but the text does not say; (iii) whether the comparison is strict: §3.2.2 says "If the latter tail conflict is
larger", which the AMZN row of Table 3 (4 → 4, yet disabled) contradicts (§5.2). Contrast with our control,
where both values are computed **per 4,096-key region on the region's own keys** with a plain rank fit
(intercept rescaled so the first key maps to 0) and a 10 % margin [index.hpp:235-237, :248, :280-282;
transform.hpp:90-106].

#### 5.2 Which datasets end up without the flow, and why

"Since transforming the workloads with small conflict degrees (i.e., YCSB, AMZN, WIKI) increases the tail
conflict degree, our NFL determines not to use NF" [NFL §4.2]; "On YCSB, AMZN, and WIKI, the NFL disables the NF
due to that the distribution transformation does not reduce the tail conflict degree, so the NFL achieves almost
the same performance as the proposed AFLI" [NFL §4.2]. The measured values are Table 3 [NFL p.11] ("(L)" after
the loading phase, "(R)" after the running phase; the second pair of rows is after the transformation):

| | LTD | LLT | LGN | YCSB | AMZN | FB | WIKI |
|---|---|---|---|---|---|---|---|
| Tail (L), raw | 8 | 146 | 14 | 3 | 4 | 386 | 2 |
| Tail (R), raw | 7 | 147 | 13 | 3 | 4 | 454 | 1 |
| Tail (L), after NF | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| Tail (R), after NF | 4 | 5 | 4 | 4 | 4 | 4 | 4 |

Reading (script section G): the flow compresses the tail to 4 whatever the input (LLT 146 → 4, ×36.5; FB 386 →
4, ×96.5; LTD 8 → 4; LGN 14 → 4), and *raises* it where the raw keys were already better than 4 (YCSB 3 → 4,
WIKI 2 → 4). AMZN is 4 → 4: under the literal rule ("if the latter is larger") the flow would be *kept* on AMZN,
yet §4.2 lists AMZN among the disabled datasets; either the rule is applied as "not strictly smaller" or the
printed values are rounded [unverified: inconsistency between §3.2.2's rule and §4.2's list for AMZN; the
paper does not resolve it]. Note also that after the running phase the raw tail on FB grows 386 → 454 while the
transformed tail stays 4. Table 3's caption only says "(R)" is "after the running phase"; it does not say which
workload produced those rows (read-heavy, write-heavy and write-only insert different numbers of keys), so
"~100M insertions" is taken from §4.4.1's prose, not from the table: **workload for (R) not reported** [NFL Table
3 caption p.11; §4.4.1: "after inserting around 100 million new keys, the tail conflict degree is around 4"] — the flow generalizes to keys it never saw because inserts are drawn from the same key space
("known-key-space insertions" [NFL §4.1.1]). The paper is candid that on low-conflict data "the NF increases the
conflict degrees. This is due to the limitation of NF ... the transformation quality of the NF has almost reached
to its upper limit" [NFL §4.4.1].

#### 5.3 Table 2 in full: transformation latency per key

Caption, verbatim: "Average transformation latency of each key with different NFs. "H" and "L" correspond
to the hidden dimension and the number of layers, respectively. The figure in brackets (e.g., "(12)")
represents the amount of parameters of NF. All latency is measured in nanosecond (ns)." [NFL Table 2, p.6]:

| Batch size | 2H2L (8) | 2H4L (16) | 4H3L (32) | 4H4L (48) |
|---|---|---|---|---|
| 1 | 169.53 | 384.84 | 320.15 | 463.81 |
| 8 | 40.60 | 83.05 | 77.52 | 113.31 |
| 32 | 15.28 | 34.75 | 33.80 | 49.40 |
| 128 | 9.52 | 24.00 | 24.91 | 36.93 |
| 256 | 8.38 | 21.66 | 23.52 | 35.07 |
| 1024 | 7.40 | 19.81 | 22.21 | 33.13 |
| 2048 | 7.29 | 19.63 | 22.00 | 32.73 |

How to read it (script section E):

* The deployed 2H2L flow costs **169.53 ns per key at batch 1 and 8.38 ns at batch 256**, a 20.2× reduction;
  going to batch 2048 only lowers the per-key cost by a further 13 % (8.38 → 7.29 ns, a 1.15× ratio; 23.3× in
  total vs batch 1: 169.53/7.29 = 23.26, script 2, section O). The knee is between 32 and 128.
* At batch 1 the transform alone caps throughput at 1e9/169.53 = 5.90 Mops/s, which is far below the scale of
  Figure 7 (y-axes up to 45 Mops/s in panel (a) and 40 in panel (e)); at batch 256 the cap is 119 Mops/s. The
  paper prints no numeric baseline throughputs, only ratios, so "below every baseline" cannot be checked from
  the PDF [unverified: Figure 7's axis maxima bound the tallest bar, they do not give the baselines' values;
  NFL p.9 Fig. 7 axis ticks; §4.2 gives only 2.34×, 2.46×, 3.82×, 7.45×]. This single row is the quantitative
  reason NFL must batch (§2.5). Two separate numbers describe our unbatched control: the **measured** cost of one
  transform on the lookup path is 66 ns [56, 77] [results/aidb/MEETING_NOTES.md §3, pooled pre-fix data], and
  what the fused selector **charges** is c_flow = 4 probe-equivalents per lookup (`--flow-cost 4`, the default,
  recorded as `flow_cost: 4` in every run) added to the in-sample probe count [src/benchmark.cpp:179, :191;
  index.hpp:264; results/aidb/sweep/results.jsonl `flow_cost`]; with either accounting the selector never finds
  the flow worthwhile (§8). For the batching comparison at build time: our build loop transforms the 2M fb keys
  in 59,765,592 ns = 29.88 ns/key [results.jsonl, fb packed_rank_flow, `learnability.transform_ns`;
  summary.csv `preprocess_ns_per_key` 29.882796; MEETING_NOTES §4 "27-34 ns/key"], versus NFL's 8.38 ns at
  batch 256 and the ≈103 ns/key implied by its bulk-load figure (below).
* Larger flows are 2.6-4.2× more expensive at batch 256 (2H4L 21.66, 4H3L 23.52, 4H4L 35.07 ns) and, per the
  paper, do not lower the conflict degree enough to pay for it: "we slightly search for the parameters without
  increasing the conflict degrees and with a low search cost. As Table 2 shows, the search space is small limited
  by the unacceptable transformation overheads of larger NFs" [NFL §3.2.2].
* The four factors the paper lists for inference cost are input dimension, number of layers, hidden dimension and
  implementation platform (MKL) [NFL §3.2.2].

A cross-check that the paper does not make: bulk loading 100M keys takes 13.42 s on average of which 77 % is
transformation [NFL §4.4.2] ⇒ 10.33 s ≈ 103 ns per key at bulk load, versus 0.84 s if it ran at the 8.38 ns of
Table 2 (script section H). The bulk-load figure presumably includes the switch (transform + two tail-conflict
computations) and non-batched paths; the paper does not reconcile the two [unverified: my arithmetic on two
independent statements]. A second caveat: 13.42 s and 77 % are averages whose scope is not stated. If they
average over all seven datasets including the three where the switch disables the flow (YCSB, AMZN, WIKI [NFL
§4.2]), the per-key transform cost on the four flow-on datasets is even higher than 103 ns; if instead the switch
still transforms every key once to compute D₉₉(z) ("first tries to transform the input keys" [NFL §3.2.2]), the
77 % includes that one pass on every dataset. Keep the arithmetic, but label both readings [unverified: scope of
the average not stated].

---

### 6. AFLI: the After-Flow Learned Index (§3.3)

#### 6.1 Design premise

After the flow, "the conflict degrees can be kept around a low value (e.g., around 4 for the tail conflict
degree)" on every dataset, so the index no longer needs the large, empirically tuned node capacities that
existing indexes carry to survive high-conflict data [NFL §3.3]. AFLI's "main idea is to buffer local conflicts":
a few colliding keys go into a small bucket rather than a new model, because a linear model fitted on 4 keys
"lacks generalization capability" [NFL §3.3, citing LIPP]; but buckets must stay tiny or scanning them dominates.
The trade-off knob is the tail conflict degree.

#### 6.2 Node types (§3.3.1, Fig. 5)

* **Model node** = one linear model + an array of entries. Entry types: *empty slot*; *data slot* (one
  key-payload pair, placed exactly at its predicted position); *bucket pointer*; *node pointer* (to a child
  model node or dense node). Several adjacent positions may hold the *same* child address ("duplicated node
  pointer"): keys predicted to adjacent positions are sent to one child "for better generalization capability of
  the linear model" [NFL §3.3.1]. Two bitmaps record the entry type [NFL §3.3.2, Queries].
* **Bucket** = a short unsorted data array ("linear bucket", default) or sorted ("ordered bucket"). Its maximum
  size "is determined by the tail conflict degree D_γ^M, but will be kept within a preset threshold range"; the
  experiments cap it at 6 [NFL §3.3.1, §4.1.3].
* **Dense node** = an ordered, gapped data array, larger than a bucket, much smaller than a model node, with at
  most D_γ^M gaps. Gaps are not tracked by a bitmap: each gap is filled with a copy of the key just before it, so
  "empty" is detected by comparing adjacent slots [NFL §3.3.1].

Analysis in the paper's words: most nodes are model nodes with *precise* placement ("the predictions are all
precise in the model nodes"); locally too-close keys become buckets; when a node's keys are so close that the
fitted slope is 0 (all map to one position) the index "does not further partition the key space but allocates a
dense node" [NFL §3.3.1]. Buckets and dense nodes exist "to buffer keys without increasing the height of the
index" until they are either distinguishable (dense node) or numerous enough to fit a model (bucket).

#### 6.3 Bulk load and Modelling (Algorithm 3.2)

BulkLoad "first computes the tail conflict degree D_γ^M_L, then follows the same procedure as the modelling
operation in Algorithm 3.2. The returned result is the root node" [NFL §3.3.2, More Operations]. Algorithm 3.2
Modelling({⟨x₁,v₁⟩..⟨xₙ,vₙ⟩}, 𝔫), inputs: sorted key-payload pairs, the node pointer 𝔫, the space amplification
factor α [NFL p.8]:

```
 1: build a linear model M_𝔫 from the keys and SCALED positions
 2: if M_𝔫.a == 0 or all keys map to one position then            ← keys too close to separate
 3:     𝔫.size = n + D_γ^M_L
 4:     allocate 𝔫.E of that size; insert all pairs evenly gapped with D_γ^M_L gaps in total   ← DENSE NODE
 5: else
 6:     compute the conflict degree D[pos] of every predicted position
 7:     𝔫.size = MIN( ⌊n·α⌋ , pos_last − pos_first + 1 )              ← capacity: at most α n slots
 8:     allocate 𝔫.E of that size
 9:     i = 0
10:     for pos in all predicted positions do
11:         if D[pos] == 1 then                                    ← exactly one key: DATA SLOT
12:             𝔫.E[pos] = ⟨xᵢ, vᵢ⟩
13:             i = i + 1
14:         else if D[pos] < D_γ^M_L then                          ← small conflict: BUCKET
15:             build a bucket 𝔟 of max size D_γ^M_L storing {⟨xᵢ,vᵢ⟩, …, ⟨x_{i+D[pos]}, v_{i+D[pos]}⟩}  (as printed)
16:             𝔫.E[pos] = 𝔟
17:             i = i + D[pos]
18:         else if D[pos] ≥ D_γ^M_L then                          ← large conflict: CHILD NODE
19:             iterate the subsequent positions pos_seq where D[pos_seq] > D_γ^M_L, summing their degrees into 𝔇_seq
20:             allocate a new node at 𝔫.E[pos]
21:             Modelling({⟨xᵢ,vᵢ⟩, …, ⟨x_{i+𝔇_seq}, v_{i+𝔇_seq}⟩}, 𝔫.E[pos])   ← recursion (as printed)
22:             for all positions pos+1 .. pos_seq set the pointer to 𝔫.E[pos]   (duplicated pointers)
23:             i = i + 𝔇_seq
24:         end if
25:     end for
26: end if
27: return 𝔫
```

So the three thresholds are: D = 1 → data slot; 1 < D < D₉₉ → bucket sized D₉₉ (≤ 6); D ≥ D₉₉ → recurse on the
whole run of consecutive over-full positions. Contrast with LIPP, which "directly builds a new node for
conflicted keys" at any D > 1 [NFL §4.3, Write-Only]; AFLI's bucket "is a quite cheap adjustment" by comparison
[NFL §4.2, Read-Write]. Line numbering follows the print exactly: 13 is "i = i + 1", 17 is "i = i + D[pos]", 24
is "end if", 25 "end for", 26 "end if" [NFL p.8, Alg. 3.2]; the paper's own prose refers to the data-slot case as
"Line 10-13" and the bucket case as "Line 14-17" [NFL §3.3.2, p.8].

Reading note (an off-by-one in the print). Lines 15 and 21 as printed list the pairs from index i to i + D[pos]
(resp. i + 𝔇_seq) **inclusive**, i.e. D[pos] + 1 (resp. 𝔇_seq + 1) pairs, while lines 17 and 23 advance i by
exactly D[pos] (resp. 𝔇_seq). Since D[pos] is by definition the number of keys predicted to pos, the intended
range is i .. i + D[pos] − 1 (resp. i .. i + 𝔇_seq − 1); otherwise consecutive buckets would share a key and a
bucket would hold one key more than its conflict degree. Also note the asymmetry between lines 18 ("≥") and 19
(">"): the run of positions folded into one child starts at a position with D ≥ D_γ and continues only through
positions with D > D_γ [NFL p.8, Alg. 3.2 lines 15, 17, 18, 19, 21, 23].

#### 6.4 Lookup

Start at the root (a model node or a dense node; a bucket is never a root). Model node: evaluate the linear
model, read the entry type from the bitmaps at the predicted position: empty ⇒ key absent; data slot ⇒ compare
the stored key; bucket ⇒ linear scan of the bucket; node pointer ⇒ recurse. Dense node ⇒ binary search
[NFL §3.3.2, Queries]. The point the paper stresses: "the predicted positions in model nodes are all precise
positions, which means that there is no extra local search in model nodes" — the only searches are the ≤ 6-entry
bucket scan and the dense-node binary search [NFL §3.3.2].

#### 6.5 Insert and split ("Modelling")

Model node: predict the position; empty ⇒ store and set the bitmaps; data slot ⇒ create a bucket holding the
two conflicting pairs and replace the entry by a bucket pointer; bucket or node pointer ⇒ insert there. Bucket:
append at the tail (or insertion-sort in ordered mode). Dense node: binary search; empty ⇒ insert; occupied ⇒
shift data to the closest empty slot and insert [NFL §3.3.2, Insertions]. When a bucket or dense node is full,
it is converted into a model node by Algorithm 3.2 (a bucket is sorted first) [NFL §3.3.2, Modelling; Fig. 6].
Update = lookup + in-place payload write. Delete = lookup + unset a bitmap bit (model node) or overwrite the key
with the following keys (bucket/dense node) [NFL §3.3.2, More Operations].

#### 6.6 Why AFLI is built for near-uniform keys

Every structural decision assumes D₉₉ is small: capacity is at most α·n slots per node (line 7) so a heavy
cluster cannot be absorbed by over-allocation; conflicts up to D₉₉ ≈ 4-6 are scanned linearly in a bucket, which
is cheap only because D₉₉ is tiny; anything larger recurses, which on raw FB (D₉₉ = 386) or LLT (146) would
produce deep, unbalanced trees. The paper shows exactly that with AFLI alone: "The long P99 latency of proposed
AFLI on the workloads LLT and FB also prove that without the transformation of NF, the learned index can be an
unbalanced tree due to locally large conflicts" [NFL §4.3, Read-Only]. AFLI is therefore not a general-purpose
index; it is the second stage of a pipeline whose first stage guarantees D₉₉ ≈ 4.

---

### 7. Experiments (§4)

#### 7.1 Hardware, software, parameters [NFL §4.1.3]

* Ubuntu 20.04, Intel Core i7-10700 (8 cores, 2.9 GHz), 64 GB RAM; **single thread**; GCC 9.3.0 at −O3.
* NF inference through Intel MKL; NF training in PyTorch on an RTX 3080 (10 GB) with 64 GB host RAM.
* Flow: modified B-NAF, 2 layers, 2 input dimensions, 2 hidden dimensions, Gaussian base with variance 10¹⁶;
  trained on a 10 % sample of the bulk-loaded keys, "for three times".
* **Batch size 256** for all operations. Bucket size ≤ 6.
* Baselines with their default hyper-parameters: LIPP, ALEX, PGM-Index, Google's C++ B-Tree [NFL §4.1.2].

#### 7.2 Datasets and workloads [NFL §4.1.1]

Seven datasets, each "about 200 million unique keys", key type `double`, payload `int64`:

| Name | Abbrev. | Content / construction | Source as stated |
|---|---|---|---|
| longitudes | LTD | longitudes of locations | Open Street Maps [31] |
| longlat | LLT | k = 180·FLOOR(longitude) + latitude per (lon, lat) pair | Open Street Maps [31] |
| lognormal | LGN | synthetic, lognormal μ = 0, σ = 2, ×10⁹, floored | synthetic |
| YCSB | YCSB | user IDs from the YCSB generator | YCSB [4] |
| amazon | AMZN | book sale popularity | Kaggle Amazon sales-rank data [35] |
| facebook | FB | up-sampled Facebook user IDs | Van Sandt et al. SIGMOD 2019 [36] |
| wikipedia | WIKI | article edit timestamps | Wikimedia dumps [10] |

(The PDF says only that the seven are "representative datasets used in [6, 20, 42]", i.e. ALEX, SOSD and LIPP
[NFL §4.1.1 p.8]; that FB and WIKI coincide with SOSD's fb and wiki files is plausible from the identical
descriptions but cannot be verified from the sources in hand [unverified: the SOSD paper is not among our
sources]. Our ten GRE datasets are books, fb, osm, covid, genome, history, libio, planet, stack, wise
[results/aidb/provenance.json]; whether GRE's fb and osm are the same files as the FB and OSM-derived sets here,
and which of ours is "wiki-derived", is not established in this section [unverified: dataset lineage taken from
names only; see section 03 §2 and section 04 for the AIDB paper's dataset table].)

Workloads: bulk-load 50 % of the dataset (≈100M keys), then run one of four request mixes: read-only (100 %
lookups), read-heavy (80/20), write-heavy (20/80), write-only (100 % inserts). Requested keys are sampled from
the dataset with a **Zipfian** distribution; inserted keys lie inside the bulk-loaded key space
("known-key-space insertions"). Each workload is run 5 times and throughput, tail latency and index size are
averaged [NFL §4.1.1]. Everything is measured at **batch size 256** [NFL §4.1.3] and the inference "can be
simplified as several matrix computations and nonlinear function computations" on MKL [NFL §3.2.2]; the picture
"receive 256 requests → transform them with one MKL call → serve them" is this guide's reading of those two
sentences [unverified: mechanism inferred from §3.2.2 and §4.1.3; the paper does not describe the per-batch
sequence].

What a DB professor will ask and the paper does not give, each [NFL §4.1.1] **not reported**: the Zipfian skew
parameter (only "sampled from the given dataset based on a Zipfian distribution"); the number of operations in
the running phase (the "~100M insertions" used elsewhere in this section is an inference from §4.4.1's "after
inserting around 100 million new keys", which fits a write-only run over the remaining 50 % of a 200M-key
dataset, not a stated count); whether lookups include non-existent keys (misses); the record layout (the
payload type `int64` is stated, the layout is not); and the exact key-space bound used for "known-key-space"
insertions (min..max of the bulk-loaded keys is the natural reading).

#### 7.3 Throughput (§4.2, Figure 7)

The per-dataset bars of Figure 7 are not printed as numbers; the paper only gives averages and extremes in the
text (every value below is a quote of §4.2; per-dataset bar heights are **not reported** numerically). Whether
Figure 7's throughput *includes* the online transformation is not stated: Figures 8, 9 and 10 split NFL-Index
from NFL-Trans, but Figure 7 draws a single "NFL" series, and the only explicit statement that a measurement
includes the transform is for bulk loading ("the time cost includes the time cost of the online transformation
of bulk-loaded keys" [NFL §4.4.2]) [unverified: Figure 7 legend has one NFL series (NFL, AFLI, LIPP, ALEX,
PGM-Index, B-Tree); whether transform time is inside the throughput is not stated]. The README's result line has
a single `(overall throughput)` column next to separate T and I latency columns, so the natural reading is that
throughput is end-to-end; treat that as a reading, not a fact [nfl_readme.md, Results].

| Workload | NFL vs LIPP | vs ALEX | vs PGM-Index | vs B-Tree | Notes |
|---|---|---|---|---|---|
| Read-only (Fig. 7a, e) | 2.34× | 2.46× | 3.82× | 7.45× | up to 2.41× (LIPP) and 3.70× (ALEX) on LLT and FB; NF vs AFLI alone 2.25× on large-conflict workloads; NF *degrades* throughput on LTD and LGN (small D₉₉ gain, inference overhead) |
| Read-heavy (Fig. 7b, f) | +72.22 % | +101.05 % | +611.48 % | +389.45 % | |
| Write-heavy (Fig. 7c, g) | +29.10 % | +39.28 % | +50.88 % | +162.92 % | gains shrink because buckets fill and their linear scan slows |
| Write-only (Fig. 7d, h) | +22.65 % | +28.30 % | −21.43 % | +131.58 % | PGM's LSM-style 128-entry buffer wins on pure inserts |
| Introduction average (the abstract prints no numbers) | 2.77× throughput, 43 % lower tail latency | | | | [NFL §1, p.2: "2.77x improvements on average in throughput and 43% reductions on average in tail latency"] |

On YCSB, AMZN and WIKI the flow is off, so NFL ≈ AFLI [NFL §4.2]. On LTD and LGN under read-write mixes AFLI
alone is comparable to NFL because AFLI's tightly fitted nodes have few empty slots ("like the case of
overfitting") while the flow leaves larger arrays [NFL §4.2].

#### 7.4 Tail latency (§4.3, Figures 8-9)

**Definition of P99 in this paper**: "In each run, we collect the latency of each batch of operations, sort the
latencies in the ascending order, and report the 99-th percentile batch latency divided by the batch size"
[NFL §4.3]. So P99 is the 99th-percentile *batch* time / 256, i.e. an average over the 256 requests of a slow
batch, not the 99th percentile of individual requests; it smooths single-request outliers by construction.
Illustration (script 2, section L): a batch of 256 in which one lookup takes 10 µs and the other 255 take 50 ns
has batch latency 10,000 + 255·50 = 22,750 ns and is reported as 22,750/256 = 88.87 ns; a per-request P99 of
the same trace would report 50 ns and its max 10 µs. This is the one-line reason the paper's "P99 < 80 ns on LLT
and FB" is not comparable with an unbatched per-lookup P99 such as our `read_hit_p99_ns_median` column (which is
0 in `results/aidb/sweep/summary.csv` because that sweep did not run the latency replay, `latency_pass: false`;
the point is definitional).
Figures 8-9 separate NFL-Index (index part) from NFL-Trans (transform part), matching the README's T/I columns.

| Workload | P99 reduction vs LIPP | vs ALEX | vs PGM | vs B-Tree | Notes |
|---|---|---|---|---|---|
| Read-only | 58.68 % | 32.89 % | 62.73 % | 80.77 % | NFL keeps P99 < 80 ns on LLT and FB; others > 100 ns |
| Read-write (RH + WH) | 26.64 % | 45.05 % | 59.49 % | 65.31 % | |
| Write-only | 2.26 % | 27.92 % | −32.09 % (NFL higher) | 50.48 % | LIPP comparable (both precise placement) |

P99.99 and max latency (read-heavy, Fig. 9): NFL has the lowest of all; the figure's annotations show baselines
reaching 2-19 µs at P99.99 and 3 µs - 6 ms at max; NFL's exact bars are **not reported** numerically.

#### 7.5 Bulk loading, memory, ablations (§4.4)

* **Bulk loading** (Fig. 10): AFLI is the fastest loader except PGM; NFL spends 77 % of its load time in the
  transform and needs 2.25×, 0.86×, 2.81× the load time of LIPP, ALEX, B-Tree; 13.42 s on average for 100M keys
  [NFL §4.4.2].
* **Index size** (Fig. 11, after the running phase of write-heavy/write-only, including gaps): NFL is 2.26× ALEX,
  3.1× PGM-Index and 0.51× LIPP [NFL §4.4.3].
* **Conflict degree** (Table 3, §4.4.1): the transformed tail stays ≈ 4 after ~100M inserts although the flow saw
  only 10 % of the bulk-loaded keys; on low-conflict data the flow *raises* D₉₉.
* **Ablations**: (i) NFL vs AFLI (flow on/off) throughout Figures 7-11: 2.25× on high-conflict data, negative on
  LTD/LGN, neutral where switched off; (ii) flow size (Table 2): larger flows too costly; (iii) NFL-Trans vs
  NFL-Index split of latency and load time in Figures 8-10. There is no ablation of the feature expansion, of the
  sum decoder, of the base variance, or of the batch size on end-to-end throughput; the only batch-size sweep is
  the transform cost in Table 2.

**What is per-batch**: all throughput (ops/s over batched processing), all P99/P99.99/max latencies (batch
latency ÷ 256), and the transform cost (Table 2 per-key figures are batch averages). Nothing in §4 is a
single-request latency.

---

### 8. What our study takes from NFL, and what it does not

Our components are clean-room CONTROLS inside the SCALE-LI experimental map, not a reproduction of NFL. Taken
from NFL: (1) the **2D2H2L text weight format** and its evaluation (header `in_dim hidden layers`, `mean var`,
two 2×2 matrices, tanh hidden layer, linear last layer, sum decoder) so that author-trained weights would load
[transform.hpp:35-51, 64-79] [unverified: format read from our reader's expectations; no official weight file
is among the sources and none was loaded in this study, APPROACHES.md:18 lists "author weights on real SOSD
data" as not covered]; (2) the **two-feature encoder** [x, x − ⌊x⌋] with x = (key − mean)/var
[transform.hpp:69; train_flow.py `features`]; (3) the **tail conflict degree** as the switch metric, computed
per 4,096-key region on the raw and the transformed feature [transform.hpp:90-106; index.hpp:237-282]. Detail
that answers "how can you compute a conflict degree for a transform that is not monotone?": the transformed D₉₉
is computed on the **sorted** z values (index.hpp:248 and :280 copy z into `zs`, `std::sort` it, then call
`tail_conflict_degree`), and `tail_conflict_degree` fits feature → index-in-sorted-order (transform.hpp:94 uses
i as the target), so for a non-monotone flow our metric mirrors NFL's "sort by z" semantics even though our
records stay in key order. fb numbers from the run log [results/aidb/sweep/results.jsonl, packed_rank_flow,
fb, seed 11, `learnability`]: 489 regions, 41 accepted (41/489 = 8.38 %), mean per-region D₉₉ raw 8.4479 vs
flow 8.4642 (the flow does not lower the average), `flow_bytes` 144 (8 weights + mean/var + struct overhead
[transform.hpp:62]); (4) the
**bypass idea**, tightened to "keep the flow only if D₉₉ falls by at least 10 %" (`--flow-bypass 1
--flow-min-gain 0.1`) instead of the paper's plain comparison [index.hpp:54-55, 281-282; MEETING_NOTES.md §2].
Not taken: AFLI (our host keeps its own regions, blocks, fences and packed codecs; z is only the model feature
and records stay in key order [transform.hpp:14-16]); the B-NAF PyTorch trainer (ours is a stdlib 1-D
change-of-variables trainer with Adam, `--monotone` zeroing the fractional-feature weights, 4,096 sampled keys,
200 steps, ≈2.4 s per dataset [train_flow.py; results/aidb/flows/training_report.json]); the paper's training
objective (ours is 0.5·z² − log(dz/dx) + barrier/(dz/dx), i.e. a **standard normal base, σ = 1, not the paper's
σ² = 10¹⁶**, with a positivity barrier, §3.2 [train_flow.py:48-55]); the paper's normalization constants (our
mean/var come from the 4,096-key training subsample, §3.8 [train_flow.py:89]); and batching (every lookup
transforms one key, costing 66 ns [56, 77] per evaluation on our host [MEETING_NOTES.md §3], while the selector
charges it c_flow = 4 probe-equivalents [benchmark.cpp:191; index.hpp:264]; at build time our loop spends 29.88
ns/key on fb [summary.csv `preprocess_ns_per_key`], against NFL's 8.38 ns at batch 256 [NFL Table 2]). Outcomes that section 05 details: on the 2M uniform samples the
whole-sample D₉₉ is unchanged by our flow on 9 of 10 datasets (planet 18 → 9) [training_report.json,
`tail_conflict_degree_raw/transformed`]; the per-region bypass accepts the flow in only 0.2 % (stack) to 8.8 %
(planet) of regions [results/aidb/sweep/summary.csv, `flow_region_fraction`, packed_rank_flow, read_only]; and
forcing it into every region leaves fence probes equal to the control's to within 5.4×10⁻⁴ (fb: packed_rank
2.599879 vs packed_rank_flow_forced 2.599933, a difference of 5.4×10⁻⁵; the largest gap on any dataset is planet,
+5.4×10⁻⁴, and the granularity ablation states "within 0.05") [same file, packed_rank_flow_forced;
MEETING_NOTES.md §1]. That is the expected result of §3.8: an 8-weight monotone tanh is nearly affine, and a
per-region linear model already absorbs what it can express.

Fence probes per operation, control vs forced flow, read-only, uniform 2M samples, 3 seeds each
[results/aidb/sweep/summary.csv, rows packed_rank / packed_rank_flow / packed_rank_flow_forced; script 2,
section M]:

| dataset | packed_rank (control) | packed_rank_flow (bypass) | packed_rank_flow_forced | forced − control | flow_region_fraction (bypass) | preprocess_ns_per_key (bypass) |
|---|---|---|---|---|---|---|
| books | 2.230642 | 2.230594 | 2.230558 | −0.000084 | 0.0532 | 28.87 |
| covid | 3.063852 | 3.063821 | 3.063754 | −0.000098 | 0.0348 | 31.50 |
| fb | 2.599879 | 2.599902 | 2.599933 | +0.000054 | 0.0838 | 29.88 |
| genome | 3.297549 | 3.297557 | 3.297578 | +0.000029 | 0.0164 | 32.33 |
| history | 2.458673 | 2.458677 | 2.458760 | +0.000087 | 0.0266 | 29.33 |
| libio | 2.555318 | 2.555336 | 2.555368 | +0.000050 | 0.0429 | 31.24 |
| osm | 5.076565 | 5.076247 | 5.076308 | −0.000257 | 0.0143 | 30.82 |
| planet | 3.229126 | 3.228882 | 3.229667 | +0.000541 | 0.0879 | 32.12 |
| stack | 2.291464 | 2.291464 | 2.291436 | −0.000028 | 0.0020 | 31.57 |
| wise | 2.417157 | 2.417156 | 2.417249 | +0.000092 | 0.0082 | 26.95 |

Every difference is below 10⁻³ probes per lookup, i.e. below the seed spread; the `preprocess_ns_per_key`
column is the per-key cost of the one transform pass at build (27-32 ns/key here, unbatched, plain loops).

---

### 9. Questions the supervisor may ask, with answers

1. **Why does a flow help a learned index at all?** A learned index is a CDF approximator; a flow trained by
   maximum likelihood toward a flat base density learns (a smooth version of) the CDF, so z = G⁻¹(x) is near-
   uniform and rank ≈ linear in z. Segmentation heuristics find *local* near-uniform pieces by cutting the curve
   deeper; the flow makes the *whole* curve near-linear once, so the index needs fewer levels (ALEX FB height
   11 → 3, Table 1) and fewer collisions (FB D₉₉ 386 → 4, Table 3) [NFL §2.2, §3.1, Tables 1, 3].

2. **Is the transform monotone?** Not by construction. B-NAF is monotone per input coordinate, but NFL's input is
   [⌊x⌋, frac(x)] (Algorithm 3.1) and the output is a sum of the flow's coordinates, and frac(x) is a sawtooth;
   the composed key → z can decrease at integer boundaries of x_norm. NFL therefore builds and searches on z
   (sorted by z) and only needs z to be injective; it evaluates no range scans [NFL Alg. 3.1, Alg. 3.2 input,
   §4.1.1]. Our control keeps records in key order and uses z only as the model feature, so exactness never
   depends on monotonicity [transform.hpp:14-16]; the weights we trained are monotone anyway (fractional weights
   zeroed, 0 unordered pairs on all ten datasets [training_report.json]).

3. **What does the flow cost per key?** Table 2: 169.53 ns at batch 1, 40.60 at 8, 15.28 at 32, 9.52 at 128,
   8.38 at 256, 7.29 at 2048 (2H2L, MKL, i7-10700). Larger flows: 21.66 / 23.52 / 35.07 ns at batch 256 [NFL
   Table 2]. Bulk-loading 100M keys spends 77 % of 13.42 s in the transform (≈103 ns/key), which the paper does
   not reconcile with Table 2 [NFL §4.4.2; my arithmetic; unverified: the scope of those averages (all seven
   datasets or only the flow-on ones) is not stated]. Our unbatched control: 66 ns [56, 77] measured per lookup
   transform, 29.88 ns/key in the build loop on fb [MEETING_NOTES.md §3; summary.csv].

4. **Why does NFL batch?** Stated reason: batched requests are common in modern databases [NFL §3.1, refs 14, 15,
   28, 29, 33]. Quantitative reason: the transform is a dense network; at batch 1 it alone costs 169.53 ns ≈ a
   5.9 Mops/s ceiling, far below the 40-45 Mops/s scale of Figure 7's axes, while at batch 256 it is 8.38 ns
   [NFL Table 2; Fig. 7 axes] [unverified: the baselines' numeric throughputs are not printed].
   Every reported number is at batch 256 [NFL §4.1.3].

5. **How is P99 defined and why does it matter?** The 99th-percentile *batch* latency divided by 256 [NFL §4.3].
   It is an average over the 256 requests of the 99th-percentile batch, so single-request outliers are diluted
   by construction; it is not comparable with a per-request P99 measured unbatched.

6. **What exactly is the tail conflict degree?** Fit one linear model, count the keys per integer predicted
   position (Definition 3.1), take the occupied positions, and report the value at rank INT(0.99·m) in
   ascending order (Definition 3.2), i.e. the 99th percentile of per-position collision counts. On 10 keys with
   cluster {10,11,12,13} the max is 4 but D₉₉ = 2 (§4.1). Our control reports collisions (count − 1) with a
   ceiling percentile and capacity 1.5n [transform.hpp:90-106].

7. **When is the flow switched off?** If D₉₉ on the transformed keys is larger than on the raw keys [NFL §3.2.2];
   the paper reports one outcome per dataset, and "once per bulk load, for the whole dataset" is the guide's
   inference [unverified: §3.2.2 + §4.2]. Which keys and which model the two D₉₉ are computed on: not reported.
   In the paper: YCSB (3 → 4), WIKI (2 → 4) and AMZN (4 → 4, which under the literal rule would keep it;
   unresolved in the paper) [NFL §4.2, Table 3]. Ours: per 4,096-key region, on the region's keys, with a 10 %
   margin [index.hpp:281].

8. **Why train toward N(0, 10¹⁶) and not the uniform?** The uniform's log-density is −∞ outside its support and
   constant inside, giving "Nan-loss" / "INF-loss"; a Gaussian with σ = 10⁸ is flat over the key range and keeps
   the likelihood finite [NFL §3.2, §4.1.3].

9. **What is the flow's architecture and why so small?** Two input features, two tanh hidden units, two layers,
   8 weights, no bias (2D2H2L). Table 2 shows that 16-48-parameter flows cost 2.6-4.2× more per key at batch 256
   and, per §3.2.2, did not lower the conflict degrees enough. The paper itself says the flow's quality "has
   almost reached to its upper limit" [NFL §4.4.1]. Our forward-pass table (§3.8) shows a trained 2D2H2L is
   within 1.75 % of a straight line over the key range.

10. **What is the feature expansion for?** A scalar key gives a flow "hardly any useful knowledge"; Algorithm 3.1
    splits the normalized key into integer part, base-θ digits of the fraction and the remainder — a 1-to-1
    O(n·d) encoding. The fractional part restores uniqueness (the integer part alone is a step function) and adds
    a periodic feature. Deployed d = 2 [NFL §3.2.1, §4.1.3].

11. **What does AFLI do that LIPP does not?** Both place keys at precise predicted positions; LIPP creates a child
    node for any collision, AFLI buffers collisions of degree < D₉₉ (≤ 6) in a bucket and recurses only on runs of
    positions with degree ≥ D₉₉ (Algorithm 3.2 lines 11-23); dense nodes catch keys the model cannot separate
    (slope 0). Write-only P99: NFL 2.26 % lower than LIPP; index size 0.51× LIPP [NFL §3.3, §4.3, §4.4.3].

12. **How robust is the switch to insertions?** Inserts are within the bulk-loaded key space, the flow saw 10 % of
    those keys, and after ~100M inserts the transformed D₉₉ is still 4-5 on every dataset (raw FB rose 386 → 454)
    [NFL Table 3, §4.4.1]. No experiment covers a distribution shift; the paper only says the flow would be
    retrained offline "when the distribution significantly shifts" [NFL §3.2.2].

13. **Which numbers are averages and which are per dataset?** All the ×/% figures in §4.2-4.3 are averages over
    datasets and workloads of a class; the only per-dataset numbers printed are Table 1 (LLT, FB with ALEX),
    Table 3 (D₉₉ for all seven) and Figure 9's annotations. Per-dataset throughputs exist only as bars.

14. **Where does our control depart from NFL, in one breath?** Same weight format and encoder, same conflict
    metric idea, per-region instead of per-dataset switch with a 10 % margin, z as feature only, records in key
    order, stdlib trainer with σ = 1 and a positivity barrier instead of B-NAF with σ² = 10¹⁶, no batching (66 ns
    measured per unbatched evaluation, charged as 4 probe-equivalents by the selector, vs 8.38 ns batched), no
    AFLI. Hence our finding "the flow feature is never chosen" is a statement about an unbatched, region-local
    setting at 2M-200M keys, not a refutation of NFL's batched, whole-dataset result [MEETING_NOTES.md §1-3].

15. **What is the licence situation?** github.com/luffy06/NFL is GPL-3; no code was copied, only the public text
    weight format was re-implemented (format compatibility is not a derivative work of the code)
    [transform.hpp:3-10; docs/APPROACHES.md:18] [unverified: licence text itself not in our sources; the README we
    hold has no licence statement].

16. **Why a flow and not simply the empirical CDF, or a spline of it (RadixSpline, PGM), as the transform?** In
    §3.1's own terms the CDF *is* the ideal transform (a uniform z is what a learned index wants), and the paper
    lists RadixSpline [21] among the learned indexes it positions itself against [NFL §5.1, ref. 21]. A spline of
    the CDF is itself a learned index: it carries as many parameters as it has segments, must be rebuilt when the
    data change, and is applied per lookup. The flow is a smooth *parametric* stand-in that costs 8 weights and
    8.38 ns/key batched [NFL Table 2], generalizes ("normalizing flows have much better generalization, they
    don't need to re-train during every bulk loading phase" [NFL §3.2.2]), and is trained offline once
    ("about 38 seconds" [NFL §3.2.2]). The price is capacity: an 8-weight monotone tanh is nearly affine (§3.8),
    which is why the paper reports its quality "has almost reached to its upper limit" [NFL §4.4.1] and why our
    per-region linear models see nothing left to gain (§8).

17. **How much memory does the flow take?** 8 weights plus mean and var; our reader reports `flow_bytes` = 144
    (the struct plus 8 doubles [transform.hpp:62; results/aidb/sweep/results.jsonl `learnability.flow_bytes`]).
    The paper's benchmark records a `(model size)` column in every result line [nfl_readme.md, Results], but the
    PDF prints no number for it: **not reported**.

18. **What does the paper's index size (2.26× ALEX, 3.1× PGM, 0.51× LIPP) actually count?** "the overall index
    size, including the sum size of allocated gaps, rather than the model size", measured "after the running phase
    of write-heavy workloads" and normalized in Figure 11 [NFL §4.4.3]; so gaps from precise placement are inside
    the number, which is why LIPP (also precise placement) is the only index larger than NFL.

19. **Is D₉₉ well defined for tiny m?** Yes: t = INT(m·γ) is at least 1 for m ≥ 2 (m = 1 gives t = 0, which the
    paper does not address; **not reported**), and the worked example of §4.1 has m = 6 ⇒ t = 5 ⇒ D₉₉ = 2 while
    the maximum is 4. Our control uses ⌈0.99·m⌉ − 1 as the 0-based index, which at m = 6 picks the maximum
    (index 5) and at m = 1000 the same 990th element [transform.hpp:104].

20. **The flow is not monotone; how do you compute a conflict degree on it?** NFL sorts by z; we sort the z
    values before calling `tail_conflict_degree`, which fits feature → sorted index [index.hpp:248, :280;
    transform.hpp:94], so the metric is the same as NFL's while our records stay in key order (§8). With
    `--monotone` weights the question is moot: 0 unordered pairs on all ten datasets [training_report.json].


---

## 02. Learned Indexes with Distribution Smoothing via Virtual Points (CSV), deep account

**What this section gives you.** A self-contained, equation-level reading of the CSV paper: the exact optimisation problem it poses (loss over ranks with and without virtual points, the budget λ = α·n, the NP-hardness sketch and its weak spot), the single-model greedy (Algorithm 1) reproduced line by line with the closed-form refit and the derivative-sign filter, a worked 7-key example computed with Python so every number can be re-derived by hand, the hierarchical CSV (Algorithm 2) with its cost condition (eq. 22), what the paper does and does not say about lookups and inserts in ALEX / LIPP / SALI, the experimental setup and the reported numbers (Tables 2–4, Figures 6–10), and finally an honest map of what our clean-room CONTROL (`smoothing.hpp`, `index.hpp`) takes from CSV and what it deliberately leaves out, with our own measured numbers. Every number carries an inline source; anything not in the PDF, the code or a result file is marked "not reported" or [unverified: ...]. Notation follows the guide's convention: keys k₁ < … < kₙ, rank r(kᵢ) = i − 1, linear model f(k) = w·φ(k) + b, slot s(kᵢ) = rank + number of virtual points before kᵢ, budget λ = α·n.

---

### 1. Bibliographic facts

| Item | Value | Source |
|---|---|---|
| Title | Learned Indexes with Distribution Smoothing via Virtual Points | [CSV p.1] |
| Authors | Kasun Amarasinghe, Farhana Choudhury, Jianzhong Qi, James Bailey, all The University of Melbourne | [CSV p.1] |
| arXiv | 2408.06134v3 [cs.DB], stamped 15 Dec 2024; PDF creation date 2024-12-17 | [CSV p.1 margin stamp; PDF metadata `creationDate D:20241217015339Z`] |
| Venue | EDBT 2025 | [unverified: the PDF never names EDBT for this paper; its running header reads "Arvix, Oct, 2025" and the ACM-format block says only "2025"; the EDBT 2025 attribution comes from the workspace registry, `docs/APPROACHES.md` line 22 and `results/aidb/READING_GUIDE.md` line 42, which also records the inconsistent date labels] |
| Public implementation | None. Checked September 2026. | [smoothing.hpp:7 "No public implementation of CSV exists (checked September 2026)"; READING_GUIDE.md:43] |
| Hosts the paper integrates with | ALEX [2], LIPP [33], SALI [9] | [CSV §1, §6.1] |
| Headline claims | promote up to 60% of lower-level keys, up to 34% query-time improvement on those keys, "less than 15% increase to the storage space overhead" | [CSV §1 contribution (3)] |

Note the storage figure is stated three ways in the PDF: "< 15%" [CSV §1], "in most cases less than 10% ... in the worst case less than 31%" [CSV §6.2.1 "Index size"], and "at or below 10%" for the read-write runs [CSV §6.3]. Quote the §6.2.1 sentence if challenged; the 31% worst case is the paper's own number.

---

### 2. Motivation and the core idea

**The model.** A learned index is a function f from a key to its storage position; with keys stored in ascending order the target is the rank, so f approximates the CDF: rank(kᵢ) ≈ f(kᵢ) [CSV §1]. The paper uses only linear f (w·k + b) "for their efficiency" and says the idea "can naturally extend to more complex (e.g., quadratic) functions" [CSV §1].

**The observation.** Hierarchical indexes (LIPP is the example) push hard-to-model key ranges into deeper levels, and deeper keys cost more per query: Figure 1 plots average query time per level of LIPP on Facebook, Covid, OSM, Genome (200M keys each) and shows time rising with level [CSV Fig. 1]. Prior work fixes this by changing the model (splines, more segments) or the structure; CSV instead "modifies the key space": it changes the training targets so that a linear model fits better [CSV §1].

**What a virtual point is.** A virtual point k_v is an extra value inserted into the sorted key list K (keeping it sorted) that holds no record. Its only effect is on ranks: every real key after k_v moves one position later, so the rank targets of the real keys change. In Figure 2, ten black dots (real keys) are fit by a line with loss L_f(K) = 8.33; after inserting V = {k_v1..k_v5} (budget 0.5·n = 5, red hollow dots) and refitting to f′, the loss on the real keys drops to L_f′(K) = 2.04 and the loss on real plus virtual points is L_f′(K ∪ V) = 2.29 [CSV §1, Fig. 2a/2b].

**Why it can lower the error.** Squared error of an OLS line depends on how far the (key, rank) points sit from any straight line. A cluster of keys with a wide empty range after it forms a "step" in the CDF; a line cannot bend around a step. Inserting virtual points into the empty range stretches the rank axis exactly where the keys are sparse, so the augmented point set lies closer to a line. Section 4.6 below shows this on seven keys: one virtual point in the only wide gap cuts SSE from 2.671 to 1.872.

The one-line algebra behind that sentence: for an OLS line, SSE = S_yy · (1 − ρ²), where S_yy = Σ (yᵢ − ȳ)² is the total variance of the targets and ρ is the Pearson correlation between feature and target over the fitted set. Equivalently, with the running sums c_xx = S_xx − S_x²/n, c_xy = S_xy − S_x·S_y/n, c_yy = S_yy − S_y²/n, SSE = c_yy − c_xy²/c_xx, which is the closed form our control evaluates [smoothing.hpp:38-43]. A virtual point in a wide gap adds one target step exactly where the CDF is flat, which raises |ρ| of the augmented set. On the seven keys of §4.6: before smoothing S_yy = 28, ρ = 0.951101, ρ² = 0.904594, SSE = 28 × 0.095406 = 2.6714; after inserting k_v = 9 (eight targets 0..7) S_yy = 42, ρ = 0.977466, ρ² = 0.955439, SSE = 42 × 0.044561 = 1.8716; after {8, 9, 10} ρ = 0.993812 and SSE = 1.0179 [scratch/csv_fix_checks.py]. Note that S_yy itself GROWS with every inserted slot (28 → 42), so the SSE falls only because 1 − ρ² falls faster; this is also why a whole-sample RMSE measured in slot units can rise after smoothing (§8.5, Q10) while every local fit improves.

**The trade-off.** (i) Space: each virtual point is a slot in the host's array that holds nothing, so the index grows by up to λ = α·n slots; the paper reports the resulting index-size increase in Figs. 8b/8e/8h [CSV §6.2.1]. (ii) Preprocessing time: the search for virtual points is a greedy over candidate positions; Tables 3 and 4 report between 247 s (ALEX, Facebook, α = 0.05) and 81,620 s (ALEX, OSM, α = 0.8) of one-off preprocessing on 200M keys depending on α, host and dataset; the LIPP table alone spans 304 s (Covid, α = 0.05) to 15,709 s (Genome, α = 0.8) [CSV Tables 3–4, p.11; §7.4 below reproduces every cell]. (iii) Query time can worsen for hosts with an in-node search (ALEX) when merged nodes become large; the cost model of §5.1 guards against this [CSV §5, §5.1].

**Origin of the idea.** The authors present CSV as the constructive mirror image of poisoning attacks on learned indexes [11] (Kornaropoulos et al., SIGMOD 2022), which insert points to maximise the SSE; CSV inserts points to minimise it [CSV §2.3]. The related "gap insertion" [16] manipulates ranks directly, which lets several keys share a position and needs an overflow array with "up to 87%" space increase; NFL [34] transforms the key value with a normalising flow, which adds a transform at query time and may raise the tail conflict degree [CSV §2.2, Table 1].

---

### 3. The formal problem (§3)

#### 3.1 Loss without virtual points

For one indexing function f over key set K of size n [CSV eq. 1]:

```
L_f(K) = Σ_{i=1..n} ( f(kᵢ) − rank(kᵢ) )²                              (1)
```

For a partition of K into m segments K₁..K_m, each with its own f_i ∈ F [CSV eq. 2, stated exactly]:

```
L_F(K) = Σ_{i=1..m} Σ_{k ∈ K_i} ( f_i(k) − rank(k) )²                   (2)
```

"Equation 2 is the loss function of our optimisation problem. We aim to insert values (virtual points) into K while keeping it sorted, such that L_F(K) is minimised" [CSV §3]. Note carefully: in eq. 2 the rank is the rank in the FULL list K after insertion, and the f_i are the ORIGINAL functions (no refit yet); refitting is introduced only in §4 (eq. 4).

#### 3.2 Why a budget is needed

The "naive optimal smoothing" inserts enough virtual points that every key k lands at position f_i(k) exactly (rank(k) = f_i(k)), making the loss zero; with 64-bit integer keys that layout would need 2⁶⁴ × 8 bytes ≈ 128 exabytes [CSV §3]. Hence the budget:

```
|V| ≤ λ,   λ = α · n,   α ∈ (0, 1)        (linear space overhead)          [CSV §3]
```

**Definition 1 (learned index smoothing).** Given sorted K partitioned into m segments each indexed by f_i ∈ F, insert a set V (|V| ≤ λ) of virtual points into K, keeping K in order, such that the loss of eq. 2 is minimised [CSV Def. 1].

#### 3.3 Assumptions

- Linear indexing functions ("used in most existing learned indexes") [CSV §3].
- Integer keys; real keys are handled "when they can be scaled up to become integers" [CSV §3]. This is why candidates form integer runs between consecutive keys (§4.2 of the paper).
- Keys are unique: the experiments remove duplicates "to suit LIPP and SALI's requirements", and the candidate filter "skip[s] the index keys already in K" so that hosts without duplicate support stay compatible [CSV §4.2, §6.1].
- Ranks start at 0 [CSV §4.1].

#### 3.4 NP-hardness (Lemma 3.1) and how to read it

The paper's proof sketch [CSV §3, Lemma 3.1]:

1. Knapsack: items S, each with value c_s and weight w_s; choose A ⊆ S maximising Σ value with Σ weight ≤ t.
2. CDF smoothing: candidate set C of virtual points, "naively ... λ virtual point candidates between every two adjacent keys", so |C| ≤ λ·(n − 1); choose V ⊆ C, |V| ≤ λ, minimising the loss, i.e. maximising the loss reduction.
3. Mapping: items ↦ candidates; every weight ← 1; t ← λ; value c_s ← the loss reduction of that candidate.
4. Because loss reductions are not additive when several points are inserted together, the paper writes

```
| Σ_{s∈A} c_s | = r · Σ_{s∈A} |c_s|,   r ∈ ℝ                                (3)
```

   with r "deterministic since it could be calculated based on Equation 1", and argues that the combined value therefore transforms into a sum of individual values, "there is a one-to-one mapping", so solving smoothing solves Knapsack and smoothing is NP-hard.

**How to defend / critique it.** The direction of the reduction is right (Knapsack → smoothing). But two things a database professor will notice: (a) with all weights equal to 1 the Knapsack instance is the cardinality-constrained version, solvable in polynomial time by sorting values, so unit weights alone do not carry hardness; (b) eq. 3 asserts that the interaction between chosen points is a single scalar r, whereas the interaction really depends on which subset is chosen (the refit changes w and b). The paper does not fill these gaps. What IS uncontroversial: the exact problem requires choosing a size-λ subset of p candidate positions with a non-additive objective; the paper's own count for the brute force is O(pCλ · n · p) [CSV §4.3], and Table 2 shows the exhaustive search on 10 keys with λ = 5 already takes 140,656,167 ns versus 424,667 ns for the greedy [CSV Table 2]. Say: "the paper gives a Knapsack-style sketch; the honest position is that the exact problem is combinatorial with a non-additive objective and that the paper's evidence for the greedy is empirical (Table 2), not an approximation guarantee." [our reading, not a statement in the PDF]

---

### 4. Single-model smoothing (§4)

#### 4.1 The refit objective (eq. 4)

For one segment K_i with budget λ [CSV eq. 4, stated exactly]:

```
argmin_{V_i, w, b}  L_{f_{w,b}}(K_i ∪ V_i)     s.t.  |V_i| ≤ λ                 (4)
```

Two differences from eq. 2 [CSV §4]: (i) the slope w and intercept b are REFITTED to the keys with adjusted ranks plus the virtual points, instead of keeping the original f; (ii) the virtual points themselves are inside the loss ("we include V_i in the loss calculation, such that the storage space allocated to the virtual points can be used to accumulate data insertions, with minimized prediction errors when querying the inserted data points"). This is why Figure 2 reports two numbers: 2.04 on K only and 2.29 on K ∪ V.

**The naive greedy and its cost.** Repeat λ times: try every candidate k_v, refit, compute the loss, keep the best. With p candidates per iteration and O(|K_i| + λ) per loss evaluation the naive cost is O((|K_i| + λ) · λ · p) [CSV §4 "Challenge 2"]. The paper's three steps: reduce p by a derivative test (§4.2), make each loss evaluation O(1) by reusing sums (§4.1), and run the greedy over the reduced set (§4.3).

#### 4.2 Closed-form refit and incremental loss (eqs. 5–16)

Let yᵢ = rank(kᵢ) (from 0). Insert one virtual point k_v with rank y_v after insertion. Loss with refitted w, b [CSV eq. 5]:

```
L(K ∪ V) = Σ_{i=1..n} (w·kᵢ + b − yᵢ)² + (w·k_v + b − y_v)²                  (5)
```

where, in eq. 5 and in eqs. 10–16, the yᵢ are the SHIFTED ranks (every key after k_v has its rank increased by one; the paper encodes this in eqs. 11 and 14); eq. 9 is the one place where the yᵢ must be read as the original ranks, see below. OLS refit over the n + 1 points [CSV eqs. 6–9]:

```
w = Σ_{i=1..n+1} (kᵢ − k̄_v)(yᵢ − ȳ_v) / Σ_{i=1..n+1} (kᵢ − k̄_v)²             (6)
b = ȳ_v − w · k̄_v                                                          (7)
k̄_v = ( Σ_{i=1..n} kᵢ + k_v ) / (n + 1)                                     (8)
ȳ_v = ( Σ_{i=1..n} yᵢ + n ) / (n + 1)                                       (9)
```

Eq. 9 is worth a second look: the mean target after inserting ONE point is always (0 + 1 + … + n)/(n + 1) = n/2, independent of where the point goes. The paper writes it as (Σ yᵢ + n)/(n + 1), and that identity holds only when the yᵢ in it are the ORIGINAL ranks: Σ y_original + n = n(n − 1)/2 + n = n(n + 1)/2, so the quotient is n/2. With the shifted ranks it would double-count the shift. On the 7-key example with k_v = 9 (y_v = 4): (Σ y_original + n)/(n + 1) = (21 + 7)/8 = 3.5 = the true mean of 0..7, whereas (Σ y_shifted + n)/(n + 1) = (24 + 7)/8 = 3.875 ≠ 3.5 [scratch/csv_fix_checks.py; scratch/csv_eq_check.py line 21 uses y0, the original ranks, for its 3.5 check]. So read eq. 9 with the original ranks; the shifted-rank bookkeeping enters through eqs. 11 and 14, not through eq. 9.

Rewritten loss separating K-only sums from k_v terms [CSV eq. 10]:

```
L(K ∪ V) = w² Σ kᵢ² + 2wb·n·k̄ − 2w Σ kᵢyᵢ + n·b² − 2n·b·ȳ
           + Σ y²_original,i + n² − y_v² + (w·k_v + b − y_v)²                (10)
```

with the bookkeeping identities that express the rank shift [CSV eqs. 11–14]:

```
Σ_{i=1..n} yᵢ     = Σ y_original,i + n − y_v                                 (11)
ȳ = Σ yᵢ / n,   k̄ = Σ kᵢ / n                                              (12, 13)
Σ_{i=1..n} kᵢyᵢ   = Σ kᵢ·y_original,i + Σ_{i=y_v..n} kᵢ                      (14)
```

Eq. 11 says: the n − y_v keys after the virtual point each gain 1 (n − y_v ones added). Eq. 14 says: the cross sum gains the sum of the keys that were shifted (a suffix sum of the keys). Index convention in eq. 14: with keys k₁..kₙ numbered from 1 and ranks from 0, the keys whose rank shifts are those of original rank ≥ y_v, i.e. k_{y_v+1}..kₙ, so the printed lower limit i = y_v is off by one unless the keys are 0-indexed. On the example with y_v = 4 the printed Σ_{i=4}^{7} kᵢ = 5 + 12 + 13 + 14 = 44, but the shifted keys are 12, 13, 14 with sum 39, which is the value used in §4.6 Step 2 [scratch/csv_fix_checks.py]. Our check scripts never evaluate the printed limit: csv_eq_check.py builds the shifted rank vector directly (ys = y + 1 for y ≥ y_v, line 7) and sums kᵢ·yᵢ over it, which is the intended quantity. Eq. 10's "Σ y²_original + n² − y_v²" is Σ yᵢ² after the shift, and it is exact: the shifted suffix contributes Σ_{i=y_v}^{n−1} ((i+1)² − i²) = Σ (2i + 1) = (n − y_v)(n + y_v) = n² − y_v². On the example with k_v = 9 (y_v = 4): Σ y²_original = 91, n² − y_v² = 49 − 16 = 33, and the shifted ranks 0,1,2,3,5,6,7 give Σ y² = 124 = 91 + 33 [checked by hand from scratch/csv_eq_check.py's shifted ranks]. The w, b in the incremental form [CSV eqs. 15–16]:

```
w = ( Σ kᵢyᵢ + k_v·y_v − (n+1)·k̄_v·ȳ_v ) / ( Σ kᵢ² + k_v² − (n+1)·k̄_v² )     (15)
b = ȳ_v − w·k̄_v                                                            (16)
```

Numerically eq. 15 equals eq. 6 (checked at k_v = 6, 9, 11 on the example: w = 0.485126, 0.495413, 0.476744 by both forms) [scratch/csv_eq_check.py].

**Multiple virtual points.** After accepting (k_v1, y_v1) the "original key set terms" are updated to include it and the next candidate is evaluated against the augmented base [CSV §4.1 "Adjustment for multiple virtual points"].

#### 4.3 Filtering candidates by the derivative sign (§4.2)

Candidate range: (min K, max K) open, because points before min K shift every rank equally and points after max K shift none; existing keys are excluded [CSV §4.2]. Within that range the integer candidates form runs ("sub-sequences") between consecutive keys: for keys 20 and 26 the run is 21..25 [CSV §4, Fig. 3].

Filter [CSV §4.2 items (1)–(2), and Algorithm 1 lines 7–22]:

```
for each run E = [first, second] of consecutive integer candidates:
    if second − first ≤ 1 (run of length ≤ 2):        keep both endpoints          (Alg.1 l.7-8)
    else:
        compute L′ at both endpoints (eq. 17)                                       (Alg.1 l.13-14)
        if L′(first) · L′(second) < 0:                  keep only the interior minimum,
                                                        found from the two derivative
                                                        values' intersection with the x-axis  (l.15, l.20-21)
        else (same sign):                                keep both endpoints          (l.17)
```

The justification: the loss as a function of k_v over one run is a sum of n quadratic residual terms plus one term (w·k_v + b − y_v)² where w, b also depend on k_v; the paper cites [11] for convexity of each run and so a sign change of the derivative between the endpoints implies exactly one interior minimum [CSV §4.2]. The exact condition, as written in Algorithm 1 line 14: `L′(K ∪ G[i].first) · L′(K ∪ G[i].second) < 0`.

Derivative with respect to k_v, K-only sums factored out [CSV eqs. 17–21]:

```
L′(K ∪ V) = 2·( w′·( w·Σ kᵢ² + n·b·k̄ − Σ kᵢyᵢ ) + n·b′·( w·k̄ + b − ȳ )
              + ( w·k_v + b − y_v )·( w′·k_v + w + b′ ) )                          (17)
w′ = ( A·( n·(y_v − ȳ) ) − B·( 2n·(k_v − k̄) ) ) / A²                              (18)
b′ = −( w + (n+1)·k̄_v·w′ ) / (n+1)                                                (19)
A  = (n+1)·( Σ kᵢ² + k_v² ) − ( (n+1)·k̄_v )²                                       (20)
B  = (n+1)·( Σ kᵢyᵢ + k_v·y_v ) − (n+1)²·k̄_v·ȳ_v                                 (21)
```

Here k̄, ȳ are the n-point means of eqs. 12–13 and w = B/A. We verified eqs. 17–21 as printed against a central finite difference of the true loss on the example: at k_v = 6 both give −1.132330, at k_v = 11 both give +0.886966, at the interior minimiser 8.7224 both give −0.000009 [scratch/csv_eq_check.py]. So the printed derivative is correct (the identity (n+1)(y_v − ȳ_v) = n(y_v − ȳ) is what makes the n-point means in eq. 18 consistent with the (n+1)-point means in eqs. 20–21).

**What `minimum_point(first, second)` computes (Alg. 1 line 21).** The PDF gives no formula, only the sentence "The minimal point can be calculated by using the two partial derivative values to find their intersection with the x-axis" [CSV §4.2, p.6]. Read literally, that is the zero of the chord (secant) of L′ through the two endpoint values, i.e. a linear interpolation of the derivative:

```
k* = first + (second − first) · ( −L′(first) ) / ( L′(second) − L′(first) )        [our transcription of CSV §4.2]
```

On the 7-key example, run 6..11: k* = 6 + 5 · 1.132330 / (1.132330 + 0.886966) = 8.8038, whereas the exact continuous minimiser found by ternary search on the true loss is 8.7224 (loss 1.8551) [scratch/csv_fix_checks.py; scratch/csv_worked_example.py]. The secant is an approximation because L′ is not linear in k_v (w and b move with k_v); both values round to 9 here. How the real-valued k* is turned into an integer candidate (floor, round, or both neighbours) is not reported in the PDF.

#### 4.4 Algorithm 1 line by line

Reproduced from [CSV Algorithm 1, p.7] with a gloss per line.

```
Algorithm 1 CDF_smoothing
Require: key set K, loss with a new virtual point L(K ∪ k_v), smoothing threshold α
 1: U, C, V = []                       U: loss per candidate; C: candidate keys; V: accepted virtual points
 2: G = [], M = []                     arrays of endpoint PAIRS: G = runs longer than 2, M = runs whose
                                       endpoint derivatives have opposite sign
 3: L′(K ∪ k_v) = ∂L(K ∪ k_v)/∂k_v,  λ = α · K.size,  L_previous = L(K ∪ ∅)
                                       define the derivative (eq. 17); budget; loss of the unsmoothed set
 4: Find the endpoint pairs E for each sub-sequence
                                       one pair per run of consecutive integer candidates between keys
 5: while V.size < λ do                at most λ rounds
 6:   for i = 1..E.size do             classify every run
 7:     if E[i].second − E[i].first ≤ 1 then
 8:       append E[i].first and E[i].second to C       short run: both points are candidates
 9:     else                           run of more than 2 points
10:       append E[i].first to G.first and E[i].second to G.second
11:     end if
12:   end for
13:   for i = 1..G.size do             derivative test on the long runs
14:     if L′(K ∪ G[i].first) · L′(K ∪ G[i].second) < 0 then
15:       append the pair to M         sign change: interior minimum exists
16:     else
17:       append G[i].first and G[i].second to C        same sign: minimum is at an endpoint
18:     end if
19:   end for
20:   for i = 1..M.size do
21:     append minimum_point(M[i].first, M[i].second) to C   interior minimiser: x-axis intercept of the
                                       chord of L′ between the two endpoints (formula in §4.3); integer
                                       rounding rule not reported
22:   end for
23:   for i = 1..C.size do
24:     U[i] = L(K ∪ C[i])             loss with that single candidate added and the model refitted
25:   end for
26:   find index i of minimum L        best candidate this round
27:   if L_previous ≤ U[i] then
28:     break                          no candidate lowers the loss: stop early (may return fewer than λ)
29:   end if
30:   append C[i] to V, append C[i] to K, L_previous = U[i]
                                       accept: it becomes part of the base set for the next round
31: end while
32: return C                           [sic] the prose says "the final virtual points in V are returned";
                                       line 32 as printed returns C — a typo in the PDF
```

Two details the gloss must make explicit. First, K is modified in place at line 30, so from round 2 on the "keys" include earlier virtual points and the runs E must be re-derived (the pseudo-code computes E once at line 4; the prose's "Adjustment for multiple virtual points" implies the sums are updated; how E is refreshed is not stated in the PDF). Second, C, G, M are not cleared between rounds in the pseudo-code; read them as per-round scratch.

#### 4.5 The claimed complexity O(n + λ)

"[CSV] reduces the computation of the loss over K to just once, which takes O(n) time. This process is repeated to find λ optimal candidate virtual points. However, there is no need to recalculate the loss function after adding a virtual point, as we could treat the key set with the previous virtual point inserted as the new original or base key set for a constant time calculation. Thereby, giving a time complexity of O(λ + n)" [CSV §4.3 "Complexity analysis"].

How to read it: the O(n) is the one-off computation of Σkᵢ, Σkᵢ², Σkᵢyᵢ, Σyᵢ; each candidate's loss is then O(1) via eqs. 10–16 IF the suffix sums in eq. 14 are available in O(1); each of the λ rounds then costs O(number of surviving candidates). The bound O(n + λ) therefore presumes that the number of candidates examined per round is O(1), which the PDF does not justify: after the derivative filter there is still at least one candidate per run, and there are up to n − 1 runs. A strict reading of Algorithm 1 gives O(n + λ · p′) with p′ the candidates surviving the filter (≤ 2·(n − 1) + |M|). Our implementation is O(λ · n) for exactly this reason (§8). [our reading; the PDF states O(λ + n) without the per-round count]

**What that means per region on our data.** A 4,096-key region has 4,095 gaps; at α = 0.1 the budget is ⌊0.1 × 4,096⌋ = 409 rounds, so our O(λ·n) greedy performs on the order of 409 × 4,095 ≈ 1.67 M candidate-gap evaluations per region, each with 4 loss probes plus up to 40 ternary-search steps for a gap with an interior minimum [smoothing.hpp:72-89]. Measured on the books 2M uniform sample with the hardness tool: 489 regions (⌈2,000,000/4,096⌉), 185,327 virtual points in total = 379 per region (the budget is 409, so most regions ran the full budget, a few stopped early), smoothing wall-clock 16.58 s on 16 threads = 33.9 ms of wall-clock per region and 8.3 µs per key of the sample [results/aidb/hardness/books_sample_csv.json: regions 489, virtual_points 185327, smoothing_ns 16,575,434,458, threads 16; src/hardness.cpp:172-173 time the parallel `smooth_regions` call; hardness.hpp:369 spawns the worker pool]. The benchmark reports a different figure for the same dataset and α: preprocess_ns_per_key = 30,591.6 ns = 30.6 µs/key for books_uniform packed_rank_vp10 [results/aidb_final/sweep/summary.csv], and that column is a SUM of per-region smoothing_ns divided by the key count [index.hpp:305, 485; tools/summarize.py:59], i.e. summed thread time, not wall-clock. If the hardness tool's 16 threads had been fully parallel, its summed thread time would be about 16 × 8.3 = 133 µs/key, four times the benchmark's 30.6 µs/key; the runs differ in scheduling (the benchmark ran with performance-core QoS, `qos 1`, the hardness run's QoS is not recorded), in the number of regions that ran the full budget, and in host load, so the gap between the two figures is not established here [our reading]. Either way the order of magnitude is tens of microseconds of CPU per key at 4,096-key regions, which is what §8.4 quotes.

#### 4.6 Worked example on seven keys (computed, not taken from the paper)

Keys K = {2, 3, 4, 5, 12, 13, 14}, n = 7, ranks 0..6. The only wide gap is (5, 12). All numbers below are from `scratch/csv_worked_example.py` (exact rational arithmetic, then decimal).

**Step 0 – fit and loss before smoothing.** OLS on (kᵢ, rank): w = 112/283 = 0.395760, b = 1/283 = 0.003534.

| key | rank | f(k) = w·k + b | residual | squared |
|---|---|---|---|---|
| 2 | 0 | 0.7951 | +0.7951 | 0.6321 |
| 3 | 1 | 1.1908 | +0.1908 | 0.0364 |
| 4 | 2 | 1.5866 | −0.4134 | 0.1709 |
| 5 | 3 | 1.9823 | −1.0177 | 1.0356 |
| 12 | 4 | 4.7527 | +0.7527 | 0.5665 |
| 13 | 5 | 5.1484 | +0.1484 | 0.0220 |
| 14 | 6 | 5.5442 | −0.4558 | 0.2078 |

L(K) = 756/283 = **2.6714**. The large residuals sit on both sides of the gap (key 5: −1.02; key 12: +0.75): the line cannot follow the step.

**Step 1 – candidate runs (Alg. 1 line 4).** Integer candidates in (2, 14) not in K: gaps (2,3), (3,4), (4,5), (12,13), (13,14) are empty; gap (5,12) gives the run 6..11 of length 6 (> 2, so it goes to G, line 10).

**Step 2 – derivative sign test (lines 13–15).** dL/dk_v at k_v = 6 is −1.1323, at k_v = 11 is +0.8870; product < 0, so the run goes to M and only its interior minimiser is kept (line 21). The paper's `minimum_point` is the x-axis intercept of the chord of L′ between the endpoints (§4.3): 6 + 5 × 1.132330 / (1.132330 + 0.886966) = 8.8038; the exact continuous minimiser (ternary search on the true loss) is k_v* = 8.7224 with loss 1.8551 [scratch/csv_fix_checks.py; scratch/csv_worked_example.py]. Both round to 9, and 9 is the candidate we carry forward; "nearest integer" is our own interpretation, because the PDF never states how the real-valued minimum point becomes an integer candidate (not reported). Any virtual point in that run has rank y_v = 4 and shifts ranks of 12, 13, 14 from 4, 5, 6 to 5, 6, 7 (eq. 11: Σy gains n − y_v = 3; eq. 14: Σky gains 12 + 13 + 14 = 39).

**Step 3 – loss per candidate (lines 23–26), shown for the whole run so you can see the convex shape.** (The table evaluates EVERY integer in the run. `scratch/csv_worked_example.py` does the same in every greedy round: it brute-forces all integers in (min K, max K) not yet present, with no derivative filter [csv_worked_example.py lines 84-87]. Algorithm 1 would evaluate only the filtered set C; on this example the two agree at every round, see Step 5.)

| k_v | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|
| L(K ∪ {k_v}) | 3.4325 | 2.4938 | 1.9676 | **1.8716** | 2.1963 | 2.9070 |

The refit for k_v = 9: w = 0.495413, b = −0.339450 (eq. 6/15 agree). Since 1.8716 < 2.6714 the test at line 27 passes and 9 is accepted (line 30).

**Step 4 – new slots after one virtual point.**

| slot | content | f′(x) | residual |
|---|---|---|---|
| 0 | key 2 | 0.6514 | +0.6514 |
| 1 | key 3 | 1.1468 | +0.1468 |
| 2 | key 4 | 1.6422 | −0.3578 |
| 3 | key 5 | 2.1376 | −0.8624 |
| 4 | VIRTUAL 9 | 4.1193 | +0.1193 |
| 5 | key 12 | 5.6055 | +0.6055 |
| 6 | key 13 | 6.1009 | +0.1009 |
| 7 | key 14 | 6.5963 | −0.4037 |

L_f′(K ∪ V) = 204/109 = **1.8716** (this is the paper's eq. 4 objective); L_f′(K) on the seven real keys alone = **1.8573** (the paper's Figure-2 style "L_f′(K)"). Reduction 30% with one point (λ = 1, α ≈ 0.14).

**Step 5 – continue the greedy to λ = ⌊0.5·7⌋ = 3.** Round 2 inserts 8 (loss 1.2799), round 3 inserts 10 (loss 1.0179); V = {9, 8, 10}, final w = 0.696429, b = −1.071429, final slots 0..9 with the three virtual points at slots 4, 5, 6 and keys 12, 13, 14 at slots 7, 8, 9. Final L(K ∪ V) = **1.0179** (−61.9%), L(K) real-only = 0.7178.

Does Algorithm 1's candidate filter change any of this? Recomputing the filter per round [scratch/csv_fix_checks.py]:

| round | base set | runs (line 4) | filter | C | losses | accepted |
|---|---|---|---|---|---|---|
| 1 | {2,3,4,5,12,13,14} | 6..11 (second − first = 5 > 1 → G) | L′(6) = −1.1323, L′(11) = +0.8870, sign change → M; interior min 8.7224 (secant 8.8038) → 9 | {9} | 9: 1.8716 | 9 |
| 2 | {2,3,4,5,9,12,13,14} | 6..8 (second − first = 2 > 1 → G); 10..11 (second − first = 1 → C, line 8) | L′(6) = −1.0814, L′(8) = +0.1596, sign change → M; interior min 7.7500 (secant 7.7428) → 8 | {8, 10, 11} | 8: 1.2799, 10: 1.6607, 11: 2.1594 | 8 |
| 3 | {2,3,4,5,8,9,12,13,14} | {6,7} and {10,11}, both second − first = 1 → C (line 8) | no derivative test needed | {6, 7, 10, 11} | 6: 1.6346, 7: 1.2066, 10: 1.0179, 11: 1.2840 | 10 |

So the greedy path 9 → 8 → 10 is identical under Algorithm 1's filter and under the brute force of the script; the filter only saves loss evaluations (1 + 3 + 4 = 8 instead of 6 + 5 + 4 = 15). This is what the example demonstrates: Algorithm 1's choices, computed by a brute force that happens to coincide with them here.

**Step 6 – greedy vs exhaustive (the paper's Table 2 in miniature).** Exhaustive search over all subsets of ≤ 3 points from the pool {6..11} examines 41 subsets and finds {7, 9, 10} with loss **0.7578** < greedy 1.0179. The greedy is not optimal (it committed to 9 first); this is the same qualitative picture as the paper's 10-key case where CSV reaches 2.293 and the exhaustive 2.118 from an original 8.327 [CSV Table 2].

---

### 5. Hierarchical CSV (§5)

#### 5.1 Why a second algorithm

Applying Algorithm 1 to each node in isolation lowers the leaf-node search time but "fail[s] to address traversal time". CSV therefore smooths the key set of a parent plus its subtree and, if a cost condition holds, rebuilds them as ONE leaf, reducing the index height for those keys [CSV §5]. The stated challenge is balancing the traversal saving against the larger leaf; "unbalanced learned index structures are better suited as it gives the ability to reduce the height of taller branches without affecting the rest" [CSV §5].

#### 5.2 Algorithm 2 line by line

Reproduced from [CSV Algorithm 2, p.8].

```
Algorithm 2 CSV
Require: nodes with sub-trees Nodes, smoothing threshold α, cost threshold c
 1: Nodes = []
 2: keyset = []
 3: keyset_smooth = []
 4: max_level ← maximum level of index with sub trees      deepest level whose nodes still have children
 5: current_level ← max_level
 6: while current_level > 1 do                            bottom-up until the level under the root
 7:   Nodes ← all nodes with sub trees                    (at current_level; the PDF says "all nodes with sub trees")
 8:   for i = 1..Nodes.size do
 9:     keyset ← collect all keys in the node and its sub tree
10:     keyset_smooth ← CDF_smoothing(keyset, α)          Algorithm 1 on the collected keys
11:     if cost < c then                                   cost condition (§5.1; eq. 22 for ALEX)
12:       Reconstruct the sub-tree and node with keyset_smooth
                                                          "creating a new leaf-node in place of the parent
                                                          node and placing the keys from the collected nodes"
13:     end if
14:   end for
15:   current_level ← current_level − 1
16: end while
```

Practical rules from §6.1 that modify this: for LIPP and SALI, which "can create nodes that are indexing only a few keys", CSV starts at the SECOND level of the index rather than the bottom "such that each smoothing step can benefit more points"; for ALEX it starts at the bottom; and for all hosts CSV "stops at the second level from the top (i.e., the root)" because query times in the top two levels are very close [CSV §6.1 "Parameters"].

**Level convention.** The root is level 1: Algorithm 2 loops `while current_level > 1` [CSV Alg. 2 line 6] and the prose says the process "is performed until the root node depicted as level 1 is reached" [CSV §5, p.8]. Hence "keys at level 3 or below" (the promotable keys of §6.1) are keys at least two hops under the root, and "stops at the second level from the top" means level 2 is the highest level whose nodes are rebuilt; a key promoted from level 3 to level 2 saves one hop.

**What "reconstruct" (line 12) means per host, and what the PDF does not say.** The only sentence on the mechanics is: the merging "is performed by creating a new leaf-node in place of the parent node and placing the keys from the collected nodes" [CSV §5, p.8]. Not reported: (i) whether the host rebuilds its own node model over the n + |V| slots (LIPP and SALI place keys with a conflict-free linear model, ALEX fits its own linear model) or installs CSV's refitted OLS line f′ from eq. 4; (ii) how a LIPP leaf, which needs every real key in a distinct slot, is built from `keyset_smooth` when the OLS refit does not guarantee conflict-free placement (the paper's §5.1 wording "if the new model could hold more keys than before" suggests that keys which still conflict remain in child nodes, but the PDF does not say so); (iii) whether the virtual points are stored as explicit empty slots or the node array is simply sized n + |V| and the model trained on slot targets. Our control (§8.2) takes the third option and keeps only the virtual features; that is a design choice of ours, not a statement of the paper.

#### 5.3 The cost condition (eq. 22) per host

- **LIPP and SALI** have no in-node search: a leaf model places a key exactly or creates a child node on conflict. The paper therefore uses the loss value itself as the cost condition: "their loss function values can be taken as the cost conditions. This is because if the new model could hold more keys than before, then it does not have any other component (that is, leaf-node search time) that would negatively affect the performance" [CSV §5.1, p.8]. The PDF gives NO numeric acceptance threshold on the loss for these two hosts; "hold more keys than before" is the only stated criterion. Whether "hold more keys" means conflict-free placement of more real keys in the merged node (LIPP semantics: a node holds a key iff the key's predicted slot is free) is [our reading]; the PDF does not define it. For SALI specifically the PDF adds nothing beyond "SALI is based on LIPP" [CSV §6.2.1, p.10]; see §6 below.
- **ALEX** has an exponential/binary search inside the leaf after the model prediction, so a merged leaf can cost more per lookup. Reconstruction happens only if [CSV eq. 22]

```
cost = search_constant × expected_number_of_searches + traversal_constant × index_level       (22)
```

is below the threshold c. The constants are measured on the target machine by sampling queries: search_constant = time per leaf-node search, traversal_constant = traversal time per level; expected_number_of_searches comes from "the inbuilt function in ALEX that uses the log₂ error to estimate it". Because eq. 22 estimates the expected query time of the node, "the cost threshold c should be set below 0 to identify an improvement. Setting a lower value would result in fewer keys being able to be promoted to upper levels but the expected query time improvement will be greater" [CSV §5.1]. The numerical values of the constants and of c used in the experiments are not reported.

Three things eq. 22 leaves undefined, which you should say out loud rather than paper over [CSV §5.1, p.8]:

  (a) which `index_level` enters: the merged leaf's new level, the parent's level, or a difference of levels — not reported;
  (b) since c must be "below 0 to identify an improvement", `cost` must be a DIFFERENCE between the merged and the current configuration, but the equation as printed is an absolute expected time (a positive constant times a positive count plus a positive constant times a level); the phrase "relative to the current cost" used in §5.5 below is therefore [our reading], not the PDF's wording;
  (c) `expected_number_of_searches`: ALEX's leaf search is an exponential search from the predicted position, which takes about log₂(|prediction error|) steps, and that is what "uses the log₂ error" refers to [unverified: ALEX's behaviour (Ding et al., SIGMOD 2020) is not restated in the CSV PDF].

#### 5.4 Complexity of Algorithm 2

For m non-leaf nodes with nᵢ keys and budgets λᵢ, the paper sums O(λᵢ + nᵢ) over nodes and simplifies to O(λ + n) [CSV §5.1 "Complexity analysis"]. This inherits the per-round assumption of §4.5 and additionally ignores that a key belongs to every ancestor's collected key set, so the same key is smoothed once per level; the sum over levels is not O(n) unless the number of levels is O(1). [our reading]

**Budget accounting the paper never states.** Line 10 calls `CDF_smoothing(keyset, α)`, so each node i gets its own budget λᵢ = α · |keysetᵢ| [CSV Alg. 2 line 10; §5.1 "node_i with n_i keys and a smoothing budget of λ_i"]. A key at depth d belongs to the collected key sets of up to d − 2 ancestors (every level from max_level down to 2), so the pseudo-code bounds the total number of virtual points across one run of Algorithm 2 by α · Σᵢ nᵢ, not by α · n; the two coincide only for a two-level tree. Whether the key set collected at a higher level already contains the virtual points inserted at the lower level (so that they count as "keys" and raise λᵢ), or only the real keys, is not reported. The only empirical bound the paper gives on the resulting space is Fig. 8b/8e/8h: index-size increase "in most cases less than 10%" and "in the worst case less than 31%" at α ≤ 0.8 [CSV §6.2.1 "Index size"]; note that with α = 0.8 a single level alone could add 80% of slots, so the measured 31% says that most keys are smoothed at one level and that early stopping (Alg. 1 line 27) or failed cost conditions leave much of the budget unused [our reading].

#### 5.5 Example of merging a subtree into a leaf (illustrative, constructed by us)

Take a LIPP-like tree where a parent P at level 3 has a model over 4,096 keys, 3,900 of which it places exactly and 196 of which conflicted and live in 12 child nodes at level 4 (some of which have their own children at level 5). Algorithm 2 at current_level = 3 collects the 4,096 keys of P and its subtree (line 9), runs Algorithm 1 with α = 0.1, i.e. λ = 409 (line 10). Suppose the augmented loss falls enough that a fresh linear model over the 4,096 + |V| slots places every key without conflict (in LIPP terms, no two keys map to the same slot) — this "suppose" is an assumption of ours for the illustration; the PDF never says that reconstruction requires or achieves conflict-free placement of all collected keys (§5.2, "what reconstruct means"). Under the LIPP cost condition (the loss improved and the node holds more keys) the subtree is replaced by a single leaf with 4,096 real keys and |V| empty slots (line 12). Every key previously at levels 4–5 is now at level 3: those are the "promoted" keys the paper's Figure 8a counts; the 12 + child nodes disappear (Figure 8c "node reduction"); the |V| empty slots are the storage increase (Figure 8b). For ALEX the same merge would additionally have to pass eq. 22: the merged leaf's expected exponential-search length (from its log₂-error estimate) times search_constant, plus traversal_constant times the new level, must be less than c < 0 relative to the current cost ("relative to the current cost" is [our reading] of why c < 0 can mean an improvement; §5.3 item (b)). The numbers in this paragraph are invented for illustration; the mechanics follow [CSV §5, §5.1, Fig. 8].

---

### 6. Lookups and updates with virtual points in each host

What the PDF states, and only that:

- **Where virtual points live.** In the host's node storage as slots with no record ("gaps"). The paper contrasts itself with gapped arrays in ALEX/LIPP/APEX: those leave gaps for future inserts as a side effect, CSV chooses the gap positions to minimise model error [CSV §2.2]. After CSV, "the initial gaps left by the virtual points are gradually filled up by the inserted points" [CSV §6.3 "Index size"], confirming that a virtual point is an empty slot in the host's array.
- **Lookup.** The paper does not describe a modified lookup path. The reasoning in §5.1 tells you what each host does natively: LIPP and SALI "do not contain any searching component", so a prediction is a position; ALEX has a "leaf-node search component" whose expected length ALEX estimates from the log₂ error [CSV §5.1]. The refitted model f′ is trained on the slot layout that includes the gaps (eq. 4), so a real key's predicted slot is close to its actual slot. What happens when the model points at an empty slot for a key that IS present: for ALEX, the native exponential search scans outward over the gapped array until the key is found [unverified: this is ALEX's documented behaviour (Ding et al., SIGMOD 2020), not a statement in the CSV PDF]; for LIPP the stored key is at the predicted position by construction, since keys that would not fit are pushed into child nodes [unverified: LIPP's documented behaviour (Wu et al., PVLDB 2021), not restated in the CSV PDF]. For SALI the CSV PDF says nothing beyond "SALI is based on LIPP" [CSV §6.2.1, p.10] and "there is no such searching in LIPP and SALI's query process" [CSV §6.2.1, p.10]; its own related-work paragraph notes that SALI flattens sub-trees with a PGM-like segmentation, which "leads to an additional search step for queries, as we need to find the correct node from the flattened structure" [CSV §2.2, p.3]. How CSV's virtual points interact with that segment search is not reported.
- **Inserts.** Described only at the level of the read-write experiment: the index is built on a random half of each dataset, CSV is applied once, and the other half is inserted in batches of 0.1·n without re-running CSV [CSV §6.1 "Read-write workload"]. Inserted keys can land in the gaps left by the virtual points, which "helps improve the insertion times in some cases"; when more keys sit in upper-level nodes there are more collisions "which requires new index node creation", raising insert time in other cases; overall insert times are "on par" [CSV §6.3 "Insertion time"]. Storage overhead shrinks after each batch as the gaps fill: at or below 10% for LIPP and "almost negligible (< 0.5%)" for ALEX, sometimes below the original index because plain ALEX creates more nodes to host inserts [CSV §6.3 "Index size"].
- **Deletes.** Not reported.
- **Robustness claim.** "Since the models are built with virtual points that can be used to host data insertions, a side benefit of our structure is that it is more resilient against data insertions" [CSV §2.3].

---

### 7. Experiments (§6)

#### 7.1 Setup

| Item | Value | Source |
|---|---|---|
| Framework | based on the benchmark of Bachfischer, Borovica-Gajic, Rubinstein 2022 [1] | [CSV §6] |
| Hardware | Ubuntu 20.04.5 VM, AMD EPYC 7763 64-core CPU, 128 GB RAM | [CSV §6] |
| Hosts | ALEX, LIPP, SALI | [CSV §6.1] |
| Datasets (200M keys each) | Facebook (integer Facebook user IDs, origin [30] Van Sandt, Chronis, Patel, SIGMOD 2019), Covid (integer tweet IDs sampled from tweets tagged "Covid-19", origin [18] Lopez & Gallemore, SNAM 2021), OSM (OpenStreetMap locations as Google S2 [26] cell IDs, origin [24] Pandey, Kipf, Neumann, Kemper, PVLDB 2018), Genome (loci pairs in human chromosomes as integers, origin [25] Rao et al., Cell 2014); taken from the benchmark works [20] Marcus et al. (SOSD) and [32] Wongkham et al. (GRE); duplicates removed | [CSV §6.1 "Datasets", p.9; reference list pp.12-13] |
| Overlap with our ten GRE sets | fb, covid, osm, genome are the same four sources (ours are the GRE mirrors: provenance.json records them as the AIDB 2026 Table 1 sets, e.g. fb "Upsampled Facebook user ID"; whether the CSV authors' copies are byte-identical to the GRE files is not reported); books, history, libio, planet, stack, wise are not in the CSV paper | [results/aidb/provenance.json roles; CSV §6.1] |
| Easy / hard | Facebook, Covid easy; OSM, Genome hard; all except OSM "almost globally linear" CDFs; locally all except Covid deviate, "especially Genome" (Fig. 5 zooms on the 100M-th key + next thousand) | [CSV §6.1, Fig. 5] |
| α | 0.05, 0.1, 0.2, 0.4, 0.8; default 0.1 | [CSV §6.1 "Parameters"] |
| Scaling sizes | 12.5M, 25M, 50M, 100M, 200M; smaller sets made by eliminating every j-th key of the sorted set | [CSV §6.1] |
| Timing | each queried key repeated 100 times, average | [CSV §6.1] |
| Where CSV starts / stops | LIPP, SALI: from the second level; ALEX: from the bottom; all: stop at the level below the root | [CSV §6.1] |
| Read-only workload | build on full dataset, apply CSV, run queries | [CSV §6.1] |
| Read-write workload | build on a random half, apply CSV once, insert the other half in batches of 0.1·n, query after each batch | [CSV §6.1] |
| Queried keys | the "promoted" keys (every key moved to an upper level by CSV); "promotable" = keys at level 3 or below of the original index, with the root at level 1 (§5.2 "Level convention"), i.e. keys at least two hops under the root | [CSV §6.1 "Queries", "Evaluation metrics"; Alg. 2 line 6; §5 p.8] |
| Metrics | total query time saved; query-time improvement (%); promoted data (%); storage increase (%); node reduction (% of nodes at levels ≥ 3); insert-time increase (%) | [CSV §6.1] |

#### 7.2 Approximation quality (Table 2)

| | Exhaustive | CSV (greedy) | Original |
|---|---|---|---|
| Loss | 2.118 | 2.293 | 8.327 |
| Time (ns) | 140,656,167 | 424,667 | N/A |

Ten keys of Fig. 2, α = 0.5 (λ = 5); the text says the greedy improves the loss by 72.34% and the exhaustive by 74.44%, and that the exhaustive takes "nearly 3 orders of magnitude" longer [CSV §5.2, Table 2]. (The time ratio is 331×.) Internal inconsistency to be aware of: the percentages do not follow from the table's own values, (8.327 − 2.293)/8.327 = 72.46% and (8.327 − 2.118)/8.327 = 74.56% [scratch/csv_fix_checks.py]; the 0.12-point gap is unexplained in the PDF (presumably unrounded losses behind the text, but that is not stated). Quote the table's losses, not the text's percentages, if pressed.

#### 7.3 Read-only results vs α (Figs. 6–8)

- **Total time saved (Fig. 6).** Grows with α; LIPP and SALI behave alike because SALI builds on LIPP; on Facebook and Covid the saving "stabilise[s] after a certain number of virtual points" for LIPP/SALI because their CDFs are already near-linear. For ALEX the paper's explanation of the missing plateau is structural, not accuracy-related: "The same pattern was not observed for ALEX, this is because ALEX has an additional leaf-node search step not required by LIPP and SALI (CSV forms larger nodes that could lead to longer leaf-node search times)" [CSV §6.2.1, Fig. 6 caption-paragraph, p.10]. Also: "the total time saved for OSM and Genome is higher due to higher number of data being promoted for those datasets" [CSV §6.2.1, p.10]. Axis tick ranges: LIPP up to 4.5×10⁸ ns, SALI up to 7.5×10⁸ ns, ALEX up to 6×10⁸ ns; approximate bar heights at α = 0.8 read off the rendered figure: LIPP Genome ≈ 5.4×10⁸, OSM ≈ 3.0×10⁸, Facebook ≈ 2.2×10⁸, Covid ≈ 1.2×10⁸; ALEX Facebook ≈ 6.5×10⁸, OSM ≈ 3.5×10⁸, Genome ≈ 1.0×10⁸, Covid ≈ 0.5×10⁸ [unverified: read off Fig. 6 of the PDF page 10, rendered at 3×; no numbers are printed in the text].
- **Query-time improvement on promoted keys (Fig. 7).** "up to 34%, with stronger benefits observed over the two SOTA index structures LIPP and SALI"; smaller for ALEX "due to its leaf-node search process" but "consistent"; for ALEX the improvement rises with α, and the paper's reason for THAT is the accuracy one: "As α increases, the relative query time improvements for ALEX also increases, as the leaf-node search efficiency is improved due to increased accuracy of the refitted indexing functions. Since there is no such searching in LIPP and SALI's query process, their query performance is stagnant and the improvement represents the reduction in the index traversal time" [CSV §6.2.1, p.10]. Axis tick labels: LIPP 0, 10, 20, 30; SALI and ALEX 0, 5, 10, 15; the plotted areas extend above the last tick (to about 35 and 19), which is how a 34% bar fits under a "30" top tick [CSV Fig. 7, p.10]. The per-α, per-dataset percentages are not printed anywhere in the text; approximate bar heights read off the rendered figure [unverified: read off Fig. 7 of the PDF page 10, rendered at 3×; ±1 point]:

| host | dataset | α = 0.05 | α = 0.1 | α = 0.8 | trend |
|---|---|---|---|---|---|
| LIPP | Facebook | ≈ 30.5 | ≈ 30.5 | ≈ 31 | flat |
| LIPP | Covid | ≈ 32.5 | ≈ 32 | ≈ 32.5 | flat |
| LIPP | OSM | ≈ 28.5 | ≈ 28.5 | ≈ 29.5 | flat, slight rise |
| LIPP | Genome | ≈ 34 | ≈ 33.5 | ≈ 33.5 | flat; this is the "34%" bar |
| SALI | Facebook | ≈ 18 | ≈ 18 | ≈ 18 | flat |
| SALI | Covid | ≈ 18.5 | ≈ 19 | ≈ 18.5 | flat |
| SALI | OSM | ≈ 15.5 | ≈ 16.5 | ≈ 17 | slight rise |
| SALI | Genome | ≈ 18 | ≈ 18 | ≈ 18 | flat |
| ALEX | Facebook | ≈ 5 | ≈ 5 | ≈ 7.5 | rises |
| ALEX | Covid | ≈ 9.5 | ≈ 11 | ≈ 18.5 | rises strongly |
| ALEX | OSM | ≈ 9 | ≈ 10.5 | ≈ 13.5 | rises |
| ALEX | Genome | ≈ 11 | ≈ 11.5 | ≈ 12 | slight rise |

  The "up to 34%" is therefore the LIPP/Genome bar (highest at α = 0.05, ≈ 34), and for LIPP/SALI the improvement is essentially INDEPENDENT of α (traversal saving of one or more levels for a promoted key does not depend on how many virtual points bought the promotion), while the number of promoted keys and hence the total time saved (Fig. 6) do grow with α. The best α for total time is the largest tried (0.8), at the cost of space and preprocessing.
- **Promoted data (Figs. 8a/8d/8g).** Facebook ≈ 60% of promotable keys, Covid ≈ 30%, OSM up to 27%, Genome up to 57%; OSM and Genome promote MORE keys in absolute terms because they have far more keys in lower levels; more α promotes more [CSV §6.2.1 "Size of the promoted data"].
- **Index size (Figs. 8b/8e/8h).** "In most cases, less than 10% ... in the worst case, less than 31%"; proportional to α; axes 0–45% for LIPP/SALI and 0–12% for ALEX [CSV §6.2.1 "Index size", Fig. 8].
- **Node reduction (Figs. 8c/8f/8i).** Follows the promoted-data pattern; expressed relative to nodes at level ≥ 3 [CSV §6.2.1].

#### 7.4 Preprocessing time (Tables 3 and 4), seconds, 200M keys

Table 3, LIPP [CSV Table 3]:

| α | 0.05 | 0.1 | 0.2 | 0.4 | 0.8 |
|---|---|---|---|---|---|
| Facebook | 589 | 1,194 | 1,859 | 2,106 | 2,228 |
| Covid | 304 | 337 | 343 | 337 | 336 |
| OSM | 1,217 | 2,329 | 4,495 | 7,983 | 13,019 |
| Genome | 1,155 | 2,174 | 4,616 | 9,316 | 15,709 |

Table 4, ALEX [CSV Table 4]:

| α | 0.05 | 0.1 | 0.2 | 0.4 | 0.8 |
|---|---|---|---|---|---|
| Facebook | 247 | 889 | 4,123 | 17,508 | 48,737 |
| Covid | 609 | 1,423 | 2,795 | 4,463 | 4,955 |
| OSM | 988 | 2,297 | 9,526 | 33,097 | 81,620 |
| Genome | 1,356 | 2,902 | 6,253 | 8,854 | 9,777 |

At the default α = 0.1: LIPP 337 s (Covid) to 2,329 s (OSM); ALEX 889 s (Facebook) to 2,902 s (Genome). SALI omitted "for brevity" as it behaves like LIPP. The paper calls these "one-off pre-processing costs that can be amortised" and suggests building the CSV structure in parallel while serving from the original [CSV §6.2.1 "Pre-processing time"]. Observe that Covid's LIPP time is flat in α (304–343 s) while OSM/Genome scale almost linearly in α (13,019 / 1,217 ≈ 10.7× for 16× α): on an easy set the greedy stops early (Alg. 1 line 27), on hard sets it spends the whole budget. [our reading of Table 3]

#### 7.5 Cardinality (Fig. 9) and read-write (Fig. 10)

- Time saved grows with dataset size 12.5M → 200M on all sets, faster on Facebook/Covid because small versions of the easy sets have few lower-level keys [CSV §6.2.2, Fig. 9].
- Read-write, α = 0.1, LIPP and ALEX (SALI omitted): total time saved decreases slightly with each 0.1·n batch for LIPP (inserts collide more often with promoted keys now in upper levels); ALEX similar except two drops on OSM after batches 1 and 3 where the original index "happen[s] to be slightly lower" [CSV §6.3, Figs. 10a/10d]. Storage overhead ≤ 10% for LIPP and < 0.5% for ALEX, decreasing as gaps fill [Figs. 10b/10e]. Insert-time change within about ±8% for LIPP and ±30% for ALEX by the plotted axes, "on par" overall [Figs. 10c/10f; CSV §6.3].

#### 7.6 The paper's own stated limitations

Collected from the text: preprocessing "may seem quite large under certain settings" [CSV §6.2.1]; space grows with α up to 31% [CSV §6.2.1]; ALEX gains less because of its leaf search [CSV §6.2.1]; the smoothing of the full key set is "computationally expensive", which is why Algorithm 2 works on subtrees of an already-built index [CSV §5]; requires unique keys [CSV §6.1]; integer keys or real keys scalable to integers [CSV §3]; no re-run of CSV after inserts in the experiments [CSV §6.1]. Not addressed in the PDF: deletes, concurrency, the values of the eq. 22 constants and c, the tail (p99) latency, and any comparison to NFL or gap insertion beyond Table 1's checklist.

---

### 8. What our study takes from CSV and what it does not

Everything here is a clean-room CONTROL inside the SCALE-LI experimental map; it is written from the paper, not from any CSV code (none exists), and it is not a reproduction of CSV or of its ALEX/LIPP/SALI hosts [smoothing.hpp:2-8; docs/APPROACHES.md:22; README.md:35].

#### 8.1 Taken: Algorithm 1 per region, on slot targets

`smooth_cdf(x, alpha)` [smoothing.hpp:47-97] implements the greedy of Algorithm 1 for ONE model:

- Objective: the OLS residual SSE over real AND virtual points, "matching equation 4 of the paper", in closed form from running sums S_x, S_xx, S_y, S_yy, S_xy: SSE = S_yy − S_y²/n − (S_xy − S_x·S_y/n)² / (S_xx − S_x²/n) [smoothing.hpp:35-44]. This is algebraically the same quantity as eq. 5 with the refit of eqs. 6–7 substituted in.
- Rank shift of eqs. 11/14: "a candidate at gap i shifts every later target by one; suffix sums make each candidate evaluation O(1)": `loss_at(i, xv)` adds cnt = (points after i) to S_y, 2·suffix_y + cnt to S_yy and suffix_x to S_xy, then adds the candidate at target i + 1 [smoothing.hpp:59-70].
- Candidate range: only gaps between consecutive features; a gap whose right feature is not greater than the left is skipped; the interior of a gap is searched, never the endpoints themselves (the probes are at lo + e, lo + 2e, hi − 2e, hi − e with e = 10⁻³ · gap width) [smoothing.hpp:75-78]. This corresponds to the paper's open interval (min K, max K) and its exclusion of existing keys.
- Derivative-sign filter, replaced by a loss-sign test: "if [the loss] is decreasing on the left and increasing on the right, a ternary search finds the interior minimum (the paper uses the sign of the derivative at the endpoints for the same purpose). Otherwise the better end is used" [smoothing.hpp:14-17, 80-87]; up to `max_ternary_steps = 40` iterations [smoothing.hpp:47, 82].
- Budget λ = ⌊α · n⌋ [smoothing.hpp:57]; nothing is done for n < 3 or λ = 0 [smoothing.hpp:58].
- Early stop = Algorithm 1 line 27: a round with no candidate below the current SSE (by more than 10⁻¹² relative) breaks [smoothing.hpp:88-90].
- After acceptance the base sums are recomputed over the augmented sequence (`base = total()`), the paper's "treat the key set with the previous virtual point inserted as the new base" [smoothing.hpp:92].
- Output: `slot[i]` = rank + number of virtual points before key i, the virtual feature values, SSE before/after, rounds [smoothing.hpp:27-32, 94-95].

Features are doubles, so the same routine runs on normalised keys or on flow-transformed keys z(k) [smoothing.hpp:18]. Regions cap α < 1 [index.hpp:67]; the root may use α up to 64 because its "n" is the number of regions, not keys [index.hpp:69; smoothing.hpp:48].

#### 8.2 Taken: slot ranks instead of stored points

The virtual points are never materialised. In a region, `Region::rebuild` runs `smooth_cdf` on the region's features [index.hpp:304-308], fits `rank_model` on (feature, slot) instead of (feature, rank) [index.hpp:312], and records for each 128-key block its `slot_begin` = slot of the block's first key [index.hpp:83, 308]. A lookup computes y = rank_model(f(k)) and binary-searches the blocks' `slot_begin` values to pick the block (`predicted_block`, coordinate probes) [index.hpp:118-129], then `locate_block` corrects with exponential + binary search on the exact block fences (fence probes) [index.hpp:133-140]. Because the key arena never contains a virtual point, the only bytes they cost are the persisted `virtual_features` (8 bytes each, kept so compactions can re-place them) [index.hpp:101, 320]: at α = 0.1 that is ≈ 0.8 B/key, which is exactly the measured metadata increase, e.g. books_uniform 14.53 → 15.27 B/key, fb_uniform 11.39 → 12.19 [results/aidb_final/sweep/summary.csv initial_bytes_per_key]. The paper's virtual points, by contrast, occupy real slots in the host's gapped array and are what its Fig. 8 storage overhead measures. This also means our virtual points cannot host inserts.

Compactions do not re-run the search: the previous region's virtual features are re-placed among the new keys in O(n + v) and kept only if the augmented SSE is still ≤ the plain SSE, otherwise dropped [index.hpp:289-303]; `--relearn 1` forces the full search on every compaction [benchmark.cpp:178; Config::relearn_on_compaction, index.hpp:57-60].

#### 8.3 Taken: Algorithm 1 at the root ("virtual fences")

The root of our index is a single linear model over the region fences (one fence per 4,096-key region) that predicts the region, corrected by exponential + binary search on the fences [index.hpp:21-25, 348-357]. `fit_root` evaluates candidates {raw, flow} × {ranks, virtual-fence slots}; the virtual-fence candidate runs `smooth_cdf` on the fence features with budget `root_alpha × regions`, fits on slot targets, and builds a `root_slot_to_region_` table of n + |V| uint32 entries so that a predicted slot maps to a region in O(1) [index.hpp:378-388, 333, 342]. Each candidate is scored by the probes the real locate routine spends on the fences and fence midpoints; binary search wins if no candidate is cheaper [index.hpp:389-400]. So "virtual fences" = CSV Algorithm 1 applied to the ~489 fence features of a 2M sample (or 48,829 at 200M keys), with the same slot semantics.

#### 8.4 Not taken

- **Hierarchical Algorithm 2 and the cost model of eq. 22.** Our index has exactly two levels (root over regions, region model over blocks) and no subtree merging; there is nothing to promote [docs/APPROACHES.md:22 "Hierarchical Algorithm 2 (cost-model merges)" listed under not covered].
- **Gap materialisation for inserts** and the authors' ALEX/LIPP/SALI hosts [docs/APPROACHES.md:22].
- **The O(n + λ) complexity.** Ours is O(λ · n) per model: every greedy round rescans every gap after the refit [smoothing.hpp:10-13 "The paper's O(n + lambda) claim is not reproduced"]. Measured: 12–32 µs of thread time per key at α = 0.1 on 4,096-key regions (e.g. books_uniform 30.6, fb_uniform 18.0, osm_uniform 13.4 µs/key) [results/aidb_final/sweep/summary.csv preprocess_ns_per_key]; it grows with region size, fb 18.0 → 90.5 → 207.7 µs/key at 4,096 / 16,384 / 32,768 keys per region [results/aidb_granularity/summary_r*.csv], consistent with O(α·n²/regions). At full scale the 200M-key root at budget 4 × regions spent 2,067.78 s single-threaded inserting 195,316 virtual fences (the whole budget, 4 × 48,829) over planet's 48,829 region fences [MEETING_NOTES.md:118, 123-124; recomputed with `python3 results/aidb/root_analysis.py results/aidb_fullscale/sweep/results.jsonl`, row "planet root fences": virt 195316, build 2067.78 s].
- **The derivative formula (eqs. 17–21).** Replaced by four loss probes and a ternary search [smoothing.hpp:78-87]. Same purpose, different arithmetic; §4.3 above verifies the paper's formula independently.
- **Integer candidates.** Ours are real-valued features inside the gap [smoothing.hpp:18].

#### 8.5 What the control measured (numbers, not a reproduction of the paper's)

Final sweep: ten GRE datasets, 2M-key samples (uniform and contiguous window, seed 42), region 4,096 keys, block 128 keys, `--virtual-alpha 0.1`, 5M lookups per run, query seeds 11, 29, 47 (3 seeds), performance-core QoS, macOS arm64, Apple clang 17; variant `packed_rank_vp10` (policy min_bytes, routing rank, virtual-alpha 0.1) vs control `packed_rank` (same, no virtual points) [results/aidb_final/config.json: seeds [11, 29, 47], description "3 query seeds"; results/aidb_final/sweep/summary.csv: runs = 3 in all 180 rows; environment.json for the host/compiler]. Exact common flags of every run [results/aidb_final/config.json "common"]: load-ratio 1 (the whole sample is bulk-loaded), miss 0 (every lookup hits), query-distribution uniform, ops 5,000,000, warmup 500,000, verify 0, instrument 1 (software probe counters on), latency 0, build-threads 16, qos 1. Provenance caveat: results/aidb_final/sweep/environment.json embeds a DIFFERENT config than config.json: its description says "4 query seeds", its seed list is [11, 29, 47, 61] and it has an extra variant `packed_rank_fusion_auto` that has no rows in summary.csv; it was probably written by a later or aborted sweep launch. The numbers below rest on summary.csv (runs = 3), which agrees with config.json; do not cite environment.json as corroboration of the seed count.

| dataset | fence probes ctrl → vp10 | change | correction distance ctrl → vp10 | augmented-SSE ratio after/before | virtual points per key | B/key ctrl → vp10 | preprocessing µs/key | paired throughput speedup [bootstrap] |
|---|---|---|---|---|---|---|---|---|
| books_uniform | 2.230 → 2.008 | −10.0% | 0.108 → 0.018 | 0.046 | 0.0925 | 14.53 → 15.27 | 30.6 | 1.012 [0.892, 1.143] |
| books_window | 2.952 → 2.438 | −17.4% | 0.405 → 0.191 | 0.272 | 0.0999 | 14.03 → 14.83 | 15.2 | 1.014 [0.889, 1.112] |
| covid_uniform | 3.064 → 2.328 | −24.0% | 0.469 → 0.155 | 0.289 | 0.0995 | 13.78 → 14.58 | 25.1 | 0.963 [0.917, 1.033] |
| covid_window | 2.595 → 2.054 | −20.8% | 0.256 → 0.037 | 0.051 | 0.0987 | 12.94 → 13.73 | 29.9 | 1.001 [0.961, 1.029] |
| fb_uniform | 2.600 → 2.181 | −16.1% | 0.257 → 0.087 | 0.154 | 0.0999 | 11.39 → 12.19 | 18.0 | 1.006 [0.941, 1.104] |
| fb_window | 5.165 → 4.769 | −7.7% | 1.888 → 1.597 | 0.769 | 0.0999 | 10.50 → 11.30 | 12.4 | 0.968 [0.925, 1.010] |
| genome_uniform | 3.296 → 2.610 | −20.8% | 0.626 → 0.320 | 0.598 | 0.0999 | 11.56 → 12.36 | 21.9 | 1.053 [1.009, 1.087] |
| genome_window | 3.050 → 2.549 | −16.4% | 0.450 → 0.237 | 0.314 | 0.0999 | 10.68 → 11.48 | 13.5 | 1.041 [1.005, 1.103] |
| history_uniform | 2.458 → 2.072 | −15.7% | 0.204 → 0.045 | 0.177 | 0.0977 | 10.87 → 11.65 | 27.1 | 0.998 [0.879, 1.105] |
| history_window | 2.451 → 2.104 | −14.2% | 0.212 → 0.065 | 0.465 | 0.0974 | 10.04 → 10.81 | 27.7 | 1.013 [0.960, 1.057] |
| libio_uniform | 2.556 → 2.145 | −16.1% | 0.253 → 0.081 | 0.434 | 0.0978 | 10.36 → 11.15 | 25.8 | 0.963 [0.919, 1.011] |
| libio_window | 3.069 → 2.551 | −16.9% | 0.522 → 0.273 | 0.476 | 0.0850 | 9.43 → 10.11 | 20.8 | 0.998 [0.961, 1.020] |
| osm_uniform | 5.076 → 4.641 | −8.6% | 1.970 → 1.680 | 0.840 | 0.0999 | 14.25 → 15.05 | 13.4 | 0.918 [0.898, 0.949] |
| osm_window | 4.366 → 3.774 | −13.6% | 1.297 → 0.965 | 0.717 | 0.0999 | 13.48 → 14.28 | 15.2 | 1.005 [0.971, 1.039] |
| planet_uniform | 3.229 → 2.604 | −19.4% | 0.561 → 0.290 | 0.545 | 0.0998 | 10.79 → 11.58 | 18.0 | 1.062 [1.056, 1.074] |
| planet_window | 4.619 → 4.094 | −11.4% | 1.525 → 1.224 | 0.796 | 0.0999 | 10.27 → 11.07 | 12.9 | 0.961 [0.934, 1.012] |
| stack_uniform | 2.291 → 2.032 | −11.3% | 0.135 → 0.029 | 0.264 | 0.0922 | 10.20 → 10.94 | 30.1 | 1.006 [0.982, 1.022] |
| stack_window | 2.079 → 1.982 | −4.7% | 0.047 → 0.007 | 0.024 | 0.0310 | 9.19 → 9.43 | 14.1 | 0.967 [0.933, 1.004] |
| wise_uniform | 2.416 → 2.059 | −14.8% | 0.187 → 0.041 | 0.283 | 0.0975 | 12.09 → 12.87 | 31.7 | 0.937 [0.899, 0.985] |
| wise_window | 2.318 → 2.017 | −13.0% | 0.143 → 0.021 | 0.049 | 0.0978 | 11.26 → 12.04 | 31.5 | 0.968 [0.926, 1.013] |

[all cells: results/aidb_final/sweep/summary.csv, read_only profile, computed by scratch reads of the CSV; percentages derived from the two fence columns]

Reading: fence probes fall on all 20 samples, by 4.7% (stack_window, where the greedy stopped at 0.031 points per key because the set is already near-linear, SSE ratio 0.024) to 24.0% (covid_uniform). Apart from stack_window, the smallest reductions are where the per-region augmented-SSE ratio stays high, i.e. where 4,096-key regions are still far from one line after 10% virtual points: osm_uniform −8.6% (ratio 0.840), fb_window −7.7% (0.769), planet_window −11.4% (0.796). Do NOT map this onto the paper's easy/hard split: the paper's hard sets are OSM and Genome [CSV §6.1, p.9], and here genome_uniform (−20.8%) and genome_window (−16.4%) are among the LARGER reductions while fb (easy in the paper) has the second-smallest one; the paper itself reports that its hard sets promote MORE keys in absolute terms and save more total time [CSV §6.2.1, p.10], so "hard" in the paper's sense is not "gains least". Root probes are unchanged at 8.956 because region smoothing does not touch the root [summary.csv root_probes_per_operation]. Throughput moves by −8% to +6% with paired-seed bootstrap intervals that include 1.0 in 15 of 20 cases; the five exceptions are genome_uniform [1.009, 1.087], genome_window [1.005, 1.103] and planet_uniform [1.056, 1.074] (faster) and osm_uniform [0.898, 0.949] and wise_uniform [0.899, 0.985] (slower) [recounted from summary.csv seed_bootstrap_low/high, variant packed_rank_vp10; scratch/csv_fix_checks.py]. MEETING_NOTES.md:59 says "16/20 intervals include 1.0"; that count is not reproducible from the final summary.csv and 15/20 is the number to quote. The notes attribute effects below ~6% to host noise [MEETING_NOTES.md:57-61]. The CSV paper's "up to 34%" is on promoted keys of a multi-level host and is not comparable to any of these numbers.

Hardness-space view (2M uniform samples, whole-sample metrics after per-region smoothing at α = 0.1, `sample` → `sample_csv` scopes) [results/aidb/hardness.json]: PLA-32 segments fall on nine of ten sets (books 602 → 236, history 1,067 → 411, libio 1,256 → 614, stack 593 → 319, wise 787 → 436, fb 3,034 → 1,718, covid 1,192 → 892, genome 2,045 → 1,627, planet 3,456 → 2,767) and rise on osm (7,099 → 7,491); global-fit RMSE rises 6.5–10% on every set (e.g. books 180,591 → 196,846) because the augmented sequence has ≈ 10% more slots, so one global line's error measured in slot units scales up [our interpretation]; conflict degree changes by at most ±14 (history 7 → 21, libio 8 → 21, osm 1,102 → 1,099). For books the smoothing inserted 185,327 virtual points into 489 regions in 16.6 s wall-clock on 16 threads (the hardness tool times the parallel `smooth_regions` call as a whole, smoothing_ns = 16,575,434,458 with threads = 16; this is NOT a sum of per-thread times, unlike the benchmark's preprocess_ns_per_key, see §4.5) and cut the summed per-region augmented SSE from 7.17×10⁸ to 3.29×10⁷ [results/aidb/hardness/books_sample_csv.json `smoothed`, `threads`; src/hardness.cpp:172-173; hardness.hpp:369].

Root virtual fences (α_root = 4, i.e. budget 4 × regions, raw-key feature chosen) [results/aidb_final/root_analysis.json]: books_uniform root probes 8.956 → 2.600 with 662 virtual fences, total probes 16.22 → 9.86 (−39%), metadata +0.002 B/key; planet_uniform at 2M keys ALSO takes the virtual-fence root: chosen raw+fences with 1,956 virtual fences (the full 4 × 489 budget), root probes 8.956 → 5.139, total 17.21 → 13.40, throughput ×1.020 [0.991, 1.048] [results/aidb_final/root_analysis.json per_dataset.planet_uniform.packed_rank_root_vf4]; the notes report 8.96 → 2.1–2.8 on 16 of 20 samples and −22 to −31% total probes on planet_uniform, osm_window and planet_window [MEETING_NOTES.md:65-68]. At 200M keys (48,829 regions, single seed 11, 2M lookups) fb's root went 15.66 → 3.42 probes with 973 fences (the greedy stopped there under a 1,953 or a 195k budget), total 25.82 → 13.59. Planet at 200M: the raw-key root (estimate 25.8 vs 15.7 for binary) and the α = 0.04 root (1,950 fences, estimate 25.2) both fall back to binary search, but at budget 4 × regions the selector DID choose the virtual-fence root: raw+fences with 195,316 virtual fences (the whole budget), root probes 15.66 → 13.69 (−13%), total probes 24.68 → 22.70, single-seed throughput ×1.110, at the price of 2,067.78 s of single-threaded smoothing [recomputed with `python3 results/aidb/root_analysis.py results/aidb_fullscale/sweep/results.jsonl`, rows "planet root fences" and "planet root fences a=0.04"; MEETING_NOTES.md:123-125 says the same; the selector accepts any candidate whose estimated probe cost is below the binary-search cost, index.hpp:400]. The stored results/aidb_fullscale/root_analysis.json is stale: it lacks the planet packed_rank_root_vf4 run, so it shows planet with no chosen root; use the recomputation. Completeness: MEETING_NOTES.md:117 says 13 of 16 full-scale runs had finished; results.jsonl now holds 14 (planet's root_vf4 landed after the notes; planet root fusion and planet vp10 + root fusion are still absent). The honest one-line summary is: on fb, 973 virtual fences buy −78% root probes; on planet, the entire 195,316-fence budget buys −13%, so one global line over 48,829 planet fences is not smoothable at this budget. This is CSV's mechanism working at the only place our index has a "whole-range" model.

#### 8.6 Tests that pin the control's semantics

`tests/test_main.cpp:91-98`: augmented SSE never increases; at most ⌊α·n⌋ points; slots strictly increasing with `slot.back() == n − 1 + |V|`; every virtual feature strictly inside (min, max) and never equal to a key; α = 0 inserts nothing; α = 64 throws; an already-linear set {0..7} gets no virtual points. `tests/test_hardness.cpp:140-149`: per-region smoothing is deterministic across thread counts and rejects unsorted input. `tests/test_tools.py:19-23`: the stdlib toy `tools/virtual_points_lab.py` (exact/greedy on ≤ 4,096-wide integer keys, default keys 1,2,3,4,5,10,20,26,27,30, budget 3) satisfies greedy ≤ original and exhaustive ≤ greedy; that file is explicitly "NOT a reproduction of CSV's optimized algorithm" [tools/virtual_points_lab.py:2].

---

### 9. Questions the supervisor may ask, with answers

1. **"Why is the problem NP-hard?"** The paper reduces from Knapsack: candidates are items of unit weight, the budget λ is the capacity, the loss reduction is the value, and eq. 3 folds the interaction between points into a scalar r [CSV Lemma 3.1]. Say that plainly, then add that the sketch is thin: unit-weight Knapsack is polynomial, and the interaction is subset-dependent, so the paper's real support for hardness is the combinatorial size of the exact search, O(pCλ · n · p) [CSV §4.3], and the Table 2 timing. Do not claim more than the PDF does.

2. **"Why does the greedy work, and how well?"** Each round picks the single point that most reduces the refitted augmented SSE and then re-bases; on convex per-gap loss curves the interior minimum of each gap is found from the derivative sign at the two integer endpoints, so each gap contributes one candidate [CSV §4.2]. Quality: on ten keys, greedy 2.293 vs exhaustive 2.118 from 8.327 [CSV Table 2], i.e. 72.46% vs 74.56% reduction by the table's values (the text prints 72.34% / 74.44%, an inconsistency of the PDF, §7.2); on our seven keys, greedy 1.018 vs exhaustive 0.758 from 2.671 [§4.6]. There is no approximation guarantee in the paper.

3. **"What exactly is the loss being minimised?"** Eq. 4: SSE of the REFITTED line over real keys with shifted ranks plus the virtual points at their own slots, |V| ≤ λ [CSV eq. 4]. Not eq. 2 (fixed model), and not the real-key-only loss of Fig. 2's 2.04, which the paper reports as a side figure.

4. **"Where do the virtual points cost memory?"** In the paper, as empty slots in the host's node arrays: up to α·n slots, measured as < 10% index growth typically and < 31% worst case at α ≤ 0.8 [CSV §6.2.1]. In our control, zero arena bytes; only the 8-byte feature per virtual point kept for compaction re-placement, ≈ 0.8 B/key at α = 0.1 [index.hpp:320; summary.csv]. Root virtual fences cost 4 bytes per slot in the slot-to-region table [index.hpp:333, 495].

5. **"How does this interact with inserts?"** Paper: the gaps left by virtual points absorb inserts, which lowers storage overhead over time (LIPP ≤ 10%, ALEX < 0.5%) and keeps insert time on par (sometimes faster, sometimes slower because promoted keys collide more) [CSV §6.3]; CSV is not re-run after inserts in the experiments [CSV §6.1]. Ours: virtual points are not slots, so they cannot absorb inserts; inserts go to a delta and compactions re-place the old virtual features in O(n + v) or drop them if they stopped helping [index.hpp:289-303]; re-running the search on every compaction made 10%-insert runs 116–232× slower in the older synthetic sweep [READING_GUIDE.md:148-151].

6. **"Why did CSV pick a linear model and integer keys?"** Linear "for their efficiency" and because most learned indexes use them; integer keys make the candidate set discrete runs between consecutive keys, which is what the endpoint-derivative filter needs; reals are scaled to integers [CSV §1, §3].

7. **"What does the O(n + λ) claim rest on?"** On computing the K-sums once (O(n)) and O(1) per candidate via eqs. 10–16, plus the assumption that each round examines O(1) candidates; the PDF does not justify the last part. Our implementation rescans all gaps per round and is O(λ · n), 12–32 µs/key at α = 0.1 on 4,096-key regions [smoothing.hpp:10-13; summary.csv]. The paper's own preprocessing is 889–2,902 s at α = 0.1 on 200M keys for ALEX, 337–2,329 s for LIPP [CSV Tables 3–4].

8. **"What is the cost condition and why does it differ per host?"** LIPP/SALI have no in-leaf search, so a lower loss with more keys in the node is enough; ALEX searches inside leaves, so eq. 22 (search_constant × expected searches + traversal_constant × level, threshold c < 0) trades leaf search against traversal [CSV §5.1]. Constants are machine-measured; values not reported.

9. **"Which α should one use?"** The paper's default is 0.1; more α gives more time saved and more promotion, monotonically, but space grows proportionally and, while α grows 16× (0.05 → 0.8), preprocessing grows by 3.8× (LIPP Facebook 589 → 2,228 s) to 13.6× (LIPP Genome 1,155 → 15,709 s) on LIPP and by 7.2× (ALEX Genome 1,356 → 9,777 s) to 197× (ALEX Facebook 247 → 48,737 s; ALEX OSM 988 → 81,620 s is 82.6×) on ALEX; Covid on LIPP barely moves (304 → 336 s) [CSV Tables 3–4; ratios in scratch/csv_fix_checks.py]. Easy sets plateau (LIPP/SALI on Facebook/Covid) so extra α buys nothing there [CSV §6.2.1].

10. **"Why does your RMSE go UP after smoothing while PLA-32 goes down?"** The hardness scope `sample_csv` measures one global line over the augmented sequence, which has ≈ 10% more slots, so rank-unit errors scale up (+6.5 to +10% on every set); PLA-32 counts ε-bounded segments, which is a local-linearity measure and it falls by 19.9% (planet 3,456 → 2,767) to 61.5% (history 1,067 → 411; books 602 → 236 is 60.8%) on nine sets and rises 5.5% on osm (7,099 → 7,491) [results/aidb/hardness.json sample.pla_32 vs sample_csv.pla_32; scratch/csv_fix_checks.py]. The index's region models see the local effect, which is why fence probes fall.

11. **"Is your fence-probe reduction comparable to the paper's 34%?"** No. The paper measures average query time on promoted keys in a 3-to-7-level host; we count key comparisons in a two-level control with exact fences. The honest comparable statement is directional: CSV-style smoothing lowers the local model error and the correction work on all ten datasets, most on the sets whose regions are far from linear, least on osm/fb-window where the SSE ratio after smoothing is still 0.77–0.84 [summary.csv].

12. **"Why virtual points at the root, and what happened on planet?"** The root is the only whole-range model, so it is where a global smoothing can pay off. On fb, 973 virtual fences take root probes 15.66 → 3.42 at 200M keys (−78%). On planet the selector does NOT keep binary search at budget 4 × regions: it picks the virtual-fence root because 13.69 < 15.66 [index.hpp:400], but the whole budget of 195,316 virtual fences over 48,829 fences buys only −13% root probes (total 24.68 → 22.70) after 2,067.78 s of smoothing; only the cheaper α = 0.04 root (1,950 fences, estimated 25.2 probes) and the raw-key root (25.8) fall back to binary [recomputed fullscale root analysis; MEETING_NOTES.md:123-125]. At 2M keys planet_uniform's root_vf4 is chosen too, 8.956 → 5.139 with 1,956 fences [results/aidb_final/root_analysis.json]. So the honest phrasing is "planet's fences are not smoothable into one line at this budget: the entire budget for −13%, versus fb's −78% with 973 fences". Do not present this as the paper's OSM finding; the paper reports that its hard sets gain MORE total time because more keys are promoted [CSV §6.2.1].

13. **"Where is the paper weakest?"** No code; the hardness proof; the complexity claim; constants of eq. 22 not reported; storage overhead stated as < 15% in the introduction but < 31% worst case in §6.2.1; results only on promoted keys, not on the whole workload; single machine; deletes not covered. [all from the PDF as cited above]

14. **"How would you verify the derivative formula if asked on the spot?"** Insert one virtual point into a wide gap, compute the loss at k_v ± h by refitting, take the central difference, compare with eqs. 17–21; we did it: −1.132330 at k_v = 6 and +0.886966 at k_v = 11 by both routes [scratch/csv_eq_check.py].

---

Scratch scripts (standard library only) used for every computed number in this section: `results/aidb/guide_deep/scratch/csv_worked_example.py` (7-key example, greedy, exhaustive), `results/aidb/guide_deep/scratch/csv_eq_check.py` (eqs. 6–9, 15–21 against finite differences) and `results/aidb/guide_deep/scratch/csv_fix_checks.py` (eq. 9 with original vs shifted ranks, eq. 14 limits, secant minimum point, ρ before/after, Algorithm 1's candidate set per greedy round, Table 2 percentages, Tables 3–4 growth ratios, PLA-32 drops, bootstrap-interval recount, per-region operation counts). The rendered page 10 of the CSV PDF used for the Fig. 6/7 readings is `scratch/csv_p10_top.png`.


---

## 03. How Hard Can Indexing Be? (Zhang, Tang, Ailamaki, AIDB@VLDB 2026), deep

What this section gives you: a self-contained account of the AIDB hardness paper at the level of the equations and of the numbers, so that you can define every one of its five scalar metrics on a napkin (with an eight-key example computed here), state the harder-than partial order, coverage and conformance exactly (equations 1-3, with worked examples), reproduce Table 2 and read it the way the authors do (Findings 1-7), describe the six indexes, the ten datasets and the measurement protocol precisely, and then say exactly which parts of that paper our study reuses (the ten datasets, a clean-room reimplementation of the five metrics, and a clean-room reimplementation of the conformance/coverage scorer applied to our own SCALE-LI control variants) and what matched. Every number is cited to the PDF, to the code, or to a result file; anything read off a figure or not stated in a source is tagged. Worked numbers were computed with `scratch/worked_examples_03.py` and `scratch/table2_gains_03.py` (standard library only; outputs in `scratch/worked_examples_03.out.txt`). Notation: keys k₁ < … < kₙ (uint64), rank r(kᵢ) = i − 1; the paper writes S = ⟨x₁ < … < x_N⟩ and R(x) = |{xᵢ ≤ x}| so that R(xᵢ) = i; the offset of one is absorbed by the intercept of every linear fit and changes none of the metrics.

---

### 1. Bibliographic facts and the question the paper asks

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

### 2. The GRE metric (§2.2): PLA-32, PLA-4096, and why it fails

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

### 3. The five scalar metrics, with an eight-key worked example

Setting common to all five [AIDB §4.1]: the input is the sorted key set; every metric is O(N) on sorted keys; "they typically take 10-20 seconds to compute for each dataset". Our tool computes all five on the 200M-key files in 12.9 s (history) to 27.4 s (wise) wall clock with 16 threads, of which 2.7 s (history) is the scalar-metric pass proper [results/aidb/hardness_details.json <name>.full.elapsed_ns; results/aidb/hardness/history_full.json original_metrics_ns].

The toy set used below (8 keys, so LIPP's small-node parameters apply; nothing about it is representative of 200M keys, it exists to make every formula concrete):

```
keys  k = [ 1,  2,  3,  4, 10, 20, 30, 100]
ranks r = [ 0,  1,  2,  3,  4,  5,  6,   7]     (n = 8)
```

#### 3.1 RMSE and ME: one least-squares line key → rank

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

#### 3.2 CD, the conflict degree, as LIPP's FMCD defines it

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

#### 3.3 PLA-ε: segments of the optimal ε-bounded piecewise linear approximation

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

### 4. Metric compositions, the harder-than order, coverage and conformance

#### 4.1 The 25 metrics of Table 2

[AIDB §4.1]: the five scalars; "A·B is a two-dimensional metric. It composes two scalar metrics, A and B, chosen from the list above"; "A·B·C composes three scalar metrics". So the corpus is 5 + C(5,2) + C(5,3) = 5 + 10 + 10 = 25 metrics, and GRE is the special name of PLA-32·PLA-4096 [AIDB Table 2]. Enumerated in the paper's order [AIDB Table 2; tools/aidb_scores.py:52-57 generates the same list]:

```
scalar (5):   RMSE, ME, CD, PLA-32, PLA-4096
2-dim  (10):  GRE = PLA-32·PLA-4096, RMSE·ME, RMSE·CD, RMSE·PLA-32, RMSE·PLA-4096,
              ME·CD, ME·PLA-32, ME·PLA-4096, CD·PLA-32, CD·PLA-4096
3-dim  (10):  RMSE·ME·CD, RMSE·ME·PLA-32, RMSE·ME·PLA-4096, RMSE·CD·PLA-32, RMSE·CD·PLA-4096,
              RMSE·PLA-32·PLA-4096, ME·CD·PLA-32, ME·CD·PLA-4096, ME·PLA-32·PLA-4096, CD·PLA-32·PLA-4096
```

Composition does not combine the numbers (no weighted sum): a composed metric is a *vector*, and vectors are compared by the partial order below. Four- and five-dimensional compositions are not evaluated; the paper gives no reason [not reported].

#### 4.2 The harder-than partial order, C and U

Notation [AIDB §3.2]: dataset corpus 𝒮 = {S₁, …, Sₙ} (n = 10), index corpus ℐ = {I₁, …, I_m} (m = 6), a d-dimensional metric h with components h_k. Definition (exact): **Sᵢ is harder than Sⱼ when h_k(Sᵢ) ≥ h_k(Sⱼ) for all k and h_k(Sᵢ) > h_k(Sⱼ) for at least one k.** Two datasets are *comparable* when one is harder than the other. C = {(i, j) | Sᵢ is harder than Sⱼ} is the set of comparable pairs *as ordered pairs* (harder first); U = {(i, j) | i < j ∧ Sᵢ, Sⱼ incomparable} is the set of incomparable *unordered* pairs. Every unordered pair of distinct datasets is in exactly one of C (once, with its orientation) or U, so |C| + |U| = C(n, 2) = 45 for ten datasets. Two consequences that are easy to miss: (i) a pair equal on *every* dimension is incomparable even for a scalar metric (neither is strictly harder), so a scalar metric does not automatically have coverage 1: on our 2M uniform samples covid, libio, stack and wise all have CD = 8, which makes the six pairs among them incomparable and gives scalar CD a coverage of (39 − 6)/45 = 0.733 [results/aidb/scores.json sample.metrics.CD]; (ii) a 2-D pair is incomparable exactly when the two dimensions disagree (or one ties and the other does not decide the other way), which is why adding a dimension "can only make some previously comparable pairs incomparable, but not the opposite" [AIDB §4.2, before Finding 7].

Implementation: `harder` and `classify_pairs` [tools/aidb_scores.py:82-97] are a literal transcription (`all(>=) and any(>)`).

#### 4.3 Coverage (equation 3) with a worked example

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

#### 4.4 Conformance (equations 1-2) with a worked example

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

### 5. Setup (§4.1): indexes, datasets, hardware, workload

#### 5.1 The six learned indexes

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

#### 5.2 The ten datasets (Table 1) and what they are like

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

#### 5.3 Hardware and software

[AIDB §4.1, "Testbed and Configuration"]: a server with two Intel Xeon Gold 5118 CPUs (12 cores per socket, 2.3 GHz) and 384 GiB of RAM; a container running Ubuntu 20.04.6 on host Linux kernel 6.8.0-57-generic; hugepages disabled; "All indexes are ported in a common codebase and compiled with GCC 9.4.0 using the -O3 flag"; default configuration for each index "when applicable", with bug fixes "applied when necessary"; "The worker thread is pinned to a fixed CPU core for stability". Not reported: which core, whether the other socket was idle, memory allocation policy, index-specific parameters, the version or commit of each index.

#### 5.4 Workload and how throughput is measured

[AIDB §4.1, "Workloads and Measurements"]: "We use index lookup throughput as the primary performance metric, as write operations introduce performance overhead that is not directly related to dataset hardness. For each index and dataset, we first bulk load the index and then perform random lookups for all keys in the dataset. For each run, we start with a warm-up phase of 20 million lookups and then a measurement phase of 100 million lookups, from which we calculate the throughput." Combined with §2.2 ("uniform read-only workload over the same dataset used to populate the index"):

- bulk load of all 200M keys, then lookups only (no inserts, no range scans);
- lookup keys drawn from the dataset's own keys (every lookup hits; "random lookups for all keys" reads as uniform random over the key set, sampling with or without replacement not stated [not reported]);
- 20M warm-up lookups discarded, 100M measured lookups, throughput = 10⁸ / measured time (in MOPS);
- one worker thread pinned to a core (the singular "worker thread" implies single-threaded measurement [inference]);
- number of repetitions per (index, dataset), variance, and whether the reported value is a mean or a median: not reported.

The only absolute throughput numbers printed are the LIPP anecdote (7.1 MOPS on libio, 5.6 on history) [AIDB §2.2]; everything else enters the paper only through p̂_I(S) = p_I(S)/σ_I in the conformance score.

---

### 6. Results (§4.2): Table 2, Findings 1-7, and the open problems of §5

#### 6.1 Table 2, reproduced

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

#### 6.2 The conformance-coverage plane and the approximate Pareto frontier (Figure 4)

Figure 4 [AIDB p. 5] plots (Conf, Cov) for the 25 metrics: the five scalars sit on the Cov = 1 line at Conf 0.08-0.50, the 2-D metrics at Cov 0.42-0.82 and Conf 0.25-0.71, the 3-D metrics at Cov 0.16-0.51 and Conf 0.50-0.83 [values from Table 2]. The "approximate Pareto frontier" is computed by a greedy, beam-search-like procedure [AIDB §4.2]: a scalar metric is nothing but an ordering of the ten datasets, so all 10! = 3,628,800 orderings are enumerated and scored; the top-1k by conformance become bases to which a second dimension (again any of the 3,628,800 orderings) is added; the 2-D Pareto-optimal metrics become bases for a third dimension, and so on up to six dimensions, "as it is guaranteed to achieve a conformance of one with six dimensions, each corresponding to the ordering given by the performance of one index" (six dimensions = six indexes: each dimension copies one index's throughput order, so no pair can violate). The frontier is the Pareto front of everything explored. It is an upper bound achievable by *any* ordering, including orderings no computable feature of the data could ever produce [AIDB §5].

#### 6.3 Findings 1-7 with their evidence

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

#### 6.4 Open problems (§5)

1. *Exact and tight Pareto frontier.* The greedy frontier is an approximation in two senses: being close to it "does not necessarily mean a metric is close to a true optimal metric", and even the exact frontier is "the upper bound achievable by any ordering of the datasets, including orderings that no computable feature of the data could ever reproduce". A "tight frontier that is realizable by some computable metric" is undefined so far [AIDB §5].
2. *Workload adaptation.* The tradeoff is inherent only for a workload-agnostic metric; since indexes favour different indicators (Findings 3-4), a metric specialised to the user's target indexes "pushes the frontier outward, rather than merely selecting a point on it". Proposal: a workload-aware metric selection framework taking target indexes as input and weighting the conformance objective toward them [AIDB §5].
3. *Filling gaps in the hardness space.* Under GRE and CD·PLA-32 (Figures 2 and 6) the ten datasets cluster, leaving regions with no dataset. Wongkham et al.'s generator for GRE targets was used to generate ten datasets matching the ten real GRE values; Table 3 [AIDB p. 7] reports that they neither match the target local hardness (differences from 0.00 % on stack to −12.23 % on genome; global hardness matched exactly) nor reproduce index behaviour: ALEX throughput differs by −34.23 % (books) to +63.84 % (fb), LIPP by −59.30 % (stack) to +64.32 % (osm). The generated sets were excluded from the main evaluation to avoid bias [AIDB §5, Table 3, footnote 1]. Conclusion: "dataset generation algorithms that reliably realize target hardness values for a wide range of metrics are needed" [AIDB §5].

---

### 7. How our study uses this paper

Everything in this section is a clean-room CONTROL study inside the SCALE-LI experimental map. We did not run RMI, PGM-index, ALEX, LIPP, XIndex or FINEdex, we did not run the paper's code (none is published with the paper), and nothing below is a reproduction of Table 2 or of NFL, AFLI or CSV. What we reuse from the paper is (a) its ten datasets, (b) the definitions of its five scalar metrics and (c) the definitions of its two scores; what we apply them to is our own index and our own variants.

#### 7.1 The ten datasets, at two scales

- *Full files* (200M keys) are used for the hardness metrics only [results/aidb/run.json scale_note]. Seven were sorted once with `scaleli_hardness --sort-only 1 --write-sorted` (parallel chunk sort and merge, identical to `std::sort` [hardness.hpp:50-76]); no duplicates anywhere [results/aidb/provenance.json].
- *2M-key samples* are what the index runs load, in two modes [tools/datasets.py:73-83; run.json settings]: `uniform` = 2,000,000 sorted random indices drawn by `random.Random(42).sample(range(200_000_000), 2_000_000)` (keeps the global shape, flattens local structure); `window` = one contiguous run of 2,000,000 keys starting at a seeded random offset (keeps local structure, loses the global shape). Every sample carries a manifest with the source sha256 and its own sha256 [data/samples/<name>_2M_{uniform,window}_s42.manifest.json]. The two modes disagree on local hardness: fb's uniform sample has PLA-32 = 3,034, its window 10,599 [results/aidb/hardness.json fb.sample.pla_32; results/aidb_window/hardness.json fb.sample.pla_32].

#### 7.2 The hardness tool: the five metrics reimplemented

`scaleli_hardness` [src/hardness.cpp] reads a SOSD file, checks sortedness and duplicates, and calls `compute_metrics` [hardness.hpp:323-333], which runs the four scans in parallel: least squares (RMSE, ME), FMCD fit plus CD, and one PLA pass per ε in `--pla-eps 32,4096`. The exact command per dataset, as recorded next to each output [results/aidb/hardness/history_full.json.command.json]:

```
scaleli_hardness --data data/external/gre/history.sorted --flow results/aidb/flows/history_2D2H2L.txt \
                 --dtype uint64 --region-keys 4096 --pla-eps 32,4096 --check-sorted 1
scaleli_hardness --data data/samples/history_2M_uniform_s42 --virtual-alpha 0.1 --dtype uint64 \
                 --region-keys 4096 --pla-eps 32,4096 --check-sorted 1          # and a third run: sample + --flow + --virtual-alpha 0.1
```

[tools/aidb_pipeline.py:449-455 `hardness_jobs`]. This yields six *scopes* per dataset in `results/aidb/hardness.json`: `full` (raw 200M keys: the block compared with the paper), `full_flow` (the same keys after our NFL-style transform, sorted), `sample`, `sample_flow`, `sample_csv` (the sample after CSV-style virtual points at α = 0.1 per 4,096-key region; the metrics are computed on the augmented sequence of real plus virtual features [hardness.hpp:335-377]), `sample_flow_csv` (both). Sections 04-06 use the non-`full` scopes to show which hardness axis each component moves; this section only needs `full`.

Cost on this machine: 12.9-27.4 s per full file (16 threads, including 0.6-0.8 s of reading and, for the `--flow` runs, 0.65 s of transform) [results/aidb/hardness_details.json <name>.full.elapsed_ns; results/aidb/hardness_timing/fb_full.json]; the paper's 10-20 s per dataset is in the same range [AIDB §4.1].

#### 7.3 The scorer: equations 1-3 reimplemented and applied to our variants

`tools/aidb_scores.py` implements the protocol literally: `harder` [:82], `classify_pairs` [:87], `coverage` [:100], `conformance_of_variant` [:105] (sigmoid weights, R, P, (R − P)/(R + P)), `score_metric` [:123] (per-variant Conf and their mean), `score_scope` [:153] (population std normalisation, corpus = the datasets present in the hardness scope and in every variant's throughput). It was checked against an independent re-implementation written from the paper text on random and adversarial inputs, with agreement within the verifier's 1e-9 tolerance on every score [results/aidb/verification/verdicts.json, scores verdict; results/aidb/verification/scores/ref_scores.py, verify_scores.py]; MEETING_NOTES quotes 4e-16 [unverified: the archived verdict text I read states the 1e-9 tolerance, not the observed maximum].

The "index corpus" ℐ in our application is not the paper's six indexes but our ten control variants, all instances of the same SCALE-LI map with different options [tools/aidb_pipeline.py:52-66; results/aidb/scores.json full.variants]: `sorted_vector` (binary search, no index), `raw_rank`, `packed_rank` (the control), `packed_rank_flow`, `packed_rank_flow_forced`, `packed_rank_vp10`, `packed_rank_flow_vp10`, `packed_byte`, `packed_rank_fusion_auto`, `packed_rank_flow_costsel`. Their throughput p_I(S) is the median over three query seeds (11, 29, 47) of 1,000,000 uniform lookups (all hits) after 200,000 warm-up lookups on the 2M-key sample, single-threaded, from `throughput_ops_s` of `scaleli_bench` [tools/aidb_pipeline.py:159-171 `median_throughput`, :497-504 `sweep_config`; src/benchmark.cpp:140; results/aidb/throughput.json]. Scale reminder: the paper measures 100M lookups over 200M keys; we measure 1M lookups over 2M keys [results/aidb/run.json scale_note].

Consequence for reading our scores: the *coverage* column depends only on the hardness table, so for the `full` scope it is directly comparable with Table 2; the *conformance* columns are scores of our own variants at reduced scale and say nothing about the six indexes of the paper.

#### 7.4 What matched, and what did not

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

#### 7.5 Where the rest of the story is

Section 04 (host index and controls) describes the SCALE-LI map, `Region`, `Index`, the probe counters and the variants; section 05 the NFL-style transform and its bypass; section 06 the CSV-style virtual points and the region/root selectors; section 07 the experiments E1-E7 and their numbers, including how each component moves the five metrics across the six scopes (transform: RMSE −98 % on fb to +111 % on stack, PLA-32 by at most 5 segments; virtual points at α = 0.1: PLA-32 −43 to −61 % on six uniform samples, RMSE +6 to +10 %) [results/aidb/MEETING_NOTES.md §1; results/aidb/hardness.json].

---

### 8. Questions the supervisor may ask, with answers

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


---

## 04. The ten datasets and our samples

**What this section gives you.** Everything about the data behind the SCALE-LI AIDB lane: where the ten
200M-key files come from and how we checked them (URLs, sizes, sha256, sorting, duplicates); what each
dataset's keys are according to the AIDB 2026 paper and, where those papers describe them, NFL and CSV;
the exact key ranges and full-file hardness numbers (RMSE, ME, CD, PLA-32, PLA-4096) of every dataset,
with and without the NFL-style transform; a master table sorted by PLA-32 with the two values the AIDB
paper prints marked; the exact recipe for the two 2M-key samples (uniform, seed 42; contiguous window at
a seeded offset) and what each sampling mode preserves or destroys, calibrated against a control sample
of a perfectly linear file; every sample's hardness in the four sample scopes and its mean per-region
tail conflict degree; how each sample's flow was trained and what the transform does to the metrics; how
the CSV-style augmented scopes were produced; and a list of questions the supervisor may ask, with
answers. Every number is copied from the PDFs, the code or the result files, cited inline in brackets;
worked numbers were computed with the scratch scripts under `results/aidb/guide_deep/scratch/`
(`data_stats.py`, `tables.py`, `linear_uniform_hardness.json`). Our index components are clean-room
CONTROLS inside the SCALE-LI experimental map; nothing here is a reproduction of NFL, AFLI, CSV or the
AIDB benchmark.

Notation: keys k₁ < … < kₙ (uint64), rank r(kᵢ) = i − 1 ∈ {0..n−1}; linear model f(k) = w·φ(k) + b with
feature φ (raw key, or flow output z(k)); slot s(kᵢ) = rank plus the number of virtual points before kᵢ;
virtual budget λ = α·n; conflict degree D and tail conflict degree D₉₉; region = 4,096 keys; block =
128 keys.

---

### 1. Provenance: the ten GRE files

#### 1.1 Where the bytes come from

- URL pattern: `https://www.cse.cuhk.edu.hk/mlsys/gre/<name>` for `<name>` ∈ {books, fb, osm, covid,
  genome, history, libio, planet, stack, wise} [results/aidb/provenance.json `<name>.url`]. The list and
  the URL come from the GRE benchmark's own `datasets/download.sh`
  (`https://github.com/gre4index/GRE/blob/master/datasets/download.sh`) [provenance.json `<name>.source`];
  our resumable copy of that script is `data/external/gre/download.sh` (curl with `--retry 8`, four
  parallel downloads).
- Why these ten: they are the ten datasets of Table 1 of the AIDB paper [AIDB Table 1, p.4], which the
  paper describes as "a superset of the datasets used in recent work on learned indexes", each containing
  "200 million 64-bit unsigned integer keys without duplicates" [AIDB §4.1, "Datasets"]. GRE (Wongkham
  et al., PVLDB 2022, AIDB ref. [27]) is where the paper's hardness baseline (PLA-32 / PLA-4096) comes
  from [AIDB §2.2].
- File format ("sosd"): an 8-byte little-endian count followed by count × 8-byte little-endian uint64
  keys [tools/datasets.py `header`, `sample`]. Size check: 8 + 8 × 200,000,000 = 1,600,000,008 bytes,
  which is the `bytes` field of all ten entries [provenance.json `<name>.bytes`] and the on-disk size
  (`ls -l data/external/gre/`).
- Retrieved 2026-09-21 between 14:47 and 14:59 UTC; sha256 verified 15:17 UTC the same day
  [provenance.json `retrieved_utc`, `verified_utc`].

#### 1.2 Sortedness, duplicates and the `.sorted` copies

The GRE loader sorts at load time, and seven of the ten files are served **unsorted on disk**: covid,
genome, history, libio, planet, stack, wise [tools/aidb_pipeline.py:42 `GRE_UNSORTED`, audited
2026-09-21; hardness.hpp:50-53]. The pipeline's `sort` step runs `scaleli_hardness --sort-only 1
--write-sorted` on every file, records `sort_audit = {sorted, duplicates, written, written_keys}` and
writes `data/external/gre/<name>.sorted` for the unsorted ones [aidb_pipeline.py:374-380;
src/hardness.cpp:93-95]. The sort is a chunked `std::sort` + pairwise `std::inplace_merge`, identical in
result to `std::sort` [hardness.hpp:57-77]. Every step after `verify-downloads` uses the sorted copy when
it exists, else the download [aidb_pipeline.py:278-283].

Audit result: `sorted: true` for books, fb, osm; `sorted: false, written_keys: 200000000` for the other
seven; **`duplicates: 0` for all ten** [provenance.json `<name>.sort_audit`]. So "without duplicates"
[AIDB §4.1] holds for the files as served, and n = 200,000,000 exactly in every full-file metric below.

**T1. Checksums.** Download sha256 [provenance.json `<name>.sha256`]; sorted-copy sha256 [provenance.json
`<name>.sample.source_sha256`, which is the sample step's hash of the file it read]; sample sha256
[`data/samples/<name>_2M_{uniform,window}_s42.manifest.json` `sample_sha256`].

| dataset | served sorted? | download sha256 | .sorted sha256 | uniform sample sha256 | window sample sha256 |
|---|---|---|---|---|---|
| books | yes | `c71eb78eebb746f0e525e47d3d7e6b097f680c45cf17f1ead1d1f22a8950f939` | (same file) | `c2484df60c8b2f3100c408330453e0aca321ba5af6d0d23312a63bdd567f3eca` | `76fb950be335b5bcf4ee01fe615edacd3581441385d130df07f3ed571b97884a` |
| fb | yes | `22d5fd6f608e528c2ab60b77d4592efa5765516b75a75350f564feb85d573415` | (same file) | `7ecf95742784e7f3ef7bae5c5f3ac5c5924126a023927cb90e371d0074aeac06` | `a26c9b3809d2089a0efeff0ec90cdcc84a40fdc81203b0a4c7600433e0858b8c` |
| osm | yes | `1d1f5681cbfcd3774de112533dccc4065b58b74adf09684e9ad47298d1caa9e0` | (same file) | `b832a46ba727d96dd16acd70e909b4dc9a232dad0d116b850a2373a865d84ceb` | `7c3c37cb82936d9741a4451de9aeba95d00f95191a1631c89611862c79d9f67c` |
| covid | no (sorted copy written) | `96834e0f0a9a697ac39ec30f37316b9cf092056bcd9332e59ce131e891c0a3a8` | `cdc34a55f609691b57e316fd11c46253477df63b1296a7f05314e701d7c342ba` | `5dee4e179c89b1e59ddba6578bc96614267d3bfc3e91190acb33447f405ff268` | `62128272374874002adc29b6fddccd9ee408b278f602e7b028ed959167356644` |
| genome | no (sorted copy written) | `6862bff81727ec2bd0df54e9e08c60758ef8b3ee9bd800ffdc4eb68bd7680daa` | `2b0b5ae57a5cd00f504eb8226801ab8311cec1a2e3fb0ec3594b5bb38457c739` | `fb507fcd79f8bddfb0d2b93f1f087e02c9eed8f2c7cc573a994e5289cf606c24` | `61dc34adb69499a2c439240c3591b689a9fd967ce61e615ca3fbef6479651a2a` |
| history | no (sorted copy written) | `2556096e02969666fc4f48737330b1ec2788588481f54181816563f193efedd4` | `c2662355dd437e5d6f4a8e581a18cee07c2b2c184634afe05dd8a0e59311cd42` | `186080bdf75a972ba156edbc681fa9c49f0458bf7f61f9b1037cfcacc86ee662` | `997e2743bdbc42f5f5e31a8c978d29d2905b51141876898d9db5390a9e70d55e` |
| libio | no (sorted copy written) | `0d0c99a064a1142c48f4a5887429f272113ed8daa2f407072f3d5c2f18b2975b` | `f9f868375134273548644708f42365b4d3e0b952b5ce211f205b822e9440aea0` | `9156ef37030a4861635ec9f621b7678b7f058c2ce68400e4b67f0a9de7fdb237` | `4f911d41d6c4579014c4d7e745075e6bb33b68ce6759d8923642c4335e0a8362` |
| planet | no (sorted copy written) | `0e4a90c81e6e36f198aa674e9773c366e55e27430b208780d559a25e97324ef5` | `d6680d134075890173d5a9d5f2af991eb13ed794da153b5a67d90cec98a4628b` | `3a0fcf47ba903c33e9f6c6ed9bdc233918489751dbc0b9a41b0f5e205125c045` | `e15c214776a67511ec4211f7c888776cb71ebcb165b4569bf426fc9aa1a7c8e2` |
| stack | no (sorted copy written) | `34f9a3ae69ac17ed80a054d3f8ed9cebf0d86dca61eb502ca634daea9469a09d` | `1e261e45f0058731afb84a2caa37fec957e49fa5c4c622333696f43a1530bacc` | `1cb323ad7b26ed8bb637684f2a12b5558e6a052520504eafa82b0644c490abdb` | `7b3764a00f6e4f94883127deb13e31c919afb0a1855fdd87d0e0ff60b02b120a` |
| wise | no (sorted copy written) | `87e34fb1503881b19e2f172fee4855919544432e79b96f6bceed33e5c9805dbb` | `6353bad843367ceda822189c223b6ecea76cd606d824f9483d2a45fb7afa7f88` | `b5b5240959dc12f187b487b25c1d1dc3e08c715994d7a85827effeb2eadf0709` | `9d038460726c16aad533cee4d1a9527937735e8890a9a0e0bdb82215008abd6a` |

Note the window-sample sha256 values are not in `provenance.json` (the provenance entry records the
uniform sample only, because `results/aidb/` was the uniform run); they come from the window manifests.

---

### 2. The metrics used below, exactly as computed

All five hardness metrics follow the AIDB paper's list [AIDB §4.1 "Metrics under Test"] and are computed
by `scaleli_hardness` on the sorted feature sequence x₀ ≤ … ≤ xₙ₋₁ with rank target yᵢ = i
[hardness.hpp:1-10]. Raw uint64 keys are handled with exact `__int128` differences; flow outputs and
smoothed slots are doubles [hardness.hpp:31-34; src/hardness.cpp:100].

```
RMSE, ME   least-squares line y ≈ a·(x − x₀) + b over all n points (Welford accumulation);
           RMSE = sqrt( Σᵢ (i − a·(xᵢ−x₀) − b)² / n ),  ME = maxᵢ |i − a·(xᵢ−x₀) − b|     [hardness.hpp:84-106]

CD         LIPP's FMCD bulk-load model (Wu et al. 2021, lipp.h build_tree_bulk_fmcd), capacity
           L = n·(gap+1) with gap = 1 for n ≥ 1e6 (so L = 2n), search for the smallest D such that
           every D-apart pair of keys is at least U_T = (k[n−1−D] − k[D])/(L−2) + ε apart, ε = 1e-6
           on integer keys; slot(k) = floor(a·(k − anchor) + L/2) clamped to [0, L−1] with a = 1/U_T;
           CD = the longest run of consecutive keys with the same slot.       [hardness.hpp:109-227]
           On real-valued features ε is scaled to 1e-6 × mean gap (documented deviation from LIPP's
           absolute 1e-6, which would dominate O(1)-range flow outputs).      [hardness.hpp:151-166]

PLA-ε      number of segments of the optimal ε-bounded piecewise-linear approximation (O'Rourke 1981),
           counted exactly as PGM-index's make_segmentation does, including the (x_last+1, n) sentinel;
           ε ∈ {32, 4096} = the GRE local/global dimensions.                  [hardness.hpp:230-312]

tail CD    NFL Definitions 3.1/3.2: fit one line feature→rank, floor predictions into
D₉₉        [0, 1.5·n) slots, count keys per occupied slot, take the value at the 99th percentile of the
           occupied-slot counts and subtract one (NFL: "t = INT(m × γ)", γ = 0.99 [NFL §3.1.1]).
                                                              [transform.hpp:84-108; train_flow.py 71-86]
```

The pipeline computes the metric blocks with `--region-keys 4096 --pla-eps 32,4096 --check-sorted 1`
[aidb_pipeline.py:451]. Scopes in `hardness.json` map to `scaleli_hardness` runs as follows
[aidb_pipeline.py:449-454, 472-474]:

| scope | run | input | block |
|---|---|---|---|
| `full` | `--data <full> --flow <flow>` | 200M sorted keys | `original` |
| `full_flow` | same run | z(k) of the 200M keys through the SAMPLE-trained flow | `transformed` |
| `sample` | `--data <sample> --flow <flow> --virtual-alpha 0.1` | 2M sample keys | `original` |
| `sample_flow` | same run | z(k) of the 2M keys | `transformed` |
| `sample_flow_csv` | same run | z(k) + virtual points per region | `smoothed` |
| `sample_csv` | `--data <sample> --virtual-alpha 0.1` (no flow) | double(k − min) + virtual points | `smoothed` |

CD carries a ±1 rounding uncertainty (floor of a double product, as in LIPP) [hardness.hpp:119-127;
hardness_details.json `cd_note`]; treat CD differences of one as ties.

---

### 3. The ten datasets, one by one

Descriptions are quoted from Table 1 of the AIDB paper [AIDB Table 1, p.4] with the paper's source
reference; anything beyond that is either from the CSV or NFL papers (cited) or measured by us on the
files (`scratch/data_stats.py`, `hardness_details.json`). Key statistics (min, max, percentiles) are from
the sorted files [scratch/data_stats.json]; "p50" is the key at rank ⌊0.5·(n−1)⌋. Full-file metrics are
[results/aidb/hardness.json `<name>.full` / `.full_flow`]; the flow in `full_flow` is the one trained on
the 2M **uniform** sample (Section 6).

**T3. Key statistics of the ten sorted files** [scratch/data_stats.json].

| dataset | min | max | range = max−min | log₂(range) | p50 | p99 | p99.99 | mean gap = range/(n−1) |
|---|---|---|---|---|---|---|---|---|
| books | 0 | 9,223,372,036,854,784,874 | 9,223,372,036,854,784,874 | 63.00 | 1,920,975,649,179,691,136 | 7,690,015,317,089,125,888 | 9,132,078,801,956,411,904 | 46,116,860,414.858 |
| fb | 1 | 18,446,744,073,709,551,615 | 18,446,744,073,709,551,614 | 64.00 | 38,752,453,239 | 76,546,751,873 | 77,299,521,445 | 92,233,720,829.716 |
| osm | 33,246,697,004,540,789 | 13,748,550,930,623,082,253 | 13,715,304,233,618,541,464 | 63.57 | 5,170,332,552,509,244,517 | 10,737,454,289,360,892,995 | 13,686,332,700,453,684,633 | 68,576,521,510.975 |
| covid | 1,344,795,470,900,715,522 | 1,446,626,459,003,486,210 | 101,830,988,102,770,688 | 56.50 | 1,391,398,680,826,101,761 | 1,443,581,128,888,127,494 | 1,446,589,962,841,907,209 | 509,154,943.060 |
| genome | 2,489,750 | 446,640,429,470 | 446,637,939,720 | 38.70 | 280,552,045,122 | 445,347,172,802 | 446,612,275,232 | 2,233.190 |
| history | 83 | 9,178,997,263 | 9,178,997,180 | 33.10 | 4,473,873,432 | 9,085,796,222 | 9,178,114,938 | 45.895 |
| libio | 21,335 | 527,206,042 | 527,184,707 | 28.97 | 286,981,400 | 523,018,907 | 527,161,946 | 2.636 |
| planet | 1 | 9,178,997,250 | 9,178,997,249 | 33.10 | 776,139,014 | 8,988,035,135 | 9,177,243,630 | 45.895 |
| stack | 1 | 238,071,030 | 238,071,029 | 27.83 | 121,403,258 | 235,463,774 | 238,041,730 | 1.190 |
| wise | 8,796,093,034,805 | 17,592,186,032,162 | 8,796,092,997,357 | 43.00 | 13,104,235,008,794 | 17,530,874,982,592 | 17,591,561,268,342 | 43,980.465 |

A quick global-linearity test: for a perfectly linear file p50 sits at 50 % of the range. Measured
(p50 − min)/range: stack 0.510, history 0.487, libio 0.544, covid 0.458, wise 0.490, genome 0.628, osm
0.375, books 0.208, planet 0.085, fb 2.1e-9 (fb's range is set by five outliers, see below).

#### 3.1 books — "Amazon book sales popularity" [AIDB Table 1, source [7] = SOSD, Kipf et al. 2019]

- NFL calls the same source "the amazon (AMZN) dataset ... book sale popularity data on the Amazon [35]"
  (Kaggle sales-rank data) [NFL §4.1.1]. What the integer encodes beyond that is not described in the
  papers.
- Keys 0 … 9,223,372,036,854,784,874 (= 2⁶³ + 9,066); 245 keys are ≥ 2⁶³ and 30,130,154 are ≥ 2⁶²
  [scratch/data_stats.json `books.count_ge`]. The distribution is strongly skewed towards small keys:
  p50 at 20.8 % of the range, p90 at 58 %.
- Full: RMSE 18,053,637.5, ME 96,380,337.9, CD 246, PLA-32 262,604, PLA-4096 **97** [hardness.json
  `books.full`]. RMSE/n = 0.090 and ME/n = 0.48: a single line is a poor global fit, yet PLA-4096 = 97 is
  the smallest of all ten: the CDF is smooth and concave (few global pieces) but far from a line.
- With the uniform-sample flow: RMSE 15,985,128.0 (−11.5 %), ME 78,944,811.5, CD 245, PLA-32 262,604,
  PLA-4096 94 [`books.full_flow`]. The flow's tanh bends the concave CDF a little straighter; PLA is
  untouched because a monotone reparametrisation cannot change the number of ε-segments unless it
  creates or removes collinearity (it changed PLA-32 by 0 to 5 segments on every dataset).
- Character: globally curved and skewed, moderately clustered (CD 246), locally moderate. The uniform
  sample looks trivially easy (PLA-32 602, at the sampling-noise floor of Section 5.3), the window does
  not (PLA-32 5,789, mean per-region tail CD 10.76).

#### 3.2 fb — "Upsampled Facebook user ID" [AIDB Table 1, source [7] = SOSD]

- NFL: "an upsampled version of a Facebook user ID dataset [36]" [NFL §4.1.1]; CSV: "200 million integer
  Facebook user IDs [30]" [CSV §6.1 "Datasets"]. The upsampling procedure is not described in the three
  papers.
- Keys 1 … 18,446,744,073,709,551,615 = 2⁶⁴ − 1. **The range is set by a handful of outliers**: only 21
  keys are ≥ 2⁴⁰ (1.1e12), 9 are ≥ 2⁵⁶ and 5 are ≥ 2⁶³ [scratch/data_stats.json `fb.count_ge`]; the last
  five keys are 10,248,691,552,019,458,048; 12,298,204,682,441,981,952; 14,347,717,812,864,503,808;
  16,397,230,943,287,027,712; 18,446,744,073,709,551,615, spaced by 2,049,513,130,422,523,904 ≈ 2⁶⁴/9
  [`fb.tail`, `fb.top_gaps`]. The other 199,999,979 keys lie in [1, 1.1e12]; p99.99 = 77,299,521,445.
- Full: RMSE 57,735,023.7, ME 99,999,995.5, CD 110, PLA-32 1,055,308, PLA-4096 1,687 [`fb.full`].
  **Worked check**: with five keys at ~10¹⁹ and everything else below 10¹², the least-squares line is
  pinned by the outliers and is essentially constant over the bulk; the residual of a constant predictor
  against ranks 0..n−1 has RMSE n/√12 = 200,000,000/3.4641 = 57,735,026.9 and ME n/2 = 100,000,000. The
  measured values are 57,735,023.7 (ratio 1.000000) and 99,999,995.5. So fb's RMSE and ME say nothing
  about the 199,999,979 ordinary keys; they measure the outliers.
- With the uniform-sample flow: RMSE 1,161,273.7 (−98.0 %), ME 352,904,667.5 (larger than n: the flow's
  saturated tanh maps the five outliers to z ≈ v₀+v₁ = 4.67 while the bulk spans z ∈ [0, 1.67], so the
  line now fits the bulk and the outliers become the residual), CD 110, PLA-32 1,055,308, PLA-4096 1,687
  [`fb.full_flow`].
- Character: a near-linear body (uniform-sample RMSE 1,448.2 is the smallest of all ten, 5× the noise
  floor) with extreme outliers and strong **local** irregularity: window PLA-32 10,599, window CD 96,
  mean per-region tail CD 31.41 (second-highest after genome's window). CSV's Figure 5 calls Facebook
  "easy" with "almost globally linear CDFs" [CSV §6.1]; the AIDB Figure 2 places it at the top of the
  PLA-32 axis [AIDB Fig. 2]. Both are right: they look at different scales.

#### 3.3 osm — "Uniformly sampled OpenStreetMap locations" [AIDB Table 1, source [7] = SOSD]

- CSV: "200 million locations randomly sampled from OpenStreetMap and represented as Google S2 [26] cell
  IDs [24]" [CSV §6.1]; CSV also calls OSM and Genome "hard datasets" [CSV §6.1].
- Keys 33,246,697,004,540,789 … 13,748,550,930,623,082,253 (log₂ range 63.57). Extremely clustered: p25 =
  5.06e18 and p75 = 6.09e18, i.e. half of all keys sit in 7.5 % of the range (S2 cells over land masses).
- Full: RMSE 24,177,498.5, ME 67,384,214.8, **CD 4,107** (by far the highest), PLA-32 661,115, **PLA-4096
  5,495** (highest) [`osm.full`]. FMCD's fit: U_T = 3.41e10 key units per slot, D = 4,106 [hardness_details
  `osm.full_original.fmcd`]: 4,107 keys fall in one 3.4e10-wide slot somewhere.
- With the flow: RMSE 23,977,602.0 (−0.8 %), CD 3,943, PLA unchanged [`osm.full_flow`]: a 2×2 tanh
  cannot un-cluster a stepped CDF.
- Character: globally hard (steps and plateaus; the only dataset whose uniform-sample CD stays in the
  thousands: 1,102), moderately hard locally (window PLA-32 4,919, tail CD 9.66: inside one 0.144 %-wide
  slice of the key range the cells are fairly regular).

#### 3.4 covid — "Uniformly sampled Tweet ID with tag COVID-19" [AIDB Table 1, source [16] = Lopez & Gallemore 2021]

- CSV: "200 million integer tweet IDs randomly sampled from tweets tagged with 'Covid-19' [18]"; CSV
  calls it an easy dataset with an almost globally linear CDF and, zoomed in, the one dataset that does
  not "deviate from linear CDFs at local level" [CSV §6.1]. What the tweet-ID integer encodes is not
  described in the papers.
- Keys 1,344,795,470,900,715,522 … 1,446,626,459,003,486,210, range 1.018e17 (log₂ 56.5), p50 at 45.8 %
  of the range.
- Full: RMSE 1,795,722.5, ME 8,133,078.0, CD 27, PLA-32 81,908, PLA-4096 850 [`covid.full`]. With the
  flow: RMSE 1,272,460.5 (−29.1 %), CD 27, PLA-32 81,909, PLA-4096 850 [`covid.full_flow`].
- Character: near-linear globally (RMSE/n = 0.009) and the easiest locally together with stack, history
  and wise: window PLA-32 827, mean per-region tail CD 3.02 in the window and 3.16 in the uniform sample
  (the noise floor is 3.00). The window's CD 24 and the full CD 27 agree (ratio 1.12): the clusters that
  set CD are local, so a window sees them at full resolution.

#### 3.5 genome — "Loci pairs in human chromosomes" [AIDB Table 1, source [21] = Rao et al. 2014, Cell]

- CSV: "200 million entries of loci pairs in human chromosomes represented as integers [25]", a "hard"
  dataset that deviates from linearity "at local level, especially Genome" [CSV §6.1]. The integer
  encoding of a locus pair is not described in the papers.
- Keys 2,489,750 … 446,640,429,470 (log₂ 38.7), mean gap 2,233; p50 at 62.8 % of the range.
- Full: RMSE 7,531,940.1, ME 19,622,982.5, CD 585, **PLA-32 1,290,208** (highest of all ten), PLA-4096
  1,426 [`genome.full`]. With the flow: RMSE 8,390,195.9 (+11.4 %: the transform hurts here), CD 587,
  PLA-32 1,290,208, PLA-4096 1,425 [`genome.full_flow`].
- Character: only mildly curved globally (RMSE/n 0.038) but the roughest locally: window PLA-32 12,719
  (highest window value), window CD 171, mean per-region tail CD **60.41** (highest anywhere). The
  uniform sample hides all of it (PLA-32 2,045, tail CD 4.23).

#### 3.6 history — "History node ID in OpenStreetMap" [AIDB Table 1, source [5] = Google Cloud OSM]

- Not described further in the papers. Keys 83 … 9,178,997,263 (log₂ 33.1), mean gap 45.9, p50 at 48.7 %
  of the range: history and planet (below) share the same ID space (their maxima differ by 13) but have
  very different shapes.
- Full: RMSE 815,542.7, ME 2,303,085.2, CD 8, **PLA-32 105,468, PLA-4096 468** [`history.full`]. These are
  exactly the two values the AIDB paper prints for history, "h_l = 105k, h_g = 468" [AIDB §2.2], which is
  the strongest evidence that our metric code and the paper's agree on the same file. With the flow:
  RMSE 432,676.2 (−46.9 %), CD 8, PLA-32 105,469, PLA-4096 467 [`history.full_flow`].
- Character: near-linear globally (RMSE/n 0.0041, lowest with stack) and easy locally (window PLA-32 943,
  tail CD 3.10 in both modes). Note the AIDB paper's point: LIPP does 5.6 MOPS on this "relatively easy"
  dataset versus 7.1 MOPS on the "relatively hard" libio [AIDB §2.2].

#### 3.7 libio — "Repository ID from libraries.io" [AIDB Table 1, source [14] = libraries.io data]

- Not described further. Keys 21,335 … 527,206,042 (log₂ 29.0), mean gap **2.636**: about 38 % of the
  integers in the range are present, so the sequence is close to consecutive IDs with holes.
- Full: RMSE 3,445,772.9, ME 11,828,007.2, **CD 2** (lowest with stack), **PLA-32 145,808, PLA-4096 639**
  [`libio.full`], matching the paper's "h_l = 146k, h_g = 639" [AIDB §2.2]. With the flow: RMSE
  4,072,387.5 (+18.2 %), CD 2, PLA-32 145,811, PLA-4096 638 [`libio.full_flow`].
- Character: mildly curved globally (p50 at 54 %, RMSE/n 0.017) but so dense that FMCD never collides:
  window CD 2, mean per-region tail CD 1.61 (window) — the second-easiest local structure after stack.

#### 3.8 planet — "Planet ID in OpenStreetMap" [AIDB Table 1, source [5] = Google Cloud OSM]

- Not described further. Keys 1 … 9,178,997,250 (same ID space as history), but p50 = 776,139,014 sits at
  8.5 % of the range and p75 at 33.8 %: most IDs are small (a heavily front-loaded, convex-then-flat CDF).
- Full: RMSE 29,771,622.2 (RMSE/n 0.149, second-highest after fb), ME 60,462,049.5, CD 21, PLA-32 613,597,
  PLA-4096 2,314 [`planet.full`]. With the flow: RMSE 15,545,175.8 (−47.8 %, the largest genuine
  improvement: a tanh bends a front-loaded CDF the right way), CD 32, PLA-32 613,602, PLA-4096 2,313
  [`planet.full_flow`].
- Character: globally curved, locally rough as well (window PLA-32 8,498, window CD 49, mean per-region
  tail CD 35.78). Uniform-sample tail CD 7.66 is the third-highest of the uniform samples.

#### 3.9 stack — "Vote ID from StackOverflow" [AIDB Table 1, source [22] = archive.org stackexchange dump]

- Not described further. Keys 1 … 238,071,030, mean gap **1.190**: 84 % of all integers in the range are
  keys; the file is almost the sequence 1, 2, 3, … with 16 % holes (head: 1, 2, 3, 4, 5).
- Full: RMSE 839,727.0, ME 2,444,700.0, **CD 1, PLA-32 17,833, PLA-4096 133** (all three the lowest of
  the ten) [`stack.full`]. With the flow: RMSE 1,769,181.3 (+110.7 %: the flow can only add curvature to
  an already straight CDF), CD 1, PLA-32 17,833, PLA-4096 129 [`stack.full_flow`].
- Character: the easiest dataset at every scale. Window: CD 1, PLA-32 144, mean per-region tail CD
  exactly 1.0000; the CSV-style smoother could place only 62,282 of the 199,707 budgeted virtual points
  in the window (31 %) because there was nothing left to straighten (Section 7).

#### 3.10 wise — "Partition key from the data returned by the Wide-field Infrared Survey Explorer (WISE)" [AIDB Table 1, source [28] = Wright et al. 2010]

- Not described further. Keys 8,796,093,034,805 … 17,592,186,032,162: the whole file sits inside
  [2⁴³, 2⁴⁴) (2⁴³ = 8,796,093,022,208, 2⁴⁴ = 17,592,186,044,416), min = 2⁴³ + 12,597, max = 2⁴⁴ − 12,254.
  p50 at 49.0 % of the range.
- Full: RMSE 2,200,401.6, ME 4,988,460.6, CD 10, PLA-32 79,035, PLA-4096 382 [`wise.full`]. With the
  flow: RMSE 1,511,720.5 (−31.3 %), CD 9, PLA-32 79,036, PLA-4096 382 [`wise.full_flow`].
- Character: near-linear globally (RMSE/n 0.011) and easy locally (window PLA-32 805, tail CD 3.01).

---

### 4. Master table (full files, sorted by PLA-32)

**T2.** All values [results/aidb/hardness.json `<name>.full`]; n = 200,000,000. The two datasets whose
GRE values the AIDB paper prints are marked; both match to the printed precision [AIDB §2.2].

| rank | dataset | RMSE | ME | CD | PLA-32 (h_l) | PLA-4096 (h_g) | RMSE/n | ME/n |
|---|---|---|---|---|---|---|---|---|
| 1 | stack | 839,727.0 | 2,444,700.0 | 1 | 17,833 | 133 | 0.0042 | 0.0122 |
| 2 | wise | 2,200,401.6 | 4,988,460.6 | 10 | 79,035 | 382 | 0.0110 | 0.0249 |
| 3 | covid | 1,795,722.5 | 8,133,078.0 | 27 | 81,908 | 850 | 0.0090 | 0.0407 |
| 4 | **history** (paper: h_l = 105k, h_g = 468) | 815,542.7 | 2,303,085.2 | 8 | **105,468** | **468** | 0.0041 | 0.0115 |
| 5 | **libio** (paper: h_l = 146k, h_g = 639) | 3,445,772.9 | 11,828,007.2 | 2 | **145,808** | **639** | 0.0172 | 0.0591 |
| 6 | books | 18,053,637.5 | 96,380,337.9 | 246 | 262,604 | 97 | 0.0903 | 0.4819 |
| 7 | planet | 29,771,622.2 | 60,462,049.5 | 21 | 613,597 | 2,314 | 0.1489 | 0.3023 |
| 8 | osm | 24,177,498.5 | 67,384,214.8 | 4,107 | 661,115 | 5,495 | 0.1209 | 0.3369 |
| 9 | fb | 57,735,023.7 | 99,999,995.5 | 110 | 1,055,308 | 1,687 | 0.2887 | 0.5000 |
| 10 | genome | 7,531,940.1 | 19,622,982.5 | 585 | 1,290,208 | 1,426 | 0.0377 | 0.0981 |

Reading the table: the five metrics rank the datasets differently, which is the AIDB paper's whole
point ("there is no metric that simultaneously achieves ideal conformance and ideal coverage" [AIDB §4.2]). By PLA-32 genome > fb > osm > planet > books;
by CD osm ≫ genome > books > fb; by RMSE fb > planet > osm > books ≫ genome; by PLA-4096 osm > planet >
fb > genome. The paper's own Figure 2 (GRE plane) and Figure 6 (CD·PLA-32 plane) show the same ten
points [AIDB Figs. 2, 6]; the exact coordinates of the other eight datasets are not printed in the
paper, so only history and libio can be cross-checked digit by digit.

The same ten metrics after the uniform-sample flow (`full_flow`) are in T7 (Section 6.3).

---

### 5. The samples

#### 5.1 Exactly how the two samples are drawn

Both samples are produced by `tools/datasets.py sample <src> <dest> --n 2000000 --dtype uint64 --mode
{uniform|window} --seed 42`, invoked from the pipeline's `sample` step [aidb_pipeline.py:418-433, 430].
The source `<src>` is the sorted copy for the seven unsorted files and the download itself for books,
fb, osm [aidb_pipeline.py:278-283; provenance.json `<name>.sample.source`].

```
## tools/datasets.py:73-89 (verbatim logic)
total = 200_000_000; n = 2_000_000; rng = random.Random(seed)          # seed = 42
if mode == 'uniform':  indices = sorted(rng.sample(range(total), n))    # 2M distinct ranks, no replacement
elif mode == 'strided': indices = [(2*i+1)*total // (2*n) for i in range(n)]   # exists, not used
else:                   start = rng.randrange(total - n + 1)            # 'window'
                        indices = range(start, start + n)
write 8-byte count, then key[i] for i in indices  (mmap of the source, so keys keep source order = sorted)
manifest = {source, source_count, source_sha256, mode, seed, sample_count, dtype, sample_sha256, warning}
```

- **Uniform, seed 42**: 2,000,000 distinct ranks chosen without replacement from 0..199,999,999; every key
  has inclusion probability p = 0.01. The first sampled ranks are 132, 203, 373, 403, 407 and the last
  are 199,999,700, 199,999,709, 199,999,874; the mean rank gap is 99.99992 [scratch/data_stats.py, same
  RNG call]. Because the RNG is seeded identically for every dataset, **the same 2M ranks are taken from
  all ten files**.
- **Window, seed 42**: `random.Random(42).randrange(198,000,001)` = **171,644,825**, so every window is
  ranks 171,644,825 … 173,644,824, i.e. the slice starting at the 85.82 % quantile of each file
  [scratch/data_stats.py]. Verified for all ten datasets: the window sample's first and last keys equal
  `key[171,644,825]` and `key[173,644,824]` of the sorted source file [scratch/data_stats.json
  `<name>.window_matches = true`]. The manifests do not store the offset (they store mode, seed, counts
  and hashes) [data/samples/*_window_s42.manifest.json]; the offset is reproducible from the seed.
- Each sample file is 8 + 8 × 2,000,000 = 16,000,008 bytes; `sample_count: 2000000`, `dtype: uint64`,
  and the manifest carries the warning "A derived sample is NOT the full benchmark corpus; sampling
  changes local hardness." [manifest.json].
- Sample min/max: **T10** below [hardness_details.json `<name>.sample.min/max` in both result
  directories].

**T10. Uniform vs window samples, raw keys** (metrics [hardness.json `<name>.sample`]; tail CD = mean
per-region tail conflict degree from the sweeps, Section 5.4).

| dataset | uniform min | uniform max | window min | window max | window span / full range | uniform PLA-32 | window PLA-32 | uniform CD | window CD | uniform RMSE | window RMSE | uniform tail CD | window tail CD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| books | 3,241 | 9,223,372,036,854,781,810 | 4,735,252,013,344,826,368 | 4,880,676,988,429,512,704 | 1.577% | 602 | 5,789 | 11 | 27 | 180,591.3 | 1,709.1 | 3.28 | 10.76 |
| fb | 97,995 | 77,308,811,965 | 66,478,271,979 | 67,251,951,970 | 0.000% | 3,034 | 10,599 | 24 | 96 | 1,448.2 | 8,202.5 | 8.45 | 31.41 |
| osm | 42,262,281,243,463,475 | 13,748,274,210,235,485,307 | 9,771,619,518,438,332,541 | 9,791,353,970,024,941,583 | 0.144% | 7,099 | 4,919 | 1,102 | 152 | 241,476.1 | 118,136.5 | 36.16 | 9.66 |
| covid | 1,344,795,518,640,259,075 | 1,446,626,324,517,429,249 | 1,428,280,948,953,600,004 | 1,429,257,627,939,184,641 | 0.959% | 1,192 | 827 | 8 | 24 | 18,073.1 | 24,143.5 | 3.16 | 3.02 |
| genome | 46,509,831 | 446,638,295,046 | 417,037,340,916 | 419,441,740,061 | 0.538% | 2,045 | 12,719 | 77 | 171 | 75,352.0 | 48,827.0 | 4.23 | 60.41 |
| history | 81,021 | 9,178,989,449 | 7,808,774,263 | 7,898,935,777 | 0.982% | 1,067 | 943 | 7 | 8 | 8,104.9 | 9,651.7 | 3.10 | 3.10 |
| libio | 29,518 | 527,205,800 | 462,326,661 | 466,941,114 | 0.875% | 1,256 | 1,057 | 8 | 2 | 34,472.4 | 8,854.0 | 3.37 | 1.61 |
| planet | 457 | 9,178,968,583 | 5,721,028,025 | 5,986,820,672 | 2.896% | 3,456 | 8,498 | 26 | 49 | 297,556.5 | 42,645.1 | 7.66 | 35.78 |
| stack | 299 | 238,070,846 | 204,327,766 | 206,662,194 | 0.981% | 593 | 144 | 8 | 1 | 8,395.8 | 25,321.9 | 3.01 | 1.00 |
| wise | 8,796,104,780,969 | 17,592,181,572,984 | 16,325,355,473,712 | 16,408,098,198,591 | 0.941% | 787 | 805 | 8 | 8 | 21,926.3 | 5,551.0 | 3.03 | 3.01 |

Two things to notice immediately. (i) The fb uniform sample's maximum is 77,308,811,965 < 2⁴⁰: none of
fb's 21 outliers was drawn (with p = 0.01 per key the chance of drawing none is 0.99²¹ = 0.81), so the
uniform sample of fb has none of the global pathology of the full file. (ii) For a file with a linear
CDF, a 1 %-of-ranks window covers 1 % of the key range; the "window span / full range" column is
therefore a local-density indicator: covid, history, stack, wise, libio ≈ 0.9–1 % (uniform density where
the window sits), planet 2.9 % (sparse there), books 1.6 %, osm 0.14 % and genome 0.54 % (dense there),
fb 0.000 % (the window sits in the bulk, the range is the outliers).

#### 5.2 What the two modes preserve and destroy: derivation

Let R(k) be a key's rank in the full file (n_F = 2·10⁸) and R_s(k) its rank in a sample.

*Uniform sample.* R_s(k) counts how many of the R(k) smaller keys were drawn, so R_s(k) ~ Binomial(R(k),
p) with p = 0.01: mean p·R(k), standard deviation √(R(k)·p·(1−p)) ≤ √(2·10⁸ · 0.0099) = 1,407 ranks.
Hence:

- Global shape is preserved and **RMSE and ME scale by exactly p**: the least-squares residual of the
  sample is p × the full residual plus binomial noise. Measured full/uniform ratios [T9]: RMSE 99.36–
  100.62 and ME 98.52–100.85 on nine datasets; fb is the exception (39,866 and 27,538) because the sample
  lost the outliers.
- Local structure at the 32-rank scale is destroyed: an ε = 32 corridor in sample ranks is a ±3,200-rank
  corridor in the full file, and on top of that the binomial jitter alone forces segments. The
  sampling-noise floor (Section 5.3) is PLA-32 = 535 for a perfectly linear file; books (602), stack
  (593), wise (787), history (1,067), covid (1,192), libio (1,256) all sit within 2.4× of it.
- CD is not scale-free either: FMCD's slot width U_T = (k[n−1−D] − k[D])/(2n − 2) is 100× wider on the
  sample (same key range, n/100), so CD measures cluster density at a 100× coarser key scale. Full/uniform
  CD ratios range from 0.12 (stack) to 22 (books) [T9]; only osm keeps a CD in the thousands (1,102).
- Consequence: the uniform samples are a **global-shape** benchmark. The mean per-region tail CD is
  3.0–3.4 on six of the ten uniform samples, i.e. at the noise floor of 3.00 (Section 5.3); only fb
  (8.45), osm (36.16), planet (7.66) and genome (4.23) keep anything above it [T5].

*Window sample.* Ranks are exact (R_s = R − 171,644,825), so every local statistic is measured at full
resolution, but the sample sees 1 % of the key range at one place:

- PLA-32 is preserved up to how typical the slice is: window PLA-32 × 100 vs full PLA-32 gives ratios
  full/window of 99.6 (fb), 99.0 (covid), 101.4 (genome), 98.2 (wise), 111.8 (history), 123.8 (stack),
  134.4 (osm), 137.9 (libio), 72.2 (planet), 45.4 (books) [T9]. For fb, covid, genome and wise the window
  is a faithful 1/100 of the file; for books and planet (non-stationary, front-loaded CDFs) the 85.8 %
  quantile slice is rougher than average.
- CD is preserved when the clusters that set it are local: full/window CD ratios 1.00 (history, libio,
  stack), 1.12 (covid), 1.15 (fb), 1.25 (wise), but 27 (osm), 9.1 (books), 3.4 (genome), 0.43 (planet)
  [T9].
- Global shape is destroyed: every window RMSE is tiny in absolute terms (1,709 books; 5,551 wise) and
  unrelated to the full RMSE (full/window RMSE ratio 33 to 10,564 [T9]).

**T9. Full-file / sample ratios** (n ratio = 100) [computed from hardness.json of both runs].

| dataset | RMSE full/uniform | ME full/uniform | CD full/uniform | PLA-32 full/uniform | PLA-4096 full/uniform | RMSE full/window | PLA-32 full/window | CD full/window |
|---|---|---|---|---|---|---|---|---|
| books | 99.97 | 100.19 | 22.36 | 436.2 | 12.1 | 10,563.5 | 45.4 | 9.11 |
| fb | 39,865.58 | 27,538.25 | 4.58 | 347.8 | 1,687.0 | 7,038.7 | 99.6 | 1.15 |
| osm | 100.12 | 100.21 | 3.73 | 93.1 | 80.8 | 204.7 | 134.4 | 27.02 |
| covid | 99.36 | 100.66 | 3.38 | 68.7 | 94.4 | 74.4 | 99.0 | 1.12 |
| genome | 99.96 | 99.84 | 7.60 | 630.9 | 79.2 | 154.3 | 101.4 | 3.42 |
| history | 100.62 | 98.52 | 1.14 | 98.8 | 156.0 | 84.5 | 111.8 | 1.00 |
| libio | 99.96 | 100.42 | 0.25 | 116.1 | 63.9 | 389.2 | 137.9 | 1.00 |
| planet | 100.05 | 100.02 | 0.81 | 177.5 | 178.0 | 698.1 | 72.2 | 0.43 |
| stack | 100.02 | 100.85 | 0.12 | 30.1 | 33.2 | 33.2 | 123.8 | 1.00 |
| wise | 100.35 | 100.21 | 1.25 | 100.4 | 18.2 | 396.4 | 98.2 | 1.25 |

**The fb example.** Full PLA-32 1,055,308 (rank 9 of 10). Uniform sample: PLA-32 3,034, CD 24, RMSE
1,448 — the "easiest-looking" body of all ten. Window: PLA-32 **10,599** (×100 = 1,059,900 ≈ full),
CD 96, RMSE 8,202, mean per-region tail CD 31.41 [T10]. The uniform sample turned fb into a near-linear
file with a little noise; the window kept the upsampled-ID roughness that makes fb hard for LIPP-style
slot mapping [NFL Table 1 on facebook; AIDB Fig. 6 on fb's CD].

**The osm example.** Full CD 4,107, PLA-4096 5,495 (the hardest global structure). Uniform sample: CD
1,102, PLA-32 7,099 (the highest uniform PLA-32), tail CD 36.16 (the highest uniform tail CD): the global
clustering survives sampling because clusters spanning 10¹⁷ key units are 100× wider than the sample's
slot width. Window: CD 152, PLA-32 4,919, tail CD 9.66: inside 0.144 % of the range the S2 cells are more
regular than fb's IDs. So osm is "hard" in the uniform lane and only "medium" in the window lane, while
fb and genome are the opposite — the two lanes are two different benchmarks and are reported separately
everywhere in this guide.

#### 5.3 Control: the sampling-noise floor

To know how much of a uniform sample's hardness is sampling noise, we sampled a *perfectly linear* file
with the same RNG: keys = the 2M sampled ranks themselves (`sorted(random.Random(42).sample(range(2e8),
2e6))`, so the "full file" is 0, 1, …, 199,999,999) and ran the same `scaleli_hardness` binary with the
same flags [scratch/linear_uniform_2M_s42, scratch/linear_uniform_hardness.json]:

| metric | linear-file uniform sample (noise floor) | smallest real uniform sample | largest |
|---|---|---|---|
| RMSE | 289.5 | fb 1,448.2 | planet 297,556.5 |
| ME | 732.5 | fb 3,631.3 | books 961,972.8 |
| CD | 7 | history 7 | osm 1,102 |
| PLA-32 | 535 | stack 593 | osm 7,099 |
| PLA-4096 | 1 | fb 1 | osm 68 |
| mean per-region tail CD (489 regions) | 3.002 (min 3, max 4) | stack 3.008 | osm 36.16 |
| whole-sample tail CD | 3 | covid/history/libio/stack/wise 3 | osm 99 |

So for the uniform samples: PLA-32 below ~600, CD ≤ 8 and per-region tail CD ≈ 3 mean "indistinguishable
from a linear file with sampling noise"; those numbers should not be interpreted as dataset structure.
The FMCD fit on the control gives U_T = 49.9995 ranks per slot with D = 7 [linear_uniform_hardness.json
`original.fmcd`]: seven sampled ranks land in the same 50-rank slot somewhere in 2M draws, which is the
Poisson tail, not structure. (The CSV smoother on this control sample: PLA-32 535 → 225 with 179,985
virtual points; Section 7.)

#### 5.4 Sample hardness in the four sample scopes

**T4a. Uniform samples** [results/aidb/hardness.json `<name>.{sample,sample_flow,sample_csv,
sample_flow_csv}`; `vp`, `regions`, `unordered` from hardness_details.json]. The uniform run's details do
not record the transformed-duplicates count (field absent; the window run's do, see T4b); `unordered = 0`
everywhere means the monotone flows kept key order.

| dataset | scope | keys | RMSE | ME | CD | PLA-32 | PLA-4096 | extra |
|---|---|---|---|---|---|---|---|---|
| books | sample | 2,000,000 | 180,591.3 | 961,972.8 | 11 | 602 | 8 |  |
| books | sample_flow | 2,000,000 | 159,882.7 | 787,945.4 | 9 | 602 | 8 | unordered=0 |
| books | sample_csv | 2,185,327 | 196,846.2 | 1,048,403.0 | 10 | 236 | 8 | vp=185,327, regions=489 |
| books | sample_flow_csv | 2,185,445 | 174,288.9 | 858,949.9 | 9 | 239 | 8 | vp=185,445, regions=489 |
| fb | sample | 2,000,000 | 1,448.2 | 3,631.3 | 24 | 3,034 | 1 |  |
| fb | sample_flow | 2,000,000 | 11,584.6 | 30,693.4 | 24 | 3,034 | 3 | unordered=0 |
| fb | sample_csv | 2,199,707 | 1,591.2 | 3,959.5 | 23 | 1,718 | 1 | vp=199,707, regions=489 |
| fb | sample_flow_csv | 2,199,707 | 12,742.1 | 33,743.0 | 23 | 1,721 | 3 | vp=199,707, regions=489 |
| osm | sample | 2,000,000 | 241,476.1 | 672,432.6 | 1,102 | 7,099 | 68 |  |
| osm | sample_flow | 2,000,000 | 239,472.1 | 589,442.9 | 1,100 | 7,099 | 68 | unordered=0 |
| osm | sample_csv | 2,199,605 | 265,521.5 | 739,686.0 | 1,099 | 7,491 | 76 | vp=199,605, regions=489 |
| osm | sample_flow_csv | 2,199,396 | 263,263.9 | 648,294.0 | 1,099 | 7,468 | 76 | vp=199,396, regions=489 |
| covid | sample | 2,000,000 | 18,073.1 | 80,794.0 | 8 | 1,192 | 9 |  |
| covid | sample_flow | 2,000,000 | 12,906.4 | 47,120.6 | 8 | 1,193 | 9 | unordered=0 |
| covid | sample_csv | 2,198,862 | 19,819.2 | 88,707.4 | 10 | 892 | 11 | vp=198,862, regions=489 |
| covid | sample_flow_csv | 2,198,904 | 14,147.7 | 51,691.0 | 11 | 892 | 9 | vp=198,904, regions=489 |
| genome | sample | 2,000,000 | 75,352.0 | 196,543.7 | 77 | 2,045 | 18 |  |
| genome | sample_flow | 2,000,000 | 83,934.3 | 214,604.9 | 82 | 2,045 | 18 | unordered=0 |
| genome | sample_csv | 2,199,707 | 82,874.1 | 216,173.7 | 71 | 1,627 | 19 | vp=199,707, regions=489 |
| genome | sample_flow_csv | 2,199,707 | 92,313.6 | 236,037.8 | 73 | 1,629 | 19 | vp=199,707, regions=489 |
| history | sample | 2,000,000 | 8,104.9 | 23,376.3 | 7 | 1,067 | 3 |  |
| history | sample_flow | 2,000,000 | 4,290.7 | 11,084.8 | 7 | 1,068 | 3 | unordered=0 |
| history | sample_csv | 2,195,413 | 8,628.3 | 24,888.9 | 21 | 411 | 3 | vp=195,413, regions=489 |
| history | sample_flow_csv | 2,195,568 | 4,846.6 | 12,761.0 | 21 | 412 | 3 | vp=195,568, regions=489 |
| libio | sample | 2,000,000 | 34,472.4 | 117,779.9 | 8 | 1,256 | 10 |  |
| libio | sample_flow | 2,000,000 | 40,748.0 | 138,497.3 | 7 | 1,255 | 10 | unordered=0 |
| libio | sample_csv | 2,195,924 | 37,442.8 | 128,489.6 | 21 | 614 | 11 | vp=195,924, regions=489 |
| libio | sample_flow_csv | 2,195,953 | 44,356.9 | 151,183.9 | 16 | 614 | 11 | vp=195,953, regions=489 |
| planet | sample | 2,000,000 | 297,556.5 | 604,478.5 | 26 | 3,456 | 13 |  |
| planet | sample_flow | 2,000,000 | 155,367.6 | 337,958.1 | 33 | 3,455 | 15 | unordered=0 |
| planet | sample_csv | 2,199,656 | 327,255.8 | 664,813.1 | 37 | 2,767 | 15 | vp=199,656, regions=489 |
| planet | sample_flow_csv | 2,199,699 | 170,882.5 | 371,498.2 | 33 | 2,771 | 15 | vp=199,699, regions=489 |
| stack | sample | 2,000,000 | 8,395.8 | 24,241.8 | 8 | 593 | 4 |  |
| stack | sample_flow | 2,000,000 | 17,705.8 | 42,459.6 | 7 | 594 | 4 | unordered=0 |
| stack | sample_csv | 2,184,990 | 9,036.9 | 26,127.3 | 7 | 319 | 4 | vp=184,990, regions=489 |
| stack | sample_flow_csv | 2,184,637 | 19,180.3 | 45,948.0 | 8 | 321 | 4 | vp=184,637, regions=489 |
| wise | sample | 2,000,000 | 21,926.3 | 49,780.1 | 8 | 787 | 21 |  |
| wise | sample_flow | 2,000,000 | 15,026.4 | 38,641.2 | 7 | 787 | 21 | unordered=0 |
| wise | sample_csv | 2,194,693 | 24,008.5 | 54,783.4 | 9 | 436 | 22 | vp=194,693, regions=489 |
| wise | sample_flow_csv | 2,194,876 | 16,401.8 | 42,370.8 | 8 | 430 | 22 | vp=194,876, regions=489 |

**T4b. Window samples** [results/aidb_window/hardness.json, hardness_details.json].

| dataset | scope | keys | RMSE | ME | CD | PLA-32 | PLA-4096 | extra |
|---|---|---|---|---|---|---|---|---|
| books | sample | 2,000,000 | 1,709.1 | 4,261.7 | 27 | 5,789 | 1 |  |
| books | sample_flow | 2,000,000 | 11,384.6 | 30,804.2 | 26 | 5,789 | 3 | dup=0, unordered=0 |
| books | sample_csv | 2,199,707 | 1,875.1 | 4,565.1 | 26 | 4,730 | 1 | vp=199,707, regions=489 |
| books | sample_flow_csv | 2,199,707 | 12,517.3 | 33,812.0 | 26 | 4,730 | 3 | vp=199,707, regions=489 |
| fb | sample | 2,000,000 | 8,202.5 | 24,280.7 | 96 | 10,599 | 18 |  |
| fb | sample_flow | 2,000,000 | 11,037.8 | 33,896.0 | 89 | 10,599 | 18 | dup=0, unordered=0 |
| fb | sample_csv | 2,199,707 | 9,015.3 | 26,484.2 | 85 | 11,034 | 20 | vp=199,707, regions=489 |
| fb | sample_flow_csv | 2,199,707 | 12,139.2 | 37,214.4 | 86 | 11,033 | 20 | vp=199,707, regions=489 |
| osm | sample | 2,000,000 | 118,136.5 | 219,857.8 | 152 | 4,919 | 44 |  |
| osm | sample_flow | 2,000,000 | 111,995.7 | 213,508.7 | 164 | 4,918 | 44 | dup=0, unordered=0 |
| osm | sample_csv | 2,199,707 | 129,938.9 | 241,721.1 | 141 | 4,856 | 49 | vp=199,707, regions=489 |
| osm | sample_flow_csv | 2,199,707 | 123,184.6 | 234,738.0 | 149 | 4,859 | 49 | vp=199,707, regions=489 |
| covid | sample | 2,000,000 | 24,143.5 | 49,834.7 | 24 | 827 | 9 |  |
| covid | sample_flow | 2,000,000 | 17,664.6 | 34,341.4 | 23 | 827 | 9 | dup=526, unordered=0 |
| covid | sample_csv | 2,197,391 | 26,482.5 | 54,690.5 | 22 | 490 | 9 | vp=197,391, regions=489 |
| covid | sample_flow_csv | 2,197,565 | 19,341.2 | 37,597.6 | 22 | 489 | 9 | vp=197,565, regions=489 |
| genome | sample | 2,000,000 | 48,827.0 | 98,852.6 | 171 | 12,719 | 14 |  |
| genome | sample_flow | 2,000,000 | 45,938.1 | 89,886.7 | 160 | 12,719 | 14 | dup=0, unordered=0 |
| genome | sample_csv | 2,199,707 | 53,700.7 | 108,703.0 | 142 | 12,337 | 14 | vp=199,707, regions=489 |
| genome | sample_flow_csv | 2,199,707 | 50,523.0 | 98,766.3 | 167 | 12,330 | 14 | vp=199,707, regions=489 |
| history | sample | 2,000,000 | 9,651.7 | 21,773.6 | 8 | 943 | 6 |  |
| history | sample_flow | 2,000,000 | 18,579.0 | 48,044.9 | 8 | 943 | 6 | dup=0, unordered=0 |
| history | sample_csv | 2,194,763 | 10,512.4 | 23,859.3 | 21 | 422 | 6 | vp=194,763, regions=489 |
| history | sample_flow_csv | 2,194,819 | 20,400.1 | 52,812.8 | 20 | 418 | 6 | vp=194,819, regions=489 |
| libio | sample | 2,000,000 | 8,854.0 | 21,656.5 | 2 | 1,057 | 9 |  |
| libio | sample_flow | 2,000,000 | 13,617.6 | 44,065.3 | 2 | 1,058 | 9 | dup=0, unordered=0 |
| libio | sample_csv | 2,170,213 | 9,257.0 | 23,767.3 | 31 | 941 | 9 | vp=170,213, regions=489 |
| libio | sample_flow_csv | 2,170,632 | 15,157.8 | 48,427.9 | 20 | 944 | 9 | vp=170,632, regions=489 |
| planet | sample | 2,000,000 | 42,645.1 | 81,355.0 | 49 | 8,498 | 35 |  |
| planet | sample_flow | 2,000,000 | 47,239.9 | 100,454.0 | 47 | 8,498 | 35 | dup=0, unordered=0 |
| planet | sample_csv | 2,199,707 | 46,896.6 | 89,150.9 | 124 | 8,628 | 38 | vp=199,707, regions=489 |
| planet | sample_flow_csv | 2,199,707 | 51,950.2 | 110,460.9 | 190 | 8,636 | 37 | vp=199,707, regions=489 |
| stack | sample | 2,000,000 | 25,321.9 | 48,379.9 | 1 | 144 | 3 |  |
| stack | sample_flow | 2,000,000 | 34,350.2 | 73,343.2 | 1 | 146 | 4 | dup=0, unordered=0 |
| stack | sample_csv | 2,062,282 | 23,747.7 | 45,364.7 | 6 | 86 | 3 | vp=62,282, regions=489 |
| stack | sample_flow_csv | 2,062,394 | 33,100.3 | 71,408.8 | 10 | 88 | 4 | vp=62,394, regions=489 |
| wise | sample | 2,000,000 | 5,551.0 | 11,282.6 | 8 | 805 | 4 |  |
| wise | sample_flow | 2,000,000 | 11,518.2 | 29,991.5 | 7 | 807 | 3 | dup=0, unordered=0 |
| wise | sample_csv | 2,195,396 | 6,107.5 | 12,504.7 | 16 | 315 | 4 | vp=195,396, regions=489 |
| wise | sample_flow_csv | 2,195,323 | 12,666.0 | 32,988.4 | 10 | 317 | 4 | vp=195,323, regions=489 |

One caveat about the **window run's `full_flow` scope** (not shown in the tables above, but present in
`results/aidb_window/hardness.json <name>.full_flow`): there the 200M-key file is pushed through a flow
trained on a 2M window that spans 0.14–2.9 % of the key range. Outside the window, the tanh is
saturated, so most keys collapse onto duplicate feature values (`duplicates` = 39,473,722 for books up to
168,160,900 for osm; 0 for planet) and the block is degenerate (CD = 200,000,000 for osm; RMSE ≈ 44–45 M
on all ten) [results/aidb_window/hardness_details.json `<name>.full_flow`]. The pipeline logs a warning
when duplicates exceed half the keys [aidb_pipeline.py:491-493]. Use the uniform run's `full_flow` for
"flow applied to the full file"; the window run's `full_flow` only documents that a locally trained flow
cannot be applied globally.

#### 5.5 Per-region tail conflict degree (the index's own learnability signal)

At bulk load, every region of 4,096 keys fits its own line, normalises keys to x ∈ [0,1] over the region
(`LinearModel::normalized`, model.hpp:15) and computes NFL's tail conflict degree D₉₉ on x
(`tail_conflicts_raw`); with a flow it also computes D₉₉ on the sorted z(k) (`tail_conflicts_flow`)
[index.hpp:231-238, 245-248, 280]. The auto-switch keeps the flow for a region only if D₉₉(z) < D₉₉(x)
and the gain is at least `flow_min_gain` = 10 % of D₉₉(x) [index.hpp:281]. The sweep records the mean
over the 489 regions in `learnability.tail_conflicts_raw_mean` / `tail_conflicts_flow_mean`
[src/benchmark.cpp:72]. With 2,000,000 keys, regions = ⌈2,000,000 / 4,096⌉ = 489 (488 full regions and a
last one of 1,152 keys) [hardness.hpp:348].

**T5.** Mean per-region tail CD: raw from the `raw_rank` rows, flow from the `packed_rank_flow_forced`
rows (`--flow-bypass 0`, flow applied to all 489 regions); "kept" = `flow_regions` of the auto-switch
variant `packed_rank_flow` [results/aidb/sweep/results.jsonl and results/aidb_window/sweep/results.jsonl,
`learnability`; identical across the three query seeds since bulk load is deterministic].

| dataset | uniform raw | uniform flow (forced) | uniform regions where the auto-switch kept the flow | window raw | window flow (forced) | window regions kept |
|---|---|---|---|---|---|---|
| books | 3.2843 | 3.2843 | 26/489 | 10.7648 | 10.8037 | 4/489 |
| fb | 8.4479 | 8.4642 | 41/489 | 31.4110 | 31.4274 | 12/489 |
| osm | 36.1636 | 36.9816 | 7/489 | 9.6626 | 9.6462 | 21/489 |
| covid | 3.1575 | 3.1391 | 17/489 | 3.0184 | 3.0307 | 1/489 |
| genome | 4.2270 | 4.2290 | 8/489 | 60.4090 | 60.3824 | 9/489 |
| history | 3.1022 | 3.1043 | 13/489 | 3.1002 | 3.1084 | 2/489 |
| libio | 3.3722 | 3.3620 | 21/489 | 1.6074 | 1.6012 | 5/489 |
| planet | 7.6646 | 7.6483 | 43/489 | 35.7812 | 35.6708 | 17/489 |
| stack | 3.0082 | 3.0082 | 1/489 | 1.0000 | 1.0000 | 0/489 |
| wise | 3.0327 | 3.0348 | 4/489 | 3.0061 | 3.0041 | 1/489 |

Reading: (a) the noise floor is 3.00 (Section 5.3), so six uniform samples and four windows are at the
floor; (b) a single global 2×2 flow changes the per-region tail CD by less than 1 % on every sample
(largest change: osm uniform 36.16 → 36.98, worse) — at region scale a global monotone transform is
nearly affine, and D₉₉ is invariant under affine maps; (c) the auto-switch consequently keeps the flow
in 0–43 of 489 regions. This is the data-side reason the flow variants are neutral in the throughput
sweeps (see the results section of this guide).

---

### 6. The flows

#### 6.1 What was trained, on what, how

One weight file per sample: `results/aidb/flows/<name>_2D2H2L.txt` (uniform) and
`results/aidb_window/flows/<name>_2D2H2L.txt` (window), produced by
`tools/train_flow.py <sample> --dtype uint64 --output <weights> --sample 4096 --steps 200 --monotone`
[aidb_pipeline.py:441; run.json `flow_sample: 4096, flow_steps: 200`]. The trainer:

- reads the 2M sample, de-duplicates and sorts it (`read_keys`), draws **4,096 training keys** uniformly
  with `random.Random(1000000007)` [train_flow.py `main`, `--seed` default 1000000007];
- model = NFL's deployed shape: x = (k − mean)/var with mean = smallest training key and var =
  (max − min)/shifts, shifts = 64 (NFL's author default is 1e6 for 200M keys [train_flow.py `--shifts`
  help]); features φ(x) = [x, x − ⌊x⌋]; hidden u = φ·W₀ (2×2), h = tanh(u); z = Σ_h h_h · (W₁[h,0] +
  W₁[h,1]) (2×2 output, sum decoder) [train_flow.py `Flow.forward`; transform.hpp reads the same
  format];
- `--monotone` zeroes the two weights on the fractional feature, so z is monotone in k (a documented
  deviation from NFL, which keeps them) [train_flow.py `--monotone` help; `loss_and_grad`];
- loss = 1-D change-of-variables negative log-likelihood under N(0,1) with a barrier keeping dz/dx > 0:
  L = mean over training keys of ½z² − log(dz/dx) + barrier/(dz/dx), barrier = 1e-3; Adam, lr 0.05,
  200 steps, best-loss weights kept [train_flow.py `loss_and_grad`, `train`]. This is "a reproducible,
  dependency-free stand-in, not a BNAF" [train_flow.py docstring; training_report.json `note`].

Weight file format (fb uniform) [results/aidb/flows/fb_2D2H2L.txt]:

```
2	2	2                                  # depth, hidden, output
15162980.0000000000000000	1207648189.7031250000000000     # mean, var  (var·64 = 77,289,484,141 = training span)
2	2
0.0058556031582764	0.0058592806847908                      # W0 row 0: x-feature weights (≈ 0.375/64)
0.0000000000000000	0.0000000000000000                      # W0 row 1: fractional-feature weights = 0 (--monotone)
2	2
1.2798795760698392	0.8812895293543010                      # W1 row 0
1.4785700705830542	1.0331745748154413                      # W1 row 1
```

**Worked transform** (fb uniform flow, computed with the formulas above [scratch/tables.py]):
v = (W₁[0,0]+W₁[0,1], W₁[1,0]+W₁[1,1]) = (2.161169, 2.511745).

| key k | x = (k − mean)/var | u = x·W₀[0,·] | h = tanh u | z = h·v |
|---|---|---|---|---|
| 97,995 (sample min) | −0.012475 | (−0.000073, −0.000073) | (−0.000073, −0.000073) | −0.000341 |
| 38,752,453,239 (full p50) | 32.076635 | (0.187828, 0.187946) | (0.185650, 0.185764) | 0.867812 |
| 77,308,811,965 (sample max) | 64.003449 | (0.374779, 0.375014) | (0.358165, 0.358370) | 1.674188 |

So on fb the learned map is z ≈ 0.026·x with a slight concave bend (h/u = 0.956 at the top): nearly
affine, which is why it changes CD and PLA by nothing and per-region tail CD by < 1 %. For the five
outlier keys of the full file x ≈ 1.5e10, u ≈ 8.8e7 and tanh saturates at 1, giving z = v₀ + v₁ =
4.672914 for all five.

**T6. Training report** [results/aidb/flows/training_report.json, results/aidb_window/flows/
training_report.json]. "Whole-sample tail CD" is the trainer's own D₉₉ on the whole 2M sample, raw
(min-max normalised keys) and transformed (sorted z).

| dataset | mode | best NLL | train s | whole-sample tail CD raw | tail CD transformed | unordered pairs |
|---|---|---|---|---|---|---|
| books | uniform | 3.6104 | 2.41 | 4 | 4 | 0 |
| fb | uniform | 4.1820 | 2.39 | 8 | 8 | 0 |
| osm | uniform | 3.9238 | 2.38 | 99 | 99 | 0 |
| covid | uniform | 4.1279 | 2.38 | 3 | 3 | 0 |
| genome | uniform | 4.3172 | 2.35 | 5 | 5 | 0 |
| history | uniform | 4.1648 | 2.36 | 3 | 3 | 0 |
| libio | uniform | 4.2237 | 2.38 | 3 | 3 | 0 |
| planet | uniform | 3.3771 | 2.36 | 18 | 9 | 0 |
| stack | uniform | 4.1902 | 2.39 | 3 | 3 | 0 |
| wise | uniform | 4.1711 | 2.39 | 3 | 3 | 0 |
| books | window | 4.1814 | 2.37 | 11 | 11 | 0 |
| fb | window | 4.1721 | 2.34 | 31 | 31 | 0 |
| osm | window | 4.1628 | 2.34 | 15 | 14 | 0 |
| covid | window | 4.1543 | 2.35 | 3 | 3 | 0 |
| genome | window | 4.1544 | 2.33 | 67 | 67 | 0 |
| history | window | 4.1913 | 2.35 | 3 | 3 | 0 |
| libio | window | 4.1874 | 2.35 | 2 | 2 | 0 |
| planet | window | 4.1799 | 2.34 | 52 | 52 | 0 |
| stack | window | 4.2061 | 2.34 | 1 | 1 | 0 |
| wise | window | 4.1783 | 2.37 | 3 | 3 | 0 |

Training takes 2.33–2.41 s per sample (pure Python, 4,096 keys × 200 steps). The only whole-sample tail
CD improvement is planet uniform, 18 → 9 (its front-loaded CDF is the one shape a monotone tanh can
straighten). Compare NFL's claim that after its flow "the conflict degrees can be kept around a low value
(e.g., around 4 for the tail conflict degree)" [NFL §3.3, p.6]: our 2×2 stand-in with 4,096
training keys does not achieve that on osm (99), fb window (31), genome window (67) or planet window
(52), and this guide never claims otherwise.

#### 6.2 Effect of the transform on the hardness metrics

**T7.** Before/after for the full file (uniform-trained flow), the uniform sample and the window sample
[hardness.json `<name>.full` vs `.full_flow`, `.sample` vs `.sample_flow` in both runs]. Percentages are
(after − before)/before.

| dataset | scope | RMSE before | RMSE after | ΔRMSE | CD before | CD after | PLA-32 before | PLA-32 after | ΔPLA-32 | PLA-4096 before | after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| books | full (uniform-trained flow) | 18,053,637.5 | 15,985,128.0 | −11.5% | 246 | 245 | 262,604 | 262,604 | +0.0% | 97 | 94 |
| books | uniform sample | 180,591.3 | 159,882.7 | −11.5% | 11 | 9 | 602 | 602 | +0.0% | 8 | 8 |
| books | window sample | 1,709.1 | 11,384.6 | +566.1% | 27 | 26 | 5,789 | 5,789 | +0.0% | 1 | 3 |
| fb | full (uniform-trained flow) | 57,735,023.7 | 1,161,273.7 | −98.0% | 110 | 110 | 1,055,308 | 1,055,308 | +0.0% | 1,687 | 1,687 |
| fb | uniform sample | 1,448.2 | 11,584.6 | +699.9% | 24 | 24 | 3,034 | 3,034 | +0.0% | 1 | 3 |
| fb | window sample | 8,202.5 | 11,037.8 | +34.6% | 96 | 89 | 10,599 | 10,599 | +0.0% | 18 | 18 |
| osm | full (uniform-trained flow) | 24,177,498.5 | 23,977,602.0 | −0.8% | 4,107 | 3,943 | 661,115 | 661,115 | +0.0% | 5,495 | 5,495 |
| osm | uniform sample | 241,476.1 | 239,472.1 | −0.8% | 1,102 | 1,100 | 7,099 | 7,099 | +0.0% | 68 | 68 |
| osm | window sample | 118,136.5 | 111,995.7 | −5.2% | 152 | 164 | 4,919 | 4,918 | −0.0% | 44 | 44 |
| covid | full (uniform-trained flow) | 1,795,722.5 | 1,272,460.5 | −29.1% | 27 | 27 | 81,908 | 81,909 | +0.0% | 850 | 850 |
| covid | uniform sample | 18,073.1 | 12,906.4 | −28.6% | 8 | 8 | 1,192 | 1,193 | +0.1% | 9 | 9 |
| covid | window sample | 24,143.5 | 17,664.6 | −26.8% | 24 | 23 | 827 | 827 | +0.0% | 9 | 9 |
| genome | full (uniform-trained flow) | 7,531,940.1 | 8,390,195.9 | +11.4% | 585 | 587 | 1,290,208 | 1,290,208 | +0.0% | 1,426 | 1,425 |
| genome | uniform sample | 75,352.0 | 83,934.3 | +11.4% | 77 | 82 | 2,045 | 2,045 | +0.0% | 18 | 18 |
| genome | window sample | 48,827.0 | 45,938.1 | −5.9% | 171 | 160 | 12,719 | 12,719 | +0.0% | 14 | 14 |
| history | full (uniform-trained flow) | 815,542.7 | 432,676.2 | −46.9% | 8 | 8 | 105,468 | 105,469 | +0.0% | 468 | 467 |
| history | uniform sample | 8,104.9 | 4,290.7 | −47.1% | 7 | 7 | 1,067 | 1,068 | +0.1% | 3 | 3 |
| history | window sample | 9,651.7 | 18,579.0 | +92.5% | 8 | 8 | 943 | 943 | +0.0% | 6 | 6 |
| libio | full (uniform-trained flow) | 3,445,772.9 | 4,072,387.5 | +18.2% | 2 | 2 | 145,808 | 145,811 | +0.0% | 639 | 638 |
| libio | uniform sample | 34,472.4 | 40,748.0 | +18.2% | 8 | 7 | 1,256 | 1,255 | −0.1% | 10 | 10 |
| libio | window sample | 8,854.0 | 13,617.6 | +53.8% | 2 | 2 | 1,057 | 1,058 | +0.1% | 9 | 9 |
| planet | full (uniform-trained flow) | 29,771,622.2 | 15,545,175.8 | −47.8% | 21 | 32 | 613,597 | 613,602 | +0.0% | 2,314 | 2,313 |
| planet | uniform sample | 297,556.5 | 155,367.6 | −47.8% | 26 | 33 | 3,456 | 3,455 | −0.0% | 13 | 15 |
| planet | window sample | 42,645.1 | 47,239.9 | +10.8% | 49 | 47 | 8,498 | 8,498 | +0.0% | 35 | 35 |
| stack | full (uniform-trained flow) | 839,727.0 | 1,769,181.3 | +110.7% | 1 | 1 | 17,833 | 17,833 | +0.0% | 133 | 129 |
| stack | uniform sample | 8,395.8 | 17,705.8 | +110.9% | 8 | 7 | 593 | 594 | +0.2% | 4 | 4 |
| stack | window sample | 25,321.9 | 34,350.2 | +35.7% | 1 | 1 | 144 | 146 | +1.4% | 3 | 4 |
| wise | full (uniform-trained flow) | 2,200,401.6 | 1,511,720.5 | −31.3% | 10 | 9 | 79,035 | 79,036 | +0.0% | 382 | 382 |
| wise | uniform sample | 21,926.3 | 15,026.4 | −31.5% | 8 | 7 | 787 | 787 | +0.0% | 21 | 21 |
| wise | window sample | 5,551.0 | 11,518.2 | +107.5% | 8 | 7 | 805 | 807 | +0.2% | 4 | 3 |


How to read T7:

- **PLA-32 and PLA-4096 are invariant** under the monotone flow (changes of 0 to 5 segments out of up to
  1.29 M). That is expected: a strictly monotone reparametrisation of the x-axis preserves the ordering
  and only bends line segments slightly; PGM's optimal segmentation sees the same rank sequence.
- **CD moves by at most a few percent** (osm full 4,107 → 3,943; planet 21 → 32): the flow output is
  near-affine at cluster scale, and the CD deviation ε on real features is documented.
- **RMSE changes a lot and in both directions**, because RMSE is about global curvature and a 2×2 tanh
  is a global curvature knob: it helps front-loaded or convex CDFs (planet −47.8 %, history −46.9 %,
  wise −31.3 %, covid −29.1 %, books −11.5 %), does nothing on stepped CDFs (osm −0.8 %), and hurts
  already-straight sequences (stack +110.7 %, libio +18.2 %, genome +11.4 %); on the windows, which are
  nearly linear locally, it hurts seven of ten (books +566 %, wise +108 %, history +93 %). fb full −98 %
  is the outlier-squashing effect explained in Section 3.2, not a real gain for the bulk (fb uniform
  sample: +700 %).
- The uniform-sample rows track the full-file rows to three significant figures (books −11.5/−11.5,
  covid −29.1/−28.6, history −46.9/−47.1, planet −47.8/−47.8, wise −31.3/−31.5): the flow was trained
  on 4,096 keys of the uniform sample, and the uniform sample has the full file's global shape
  (Section 5.2). This is the data-level reason our full-file `full_flow` scope can be trusted for RMSE
  but tells you nothing new about PLA or CD.

---

### 7. The CSV-augmented scopes (`sample_csv`, `sample_flow_csv`)

#### 7.1 How they are produced

`scaleli_hardness --virtual-alpha 0.1 --region-keys 4096` computes a third metric block, `smoothed`, on
"the CSV-style augmented sequence (alpha × region virtual points per region; features are the
transformed keys with --flow, else double(key − min))" [src/hardness.cpp:87-88, 166-181]; `--alpha 0.1`
is the pipeline default, "CSV-style virtual-point budget per region" [aidb_pipeline.py:606], the same
α = 0.1 default the CSV paper uses ("We vary the smoothing threshold, α, from 0.05 to 0.8, with a default
value of 0.1" [CSV §6.1 "Parameters"]). The full 200M files are not smoothed (cost O(α·n·region_keys)
loss evaluations, "meant for samples (a few million keys), not 200M keys" [hardness.hpp:341-345]).

The smoother is our clean-room reading of CSV Algorithm 1 (no public implementation exists as of
September 2026) [smoothing.hpp header]:

```
smooth_regions(features f[0..n), region_keys R = 4096, α = 0.1):          # hardness.hpp:346-378
  regions = ⌈n / R⌉ = 489 for n = 2,000,000 (488 regions of 4,096 keys + one of 1,152)
  for each region independently (threads):  smooth_cdf(f[b .. b+len), α)   # smoothing.hpp
      budget λ = ⌊α · len⌋  (= 409 for a full region, 115 for the last one)
      seq = the region's features, targets = positions 0..len−1
      repeat up to λ times:
          for every gap (seq[i], seq[i+1]) with seq[i+1] > seq[i]:
              probe the OLS SSE of (seq ∪ {virtual point at x_v inserted after i}) at
              x_v = lo+e, lo+2e, hi−2e, hi−e  (e = 1e-3 · gap);  if the loss decreases from the left AND
              from the right, ternary-search the interior minimum (≤ 40 steps), else take the better end
          if no candidate lowers the SSE (beyond 1e-12 relative): stop      # CSV Alg. 1 line 27
          insert the best virtual point; every later target shifts by +1
      return slot(k) = rank + #virtual points before k, and the virtual features
  concatenate the per-region augmented sequences: position j = slot j of the concatenated slot spaces
```

Loss = OLS residual SSE over real AND virtual points (CSV's equation 4), evaluated in O(1) per candidate
from running and suffix sums, so one round is O(gaps) and a region costs O(λ·R) [smoothing.hpp header,
`loss_at`]. Virtual points are slot offsets only: "never stored as records and never affect logical rank"
[smoothing.hpp header]. The metrics of the `smoothed` block are then computed on the augmented sequence
of n + (virtual points) features with target = augmented position, i.e. **the slot sequence**
[src/hardness.cpp:172-176].

**Budget arithmetic** (verified against the files): 488 × ⌊0.1 × 4,096⌋ + ⌊0.1 × 1,152⌋ = 488 × 409 + 115
= **199,707** = the `virtual_points` count on every dataset that used its whole budget (fb, genome,
planet-window, books-window, osm-window, …) [hardness_details.json `<name>.sample_csv.virtual_points`].
Smaller counts mean the early stop fired in some regions: the loss could not be lowered any more. That
happens where the CDF is already straight: stack window used 62,282 (31.2 %), libio window 170,213
(85.2 %), stack uniform 184,990, books uniform 185,327; and on the linear control sample 179,985
[linear_uniform_hardness.json `smoothed.virtual_points`].

#### 7.2 Effect per dataset

**T8.** `sample` vs `sample_csv` [hardness.json both runs; vp from hardness_details.json].

| dataset | mode | virtual points | of budget 199,707 | RMSE sample | RMSE sample_csv | ΔRMSE | CD | CD csv | PLA-32 | PLA-32 csv | ΔPLA-32 | PLA-4096 | csv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| books | uniform | 185,327 | 92.8% | 180,591.3 | 196,846.2 | +9.0% | 11 | 10 | 602 | 236 | −60.8% | 8 | 8 |
| books | window | 199,707 | 100.0% | 1,709.1 | 1,875.1 | +9.7% | 27 | 26 | 5,789 | 4,730 | −18.3% | 1 | 1 |
| fb | uniform | 199,707 | 100.0% | 1,448.2 | 1,591.2 | +9.9% | 24 | 23 | 3,034 | 1,718 | −43.4% | 1 | 1 |
| fb | window | 199,707 | 100.0% | 8,202.5 | 9,015.3 | +9.9% | 96 | 85 | 10,599 | 11,034 | +4.1% | 18 | 20 |
| osm | uniform | 199,605 | 99.9% | 241,476.1 | 265,521.5 | +10.0% | 1,102 | 1,099 | 7,099 | 7,491 | +5.5% | 68 | 76 |
| osm | window | 199,707 | 100.0% | 118,136.5 | 129,938.9 | +10.0% | 152 | 141 | 4,919 | 4,856 | −1.3% | 44 | 49 |
| covid | uniform | 198,862 | 99.6% | 18,073.1 | 19,819.2 | +9.7% | 8 | 10 | 1,192 | 892 | −25.2% | 9 | 11 |
| covid | window | 197,391 | 98.8% | 24,143.5 | 26,482.5 | +9.7% | 24 | 22 | 827 | 490 | −40.7% | 9 | 9 |
| genome | uniform | 199,707 | 100.0% | 75,352.0 | 82,874.1 | +10.0% | 77 | 71 | 2,045 | 1,627 | −20.4% | 18 | 19 |
| genome | window | 199,707 | 100.0% | 48,827.0 | 53,700.7 | +10.0% | 171 | 142 | 12,719 | 12,337 | −3.0% | 14 | 14 |
| history | uniform | 195,413 | 97.8% | 8,104.9 | 8,628.3 | +6.5% | 7 | 21 | 1,067 | 411 | −61.5% | 3 | 3 |
| history | window | 194,763 | 97.5% | 9,651.7 | 10,512.4 | +8.9% | 8 | 21 | 943 | 422 | −55.2% | 6 | 6 |
| libio | uniform | 195,924 | 98.1% | 34,472.4 | 37,442.8 | +8.6% | 8 | 21 | 1,256 | 614 | −51.1% | 10 | 11 |
| libio | window | 170,213 | 85.2% | 8,854.0 | 9,257.0 | +4.6% | 2 | 31 | 1,057 | 941 | −11.0% | 9 | 9 |
| planet | uniform | 199,656 | 100.0% | 297,556.5 | 327,255.8 | +10.0% | 26 | 37 | 3,456 | 2,767 | −19.9% | 13 | 15 |
| planet | window | 199,707 | 100.0% | 42,645.1 | 46,896.6 | +10.0% | 49 | 124 | 8,498 | 8,628 | +1.5% | 35 | 38 |
| stack | uniform | 184,990 | 92.6% | 8,395.8 | 9,036.9 | +7.6% | 8 | 7 | 593 | 319 | −46.2% | 4 | 4 |
| stack | window | 62,282 | 31.2% | 25,321.9 | 23,747.7 | −6.2% | 1 | 6 | 144 | 86 | −40.3% | 3 | 3 |
| wise | uniform | 194,693 | 97.5% | 21,926.3 | 24,008.5 | +9.5% | 8 | 9 | 787 | 436 | −44.6% | 21 | 22 |
| wise | window | 195,396 | 97.8% | 5,551.0 | 6,107.5 | +10.0% | 8 | 16 | 805 | 315 | −60.9% | 4 | 4 |

How to read T8 (and why the global metrics look worse while the local one looks better):

- **RMSE/ME rise by ≈ +10 %**, i.e. by exactly the factor (n + λ)/n = 1.1 when the budget is used up.
  The global line is fitted to slots, not ranks; inserting 10 % more targets stretches the y-axis by 1.1
  and every residual with it. This is not a degradation, it is the unit change; RMSE/(n+λ) is unchanged.
  Where fewer points were inserted the rise is smaller (history uniform +6.5 %, stack window −6.2 % with
  31 % of the budget: there the smoother actually straightened the window a little).
- **PLA-32 falls sharply on the smooth datasets**: −60.8 % books, −61.5 % history, −60.9 % wise window,
  −51.1 % libio, −46.2 % stack, −43.4 % fb uniform, −40.7 % covid window. The per-region smoother
  straightens the slot sequence inside each 4,096-key region, so a 32-slot corridor covers more keys.
  On the linear control sample it does the same (535 → 225 [linear_uniform_hardness.json]): the gain on
  those datasets is mostly the removal of sampling jitter, not of dataset structure.
- **PLA-32 does not fall where the roughness is stronger than 10 % of the keys can absorb**: fb window
  +4.1 %, osm uniform +5.5 %, planet window +1.5 %, genome window −3.0 %, osm window −1.3 %. This is the
  data-side limit of a fixed α: CSV's own claim is a per-key improvement for "the keys in deeper levels"
  [CSV Abstract], not a global PLA reduction.
- **CD goes up on the easy datasets** (history 7 → 21, libio 8 → 21 uniform / 2 → 31 window, planet window
  49 → 124, wise window 8 → 16): the smoother is greedy on the OLS loss, and a virtual point placed 1e-3
  of a gap away from a real key (the `lo+e` / `hi−e` candidates) creates a feature pair closer than
  FMCD's slot width, so LIPP-style slot mapping sees a new "cluster". The index does not use CD to place
  keys (virtual points are slot offsets), but the metric records it honestly. On the hard datasets CD
  drops slightly (osm 1,102 → 1,099, genome 171 → 142, fb window 96 → 85).
- **PLA-4096 is unchanged or +1–8**: the smoother works inside 4,096-key regions, and the region
  boundaries are exactly where an ε = 4,096 segmentation can already absorb everything.

`sample_flow_csv` (flow first, then virtual points on z) behaves like `sample_csv` on every dataset
(T4a/T4b): the two mechanisms act on different scales and do not interact at the metric level.

---

### 8. Questions the supervisor may ask, with answers

1. **Why 2M-key samples and not the full 200M keys?** The full files are used for the hardness metrics
   (`full`, `full_flow`: 200,000,000 keys each, exact `__int128` arithmetic, ~17 s per file
   [hardness_details.json `full.elapsed_ns` = 17.4 s for fb]). The index sweeps run on 2M-key samples
   because (a) the CSV-style smoother costs O(α·n·R) loss evaluations and the pipeline documents it as
   meant for "a few million keys, not 200M" [hardness.hpp:344-345]; (b) the sweep is 10 datasets × 10
   variants × 3 seeds = 300 runs per lane, and each run bulk-loads and instruments every probe; (c) the
   AIDB paper's own protocol ("bulk load ... then perform random lookups for all keys", 20M warm-up +
   100M measured lookups, two Xeon Gold 5118 sockets [AIDB §4.1]) is a server protocol we did not
   attempt to reproduce on a laptop. `results/aidb_fullscale/` holds the separate full-scale
   configurations (see the results section). The sample size is `--sample 2000000` [aidb_pipeline.py:601;
   run.json].

2. **Is a uniform 1 % sample representative?** For global shape, yes, exactly: RMSE and ME scale by
   p = 0.01 to within 1 % on nine datasets (T9), and the flow's effect on RMSE is the same on the sample
   and the full file to three digits (T7). For local structure, no: an ε = 32 corridor in the sample is
   ±3,200 full ranks plus binomial jitter of up to 1,407 ranks, so PLA-32 collapses to a noise floor of
   ~535 segments (Section 5.3) and per-region tail CD to 3.0 on six datasets. That is exactly why the
   window lane exists.

3. **Is a contiguous 2M window representative?** For local structure it is a faithful 1/100 of the file
   on fb, covid, genome, wise (window PLA-32 × 100 within 2 % of full PLA-32) and within 40 % on osm,
   libio, stack, history; on the non-stationary books and planet the 85.8 % quantile slice is 1.4–2.2×
   rougher than the file average (T9). For global shape it is useless by construction (1 % of the key
   range at one place). We report both lanes separately and never average them.

4. **Why is the window at rank 171,644,825 for every dataset?** Because `random.Random(42).randrange
   (198,000,001)` is evaluated with the same seed for every dataset [tools/datasets.py:81]; the seed is
   `--sample-seed 42` [aidb_pipeline.py:602]. The offset is not in the manifest but is reproducible, and
   we verified that each window's first/last keys equal the source keys at ranks 171,644,825 and
   173,644,824 [scratch/data_stats.json]. A different seed would give a different slice; we did not run a
   multi-offset study (see the "unverified" list at the end of this guide).

5. **Which datasets are hard, and why?** It depends on the scale you ask about, which is the AIDB
   paper's thesis [AIDB §4.2]. Globally (RMSE/ME/PLA-4096): fb (five outliers up to 2⁶⁴ − 1 make the
   least-squares line useless: RMSE = n/√12 exactly), planet (front-loaded CDF, RMSE/n 0.149), osm
   (stepped, clustered S2 cells: CD 4,107, PLA-4096 5,495), books (concave, skewed). Locally (window
   PLA-32, per-region tail CD): genome (12,719; 60.4), fb (10,599; 31.4), planet (8,498; 35.8), books
   (5,789; 10.8). Easy at every scale: stack (CD 1, 84 % dense integers), then wise, covid, history,
   libio (per-region tail CD ≈ 3 or below, PLA-32 windows 144–1,057).

6. **Your history and libio numbers: do they match the paper?** Yes, digit for digit at the paper's
   precision: history PLA-32 105,468 / PLA-4096 468 vs "h_l = 105k, h_g = 468", libio 145,808 / 639 vs
   "h_l = 146k, h_g = 639" [AIDB §2.2; hardness.json]. The paper prints no other dataset's coordinates
   (only Figures 2 and 6), so the other eight cannot be cross-checked numerically.

7. **Why is fb's RMSE exactly 57.7 M?** Because 199,999,979 of its keys are below 1.1·10¹² and five are
   above 10¹⁹, so the least-squares line is flat over the bulk and the residual against ranks 0..n−1 is
   that of a constant predictor: RMSE = n/√12 = 57,735,026.9 (measured 57,735,023.7), ME = n/2 (measured
   99,999,995.5). The uniform sample happened to draw none of the 21 keys ≥ 2⁴⁰ (probability 0.99²¹ =
   0.81), which is why the sample's RMSE is 1,448 and the ratio is 39,866 instead of 100. Nothing in our
   code special-cases fb.

8. **Did you sort or de-duplicate the data, and is that legitimate?** Seven files are served unsorted;
   GRE's own loader sorts at load time [aidb_pipeline.py:22-23]. Our `sort` step sorts them once,
   audits duplicates (0 in all ten) and stores the sorted copy with its own sha256 [provenance.json
   `sort_audit`]. No key was added or removed (`written_keys: 200000000`), so the sorted files are the
   same key sets the paper describes ("200 million 64-bit unsigned integer keys without duplicates"
   [AIDB §4.1]).

9. **Does the flow change the hardness metrics?** PLA-32/PLA-4096: no (monotone reparametrisation;
   changes ≤ 5 segments). CD: by a few percent. RMSE: yes, ±50 %, in both directions, because a 2×2
   tanh is a global curvature knob (helps planet/history/wise/covid, hurts stack/libio/genome and almost
   every window). Per-region tail CD: < 1 % (T5), so the NFL-style auto-switch keeps the flow in only
   0–43 of 489 regions. None of this contradicts NFL, which trains a much larger BNAF on the full key
   set; it says that our small clean-room control cannot reproduce NFL's "around 4" tail conflict
   degree on osm, fb-window, genome-window or planet-window.

10. **What does α = 0.1 mean concretely and why do some samples have fewer than 199,707 virtual
    points?** λ = ⌊0.1 × region size⌋ virtual slots per 4,096-key region: 409 per full region, 115 in
    the 1,152-key tail region, 199,707 in total. Fewer means the greedy loop stopped because no candidate
    lowered the region's OLS SSE (CSV Algorithm 1's stop condition) — e.g. stack window used 62,282
    because the sequence is almost 1, 2, 3, …. The budget is per region, never global, and virtual points
    are never materialised as records.

11. **Why do the CSV scopes show higher RMSE and CD if smoothing is supposed to help?** RMSE rises by
    the slot-count factor 1.1 (units, not fit quality); CD rises on smooth datasets because the greedy
    OLS placement puts virtual points 1e-3 of a gap from real keys, which FMCD reads as clusters. The
    metric that measures what CSV targets, PLA-32, drops by 20–60 % on the smooth datasets and by 0–5 %
    on the rough ones (fb window, osm, planet window). The index's own signal (rank SSE before/after in
    the sweep's learnability block) drops by 16–98 % (osm uniform 2.72e11 → 2.29e11 = −16 %; wise window
    1.27e9 → 6.17e7 = −95 %; stack window −98 %) [results/*/sweep/results.jsonl `learnability.rank_sse_before/after`].

12. **How were "duplicates after the transform" handled?** The trainer zeroes the fractional-feature
    weights (`--monotone`), so z is monotone and `unordered_pairs = 0` on all 20 samples. On the window
    lane, covid's flow still produced 526 duplicate z values among 2M keys (tanh resolution on a 1e17
    range compressed to 1 % of the range) [results/aidb_window/hardness_details.json
    `covid.sample_flow.duplicates`]; `scaleli_hardness` reports duplicates and PGM-style PLA handles
    them with the `next(x)` rule [hardness.hpp:303-308]. Applying a window-trained flow to the full file
    collapses up to 84 % of the keys onto duplicates and is reported as degenerate (Section 5.4 caveat).

13. **What would you run next on the data side?** (a) Several window offsets per dataset (seeds) to put
    error bars on the window lane; (b) the `strided` sampler, which exists in the code but was not used,
    as a third lane that keeps global shape without binomial jitter; (c) larger flows (more hidden
    units, all keys) to test whether NFL's "around 4" tail conflict degree is reachable on osm and
    genome; (d) a full-file CSV smoothing run with a bounded-window implementation, since the current
    O(λ·R) smoother is the reason the `*_csv` scopes exist only for samples.

---

### Unverified items in this section

- [unverified: not stated in the papers] what integer each dataset's key encodes beyond the one-line
  Table 1 description (books "sales popularity" value, fb upsampling procedure, covid tweet-ID layout,
  genome locus-pair encoding, wise "partition key", history/planet "node ID" vs "planet ID"); the CSV
  paper's "Google S2 cell IDs" for osm is the only extra detail available.
- [unverified: observed only] the five fb keys ≥ 2⁶³ spaced by ≈ 2⁶⁴/9 are in the file as served (sha256
  verified); whether they are intentional sentinels of the SOSD generator or artefacts is not stated in
  the papers.
- [unverified: single seed] all "window" statements are for one offset (rank 171,644,825, seed 42); no
  multi-offset variance was measured.
- [unverified: field absent] transformed-duplicate counts for the uniform lane's `sample_flow` and
  `full_flow` blocks are not recorded in `results/aidb/hardness_details.json` (the window lane records
  them); `unordered_pairs = 0` is recorded for both lanes.


---

## 05 — The host index and the clean-room implementations, with worked examples

**What this section gives you.** This is the "how does the code actually work" section. It walks through the SCALE-LI experimental map (the host index that every control runs inside: regions of 4,096 keys, blocks of 128 keys, packed codecs, fences, delta, compaction, parallel bulk load), the exact lookup path and the precise meaning of every software counter, then each clean-room control in turn: the affine `LinearModel`, the NFL-style flow transform and its stand-in trainer, the bypass rule, the CSV-style smoothing (`smooth_cdf`) with a full derivation of the O(1) closed-form loss, how the index consumes virtual points, the per-region selector (`Fusion::Auto`), the learned root (`Index::fit_root`, including the sentinel-fence bug), the hardness tool (least squares, LIPP's FMCD, PGM's optimal PLA) and the AIDB scorer. Every mechanism comes with a worked numeric example that was actually computed (Python, or a tiny C++ program compiled against the headers; scripts are under `results/aidb/guide_deep/scratch/`). Everything here is a clean-room CONTROL inside the SCALE-LI experimental map, written from the papers' descriptions; nothing is a reproduction of NFL/AFLI, CSV or the AIDB benchmark. Notation: keys k₁ < … < kₙ (uint64), rank r(kᵢ) = i − 1 ∈ {0..n−1}, linear model f(k) = w·φ(k) + b with feature φ, slot s(kᵢ) = rank + number of virtual points before kᵢ, budget λ = α·n, tail conflict degree D₉₉, region = 4,096 keys, block = 128 keys.

Code paths are relative to `S = /Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli`. Line numbers refer to the files as read on 2026-09-22.

---

### 1. The SCALE-LI experimental map (the host)

#### 1.1 What it stores

The index maps `Key = uint64` to `Value = uint64` (`Record = pair<Key,Value>`) [types.hpp:11-13]. Input rows are canonicalised (stable sort by key, last duplicate wins) [types.hpp:38-47]. The structure is two levels:

- `Index` owns `vector<unique_ptr<Region>> regions_`, a `Config`, a root model and a slot→region table [index.hpp:327-334].
- `Region` owns `low_fence` (the smallest key it can hold), `vector<BlockDescriptor> blocks`, a byte `arena` holding all encoded key blocks back to back, `vector<Value> values` (uncompressed, one per key, in rank order), a sorted `delta` of pending writes, two `LinearModel`s (`rank_model`, `byte_model`), heat statistics and the learnability diagnostics (`flow_used`, `tail_conflicts_raw/flow`, `virtual_points`, `virtual_features`, `rank_sse_before/after`, `smoothing_ns`, `transform_ns`, `choice`, `cost_none`, `cost_selected`) [index.hpp:87-104].

`Config` defaults: `region_keys=4096, block_keys=128, delta_limit=64, restart_interval=16, routing=Byte, policy=MinBytes, min_saving_fraction=0.05`, learnability knobs `flow=nullptr, flow_bypass=true, flow_min_gain=0.10, virtual_alpha=0, relearn_on_compaction=false, fusion=Manual, root=Binary, root_alpha=0, flow_cost=4.0, build_threads=1` [index.hpp:43-65]. `validate()` enforces `virtual_alpha ∈ [0,1)`, `flow_min_gain ∈ [0,1]`, `root_alpha ∈ [0,64)`, `1 ≤ block_keys ≤ region_keys ≤ 2²⁴`, `delta_limit ≤ 2·region_keys` [index.hpp:66-77]. The benchmark binary maps these to `--region-keys --block-keys --routing rank|byte|binary --policy raw|min_bytes|... --flow --flow-bypass --flow-min-gain --virtual-alpha --relearn --fusion manual|auto --flow-cost --root binary|model --root-alpha --build-threads` [benchmark.cpp:189-193]. The ten uniform-sweep variants are `sorted_vector` (a different index, `--index sorted_vector`), `raw_rank` (`--routing rank --policy raw`), `packed_rank` (`--routing rank --policy min_bytes`), `packed_byte` (`--routing byte --policy min_bytes`, the only variant with byte routing), and six learnability variants that add flags on top of `packed_rank`: `packed_rank_flow` (`--flow W --flow-bypass 1`), `packed_rank_flow_forced` (`--flow-bypass 0`), `packed_rank_vp10` (`--virtual-alpha 0.1`), `packed_rank_flow_vp10` (flow + bypass + α = 0.1), `packed_rank_fusion_auto` (flow + α = 0.1 + `--fusion auto`), `packed_rank_flow_costsel` (flow + `--fusion auto`, no virtual points) [results/aidb/sweep/environment.json config.variants; results.jsonl has these 10 variant names]. So every scaleli variant except `packed_byte` uses rank routing, every packed variant uses `min_bytes`, and only `raw_rank` uses `raw`. The root sweeps (`results/aidb_final`, `results/aidb_fullscale`) add `packed_rank_root_{raw,flow,vf4,fusion}`, `packed_rank_vp10_root_fusion`, `packed_rank_root_vf004`, `packed_rank_vp10_root_vf004` (`--root model`, `--root-alpha 4` or `0.04`) [results/aidb_final/config.json; results/aidb_fullscale/config*.json].

#### 1.2 Regions and blocks; bulk load

`bulk_load` cuts the canonical row vector into consecutive slices of `region_keys` (so 2,000,000 keys → ⌈2,000,000/4096⌉ = 489 regions, the last one holding 2,000,000 − 488·4096 = 1,152 keys; 200,000,000 keys → 48,829 regions). Region j gets `low_fence = rows[j·4096].first`, except region 0 whose fence is the sentinel 0 [index.hpp:432]. Each region is built independently by `Region::rebuild`; with `build_threads = W > 1` a static interleaved partition (thread w builds regions w, w+W, …) runs in a thread pool, exceptions are captured and rethrown, and the result is bit-identical to the serial build because each rebuild touches only its own `Region` plus read-only config/flow [index.hpp:433-441]. Test `fusion_auto` proves determinism: `dump_layout` output for `build_threads ∈ {1, 4}` must be byte-identical [test_main.cpp:143-149]. After the regions exist, `fit_root()` runs [index.hpp:443]. Timings (`build_ns`) are the wall clock of the parallel build [benchmark.cpp:65]; the AIDB sweeps used `--build-threads 16` [results/aidb/sweep/environment.json config.common].

Inside `rebuild` [index.hpp:199-317], keys are cut into blocks of `block_keys` (128; 4,096/128 = 32 blocks per full region). For each block:

1. `encode_block(part, Codec::Raw)` is always produced first [index.hpp:209].
2. Under `Policy::MinBytes` (the default, called "packed"), the three packed codecs `For`, `Delta`, `Linear` are tried; a candidate replaces the current choice iff it saves at least `min_saving_fraction` = 5 % against RAW (`size ≤ raw·0.95`) AND has a strictly smaller score (= its byte size under MinBytes; under `Policy::Smooth`/`Adaptive` the score is |size − target| with a per-block byte target computed from the block's key-span share) [index.hpp:211-226]. The target as coded [index.hpp:205, 216-219]:

```
span   = keys.back() − keys.front()                       (region key span; 1 if one key)
left   = 0 if first block, else ((keys[begin−1] − keys[0]) + (keys[begin] − keys[0]))/2
right  = span if last block, else ((keys[end−1] − keys[0]) + (keys[end] − keys[0]))/2
target = (16·⌈n/block_keys⌉ + 8·n) · (right − left)/max(1, span) · smooth_scale
```
i.e. the region's RAW byte budget (16-byte headers + 8 bytes per key) is shared out in proportion to the key-span each block covers, measured between the midpoints to its neighbours, so dense intervals get smaller byte targets. Two guards: the whole codec trial is skipped when the region is flagged `hot` by the `Adaptive` policy (`!hot` at index.hpp:211: a hot region keeps Raw blocks, the "raw bypass"), and `validate()` additionally requires `flow_cost ≥ 0` and finite [index.hpp:68].
3. `Policy::Raw` skips step 2; `Policy::Forced` uses `forced_codec` [index.hpp:210].
4. A `BlockDescriptor{first, last, rank_begin, count, offset, length, codec, slot_begin = rank_begin}` is appended and the bytes go to the arena [index.hpp:227-229].

The four codecs [codec.hpp:14,83-130]. Every block starts with a 16-byte header of which only three fields are written for every codec: byte 0 = codec tag, bytes 4-7 = count (LE32), bytes 8-15 = first key (LE64) [codec.hpp:88-90]. Byte 1 (bit width w) is written only by `For` and `Linear` [codec.hpp:97,111], bytes 2-3 (restart interval, LE16) only by `Delta` [codec.hpp:119], and `Linear` stores 16 extra bytes after the header (`span` at bytes 16-23, `lo` at 24-31) so its bit-packed payload starts at byte 32 = bit 256, whereas `For`'s starts at byte 16 = bit 128 [codec.hpp:112-116,99]:

| Codec | Payload | Random access `key_at(i)` |
|---|---|---|
| `Raw` | 8 bytes per key, little endian | direct [codec.hpp:136-138] |
| `For` (frame of reference) | `w = bit_width(last − first)` bits per key, values `k − first` bit-packed [codec.hpp:96-101] | `get_bits` at bit 128 + i·w [codec.hpp:139-143] |
| `Linear` | residual `r_i = (k_i − first) − interpolation(span, i, n)` with exact `__int128` interpolation `⌊span·i/(n−1)⌋`; stores `span`, `lo`, and `w = bit_width(hi − lo)` bits per key at bit 256 + i·w; falls back to Raw if residuals exceed int64 [codec.hpp:102-117,80-82] | recompute interpolation + residual [codec.hpp:144-150] |
| `Delta` | restart every 16 keys: an 8-byte absolute key, then LEB128 varints of consecutive differences; a 4-byte offset table per restart group [codec.hpp:118-127] | jump to the restart group, then decode ≤ 15 varints [codec.hpp:151-161] |

`key_at` increments `key_at_calls`, `decoded_keys` and `codec_bytes_examined` [codec.hpp:134-160]. Test `codecs()` round-trips every codec over 12 sizes × 7 key shapes (including keys at 2⁶⁴−1 and all-equal keys) [test_main.cpp:13-39].

#### 1.3 Fences, delta, compaction

- **Fences.** Region-level: `low_fence` (region j owns keys in [low_fence_j, low_fence_{j+1})); `validate()` throws "fence ownership" if violated [index.hpp:503-507]. Block-level: `BlockDescriptor::first/last` are exact stored keys; the block search compares against `last` [index.hpp:135].
- **Delta.** Writes never touch the arena. `upsert`/`erase` locate the region, then `set_delta` inserts or overwrites a `{key, value, deleted}` entry in a sorted vector [index.hpp:164-168,449-459]. `find` checks the delta first by binary search (`delta_probes`) [index.hpp:113-117,156-157].
- **Compaction.** When `delta.size() ≥ delta_limit` (64), `compact(j)` materialises base+delta (`scan_into` over all blocks, merging deletions), and either rebuilds the region in place or, if more than `2·region_keys` rows, splits it into two regions (the right one gets `low_fence = rows[mid].first`) and refits the root [index.hpp:402-425]. The previous region is passed as `previous` so the rebuild can reuse the flow decision and re-place the virtual points (section 6). `maintain()` compacts every region [index.hpp:467].

#### 1.4 The lookup path, step by step

`Index::find(k)` [index.hpp:445-447]:

```
1. j = locate_region(k)                         -> root_probes
2. r = regions_[j]; r.observe(false)            (heat bookkeeping)
3. Region::find(k):
   a. d = delta_lower(k)                        -> delta_probes; hit in delta returns immediately
   b. b = locate_block(k):
        i.   p = predicted_block(k)             -> transform_calls (if flow), coordinate_probes
        ii.  exponential bracket + binary correction on blocks[].last
                                                -> fence_probes, block_routes, correction_distance
   c. binary search inside block b by key_at()  -> key_at_calls, decoded_keys, codec_bytes_examined
   d. return values[b.rank_begin + lo] if key matches
```

**Step 1 — `Index::locate_region`** [index.hpp:348-357]. With `Root::Binary` (or a root model that was not accepted) it is a plain binary search over `regions_[m]->low_fence <= k`, incrementing `root_probes` once per comparison [index.hpp:352-354]; the answer is the last region whose fence ≤ k. With an accepted `Root::Model` it calls `locate_from_prediction(ok, root_model_.predict_x(root_feature(k)), n, &root_slot_to_region_)` [index.hpp:356]: the prediction y is turned into a starting region p exactly as follows [index.hpp:341-343]:

```
with a slot table of size m+1 = n_regions + virt:   slot = 0 if y ≤ 0;  m if y ≥ m;  else ⌊y⌋ (size_t truncation)
                                                    p = table[slot]
without a table:                                    p = 0 if y ≤ 0;  n−1 if y ≥ n−1;  else ⌊y⌋
```
The table is filled at fit time so that every slot in [sm.slot[j], sm.slot[j+1]) maps to region j (the last region owns the slots up to n + virt) [index.hpp:386-387]: a virtual fence inserted between fences j and j+1 therefore belongs to region j. Numeric instance: the fb_uniform root chose raw + 1 virtual fence over 489 regions, so the table has 490 `uint32` entries = 1,960 B, which is exactly the `slot=1960` metadata delta of `root_vf4` in the cost-story table [results/aidb_final/sweep/results.jsonl fb_uniform packed_rank_root_vf4 root_virtual = 1; results/aidb/verification/coststory/coststory.txt metadata table]. Note the two search semantics: the root's predicate is `ok(i) = low_fence_i ≤ k` and the answer is `last_true` (the largest true index), whereas the block search's predicate is `less(i) = last_i < k` and the answer is `lower` (the smallest false index) [index.hpp:337-339,135-137]. Then `ok(p)` is tested; if true, exponential probing upwards (step 1, 2, 4, …) until `ok` fails, then `last_true` binary search; if false, exponential probing downwards, then `last_true` [index.hpp:340-347,337-339]. Every `ok(i)` call increments `root_probes` [index.hpp:350].

**Step 3.b.i — `Region::predicted_block`** [index.hpp:118-129]. `f = feature(k)` (either `rank_model.normalized(k)` = (k − origin)/span, or `(*flow)(k)` when `flow_used`, which counts a `transform_calls`) [index.hpp:106; transform.hpp:65]; `y = rank_model.predict_x(f)` under `Routing::Rank` (`byte_model` under `Routing::Byte`); then a binary search over the blocks' `slot_begin` (rank routing) or `offset` (byte routing) for the last block whose boundary ≤ y, each comparison incrementing `coordinate_probes` [index.hpp:122-127]. The result is a *hint*, never trusted.

**Step 3.b.ii — `Region::locate_block`** [index.hpp:133-154]. `less(i) := blocks[i].last < k` and every call increments `fence_probes` [index.hpp:135]. `Routing::Binary` simply binary-searches all blocks [index.hpp:139]. Otherwise, with prediction p:

```
if less(p):                       # answer is to the right
    lo = p+1; hi = lo; step = 1
    while hi < n and less(hi): lo = hi+1; step = min(n, 2*step); hi = min(n, p+step)
    out = lower(lo, hi < n ? hi+1 : n)     # binary search in the bracket
elif p == 0 or less(p-1): out = p         # exact hit
else:                             # answer is to the left
    hi = p; step = 1; lo = p - step
    while lo > 0 and not less(lo): hi = lo; step = min(n, 2*step); lo = p > step ? p-step : 0
    if less(lo): lo += 1
    out = lower(lo, hi+1)
correction_distance += |out - p|; max_correction_distance = max(...); block_routes += 1
```
[index.hpp:140-153]. The exponential phase costs O(log d) probes for a prediction that is d blocks off, then a binary search inside a bracket of ≤ 2d blocks: total O(log d), and correctness never depends on the model (test `adversarial_routes` sets `slope=0, intercept=±1e15` and still requires exact answers [test_main.cpp:77-79]).

**Step 3.c — in-block search** [index.hpp:159-161]: lower-bound binary search over `count ≤ 128` keys with `key_at` [index.hpp:160], then one more `key_at` for the hit check `if(lo<b.count && key_at(bytes(b),lo,s)==k)` [index.hpp:161]. The loop `while(lo<hi){m=lo+(hi−lo)/2; if(key_at(m)<k) lo=m+1; else hi=m;}` over 128 keys makes 7 or 8 calls depending on where the answer lands (simulated over all 129 outcomes: 127 cost 7, 2 cost 8 [scratch/fix05_checks.py]), so a successful lookup costs **8 or 9** `key_at` calls, never fewer than 8; the AIDB runs measure `key_at_calls/op = 8.016` on every variant [results/aidb_final/sweep/results.jsonl, fb_uniform packed_rank, seed 11].

#### 1.5 The counters, precisely

`QueryStats` [types.hpp:17-24] is filled only when a non-null pointer is passed; the benchmark collects them in a separate pass (pass 4) so they never contaminate timing, and it checks that the instrumented replay produces the same result checksum as the timed replay [benchmark.cpp:111-115].

| Counter | Incremented at | Meaning |
|---|---|---|
| `root_probes` | `locate_region`: binary search comparison [index.hpp:353] or each `ok(i)` of the learned root [index.hpp:350] | number of *region fence comparisons* (`low_fence <= k`) per operation |
| `coordinate_probes` | `predicted_block` binary search over `slot_begin`/`offset` [index.hpp:124] | comparisons of the model's real-valued prediction y against block boundaries; the lower-bound loop of index.hpp:123-127 over 32 boundaries has 33 possible outcomes and costs 5 or 6 comparisons (31 outcomes cost 5, 2 cost 6; the 9-block last region costs 3 or 4) [scratch/fix05_checks.py], which is why the measured mean is 5.029, slightly above 5 [results/aidb_final/sweep/results.jsonl fb_uniform packed_rank seed 11]. It never grows with the model error |
| `fence_probes` | `locate_block::less` [index.hpp:135] | comparisons of the query key against a block's exact `last` key during exponential bracketing + binary correction; the *first* probe is always made (checking `less(p)`), so a perfect prediction costs 1 or 2 fence probes |
| `correction_distance` | `locate_block` after the search [index.hpp:151] | Σ over operations of |found block − predicted block| (`max_correction_distance` is the max) |
| `block_routes` | same place | number of `locate_block` calls (denominator for the mean correction) |
| `transform_calls` | `FlowTransform::transform` [transform.hpp:65] | number of flow evaluations (region feature and/or root feature) |
| `delta_probes` | `delta_lower` [index.hpp:115] | comparisons in the delta binary search |
| `key_at_calls`, `decoded_keys`, `codec_bytes_examined` | `key_at`/`get_bits`/`get_varint` [codec.hpp:134,137,142,149,154,158,52,66] | decode work in the block search |
| `blocks_decoded_for_scan` | `scan_into` [index.hpp:177] | blocks materialised by scans |

**Determinism.** All counters are functions of the built structure and the query trace only; they contain no timing. Because the build is deterministic (section 1.2) and the trace is seeded, the same command yields identical counters (the benchmark compares the throughput digest with the instrumented replay's digest and throws "instrumentation changes results" on a mismatch [benchmark.cpp:115]; the latency-replay comparison at benchmark.cpp:96 runs only when `--latency` is set, and every AIDB sweep passes `--latency 0` [results/aidb/sweep/environment.json config.common; every `command` in results.jsonl], so in these sweeps only the two digests are compared). The cost-story verification confirms that root variants differ from `packed_rank` *only* in `root_probes` and `transform_calls` (max abs diff per lookup on `fence_probes`, `coordinate_probes`, `key_at_calls`, `correction_distance` = ±0.000) [results/aidb/verification/coststory/coststory.txt]. What is NOT deterministic: `smoothing_ns`, `transform_ns`, `build_ns`, `throughput_ops_s` (wall-clock).

**Worked example (counters on a toy index).** `scratch/index_demo.cpp` builds 32 keys (0,10,…,230 then 240 + 1000·(i−24)) with `region_keys=16, block_keys=4, routing=Rank, policy=Raw`: 2 regions, 8 blocks. Measured:

```
find(0)    -> root_probes=2 coordinate_probes=3 fence_probes=1 correction=0 key_at=4
find(70)   -> root_probes=2 coordinate_probes=2 fence_probes=2 correction=0 key_at=3
find(230)  -> root_probes=1 coordinate_probes=2 fence_probes=2 correction=0 key_at=3
find(2240) -> root_probes=1 coordinate_probes=2 fence_probes=3 correction=1 key_at=3
find(7240) -> root_probes=1 coordinate_probes=2 fence_probes=2 correction=0 key_at=3
```
Reading: 2 regions → the fence binary search makes 1 or 2 comparisons; 4 blocks per region → 2-3 coordinate probes; key 2240 sits in the sparse tail of region 1 where the linear model misses by one block, so one extra `fence_probe` and `correction_distance = 1`. On the real 2M fb sample, `packed_rank` measures per lookup `root_probes 8.956` (≈ log₂ 489 = 8.93), `coordinate_probes 5.029`, `fence_probes 2.600`, `correction_distance 0.257`, `key_at_calls 8.016` [results/aidb_final/sweep/results.jsonl, fb_uniform, seed 11].

---

### 2. `LinearModel` (model.hpp)

```
struct LinearModel { Key origin=0; uint64 span=1; double slope=0, intercept=0; ... }
normalized(k) = (k − origin)/span      if k ≥ origin, else −(origin − k)/span      [model.hpp:15-18]
predict_x(x)  = fma(slope, x, intercept), 0 if non-finite                         [model.hpp:19-22]
```

`fit_xy(keys, x, targets)` [model.hpp:26-42] sets `origin = keys.front()`, `span = max(1, keys.back() − origin)` (integers subtracted *before* conversion to double, so precision near 2⁶⁴ is kept), then a single-pass centred (Welford) accumulation in `long double`:

```
for i: n=i+1; dx = xᵢ − mx; dy = yᵢ − my; mx += dx/n; my += dy/n; xx += dx·(xᵢ − mx); xy += dx·(yᵢ − my)
slope = xy/xx (0 if xx ≤ 0);  intercept = my − slope·mx
```
which is exactly closed-form OLS: w = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)², b = ȳ − w·x̄. `fit(keys, targets)` is the same with x = normalized(k) [model.hpp:43-60].

**Targets used.** In `rebuild`, `rank_model.fit_xy(keys, x, slots)` and `byte_model.fit_xy(keys, x, positions)` [index.hpp:312]: the rank model's target is the *slot rank* (= rank when no virtual points; rank + #virtual points before the key otherwise), the byte model's target is the byte coordinate of the key's first encoded bit inside the arena (`desc.offset + key_positions[j]`) [index.hpp:228]. Features x are normalized keys or flow outputs (section 3). The hardness tool uses plain rank targets (section 9).

**Worked example.** Keys/features {1,2,3,10,11,12}, ranks {0..5}: x̄ = 6.5, ȳ = 2.5, Σ(x−x̄)² = 125.5, Σ(x−x̄)(y−ȳ) = 44.5 → w = 0.354581673, b = 0.195219124; `predict_x(10) = 3.741` (true rank 3). With normalized features (origin 1, span 11, φ = {0, 0.0909, 0.1818, 0.8182, 0.9091, 1}) the fit is w = 3.900398, b = 0.549801: the same line in different units [scratch/index_demo.cpp, block A].

---

### 3. NFL-style transform (transform.hpp) and the stand-in trainer (tools/train_flow.py)

#### 3.1 What the paper does and what we re-implement

NFL [Wu et al., PVLDB 2022, §3.2] transforms the keys with a small numerical normalizing flow so that the transformed distribution is near-uniform, then builds AFLI on the transformed keys. The transformation pipeline is Algorithm 3.1: scaled min-max normalisation μ = min(X), σ = (max(X) − min(X))/θ, x_norm = (x − μ)/σ (line 2-4); an encoder that expands x into d features by repeatedly splitting integral and fractional parts (lines 5-17); the flow F on the d-dimensional vector (line 18); a decoder that sums the output vector back to one number (lines 19-22) [NFL §3.2.1, Alg. 3.1]. The deployed flows are small ("2H2L" = hidden 2, 2 layers, 8 parameters, 169.53 ns per key at batch size 1 and 7.29 ns at batch 2048 on their MKL implementation) [NFL Table 2]. That the network is a tanh network is NOT stated in the paper (the word does not occur in the PDF text); it comes from `transform.hpp`'s description of the official repository's C++ side ("evaluates that file as a small tanh network on Intel MKL") [transform.hpp:6-7; unverified against the paper: activation not stated there]. The paper's own training statement is split: §3.2.2 says only that the flow "takes time to train (about 38 seconds)" [NFL §3.2.2, p. 6]; the architecture is in the experimental setup: "we train a modified B-NAF in PyTorch. The B-NAF is set to two layers, two input dimensions, two hidden dimensions, and a normal distribution with the variance of 10¹⁶ as the latent distribution", trained on an RTX 3080 (10 GB) with "We only sample 10% bulk-loaded keys for three times to train the NF. We set the batch size to 256" [NFL §4.1, p. 9]. Our stand-in for comparison: 4,096 uniformly sampled keys of the 2M sample, 200 Adam steps, lr 0.05, 2.39 s on CPU [results/aidb/flows/fb_training.json; train_flow.py:119-127], and the scale factor θ (Alg. 3.1) is our `--shifts 64`, whereas the trainer's help text records the author default of 10⁶ for 200M keys [train_flow.py:119; unverified: author code not in the workspace]. The official repository (github.com/luffy06/NFL, GPL-3) exports a text weight file that its C++ side evaluates [transform.hpp:3-10]. We copy no code: `transform.hpp` re-implements only *inference* of that text format, and `train_flow.py` is a stand-in trainer that writes the same format [transform.hpp:1-19; train_flow.py:1-14]. Differences stated in the header: no MKL, no batching; the transformed value is used **only as the model feature** (records stay in key order, so fences/scans/exactness hold even for a non-monotone network, whereas NFL sorts by transformed key); keys are converted to double before the transform (precision above 2⁵³ is lost in the feature only) [transform.hpp:12-19].

#### 3.2 The weight file ("2D2H2L")

`FlowTransform::load` [transform.hpp:35-53] reads: `in_dim hidden layers` then `mean var`, then for each layer `rows cols` followed by rows×cols weights row-major; shapes are checked (`layer 0: in_dim×hidden`, last layer: `hidden×in_dim`), `in_dim ∈ {1,2}`, `layers ≤ 16`, `hidden ≤ 64`, `var > 0`. "2D2H2L" = input dimension 2, hidden 2, 2 layers, no bias. `save` writes the same layout with 17 significant digits [transform.hpp:54-61]. Whether the official author files carry exactly this header layout is stated in the header comment but not independently checked here [unverified: no author weight file in the workspace].

`results/aidb/flows/fb_2D2H2L.txt` (trained by `tools/train_flow.py` on the fb 2M uniform sample with `--sample 4096 --steps 200 --monotone`, seed 1000000007, 2.39 s, best NLL 4.182016 [results/aidb/flows/fb_training.json, fb_training.json.command.json]):

```
2	2	2                                    in_dim=2 hidden=2 layers=2
15162980.0	1207648189.703125            mean, var
2	2                                        W0 (2×2): row 0 = weights of feature x, row 1 = weights of the fractional feature
0.0058556031582764	0.0058592806847908
0.0000000000000000	0.0000000000000000        <- --monotone zeroes row 1
2	2                                        W1 (2×2)
1.2798795760698392	0.8812895293543010
1.4785700705830542	1.0331745748154413
```
`mean` = the smallest of the 4,096 *sampled* training keys (not the dataset minimum: the sample minimum is 97,995 [scratch/flow_demo.py]) because `train()` sets `mean = float(keys[0])` on the training subset; `var = max(1, (train_keys[-1] − train_keys[0])/shifts)` with `shifts = 64` [train_flow.py:89,119]. So x = (key − mean)/var spans ≈ [0, 64] on the training keys (Algorithm 3.1's θ is our `--shifts`).

#### 3.3 The feature and the network, exactly as coded

`transform(key)` [transform.hpp:64-80]:

```
x   = (key − mean) / var
a   = [x, x − ⌊x⌋]                          (in_dim = 2: the repository's "partition" encoder as described in transform.hpp:68)
for each layer l with matrix W_l (rows×cols):
    b_c = Σ_r a_r · W_l[r][c]
    a   = tanh(b)  for all but the last layer; a = b on the last layer
z   = Σ_c a_c                                ("sum" decoder); 0 if non-finite
```
**Encoder: paper versus code.** NFL Algorithm 3.1 with d = 2 does *not* produce [x, x − ⌊x⌋]. Its lines 6-8 set x_int = INT(x), x_float = x − INT(x) and add x_int to the vector; the loop of lines 9-14 runs d − 2 = 0 times; line 15 adds x_float [NFL Alg. 3.1, p. 6]. So the paper's 2-D vector is [INT(x_norm), x_norm − INT(x_norm)] (integral part, fractional part), whose *sum* equals x_norm but whose first coordinate is a 64-step staircase over the key range (x_norm ∈ [0, 64] with θ = 64). Our code follows transform.hpp:68's reading of the official repository's "partition" encoder, [x, x − floor(x)], which keeps the full value in the first slot [transform.hpp:68-69; unverified: the official encoder is not in the workspace, only the header author's description of it]. The consequence matters for `--monotone`: with W0 row 1 zeroed only the first feature carries information, so keeping x (rather than INT(x)) is what makes z continuous and strictly increasing in the key; with the paper's INT(x) in the first slot a monotone 2D2H2L flow would be a 64-step staircase and the tail conflict degree would be catastrophic. For 2D2H2L the code computes z(k) = Σ_c Σ_h tanh(u_h)·W1[h][c] = Σ_h v_h·tanh(u_h) with u_h = W0[0][h]·x + W0[1][h]·(x − ⌊x⌋) and v_h = Σ_c W1[h][c]. With `--monotone`, W0[1][·] = 0 so u_h = W0[0][h]·x and z is strictly increasing in the key. In the index, φ(k) = z(k) replaces the normalized key when the region uses the flow [index.hpp:106].

**Worked example (fb, by hand with Python; `scratch/flow_demo.py`).** v₀ = 1.27987958 + 0.88128953 = 2.16116911; v₁ = 1.47857007 + 1.03317457 = 2.51174465. Take the sample's median key k = keys[1,000,000] = 38,737,396,981:

```
x   = (38737396981 − 15162980) / 1207648189.703125 = 32.064167636867
frac= 0.064167636867 (unused: its weights are 0)
u₀  = 0.0058556031582764 · x = 0.187755041282 ;  tanh(u₀) = 0.185579469285
u₁  = 0.0058592806847908 · x = 0.187872958109 ;  tanh(u₁) = 0.185693322595
layer 1 pre-activations: [0.185579·1.279880 + 0.185693·1.478570, 0.185579·0.881290 + 0.185693·1.033175]
                       = [0.512079961573, 0.355402862763]   (no tanh on the last layer)
z   = 0.512079961573 + 0.355402862763 = 0.867482824336
```
Compare the raw feature φ_raw = (k − k_min)/(k_max − k_min) = 0.5010729. The sample's first key (97,995) maps to x = −0.01247 → z = −0.000341; the last key (77,308,811,965) to x = 64.0034 → z = 1.674188. Sampling every 1,000th key gives z ∈ [−0.000341, 1.673468], strictly increasing (`unordered_transformed_pairs = 0` in the training report). The pre-activations are at most 0.3750 (u₁ = 0.3750142 at the last key, u₀ = 0.3747788 [scratch/flow_demo.py; scratch/fix05_checks.py]), i.e. in the near-linear part of tanh (tanh(0.375) = 0.3584, slope 1 − tanh² = 0.872): this flow is a gentle concave bend, not a strong re-shaping, which is why the 2M-sample tail conflict degree is 8 before and 8 after the transform [results/aidb/flows/fb_training.json].

#### 3.4 `tail_conflict_degree` (transform.hpp:90-106) — NFL Definitions 3.1/3.2

Paper: the conflict degree of position j under model M is D_j = |{xᵢ : M(xᵢ) = j}| (Def. 3.1, eq. 5); over the m positions with D_j > 0, with tail percent γ = 0.99 and t = INT(m·γ), the tail conflict degree D^γ is the t-th largest conflict degree (Def. 3.2); The paper lists its two uses verbatim: "It could be useful in: 1) determine the execution of flow (see Section 3.2.2); 2) determine the capacity threshold of a node (see Section 3.3.1)" [NFL §3.1, Def. 3.2 → §3.2.2 and §3.3.1]. The paper example: 1000 positions, t = INT(1000 × 0.99) = 990, "the 990-th larger conflict degree", where "INT represents the flooring operation" [NFL Def. 3.2].

Our function on a sorted feature vector x (n ≥ 2; if x_n ≤ x₁ it returns n − 1):

```
1. OLS slope of rank on x (Welford); if slope ≤ 0, slope = n/(x_n − x₁)
2. intercept = −slope·x₁ + 0.5                      (first key maps near position 0, as NFL rescales)
3. max_size = max(1, min(⌊1.5·n⌋, max(1, ⌊slope·x_n + intercept⌋ + 1)))   ("1.5·n slots")
4. p_i = clamp(⌊slope·x_i + intercept⌋, 0, max_size − 1)                 (floor)
5. counts = run lengths of equal consecutive p_i  (these are exactly the m positions with D_j > 0)
6. sort counts ascending; idx = max(0, ⌈0.99·m⌉ − 1); return counts[min(idx, m−1)] − 1
```
Two conventions to keep in mind: (i) sorting ascending and taking index ⌈0.99 m⌉−1 selects the 99th-percentile *smallest*, which is the same element as the ⌊0.01 m⌋+1-th largest — the paper's wording "t-th larger" with t = INT(0.99 m) is ambiguous between the two readings, ours takes the upper tail (the one that "indicates the upper bound of the conflicts for most positions") [NFL Def. 3.2]; (ii) the final "− 1" reports *extra* keys per position, so D₉₉ = 0 means no conflict (a uniform sequence gives 0: test `learnability` [test_main.cpp:100]).

**Worked example** (`scratch/tcd_demo.py`, same arithmetic as the C++): x = {0,1,…,9, 9.1, 9.2, 9.3, 9.4, 9.5, 20, 30, 40, 50, 60}, n = 20. OLS slope = 0.287944413, intercept = 0.5, ⌊slope·60 + 0.5⌋ + 1 = 18 < ⌊1.5·20⌋ = 30 → max_size = 18. Positions: [0,0,1,1,1,1,2,2,2,3,3,3,3,3,3,6,9,12,14,17]. Run lengths: {2,4,3,6,1,1,1,1,1} → sorted {1,1,1,1,1,2,3,4,6}, m = 9, idx = ⌈8.91⌉ − 1 = 8 → counts[8] − 1 = 6 − 1 = **5**. The C++ returns 5 as well [scratch/index_demo.cpp, block E]. **Which percentile is meant.** The paper takes t = INT(m·γ) (floor) and the "t-th larger" element; the code takes index ⌈m·γ⌉ − 1 in ascending order [transform.hpp:104-105], i.e. the ⌈m·γ⌉-th smallest. For m = 1000 both give the 990th smallest (t = 990 1-based ⇔ 0-based 989) and coincide; for m·γ not an integer they differ by one element. On this m = 9 example the readings diverge: code ⌈8.91⌉ = 9th smallest = 6 → **5**; paper's t = INT(8.91) = 8 read as 8th smallest = counts[7] = 4 → **3**; read as 8th *largest* (descending {6,4,3,2,1,1,1,1,1}) = 1 → **0** [scratch/fix05_checks.py]. The code deliberately takes the upper-tail (ascending, ceil) reading because the definition says the quantity "indicates the upper bound of the conflicts for most positions" [NFL §3.1]; with m in the hundreds per region the one-element difference is immaterial, but say so if asked.

Real numbers (per-region means over 489 regions, `tail_conflicts_raw_mean` / `tail_conflicts_flow_mean`, variant `packed_rank_flow`, seed 11) for all ten 2M uniform samples: wise 3.03 / 3.03, fb 8.448 / 8.464, genome 4.23 / 4.23, books 3.28 / 3.28, planet 7.66 / 7.65, libio 3.37 / 3.36, stack 3.01 / 3.01, covid 3.16 / 3.14, history 3.10 / 3.10, osm 36.16 / 36.98 [results/aidb/sweep/results.jsonl, learnability; scratch/fix05_checks.py]. The transform changes the mean tail conflict degree by less than 1 % everywhere except osm (+2.3 %, worse). The Python trainer's whole-sample D₉₉ for fb is 8 → 8 [results/aidb/flows/fb_training.json].

#### 3.5 The stand-in trainer (tools/train_flow.py)

Not a BNAF. It maximises the likelihood of the keys under z ~ N(0,1) through the 1-D change-of-variables formula p_X(x) = p_Z(z(x))·|dz/dx|, i.e. it minimises

  NLL = (1/n) Σ_keys [ ½·z² − log(dz/dx) + barrier/(dz/dx) ]   (constant ½log 2π and log var dropped) [train_flow.py:48-55]

with, for the 2D2H2L shape, z = Σ_h v_h·h_h, h_h = tanh(u_h), s_h = 1 − h_h², and dz/dx = Σ_h s_h·a_h·v_h where a_h = W0[0][h] + W0[1][h] (since d(x − ⌊x⌋)/dx = 1 almost everywhere) [train_flow.py:31-40]. If dz/dx ≤ 0 the loss switches to ½z² + 50 − 10·(dz/dx) (a linear push-back) [train_flow.py:55]. The `barrier/(dz/dx)` term (default 1e-3) keeps the derivative positive, i.e. keeps the transform monotone. Gradients are analytic (∂z/∂W0[i][h] = v_h s_h f_i, ∂(dz/dx)/∂W0[i][h] = v_h((−2h_h s_h f_i)a_h + s_h), ∂z/∂W1[h][c] = h_h, ∂(dz/dx)/∂W1[h][c] = s_h a_h) [train_flow.py:57-66]. **Scaled parametrisation:** the two x-feature weights are learned as θ = w·shifts (so `scale0 = [1/shifts, 1/shifts, 1, 1]`) to prevent a single Adam step from saturating the tanh, "the official code relies on BNAF weight normalisation for this" [train_flow.py:90-93,105-109]. **Adam** with β₁ = 0.9, β₂ = 0.999, ε = 1e-8, lr 0.05, 200 steps, bias-corrected; the weights with the best loss seen are kept [train_flow.py:98-111]. **`--monotone`** zeroes W0 row 1 and its gradient, "deviation from NFL, which keeps them" [train_flow.py:67,96,124]. Initialisation is seeded (`abs(gauss(1.5, 0.5))/shifts` for row 0, `abs(gauss(0.1, 0.05))` for row 1, `abs(gauss(1, 0.2))` for W1) [train_flow.py:93-95]. Training keys: `--sample 4096` uniformly sampled from the de-duplicated key set with the same seed [train_flow.py:127-128]. The report JSON records `best_nll`, `unordered_transformed_pairs` on *all* keys, and D₉₉ raw vs transformed [train_flow.py:132-139]. The pipeline trains one flow per dataset on the 2M sample (`step_flows`) [aidb_pipeline.py:435-447].

**Worked example (the loss on one key; same fb flow and median key as 3.3; `scratch/fix05_checks.py`).** v = [2.16116911, 2.51174465] (row sums of W1), a_h = W0[0][h] + W0[1][h] = [0.0058556032, 0.0058592807] (row 1 is zero), h = tanh(u) = [0.185579469, 0.185693323], s_h = 1 − h² = [0.96556026, 0.96551799]:

```
dz/dx = Σ_h s_h·a_h·v_h = 0.96556026·0.0058556032·2.16116911 + 0.96551799·0.0058592807·2.51174465 = 0.026428660
per-key NLL term = ½z² − log(dz/dx) + 10⁻³/(dz/dx)
                 = ½·0.867483² + (−log 0.026428660) + 0.001/0.026428660
                 = 0.376263 + 3.633306 + 0.037838 = 4.047407
```
for comparison `best_nll` = 4.182016 is the mean of this term over the 4,096 training keys [results/aidb/flows/fb_training.json]; at the last key (77,308,811,965: z = 1.674188, dz/dx = 0.023858) the term is 1.401452 + 3.735616 + 0.041914 = 5.178982. Why NLL ≈ 4 rather than ≈ 1.4 (= ½ + ½log 2π, the entropy of a standard normal): the −log(dz/dx) term dominates because z spans only ≈ 1.67 while x spans 64, so dz/dx ≈ 1/38 and −log(1/38) ≈ 3.6. The likelihood is measured in normalized-x units (the constant log var of the key→x scaling is dropped, train_flow.py:50), so the absolute value of the NLL is not comparable to NFL's; only differences between candidate weights matter to the optimiser.

---

### 4. The bypass rule ("manual fusion")

NFL: "the NFL first tries to transform the input keys, and computes the tail conflict degree based on the input keys and the transformed keys, respectively. If the latter tail conflict is larger, the NFL determines not to use the Numerical NF" [NFL §3.2.2]. Our per-region rule, in `Region::rebuild`, `Fusion::Manual` branch [index.hpp:273-285]:

```
z = flow(keys)                                      (transform_ns measured)
D99_flow = tail_conflict_degree(sorted z)           (sorted because the flow need not be monotone)
D99_raw  = tail_conflict_degree(normalized keys)    [index.hpp:237]
gain = D99_flow < D99_raw  and  (D99_raw − D99_flow) ≥ flow_min_gain · D99_raw     [index.hpp:281]
flow_used = !flow_bypass or gain                                                     [index.hpp:282]
if flow_used: x = z
```
With the defaults `flow_bypass = true, flow_min_gain = 0.10` a region keeps the flow only if its tail conflict degree falls by at least 10 % (the paper's rule is "not larger"; the 10 % margin is ours). `--flow-bypass 0` (variant `packed_rank_flow_forced`) forces the flow everywhere. **Reuse on compaction:** if `previous` is given and `relearn_on_compaction` is false, the previous region's `flow_used` and `tail_conflicts_*` are copied and the transform is evaluated only if the flow is in use [index.hpp:274-276]; `tail_conflicts_raw` is likewise copied [index.hpp:238].

Measured outcome on the 2M uniform samples (variant `packed_rank_flow`, 489 regions): the flow is kept (`flow_regions`, seed 11) in wise 4, fb 41, genome 8, books 26, planet 43, libio 21, stack 1, covid 17, history 13, osm 7 of the 489 regions, i.e. from 0.2 % (stack) to 8.8 % (planet) [results/aidb/sweep/results.jsonl, variant packed_rank_flow; scratch/fix05_checks.py]. Consistently, `transform_calls/op` = 0.084 on fb (= 41/489 = 0.0838, since a lookup pays one transform iff its region uses the flow) and 1.000 for the forced variant [same file; verified in coststory: max |T − (root_flow + flow_regions/regions)| = 0.0015].

---

### 5. CSV-style smoothing (smoothing.hpp), line by line

#### 5.1 The paper's problem

CSV [Amarasinghe et al., EDBT 2025] keeps the index structure and instead changes the *key set*: it inserts virtual points into the sorted keys so that the CDF becomes easier to fit by a linear model. The loss is the SSE of the indexing function(s), L_F(K) = Σᵢ Σ_{k∈Kᵢ} (fᵢ(k) − rank(k))² (eq. 2); with a budget λ = α·n, α ∈ (0,1), the single-segment problem is argmin_{Vᵢ,w,b} L_{f_{w,b}}(Kᵢ ∪ Vᵢ) s.t. |Vᵢ| ≤ λ (eq. 4) — note that the line is *refitted* on real and virtual points and the virtual points themselves count in the loss (eq. 5: Σᵢ(w kᵢ + b − yᵢ)² + (w k_v + b − y_v)²) [CSV §3-4]. The general problem is NP-hard (reduction from knapsack) [CSV §3]. Algorithm 1 `CDF_smoothing` is a greedy: while |V| < λ, for each sub-sequence of candidate positions take the endpoints (or, when the first-order derivative of the loss changes sign between the endpoints, the interior minimum), evaluate the loss of every candidate, pick the best, stop if it does not lower the previous loss (line 27-28), insert it and repeat [CSV Alg. 1, lines 5-31]. Candidates are restricted to (min K, max K) and never coincide with an existing key [CSV §4.2]. There is no public implementation ("checked September 2026") [smoothing.hpp:7-8].

#### 5.2 `Sums` and the closed-form SSE

```
struct Sums { long double n, sx, sxx, sy, syy, sxy;   add(x,y): ++n; sx+=x; sxx+=x²; sy+=y; syy+=y²; sxy+=xy;
  sse(): if n<2 → 0; cxx = sxx − sx²/n; cxy = sxy − sx·sy/n; cyy = syy − sy²/n;
         s = cxx>0 ? cyy − cxy²/cxx : cyy; return max(0, s) }                         [smoothing.hpp:35-44]
```
Derivation: for OLS w = S_xy/S_xx, b = ȳ − w x̄ (centred sums S_xx = Σ(x−x̄)², S_xy = Σ(x−x̄)(y−ȳ), S_yy = Σ(y−ȳ)²), the residual sum of squares is

  SSE = Σ(yᵢ − w xᵢ − b)² = Σ((yᵢ−ȳ) − w(xᵢ−x̄))² = S_yy − 2w S_xy + w² S_xx = S_yy − S_xy²/S_xx,

and the centred sums come from the raw sums by S_xx = Σx² − (Σx)²/n etc. This is CSV's eq. 10-16 in a compact form: the loss for a candidate can be evaluated from six running sums without refitting w, b explicitly [CSV §4.1].

#### 5.3 `loss_at(i, x_v)`: why inserting one point is O(1)

Working sequence `seq` = (feature, is_virtual) in feature order; the target of position j is simply j (so slot ranks are sequence positions) [smoothing.hpp:51-55]. `base` holds the sums of the current sequence. Inserting a virtual point after position i (target i+1) shifts the target of every later element j ≥ i+1 by +1. The updated sums are, with cnt = |seq| − (i+1) elements shifted,

```
S_y'  = S_y + cnt                                        (each shifted y grows by 1)
S_yy' = S_yy + Σ_{j≥i+1} ((j+1)² − j²) = S_yy + 2·Σ_{j≥i+1} j + cnt = S_yy + 2·suf_y[i+1] + cnt
S_xy' = S_xy + Σ_{j≥i+1} x_j·1 = S_xy + suf_x[i+1]
then add the new point (x_v, i+1): n'=n+1, S_x += x_v, S_xx += x_v², S_y += i+1, S_yy += (i+1)², S_xy += x_v(i+1)
loss = sse(S')
```
[smoothing.hpp:64-70]. `suf_x[j] = Σ_{t≥j} x_t` and `suf_y[j] = Σ_{t≥j} t` are suffix sums rebuilt once per round (O(|seq|)) [smoothing.hpp:59-63], so each candidate evaluation is O(1).

**Numeric instance of the O(1) update** (the 6-key example of 5.5, base sums n = 6, S_x = 39, S_xx = 379, S_y = 15, S_yy = 55, S_xy = 142; insert x_v = 6.5 after position i = 2, i.e. in the gap (3, 10)) [scratch/fix05_checks.py]:

```
cnt      = 6 − (2+1) = 3 shifted elements (positions 3, 4, 5)
suf_y[3] = 3 + 4 + 5 = 12          suf_x[3] = 10 + 11 + 12 = 33
shift:   S_y' = 15 + 3 = 18        S_yy' = 55 + 2·12 + 3 = 82       S_xy' = 142 + 33 = 175
add (6.5, 3): n' = 7, S_x' = 45.5, S_xx' = 421.25, S_y' = 21, S_yy' = 91, S_xy' = 194.5
centred: cxx = 421.25 − 45.5²/7 = 125.5   cxy = 194.5 − 45.5·21/7 = 58   cyy = 91 − 21²/7 = 28
SSE      = 28 − 58²/125.5 = 1.1952191235
```
identical to a brute-force OLS refit of the seven points (1,0),(2,1),(3,2),(6.5,3),(10,4),(11,5),(12,6): 1.1952191235. A round examines every gap: 4 loss evaluations for the endpoint test, and for a gap that enters the ternary search ≤ 2·40 more inside the loop plus 1 final evaluation of the midpoint candidate (`cl = loss_at(i, cx)`, smoothing.hpp:86), i.e. ≤ 85 evaluations per gap [smoothing.hpp:78,82-86], so one round is O(|seq|) and the whole run with budget λ is **O(λ·n)** loss evaluations; "the paper's O(n + λ) claim is not reproduced" [smoothing.hpp:11-13]. (CSV's complexity paragraph says the loss over K is computed once in O(n) and reused; the per-round cost over the candidate set is what dominates here.)

#### 5.4 The per-gap search (four-sample endpoint test, ternary search)

For each gap (lo = seq[i].x, hi = seq[i+1].x) with hi > lo (gaps of zero width are skipped, so a virtual point never equals a key), w = hi − lo, e = 10⁻³·w:

```
l0 = loss(lo+e), l1 = loss(lo+2e), r1 = loss(hi−2e), r0 = loss(hi−e)
if l1 < l0 and r1 < r0:   # loss decreasing at the left end and increasing at the right end: interior minimum
    ternary search on [lo+e, hi−e] for ≤ max_ternary_steps=40 iterations or until b−a ≤ e; candidate = midpoint
else: candidate = the better of lo+e and hi−e
```
[smoothing.hpp:75-87]. The paper uses the *sign of the derivative* at the two endpoints for the same decision (Alg. 1 lines 13-22, "if the sign … are opposite, there is a local minimum"; convexity within a sub-sequence is cited from [11]) [CSV §4.2]; we sample the loss just inside the ends instead of computing L'. Three further deviations from CSV Algorithm 1, stated explicitly: (a) **candidate domain.** CSV works on integer keys ("To simplify the discussion, we use integer index keys, while our techniques also apply to real number index keys when they can be scaled up to become integers" [CSV §3]) and its candidate virtual points are the integer values strictly between consecutive keys (Fig. 3: "the segment formed by 21 to 25 is between index keys 20 and 26 (the index keys themselves are not considered to be candidate virtual points)" [CSV §4.2, Fig. 3]); Algorithm 1 line 7-8 append both endpoints directly when `E[i].second − E[i].first ≤ 1`, so a sub-sequence of ≤ 2 candidates skips the derivative test entirely, and between two *adjacent* integer keys there is no candidate at all [CSV Alg. 1 lines 7-8]. `smooth_cdf` instead works on real-valued features and always tries lo + 10⁻³·w and hi − 10⁻³·w in every gap of positive width [smoothing.hpp:77-78], which is why it produces non-integer virtual features such as 6.4994 on integer keys (5.5). (b) **interior minimum.** CSV's `minimum_point(M[i].first, M[i].second)` is "calculated by using the two partial derivative values" [CSV §4.2; Alg. 1 lines 20-21]; ours is a ≤ 40-step ternary search on the sampled loss [smoothing.hpp:82-86]. (c) **acceptance margin.** `best` starts at `current` and every candidate must satisfy `cl < best − 10⁻¹²·max(1, best)`, with `best` updated as the round progresses [smoothing.hpp:74,88]: the relative margin is applied against the best candidate seen so far in the round, not only against the current loss, and a round that finds no such candidate breaks the loop (`found == false`, Algorithm 1 line 27) [smoothing.hpp:90]; otherwise the point is inserted, `base` is recomputed from scratch (O(|seq|)) and `rounds` increments [smoothing.hpp:91-92]. At the end, `out.slot[k]` = position of the k-th real key in the final sequence (rank + virtual points before it) and `out.virtual_features` = the inserted values in insertion order; `sse_before` is the OLS SSE of the ORIGINAL sequence (computed at smoothing.hpp:56 before any insertion, no virtual points), while `sse_after` is the SSE of the augmented sequence after the last accepted insertion (`current`, smoothing.hpp:71,92,95). So a ratio `sse_after/sse_before` compares an n-point fit with an (n + v)-point fit whose targets are slots, not ranks; the header comment's "augmented-set OLS SSE" applies to `sse_after` only [smoothing.hpp:30].

**Bounds.** `alpha ∈ [0, 64)` is enforced here (the root uses `root_alpha` up to 64 fences per region fence), while `Config::validate` caps `virtual_alpha < 1` for regions [smoothing.hpp:48; index.hpp:67,69]: CSV requires α ∈ (0,1) "to retain a linear space overhead" [CSV §3]. `budget = ⌊α·n⌋`; nothing is done if n < 3 or budget = 0 [smoothing.hpp:57-58].

#### 5.5 WORKED EXAMPLE on 6 keys, features [1, 2, 3, 10, 11, 12]

Base sums: n = 6, S_x = 39, S_xx = 379, S_y = 15, S_yy = 55, S_xy = 142. Centred: S_xx^c = 379 − 39²/6 = 125.5, S_xy^c = 142 − 39·15/6 = 44.5, S_yy^c = 55 − 15²/6 = 17.5. **SSE_before = 17.5 − 44.5²/125.5 = 1.7211155378** (direct check with w = 0.35458, b = 0.19522: 1.7211155378 ✓) [scratch/smooth_py.py; C++ `Sums::sse` gives 1.7211155378 too, scratch/index_demo.cpp block A].

Per gap (e = 10⁻³·width; losses via `loss_at`, cross-checked against a brute-force refit on the augmented sequence — identical to 6 decimals):

| gap i | (lo, hi) | e | loss(lo+e) | loss(lo+2e) | loss(hi−2e) | loss(hi−e) | interior? |
|---|---|---|---|---|---|---|---|
| 0 | (1, 2) | 0.001 | 3.427440 | 3.427522 | 3.632328 | 3.632664 | no (rising) |
| 1 | (2, 3) | 0.001 | 3.632510 | 3.632021 | 3.265183 | 3.264944 | no (falling) |
| 2 | (3, 10) | 0.007 | 3.257072 | 3.249449 | 3.249449 | 3.257072 | **yes** |
| 3 | (10, 11) | 0.001 | 3.264944 | 3.265183 | 3.632021 | 3.632510 | no |
| 4 | (11, 12) | 0.001 | 3.632664 | 3.632328 | 3.427522 | 3.427440 | no |

Only the big gap qualifies for a ternary search; it converges in 18 iterations to x_v = 6.499363 (loss 1.1952191978; the exact minimiser is the midpoint 6.5 with loss 1.1952191235 — the search stops when the bracket is narrower than e = 0.007). Every other gap's best endpoint (≥ 3.26) is *worse than the current SSE 1.72*: inserting a point in a small gap only shifts the later targets up and hurts. Best = gap 2, so with α = 0.2 (budget ⌊1.2⌋ = 1) the C++ returns

```
alpha=0.20 budget=1 rounds=1 sse_before=1.7211155378 sse_after=1.1952191978
slots = 0 1 2 4 5 6      virtual = 6.4993630796
```
[scratch/smooth_demo.cpp]. Slots: keys 10, 11, 12 moved from ranks 3,4,5 to slots 4,5,6 — the augmented sequence is (1,0),(2,1),(3,2),(6.499,3),(10,4),(11,5),(12,6) and its OLS SSE is 1.19522 (Python recomputation ✓). Larger budgets keep filling the same gap: α = 0.4 → virtual {6.4994, 7.5031}, SSE 0.97892; α = 0.5 → adds 4.9698, SSE 0.51954; α = 0.9 (budget 5) → adds 8.5625 and 4.1837, SSE 0.17773 and slots 0 1 2 8 9 10 [same program]. The unit test's vector {1,2,3,4,5,10,20,26,27,30} at α = 0.3 gives 3 rounds, SSE 7.8405 → 4.5861, virtual {16.599, 14.781, 22.110} [scratch/smooth_demo.cpp; test_main.cpp:92-96 checks the invariants].

Real data: on the fb 2M sample at α = 0.1 the budget is 488·⌊409.6⌋ + ⌊115.2⌋ = 199,707 and exactly 199,707 points were inserted (`rounds = 199707`: early stop never fired), per-region summed SSE 3.798×10⁹ → 5.873×10⁸ (ratio 0.1546) in the hardness tool [results/aidb/hardness/fb_sample_csv.json smoothed], and 3.798×10⁹ → 5.866×10⁸ (0.1544) inside the index [results/aidb/sweep/results.jsonl fb packed_rank_vp10 learnability] — the two differ by 0.1 %. This is floating point, not an algorithmic difference: the OLS SSE and the greedy's relative grid (e = 10⁻³·gap) are invariant under an affine change of the feature, so the tool (features `double(key − global_min)`, up to 7.7×10¹⁰ [hardness.cpp:170]) and the index (per-region features `(k − origin)/span` in [0, 1] [index.hpp:236]) run the same algorithm and should agree exactly. The proof is that `sse_before`, which involves no greedy choice, already differs: 3,798,262,711.26 (tool) vs 3,798,262,931.22 (index). With raw-scale features S_xx ≈ 10²⁵ and the centred sums cxx = S_xx − S_x²/n [smoothing.hpp:40] cancel catastrophically in `long double`, which on Apple arm64 is only 64-bit ("Apple arm64 has long double == double" [hardness.hpp:14]); the index's [0, 1] features do not suffer from this. The 0.1 % is therefore a precision artefact of the tool's feature scale, and the index's number is the more accurate one [results/aidb/hardness/fb_sample_csv.json smoothed.rank_sse_before; results/aidb/sweep/results.jsonl fb packed_rank_vp10 learnability.rank_sse_before]. Smoothing cost 11.7 s (tool, 16 threads) / 36.9 s summed over the build threads (index) for 2M keys; at 200M keys it is 2,407 s of summed thread time (fb `packed_rank_vp10_root_vf004`, `smoothing_ns`) [results/aidb_fullscale/sweep/results.jsonl].

---

### 6. How the index uses virtual points

1. **Targets, not records.** `smooth_cdf(x, virtual_alpha)` runs on the region's features [index.hpp:305]; `slots[i] = sm.slot[i]` become the rank model's targets [index.hpp:307,312]; each block stores `slot_begin = sm.slot[b.rank_begin]` (the slot of its first key) [index.hpp:308,83]. Nothing is written to the arena or to `values`; `virtual_points` is only counted and the feature values are kept in `virtual_features` [index.hpp:306].
2. **Lookup.** `predicted_block` binary-searches the prediction y against `blocks[m].slot_begin` [index.hpp:125-126]: the model now predicts in slot space and the block boundaries are expressed in slot space, so the extra "room" created by the virtual points is consistent on both sides. `locate_block`'s fence correction is unchanged (exact `last` keys) [index.hpp:135].
3. **Memory.** `Region::memory()` counts `virtual_features.size()·sizeof(double)` in `metadata_bytes` [index.hpp:320]. At α = 0.1 that is ≤ 0.1 doubles per key = 0.8 bytes/key. Measured: fb vp10 adds 199,707·8 = 1,597,656 bytes of metadata for 2M keys = 0.7988 B/key. To place that, the full fb `packed_rank` memory at bulk load (seed 11, `memory_before`) is: `key_bytes` 5,636,561 (2.82 B/key of packed arena, against 8 B/key raw), `value_bytes` 16,000,000 (8 B/key), `metadata_bytes` 1,141,040 (0.57 B/key: Σ over the 489 regions of sizeof(Region) + blocks·sizeof(BlockDescriptor) with 15,625 descriptors in total, plus the Index-level bookkeeping [index.hpp:320,495]), `accounted_bytes` 22,777,601 = 11.39 B/key; `packed_rank_vp10` has `metadata_bytes` 2,738,696 = 1,141,040 + 1,597,656 and `accounted_bytes` 24,375,257 = 12.19 B/key, i.e. +7.0 % [results/aidb/sweep/results.jsonl fb seed 11; scratch/fix05_checks.py]. Beware that the cost-story metadata table, built from the `aidb_final` sweep, prints `packed meta = 1141168`, 128 B more than the uniform sweep's 1,141,040: `Index::memory()` adds `sizeof(Index)` and the root's slot table [index.hpp:495], and the root fields were added to `Index` between the two builds [unverified: the two binaries were not diffed]. Cite one sweep consistently; the vp10 delta of +1,597,656 B is identical in both.
4. **Compaction reuse** (`relearn_on_compaction = false`, `previous` given, features sorted) [index.hpp:289-303]: keep the previous `virtual_features` that fall strictly inside (x.front(), x.back()), sort them, and merge them with the new keys in one pass to assign slots — O(n + v), no search; compute SSE before/after with `Sums`; **drop them all if the SSE would not improve** (`after.sse() <= before.sse()` required, else `slots = ranks`) [index.hpp:300-301]. With `relearn` the full `smooth_cdf` reruns. The test `learnability` covers both paths and asserts `rank_sse_after ≤ rank_sse_before` after 2,000 mixed operations [test_main.cpp:109-122].
5. **Why reuse:** in the synthetic learnability sweep (read_heavy profile: 80 % reads, 10 % inserts, 5 % updates, 2 % erases, 3 % scans [benchmark.cpp:196]), relearning on every compaction (`packed_rank_vp10_relearn`) dropped throughput from 1,063,104 to 9,145 ops/s on clustered (116.3×), 921,942 → 3,981 on dense_sparse (231.6×), 1,047,678 → 6,510 on locally_hard (160.9×), 945,537 → 7,323 on lognormal (129.1×); insert p99 went from ~0.2 ms to 14-56 ms. Read-only throughput is unaffected within noise, as expected, since no compaction ever runs: the file's `paired_throughput_speedup` versus `packed_rank` on the read_only profile is 1.0004 / 1.0059 (clustered), 1.0279 / 1.0320 (dense_sparse), 1.0588 / 1.0145 (locally_hard), 0.9897 / 0.9842 (lognormal) for `packed_rank_vp10` / `packed_rank_vp10_relearn`, i.e. 0.98-1.06; the direct relearn/vp10 throughput ratios (medians 4,682,376/4,653,687, 3,005,787/2,935,709, 3,831,908/3,996,803, 4,237,439/4,397,696 ops/s) are 1.006, 1.024, 0.959, 0.964, i.e. 0.96-1.02 [results/learnability/summary.csv, read_only rows; scratch/fix05_checks.py].

---

### 7. The per-region selector (`Fusion::Auto`) [index.hpp:239-272]

The condition is `c.fusion == Fusion::Auto && !reuse && keys.size() > 2` with `reuse = previous && !c.relearn_on_compaction` [index.hpp:201,239]. So the selector runs at bulk load (no `previous`), and it also re-runs on every compaction rebuild (which does pass `previous`) when `relearn_on_compaction` is true; it is skipped only when a previous region is given AND relearn is off, in which case the previous decision and virtual points are reused (section 6.4). Two facts about Auto mode that are easy to get wrong: (1) the NFL bypass rule of section 4 is NOT applied; `tail_conflicts_flow` is still computed on the sorted z, but only for reporting [index.hpp:248], and the flow is chosen purely by the probe cost below; (2) `smooth_cdf` runs once per *base* candidate, i.e. once on the raw features and once on the flow features [index.hpp:252-257], so the build pays the O(λ·n) search twice: on fb, `packed_rank_fusion_auto` reports `smoothing_ns` = 124.15 s of summed thread time against 36.94 s for `packed_rank_vp10` (osm: 104.83 s vs 27.75 s) [results/aidb/sweep/results.jsonl seed 11]. That doubling (and a bit more, since the flow candidate's smoothing is on the same n) is a cost of the selector design, not of the structure it chooses. Candidates, in this order:

```
cands[0] = {flow=false, vp=false, feat=normalized keys, slots=ranks}
cands[1] = {flow=true,  vp=false, feat=z(keys),         slots=ranks}          (if c.flow; transform_ns measured; D99_flow computed on sorted z)
cands[2] = {flow=false, vp=true,  feat=normalized keys, slots=smooth_cdf(...).slot}   (if virtual_alpha > 0)
cands[3] = {flow=true,  vp=true,  feat=z(keys),         slots=smooth_cdf(z).slot}     (if both)
```
[index.hpp:243-257]. For each candidate the *real* structure is fitted (`rank_model.fit_xy(keys, feat, slots)`, `slot_begin` per block) and the real `locate_block` is run on every key of the region under `Routing::Rank` with a `QueryStats` collector; the cost is

  cost = (fence_probes + coordinate_probes)/n + (flow ? flow_cost : 0)

[index.hpp:258-265]. Root probes and decode work are excluded (identical across candidates), and the routing is forced to Rank even if the run uses byte routing. The minimum is taken with a strict `<` scan from index 0, so ties go to the earlier candidate: raw ranks beat everything at equal cost, flow beats virtual points at equal cost [index.hpp:266]. `cost_none = cands[0].cost`, `cost_selected = best.cost`; the chosen candidate's features, slots, virtual features and SSEs are installed and `targets_done` skips the manual branch [index.hpp:267-272]. `choice = (flow_used ? 1 : 0) | (virtual_points ? 2 : 0)` [index.hpp:313] and `learnability()` counts regions per choice as `choices: {none, flow, vp, both}` [index.hpp:488; benchmark.cpp:74]. `flow_cost` is a probe-equivalent per lookup charged to the transform, default 4, "measure it; default is a placeholder" [index.hpp:64-65]; the measured transform costs 66 ns [56, 77] per evaluation [results/aidb/MEETING_NOTES.md:114-115].

**Worked example** (`scratch/index_demo.cpp`, block C): 128 keys (96 quadratic then 32 sparse), `region_keys = 64, block_keys = 8, virtual_alpha = 0.1`, a toy flow. Both regions choose `vp`: `cost_none_mean = 6.0391` probes per lookup, `cost_selected_mean = 5.9297`, 12 virtual points, D₉₉ raw 16.5 vs flow 11.0 — yet the flow is not chosen even with `flow_cost = 0`, because the tail conflict degree is not what is being minimised; the measured probe count is. **Real data:** on the fb 2M sample (`packed_rank_fusion_auto`, α = 0.1, flow_cost 4) all 489 regions choose `vp`, `cost_none_mean = 7.627`, `cost_selected_mean = 7.208` probes (fence + coordinate) per lookup; the measured counters agree: fence 2.602 → 2.183 with coordinate 5.029-5.030 (7.63 → 7.21). On osm 470 regions choose vp, 19 choose none (10.097 → 9.657). With the flow alone as an option (`packed_rank_flow_costsel`), zero regions take it on all five datasets shown [results/aidb/sweep/results.jsonl]. Test `fusion_auto` asserts Σchoices = regions, `cost_selected ≤ cost_none`, and that `flow_cost = 100` is never chosen [test_main.cpp:124-141].

---

### 8. The learned root (`Index::fit_root`) [index.hpp:358-401]

**Which keys.** For n ≥ 2 regions, fences[j] = the *first real key* of region 0 for j = 0 and `low_fence` (= first key at bulk load) for j ≥ 1; ranks = j [index.hpp:369-370]. **The sentinel-fence bug:** an earlier version used `low_fence` for all regions, i.e. the sentinel 0 for region 0. For a contiguous window whose keys start far from 0 (e.g. 9·10¹²), the point (0, 0) is a huge leverage outlier: the raw least-squares line through it is useless, so the raw candidate's estimated cost exceeded binary search and was rejected, while the flow's tanh compression of the outlier let the flow-based candidates look *relatively* useful. The complete pre-fix candidate table (estimated root probes per lookup, seed 11, binary = 8.956; the same estimates appear in every root variant of that sweep) shows that the plain flow candidate did NOT win either: only the vp_flow combination beat binary search [results/aidb_final/sweep/results.stale-root-before-fix.jsonl, learnability.root_probes_*, root_virtual; work_counters.root_probes/operations]:

| sample (pre-fix) | raw | flow | vp_raw | vp_flow | chosen | measured `root_probes/op` |
|---|---|---|---|---|---|---|
| fb_window | 14.339 | 14.066 | 14.230 | **7.000** | flow + 1,386 virtual fences (`root_flow = true`, `root_virtual = 1386`) | 2.882 |
| books_window | 14.241 | 13.955 | 13.902 | **6.990** | flow + 1,374 fences | 2.930 |
| osm_window | 14.378 | 15.632 | 14.358 | 10.510 | none (binary) | 8.956 |

Read it this way: the sentinel point (0, 0) ruined raw AND vp_raw alike (both ≈ 14.2-14.4: no number of virtual fences can repair a line pinned to an outlier 9·10¹² away), whereas the tanh squashed the outlier's feature so that fences could still straighten the flow's CDF, hence vp_flow ≈ 7. After the fix (fences[0] = first real key) the same seed gives [results/aidb_final/sweep/results.jsonl]:

| sample (post-fix) | raw | flow | vp_raw | vp_flow | chosen | measured `root_probes/op` |
|---|---|---|---|---|---|---|
| fb_window | 3.584 | 7.860 | **2.409** (34 fences) | 6.490 | raw + fences | 2.273 |
| books_window | 2.320 | 8.115 | **2.301** (1 fence) | 6.331 | raw + fences | 2.102 |
| osm_window | 10.555 | 14.382 | **3.559** (984 fences) | 7.648 | raw + fences | 3.435 |

fb_uniform is unchanged (raw 2.266, vp_raw 2.224) because its keys start near 0 [same files]. **Warning about `coststory.txt`:** the cost-story verification file was generated on the PRE-fix sweep (its mtime, Sep 21 20:45, precedes the refit `results.jsonl` at 21:17 and matches `results.stale-root-before-fix.jsonl` at 20:46), so its root columns contradict this section (e.g. "fb_window packed_rank_root_fusion flow+fences ... est b/r/f/vr/vf = 8.96/14.34/14.07/14.23/7.00" and "fb_window root_vf4: d = +0 binary") [results/aidb/verification/coststory/coststory.txt per-(dataset, variant) table and metadata table; `ls -lT`]. Its `P` constant already points at `results/aidb_final/sweep/results.jsonl` [coststory.py:4], so rerunning it refreshes those rows; it was not rerun here because the verification directory is outside this guide's write scope. What is unaffected by the staleness: the vp10 metadata delta (+1,597,656 B), the invariant that root variants differ from `packed_rank` only in `root_probes` and `transform_calls`, and the osm/planet 1,956-fence rows (uniform samples, whose keys start near 0). So the "flow + fences synergy" of the first pass was the artefact; after the fix the flow feature wins 0 of 60 candidate cells [results/aidb/MEETING_NOTES.md:101-106]. The regression test builds 4,000 keys from 9·10¹² and requires the raw root to beat binary search [test_main.cpp:169-171].

**Probe set for the cost estimate.** 2n − 1 keys: every fence and every midpoint fence_j + (fence_{j+1} − fence_j)/2 [index.hpp:371-372].

**Candidates** {raw, flow} × {ranks, virtual fences} [index.hpp:375-396]:
- feature x_j = `tmp.normalized(fence_j)` (tmp fitted on the fences: origin = first fence, span = last − first) or `(*flow)(fence_j)`;
- with virtual fences (`root_alpha > 0`, n > 2, x sorted): `smooth_cdf(x, root_alpha)` with budget ⌊root_alpha·n⌋; targets = slots; `table` of size n + virt maps every slot to its region (slot j.. of region j) [index.hpp:381-388]; the candidate is skipped if no virtual fence was inserted;
- `model.fit_xy(fences, x, targets)`; cost = (Σ over the 2n−1 probe keys of the `ok()` calls made by the real `locate_from_prediction`)/(2n−1) + (flow ? flow_cost : 0) [index.hpp:389-392];
- the binary-search cost on the same probe keys is measured too [index.hpp:397].
The best candidate replaces binary search only if its cost is strictly lower [index.hpp:399-400]; `learnability()` reports `root_probes_binary/raw/flow/vp_raw/vp_flow`, `root_model`, `root_flow`, `root_vp`, `root_virtual` [index.hpp:490-491].

**Refit.** `fit_root()` runs at the end of `bulk_load` and after every *split* [index.hpp:443,418]; a compaction without split keeps the fences, so the model remains valid; exactness is guaranteed by the correction in any case.

**Numbers.** 2M samples (489 regions; binary cost 8.956): fb_uniform raw 2.266, vp_raw 2.224 (1 virtual fence), flow 8.139, vp_flow 6.349 → chosen raw+fences, measured `root_probes/op` 2.079; fb_window raw 3.584, vp_raw 2.409 (34 fences), flow 7.860 → raw+fences, measured 2.273; osm_uniform: 1,956 virtual fences (= full budget 4·489) [results/aidb_final/sweep/results.jsonl; coststory metadata table]. Slot-table bytes = 4·(489 + virt), e.g. 9,780 B on osm/planet [coststory]. **200M keys** (48,829 regions, binary 15.658 ≈ log₂ 48,829 = 15.58): fb raw 10.802, vp_raw 3.497 with 973 virtual fences (search stopped early), measured 3.419/op; build 9.98 s vs 6.63 s for `packed_rank`. planet raw 25.80, flow 27.87 (both rejected), vp_raw 13.748 with 195,316 virtual fences = the full budget ⌊4·48,829⌋ → chosen, measured 13.685/op, **build 2,067.8 s** vs 6.26 s [results/aidb_fullscale/sweep/results.jsonl]. **Why this build number is wall clock while the region figure (2,407 s, section 5.5) is summed thread time:** `fit_root()` runs on the calling thread after the worker pool has joined [index.hpp:441-443], so its `smooth_cdf` over 48,829 fences is single-threaded and its whole cost lands in `build_ns`; it is *not* included in `learnability.smoothing_ns`, which only sums the per-region `r->smoothing_ns` [index.hpp:485] (planet `packed_rank_root_vf4` reports `smoothing_ns = 0` next to its 2,067.8 s build). Region smoothing, by contrast, runs inside the 16 build threads and is reported as the SUM of per-region times, so fb `packed_rank_vp10_root_vf004`'s 2,407.4 s of thread time is ≈ 150 s of wall clock, consistent with its `build_ns` of 161.0 s [results/aidb_fullscale/sweep/results.jsonl]. Complexity: 195,316 rounds × O(|seq|) with |seq| growing from 48,829 to 244,145, i.e. of the order of 10¹⁰ gap examinations (≥ 4 loss evaluations each) — our arithmetic from section 5.3, consistent with the 34 minutes observed [unverified: not profiled]. At `root_alpha = 0.04` (budget 1,953) fb again stops at 973 fences (same result, build 9.99 s) while planet's estimate 25.18 loses to binary search [same file, `packed_rank_root_vf004`].

---

### 9. The hardness tool (hardness.hpp, src/hardness.cpp)

AIDB [Zhang, Tang, Ailamaki, AIDB@VLDB 2026, §4.1] evaluates five scalar metrics on sorted keys, all O(N): RMSE and ME of a least-squares line, CD = "the maximum number of keys mapped to the same rank by a linear model fitted on the dataset using FMCD by Wu et al." (LIPP), PLA-32 and PLA-4096 = the number of segments of the optimal ε-bounded PLA (the two dimensions of the GRE metric), plus their 2- and 3-way compositions [AIDB §4.1]. The performance measure is "the index throughput under a uniform read-only workload over the same dataset used to populate the index" [AIDB §2.2 "Measuring Dataset Hardness"; §3.1 is the "Strawman Measurement" subsection].

- **`least_squares<T>(x)`** [hardness.hpp:84-106]: features are x − x_min (integers subtracted as `unsigned __int128`, exact), Welford OLS of rank i on the feature, then RMSE = √(Σeᵢ²/n), ME = max|eᵢ|. Hand case {0,1,3}: slope 9/14 = 0.642857, intercept 1/7, residuals −1/7, 3/14, −1/14 → ME = 3/14 = 0.214286, RMSE = √(1/42) = 0.154303 [test_hardness.cpp:128-131; scratch/index_demo.cpp block D]. fb full: RMSE 57,735,023.67, ME 99,999,995.54 [results/aidb/hardness.json full.fb].
- **`fmcd_fit` / `conflict_degree`** [hardness.hpp:108-227]: a port of LIPP's `build_tree_bulk_fmcd` (MIT, commit fe6ca49) [hardness.hpp:15-22]. LIPP's FMCD ("fastest minimum conflict degree") looks for the smallest window D such that a line with slope 1/U_T, U_T = (keys[size−1−D] − keys[D])/(L − 2) + 10⁻⁶, spreads the keys over L = size·(gap+1) slots with at most D-ish collisions: `i = 0, D = 1; while i < size−1−D: advance i while keys[i+D] − keys[i] ≥ U_T; if stuck, D += 1 and recompute U_T; stop when D·3 > size` [hardness.hpp:185-192]. gap = 1 (size ≥ 10⁶), 2 (≥ 10⁵), 5 otherwise [hardness.hpp:109-113]. Model: a = 1/U_T, base = L/2, anchor = (keys[size−1−D] + keys[D])/2 kept as an exact `__int128` `anchor2 = 2·anchor` so that position(key) = a·(key − anchor) + base uses exact 64-bit differences (LIPP itself computes a·key + b in long double, exact only on 80-bit platforms) [hardness.hpp:114-117,138-147,194-198]. If D·3 > size, the fallback fits the line through the two tertile midpoints [hardness.hpp:199-212]. `predict` = floor of the double product, clamped to [0, L−1] (LIPP's PREDICT_POS) [hardness.hpp:138-147]; **CD** = the longest run of equal predictions (sorted keys give non-decreasing slots) [hardness.hpp:218-227]. **Epsilon scaling:** LIPP's absolute +10⁻⁶ on U_T is used verbatim for integer keys; for double features (flow outputs, range O(1)) it would be up to 147 % of U_T, so it is scaled to 10⁻⁶·(range/(n−1)); the literal value is emitted alongside as `conflict_degree_lipp_epsilon` [hardness.hpp:149-167; hardness.cpp:160]. Rounding caveat: floor of a double product makes CD ±1-uncertain near slot boundaries (books 2M sample: 11 here and with LIPP's double arithmetic, 10 with exact rationals) [hardness.hpp:119-127]. Hand-traced example (test `cd_cases`, reproduced in block D): 50 consecutive keys then 950 keys 100 apart from 10⁶: gap 5, L = 6,000, D = 50, U_T = 89,900/5,998 + 10⁻⁶ = 14.98833, a = 0.0667186, base 3,000, anchor 1,044,950; the cluster is clamped to slot 0 (predict(cl[0]) = predict(cl[49]) = 0, predict(cl[50]) = 1, predict(cl[51]) = 7) → **CD = 50** [test_hardness.cpp:96-102]. fb full: gap 1, capacity 4·10⁸, D = 114, U_T = 193.27, CD = 110 [results/aidb/hardness_details.json fb.full_original.fmcd; hardness.json full.fb.conflict_degree].
- **`pla_segments<T>(in, ε)`** [hardness.hpp:229-311]: `OptimalPLA` is O'Rourke's 1981 streaming algorithm as PGM-index implements it (`OptimalPiecewiseLinearModel::add_point`, Apache-2.0, commit c6fcf3d): it maintains the upper and lower convex hulls of the strips [y−ε, y+ε] and the rectangle of feasible slopes; a point outside starts a new segment [hardness.hpp:252-284]. Integral keys are widened to `unsigned __int128` so PGM's sentinel (x_last + 1, n) exists even at 2⁶⁴ − 1, and cross products are `__int128` (exact) [hardness.hpp:230-249,287-294]. Duplicates follow PGM's rule (a run of equal keys is one point; (x+1, rank of the last duplicate) is added if there is room) [hardness.hpp:304-309]. Count = breaks + 1. Examples: a perfect line → 1 segment for ε ≥ 1 (2 at ε = 0 because the sentinel is off the line); the staircase of 10 runs of 100 consecutive keys 10⁶ apart → 10 segments at ε = 32, 1 at ε = 4,096 [test_hardness.cpp:67-75; block D].
- **`compute_metrics`** runs the four passes (LS, FMCD+CD, one PLA per ε) as `std::async` tasks [hardness.hpp:323-333]; results are identical serial vs parallel [test_hardness.cpp:136-138].
- **CLI** `scaleli_hardness --data PATH --dtype uint64|uint32 [--limit N] [--flow W.txt] [--virtual-alpha A] [--region-keys 4096] [--pla-eps 32,4096] [--check-sorted 0|1] [--write-sorted PATH] [--sort-only 1] [--threads T] [--verbose 1]` [hardness.cpp:84-106]: reads SOSD (8-byte count header) [hardness.cpp:41-58]; audits sortedness and duplicates (seven GRE files are served unsorted: covid, genome, history, libio, planet, stack, wise; the pipeline writes `<name>.sorted`) [hardness.cpp:121-135; aidb_pipeline.py:22-27]; `original` block on raw keys; with `--flow`, `transformed` block on sorted z(keys) (sorting is needed only if the flow is non-monotone; `unordered_pairs` is reported); with `--virtual-alpha`, `smoothed` block on the per-region augmented sequence produced by `smooth_regions` (real and virtual features in order, so it has n + v entries; features are z(keys) with `--flow`, else `double(key − min)`) [hardness.cpp:143-182; hardness.hpp:335-377]. Cost: ~3 s and 1.6 GB RSS for a sorted 200M-key file on 16 threads [hardness.cpp:104]; books full took 3.13 s [results/aidb/verification/pla/books_full_hardness.json elapsed_ns].
- **Verification.** PLA against PGM-index's own `make_segmentation` on the full 200M-key books file: PGM 262,604 / 97 segments, ours 262,604 / 97 [results/aidb/verification/pla/books_full_ref.json vs books_full_hardness.json]; 320 random small instances against a DP oracle [test_hardness.cpp:83-90]. Least squares against an exact-rational Python oracle (`vlib.py`, `Fraction`/`Decimal`): relative error 5.3×10⁻¹⁵ (RMSE) and 2.5×10⁻¹⁴ (ME) on the books 2M sample, 2.7×10⁻¹⁴ / 5.5×10⁻¹³ on the fb 5M prefix; CD: books sample cpp 11 vs exact-rational 10 (the documented ±1), fb 5M 94 = 94 [results/aidb/verification/metrics/realdata.log].

---

### 10. The scorer (tools/aidb_scores.py)

Protocol [AIDB §3.2, eqs. 1-3]: normalise each index's throughput p̂_I(S) = p_I(S)/std_I over the datasets; S_i is *harder* than S_j under a (multi-dimensional) metric iff h_k(S_i) ≥ h_k(S_j) on every dimension and > on at least one (ties on all dimensions are incomparable); importance w = sigmoid(δp) with δp = p̂(harder) − p̂(easier); a comparable pair is conforming when δp ≤ 0 and violating when δp > 0; R_I = Σ w over conforming, P_I = Σ w over violating; Conf_I = (R_I − P_I)/(R_I + P_I) (eq. 1), Conf = mean over indexes (eq. 2); Cov = (|C| − |U|)/(|C| + |U|) with |C| + |U| = C(n, 2) (eq. 3).

Code: `harder(hi, hj, dims)` = `all(≥) and any(>)` [aidb_scores.py:82-84]; `classify_pairs` splits the C(n,2) unordered pairs into (harder, easier) and incomparable [87-97]; `coverage` [100-102]; `conformance_of_variant(p_hat, comparable)` accumulates reward/penalty with `sigmoid(gap)` and returns Conf_I (0 when R + P = 0) [105-120]; `normalize` divides by the population std (ddof = 0 default; the paper does not state the ddof) [67-79]; `score_scope` builds the dataset corpus as the intersection of the hardness scope and every scored variant's throughput, skips variants with < 3 datasets or with zero throughput std (eq. 1 undefined) and drops non-finite entries with a reason [153-211]; 25 metrics = 5 scalars + C(5,2) = 10 pairs + C(5,3) = 10 triples, "PLA-32·PLA-4096" aliased "GRE" [45-58]. Scopes: `full, full_flow, sample, sample_flow, sample_csv, sample_flow_csv` (the metrics on 200M keys, on the flow-transformed 200M keys, on the 2M sample, after flow, after smoothing, after both) [49]. **Protocol deviation to state up front:** in EVERY scope the throughput side comes from the 2M uniform samples (every sweep row has `dataset_path = data/samples/<name>_2M_uniform_s42` and `initial_rows = 2,000,000` [results/aidb/sweep/results.jsonl]; `throughput.json` is built from that sweep [aidb_pipeline.py:159-175]). Scope `full` therefore pairs hardness measured on the 200M-key files with throughput measured on 2M-key samples of them, whereas the AIDB paper measures both on the same 200M-key dataset [AIDB §2.2]. Only `results/aidb_fullscale` (fb and planet, single seed) measures throughput at 200M keys, and it is not scored. Consequence for reading Conf: scope `sample` is the internally consistent one (hardness and throughput on the same 2M keys); scope `full` tests whether the full-file hardness ordering predicts the sample throughput ordering, which is a weaker, cross-scale statement and should be presented as such.

**How our throughputs get in.** `aidb_pipeline.py step_throughput` reads `sweep/results.jsonl`, checks that every (dataset, profile, seed, repeat) has a single `trace_fingerprint` across variants (paired traces), and writes `throughput.json = {variant: {dataset: median over seeds of throughput_ops_s/1e6}}` [aidb_pipeline.py:159-175,539-546]; `step_scores` calls `aidb_scores.py --hardness hardness.json --throughput throughput.json --output scores.json` [548-553]. The "indexes" of the paper are replaced by our ten control variants (`sorted_vector, raw_rank, packed_rank, packed_rank_flow, packed_rank_flow_forced, packed_rank_vp10, packed_rank_flow_vp10, packed_byte, packed_rank_fusion_auto, packed_rank_flow_costsel`) [results/aidb/throughput.json].

**Worked example (scope `full`, metric RMSE, variant `packed_rank`; computed with the module).** Throughput (MOPS): books 2.144, covid 2.217, fb 2.638, genome 2.476, history 2.816, libio 2.824, osm 1.932, planet 2.789, stack 2.942, wise 2.482; population std 0.32038 → p̂ = books 6.692, covid 6.921, fb 8.234, genome 7.728, history 8.789, libio 8.816, osm 6.030, planet 8.705, stack 9.182, wise 7.747. RMSE (full): fb 57.7M > planet 29.8M > osm 24.2M > books 18.1M > genome 7.5M > libio 3.4M > wise 2.2M > covid 1.8M > stack 0.84M > history 0.82M — all distinct, so all 45 pairs are comparable (Cov = 1.0). Pair (fb harder than books): δp = 8.234 − 6.692 = +1.543 → violating, w = sigmoid(1.543) = 0.824 (fb is *faster* although harder); pair (books harder than covid): δp = −0.229 → conforming, w = 0.443. Totals: R = 7.358 over 29 conforming pairs, P = 11.984 over 16 violating → Conf_packed_rank = (7.358 − 11.984)/(19.342) = **−0.239**. Averaging over the ten variants gives Conf = −0.317 (per variant from −0.565 for `sorted_vector` to −0.147 for `packed_byte`) [results/aidb/scores.json full.RMSE]. For RMSE·CD, 32 pairs are comparable and 13 incomparable → Cov = (32 − 13)/45 = 0.422 [scratch computation with `classify_pairs`].

---

### 11. Tests (what is checked, by name)

- `tests/test_main.cpp` (8 groups, run as one binary): `codecs` (round trips, bit widths 0-64, corrupt input), `endpoints` (empty index, duplicates, keys 0 and 2⁶⁴−1 across all routings/policies), `differential` (45 configurations × 3,500 random ops vs `OrderedMap`, with `maintain`/`validate` every 127 ops; requires splits), `adversarial_routes` (corrupted models must still answer exactly), `maintenance_and_memory` (adaptive raw bypass, memory accounting, `dump_layout`, invalid config throws), `learnability` (smoothing invariants: SSE never increases, virtual points inside the range and never equal to keys, strictly increasing slots; D₉₉ = 0 on uniform, > 0 on a cluster; hand-checked 2D2H2L forward pass 0.5·2.75 + 0.25·0.75 → tanh sums; text round trip; differential with flow + smoothing × bypass × relearn × routing; `transform_calls > 0` when forced), `fusion_auto` (Σchoices = regions, `cost_selected ≤ cost_none`, no flow when absent or when flow_cost = 100, no vp when α = 0, differential, parallel-build layout identical), `learned_root` (`root_probes_binary > 0`, chosen estimate < binary, virtual fences > 0 when chosen, exact lookups over all keys and 9,973-spaced probes, splits refit, the far-from-zero window regression) [test_main.cpp:13-183].
- `tests/test_hardness.cpp`: `exactness_near_2_64` (1,000 keys below 2⁶⁴: RMSE < 1e-9, PLA = 1, FMCD D = 1, CD = 1, and the same keys as doubles collapse to CD = 1000), `pla_cases` (line, staircase 10/1, duplicates, 320 DP-oracle differentials, monotone in ε), `cd_cases` (cluster D = 50/CD = 50, tertile fallback with hand-derived slots, uniform CD = 1, gap table, scaled epsilon on doubles), `least_squares_cases` ({0,1,3} hand values, parallel = serial), `smoothing_cases` (`smooth_regions`: 8 regions, sorted output, every real feature survives, thread count irrelevant, unsorted input throws), `parallel_sort_cases` (equals `std::sort` for 6 sizes × 8 thread counts), `cli_case` (JSON well-formed, `--flow --virtual-alpha 0.1 --region-keys 1024` → 16 regions with virtual points, `--limit`, unknown option rejected, `--sort-only --write-sorted` on a shuffled file with duplicates, `conflict_degree_lipp_epsilon` present) [test_hardness.cpp:47-214].
- `tests/test_tools.py`: `ToolsTests.test_feasibility, test_uniform_scaling_not_smoothing, test_virtual_greedy` (greedy ≤ baseline, exhaustive ≤ greedy on 10 keys), `test_dataset_sampling_and_corruption` (uniform/window/strided samplers), `test_flow_trainer_and_conflicts` (D₉₉ 0 vs > 0; monotone training lowers NLL; 8-line weight file), `test_cluster_bootstrap`; `BenchmarkTests.test_all_workloads, test_query_distributions, test_bulk_sampling_and_order, test_pairing` (identical trace fingerprint and result checksum across indexes), `test_full_width_and_duplicates, test_cli_rejects_invalid_inputs, test_learnability_options` (trains a flow on dumped keys and runs vp/flow/fusion/byte/adaptive variants; virtual points > 0, SSE non-increasing, forced flow in every region, Σchoices = regions; α = 1.5 rejected), `test_latency_skip, test_u32_u64_import` [test_tools.py].
- `tests/test_aidb_scores.py` (21 test methods: `HandBuiltExample` 6, `ProtocolProperties` 6, `DatasetSelection` 7, `CommandLine` 2 [test_aidb_scores.py:31-291]): `test_normalization, test_scalar_rmse, test_two_dim_rmse_me_trades_coverage_for_conformance, test_three_dim_with_tied_dimension_matches_two_dim, test_all_ties_are_incomparable, test_sample_std_changes_scale_not_sign, test_scalar_metric_has_full_coverage_when_values_are_distinct, test_adding_a_dimension_never_increases_coverage, test_perfectly_conforming_ordering_scores_one_and_reversed_minus_one, test_zero_std_variant_is_skipped_not_scored, test_importance_is_sigmoid_of_signed_gap, test_canonical_metric_names, test_nan_and_inf_throughput_are_dropped_with_a_reason, test_null_throughput_shrinks_the_corpus_and_says_so, test_string_throughput_skips_the_variant_with_the_real_reason, test_cli_accepts_nan_in_throughput_json, test_variant_with_fewer_than_three_datasets_is_skipped, test_datasets_are_the_intersection_of_hardness_and_every_scored_variant, test_scopes_are_independent_and_missing_fields_drop_a_dataset, test_end_to_end_writes_contract_shape_and_prints_table, test_rejects_unknown_option_and_missing_scope`.
- `tests/test_aidb_pipeline.py`: `test_contract_variants_and_common_block, test_extra_variants_and_build_threads_are_opt_in, test_dry_dataset_urls_and_report_step_detection, test_median_throughput_orders_contract_variants_first, test_write_json_leaves_identical_files_untouched, test_sosd_helpers_and_metric_block, test_sample_stale_reason_and_cached_run_identity`. `tests/test_aidb_report.py`: `test_full_inputs_via_cli, test_empty_results_dir, test_corrupt_and_partial_inputs`.

---

### 12. Questions the supervisor may ask, with answers

1. **"Why do virtual points cost zero key bytes?"** Because they are never stored as records. `smooth_cdf` returns only slot ranks and feature values; the index uses the slots as the rank model's targets and writes one `slot_begin` per block. Lookups compare the model's prediction against `slot_begin` and then correct on the exact `last` keys of the blocks, so the arena, values and scans are untouched [index.hpp:83,125,307-308,312]. The only cost is 8 bytes per virtual point of `virtual_features` kept for compaction reuse = 0.8 B/key at α = 0.1, measured 1,597,656 B for 199,707 points [coststory]. CSV materialises gaps in its host indexes (ALEX/LIPP) and uses them for inserts: "we include V_i in the loss calculation, such that the storage space allocated to the virtual points can be used to accumulate data insertions, with minimized prediction errors when querying the inserted data points" [CSV §4, after eq. 4]; we deliberately do not.

2. **"Why is the selector's cost a probe count and not time?"** Time on this host is noisy at the level of the effect (two measurements of the identical structure differ by up to 8-10 % [MEETING_NOTES.md:107-111]), whereas the probe counts are deterministic, reproducible across seeds and comparable across candidates because everything except routing is held fixed. The selector runs the *real* `locate_block` on the region's own keys, so it measures the work the index will do, not a proxy such as SSE or D₉₉ [index.hpp:240-242,262-264]. The one time-like quantity, the transform, enters as `flow_cost` probe-equivalents (default 4, measured 66 ns per call ≈ a few probes) [index.hpp:64; MEETING_NOTES.md:114-115].

3. **"What exactly is a fence probe?"** One evaluation of `blocks[i].last < k` inside `Region::locate_block` [index.hpp:135]. It is the exact-key comparison that verifies or corrects the model's block hint: at least one per lookup (checking the predicted block), two for a hit that needs the left neighbour checked, and O(log d) for a prediction d blocks off. Root fence comparisons (`low_fence <= k`) are counted separately as `root_probes` [index.hpp:350,353].

4. **"Is your transform the NFL flow?"** No. Same deployed *shape* (2D2H2L, features [x, x − ⌊x⌋], tanh, sum decoder) and the same text weight format, but a stand-in trainer (1-D change-of-variables likelihood with Adam, not a BNAF), `--monotone` zeroes the fractional-feature weights, and the output is used as a model feature only; NFL sorts by the transformed key and builds AFLI on it [transform.hpp:1-19; train_flow.py:1-14].

5. **"Why does the flow not help at the region level?"** With 4,096-key regions each linear model already absorbs the global curvature; the tail conflict degree barely moves (fb 8.45 → 8.46 per region, osm 36.2 → 37.0), so the 10 %-gain bypass keeps it in 1-9 % of regions, and the cost-based selector never picks it at flow_cost 4 [results/aidb/sweep/results.jsonl]. The granularity ablation (regions 4,096 / 16,384 / 32,768) tested whether larger regions change this, and they do not: forcing the flow changes `fence_probes/op` negligibly at every region size (fb: 2.6010 vs 2.6010 at 4,096; 3.2104 vs 3.2094 at 16,384; 3.6888 vs 3.6922 at 32,768; osm: 5.0777 vs 5.0780; 8.3355 vs 8.3361; 10.1188 vs 10.1200, forced-flow vs plain `packed_rank`), `correction_distance/op` likewise (osm 18.17 vs 18.17 at 32,768), and `fusion_auto` picks the flow in 0 regions at all three sizes (`flow_region_fraction` = 0.0, `virtual_points_per_key` ≈ 0.095-0.100). So larger regions do raise the raw probe count (fb fence 2.60 → 3.69, osm 5.08 → 10.12: one line per region fits worse), but the flow still does not help; only virtual points do (fb `fusion_auto` fence 2.18-2.25 at all sizes) [results/aidb_granularity/summary_r4096.csv, summary_r16384.csv, summary_r32768.csv, columns fence_probes_per_operation, correction_distance_per_operation, flow_region_fraction, virtual_points_per_key; config.json].

6. **"What did the sentinel-fence bug do?"** It fitted the root's raw least-squares line through (0, region 0) while the real keys started at e.g. 9·10¹² (window samples), so the raw root looked worse than binary search (est. 14.3 vs 8.96 on fb_window) while the tanh-saturated flow feature was less damaged (14.07): the fusion candidate "won" for the wrong reason. Fitting on the first real key gives raw 3.58, flow 7.86; the flow then never wins [results/aidb_final/sweep/results.stale-root-before-fix.jsonl vs results.jsonl; index.hpp:366-370].

7. **"Where is the O(1) in CSV's loss evaluation, and why is your algorithm O(λ·n)?"** Inserting one point after position i shifts every later target by one; with suffix sums of x and of the positions the six OLS sums update in O(1) (S_y += cnt, S_yy += 2·suf_y + cnt, S_xy += suf_x) and SSE = S_yy − S_xy²/S_xx (centred) follows in O(1) [smoothing.hpp:64-70]. But every round still scans every gap (≥ 4 evaluations each) and the budget is λ = α·n rounds, hence O(λ·n) overall; the paper's O(n + λ) is not reproduced [smoothing.hpp:11-13]. This is why smoothing 200M keys costs ~2,400 s of thread time and why 195,316 virtual fences at the root took 2,068 s [results/aidb_fullscale].

8. **"How is CD computed and why can it differ from LIPP by one?"** It is LIPP's FMCD fit (window search on D and U_T, a = 1/U_T, base L/2, tertile fallback) followed by the longest run of equal floor(a·(k − anchor) + base) slots. The differences are kept as exact 128-bit integers, but the final floor of a double product decides keys within ~10³ units of a slot boundary by rounding, so CD is LIPP-faithful, not bit-reproducible: books 2M gives 11 (ours, and LIPP-style doubles) vs 10 (exact rationals) [hardness.hpp:119-127; verification/metrics/realdata.log]. On double features the +10⁻⁶ epsilon is scaled by the mean gap; the literal-epsilon value is reported beside it [hardness.hpp:149-159].

9. **"Are the PLA counts really PGM's?"** Yes, verified on the full books file: 262,604 (ε = 32) and 97 (ε = 4,096) both from PGM's `make_segmentation` and from ours [verification/pla/books_full_ref.json, books_full_hardness.json], plus 320 DP-oracle differentials [test_hardness.cpp:83-90]. Ours additionally survives x_last = 2⁶⁴ − 1 by widening to 128 bits [hardness.hpp:287-294].

10. **"Why does smoothing lower the per-region SSE by 85 % on fb but raise the whole-sequence RMSE?"** The regions minimise their own OLS SSE (3.80×10⁹ → 5.87×10⁸ summed over 489 regions), while the hardness tool's `smoothed` block fits ONE line over the concatenated 2.2M-entry sequence (RMSE 1,448 → 1,591) [results/aidb/hardness.json sample vs sample_csv]. A per-region optimum does not imply a better global line; the index only ever uses the per-region models, and its measured `fence_probes/op` fall from 2.60 to 2.18 with `correction_distance/op` 0.257 → 0.088 [results/aidb/sweep/results.jsonl fb packed_rank vs packed_rank_vp10].

11. **"What happens to the virtual points on inserts?"** Nothing until the region's delta reaches 64 entries; then the compaction re-places the stored virtual feature values among the new keys in O(n + v), recomputes the SSE with and without them, and drops them if they no longer help; the greedy search is not re-run unless `--relearn 1` [index.hpp:289-303]. Re-running it made the 10 %-insert workloads 116-232× slower [results/learnability/summary.csv].

12. **"Why is the parallel bulk load safe and deterministic?"** Regions are independent: each `rebuild` writes only its own `Region` and reads a const `Config`/`FlowTransform`; the partition is static (thread w builds regions w, w+W, …); exceptions are captured per thread and rethrown; the resulting layout is byte-identical for 1 and 4 threads in the test [index.hpp:433-441; test_main.cpp:143-149]. Queries stay single-threaded.

13. **"Why measure throughput on the same keys that were loaded, with no misses?"** Because that is the AIDB protocol ("the index throughput under a uniform read-only workload over the same dataset used to populate the index") [AIDB §2.2]; the paper's own workload is "random lookups for all keys in the dataset" with "a warm-up phase of 20 million lookups and then a measurement phase of 100 million lookups" on 200M-key datasets [AIDB §4.1 "Workloads and Measurements"]. Our sweeps all set `load-ratio 1, miss 0, query-distribution uniform, latency 0, instrument 1, build-threads 16` and differ in size: the uniform/window 2M sweeps use ops 1,000,000, warmup 200,000, seeds 11/29/47 [results/aidb/sweep/environment.json config.common]; `aidb_final` uses ops 5,000,000, warmup 500,000, qos 1, seeds 11/29/47 [results/aidb_final/config.json]; `aidb_fullscale` uses ops 2,000,000, warmup 200,000, qos 1, single seed 11 on the 200M-key files [results/aidb_fullscale/config.json]; `aidb_granularity` uses ops 1,000,000, warmup 200,000, seeds 11/29 [results/aidb_granularity/config.json]. And, as said in section 10, all scored throughputs are on the 2M samples, not on the 200M-key files.

14. **"Which numbers in the learnability block are wall-clock and therefore not reproducible?"** `smoothing_ns`, `transform_ns` (summed across build threads), `build_ns`, and of course `throughput_ops_s`. Everything else (regions, flow_regions, virtual_points, SSEs, tail conflicts, choices, cost estimates, root estimates) is deterministic [benchmark.cpp:69-78; index.hpp:482-493].

15. **"What is the difference between `coordinate_probes` and `fence_probes`, and why is the former always ≈ 5?"** Coordinate probes are the lower-bound binary search of the real-valued prediction against the 32 block boundaries in slot (or byte) space: 5 or 6 comparisons depending on where the prediction lands (31 of the 33 outcomes cost 5, 2 cost 6; measured mean 5.029), and independent of model *accuracy* only in the sense that the count never grows with the error [index.hpp:122-127; scratch/fix05_checks.py]; fence probes are the exact-key comparisons that fix the hint and grow with the correction distance [index.hpp:135-150]. A better model shows up in `fence_probes` and `correction_distance`, never in `coordinate_probes`.


---

## 06 — The exact experimental protocol

**What this section gives you.** Everything that was actually run, precisely enough to rerun it: the benchmark binary option by option (how keys are loaded, how the query trace is generated and fingerprinted, what the four passes time and check, what every JSON field means), the two Python drivers that turn a config into paired runs and a summary (`tools/run_suite.py`, `tools/summarize.py`), the nine-step AIDB pipeline (`tools/aidb_pipeline.py`) with the exact parameters it fixed, the machine and the three build directories, the measured noise floor, one card per experiment E1–E7 (config, variants with the option string each expands to, seeds, dates, durations, row counts, caveats), a dictionary of every variant name, the reproduction commands in order, and the questions the supervisor is likely to ask. Every number is cited to a PDF, a source file or a result file; anything I could not confirm carries an `[unverified: …]` tag. All our components are clean-room CONTROLS inside the SCALE-LI experimental map, never reproductions of NFL, AFLI, CSV or the AIDB benchmark [aidb_pipeline.py:14-15]. Paths: S = `/Users/louisvasseur/Downloads/scaleli_sota/experimental/scaleli`; the repo root is two levels up.

Notation (shared with the other sections): keys k₁ < … < kₙ (uint64), rank r(kᵢ) = i − 1; region = 4,096 keys, block = 128 keys; root = the whole-range structure over region fences; probes = key comparisons counted by the software counters; λ = α·n is the virtual-point budget; D₉₉ the tail conflict degree.

---

### 1. The benchmark binary: `src/benchmark.cpp` (222 lines)

#### 1.1 What one invocation does

One call of `scaleli_bench` = load keys → build a deterministic workload (initial set + operation trace) → run up to four separate passes on freshly built indexes → print ONE JSON object on stdout [benchmark.cpp:123-152]. The binary never writes result files itself; `tools/run_suite.py` captures stdout into `results.jsonl` (section 2).

#### 1.2 Every command-line option (allow-list at benchmark.cpp:27; unknown options abort with `unknown option:`)

Options are `--name value` or `--name=value`; a bare `--name` means `1` [benchmark.cpp:25-26]. Flags accept only `0`/`1` [benchmark.cpp:33]. Defaults are those in the parsing code (benchmark.cpp:189-208), not the help text (which lists the same values).

| option | default | meaning (where read) |
|---|---|---|
| `--help` | off | prints the option summary and exits [benchmark.cpp:164-188] |
| `--index` | `scaleli` | `scaleli` (our region/block index), `sorted_vector` (plain binary search over a sorted array), `ordered_map` (std::map wrapper); `alex`/`pgm` only if built with `SCALELI_EXTERNAL` [benchmark.cpp:213-219] |
| `--n` | 50000 | number of synthetic keys when no `--data` is given [benchmark.cpp:208] |
| `--distribution` | `dense_sparse` | synthetic generator: linear, uniform, lognormal, dense_sparse, staircase, locally_hard, clustered, near_u64, duplicates [workload.hpp:14-32] |
| `--seed` | 42 | seed of the `std::mt19937_64` that drives BOTH the synthetic generator and the workload trace [workload.hpp:15, workload.hpp:77] |
| `--ops` | 20000 | number of operations in the trace [benchmark.cpp:203] |
| `--profile` | `read_only` | operation mix: read_only (read = 1), read_heavy (.8/.1/.05/.02/.03 read/insert/update/erase/scan), scan_heavy (.45/.05/scan .5), write_heavy (.2/.5/.2/.1), churn (.2/.3/.2/.3), append (.5/.5, insert_mode append), shift (.4/.4/.1/scan .1, insert_mode shift, moving_hotspot queries) [benchmark.cpp:194-201] |
| `--read --insert --update --erase --scan` | from profile | override the ratios; must be ≥ 0 and sum to 1 within 1e-8 [workload.hpp:61] |
| `--policy` | `min_bytes` | block codec policy: raw, min_bytes, smooth, adaptive, forced [benchmark.cpp:189] |
| `--codec` | `for` | codec used when policy = forced: raw, for, delta, linear [benchmark.cpp:189] |
| `--routing` | `byte` | how a key is located inside a region: binary (fence binary search), rank (linear rank model + exponential search on fences), byte [benchmark.cpp:189] |
| `--region-keys` | 4096 | keys per region [benchmark.cpp:189]; regions are cut every `region_keys` rows at bulk load [index.hpp:432] |
| `--block-keys` | 128 | keys per block [benchmark.cpp:189] |
| `--delta-limit` | 64 | per-region delta buffer size before compaction [benchmark.cpp:189] |
| `--restart` | 16 | restart interval of the delta codec [benchmark.cpp:190] |
| `--min-saving` | 0.05 | minimum fraction of bytes a codec must save to be chosen under min_bytes [benchmark.cpp:191] |
| `--smooth-scale`, `--hot-enter`, `--hot-exit`, `--heat-decay` | 1, .25, .05, .99 | adaptive-policy heat parameters [benchmark.cpp:193]; not used by any AIDB run |
| `--data` | (none) | path of a key file; when present `--n/--distribution` are ignored [benchmark.cpp:207-208] |
| `--format` | `sosd` | `sosd` = 8-byte little-endian count header then keys; `raw` = keys only [workload.hpp:37-48] |
| `--dtype` | `uint64` | `uint64` or `uint32` (4-byte keys are widened) [benchmark.cpp:207] |
| `--limit` | 0 | 0 = whole file; N > 0 = the first N keys of the file, a PREFIX, and the JSON then says `"dataset_scope":"prefix"` [workload.hpp:45, benchmark.cpp:125] |
| `--load-ratio` | 0.75 | fraction of the unique keys bulk-loaded; the rest are held out for inserts [workload.hpp:79-82] |
| `--bulk-sampling` | `uniform` | `uniform` = shuffle the unique keys with the seeded RNG before taking the loaded fraction; `prefix` = take the first fraction in key order [workload.hpp:78] |
| `--insert-order` | `shuffled` | order of held-out keys for inserts [workload.hpp:83-84] |
| `--insert-mode` | `random` | random, append, hotspot, shift [workload.hpp:119-127] |
| `--query-distribution` | `uniform` | uniform, hotspot (80% of reads in the first 10% of live keys), moving_hotspot (hot set moves to the last 10% after ops/2), zipf [workload.hpp:98-106] |
| `--zipf-theta` | 0.99 | Zipf exponent [workload.hpp:94-97] |
| `--miss` | 0.1 | probability that a read is turned into a read-miss on an absent key [workload.hpp:129] |
| `--scan-length` | 100 | rows returned per scan [benchmark.cpp:203] |
| `--flow` | (none) | NFL-format weight file; loads a `FlowTransform` and sets `cfg.flow` [benchmark.cpp:192] |
| `--flow-bypass` | 1 | 1 = per-region NFL-style auto-switch (keep the flow only if it lowers D₉₉ by ≥ `flow_min_gain`); 0 = force the flow into every region [index.hpp:281-282] |
| `--flow-min-gain` | 0.1 | relative D₉₉ reduction required by the bypass rule [index.hpp:281] |
| `--virtual-alpha` | 0 | CSV-style virtual-point budget per region, λ = α·(keys in region); 0 = off [benchmark.cpp:191] |
| `--relearn` | 0 | 1 = rerun the flow decision and smoothing on every compaction; 0 = reuse bulk-load decisions [benchmark.cpp:191] |
| `--fusion` | `manual` | `auto` = per-region cost-based choice among {none, flow, vp, both} scored by the real `locate_block` on the region's own keys [index.hpp:239-272] |
| `--flow-cost` | 4 | probe-equivalents charged per lookup to any candidate that uses the flow [index.hpp:264] |
| `--root` | `binary` | `binary` = fence binary search; `model` = one global linear model over region fences with candidates {raw, flow} × {ranks, virtual fences}, kept only if its estimated probes beat binary search [index.hpp:348-372] |
| `--root-alpha` | 0 | with `--root model`: virtual-fence budget = alpha × regions (alpha < 64) [benchmark.cpp:191, help text] |
| `--build-threads` | 1 | regions are built concurrently; `build_ns` becomes the wall clock of the parallel build; queries are always single-threaded [help text; benchmark.cpp:191] |
| `--qos` | 1 | macOS only: 1 = call `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` at the start of `main` so the process prefers performance cores; 0 = default class. Not a pinning guarantee [benchmark.cpp:158-163] |
| `--verify` | 1 | pass 3: differential validation against `OrderedMap` (std::map) [benchmark.cpp:98-110] |
| `--instrument` | 1 | pass 4: software counters on a fresh replay [benchmark.cpp:111-115] |
| `--latency` | 1 | pass 2: per-operation timings on a fresh replay; 0 skips it and emits every latency block with `count 0` [benchmark.cpp:84-97] |
| `--warmup` | 4096 | warm-up lookups before the timed loop in every pass [benchmark.cpp:44-48, benchmark.cpp:59] |
| `--dump-layout`, `--dump-trace`, `--dump-keys` | (none) | write the region/block layout CSV, the trace CSV (`operation,key,value,length`), or the loaded keys as a SOSD file [benchmark.cpp:116-120, 210-212] |

There is no `--sort` option in the benchmark: sortedness of a data file is handled by `canonicalize()` (stable sort + de-duplicate, later value wins) applied to every loaded input [types.hpp:38-44, workload.hpp:75]. The sorted on-disk copies of the seven unsorted GRE files come from `scaleli_hardness --sort-only 1 --write-sorted` in the pipeline (section 3).

#### 1.3 How keys are loaded

`read_dataset(path, format, width, limit)` [workload.hpp:37-48]: checks that `(bytes − header) % width == 0`, reads the 8-byte count when `format == sosd`, checks that the count equals `(bytes − header)/width`, applies `--limit` as `min(count, limit)`, then reads every key little-endian and pairs it with the value `mix64(i)` (i = file position) [workload.hpp:46-47]. `make_workload` then sets `source_rows = input.size()`, canonicalizes (sort + dedupe) and sets `unique_rows` [workload.hpp:75]. For the GRE files `source_rows = unique_rows = 200,000,000` (no duplicates: provenance `sort_audit.duplicates = 0` for all ten [results/aidb/provenance.json]); for the samples both are 2,000,000 [results/aidb/sweep/results.jsonl, any row].

#### 1.4 How the query trace is generated (workload.hpp:74-137)

```
rng = mt19937_64(seed)                                 # workload.hpp:77
if bulk_sampling == uniform: shuffle(input, rng)       # consumes RNG state even at load_ratio 1
load_n = max(1, floor(|input| * load_ratio))           # load_ratio 1 -> load_n = n
initial = canonicalize(input[0:load_n])                # sorted again, so the loaded set is sorted
heldout = keys input[load_n:]                          # empty at load_ratio 1
dynamic = insert+update+erase > 0                      # false for read_only
for op in 0..ops-1:
    p = unit(rng)                                      # U(0,1)
    if p < read:                                       # read_only: always
        key = sample_live(op)                          # uniform: i = rng() % n; key = initial[i]
        if unit(rng) < miss: kind = ReadMiss; key = absent(key)   # miss 0 -> never
        else kind = ReadHit
    ... (insert / update / erase / scan branches, unused in the AIDB runs)
```

So for the AIDB sweeps (`--profile read_only --load-ratio 1 --miss 0 --query-distribution uniform`) the trace is exactly `ops` uniformly random ReadHit lookups of loaded keys, drawn by `rng() % n` [workload.hpp:104] after two `unit(rng)` draws per operation. The trace depends only on (the key set, `--seed`, `--ops`, the profile/ratios) and NOT on any index option, which is what makes runs across variants paired. `absent()` produces near misses (`near+1+attempt` for 64 attempts, then random) [workload.hpp:108-113]; unused here because miss = 0.

**Trace fingerprint** [benchmark.cpp:54]: `h = 0; for each initial record r: h = mix64(h ^ digest_record(r)); for each op o: h = mix64(h ^ mix64(o.key) ^ mix64(o.value) ^ mix64(unsigned(o.kind)) ^ mix64(o.length))`, with `mix64` the splitmix64 finalizer [workload.hpp:11] and `digest_record(r) = mix64(r.first) ^ mix64(r.second + 17)` [workload.hpp:12]. It is printed as a decimal string (`"trace_fingerprint"`). Two runs with the same fingerprint executed the same operations on the same loaded records. Example: every fb_uniform run at seed 11 in the final sweep has fingerprint `12214875262882230774` [results/aidb_final/sweep/results.jsonl].

**Result checksum** [benchmark.cpp:80]: over the throughput pass, `throughput_digest = mix64(throughput_digest ^ r.checksum)` where for a read `r.checksum = mix64(*value) ^ mix64(key)` when found, 0 otherwise [benchmark.cpp:36-38]. Since values are `mix64(file position)`, a different index that returns a wrong value or misses a key produces a different checksum. Every pass recomputes it and the binary aborts if they differ ("fresh replays produced different results", "instrumentation changes results") [benchmark.cpp:96, 115]. Example: fb_uniform seed 11 → `17216331355546010123` for all nine variants [results/aidb_final/sweep/results.jsonl; verified by the recompute verifier: "one result_checksum across their variants (0 mismatches)", results/aidb/verification/verdicts.json entry 11].

**Warm-up** [benchmark.cpp:44-48]: `warmup(ix, w, n)` performs `min(n, |initial|)` lookups of `initial[(j·997) mod |initial|]` for j = 0, 1, …: a deterministic stride-997 walk over the loaded keys, not the trace, and not random. Its XOR of values goes into a `volatile` sink so it cannot be optimized away. `warmup_reads` in the JSON is `min(warm, |initial|)`.

#### 1.5 The four passes (each on a freshly built index) [benchmark.cpp:58-115]

| pass | enabled by | what happens | what is measured |
|---|---|---|---|
| 1 throughput | always | `make()` → `bulk_load(initial)` timed → learnability block read → `memory()` → warm-up → timed loop over the whole trace, no per-op timer, no counters → `memory()`, `maintenance()`, `size()` → `maintain()` (drain) timed | `build_ns`, `elapsed_ns`, `throughput_ops_s = ops·10⁹/elapsed_ns`, `result_checksum`, `memory_before/after/after_drain`, `maintenance` |
| 2 latency | `--latency 1` | fresh index, warm-up, then `Clock::now()` around EVERY operation; digest must equal pass 1 | `latency_ns` per op kind (p50/p95/p99/max), `phase_latency_ns` (first/second half of the trace), `compaction_operation_latency_ns` |
| 3 verify | `--verify 1` | fresh index and a `std::map` reference; every op executed on both; any mismatch throws `differential mismatch at operation j`; then `validate()` and a full-range scan comparison | `verified: true` (or abort) |
| 4 instrument | `--instrument 1` | fresh index, warm-up, replay with a `QueryStats*` so every probe is counted; digest must equal pass 1 | `work_counters` |

All AIDB sweeps ran with `--verify 0 --latency 0 --instrument 1` [results/*/sweep/environment.json config.common], i.e. two bulk loads per run (pass 1 and pass 4), no per-operation timer in the timed pass, no differential check inside the sweeps. Differential verification with `--verify 1` was run separately on synthetic data by the final verifier (T7: `--n 20000 --distribution lognormal --ops 5000 --root model --root-alpha 4 --fusion auto --virtual-alpha 0.1 --verify 1` → `verified true`) [verdicts.json entry 15] and by the unit tests (46 Python tests + 4 ctest binaries) [verdicts.json entry 15, T3/T4]. Consequence of two bulk loads: for planet at full scale with root smoothing, `build_ns = 2,067.8 s` but the job took ≈ 70 min of wall clock (23:49:40 → 00:59:37 local) because pass 4 rebuilds the same index [results/aidb_fullscale/sweep/results.jsonl planet/packed_rank_root_vf4; results/aidb/run_chain3.log].

`clock_pair_p50_ns` is the median of 10,000 back-to-back `Clock::now()` pairs measured after the passes: 0 or 41 ns on this machine [benchmark.cpp:121; results/aidb_final/sweep/results.jsonl], i.e. the timer overhead reported (never subtracted).

#### 1.6 The JSON output: every field (benchmark.cpp:123-152)

Top level (`schema_version` 1): `index`, `distribution`, `profile`, `dataset_path`, `dataset_scope` (`prefix` when `--limit` > 0 else `full_input`), `seed`, `source_rows`, `unique_rows`, `initial_rows` (loaded), `final_rows` (after the trace), `operations`, `trace_fingerprint`, `result_checksum`, `verified`, `instrumented`, `latency_pass`, `software_counters_supported` (true only for `--index scaleli`), `policy`, `routing`, `forced_codec`, `region_keys`, `block_keys`, `delta_limit`, `restart_interval`, `min_saving_fraction`, `smooth_scale`, `flow_weights` (path or ""), `flow_bypass`, `flow_min_gain`, `virtual_alpha`, `relearn_on_compaction`, `fusion`, `flow_cost`, `build_threads`, `root`, `root_alpha` (the last two only in binaries that have the root option; the build-fusion rows lack them), `learnability` (block, or `null` for non-scaleli indexes), `query_distribution`, `insert_mode`, `bulk_sampling`, `insert_order`, `load_ratio`, `scan_length`, `requested_miss_ratio`, `warmup_reads`, `build_ns`, `elapsed_ns`, `throughput_ops_s`, `scan_records_returned`, `scan_records_per_total_second`, `clock_pair_p50_ns`, `memory_before`, `memory_after`, `memory_after_drain`, `maintenance`, `latency_ns`, `phase_latency_ns`, `compaction_operation_latency_ns`, `work_counters`. `run_suite.py` appends `dataset`, `variant`, `repeat`, `command` [run_suite.py:69].

`memory_*` [benchmark.cpp:53; types.hpp:29-37]: `key_bytes` (encoded key arenas), `value_bytes` (8 per key), `metadata_bytes` (fences, block descriptors, models, virtual features, root tables), `delta_bytes`, `reserved_slack_bytes`, `accounted_bytes` = the sum, `estimated` (false: counted allocations, "excludes malloc overhead and process RSS"). Example (fb_uniform, packed_rank_vp10, seed 11): key 5,636,561 + value 16,000,000 + metadata 2,738,824 = 24,375,385 B = 12.19 B/key [results/aidb_final/sweep/results.jsonl].

`maintenance`: `compactions`, `splits`, `bytes_rewritten`, `max_rewrite_bytes`, `raw_bypasses`, `drain_ns`, `drain_additional_rewrite_bytes` [benchmark.cpp:144-146]; all zero in read-only runs except `drain_ns` (a few µs).

`work_counters` [types.hpp:17-24; increment sites]: `root_probes` (fence comparisons while locating the region: binary loop index.hpp:353 or the model's exponential search index.hpp:350), `coordinate_probes` (comparisons inside a block coordinate search, index.hpp:124), `fence_probes` (block-fence comparisons `blocks[i].last < k` in the region's locate, index.hpp:135), `delta_probes` (delta-buffer binary search, index.hpp:115), `key_at_calls` and `decoded_keys` (codec key accesses/decodes, codec.hpp:134-181), `codec_bytes_examined`, `block_routes` (one per block prediction, index.hpp:151), `correction_distance` (Σ |predicted block − actual block|, index.hpp:151) and `max_correction_distance`, `blocks_decoded_for_scan`, `transform_calls` (flow evaluations on the lookup path, transform.hpp:65). Per-operation values are these divided by `operations`; e.g. binary root: `root_probes/ops = 8.96` at 2M keys and 15.66 at 200M keys (section 8, Q5).

`learnability` [benchmark.cpp:66-76; index.hpp:472-493], read from the pass-1 index right after bulk load: `regions`, `flow_regions` (regions whose model uses the flow feature), `virtual_points` (Σ over regions), `keys`, `rank_sse_before/after` (Σ over regions of SSE of the linear fit before/after smoothing), `smoothing_ns` and `transform_ns` (Σ over regions of thread time, so with 16 build threads they exceed wall clock), `tail_conflicts_raw_mean`, `tail_conflicts_flow_mean` (mean D₉₉ over regions), `flow_bytes`, `choices {none, flow, vp, both}` (fusion-auto selections per region), `cost_none_mean`, `cost_selected_mean` (auto mode only), and the root block: `root_model` (true iff a learned root is in use), `root_flow` (root uses the flow feature), `root_vp` (root uses virtual fences), `root_virtual` (number of virtual fences), `root_probes_binary/raw/flow/vp_raw/vp_flow` (the selector's expected probes per lookup for each candidate, measured by the real root locate on the fences and fence midpoints; 0 for candidates not evaluated).

---

### 2. `tools/run_suite.py` and `tools/summarize.py`

#### 2.1 run_suite.py (78 lines): config schema and execution

Config JSON keys [run_suite.py:39-58]:

- `datasets`: list of strings (then `{"name": s, "distribution": s}`) or objects `{"name", "data", "format", "dtype", "flow_weights", "flow_train_args", …}`; every key except `name`, `flow_weights`, `flow_train_args` becomes a `--key value` option [run_suite.py:53].
- `profiles`: list of profile names → `--profile`.
- `variants`: list of `{"name", option: value, …}`; every key except `name` becomes an option [run_suite.py:54]; booleans are printed as `1`/`0` [run_suite.py:61].
- `seeds` (default `[42]`) → `--seed`; `repeats` (default 1, or `--repeats`); `common`: options prepended to every job; `schedule_seed` (default 137).
- `"$flow"` substitution [run_suite.py:55-58]: a variant with `"flow": "$flow"` gets the dataset's `flow_weights` path resolved to an absolute path; missing file → `RuntimeError` telling you to run `tools/prepare_flows.py`.

Jobs = the cartesian product datasets × profiles × variants × seeds × repeats, shuffled by `random.Random(schedule_seed).shuffle` [run_suite.py:48-49], so variants are interleaved in time (a slow-down of the host affects all variants, not one). Each job runs `subprocess.run(command, timeout=--timeout)` (default 300 s; the AIDB runs passed 3600, 7200 or 14400 s); a non-zero exit writes `failed_command.json` and aborts [run_suite.py:62-66]. With `--resume`, jobs whose `(dataset, profile, variant, seed, repeat)` already appear in `results.jsonl` are skipped and new rows are appended [run_suite.py:36-42]; without it an existing `results.jsonl` is refused. `--cpu` pins with `taskset` on Linux only (never used here).

`environment.json` [run_suite.py:20-27]: `timestamp_utc`, `platform`, `machine`, `python`, `cpu` (output of `lscpu`, which is `"unavailable"` on macOS), `compiler` (`c++ --version`), `affinity` (null on macOS), `binary_sha256`, `git_commit` (`HEAD` here: the workspace has no commits), `warning` ("Shared-host exploratory results; CPU governor, NUMA binding and isolation not controlled by this script."), `config` (the whole config), `requested_cpu`. It is written once and NOT rewritten on `--resume` [run_suite.py:44]; this matters for E3 (section 5).

`results.jsonl`: one JSON object per job = the benchmark's JSON plus `dataset`, `variant`, `repeat`, `command` (the full argv) [run_suite.py:68-69].

#### 2.2 summarize.py (75 lines): pairing, ratio, geometric mean, bootstrap, columns

- Pairing key: `(dataset, profile, seed, repeat)`; the script first asserts that every such group has exactly ONE `trace_fingerprint` across variants and raises `unmatched trace fingerprints across variants` otherwise [summarize.py:28-31].
- Ratio: for every run of a variant, `throughput_ops_s / throughput_ops_s of the baseline run with the same (dataset, profile, seed, repeat)` [summarize.py:38-40]. Baseline: `--baseline` (default `raw_rank`; the pipeline and all AIDB summaries used `packed_rank`, except the granularity summaries, which used `packed_rank_r4096/r16384/r32768`) [aidb_pipeline.py:531; run_chain.sh].
- Estimate: per seed, the mean of `log(ratio)` over repeats; then `exp(mean over seeds)` = the geometric mean of per-seed ratios [summarize.py:10-15].
- Bootstrap: `random.Random(42)`, 2000 resamples of the per-seed log-means WITH replacement (`rng.choices(logs, k=len(logs))`), `exp(mean)` of each; interval = the 2.5% and 97.5% quantiles, with `quantile(xs, q) = sorted(xs)[min(len−1, max(0, ceil(q·len) − 1))]` [summarize.py:8, 16-19]. With fewer than 2 seeds the bounds are NaN. With k = 3 seeds there are only 3³ = 27 equally likely resamples, so the 2.5% quantile is the smallest resample mean, i.e. the smallest per-seed ratio, and the 97.5% quantile is the largest: the interval IS the per-seed range, not a 95% confidence interval. Worked example (E3, fb_uniform, `packed_rank_root_vf4` vs `packed_rank`): per-seed ratios 1.0330, 1.0214, 1.1116 → geometric mean 1.0546, bootstrap [1.0214, 1.1116] [computed by results/aidb/guide_deep/scratch/protocol_examples.py from results/aidb_final/sweep/results.jsonl].
- Columns of `summary.csv` [summarize.py:41-59], all medians over the variant's runs unless stated: `dataset, profile, variant, runs, all_verified` (all rows had `verified`; false in every AIDB sweep because `--verify 0`), `throughput_ops_s_median`, `read_hit_p99_ns_median`, `insert_p99_ns_median`, `scan_p99_ns_median` (0 here: no latency pass), `initial_bytes_per_key` = accounted_bytes/initial_rows, `final_bytes_per_key`, `key_arena_ratio` = key_bytes/(8·initial_rows), `rewrite_bytes_per_operation`, `decode_work_per_operation` = decoded_keys/ops, `fence_probes_per_operation`, `root_probes_per_operation`, `root_model_fraction` (median of 1/0 per run of `learnability.root_model`), `correction_distance_per_operation`, `build_ns_per_key`, `flow_region_fraction` = flow_regions/regions, `virtual_points_per_key` = virtual_points/keys, `rank_sse_ratio` = rank_sse_after/rank_sse_before (1.0 when before = 0), `preprocess_ns_per_key` = (smoothing_ns + transform_ns)/keys, `fusion_both_fraction` = choices.both/regions, `fusion_flow_fraction` = (flow + both)/regions, `fusion_vp_fraction` = (vp + both)/regions, `est_probes_none`, `est_probes_selected` (auto-mode cost means), `baseline`, `paired_throughput_speedup`, `seed_bootstrap_low`, `seed_bootstrap_high`.

---

### 3. `tools/aidb_pipeline.py` (649 lines): the nine steps and their fixed parameters

Steps, in the enforced order [aidb_pipeline.py:35]: `verify-downloads, sort, sample, flows, hardness, sweep, throughput, scores, report`. Each checks its outputs first (`--force` redoes). Defaults [aidb_pipeline.py:600-640]: `--results results/aidb`, `--data-dir data/external/gre`, `--samples-dir data/samples`, `--sample 2000000`, `--sample-seed 42`, `--sample-mode uniform` (choices uniform, window, strided), `--ops 1000000`, `--warmup 200000`, `--seeds 11,29,47`, `--alpha 0.1`, `--region-keys 4096`, `--flow-sample 4096`, `--flow-steps 200`, `--binary-dir <repo>/build/experimental/scaleli`, `--jobs 1`, `--timeout 14400` s per subprocess, `--extra-variants` (only group `fusion`), `--build-threads` (omitted from the config unless given), `--wait/--poll-seconds 60`, `--dry-run`, `--steps`.

1. **verify-downloads** [aidb_pipeline.py:325-346]: the pipeline never downloads. For each dataset it requires `<data-dir>/<name>` with size `8 + 8·count`, compares the header count to the catalog count (200,000,000 for every GRE entry in configs/datasets.json), computes sha256 of the DOWNLOAD (never of the sorted copy) and writes `provenance.json[name] = {url, source, path, format, dtype, count, bytes, sha256, mtime_ns, retrieved_utc, verified_utc, role}`. Ran 15:17:17–15:17:27 UTC on 2026-09-21 (10.2 s) [results/aidb/pipeline.log].
2. **sort** [aidb_pipeline.py:352-381]: `scaleli_hardness --data <raw> --dtype uint64 --check-sorted 0 --sort-only 1 --write-sorted <name>.sorted`; the audit JSON (`sorted`, `duplicates`, `written`, `written_keys`, `sort_ns`, `threads`) goes to `results/aidb/sort/<name>.json` and into `provenance[name].sort_audit`, plus the sorted copy's own sha256/bytes/mtime. Result: books, fb, osm sorted with 0 duplicates; covid, genome, history, libio, planet, stack, wise unsorted with 0 duplicates → seven `.sorted` copies of 1,600,000,008 bytes each, ≈ 21 s per file, step total 151.5 s [results/aidb/pipeline.log 15:17:27–15:19:58 UTC]. Every later step uses `dataset_path(name)` = the `.sorted` copy when it exists, else the download [aidb_pipeline.py:278-283].
3. **sample** [aidb_pipeline.py:391-407]: `python3 tools/datasets.py sample <src> <dest> --n 2000000 --dtype uint64 --mode <mode> --seed 42`, dest = `data/samples/<name>_2M_<mode>_s42`, plus a `.manifest.json` (`source, source_count, source_sha256, mode, seed, sample_count, dtype, sample_sha256, warning`). Sampling rule [datasets.py:73-89]: `rng = random.Random(seed)`; `uniform`: `indices = sorted(rng.sample(range(total), n))` (2M of the 200M positions without replacement, kept in file order, hence sorted because the source is sorted); `window`: `start = rng.randrange(total − n + 1)` and `indices = range(start, start + n)`; `strided`: `(2i+1)·total/(2n)`. Because seed and total are the same for all ten datasets, every window sample is the SAME rank interval [171,644,825, 173,644,825) of its file (recomputed: `random.Random(42).randrange(198000001) = 171644825`; checked on disk: fb window first key 66,478,271,979 = fb[171,644,825], last 67,251,951,970 = fb[173,644,824]; planet 5,721,028,025 … 5,986,820,672 [guide_deep/scratch/protocol_examples.py; data/external/gre/fb, planet.sorted]). A sample is redrawn when the manifest's source path or sha256 no longer matches provenance, when size/mode/seed changed, or when the keys are not strictly ascending [aidb_pipeline.py:382-390]. Uniform samples were drawn 15:20:03–15:20:39 UTC (books/fb/osm earlier at 17:09 local from the sorted originals; the other seven after the sort step), window samples 16:11:17–16:11:32 UTC [pipeline.log of both result dirs].
4. **flows** [aidb_pipeline.py:409-421]: `python3 tools/train_flow.py <sample> --dtype uint64 --output results/<lane>/flows/<name>_2D2H2L.txt --sample 4096 --steps 200 --monotone` (exact command recorded in `<name>_training.json.command.json`). train_flow defaults that were NOT overridden: `--shifts 64`, `--lr 0.05`, `--barrier 1e-3`, `--seed 1000000007` [train_flow.py:60-64]. `--sample 4096` = 4,096 training keys uniformly sampled from the 2M-key sample; `--monotone` zeroes the fractional-feature weights (a stated deviation from NFL) [train_flow.py:65]. Output report per dataset: `keys, training_keys, steps, seed, shifts, monotone, best_nll, train_seconds, unordered_transformed_pairs, tail_conflict_degree_raw/transformed`; e.g. fb uniform: nll 4.1820, 2.39 s, D₉₉ 8 → 8 [results/aidb/flows/fb_training.json]. Whole step 71.5 s (uniform), 69.2 s (window). The flow is trained on the same keys it is later applied to (in-domain), which prepare_flows.py states must be reported [prepare_flows.py:7-9].
5. **hardness** [aidb_pipeline.py:423-503]: three `scaleli_hardness` jobs per dataset, common options `--dtype uint64 --region-keys 4096 --pla-eps 32,4096 --check-sorted 1`: (a) `full`: `--data <full file> --flow <flow>` → blocks `original` and `transformed`; (b) `sample_flow`: `--data <sample> --flow <flow> --virtual-alpha 0.1` → `original`, `transformed`, `smoothed` (smoothing applied to the TRANSFORMED features); (c) `sample_csv`: `--data <sample> --virtual-alpha 0.1` → `original`, `smoothed` (raw features). These map to the six scopes of `hardness.json`: `full`, `full_flow`, `sample`, `sample_flow`, `sample_flow_csv`, `sample_csv`, each with `rmse, max_error, conflict_degree, pla_32, pla_4096` (+ `keys`, and `duplicates/unordered_pairs` for flow scopes, `virtual_points/regions` for csv scopes) [aidb_pipeline.py:470-475]. `--jobs 4` ran four processes concurrently in the actual runs [prep_after_sort.sh]. Uniform lane: 132.3 s for all 30 jobs (15:21:50–15:24:03 UTC); window lane 105.1 s (the full-file jobs were reused from cache) [pipeline.log]. Details and timings in `hardness_details.json` (e.g. fb full 17.37 s elapsed incl. 0.81 s flow transform; fb sample_csv smoothing 11.7 s for 199,707 virtual points over 489 regions) [results/aidb/hardness_details.json].
6. **sweep** [aidb_pipeline.py:505-531]: writes the generated config to `configs/aidb.json` AND `results/<lane>/sweep/config.json` (byte-identical, checked with `cmp`), plus `identity.json` = {config, scaleli_bench sha256, path}. Config [aidb_pipeline.py:488-496]: `common = {load-ratio 1, miss 0, query-distribution uniform, ops, warmup, verify 0, instrument 1, latency 0, region-keys 4096, [build-threads]}`; `datasets = [{name, data: data/samples/<name>_2M_<mode>_s42, format sosd, dtype uint64, flow_weights, flow_train_args ["--monotone"]}]`; `profiles ["read_only"]`; `variants` = the eight contracted ones (+ the two fusion ones with `--extra-variants fusion`); `seeds [11,29,47]`; `repeats 1`; no `schedule_seed` (so 137). Then `run_suite.py --binary <bench> --config configs/aidb.json --output results/<lane>/sweep --timeout 14400 --resume`, then `summarize.py … --baseline packed_rank --output sweep/summary.csv`, then `sweep_report.py`. A pre-existing incomplete `results.jsonl` is renamed `results.partial-<timestamp>.jsonl` and the sweep restarts from zero (with `--resume` finding nothing) [aidb_pipeline.py:525-527]; a complete sweep with a different config or binary sha256 refuses to run without `--force`.
7. **throughput** [aidb_pipeline.py:533-541]: `throughput.json = {variant: {dataset: median over seeds of throughput_ops_s/1e6}}` after re-checking trace pairing; `throughput_detail.json` keeps every seed.
8. **scores** [aidb_pipeline.py:543-556]: `python3 tools/aidb_scores.py --hardness hardness.json --throughput throughput.json --output scores.json`; one block per scope with the 25 metrics (5 scalar, 10 two-dim, 10 three-dim) and per-variant conformance (section 5, E6).
9. **report** [aidb_pipeline.py:558-566]: `python3 tools/aidb_report.py --results <lane> --output report.html`; `--steps report` alone always regenerates.

`run.json` [aidb_pipeline.py:569-580] records the LAST pipeline invocation only: label, started/finished UTC, steps, datasets, settings (sample 2000000, sample_seed 42, sample_mode, ops 1000000, warmup 200000, seeds [11,29,47], alpha 0.1, region_keys 4096, flow_sample 4096, flow_steps 200, jobs, force, build_threads, extra_variants), variants, paths (binary_dir), sha256 of both binaries, and the scale note ("paper protocol is 200M keys, 20M warm-up and 100M measured lookups per run"). Caveat: the current `results/aidb/run.json` and `results/aidb_window/run.json` were written by `--steps report --force` invocations from `assemble.sh` at 21:49:40 UTC, so they say `extra_variants: []` and `build_threads: null` although the sweeps they sit next to were produced with `--extra-variants fusion --build-threads 16` (the `environment.json` config is authoritative for that) [results/aidb/run.json; results/aidb/sweep/environment.json; results/aidb/run_chain.sh].

`--extra-variants fusion` [aidb_pipeline.py:47-51] adds `packed_rank_fusion_auto` and `packed_rank_flow_costsel` and first checks that `scaleli_bench --help` mentions `--fusion`. `--sample-mode window` changes only the sample file names, the flow files and the results directory (`--results results/aidb_window`).

---

### 4. Machine, builds, QoS and the noise floor

**Machine** (measured now with `sysctl`, since `environment.json` records `cpu: "unavailable"` because `lscpu` does not exist on macOS): Apple M3 Max, 16 cores = 12 performance + 4 efficiency, 68,719,476,736 B (64 GiB) RAM, macOS 15.7.5 (24G624); `environment.json` of every sweep: `platform macOS-15.7.5-arm64-arm-64bit-Mach-O`, `machine arm64`, Python 3.14.6, `Apple clang version 17.0.0 (clang-1700.0.13.5)`, `affinity null`, `git_commit HEAD`, and the warning that governor/NUMA/isolation are uncontrolled [results/*/sweep/environment.json]. Compare the paper's testbed: two Intel Xeon Gold 5118 (12 cores/socket, 2.3 GHz), 384 GiB, Ubuntu 20.04.6, GCC 9.4.0 `-O3`, worker thread pinned to one core [AIDB §4.1, PDF page 4]. Our runs were NOT pinned; the only control is the QoS request.

**Builds** (all `CMAKE_BUILD_TYPE Release`, `CMAKE_CXX_FLAGS_RELEASE -O3 -DNDEBUG`, `/usr/bin/c++`, no `-march=native` because `SCALELI_NATIVE` is OFF by default [CMakeLists.txt:4,12-13; build-*/CMakeCache.txt]):

| build dir | scaleli_bench sha256 (prefix) | built (local time) | options in `--help` | used by |
|---|---|---|---|---|
| `build-fusion` | 3cc4fa88fedc1fd6… | 2026-09-21 17:18:13 | 18 lines; has `--fusion`, `--build-threads`, `--latency`; NO `--qos`, `--root`, `--root-alpha`; no QoS symbol linked (`nm -u` finds none) | E1 hardness (`scaleli_hardness` 1a414ce4…), E2 uniform + window sweeps, E4 granularity |
| `build-final` | at first 837daca73c112453… (environment.json of E3, built ≈ 19:14 local); now c63ceab046fe50b2… (rebuilt 20:45:24 after the root fix) | 21 lines; adds `--qos`, `--root`, `--root-alpha`; QoS symbol linked | E3: non-root rows from the first binary, root rows from the rebuilt one |
| `build-root` | c63ceab046fe50b2… (identical bytes to the rebuilt build-final), built 20:45:27 | same 21 lines | E5 full scale |

What differs between them is the source at build time: build-fusion predates the learned root and the QoS request; build-final/build-root include them and the root-fit fix (fit on the regions' first real keys instead of region 0's sentinel fence 0 [index.hpp:366-369]). `[unverified: the exact pre-fix build-final source is gone (the directory was rebuilt in place); only its sha256 837daca7… survives in results/aidb_final/sweep/environment.json]`.

**QoS** [benchmark.cpp:158-163]: the request is made unconditionally at the start of `main` unless `--qos 0`, but only by binaries compiled from a source that has it. So E2 and E4 (build-fusion) ran with the DEFAULT QoS class and no `--qos` option in their commands; E3 and E5 ran with `--qos 1` explicit and the call linked. The help text is explicit that this is "Not a pinning guarantee".

**Noise floor, three ways to see it (all from the result files):**

1. *Same structure, same trace, two variants.* In E3, `packed_rank_root_raw` falls back to binary search on books_uniform, osm_uniform, osm_window and planet_uniform (`root_model false`), so those runs execute exactly the same code path as `packed_rank` on the same trace; their work counters are byte-identical. The 12 throughput ratios range 0.824–1.029 (geometric means per dataset 0.903–0.948) [protocol_examples.py; results/aidb_final/sweep/results.jsonl]. The MEETING_NOTES figure "up to 8% (10% for the binary-fallback root)" is the geometric-mean view; per seed the deviation reaches 17.6%. Note `packed_rank_root_fusion` is NOT structurally identical to `packed_rank_root_vf4` even when the root chooses the same candidate, because the flow file also switches on the per-region bypass rule (flow_regions 1–43 of 489, transform_calls > 0), so that pair is not a clean repeat except on stack_window (3 pairs, 0.894–1.055).
2. *Same variant, different query seeds.* `packed_rank` max/min throughput over the three seeds: 1.005 (planet_uniform) to 1.221 (books_uniform), 13 of 20 samples above 1.06 [protocol_examples.py]. This is the "per seed up to 13%" spread of the notes (here it reaches 22% on books_uniform, seed spread mixes noise with genuine trace differences).
3. *The interrupted first uniform sweep.* `results/aidb/sweep/results.partial-20260921T183803.jsonl` holds 254 rows produced 15:24–16:10 UTC with the SAME commands as the final 300 rows, but concurrently with the window pipeline (four hardness processes, a sort) on the same host: identical counters, throughput ratios final/partial from 0.388 to 62.2, median 0.87. It is unusable as a repeat and shows why the sweeps were serialized afterwards (`run_chain.sh` runs one thing at a time). The 19-row window partial (`build-threads 1`, a first attempt) ranges 0.715–1.495.

Conclusion used throughout the guide: per-dataset throughput effects below ≈ 6% on the geometric mean, and ≈ 15–20% on single runs, are not resolvable on this host. Probe counts, memory and build decisions are deterministic and are the primary evidence.

---

### 5. Experiment cards

#### E1 — Hardness metrics (full files and samples)

- **Purpose.** RMSE, ME, CD, PLA-32, PLA-4096 of the ten GRE datasets at full scale (the paper's scope), of the 2M samples, and of the same keys after our flow and after CSV-style virtual points, to place our controls on the paper's hardness axes.
- **Binary.** `scaleli_hardness` from build-fusion (sha256 1a414ce4…) [results/aidb/run.json binaries].
- **Inputs.** The ten downloads (200,000,000 keys, 1,600,000,008 bytes each; sha256 in provenance.json; URL `https://www.cse.cuhk.edu.hk/mlsys/gre/<name>`), the seven `.sorted` copies, the 2M samples, the flows.
- **Scopes and commands.** See section 3 step 5: `full` (`--flow`), `sample_flow` (`--flow --virtual-alpha 0.1`), `sample_csv` (`--virtual-alpha 0.1`), all with `--region-keys 4096 --pla-eps 32,4096 --check-sorted 1`. Window lane: same three jobs on the window samples (the full-file jobs are cache hits).
- **When / how long.** Uniform lane 15:21:50–15:24:03 UTC 2026-09-21 (132.3 s with `--jobs 4`); window lane 16:12:41–16:14:26 UTC (105.1 s) [pipeline.log of each lane]. Per file, e.g. fb full: 17.37 s elapsed, 14.35 s real / 30.88 s user in the timing log [hardness_details.json; results/aidb/hardness_timing/fb_full.log].
- **Outputs.** `results/aidb/hardness/<name>_{full,sample_flow,sample_csv}.json` (+ `.command.json` with the sha256 of the binary), `hardness.json` (6 scopes × 10 datasets × 5 metrics), `hardness_details.json` (FMCD slope/intercept/capacity/D/U_T, timings, virtual point counts: 184,990–199,707 per sample over 489 regions), the same under `results/aidb_window/`.
- **Rows.** 60 metric blocks per lane.
- **Caveats.** CD on transformed/smoothed features uses U_T + 1e-6·(mean gap) instead of LIPP's literal 1e-6 (documented; `fmcd.conflict_degree_lipp_epsilon` gives the literal value) [hardness --help; hardness_details cd_note]. CD carries ±1 rounding uncertainty. Flow scopes can be degenerate when the flow collapses keys onto equal doubles (the pipeline warns when duplicates > 50% of keys) [aidb_pipeline.py:495-497]. The sample scopes describe 1%-samples, not the corpus (manifest warning).

#### E2 — Region-level sweeps on the uniform and window samples

- **Purpose.** Paired throughput, probes and memory of the eight contracted variants plus the two fusion controls on the ten 2M samples: does the NFL-style transform or CSV-style smoothing inside a 4,096-key region reduce work or time?
- **Config.** `configs/aidb.json` = `results/aidb/sweep/config.json` (uniform) and `results/aidb_window/sweep/config.json` (window), generated by the pipeline with `--jobs 4 --build-threads 16 --extra-variants fusion` [run_chain.sh]. `common`: `--load-ratio 1 --miss 0 --query-distribution uniform --ops 1000000 --warmup 200000 --verify 0 --instrument 1 --latency 0 --region-keys 4096 --build-threads 16`.
- **Datasets/samples.** `data/samples/<name>_2M_uniform_s42` with `results/aidb/flows/<name>_2D2H2L.txt`; `data/samples/<name>_2M_window_s42` with `results/aidb_window/flows/<name>_2D2H2L.txt`.
- **Variants** (exact expansion beyond the common options, `--profile read_only --seed S --data … --format sosd --dtype uint64`): `sorted_vector` → `--index sorted_vector`; `raw_rank` → `--policy raw --routing rank`; `packed_rank` → `--policy min_bytes --routing rank`; `packed_rank_flow` → `--policy min_bytes --routing rank --flow <abs path> --flow-bypass 1`; `packed_rank_flow_forced` → `… --flow <abs> --flow-bypass 0`; `packed_rank_vp10` → `… --virtual-alpha 0.1`; `packed_rank_flow_vp10` → `… --flow <abs> --flow-bypass 1 --virtual-alpha 0.1`; `packed_byte` → `--policy min_bytes --routing byte`; `packed_rank_fusion_auto` → `… --flow <abs> --virtual-alpha 0.1 --fusion auto`; `packed_rank_flow_costsel` → `… --flow <abs> --fusion auto`.
- **Seeds / ops / warm-up.** 11, 29, 47; 1,000,000 lookups after 200,000 warm-up reads; 2,000,000 keys loaded.
- **Binary.** build-fusion scaleli_bench 3cc4fa88… (no QoS request possible, see section 4).
- **When / how long.** Window sweep: 16:18:08–16:38:03 UTC (1,194.7 s for 300 runs); uniform clean rerun: 16:38:03–17:01:05 UTC (1,382.2 s for 300 runs) [pipeline.log; environment.json timestamps]. Local time = UTC + 2 (run_chain.log: WINDOW 18:16:19–18:38:03, UNIFORM 18:38:03–19:01:06).
- **Outputs.** `results/aidb{,_window}/sweep/{results.jsonl, summary.csv, environment.json, identity.json, config.json, sweep_report.html}`, `throughput.json`, `throughput_detail.json`, `scores.json`, `report.html`, `chart_data.json`, `fusion_report.html` (both lanes).
- **Rows.** 300 + 300 (10 datasets × 10 variants × 3 seeds). Same trace fingerprint for all ten variants of a (dataset, seed).
- **Caveats.** The uniform sweep was run three times: an 8-variant attempt at 15:24 UTC (contaminated by concurrent work, archived as `results.partial-20260921T183803.jsonl`, 254 rows), a refused attempt at 16:10 ("holds a sweep with different settings"), and the clean 10-variant run. The window lane had a 19-row partial with `build-threads 1` (`results.partial-20260921T181808.jsonl`). `run.json` in both lanes reflects a later `--steps report` call (section 3).

#### E3 — Final sweep (QoS, 5M lookups, root ablation) — `results/aidb_final`

- **Purpose.** Repeat the region controls with more lookups and the performance-core QoS request, and add the learned-root ablation (raw root, flow root, virtual fences, fusion, region vp + root fusion) on all twenty samples.
- **Config.** `results/aidb_final/config.json`: `common` = `--load-ratio 1 --miss 0 --query-distribution uniform --ops 5000000 --warmup 500000 --verify 0 --instrument 1 --latency 0 --build-threads 16 --qos 1` (note: no `--region-keys`, so the default 4096); 20 datasets (`<name>_uniform` and `<name>_window`, flows from the matching lane); 9 variants; seeds 11, 29, 47; repeats 1.
- **Variants** (expansion beyond common/profile/seed/data): `sorted_vector`, `raw_rank`, `packed_rank`, `packed_rank_vp10` as in E2; `packed_rank_root_raw` → `--policy min_bytes --routing rank --root model`; `packed_rank_root_flow` → `… --root model --flow <abs>`; `packed_rank_root_vf4` → `… --root model --root-alpha 4`; `packed_rank_root_fusion` → `… --root model --root-alpha 4 --flow <abs>`; `packed_rank_vp10_root_fusion` → `… --virtual-alpha 0.1 --root model --root-alpha 4 --flow <abs>`.
- **Binary.** build-final. `environment.json` records sha256 837daca7… (the pre-fix build); the 300 root-variant rows were re-run with the rebuilt build-final (sha256 c63ceab0…, 20:45:24 local); the command path is the same string for all 540 rows.
- **Timeline (local).** 19:14:38 an 800-job attempt (10 variants incl. `packed_rank_fusion_auto`, 4 seeds incl. 61) wrote `environment.json` and a handful of rows, then was stopped; 19:16:43–20:11:18 `run.sh` ran the 9-variant × 3-seed grid with `--resume` ("[540/540]"); after the root-fit fix the root rows (300) plus the 7 stray rows (6 seed-61 runs + 1 `packed_rank_fusion_auto`) were moved to `results.stale-root-before-fix.jsonl` (307 rows, mtime 20:46:02) and the 300 root jobs re-run by `run_chain2.sh` 20:46:02–21:17:20 [results/aidb_final/run.sh, run.log; run_chain2.log; verdicts.json entry 11 for the 547-row inventory before the move].
- **What the fix changed.** Pre-fix `fit_root` fitted the raw root through region 0's `low_fence = 0` sentinel [index.hpp:432]; for a contiguous window whose first key is far from 0 the OLS line predicts ≈ rank 244 for every key, so the raw estimate was 14.1–14.4 on all windows and only the flow+fences candidate beat binary (the "synergy" that the synergy verifier refuted, verdicts.json entry 14). Post-fix the root is fitted on the regions' first real keys [index.hpp:366-370]. Example fb_window seed 11: raw estimate 14.34 → 3.58; `packed_rank_root_vf4` root probes 8.96 (fallback) → 2.27 with `root_vp true`; `packed_rank_root_fusion` 2.88 with `root_flow true` → 2.27 with `root_flow false` [results.stale-root-before-fix.jsonl vs results.jsonl].
- **Outputs.** `sweep/results.jsonl` (540 rows), `sweep/summary.csv` (baseline packed_rank), `root_analysis.json` (from `results/aidb/root_analysis.py`), `run.log`.
- **Caveats.** `assemble.sh` labels this lane "5 seeds": wrong, it is 3 seeds (verifier correction, verdicts.json entry 11). The 240 non-root rows come from the pre-fix binary; the non-root code paths are unaffected by the fix `[unverified: not re-run after the rebuild]`. `environment.json` describes the 800-job attempt, not the grid that was run.

#### E4 — Granularity ablation — `results/aidb_granularity`

- **Purpose.** Does the transform start to matter when regions grow (4,096 → 16,384 → 32,768 keys), where a single per-region linear model can no longer absorb curvature?
- **Config.** `results/aidb_granularity/config.json`: `common` = `--load-ratio 1 --miss 0 --query-distribution uniform --ops 1000000 --warmup 200000 --verify 0 --instrument 1 --latency 0 --build-threads 16`; datasets fb, osm, planet, covid, genome (uniform samples, uniform flows); seeds 11, 29; 15 variants = 3 region sizes × 5 kinds:
  `packed_rank_r{R}` → `--region-keys R --policy min_bytes --routing rank`; `packed_rank_flow_forced_r{R}` → `… --flow <abs> --flow-bypass 0`; `packed_rank_vp10_r{R}` → `… --virtual-alpha 0.1`; `packed_rank_flow_vp10_r{R}` → `… --flow <abs> --flow-bypass 0 --virtual-alpha 0.1` (NOTE: forced flow, unlike E2's `packed_rank_flow_vp10` which uses the bypass); `packed_rank_fusion_auto_r{R}` → `… --flow <abs> --virtual-alpha 0.1 --fusion auto`; for R ∈ {4096, 16384, 32768}.
- **Binary.** build-fusion 3cc4fa88… (no QoS).
- **Timeline (local).** Started 19:01:06 (`run_chain.sh`, logged "GRANULARITY DONE" 19:14:33 but the grid was not complete), resumed in `run_chain2.sh` (after 21:17, interrupted) and `run_chain3.sh` 22:50:20–23:49:39, ending with 150 rows [run_chain*.log].
- **Outputs.** `sweep/results.jsonl` (150 rows = 5 × 15 × 2), `summary_r4096.csv`, `summary_r16384.csv`, `summary_r32768.csv` (each with baseline `packed_rank_r<R>`, so ratios are within a region size).
- **Caveats.** Only two seeds, so the bootstrap interval is the two-seed range. Regions per sample: 489 / 123 / 62 (2M/R rounded up).

#### E5 — Full scale (200M keys) — `results/aidb_fullscale`

- **Purpose.** The paper's key count on two datasets: does the learned root and the region smoothing behave at 48,829 regions as at 489?
- **Configs.** Planned `config.json`: fb (`data/external/gre/fb`) and planet (`planet.sorted`), 6 variants (`packed_rank`, `packed_rank_root_raw`, `packed_rank_root_flow`, `packed_rank_root_vf4`, `packed_rank_root_fusion`, `packed_rank_vp10_root_fusion`), seed 11, `common` = `--load-ratio 1 --miss 0 --query-distribution uniform --ops 2000000 --warmup 200000 --verify 0 --instrument 1 --latency 0 --build-threads 16 --qos 1` (12 jobs). Split because the planet root smoothing at α_root = 4 is 195,316 greedy rounds over 48,828 fences (Algorithm 1 is O(budget × fences)) and did not finish in the 2 h timeout of `run_chain2.sh`: `config_a.json` (both datasets: `packed_rank`, `packed_rank_root_raw`, `packed_rank_root_flow`, `packed_rank_root_vf004` → `--root model --root-alpha 0.04`, `packed_rank_vp10_root_vf004` → `--virtual-alpha 0.1 --root model --root-alpha 0.04`; 0.04 × 48,829 = 1,953 fences, the absolute count the 2M runs spent at α = 4), `config_b.json` (fb only at α = 4: `root_vf4`, `root_fusion`, `vp10_root_fusion`), `config_c1/c2/c3.json` (planet at α = 4, one variant each, 4 h cap).
- **Binary.** build-root c63ceab0… (QoS linked, `--qos 1`).
- **Timeline (local).** `run_chain2.sh` 21:17:20 started the 12-job plan and wrote 2 rows (planet root_flow, fb root_fusion) before being stopped; `run_chain3.sh` 22:21:05 config_a → 22:42:17, config_b → 22:50:20 (13 rows), c1 23:49:40 → 00:59:37 (planet `root_vf4`: build 2,067.8 s, ×2 passes), c2 (planet `root_fusion`) started 00:59:37 and was still running when this section was written (PID 31890, elapsed 47:57 at 01:47 local); c3 not started [run_chain2.log, run_chain3.log, `ps`].
- **Outputs.** `sweep/results.jsonl` (14 rows), `sweep/summary.csv`, `root_analysis.json/.txt` (written 22:50:20, 13 rows; recompute with `python3 results/aidb/root_analysis.py results/aidb_fullscale/sweep/results.jsonl` to include the 14th).
- **Key numbers for the cards.** 48,829 regions; binary root 15.658 probes/lookup (emulated 15.6579); fb raw root 10.81, fences 3.42 with 973 virtual fences (greedy stops at 973 under both the 1,953 and the 195,316 budget), throughput 0.648 (binary) / 0.611 / 0.812 / 0.723 Mops for control / raw / fences / fusion, i.e. the same fences structure measured three times gave 0.65, 0.72, 0.81; planet: raw estimate 25.8 > 15.66 → fallback, 1,950-fence root estimate 25.2 → fallback, 195,316-fence root 13.75 estimate / 13.69 measured; region vp10 adds 161–190 s of build and 0.8 B/key [results/aidb_fullscale/sweep/results.jsonl].
- **Caveats.** Single seed; no throughput claim at 200M; `build_ns` is wall clock of a 16-thread build while `smoothing_ns` (2.4–2.9 × 10¹² ns for region vp) is summed thread time; each job builds twice (pass 1 + pass 4).

#### E6 — Conformance/coverage scores — `results/aidb/scores.json`, `results/aidb_window/scores.json`

- **Purpose.** Apply the paper's eq. (1)–(3) to OUR variants: for each of the 6 hardness scopes and 25 metrics, coverage (fraction of comparable dataset pairs under the metric's partial order) and conformance (weighted agreement between "harder" and "lower throughput").
- **Inputs.** `hardness.json` (E1) and `throughput.json` (E2 medians in MOPS per variant × dataset). Command: `python3 tools/aidb_scores.py --hardness results/aidb/hardness.json --throughput results/aidb/throughput.json --output results/aidb/scores.json` [aidb_pipeline.py:547]. Defaults: `--ddof 0` (population std), `--min-datasets 3`.
- **Structure.** `scores.json[scope] = {metrics: {name: {dims, coverage, conformance, per_variant, comparable_pairs, incomparable_pairs, per_variant_detail, pairs}}, datasets, variants, skipped, dropped_datasets, ddof, normalized_throughput, hardness, note}`; 10 variants, 10 datasets, nothing skipped [results/aidb/scores.json]. Example: scope `sample`, metric PLA-32: coverage 1.0 (45/45 pairs comparable), conformance −0.397, per variant from −0.554 (raw_rank) to −0.194 (sorted_vector).
- **When.** 17:01:05 UTC (uniform), 16:38:03 UTC (window), 0.1 s each [pipeline.log].
- **Caveats.** Conformance measures whether OUR throughput ordering follows the metric; it says nothing about the paper's indexes. A variant with zero throughput std would be skipped, not scored +1 (fixer round 1, verdicts.json entry 10). The two "GRE" and "CD·PLA-32" lines printed by `walkthrough.py demo 09` come from a separate demo on `results/local`, not from this scores file [assembly1.log].

#### E7 — Verification — `results/aidb/verification/verdicts.json` (16 entries)

Entries 0–4 are the builder reports of contracts C1–C5 (hardness binary, benchmark `--latency`/learnability capture, scorer, report, pipeline); entries 5–9 are the first-round verifiers (findings: unsorted GRE files → the `sort` step; provenance and idempotency gaps; the ten-variant deviation → `--extra-variants`; the ε rule for CD; std = 0 scoring); entry 10 is the fixer round. Entries 11–15 (workflow wf_26569070) verified the root-ablation claims on `results/aidb_final`:

| verifier (dir) | recomputed | verdict |
|---|---|---|
| recompute (`recompute/recompute.py`) | run counts, pairing (one fingerprint and one checksum per (dataset, seed)), probes per lookup, paired geometric speedups with seed bootstrap, sorted_vector vs every packed variant | pairing confirmed (0 mismatches); "5M × 5 seeds" wording refuted (3 seeds); C3a fences-only +6–12% refuted; C7 sorted_vector 1.15–2.3× confirmed |
| coststory (`coststory/coststory.py`) | `1e9/throughput == elapsed/ops` to 0.0000 ns; transform_calls per lookup = root_flow + flow_regions/regions within 0.0015; counters of root variants identical to packed_rank except root_probes and transform_calls; ns per probe by pooled regression | C4 probe cost needs qualification (≈ 4.0 ns/probe pooled, per-dataset spread is noise); C5 flow charge of 4 uncalibrated confirmed |
| synergy (`synergy/`) | exact Python emulation of `fit_root`'s estimator reproducing every stored estimate; counterfactual fit without the sentinel | C1 "synergy" REFUTED: artifact of region 0's sentinel fence → the fit was changed and E3's root rows re-run |
| claims (`final/claims.py`) + build/ctest/unittest/dry-run logs | fresh Release build, ctest 4/4, 46 unit tests, dry-run pipeline with `--extra-variants fusion`, `--help` option list, a `--verify 1` differential run | T1–T7 confirmed |
| pla (`pla/compare.py`), metrics (`metrics/gen.py`, `vlib.py`), scores (`scores/ref_scores.py`) | PLA segment counts vs a PGM-index reference on fixtures, synthetic sets and 5M-key prefixes; FMCD/CD oracle; an independent implementation of eq. (1)–(3) | agreement (PLA identical for ε ∈ {0,1,8,32,128,1024,4096}; scorer agreement 4e-16 per MEETING_NOTES §1) `[unverified: I did not re-run these scripts]` |

---

### 6. Variant dictionary (every name used anywhere → options → what it tests)

| variant | options beyond the common ones | what it isolates |
|---|---|---|
| `sorted_vector` | `--index sorted_vector` | plain binary search over the sorted uint64 array: the "no index" reference |
| `raw_rank` | `--policy raw --routing rank` | our region/block map with uncompressed blocks and rank routing: removes decoding cost only |
| `packed_rank` | `--policy min_bytes --routing rank` | the packed control (codec chosen per block by min bytes); baseline of every AIDB summary |
| `packed_byte` | `--policy min_bytes --routing byte` | packed with byte routing (E2 only) |
| `packed_rank_flow` | `+ --flow $flow --flow-bypass 1` | NFL-style transform under the per-region bypass rule (kept iff D₉₉ drops ≥ 10%) |
| `packed_rank_flow_forced` | `+ --flow $flow --flow-bypass 0` | transform forced into every region: upper bound of what it could change |
| `packed_rank_vp10` | `+ --virtual-alpha 0.1` | CSV-style virtual points, λ = 0.1 × 4096 ≈ 409 per region |
| `packed_rank_flow_vp10` | `+ --flow $flow --flow-bypass 1 --virtual-alpha 0.1` (E2) / `--flow-bypass 0` (E4) | both mechanisms stacked |
| `packed_rank_fusion_auto` | `+ --flow $flow --virtual-alpha 0.1 --fusion auto` | per-region cost-based choice among none/flow/vp/both by measured probes + 4-probe flow charge |
| `packed_rank_flow_costsel` | `+ --flow $flow --fusion auto` | cost-based choice between none and flow only (no vp) |
| `packed_rank_r{4096,16384,32768}` and the `_flow_forced_`, `_vp10_`, `_flow_vp10_`, `_fusion_auto_` forms | `--region-keys R` + as above | E4 region-size ablation |
| `packed_rank_root_raw` | `+ --root model` | learned root on the raw key feature, ranks as targets; binary fallback if not better |
| `packed_rank_root_flow` | `+ --root model --flow $flow` | root candidates {raw, flow} × ranks (regions also get the bypass flow) |
| `packed_rank_root_vf4` | `+ --root model --root-alpha 4` | root candidates {raw} × {ranks, virtual fences with budget 4 × regions} |
| `packed_rank_root_vf004` | `+ --root model --root-alpha 0.04` | E5 only: budget 0.04 × 48,829 ≈ 1,953 fences |
| `packed_rank_root_fusion` | `+ --root model --root-alpha 4 --flow $flow` | all four root candidates |
| `packed_rank_vp10_root_fusion` | `+ --virtual-alpha 0.1 --root model --root-alpha 4 --flow $flow` | region smoothing plus the full root selector |
| `packed_rank_vp10_root_vf004` | `+ --virtual-alpha 0.1 --root model --root-alpha 0.04` | E5 only |
| older learnability sweep (`results/learnability/summary.csv`): `packed_rank_vp05/vp30`, `packed_rank_vp10_relearn`, `packed_byte_flow` | `--virtual-alpha 0.05/0.3`, `--relearn 1`, `--routing byte --flow` | synthetic distributions, 20k keys, read_only + read_heavy; superseded by the AIDB lanes |

`$flow` always resolves to `results/aidb/flows/<name>_2D2H2L.txt` for uniform samples and `results/aidb_window/flows/<name>_2D2H2L.txt` for window samples (and to the uniform flow for the full files in E5) [config files].

---

### 7. Reproduction commands, in order (from S)

```
## 0. build (each build dir is a plain Release configure of the repo root)
cmake -S ../.. -B ../../build-final -DCMAKE_BUILD_TYPE=Release && cmake --build ../../build-final --parallel 4
##    (equivalent: the actual runs used build-fusion for E1/E2/E4, build-final for E3, build-root for E5;
##     build-fusion cannot be regenerated from the current source because the source has since gained --root/--qos)

## 1. data: place the ten GRE SOSD files under data/external/gre/<name> (the pipeline never downloads)

## 2. E1 + E2 uniform lane (verify-downloads, sort, sample, flows, hardness, sweep, throughput, scores, report)
python3 tools/aidb_pipeline.py --binary-dir ../../build-final/experimental/scaleli --jobs 4 --build-threads 16 --extra-variants fusion
##    equivalent: the actual run was staged (prep_sort.sh -> prep_after_sort.sh -> run_sweeps*.sh -> run_chain.sh)

## 3. E1 + E2 window lane
python3 tools/aidb_pipeline.py --binary-dir ../../build-final/experimental/scaleli --sample-mode window --results results/aidb_window --jobs 4 --build-threads 16 --extra-variants fusion

## 4. E3 final sweep
python3 tools/run_suite.py --binary ../../build-final/experimental/scaleli/scaleli_bench --config results/aidb_final/config.json --output results/aidb_final/sweep --timeout 3600
python3 tools/summarize.py results/aidb_final/sweep/results.jsonl --baseline packed_rank
python3 results/aidb/root_analysis.py results/aidb_final/sweep/results.jsonl --json results/aidb_final/root_analysis.json
##    equivalent: the actual run used --resume twice (section 5, E3)

## 5. E4 granularity
python3 tools/run_suite.py --binary ../../build-final/experimental/scaleli/scaleli_bench --config results/aidb_granularity/config.json --output results/aidb_granularity/sweep --timeout 3600
for r in 4096 16384 32768; do python3 tools/summarize.py results/aidb_granularity/sweep/results.jsonl --baseline packed_rank_r$r --output results/aidb_granularity/summary_r$r.csv; done

## 6. E5 full scale (A, B, then the planet alpha-4 runs with a 4 h cap each)
for c in a b c1 c2 c3; do python3 tools/run_suite.py --binary ../../build-root/experimental/scaleli/scaleli_bench --config results/aidb_fullscale/config_$c.json --output results/aidb_fullscale/sweep --timeout 14400 --resume; done
python3 tools/summarize.py results/aidb_fullscale/sweep/results.jsonl --baseline packed_rank --output results/aidb_fullscale/sweep/summary.csv
python3 results/aidb/root_analysis.py results/aidb_fullscale/sweep/results.jsonl --json results/aidb_fullscale/root_analysis.json > results/aidb_fullscale/root_analysis.txt

## 7. E6 scores are produced by step 2/3; to redo alone:
python3 tools/aidb_scores.py --hardness results/aidb/hardness.json --throughput results/aidb/throughput.json --output results/aidb/scores.json

## 8. cross-lane figures and reports
results/aidb/assemble.sh        # fusion_report.py over all summary.csv files, report.html of both lanes, manifest, walkthrough checks

## 9. worked numbers of this section
python3 results/aidb/guide_deep/scratch/protocol_examples.py
```

Timing budget from the logs: sort 2.5 min, samples + flows ≈ 2 min per lane, hardness ≈ 2 min per lane with 4 jobs, E2 ≈ 20–23 min per lane, E3 ≈ 55 min for 540 runs (+ 31 min for the root re-run), E4 ≈ 1 h 15 min spread over three sessions, E5 ≈ 30 min for A+B, 70 min for planet at α = 4, c2/c3 open.

---

### 8. Questions the supervisor may ask, with answers

1. **How do you know the traces are identical across variants?** The trace depends only on the key set, `--seed`, `--ops` and the profile [workload.hpp:74-137]; the binary prints `trace_fingerprint`, a splitmix64 hash over the loaded records and every operation [benchmark.cpp:54], and `summarize.py` refuses to pair runs whose fingerprints differ within a (dataset, profile, seed, repeat) group [summarize.py:28-31]. The recompute verifier found exactly one fingerprint AND one `result_checksum` per (dataset, seed) over all 540 final rows [verdicts.json entry 11].

2. **Why geometric means of ratios?** Throughput ratios are multiplicative; the geometric mean is symmetric under inverting the baseline (a 1.10 and a 0.909 average to 1.0, not 1.005), and averaging log-ratios per seed first gives each trace equal weight [summarize.py:10-15].

3. **What is the noise floor and how did you measure it?** From the result files, not assumed: identical code path and trace (`packed_rank_root_raw` in binary fallback vs `packed_rank`) gives 12 ratios in 0.824–1.029; the same variant across the three query seeds varies by up to 1.22×; the interrupted uniform sweep that overlapped other work varied up to 62× (section 4). Hence throughput differences under ≈ 6% on geometric means are not resolvable; probes, memory and selector decisions are deterministic and are the primary evidence.

4. **Why 5M lookups in the final sweep (and 1M before)?** The paper measures 100M lookups after 20M warm-up on 200M keys [AIDB §4.1]; on a 2M-key sample 1M lookups already touch every key ≈ 0.5 times and take 0.1–0.3 s, which is short enough for scheduler effects to dominate. 5M with 500k warm-up lengthens each timed loop to ≈ 1–2 s while keeping 540 runs under an hour; it is a compromise, not the paper's protocol, and the noise floor did not shrink much (section 4).

5. **Where does 8.96 root probes come from?** The fence binary search [index.hpp:352-353] counts one probe per loop iteration; 2,000,000/4,096 → 489 regions (last one 1,152 keys), so most lookups need 9 iterations (ceil(log₂ 489)) and a few 8; the size-weighted expectation is 8.9564, matching the measured 8.96 to the printed digits; at 200M keys 48,829 regions give 15.6579 vs measured 15.6578739875 [protocol_examples.py; results/aidb_fullscale/sweep/results.jsonl].

6. **Why load-ratio 1, miss 0, uniform queries?** To mirror the paper's read-only protocol ("bulk load the index and then perform random lookups for all keys") [AIDB §4.1]. No inserts means no compactions, so `relearn` and delta options are irrelevant and the counters are pure lookup work.

7. **What exactly is timed?** Only the loop over the trace in pass 1, with a monotonic `steady_clock` before and after, no per-operation timer and no counters [benchmark.cpp:80]; build, warm-up, memory accounting and the drain are outside it. Latency percentiles were disabled (`--latency 0`) so the only timing in every AIDB row is `elapsed_ns` (the cost-story verifier checked `1e9/throughput == elapsed/ops` to 0.0000 ns).

8. **Were results verified against a reference in the sweeps?** No: `--verify 0` in every sweep for speed. The differential check (std::map reference, every operation, plus final-state scan) was exercised in the unit tests and by the verifier's `--verify 1` run with root + fusion + virtual points [verdicts.json entry 15]; each sweep pass still cross-checks the result checksum between passes and aborts on mismatch [benchmark.cpp:96, 115].

9. **Why are the seven `.sorted` copies legitimate?** GRE serves those files unsorted and its own loader sorts and de-duplicates at load time [verdicts.json entry 6]; our audit found 0 duplicates in every file, so the sorted copy is the same multiset in key order; the copy's sha256 is in `provenance.json[name].sort_audit`, the download's own sha256 is kept separately, and the samples' manifests point at the copy they were drawn from.

10. **Are the window samples comparable across datasets?** They are all the rank interval [171,644,825; 173,644,825) of their file because `datasets.py` draws `start` from `random.Random(42)` with the same `total − n + 1` for every dataset (verified on fb and planet). They preserve local structure but not the global shape; the uniform samples do the opposite (manifest warning).

11. **Is the flow trained on the test keys?** Yes: 4,096 keys sampled from the same 2M sample the index is built from (in-domain), 200 Adam steps with the `--monotone` restriction; `prepare_flows.py` and the training report state this. For the 200M-key runs the 2M-sample flow was applied to the full file (see `flow_weights` in `results/aidb_fullscale/config*.json`).

12. **Why do the E2 runs lack `--qos` and `--root` in their JSON?** They were produced by the build-fusion binary, compiled before those options existed (its `--help` has 18 option lines vs 21, and `nm -u` shows no QoS symbol). Rows from that binary therefore have 62 top-level keys instead of 64. This is also why E2/E4 had no performance-core request while E3/E5 had one.

13. **What would you change to make the timing trustworthy?** Pin the process (not possible on macOS; on Linux `run_suite.py --cpu N` uses `taskset`), run repeats > 1 on a quiet host, use the paper's lookup count, and report the identical-structure spread next to every ratio; the probe counters already provide the deterministic story.

14. **How many runs and how long did the whole thing take?** E2 600 runs (≈ 43 min), E3 540 + 300 re-run (≈ 86 min), E4 150 (≈ 75 min), E5 14 done (≈ 2 h 10 min so far, planet α = 4 root fusion still running), E1 60 hardness jobs (≈ 4 min), all on 2026-09-21/22 local time; every wall-clock number above comes from `pipeline.log` and `run_chain*.log`.

15. **Which single file proves a given number?** `results/<lane>/sweep/results.jsonl` (one row per run, with the full command line), never the HTML; `summary.csv` and `root_analysis.json` are derived by the two 75-line scripts quoted in section 2 and can be regenerated in seconds.


---

## 07 — Results per dataset, with interpretation

**What this section gives you.** Every number the study produced, per dataset and per variant, recomputed from the raw result files with the stdlib script `results/aidb/guide_deep/scratch/compute_tables.py` (tables are pasted from its output, not from prose), and for each experiment an interpretation of the *mechanism* that produced the number. The experiments are E1 (hardness metrics on the ten AIDB datasets, raw and transformed, with and without CSV-style virtual points), E2 (region-level sweeps on 2M-key uniform and window samples), E3 (the final sweep with the root ablation, including the sentinel-fence artefact and the withdrawn "synergy"), E4 (region-size ablation), E5 (200M-key runs on fb and planet), E6 (conformance/coverage of the 25 metric compositions against our variants versus the paper's Table 2), a cost decomposition (ns per probe, ns per flow evaluation, break-even batch size), the throughput picture (paired speedups, noise band, the decoding-vs-routing ablation), a one-page "settled / noisy / open" summary with the exact sentences to say, and a set of hostile questions with answers. Everything here is about clean-room CONTROLS inside the SCALE-LI experimental map: an 8-parameter monotone tanh network in NFL's weight format trained by a stand-in trainer (`tools/train_flow.py`), CSV Algorithm 1 written from the paper (`include/scaleli/smoothing.hpp`), and the paper's five scalar hardness metrics (`include/scaleli/hardness.hpp`). Nothing is a reproduction of NFL, AFLI, CSV or the AIDB benchmark, whose six indexes were not run.

Notation (shared with the other sections): keys k₁ < … < kₙ (uint64), rank r(kᵢ) = i − 1; a linear model f(k) = w·φ(k) + b with φ the normalized raw key or the flow output z(k); slot s(kᵢ) = rank plus the number of virtual points before kᵢ; virtual point set V with budget λ = α·n; tail conflict degree D₉₉; region = 4,096 keys, block = 128 keys [index.hpp:44]; root = the whole-range structure over the 489 region fences of a 2M-key sample (⌈2,000,000/4,096⌉ = 489) or the 48,829 fences of a 200M-key file; probes = key comparisons counted by the software counters `root_probes`, `fence_probes`, `coordinate_probes` [index.hpp:124, :135, :350-353], `correction_distance` [index.hpp:151], `transform_calls`.

---

### 0. How to read a number in this section

#### 0.1 The three probe counters and "total probes"

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

#### 0.2 Paired speedup and the "seed spread" bracket

For variant v on sample d, each query seed σ ∈ {11, 29, 47} replays the identical lookup trace (`trace_fingerprint` equal across variants; verified in [results/aidb/verification/recompute/recompute.json `pairing_ok: true`]) against the control:

```
ratio_σ  = throughput_v(d, σ) / throughput_control(d, σ)
speedup  = exp( mean_σ log ratio_σ )                      (geometric mean over the 3 seeds)
bracket  = 2.5 % and 97.5 % quantiles of 2,000 bootstrap resamples of the 3 log-ratios
           (resampling seeds, RNG 42)                     [root_analysis.py paired(); tools/summarize.py:9]
```

With three seeds the bracket is the *seed spread*, not a 95 % confidence interval; with one seed (E5) there is no bracket at all.

#### 0.3 The samples the region-level experiments run on

Two 2M-key samples per dataset from the sorted 200M-key GRE file [results/aidb/provenance.json; data/samples/*.manifest.json]: `uniform` = 2,000,000 keys drawn uniformly at random (seed 42), which keeps the global shape and flattens local structure; `window` = one contiguous 2,000,000-key range at a seeded offset, which keeps local structure and loses the global shape. The hardness of the two differs a lot: fb PLA-32 is 3,034 on the uniform sample and 10,599 on the window [results/aidb/hardness.json fb.sample.pla_32; results/aidb_window/hardness.json fb.sample.pla_32].

---

### 1. E1 — Hardness moves: what the transform and the virtual points do to the five metrics

Sources: [results/aidb/hardness.json] (uniform run; `full` and `full_flow` are computed on the whole 200M-key file, `sample*` on the uniform 2M sample), [results/aidb_window/hardness.json] (window run; its `full` scope is byte-identical to the uniform run's, its `sample*` scopes are the window sample). Metric definitions: RMSE and ME of one least-squares fit key → rank [hardness.hpp:84]; CD via the FMCD fit [hardness.hpp:168, :218]; PLA-ε = optimal ε-bounded segment count [hardness.hpp:295]; all as in [AIDB §4.1 "Metrics under Test"].

#### 1.1 Full 200M-key files, raw keys

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

#### 1.2 Full 200M-key files, transformed keys z(k) (flow trained on the uniform sample), with % change vs raw

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

#### 1.3 The 2M samples: sample → sample_flow → sample_csv → sample_flow_csv (α = 0.1)

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

### 2. E2 — Region-level sweeps (uniform and window 2M samples)

Protocol [results/aidb/sweep/config.json; results/aidb_window/sweep/config.json]: 10 datasets × 10 variants × 3 query seeds (11, 29, 47) = 300 runs per sample mode; read-only; 2M keys bulk-loaded with 16 build threads; 200,000 warm-up lookups then 1,000,000 measured lookups per run; `--verify 0 --instrument 1 --latency 0` (throughput pass and counter pass only); region 4,096 keys, block 128 keys. Binary sha256 3cc4fa88… [results/aidb/sweep/environment.json]. Variants: `packed_rank` (control: packed codecs, per-region rank model, fence binary-search root), `packed_rank_flow` (`--flow <weights> --flow-bypass 1`: the flow is kept in a region iff D₉₉(z) ≤ 0.9·D₉₉(raw) [index.hpp:281]), `packed_rank_flow_forced` (`--flow-bypass 0`), `packed_rank_vp10` (`--virtual-alpha 0.1`), `packed_rank_flow_vp10` (both, bypass on), `packed_rank_fusion_auto` (`--fusion auto --flow … --virtual-alpha 0.1`: the per-region selector over {raw, flow} × {ranks, slots} [index.hpp:239-264]), `packed_rank_flow_costsel` (`--fusion auto` with the flow only), plus `packed_byte`, `raw_rank`, `sorted_vector`.

#### 2.1 Fence probes per lookup (the correction work after the region model's prediction)

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

#### 2.2 Coordinate probes, correction distance, transform calls

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

#### 2.3 Flow acceptance under the bypass rule, selector choices, tail conflict degrees

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

#### 2.4 Rank SSE, memory, build time, preprocessing

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

#### 2.5 Throughput and paired speedups (region level)

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

#### 2.6 The three region-level readings in one sentence each

1. Virtual points at α = 0.1 cut fence probes per lookup on all 20 samples, by 4.7 % (stack_window) to 24.0 % (covid_uniform); the saving is ≥ 10 % on 16 samples strictly and 9.98 % on a 17th (books_uniform); the two weakest are stack_window (4.7 %) and fb_window (7.7 %) [Tables 2.1].
2. Forcing the transform into every region leaves fence probes equal to the control's to three decimals on all 20 samples (max |Δ| = 0.0006 uniform, 0.0002 window), while adding exactly one transform evaluation per lookup [Tables 2.1, 2.2].
3. The selector picks the flow in 0 of 9,780 regions (20 samples × 489), and virtual points in 96.1-100 % [Tables 2.3].

---

### 3. E3 — Final sweep and the root ablation

Protocol [results/aidb_final/config.json]: 20 samples (10 datasets × {uniform, window}) × 9 variants × 3 query seeds (11, 29, 47) = 540 paired runs; 500,000 warm-up + 5,000,000 measured lookups per run; performance-core QoS (`--qos 1`); binary `build-final`. Variants add the root ablation to the E2 controls: `packed_rank_root_raw` (`--root model`: one global linear model on the regions' first keys, kept only if its estimated probes beat the fence binary search), `packed_rank_root_flow` (`--root model --flow …`: the same with the flow feature as a candidate), `packed_rank_root_vf4` (`--root model --root-alpha 4`: CSV-style virtual fences on the raw feature, budget 4 × 489 = 1,956), `packed_rank_root_fusion` (all four candidates {raw, flow} × {ranks, fences}), `packed_rank_vp10_root_fusion` (region virtual points + that root). The root selector [index.hpp:362-392] scores each candidate by the probes the real root `locate` spends on the fences and on the fence midpoints, adds `flow_cost` = 4 for a flow candidate [index.hpp:392], and falls back to the binary search if no candidate beats it. The sweep ran twice: the first pass 19:16-20:11, then the five root variants again 20:46-21:17 after the sentinel-fence fix with the binary rebuilt at 20:45 (sha256 c63ceab0…; `sweep/environment.json` still records the pre-fix binary 837daca7… because `--resume` keeps the first record) [READING_GUIDE §5 E3; results/aidb_final/sweep/environment.json]. The 307 pre-fix rows (300 root-variant rows + 7 strays from seed-61 and `fusion_auto` attempts [results/aidb/verification/recompute/recompute.json `extra_runs_outside_grid`]) are archived in `results.stale-root-before-fix.jsonl`.

#### 3.1 Per-sample summary (post-fix)

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

#### 3.2 The full ablation table (post-fix; medians over the 3 seeds; `est` = the selector's own estimates for binary / raw+ranks / flow+ranks / raw+fences / flow+fences, flow candidates include the +4 charge)

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

#### 3.3 Readings

- **Raw + virtual fences is chosen on 20 of 20 samples** (`chosen` column of `root fences` and `root fusion`) and is the same structure whether the flow is offered or not: `root fusion` and `root fences` have identical root probes on every sample. The flow feature is chosen in **0 of 60 candidate cells** (20 samples × {root_flow, root_fusion, vp10_root_fusion}; every run's `learnability.root_flow` is false [results/aidb_final/sweep/results.jsonl; key_numbers.json `e3_flow_cells` = (180 runs, 0)]).
- **Root probes fall from 8.956 to 2.08-2.84 on 16 samples** (fb_uniform 2.08, stack_window 2.08, books_window 2.10, history_window 2.10, stack_uniform 2.17, history_uniform 2.18, wise_window 2.18, libio_window 2.25, fb_window 2.27, covid_window 2.28, libio_uniform 2.32, genome_window 2.38, covid_uniform 2.44, books_uniform 2.60, wise_uniform 2.65, genome_uniform 2.84) and to 3.17 (planet_window), 3.43 (osm_window), 5.14 (planet_uniform), 8.17 (osm_uniform).
- **Total probes per lookup fall from 16.07-19.15 to 9.19-13.40 on 19 samples**: −34.9 % to −42.8 % on the 16 samples above (fb_window −34.9 %, genome_uniform −35.4 %, …, stack_window −42.8 %), −31.1 % on planet_window, −30.1 % on osm_window, −22.2 % on planet_uniform; osm_uniform goes 19.06 → 18.28 (−4.1 %), essentially unchanged. With region virtual points on top (`vp10 + root fusion`) the totals are 9.09-12.77 on the same 19 (−37.0 % to −44.0 %) and 17.84 on osm_uniform.
- **Virtual fences used**: 1 (fb_uniform, books_window), 13-130 on eleven samples, 206-294 on planet_window and genome_uniform, 662 on books_uniform, 984 on osm_window and 1,956 (the whole budget) on osm_uniform and planet_uniform. The greedy stops early when no insertion lowers the SSE [smoothing.hpp:47], so the count is a measure of how far the fence CDF is from a line.
- **The raw-key root alone** (`root raw`) beats the binary search on 16 samples (estimate 2.27-8.63 vs 8.956) and falls back on books_uniform (11.08), osm_uniform (11.67), planet_uniform (12.59) and osm_window (10.55). The flow-feature root (`root flow`, estimate including the +4 charge) is 6.71-15.84 and never below the raw candidate on the same sample. With the +4 charge removed, flow+fences would edge out raw+fences on 10 of 20 samples, by 0.02-0.18 probes on nine of them (e.g. covid_uniform 2.43 vs 2.56, libio_uniform 2.27 vs 2.46) and by 2.13 on planet_uniform (3.11 vs 5.24) [results/aidb_final/root_analysis.json <sample>.packed_rank_root_fusion.est, fifth entry minus 4 vs fourth entry]; a charge of 4 probe-equivalents ≈ 16 ns is already far below the measured 66 ns per evaluation (§7), so even the planet_uniform case (2.13 probes ≈ 9 ns) does not pay.
- **Build**: 52-60 ms for the control, 54-642 ms with the virtual-fence root (the 0.25-0.64 s cases are the three samples that used 984-1,956 fences: Algorithm 1 over 489 fences with budget 1,956 is ≈ 1,956 × 489 gap rescans), 1.67-4.29 s with region virtual points. **Metadata**: 0.571 B/key (control) → 0.571-0.576 with root fences (4 bytes per fence + per slot: 9,780 bytes for 1,956 fences on a 2M-key sample, i.e. 0.005 B/key), → 0.82-1.37 B/key with region virtual points.
- **Throughput** (last column; also §8): root fences 0.894-1.078, root fusion 0.932-1.076, vp10 + root fusion 0.892-1.093; pooled geometric means over all 60 pairs are 0.992 (root fences), 1.010 (root fusion), 0.993 (vp10 + root fusion). The time effect of removing ≈ 6.9 root probes is not resolvable on this host.

#### 3.4 The sentinel-fence artefact and the withdrawn synergy (what the stale rows said and why it was wrong)

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

### 4. E4 — Granularity: does the transform start to matter when regions grow?

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

### 5. E5 — Full scale: 200M keys, fb and planet

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

### 6. E6 — Conformance and coverage of the 25 metric compositions

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

### 7. Cost decomposition: what a probe costs, what the flow costs, when batching would pay

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

### 8. Throughput overall

#### 8.1 The strip of paired speedups per variant (final sweep, 5M lookups × 3 seeds, QoS on)

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

#### 8.2 The noise band: how far apart two measurements of the same structure land

Two ways to measure it from the final sweep itself:

- `root fusion` vs `root fences` when both chose raw+fences (all 20 samples): the root and fence counters are identical; the only difference is that `root fusion` has the flow file loaded and evaluates it in the 0.0-8.8 % of regions the region-level bypass accepted (`transform_calls` per lookup 0.0006-0.088). Per-seed ratios span **0.842 to 1.245** (60 pairs); per-sample geometric means span **0.942 (history_uniform) to 1.092 (books_uniform)** [key_numbers.json `e8_noise_same_structure`, `e8_noise_same_per_sample`].
- `raw root` that fell back to the binary search vs the control (identical lookup path, 4 samples × 3 seeds): per-seed ratios 0.824-1.029, per-sample 0.903 (planet_uniform), 0.904 (osm_window), 0.913 (osm_uniform), 0.948 (books_uniform) [`e8_noise_fallback`]. All four are below 1.0, which is consistent with a session effect: the root variants were measured in the second pass (20:46-21:17) and the control in the first (19:16-20:11) [READING_GUIDE §5 E3], so every root-variant speedup in §8.1 is a cross-session pair.
- The pre-fix verifier had 75 such sham pairs: ratio mean +0.3 %, sd 5.8 %, range −10.2 to +18.9 % [coststory.txt "NOISE FLOOR"].

Conclusion: per-dataset time effects below ≈ 6-10 % are not resolvable here, and single pairs can differ by 25 %. The meeting notes' "up to 8 % (10 % for the binary-fallback root)" [MEETING_NOTES §3] is the per-sample statement; the per-seed extremes are wider.

#### 8.3 Where the time goes: the uncompressed-rank ablation

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

### 9. What is settled, what is noisy, what is open — and the sentence to say

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

### 10. Questions the supervisor may ask, with answers

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


---

## Caveats and likely questions (from the short guide)

- "Is this NFL / CSV?" No: clean-room controls inside our map. NFL's AFLI and its BNAF trainer are not
  run (GPL-3 code, MKL); CSV has no code, so ours is written from the paper and checked against an
  exhaustive toy oracle. The AIDB paper's six indexes are not run.
- "Why 2M-key samples?" The index runs at 2M for time; hardness is computed on the full 200M keys; 14 of
  the 16 full-scale runs are done (section 5, E5): at 200M keys fb's root goes from 15.7 to 3.4 probes with
  973 virtual fences; planet's root cannot be learned within a 1,950-fence budget and gains only 13% with
  195,316 fences (34 min of smoothing); two planet runs at budget 4 are still running
  (`results/aidb/run_chain3.log` has the live status). Uniform and window samples answer different questions and disagree on
  local hardness (fb window PLA-32 10,599 vs uniform 3,034).
- "How reliable is the timing?" Two measurements of an identical structure differ by up to 8% (10% for
  the binary-fallback root) on this laptop (performance/efficiency cores, shared host); per-dataset time
  effects below that are not resolvable. That is why mechanism claims use counters.
- "Why did the transform fail?" At region scale a per-region line already absorbs global curvature; at
  the root, once fitted on real keys, the raw key is already good and virtual fences finish the job; and
  the unbatched evaluation costs more than the probes it saves. It is a cost failure, not a prediction
  failure.
- "What would change the conclusion?" Batched lookups (NFL's regime), a higher-capacity or per-node
  transform, or a host where routing rather than decoding dominates.
- "What was verified and how?" Section 5, E7: five independent agents recomputed the ablation, the cost
  decomposition and the synergy claim from the raw run files and reviewed the notes; their reports and
  scripts are archived.
- "How expensive is CSV's preprocessing?" Algorithm 1 as written is O(budget x n) per model (each greedy
  round rescans every gap after the refit). Per region (4,096 keys, alpha 0.1) that is 12-32 us of thread
  time per key; at the root with alpha 4 it is 1956 rounds over 489 fences at 2M keys (0.6 s) but 195k
  rounds over 48,828 fences at 200M keys, which did not finish in 2 hours (section 5, E5). The CSV paper
  itself reports 889-2,902 s of preprocessing at alpha 0.1 on 200M keys.
- Open implementation question: one CD ordering differs from the paper's (paper prints no CD values).
