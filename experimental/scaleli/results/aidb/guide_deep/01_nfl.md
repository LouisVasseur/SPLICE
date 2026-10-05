# 01. NFL: Robust Learned Index via Distribution Transformation (deep)

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

## 1. Bibliographic facts

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

## 2. The problem and the thesis

### 2.1 Learned indexes approximate the CDF

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

### 2.2 Why the error depends on the key distribution

Segmentation looks for sub-curves of the CDF that are "roughly linear", i.e. whose local probability density
is near-uniform [NFL §2.2, Fig. 3]. A linear model f(k) = w·k + b fits the rank exactly iff the keys are
equally spaced, i.e. iff the density is uniform on the segment. Curvature of the CDF (density that varies over
the segment) is what produces residuals r(kᵢ) − f(kᵢ). Therefore: **the harder the distribution, the more
segments (or the deeper the tree) a learned index needs**, and depth is paid on every lookup as extra
predictions and cache misses [NFL §2.1-2.2].

### 2.3 Distribution transformation

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

### 2.4 The three challenges the paper states (§2.4)

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

### 2.5 The two-stage framework (§3.1) and the batching decision

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

## 3. Normalizing flows from first principles, then NFL's numerical flow

### 3.1 Change of variables in one dimension

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

### 3.2 Maximum-likelihood training

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

### 3.3 Why invertible and monotone matter

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

### 3.4 The flow family and the exact deployed architecture

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

### 3.5 Feature space expansion (§3.2.1) and Algorithm 3.1 line by line

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

### 3.6 The decoder

Line 20 sums the d outputs of the flow into one scalar zᵢ. Nothing in the paper motivates the sum beyond
"merge the high-dimensional features into 1D keys" [NFL §3.2.1]. Two consequences: (i) the sum is not itself a
bijection of ℝᵈ → ℝ, so uniqueness of z is not guaranteed by construction, only empirically; (ii) the
log-Jacobian used in training is that of the d-dimensional flow F, not of the composed scalar map key → z. For
d = 2 with a final linear layer, the sum decoder folds the two output columns into a single effective output
weight per hidden unit, v_h = w1[h,0] + w1[h,1] [train_flow.py, `forward`], which is how our trainer and reader
treat it.

### 3.7 What happens if the composed transform is not monotone

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

### 3.8 Worked forward pass of a real 2D2H2L weight file

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

## 4. Conflict degree and tail conflict degree (Definitions 3.1 and 3.2)

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

### 4.1 Worked example on 10 keys (script section A)

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

### 4.2 Why a "soft" 99th percentile and not the maximum

The paper calls the tail conflict degree "a soft measurement" [NFL §3.1.1]. A maximum is dominated by one
pathological cluster (e.g. many duplicates or a dense burst of timestamps), which AFLI handles with a child node
anyway (Algorithm 3.2 lines 18-22); sizing every bucket for that one cluster would waste space everywhere else.
The 99th percentile is what "most" positions need, so buckets of that size absorb almost all conflicts with a
linear scan, while the rare larger clusters recurse. Empirically the transformed data settle at D₉₉ ≈ 4 on every
dataset [NFL Table 3, §3.3], and the bucket cap is set to 6 [NFL §4.1.3].

---

## 5. The switching mechanism (§3.2.2) and Table 2

### 5.1 The exact rule

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

### 5.2 Which datasets end up without the flow, and why

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

### 5.3 Table 2 in full: transformation latency per key

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

## 6. AFLI: the After-Flow Learned Index (§3.3)

### 6.1 Design premise

After the flow, "the conflict degrees can be kept around a low value (e.g., around 4 for the tail conflict
degree)" on every dataset, so the index no longer needs the large, empirically tuned node capacities that
existing indexes carry to survive high-conflict data [NFL §3.3]. AFLI's "main idea is to buffer local conflicts":
a few colliding keys go into a small bucket rather than a new model, because a linear model fitted on 4 keys
"lacks generalization capability" [NFL §3.3, citing LIPP]; but buckets must stay tiny or scanning them dominates.
The trade-off knob is the tail conflict degree.

### 6.2 Node types (§3.3.1, Fig. 5)

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

### 6.3 Bulk load and Modelling (Algorithm 3.2)

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

### 6.4 Lookup

Start at the root (a model node or a dense node; a bucket is never a root). Model node: evaluate the linear
model, read the entry type from the bitmaps at the predicted position: empty ⇒ key absent; data slot ⇒ compare
the stored key; bucket ⇒ linear scan of the bucket; node pointer ⇒ recurse. Dense node ⇒ binary search
[NFL §3.3.2, Queries]. The point the paper stresses: "the predicted positions in model nodes are all precise
positions, which means that there is no extra local search in model nodes" — the only searches are the ≤ 6-entry
bucket scan and the dense-node binary search [NFL §3.3.2].

### 6.5 Insert and split ("Modelling")

Model node: predict the position; empty ⇒ store and set the bitmaps; data slot ⇒ create a bucket holding the
two conflicting pairs and replace the entry by a bucket pointer; bucket or node pointer ⇒ insert there. Bucket:
append at the tail (or insertion-sort in ordered mode). Dense node: binary search; empty ⇒ insert; occupied ⇒
shift data to the closest empty slot and insert [NFL §3.3.2, Insertions]. When a bucket or dense node is full,
it is converted into a model node by Algorithm 3.2 (a bucket is sorted first) [NFL §3.3.2, Modelling; Fig. 6].
Update = lookup + in-place payload write. Delete = lookup + unset a bitmap bit (model node) or overwrite the key
with the following keys (bucket/dense node) [NFL §3.3.2, More Operations].

### 6.6 Why AFLI is built for near-uniform keys

Every structural decision assumes D₉₉ is small: capacity is at most α·n slots per node (line 7) so a heavy
cluster cannot be absorbed by over-allocation; conflicts up to D₉₉ ≈ 4-6 are scanned linearly in a bucket, which
is cheap only because D₉₉ is tiny; anything larger recurses, which on raw FB (D₉₉ = 386) or LLT (146) would
produce deep, unbalanced trees. The paper shows exactly that with AFLI alone: "The long P99 latency of proposed
AFLI on the workloads LLT and FB also prove that without the transformation of NF, the learned index can be an
unbalanced tree due to locally large conflicts" [NFL §4.3, Read-Only]. AFLI is therefore not a general-purpose
index; it is the second stage of a pipeline whose first stage guarantees D₉₉ ≈ 4.

---

## 7. Experiments (§4)

### 7.1 Hardware, software, parameters [NFL §4.1.3]

* Ubuntu 20.04, Intel Core i7-10700 (8 cores, 2.9 GHz), 64 GB RAM; **single thread**; GCC 9.3.0 at −O3.
* NF inference through Intel MKL; NF training in PyTorch on an RTX 3080 (10 GB) with 64 GB host RAM.
* Flow: modified B-NAF, 2 layers, 2 input dimensions, 2 hidden dimensions, Gaussian base with variance 10¹⁶;
  trained on a 10 % sample of the bulk-loaded keys, "for three times".
* **Batch size 256** for all operations. Bucket size ≤ 6.
* Baselines with their default hyper-parameters: LIPP, ALEX, PGM-Index, Google's C++ B-Tree [NFL §4.1.2].

### 7.2 Datasets and workloads [NFL §4.1.1]

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

### 7.3 Throughput (§4.2, Figure 7)

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

### 7.4 Tail latency (§4.3, Figures 8-9)

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

### 7.5 Bulk loading, memory, ablations (§4.4)

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

## 8. What our study takes from NFL, and what it does not

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

## 9. Questions the supervisor may ask, with answers

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
