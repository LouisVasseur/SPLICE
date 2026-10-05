import re, sys
P='results/aidb/guide_deep/01_nfl.md'
s=open(P).read()
edits=[]
def rep(old,new):
    edits.append((old,new))

# --- intro: mention second script
rep("`results/aidb/guide_deep/scratch/nfl_worked.py` (stdlib only); its output is saved next to it as `nfl_worked.out`.",
"`results/aidb/guide_deep/scratch/nfl_worked.py` and `nfl_worked2.py` (stdlib only); their outputs are saved next to\nthem as `nfl_worked.out` and `nfl_worked2.out`.")

# --- E13 notation: symbol flip warning
rep("generator is G_θ (z → x) and its inverse G_θ⁻¹ (x → z), the linear model is M, the tail percent is γ.",
"generator is G_θ (z → x) and its inverse G_θ⁻¹ (x → z), the linear model is M, the tail percent is γ.\n"
"One warning before you open the PDF: the paper itself flips the symbol. §2.3 defines the generator as z → x\n"
"(\"x_i = G_θ(z_i)\", eq. 4), but §3.1 writes \"the normalizing flow first transform it to a target distribution\n"
"p(z), i.e., z = G_θ(x)\" [NFL §2.3 p.4; NFL §3.1 p.4]. This guide keeps §2.3's convention throughout (G_θ: z → x,\n"
"G_θ⁻¹: x → z, the direction executed online), so where §3.1 of the PDF says G_θ(x) read G_θ⁻¹(x).")

# --- E35 licence row
rep("| Licence | GPL-3 (the repository licence; this is why the workspace never copies its code and calls its own transform a clean-room control) | [transform.hpp:5], [docs/APPROACHES.md, NFL row] [unverified: licence file not among the sources; stated by our code comment and the workspace docs] |",
"| Licence | GPL-3 (the repository licence; this is why the workspace never copies its code and calls its own transform a clean-room control). Checked: the README we hold (`nfl_readme.md`, 64 lines: Introduction, Requirements, Getting Started, Training, Results, Contact) contains **no licence statement at all**; the only sources for \"GPL-3\" are our own code comment, the workspace approach table and the previous guide | [transform.hpp:5], [docs/APPROACHES.md:18], [results/aidb/READING_GUIDE.md:15] [unverified: licence file not among the sources; nfl_readme.md has no licence text; stated only by our code comment and the workspace docs] |")

# --- E25 Table 1 paradox
rep("(errors per prediction 1.89 → 4.51; the paper attributes the FB gain to the height reduction 11 → 3 and the\nLLT gain to fewer out-of-boundary keys per node [NFL §2.2]). What \"# Prediction Errors\" measures (sum of\nabsolute errors? count of non-zero errors?) is **not defined** in the paper.",
"(errors per prediction 1.89 → 4.51; the paper attributes the FB gain to the height reduction 11 → 3 and the\nLLT gain to fewer out-of-boundary keys per node [NFL §2.2]). What \"# Prediction Errors\" measures (sum of\nabsolute errors? count of non-zero errors?) is **not defined** in the paper.\n\n"
"Two things to be ready to defend about this table. (i) The apparent FB paradox: the flow cuts the number of\n"
"predictions ×4.90 and the total \"errors\" ×2.05, yet errors *per prediction* rise 1.89 → 4.51. The paper's only\n"
"explanation is structural: \"the tree height of ALEX can be reduced from 11 to 3 levels\", so each lookup makes\n"
"fewer predictions, and \"the numbers of predictions are also reduced\" [NFL §2.2]; because the unit of \"#\n"
"Prediction Errors\" is undefined (a count of mispredictions and a sum of error magnitudes behave differently\n"
"when the tree gets shallower), the per-prediction ratio cannot be interpreted further [unverified: metric\n"
"definition not given in the paper]. (ii) Table 1 is measured with the flow in front of **ALEX**, not AFLI: it is\n"
"the paper's evidence that the transformation is index-agnostic, which is what §3.1 later calls \"flexible\"\n"
"[NFL §2.2, Table 1; §3.1].")

# --- E14 challenges section + renumber framework subsection
rep("### 2.4 The two-stage framework (§3.1) and the batching decision",
"### 2.4 The three challenges the paper states (§2.4)\n\n"
"The paper's own problem statement is a list of three challenges, each answered by one section of the design\n"
"[NFL §2.4 \"Challenges\", p.4]:\n\n"
"1. **Efficacy of normalizing flow.** \"Naively using NF is limited in a few ways: 1) the NF perform poorly due to\n"
"   limited features from the numerical data of keys; 2) the uniform distribution is hard to function directly as\n"
"   an training objective.\" Answered by the feature-space expansion (§3.2.1, our §3.5) and by the wide-Gaussian\n"
"   training target instead of the uniform (§3.2, our §3.4).\n"
"2. **Efficiency of normalizing flow.** \"The transformation must be an efficient online step. Such requirement\n"
"   also limits the complexity of normalizing flows. Directly reducing the number of parameters in normalizing\n"
"   flows might degrade the transformation quality so that learned indexes require deeper hierarchy and more\n"
"   models to approximate the CDF.\" Answered by the inference optimizations of §3.2.2 (tiny flow, MKL, batching;\n"
"   Table 2, our §5.3).\n"
"3. **Lack of proper indexes for transformed keys.** \"With the transformation of Numerical NF that fundamentally\n"
"   makes linear models approximate better, the design of learned indexes should be reconsidered in a new\n"
"   perspective ... the locality of the transformed data distribution should be considered in the design of the\n"
"   learned index.\" Answered by AFLI (§3.3, our §6).\n\n"
"So the thesis has exactly three moving parts, flow quality, flow cost, and an index shaped for the flow's output,\n"
"and the experiments in §4 are organized to test each (Table 3 for quality, Table 2 and Figure 10 for cost,\n"
"Figures 7-9 for the index).\n\n"
"### 2.5 The two-stage framework (§3.1) and the batching decision")
rep("119 Mops/s. This single row is the quantitative reason NFL must batch (§2.4) and the reason our unbatched",
"119 Mops/s. This single row is the quantitative reason NFL must batch (§2.5) and the reason our unbatched")

# --- E3/E11 trainer loss
rep("section D). Our stand-in trainer optimizes exactly this 1-D expression with Adam and analytic gradients\n[train_flow.py, `loss_and_grad`]; the paper's trainer is the PyTorch B-NAF code [NFL §4.1.3].",
"section D). The paper's trainer is the PyTorch B-NAF code, with the base variance σ² = 10¹⁶ [NFL §4.1.3].\n\n"
"Our stand-in trainer optimizes a *related but not identical* 1-D objective with Adam and analytic gradients\n"
"[train_flow.py:48-68, `loss_and_grad`]. Per key, with d = dz/dx in normalized-x units:\n\n"
"```\n"
"if d > 0:   L = 0.5·z² − log(d) + barrier/d          (base density N(0, 1): σ = 1, constant ½·log 2π dropped,\n"
"                                                     the constant log var from dz/dkey = d/var dropped,\n"
"                                                     positivity barrier added; barrier default 1e-3)\n"
"else:       L = 0.5·z² + 50 − 10·d                   (linear push-back when the sign of dz/dx flips)\n"
"```\n\n"
"[train_flow.py:54-55; defaults lr 0.05, barrier 1e-3, Adam β = (0.9, 0.999), ε = 1e-8 at :98, :121-122]. The\n"
"two differences that matter: the paper's base variance is 10¹⁶ and ours is 1 (so our z² term is of order 1 and\n"
"pulls z toward 0, whereas the paper's is negligible, §3.4), and ours adds a barrier that enforces dz/dx > 0.\n"
"The best per-key NLL values reached are 3.38 (planet) to 4.32 (genome) [results/aidb/flows/training_report.json,\n"
"`best_nll`]; they are not comparable with any number in the paper (which prints no likelihoods).")

# --- E26 activation and parameter count rows
rep("| Parameter count | 8 = 2×2 (input→hidden) + 2×2 (hidden→output), no biases; check: 2H4L = 4·(2×2) = 16, 4H3L = 2·4 + 4·4 + 4·2 = 32, 4H4L = 2·4 + 4·4 + 4·4 + 4·2 = 48, all matching the bracketed counts in Table 2 (script section E) | [NFL Table 2] |",
"| Parameter count | 8 = 2×2 (input→hidden) + 2×2 (hidden→output), no biases; check: 2H4L = 4·(2×2) = 16, 4H3L = 2·4 + 4·4 + 4·2 = 32, 4H4L = 2·4 + 4·4 + 4·4 + 4·2 = 48, all matching the bracketed counts in Table 2 (script section E). The bracketed 8 is the only paper-side evidence for \"no bias\": with biases 2H2L would have 2·2+2 + 2·2+2 = 12 parameters (script 2, section O). The Introduction's complaint that \"existing normalizing flows are of too high complexity (e.g., 4 layers, 16 parameters)\" is exactly Table 2's 2H4L (16) column, so the 8-parameter net is the paper's deliberate minimum | [NFL Table 2], [NFL §1 p.1], [transform.hpp:8-9] |")
rep("| Activation | not stated in the paper; the shipped weight file is evaluated as \"a small tanh network\" by the official C++ side, which is what our reader replicates (tanh after every layer except the last, then sum) | [transform.hpp:6-10, 64-79] [unverified: read from our code comment describing the official repository, not from the PDF] |",
"| Activation | not named in the paper; its only hint is §3.2.2: \"The computations in inference can be simplified as several matrix computations and nonlinear function computations\" (so a nonlinearity exists, its type is not stated). The shipped weight file is evaluated as \"a small tanh network\" by the official C++ side, which is what our reader replicates (tanh after every layer except the last, then sum) | [NFL §3.2.2 p.6], [transform.hpp:6-10, 64-79] [unverified: tanh read from our code comment describing the official repository, not from the PDF] |")

# --- E15 positivity row
rep("| How positivity/monotonicity is enforced | not stated in the paper. In B-NAF as published, the weight matrices are block lower-triangular, the diagonal blocks are made strictly positive by an element-wise exponential, and tanh is used between layers, which makes each output a strictly increasing function of its own input dimension and gives a closed-form log-Jacobian [unverified: from the B-NAF paper (ref. [2]), which is not among our sources] | — |",
"| How positivity/monotonicity is enforced | not stated in the paper. In B-NAF as published, the weight matrices are block lower-triangular, the diagonal blocks are made strictly positive by an element-wise exponential, and tanh is used between layers, which makes each output a strictly increasing function of its own input dimension and gives a closed-form log-Jacobian [unverified: from the B-NAF paper (ref. [2]), which is not among our sources]. Two facts about the *deployed* file: (a) it holds two full, unmasked 2×2 matrices; our reader checks only their shapes [transform.hpp:43-51] and evaluates them as a plain dense tanh net [transform.hpp:71-78], and `fb_2D2H2L.txt` shows a dense W0 with a non-zero first row and a zero second row, so whatever block-triangular/positive-diagonal structure B-NAF has during training is not visible in the exported format [unverified: whether the official exporter bakes masks and the exponential into the stored values]; (b) the general Jacobian of the deployed net with respect to the key is given in §3.8 and holds whether or not the fractional row is zero | [transform.hpp:43-51, 71-78], [results/aidb/flows/fb_2D2H2L.txt] |")

# --- E17 variance 1e16 arithmetic
rep("log p(z) finite everywhere (the uniform has −∞ log-density outside its support, which is the \"INF-loss\" the\npaper mentions) [NFL §3.2].",
"log p(z) finite everywhere (the uniform has −∞ log-density outside its support, which is the \"INF-loss\" the\npaper mentions) [NFL §3.2].\n\n"
"The number that makes \"flat\" concrete (script 2, section K): with σ² = 10¹⁶ the data term of the loss in §3.2 is\n"
"z²/(2σ²) ≤ 10¹²/(2·10¹⁶) = 5×10⁻⁵ for every |z| ≤ 10⁶, and 5×10⁻¹¹ for |z| ≤ 10³, i.e. negligible against the\n"
"log-Jacobian term, whose magnitude is of order 1. The objective therefore reduces to maximizing Σᵢ log|dz/dx|(xᵢ),\n"
"\"stretch the key axis where the keys are dense\", which is the CDF-learning reading of §3.1. Contrast with our\n"
"stand-in trainer, where σ = 1: for z of order 1 the data term is ≈0.5, of the same order as −log|dz/dx|, so our\n"
"objective *also* pulls z toward 0, and the best per-key NLL it reaches is 3.38 (planet) … 4.32 (genome)\n"
"[results/aidb/flows/training_report.json, `best_nll`; train_flow.py:48-55] (see §3.2 and §8).")

# --- E18 out-of-range keys note (add bullet after the Line 2 bullet)
rep("  [train_flow.py, `--shifts` help: \"author default 1e6 for 200M keys\"] [unverified: not in the PDF].\n* Lines 6-7 as printed use xᵢ, not x_norm.",
"  [train_flow.py, `--shifts` help: \"author default 1e6 for 200M keys\"] [unverified: not in the PDF].\n"
"* **Keys outside [min(X), max(X)] of the bulk load.** μ and σ are fixed at bulk load (line 2), so a queried or\n"
"  inserted key below min(X) gives x_norm < 0 and one above max(X) gives x_norm > θ; the integral part is then\n"
"  negative or larger than θ. The paper never discusses this case and avoids it experimentally: \"all inserted\n"
"  keys are in the key space consisting of all bulk-loaded keys, which means all insertions are known-key-space\n"
"  insertions\" [NFL §4.1.1]; behaviour for out-of-range keys: **not reported**. Our code: `transform.hpp:66`\n"
"  normalizes with the stored mean/var without clamping, tanh saturates gracefully, and a non-finite z is\n"
"  replaced by 0 [transform.hpp:79]; the trainer sets mean = first training key and var = (last − first)/shifts\n"
"  [train_flow.py:89], so keys of the full sample outside the *training subsample's* range already exercise\n"
"  this path (fb: 408 keys give x < 0 and 102 give x > 64, script 2, section N).\n"
"* Lines 6-7 as printed use xᵢ, not x_norm.")

# --- E12 transform.hpp:13-16 -> 14-16 (three places)
s_count = s.count("[transform.hpp:13-16]")
assert s_count==3, s_count
s = s.replace("[transform.hpp:13-16]","[transform.hpp:14-16]")

# --- E2 weight file header comment and labels
rep("15162980.0  1207648189.703125   ← mean = min key, var = (max − min)/64   [train_flow.py, train(): shifts=64]",
"15162980.0  1207648189.703125   ← mean = min, var = (max − min)/64 of the 4,096-key TRAINING SUBSAMPLE [train_flow.py:89, :128]")
rep("Evaluation [transform.hpp:64-79]: x = (key − mean)/var; a = [x, x − ⌊x⌋]; u = a·W0; h = tanh(u); out = h·W1;\nz = out₀ + out₁. Because the fractional row is zero, z depends on x only through x·W0[0,:], so dz/dx =\nΣ_h (1 − h_h²)·W0[0,h]·(W1[h,0] + W1[h,1]) > 0: this particular file is monotone (script section C):",
"Precision note on the header line: `mean` and `var` are the minimum and (max − min)/64 of the 4,096 keys drawn\n"
"with seed 1000000007 from the 2M sample [train_flow.py:87-89, :128], **not** of the 2M sample itself. Recomputed\n"
"(script 2, section N): the full `fb_2M_uniform_s42` file has min 97,995 and max 77,308,811,965; the subsample\n"
"has min 15,162,980 and max 77,304,647,121, which is exactly what the file stores (var = 1,207,648,189.703125).\n"
"Consequently the rows labelled \"subsample min/max\" below are the training-subsample extremes; 408 real keys\n"
"lie below 15,162,980 (x < 0, down to x = −0.012475) and 102 above 77,304,647,121 (x > 64, up to 64.003449),\n"
"harmless here only because the fractional-feature row is zero and tanh is defined everywhere.\n\n"
"Evaluation [transform.hpp:64-79]: x = (key − mean)/var; a = [x, x − ⌊x⌋]; u = a·W0; h = tanh(u); out = h·W1;\n"
"z = out₀ + out₁. The general Jacobian with respect to the key, valid also when the fractional row is non-zero\n"
"(this is the formula the trainer differentiates [train_flow.py:36-39]): with u_h = x·W0[0,h] + frac(x)·W0[1,h],\n"
"h_h = tanh u_h and v_h = W1[h,0] + W1[h,1] (the sum decoder folds the two output columns),\n\n"
"```\n"
"dz/dx = Σ_h (1 − h_h²) · (W0[0,h] + W0[1,h]) · v_h        almost everywhere (d frac(x)/dx = 1 between integers)\n"
"dz/dkey = (dz/dx) / var\n"
"jump at each integer x = m:  z(m⁺) − z(m⁻) = − Σ_h v_h · ( tanh(m·W0[0,h] + W0[1,h]) − tanh(m·W0[0,h]) )\n"
"```\n\n"
"(checked against a central finite difference to 1.5×10⁻⁹, script 2, section J). The jump term is what makes\n"
"the composed map non-monotone when W0[1,:] ≠ 0 (§3.7). Because the fractional row of *this* file is zero, z\n"
"depends on x only through x·W0[0,:], the jump vanishes, and dz/dx = Σ_h (1 − h_h²)·W0[0,h]·v_h > 0: this\n"
"particular file is monotone (script section C):")
rep("| 15,162,980 (min) | 0 |","| 15,162,980 (subsample min) | 0 |")
rep("| 77,304,647,121 (max) | 64 |","| 77,304,647,121 (subsample max) | 64 |")

# --- E16 sawtooth table after the 65-grid-point paragraph
rep("flow feature never wins inside a 4,096-key region (§8).\n\n---\n\n## 4. Conflict degree",
"flow feature never wins inside a 4,096-key region (§8).\n\n"
"The table above only uses integer x (16, 32, 48, 64), where frac(x) = 0, and the fractional weights are zero\n"
"anyway, so it never shows the non-monotone case of §3.7 and Q2. Here is that case, computed (script 2, section\n"
"I) with the same file except a **hypothetical** fractional row W0[1,:] = [0.5, 0.5] (what `--monotone` would\n"
"have left free):\n\n"
"| x | key | frac(x) | u = (u₁, u₂) | z, real file (row 1 = 0) | z, hypothetical row 1 = [0.5, 0.5] | dz/dx (hyp., a.e.) |\n"
"|---|---|---|---|---|---|---|\n"
"| 15.90 | 19,216,769,196 | 0.90 | (0.54310, 0.54316) | 0.43396 | 2.31476 | 1.78379 |\n"
"| 15.99 | 19,325,457,533 | 0.99 | (0.58863, 0.58869) | 0.43640 | 2.47166 | 1.70250 |\n"
"| 16.00 | 19,337,534,015 | 0.00 | (0.09369, 0.09375) | 0.43667 | **0.43667** | 2.34319 |\n"
"| 16.01 | 19,349,610,497 | 0.01 | (0.09875, 0.09881) | 0.43694 | 0.46009 | 2.34091 |\n"
"| 16.50 | 19,941,358,110 | 0.50 | (0.34662, 0.34668) | 0.45024 | 1.55796 | 2.10107 |\n\n"
"Between integers z rises steeply (dz/dx ≈ 1.7-2.3 instead of 0.027), and at x = 16 it drops from 2.47166 back\n"
"to 0.43667: a jump of −2.05196, equal to the closed-form jump term above evaluated at m = 16. The map key → z\n"
"is a sawtooth with period var = 1.2×10⁹ key units, so ≈2.5 % of the fb key pairs would be re-ordered on every\n"
"tooth. That is exactly why `train_flow.py` has `--monotone` (it zeroes W0[1,:] at initialization and keeps the\n"
"gradient of those two weights at zero [train_flow.py:67, :96]) and why every `*_training.json` reports\n"
"`unordered_transformed_pairs = 0` with `monotone: true` [results/aidb/flows/fb_training.json; train_flow.py:133].\n"
"NFL itself keeps those weights (\"deviation from NFL, which keeps them\" [train_flow.py:124]) and lives with the\n"
"sawtooth by sorting on z (§3.7).\n\n---\n\n## 4. Conflict degree")

# --- E32 comparison paragraph after Definition 3.2 uses
rep("flow on/off switch (§5); (2) the capacity threshold of buckets and dense nodes in AFLI (§6) [NFL §3.1.1].\n\n### 4.1",
"flow on/off switch (§5); (2) the capacity threshold of buckets and dense nodes in AFLI (§6) [NFL §3.1.1].\n\n"
"**Three conflict degrees, not one.** A DB professor will ask how this relates to the other two \"conflict\n"
"degrees\" in this guide. (a) LIPP's conflict degree: \"LIPP is more concerned with conflict degrees, i.e., the\n"
"number of keys predicted to the same position\" [NFL §2.1 p.3]; LIPP's bulk load (FMCD) searches the smallest\n"
"bound D such that its line puts at most D keys into any slot. (b) NFL's tail conflict degree: the same count\n"
"per position under one least-squares line fitted on keys and *scaled positions*, but the statistic is the 99th\n"
"percentile over occupied positions, not the maximum [NFL Definitions 3.1-3.2]. (c) The AIDB paper's CD, which\n"
"our `hardness.json` reports: \"the maximum number of keys mapped to the same rank by a linear model fitted on the\n"
"dataset using Fastest Minimum Conflict Degree (FMCD)\", i.e. LIPP's fit and a *maximum* statistic, ported in\n"
"`fmcd_fit` [include/scaleli/hardness.hpp:7-8, :168; results/aidb/MEETING_NOTES.md §1 \"CD via LIPP's FMCD fit\"].\n"
"All three count keys per predicted position under one linear model; they differ in the fit (plain least squares\n"
"with scaled positions in NFL; FMCD's search over the slope with capacity L = size·(gap+1) in LIPP/AIDB) and in\n"
"the statistic (99th percentile vs maximum). Section 03 §3.2 gives the FMCD definition and its capacity rule.\n\n### 4.1")

# --- E7/E19 switch granularity
rep("The decision is made once per bulk load for the whole dataset (the transformed keys \"cannot be stored, as it\nwould cause another indexing problem\", so the choice is between transforming every key online or none [NFL\n§3.2.2]). Motivation: \"keys in some datasets are already near-uniform distributed, it is unnecessary to spend\nextra time and memory to transform them\" [NFL §3.2.2].",
"What the paper states about the granularity: the switch compares the tail conflict degrees of \"the input keys\"\n"
"and \"the transformed keys\" [NFL §3.2.2], and §4.2 reports **one on/off outcome per dataset** (YCSB, AMZN, WIKI\n"
"off; the other four on). That the decision is taken once per bulk load for the whole dataset, and that it is\n"
"all-or-nothing, is this guide's inference from those two passages together with the separate sentence that\n"
"\"The transformed keys cannot be stored, as it would cause another indexing problem\" (a statement about online\n"
"inference, not about the switch) [unverified: inferred from §3.2.2 + §4.2; the paper does not say \"once per bulk\n"
"load\"]. Motivation: \"keys in some datasets are already near-uniform distributed, it is unnecessary to spend\n"
"extra time and memory to transform them\" [NFL §3.2.2].\n\n"
"What the paper does **not** say about the switch, each **not reported**: (i) on which keys the two tail conflict\n"
"degrees are computed, all ≈100M bulk-loaded keys or the 10 % training sample; (ii) which linear model is used,\n"
"presumably one global fit over all keys, i.e. the root model of Algorithm 3.2 line 1 with \"scaled positions\",\n"
"but the text does not say; (iii) whether the comparison is strict: §3.2.2 says \"If the latter tail conflict is\n"
"larger\", which the AMZN row of Table 3 (4 → 4, yet disabled) contradicts (§5.2). Contrast with our control,\n"
"where both values are computed **per 4,096-key region on the region's own keys** with a plain rank fit\n"
"(intercept rescaled so the first key maps to 0) and a 10 % margin [index.hpp:235-237, :248, :280-282;\n"
"transform.hpp:90-106].")

# --- E28 Table 3 (R) workload not reported
rep("paper does not resolve it]. Note also that after ~100M insertions the raw tail on FB grows 386 → 454 while the\ntransformed tail stays 4 [NFL §4.4.1: \"after inserting around 100 million new keys, the tail conflict degree is\naround 4\"]",
"paper does not resolve it]. Note also that after the running phase the raw tail on FB grows 386 → 454 while the\ntransformed tail stays 4. Table 3's caption only says \"(R)\" is \"after the running phase\"; it does not say which\nworkload produced those rows (read-heavy, write-heavy and write-only insert different numbers of keys), so\n\"~100M insertions\" is taken from §4.4.1's prose, not from the table: **workload for (R) not reported** [NFL Table\n3 caption p.11; §4.4.1: \"after inserting around 100 million new keys, the tail conflict degree is around 4\"]")

# --- E4 Table 2 caption verbatim
rep("\"Average transformation latency of each key with different NFs. H = hidden dimension, L = number of layers,\nbrackets = number of parameters, all in ns\" [NFL Table 2, p.6]:",
"Caption, verbatim: \"Average transformation latency of each key with different NFs. \\\"H\\\" and \\\"L\\\" correspond\nto the hidden dimension and the number of layers, respectively. The figure in brackets (e.g., \\\"(12)\\\")\nrepresents the amount of parameters of NF. All latency is measured in nanosecond (ns).\" [NFL Table 2, p.6]:")

# --- E5 15% -> 13%
rep("going to batch 2048 only adds another 15 % (7.29 ns, 23.3× total). The knee is between 32 and 128.",
"going to batch 2048 only lowers the per-key cost by a further 13 % (8.38 → 7.29 ns, a 1.15× ratio; 23.3× in\n  total vs batch 1: 169.53/7.29 = 23.26, script 2, section O). The knee is between 32 and 128.")

# --- E6 & E22 batch-1 cap and control cost
rep("* At batch 1 the transform alone caps throughput at 1e9/169.53 = 5.90 Mops/s, which is *below* the read-only\n  throughput of the baselines the paper beats (its Figure 7 axes reach 45 Mops/s); at batch 256 the cap is\n  119 Mops/s. This single row is the quantitative reason NFL must batch (§2.5) and the reason our unbatched\n  control charges the flow at 66 ns per evaluation and never finds it worthwhile (§8).",
"* At batch 1 the transform alone caps throughput at 1e9/169.53 = 5.90 Mops/s, which is far below the scale of\n  Figure 7 (y-axes up to 45 Mops/s in panel (a) and 40 in panel (e)); at batch 256 the cap is 119 Mops/s. The\n  paper prints no numeric baseline throughputs, only ratios, so \"below every baseline\" cannot be checked from\n  the PDF [unverified: Figure 7's axis maxima bound the tallest bar, they do not give the baselines' values;\n  NFL p.9 Fig. 7 axis ticks; §4.2 gives only 2.34×, 2.46×, 3.82×, 7.45×]. This single row is the quantitative\n  reason NFL must batch (§2.5). Two separate numbers describe our unbatched control: the **measured** cost of one\n  transform on the lookup path is 66 ns [56, 77] [results/aidb/MEETING_NOTES.md §3, pooled pre-fix data], and\n  what the fused selector **charges** is c_flow = 4 probe-equivalents per lookup (`--flow-cost 4`, the default,\n  recorded as `flow_cost: 4` in every run) added to the in-sample probe count [src/benchmark.cpp:179, :191;\n  index.hpp:264; results/aidb/sweep/results.jsonl `flow_cost`]; with either accounting the selector never finds\n  the flow worthwhile (§8). For the batching comparison at build time: our build loop transforms the 2M fb keys\n  in 59,765,592 ns = 29.88 ns/key [results.jsonl, fb packed_rank_flow, `learnability.transform_ns`;\n  summary.csv `preprocess_ns_per_key` 29.882796; MEETING_NOTES §4 \"27-34 ns/key\"], versus NFL's 8.38 ns at\n  batch 256 and the ≈103 ns/key implied by its bulk-load figure (below).")

# --- E29 bulk load scope caveat
rep("Table 2 (script section H). The bulk-load figure presumably includes the switch (transform + two tail-conflict\ncomputations) and non-batched paths; the paper does not reconcile the two [unverified: my arithmetic on two\nindependent statements].",
"Table 2 (script section H). The bulk-load figure presumably includes the switch (transform + two tail-conflict\ncomputations) and non-batched paths; the paper does not reconcile the two [unverified: my arithmetic on two\nindependent statements]. A second caveat: 13.42 s and 77 % are averages whose scope is not stated. If they\naverage over all seven datasets including the three where the switch disables the flow (YCSB, AMZN, WIKI [NFL\n§4.2]), the per-key transform cost on the four flow-on datasets is even higher than 103 ns; if instead the switch\nstill transforms every key once to compute D₉₉(z) (\"first tries to transform the input keys\" [NFL §3.2.2]), the\n77 % includes that one pass on every dataset. Keep the arithmetic, but label both readings [unverified: scope of\nthe average not stated].")

# --- E1 & E27 Algorithm 3.2 numbering + off-by-one note
rep("11:         if D[pos] == 1 then                                    ← exactly one key: DATA SLOT\n12:             𝔫.E[pos] = ⟨xᵢ, vᵢ⟩ ; i += 1\n14:         else if D[pos] < D_γ^M_L then                          ← small conflict: BUCKET\n15:             build a bucket of max size D_γ^M_L holding ⟨xᵢ..x_{i+D[pos]}⟩\n16:             𝔫.E[pos] = bucket ; i += D[pos]\n18:         else  (D[pos] ≥ D_γ^M_L)                                ← large conflict: CHILD NODE\n19:             walk the following positions while D[pos_seq] > D_γ^M_L, summing their degrees into 𝔇_seq\n20:             allocate a new node at 𝔫.E[pos]\n21:             Modelling({⟨xᵢ..x_{i+𝔇_seq}⟩}, 𝔫.E[pos])          ← recursion\n22:             set every position pos+1..pos_seq to the SAME child pointer (duplicated pointers)\n23:             i += 𝔇_seq\n25:     end for\n26: end if\n27: return 𝔫",
"11:         if D[pos] == 1 then                                    ← exactly one key: DATA SLOT\n12:             𝔫.E[pos] = ⟨xᵢ, vᵢ⟩\n13:             i = i + 1\n14:         else if D[pos] < D_γ^M_L then                          ← small conflict: BUCKET\n15:             build a bucket 𝔟 of max size D_γ^M_L storing {⟨xᵢ,vᵢ⟩, …, ⟨x_{i+D[pos]}, v_{i+D[pos]}⟩}  (as printed)\n16:             𝔫.E[pos] = 𝔟\n17:             i = i + D[pos]\n18:         else if D[pos] ≥ D_γ^M_L then                          ← large conflict: CHILD NODE\n19:             iterate the subsequent positions pos_seq where D[pos_seq] > D_γ^M_L, summing their degrees into 𝔇_seq\n20:             allocate a new node at 𝔫.E[pos]\n21:             Modelling({⟨xᵢ,vᵢ⟩, …, ⟨x_{i+𝔇_seq}, v_{i+𝔇_seq}⟩}, 𝔫.E[pos])   ← recursion (as printed)\n22:             for all positions pos+1 .. pos_seq set the pointer to 𝔫.E[pos]   (duplicated pointers)\n23:             i = i + 𝔇_seq\n24:         end if\n25:     end for\n26: end if\n27: return 𝔫")
rep("[NFL §4.2, Read-Write]. (Line numbering follows the print; lines 13, 17, 24 are the `end` lines.)",
"[NFL §4.2, Read-Write]. Line numbering follows the print exactly: 13 is \"i = i + 1\", 17 is \"i = i + D[pos]\", 24\nis \"end if\", 25 \"end for\", 26 \"end if\" [NFL p.8, Alg. 3.2]; the paper's own prose refers to the data-slot case as\n\"Line 10-13\" and the bucket case as \"Line 14-17\" [NFL §3.3.2, p.8].\n\n"
"Reading note (an off-by-one in the print). Lines 15 and 21 as printed list the pairs from index i to i + D[pos]\n"
"(resp. i + 𝔇_seq) **inclusive**, i.e. D[pos] + 1 (resp. 𝔇_seq + 1) pairs, while lines 17 and 23 advance i by\n"
"exactly D[pos] (resp. 𝔇_seq). Since D[pos] is by definition the number of keys predicted to pos, the intended\n"
"range is i .. i + D[pos] − 1 (resp. i .. i + 𝔇_seq − 1); otherwise consecutive buckets would share a key and a\n"
"bucket would hold one key more than its conflict degree. Also note the asymmetry between lines 18 (\"≥\") and 19\n"
"(\">\"): the run of positions folded into one child starts at a position with D ≥ D_γ and continues only through\n"
"positions with D > D_γ [NFL p.8, Alg. 3.2 lines 15, 17, 18, 19, 21, 23].")

# --- E8 abstract -> introduction
rep("| Abstract average | 2.77× throughput, 43 % lower tail latency | | | | [NFL §1] |",
"| Introduction average (the abstract prints no numbers) | 2.77× throughput, 43 % lower tail latency | | | | [NFL §1, p.2: \"2.77x improvements on average in throughput and 43% reductions on average in tail latency\"] |")

# --- E24 Figure 7 transform inclusion
rep("The per-dataset bars of Figure 7 are not printed as numbers; the paper only gives averages and extremes in the\ntext (every value below is a quote of §4.2; per-dataset bar heights are **not reported** numerically):",
"The per-dataset bars of Figure 7 are not printed as numbers; the paper only gives averages and extremes in the\ntext (every value below is a quote of §4.2; per-dataset bar heights are **not reported** numerically). Whether\nFigure 7's throughput *includes* the online transformation is not stated: Figures 8, 9 and 10 split NFL-Index\nfrom NFL-Trans, but Figure 7 draws a single \"NFL\" series, and the only explicit statement that a measurement\nincludes the transform is for bulk loading (\"the time cost includes the time cost of the online transformation\nof bulk-loaded keys\" [NFL §4.4.2]) [unverified: Figure 7 legend has one NFL series (NFL, AFLI, LIPP, ALEX,\nPGM-Index, B-Tree); whether transform time is inside the throughput is not stated]. The README's result line has\na single `(overall throughput)` column next to separate T and I latency columns, so the natural reading is that\nthroughput is end-to-end; treat that as a reading, not a fact [nfl_readme.md, Results].")

# --- E9 & E23 workloads
rep("averaged [NFL §4.1.1]. Everything is **per batch of 256** operations: the index receives 256 requests, transforms\nthem with one MKL call, then serves them.",
"averaged [NFL §4.1.1]. Everything is measured at **batch size 256** [NFL §4.1.3] and the inference \"can be\nsimplified as several matrix computations and nonlinear function computations\" on MKL [NFL §3.2.2]; the picture\n\"receive 256 requests → transform them with one MKL call → serve them\" is this guide's reading of those two\nsentences [unverified: mechanism inferred from §3.2.2 and §4.1.3; the paper does not describe the per-batch\nsequence].\n\n"
"What a DB professor will ask and the paper does not give, each [NFL §4.1.1] **not reported**: the Zipfian skew\n"
"parameter (only \"sampled from the given dataset based on a Zipfian distribution\"); the number of operations in\n"
"the running phase (the \"~100M insertions\" used elsewhere in this section is an inference from §4.4.1's \"after\n"
"inserting around 100 million new keys\", which fits a write-only run over the remaining 50 % of a 200M-key\n"
"dataset, not a stated count); whether lookups include non-existent keys (misses); the record layout (the\n"
"payload type `int64` is stated, the layout is not); and the exact key-space bound used for \"known-key-space\"\n"
"insertions (min..max of the bulk-loaded keys is the natural reading).")

# --- E10/E31 dataset provenance
rep("(FB and WIKI are the same sources as SOSD's fb and wiki; the paper cites SOSD [20] as the dataset provenance\ntogether with ALEX and LIPP. Our ten GRE datasets overlap with these on fb, osm-derived and wiki-derived data\nbut are not the same files; see section 02 of this guide.)",
"(The PDF says only that the seven are \"representative datasets used in [6, 20, 42]\", i.e. ALEX, SOSD and LIPP\n[NFL §4.1.1 p.8]; that FB and WIKI coincide with SOSD's fb and wiki files is plausible from the identical\ndescriptions but cannot be verified from the sources in hand [unverified: the SOSD paper is not among our\nsources]. Our ten GRE datasets are books, fb, osm, covid, genome, history, libio, planet, stack, wise\n[results/aidb/provenance.json]; whether GRE's fb and osm are the same files as the FB and OSM-derived sets here,\nand which of ours is \"wiki-derived\", is not established in this section [unverified: dataset lineage taken from\nnames only; see section 03 §2 and section 04 for the AIDB paper's dataset table].)")

# --- E30 batched P99 illustration
rep("batch, not the 99th percentile of individual requests; it smooths single-request outliers by construction.\nFigures 8-9 separate NFL-Index (index part) from NFL-Trans (transform part), matching the README's T/I columns.",
"batch, not the 99th percentile of individual requests; it smooths single-request outliers by construction.\nIllustration (script 2, section L): a batch of 256 in which one lookup takes 10 µs and the other 255 take 50 ns\nhas batch latency 10,000 + 255·50 = 22,750 ns and is reported as 22,750/256 = 88.87 ns; a per-request P99 of\nthe same trace would report 50 ns and its max 10 µs. This is the one-line reason the paper's \"P99 < 80 ns on LLT\nand FB\" is not comparable with an unbatched per-lookup P99 such as our `read_hit_p99_ns_median` column (which is\n0 in `results/aidb/sweep/summary.csv` because that sweep did not run the latency replay, `latency_pass: false`;\nthe point is definitional).\nFigures 8-9 separate NFL-Index (index part) from NFL-Trans (transform part), matching the README's T/I columns.")

# --- §8: E34, E20, E21, E22, E11 not-taken list
rep("two 2×2 matrices, tanh hidden layer, linear last layer, sum decoder) so that author-trained weights would load\n[transform.hpp:35-51, 64-79];",
"two 2×2 matrices, tanh hidden layer, linear last layer, sum decoder) so that author-trained weights would load\n[transform.hpp:35-51, 64-79] [unverified: format read from our reader's expectations; no official weight file\nis among the sources and none was loaded in this study, APPROACHES.md:18 lists \"author weights on real SOSD\ndata\" as not covered];")
rep("per 4,096-key region on the raw and the transformed feature [transform.hpp:90-106; index.hpp:237-282]; (4) the",
"per 4,096-key region on the raw and the transformed feature [transform.hpp:90-106; index.hpp:237-282]. Detail\nthat answers \"how can you compute a conflict degree for a transform that is not monotone?\": the transformed D₉₉\nis computed on the **sorted** z values (index.hpp:248 and :280 copy z into `zs`, `std::sort` it, then call\n`tail_conflict_degree`), and `tail_conflict_degree` fits feature → index-in-sorted-order (transform.hpp:94 uses\ni as the target), so for a non-monotone flow our metric mirrors NFL's \"sort by z\" semantics even though our\nrecords stay in key order. fb numbers from the run log [results/aidb/sweep/results.jsonl, packed_rank_flow,\nfb, seed 11, `learnability`]: 489 regions, 41 accepted (41/489 = 8.38 %), mean per-region D₉₉ raw 8.4479 vs\nflow 8.4642 (the flow does not lower the average), `flow_bytes` 144 (8 weights + mean/var + struct overhead\n[transform.hpp:62]); (4) the")
rep("Not taken: AFLI (our host keeps its own regions, blocks, fences and packed codecs; z is only the model feature\nand records stay in key order [transform.hpp:14-16]); the B-NAF PyTorch trainer (ours is a stdlib 1-D\nchange-of-variables trainer with Adam, `--monotone` zeroing the fractional-feature weights, 4,096 sampled keys,\n200 steps, ≈2.4 s per dataset [train_flow.py; results/aidb/flows/training_report.json]); and batching (every\nlookup transforms one key, costing 66 ns [56, 77] per evaluation on our host [MEETING_NOTES.md §3], against\nNFL's 8.38 ns at batch 256 [NFL Table 2]).",
"Not taken: AFLI (our host keeps its own regions, blocks, fences and packed codecs; z is only the model feature\nand records stay in key order [transform.hpp:14-16]); the B-NAF PyTorch trainer (ours is a stdlib 1-D\nchange-of-variables trainer with Adam, `--monotone` zeroing the fractional-feature weights, 4,096 sampled keys,\n200 steps, ≈2.4 s per dataset [train_flow.py; results/aidb/flows/training_report.json]); the paper's training\nobjective (ours is 0.5·z² − log(dz/dx) + barrier/(dz/dx), i.e. a **standard normal base, σ = 1, not the paper's\nσ² = 10¹⁶**, with a positivity barrier, §3.2 [train_flow.py:48-55]); the paper's normalization constants (our\nmean/var come from the 4,096-key training subsample, §3.8 [train_flow.py:89]); and batching (every lookup\ntransforms one key, costing 66 ns [56, 77] per evaluation on our host [MEETING_NOTES.md §3], while the selector\ncharges it c_flow = 4 probe-equivalents [benchmark.cpp:191; index.hpp:264]; at build time our loop spends 29.88\nns/key on fb [summary.csv `preprocess_ns_per_key`], against NFL's 8.38 ns at batch 256 [NFL Table 2]).")
rep("forcing it into every region leaves fence probes identical to the control (e.g. fb 2.5999 vs 2.5999) [same\nfile, packed_rank_flow_forced]. That is the expected result of §3.8: an 8-weight monotone tanh is nearly affine,\nand a per-region linear model already absorbs what it can express.",
"forcing it into every region leaves fence probes equal to the control's to within 5.4×10⁻⁴ (fb: packed_rank\n2.599879 vs packed_rank_flow_forced 2.599933, a difference of 5.4×10⁻⁵; the largest gap on any dataset is planet,\n+5.4×10⁻⁴, and the granularity ablation states \"within 0.05\") [same file, packed_rank_flow_forced;\nMEETING_NOTES.md §1]. That is the expected result of §3.8: an 8-weight monotone tanh is nearly affine, and a\nper-region linear model already absorbs what it can express.\n\n"
"Fence probes per operation, control vs forced flow, read-only, uniform 2M samples, 3 seeds each\n"
"[results/aidb/sweep/summary.csv, rows packed_rank / packed_rank_flow / packed_rank_flow_forced; script 2,\n"
"section M]:\n\n"
"| dataset | packed_rank (control) | packed_rank_flow (bypass) | packed_rank_flow_forced | forced − control | flow_region_fraction (bypass) | preprocess_ns_per_key (bypass) |\n"
"|---|---|---|---|---|---|---|\n"
"| books | 2.230642 | 2.230594 | 2.230558 | −0.000084 | 0.0532 | 28.87 |\n"
"| covid | 3.063852 | 3.063821 | 3.063754 | −0.000098 | 0.0348 | 31.50 |\n"
"| fb | 2.599879 | 2.599902 | 2.599933 | +0.000054 | 0.0838 | 29.88 |\n"
"| genome | 3.297549 | 3.297557 | 3.297578 | +0.000029 | 0.0164 | 32.33 |\n"
"| history | 2.458673 | 2.458677 | 2.458760 | +0.000087 | 0.0266 | 29.33 |\n"
"| libio | 2.555318 | 2.555336 | 2.555368 | +0.000050 | 0.0429 | 31.24 |\n"
"| osm | 5.076565 | 5.076247 | 5.076308 | −0.000257 | 0.0143 | 30.82 |\n"
"| planet | 3.229126 | 3.228882 | 3.229667 | +0.000541 | 0.0879 | 32.12 |\n"
"| stack | 2.291464 | 2.291464 | 2.291436 | −0.000028 | 0.0020 | 31.57 |\n"
"| wise | 2.417157 | 2.417156 | 2.417249 | +0.000092 | 0.0082 | 26.95 |\n\n"
"Every difference is below 10⁻³ probes per lookup, i.e. below the seed spread; the `preprocess_ns_per_key`\n"
"column is the per-key cost of the one transform pass at build (27-32 ns/key here, unbatched, plain loops).")

# --- Q3 caveat, Q4 wording, Q7 wording, Q14 cost
rep("   8.38 at 256, 7.29 at 2048 (2H2L, MKL, i7-10700). Larger flows: 21.66 / 23.52 / 35.07 ns at batch 256 [NFL\n   Table 2]. Bulk-loading 100M keys spends 77 % of 13.42 s in the transform (≈103 ns/key), which the paper does\n   not reconcile with Table 2 [NFL §4.4.2; my arithmetic].",
"   8.38 at 256, 7.29 at 2048 (2H2L, MKL, i7-10700). Larger flows: 21.66 / 23.52 / 35.07 ns at batch 256 [NFL\n   Table 2]. Bulk-loading 100M keys spends 77 % of 13.42 s in the transform (≈103 ns/key), which the paper does\n   not reconcile with Table 2 [NFL §4.4.2; my arithmetic; unverified: the scope of those averages (all seven\n   datasets or only the flow-on ones) is not stated]. Our unbatched control: 66 ns [56, 77] measured per lookup\n   transform, 29.88 ns/key in the build loop on fb [MEETING_NOTES.md §3; summary.csv].")
rep("   5.9 Mops/s ceiling, below the baselines' read-only throughput, while at batch 256 it is 8.38 ns [NFL Table 2].",
"   5.9 Mops/s ceiling, far below the 40-45 Mops/s scale of Figure 7's axes, while at batch 256 it is 8.38 ns\n   [NFL Table 2; Fig. 7 axes] [unverified: the baselines' numeric throughputs are not printed].")
rep("7. **When is the flow switched off?** Once per bulk load, for the whole dataset, if D₉₉ on the transformed keys is\n   larger than on the raw keys [NFL §3.2.2]. In the paper: YCSB (3 → 4), WIKI (2 → 4) and AMZN (4 → 4, which\n   under the literal rule would keep it; unresolved in the paper) [NFL §4.2, Table 3].",
"7. **When is the flow switched off?** If D₉₉ on the transformed keys is larger than on the raw keys [NFL §3.2.2];\n   the paper reports one outcome per dataset, and \"once per bulk load, for the whole dataset\" is the guide's\n   inference [unverified: §3.2.2 + §4.2]. Which keys and which model the two D₉₉ are computed on: not reported.\n   In the paper: YCSB (3 → 4), WIKI (2 → 4) and AMZN (4 → 4, which under the literal rule would keep it;\n   unresolved in the paper) [NFL §4.2, Table 3]. Ours: per 4,096-key region, on the region's keys, with a 10 %\n   margin [index.hpp:281].")
rep("    order, stdlib trainer instead of B-NAF, no batching (66 ns per unbatched evaluation vs 8.38 ns batched), no\n    AFLI.",
"    order, stdlib trainer with σ = 1 and a positivity barrier instead of B-NAF with σ² = 10¹⁶, no batching (66 ns\n    measured per unbatched evaluation, charged as 4 probe-equivalents by the selector, vs 8.38 ns batched), no\n    AFLI.")

# --- E33 extra questions
rep("15. **What is the licence situation?** github.com/luffy06/NFL is GPL-3; no code was copied, only the public text\n    weight format was re-implemented (format compatibility is not a derivative work of the code)\n    [transform.hpp:3-10; docs/APPROACHES.md] [unverified: licence text itself not in our sources].",
"15. **What is the licence situation?** github.com/luffy06/NFL is GPL-3; no code was copied, only the public text\n    weight format was re-implemented (format compatibility is not a derivative work of the code)\n    [transform.hpp:3-10; docs/APPROACHES.md:18] [unverified: licence text itself not in our sources; the README we\n    hold has no licence statement].\n\n"
"16. **Why a flow and not simply the empirical CDF, or a spline of it (RadixSpline, PGM), as the transform?** In\n"
"    §3.1's own terms the CDF *is* the ideal transform (a uniform z is what a learned index wants), and the paper\n"
"    lists RadixSpline [21] among the learned indexes it positions itself against [NFL §5.1, ref. 21]. A spline of\n"
"    the CDF is itself a learned index: it carries as many parameters as it has segments, must be rebuilt when the\n"
"    data change, and is applied per lookup. The flow is a smooth *parametric* stand-in that costs 8 weights and\n"
"    8.38 ns/key batched [NFL Table 2], generalizes (\"normalizing flows have much better generalization, they\n"
"    don't need to re-train during every bulk loading phase\" [NFL §3.2.2]), and is trained offline once\n"
"    (\"about 38 seconds\" [NFL §3.2.2]). The price is capacity: an 8-weight monotone tanh is nearly affine (§3.8),\n"
"    which is why the paper reports its quality \"has almost reached to its upper limit\" [NFL §4.4.1] and why our\n"
"    per-region linear models see nothing left to gain (§8).\n\n"
"17. **How much memory does the flow take?** 8 weights plus mean and var; our reader reports `flow_bytes` = 144\n"
"    (the struct plus 8 doubles [transform.hpp:62; results/aidb/sweep/results.jsonl `learnability.flow_bytes`]).\n"
"    The paper's benchmark records a `(model size)` column in every result line [nfl_readme.md, Results], but the\n"
"    PDF prints no number for it: **not reported**.\n\n"
"18. **What does the paper's index size (2.26× ALEX, 3.1× PGM, 0.51× LIPP) actually count?** \"the overall index\n"
"    size, including the sum size of allocated gaps, rather than the model size\", measured \"after the running phase\n"
"    of write-heavy workloads\" and normalized in Figure 11 [NFL §4.4.3]; so gaps from precise placement are inside\n"
"    the number, which is why LIPP (also precise placement) is the only index larger than NFL.\n\n"
"19. **Is D₉₉ well defined for tiny m?** Yes: t = INT(m·γ) is at least 1 for m ≥ 2 (m = 1 gives t = 0, which the\n"
"    paper does not address; **not reported**), and the worked example of §4.1 has m = 6 ⇒ t = 5 ⇒ D₉₉ = 2 while\n"
"    the maximum is 4. Our control uses ⌈0.99·m⌉ − 1 as the 0-based index, which at m = 6 picks the maximum\n"
"    (index 5) and at m = 1000 the same 990th element [transform.hpp:104].\n\n"
"20. **The flow is not monotone; how do you compute a conflict degree on it?** NFL sorts by z; we sort the z\n"
"    values before calling `tail_conflict_degree`, which fits feature → sorted index [index.hpp:248, :280;\n"
"    transform.hpp:94], so the metric is the same as NFL's while our records stay in key order (§8). With\n"
"    `--monotone` weights the question is moot: 0 unordered pairs on all ten datasets [training_report.json].")

for old,new in edits:
    c=s.count(old)
    if c!=1:
        print("ANCHOR COUNT",c,"for:",old[:90].replace("\n","\\n")); sys.exit(1)
    s=s.replace(old,new)
open(P,'w').write(s)
print("applied",len(edits)+1,"edits; lines:",s.count("\n")+1)
