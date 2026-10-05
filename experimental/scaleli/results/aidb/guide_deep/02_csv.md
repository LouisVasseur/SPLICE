# 02. Learned Indexes with Distribution Smoothing via Virtual Points (CSV), deep account

**What this section gives you.** A self-contained, equation-level reading of the CSV paper: the exact optimisation problem it poses (loss over ranks with and without virtual points, the budget λ = α·n, the NP-hardness sketch and its weak spot), the single-model greedy (Algorithm 1) reproduced line by line with the closed-form refit and the derivative-sign filter, a worked 7-key example computed with Python so every number can be re-derived by hand, the hierarchical CSV (Algorithm 2) with its cost condition (eq. 22), what the paper does and does not say about lookups and inserts in ALEX / LIPP / SALI, the experimental setup and the reported numbers (Tables 2–4, Figures 6–10), and finally an honest map of what our clean-room CONTROL (`smoothing.hpp`, `index.hpp`) takes from CSV and what it deliberately leaves out, with our own measured numbers. Every number carries an inline source; anything not in the PDF, the code or a result file is marked "not reported" or [unverified: ...]. Notation follows the guide's convention: keys k₁ < … < kₙ, rank r(kᵢ) = i − 1, linear model f(k) = w·φ(k) + b, slot s(kᵢ) = rank + number of virtual points before kᵢ, budget λ = α·n.

---

## 1. Bibliographic facts

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

## 2. Motivation and the core idea

**The model.** A learned index is a function f from a key to its storage position; with keys stored in ascending order the target is the rank, so f approximates the CDF: rank(kᵢ) ≈ f(kᵢ) [CSV §1]. The paper uses only linear f (w·k + b) "for their efficiency" and says the idea "can naturally extend to more complex (e.g., quadratic) functions" [CSV §1].

**The observation.** Hierarchical indexes (LIPP is the example) push hard-to-model key ranges into deeper levels, and deeper keys cost more per query: Figure 1 plots average query time per level of LIPP on Facebook, Covid, OSM, Genome (200M keys each) and shows time rising with level [CSV Fig. 1]. Prior work fixes this by changing the model (splines, more segments) or the structure; CSV instead "modifies the key space": it changes the training targets so that a linear model fits better [CSV §1].

**What a virtual point is.** A virtual point k_v is an extra value inserted into the sorted key list K (keeping it sorted) that holds no record. Its only effect is on ranks: every real key after k_v moves one position later, so the rank targets of the real keys change. In Figure 2, ten black dots (real keys) are fit by a line with loss L_f(K) = 8.33; after inserting V = {k_v1..k_v5} (budget 0.5·n = 5, red hollow dots) and refitting to f′, the loss on the real keys drops to L_f′(K) = 2.04 and the loss on real plus virtual points is L_f′(K ∪ V) = 2.29 [CSV §1, Fig. 2a/2b].

**Why it can lower the error.** Squared error of an OLS line depends on how far the (key, rank) points sit from any straight line. A cluster of keys with a wide empty range after it forms a "step" in the CDF; a line cannot bend around a step. Inserting virtual points into the empty range stretches the rank axis exactly where the keys are sparse, so the augmented point set lies closer to a line. Section 4.6 below shows this on seven keys: one virtual point in the only wide gap cuts SSE from 2.671 to 1.872.

The one-line algebra behind that sentence: for an OLS line, SSE = S_yy · (1 − ρ²), where S_yy = Σ (yᵢ − ȳ)² is the total variance of the targets and ρ is the Pearson correlation between feature and target over the fitted set. Equivalently, with the running sums c_xx = S_xx − S_x²/n, c_xy = S_xy − S_x·S_y/n, c_yy = S_yy − S_y²/n, SSE = c_yy − c_xy²/c_xx, which is the closed form our control evaluates [smoothing.hpp:38-43]. A virtual point in a wide gap adds one target step exactly where the CDF is flat, which raises |ρ| of the augmented set. On the seven keys of §4.6: before smoothing S_yy = 28, ρ = 0.951101, ρ² = 0.904594, SSE = 28 × 0.095406 = 2.6714; after inserting k_v = 9 (eight targets 0..7) S_yy = 42, ρ = 0.977466, ρ² = 0.955439, SSE = 42 × 0.044561 = 1.8716; after {8, 9, 10} ρ = 0.993812 and SSE = 1.0179 [scratch/csv_fix_checks.py]. Note that S_yy itself GROWS with every inserted slot (28 → 42), so the SSE falls only because 1 − ρ² falls faster; this is also why a whole-sample RMSE measured in slot units can rise after smoothing (§8.5, Q10) while every local fit improves.

**The trade-off.** (i) Space: each virtual point is a slot in the host's array that holds nothing, so the index grows by up to λ = α·n slots; the paper reports the resulting index-size increase in Figs. 8b/8e/8h [CSV §6.2.1]. (ii) Preprocessing time: the search for virtual points is a greedy over candidate positions; Tables 3 and 4 report between 247 s (ALEX, Facebook, α = 0.05) and 81,620 s (ALEX, OSM, α = 0.8) of one-off preprocessing on 200M keys depending on α, host and dataset; the LIPP table alone spans 304 s (Covid, α = 0.05) to 15,709 s (Genome, α = 0.8) [CSV Tables 3–4, p.11; §7.4 below reproduces every cell]. (iii) Query time can worsen for hosts with an in-node search (ALEX) when merged nodes become large; the cost model of §5.1 guards against this [CSV §5, §5.1].

**Origin of the idea.** The authors present CSV as the constructive mirror image of poisoning attacks on learned indexes [11] (Kornaropoulos et al., SIGMOD 2022), which insert points to maximise the SSE; CSV inserts points to minimise it [CSV §2.3]. The related "gap insertion" [16] manipulates ranks directly, which lets several keys share a position and needs an overflow array with "up to 87%" space increase; NFL [34] transforms the key value with a normalising flow, which adds a transform at query time and may raise the tail conflict degree [CSV §2.2, Table 1].

---

## 3. The formal problem (§3)

### 3.1 Loss without virtual points

For one indexing function f over key set K of size n [CSV eq. 1]:

```
L_f(K) = Σ_{i=1..n} ( f(kᵢ) − rank(kᵢ) )²                              (1)
```

For a partition of K into m segments K₁..K_m, each with its own f_i ∈ F [CSV eq. 2, stated exactly]:

```
L_F(K) = Σ_{i=1..m} Σ_{k ∈ K_i} ( f_i(k) − rank(k) )²                   (2)
```

"Equation 2 is the loss function of our optimisation problem. We aim to insert values (virtual points) into K while keeping it sorted, such that L_F(K) is minimised" [CSV §3]. Note carefully: in eq. 2 the rank is the rank in the FULL list K after insertion, and the f_i are the ORIGINAL functions (no refit yet); refitting is introduced only in §4 (eq. 4).

### 3.2 Why a budget is needed

The "naive optimal smoothing" inserts enough virtual points that every key k lands at position f_i(k) exactly (rank(k) = f_i(k)), making the loss zero; with 64-bit integer keys that layout would need 2⁶⁴ × 8 bytes ≈ 128 exabytes [CSV §3]. Hence the budget:

```
|V| ≤ λ,   λ = α · n,   α ∈ (0, 1)        (linear space overhead)          [CSV §3]
```

**Definition 1 (learned index smoothing).** Given sorted K partitioned into m segments each indexed by f_i ∈ F, insert a set V (|V| ≤ λ) of virtual points into K, keeping K in order, such that the loss of eq. 2 is minimised [CSV Def. 1].

### 3.3 Assumptions

- Linear indexing functions ("used in most existing learned indexes") [CSV §3].
- Integer keys; real keys are handled "when they can be scaled up to become integers" [CSV §3]. This is why candidates form integer runs between consecutive keys (§4.2 of the paper).
- Keys are unique: the experiments remove duplicates "to suit LIPP and SALI's requirements", and the candidate filter "skip[s] the index keys already in K" so that hosts without duplicate support stay compatible [CSV §4.2, §6.1].
- Ranks start at 0 [CSV §4.1].

### 3.4 NP-hardness (Lemma 3.1) and how to read it

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

## 4. Single-model smoothing (§4)

### 4.1 The refit objective (eq. 4)

For one segment K_i with budget λ [CSV eq. 4, stated exactly]:

```
argmin_{V_i, w, b}  L_{f_{w,b}}(K_i ∪ V_i)     s.t.  |V_i| ≤ λ                 (4)
```

Two differences from eq. 2 [CSV §4]: (i) the slope w and intercept b are REFITTED to the keys with adjusted ranks plus the virtual points, instead of keeping the original f; (ii) the virtual points themselves are inside the loss ("we include V_i in the loss calculation, such that the storage space allocated to the virtual points can be used to accumulate data insertions, with minimized prediction errors when querying the inserted data points"). This is why Figure 2 reports two numbers: 2.04 on K only and 2.29 on K ∪ V.

**The naive greedy and its cost.** Repeat λ times: try every candidate k_v, refit, compute the loss, keep the best. With p candidates per iteration and O(|K_i| + λ) per loss evaluation the naive cost is O((|K_i| + λ) · λ · p) [CSV §4 "Challenge 2"]. The paper's three steps: reduce p by a derivative test (§4.2), make each loss evaluation O(1) by reusing sums (§4.1), and run the greedy over the reduced set (§4.3).

### 4.2 Closed-form refit and incremental loss (eqs. 5–16)

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

### 4.3 Filtering candidates by the derivative sign (§4.2)

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

### 4.4 Algorithm 1 line by line

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

### 4.5 The claimed complexity O(n + λ)

"[CSV] reduces the computation of the loss over K to just once, which takes O(n) time. This process is repeated to find λ optimal candidate virtual points. However, there is no need to recalculate the loss function after adding a virtual point, as we could treat the key set with the previous virtual point inserted as the new original or base key set for a constant time calculation. Thereby, giving a time complexity of O(λ + n)" [CSV §4.3 "Complexity analysis"].

How to read it: the O(n) is the one-off computation of Σkᵢ, Σkᵢ², Σkᵢyᵢ, Σyᵢ; each candidate's loss is then O(1) via eqs. 10–16 IF the suffix sums in eq. 14 are available in O(1); each of the λ rounds then costs O(number of surviving candidates). The bound O(n + λ) therefore presumes that the number of candidates examined per round is O(1), which the PDF does not justify: after the derivative filter there is still at least one candidate per run, and there are up to n − 1 runs. A strict reading of Algorithm 1 gives O(n + λ · p′) with p′ the candidates surviving the filter (≤ 2·(n − 1) + |M|). Our implementation is O(λ · n) for exactly this reason (§8). [our reading; the PDF states O(λ + n) without the per-round count]

**What that means per region on our data.** A 4,096-key region has 4,095 gaps; at α = 0.1 the budget is ⌊0.1 × 4,096⌋ = 409 rounds, so our O(λ·n) greedy performs on the order of 409 × 4,095 ≈ 1.67 M candidate-gap evaluations per region, each with 4 loss probes plus up to 40 ternary-search steps for a gap with an interior minimum [smoothing.hpp:72-89]. Measured on the books 2M uniform sample with the hardness tool: 489 regions (⌈2,000,000/4,096⌉), 185,327 virtual points in total = 379 per region (the budget is 409, so most regions ran the full budget, a few stopped early), smoothing wall-clock 16.58 s on 16 threads = 33.9 ms of wall-clock per region and 8.3 µs per key of the sample [results/aidb/hardness/books_sample_csv.json: regions 489, virtual_points 185327, smoothing_ns 16,575,434,458, threads 16; src/hardness.cpp:172-173 time the parallel `smooth_regions` call; hardness.hpp:369 spawns the worker pool]. The benchmark reports a different figure for the same dataset and α: preprocess_ns_per_key = 30,591.6 ns = 30.6 µs/key for books_uniform packed_rank_vp10 [results/aidb_final/sweep/summary.csv], and that column is a SUM of per-region smoothing_ns divided by the key count [index.hpp:305, 485; tools/summarize.py:59], i.e. summed thread time, not wall-clock. If the hardness tool's 16 threads had been fully parallel, its summed thread time would be about 16 × 8.3 = 133 µs/key, four times the benchmark's 30.6 µs/key; the runs differ in scheduling (the benchmark ran with performance-core QoS, `qos 1`, the hardness run's QoS is not recorded), in the number of regions that ran the full budget, and in host load, so the gap between the two figures is not established here [our reading]. Either way the order of magnitude is tens of microseconds of CPU per key at 4,096-key regions, which is what §8.4 quotes.

### 4.6 Worked example on seven keys (computed, not taken from the paper)

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

## 5. Hierarchical CSV (§5)

### 5.1 Why a second algorithm

Applying Algorithm 1 to each node in isolation lowers the leaf-node search time but "fail[s] to address traversal time". CSV therefore smooths the key set of a parent plus its subtree and, if a cost condition holds, rebuilds them as ONE leaf, reducing the index height for those keys [CSV §5]. The stated challenge is balancing the traversal saving against the larger leaf; "unbalanced learned index structures are better suited as it gives the ability to reduce the height of taller branches without affecting the rest" [CSV §5].

### 5.2 Algorithm 2 line by line

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

### 5.3 The cost condition (eq. 22) per host

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

### 5.4 Complexity of Algorithm 2

For m non-leaf nodes with nᵢ keys and budgets λᵢ, the paper sums O(λᵢ + nᵢ) over nodes and simplifies to O(λ + n) [CSV §5.1 "Complexity analysis"]. This inherits the per-round assumption of §4.5 and additionally ignores that a key belongs to every ancestor's collected key set, so the same key is smoothed once per level; the sum over levels is not O(n) unless the number of levels is O(1). [our reading]

**Budget accounting the paper never states.** Line 10 calls `CDF_smoothing(keyset, α)`, so each node i gets its own budget λᵢ = α · |keysetᵢ| [CSV Alg. 2 line 10; §5.1 "node_i with n_i keys and a smoothing budget of λ_i"]. A key at depth d belongs to the collected key sets of up to d − 2 ancestors (every level from max_level down to 2), so the pseudo-code bounds the total number of virtual points across one run of Algorithm 2 by α · Σᵢ nᵢ, not by α · n; the two coincide only for a two-level tree. Whether the key set collected at a higher level already contains the virtual points inserted at the lower level (so that they count as "keys" and raise λᵢ), or only the real keys, is not reported. The only empirical bound the paper gives on the resulting space is Fig. 8b/8e/8h: index-size increase "in most cases less than 10%" and "in the worst case less than 31%" at α ≤ 0.8 [CSV §6.2.1 "Index size"]; note that with α = 0.8 a single level alone could add 80% of slots, so the measured 31% says that most keys are smoothed at one level and that early stopping (Alg. 1 line 27) or failed cost conditions leave much of the budget unused [our reading].

### 5.5 Example of merging a subtree into a leaf (illustrative, constructed by us)

Take a LIPP-like tree where a parent P at level 3 has a model over 4,096 keys, 3,900 of which it places exactly and 196 of which conflicted and live in 12 child nodes at level 4 (some of which have their own children at level 5). Algorithm 2 at current_level = 3 collects the 4,096 keys of P and its subtree (line 9), runs Algorithm 1 with α = 0.1, i.e. λ = 409 (line 10). Suppose the augmented loss falls enough that a fresh linear model over the 4,096 + |V| slots places every key without conflict (in LIPP terms, no two keys map to the same slot) — this "suppose" is an assumption of ours for the illustration; the PDF never says that reconstruction requires or achieves conflict-free placement of all collected keys (§5.2, "what reconstruct means"). Under the LIPP cost condition (the loss improved and the node holds more keys) the subtree is replaced by a single leaf with 4,096 real keys and |V| empty slots (line 12). Every key previously at levels 4–5 is now at level 3: those are the "promoted" keys the paper's Figure 8a counts; the 12 + child nodes disappear (Figure 8c "node reduction"); the |V| empty slots are the storage increase (Figure 8b). For ALEX the same merge would additionally have to pass eq. 22: the merged leaf's expected exponential-search length (from its log₂-error estimate) times search_constant, plus traversal_constant times the new level, must be less than c < 0 relative to the current cost ("relative to the current cost" is [our reading] of why c < 0 can mean an improvement; §5.3 item (b)). The numbers in this paragraph are invented for illustration; the mechanics follow [CSV §5, §5.1, Fig. 8].

---

## 6. Lookups and updates with virtual points in each host

What the PDF states, and only that:

- **Where virtual points live.** In the host's node storage as slots with no record ("gaps"). The paper contrasts itself with gapped arrays in ALEX/LIPP/APEX: those leave gaps for future inserts as a side effect, CSV chooses the gap positions to minimise model error [CSV §2.2]. After CSV, "the initial gaps left by the virtual points are gradually filled up by the inserted points" [CSV §6.3 "Index size"], confirming that a virtual point is an empty slot in the host's array.
- **Lookup.** The paper does not describe a modified lookup path. The reasoning in §5.1 tells you what each host does natively: LIPP and SALI "do not contain any searching component", so a prediction is a position; ALEX has a "leaf-node search component" whose expected length ALEX estimates from the log₂ error [CSV §5.1]. The refitted model f′ is trained on the slot layout that includes the gaps (eq. 4), so a real key's predicted slot is close to its actual slot. What happens when the model points at an empty slot for a key that IS present: for ALEX, the native exponential search scans outward over the gapped array until the key is found [unverified: this is ALEX's documented behaviour (Ding et al., SIGMOD 2020), not a statement in the CSV PDF]; for LIPP the stored key is at the predicted position by construction, since keys that would not fit are pushed into child nodes [unverified: LIPP's documented behaviour (Wu et al., PVLDB 2021), not restated in the CSV PDF]. For SALI the CSV PDF says nothing beyond "SALI is based on LIPP" [CSV §6.2.1, p.10] and "there is no such searching in LIPP and SALI's query process" [CSV §6.2.1, p.10]; its own related-work paragraph notes that SALI flattens sub-trees with a PGM-like segmentation, which "leads to an additional search step for queries, as we need to find the correct node from the flattened structure" [CSV §2.2, p.3]. How CSV's virtual points interact with that segment search is not reported.
- **Inserts.** Described only at the level of the read-write experiment: the index is built on a random half of each dataset, CSV is applied once, and the other half is inserted in batches of 0.1·n without re-running CSV [CSV §6.1 "Read-write workload"]. Inserted keys can land in the gaps left by the virtual points, which "helps improve the insertion times in some cases"; when more keys sit in upper-level nodes there are more collisions "which requires new index node creation", raising insert time in other cases; overall insert times are "on par" [CSV §6.3 "Insertion time"]. Storage overhead shrinks after each batch as the gaps fill: at or below 10% for LIPP and "almost negligible (< 0.5%)" for ALEX, sometimes below the original index because plain ALEX creates more nodes to host inserts [CSV §6.3 "Index size"].
- **Deletes.** Not reported.
- **Robustness claim.** "Since the models are built with virtual points that can be used to host data insertions, a side benefit of our structure is that it is more resilient against data insertions" [CSV §2.3].

---

## 7. Experiments (§6)

### 7.1 Setup

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

### 7.2 Approximation quality (Table 2)

| | Exhaustive | CSV (greedy) | Original |
|---|---|---|---|
| Loss | 2.118 | 2.293 | 8.327 |
| Time (ns) | 140,656,167 | 424,667 | N/A |

Ten keys of Fig. 2, α = 0.5 (λ = 5); the text says the greedy improves the loss by 72.34% and the exhaustive by 74.44%, and that the exhaustive takes "nearly 3 orders of magnitude" longer [CSV §5.2, Table 2]. (The time ratio is 331×.) Internal inconsistency to be aware of: the percentages do not follow from the table's own values, (8.327 − 2.293)/8.327 = 72.46% and (8.327 − 2.118)/8.327 = 74.56% [scratch/csv_fix_checks.py]; the 0.12-point gap is unexplained in the PDF (presumably unrounded losses behind the text, but that is not stated). Quote the table's losses, not the text's percentages, if pressed.

### 7.3 Read-only results vs α (Figs. 6–8)

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

### 7.4 Preprocessing time (Tables 3 and 4), seconds, 200M keys

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

### 7.5 Cardinality (Fig. 9) and read-write (Fig. 10)

- Time saved grows with dataset size 12.5M → 200M on all sets, faster on Facebook/Covid because small versions of the easy sets have few lower-level keys [CSV §6.2.2, Fig. 9].
- Read-write, α = 0.1, LIPP and ALEX (SALI omitted): total time saved decreases slightly with each 0.1·n batch for LIPP (inserts collide more often with promoted keys now in upper levels); ALEX similar except two drops on OSM after batches 1 and 3 where the original index "happen[s] to be slightly lower" [CSV §6.3, Figs. 10a/10d]. Storage overhead ≤ 10% for LIPP and < 0.5% for ALEX, decreasing as gaps fill [Figs. 10b/10e]. Insert-time change within about ±8% for LIPP and ±30% for ALEX by the plotted axes, "on par" overall [Figs. 10c/10f; CSV §6.3].

### 7.6 The paper's own stated limitations

Collected from the text: preprocessing "may seem quite large under certain settings" [CSV §6.2.1]; space grows with α up to 31% [CSV §6.2.1]; ALEX gains less because of its leaf search [CSV §6.2.1]; the smoothing of the full key set is "computationally expensive", which is why Algorithm 2 works on subtrees of an already-built index [CSV §5]; requires unique keys [CSV §6.1]; integer keys or real keys scalable to integers [CSV §3]; no re-run of CSV after inserts in the experiments [CSV §6.1]. Not addressed in the PDF: deletes, concurrency, the values of the eq. 22 constants and c, the tail (p99) latency, and any comparison to NFL or gap insertion beyond Table 1's checklist.

---

## 8. What our study takes from CSV and what it does not

Everything here is a clean-room CONTROL inside the SCALE-LI experimental map; it is written from the paper, not from any CSV code (none exists), and it is not a reproduction of CSV or of its ALEX/LIPP/SALI hosts [smoothing.hpp:2-8; docs/APPROACHES.md:22; README.md:35].

### 8.1 Taken: Algorithm 1 per region, on slot targets

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

### 8.2 Taken: slot ranks instead of stored points

The virtual points are never materialised. In a region, `Region::rebuild` runs `smooth_cdf` on the region's features [index.hpp:304-308], fits `rank_model` on (feature, slot) instead of (feature, rank) [index.hpp:312], and records for each 128-key block its `slot_begin` = slot of the block's first key [index.hpp:83, 308]. A lookup computes y = rank_model(f(k)) and binary-searches the blocks' `slot_begin` values to pick the block (`predicted_block`, coordinate probes) [index.hpp:118-129], then `locate_block` corrects with exponential + binary search on the exact block fences (fence probes) [index.hpp:133-140]. Because the key arena never contains a virtual point, the only bytes they cost are the persisted `virtual_features` (8 bytes each, kept so compactions can re-place them) [index.hpp:101, 320]: at α = 0.1 that is ≈ 0.8 B/key, which is exactly the measured metadata increase, e.g. books_uniform 14.53 → 15.27 B/key, fb_uniform 11.39 → 12.19 [results/aidb_final/sweep/summary.csv initial_bytes_per_key]. The paper's virtual points, by contrast, occupy real slots in the host's gapped array and are what its Fig. 8 storage overhead measures. This also means our virtual points cannot host inserts.

Compactions do not re-run the search: the previous region's virtual features are re-placed among the new keys in O(n + v) and kept only if the augmented SSE is still ≤ the plain SSE, otherwise dropped [index.hpp:289-303]; `--relearn 1` forces the full search on every compaction [benchmark.cpp:178; Config::relearn_on_compaction, index.hpp:57-60].

### 8.3 Taken: Algorithm 1 at the root ("virtual fences")

The root of our index is a single linear model over the region fences (one fence per 4,096-key region) that predicts the region, corrected by exponential + binary search on the fences [index.hpp:21-25, 348-357]. `fit_root` evaluates candidates {raw, flow} × {ranks, virtual-fence slots}; the virtual-fence candidate runs `smooth_cdf` on the fence features with budget `root_alpha × regions`, fits on slot targets, and builds a `root_slot_to_region_` table of n + |V| uint32 entries so that a predicted slot maps to a region in O(1) [index.hpp:378-388, 333, 342]. Each candidate is scored by the probes the real locate routine spends on the fences and fence midpoints; binary search wins if no candidate is cheaper [index.hpp:389-400]. So "virtual fences" = CSV Algorithm 1 applied to the ~489 fence features of a 2M sample (or 48,829 at 200M keys), with the same slot semantics.

### 8.4 Not taken

- **Hierarchical Algorithm 2 and the cost model of eq. 22.** Our index has exactly two levels (root over regions, region model over blocks) and no subtree merging; there is nothing to promote [docs/APPROACHES.md:22 "Hierarchical Algorithm 2 (cost-model merges)" listed under not covered].
- **Gap materialisation for inserts** and the authors' ALEX/LIPP/SALI hosts [docs/APPROACHES.md:22].
- **The O(n + λ) complexity.** Ours is O(λ · n) per model: every greedy round rescans every gap after the refit [smoothing.hpp:10-13 "The paper's O(n + lambda) claim is not reproduced"]. Measured: 12–32 µs of thread time per key at α = 0.1 on 4,096-key regions (e.g. books_uniform 30.6, fb_uniform 18.0, osm_uniform 13.4 µs/key) [results/aidb_final/sweep/summary.csv preprocess_ns_per_key]; it grows with region size, fb 18.0 → 90.5 → 207.7 µs/key at 4,096 / 16,384 / 32,768 keys per region [results/aidb_granularity/summary_r*.csv], consistent with O(α·n²/regions). At full scale the 200M-key root at budget 4 × regions spent 2,067.78 s single-threaded inserting 195,316 virtual fences (the whole budget, 4 × 48,829) over planet's 48,829 region fences [MEETING_NOTES.md:118, 123-124; recomputed with `python3 results/aidb/root_analysis.py results/aidb_fullscale/sweep/results.jsonl`, row "planet root fences": virt 195316, build 2067.78 s].
- **The derivative formula (eqs. 17–21).** Replaced by four loss probes and a ternary search [smoothing.hpp:78-87]. Same purpose, different arithmetic; §4.3 above verifies the paper's formula independently.
- **Integer candidates.** Ours are real-valued features inside the gap [smoothing.hpp:18].

### 8.5 What the control measured (numbers, not a reproduction of the paper's)

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

### 8.6 Tests that pin the control's semantics

`tests/test_main.cpp:91-98`: augmented SSE never increases; at most ⌊α·n⌋ points; slots strictly increasing with `slot.back() == n − 1 + |V|`; every virtual feature strictly inside (min, max) and never equal to a key; α = 0 inserts nothing; α = 64 throws; an already-linear set {0..7} gets no virtual points. `tests/test_hardness.cpp:140-149`: per-region smoothing is deterministic across thread counts and rejects unsorted input. `tests/test_tools.py:19-23`: the stdlib toy `tools/virtual_points_lab.py` (exact/greedy on ≤ 4,096-wide integer keys, default keys 1,2,3,4,5,10,20,26,27,30, budget 3) satisfies greedy ≤ original and exhaustive ≤ greedy; that file is explicitly "NOT a reproduction of CSV's optimized algorithm" [tools/virtual_points_lab.py:2].

---

## 9. Questions the supervisor may ask, with answers

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
