# Joint input-warp / output-warp optimisation for the learned-index root

Formal derivation, convergence analysis, and an adversarial degeneracy hunt.
Everything measured here is on the 489 region fences of the 2M-key samples
(`keys[0], keys[4096], ...`), plain Python, standard library only.
Scripts: `diag.py` .. `diag5.py` in this directory; raw output in `diag*.json`.

**Verdict up front.** The joint objective is well posed, the block structure is
real, and one of the two blocks has an exact closed form that the current
trainer is not using. But the mechanism hypothesis, made quantitative, predicts
a pay-off on **planet only** — not on osm. At the workspace's root budget the
virtual-point budget is the binding constraint on 2 of 10 datasets, and on osm
the monotone flow family reduces the binding by 13%, which is worth about 0.4
root probes. The largest measured lever is not the joint optimisation at all: it
is adding a bias inside the tanh (see §4, P4). Two implementation-level defects
would silently break a naive prototype (§2.2, §3d2).

---

## 1. The objective

### 1.1 Data and variables

Sorted keys `x_1 < ... < x_n` — at the root these are the region fences, `n = 489`
for a 2M-key sample at 4096 keys/region. Ranks are `i = 0..n-1`.

**Flow.** `f_θ ∈ F`, the deployed NFL 2D2H2L shape (`transform.hpp`):

```
t      = (x - μ)/σ                     μ = mean, σ = var   (two scalars in the header)
φ(t)   = t - floor(t)                  the "fractional" second input feature
f_θ(x) = Σ_{r=1,2} v_r · tanh(u_r·t + p_r·φ(t))
         u_r = W0[0][r],  p_r = W0[1][r],  v_r = W1[r][0] + W1[r][1]   (sum decoder)
```
θ = (W0, W1, μ, σ) ∈ ℝ⁸ × ℝ × ℝ₊. There are **no biases inside the tanh** — this
matters, see §3d4.

**Order-preserving constraint.** φ is a sawtooth, so any `p_r ≠ 0` makes f
decreasing on part of every unit interval of t. Monotonicity therefore forces
`p_1 = p_2 = 0` (which is exactly what `train_flow.py:67` does), leaving

```
f_θ(x) = c_1·tanh(g_1·(x-μ)) + c_2·tanh(g_2·(x-μ)),    g_r = u_r/σ,  c_r = v_r
```

a **3-shape-parameter family** (g_1, g_2, c_2/c_1) with a **shared centre μ**.

**Virtual points.** `V` is a multiset of features, each assigned to an interior
gap: `v ∈ (f(x_i), f(x_{i+1}))` for some `1 ≤ i < n`. (`smoothing.hpp:75` loops
`i` over `0 .. |seq|-2`, so nothing can be inserted before the first or after the
last key — a small but real restriction on the reachable slot maps.) Write
`m_i ≥ 0` for the count in gap i and `M_i = Σ_{j<i} m_j`.

**Slot ranks.** `s_i = i + M_i`. Equivalently: `s` is any strictly increasing
integer sequence with `s_1 = 0` and `s_i - i` non-decreasing and `s_n - (n-1) ≤ λ`.

**Line.** `(a, b) ∈ ℝ²`.

### 1.2 The program

```
minimise    L(θ, V, a, b) = Σ_{i=1}^{n} ( a·f_θ(x_i) + b - s_i )²
                          + Σ_{v∈V}     ( a·v        + b - σ(v) )²
over        θ ∈ Θ,  V,  (a,b)
subject to  (C1)  f_θ strictly increasing on [x_1, x_n]
            (C2)  f_θ(x_1), ..., f_θ(x_n) pairwise DISTINCT in double precision
            (C3)  |V| ≤ λ, every v strictly interior to a gap
            (C4)  (a,b) = argmin  (refitted OLS at every step)
```

σ(v) is the virtual point's own slot index. The second sum is CSV's equation 4
as `smoothing.hpp` implements it (the SSE runs over real *and* virtual points —
`total()` at line 55 sums over all of `seq`).

(C2) is not cosmetic. `index.hpp:382` guards the virtual-fence root candidate
with `std::is_sorted`, which accepts ties; `smoothing.hpp:76` skips any gap with
`!(hi > lo)`. A tie is an irrecoverable error floor — see §3d2.

### 1.3 What each block solves, and what is closed form

**Block (a,b) — exact closed form.** Ordinary least squares on the augmented
set. `smoothing.hpp` never materialises a or b: `detail::Sums::sse()` returns
`C_yy − C_xy²/C_xx` directly from six running sums, which is the residual SSE at
the optimal (a,b). O(1).

**Block V, f fixed — not closed form.** A discrete problem over multisets.
`smooth_cdf` solves it greedily: one insertion per round, all gaps scanned, each
candidate O(1), within-gap placement by 4 endpoint probes plus an optional
ternary search. Cost O(λ·n).

**Block f, V fixed — not closed form in general, but it contains an exact
3×3 solve that should be used.** Fix the two gains and the centre. Then
`t_r = tanh(g_r(x - μ))` are fixed vectors and

```
a·f_θ(x) + b = (a·c_1)·t_1 + (a·c_2)·t_2 + b
```

is **linear** in `(C_1, C_2, b) := (a·c_1, a·c_2, b)`. So block f reduces to a
3-dimensional nonlinear search over `(g_1, g_2, μ)` wrapped around an exact
3×3 normal-equation solve. That is how every number in §4 was computed — a
48×48 gain grid with an exact inner solve costs 3 seconds per dataset in plain
Python for all ten datasets. Adam on 8 weights (`train_flow.py`) is solving a
harder problem than necessary and is solving it against the wrong loss (§3d5).

**Identifiability.** Only the products `a·c_r` are determined: L is invariant
under `(c_1,c_2) → α(c_1,c_2)` with `a → a/α`, and under `z → z + β` with
`b → b - aβ`. The argmin is therefore a 1-parameter family in θ, not a point.
Consequence for the prototype: **do not use parameter change as a convergence
criterion, only loss change.** And do not optimise `c` and `a` as separate
blocks — merge them, as above.

### 1.4 Does the O(1)-per-candidate trick survive f? Yes, unchanged.

The only thing that makes an insertion non-local is that every later target
shifts by +1. `smoothing.hpp:66-68` handles that with

```
c = |seq| - (i+1)
Δ(Σy) = c        Δ(Σy²) = 2·suf_y[i+1] + c        Δ(Σxy) = suf_x[i+1]
```

then adds the new point `(x_v, i+1)`. **f does not appear in the update rule.**
`suf_y` is a suffix sum of sequence *positions* — purely combinatorial, identical
whatever the feature is. `suf_x` is one suffix sum of whatever the feature
happens to be; it is rebuilt once per round anyway (`rebuild_suffix`, O(n)), so
swapping `x` for `f(x)` changes the numbers in one array and nothing else. The
per-candidate cost stays O(1) and the per-round cost O(#gaps). Three riders:

1. **It needs (C1).** The derivation of the shift rule assumes the sequence is
   in feature order, i.e. inserting at position i+1 with a feature inside the gap
   keeps the sequence sorted. With a non-monotone f the gaps where
   `f(x_{i+1}) ≤ f(x_i)` are silently dropped from the candidate set
   (`smoothing.hpp:76`) — the algorithm still runs and still returns a lower SSE,
   but it is optimising over a truncated feasible set and the resulting slot
   table is meaningless (`index.hpp:382` rejects the candidate outright).

2. **No f⁻¹ is ever needed, and that is worth saying out loud.** A virtual point
   is a slot, not a key. It is chosen *directly in z-space*, so the ternary
   search costs no flow evaluations, and at query time only `(a, b)`, the slot
   table and `f(q)` are used. If instead one insisted on choosing a virtual
   *key* `k_v` and using `f(k_v)`, each ternary probe would cost one flow
   evaluation (≈2 tanh — a constant factor), and the minimiser would be the same
   only because f is monotone. Choosing z directly is cheaper and strictly more
   general.

3. **The candidate set itself becomes f-dependent.** Gap widths in z are
   `w = f(x_{i+1}) - f(x_i)`, and the code probes at `lo + w·10⁻³` with ternary
   tolerance `w·10⁻³`. Where f saturates, w underflows to 0 and the gap is
   dropped permanently. So the per-candidate cost is invariant but the *number*
   of candidates is not. This is the anti-alignment of §3d1-d2.

---

## 2. Convergence

### 2.1 Monotone non-increase — true, but only with exact blocks

Trivially: each step minimises L over one block with the others fixed, so
`L(new) ≤ L(old)`; L ≥ 0, so `L_t` is non-increasing and bounded below, hence
convergent. That is the whole proof, and it is not the interesting part.

### 2.2 Two ways the naive prototype breaks it

**(i) `smooth_cdf` is a descent step, not a block minimiser.** It accepts only
strictly improving single insertions (line 88) and halts when none exists
(line 90, CSV Algorithm 1 line 27). It returns a *greedy-stationary* V, not
`argmin over {|V| ≤ λ}`. Non-increase relative to its own starting point still
holds, so this alone does not break monotonicity.

**(ii) `smooth_cdf` cannot warm-start, and this DOES break monotonicity.**
Line 54 always rebuilds the working sequence from the raw feature vector with
V = ∅. So round t+1 gives

```
L(f_{t+1}, V_{t+1}) ≤ L(f_{t+1}, ∅)        but NOT necessarily
L(f_{t+1}, V_{t+1}) ≤ L(f_{t+1}, V_t) = the value at the end of the f-block.
```

A round can therefore *increase* L, and the iteration can oscillate. Fixes, in
order of preference: (a) thread an initial slot map into `smooth_cdf` so it can
resume from V_t; (b) wrap it in a best-so-far guard and only accept an improving
round. (b) is five lines and restores the proof exactly. **Do this before
running anything**, or a negative result will be uninterpretable.

**(iii) The f-block is nonconvex** (tanh), so it reaches a stationary point.
`train_flow.py` already keeps the best iterate, which is enough for non-increase
— but against the NLL, not against L (§3d5).

### 2.3 What it can converge to

`L_t` converges. The **iterates need not**, and when they do, the limit is only a
**block-wise fixed point** (a partial optimum / Nash point): no single insertion
improves given f, and f is stationary given V. This is strictly weaker than a
local minimum of the joint objective, for three independent reasons:

* **The V block is discrete.** The standard BCD guarantees (Bertsekas Prop.
  2.7.1; Tseng 2001) require continuous blocks with unique minimisers, or
  per-block pseudoconvexity. A finite block satisfies none of them, and
  "optimal in each block separately" does not imply "locally optimal jointly" —
  escaping can require moving both blocks at once. Here that is concrete: an
  insertion that is loss-increasing at the current f can be loss-decreasing
  after f bends, and the alternation will never try it.
* **The V block is not even block-optimal**, only greedy-stationary. The loss
  decrement of an insertion set is not submodular in general — inserting in gap
  i shifts *every* later target, so decrements interact non-monotonically — so
  there is no (1−1/e) fallback either.
* **The f block is nonconvex.**

Two further structural remarks. The value function `L*(θ) = min_V L(θ,V)` is a
pointwise minimum over finitely many smooth functions of θ: continuous but
**non-smooth**, with kinks exactly where the greedy's chosen gap set changes. So
a directly joint gradient method would need a subgradient or smoothing; the BCD
formulation is the right one. And by §1.3 the argmin is a continuum, so only
loss convergence is a meaningful stopping test.

**Honest summary: convergence in objective value, to a block-wise fixed point.
Not a local minimum, not a stationary point of the joint problem, and not
monotone at all unless the warm-start defect is fixed.**

---

## 3. Degeneracies

### (a) Unconstrained f — the objective collapses completely

Take f = the empirical rank function, `f(x_i) = i`, extended monotonically
between keys. Then `a = 1, b = 0, V = ∅` gives `L = 0` exactly, with zero budget.
So over all monotone f the minimum is 0 and the virtual points are exactly
redundant.

Two consequences, both important:

* **All the content of the method is in the capacity constraint on f.** The
  joint objective without a capacity bound is vacuous.
* **The degenerate minimiser is the object being built.** An exact monotone map
  from key to rank *is* the index. So "give f more capacity" is not free — it is
  storage, traded against exactly the storage the virtual points cost. There is
  no free lunch here, only an exchange rate (§3b).

A subtler second version: a flow with enough gain approximates a step function
and scores well on the *fences* while being useless *between* them. The
workspace already guards this correctly — `index.hpp:372` scores the root on
fence midpoints as well as fences. Keep that; the joint objective as written
does not contain it.

### (b) At a fixed λ, can virtual points alone do what the bend does?

**With unbounded budget yes, with an explicit exchange rate. At a fixed λ, no.**

L depends on (f, V) only through the composite map key → predicted slot versus
actual slot. f is a monotone reparametrisation of the input axis, V a monotone
reparametrisation of the output axis, and they compose. Let
`ρ = sup f' / avg f'` over `[x_1, x_n]` — the peak densification the flow
performs. To reproduce the same residuals with f = identity you must take slope
`a' = a·sup f'`, so the slot count grows from `a·range·avg f' = n + |V|` to
`a·range·sup f' ≈ ρ(n + |V|)`. Hence

> **Exchange rate: a flow of peak-to-mean slope ratio ρ is worth
> (ρ−1)(n+λ) virtual points.**

So the flow is redundant exactly when the budget is slack, and useful exactly
when it binds. Measured (§4): at λ = 4n the budget is slack on 8 of 10 datasets.

The converse also holds and should be said: **V can realise any non-decreasing
integer step correction** — the whole monotone group, up to integrality —
whereas F is a 3-shape-parameter family. V is capacity-rich / budget-poor;
f is capacity-poor / budget-free. Neither dominates. Which one is binding is an
empirical question with a computable answer (§3c).

### (c) Are they fighting for the same degrees of freedom? Yes — and here is the scalar they are fighting over

For a sorted feature vector z and a target uniform error e define

```
a_min(z,e) = max_{i<j} (j - i - 2e)/(z_j - z_i)
B_e(z)     = a_min(z,e)·(z_n - z_1) - (n-1) - 2e
```

**Lemma (budget lower bound).** If there exist a, b and admissible slot ranks s
with `|a·z_i + b - s_i| ≤ e` for all i, then `|V| ≥ B_e(z)`.

*Proof.* For i<j, `a(z_j - z_i) = (s_j - s_i) + (err_j - err_i) ≥ (j-i) - 2e`
since `s_j - s_i ≥ j - i`. Maximising over pairs gives `a ≥ a_min(z,e)`. And
`|V| = (s_n - s_1) - (n-1) ≥ a(z_n - z_1) - 2e - (n-1)`. ∎

Properties that make `B_e` the right object:

* **Invariant under `z ↦ αz + β`** (α>0) — exactly the invariance of L itself
  (§1.3). It is a property of the *shape* of the warp, nothing else.
* Decreasing in e.
* `B_e(rank feature) < 0` for every e ≥ 0 — which is §3a restated in budget units.

This makes the conflict explicit: **both blocks buy down the same scalar.** The
flow reduces `B_e` by reshaping z; the virtual points pay for whatever is left,
one point per unit. They are not two resources, they are two prices for the same
good. Complementarity is possible **only** in the regime `B_e(x) > λ`; outside
it the flow is strictly redundant. This is the sharpest form of the scaling
argument, and it is directly computable (O(n²) = 120k pairs at n=489).

**A mismatch worth knowing about.** The stated objective is SSE, an L² quantity;
`B_e` is L^∞. Minimising SSE is not minimising `B_e`, and a B-directed search
over the same family finds different (g_1, g_2, c_2/c_1) (`diag3.py`). But do
**not** switch the block objective to `B_e`: the B-directed search on osm
returned an apparently perfect solution (`B_4 = −16`) at `g ≈ 1.2×10³`, which is
a step function that collapses most fences onto one double — degeneracy (d2).
SSE penalises collapse (a collapsed block of m fences costs Θ(m³) in rank
variance); `B_e` does not. **Keep SSE as the block objective; use `B_e` only as a
diagnostic.** That is a point in favour of the proposal as written.

### (d) Order-preserving f versus greedy insertion in z-space

**d1 — Anti-alignment (not a bug, but it reverses the story's sign).** Greedy
insertion fills *sparse* key regions with virtual slots so that slot ∝ feature. A
monotone saturating flow linearises the CDF by *compressing* sparse regions.
These are the same job done from two sides, and doing the first leaves less raw
material for the second. The mechanism hypothesis says the flow "removes global
curvature so the budget goes further"; the composite-map view (§3c) says the
flow *spends* the same budget in a different currency. Both are true; only the
exchange rate decides.

**d2 — Gap annihilation and the tie floor (a real defect).** Where f saturates,
`w = f(x_{i+1}) - f(x_i)` underflows to 0 in double; `smoothing.hpp:76` then
drops the gap permanently, so no budget can ever be spent there. Worse, if m
fences share one double they share one prediction and one slot-table entry, so

> **max error ≥ (m−1)/2 for a tie block of size m, at any λ, forever.**

`index.hpp:382` guards with `std::is_sorted`, which **accepts ties**. Any
implementation of this must enforce *strict* separation in double, not
sortedness. (My own first version of `B_e` had the same blind spot — it skips
pairs with `z_j = z_i` — which is how the osm step-function "solution" got
through; `diag4.py` adds the tie floor. If a bound and the production code share
a blind spot, the bound will certify the bug.)

**d3 — The monotone set is not convex in θ.** With `p_r = 0`, monotonicity is
`c_1 g_1 sech²(g_1(x−μ)) + c_2 g_2 sech²(g_2(x−μ)) > 0` on the range: a union of
sign orthants, not a convex set, so projection is not well defined. The
parameter space is 3-dimensional, so just enumerate the four sign patterns —
free. Note also that `train_flow.py` enforces monotonicity twice and
inconsistently: hard (`g0[2]=g0[3]=0`, line 67) and soft (a `1/d` barrier,
line 53) — and the barrier is evaluated **only at sampled keys**, so a trained
flow can be non-monotone between them. Under (C1)/(C2) that is not acceptable at
the root; check monotonicity on a dense grid, not on the training sample.

**d4 — The shared centre is the binding capacity limit.** There is no bias
inside the tanh (`transform.hpp` computes `acc += a[r]*W0[...]`, no bias term),
and μ is a single scalar shared by both units; `train_flow.py:89` sets
`mean = keys[0]`, so `x ∈ [0, shifts]` and **both tanh units are centred at the
data minimum**. Therefore

```
f'(x) = c_1 g_1 sech²(g_1 x) + c_2 g_2 sech²(g_2 x)
```

is a combination of two even bumps both peaked at x = 0: the density expansion is
**pinned to the left edge**, with at most one interior maximum whose position is
controlled only through the ratio g_1/g_2, and the warp has **at most one
inflection**. This — not the alternation — is what fails on osm (§4, P4).

**d5 — The sequential baseline optimises a different loss (confound).**
`train_flow.py` minimises the NFL negative log-likelihood (change-of-variables
density matching to a standard normal), not L. So "joint vs sequential" as
proposed confounds *two* changes: the alternation, and replacing the NLL with the
downstream least-squares loss. **A three-arm experiment is mandatory:**
(A) NLL-flow → CSV (the true sequential baseline), (B) SSE-flow fitted to plain
ranks → CSV (one pass, new objective, no alternation), (C) alternating. If
B ≈ C, the alternation contributes nothing and the finding reduces to "train the
flow on the downstream loss", which is a much smaller and much older claim.

**d6 — The objective has no space term.** The root slot table is one `uint32`
per slot (`index.hpp:333, 387`), so λ = 4n costs `4(n+λ) ≈ 9.8 KB` at n = 489,
while the flow costs ~10 doubles = 80 bytes plus two tanh per lookup. Since L
has no space term, raising λ weakly always wins and the flow will look
unnecessary for the wrong reason. **Compare at matched bytes** (e.g. joint at
λ/2 against sequential at λ), not at matched λ.

**d7 — The alternation may be a first-order no-op.** The V block changes the
targets from ranks to slot ranks by exactly the residual component the budget
could afford — which is, by construction, the part f had already failed to fit.
If that change lies to first order in the orthogonal complement of the flow's
Jacobian span, the f-block's gradient is unchanged and round 2 returns round 1's
f. The prototype must log ‖θ_t − θ_{t−1}‖ and the loss drop attributable to
rounds ≥ 2.

---

## 4. The sharpest falsifiable prediction

Restate the mechanism hypothesis in the units of §3c: **the flow pays off only
where the virtual budget binds (`B_e(x) > λ`), and only in proportion to how much
it reduces `B_e`.**

Define `e*(z, λ) = min{ e : B_e(z) ≤ λ }`, tie-aware — a **floor** on the max root
error achievable at budget λ. Measured at n = 489 fences, λ = 4n = 1956, with the
best strictly-monotone in-family flow found by a 48×48 gain grid with an exact
inner 3×3 solve (`diag4.py`):

| dataset | e*(raw) | e*(best monotone flow) | in-family rms | budget binding? |
|---|---|---|---|---|
| **osm**    | **80.1** | **69.7** | 53.9 (line: 59.0) | yes — flow barely dents it |
| **planet** | **21.2** | **~0**   | 9.10 (line: 72.8) | yes — flow removes it entirely |
| books  | ~0 | ~0 | 2.40 | no |
| fb, covid, genome, history, libio, stack, wise | ~0 | ~0 | — | no |

`~0` means *the bound is vacuous* — the budget is not the binding constraint. It
does **not** mean the achieved error is zero. `e*` is a lower bound only.

Probe conversion: `locate_from_prediction` does exponential search then binary
search, so probes ≈ `2·log₂(error) + 2`. **Halving the error saves ~2 probes.**

### P1 — primary prediction

At λ = 4n, a joint method beats sequential by more than 0.5 root probes on
**planet** and on no other dataset.

* **planet**: CSV-alone root probes 5.14. Predict joint ≤ 3.0.
  **If joint does not improve planet by at least 1.5 probes, the mechanism
  hypothesis is refuted.**
* **osm**: CSV-alone 8.18 (binary 8.96). The flow cuts the error floor by 13%,
  ceiling `2·log₂(80.1/69.7) = 0.40` probes. Predict improvement < 0.5.
  **If joint improves osm by more than 1 probe, my §3c analysis is wrong** —
  and that would be the more interesting outcome, so measure it.
* **the other eight**: predict |Δ| ≤ 0.1 probes. Any gain above 0.2 there is not
  coming from the budget mechanism and needs a different explanation.

Note this **contradicts the framing in the brief**, which expects the pay-off on
osm *and* planet because those are where NFL's conflict-degree reduction is
largest. Half of that expectation survives: planet yes, osm no. The conflict
degree on osm drops 99 → 6 inside regions, but at the root osm needs ≈185,000
virtual fences to reach max error 1 against a budget of 1,956, and the monotone
flow family only brings that to ≈110,000. Two orders of magnitude short.

**Weakest link, stated plainly:** `e*` bounds the *max* error, while the measured
root probes are a *mean* over fences and fence midpoints. A vacuous bound does not
prove the mean cannot improve, and a binding bound constrains the tail more than
the mean. P1 converts a max-error bound into a mean-probe prediction, and that
step is the one most likely to be wrong.

### P2 — cheap pre-check, run this first, costs nothing

Report `SmoothingResult.rounds` against the budget for the root. The greedy stops
early (`rounds < λ`, line 90) exactly when the budget is not binding. Predict:
early stop on all eight flat datasets; budget exhausted on osm and planet. If CSV
exhausts the budget where I predict a vacuous bound, either the bound is being
computed on the wrong feature or the greedy is far from block-optimal — find out
before building anything.

### P3 — no-op falsifier

Log ‖θ_t − θ_{t−1}‖ and ΔL per round. Predict round 2 changes nothing on the eight
flat datasets. On planet, if round 2's loss drop is under 5% of round 1's, the
correct conclusion is *"train the flow on the least-squares loss instead of the
NLL"* (arm B of §3d5), not *"optimise jointly"*.

### P4 — the better lead, measured

Allowing a **bias inside each tanh** (two independently placed bends: +2 weights,
16 bytes) versus the deployed shared-centre family, on osm:

| | rms on fences | e*(λ=1956) |
|---|---|---|
| plain line | 59.0 | 80.1 |
| best monotone in-family (shared centre) | 53.9 | 69.7 |
| **best monotone with per-unit bias** | **36.2** | **27.6** |

That is a 2.5× cut in the error floor, worth ≈ `2·log₂(2.5) = 2.6` root probes of
headroom — **six times the entire headroom the biasless family offers.**

*Honest caveat:* the bias search used a coarser gain grid (12 points, 10⁻¹..10⁴)
than the biasless search (48 points, 10⁻²..10⁴), which is why planet's biased
number came out *worse* (e* 4.57 vs ~0) — a grid artifact, not a regression. The
osm comparison survives a fortiori: a **coarser** grid found a strictly better
solution, so the shared-centre family genuinely cannot reach it.

**If there is time for one experiment, it should be "add a tanh bias" before
"alternate the blocks."**

---

## 5. Reconciling this with the tail-conflict-degree null result

Three sentences Louis can say out loud:

> The sweep showed a 2×2×2 order-preserving flow cannot reduce the tail conflict
> degree at any training budget, and that result stands — the conflict degree is
> measured against a line refitted in z, so it is invariant to `z → αz + β`, and
> a single monotone bend that only rescales the axis leaves it untouched.
>
> But the conflict degree is a *local* statistic: the 99th percentile of how many
> keys land in one predicted bucket. The root probe count is a *global* one: it
> tracks the whole residual curve, and it is not affine-invariant in the same way,
> because the slot targets are integers fixed by the virtual points while the
> feature axis is not. A single global bend is exactly the wrong tool for the
> first quantity and exactly the right shape for the second.
>
> So the flow was shown powerless on local density and remains untested on global
> curvature — and the measurement above says the global curvature it *can* remove
> matters on planet, is 100× too small on osm, and is irrelevant on the other
> eight because the virtual-point budget there is not binding in the first place.
