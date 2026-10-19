# Why transform keys before a learned index: definitions, theorems, proofs

This note answers one question: **why would one apply a monotone transform to the keys before fitting a learned
index, and when should one not?** The transform in question is NFL's distribution transformation (Wu et al., *NFL:
Robust Learned Index via Distribution Transformation*, PVLDB 15(10), 2022, arXiv 2205.11807). It is a normalizing flow
applied to the keys before the index. In this repo, SCALE-LI's cell N applies an NFL-style flow
(`src/scaleli/transform.hpp`, weights in `data/flows/`); cell B is the same index without it.

The paper argues by design and experiment and contains no theorem. Sections 2 to 6 supply the mathematics its argument
relies on. Every statement is numbered and proved here. Section 7 checks the theory against the NF-gate study
(`results/warmup_nfgate_2026-10-17/NFGATE.md`). Section 8 says exactly what is machine-checked in Lean
(`docs/proof/lean`) and what enters as an explicit hypothesis.

**Machine-checked.** Each numbered result below carries a line *Lean:* naming its counterpart in
`docs/proof/lean/NFLProof/`. All of them compile against Mathlib `v4.35.0` with no `sorry`, no `admit` and no
`axiom` of ours, and `NFLProof/Axioms.lean` checks during the build that each theorem and lemma depends only on
`propext`, `Classical.choice` and `Quot.sound`. `NFLProof/Sanity.lean` instantiates every hypothesis set on small
concrete inputs, so no theorem is vacuous (Section 8.6). Pinsker's inequality, absent from Mathlib, is *proved* for
finite distributions (`NFLProof/Pinsker.lean`). The only classical result taken as a hypothesis is the DKW inequality,
which enters as the explicit premise `hDKW` of the two probabilistic statements (Theorem 5.4(b) and Theorem 5.5(b)).

**The answer in five lines.**
1. A strictly increasing transform costs nothing in correctness: ranks, lower bounds and ranges on the transformed
   keys equal those on the keys (Theorem 2.1). Monotonicity is also *necessary* for ranks to be preserved
   (Theorem 2.2).
2. The best single linear model on the transformed keys has rank error equal to $n$ times the transformed keys'
   distance to the nearest linear CDF, within one rank (Theorem 3.1). The transform helps exactly when it lowers
   that distance (Theorem 3.5). The ideal transform is the empirical CDF itself, the probability integral transform,
   which gives error $\le 1$ (Theorem 3.3). Evaluating that transform is the indexing problem itself.
3. Search cost is logarithmic in the error and conflict degree is linear in it (Theorems 4.1 to 4.6). Lowering the
   distance therefore lowers both.
4. A KL statistic on $m$ sampled keys bounds the transformed error, with deterministic or DKW confidence (Theorem 5.4).
   The bound loses a square root through Pinsker's inequality (Theorem P, proved in Lean for finite distributions),
   and it says nothing about the untransformed incumbent. So the decision must compare costs, including the
   transform's own compute (Theorem 5.5).
5. A transform of finite resolution (affine, or nearly affine, on each of $K$ pieces) cannot remove the
   irregularity *inside* a piece (Theorems 6.1 to 6.4). This is the formal side of the repo's finding that 67-99% of
   log-gap variance is fine-scale.

---

## 1. Setting and the paper

### 1.1 Notation (aligned with the paper where it has notation)

| here | paper | meaning |
|---|---|---|
| $x_1 < x_2 < \dots < x_n$ | $\mathcal X = \lbrace x_1,\dots,x_n\rbrace$, $N$ | the sorted, distinct keys ($n = N$) |
| $r_X(q) = \#\lbrace i : x_i \le q\rbrace$ | position | rank of a query; $r_X(x_i) = i$ (1-indexed) |
| $\ell_X(q) = \#\lbrace i : x_i < q\rbrace$ | | lower-bound slot (0-indexed) |
| $F_X(t) = r_X(t)/n$ | $F$, eq. (1) | empirical CDF; the paper's $\mathrm{pos} = F(\mathrm{key})\cdot N$ |
| $T$, $z_i = T(x_i)$, $Z = T(X)$ | the normalizing flow (NF), transformed key $z$ (Sec. 2.3, 3.1) | the transform (flow) and the transformed keys |
| $u_i = (z_i - z_{\min})/(z_{\max}-z_{\min})$ | | min-max normalized transformed keys in $[0,1]$ |
| $U(t) = \min(\max(t,0),1)$ | "near-uniform" target | the uniform CDF on $[0,1]$ |
| $L(t) = \alpha t + \beta$ | $\mathcal M$ | an affine model in CDF units; predicted position $nL(z)$ |
| $D^j_{\mathcal M}$, $D^\gamma_{\mathcal M}$ | Def. 3.1, 3.2 | conflict degree of slot $j$; tail conflict degree ($\gamma = 0.99$) |

Throughout, $n \ge 2$, and $\#S$ is the number of elements of a finite set $S$. Keys are real numbers. The repo's keys
are `uint64`, and the flow sees them as `double`, which is discussed where it matters (Remark 2.4).

### 1.2 The paper's argument, and what it leaves unproved

The following paraphrases the paper section by section.
- **Motivation (Sec. 2.1-2.2).** A learned index approximates the CDF, so that the position is $F(\mathrm{key})\cdot N$.
  Existing indexes split the CDF into pieces until each piece is roughly linear, that is, locally near-uniform. This
  costs depth, models and tuning. If the keys were globally near-uniform, one linear model would suffice, so NFL
  transforms them first. Table 1 shows ALEX with and without the flow: on FB the tree height drops from 11 to 3.
- **Transformation (Sec. 2.3, 3.2).** A normalizing flow is trained by maximum likelihood (eqs. 2-3) toward a
  near-uniform latent: a normal distribution with a very large variance (Sec. 3.2; $10^{16}$ in Sec. 4.1), because a
  uniform target produces NaN or INF losses. Keys are min-max scaled and expanded into integer and fractional parts
  (Sec. 3.2.1, Algorithm 3.1). A small network (2 layers, 2 inputs, 2 hidden units; Sec. 4.1) maps them, and a "sum"
  decoder merges the output back to one dimension (Algorithm 3.1). The flow is invertible as a $d$-dimensional map;
  the one-dimensional composite after the sum decoder need not be monotone (this is our observation, not the paper's;
  Section 7 measures that it is not).
- **Quality metric (Sec. 3.1.1).** Def. 3.1 counts the keys a fitted linear model sends to slot $j$ (the conflict
  degree). Def. 3.2 takes the $\gamma = 0.99$ quantile of the nonzero conflict degrees (the tail conflict degree).
- **Switching (Sec. 3.2.2).** NFL transforms the bulk-load keys and compares the tail conflict degree before and after.
  It disables the flow when the transformed tail conflict degree is larger. Sec. 4.2 reports that it is disabled on
  YCSB, AMZN and WIKI, and that on LTD and LGN the flow lowers throughput because of its inference cost.
- **After-flow index (Sec. 3.3).** AFLI places keys exactly in model nodes. A slot with a few conflicting keys becomes
  a bucket (capacity tied to the tail conflict degree, at most 6 in the experiments, Sec. 4.1), and dense conflicts
  become child or dense nodes.

None of this is stated as a theorem. Four points are implicit and need proof or qualification:
(a) that a transform preserves what the index must answer (Section 2);
(b) that "near-uniform" quantitatively means "small linear-model error" (Section 3);
(c) that conflicts and search costs follow the error (Section 4);
(d) that a cheap statistic can decide when to transform (Section 5).
A fifth point is the paper's remark that the flow's quality has "almost reached to its upper limit" (Sec. 4.4.1).
Section 6 explains why such a limit must exist. Throughout, "Sec." cites the paper and "Section" this note.

---

## 2. Correctness (task 1)

**Definition 2.1.** $T : \mathbb R \to \mathbb R$ is *strictly increasing on* $S$ if $s < s'$ in $S$ implies
$T(s) < T(s')$.

**Theorem 2.1 (order preservation).** Let $T$ be strictly increasing on a set containing the keys and the query $q$.
Then
$$r_{T(X)}(T(q)) = r_X(q), \qquad \ell_{T(X)}(T(q)) = \ell_X(q),$$
and for queries $q_1 \le q_2$ the range $\lbrace i : q_1 \le x_i \le q_2\rbrace$ equals
$\lbrace i : T(q_1) \le T(x_i) \le T(q_2)\rbrace$. Moreover $T(x_1) < \dots < T(x_n)$, so $T$ restricted to $X$ is a
bijection onto $T(X)$ that preserves order, and its inverse is $T(x_i) \mapsto x_{r_{T(X)}(T(x_i))}$.

*Proof.* For each $i$: if $x_i \le q$ then $T(x_i) \le T(q)$ (equality if $x_i = q$, strict increase otherwise).
Conversely, if $x_i > q$ then $T(x_i) > T(q)$. Hence $x_i \le q \iff T(x_i) \le T(q)$, and the index sets counted by
$r_X(q)$ and $r_{T(X)}(T(q))$ coincide. The same argument with strict inequalities gives $\ell$, and applying it to
both ends gives ranges. Strict increase on $x_1 < \dots < x_n$ gives $T(x_1) < \dots < T(x_n)$. Then
$r_{T(X)}(T(x_i)) = i$, which proves the inverse formula. $\square$

*Lean:* `rank_comp_strictMono` (ranks and lower bounds), `range_comp_strictMono` (ranges), `strictMono_comp`
(sortedness), `rank_of_transformed` and `rankLE_self` (the inverse on the keys); files `Correctness.lean`, `Defs.lean`.
Lean assumes `StrictMono T` on all of $\mathbb R$, a special case of "strictly increasing on a set containing the keys
and $q$"; the proof uses only the keys and $q$.

**Theorem 2.2 (monotonicity is necessary for ranks).** For any $T$, with $z_i = T(x_i)$:
$$\bigl(\forall i:\ r_{Z}(z_i) = i\bigr) \iff z_1 < z_2 < \dots < z_n .$$

*Proof.* ($\Leftarrow$) is the last step of Theorem 2.1. ($\Rightarrow$) Let $i < j$ and suppose $z_j \le z_i$.
Every $k$ with $z_k \le z_j$ also has $z_k \le z_i$. So $j = r_Z(z_j) \le r_Z(z_i) = i$, a contradiction.
Hence $z_i < z_j$. $\square$

*Lean:* `ranks_preserved_iff`.

**Proposition 2.3 (point lookups need only injectivity).** If $T$ is injective on $X \cup \lbrace q\rbrace$, then
$T(q) = T(x_i) \iff q = x_i$. So an index keyed by $T(x)$ answers point lookups correctly. By Theorem 2.2, it answers
rank and range queries correctly only if $T$ is increasing on the keys. *Proof:* injectivity. $\square$

*Lean:* `point_lookup`.

**Remark 2.4 (the two architectures).**
- **NFL** builds its index on the transformed keys (`transform.hpp` notes that it sorts by transformed key). For NFL,
  point lookups need injectivity and rank or range answers need monotonicity.
- **SCALE-LI** keeps records in *original key order* and uses $T(x)$ only as the model's feature. Every prediction
  is then corrected by an exact search on the original keys (`Index::locate_from_prediction`, `Region::locate_block`).
  That search returns the true answer from *any* starting point (Theorem 4.3), so correctness never depends on $T$:
  not on its monotonicity, and not on the precision lost when keys above $2^{53}$ become `double`. Only the *cost*
  depends on $T$, through the distance between prediction and answer. Sections 3 and 4 quantify it, and Theorem 3.6
  covers a non-monotone $T$.

---

## 3. Error (task 2)

Assume $T$ is strictly increasing (Theorem 3.6 drops this). Then $z_1 < \dots < z_n$, $u_1 = 0$, $u_n = 1$, and
$F_u(t) = \#\lbrace i : u_i \le t\rbrace / n$ is the empirical CDF of the normalized transformed keys.

**Definition 3.1 (Kolmogorov distance).** $D_{KS}(F,G) = \sup_{t\in\mathbb R} |F(t) - G(t)|$. For $F_u$ and $U$ the
supremum may be taken over $[0,1]$: for $t < 0$ both are 0, and for $t \ge 1$ both are 1.

*Lean:* `ksUnif u`, defined as a maximum over the $2n$ one-sided values $|(i+1)/n - U(u_i)|$ and $|i/n - U(u_i)|$ at
the keys. `ecdf_sub_unif_le_ksUnif` and `ksUnif_le` prove that this finite maximum *is* $\sup_t |F_u(t) - U(t)|$
(`Error.lean`). Keys are `u : Fin n → ℝ` with `StrictMono u`; `rankLE`, `rankLT`, `ecdf`, `unifCDF` are in `Defs.lean`.

**Definition 3.2 (model errors).**
- Endpoint model (the min-max line): predicted position $n u_i$, with error
  $\varepsilon_U = \max_i |n u_i - i|$.
- Affine model $L$ on keys $Y = (y_i)$: error $\varepsilon_Y(L) = \max_i |n L(y_i) - i|$. The best error is
  $\varepsilon^*(Y) = \inf_L \varepsilon_Y(L)$, the infimum over all affine $L$.
- **Linear Kolmogorov distance:**
  $D_{\mathrm{lin}}(Y) = \inf_L \sup_{t \in [y_1, y_n]} |F_Y(t) - L(t)|$, the distance from the empirical CDF to the
  nearest linear CDF on the key range.

**Theorem 3.1 (error = n times distance, within one rank).**
$$\text{(a)}\quad \varepsilon_U \;\le\; n\,D_{KS}(F_u, U) \;\le\; \varepsilon_U + 1,$$
$$\text{(b)}\quad n\,D_{\mathrm{lin}}(Z) - 1 \;\le\; \varepsilon^*(Z) \;\le\; n\,D_{\mathrm{lin}}(Z) \;\le\; n\,D_{KS}(F_u, U).$$

*Proof.* (a), left inequality: since the $u_i$ are distinct, $F_u(u_i) = i/n$. So
$|n u_i - i| = n|U(u_i) - F_u(u_i)| \le n D_{KS}$.

(a), right inequality: let $t \in [u_i, u_{i+1})$ for some $1 \le i \le n-1$. There $F_u(t) = i/n$, and
$t \mapsto i/n - t$ is monotone. So
$|F_u(t) - t| \le \max(|i/n - u_i|, |i/n - u_{i+1}|)$. Moreover
$|i/n - u_{i+1}| \le |(i+1)/n - u_{i+1}| + 1/n$. Both terms are therefore at most $(\varepsilon_U + 1)/n$. At $t = 1$
the difference is 0, and outside $[0,1]$ it is 0 by Definition 3.1.

(b): repeat (a) with an arbitrary affine $L$ in place of $U$, on $[z_1, z_n]$. An affine function minus a constant is
still monotone, so the same argument gives
$\varepsilon_Z(L) \le n \sup_{[z_1,z_n]}|F_Z - L| \le \varepsilon_Z(L) + 1$. Take the infimum over $L$. The last
inequality holds because $t \mapsto U((t - z_1)/(z_n - z_1))$ is one admissible $L$, and
$F_Z(t) = F_u((t - z_1)/(z_n - z_1))$. $\square$

*Lean:* (a) `ks_sandwich` (with `endpointErr u` $= \varepsilon_U$); (b) for each affine $L$, `err_le_of_ks` (upper) and
`ks_le_of_err` (lower), on the key range. The infimum over $L$ is taken on paper.

**Theorem 3.2 (slots: the "+1").** The endpoint model's 0-indexed slot satisfies
$$\bigl|\lfloor n u_i \rfloor - (i - 1)\bigr| \;\le\; n\,D_{KS}(F_u, U) + 1 .$$

*Proof.* $\lfloor y \rfloor \in (y - 1, y]$, so $\lfloor n u_i\rfloor - (i-1) \in (n u_i - i,\ n u_i - i + 1]$.
By Theorem 3.1(a) this lies in $(-nD_{KS},\ nD_{KS} + 1]$. $\square$

*Lean:* `slot_err` (any CDF-scale model $G$ within $D$ of $F_u$), `slot_err_ksUnif` ($G = U$, $D = D_{KS}$).

**Theorem 3.3 (the probability integral transform is optimal).**
- (a) For every strictly increasing $T$: $D_{KS}(F_u, U) \ge 1/n$. More generally, if some $u_{i_0} = 0$ and a CDF $G$
  has $G(0) = 0$, then $|F_u(0) - G(0)| \ge 1/n$.
- (b) Take the empirical CDF itself, $T = F_X$, so that $z_i = i/n$ and $u_i = (i-1)/(n-1)$. Then
  $D_{KS}(F_u,U) = 1/n$ (the minimum in (a)) and $\varepsilon_U = 1$. The affine model $L(z) = z$ has
  $\varepsilon_Z(L) = 0$.
- (c) $\varepsilon_Z(L) = 0$ for some affine $L$ if and only if there are $c \neq 0$ and $d$ with $z_i = c\,i + d$ for
  all $i$. That is, $T$ agrees on the keys with an affine function of the rank, or equivalently of $F_X$.

*Proof.*
- (a) $u_{i_0} = 0 \le 0$, so $F_u(0) \ge 1/n$, while $G(0) = 0$. For $U$, take $i_0 = 1$.
- (b) For $t \in [u_i, u_{i+1})$, $F_u(t) = i/n$ and $t \in [\tfrac{i-1}{n-1}, \tfrac{i}{n-1})$. So $i/n - t$ lies
  between $-\tfrac{i}{n(n-1)}$ and $\tfrac{n-i}{n(n-1)}$, and both have absolute value at most $1/n$; the value $1/n$
  is attained at $t = u_1 = 0$. Next, $|n u_i - i| = \tfrac{n-i}{n-1} \le 1$, with equality at $i = 1$. Finally
  $nL(z_i) = n \cdot i/n = i$.
- (c) If $n(\alpha z_i + \beta) = i$ for all $i$, then $\alpha \neq 0$ because $n \ge 2$ forces two different targets.
  Solve for $z_i$. Conversely, $L(z) = (z - d)/(nc)$ works. $\square$

*Lean:* (a) `ks_ge_inv_n`, `ks_ge_inv_n_of_normalized`; (b) `pit_ks`, `pit_ksUnif_eq` ($D_{KS} = 1/n$ exactly),
`pit_endpoint_err`, `pit_endpointErr_le` ($\varepsilon_U \le 1$), `pit_exact`; (c) `zero_error_iff`.

**Remark 3.4 (circularity).** By Theorem 3.3(c), a transform that makes the linear model exact *is* the rank function on
the keys. Evaluating it at an arbitrary query is the lookup itself; the paper says as much when it notes that the
transformed keys cannot be stored without creating another indexing problem (Sec. 3.2.2). A transform is therefore
worth having only if it is cheaper to evaluate than the index structure it replaces. That is the compute term $c_T$ in
Theorem 5.5.

**Theorem 3.5 (affine invariance; when a transform helps).**
- (a) For $\alpha \neq 0$ and any $\beta$, the keys $y_i$ and $\alpha y_i + \beta$ (each keeping its index $i$) have
  the same set of achievable errors $\lbrace \varepsilon(L)\rbrace$, hence the same $\varepsilon^*$. For $\alpha > 0$
  they also have the same $D_{\mathrm{lin}}$, and min-max normalization does not change $u$. So rescaling cannot help;
  only a non-affine $T$ can. (For $\alpha < 0$ the order reverses and $D_{\mathrm{lin}}$ itself can change, by at most
  $1/n$ by Theorem 3.1(b): $Y = \lbrace 0, 1, 100\rbrace$ has $D_{\mathrm{lin}} = 1/6$, while $-Y$ has
  $D_{\mathrm{lin}} \ge 0.33$.)
- (b) (Corollary of 3.1(b).) If $n D_{\mathrm{lin}}(T(X)) < n D_{\mathrm{lin}}(X) - 1$, then
  $\varepsilon^*(T(X)) < \varepsilon^*(X)$.
- (c) If $n D_{\mathrm{lin}}(T(X)) - 1 \ge n D_{\mathrm{lin}}(X)$, then $\varepsilon^*(T(X)) \ge \varepsilon^*(X)$.
- (d) In particular, $n D_{KS}(F_{u}, U) < n D_{\mathrm{lin}}(X) - 1$ is sufficient for the transform to reduce the best
  single-model error.

*Proof.*
- (a) $L \mapsto L \circ A^{-1}$, with $A(y) = \alpha y + \beta$, is a bijection of affine maps, and it preserves every
  value $L(y_i)$. For $\alpha > 0$, $A$ is increasing, so $F_{A(Y)} = F_Y \circ A^{-1}$, and the supremum of
  $|F_{A(Y)} - L \circ A^{-1}|$ over $[A(y_1), A(y_n)]$ is that of $|F_Y - L|$ over $[y_1, y_n]$.
- (b) and (c) chain $\varepsilon^*(T X) \le n D_{\mathrm{lin}}(TX)$ with $\varepsilon^*(X) \ge n D_{\mathrm{lin}}(X) - 1$,
  and the mirror chain.
- (d) $D_{\mathrm{lin}} \le D_{KS}$ by 3.1(b). $\square$

*Lean:* (a) `affine_invariance` (the sets of errors; the $D_{\mathrm{lin}}$ part is on paper); (b) and (d)
`transform_helps`, stated with the distance bounds as hypotheses (raw keys at distance $\ge D_X$ from every affine
CDF, transformed keys within $D_Z$ of one, $nD_Z < nD_X - 1$). `Sanity.lean` satisfies these hypotheses on nine keys.
(c) is the mirror chain and is not separately formalized.

The untransformed side of the comparison is therefore $n D_{\mathrm{lin}}(X)$: $n$ times the distance from $F_X$ to the
best linear CDF. Theorem 3.5(d) is the precise form of the paper's motivation: a transform helps a single linear model
when it makes the keys closer to uniform *than the raw keys are to their best line*.

**Theorem 3.6 (non-monotone transforms: the displacement term).** Let $T$ be arbitrary, $z_i = T(x_i)$,
$\sigma(i) = r_Z(z_i)$ (the rank of $z_i$ among the $z$'s), and $\tilde u$ the min-max normalized $z$'s. Then for every $i$:
$$\bigl|\sigma(i) - i\bigr| - nD_{KS}(F_{\tilde u}, U) \;\le\; \bigl|n\tilde u_i - i\bigr| \;\le\; nD_{KS}(F_{\tilde u}, U) + \bigl|\sigma(i) - i\bigr| .$$

*Proof.* $F_{\tilde u}(\tilde u_i) = \sigma(i)/n$ by definition, so $|n\tilde u_i - \sigma(i)| \le nD_{KS}$. Apply the
triangle inequality in both directions. $\square$

*Lean:* `err_le_ks_add_disp` (upper), `disp_le_err_add_ks` (lower). No monotonicity is assumed.

Theorem 3.6 separates two things the flatness statistics conflate. $D_{KS}(F_{\tilde u}, U)$ depends only on the *set* of
transformed values, which is what a histogram, a KL divergence or a KS distance of the sorted flow output measures. The
displacement $|\sigma(i) - i|$ measures how badly $T$ scrambles key order. For SCALE-LI, which predicts *key* ranks from
$T(x)$, the error is governed by the sum of the two. For NFL, which indexes $T(x)$ itself, only the first term matters.

**Lemma 3.7 (clamping never hurts).** If $lo \le r \le hi$, then $|\max(lo, \min(y, hi)) - r| \le |y - r|$.
*Proof:* clamping is the projection onto $[lo, hi]$, which contains $r$; check the three cases. $\square$
(`locate_from_prediction` clamps predictions to $[0, n-1]$. This is why a few huge outlier keys, as in fb, cost the raw
model almost nothing even though they push $D_{KS}$ of the min-max normalized keys to $\approx 1$. See Section 7.)

*Lean:* `clamp_err`.

---

## 4. Cost (task 3)

**Definition 4.1.** A *comparison search* asks questions "is $x_k < q$?" adaptively and outputs a position. Its
worst-case cost is the maximum number of questions. Modelled as a binary decision tree, this is the depth.

**Theorem 4.1 (binary search on a window).** If the answer is known to lie among $L \ge 1$ consecutive candidates,
binary search needs at most $\lceil \log_2 L \rceil$ comparisons. With an error bound $e$, the window
$[p - e, p + e]$ has $L = 2e + 1$, so the cost is $\lceil \log_2(2e+1)\rceil$, nondecreasing in $e$. The same holds for
the repo's `last_true(lo, hi)` with $L = hi - lo$.

*Proof.* A probe splits $L$ candidates into parts of sizes $\lfloor L/2 \rfloor$ and $\lceil L/2 \rceil$. The worst case
satisfies $C(1) = 0$ and $C(L) = 1 + C(\lceil L/2\rceil)$ for $L \ge 2$. For $L \ge 2$ we have
$\lceil \log_2 L\rceil = 1 + \lceil \log_2 \lceil L/2\rceil\rceil$, because $2^{k-1} < L \le 2^k$ iff
$2^{k-2} < \lceil L/2\rceil \le 2^{k-1}$. Strong induction gives $C(L) = \lceil\log_2 L\rceil$. Monotonicity:
$L \mapsto \lceil \log_2 L\rceil$ is nondecreasing. $\square$

*Lean:* `lastTrue` (the repo's `last_true` with a probe counter), `lastTrue_spec` (invariant, probes
$\le$ `Nat.clog 2 (hi - lo)`; no monotonicity needed), `lastTrue_eq` (returns the threshold), `window_cost_mono`.

**Theorem 4.2 (no comparison search does better).** A binary decision tree of depth $d$ has at most $2^d$ leaves. So a
comparison search that can return any of $L$ distinct answers needs depth $\ge \lceil \log_2 L \rceil$.
*Proof:* induction on the tree, $\#\mathrm{leaves}(\mathrm{node}(l, r)) \le 2^{d(l)} + 2^{d(r)} \le 2^{d}$; then
$2^d \ge L$ iff $d \ge \lceil \log_2 L\rceil$. $\square$
A linear scan from the prediction, as in AFLI's buckets or a gapped layout, costs at most $e + 1$ comparisons
(one to pick the direction, then at most $e$ steps). That cost is also nondecreasing in $e$.

*Lean:* `DTree`, `DTree.card_leaves_le`, `DTree.clog_le_depth`. The linear-scan remark is not formalized.

**Theorem 4.3 (the repo's exponential search: correct from any start, and its exact cost).**
`locate_from_prediction(ok, y, n)` clamps $y$ to $p \in [0, n-1]$. It then gallops away from $p$ in steps
$1, 2, 4, \dots$ and finishes with `last_true`. For a monotone predicate (true up to the answer $t$, false after) it
returns $t$ for every $p$. Let $d = |p - t|$. When the gallop stays inside $[0, n)$ (untruncated), the number of
probes is:

| case | probes |
|---|---|
| $d = 0$ | 2 |
| forward, $d \ge 1$ | $2\lceil \log_2(d+1)\rceil + 1$ |
| backward, $d \ge 2$ | $2\lceil \log_2 d\rceil + 1$ |
| backward, $d = 1$ | 2 |

Hence for every $d$ (the lower bound in the untruncated case):
$$2\log_2(d+1) - 1 \;\le\; \mathrm{cost}(d) \;\le\; \min\bigl(2\lceil\log_2(d+1)\rceil + 2,\ 2\log_2(d+1) + 3\bigr).$$
Truncation at 0 or $n$ only removes probes, so the upper bound always holds.

*Proof.*
- **Forward** ($t > p$). One probe at $p$ (true). The gallop probes $p + 2^j$ for $j = 0, 1, \dots$ until the first
  false. That happens at $j = k := \lceil \log_2(d+1)\rceil$, the least $j$ with $2^j > d$: $k + 1$ probes.
  `last_true` then runs on $(p + 2^{k-1}, p + 2^k)$, an interval of length $2^{k-1}$, which takes $k - 1$ probes.
  Total $2k + 1$. For $d = 0$: the probes at $p$ and $p+1$, then nothing.
- **Backward** ($t < p$). One probe at $p$ (false). The gallop probes $p - 2^j$ until the first true, at
  $j = \lceil\log_2 d\rceil$. Then `last_true` runs on an interval of length $2^{j-1}$. Total $2j + 1$ for $j \ge 1$,
  and 2 for $d = 1$.
- **Correctness.** `last_true` keeps the invariant $\mathrm{ok}(lo) \wedge \neg\mathrm{ok}(hi)$ and halves
  $hi - lo$ down to 1. The gallop establishes the invariant around $t$.
- **Bounds.** Take the case formulas, use $x \le \lceil x\rceil < x + 1$, and use $\log_2(d+1) \le \log_2 d + 1$ for
  $d \ge 1$.
- **Truncation.** When `hi` reaches $n$, or `lo` reaches 0, the gallop stops without that probe, and the remaining
  interval is no longer than in the untruncated case. $\square$

*Lean:* `gallop` transcribes `locate_from_prediction` (table-free branch) with `gallopUp` and `gallopDown` for the two
loops (`Gallop.lean`). `gallopUp_spec`, `gallopDown_spec` are the loop invariants; `gallop_core` gives correctness
and the probe count in terms of the stopping exponent $J$; `gallop_spec` is the bound $2\lceil\log_2(d+1)\rceil + 2$ and
`gallop_cost_logb` the bound $2\log_2(d+1) + 3$, both including truncation, for every start $p$;
`gallop_bound_mono` is monotonicity in $d$. The exact case table and the lower bound $2\log_2(d+1) - 1$ are not
formalized (Corollary 4.5(b) takes the lower bound as a premise). Because `gallop_spec` holds for every start, it also
covers the table branch of `locate_from_prediction`, whose start is `table[slot]` instead of the clamped prediction.
The transcription itself is checked against the C++: `docs/proof/check/gallop_crosscheck/run.sh` compiles the repo's
`last_true` and `locate_from_prediction` verbatim, and runs them and the Lean `gallop` on all 116,795 cases
$(n, p, t)$ with $n \le 70$; answers and probe counts are identical.

**Corollary 4.4 (mean cost, Jensen).** For lookups with distances $d_1, \dots, d_Q$ and mean $\bar d$:
$$\frac1Q\sum_j \mathrm{cost}(d_j) \;\le\; 2\log_2(1 + \bar d) + 3 .$$
*Proof:* the upper bound of Theorem 4.3, plus concavity of $\log_2(1 + \cdot)$ (Jensen). $\square$
With uniform lookups over keys, $\bar d$ is the mean rank error. It is at most $\varepsilon \le n D_{KS}$ (Theorem 3.1),
and for the endpoint model it is within one rank of $n$ times the repo's `w1` statistic (on the full key set).

*Lean:* `mean_cost_jensen` (the Jensen step), `mean_cost_le` (any costs below the per-lookup bound),
`gallop_mean_cost` (the repo's search, unconditionally).

**Corollary 4.5 (when a model beats binary search).** Let binary search over the same $R$ candidates cost $B_R$
probes per lookup ($\approx \log_2 R$).
- (a) **Sufficient:** if $\bar d < 2^{(B_R - 3)/2} - 1$, the model's mean cost is below $B_R$.
- (b) **Necessary (untruncated):** the model can win only if $\frac1Q\sum_j \log_2(d_j + 1) < (B_R + 1)/2$.

*Proof:* Corollary 4.4 for (a), and the lower bound of Theorem 4.3 for (b). $\square$
For SCALE-LI's root, $R = 48{,}829$ fences at 200M keys, and fit_root measures $B_R = 15.66$ (nf_gate's proxy uses
$\lceil\log_2 R\rceil = 16$). So (a) needs $\bar d < 79$ fences, that is 0.16% of $R$, and (b) needs the geometric
mean of $d + 1$ to be below $2^{8.33} \approx 322$ fences.

*Lean:* (a) `beats_binary_of_mean_dist`; (b) `win_needs_small_log_dist`, with the per-lookup lower bound as premise
`hlow`.

**Theorem 4.6 (conflict degree is bounded by the error).** Let $M$ be any real-valued model and $\varepsilon$ its
maximum error, $|M(z_i) - i| \le \varepsilon$ for all $i$. Then every slot $j \in \mathbb Z$ satisfies
$$D^j_{\mathcal M} = \#\lbrace i : \lfloor M(z_i)\rfloor = j\rbrace \;\le\; \lfloor\varepsilon\rfloor + \lceil\varepsilon\rceil + 1 \;\le\; 2\varepsilon + 2,$$
so the tail conflict degree satisfies $D^\gamma_{\mathcal M} \le \max_j D^j_{\mathcal M} \le 2\varepsilon + 2$.
*Proof.* $\lfloor M(z_i)\rfloor = j$ gives $j \le M(z_i) < j+1$, hence $i \in [\,j - \varepsilon,\ j + 1 + \varepsilon)$.
The integers in this interval are $j - \lfloor \varepsilon\rfloor, \dots, j + \lceil\varepsilon\rceil$, and distinct
keys have distinct ranks $i$. $\square$
No monotonicity of $M$ is needed. With Theorem 3.1(a), the endpoint model gives $D^j \le 2nD_{KS} + 2$.

*Lean:* `conflict_le`.

**Proposition 4.7 (conflicts and density, directly).**
- (a) For the uniform-slope model $\lfloor n u\rfloor$: $D^j \le 1 + 2nD_{KS}(F_u, U)$.
- (b) (Pigeonhole.) Let $M(z) = az + b$ with $a \ge 0$, and let an interval of $z$-length $w$ contain $k$ keys. Then
  some slot receives at least $k / (\lfloor aw\rfloor + 2)$ of them.
- In density terms: slot $j$ covers a $z$-interval $J_j$ of length $1/a$, and
  $D^j = n\,\mu_n(J_j) = (n/a)\,\rho(J_j)$, where $\mu_n$ is the empirical measure and $\rho(J) = \mu_n(J)/|J|$ is the
  relative local density. The conflict degree *is* the local density at slot scale. (a) bounds it above by the global
  distance, and (b) bounds the maximum below by the density at every coarser scale.

*Proof.*
- (a) $D^j = \#\lbrace i : u_i \in [j/n, (j+1)/n)\rbrace$. This is at most
  $n\bigl(F_u((j+1)/n) - F_u(j/n - \eta)\bigr)$ for every $\eta > 0$, which is at most
  $n\bigl(U((j+1)/n) - U(j/n - \eta) + 2D_{KS}\bigr) \le 1 + n\eta + 2nD_{KS}$. Let $\eta \to 0$.
- (b) The slots of the interval's keys lie in $\lbrace\lfloor A\rfloor, \dots, \lfloor A + aw\rfloor\rbrace$, where $A$
  is the image of the interval's left end. That set has at most $\lfloor aw\rfloor + 2$ elements, so the pigeonhole
  principle applies. $\square$

*Lean:* (a) `conflict_le_ks`; (b) `conflict_ge_density`.

**Corollary 4.8 (reducing $D_{KS}$ reduces everything).** Every bound above is a nondecreasing function of $D_{KS}$ (for
a monotone $T$), or of $D_{KS}$ plus the displacement (Theorem 3.6):
- error $\le nD_{KS}$, or $nD_{KS} + 1$ in slots;
- window search $\le \lceil\log_2(2nD_{KS} + 3)\rceil$;
- exponential search $\le 2\log_2(nD_{KS} + 2) + 3$ (the floored prediction is within $nD_{KS} + 1$ slots);
- conflict degree $\le 1 + 2nD_{KS}$.

This is the formal content of NFL's motivation: a near-uniform transformed distribution makes one linear model accurate,
keeps AFLI's buckets short, and keeps searches short. The search gain is only logarithmic, while the conflict gain is
linear.

*Lean:* `lookup_cost_of_ks` is the end-to-end form: for sorted keys and any CDF-scale model $G$ within $D$ of the
empirical CDF, the repo's search started from the clamped slot `predSlot n (n * G (u i))` returns the key's index
with at most $2\lceil\log_2(\lfloor nD\rfloor + 2)\rceil + 2$ probes; `lookup_bound_mono` says that bound is
nondecreasing in $D$. With `window_cost_mono`, `slot_err` and `conflict_le_ks` this covers the four bullets.

---

## 5. The gate: deciding from a sample (task 4)

**Definition 5.1 (binned statistics).** Take sample values $v_1, \dots, v_m \in [0,1]$ (normalized transformed keys) and
$B$ bins: $\mathrm{bin}(v) = \min(B-1, \lfloor vB\rfloor)$, so bin $b$ is $[b/B, (b+1)/B)$ and the last bin is closed.
Let $p_b = \#\lbrace k : \mathrm{bin}(v_k) = b\rbrace / m$. Define
$$\mathrm{TV}_B = \tfrac12\sum_b |p_b - 1/B|, \qquad \mathrm{KL}_B = \sum_b p_b \ln(B p_b), \text{ with } 0\ln 0 = 0 .$$
This is exactly `kl_uniform` in `src/tools/nf_gate.cpp`, which normalizes by the sample's minimum and maximum.

*Lean:* `bin`, `hist`, `tvUnif` (with the factor $\tfrac12$; Mathlib's `tvDist` has none) and `klUnif`, the last with
Mathlib's convention $\log 0 = 0$. `klUnif_eq_sum_klFun` rewrites $\mathrm{KL}_B$ through Mathlib's
`InformationTheory.klFun`, and `klUnif_nonneg` is Gibbs' inequality.

**Lemma 5.1 (prefix sums are bounded by TV).** For probability vectors $p, q$ on $B$ points and every $k$:
$\bigl|\sum_{b<k}(p_b - q_b)\bigr| \le \tfrac12\sum_b|p_b - q_b|$.
*Proof.* Since $\sum_b (p_b - q_b) = 0$, the prefix sum equals minus the suffix sum. So
$2\bigl|\sum_{b<k}\bigr| = \bigl|\sum_{b<k}\bigr| + \bigl|\sum_{b\ge k}\bigr| \le \sum_b |p_b - q_b|$. $\square$

*Lean:* `prefix_le_tv`.

**Theorem 5.2 (binning: $D_{KS} \le \mathrm{TV} + 1/B$).** $D_{KS}(F_v, U) \le \mathrm{TV}_B + 1/B$, where $F_v$ is the
sample's empirical CDF.
*Proof.* Let $P_k = \sum_{b<k} p_b$ and $t \in [k/B, (k+1)/B)$ with $t < 1$. Samples in bins below $k$ are $< k/B \le t$,
and samples $\le t$ lie in bins $\le k$. So $P_k \le F_v(t) \le P_{k+1}$. Hence
- $F_v(t) - t \le P_{k+1} - k/B = (P_{k+1} - \tfrac{k+1}{B}) + \tfrac1B \le \mathrm{TV}_B + \tfrac1B$;
- $F_v(t) - t \ge P_k - \tfrac{k+1}{B} \ge -\mathrm{TV}_B - \tfrac1B$;

using Lemma 5.1 with $q$ uniform. At $t = 1$, and outside $[0,1]$, the difference is 0. $\square$

*Lean:* `ks_le_tv_add` (every real $t$). The binned form, $D_{KS} \le \mathrm{TV}$ for the binned CDFs
$P_k = \sum_{b<k}p_b$ at the $B+1$ bin boundaries, is `ksBinned_le_tv` (any two vectors of equal mass) and
`ksBinned_hist_le_tvUnif` (the gate's histogram against uniform).

**Theorem P (Pinsker's inequality for finite distributions).** For probability vectors $p \ge 0$ and $q > 0$ on a finite
set, $\mathrm{TV}(p,q) = \tfrac12\sum_b|p_b - q_b| \le \sqrt{\mathrm{KL}(p\,\|\,q)/2}$, with the natural logarithm and
$0 \ln 0 = 0$ (Pinsker 1964; the constant $1/2$ is due to Csiszár 1967 and Kullback 1967; see Tsybakov 2009,
Lemma 2.5). In particular $\mathrm{TV}_B \le \sqrt{\mathrm{KL}_B/2}$ ($q$ uniform).

*Proof.*
- **Step 1.** For $x > 0$, $\psi(x) = 2x(x+2)\ln x - (x-1)(5x+1) \ge 0$. Indeed $\psi'(x) = 4\chi(x)$ with
  $\chi(x) = (x+1)\ln x - 2(x-1)$, and $\chi'(x) = \ln x + 1/x - 1 \ge 0$, so $\chi$ increases and $\chi(1) = 0$.
  Hence $\psi$ decreases on $(0,1]$ and increases on $[1,\infty)$, and $\psi(1) = 0$.
- **Step 2.** For $p \ge 0$ and $q > 0$: $3(p-q)^2 \le 2(p+2q)\bigl(p\ln(p/q) - p + q\bigr)$. For $p = 0$ this reads
  $3q^2 \le 4q^2$. For $p > 0$ put $x = p/q$: the right side minus the left side is $q^2\psi(x)$.
- **Step 3.** Divide Step 2 by $p_b + 2q_b > 0$ and sum. Since $\sum_b p_b = \sum_b q_b$, the sum of
  $p_b\ln(p_b/q_b) - p_b + q_b$ is $\mathrm{KL}$. Sedrakyan's form of Cauchy-Schwarz with weights $p_b + 2q_b$,
  which sum to 3, gives $(\sum_b|p_b-q_b|)^2/3 \le \sum_b (p_b-q_b)^2/(p_b+2q_b) \le \tfrac23\,\mathrm{KL}$. So
  $(2\,\mathrm{TV})^2 \le 2\,\mathrm{KL}$. $\square$

The pinned Mathlib (`v4.35.0`) has Kullback-Leibler divergence (`Mathlib/InformationTheory/KullbackLeibler`) but no
Pinsker inequality, so the earlier draft of this note took it as a hypothesis. For finite distributions, which is all
the gate uses, it is now proved. The generic chain `certified_error` still takes it as a premise `hPinsker` about
abstract numbers $\mathrm{tv}, \mathrm{kl}$, and the `PinskerUnif B` premise of `certified_error_even` and
`certified_error_iid` is discharged by `pinsker_unif`.

*Lean:* `psi_nonneg` (Step 1, with `chiP_monotoneOn` and Mathlib's `monotoneOn_of_hasDerivWithinAt_nonneg`),
`klTerm_lower` (Step 2), `pinsker_fin` and `pinsker_tv` (Step 3, any $q > 0$; Mathlib's `Finset.sq_sum_div_le_sum_sq_div`),
`pinsker_unif : PinskerUnif B` for every $B > 0$ (`Pinsker.lean`).

**Theorem 5.3 (evenly spaced samples: deterministic error $< 1/m$).** Let $2 \le m \le n$. Take the nf_gate sample
$s_k = x_{\rho_k}$ with 0-indexed ranks $\rho_k = \lfloor k(n-1)/(m-1)\rfloor$, $k = 0, \dots, m-1$. Let $T$ be strictly
increasing. Then for every $t$:
$$\bigl|F_{T(s)}(t) - F_{T(X)}(t)\bigr| < 1/m .$$
The sample contains $x_1$ and $x_n$ ($\rho_0 = 0$, $\rho_{m-1} = n-1$), so its min-max normalization is the
population's.

*Proof.* Let $c = \#\lbrace i : T(x_i) \le t\rbrace \in \lbrace 0, \dots, n\rbrace$. By monotonicity, $T(s_k) \le t$ iff
$\rho_k \le c - 1$, iff $k(n-1)/(m-1) < c$, iff $k(n-1) < c(m-1)$. So the sample count is
$c_s = \#\lbrace k < m : k(n-1) < c(m-1)\rbrace$.
- If $c = 0$, then $c_s = 0$. If $c = n$, then $c_s = m$, because $k(n-1) \le (m-1)(n-1) < n(m-1)$. In both cases the
  difference is 0.
- Otherwise $c_s = \lceil y\rceil$ with $y = c(m-1)/(n-1) \le m - 1$. So $c_s/m - c/n \in [\,y/m - c/n,\ y/m - c/n + 1/m)$.
  Now $y/m - c/n = c(m-n)/(mn(n-1))$, which lies in $[-(n-m)/(mn),\ 0]$, a subset of $(-1/m, 0]$.
  Hence $c_s/m - c/n \in (-1/m, 1/m)$. $\square$

*Lean:* `even_sample_count` (the counting step), `evenIdx` and `evenSample` (nf_gate's sample), `evenIdx_first_last`,
`even_sample_dev`.

**Hypothesis DKW (Dvoretzky-Kiefer-Wolfowitz with Massart's constant; classical, not in Mathlib).** Draw $m$ keys
i.i.d. (with replacement) from the key set, with empirical CDF $F_m$. Then for all $\tau > 0$:
$$\Pr\bigl[\sup_t |F_m(t) - F_X(t)| > \tau\bigr] \le 2e^{-2m\tau^2}$$
(Dvoretzky, Kiefer and Wolfowitz 1956; Massart 1990 for the constant 2). Massart proves it for continuous $F$. For a
discrete $F$ such as $F_X$, the statistic is stochastically smaller (write $F_m = G_m \circ F$ with $G_m$ a uniform
empirical CDF), so the bound still holds. Applying the same strictly increasing map (here $T$ followed by the
normalization by $T(x_1)$, $T(x_n)$) to the sample and to the keys leaves the deviation unchanged, as in Theorem 2.1:
both CDFs are step functions that jump only at the keys. So the hypothesis may be stated on the transformed,
normalized values, as `hDKW` does. Mathlib has no Glivenko-Cantelli or DKW theorem, so this enters as an explicit
premise `hDKW`.

*Lean:* never an axiom. It is the premise `hDKW` of `certified_error_prob`, `certified_error_prob_proved` and
`gate_false_positive`, stated for an arbitrary probability space as
`P {ω | ∃ t, τ < |ecdf (V ω) t - ecdf u t|} ≤ ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2))`
(resp. `P {ω | τ < dev ω} ≤ …`).

**Theorem 5.4 (what a sample KL certifies).** Let $T$ be strictly increasing, and normalize by the population endpoints
$T(x_1)$, $T(x_n)$. These are known exactly in a sorted array, and they are the sample's own for an evenly spaced
sample. Then (Pinsker's inequality enters through Theorem P):
- (a) For the evenly spaced sample: $\varepsilon_U(T) \le n\bigl(\sqrt{\mathrm{KL}_B/2} + 1/B + 1/m\bigr)$.
- (b) For an i.i.d. sample, assuming Hypothesis DKW: with probability at least $1 - 2e^{-2m\tau^2}$,
  $\varepsilon_U(T) \le n\bigl(\sqrt{\mathrm{KL}_B/2} + 1/B + \tau\bigr)$.

Slots add 1 (Theorem 3.2). Equivalently, at confidence $1 - \delta$ take $\tau = \sqrt{\ln(2/\delta)/(2m)}$; for
$m = 1024$ and $\delta = 0.05$ this is 0.042.

*Proof.* Chain the triangle inequality:
$$\varepsilon_U \le nD_{KS}(F_u, U) \le n\bigl(D_{KS}(F_v, U) + \sup|F_v - F_u|\bigr) \le n\bigl(\mathrm{TV}_B + 1/B + \mathrm{dev}\bigr).$$
The steps are Theorem 3.1(a), the triangle inequality, and Theorem 5.2. Then $\mathrm{TV}_B \le \sqrt{\mathrm{KL}_B/2}$ by
Theorem P. Finally $\mathrm{dev} < 1/m$ by Theorem 5.3 in case (a), and $\mathrm{dev} \le \tau$ on the DKW event in
case (b). $\square$

*Lean:* `certified_error` is the chain for abstract numbers, with Pinsker as the premise `hPinsker : tv ≤ √(kl/2)`;
`certified_slot_err` adds the slot "+1"; `ks_chain` is the triangle step.
(a) `certified_error_even` (premise `PinskerUnif B`) and, with Theorem P discharging it, `certified_error_even_proved`
and `certified_slot_err_even_proved`, which have **no hypothesis** beyond $2 \le m \le n$, $B > 0$, sorted keys in
$[0,1]$.
(b) `certified_error_iid` and `certified_slot_err_iid` on the event $\sup_t|F_v - F_u| \le \tau$;
`certified_error_prob` and `certified_error_prob_proved` give the probability form (failure probability
$\le 2e^{-2m\tau^2}$) from the premise `hDKW`.

**Theorem 5.5 (the decision rule).** Let:
- $C_0$ be the true per-lookup search cost *without* the transform, that is, of the incumbent the build would pick:
  the cheaper of the raw model and binary search, as `fit_root` does;
- $C_T$ the search cost *with* it;
- $c_T \ge 0$ the transform's compute per lookup, in probe-equivalents (NFGATE charges 1 per transform call);
- $\delta \ge 0$ a margin.

Suppose statistics $\hat U_T$ and $\hat L_0$ satisfy $C_T \le \hat U_T$ and $\hat L_0 \le C_0$ on an event $E$.
**Rule:** transform iff $\hat U_T + c_T + \delta < \hat L_0$. Then:
- (a) On $E$, "transform" implies $C_0 - C_T - c_T > \delta$: the saving net of compute exceeds the margin.
- (b) $\Pr[\text{transform} \wedge C_0 - C_T - c_T \le \delta] \le \Pr[E^c]$. This is 0 in case (a) of Theorem 5.4,
  and at most $2e^{-2m\tau^2}$ in case (b).
- (c) The rule needs an upper bound on the transformed cost *and* a lower bound on the incumbent's.

*Proof.*
- (a) $C_T + c_T + \delta \le \hat U_T + c_T + \delta < \hat L_0 \le C_0$.
- (b) By (a), the bad event lies in $E^c$; use monotonicity of measure.
- (c) Restates the hypotheses of (a). $\square$

*Lean:* (a) `gate_sound`; (b) `gate_false_positive` (premise `hDKW`).

Theorem 5.4 supplies $\hat U_T$: compose its error bound with the nondecreasing cost bounds of Theorems 4.1 and 4.3.
The cost of the incumbent $C_0$ is computable exactly at build time (fit_root does it on the fences). This is the
precise sense of "transform iff the predicted cost saving exceeds the transform's compute cost".

**Remark 5.6 (why a flatness statistic alone cannot be the gate).** Three facts follow from the theorems above.
1. **It bounds only one side.** KL, KS or TV of the transformed sample bounds $\varepsilon_T$ from above. It gives no
   lower bound on $C_0$, so it cannot certify "transform" by itself (Theorem 5.5(c)).
   - It also says nothing about the raw side: $X = \lbrace 1, \dots, n-1, M\rbrace$ with $M \to \infty$ has min-max
     $D_{KS} \to 1 - 1/n$, yet the clamped identity model is exact (Lemma 3.7).
2. **It loses a square root.** To guarantee the root's exponential search beats binary search on $R = 48{,}829$ fences,
   Corollary 4.5(a) needs a fence error below 79 fences. Through Theorem 5.4 this needs
   $\sqrt{\mathrm{KL}_B/2} + 1/B + 1/m < 0.0016$, i.e. $\mathrm{KL}_B < 5\cdot 10^{-6}$ and $B, m > 600$.
3. **It ignores order.** For a non-monotone $T$, a statistic of the sorted output misses the displacement term of
   Theorem 3.6.

NFL's own switching rule compares tail conflict degrees, which is Theorem 4.6's quantity on the sorted transformed
keys. It has no $c_T$ term. The paper reports the consequence: on LTD and LGN the flow lowered throughput through its
inference overhead (Sec. 4.2).

---

## 6. Limits of a finite-resolution transform (task 5)

**Definition 6.1.** $T$ is *$K$-piece affine on $X$* if the index set $\lbrace 1, \dots, n\rbrace$ splits into $K$ runs of
consecutive indices $P_1, \dots, P_K$, with $T(x_i) = \alpha_k x_i + \beta_k$ for $i \in P_k$. Any piecewise-linear $T$
with $K$ pieces in key space induces such runs: spline flows, segmented tiers, a radix layer. On a run $P$, let
$\delta(P) = \inf_{a,b}\max_{i\in P}|a x_i + b - i|$, the best affine rank error of the *raw* keys of that piece.

**Theorem 6.1 (no transform beats the within-piece deviation).** Let $T$ be $K$-piece affine and $A(z) = az + b$ any
affine model. Then
$$\max_i |A(T(x_i)) - i| \;\ge\; \max_k \delta(P_k) \;\ge\; \varepsilon^*_K(X),$$
where $\varepsilon^*_K(X)$ is the optimal error of a $K$-segment piecewise-linear fit of the rank function (the
disconnected $\varepsilon$-PLA). Equivalently: if an $\varepsilon$-PLA needs $K_\varepsilon$ segments, then no
$K$-piece affine transform with $K < K_\varepsilon$ lets one linear model reach error $\varepsilon$. This $K_\varepsilon$
is what `hardness::pla_segments` computes with an optimal streaming algorithm.
*Proof.* On $P_k$, $A \circ T(x) = (a\alpha_k)x + (a\beta_k + b)$ is affine in $x$. So its error on $P_k$ is at least
$\delta(P_k)$, and the composite is a $K$-segment PLA. $\square$
With clamping (Lemma 3.7), each piece splits into at most 3 affine parts (two constant), so replace $K$ by $3K$.

*Lean:* `piece_composite` (one piece), `kpiece_composite` ($K$ pieces: an affine model after a $K$-piece affine $T$ is a
$K$-segment model of the raw keys with the same errors). The optimal-PLA comparison is on paper.

**Lemma 6.2 (two runs).** Let a model be affine with slope $s$ (positions per key unit) on a run, with error at most
$\varepsilon$ there. Then for keys $i < i'$ of the run, $|s(x_{i'} - x_i) - (i' - i)| \le 2\varepsilon$. Hence, for two
disjoint sub-runs of $h$ gaps each, with key spans $S_1, S_2 > 0$:
$$2\varepsilon \;\ge\; \max\bigl(|sS_1 - h|,\ |sS_2 - h|\bigr) \;\ge\; h\,\frac{|S_1 - S_2|}{S_1 + S_2} \;=\; h\tanh\Bigl(\tfrac12\bigl|\ln\tfrac{S_1}{S_2}\bigr|\Bigr).$$

*Proof.* The first claim is the difference of the two keys' errors. For the second, assume $S_1 \le S_2$. If
$s \ge 2h/(S_1+S_2)$, then $sS_2 - h \ge h(S_2 - S_1)/(S_1+S_2)$. Otherwise
$h - sS_1 > h(S_2 - S_1)/(S_1 + S_2)$. The equality is $(r-1)/(r+1) = \tanh(\tfrac12\ln r)$. $\square$

*Lean:* `run_diff`, `two_run` (no sign condition on $h$ is needed). The $\tanh$ identity is not formalized.

So a run of $h$ gaps whose local density differs by a factor $r$ from another run in the same piece forces
$\varepsilon \ge \tfrac h2\cdot\tfrac{r-1}{r+1}$, whatever the transform does elsewhere.

**Theorem 6.3 (the variance form).** Let one piece contain $J$ disjoint runs of $h$ gaps, with spans $S_1, \dots, S_J$
of mean $\bar S$, variance $\mathrm{Var}(S)$ and coefficient of variation $\mathrm{CV} = \sqrt{\mathrm{Var}(S)}/\bar S$.
Then every $K$-piece affine $T$ and every affine model satisfy
$$\varepsilon \;\ge\; \frac h2\cdot\frac{\mathrm{CV}}{\sqrt{1 + \mathrm{CV}^2}} .$$

*Proof.* By Lemma 6.2, $2\varepsilon \ge \max_j|sS_j - h| \ge \bigl(\frac1J\sum_j (sS_j - h)^2\bigr)^{1/2}$. The
quadratic $s^2\,\overline{S^2} - 2sh\bar S + h^2$ is minimized over $s$ at value
$h^2(1 - \bar S^2/\overline{S^2}) = h^2\,\mathrm{Var}(S)/\overline{S^2}$. And
$\mathrm{Var}/\overline{S^2} = \mathrm{CV}^2/(1 + \mathrm{CV}^2)$. $\square$

*Lean:* `rms_lower` (the quadratic), `spanCV`, `cv_floor` (the bound, from $|sS_j - h| \le 2\varepsilon$ for all $j$).

**Theorem 6.4 (smooth transforms: resolution as log-derivative range).** Let $T$ be differentiable with $T' > 0$ on a
window $W$, and suppose $T'(y) \le e^{\Lambda}\,T'(y')$ for all $y, y' \in W$. In words, $\ln T'$ varies by at most
$\Lambda$ on $W$; if $\ln T'$ is $\kappa$-Lipschitz, then $\Lambda = \kappa|W|$. Then for two key runs in $W$:
- the transformed spans satisfy $e^{-\Lambda} \le \dfrac{S'_1/S'_2}{S_1/S_2} \le e^{\Lambda}$;
- if both runs have $h$ gaps and $\rho = S_1/S_2 \ge e^{\Lambda}$, every affine model on $z = T(x)$ has
  $$\varepsilon \;\ge\; \frac h2\cdot\frac{\rho e^{-\Lambda} - 1}{\rho e^{-\Lambda} + 1}.$$

*Proof.* By the mean value theorem, $S'_j = T(b_j) - T(a_j) = T'(\xi_j)S_j$ with $\xi_j \in W$, and
$T'(\xi_1)/T'(\xi_2) \in [e^{-\Lambda}, e^{\Lambda}]$. So $r := S'_1/S'_2 \ge \rho e^{-\Lambda} \ge 1$. Apply Lemma 6.2
to the transformed keys and use that $(r-1)/(r+1)$ increases in $r$. $\square$

*Lean:* `span_mvt` (mean value theorem, Mathlib's `exists_hasDerivAt_eq_slope`), `span_ratio`, `ratio_form`,
`smooth_floor` (the floor).

**Interpretation (connection to the fine-scale finding).** The repo's F05 decomposition (slide F05,
`experimental/scaleli/docs/figures/slides/README.md` at main@ca8d80e) splits $\ln(\mathrm{gap})$ into a trend (a moving
mean over $W = 4096$ gaps), spikes (the top 0.1%) and a fine residual. The fine residual carries 67-99% of the variance
on every sample (F05; `docs/DESIGN.md`, V row). Theorems 6.3 and 6.4 charge the dispersion of run spans at scales
$h \le W$ inside a piece. By construction of the decomposition, that dispersion is carried by the fine residual and the
spikes, not by the trend. Consider a transform whose pieces are wider than $W$ gaps, or whose $\ln T'$ varies little
across $W$ gaps. It can remove at most the trend share, which is below 33% of the variance (F05: the trend is real only
on osm, 30%, planet, 22%, and books, 15%). The floor set by the rest stays.

This is an interpretation, not a theorem: the theorems bound the error by run-span dispersion, and the link from
run-span dispersion to the log-gap variance share is empirical. It is the formal side of the design conclusion that
"the lever is local slack, not a global warp" (slide F05).

---

## 7. What the measurements say

Sources:
- `results/warmup_nfgate_2026-10-17/NFGATE.md` and `nfgate/cases.jsonl`;
- `docs/proof/check/theory_check.py`, which reads the full 200M keys of each dataset and writes
  `docs/proof/check/theory_check_m{1024,65536}.json`. It computes KS, displacement, pieces and teeth on nf_gate's evenly
  spaced sample of $m$ keys; the root lookups are those $m$ keys against all $R$ fences, with the root line fitted on
  $\min(m, R)$ evenly spaced fences as nf_gate does; the fold uses every region of the full key set that straddles a
  tooth boundary. Its KS is nf_gate's statistic $\max_i |u_i - i/(m-1)|$ on the sorted, min-max normalized values. Since
  $i/m \le i/(m-1) \le (i+1)/m$, it lies within $1/m$ below $D_{KS}$ of the sample (Definition 3.1). The script is
  read-only and untimed, uses the same flows, sample formula and fence fit as nf_gate, and reproduces nf_gate's root
  proxy (e.g. fb raw 10.78, flow 26.74, stack raw 15.95, flow 26.89 at $m = 1024$).
- `docs/proof/check/gallop_crosscheck/`, the check of the Lean search against the C++ (Theorem 4.3).

Numbers below are at $m = 65536$ unless marked.

**Which gate won, and why (Theorem 5.5 and Remark 5.6).** NFGATE chose the root proxy at $m = 1024$: 0 false positives
in 90 cases, and agreement with the timed GRE verdict on 10/10 datasets. In the terms of this note, the root proxy is
the plug-in estimate of $C_0$ and $C_T$ at the root. $C_0 = \min(\text{raw model}, \text{binary search})$ is
Corollary 4.5's comparison, and $C_T$ is measured *in key order*, so it includes Theorem 3.6's displacement. Together
with the `pred` scores built on it, it is the only candidate statistic that estimates both sides of Theorem 5.5's
inequality.

The flatness gates (KL, KS, W1, trend, PLA, NFL's tail-conflict rule) estimate functions of the *sorted* flow output:
the order-free term of Theorem 3.6, on the transformed side only. Remark 5.6 predicts they cannot decide, and they
do not: 9 to 85 false positives at their natural thresholds.

**Where the flow "helps": the order-free term.** The flow lowers the sorted-output KS on four datasets and raises it on
six (the ones whose raw keys were already flat):

| dataset | raw KS | sorted-flow KS |
|---|---|---|
| fb | 1.000 | 0.630 |
| osm | 0.317 | 0.226 |
| books | 0.367 | 0.295 |
| planet | 0.530 | 0.281 |
| covid, genome, history, libio, stack, wise | 0.015-0.13 | 0.121-0.149 |

By Theorem 3.1, in NFL's architecture (an index on the sorted $T(x)$) the error bound improves on exactly the first
four. These four are the paper's target regime (LLT and FB, its "workloads with large conflict degrees", Sec. 4.2). On
the other six, the bound gets worse. This mirrors what NFL's own switching rule found on YCSB, AMZN and WIKI.

**Why it cannot help SCALE-LI: the displacement term.**
- The flows are **not monotone** on any sample (`flow_monotone` false in 360/360 nf_gate samples). The cause is the
  fractional feature of the input encoding (Algorithm 3.1's integer/fraction expansion; `transform.hpp` feeds
  $[x, x - \lfloor x\rfloor]$ with $x = (key - mean)/var$). It resets from 1 to 0 at every integer $x$. The trained
  network increases in $x$ within each unit interval, so its output drops at every reset.
- In key order, $T$ consists of $K$ increasing "teeth", one per unit interval of $(key - mean)/var$. $K$ is 66 for the
  $s64$ flows and 514-515 for fb and planet ($s512$). osm ($s512$) has 315 teeth with keys.
- The teeth's images overlap: in key order the normalized flow value has KS = 1.000 on every dataset, and the mean
  displacement $|\sigma(i) - i|$ is 0.42-0.44 $n$.
- Applied to the sample as the key set, Theorem 3.6 (mean displacement minus $D_{KS}$ of the sorted output) puts the
  mean key-order error of the flow's endpoint model at 0.14-0.31 times the number of keys or more on nine datasets
  (less $1/m$, the gap between nf_gate's KS and $D_{KS}$). On fb the bound is vacuous, because its sorted KS (0.63)
  exceeds its displacement (0.44), but the direct measurement below is the same. At the root, the flow model's mean
  correction distance is 9,316-10,071 fences, 19-21% of $R$, on all ten datasets.
- Evaluated at the measured distances, Theorem 4.3's lower bound averages 24.2-24.5 probes for the flow root. (It
  bounds only lookups whose gallop is not truncated at 0 or $R$, so this is an estimate.) The measured mean is
  26.8-27.1. The incumbent costs 15.66 (binary search; fb's raw root, 10.80). So the flow root loses on every dataset,
  by 11.2-11.4 probes (16.1 on fb), of which the lower bound alone accounts for at least 8.5. No threshold on any
  statistic could flip that, and N's build kept B's root in 90/90 cases.

**Cost formulas hold on real data (Theorem 4.3, Corollaries 4.4 and 4.5).**
- Theorem 4.3's upper bound $2\lceil\log_2(d+1)\rceil + 2$ held on all $2 \times 10 \times 65{,}536$ root lookups
  (0 violations).
- Corollary 4.4's mean bound holds: e.g. fb raw 10.80 measured vs 12.77, stack raw 15.96 vs 17.77, fb flow 26.86 vs
  29.37.
- The crossover of Corollary 4.5 sorts the raw roots exactly (and the order of mean distance is the order of measured
  probes on all ten datasets):
  - fb, mean distance 29 fences (0.06% of $R$, below the 79-fence sufficient bound): the only dataset whose raw root
    beats binary (10.8 probes).
  - history and stack, 165-166 fences (0.34%): tie with binary (15.80, 15.96).
  - all others, 318-5,916 fences: lose to binary, so the build keeps binary.
- fb's raw KS of 1.000 comes from a few huge keys at the top (its maximum key is $2^{64} - 1$, `cases.jsonl`; NFGATE
  notes that fb's large KL is driven by outliers). Clamping (Lemma 3.7) makes them harmless to the model, a concrete
  case of Remark 5.6(1).

**Region level: resolution and fold (Theorems 6.1-6.4, 4.6).**
- Inside a tooth, the flow is strictly increasing and nearly affine. $T'$ changes sign in no tooth, and $\ln T'$ varies
  by at most 0.107 across a *whole* tooth (median 0.013-0.052), both measured on a 2,000-point grid per tooth. The
  direction is increasing because key order has at most one descent per tooth boundary (66-515 pieces among 65,536
  keys). Teeth hold $9\cdot10^4$ to $3\cdot10^6$ keys (median).
- So on a 4096-key region inside a tooth the flow is affine to within a density factor $e^{\Lambda}$ with
  $\Lambda \le 0.107$, the whole-tooth range (such a region holds at most 4.5% of a median tooth). By Theorem 6.4 it
  cannot change a within-region density contrast by more than that factor, so (Theorems 6.1 and 4.6) it barely changes
  the conflict structure. Accordingly, the regions that straddle no tooth boundary change the mean tail conflict degree
  by at most 0.008 per region (the difference of the last two columns below).
- The measured mean increase in region tail conflict degree under the flow comes, to within 0.01 on every dataset,
  from the 65-514 regions that straddle a tooth boundary, where the fold superimposes two teeth (Theorem 3.6):

| dataset | regions straddling a fold (of 48,829) | extra tail conflicts there, per region of the index (recomputed) | `cases.jsonl`: mean flow - mean raw |
|---|---|---|---|
| fb | 514 | +7.556 | +7.564 |
| osm | 312 | +2.123 | +2.124 |
| books | 65 | +1.625 | +1.625 |
| covid | 65 | +2.067 | +2.068 |
| genome | 65 | +1.945 | +1.944 |
| history | 65 | +2.109 | +2.110 |
| libio | 65 | +2.101 | +2.101 |
| planet | 513 | +5.497 | +5.499 |
| stack | 65 | +2.084 | +2.084 |
| wise | 65 | +2.016 | +2.016 |

- This is consistent with N's per-region bypass keeping the flow in at most 6.3% of regions. With the root unchanged,
  it is also why `benefit_net` reduces to the transform charge (NFGATE).
- More generally: the variation within a region is the fine-scale part (67-99% of log-gap variance), which, by
  Theorem 6.4, a transform that is nearly affine at that scale changes by at most the factor $e^{\Lambda}$.

**Limits of this check.**
- The theory explains why no positive case exists here. It does not create one.
- A positive control would need a *monotone* transform ($K = 1$ piece in key order, so displacement 0) whose
  fence-level mean error is below about 0.16% of $R$ (Corollary 4.5(a)). Equivalently, it would have to beat the raw
  fence error, which is already 0.06-0.34% on fb, history and stack.
- Theorem 5.4 does not apply to these flows, which are not monotone. Even for a monotone flow with the same KL, its
  bound would be vacuous. At $m = 1024$, KL64 of the flow sample is 0.18-1.11 on the ten full datasets (0.005 at the
  lowest, on planet windows). So $\sqrt{\mathrm{KL}/2} + 1/B$ is at least 0.31 at full scale and at least 0.066 in every
  case, against the 0.0016 that Remark 5.6(2) needs.
- The chosen gate is therefore an *estimate* of the Theorem 5.5 comparison, accurate to 0.019 probes at the median at
  $m = 1024$ (NFGATE), not a certificate.

---

## 8. What is machine-checked, and what is assumed

The Lean project is `docs/proof/lean`: toolchain `leanprover/lean4:v4.35.0`, Mathlib `v4.35.0`, library `NFLProof`,
sources under `NFLProof/`. To check it:

```
source ~/.elan/env
cd docs/proof/lean
lake exe cache get    # once: Mathlib's prebuilt cache
lake build            # compiles every file, then runs the axiom audit and the sanity checks
```

A successful `lake build` is the certificate. It compiles all of the following with no `sorry`, no `admit` and no
`axiom` of ours (`grep -rnE 'sorry|admit|^axiom' NFLProof/` finds nothing). `NFLProof/Axioms.lean` makes the build fail
unless every theorem and lemma depends only on `propext`, `Classical.choice` and `Quot.sound` (Section 8.5), and
`NFLProof/Sanity.lean` makes it fail unless every hypothesis set is satisfiable on a concrete example (Section 8.6).

| file | content |
|---|---|
| `Defs.lean` | keys, ranks, empirical CDF, uniform CDF; counting lemmas |
| `Correctness.lean` | Section 2 |
| `Error.lean` | Section 3, and the finite Kolmogorov distance `ksUnif` |
| `Cost.lean` | Theorems 4.1, 4.2, 4.6, Proposition 4.7, Corollaries 4.4, 4.5 |
| `Gallop.lean` | Theorem 4.3 (the repo's exponential search) and Corollary 4.8 (lookup cost from $D_{KS}$) |
| `Gate.lean` | Section 5: binning, TV, KL, evenly spaced sample, certified error, decision rule |
| `Pinsker.lean` | Theorem P (Pinsker, proved) and Theorem 5.4 with no Pinsker premise |
| `Limits.lean` | Section 6 |
| `Axioms.lean` | the axiom audit |
| `Sanity.lean` | every hypothesis set instantiated on concrete inputs; key definitions evaluated |

### 8.1 The discrete setting

- **Keys:** `x : Fin n → ℝ`; sorted and distinct is `StrictMono x`. Index `i : Fin n` is 0-based, so the 1-indexed
  rank of key `i` is `i.val + 1`.
- **Transform:** `T : ℝ → ℝ` with `StrictMono T` (Section 2), or arbitrary (Theorem 3.6, through `z = T ∘ x`).
- **Ranks and CDFs:** `rankLE x q = #{i | x i ≤ q}`, `rankLT x q = #{i | x i < q}` (Finset filters),
  `ecdf x t = rankLE x t / n`, `unifCDF t = max 0 (min t 1)`.
- **Kolmogorov distance:** `ksUnif u`, a maximum over the $2n$ one-sided values at the keys,
  `⨆ i, max |(i+1)/n - U(u i)| |i/n - U(u i)|`. `ecdf_sub_unif_le_ksUnif` and `ksUnif_le` prove it equals
  $\sup_{t\in\mathbb R}|F_u(t) - U(t)|$. For the binned CDFs, `ksBinned p q` is the maximum over the $B+1$ bin
  boundaries of `|binCDF p k - binCDF q k|`.
- **Linear model and its error:** predicted rank $n(\alpha z + \beta)$; rank error `|n * (α * z i + β) - (i + 1)|`;
  endpoint error `endpointErr u = ⨆ i, |n * u i - (i + 1)|`; floored slot `⌊n * G (u i)⌋` against `i`.
- **Discrete distributions:** `p : Fin B → ℝ` (or any `Fintype`) with `0 ≤ p b` and `∑ b, p b = 1`.
  `tvUnif p = ½ ∑ |p b - 1/B|` and `tvFin p q = ½ ∑ |p b - q b|` (Mathlib's `tvDist` has no ½);
  `klUnif p = ∑ p b * log (B * p b)` and `klFin p q = ∑ p b * log (p b / q b)`, with Mathlib's `log 0 = 0`.
  Mathlib's measure-theoretic `klDiv` is not used directly; `klUnif_eq_sum_klFun` ties `klUnif` to its generator
  `InformationTheory.klFun`.
- **Search:** `lastTrue`, `gallop` (with `gallopUp`, `gallopDown`) return `(answer, probes)`; probes are counted
  calls of the predicate `ok : ℕ → Bool`.
- **Probability:** an arbitrary measure `P : MeasureTheory.Measure Ω`; no measurability side condition is needed,
  because only monotonicity of the (outer) measure is used.

### 8.2 The five requested results

| request | Lean |
|---|---|
| (1) rank invariance under a strictly increasing $T$ | `rank_comp_strictMono`, `range_comp_strictMono`, `strictMono_comp`, `rank_of_transformed`; necessity: `ranks_preserved_iff` |
| (2) error $\le nD_{KS} + 1$; PIT error $\le 1$ | `slot_err`, `slot_err_ksUnif`, `ks_sandwich`; `pit_endpoint_err`, `pit_endpointErr_le`, `pit_ksUnif_eq` ($D_{KS} = 1/n$) |
| (3) search cost monotone in the error; conflict bound | `window_cost_mono`, `gallop_spec`, `gallop_bound_mono`, `lookup_cost_of_ks`, `lookup_bound_mono`; `conflict_le`, `conflict_le_ks` |
| (4) $D_{KS} \le \mathrm{TV}$ for binned CDFs; the gate theorem | `ksBinned_le_tv`, `ks_le_tv_add`; `certified_error`, `certified_slot_err` (premise `hPinsker`), `certified_error_iid`, `certified_slot_err_iid` (deviation $\le\tau$), `certified_error_prob` (premise `hDKW`); unconditional: `pinsker_unif`, `certified_error_even_proved`, `certified_slot_err_even_proved`, `certified_error_prob_proved` |
| (5) resolution lower bound | `kpiece_composite`, `two_run`, `cv_floor`, `smooth_floor` |

### 8.3 Explicit hypotheses

- **`hDKW` (the DKW inequality, Massart's constant).** Not in Mathlib; never an axiom. It is a premise of exactly
  three theorems: `certified_error_prob`, `certified_error_prob_proved` (Theorem 5.4(b)) and `gate_false_positive`
  (Theorem 5.5(b)), in the form
  `P {ω | ∃ t, τ < |ecdf (V ω) t - ecdf u t|} ≤ ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2))`
  (for `gate_false_positive`, `P {ω | τ < dev ω} ≤ …` with `dev` the sample deviation).
- **`hPinsker` (Pinsker's inequality).** Not in Mathlib, but proved here for finite distributions (Theorem P,
  `pinsker_tv`, `pinsker_unif`). It remains a premise only of the generic chains, where it is a statement about
  abstract numbers or a `PinskerUnif B` argument: `certified_error`, `certified_slot_err` (`tv ≤ √(kl/2)`), and
  `certified_error_even`, `certified_error_iid`, `certified_slot_err_iid`, `certified_error_prob`
  (`PinskerUnif B`). The `_proved` versions discharge it with `pinsker_unif`.
- **Modelling premises** (part of the statement, not classical results): sorted keys (`StrictMono`), normalized keys
  in $[0,1]$ (`h01`), `hgood` in `gate_false_positive` (the cost bounds hold on the DKW event), `hcost` in
  `mean_cost_le`, `hlow` in `win_needs_small_log_dist` (the lower bound of Theorem 4.3, not formalized), and the
  distance bounds `hX`, `hZ` of `transform_helps`.
- Every premise set above, `hDKW` and `hPinsker` included, is satisfied by a concrete example in `Sanity.lean`, so no
  theorem holds vacuously (Section 8.6).

### 8.4 Index of results

| # | statement | Lean | status |
|---|---|---|---|
| 2.1 | order preservation, inverse on keys | `rank_comp_strictMono`, `range_comp_strictMono`, `strictMono_comp`, `rank_of_transformed`, `rankLE_self` | proved |
| 2.2 | ranks preserved iff increasing | `ranks_preserved_iff` | proved |
| 2.3 | point lookups need injectivity | `point_lookup` | proved |
| D3.1 | $D_{KS}$ as a finite maximum | `ksUnif`, `ecdf_sub_unif_le_ksUnif`, `ksUnif_le` | proved |
| 3.1 | KS sandwich; per-model bounds | `ks_sandwich`, `err_le_of_ks`, `ks_le_of_err` | proved (infimum over $L$ on paper) |
| 3.2 | slot error $\le nD + 1$ | `slot_err`, `slot_err_ksUnif` | proved |
| 3.3 | PIT optimal; zero error iff affine in rank | `ks_ge_inv_n`, `ks_ge_inv_n_of_normalized`, `pit_ks`, `pit_ksUnif_eq`, `pit_endpoint_err`, `pit_endpointErr_le`, `pit_exact`, `zero_error_iff` | proved |
| 3.5 | affine invariance; when $T$ helps | `affine_invariance`, `transform_helps` | (a) for the error sets, (b), (d) proved; (a) for $D_{\mathrm{lin}}$, and (c), on paper |
| 3.6 | displacement term | `err_le_ks_add_disp`, `disp_le_err_add_ks` | proved |
| 3.7 | clamping | `clamp_err` | proved |
| 4.1 | binary search cost, monotone | `lastTrue`, `lastTrue_spec`, `lastTrue_eq`, `window_cost_mono` | proved |
| 4.2 | decision-tree lower bound | `DTree`, `DTree.card_leaves_le`, `DTree.clog_le_depth` | proved (linear-scan remark on paper) |
| 4.3 | exponential search: correct, cost upper bounds | `gallop`, `gallopUp_spec`, `gallopDown_spec`, `gallop_core`, `gallop_spec`, `gallop_cost_logb`, `gallop_bound_mono` | upper bounds proved; case table and lower bound on paper |
| 4.4 | Jensen mean-cost bound | `mean_cost_jensen`, `mean_cost_le`, `gallop_mean_cost` | proved |
| 4.5 | crossover vs binary search | `beats_binary_of_mean_dist`, `win_needs_small_log_dist` | proved ((b) takes the 4.3 lower bound as premise) |
| 4.6 | conflict $\le 2\varepsilon + 2$ | `conflict_le` | proved |
| 4.7 | conflict vs $D_{KS}$; pigeonhole | `conflict_le_ks`, `conflict_ge_density` | proved |
| 4.8 | lookup cost from $D_{KS}$, monotone | `lookup_cost_of_ks`, `lookup_bound_mono`, `keyProbe_iff`, `predSlot` | proved |
| D5.1 | bins, histogram, TV, KL | `bin`, `hist`, `tvUnif`, `klUnif`, `klUnif_eq_sum_klFun`, `klUnif_nonneg` | proved |
| 5.1 | prefix $\le$ TV | `prefix_le_tv` | proved |
| 5.2 | $D_{KS} \le \mathrm{TV} + 1/B$; binned $D_{KS} \le \mathrm{TV}$ | `ks_le_tv_add`, `ksBinned_le_tv`, `ksBinned_hist_le_tvUnif` | proved |
| P | Pinsker, finite distributions | `psi_nonneg`, `klTerm_lower`, `pinsker_fin`, `pinsker_tv`, `pinsker_unif` | proved |
| 5.3 | evenly spaced sample, dev $< 1/m$ | `even_sample_count`, `evenSample`, `evenIdx_first_last`, `even_sample_dev` | proved |
| 5.4 | certified error from KL | `certified_error`, `certified_slot_err`, `certified_error_even`, `certified_error_even_proved`, `certified_slot_err_even_proved`, `certified_error_iid`, `certified_slot_err_iid`, `certified_error_prob`, `certified_error_prob_proved` | (a) proved, no hypothesis; (b) proved from **`hDKW`** |
| 5.5 | decision rule, false-positive probability | `gate_sound`, `gate_false_positive` | (a) proved; (b) proved from **`hDKW`** |
| 6.1 | $K$-piece composite is a $K$-segment model | `piece_composite`, `kpiece_composite` | proved (PLA optimality on paper) |
| 6.2 | two-run bound | `run_diff`, `two_run` | proved ($\tanh$ identity on paper) |
| 6.3 | CV floor | `rms_lower`, `spanCV`, `cv_floor` | proved |
| 6.4 | smooth transforms | `span_mvt`, `span_ratio`, `ratio_form`, `smooth_floor` | proved |

Differences from the statements planned in the earlier draft of this section; each is a strengthening or a
reformulation:
- `lastTrue_spec` has no monotonicity premise: the invariant $\mathrm{ok}(lo) \wedge \neg\mathrm{ok}(hi)$ suffices.
- `gallop_spec` has no `hmono`, `ok 0` or `p < n` premise; the first two follow from `hok`, and the result holds for
  every start $p$. `gallop` transcribes the table-free branch of `locate_from_prediction`; the clamp of the
  prediction is `predSlot`, used in `lookup_cost_of_ks`.
- `two_run` has no `0 ≤ h` premise.
- `certified_error_even` and `certified_error_iid` take `PinskerUnif B` (Pinsker for every histogram on `B` bins)
  instead of a scalar premise, so that `pinsker_unif` can discharge it.
- Theorem 3.1(b) and Theorem 3.5 are stated per affine model; the infimum over models is taken on paper.
- `affine_invariance` keeps each key's index, which is why it holds for every $\alpha \neq 0$; the $D_{\mathrm{lin}}$ part
  of Theorem 3.5(a), which needs $\alpha > 0$, is on paper.

### 8.5 Axiom audit

`NFLProof/Axioms.lean` runs `#print axioms` on all 81 theorems and all 46 supporting lemmas (127 declarations; the
smoke test in `Basic.lean` aside), each wrapped in `#guard_msgs`, so the build fails if any output changes. The
result: 124 depend on `[propext, Classical.choice, Quot.sound]`, and three (`gallopUp_spec`, `gallopDown_spec`,
`evenIdx_first_last`) on `[propext, Quot.sound]`. None depends on `sorryAx` or on an axiom of this project. A block at
the end of the file prints the headline theorems in the build log, for example:

```
info: NFLProof/Axioms.lean:564:0: 'NFLProof.lookup_cost_of_ks' depends on axioms: [propext, Classical.choice, Quot.sound]
info: NFLProof/Axioms.lean:570:0: 'NFLProof.pinsker_unif' depends on axioms: [propext, Classical.choice, Quot.sound]
info: NFLProof/Axioms.lean:573:0: 'NFLProof.certified_error_prob_proved' depends on axioms: [propext, Classical.choice, Quot.sound]
Build completed successfully (9059 jobs).
```

### 8.6 Sanity checks: no vacuous theorem, no degenerate definition

A theorem with contradictory premises, or about a quantity that is identically zero, would be proved but empty.
`NFLProof/Sanity.lean` rules both out. It applies every theorem of Sections 2 to 6 to concrete inputs (mostly $n = 3$
keys) and discharges all of its premises, and it evaluates the definitions:
- `ksUnif` is exactly $1/3$ on the keys $0, \tfrac12, 1$ (the minimum $1/n$ of Theorem 3.3(a)) and at least $49/50$ on
  the clustered keys $0, 0.01, 0.02$, so the Kolmogorov distance is neither $0$ nor constant;
- `endpointErr` is exactly $1$ on $0, \tfrac12, 1$, so with $D_{KS} = 1/3$ the left half of `ks_sandwich` is an
  equality there;
- `klUnif` of a point mass on two bins is $\ln 2 > 0$;
- `transform_helps` is instantiated on the raw keys $0, 1, \dots, 7, 10^6$ and the transformed keys $0, 1, \dots, 8$:
  every line is at least $7/20$ from the raw CDF on the key range, the line $(t+1)/9$ is within $1/9$ of the
  transformed one, and $9 \cdot \tfrac19 < 9 \cdot \tfrac{7}{20} - 1$. So the premises of Theorem 3.5(b) hold together, and a transform
  strictly beats every raw affine model;
- `cv_floor` is instantiated where it is nearly tight: spans $1, 2$, $h = 3$, error $\tfrac12$, floor
  $3/(2\sqrt{10}) \approx 0.474$;
- `hDKW` holds for a sample that equals the keys, and `hgood` for constant costs.

Outside Lean, `docs/proof/check/gallop_crosscheck/run.sh` checks the transcription of Theorem 4.3 against the C++
(Section 4).

### 8.7 Not formalized

- the flow network itself (its non-monotonicity is *measured*, Section 7);
- floating-point arithmetic (keys above $2^{53}$ as `double`);
- the hardware meaning of a "probe";
- the exact probe-count table and the lower bound of Theorem 4.3, and the linear-scan remark after Theorem 4.2;
- the infimum over affine models in Theorems 3.1(b) and 3.5, the $D_{\mathrm{lin}}$ part of Theorem 3.5(a), and
  Theorem 3.5(c);
- the $\tanh$ identity of Lemma 6.2;
- the optimality of the streaming PLA algorithm used by `pla_segments`;
- the numerical claims of Remark 5.6 and Section 7, and the empirical link in Section 6's interpretation.

### 8.8 Headline statements, verbatim from the Lean sources

All in namespace `NFLProof`, with `open Finset`. Proofs are in the files.

```lean
-- Defs.lean
noncomputable def rankLE {n : ℕ} (x : Fin n → ℝ) (q : ℝ) : ℕ := (univ.filter (fun i => x i ≤ q)).card
noncomputable def ecdf {n : ℕ} (x : Fin n → ℝ) (t : ℝ) : ℝ := (rankLE x t : ℝ) / n
noncomputable def unifCDF (t : ℝ) : ℝ := max 0 (min t 1)

-- Gate.lean
noncomputable def tvUnif {B : ℕ} (p : Fin B → ℝ) : ℝ := (1 / 2) * ∑ b, |p b - 1 / B|
noncomputable def klUnif {B : ℕ} (p : Fin B → ℝ) : ℝ := ∑ b, p b * Real.log (B * p b)

-- Correctness.lean
theorem rank_comp_strictMono {n : ℕ} {T : ℝ → ℝ} (hT : StrictMono T) (x : Fin n → ℝ) (q : ℝ) :
    rankLE (T ∘ x) (T q) = rankLE x q ∧ rankLT (T ∘ x) (T q) = rankLT x q

-- Correctness.lean
theorem range_comp_strictMono {n : ℕ} {T : ℝ → ℝ} (hT : StrictMono T) (x : Fin n → ℝ)
    (q₁ q₂ : ℝ) :
    univ.filter (fun i => q₁ ≤ x i ∧ x i ≤ q₂) =
      univ.filter (fun i => T q₁ ≤ T (x i) ∧ T (x i) ≤ T q₂)

-- Correctness.lean
theorem ranks_preserved_iff {n : ℕ} (z : Fin n → ℝ) :
    (∀ i, rankLE z (z i) = i.val + 1) ↔ StrictMono z

-- Error.lean
noncomputable def ksUnif {n : ℕ} (u : Fin n → ℝ) : ℝ :=
  ⨆ i : Fin n, max |((i.val : ℝ) + 1) / n - unifCDF (u i)| |(i.val : ℝ) / n - unifCDF (u i)|

-- Error.lean
theorem ks_sandwich {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) (hn : 0 < n)
    (h0 : ∀ i, 0 ≤ u i) (h1 : ∀ i, u i ≤ 1) (hmin : ∃ i, u i = 0) (hmax : ∃ j, u j = 1) :
    endpointErr u ≤ n * ksUnif u ∧ n * ksUnif u ≤ endpointErr u + 1

-- Error.lean
theorem slot_err {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) {G : ℝ → ℝ} {D : ℝ}
    (hD : ∀ t, |ecdf u t - G t| ≤ D) (i : Fin n) :
    |((⌊(n : ℝ) * G (u i)⌋ : ℤ) : ℝ) - i.val| ≤ n * D + 1

-- Error.lean
theorem slot_err_ksUnif {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) (hn : 0 < n)
    (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1) (i : Fin n) :
    |((⌊(n : ℝ) * u i⌋ : ℤ) : ℝ) - i.val| ≤ n * ksUnif u + 1

-- Error.lean
theorem pit_ksUnif_eq {n : ℕ} (hn : 2 ≤ n) :
    ksUnif (fun i : Fin n => (i.val : ℝ) / (n - 1)) = 1 / n

-- Error.lean
theorem pit_endpoint_err {n : ℕ} (hn : 2 ≤ n) (i : Fin n) :
    |(n : ℝ) * ((i.val : ℝ) / (n - 1)) - (i.val + 1)| ≤ 1

-- Cost.lean
theorem window_cost_mono {e e' : ℕ} (h : e ≤ e') :
    Nat.clog 2 (2 * e + 1) ≤ Nat.clog 2 (2 * e' + 1)

-- Gallop.lean
theorem gallop_spec (ok : ℕ → Bool) {n p t : ℕ} (ht : t < n)
    (hok : ∀ i, ok i = true ↔ i ≤ t) :
    (gallop ok p n).1 = t ∧
      (gallop ok p n).2 ≤ 2 * Nat.clog 2 (Int.natAbs ((p : ℤ) - t) + 1) + 2

-- Gallop.lean
theorem gallop_cost_logb (ok : ℕ → Bool) {n p t : ℕ} (ht : t < n)
    (hok : ∀ i, ok i = true ↔ i ≤ t) :
    ((gallop ok p n).2 : ℝ) ≤ 2 * Real.logb 2 ((Int.natAbs ((p : ℤ) - t) : ℝ) + 1) + 3

-- Gallop.lean
theorem lookup_cost_of_ks {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) {G : ℝ → ℝ} {D : ℝ}
    (hD : ∀ t, |ecdf u t - G t| ≤ D) (i : Fin n) :
    (gallop (keyProbe u (u i)) (predSlot n ((n : ℝ) * G (u i))) n).1 = i.val ∧
      (gallop (keyProbe u (u i)) (predSlot n ((n : ℝ) * G (u i))) n).2 ≤
        2 * Nat.clog 2 (⌊(n : ℝ) * D⌋₊ + 2) + 2

-- Gallop.lean
theorem lookup_bound_mono {n : ℕ} {D D' : ℝ} (h : D ≤ D') :
    2 * Nat.clog 2 (⌊(n : ℝ) * D⌋₊ + 2) + 2 ≤ 2 * Nat.clog 2 (⌊(n : ℝ) * D'⌋₊ + 2) + 2

-- Cost.lean
theorem conflict_le {n : ℕ} (M : ℝ → ℝ) (z : Fin n → ℝ) {ε : ℝ} (hε0 : 0 ≤ ε)
    (hε : ∀ i, |M (z i) - (i.val + 1)| ≤ ε) (j : ℤ) :
    ((univ.filter (fun i : Fin n => ⌊M (z i)⌋ = j)).card : ℝ) ≤ 2 * ε + 2

-- Cost.lean
theorem conflict_le_ks {n : ℕ} (u : Fin n → ℝ) {D : ℝ}
    (hD : ∀ t, |ecdf u t - unifCDF t| ≤ D) (j : ℤ) :
    ((univ.filter (fun i : Fin n => ⌊(n : ℝ) * u i⌋ = j)).card : ℝ) ≤ 1 + 2 * n * D

-- Gate.lean
noncomputable def ksBinned {B : ℕ} (p q : Fin B → ℝ) : ℝ :=
  ⨆ k : Fin (B + 1), |binCDF p k.val - binCDF q k.val|

-- Gate.lean
theorem ksBinned_le_tv {B : ℕ} (p q : Fin B → ℝ) (hsum : ∑ b, p b = ∑ b, q b) :
    ksBinned p q ≤ (1 / 2) * ∑ b, |p b - q b|

-- Gate.lean
theorem ks_le_tv_add {m B : ℕ} (hm : 0 < m) (hB : 0 < B) (v : Fin m → ℝ)
    (hv : ∀ k, v k ∈ Set.Icc (0 : ℝ) 1) (t : ℝ) :
    |ecdf v t - unifCDF t| ≤ tvUnif (hist B v) + 1 / B

-- Gate.lean
theorem even_sample_dev {n m : ℕ} (hm : 2 ≤ m) (hmn : m ≤ n) {z : Fin n → ℝ}
    (hz : StrictMono z) (t : ℝ) :
    |ecdf (evenSample z m hm hmn) t - ecdf z t| < 1 / m

-- Gate.lean
def PinskerUnif (B : ℕ) : Prop :=
  ∀ p : Fin B → ℝ, (∀ b, 0 ≤ p b) → ∑ b, p b = 1 → tvUnif p ≤ Real.sqrt (klUnif p / 2)

-- Pinsker.lean
theorem pinsker_tv {ι : Type*} [Fintype ι] (p q : ι → ℝ) (hp : ∀ b, 0 ≤ p b)
    (hq : ∀ b, 0 < q b) (hp1 : ∑ b, p b = 1) (hq1 : ∑ b, q b = 1) :
    tvFin p q ≤ Real.sqrt (klFin p q / 2)

-- Pinsker.lean
theorem pinsker_unif {B : ℕ} (hB : 0 < B) : PinskerUnif B

-- Gate.lean
theorem certified_error {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) {Fs : ℝ → ℝ}
    {dev tv invB kl : ℝ}
    (hsamp : ∀ t, |ecdf u t - Fs t| ≤ dev)
    (hbin : ∀ t, |Fs t - unifCDF t| ≤ tv + invB)
    (hPinsker : tv ≤ Real.sqrt (kl / 2)) (i : Fin n) :
    |(n : ℝ) * unifCDF (u i) - (i.val + 1)| ≤ n * (Real.sqrt (kl / 2) + invB + dev)

-- Gate.lean
theorem certified_slot_err {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) {Fs : ℝ → ℝ}
    {dev tv invB kl : ℝ}
    (hsamp : ∀ t, |ecdf u t - Fs t| ≤ dev)
    (hbin : ∀ t, |Fs t - unifCDF t| ≤ tv + invB)
    (hPinsker : tv ≤ Real.sqrt (kl / 2)) (i : Fin n) :
    |((⌊(n : ℝ) * unifCDF (u i)⌋ : ℤ) : ℝ) - i.val| ≤
      n * (Real.sqrt (kl / 2) + invB + dev) + 1

-- Gate.lean
theorem certified_error_iid {n m B : ℕ} (hm : 0 < m) (hB : 0 < B) (hPinsker : PinskerUnif B)
    {u : Fin n → ℝ} (hu : StrictMono u) (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1)
    (v : Fin m → ℝ) (hv : ∀ k, v k ∈ Set.Icc (0 : ℝ) 1) {τ : ℝ}
    (hev : ∀ t, |ecdf v t - ecdf u t| ≤ τ) (i : Fin n) :
    |(n : ℝ) * u i - (i.val + 1)| ≤
      n * (Real.sqrt (klUnif (hist B v) / 2) + 1 / B + τ)

-- Pinsker.lean
theorem certified_error_even_proved {n m B : ℕ} (hm : 2 ≤ m) (hmn : m ≤ n) (hB : 0 < B)
    {u : Fin n → ℝ} (hu : StrictMono u) (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1) (i : Fin n) :
    |(n : ℝ) * u i - (i.val + 1)| ≤
      n * (Real.sqrt (klUnif (hist B (evenSample u m hm hmn)) / 2) + 1 / B + 1 / m)

-- Pinsker.lean
theorem certified_slot_err_even_proved {n m B : ℕ} (hm : 2 ≤ m) (hmn : m ≤ n) (hB : 0 < B)
    {u : Fin n → ℝ} (hu : StrictMono u) (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1) (i : Fin n) :
    |((⌊(n : ℝ) * u i⌋ : ℤ) : ℝ) - i.val| ≤
      n * (Real.sqrt (klUnif (hist B (evenSample u m hm hmn)) / 2) + 1 / B + 1 / m) + 1

-- Gate.lean
theorem certified_error_prob {Ω : Type*} [MeasurableSpace Ω] (P : MeasureTheory.Measure Ω)
    {n m B : ℕ} (hm : 0 < m) (hB : 0 < B) (hPinsker : PinskerUnif B)
    {u : Fin n → ℝ} (hu : StrictMono u) (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1)
    (V : Ω → Fin m → ℝ) (hV : ∀ ω k, V ω k ∈ Set.Icc (0 : ℝ) 1) {τ : ℝ}
    (hDKW : P {ω | ∃ t, τ < |ecdf (V ω) t - ecdf u t|} ≤
      ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2))) :
    P {ω | ∃ i : Fin n, (n : ℝ) * (Real.sqrt (klUnif (hist B (V ω)) / 2) + 1 / B + τ) <
        |(n : ℝ) * u i - (i.val + 1)|} ≤ ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2))

-- Pinsker.lean
theorem certified_error_prob_proved {Ω : Type*} [MeasurableSpace Ω] (P : MeasureTheory.Measure Ω)
    {n m B : ℕ} (hm : 0 < m) (hB : 0 < B)
    {u : Fin n → ℝ} (hu : StrictMono u) (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1)
    (V : Ω → Fin m → ℝ) (hV : ∀ ω k, V ω k ∈ Set.Icc (0 : ℝ) 1) {τ : ℝ}
    (hDKW : P {ω | ∃ t, τ < |ecdf (V ω) t - ecdf u t|} ≤
      ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2))) :
    P {ω | ∃ i : Fin n, (n : ℝ) * (Real.sqrt (klUnif (hist B (V ω)) / 2) + 1 / B + τ) <
        |(n : ℝ) * u i - (i.val + 1)|} ≤ ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2))

-- Gate.lean
theorem gate_sound {C0 CT cT δ U L : ℝ} (hU : CT ≤ U) (hL : L ≤ C0) (huse : U + cT + δ < L) :
    δ < C0 - CT - cT

-- Gate.lean
theorem gate_false_positive {Ω : Type*} [MeasurableSpace Ω] (P : MeasureTheory.Measure Ω)
    (C0 CT U L dev : Ω → ℝ) (cT δ τ : ℝ) {m : ℕ}
    (hgood : ∀ ω, dev ω ≤ τ → CT ω ≤ U ω ∧ L ω ≤ C0 ω)
    (hDKW : P {ω | τ < dev ω} ≤ ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2))) :
    P {ω | U ω + cT + δ < L ω ∧ C0 ω - CT ω - cT ≤ δ} ≤
      ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2))

-- Limits.lean
theorem kpiece_composite {n K : ℕ} (x : Fin n → ℝ) (T : ℝ → ℝ) (piece : Fin n → Fin K)
    (α β : Fin K → ℝ) (hT : ∀ i, T (x i) = α (piece i) * x i + β (piece i)) {a b ε : ℝ}
    (hε : ∀ i, |a * T (x i) + b - (i.val + 1)| ≤ ε) :
    ∃ a' b' : Fin K → ℝ, ∀ i, |a' (piece i) * x i + b' (piece i) - (i.val + 1)| ≤ ε

-- Limits.lean
theorem two_run {S₁ S₂ h s : ℝ} (h₁ : 0 < S₁) (h₂ : 0 < S₂) :
    h * |S₁ - S₂| / (S₁ + S₂) ≤ max |s * S₁ - h| |s * S₂ - h|

-- Limits.lean
theorem cv_floor {J : ℕ} (hJ : 0 < J) (S : Fin J → ℝ) (hS : ∀ j, 0 < S j) {h s ε : ℝ}
    (hrun : ∀ j, |s * S j - h| ≤ 2 * ε) :
    h / 2 * (spanCV S / Real.sqrt (1 + spanCV S ^ 2)) ≤ ε

-- Limits.lean
theorem smooth_floor {T T' : ℝ → ℝ} {lo hi Λ : ℝ}
    (hd : ∀ y ∈ Set.Icc lo hi, HasDerivAt T (T' y) y) (hpos : ∀ y ∈ Set.Icc lo hi, 0 < T' y)
    (hrange : ∀ y ∈ Set.Icc lo hi, ∀ y' ∈ Set.Icc lo hi, T' y ≤ Real.exp Λ * T' y')
    {a₁ b₁ a₂ b₂ s h ε : ℝ} (h₁ : lo ≤ a₁) (h₁' : a₁ < b₁) (h₁'' : b₁ ≤ hi)
    (h₂ : lo ≤ a₂) (h₂' : a₂ < b₂) (h₂'' : b₂ ≤ hi) (hh : 0 ≤ h)
    (hrun₁ : |s * (T b₁ - T a₁) - h| ≤ 2 * ε) (hrun₂ : |s * (T b₂ - T a₂) - h| ≤ 2 * ε)
    (hρ : Real.exp Λ ≤ (b₁ - a₁) / (b₂ - a₂)) :
    h / 2 * (((b₁ - a₁) / (b₂ - a₂) * Real.exp (-Λ) - 1) /
      ((b₁ - a₁) / (b₂ - a₂) * Real.exp (-Λ) + 1)) ≤ ε
```

---

## 9. References

- S. Wu, Y. Cui, J. Yu, X. Sun, T.-W. Kuo, C. J. Xue. *NFL: Robust Learned Index via Distribution Transformation.*
  PVLDB 15(10), 2022. arXiv 2205.11807 (local copy: `scaleli_sota/sources/2205.11807v1.pdf`).
- J. Wu et al. *Updatable Learned Index with Precise Positions (LIPP).* PVLDB 14(8), 2021 (origin of the conflict
  degree).
- P. Ferragina, G. Vinciguerra. *The PGM-index.* PVLDB 13(8), 2020 (optimal $\varepsilon$-PLA).
- M. S. Pinsker. *Information and Information Stability of Random Variables and Processes.* Holden-Day, 1964;
  I. Csiszár, Studia Sci. Math. Hungar. 2, 1967; S. Kullback, IEEE Trans. Inf. Theory 13, 1967;
  A. B. Tsybakov, *Introduction to Nonparametric Estimation*, Springer 2009, Lemma 2.5.
- A. Dvoretzky, J. Kiefer, J. Wolfowitz. Ann. Math. Statist. 27, 1956, 642-669; P. Massart, *The tight constant in
  the Dvoretzky-Kiefer-Wolfowitz inequality*, Ann. Probab. 18(3), 1990, 1269-1283.
- Repo: `results/warmup_nfgate_2026-10-17/NFGATE.md`, `src/scaleli/transform.hpp`, `src/scaleli/index.hpp`
  (`locate_from_prediction`, `fit_root`), `src/tools/nf_gate.cpp`, `docs/DESIGN.md` (fine-scale finding),
  `docs/proof/check/theory_check.py`.
