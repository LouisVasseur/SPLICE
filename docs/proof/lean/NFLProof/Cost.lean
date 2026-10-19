module

public import NFLProof.Defs

/-!
# §4 Cost: search cost is logarithmic in the error, conflict degree is linear in it

Lean names (PROOF.md numbering):

* Theorem 4.1 `lastTrue`, `lastTrue_spec`, `window_cost_mono`: the repo's `last_true` (binary search
  on `[lo, hi)`) is correct and makes at most `⌈log₂ (hi - lo)⌉` probes; the window cost
  `⌈log₂ (2e + 1)⌉` is nondecreasing in the error bound `e`.
* Theorem 4.2 `DTree.card_leaves_le`, `DTree.clog_le_depth`: a binary decision tree of depth `d` has
  at most `2^d` leaves, so returning any of `L` answers needs depth `≥ ⌈log₂ L⌉`.
* Corollary 4.4 `mean_cost_jensen`, `mean_cost_le`: Jensen's inequality for the per-lookup bound
  `2 log₂ (d + 1) + 3`.
* Corollary 4.5 `beats_binary_of_mean_dist`, `win_needs_small_log_dist`: the crossover against
  binary search.
* Theorem 4.6 `conflict_le`: a slot receives at most `2ε + 2` keys.
* Proposition 4.7 `conflict_le_ks` (a) and `conflict_ge_density` (b).
-/

@[expose] public section

open Finset

namespace NFLProof

/-! ### Theorem 4.1: binary search on a window -/

/-- The repo's `last_true(lo, hi)` (`src/scaleli/index.hpp`), with a probe counter: while
`hi - lo > 1`, probe the midpoint and keep the half that contains the last true index.
Returns `(answer, number of probes)`. -/
def lastTrue (ok : ℕ → Bool) (lo hi : ℕ) : ℕ × ℕ :=
  if h : lo + 1 < hi then
    if ok (lo + (hi - lo) / 2) then
      let r := lastTrue ok (lo + (hi - lo) / 2) hi
      (r.1, r.2 + 1)
    else
      let r := lastTrue ok lo (lo + (hi - lo) / 2)
      (r.1, r.2 + 1)
  else (lo, 0)
termination_by hi - lo
decreasing_by all_goals omega

/-- `⌈log₂ L⌉ = 1 + ⌈log₂ ⌈L/2⌉⌉` for `L ≥ 2` (PROOF.md, proof of Theorem 4.1). -/
lemma clog_two_succ {L : ℕ} (hL : 2 ≤ L) : Nat.clog 2 L = Nat.clog 2 ((L + 1) / 2) + 1 := by
  rw [Nat.clog_of_one_lt (by norm_num) (by omega), show L + 2 - 1 = L + 1 by omega]

/-- **Theorem 4.1 (binary search, invariant form).** If `ok lo` is true and `ok hi` is false,
`lastTrue` returns an index `r ∈ [lo, hi)` with `ok r` true and `ok (r + 1)` false, using at most
`⌈log₂ (hi - lo)⌉` probes. (No monotonicity is needed for this; `lastTrue_eq` adds it.) -/
theorem lastTrue_spec (ok : ℕ → Bool) {lo hi : ℕ} (hlh : lo < hi) (hlo : ok lo = true) (hhi : ok hi = false) :
    lo ≤ (lastTrue ok lo hi).1 ∧ (lastTrue ok lo hi).1 < hi ∧
    ok (lastTrue ok lo hi).1 = true ∧ ok ((lastTrue ok lo hi).1 + 1) = false ∧
    (lastTrue ok lo hi).2 ≤ Nat.clog 2 (hi - lo) := by
  -- strong induction on the window length `hi - lo`
  induction h : hi - lo using Nat.strong_induction_on generalizing lo hi with
  | _ L ih =>
    rw [lastTrue]
    by_cases hw : lo + 1 < hi
    · simp only [hw, ↓reduceDIte]
      set mid := lo + (hi - lo) / 2 with hmid
      have hL : 2 ≤ L := by omega
      -- both halves have length at most `⌈L/2⌉`
      have hc : Nat.clog 2 L = Nat.clog 2 ((L + 1) / 2) + 1 := clog_two_succ hL
      by_cases hm : ok mid = true
      · simp only [hm, ↓reduceIte]
        obtain ⟨h1, h2, h3, h4, h5⟩ :=
          ih (hi - mid) (by omega) (lo := mid) (hi := hi) (by omega) hm hhi rfl
        refine ⟨by omega, h2, h3, h4, ?_⟩
        have : Nat.clog 2 (hi - mid) ≤ Nat.clog 2 ((L + 1) / 2) :=
          Nat.clog_mono_right 2 (by omega)
        omega
      · have hm' : ok mid = false := by simpa using hm
        simp only [hm', Bool.false_eq_true, ↓reduceIte]
        obtain ⟨h1, h2, h3, h4, h5⟩ :=
          ih (mid - lo) (by omega) (lo := lo) (hi := mid) (by omega) hlo hm' rfl
        refine ⟨h1, by omega, h3, h4, ?_⟩
        have : Nat.clog 2 (mid - lo) ≤ Nat.clog 2 ((L + 1) / 2) :=
          Nat.clog_mono_right 2 (by omega)
        omega
    · simp only [hw, ↓reduceDIte]
      have : hi = lo + 1 := by omega
      subst this
      exact ⟨le_rfl, by omega, hlo, hhi, Nat.zero_le _⟩

/-- **Theorem 4.1 (correctness).** For a threshold predicate (`ok i` iff `i ≤ t`) with
`lo ≤ t < hi`, `lastTrue` returns exactly `t`. -/
theorem lastTrue_eq (ok : ℕ → Bool) {t : ℕ} (hok : ∀ i, ok i = true ↔ i ≤ t) {lo hi : ℕ}
    (hlo : lo ≤ t) (hhi : t < hi) : (lastTrue ok lo hi).1 = t := by
  have hl : ok lo = true := (hok lo).mpr hlo
  have hh : ok hi = false := by
    rcases h : ok hi
    · rfl
    · exact absurd ((hok hi).mp h) (by omega)
  obtain ⟨-, -, h3, h4, -⟩ := lastTrue_spec ok (by omega) hl hh
  have h3' := (hok _).mp h3
  have h4' : ¬ ((lastTrue ok lo hi).1 + 1 ≤ t) := by
    intro h; rw [(hok _).mpr h] at h4; exact Bool.noConfusion h4
  omega

/-- **Theorem 4.1 (monotonicity).** The window cost `⌈log₂ (2e + 1)⌉` is nondecreasing in `e`. -/
theorem window_cost_mono {e e' : ℕ} (h : e ≤ e') :
    Nat.clog 2 (2 * e + 1) ≤ Nat.clog 2 (2 * e' + 1) :=
  Nat.clog_mono_right 2 (by omega)

/-! ### Theorem 4.2: no comparison search does better -/

/-- A binary decision tree: internal nodes are comparisons, leaves are outputs. -/
inductive DTree (α : Type) where
  | leaf : α → DTree α
  | node : DTree α → DTree α → DTree α

/-- Depth = worst-case number of comparisons. -/
def DTree.depth {α : Type} : DTree α → ℕ
  | .leaf _ => 0
  | .node l r => max l.depth r.depth + 1

/-- The set of outputs the tree can return. -/
def DTree.leaves {α : Type} [DecidableEq α] : DTree α → Finset α
  | .leaf a => {a}
  | .node l r => l.leaves ∪ r.leaves

/-- **Theorem 4.2 (counting).** A tree of depth `d` has at most `2^d` distinct outputs. -/
theorem DTree.card_leaves_le {α : Type} [DecidableEq α] (t : DTree α) :
    t.leaves.card ≤ 2 ^ t.depth := by
  induction t with
  | leaf a => simp [DTree.leaves, DTree.depth]
  | node l r ihl ihr =>
    simp only [DTree.leaves, DTree.depth]
    calc (l.leaves ∪ r.leaves).card ≤ l.leaves.card + r.leaves.card := card_union_le _ _
      _ ≤ 2 ^ l.depth + 2 ^ r.depth := Nat.add_le_add ihl ihr
      _ ≤ 2 ^ max l.depth r.depth + 2 ^ max l.depth r.depth :=
          Nat.add_le_add (Nat.pow_le_pow_right (by norm_num) (le_max_left _ _))
            (Nat.pow_le_pow_right (by norm_num) (le_max_right _ _))
      _ = 2 ^ (max l.depth r.depth + 1) := by ring

/-- **Theorem 4.2.** A comparison search able to return every answer in `S` needs depth at least
`⌈log₂ |S|⌉`. -/
theorem DTree.clog_le_depth {α : Type} [DecidableEq α] (t : DTree α) (S : Finset α)
    (hS : S ⊆ t.leaves) : Nat.clog 2 S.card ≤ t.depth :=
  (Nat.clog_le_iff_le_pow (by norm_num)).mpr ((card_le_card hS).trans t.card_leaves_le)

/-! ### Corollary 4.4: mean cost (Jensen) -/

/-- **Corollary 4.4 (Jensen step).** The mean of `2 log₂ (d_j + 1) + 3` is at most
`2 log₂ (1 + mean d) + 3`. -/
theorem mean_cost_jensen {Q : ℕ} (hQ : 0 < Q) (d : Fin Q → ℝ) (hd : ∀ j, 0 ≤ d j) :
    (∑ j, (2 * Real.logb 2 (d j + 1) + 3)) / Q ≤ 2 * Real.logb 2 (1 + (∑ j, d j) / Q) + 3 := by
  have hQ' : (0 : ℝ) < Q := by exact_mod_cast hQ
  -- Jensen for the concave `log` with equal weights `1/Q`
  have hJ := (strictConcaveOn_log_Ioi.concaveOn).le_map_sum (t := univ)
    (w := fun _ : Fin Q => (1 : ℝ) / Q) (p := fun j => d j + 1)
    (fun _ _ => by positivity) (by simp [card_univ]; field_simp)
    (fun j _ => by simp only [Set.mem_Ioi]; linarith [hd j])
  simp only [smul_eq_mul] at hJ
  have hsum : ∑ j : Fin Q, 1 / (Q : ℝ) * (d j + 1) = 1 + (∑ j, d j) / Q := by
    rw [← Finset.mul_sum, Finset.sum_add_distrib]
    simp [card_univ]
    field_simp
    ring
  rw [hsum] at hJ
  have hlog2 : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  -- rewrite `logb` as `log / log 2` and divide
  simp only [Real.logb]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.sum_div]
  simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [← Finset.mul_sum] at hJ
  have : (∑ j, Real.log (d j + 1)) / Q ≤ Real.log (1 + (∑ j, d j) / Q) := by
    rw [div_eq_mul_inv, mul_comm]; simpa [one_div] using hJ
  have h2 : (∑ j, Real.log (d j + 1)) / Real.log 2 / Q ≤
      Real.log (1 + (∑ j, d j) / Q) / Real.log 2 := by
    rw [div_div, mul_comm, ← div_div]
    exact div_le_div_of_nonneg_right this hlog2.le
  have e : (2 * ((∑ j, Real.log (d j + 1)) / Real.log 2) + Q * 3) / Q =
      2 * ((∑ j, Real.log (d j + 1)) / Real.log 2 / Q) + 3 := by
    field_simp
  rw [e]
  linarith

/-- **Corollary 4.4.** If each lookup's cost is at most `2 log₂ (d_j + 1) + 3` (Theorem 4.3's
upper bound), the mean cost is at most `2 log₂ (1 + mean d) + 3`. -/
theorem mean_cost_le {Q : ℕ} (hQ : 0 < Q) (d cost : Fin Q → ℝ) (hd : ∀ j, 0 ≤ d j)
    (hcost : ∀ j, cost j ≤ 2 * Real.logb 2 (d j + 1) + 3) :
    (∑ j, cost j) / Q ≤ 2 * Real.logb 2 (1 + (∑ j, d j) / Q) + 3 := by
  have hQ' : (0 : ℝ) < Q := by exact_mod_cast hQ
  refine le_trans ?_ (mean_cost_jensen hQ d hd)
  exact div_le_div_of_nonneg_right (Finset.sum_le_sum fun j _ => hcost j) hQ'.le

/-! ### Corollary 4.5: when a model beats binary search -/

/-- **Corollary 4.5 (a), sufficient.** If the mean distance `dbar` satisfies
`dbar < 2^((B - 3)/2) - 1`, the Jensen bound `2 log₂ (1 + dbar) + 3` is below `B`. -/
theorem beats_binary_of_mean_dist {dbar B : ℝ} (hd : 0 ≤ dbar)
    (h : dbar < (2 : ℝ) ^ ((B - 3) / 2) - 1) : 2 * Real.logb 2 (1 + dbar) + 3 < B := by
  have h1 : 1 + dbar < (2 : ℝ) ^ ((B - 3) / 2) := by linarith
  have h2 : Real.logb 2 (1 + dbar) < (B - 3) / 2 := by
    rw [Real.logb_lt_iff_lt_rpow (by norm_num) (by linarith)]
    exact h1
  linarith

/-- **Corollary 4.5 (b), necessary.** If every lookup costs at least `2 log₂ (d_j + 1) - 1`
(Theorem 4.3's untruncated lower bound) and the mean cost is below `B`, then the mean of
`log₂ (d_j + 1)` is below `(B + 1)/2`. -/
theorem win_needs_small_log_dist {Q : ℕ} (hQ : 0 < Q) (d cost : Fin Q → ℝ) {B : ℝ}
    (hlow : ∀ j, 2 * Real.logb 2 (d j + 1) - 1 ≤ cost j) (hwin : (∑ j, cost j) / Q < B) :
    (∑ j, Real.logb 2 (d j + 1)) / Q < (B + 1) / 2 := by
  have hQ' : (0 : ℝ) < Q := by exact_mod_cast hQ
  have hs : ∑ j, (2 * Real.logb 2 (d j + 1) - 1) ≤ ∑ j, cost j :=
    Finset.sum_le_sum fun j _ => hlow j
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum] at hs
  simp only [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at hs
  rw [div_lt_iff₀ hQ'] at hwin ⊢
  nlinarith

/-! ### Theorem 4.6: conflict degree is bounded by the error -/

/-- **Theorem 4.6.** If a model `M` predicts every 1-indexed rank `i + 1` within `ε`, then any slot
`j` receives at most `2ε + 2` keys. No monotonicity of `M` is needed. (`hε0` only matters for
`n = 0`.) -/
theorem conflict_le {n : ℕ} (M : ℝ → ℝ) (z : Fin n → ℝ) {ε : ℝ} (hε0 : 0 ≤ ε)
    (hε : ∀ i, |M (z i) - (i.val + 1)| ≤ ε) (j : ℤ) :
    ((univ.filter (fun i : Fin n => ⌊M (z i)⌋ = j)).card : ℝ) ≤ 2 * ε + 2 := by
  classical
  -- the keys of slot `j` have 0-based ranks in `[j - 1 - ε, j + ε]`, an integer interval
  set S := univ.filter (fun i : Fin n => ⌊M (z i)⌋ = j) with hS
  set lo : ℤ := ⌈(j : ℝ) - 1 - ε⌉
  set hi : ℤ := ⌊(j : ℝ) + ε⌋
  have hmaps : Set.MapsTo (fun i : Fin n => (i.val : ℤ)) S (Icc lo hi) := by
    intro i hi'
    simp only [hS, coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hi'
    have hf1 := Int.floor_le (M (z i))
    have hf2 := Int.lt_floor_add_one (M (z i))
    rw [hi'] at hf1 hf2
    have he := abs_le.mp (hε i)
    simp only [coe_Icc, Set.mem_Icc]
    constructor
    · exact Int.ceil_le.mpr (by push_cast; linarith [he.1, he.2])
    · exact Int.le_floor.mpr (by push_cast; linarith [he.1, he.2])
  have hinj : Set.InjOn (fun i : Fin n => (i.val : ℤ)) S := by
    intro a _ b _ hab
    simp only at hab
    exact Fin.ext (by omega)
  have hcard := card_le_card_of_injOn _ hmaps hinj
  rw [Int.card_Icc] at hcard
  -- `#Icc lo hi = hi + 1 - lo ≤ (j + ε) + 1 - (j - 1 - ε) = 2ε + 2`
  have h1 : (hi : ℝ) ≤ j + ε := Int.floor_le _
  have h2 : (j : ℝ) - 1 - ε ≤ lo := Int.le_ceil _
  have h3 : ((hi + 1 - lo).toNat : ℝ) ≤ max ((hi : ℝ) + 1 - lo) 0 := by
    rcases le_or_gt 0 (hi + 1 - lo) with h | h
    · rw [show ((hi + 1 - lo).toNat : ℝ) = ((hi + 1 - lo : ℤ) : ℝ) by
        exact_mod_cast Int.toNat_of_nonneg h]
      push_cast; exact le_max_left _ _
    · rw [Int.toNat_of_nonpos h.le]; simp
  have h4 : max ((hi : ℝ) + 1 - lo) 0 ≤ 2 * ε + 2 := max_le (by linarith) (by linarith)
  have : (S.card : ℝ) ≤ ((hi + 1 - lo).toNat : ℝ) := by exact_mod_cast hcard
  linarith

/-! ### Proposition 4.7: conflicts and density -/

/-- **Proposition 4.7 (a).** For the uniform-slope model `⌊n u⌋`, every slot receives at most
`1 + 2nD` keys, where `D` bounds `|F_u - U|` everywhere. No monotonicity is needed. -/
theorem conflict_le_ks {n : ℕ} (u : Fin n → ℝ) {D : ℝ}
    (hD : ∀ t, |ecdf u t - unifCDF t| ≤ D) (j : ℤ) :
    ((univ.filter (fun i : Fin n => ⌊(n : ℝ) * u i⌋ = j)).card : ℝ) ≤ 1 + 2 * n * D := by
  classical
  rcases Nat.eq_zero_or_pos n with hn0 | hnpos
  · subst hn0; simp
  have hn : (0 : ℝ) < n := by exact_mod_cast hnpos
  have hD0 : 0 ≤ D := (abs_nonneg _).trans (hD 0)
  set S := univ.filter (fun i : Fin n => ⌊(n : ℝ) * u i⌋ = j) with hS
  -- for every `η > 0`: `#S ≤ n (F((j+1)/n) - F(j/n - η)) ≤ 1 + nη + 2nD`
  apply le_of_forall_pos_lt_add
  intro δ hδ
  set η := δ / (2 * n) with hη
  have hηpos : 0 < η := by positivity
  set a : ℝ := (j : ℝ) / n - η
  set b : ℝ := ((j : ℝ) + 1) / n
  -- keys of slot `j` are `> a` and `≤ b`
  have hsub : S ⊆ (univ.filter (fun i => u i ≤ b)) \ (univ.filter (fun i => u i ≤ a)) := by
    intro i hi
    simp only [hS, mem_filter, mem_univ, true_and] at hi
    have hf1 := Int.floor_le ((n : ℝ) * u i)
    have hf2 := Int.lt_floor_add_one ((n : ℝ) * u i)
    rw [hi] at hf1 hf2
    simp only [mem_sdiff, mem_filter, mem_univ, true_and, not_le]
    constructor
    · show u i ≤ ((j : ℝ) + 1) / n
      rw [le_div_iff₀ hn]; linarith
    · show (j : ℝ) / n - η < u i
      have : (j : ℝ) / n ≤ u i := by rw [div_le_iff₀ hn]; linarith
      linarith
  have hsubset : univ.filter (fun i => u i ≤ a) ⊆ univ.filter (fun i => u i ≤ b) := by
    intro i hi
    simp only [mem_filter, mem_univ, true_and] at hi ⊢
    have : a ≤ b := by
      simp only [a, b]
      have : (j : ℝ) / n ≤ ((j : ℝ) + 1) / n := div_le_div_of_nonneg_right (by linarith) hn.le
      linarith
    linarith
  have hcard : (S.card : ℝ) ≤ (rankLE u b : ℝ) - rankLE u a := by
    have h1 := card_le_card hsub
    rw [card_sdiff_of_subset hsubset] at h1
    have h2 := card_le_card hsubset
    unfold rankLE
    have : S.card + (univ.filter (fun i => u i ≤ a)).card ≤
        (univ.filter (fun i => u i ≤ b)).card := by omega
    have : (S.card : ℝ) + (univ.filter (fun i => u i ≤ a)).card ≤
        (univ.filter (fun i => u i ≤ b)).card := by exact_mod_cast this
    linarith
  -- convert ranks to the ECDF and use `|F - U| ≤ D` and the 1-Lipschitz `U`
  have hFb : (rankLE u b : ℝ) = n * ecdf u b := by unfold ecdf; field_simp
  have hFa : (rankLE u a : ℝ) = n * ecdf u a := by unfold ecdf; field_simp
  have hb := abs_le.mp (hD b)
  have ha := abs_le.mp (hD a)
  have hab : a ≤ b := by
    simp only [a, b]
    have : (j : ℝ) / n ≤ ((j : ℝ) + 1) / n := div_le_div_of_nonneg_right (by linarith) hn.le
    linarith
  have hU := unifCDF_sub_le hab
  have hba : b - a = 1 / n + η := by simp only [a, b]; field_simp; ring
  have key : ecdf u b - ecdf u a ≤ 1 / n + η + 2 * D := by linarith
  have : (S.card : ℝ) ≤ n * (1 / n + η + 2 * D) := by
    rw [hFb, hFa] at hcard
    nlinarith
  have e : (n : ℝ) * (1 / n + η + 2 * D) = 1 + δ / 2 + 2 * n * D := by
    simp only [hη]; field_simp
  linarith

/-- **Proposition 4.7 (b), pigeonhole.** For a model `⌊a z + b⌋` with slope `a ≥ 0`, the keys in a
`z`-interval of length `w` land in at most `⌊a w⌋ + 2` slots, so some slot `j` receives at least
a `1/(⌊a w⌋ + 2)` share of them. -/
theorem conflict_ge_density {n : ℕ} (z : Fin n → ℝ) {a b c w : ℝ} (ha : 0 ≤ a) (hw : 0 ≤ w) :
    ∃ j : ℤ, ((univ.filter (fun i => z i ∈ Set.Icc c (c + w))).card : ℝ) ≤
      ((⌊a * w⌋ : ℝ) + 2) *
        (univ.filter (fun i => z i ∈ Set.Icc c (c + w) ∧ ⌊a * z i + b⌋ = j)).card := by
  classical
  set S := univ.filter (fun i => z i ∈ Set.Icc c (c + w)) with hS
  set lo : ℤ := ⌊a * c + b⌋
  set hi : ℤ := ⌊a * (c + w) + b⌋
  set t := Icc lo hi with ht
  set f : Fin n → ℤ := fun i => ⌊a * z i + b⌋
  have hmaps : Set.MapsTo f S t := by
    intro i hi'
    simp only [hS, coe_filter, mem_univ, true_and, Set.mem_ofPred_eq, Set.mem_Icc] at hi'
    simp only [ht, coe_Icc, Set.mem_Icc, f]
    constructor
    · exact Int.floor_le_floor (by nlinarith [hi'.1])
    · exact Int.floor_le_floor (by nlinarith [hi'.2])
  -- the slot range has at most `⌊a w⌋ + 2` elements
  have hlohi : hi ≤ lo + ⌊a * w⌋ + 1 := by
    have h1 : (hi : ℝ) ≤ a * (c + w) + b := Int.floor_le _
    have h2 : a * c + b < lo + 1 := Int.lt_floor_add_one _
    have h3 : a * w < ⌊a * w⌋ + 1 := Int.lt_floor_add_one _
    have : (hi : ℝ) < lo + ⌊a * w⌋ + 2 := by nlinarith
    have : hi < lo + ⌊a * w⌋ + 2 := by exact_mod_cast this
    omega
  have hfw : 0 ≤ ⌊a * w⌋ := Int.floor_nonneg.mpr (mul_nonneg ha hw)
  have htcard : (t.card : ℝ) ≤ (⌊a * w⌋ : ℝ) + 2 := by
    rw [ht, Int.card_Icc]
    have hnn : 0 ≤ hi + 1 - lo ∨ hi + 1 - lo < 0 := le_or_gt 0 _
    rcases hnn with h | h
    · have : ((hi + 1 - lo).toNat : ℤ) = hi + 1 - lo := Int.toNat_of_nonneg h
      have : ((hi + 1 - lo).toNat : ℤ) ≤ ⌊a * w⌋ + 2 := by omega
      exact_mod_cast this
    · rw [Int.toNat_of_nonpos h.le]
      have : (0 : ℝ) ≤ ⌊a * w⌋ := by exact_mod_cast hfw
      simp only [Nat.cast_zero]; linarith
  -- pick the fullest slot
  by_cases hSe : S.Nonempty
  · have htne : t.Nonempty := by
      obtain ⟨i, hi'⟩ := hSe; exact ⟨f i, hmaps hi'⟩
    obtain ⟨j, -, hj⟩ := exists_max_image t (fun y => (S.filter (fun i => f i = y)).card) htne
    refine ⟨j, ?_⟩
    have hsum := card_eq_sum_card_fiberwise hmaps
    have hle : S.card ≤ t.card * (S.filter (fun i => f i = j)).card := by
      rw [hsum]
      calc ∑ y ∈ t, (S.filter (fun i => f i = y)).card
          ≤ ∑ _y ∈ t, (S.filter (fun i => f i = j)).card := sum_le_sum fun y hy => hj y hy
        _ = t.card * (S.filter (fun i => f i = j)).card := by rw [sum_const, smul_eq_mul]
    have hfilt : S.filter (fun i => f i = j) =
        univ.filter (fun i => z i ∈ Set.Icc c (c + w) ∧ ⌊a * z i + b⌋ = j) := by
      rw [hS, filter_filter]
    rw [← hfilt]
    have hle' : (S.card : ℝ) ≤ (t.card : ℝ) * (S.filter (fun i => f i = j)).card := by
      exact_mod_cast hle
    calc (S.card : ℝ) ≤ (t.card : ℝ) * (S.filter (fun i => f i = j)).card := hle'
      _ ≤ ((⌊a * w⌋ : ℝ) + 2) * (S.filter (fun i => f i = j)).card :=
          mul_le_mul_of_nonneg_right htcard (Nat.cast_nonneg _)
  · refine ⟨0, ?_⟩
    rw [not_nonempty_iff_eq_empty] at hSe
    rw [hSe]
    simp only [card_empty, Nat.cast_zero]
    have : (0 : ℝ) ≤ ⌊a * w⌋ := by exact_mod_cast hfw
    positivity

end NFLProof
