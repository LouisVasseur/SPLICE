module

public import Mathlib

/-!
# Shared definitions (PROOF.md §1.1)

The discrete setting of the whole formalization:

* the keys are a function `x : Fin n → ℝ`; "sorted, distinct keys" is `StrictMono x`;
* `rankLE x q = #{i | x i ≤ q}` is the paper's position (1-indexed rank, PROOF.md `r_X`);
* `rankLT x q = #{i | x i < q}` is the lower-bound slot (PROOF.md `ℓ_X`);
* `ecdf x t = rankLE x t / n` is the empirical CDF (PROOF.md `F_X`);
* `unifCDF t = max 0 (min t 1)` is the uniform CDF on `[0, 1]` (PROOF.md `U`).

Index `i : Fin n` is 0-based in Lean, so the 1-indexed rank of key `i` is `i.val + 1`.

We also prove the two counting facts every later file uses:
`rankLE_self` (a key's rank among sorted keys) and `exists_bracket` (a query falls between two
consecutive keys, which fixes its rank).
-/

@[expose] public section

open Finset

namespace NFLProof

/-- Rank of a query: number of keys `≤ q` (PROOF.md `r_X(q)`). -/
noncomputable def rankLE {n : ℕ} (x : Fin n → ℝ) (q : ℝ) : ℕ :=
  (univ.filter (fun i => x i ≤ q)).card

/-- Lower-bound slot of a query: number of keys `< q` (PROOF.md `ℓ_X(q)`). -/
noncomputable def rankLT {n : ℕ} (x : Fin n → ℝ) (q : ℝ) : ℕ :=
  (univ.filter (fun i => x i < q)).card

/-- Empirical CDF (PROOF.md `F_X`, the paper's eq. (1): `pos = F(key) · N`). -/
noncomputable def ecdf {n : ℕ} (x : Fin n → ℝ) (t : ℝ) : ℝ :=
  (rankLE x t : ℝ) / n

/-- The uniform CDF on `[0, 1]` (PROOF.md `U`), NFL's "near-uniform" target. -/
noncomputable def unifCDF (t : ℝ) : ℝ :=
  max 0 (min t 1)

/-! ### Elementary facts about `unifCDF` -/

lemma unifCDF_of_nonpos {t : ℝ} (ht : t ≤ 0) : unifCDF t = 0 := by
  unfold unifCDF; rw [min_eq_left (by linarith)]; exact max_eq_left ht

lemma unifCDF_of_mem {t : ℝ} (h0 : 0 ≤ t) (h1 : t ≤ 1) : unifCDF t = t := by
  unfold unifCDF; rw [min_eq_left h1]; exact max_eq_right h0

lemma unifCDF_of_one_le {t : ℝ} (ht : 1 ≤ t) : unifCDF t = 1 := by
  unfold unifCDF; rw [min_eq_right ht]; exact max_eq_right zero_le_one

lemma unifCDF_mono : Monotone unifCDF := by
  intro a b hab
  unfold unifCDF
  exact max_le_max le_rfl (min_le_min hab le_rfl)

lemma unifCDF_nonneg (t : ℝ) : 0 ≤ unifCDF t := le_max_left _ _

lemma unifCDF_le_one (t : ℝ) : unifCDF t ≤ 1 :=
  max_le zero_le_one (min_le_right _ _)

/-- `unifCDF` is 1-Lipschitz. -/
lemma unifCDF_sub_le {a b : ℝ} (hab : a ≤ b) : unifCDF b - unifCDF a ≤ b - a := by
  unfold unifCDF
  -- case split on where `a` and `b` fall relative to `0` and `1`
  simp only [min_def, max_def]
  split_ifs <;> linarith

lemma continuous_unifCDF : Continuous unifCDF := by
  unfold unifCDF
  exact continuous_const.max (continuous_id.min continuous_const)

/-! ### Counting with sorted keys -/

/-- With sorted keys, the filter `{j | x j ≤ x i}` is exactly `Iic i`. -/
lemma filter_le_eq_Iic {n : ℕ} {x : Fin n → ℝ} (hx : StrictMono x) (i : Fin n) :
    univ.filter (fun j => x j ≤ x i) = Iic i := by
  ext j
  simp only [mem_filter, mem_univ, true_and, mem_Iic]
  exact hx.le_iff_le

/-- **Rank of a key** (PROOF.md, last step of Theorem 2.1): among sorted distinct keys, key `i`
(0-based) has 1-indexed rank `i + 1`. -/
theorem rankLE_self {n : ℕ} {z : Fin n → ℝ} (hz : StrictMono z) (i : Fin n) :
    rankLE z (z i) = i.val + 1 := by
  unfold rankLE
  rw [filter_le_eq_Iic hz i, Fin.card_Iic]

/-- The empirical CDF at a key is `(i + 1) / n`. -/
lemma ecdf_self {n : ℕ} {z : Fin n → ℝ} (hz : StrictMono z) (i : Fin n) :
    ecdf z (z i) = ((i.val : ℝ) + 1) / n := by
  unfold ecdf; rw [rankLE_self hz i]; push_cast; ring

/-- **Bracketing.** If some key is `≤ t`, let `i` be the last such key. Then every later key is
`> t` and the rank of `t` is `i + 1`. -/
lemma exists_bracket {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) {t : ℝ}
    (hlo : ∃ i, u i ≤ t) :
    ∃ i : Fin n, u i ≤ t ∧ (∀ j, i < j → t < u j) ∧ rankLE u t = i.val + 1 := by
  classical
  -- the set of keys `≤ t` is nonempty; take its largest index
  set S := univ.filter (fun j => u j ≤ t) with hS
  have hne : S.Nonempty := by
    obtain ⟨i, hi⟩ := hlo; exact ⟨i, by simp [hS, hi]⟩
  set i := S.max' hne
  have hi : u i ≤ t := by
    have := S.max'_mem hne; simpa [hS] using this
  have hafter : ∀ j, i < j → t < u j := by
    intro j hij
    by_contra h
    push Not at h
    have hjS : j ∈ S := by simp [hS, h]
    exact absurd (S.le_max' j hjS) (not_le.mpr hij)
  refine ⟨i, hi, hafter, ?_⟩
  -- the keys `≤ t` are exactly the keys with index `≤ i`
  have : S = Iic i := by
    ext j
    simp only [hS, mem_filter, mem_univ, true_and, mem_Iic]
    constructor
    · intro hj; exact S.le_max' j (by simp [hS, hj])
    · intro hj; exact (hu.monotone hj).trans hi
  unfold rankLE
  rw [← hS, this, Fin.card_Iic]

/-- If no key is `≤ t`, the rank of `t` is `0`. -/
lemma rankLE_eq_zero {n : ℕ} {u : Fin n → ℝ} {t : ℝ} (h : ∀ i, t < u i) : rankLE u t = 0 := by
  unfold rankLE
  rw [card_eq_zero, filter_eq_empty_iff]
  intro i _; exact not_le.mpr (h i)

/-- For sorted keys, `u j ≤ t` iff the index of `j` is below the rank of `t`. -/
lemma le_iff_lt_rankLE {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) (t : ℝ) (j : Fin n) :
    u j ≤ t ↔ j.val < rankLE u t := by
  by_cases hlo : ∃ i, u i ≤ t
  · obtain ⟨i, hi, hafter, hr⟩ := exists_bracket hu hlo
    rw [hr]
    constructor
    · intro hj
      by_contra hc
      have : i < j := by rw [Fin.lt_def]; omega
      exact absurd hj (not_le.mpr (hafter j this))
    · intro hj
      have : j ≤ i := by rw [Fin.le_def]; omega
      exact (hu.monotone this).trans hi
  · push Not at hlo
    rw [rankLE_eq_zero hlo]
    simp only [Nat.not_lt_zero, iff_false, not_le]
    exact hlo j

/-- The rank is at most `n`. -/
lemma rankLE_le {n : ℕ} (u : Fin n → ℝ) (t : ℝ) : rankLE u t ≤ n := by
  unfold rankLE
  exact (card_filter_le _ _).trans (by simp)

/-- If every key is `≤ t`, the rank of `t` is `n`. -/
lemma rankLE_eq_n {n : ℕ} {u : Fin n → ℝ} {t : ℝ} (h : ∀ i, u i ≤ t) : rankLE u t = n := by
  unfold rankLE
  rw [filter_true_of_mem (fun i _ => h i)]
  simp

end NFLProof
