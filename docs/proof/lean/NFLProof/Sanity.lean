module

import NFLProof.Correctness
import NFLProof.Error
import NFLProof.Cost
import NFLProof.Gallop
import NFLProof.Gate
import NFLProof.Pinsker
import NFLProof.Limits

/-!
# Sanity checks: the hypotheses are satisfiable, and the definitions are not degenerate

A theorem with contradictory hypotheses, or about a quantity that is identically `0`, would be
vacuous. This file rules that out by instantiating every hypothesis set of the formalization on
small concrete inputs (mostly `n = 3` keys), and by evaluating the key definitions:

* `ksUnif` is `1/3` on the keys `0, 1/2, 1` (the minimum of Theorem 3.3 (a)) and at least `49/50`
  on clustered keys, so it is neither `0` nor constant;
* `endpointErr` is exactly `1` on the keys `0, 1/2, 1`, so the left half of `ks_sandwich` is tight;
* `klUnif` of a point mass on two bins is `log 2 > 0`;
* `transform_helps` (Theorem 3.5 (b)) is instantiated on nine keys (`0, …, 7` and an outlier
  `10⁶`), where a transform strictly beats every raw affine model, so its distance hypotheses are
  jointly satisfiable;
* `cv_floor` is instantiated where it is nearly tight (`0.474 ≤ 0.5`).

Every `example` below is checked by `lake build`; none is left unfinished.
-/

open Finset NFLProof

namespace NFLProof.Sanity

/-! ### Test keys -/

/-- The probability integral transform's normalized keys `u_i = i/(n-1)`. -/
noncomputable def pit (n : ℕ) : Fin n → ℝ := fun i => (i.val : ℝ) / (n - 1)

/-- The natural keys `0, 1, …, n-1`. -/
noncomputable def natKeys (n : ℕ) : Fin n → ℝ := fun i => (i.val : ℝ)

lemma pit_mono {n : ℕ} (hn : 2 ≤ n) : StrictMono (pit n) := pit_strictMono hn

lemma pit_mem {n : ℕ} (hn : 2 ≤ n) (i : Fin n) : pit n i ∈ Set.Icc (0 : ℝ) 1 := by
  have hn1 : (1 : ℝ) < n := by exact_mod_cast hn
  have hd : (0 : ℝ) < n - 1 := by linarith
  have hi : (i.val : ℝ) + 1 ≤ n := by exact_mod_cast i.isLt
  simp only [pit, Set.mem_Icc]
  exact ⟨div_nonneg (Nat.cast_nonneg _) hd.le, (div_le_one hd).mpr (by linarith)⟩

lemma pit_zero {n : ℕ} (hn : 2 ≤ n) : pit n ⟨0, by omega⟩ = 0 := by
  simp [pit]

lemma pit_last {n : ℕ} (hn : 2 ≤ n) : pit n ⟨n - 1, by omega⟩ = 1 := by
  have hn1 : (1 : ℝ) < n := by exact_mod_cast hn
  have hd : (0 : ℝ) < n - 1 := by linarith
  simp only [pit]
  rw [Nat.cast_sub (by omega : 1 ≤ n), Nat.cast_one]
  exact div_self hd.ne'

lemma natKeys_mono (n : ℕ) : StrictMono (natKeys n) := fun i j h => by
  show (i.val : ℝ) < j.val
  exact_mod_cast h

/-- A strictly increasing transform. -/
lemma affine_mono : StrictMono (fun y : ℝ => 2 * y + 1) := fun a b h => by
  show 2 * a + 1 < 2 * b + 1
  linarith

/-! ### §2 Correctness -/

example := rank_comp_strictMono affine_mono (pit 3) (1 / 2)
example := range_comp_strictMono affine_mono (pit 3) 0 (1 / 2)
example := strictMono_comp affine_mono (pit_mono (n := 3) (by norm_num))
example := rank_of_transformed affine_mono (pit_mono (n := 3) (by norm_num)) ⟨1, by norm_num⟩
example := (ranks_preserved_iff (pit 3)).mpr (pit_mono (by norm_num))
example := point_lookup affine_mono.injective 1 2
example := rankLE_self (pit_mono (n := 3) (by norm_num)) ⟨1, by norm_num⟩

/-! ### §3 Error -/

/-- `D_KS` of the keys `0, 1/2, 1` is `1/3`, the minimum `1/n` of Theorem 3.3 (a). -/
example : ksUnif (pit 3) = 1 / ((3 : ℕ) : ℝ) := pit_ksUnif_eq (by norm_num)

/-- Keys clustered near `0`: `D_KS ≥ 49/50`. So `ksUnif` is not constant, and a transform that
spreads the keys out can lower it from about `1` to `1/n`. -/
noncomputable def clustered : Fin 3 → ℝ := fun i => (i.val : ℝ) / 100

lemma clustered_mono : StrictMono clustered := fun i j hij => by
  have h : (i.val : ℝ) < j.val := by exact_mod_cast hij
  exact div_lt_div_of_pos_right h (by norm_num)

example : (49 : ℝ) / 50 ≤ ksUnif clustered := by
  have h := (le_abs_self _).trans (ksUnif_right_le clustered ⟨2, by norm_num⟩)
  have hc : clustered ⟨2, by norm_num⟩ = 1 / 50 := by simp only [clustered]; norm_num
  rw [hc, unifCDF_of_mem (t := 1 / 50) (by norm_num) (by norm_num)] at h
  norm_num at h
  linarith

/-- The endpoint error of `0, 1/2, 1` is exactly `1` (Theorem 3.3 (b)). With `D_KS = 1/3`, the left
inequality `ε_U ≤ n D_KS` of `ks_sandwich` is an equality here. -/
example : endpointErr (pit 3) = 1 := by
  apply le_antisymm (pit_endpointErr_le (n := 3) (by norm_num))
  have h := endpointErr_ge (pit 3) ⟨0, by norm_num⟩
  rw [pit_zero (n := 3) (by norm_num)] at h
  norm_num at h
  exact h

/-- All six hypotheses of `ks_sandwich` hold for `0, 1/2, 1`. -/
example := ks_sandwich (pit_mono (n := 3) (by norm_num)) (by norm_num)
  (fun i => (pit_mem (by norm_num) i).1) (fun i => (pit_mem (by norm_num) i).2)
  ⟨_, pit_zero (n := 3) (by norm_num)⟩ ⟨_, pit_last (n := 3) (by norm_num)⟩

example := ecdf_sub_unif_le_ksUnif (pit_mono (n := 3) (by norm_num)) (by norm_num) (1 / 4)
example := ksUnif_le (pit_mono (n := 3) (by norm_num)) (by norm_num) (pit_ks (by norm_num))
example := pit_endpoint_err (n := 3) (by norm_num) ⟨1, by norm_num⟩
example := pit_exact (n := 3) ⟨1, by norm_num⟩
example := err_le_of_ks (pit_mono (n := 3) (by norm_num)) (G := ecdf (pit 3)) (D := 0)
  (fun t _ _ => by simp) ⟨1, by norm_num⟩
example := slot_err (pit_mono (n := 3) (by norm_num)) (G := ecdf (pit 3)) (D := 0)
  (fun t => by simp) ⟨1, by norm_num⟩
example := slot_err_ksUnif (pit_mono (n := 3) (by norm_num)) (by norm_num) (pit_mem (by norm_num))
  ⟨1, by norm_num⟩
example := ks_ge_inv_n (u := pit 3) (i₀ := ⟨0, by norm_num⟩) (pit_zero (by norm_num))
  (G := unifCDF) (unifCDF_of_nonpos le_rfl)
example := ks_ge_inv_n_of_normalized (pit_mono (n := 3) (by norm_num)) (by norm_num)
  (pit_zero (by norm_num))
example := (zero_error_iff (n := 3) (by norm_num) (natKeys 3)).mp
  ⟨1, 1, fun i => by simp only [natKeys]; ring⟩
example := (affine_invariance (natKeys 3) (α := 2) (by norm_num) 5 0).mp
  ⟨1, 1, fun i => abs_nonpos_iff.mpr (by simp only [natKeys]; ring)⟩
example := clamp_err (lo := 0) (hi := 2) (y := 5) (r := 1) (by norm_num) (by norm_num)

/-! #### Theorem 3.5 (b) on nine keys: the hypotheses of `transform_helps` are satisfiable -/

/-- Raw keys `0, 1, …, 7, 10⁶`: no line is within `7/20` of their empirical CDF. -/
noncomputable def xKeys : Fin 9 → ℝ := fun i => if i.val < 8 then (i.val : ℝ) else 10 ^ 6

lemma xKeys_mono : StrictMono xKeys := by
  intro i j hij
  have hij' : i.val < j.val := hij
  have hj : j.val < 9 := j.isLt
  have hi8 : i.val < 8 := by omega
  by_cases hj8 : j.val < 8
  · simp only [xKeys, hi8, hj8, ↓reduceIte]
    exact_mod_cast hij'
  · simp only [xKeys, hi8, hj8, ↓reduceIte]
    have h1 : (i.val : ℝ) < 8 := by exact_mod_cast hi8
    have h2 : (8 : ℝ) < 10 ^ 6 := by norm_num
    linarith

/-- The raw keys are at distance at least `7/20` from every affine CDF on their range (the CDF is
`1/9` at `0`, `8/9` at `7` and `1` at `10⁶`). -/
lemma xKeys_far (α β : ℝ) : ∃ t, (∃ i, xKeys i ≤ t) ∧ (∃ j, t ≤ xKeys j) ∧
    (7 : ℝ) / 20 ≤ |ecdf xKeys t - (α * t + β)| := by
  by_contra hcon
  push Not at hcon
  -- evaluate at the keys `0`, `7` and `10⁶`
  have at_key : ∀ i : Fin 9, |ecdf xKeys (xKeys i) - (α * xKeys i + β)| < 7 / 20 :=
    fun i => hcon (xKeys i) ⟨i, le_rfl⟩ ⟨i, le_rfl⟩
  have h0 := at_key ⟨0, by norm_num⟩
  have h7 := at_key ⟨7, by norm_num⟩
  have h8 := at_key ⟨8, by norm_num⟩
  rw [ecdf_self xKeys_mono] at h0
  rw [ecdf_self xKeys_mono] at h7
  rw [ecdf_self xKeys_mono] at h8
  have e0 : xKeys ⟨0, by norm_num⟩ = 0 := by simp [xKeys]
  have e7 : xKeys ⟨7, by norm_num⟩ = 7 := by simp [xKeys]
  have e8 : xKeys ⟨8, by norm_num⟩ = 10 ^ 6 := by simp [xKeys]
  rw [e0] at h0
  rw [e7] at h7
  rw [e8] at h8
  obtain ⟨a0, b0⟩ := abs_lt.mp h0
  obtain ⟨a7, b7⟩ := abs_lt.mp h7
  obtain ⟨a8, b8⟩ := abs_lt.mp h8
  norm_num at a0 b0 a7 b7 a8 b8
  linarith

/-- The transformed keys `0, 1, …, 8` are within `1/9` of the affine CDF `(t + 1)/9`. -/
lemma natKeys_close : ∃ α β : ℝ, ∀ t, (∃ i, natKeys 9 i ≤ t) → (∃ j, t ≤ natKeys 9 j) →
    |ecdf (natKeys 9) t - (α * t + β)| ≤ 1 / 9 := by
  refine ⟨1 / 9, 1 / 9, fun t hlo hhi => ?_⟩
  have h := ks_le_of_err (natKeys_mono 9) (1 / 9) (1 / 9) 0
    (fun i => abs_nonpos_iff.mpr (by simp only [natKeys]; push_cast; ring)) hlo hhi
  calc _ ≤ ((0 : ℝ) + 1) / ((9 : ℕ) : ℝ) := h
    _ = 1 / 9 := by norm_num

/-- `transform_helps` with `n = 9`, `D_X = 7/20`, `D_Z = 1/9`: `9 · (1/9) = 1 < 9 · 7/20 - 1`. -/
example := transform_helps xKeys_mono (natKeys_mono 9) xKeys_far natKeys_close (by norm_num)

/-! #### Theorem 3.6 on non-monotone keys -/

/-- Keys out of order: `1, 0, 2`. -/
noncomputable def scrambled : Fin 3 → ℝ :=
  fun i => if i.val = 0 then 1 else if i.val = 1 then 0 else 2

example := err_le_ks_add_disp scrambled (G := ecdf scrambled) (D := 0) (fun t => by simp)
  ⟨0, by norm_num⟩
example := disp_le_err_add_ks scrambled (G := ecdf scrambled) (D := 0) (fun t => by simp)
  ⟨0, by norm_num⟩

/-! ### §4 Cost -/

example := lastTrue_spec (fun i => decide (i ≤ 3)) (lo := 0) (hi := 8) (by norm_num) (by decide)
  (by decide)
example := lastTrue_eq (fun i => decide (i ≤ 3)) (t := 3) (fun i => by simp) (lo := 0) (hi := 8)
  (by norm_num) (by norm_num)
example := window_cost_mono (e := 2) (e' := 5) (by norm_num)
example := DTree.card_leaves_le (DTree.node (.leaf 0) (.leaf 1))
example : Nat.clog 2 ({0, 1} : Finset ℕ).card ≤ (DTree.node (.leaf 0) (.leaf 1)).depth :=
  DTree.clog_le_depth _ _ (by
    intro x hx
    simp only [DTree.leaves, Finset.mem_union, Finset.mem_singleton, Finset.mem_insert] at hx ⊢
    exact hx)
example := mean_cost_jensen (Q := 2) (by norm_num) (fun _ => 3) (fun _ => by norm_num)
example := mean_cost_le (Q := 2) (by norm_num) (fun _ => 3)
  (fun _ => 2 * Real.logb 2 (3 + 1) + 3) (fun _ => by norm_num) (fun _ => le_rfl)
example := win_needs_small_log_dist (Q := 1) (by norm_num) (fun _ => 0) (fun _ => 2) (B := 3)
  (fun _ => by norm_num) (by norm_num)
example := beats_binary_of_mean_dist (dbar := 0) (B := 5) le_rfl (by norm_num)
example := conflict_le (fun y => y + 1) (natKeys 3) (ε := 0) le_rfl
  (fun i => abs_nonpos_iff.mpr (by simp only [natKeys]; ring)) 1
example := conflict_le_ks (pit 3)
  (ecdf_sub_unif_le_ksUnif (pit_mono (n := 3) (by norm_num)) (by norm_num)) 1
example := conflict_ge_density (natKeys 3) (a := 1) (b := 0) (c := 0) (w := 2) (by norm_num)
  (by norm_num)

/-! ### Theorem 4.3 and Corollary 4.8 -/

example := gallopUp_spec (fun i => decide (i ≤ 5)) (n := 10) (p := 2) (t := 5) (fun i => by simp)
  (by norm_num) 0 (by norm_num)
example := gallopDown_spec (fun i => decide (i ≤ 1)) (p := 7) (t := 1) (fun i => by simp) 0
  (by norm_num)
example := gallop_core (fun i => decide (i ≤ 5)) (n := 10) (p := 0) (t := 5) (by norm_num)
  (fun i => by simp)
example := gallop_spec (fun i => decide (i ≤ 5)) (n := 10) (p := 9) (t := 5) (by norm_num)
  (fun i => by simp)
example := gallop_cost_logb (fun i => decide (i ≤ 5)) (n := 10) (p := 9) (t := 5) (by norm_num)
  (fun i => by simp)
example := gallop_bound_mono (d := 1) (d' := 4) (by norm_num)
example := gallop_mean_cost (Q := 1) (n := 10) (by norm_num) (fun _ i => decide (i ≤ 5))
  (fun _ => 0) (fun _ => 5) (fun _ => by norm_num) (fun _ i => by simp)
example := keyProbe_iff (pit_mono (n := 3) (by norm_num)) ⟨1, by norm_num⟩ 2
/-- With the exact CDF as the model (`D = 0`), every lookup costs at most `4` probes. -/
example := lookup_cost_of_ks (pit_mono (n := 3) (by norm_num)) (G := ecdf (pit 3)) (D := 0)
  (fun t => by simp) ⟨1, by norm_num⟩
example := lookup_bound_mono (n := 3) (D := 0) (D' := 1 / 3) (by norm_num)
/-- With the endpoint model `G = U` and `D = D_KS`. -/
example := lookup_cost_of_ks (pit_mono (n := 3) (by norm_num)) (G := unifCDF)
  (ecdf_sub_unif_le_ksUnif (pit_mono (n := 3) (by norm_num)) (by norm_num)) ⟨2, by norm_num⟩

/-! ### §5 The gate -/

example := prefix_le_tv (B := 2) (fun _ => (1 : ℝ) / 2) (fun _ => 1 / 2) rfl 1
example := ksBinned_le_tv (B := 2) (fun _ => (1 : ℝ) / 2) (fun _ => 1 / 2) rfl
example := ksBinned_hist_le_tvUnif (m := 3) (B := 4) (by norm_num) (by norm_num) (pit 3)
example := ks_le_tv_add (m := 3) (B := 4) (by norm_num) (by norm_num) (pit 3)
  (pit_mem (by norm_num)) (1 / 2)
example := even_sample_count (n := 5) (m := 3) (by norm_num) (by norm_num) (c := 2) (by norm_num)
example := even_sample_dev (n := 5) (m := 3) (by norm_num) (by norm_num)
  (pit_mono (n := 5) (by norm_num)) (1 / 2)
example := evenIdx_first_last (n := 5) (m := 3) (by norm_num) (by norm_num)

lemma pit3_ks_nonneg : 0 ≤ ksUnif (pit 3) :=
  (abs_nonneg _).trans (ksUnif_right_le (pit 3) ⟨0, by norm_num⟩)

lemma pit3_pinsker : ksUnif (pit 3) ≤ Real.sqrt (2 * ksUnif (pit 3) ^ 2 / 2) := by
  rw [show 2 * ksUnif (pit 3) ^ 2 / 2 = ksUnif (pit 3) ^ 2 by ring, Real.sqrt_sq pit3_ks_nonneg]

/-- The abstract chain: `Fs = F_u`, `dev = 0`, `tv = D_KS`, `invB = 0`, `kl = 2 D_KS²`. -/
example := certified_error (pit_mono (n := 3) (by norm_num)) (Fs := ecdf (pit 3)) (dev := 0)
  (tv := ksUnif (pit 3)) (invB := 0) (kl := 2 * ksUnif (pit 3) ^ 2) (fun t => by simp)
  (fun t => by
    rw [add_zero]
    exact ecdf_sub_unif_le_ksUnif (pit_mono (n := 3) (by norm_num)) (by norm_num) t)
  pit3_pinsker ⟨1, by norm_num⟩
example := certified_slot_err (pit_mono (n := 3) (by norm_num)) (Fs := ecdf (pit 3)) (dev := 0)
  (tv := ksUnif (pit 3)) (invB := 0) (kl := 2 * ksUnif (pit 3) ^ 2) (fun t => by simp)
  (fun t => by
    rw [add_zero]
    exact ecdf_sub_unif_le_ksUnif (pit_mono (n := 3) (by norm_num)) (by norm_num) t)
  pit3_pinsker ⟨1, by norm_num⟩
example := certified_error_even (n := 5) (m := 3) (B := 4) (by norm_num) (by norm_num)
  (by norm_num) (pinsker_unif (by norm_num)) (pit_mono (by norm_num)) (pit_mem (by norm_num))
  ⟨2, by norm_num⟩
example := certified_error_even_proved (n := 5) (m := 3) (B := 4) (by norm_num) (by norm_num)
  (by norm_num) (pit_mono (by norm_num)) (pit_mem (by norm_num)) ⟨2, by norm_num⟩
example := certified_slot_err_even_proved (n := 5) (m := 3) (B := 4) (by norm_num) (by norm_num)
  (by norm_num) (pit_mono (by norm_num)) (pit_mem (by norm_num)) ⟨2, by norm_num⟩
example := certified_error_iid (n := 3) (m := 3) (B := 4) (by norm_num) (by norm_num)
  (pinsker_unif (by norm_num)) (pit_mono (by norm_num)) (pit_mem (by norm_num)) (pit 3)
  (pit_mem (by norm_num)) (τ := 0) (fun t => by simp) ⟨1, by norm_num⟩
example := certified_slot_err_iid (n := 3) (m := 3) (B := 4) (by norm_num) (by norm_num)
  (pinsker_unif (by norm_num)) (pit_mono (by norm_num)) (pit_mem (by norm_num)) (pit 3)
  (pit_mem (by norm_num)) (τ := 0) (fun t => by simp) ⟨1, by norm_num⟩
/-- `hDKW` is satisfiable: a sample that equals the keys never deviates. -/
example := certified_error_prob (MeasureTheory.Measure.dirac ()) (n := 3) (m := 3) (B := 4)
  (by norm_num) (by norm_num) (pinsker_unif (by norm_num)) (pit_mono (by norm_num))
  (pit_mem (by norm_num)) (fun _ => pit 3) (fun _ => pit_mem (by norm_num)) (τ := 1 / 10)
  (by norm_num)
example := certified_error_prob_proved (MeasureTheory.Measure.dirac ()) (n := 3) (m := 3) (B := 4)
  (by norm_num) (by norm_num) (pit_mono (by norm_num)) (pit_mem (by norm_num)) (fun _ => pit 3)
  (fun _ => pit_mem (by norm_num)) (τ := 1 / 10) (by norm_num)
example := gate_sound (C0 := 10) (CT := 3) (cT := 1) (δ := 1) (U := 4) (L := 9) (by norm_num)
  (by norm_num) (by norm_num)
example := gate_false_positive (MeasureTheory.Measure.dirac ()) (fun _ => 10) (fun _ => 3)
  (fun _ => 4) (fun _ => 9) (fun _ => 0) 1 1 1 (m := 3) (fun _ _ => ⟨by norm_num, by norm_num⟩)
  (by norm_num)

/-! ### Theorem P (Pinsker) -/

example := psi_nonneg (x := 2) (by norm_num)
example := klTerm_lower (p := 0) (q := 1) le_rfl one_pos
example := pinsker_tv (fun _ : Fin 2 => (1 : ℝ) / 2) (fun _ => 1 / 2) (fun _ => by norm_num)
  (fun _ => by norm_num) (by norm_num [Fin.sum_univ_two]) (by norm_num [Fin.sum_univ_two])
example := pinsker_fin (fun _ : Fin 2 => (1 : ℝ) / 2) (fun _ => 1 / 2) (fun _ => by norm_num)
  (fun _ => by norm_num) (by norm_num [Fin.sum_univ_two]) (by norm_num [Fin.sum_univ_two])

/-- The gate's KL is not degenerate: a point mass on two bins has `KL = log 2 > 0`. -/
example : klUnif ![(1 : ℝ), 0] = Real.log 2 := by
  simp [klUnif, Fin.sum_univ_two]

example := pinsker_unif (B := 2) (by norm_num) ![(1 : ℝ), 0]
  (fun b => by fin_cases b <;> simp) (by simp [Fin.sum_univ_two])
example := klUnif_eq_sum_klFun (B := 2) (by norm_num) (fun _ => 1 / 2)
  (by norm_num [Fin.sum_univ_two])
example := klUnif_nonneg (B := 2) (by norm_num) (fun _ => 1 / 2) (fun _ => by norm_num)
  (by norm_num [Fin.sum_univ_two])

/-! ### §6 Limits -/

example := piece_composite (natKeys 3) id univ (α := 1) (β := 0) (a := 1) (b := 1) (ε := 0)
  (fun i _ => by simp)
  (fun i _ => abs_nonpos_iff.mpr (by simp only [natKeys, id_eq, one_mul]; ring))
example := kpiece_composite (K := 1) (natKeys 3) id (fun _ => 0) (fun _ => 1) (fun _ => 0)
  (fun i => by simp) (a := 1) (b := 1) (ε := 0)
  (fun i => abs_nonpos_iff.mpr (by simp only [natKeys, id_eq, one_mul]; ring))
example := run_diff (natKeys 3) (s := 1) (b := 1) (ε := 0) (i := ⟨0, by norm_num⟩)
  (i' := ⟨2, by norm_num⟩) (abs_nonpos_iff.mpr (by simp only [natKeys]; ring))
  (abs_nonpos_iff.mpr (by simp only [natKeys]; ring))
example := two_run (S₁ := 1) (S₂ := 2) (h := 3) (s := 2) one_pos two_pos
example := rms_lower (J := 2) (fun _ => 1) 1 1 (by norm_num)

lemma runs_ok (j : Fin 2) : |2 * ((j.val : ℝ) + 1) - 3| ≤ 2 * (1 / 2) := by
  have hj := j.isLt
  have h0 : (0 : ℝ) ≤ j.val := Nat.cast_nonneg _
  have h1 : (j.val : ℝ) ≤ 1 := by exact_mod_cast (by omega : j.val ≤ 1)
  rw [abs_le]
  constructor <;> linarith

/-- Spans `1, 2` (CV `1/3`), `h = 3`, slope `2`, error `1/2`: the floor is `3/(2√10) ≈ 0.474`. -/
example := cv_floor (J := 2) (by norm_num) (fun j : Fin 2 => (j.val : ℝ) + 1)
  (fun j => by positivity) (h := 3) (s := 2) (ε := 1 / 2) runs_ok

example := span_mvt (T := id) (T' := fun _ => 1) (lo := 0) (hi := 3)
  (fun y _ => hasDerivAt_id y) (a := 0) (b := 2) le_rfl (by norm_num) (by norm_num)
example := span_ratio (T := id) (T' := fun _ => 1) (lo := 0) (hi := 3) (Λ := 0)
  (fun y _ => hasDerivAt_id y) (fun _ _ => one_pos) (fun _ _ _ _ => by simp)
  (a₁ := 0) (b₁ := 2) (a₂ := 2) (b₂ := 3) le_rfl (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) le_rfl
example := ratio_form (S₁ := 2) (S₂ := 1) (r := 2) two_pos one_pos one_le_two (by norm_num)
example := smooth_floor (T := id) (T' := fun _ => 1) (lo := 0) (hi := 3) (Λ := 0)
  (fun y _ => hasDerivAt_id y) (fun _ _ => one_pos) (fun _ _ _ _ => by simp)
  (a₁ := 0) (b₁ := 2) (a₂ := 2) (b₂ := 3) (s := 1) (h := 1) (ε := 1 / 2)
  le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num) le_rfl (by norm_num)
  (by norm_num) (by norm_num) (by norm_num)

end NFLProof.Sanity
