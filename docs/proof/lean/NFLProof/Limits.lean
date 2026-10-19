module

public import NFLProof.Defs

/-!
# §6 Limits of a finite-resolution transform

Lean names (PROOF.md numbering):

* Theorem 6.1 `piece_composite`, `kpiece_composite`: on a piece where `T` is affine, an affine
  model on `T(x)` is an affine model on `x`; a `K`-piece affine `T` followed by one affine model is
  a `K`-segment piecewise-linear model of the raw keys.
* Lemma 6.2 `run_diff`, `two_run`: two runs of `h` gaps with different spans force an error.
* Theorem 6.3 `rms_lower`, `spanCV`, `cv_floor`: the coefficient-of-variation floor.
* Theorem 6.4 `span_ratio`, `ratio_form`, `smooth_floor`: a transform whose `ln T'` varies by at
  most `Λ` changes a density contrast by at most `e^Λ`, so the floor survives.
-/

@[expose] public section

open Finset

namespace NFLProof

/-! ### Theorem 6.1 -/

/-- **Theorem 6.1 (one piece).** If `T` is affine on the keys of a piece `P`, any affine model
`a z + b` on the transformed keys equals an affine model on the raw keys there, with the same
errors. So no affine model after `T` beats the best raw affine fit `δ(P)` of the piece. -/
theorem piece_composite {n : ℕ} (x : Fin n → ℝ) (T : ℝ → ℝ) (P : Finset (Fin n))
    {α β a b ε : ℝ} (hT : ∀ i ∈ P, T (x i) = α * x i + β)
    (hε : ∀ i ∈ P, |a * T (x i) + b - (i.val + 1)| ≤ ε) :
    ∃ a' b' : ℝ, ∀ i ∈ P, |a' * x i + b' - (i.val + 1)| ≤ ε := by
  refine ⟨a * α, a * β + b, fun i hi => ?_⟩
  have := hε i hi
  rw [hT i hi] at this
  calc |a * α * x i + (a * β + b) - (i.val + 1)| = |a * (α * x i + β) + b - (i.val + 1)| := by
        ring_nf
    _ ≤ ε := this

/-- **Theorem 6.1 (`K` pieces).** If `T` is affine with parameters `(α k, β k)` on the keys of
piece `k = piece i`, an affine model on `T(x)` with error `≤ ε` is a `K`-segment piecewise-linear
model of the raw keys with error `≤ ε`. Hence its error is at least the optimal `K`-segment
error `ε*_K(X)`. -/
theorem kpiece_composite {n K : ℕ} (x : Fin n → ℝ) (T : ℝ → ℝ) (piece : Fin n → Fin K)
    (α β : Fin K → ℝ) (hT : ∀ i, T (x i) = α (piece i) * x i + β (piece i)) {a b ε : ℝ}
    (hε : ∀ i, |a * T (x i) + b - (i.val + 1)| ≤ ε) :
    ∃ a' b' : Fin K → ℝ, ∀ i, |a' (piece i) * x i + b' (piece i) - (i.val + 1)| ≤ ε := by
  refine ⟨fun k => a * α k, fun k => a * β k + b, fun i => ?_⟩
  have := hε i
  rw [hT i] at this
  calc |a * α (piece i) * x i + (a * β (piece i) + b) - (i.val + 1)|
      = |a * (α (piece i) * x i + β (piece i)) + b - (i.val + 1)| := by ring_nf
    _ ≤ ε := this

/-! ### Lemma 6.2 -/

/-- **Lemma 6.2 (first claim).** If an affine model with slope `s` has error at most `ε` at two
keys, then `s` times their key distance is within `2ε` of their rank distance. -/
theorem run_diff {n : ℕ} (x : Fin n → ℝ) {s b ε : ℝ} {i i' : Fin n}
    (hi : |s * x i + b - (i.val + 1)| ≤ ε) (hi' : |s * x i' + b - (i'.val + 1)| ≤ ε) :
    |s * (x i' - x i) - ((i'.val : ℝ) - i.val)| ≤ 2 * ε := by
  have h1 := abs_le.mp hi
  have h2 := abs_le.mp hi'
  rw [abs_le]
  constructor <;> nlinarith [h1.1, h1.2, h2.1, h2.2]

/-- **Lemma 6.2 (two runs).** For spans `S₁, S₂ > 0` of two runs of `h` gaps, every slope `s`
leaves an error of at least `h |S₁ - S₂| / (S₁ + S₂)` on one of them. (No sign condition on `h`
is needed.) -/
theorem two_run {S₁ S₂ h s : ℝ} (h₁ : 0 < S₁) (h₂ : 0 < S₂) :
    h * |S₁ - S₂| / (S₁ + S₂) ≤ max |s * S₁ - h| |s * S₂ - h| := by
  set M := max |s * S₁ - h| |s * S₂ - h|
  have hM1 : |s * S₁ - h| ≤ M := le_max_left _ _
  have hM2 : |s * S₂ - h| ≤ M := le_max_right _ _
  have a1 := abs_le.mp hM1
  have a2 := abs_le.mp hM2
  rw [div_le_iff₀ (by linarith)]
  -- `M (S₁ + S₂) ≥ (h - s S₁) S₂ + (s S₂ - h) S₁ = h (S₂ - S₁)`, and symmetrically
  rcases le_total S₁ S₂ with hle | hle
  · rw [abs_of_nonpos (by linarith)]
    nlinarith [mul_le_mul_of_nonneg_right a1.1 h₂.le, mul_le_mul_of_nonneg_right a2.2 h₁.le]
  · rw [abs_of_nonneg (by linarith)]
    nlinarith [mul_le_mul_of_nonneg_right a1.2 h₂.le, mul_le_mul_of_nonneg_right a2.1 h₁.le]

/-! ### Theorem 6.3 -/

/-- **Theorem 6.3 (the quadratic).** Over all slopes `s`, `∑ (s S_j - h)²` is at least
`J h² - h² (∑ S)² / ∑ S²`. -/
theorem rms_lower {J : ℕ} (S : Fin J → ℝ) (h s : ℝ) (hS : 0 < ∑ j, S j ^ 2) :
    J * h ^ 2 - h ^ 2 * (∑ j, S j) ^ 2 / (∑ j, S j ^ 2) ≤ ∑ j, (s * S j - h) ^ 2 := by
  set A := ∑ j, S j ^ 2
  set C := ∑ j, S j
  have hexp : ∑ j, (s * S j - h) ^ 2 = s ^ 2 * A - 2 * s * h * C + J * h ^ 2 := by
    have : ∀ j, (s * S j - h) ^ 2 = s ^ 2 * S j ^ 2 - 2 * s * h * S j + h ^ 2 := fun j => by ring
    simp only [this, sum_add_distrib, sum_sub_distrib, ← mul_sum, sum_const, card_univ,
      Fintype.card_fin, nsmul_eq_mul, A, C]
  rw [hexp]
  -- `A (s - hC/A)² ≥ 0`
  have hsq : 0 ≤ A * (s - h * C / A) ^ 2 := by positivity
  have e : A * (s - h * C / A) ^ 2 = s ^ 2 * A - 2 * s * h * C + h ^ 2 * C ^ 2 / A := by
    field_simp; ring
  rw [e] at hsq
  linarith

/-- The coefficient of variation of the run spans: `√Var(S) / mean(S)`. -/
noncomputable def spanCV {J : ℕ} (S : Fin J → ℝ) : ℝ :=
  Real.sqrt ((∑ j, S j ^ 2) / J - ((∑ j, S j) / J) ^ 2) / ((∑ j, S j) / J)

/-- **Theorem 6.3.** If one affine model has slope `s` on a piece containing `J` runs of `h` gaps
with spans `S_j > 0`, and its error is at most `ε` (so `|s S_j - h| ≤ 2ε` by Lemma 6.2), then
`ε ≥ (h/2) · CV / √(1 + CV²)`. -/
theorem cv_floor {J : ℕ} (hJ : 0 < J) (S : Fin J → ℝ) (hS : ∀ j, 0 < S j) {h s ε : ℝ}
    (hrun : ∀ j, |s * S j - h| ≤ 2 * ε) :
    h / 2 * (spanCV S / Real.sqrt (1 + spanCV S ^ 2)) ≤ ε := by
  have hJ' : (0 : ℝ) < J := by exact_mod_cast hJ
  have hne : (univ : Finset (Fin J)).Nonempty := ⟨⟨0, hJ⟩, mem_univ _⟩
  set A := ∑ j, S j ^ 2 with hA
  set C := ∑ j, S j with hC
  have hApos : 0 < A := sum_pos (fun j _ => by have := hS j; positivity) hne
  have hCpos : 0 < C := sum_pos (fun j _ => hS j) hne
  have hε0 : 0 ≤ ε := by have := (abs_nonneg _).trans (hrun ⟨0, hJ⟩); linarith
  set M := C / J with hM
  set Q := A / J with hQ
  set V := Q - M ^ 2 with hV
  have hMpos : 0 < M := by positivity
  have hQpos : 0 < Q := by positivity
  -- `Var ≥ 0`: `∑ (S_j - M)² = A - J M² ≥ 0`
  have hV0 : 0 ≤ V := by
    have h0 : 0 ≤ ∑ j, (S j - M) ^ 2 := sum_nonneg fun j _ => sq_nonneg _
    have : ∑ j, (S j - M) ^ 2 = A - 2 * M * C + J * M ^ 2 := by
      have : ∀ j, (S j - M) ^ 2 = S j ^ 2 - 2 * M * S j + M ^ 2 := fun j => by ring
      simp only [this, sum_add_distrib, sum_sub_distrib, ← mul_sum, sum_const, card_univ,
        Fintype.card_fin, nsmul_eq_mul, A, C]
    rw [this] at h0
    have e : A - 2 * M * C + J * M ^ 2 = J * V := by
      simp only [hV, hQ, hM]; field_simp; ring
    rw [e] at h0
    exact nonneg_of_mul_nonneg_right (by linarith) hJ'
  -- `CV / √(1 + CV²) = √V / √Q`
  have hcv : spanCV S / Real.sqrt (1 + spanCV S ^ 2) = Real.sqrt V / Real.sqrt Q := by
    have hcvdef : spanCV S = Real.sqrt V / M := rfl
    have h1 : 1 + spanCV S ^ 2 = Q / M ^ 2 := by
      rw [hcvdef, div_pow, Real.sq_sqrt hV0, hV]; field_simp; ring
    rw [h1, Real.sqrt_div' _ (sq_nonneg M), Real.sqrt_sq hMpos.le, hcvdef]
    field_simp
  rw [hcv]
  -- the mean of `(s S_j - h)²` is at most `(2ε)²`, and at least `h² V / Q`
  have hsum : ∑ j, (s * S j - h) ^ 2 ≤ J * (2 * ε) ^ 2 := by
    have : ∀ j ∈ (univ : Finset (Fin J)), (s * S j - h) ^ 2 ≤ (2 * ε) ^ 2 := fun j _ => by
      rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) (hrun j) 2
    have := sum_le_sum this
    simpa [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul] using this
  have hlow := rms_lower S h s hApos
  have hkey : h ^ 2 * (V / Q) ≤ (2 * ε) ^ 2 := by
    have e : h ^ 2 * (V / Q) = (J * h ^ 2 - h ^ 2 * C ^ 2 / A) / J := by
      simp only [hV, hQ, hM]; field_simp
    rw [e, div_le_iff₀ hJ']
    linarith
  rcases le_or_gt h 0 with hh | hh
  · have : 0 ≤ Real.sqrt V / Real.sqrt Q := by positivity
    nlinarith
  · -- take square roots
    have hsq : Real.sqrt (h ^ 2 * (V / Q)) ≤ Real.sqrt ((2 * ε) ^ 2) := Real.sqrt_le_sqrt hkey
    rw [Real.sqrt_sq (by linarith), Real.sqrt_mul (sq_nonneg h), Real.sqrt_sq hh.le,
      Real.sqrt_div' _ hQpos.le] at hsq
    have : h / 2 * (Real.sqrt V / Real.sqrt Q) = (h * (Real.sqrt V / Real.sqrt Q)) / 2 := by ring
    rw [this]
    linarith

/-! ### Theorem 6.4 -/

/-- The mean value theorem in the form used below: on `[a, b] ⊆ [lo, hi]`, the span
`T b - T a` equals `T'(ξ) (b - a)` for some `ξ ∈ (a, b)`. -/
lemma span_mvt {T T' : ℝ → ℝ} {lo hi : ℝ}
    (hd : ∀ y ∈ Set.Icc lo hi, HasDerivAt T (T' y) y) {a b : ℝ} (ha : lo ≤ a) (hab : a < b)
    (hb : b ≤ hi) : ∃ ξ ∈ Set.Icc lo hi, T b - T a = T' ξ * (b - a) := by
  have hcont : ContinuousOn T (Set.Icc a b) := fun y hy =>
    (hd y ⟨ha.trans hy.1, hy.2.trans hb⟩).continuousAt.continuousWithinAt
  obtain ⟨ξ, hξ, heq⟩ := exists_hasDerivAt_eq_slope T T' hab hcont
    (fun y hy => hd y ⟨ha.trans hy.1.le, hy.2.le.trans hb⟩)
  refine ⟨ξ, ⟨ha.trans hξ.1.le, hξ.2.le.trans hb⟩, ?_⟩
  rw [heq]; field_simp [(sub_pos.mpr hab).ne']

/-- **Theorem 6.4 (spans).** If `T' > 0` and `T'(y) ≤ e^Λ T'(y')` on `[lo, hi]`, the ratio of two
transformed spans is at least `e^{-Λ}` times the ratio of the raw spans. -/
theorem span_ratio {T T' : ℝ → ℝ} {lo hi Λ : ℝ}
    (hd : ∀ y ∈ Set.Icc lo hi, HasDerivAt T (T' y) y) (hpos : ∀ y ∈ Set.Icc lo hi, 0 < T' y)
    (hrange : ∀ y ∈ Set.Icc lo hi, ∀ y' ∈ Set.Icc lo hi, T' y ≤ Real.exp Λ * T' y')
    {a₁ b₁ a₂ b₂ : ℝ} (h₁ : lo ≤ a₁) (h₁' : a₁ < b₁) (h₁'' : b₁ ≤ hi)
    (h₂ : lo ≤ a₂) (h₂' : a₂ < b₂) (h₂'' : b₂ ≤ hi) :
    Real.exp (-Λ) * ((b₁ - a₁) / (b₂ - a₂)) ≤ (T b₁ - T a₁) / (T b₂ - T a₂) := by
  obtain ⟨ξ₁, hξ₁, e₁⟩ := span_mvt hd h₁ h₁' h₁''
  obtain ⟨ξ₂, hξ₂, e₂⟩ := span_mvt hd h₂ h₂' h₂''
  have p₁ := hpos ξ₁ hξ₁
  have p₂ := hpos ξ₂ hξ₂
  have r := hrange ξ₂ hξ₂ ξ₁ hξ₁
  have d₁ : 0 < b₁ - a₁ := by linarith
  have d₂ : 0 < b₂ - a₂ := by linarith
  rw [e₁, e₂]
  -- `e^{-Λ} ≤ T'(ξ₁)/T'(ξ₂)` because `T'(ξ₂) ≤ e^Λ T'(ξ₁)`
  have hq : Real.exp (-Λ) * T' ξ₂ ≤ T' ξ₁ := by
    have := mul_le_mul_of_nonneg_left r (Real.exp_pos (-Λ)).le
    rwa [← mul_assoc, ← Real.exp_add, neg_add_cancel, Real.exp_zero, one_mul] at this
  rw [mul_div_assoc', div_le_div_iff₀ d₂ (by positivity)]
  have : Real.exp (-Λ) * (b₁ - a₁) * (T' ξ₂ * (b₂ - a₂)) =
      (Real.exp (-Λ) * T' ξ₂) * ((b₁ - a₁) * (b₂ - a₂)) := by ring
  rw [this]
  have : T' ξ₁ * (b₁ - a₁) * (b₂ - a₂) = T' ξ₁ * ((b₁ - a₁) * (b₂ - a₂)) := by ring
  rw [this]
  exact mul_le_mul_of_nonneg_right hq (by positivity)

/-- **Theorem 6.4 (monotone form).** `(r - 1)/(r + 1)` is a lower bound for
`|S₁ - S₂| / (S₁ + S₂)` whenever `1 ≤ r ≤ S₁/S₂`. -/
theorem ratio_form {S₁ S₂ r : ℝ} (h₁ : 0 < S₁) (h₂ : 0 < S₂) (hr : 1 ≤ r)
    (hrat : r ≤ S₁ / S₂) : (r - 1) / (r + 1) ≤ |S₁ - S₂| / (S₁ + S₂) := by
  have hS : r * S₂ ≤ S₁ := by rwa [le_div_iff₀ h₂] at hrat
  have hge : S₂ ≤ S₁ := by nlinarith
  rw [abs_of_nonneg (by linarith), div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith

/-- **Theorem 6.4 (the floor).** Let `T` be differentiable with `T' > 0` and `ln T'` varying by at
most `Λ` on `[lo, hi]`, and let two key runs `[a₁, b₁]`, `[a₂, b₂]` inside it each span `h`
ranks, with raw span ratio `ρ = (b₁ - a₁)/(b₂ - a₂) ≥ e^Λ`. If an affine model with slope `s` on
`z = T(x)` reproduces each run's `h` ranks to within `2ε` (Lemma 6.2), then
`ε ≥ (h/2) (ρ e^{-Λ} - 1)/(ρ e^{-Λ} + 1)`. -/
theorem smooth_floor {T T' : ℝ → ℝ} {lo hi Λ : ℝ}
    (hd : ∀ y ∈ Set.Icc lo hi, HasDerivAt T (T' y) y) (hpos : ∀ y ∈ Set.Icc lo hi, 0 < T' y)
    (hrange : ∀ y ∈ Set.Icc lo hi, ∀ y' ∈ Set.Icc lo hi, T' y ≤ Real.exp Λ * T' y')
    {a₁ b₁ a₂ b₂ s h ε : ℝ} (h₁ : lo ≤ a₁) (h₁' : a₁ < b₁) (h₁'' : b₁ ≤ hi)
    (h₂ : lo ≤ a₂) (h₂' : a₂ < b₂) (h₂'' : b₂ ≤ hi) (hh : 0 ≤ h)
    (hrun₁ : |s * (T b₁ - T a₁) - h| ≤ 2 * ε) (hrun₂ : |s * (T b₂ - T a₂) - h| ≤ 2 * ε)
    (hρ : Real.exp Λ ≤ (b₁ - a₁) / (b₂ - a₂)) :
    h / 2 * (((b₁ - a₁) / (b₂ - a₂) * Real.exp (-Λ) - 1) /
      ((b₁ - a₁) / (b₂ - a₂) * Real.exp (-Λ) + 1)) ≤ ε := by
  -- the transformed spans are positive
  obtain ⟨ξ₁, hξ₁, e₁⟩ := span_mvt hd h₁ h₁' h₁''
  obtain ⟨ξ₂, hξ₂, e₂⟩ := span_mvt hd h₂ h₂' h₂''
  have S₁pos : 0 < T b₁ - T a₁ := by rw [e₁]; exact mul_pos (hpos ξ₁ hξ₁) (by linarith)
  have S₂pos : 0 < T b₂ - T a₂ := by rw [e₂]; exact mul_pos (hpos ξ₂ hξ₂) (by linarith)
  set r := (b₁ - a₁) / (b₂ - a₂) * Real.exp (-Λ) with hr
  have hr1 : 1 ≤ r := by
    have := mul_le_mul_of_nonneg_right hρ (Real.exp_pos (-Λ)).le
    rwa [← Real.exp_add, add_neg_cancel, Real.exp_zero] at this
  have hrat : r ≤ (T b₁ - T a₁) / (T b₂ - T a₂) := by
    rw [hr, mul_comm]; exact span_ratio hd hpos hrange h₁ h₁' h₁'' h₂ h₂' h₂''
  have hf := ratio_form S₁pos S₂pos hr1 hrat
  have ht := two_run (h := h) (s := s) S₁pos S₂pos
  have hmax : max |s * (T b₁ - T a₁) - h| |s * (T b₂ - T a₂) - h| ≤ 2 * ε := max_le hrun₁ hrun₂
  have : h * ((r - 1) / (r + 1)) ≤ 2 * ε := by
    calc h * ((r - 1) / (r + 1))
        ≤ h * (|(T b₁ - T a₁) - (T b₂ - T a₂)| / ((T b₁ - T a₁) + (T b₂ - T a₂))) :=
          mul_le_mul_of_nonneg_left hf hh
      _ = h * |(T b₁ - T a₁) - (T b₂ - T a₂)| / ((T b₁ - T a₁) + (T b₂ - T a₂)) := by ring
      _ ≤ _ := ht.trans hmax
  linarith

end NFLProof
