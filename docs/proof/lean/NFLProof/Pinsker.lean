module

public import NFLProof.Gate

/-!
# Pinsker's inequality for finite distributions (proved, not assumed)

The pinned Mathlib (`v4.35.0`) does not contain Pinsker's inequality (PROOF.md's Theorem P). For
the **finite** distributions that the gate uses (histograms on `B` bins), it can be proved
directly, and this file does so. As a result, Theorem 5.4 (a) holds with no Pinsker
premise at all (`certified_error_even_proved` in this file).

Proof (the classical one through a pointwise bound and Cauchy-Schwarz):

1. `psi_nonneg`: for `x > 0`, `2x(x+2) log x ≥ (x-1)(5x+1)`. Calculus: the derivative of the left
   minus the right is `4 χ(x)` with `χ(x) = (x+1) log x - 2(x-1)`, and `χ` is increasing
   (`χ' = log x + 1/x - 1 ≥ 0`, Mathlib's `Real.one_sub_inv_le_log_of_pos`) with `χ(1) = 0`.
2. `klTerm_lower`: for `p ≥ 0`, `q > 0`:
   `3 (p - q)² ≤ 2 (p + 2q) (p log (p/q) - p + q)`. (With `x = p/q`, this is step 1.)
3. `pinsker_fin`: for probability vectors `p ≥ 0` and `q > 0`,
   `(∑ |p - q|)² ≤ 2 ∑ p log (p/q)`, by Sedrakyan's form of Cauchy-Schwarz
   (Mathlib's `Finset.sq_sum_div_le_sum_sq_div`) with weights `p + 2q`, which sum to `3`.
4. `pinsker_tv`: `TV(p, q) = ½ ∑ |p - q| ≤ √(KL(p ‖ q) / 2)`.
5. `pinsker_unif : PinskerUnif B` for every `B > 0`: Theorem P of PROOF.md for the gate's bins.

`klUnif_eq_sum_klFun` relates our explicit KL to Mathlib's `InformationTheory.klFun`
(`klFun x = x log x + 1 - x`, the generator of Mathlib's `klDiv`):
`KL(p ‖ unif) = ∑_b (1/B) · klFun (B p_b)`.
-/

@[expose] public section

open Finset Real

namespace NFLProof

/-! ### Step 1: the calculus lemma -/

/-- `χ(x) = (x + 1) log x - 2 (x - 1)`. -/
noncomputable def chiP (x : ℝ) : ℝ := (x + 1) * Real.log x - 2 * (x - 1)

/-- `ψ(x) = 2 x (x + 2) log x - (x - 1)(5 x + 1)`. -/
noncomputable def psiP (x : ℝ) : ℝ := 2 * x * (x + 2) * Real.log x - (x - 1) * (5 * x + 1)

lemma hasDerivAt_chiP {x : ℝ} (hx : 0 < x) :
    HasDerivAt chiP (Real.log x + x⁻¹ - 1) x := by
  -- product rule for `(x + 1) log x`, then the linear term
  have h1 : HasDerivAt (fun y : ℝ => y + 1) 1 x := (hasDerivAt_id x).add_const 1
  have h2 := Real.hasDerivAt_log hx.ne'
  have h3 := (h1.mul h2).sub (((hasDerivAt_id x).sub_const 1).const_mul 2)
  convert h3 using 1
  · ext y; simp [chiP]
  · field_simp; ring

lemma hasDerivAt_psiP {x : ℝ} (hx : 0 < x) : HasDerivAt psiP (4 * chiP x) x := by
  -- product rule for `2x(x+2) · log x`, then the quadratic `(x - 1)(5x + 1)`
  have hq : HasDerivAt (fun y : ℝ => 2 * y * (y + 2)) (2 * 1 * (x + 2) + 2 * x * 1) x :=
    ((hasDerivAt_id x).const_mul 2).mul ((hasDerivAt_id x).add_const 2) |>.congr_deriv (by simp)
  have hl := Real.hasDerivAt_log hx.ne'
  have hr : HasDerivAt (fun y : ℝ => (y - 1) * (5 * y + 1)) (1 * (5 * x + 1) + (x - 1) * 5) x :=
    ((hasDerivAt_id x).sub_const 1).mul (((hasDerivAt_id x).const_mul 5).add_const 1)
      |>.congr_deriv (by simp)
  have h := (hq.mul hl).sub hr
  convert h using 1
  · ext y; simp [psiP]
  · simp only [chiP]; field_simp; ring

/-- `χ` is increasing on `(0, ∞)`, since `χ'(x) = log x + 1/x - 1 ≥ 0`. -/
lemma chiP_monotoneOn : MonotoneOn chiP (Set.Ioi 0) := by
  apply monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ioi 0)
  · intro x hx; exact (hasDerivAt_chiP hx).continuousAt.continuousWithinAt
  · intro x hx
    rw [interior_Ioi] at hx
    exact (hasDerivAt_chiP hx).hasDerivWithinAt
  · intro x hx
    rw [interior_Ioi] at hx
    have := Real.one_sub_inv_le_log_of_pos hx
    linarith

lemma chiP_one : chiP 1 = 0 := by simp [chiP]

lemma psiP_one : psiP 1 = 0 := by simp [psiP]

/-- **Step 1.** `ψ(x) ≥ 0` for `x > 0`: `ψ` decreases on `(0, 1]` and increases on `[1, ∞)`,
because `ψ' = 4χ` has the sign of `x - 1`, and `ψ(1) = 0`. -/
theorem psi_nonneg {x : ℝ} (hx : 0 < x) : 0 ≤ psiP x := by
  rcases le_total x 1 with h1 | h1
  · -- on `(0, 1]`: `ψ' = 4 χ ≤ 0`, so `ψ` is antitone and `ψ x ≥ ψ 1 = 0`
    have hanti : AntitoneOn psiP (Set.Ioc 0 1) := by
      apply antitoneOn_of_hasDerivWithinAt_nonpos (convex_Ioc 0 1)
      · intro y hy; exact (hasDerivAt_psiP hy.1).continuousAt.continuousWithinAt
      · intro y hy
        rw [interior_Ioc] at hy
        exact (hasDerivAt_psiP hy.1).hasDerivWithinAt
      · intro y hy
        rw [interior_Ioc] at hy
        have : chiP y ≤ chiP 1 :=
          chiP_monotoneOn hy.1 (Set.mem_Ioi.mpr one_pos) hy.2.le
        rw [chiP_one] at this
        linarith
    have := hanti ⟨hx, h1⟩ ⟨one_pos, le_rfl⟩ h1
    rwa [psiP_one] at this
  · -- on `[1, ∞)`: `ψ' = 4 χ ≥ 0`, so `ψ` is monotone and `ψ x ≥ ψ 1 = 0`
    have hmono : MonotoneOn psiP (Set.Ici 1) := by
      apply monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ici 1)
      · intro y hy
        exact (hasDerivAt_psiP (lt_of_lt_of_le one_pos hy)).continuousAt.continuousWithinAt
      · intro y hy
        rw [interior_Ici] at hy
        exact (hasDerivAt_psiP (lt_trans one_pos hy)).hasDerivWithinAt
      · intro y hy
        rw [interior_Ici] at hy
        have : chiP 1 ≤ chiP y :=
          chiP_monotoneOn (Set.mem_Ioi.mpr one_pos) (Set.mem_Ioi.mpr (lt_trans one_pos hy)) hy.le
        rw [chiP_one] at this
        linarith
    have := hmono (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr h1) h1
    rwa [psiP_one] at this

/-! ### Step 2: the pointwise bound on one KL term -/

/-- **Step 2.** For `p ≥ 0` and `q > 0`:
`3 (p - q)² ≤ 2 (p + 2q) (p log (p/q) - p + q)`. -/
theorem klTerm_lower {p q : ℝ} (hp : 0 ≤ p) (hq : 0 < q) :
    3 * (p - q) ^ 2 ≤ 2 * (p + 2 * q) * (p * Real.log (p / q) - p + q) := by
  rcases hp.eq_or_lt with h0 | hpos
  · -- `p = 0`: the claim is `3 q² ≤ 4 q²`
    subst h0
    simp only [zero_div, Real.log_zero, mul_zero, zero_sub, zero_add, sub_zero]
    nlinarith [sq_nonneg q]
  · -- `p > 0`: put `x = p / q`, so `p = x q`, and use `ψ(x) ≥ 0`
    set x := p / q with hxdef
    have hx : 0 < x := div_pos hpos hq
    have hpx : p = x * q := by rw [hxdef]; field_simp
    have hψ := psi_nonneg hx
    simp only [psiP] at hψ
    rw [hpx]
    -- both sides are `q²` times a function of `x`; the difference is `q² ψ(x)`
    have e : 2 * (x * q + 2 * q) * (x * q * Real.log x - x * q + q) - 3 * (x * q - q) ^ 2 =
        q ^ 2 * (2 * x * (x + 2) * Real.log x - (x - 1) * (5 * x + 1)) := by ring
    have : 0 ≤ q ^ 2 * (2 * x * (x + 2) * Real.log x - (x - 1) * (5 * x + 1)) :=
      mul_nonneg (sq_nonneg q) hψ
    linarith

/-! ### Steps 3-4: Pinsker for finite distributions -/

/-- KL divergence of finite distributions, `∑ p_b log (p_b / q_b)` (with `0 log 0 = 0`). -/
noncomputable def klFin {ι : Type*} [Fintype ι] (p q : ι → ℝ) : ℝ :=
  ∑ b, p b * Real.log (p b / q b)

/-- Total variation of finite distributions, `½ ∑ |p_b - q_b|`. (Mathlib's `tvDist` has no `½`.) -/
noncomputable def tvFin {ι : Type*} [Fintype ι] (p q : ι → ℝ) : ℝ :=
  (1 / 2) * ∑ b, |p b - q b|

/-- **Step 3 (Pinsker, squared `ℓ¹` form).** For probability vectors `p ≥ 0` and `q > 0`:
`(∑ |p - q|)² ≤ 2 KL(p ‖ q)`. -/
theorem pinsker_fin {ι : Type*} [Fintype ι] (p q : ι → ℝ) (hp : ∀ b, 0 ≤ p b)
    (hq : ∀ b, 0 < q b) (hp1 : ∑ b, p b = 1) (hq1 : ∑ b, q b = 1) :
    (∑ b, |p b - q b|) ^ 2 ≤ 2 * klFin p q := by
  have hw : ∀ b ∈ (univ : Finset ι), 0 < p b + 2 * q b := fun b _ => by
    have := hp b; have := hq b; linarith
  -- the weights `p + 2q` sum to 3
  have hW : ∑ b, (p b + 2 * q b) = 3 := by
    rw [sum_add_distrib, ← mul_sum, hp1, hq1]; norm_num
  -- Sedrakyan: `(∑ |p - q|)² / 3 ≤ ∑ (p - q)² / (p + 2q)`
  have hT := sq_sum_div_le_sum_sq_div univ (fun b => |p b - q b|) hw
  rw [hW] at hT
  simp only [sq_abs] at hT
  -- pointwise: `(p - q)² / (p + 2q) ≤ (2/3) (p log (p/q) - p + q)`
  have hpt : ∀ b ∈ (univ : Finset ι),
      (p b - q b) ^ 2 / (p b + 2 * q b) ≤ (2 / 3) * (p b * Real.log (p b / q b) - p b + q b) := by
    intro b hb
    rw [div_le_iff₀ (hw b hb)]
    have := klTerm_lower (hp b) (hq b)
    nlinarith
  have hsum := sum_le_sum hpt
  -- `∑ (p log (p/q) - p + q) = KL`, because `∑ p = ∑ q`
  have hKL : ∑ b, (2 / 3) * (p b * Real.log (p b / q b) - p b + q b) = (2 / 3) * klFin p q := by
    rw [← mul_sum]
    congr 1
    rw [sum_add_distrib, sum_sub_distrib, hp1, hq1]
    simp [klFin]
  rw [hKL] at hsum
  have := hT.trans hsum
  linarith

/-- **Step 4 (Pinsker's inequality).** `TV(p, q) ≤ √(KL(p ‖ q) / 2)` for probability vectors
`p ≥ 0` and `q > 0` on a finite type. -/
theorem pinsker_tv {ι : Type*} [Fintype ι] (p q : ι → ℝ) (hp : ∀ b, 0 ≤ p b)
    (hq : ∀ b, 0 < q b) (hp1 : ∑ b, p b = 1) (hq1 : ∑ b, q b = 1) :
    tvFin p q ≤ Real.sqrt (klFin p q / 2) := by
  have h := pinsker_fin p q hp hq hp1 hq1
  have htv : 0 ≤ tvFin p q := by unfold tvFin; positivity
  -- `TV² = (∑|p - q|)² / 4 ≤ KL / 2`
  have hsq : tvFin p q ^ 2 ≤ klFin p q / 2 := by
    unfold tvFin
    nlinarith
  calc tvFin p q = |tvFin p q| := (abs_of_nonneg htv).symm
    _ ≤ _ := Real.abs_le_sqrt hsq

/-- KL against the uniform distribution is `klFin` with `q = 1/B`. -/
lemma klUnif_eq_klFin {B : ℕ} (p : Fin B → ℝ) :
    klUnif p = klFin p (fun _ => (1 : ℝ) / B) := by
  unfold klUnif klFin
  apply sum_congr rfl
  intro b _
  rw [div_div_eq_mul_div, div_one, mul_comm (p b) (B : ℝ)]

lemma tvUnif_eq_tvFin {B : ℕ} (p : Fin B → ℝ) :
    tvUnif p = tvFin p (fun _ => (1 : ℝ) / B) := rfl

/-- **Theorem P of PROOF.md, for the gate's bins.** Pinsker's inequality against the uniform
distribution on `B > 0` bins. -/
theorem pinsker_unif {B : ℕ} (hB : 0 < B) : PinskerUnif B := by
  intro p hp hp1
  have hB' : (0 : ℝ) < B := by exact_mod_cast hB
  rw [tvUnif_eq_tvFin, klUnif_eq_klFin]
  refine pinsker_tv p _ hp (fun _ => by positivity) hp1 ?_
  simp [card_univ]
  field_simp

/-- **Bridge to Mathlib.** Our KL against the uniform distribution is the f-divergence of
Mathlib's `InformationTheory.klFun`: `KL(p ‖ unif) = ∑_b (1/B) klFun (B p_b)`. -/
theorem klUnif_eq_sum_klFun {B : ℕ} (hB : 0 < B) (p : Fin B → ℝ) (hp1 : ∑ b, p b = 1) :
    klUnif p = ∑ b, (1 / (B : ℝ)) * InformationTheory.klFun (B * p b) := by
  have hB' : (B : ℝ) ≠ 0 := by exact_mod_cast hB.ne'
  -- each term is `p log (B p) + 1/B - p`, and `∑ (1/B - p) = 1 - 1 = 0`
  have hterm : ∀ b, (1 / (B : ℝ)) * InformationTheory.klFun (B * p b) =
      p b * Real.log (B * p b) + (1 / B - p b) := by
    intro b
    rw [InformationTheory.klFun_apply]
    field_simp
    ring
  simp only [hterm, sum_add_distrib, sum_sub_distrib, hp1]
  simp [card_univ, klUnif]
  field_simp
  ring

/-- Gibbs' inequality for the gate's statistic: `KL(p ‖ unif) ≥ 0`. -/
theorem klUnif_nonneg {B : ℕ} (hB : 0 < B) (p : Fin B → ℝ) (hp : ∀ b, 0 ≤ p b)
    (hp1 : ∑ b, p b = 1) : 0 ≤ klUnif p := by
  rw [klUnif_eq_sum_klFun hB p hp1]
  apply sum_nonneg
  intro b _
  have hB' : (0 : ℝ) < B := by exact_mod_cast hB
  exact mul_nonneg (by positivity)
    (InformationTheory.klFun_nonneg (mul_nonneg hB'.le (hp b)))

/-! ### Theorem 5.4 with Pinsker proved: no Pinsker premise left -/

/-- **Theorem 5.4 (a), unconditional.** For min-max normalized sorted transformed keys `u`,
`2 ≤ m ≤ n` and `B > 0` bins, the endpoint model's error at every key is at most
`n (√(KL_B/2) + 1/B + 1/m)`, with `KL_B` computed on nf_gate's evenly spaced sample.
No hypothesis: Pinsker is `pinsker_unif`, the sampling bound is Theorem 5.3. -/
theorem certified_error_even_proved {n m B : ℕ} (hm : 2 ≤ m) (hmn : m ≤ n) (hB : 0 < B)
    {u : Fin n → ℝ} (hu : StrictMono u) (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1) (i : Fin n) :
    |(n : ℝ) * u i - (i.val + 1)| ≤
      n * (Real.sqrt (klUnif (hist B (evenSample u m hm hmn)) / 2) + 1 / B + 1 / m) :=
  certified_error_even hm hmn hB (pinsker_unif hB) hu h01 i

/-- **Theorem 5.4 (a), slots, unconditional.** The floored slot `⌊n u_i⌋` is within
`n (√(KL_B/2) + 1/B + 1/m) + 1` of the true 0-indexed slot `i`. -/
theorem certified_slot_err_even_proved {n m B : ℕ} (hm : 2 ≤ m) (hmn : m ≤ n) (hB : 0 < B)
    {u : Fin n → ℝ} (hu : StrictMono u) (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1) (i : Fin n) :
    |((⌊(n : ℝ) * u i⌋ : ℤ) : ℝ) - i.val| ≤
      n * (Real.sqrt (klUnif (hist B (evenSample u m hm hmn)) / 2) + 1 / B + 1 / m) + 1 := by
  set v := evenSample u m hm hmn
  have hv : ∀ k, v k ∈ Set.Icc (0 : ℝ) 1 := fun k => h01 _
  have hP := pinsker_unif hB (hist B v) (hist_nonneg v) (hist_sum (by omega) hB v)
  have h := certified_slot_err hu (Fs := ecdf v) (dev := 1 / m) (tv := tvUnif (hist B v))
    (invB := 1 / B) (kl := klUnif (hist B v))
    (fun t => by rw [abs_sub_comm]; exact (even_sample_dev hm hmn hu t).le)
    (fun t => ks_le_tv_add (by omega) hB v hv t) hP i
  rwa [unifCDF_key h01 i] at h

/-- **Theorem 5.4 (b), with Pinsker proved; DKW is the only premise (`hDKW`).** With
probability at least `1 - 2 e^{-2 m τ²}`, every key's endpoint error is at most
`n (√(KL_B/2) + 1/B + τ)`. -/
theorem certified_error_prob_proved {Ω : Type*} [MeasurableSpace Ω] (P : MeasureTheory.Measure Ω)
    {n m B : ℕ} (hm : 0 < m) (hB : 0 < B)
    {u : Fin n → ℝ} (hu : StrictMono u) (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1)
    (V : Ω → Fin m → ℝ) (hV : ∀ ω k, V ω k ∈ Set.Icc (0 : ℝ) 1) {τ : ℝ}
    (hDKW : P {ω | ∃ t, τ < |ecdf (V ω) t - ecdf u t|} ≤
      ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2))) :
    P {ω | ∃ i : Fin n, (n : ℝ) * (Real.sqrt (klUnif (hist B (V ω)) / 2) + 1 / B + τ) <
        |(n : ℝ) * u i - (i.val + 1)|} ≤ ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2)) :=
  certified_error_prob P hm hB (pinsker_unif hB) hu h01 V hV hDKW

end NFLProof
