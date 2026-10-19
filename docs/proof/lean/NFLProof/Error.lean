module

public import NFLProof.Defs

/-!
# §3 Error: a linear model's rank error is `n` times the Kolmogorov distance, within one rank

Lean names (PROOF.md numbering):

* Def. 3.1 `ksUnif` — the Kolmogorov distance `sup_t |F_u(t) - U(t)|`, defined as a maximum over
  the `2n` one-sided values at the keys; `ecdf_sub_unif_le_ksUnif` and `ksUnif_le` prove that it
  *is* the supremum over all real `t`.
* Def. 3.2 `endpointErr` — the endpoint (min-max) model's error `max_i |n u_i - i|`.
* Theorem 3.1 `err_le_of_ks`, `ks_le_of_err` (any affine model, on the key range), and the
  sandwich `ks_sandwich` : `ε_U ≤ n·D_KS ≤ ε_U + 1`.
* Theorem 3.2 `slot_err` — floored slot error `≤ n·D + 1`.
* Theorem 3.3 `ks_ge_inv_n`, `ks_ge_inv_n_of_normalized`, `pit_ks`, `pit_endpoint_err`,
  `pit_exact`, `zero_error_iff` — the probability integral transform is optimal, and the only
  transforms with zero error are affine functions of the rank.
* Theorem 3.5 `affine_invariance`, `transform_helps`.
* Theorem 3.6 `err_le_ks_add_disp`, `disp_le_err_add_ks` — the displacement term of a
  non-monotone transform.
* Lemma 3.7 `clamp_err`.
-/

@[expose] public section

open Finset Filter Topology

namespace NFLProof

/-- An index exists, so `n > 0` (as a real number). -/
lemma n_pos_of_fin {n : ℕ} (i : Fin n) : (0 : ℝ) < n := by
  have : 0 < n := Fin.pos i
  exact_mod_cast this

/-- Rewrite a CDF-scale error as a rank-scale error: `|n·g - (i+1)| = n · |(i+1)/n - g|`. -/
lemma rank_scale {n : ℕ} (hn : (0 : ℝ) < n) (g c : ℝ) :
    |(n : ℝ) * g - c| = n * |c / n - g| := by
  rw [show (n : ℝ) * g - c = -(n * (c / n - g)) by field_simp; ring, abs_neg, abs_mul,
    abs_of_pos hn]

/-! ### Theorem 3.1 -/

/-- **Theorem 3.1 (a, left; b, upper).** If a CDF-scale model `G` is within `D` of the empirical
CDF on the key range, then its rank prediction `n·G(u_i)` is within `n·D` of the true rank. -/
theorem err_le_of_ks {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) {G : ℝ → ℝ} {D : ℝ}
    (hD : ∀ t, (∃ i, u i ≤ t) → (∃ j, t ≤ u j) → |ecdf u t - G t| ≤ D) (i : Fin n) :
    |(n : ℝ) * G (u i) - (i.val + 1)| ≤ n * D := by
  have hn := n_pos_of_fin i
  -- at the key itself the empirical CDF is exactly `(i+1)/n`
  have h := hD (u i) ⟨i, le_rfl⟩ ⟨i, le_rfl⟩
  rw [ecdf_self hu i, abs_sub_comm] at h
  rw [rank_scale hn, abs_sub_comm]
  exact mul_le_mul_of_nonneg_left h hn.le

/-- **Theorem 3.1 (a, right; b, lower).** An affine model `L(t) = α t + β` with rank error at most
`E` at every key is within `(E + 1)/n` of the empirical CDF everywhere on the key range. -/
theorem ks_le_of_err {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) (α β E : ℝ)
    (hE : ∀ i : Fin n, |(n : ℝ) * (α * u i + β) - (i.val + 1)| ≤ E) {t : ℝ}
    (hlo : ∃ i, u i ≤ t) (hhi : ∃ j, t ≤ u j) :
    |ecdf u t - (α * t + β)| ≤ (E + 1) / n := by
  -- `t` lies in `[u i, u (i+1))` for the last key `i` with `u i ≤ t`
  obtain ⟨i, hi, hafter, hr⟩ := exists_bracket hu hlo
  have hn := n_pos_of_fin i
  have hecdf : ecdf u t = ((i.val : ℝ) + 1) / n := by unfold ecdf; rw [hr]; push_cast; ring
  rw [hecdf]
  have hE0 : 0 ≤ E := (abs_nonneg _).trans (hE i)
  have hle : E / n ≤ (E + 1) / n := div_le_div_of_nonneg_right (by linarith) hn.le
  -- the error at key `i`, in CDF units, is at most `E / n`
  have hEi : |((i.val : ℝ) + 1) / n - (α * u i + β)| ≤ E / n := by
    have h := hE i
    rw [rank_scale hn] at h
    rw [le_div_iff₀ hn, mul_comm]; exact h
  by_cases hlast : i.val + 1 < n
  · -- the next key `i' = i + 1` is `> t`
    set i' : Fin n := ⟨i.val + 1, hlast⟩ with hi'def
    have hii' : i < i' := by rw [Fin.lt_def]; simp [i']
    have ht' : t < u i' := hafter i' hii'
    -- at `u i'` the CDF value `(i+1)/n` is one step below the true `(i+2)/n`:
    -- `|(i+1)/n - L(u i')| ≤ |(i+2)/n - L(u i')| + 1/n ≤ (E + 1)/n`
    have hEi' : |((i.val : ℝ) + 1) / n - (α * u i' + β)| ≤ (E + 1) / n := by
      have h := hE i'
      have hv : (i'.val : ℝ) = i.val + 1 := by simp [i']
      rw [hv, rank_scale hn] at h
      have htri : |((i.val : ℝ) + 1) / n - (α * u i' + β)| ≤
          |((i.val : ℝ) + 1 + 1) / n - (α * u i' + β)| + 1 / n := by
        have : ((i.val : ℝ) + 1) / n - (α * u i' + β) =
            (((i.val : ℝ) + 1 + 1) / n - (α * u i' + β)) + (-(1 / n)) := by ring
        rw [this]
        refine (abs_add_le _ _).trans ?_
        rw [abs_neg, abs_of_pos (show (0 : ℝ) < 1 / n by positivity)]
      have h2 : |((i.val : ℝ) + 1 + 1) / n - (α * u i' + β)| ≤ E / n := by
        rw [le_div_iff₀ hn, mul_comm]; exact h
      calc _ ≤ _ := htri
        _ ≤ E / n + 1 / n := by linarith
        _ = (E + 1) / n := by ring
    -- an affine function of `t` on `[u i, u i']` lies between its endpoint values
    have key : |((i.val : ℝ) + 1) / n - (α * t + β)| ≤
        max |((i.val : ℝ) + 1) / n - (α * u i + β)| |((i.val : ℝ) + 1) / n - (α * u i' + β)| := by
      rcases le_total 0 α with hα | hα
      · have h1 : α * u i ≤ α * t := mul_le_mul_of_nonneg_left hi hα
        have h2 : α * t ≤ α * u i' := mul_le_mul_of_nonneg_left ht'.le hα
        rw [max_comm]
        exact abs_le_max_abs_abs (by linarith) (by linarith)
      · have h1 : α * t ≤ α * u i := mul_le_mul_of_nonpos_left hi hα
        have h2 : α * u i' ≤ α * t := mul_le_mul_of_nonpos_left ht'.le hα
        exact abs_le_max_abs_abs (by linarith) (by linarith)
    exact key.trans (max_le (hEi.trans hle) hEi')
  · -- `i` is the last key, so the upper key-range bound forces `t = u i`
    obtain ⟨j, hj⟩ := hhi
    have hji : j ≤ i := by rw [Fin.le_def]; omega
    have : t = u i := le_antisymm (hj.trans (hu.monotone hji)) hi
    rw [this]
    exact hEi.trans hle

/-! ### The Kolmogorov distance as a finite maximum (Definition 3.1) -/

/-- The Kolmogorov distance between the empirical CDF of sorted keys `u` and `U`, written as a
maximum over finitely many points: at each key, the right value `(i+1)/n` and the left limit
`i/n` of the empirical CDF, compared with `U(u_i)`. `ecdf_sub_unif_le_ksUnif` and `ksUnif_le`
show that it equals `sup_t |F_u(t) - U(t)|`. -/
noncomputable def ksUnif {n : ℕ} (u : Fin n → ℝ) : ℝ :=
  ⨆ i : Fin n, max |((i.val : ℝ) + 1) / n - unifCDF (u i)| |(i.val : ℝ) / n - unifCDF (u i)|

lemma ksUnif_term_le {n : ℕ} (u : Fin n → ℝ) (i : Fin n) :
    max |((i.val : ℝ) + 1) / n - unifCDF (u i)| |(i.val : ℝ) / n - unifCDF (u i)| ≤
      ksUnif u := by
  unfold ksUnif
  exact le_ciSup (f := fun j : Fin n =>
    max |((j.val : ℝ) + 1) / n - unifCDF (u j)| |(j.val : ℝ) / n - unifCDF (u j)|)
    (Set.finite_range _).bddAbove i

lemma ksUnif_right_le {n : ℕ} (u : Fin n → ℝ) (i : Fin n) :
    |((i.val : ℝ) + 1) / n - unifCDF (u i)| ≤ ksUnif u :=
  (le_max_left _ _).trans (ksUnif_term_le u i)

lemma ksUnif_left_le {n : ℕ} (u : Fin n → ℝ) (i : Fin n) :
    |(i.val : ℝ) / n - unifCDF (u i)| ≤ ksUnif u :=
  (le_max_right _ _).trans (ksUnif_term_le u i)

/-- `ksUnif` bounds the CDF difference at **every** real `t`. -/
theorem ecdf_sub_unif_le_ksUnif {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) (hn : 0 < n)
    (t : ℝ) : |ecdf u t - unifCDF t| ≤ ksUnif u := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  by_cases hlo : ∃ i, u i ≤ t
  · obtain ⟨i, hi, hafter, hr⟩ := exists_bracket hu hlo
    have hecdf : ecdf u t = ((i.val : ℝ) + 1) / n := by unfold ecdf; rw [hr]; push_cast; ring
    rw [hecdf]
    by_cases hlast : i.val + 1 < n
    · -- between keys `i` and `i+1`: `U(t)` lies between `U(u_i)` and `U(u_{i+1})`
      set i' : Fin n := ⟨i.val + 1, hlast⟩
      have ht' : t < u i' := hafter i' (by rw [Fin.lt_def]; simp [i'])
      have h1 : unifCDF (u i) ≤ unifCDF t := unifCDF_mono hi
      have h2 : unifCDF t ≤ unifCDF (u i') := unifCDF_mono ht'.le
      have hb : |((i.val : ℝ) + 1) / n - unifCDF t| ≤
          max |((i.val : ℝ) + 1) / n - unifCDF (u i')| |((i.val : ℝ) + 1) / n - unifCDF (u i)| :=
        abs_le_max_abs_abs (by linarith) (by linarith)
      have hv : ((i.val : ℝ) + 1) = (i'.val : ℝ) := by simp [i']
      refine hb.trans (max_le ?_ (ksUnif_right_le u i))
      rw [hv]; exact ksUnif_left_le u i'
    · -- after the last key: `F = 1` and `1 - U(t) ≤ 1 - U(u_{n-1})`
      have hval : ((i.val : ℝ) + 1) / n = 1 := by
        have : i.val + 1 = n := by omega
        rw [div_eq_one_iff_eq hn'.ne']; exact_mod_cast this
      have h1 : unifCDF (u i) ≤ unifCDF t := unifCDF_mono hi
      have hR := ksUnif_right_le u i
      rw [hval] at hR ⊢
      rw [abs_of_nonneg (by linarith [unifCDF_le_one t])]
      rw [abs_of_nonneg (by linarith [unifCDF_le_one (u i)])] at hR
      linarith
  · -- before the first key: `F = 0` and `U(t) ≤ U(u_0)`
    push Not at hlo
    have h0 : ecdf u t = 0 := by unfold ecdf; rw [rankLE_eq_zero hlo]; simp
    rw [h0]
    have i0 : Fin n := ⟨0, hn⟩
    have hL := ksUnif_left_le u ⟨0, hn⟩
    simp only [Nat.cast_zero, zero_div, zero_sub, abs_neg] at hL ⊢
    rw [abs_of_nonneg (unifCDF_nonneg _)] at hL ⊢
    exact (unifCDF_mono (hlo ⟨0, hn⟩).le).trans hL

/-- `ksUnif` is the **least** such bound, so it is the supremum. The left-limit terms need a
limit `t → u_i⁻`, using the continuity of `U`. -/
theorem ksUnif_le {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) (hn : 0 < n) {D : ℝ}
    (hD : ∀ t, |ecdf u t - unifCDF t| ≤ D) : ksUnif u ≤ D := by
  have : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
  apply ciSup_le
  intro i
  apply max_le
  · -- right value: evaluate at `t = u i`
    have h := hD (u i)
    rwa [ecdf_self hu i] at h
  · -- left limit: for `t` just below `u i`, the empirical CDF equals `i/n`
    -- a point strictly below `u i` and at or above every earlier key
    set lower : ℝ := if h : 0 < i.val then u ⟨i.val - 1, by omega⟩ else u i - 1 with hlower
    have hlower_lt : lower < u i := by
      rw [hlower]; split_ifs with h
      · exact hu (by rw [Fin.lt_def]; simp; omega)
      · linarith
    have hbefore : ∀ j : Fin n, j < i → u j ≤ lower := by
      intro j hj
      rw [hlower]; split_ifs with h
      · exact hu.monotone (by rw [Fin.le_def]; simp; rw [Fin.lt_def] at hj; omega)
      · rw [Fin.lt_def] at hj; omega
    have hlim : Tendsto (fun t => |(i.val : ℝ) / n - unifCDF t|) (𝓝[<] (u i))
        (𝓝 |(i.val : ℝ) / n - unifCDF (u i)|) :=
      ((continuous_const.sub continuous_unifCDF).abs.tendsto (u i)).mono_left
        nhdsWithin_le_nhds
    apply le_of_tendsto hlim
    filter_upwards [Ioo_mem_nhdsLT hlower_lt] with t ht
    -- on `(lower, u i)` the keys `≤ t` are exactly those with index `< i`
    have hrank : rankLE u t = i.val := by
      unfold rankLE
      have : univ.filter (fun j => u j ≤ t) = Iio i := by
        ext j
        simp only [mem_filter, mem_univ, true_and, mem_Iio]
        constructor
        · intro hj
          by_contra hc
          push Not at hc
          exact absurd (lt_of_lt_of_le ht.2 (hu.monotone hc)) (not_lt.mpr hj)
        · intro hj; exact (hbefore j hj).trans ht.1.le
      rw [this, Fin.card_Iio]
    have h := hD t
    unfold ecdf at h
    rwa [hrank] at h

/-- The endpoint (min-max) model's error `ε_U = max_i |n u_i - i|` (1-indexed `i`). -/
noncomputable def endpointErr {n : ℕ} (u : Fin n → ℝ) : ℝ :=
  ⨆ i : Fin n, |(n : ℝ) * u i - (i.val + 1)|

lemma endpointErr_ge {n : ℕ} (u : Fin n → ℝ) (i : Fin n) :
    |(n : ℝ) * u i - (i.val + 1)| ≤ endpointErr u := by
  unfold endpointErr
  exact le_ciSup (f := fun j : Fin n => |(n : ℝ) * u j - (j.val + 1)|)
    (Set.finite_range _).bddAbove i

/-- **Theorem 3.1 (a), the sandwich.** For min-max normalized sorted keys (values in `[0, 1]`,
minimum `0`, maximum `1`): `ε_U ≤ n · D_KS(F_u, U) ≤ ε_U + 1`. -/
theorem ks_sandwich {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) (hn : 0 < n)
    (h0 : ∀ i, 0 ≤ u i) (h1 : ∀ i, u i ≤ 1) (hmin : ∃ i, u i = 0) (hmax : ∃ j, u j = 1) :
    endpointErr u ≤ n * ksUnif u ∧ n * ksUnif u ≤ endpointErr u + 1 := by
  have : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  constructor
  · -- left: Theorem 3.1(a) left, with `G = U`, which is the identity on the keys
    apply ciSup_le
    intro i
    have h := err_le_of_ks hu (G := unifCDF) (D := ksUnif u)
      (fun t _ _ => ecdf_sub_unif_le_ksUnif hu hn t) i
    rwa [unifCDF_of_mem (h0 i) (h1 i)] at h
  · -- right: the bound `(ε_U + 1)/n` holds for every `t`, and `ksUnif` is the least bound
    have hE0 : 0 ≤ endpointErr u := (abs_nonneg _).trans (endpointErr_ge u ⟨0, hn⟩)
    have hall : ∀ t, |ecdf u t - unifCDF t| ≤ (endpointErr u + 1) / n := by
      intro t
      have hpos : 0 ≤ (endpointErr u + 1) / n := by positivity
      rcases lt_or_ge t 0 with ht | ht
      · -- left of the keys: both CDFs are 0
        have : ecdf u t = 0 := by
          unfold ecdf; rw [rankLE_eq_zero (fun i => lt_of_lt_of_le ht (h0 i))]; simp
        rw [this, unifCDF_of_nonpos ht.le]; simpa using hpos
      rcases lt_or_ge 1 t with ht1 | ht1
      · -- right of the keys: both CDFs are 1
        have : ecdf u t = 1 := by
          unfold ecdf; rw [rankLE_eq_n (fun i => (h1 i).trans ht1.le)]
          exact div_self hn'.ne'
        rw [this, unifCDF_of_one_le ht1.le]; simpa using hpos
      · -- on `[0, 1]` = the key range, apply Theorem 3.1(a) right with `L(t) = t`
        obtain ⟨i0, hi0⟩ := hmin
        obtain ⟨j0, hj0⟩ := hmax
        have h := ks_le_of_err hu 1 0 (endpointErr u)
          (fun i => by simpa using endpointErr_ge u i) (t := t)
          ⟨i0, by rw [hi0]; exact ht⟩ ⟨j0, by rw [hj0]; exact ht1⟩
        rw [unifCDF_of_mem ht ht1]
        simpa using h
    have := ksUnif_le hu hn hall
    rw [le_div_iff₀ hn'] at this
    linarith

/-! ### Theorem 3.2: slots -/

/-- **Theorem 3.2 (slots: the "+1").** The floored 0-indexed slot `⌊n·G(u_i)⌋` is within
`n·D + 1` of the true slot `i`. -/
theorem slot_err {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) {G : ℝ → ℝ} {D : ℝ}
    (hD : ∀ t, |ecdf u t - G t| ≤ D) (i : Fin n) :
    |((⌊(n : ℝ) * G (u i)⌋ : ℤ) : ℝ) - i.val| ≤ n * D + 1 := by
  have h := err_le_of_ks hu (fun t _ _ => hD t) i
  -- `⌊y⌋ ∈ (y - 1, y]`
  have hf1 := Int.floor_le ((n : ℝ) * G (u i))
  have hf2 := Int.lt_floor_add_one ((n : ℝ) * G (u i))
  rw [abs_le] at h ⊢
  constructor <;> linarith [h.1, h.2]

/-! ### Theorem 3.3: the probability integral transform is optimal -/

/-- **Theorem 3.3 (a), general form.** If some key is `0` and the target CDF has `G(0) = 0`,
the CDF difference at `0` is at least `1/n`. -/
theorem ks_ge_inv_n {n : ℕ} {u : Fin n → ℝ} {i₀ : Fin n} (h0 : u i₀ = 0) {G : ℝ → ℝ}
    (hG : G 0 = 0) : 1 / (n : ℝ) ≤ |ecdf u 0 - G 0| := by
  have hn := n_pos_of_fin i₀
  -- the key `i₀` is counted, so the rank of `0` is at least 1
  have hr : 1 ≤ rankLE u 0 := by
    unfold rankLE
    apply card_pos.mpr
    exact ⟨i₀, by simp [h0]⟩
  rw [hG, sub_zero]
  unfold ecdf
  rw [abs_of_nonneg (by positivity)]
  apply div_le_div_of_nonneg_right _ hn.le
  exact_mod_cast hr

/-- **Theorem 3.3 (a).** For normalized keys (some key equal to 0), `D_KS(F_u, U) ≥ 1/n`. -/
theorem ks_ge_inv_n_of_normalized {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) (hn : 0 < n)
    {i₀ : Fin n} (h0 : u i₀ = 0) : 1 / (n : ℝ) ≤ ksUnif u :=
  (ks_ge_inv_n h0 (unifCDF_of_nonpos le_rfl)).trans (ecdf_sub_unif_le_ksUnif hu hn 0)

/-- The probability integral transform's normalized keys `u_i = i/(n-1)` are sorted. -/
lemma pit_strictMono {n : ℕ} (hn : 2 ≤ n) :
    StrictMono (fun i : Fin n => (i.val : ℝ) / (n - 1)) := by
  have : (1 : ℝ) < n := by exact_mod_cast hn
  intro i j hij
  have hij' : (i.val : ℝ) < j.val := by exact_mod_cast hij
  exact div_lt_div_of_pos_right hij' (by linarith)

/-- **Theorem 3.3 (b), KS part.** With `T = F_X`, the normalized keys are `u_i = i/(n-1)`
(0-based `i`), and their empirical CDF is within `1/n` of `U` everywhere. -/
theorem pit_ks {n : ℕ} (hn : 2 ≤ n) (t : ℝ) :
    |ecdf (fun i : Fin n => (i.val : ℝ) / (n - 1)) t - unifCDF t| ≤ 1 / n := by
  have hn1 : (1 : ℝ) < n := by exact_mod_cast hn
  have hnpos : (0 : ℝ) < n := by linarith
  have hd : (0 : ℝ) < n - 1 := by linarith
  have hu := pit_strictMono hn
  have hpos : (0 : ℝ) ≤ 1 / n := by positivity
  rcases lt_or_ge t 0 with ht | ht
  · -- below the first key: both CDFs vanish
    have : ecdf (fun i : Fin n => (i.val : ℝ) / (n - 1)) t = 0 := by
      unfold ecdf
      rw [rankLE_eq_zero (fun i => lt_of_lt_of_le ht (by positivity))]; simp
    rw [this, unifCDF_of_nonpos ht.le]; simp
  rcases lt_or_ge t 1 with ht1 | ht1
  · -- `t ∈ [0, 1)`: `t ∈ [i/(n-1), (i+1)/(n-1))` and `F = (i+1)/n`
    obtain ⟨i, hi, hafter, hr⟩ := exists_bracket hu (t := t) ⟨⟨0, by omega⟩, by simpa using ht⟩
    have hecdf : ecdf (fun i : Fin n => (i.val : ℝ) / (n - 1)) t = ((i.val : ℝ) + 1) / n := by
      unfold ecdf; rw [hr]; push_cast; ring
    rw [hecdf, unifCDF_of_mem ht ht1.le]
    -- `i` is not the last key, otherwise `t ≥ 1`
    have hlast : i.val + 1 < n := by
      by_contra hc
      have : i.val = n - 1 := by omega
      have hcast : (i.val : ℝ) = n - 1 := by rw [this]; push_cast [show 1 ≤ n by omega]; ring
      rw [hcast, div_self hd.ne'] at hi
      linarith
    have hnext := hafter ⟨i.val + 1, hlast⟩ (Fin.lt_def.mpr (by simp))
    simp only at hnext
    push_cast at hnext
    have hile : (i.val : ℝ) + 2 ≤ n := by exact_mod_cast hlast
    rw [div_le_iff₀ hd] at hi
    rw [lt_div_iff₀ hd] at hnext
    -- `t·n < i + 2` (from `t (n-1) < i + 1` and `t < 1`) and `i ≤ t·n` (from `i ≤ t (n-1)`)
    have key1 : t * n ≤ (i.val : ℝ) + 2 := by nlinarith
    have key2 : (i.val : ℝ) ≤ t * n := by nlinarith
    rw [abs_le]
    constructor
    · -- `t ≤ (i+2)/n = (i+1)/n + 1/n`
      have : t ≤ ((i.val : ℝ) + 2) / n := by rw [le_div_iff₀ hnpos]; linarith
      have e : ((i.val : ℝ) + 2) / n = ((i.val : ℝ) + 1) / n + 1 / n := by ring
      linarith
    · -- `(i+1)/n = i/n + 1/n ≤ t + 1/n`
      have : (i.val : ℝ) / n ≤ t := by rw [div_le_iff₀ hnpos]; linarith
      have e : ((i.val : ℝ) + 1) / n = (i.val : ℝ) / n + 1 / n := by ring
      linarith
  · -- at or after the last key: both CDFs equal 1
    have : ecdf (fun i : Fin n => (i.val : ℝ) / (n - 1)) t = 1 := by
      unfold ecdf
      rw [rankLE_eq_n]
      · exact div_self hnpos.ne'
      · intro i
        have : (i.val : ℝ) ≤ n - 1 := by
          have := i.isLt
          have : (i.val : ℝ) + 1 ≤ n := by exact_mod_cast this
          linarith
        exact ((div_le_one hd).mpr this).trans ht1
    rw [this, unifCDF_of_one_le ht1]; simp

/-- **Theorem 3.3 (b), endpoint error.** The PIT's endpoint model has error at most 1. -/
theorem pit_endpoint_err {n : ℕ} (hn : 2 ≤ n) (i : Fin n) :
    |(n : ℝ) * ((i.val : ℝ) / (n - 1)) - (i.val + 1)| ≤ 1 := by
  have hn1 : (1 : ℝ) < n := by exact_mod_cast hn
  have hd : (0 : ℝ) < n - 1 := by linarith
  have hi : (i.val : ℝ) + 1 ≤ n := by exact_mod_cast i.isLt
  -- `n·i/(n-1) - (i+1) = (i + 1 - n)/(n-1) ∈ [-1, 0]`
  have : (n : ℝ) * ((i.val : ℝ) / (n - 1)) - (i.val + 1) = ((i.val : ℝ) + 1 - n) / (n - 1) := by
    field_simp; ring
  rw [this, abs_le]
  constructor
  · rw [le_div_iff₀ hd]; nlinarith [(Nat.cast_nonneg i.val : (0 : ℝ) ≤ i.val)]
  · rw [div_le_iff₀ hd]; linarith

/-- **Theorem 3.3 (b), the affine model on `z_i = i/n`.** `L(z) = z` predicts every rank
exactly: `n · L(z_i) = i`. -/
theorem pit_exact {n : ℕ} (i : Fin n) : (n : ℝ) * (((i.val : ℝ) + 1) / n) = i.val + 1 := by
  have hn := n_pos_of_fin i
  field_simp

/-- **Theorem 3.3 (c), zero error iff affine in the rank.** An affine model is exact on the
transformed keys iff those keys are an affine function of the rank. -/
theorem zero_error_iff {n : ℕ} (hn : 2 ≤ n) (z : Fin n → ℝ) :
    (∃ a b : ℝ, ∀ i : Fin n, a * z i + b = i.val + 1) ↔
      (∃ c d : ℝ, c ≠ 0 ∧ ∀ i : Fin n, z i = c * (i.val + 1) + d) := by
  constructor
  · rintro ⟨a, b, h⟩
    -- two different targets force `a ≠ 0`
    have ha : a ≠ 0 := by
      intro ha
      have h0 := h ⟨0, by omega⟩
      have h1 := h ⟨1, by omega⟩
      simp [ha] at h0 h1
      linarith
    -- solve for `z i`
    refine ⟨1 / a, -b / a, by simpa using ha, fun i => ?_⟩
    have := h i
    field_simp
    linarith
  · rintro ⟨c, d, hc, h⟩
    refine ⟨1 / c, -d / c, fun i => ?_⟩
    rw [h i]
    field_simp
    ring

/-! ### Theorem 3.5: affine invariance; when a transform helps -/

/-- **Theorem 3.5 (a).** Rescaling the keys by an affine map does not change the set of
achievable affine-model errors. -/
theorem affine_invariance {n : ℕ} (z : Fin n → ℝ) {α : ℝ} (hα : α ≠ 0) (β E : ℝ) :
    (∃ a b : ℝ, ∀ i, |a * z i + b - (i.val + 1)| ≤ E) ↔
      (∃ a b : ℝ, ∀ i, |a * (α * z i + β) + b - (i.val + 1)| ≤ E) := by
  constructor
  · -- the model `L ∘ A⁻¹` on the rescaled keys makes the same predictions
    rintro ⟨a, b, h⟩
    refine ⟨a / α, b - a * β / α, fun i => ?_⟩
    have : a / α * (α * z i + β) + (b - a * β / α) = a * z i + b := by field_simp; ring
    rw [this]; exact h i
  · rintro ⟨a, b, h⟩
    refine ⟨a * α, a * β + b, fun i => ?_⟩
    have : a * α * z i + (a * β + b) = a * (α * z i + β) + b := by ring
    rw [this]; exact h i

/-- **Theorem 3.5 (b), when a transform helps a single linear model.**
Suppose the raw keys `x` are at linear distance at least `DX` from every affine CDF (on their
key range), and the transformed keys `z` are within `DZ` of some affine CDF, with
`n·DZ < n·DX - 1`. Then some affine model on `z` has strictly smaller maximum rank error than
every affine model on `x`: for every `(α', β')` there is a key `i₀` whose raw error exceeds
every transformed error. -/
theorem transform_helps {n : ℕ} {x z : Fin n → ℝ} (hx : StrictMono x) (hz : StrictMono z)
    {DX DZ : ℝ}
    (hX : ∀ α β : ℝ, ∃ t, (∃ i, x i ≤ t) ∧ (∃ j, t ≤ x j) ∧ DX ≤ |ecdf x t - (α * t + β)|)
    (hZ : ∃ α β : ℝ, ∀ t, (∃ i, z i ≤ t) → (∃ j, t ≤ z j) → |ecdf z t - (α * t + β)| ≤ DZ)
    (hlt : (n : ℝ) * DZ < n * DX - 1) :
    ∃ α β : ℝ, ∀ α' β' : ℝ, ∃ i₀ : Fin n, ∀ i : Fin n,
      |(n : ℝ) * (α * z i + β) - (i.val + 1)| < |(n : ℝ) * (α' * x i₀ + β') - (i₀.val + 1)| := by
  obtain ⟨α, β, hαβ⟩ := hZ
  refine ⟨α, β, fun α' β' => ?_⟩
  -- the keys are nonempty: otherwise `hlt` reads `0 < -1`
  have hn : 0 < n := by
    rcases Nat.eq_zero_or_pos n with h | h
    · subst h; simp at hlt; linarith
    · exact h
  -- `i₀` = the key where the raw model `(α', β')` has its largest error `E'`
  obtain ⟨i₀, -, hi₀⟩ := Finset.exists_max_image univ
    (fun i : Fin n => |(n : ℝ) * (α' * x i + β') - (i.val + 1)|) ⟨⟨0, hn⟩, mem_univ _⟩
  refine ⟨i₀, fun i => ?_⟩
  set E' := |(n : ℝ) * (α' * x i₀ + β') - (i₀.val + 1)|
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  -- Theorem 3.1(b, lower) on `x`: the raw model is within `(E'+1)/n` on the key range, so
  -- `DX ≤ (E' + 1)/n`, i.e. `E' ≥ n·DX - 1 > n·DZ`
  obtain ⟨t, hlo, hhi, hDX⟩ := hX α' β'
  have h1 := ks_le_of_err hx α' β' E' (fun i => hi₀ i (mem_univ _)) hlo hhi
  have h2 : (n : ℝ) * DX ≤ E' + 1 := by
    have := hDX.trans h1; rw [le_div_iff₀ hn'] at this; linarith
  -- Theorem 3.1(b, upper) on `z`: the transformed model's error is at most `n·DZ`
  have h3 := err_le_of_ks hz (G := fun t => α * t + β) hαβ i
  linarith

/-! ### Theorem 3.6: non-monotone transforms -/

/-- **Theorem 3.6 (upper).** For arbitrary transformed keys `z` (no monotonicity): the rank error
of a CDF-scale model is at most `n·D` plus the displacement `|σ(i) - i|`, where
`σ(i) = rankLE z (z i)` is the rank of `z i` among the `z`'s. -/
theorem err_le_ks_add_disp {n : ℕ} (z : Fin n → ℝ) {G : ℝ → ℝ} {D : ℝ}
    (hD : ∀ t, |ecdf z t - G t| ≤ D) (i : Fin n) :
    |(n : ℝ) * G (z i) - (i.val + 1)| ≤ n * D + |(rankLE z (z i) : ℝ) - (i.val + 1)| := by
  have hn := n_pos_of_fin i
  -- `F(z_i) = σ(i)/n`, so `|n·G(z_i) - σ(i)| ≤ n·D`
  have h : |(n : ℝ) * G (z i) - rankLE z (z i)| ≤ n * D := by
    rw [rank_scale hn]
    exact mul_le_mul_of_nonneg_left (hD (z i)) hn.le
  calc |(n : ℝ) * G (z i) - (i.val + 1)|
      = |((n : ℝ) * G (z i) - rankLE z (z i)) + ((rankLE z (z i) : ℝ) - (i.val + 1))| := by
        ring_nf
    _ ≤ _ := (abs_add_le _ _).trans (by linarith)

/-- **Theorem 3.6 (lower).** The displacement is at most the rank error plus `n·D`. -/
theorem disp_le_err_add_ks {n : ℕ} (z : Fin n → ℝ) {G : ℝ → ℝ} {D : ℝ}
    (hD : ∀ t, |ecdf z t - G t| ≤ D) (i : Fin n) :
    |(rankLE z (z i) : ℝ) - (i.val + 1)| ≤ |(n : ℝ) * G (z i) - (i.val + 1)| + n * D := by
  have hn := n_pos_of_fin i
  have h : |(rankLE z (z i) : ℝ) - (n : ℝ) * G (z i)| ≤ n * D := by
    rw [abs_sub_comm, rank_scale hn]
    exact mul_le_mul_of_nonneg_left (hD (z i)) hn.le
  calc |(rankLE z (z i) : ℝ) - (i.val + 1)|
      = |((rankLE z (z i) : ℝ) - (n : ℝ) * G (z i)) + ((n : ℝ) * G (z i) - (i.val + 1))| := by
        ring_nf
    _ ≤ _ := (abs_add_le _ _).trans (by linarith)

/-! ### Lemma 3.7: clamping never hurts -/

/-- **Lemma 3.7.** Clamping a prediction into an interval that contains the answer never
increases its error. -/
theorem clamp_err {lo hi y r : ℝ} (hlo : lo ≤ r) (hhi : r ≤ hi) :
    |max lo (min y hi) - r| ≤ |y - r| := by
  -- three cases: `y < lo`, `y ∈ [lo, hi]`, `y > hi`
  rcases le_total y hi with hy | hy
  · rw [min_eq_left hy]
    rcases le_total lo y with hy' | hy'
    · rw [max_eq_right hy']
    · rw [max_eq_left hy', abs_of_nonpos (by linarith), abs_of_nonpos (by linarith)]
      linarith
  · rw [min_eq_right hy, max_eq_right (hlo.trans hhi), abs_of_nonneg (by linarith),
      abs_of_nonneg (by linarith)]
    linarith

/-! ### Corollaries in terms of the finite Kolmogorov distance `ksUnif` -/

/-- **Theorem 3.2 with `D = D_KS`.** For min-max normalized sorted keys, the endpoint model's
floored slot is within `n · D_KS(F_u, U) + 1` of the true 0-indexed slot. -/
theorem slot_err_ksUnif {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) (hn : 0 < n)
    (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1) (i : Fin n) :
    |((⌊(n : ℝ) * u i⌋ : ℤ) : ℝ) - i.val| ≤ n * ksUnif u + 1 := by
  have h := slot_err hu (G := unifCDF) (fun t => ecdf_sub_unif_le_ksUnif hu hn t) i
  rwa [unifCDF_of_mem (h01 i).1 (h01 i).2] at h

/-- **Theorem 3.3 (b), exact value.** The probability integral transform attains the minimum of
Theorem 3.3 (a): `D_KS(F_u, U) = 1/n` for `u_i = i/(n-1)`. -/
theorem pit_ksUnif_eq {n : ℕ} (hn : 2 ≤ n) :
    ksUnif (fun i : Fin n => (i.val : ℝ) / (n - 1)) = 1 / n :=
  le_antisymm (ksUnif_le (pit_strictMono hn) (by omega) (pit_ks hn))
    (ks_ge_inv_n_of_normalized (pit_strictMono hn) (by omega) (i₀ := ⟨0, by omega⟩) (by simp))

/-- **Theorem 3.3 (b), endpoint error.** The PIT's endpoint error `ε_U` is at most 1. -/
theorem pit_endpointErr_le {n : ℕ} (hn : 2 ≤ n) :
    endpointErr (fun i : Fin n => (i.val : ℝ) / (n - 1)) ≤ 1 := by
  have : Nonempty (Fin n) := ⟨⟨0, by omega⟩⟩
  exact ciSup_le (fun i => pit_endpoint_err hn i)

end NFLProof
