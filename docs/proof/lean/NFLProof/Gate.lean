module

public import NFLProof.Error

/-!
# §5 The gate: what a sample statistic certifies, and the decision rule

Lean names (PROOF.md numbering):

* Lemma 5.1 `prefix_le_tv`: prefix sums of `p - q` are bounded by the total variation.
* Definition 5.1 `bin`, `hist`, `tvUnif`, `klUnif` (exactly `kl_uniform` of `src/tools/nf_gate.cpp`).
* Theorem 5.2 `ks_le_tv_add`: `|F_v(t) - U(t)| ≤ TV_B + 1/B` for every `t`.
* Theorem 5.3 `even_sample_count`, `evenSample`, `even_sample_dev`: nf_gate's evenly spaced
  sample is within `1/m` of the population CDF everywhere (deterministic).
* `PinskerUnif`: Pinsker's inequality against the uniform distribution (PROOF.md's Theorem P), as a
  proposition. The generic theorems here take it as an explicit premise; `Pinsker.lean` proves it
  (`pinsker_unif`). It is never an axiom.
* Theorem 5.4 `certified_error` (the chain), `certified_error_even` (5.4 a, evenly spaced sample),
  `certified_error_iid` (5.4 b, on the event where the sample deviation is at most `τ`).
* Theorem 5.5 `gate_sound`, `gate_false_positive` (with the DKW tail bound as the explicit
  hypothesis `hDKW`).
-/

@[expose] public section

open Finset

namespace NFLProof

/-! ### Lemma 5.1 -/

/-- **Lemma 5.1.** If `p` and `q` have the same total mass, every prefix sum of `p - q` is at most
half the `ℓ¹` distance. -/
theorem prefix_le_tv {B : ℕ} (p q : Fin B → ℝ) (hsum : ∑ b, p b = ∑ b, q b) (k : ℕ) :
    |∑ b ∈ univ.filter (fun b : Fin B => b.val < k), (p b - q b)| ≤
      (1 / 2) * ∑ b, |p b - q b| := by
  set P := ∑ b ∈ univ.filter (fun b : Fin B => b.val < k), (p b - q b)
  set Q := ∑ b ∈ univ.filter (fun b : Fin B => ¬ b.val < k), (p b - q b)
  -- the prefix and the suffix sum to zero
  have h0 : P + Q = 0 := by
    rw [sum_filter_add_sum_filter_not, sum_sub_distrib, hsum, sub_self]
  have hP : |P| ≤ ∑ b ∈ univ.filter (fun b : Fin B => b.val < k), |p b - q b| :=
    abs_sum_le_sum_abs _ _
  have hQ : |Q| ≤ ∑ b ∈ univ.filter (fun b : Fin B => ¬ b.val < k), |p b - q b| :=
    abs_sum_le_sum_abs _ _
  have htot := sum_filter_add_sum_filter_not univ (fun b : Fin B => b.val < k)
    (fun b => |p b - q b|)
  have hQP : |Q| = |P| := by rw [show Q = -P by linarith, abs_neg]
  linarith

/-! ### Definition 5.1 and Theorem 5.2 -/

/-- The bin of a normalized value: `min (B - 1) ⌊v B⌋`, so the last bin is closed. -/
noncomputable def bin (B : ℕ) (v : ℝ) : ℕ := min (B - 1) ⌊v * B⌋₊

/-- The histogram `p_b = #{k : bin(v_k) = b} / m`. -/
noncomputable def hist {m : ℕ} (B : ℕ) (v : Fin m → ℝ) (b : Fin B) : ℝ :=
  ((univ.filter (fun k => bin B (v k) = b.val)).card : ℝ) / m

/-- Total variation to the uniform distribution on `B` bins. -/
noncomputable def tvUnif {B : ℕ} (p : Fin B → ℝ) : ℝ := (1 / 2) * ∑ b, |p b - 1 / B|

/-- Kullback-Leibler divergence `KL(p ‖ uniform) = ∑ p_b ln (B p_b)` (with `0 ln 0 = 0`, which is
Mathlib's convention `Real.log 0 = 0`). -/
noncomputable def klUnif {B : ℕ} (p : Fin B → ℝ) : ℝ := ∑ b, p b * Real.log (B * p b)

lemma bin_lt {B : ℕ} (hB : 0 < B) (v : ℝ) : bin B v < B := by
  unfold bin; omega

lemma tvUnif_nonneg {B : ℕ} (p : Fin B → ℝ) : 0 ≤ tvUnif p := by
  unfold tvUnif; positivity

/-- Prefix sums of the histogram count the sample values in the first `k` bins. -/
lemma hist_prefix {m B : ℕ} (hB : 0 < B) (v : Fin m → ℝ) (k : ℕ) :
    ∑ b ∈ univ.filter (fun b : Fin B => b.val < k), hist B v b =
      ((univ.filter (fun i => bin B (v i) < k)).card : ℝ) / m := by
  classical
  set f : Fin m → Fin B := fun i => ⟨bin B (v i), bin_lt hB (v i)⟩
  have hmaps : Set.MapsTo f (univ.filter (fun i => bin B (v i) < k))
      (univ.filter (fun b : Fin B => b.val < k)) := by
    intro i hi
    simp only [coe_filter, mem_univ, true_and, Set.mem_ofPred_eq] at hi ⊢
    exact hi
  rw [card_eq_sum_card_fiberwise hmaps, Nat.cast_sum, sum_div]
  apply sum_congr rfl
  intro b hb
  simp only [mem_filter, mem_univ, true_and] at hb
  have hfe : (univ.filter (fun i => bin B (v i) < k)).filter (fun i => f i = b) =
      univ.filter (fun i => bin B (v i) = b.val) := by
    ext i
    simp only [mem_filter, mem_univ, true_and, f, Fin.ext_iff]
    constructor
    · intro h; exact h.2
    · intro h; exact ⟨h ▸ hb, h⟩
  rw [hfe]
  rfl

/-- The number of bins below `k` (for `k ≤ B`) is `k`. -/
lemma card_bins_lt {B k : ℕ} (hk : k ≤ B) :
    ((univ.filter (fun b : Fin B => b.val < k)).card : ℝ) = k := by
  rw [Fin.card_filter_val_lt, min_eq_right hk]

/-- The prefix of the uniform distribution. -/
lemma unif_prefix {B k : ℕ} (hk : k ≤ B) :
    ∑ _b ∈ univ.filter (fun b : Fin B => b.val < k), (1 / B : ℝ) = k / B := by
  rw [sum_const, nsmul_eq_mul, card_bins_lt hk]; ring

/-- The histogram is a probability vector. -/
lemma hist_sum {m B : ℕ} (hm : 0 < m) (hB : 0 < B) (v : Fin m → ℝ) :
    ∑ b, hist B v b = 1 := by
  have h := hist_prefix hB v B
  have hall : univ.filter (fun b : Fin B => b.val < B) = univ := by
    ext b; simp
  have hall' : univ.filter (fun i : Fin m => bin B (v i) < B) = univ := by
    ext i; simp [bin_lt hB]
  rw [hall, hall'] at h
  rw [h]
  simp only [card_univ, Fintype.card_fin]
  have : (m : ℝ) ≠ 0 := by exact_mod_cast hm.ne'
  field_simp

lemma hist_nonneg {m B : ℕ} (v : Fin m → ℝ) (b : Fin B) : 0 ≤ hist B v b := by
  unfold hist; positivity

/-- `|P_k - k/B| ≤ TV` for `k ≤ B` (Lemma 5.1 with `q` uniform). -/
lemma hist_prefix_dev {m B : ℕ} (hm : 0 < m) (hB : 0 < B) (v : Fin m → ℝ) {k : ℕ} (hk : k ≤ B) :
    |((univ.filter (fun i => bin B (v i) < k)).card : ℝ) / m - k / B| ≤ tvUnif (hist B v) := by
  have h := prefix_le_tv (hist B v) (fun _ => (1 / B : ℝ))
    (by rw [hist_sum hm hB]; simp [card_univ]; field_simp) k
  rw [sum_sub_distrib, hist_prefix hB, unif_prefix hk] at h
  exact h

/-- **Theorem 5.2 (binning).** For sample values in `[0, 1]`:
`|F_v(t) - U(t)| ≤ TV_B + 1/B` for every real `t`. -/
theorem ks_le_tv_add {m B : ℕ} (hm : 0 < m) (hB : 0 < B) (v : Fin m → ℝ)
    (hv : ∀ k, v k ∈ Set.Icc (0 : ℝ) 1) (t : ℝ) :
    |ecdf v t - unifCDF t| ≤ tvUnif (hist B v) + 1 / B := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hB' : (0 : ℝ) < B := by exact_mod_cast hB
  have htv := tvUnif_nonneg (hist B v)
  have hinvB : (0 : ℝ) ≤ 1 / B := by positivity
  rcases lt_or_ge t 0 with ht0 | ht0
  · -- below `[0, 1]`: both CDFs are 0
    have : ecdf v t = 0 := by
      unfold ecdf; rw [rankLE_eq_zero (fun k => lt_of_lt_of_le ht0 (hv k).1)]; simp
    rw [this, unifCDF_of_nonpos ht0.le, sub_self, abs_zero]; linarith
  rcases le_or_gt 1 t with ht1 | ht1
  · -- above: both are 1
    have : ecdf v t = 1 := by
      unfold ecdf; rw [rankLE_eq_n (fun k => (hv k).2.trans ht1)]; exact div_self hm'.ne'
    rw [this, unifCDF_of_one_le ht1, sub_self, abs_zero]; linarith
  -- `t ∈ [0, 1)`: `t` lies in bin `k = ⌊t B⌋ < B`
  set k := ⌊t * B⌋₊ with hk
  have htB : 0 ≤ t * B := by positivity
  have hkB : k < B := by
    rw [hk, Nat.floor_lt htB]
    have : t * B < 1 * B := mul_lt_mul_of_pos_right ht1 hB'
    simpa using this
  have hk_le : (k : ℝ) ≤ t * B := Nat.floor_le htB
  have hk_lt : t * B < k + 1 := Nat.lt_floor_add_one _
  -- the sample values in bins `< k` are `≤ t`, and those `≤ t` are in bins `≤ k`
  have hsub1 : univ.filter (fun i => bin B (v i) < k) ⊆ univ.filter (fun i => v i ≤ t) := by
    intro i hi
    simp only [mem_filter, mem_univ, true_and] at hi ⊢
    have hfl : ⌊v i * B⌋₊ < k := by unfold bin at hi; omega
    have hvB : 0 ≤ v i * B := mul_nonneg (hv i).1 hB'.le
    have : v i * B < k := (Nat.floor_lt hvB).mp hfl
    nlinarith
  have hsub2 : univ.filter (fun i => v i ≤ t) ⊆ univ.filter (fun i => bin B (v i) < k + 1) := by
    intro i hi
    simp only [mem_filter, mem_univ, true_and] at hi ⊢
    have : ⌊v i * B⌋₊ ≤ k := Nat.floor_le_floor (mul_le_mul_of_nonneg_right hi hB'.le)
    unfold bin; omega
  have hc1 := card_le_card hsub1
  have hc2 := card_le_card hsub2
  have hF : ecdf v t = ((univ.filter (fun i => v i ≤ t)).card : ℝ) / m := rfl
  have hP1 : ((univ.filter (fun i => bin B (v i) < k)).card : ℝ) / m ≤ ecdf v t := by
    rw [hF]; exact div_le_div_of_nonneg_right (by exact_mod_cast hc1) hm'.le
  have hP2 : ecdf v t ≤ ((univ.filter (fun i => bin B (v i) < k + 1)).card : ℝ) / m := by
    rw [hF]; exact div_le_div_of_nonneg_right (by exact_mod_cast hc2) hm'.le
  have hd1 := abs_le.mp (hist_prefix_dev hm hB v hkB.le)
  have hd2 := abs_le.mp (hist_prefix_dev hm hB v (show k + 1 ≤ B by omega))
  rw [unifCDF_of_mem ht0 ht1.le, abs_le]
  have hk1 : (k : ℝ) / B ≤ t := by rw [div_le_iff₀ hB']; linarith
  have hk2 : t < ((k : ℝ) + 1) / B := by rw [lt_div_iff₀ hB']; linarith
  have e1 : (((k + 1 : ℕ) : ℝ)) / B = (k : ℝ) / B + 1 / B := by push_cast; ring
  have e2 : ((k : ℝ) + 1) / B = (k : ℝ) / B + 1 / B := by ring
  constructor <;> linarith

/-- The binned CDF of a vector on `B` bins at the bin boundary `k / B`: `P_k = ∑_{b < k} p_b`. -/
noncomputable def binCDF {B : ℕ} (p : Fin B → ℝ) (k : ℕ) : ℝ :=
  ∑ b ∈ univ.filter (fun b : Fin B => b.val < k), p b

/-- The Kolmogorov distance between the binned CDFs of `p` and `q`: a maximum over the `B + 1`
bin boundaries `k = 0, …, B`. -/
noncomputable def ksBinned {B : ℕ} (p q : Fin B → ℝ) : ℝ :=
  ⨆ k : Fin (B + 1), |binCDF p k.val - binCDF q k.val|

/-- **Theorem 5.2, binned form: `D_KS ≤ TV` for the binned CDFs.** For vectors of equal mass
(probability vectors), the Kolmogorov distance between their binned CDFs is at most their total
variation `½ ∑ |p - q|`. -/
theorem ksBinned_le_tv {B : ℕ} (p q : Fin B → ℝ) (hsum : ∑ b, p b = ∑ b, q b) :
    ksBinned p q ≤ (1 / 2) * ∑ b, |p b - q b| := by
  unfold ksBinned
  apply ciSup_le
  intro k
  -- each boundary is Lemma 5.1
  have h := prefix_le_tv p q hsum k.val
  unfold binCDF
  rwa [← sum_sub_distrib]

/-- `D_KS ≤ TV` for the gate's histogram against the uniform distribution. -/
theorem ksBinned_hist_le_tvUnif {m B : ℕ} (hm : 0 < m) (hB : 0 < B) (v : Fin m → ℝ) :
    ksBinned (hist B v) (fun _ => (1 : ℝ) / B) ≤ tvUnif (hist B v) := by
  apply ksBinned_le_tv
  rw [hist_sum hm hB]
  simp [card_univ]
  field_simp

/-! ### Theorem 5.3: the evenly spaced sample -/

/-- **Theorem 5.3 (counting form).** For `2 ≤ m ≤ n` and `c ≤ n`, the number of `k < m` with
`k (n-1) < c (m-1)`, divided by `m`, is within `1/m` of `c/n`. -/
theorem even_sample_count {n m : ℕ} (hm : 2 ≤ m) (hmn : m ≤ n) {c : ℕ} (hc : c ≤ n) :
    |((univ.filter (fun i : Fin m => i.val * (n - 1) < c * (m - 1))).card : ℝ) / m -
      (c : ℝ) / n| < 1 / m := by
  set N := c * (m - 1)
  set d := n - 1
  have hd : 0 < d := by omega
  -- `i·d < N ↔ i < q` with `q = ⌈N / d⌉`
  set q := (N + d - 1) / d with hq
  have hiff : ∀ i : ℕ, i * d < N ↔ i < q := by
    intro i
    constructor
    · intro h
      by_contra hc'
      push Not at hc'
      have : (N + d - 1) / d < i + 1 := by omega
      rw [Nat.div_lt_iff_lt_mul hd] at this
      have : N + d - 1 < i * d + d := by rw [add_mul, one_mul] at this; exact this
      omega
    · intro h
      have : q ≤ i → False := fun h' => absurd h (not_lt.mpr h')
      by_contra hc'
      push Not at hc'
      apply this
      -- `N ≤ i d` gives `(N + d - 1)/d ≤ i`
      have : (N + d - 1) / d < i + 1 := by
        rw [Nat.div_lt_iff_lt_mul hd, add_mul, one_mul]; omega
      omega
  have hfilt : univ.filter (fun i : Fin m => i.val * (n - 1) < c * (m - 1)) =
      univ.filter (fun i : Fin m => i.val < q) := by
    ext i; simp only [mem_filter, mem_univ, true_and]; exact hiff i.val
  rw [hfilt, Fin.card_filter_val_lt]
  set cs := min m q with hcs
  -- the two facts used below: `(cs - 1) d < N` if `cs ≥ 1`, and `N ≤ cs d` if `cs < m`
  have hm' : (2 : ℝ) ≤ m := by exact_mod_cast hm
  have hmn' : (m : ℝ) ≤ n := by exact_mod_cast hmn
  have hn' : (2 : ℝ) ≤ n := by linarith
  have hd' : (d : ℝ) = n - 1 := by simp only [d]; push_cast [show 1 ≤ n by omega]; ring
  have hN' : (N : ℝ) = c * (m - 1) := by simp only [N]; push_cast [show 1 ≤ m by omega]; ring
  have hc' : (c : ℝ) ≤ n := by exact_mod_cast hc
  have hmpos : (0 : ℝ) < m := by linarith
  have hnpos : (0 : ℝ) < n := by linarith
  rw [abs_lt]
  constructor
  · -- lower bound: `cs/m - c/n > -1/m`
    rcases lt_or_ge cs m with hlt | hge
    · have hcsq : cs = q := by omega
      have hNle : N ≤ cs * d := by
        by_contra h'; push Not at h'
        have := (hiff cs).mp h'; omega
      have hNle' : (c : ℝ) * (m - 1) ≤ cs * (n - 1) := by
        rw [← hN', ← hd']; exact_mod_cast hNle
      -- `c < n`: otherwise every `i < m` qualifies and `cs = m`
      have hcn : c < n := by
        by_contra h'
        have hcn : c = n := by omega
        have : (m - 1) * d < N := by
          simp only [N, d, hcn]
          have h1 : (m - 1) * (n - 1) < (m - 1) * n :=
            Nat.mul_lt_mul_of_pos_left (by omega) (by omega)
          rw [mul_comm n]; exact h1
        have := (hiff (m - 1)).mp this
        omega
      have hcn' : (c : ℝ) ≤ n - 1 := by
        have : c + 1 ≤ n := hcn
        have : (c : ℝ) + 1 ≤ n := by exact_mod_cast this
        linarith
      -- clear denominators: `cs·n - c·m > -n`, from `cs (n-1) ≥ c (m-1)` and `c ≤ n - 1`
      have e : (cs : ℝ) / m - c / n = (cs * n - c * m) / (m * n) := by field_simp
      rw [e, lt_div_iff₀ (by positivity)]
      have e2 : -(1 / (m : ℝ)) * (m * n) = -n := by field_simp
      rw [e2]
      have hn1 : (0 : ℝ) < n - 1 := by linarith
      have hcs0 : (0 : ℝ) ≤ cs := Nat.cast_nonneg _
      nlinarith [mul_le_mul_of_nonneg_right hNle' hnpos.le,
        mul_nonneg (sub_nonneg.mpr hcn') (sub_nonneg.mpr hmn'), mul_pos hn1 hmpos]
    · have hcsm : cs = m := by omega
      rw [hcsm, div_self hmpos.ne']
      have : (c : ℝ) / n ≤ 1 := (div_le_one hnpos).mpr hc'
      have : (0 : ℝ) < 1 / m := by positivity
      linarith
  · -- upper bound: `cs/m - c/n < 1/m`
    rcases Nat.eq_zero_or_pos cs with h0 | hpos
    · rw [h0]; simp only [Nat.cast_zero, zero_div, zero_sub]
      have : (0 : ℝ) ≤ c / n := by positivity
      have : (0 : ℝ) < 1 / m := by positivity
      linarith
    · have hlt : (cs - 1) * d < N := (hiff (cs - 1)).mpr (by omega)
      have hlt' : ((cs : ℝ) - 1) * (n - 1) < c * (m - 1) := by
        rw [← hN', ← hd']
        have : (((cs - 1 : ℕ) : ℝ)) = (cs : ℝ) - 1 := by push_cast [show 1 ≤ cs by omega]; ring
        rw [← this]; exact_mod_cast hlt
      -- clear denominators: `cs·n - c·m < n`, from `(cs - 1)(n - 1) < c (m - 1)` and `m ≤ n`
      have e : (cs : ℝ) / m - c / n = (cs * n - c * m) / (m * n) := by field_simp
      rw [e, div_lt_iff₀ (by positivity)]
      have e2 : 1 / (m : ℝ) * (m * n) = n := by field_simp
      rw [e2]
      have hn1 : (0 : ℝ) < n - 1 := by linarith
      have hc0 : (0 : ℝ) ≤ c := Nat.cast_nonneg _
      nlinarith [mul_lt_mul_of_pos_right hlt' hnpos,
        mul_nonneg hc0 (sub_nonneg.mpr hmn')]

/-- The 0-indexed ranks of nf_gate's sample, `⌊k (n-1)/(m-1)⌋`, as indices into the keys. -/
def evenIdx {n m : ℕ} (hm : 2 ≤ m) (hmn : m ≤ n) (k : Fin m) : Fin n :=
  ⟨k.val * (n - 1) / (m - 1), by
    have hk : k.val ≤ m - 1 := by omega
    have : k.val * (n - 1) / (m - 1) ≤ n - 1 :=
      Nat.div_le_of_le_mul (Nat.mul_le_mul_right (n - 1) hk)
    omega⟩

/-- nf_gate's evenly spaced sample of the keys `z`. -/
def evenSample {n : ℕ} (z : Fin n → ℝ) (m : ℕ) (hm : 2 ≤ m) (hmn : m ≤ n) (k : Fin m) : ℝ :=
  z (evenIdx hm hmn k)

/-- The sample contains the first and the last key, so its min-max normalization is the
population's. -/
lemma evenIdx_first_last {n m : ℕ} (hm : 2 ≤ m) (hmn : m ≤ n) :
    (evenIdx hm hmn ⟨0, by omega⟩).val = 0 ∧ (evenIdx hm hmn ⟨m - 1, by omega⟩).val = n - 1 := by
  simp only [evenIdx]
  refine ⟨by simp, ?_⟩
  rw [mul_comm]; exact Nat.mul_div_cancel _ (by omega)

/-- **Theorem 5.3.** For sorted keys `z` (for instance `z = T ∘ x` with `T` strictly increasing),
the evenly spaced sample's empirical CDF is within `1/m` of the population's, at every `t`. -/
theorem even_sample_dev {n m : ℕ} (hm : 2 ≤ m) (hmn : m ≤ n) {z : Fin n → ℝ}
    (hz : StrictMono z) (t : ℝ) :
    |ecdf (evenSample z m hm hmn) t - ecdf z t| < 1 / m := by
  set c := rankLE z t
  have hcn : c ≤ n := rankLE_le z t
  -- `z (ρ_k) ≤ t ↔ ρ_k < c ↔ k (n-1) < c (m-1)`
  have hfilt : univ.filter (fun k : Fin m => evenSample z m hm hmn k ≤ t) =
      univ.filter (fun k : Fin m => k.val * (n - 1) < c * (m - 1)) := by
    ext k
    simp only [mem_filter, mem_univ, true_and]
    show z (evenIdx hm hmn k) ≤ t ↔ _
    rw [le_iff_lt_rankLE hz t (evenIdx hm hmn k)]
    simp only [evenIdx]
    exact Nat.div_lt_iff_lt_mul (by omega)
  have h := even_sample_count hm hmn hcn
  rw [← hfilt] at h
  exact h

/-! ### Theorem 5.4: what a sample KL certifies -/

/-- **Pinsker's inequality against the uniform distribution on `B` bins** (PROOF.md's Theorem P),
as a proposition: `TV(p, unif) ≤ √(KL(p ‖ unif)/2)` for every probability vector `p`. Classical
(Pinsker 1964, constant by Csiszár and Kullback 1967) and not in the pinned Mathlib. The generic
theorems below take it as an explicit premise `hPinsker`; `pinsker_unif` proves it for `B > 0`. -/
def PinskerUnif (B : ℕ) : Prop :=
  ∀ p : Fin B → ℝ, (∀ b, 0 ≤ p b) → ∑ b, p b = 1 → tvUnif p ≤ Real.sqrt (klUnif p / 2)

/-- **Theorem 5.4 (the chain).** If the sample CDF `Fs` is within `dev` of the transformed keys'
CDF, and within `tv + invB` of `U`, and `tv ≤ √(kl/2)`, then the endpoint model's error at every
key is at most `n (√(kl/2) + invB + dev)`. -/
theorem certified_error {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) {Fs : ℝ → ℝ}
    {dev tv invB kl : ℝ}
    (hsamp : ∀ t, |ecdf u t - Fs t| ≤ dev)
    (hbin : ∀ t, |Fs t - unifCDF t| ≤ tv + invB)
    (hPinsker : tv ≤ Real.sqrt (kl / 2)) (i : Fin n) :
    |(n : ℝ) * unifCDF (u i) - (i.val + 1)| ≤ n * (Real.sqrt (kl / 2) + invB + dev) := by
  -- Theorem 3.1 (a, left) with `G = U` and `D = dev + tv + invB` (triangle inequality)
  have hD : ∀ t, (∃ i, u i ≤ t) → (∃ j, t ≤ u j) →
      |ecdf u t - unifCDF t| ≤ Real.sqrt (kl / 2) + invB + dev := by
    intro t _ _
    have h1 := hsamp t
    have h2 := hbin t
    calc |ecdf u t - unifCDF t| = |(ecdf u t - Fs t) + (Fs t - unifCDF t)| := by ring_nf
      _ ≤ |ecdf u t - Fs t| + |Fs t - unifCDF t| := abs_add_le _ _
      _ ≤ _ := by linarith
  exact err_le_of_ks hu hD i

/-- Normalized keys in `[0, 1]`: the endpoint model's prediction is `n u_i` (as `U(u_i) = u_i`). -/
lemma unifCDF_key {n : ℕ} {u : Fin n → ℝ} (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1) (i : Fin n) :
    unifCDF (u i) = u i := unifCDF_of_mem (h01 i).1 (h01 i).2

/-- **Theorem 5.4 (a), evenly spaced sample.** For min-max normalized sorted transformed keys `u`
(values in `[0, 1]`), `2 ≤ m ≤ n` and `B > 0` bins, assuming Pinsker's inequality: the endpoint
model's error at every key is at most `n (√(KL_B/2) + 1/B + 1/m)`, where `KL_B` is computed on
nf_gate's evenly spaced sample. Deterministic: no probability. -/
theorem certified_error_even {n m B : ℕ} (hm : 2 ≤ m) (hmn : m ≤ n) (hB : 0 < B)
    (hPinsker : PinskerUnif B) {u : Fin n → ℝ} (hu : StrictMono u)
    (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1) (i : Fin n) :
    |(n : ℝ) * u i - (i.val + 1)| ≤
      n * (Real.sqrt (klUnif (hist B (evenSample u m hm hmn)) / 2) + 1 / B + 1 / m) := by
  set v := evenSample u m hm hmn
  have hv : ∀ k, v k ∈ Set.Icc (0 : ℝ) 1 := fun k => h01 _
  have hP := hPinsker (hist B v) (hist_nonneg v) (hist_sum (by omega) hB v)
  have h := certified_error hu (Fs := ecdf v) (dev := 1 / m) (tv := tvUnif (hist B v))
    (invB := 1 / B) (kl := klUnif (hist B v))
    (fun t => by rw [abs_sub_comm]; exact (even_sample_dev hm hmn hu t).le)
    (fun t => ks_le_tv_add (by omega) hB v hv t) hP i
  rwa [unifCDF_key h01 i] at h

/-- **Theorem 5.4 (b), any sample, on the DKW event.** For any sample `v` of `m` normalized values
whose empirical CDF is within `τ` of the keys' CDF (the event that the DKW inequality bounds),
Pinsker gives the error bound `n (√(KL_B/2) + 1/B + τ)`. -/
theorem certified_error_iid {n m B : ℕ} (hm : 0 < m) (hB : 0 < B) (hPinsker : PinskerUnif B)
    {u : Fin n → ℝ} (hu : StrictMono u) (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1)
    (v : Fin m → ℝ) (hv : ∀ k, v k ∈ Set.Icc (0 : ℝ) 1) {τ : ℝ}
    (hev : ∀ t, |ecdf v t - ecdf u t| ≤ τ) (i : Fin n) :
    |(n : ℝ) * u i - (i.val + 1)| ≤
      n * (Real.sqrt (klUnif (hist B v) / 2) + 1 / B + τ) := by
  have hP := hPinsker (hist B v) (hist_nonneg v) (hist_sum hm hB v)
  have h := certified_error hu (Fs := ecdf v) (dev := τ) (tv := tvUnif (hist B v))
    (invB := 1 / B) (kl := klUnif (hist B v))
    (fun t => by rw [abs_sub_comm]; exact hev t)
    (fun t => ks_le_tv_add hm hB v hv t) hP i
  rwa [unifCDF_key h01 i] at h

/-- The triangle step of Theorem 5.4: sample deviation plus binned distance bound `|F_u - U|`. -/
lemma ks_chain {n : ℕ} {u : Fin n → ℝ} {Fs : ℝ → ℝ} {dev tv invB kl : ℝ}
    (hsamp : ∀ t, |ecdf u t - Fs t| ≤ dev)
    (hbin : ∀ t, |Fs t - unifCDF t| ≤ tv + invB)
    (hPinsker : tv ≤ Real.sqrt (kl / 2)) (t : ℝ) :
    |ecdf u t - unifCDF t| ≤ Real.sqrt (kl / 2) + invB + dev := by
  have h1 := hsamp t
  have h2 := hbin t
  calc |ecdf u t - unifCDF t| = |(ecdf u t - Fs t) + (Fs t - unifCDF t)| := by ring_nf
    _ ≤ |ecdf u t - Fs t| + |Fs t - unifCDF t| := abs_add_le _ _
    _ ≤ _ := by linarith

/-- **Theorem 5.4 (slots: the `+ 1`).** Under the hypotheses of `certified_error`, the endpoint
model's floored slot `⌊n U(u_i)⌋` is within `n (√(kl/2) + invB + dev) + 1` of the true slot. -/
theorem certified_slot_err {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) {Fs : ℝ → ℝ}
    {dev tv invB kl : ℝ}
    (hsamp : ∀ t, |ecdf u t - Fs t| ≤ dev)
    (hbin : ∀ t, |Fs t - unifCDF t| ≤ tv + invB)
    (hPinsker : tv ≤ Real.sqrt (kl / 2)) (i : Fin n) :
    |((⌊(n : ℝ) * unifCDF (u i)⌋ : ℤ) : ℝ) - i.val| ≤
      n * (Real.sqrt (kl / 2) + invB + dev) + 1 :=
  slot_err hu (ks_chain hsamp hbin hPinsker) i

/-- **Theorem 5.4 (b), slots, on the DKW event.** For any sample `v` within `τ` of the keys'
CDF, the floored slot is within `n (√(KL_B/2) + 1/B + τ) + 1` of the true slot. -/
theorem certified_slot_err_iid {n m B : ℕ} (hm : 0 < m) (hB : 0 < B) (hPinsker : PinskerUnif B)
    {u : Fin n → ℝ} (hu : StrictMono u) (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1)
    (v : Fin m → ℝ) (hv : ∀ k, v k ∈ Set.Icc (0 : ℝ) 1) {τ : ℝ}
    (hev : ∀ t, |ecdf v t - ecdf u t| ≤ τ) (i : Fin n) :
    |((⌊(n : ℝ) * u i⌋ : ℤ) : ℝ) - i.val| ≤
      n * (Real.sqrt (klUnif (hist B v) / 2) + 1 / B + τ) + 1 := by
  have hP := hPinsker (hist B v) (hist_nonneg v) (hist_sum hm hB v)
  have h := certified_slot_err hu (Fs := ecdf v) (dev := τ) (tv := tvUnif (hist B v))
    (invB := 1 / B) (kl := klUnif (hist B v))
    (fun t => by rw [abs_sub_comm]; exact hev t)
    (fun t => ks_le_tv_add hm hB v hv t) hP i
  rwa [unifCDF_key h01 i] at h

/-- **Theorem 5.4 (b), probabilistic form.** Let `V ω` be a random sample of `m` normalized
values (for instance `m` keys drawn i.i.d. from the key set, then transformed and normalized).
Assume Hypothesis DKW as the explicit premise `hDKW`: the sample's empirical CDF deviates from the
keys' by more than `τ` somewhere with probability at most `2 e^{-2 m τ²}`. Then the event that
some key's endpoint error exceeds `n (√(KL_B/2) + 1/B + τ)` has probability at most
`2 e^{-2 m τ²}`; that is, the bound holds with probability at least `1 - 2 e^{-2 m τ²}`. -/
theorem certified_error_prob {Ω : Type*} [MeasurableSpace Ω] (P : MeasureTheory.Measure Ω)
    {n m B : ℕ} (hm : 0 < m) (hB : 0 < B) (hPinsker : PinskerUnif B)
    {u : Fin n → ℝ} (hu : StrictMono u) (h01 : ∀ i, u i ∈ Set.Icc (0 : ℝ) 1)
    (V : Ω → Fin m → ℝ) (hV : ∀ ω k, V ω k ∈ Set.Icc (0 : ℝ) 1) {τ : ℝ}
    (hDKW : P {ω | ∃ t, τ < |ecdf (V ω) t - ecdf u t|} ≤
      ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2))) :
    P {ω | ∃ i : Fin n, (n : ℝ) * (Real.sqrt (klUnif (hist B (V ω)) / 2) + 1 / B + τ) <
        |(n : ℝ) * u i - (i.val + 1)|} ≤ ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2)) := by
  -- the failure event lies inside the DKW event (contrapositive of `certified_error_iid`)
  refine le_trans (MeasureTheory.measure_mono ?_) hDKW
  intro ω hω
  simp only [Set.mem_ofPred_eq] at hω ⊢
  obtain ⟨i, hi⟩ := hω
  by_contra hgood
  push Not at hgood
  have := certified_error_iid hm hB hPinsker hu h01 (V ω) (hV ω) hgood i
  linarith

/-! ### Theorem 5.5: the decision rule -/

/-- **Theorem 5.5 (a).** With an upper bound `U ≥ C_T` on the transformed cost and a lower bound
`L ≤ C_0` on the incumbent's, the rule "transform iff `U + c_T + δ < L`" only transforms when the
saving net of the transform's compute exceeds the margin. -/
theorem gate_sound {C0 CT cT δ U L : ℝ} (hU : CT ≤ U) (hL : L ≤ C0) (huse : U + cT + δ < L) :
    δ < C0 - CT - cT := by
  linarith

/-- **Theorem 5.5 (b).** If the bounds hold whenever the sample deviation is at most `τ`, and the
deviation exceeds `τ` with probability at most `2 exp(-2 m τ²)` (Hypothesis DKW, an explicit
premise `hDKW`), then a wrong "transform" decision has probability at most `2 exp(-2 m τ²)`. -/
theorem gate_false_positive {Ω : Type*} [MeasurableSpace Ω] (P : MeasureTheory.Measure Ω)
    (C0 CT U L dev : Ω → ℝ) (cT δ τ : ℝ) {m : ℕ}
    (hgood : ∀ ω, dev ω ≤ τ → CT ω ≤ U ω ∧ L ω ≤ C0 ω)
    (hDKW : P {ω | τ < dev ω} ≤ ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2))) :
    P {ω | U ω + cT + δ < L ω ∧ C0 ω - CT ω - cT ≤ δ} ≤
      ENNReal.ofReal (2 * Real.exp (-2 * m * τ ^ 2)) := by
  refine le_trans (MeasureTheory.measure_mono ?_) hDKW
  intro ω hω
  simp only [Set.mem_ofPred_eq] at hω ⊢
  by_contra h
  push Not at h
  obtain ⟨hU, hL⟩ := hgood ω h
  have := gate_sound hU hL hω.1
  linarith [hω.2]

end NFLProof
