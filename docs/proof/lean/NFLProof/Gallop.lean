module

public import NFLProof.Error
public import NFLProof.Cost

/-!
# Theorem 4.3: the repo's exponential search, and the end-to-end lookup cost (Corollary 4.8)

`gallop` transcribes `Index::locate_from_prediction` (`src/scaleli/index.hpp`, the table-free
branch) with a probe counter:

```
if (ok(p)) { lo = p; step = 1; hi = p + 1;
             while (hi < n && ok(hi)) { lo = hi; step *= 2; hi = min(n, p + step); }
             return last_true(ok, lo, hi); }
hi = p; step = 1; lo = p > step ? p - step : 0;
while (lo > 0 && !ok(lo)) { hi = lo; step *= 2; lo = p > step ? p - step : 0; }
return last_true(ok, lo, hi);
```

The loops are written as recursions on `step`; the loop variables are functions of it: in the
forward loop `hi = min n (p + step)` and `lo = p + step / 2` (the previous `hi`, or `p` when
`step = 1`); in the backward loop `lo = p - step` (truncated at 0, exactly `p > step ? p - step : 0`)
and `hi = p - step / 2`.

* `gallop_spec` (Theorem 4.3): for a threshold predicate (`ok i ↔ i ≤ t`, `t < n`) and any start
  `p < n`, `gallop` returns `t`, with at most `2 ⌈log₂ (d + 1)⌉ + 2` probes, `d = |p - t|`.
  Truncation at `0` or `n` is included.
* `predSlot`: the clamp of `locate_from_prediction` (`y ≤ 0 ↦ 0`, `y ≥ n-1 ↦ n-1`, else `⌊y⌋`).
* `lookup_cost_of_ks` (Corollary 4.8): a key lookup started from the floored, clamped prediction
  of a CDF-scale model within `D` of the empirical CDF returns the right key, with at most
  `2 ⌈log₂ (⌊n D⌋ + 2)⌉ + 2` probes. `lookup_bound_mono`: that bound is nondecreasing in `D`.
-/

@[expose] public section

open Finset

namespace NFLProof

/-! ### The transcription -/

/-- Forward gallop (`ok p` was true). Returns `(lo, hi, probes)`. -/
def gallopUp (ok : ℕ → Bool) (n p step : ℕ) : ℕ × ℕ × ℕ :=
  if _h : p + step < n ∧ 0 < step then
    if ok (p + step) then
      let r := gallopUp ok n p (2 * step)
      (r.1, r.2.1, r.2.2 + 1)
    else (p + step / 2, p + step, 1)
  else (p + step / 2, min n (p + step), 0)
termination_by n - step
decreasing_by omega

/-- Backward gallop (`ok p` was false). Returns `(lo, hi, probes)`. -/
def gallopDown (ok : ℕ → Bool) (p step : ℕ) : ℕ × ℕ × ℕ :=
  if _h : 0 < p - step ∧ 0 < step then
    if ok (p - step) then (p - step, p - step / 2, 1)
    else
      let r := gallopDown ok p (2 * step)
      (r.1, r.2.1, r.2.2 + 1)
  else (p - step, p - step / 2, 0)
termination_by p - step
decreasing_by omega

/-- `locate_from_prediction` from the clamped start `p`: one probe at `p`, a gallop, then
`last_true`. Returns `(answer, number of probes)`. -/
def gallop (ok : ℕ → Bool) (p n : ℕ) : ℕ × ℕ :=
  if ok p then
    let g := gallopUp ok n p 1
    let r := lastTrue ok g.1 g.2.1
    (r.1, 1 + g.2.2 + r.2)
  else
    let g := gallopDown ok p 1
    let r := lastTrue ok g.1 g.2.1
    (r.1, 1 + g.2.2 + r.2)

/-! ### Loop invariants -/

/-- **Forward loop.** Entered with `step = 2^j` and `ok (p + 2^j / 2)` true, it stops at some
`step = 2^J` with `lo = p + 2^J/2 ≤ t < hi = min n (p + 2^J)`, after `≤ J - j + 1` probes. -/
theorem gallopUp_spec (ok : ℕ → Bool) {n p t : ℕ} (hok : ∀ i, ok i = true ↔ i ≤ t)
    (ht : t < n) (j : ℕ) (hlo : p + 2 ^ j / 2 ≤ t) :
    ∃ J, j ≤ J ∧ (gallopUp ok n p (2 ^ j)).1 = p + 2 ^ J / 2 ∧
      (gallopUp ok n p (2 ^ j)).2.1 = min n (p + 2 ^ J) ∧ p + 2 ^ J / 2 ≤ t ∧
      t < (gallopUp ok n p (2 ^ j)).2.1 ∧ (gallopUp ok n p (2 ^ j)).2.2 ≤ J - j + 1 := by
  -- strong induction on the loop measure `n - 2^j`
  induction hm : n - 2 ^ j using Nat.strong_induction_on generalizing j with
  | _ m ih =>
    have hpos : 0 < 2 ^ j := Nat.two_pow_pos j
    rw [gallopUp]
    by_cases hc : p + 2 ^ j < n
    · simp only [hc, hpos, and_self, ↓reduceDIte]
      by_cases hprobe : ok (p + 2 ^ j) = true
      · -- the probe at `p + 2^j` is true: continue with `step = 2^(j+1)`
        simp only [hprobe, ↓reduceIte]
        have hle : p + 2 ^ j ≤ t := (hok _).mp hprobe
        have h2 : 2 * 2 ^ j = 2 ^ (j + 1) := by rw [pow_succ]; ring
        rw [h2]
        obtain ⟨J, hJ, h1, h2', h3, h4, h5⟩ := ih (n - 2 ^ (j + 1)) (by omega) (j + 1)
          (by rw [← h2]; omega) rfl
        exact ⟨J, by omega, h1, h2', h3, h4, by omega⟩
      · -- the probe is false: stop with `hi = p + 2^j > t`
        simp only [hprobe, Bool.false_eq_true, ↓reduceIte]
        have hgt : t < p + 2 ^ j := by
          by_contra h; exact hprobe ((hok _).mpr (by omega))
        exact ⟨j, le_rfl, rfl, by omega, hlo, hgt, by simp⟩
    · -- `hi = n`: stop without probing
      simp only [show ¬ (p + 2 ^ j < n ∧ 0 < 2 ^ j) by omega, ↓reduceDIte]
      exact ⟨j, le_rfl, rfl, rfl, hlo, by omega, by simp⟩

/-- **Backward loop.** Entered with `step = 2^j` and `t < p - 2^j / 2` (the current `hi` is
false), it stops at some `step = 2^J` with `lo = p - 2^J ≤ t < hi = p - 2^J/2`, after
`≤ J - j + 1` probes. At `lo = 0` it does not probe, as in the C++ (`ok 0` holds since `0 ≤ t`). -/
theorem gallopDown_spec (ok : ℕ → Bool) {p t : ℕ} (hok : ∀ i, ok i = true ↔ i ≤ t)
    (j : ℕ) (hhi : t < p - 2 ^ j / 2) :
    ∃ J, j ≤ J ∧ (gallopDown ok p (2 ^ j)).1 = p - 2 ^ J ∧
      (gallopDown ok p (2 ^ j)).2.1 = p - 2 ^ J / 2 ∧
      (gallopDown ok p (2 ^ j)).1 ≤ t ∧ t < (gallopDown ok p (2 ^ j)).2.1 ∧
      (gallopDown ok p (2 ^ j)).2.2 ≤ J - j + 1 := by
  induction hm : p - 2 ^ j using Nat.strong_induction_on generalizing j with
  | _ m ih =>
    have hpos : 0 < 2 ^ j := Nat.two_pow_pos j
    rw [gallopDown]
    by_cases hc : 0 < p - 2 ^ j
    · simp only [hc, hpos, and_self, ↓reduceDIte]
      by_cases hprobe : ok (p - 2 ^ j) = true
      · -- the probe at `p - 2^j` is true: stop
        simp only [hprobe, ↓reduceIte]
        exact ⟨j, le_rfl, rfl, rfl, (hok _).mp hprobe, hhi, by simp⟩
      · -- false: continue with `step = 2^(j+1)`, whose `hi` is `p - 2^j > t`
        simp only [hprobe, Bool.false_eq_true, ↓reduceIte]
        have hgt : t < p - 2 ^ j := by
          by_contra h; exact hprobe ((hok _).mpr (by omega))
        have h2 : 2 * 2 ^ j = 2 ^ (j + 1) := by rw [pow_succ]; ring
        rw [h2]
        obtain ⟨J, hJ, h1, h2', h3, h4, h5⟩ := ih (p - 2 ^ (j + 1)) (by omega) (j + 1)
          (by rw [← h2]; omega) rfl
        exact ⟨J, by omega, h1, h2', h3, h4, by omega⟩
    · -- `lo = 0`: stop without probing
      simp only [show ¬ (0 < p - 2 ^ j ∧ 0 < 2 ^ j) by omega, ↓reduceDIte]
      exact ⟨j, le_rfl, rfl, rfl, by omega, hhi, by simp⟩

/-! ### Theorem 4.3 -/

/-- If the gallop stopped at `step = 2^J` with `2^J / 2 ≤ d`, then `J ≤ ⌈log₂ (d + 1)⌉`. -/
lemma le_clog_of_half_pow_le {J d : ℕ} (h : 2 ^ J / 2 ≤ d) : J ≤ Nat.clog 2 (d + 1) := by
  rcases J with _ | J
  · exact Nat.zero_le _
  · -- `2^J ≤ d < d + 1 ≤ 2^⌈log₂ (d+1)⌉`
    have hJ : 2 ^ (J + 1) / 2 = 2 ^ J := by rw [pow_succ]; simp
    rw [hJ] at h
    have hc := Nat.le_pow_clog (by norm_num : 1 < 2) (d + 1)
    have : 2 ^ J < 2 ^ Nat.clog 2 (d + 1) := by omega
    have := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mp this
    omega

/-- The final `last_true` window after a gallop that stopped at `2^J` has length `≤ 2^(J-1)`, so
`last_true` makes at most `J - 1` probes. -/
lemma clog_window_le {J w : ℕ} (hw : w ≤ 2 ^ J - 2 ^ J / 2) : Nat.clog 2 w ≤ J - 1 := by
  apply (Nat.clog_le_iff_le_pow (by norm_num)).mpr
  rcases J with _ | J
  · simpa using hw
  · have : 2 ^ (J + 1) - 2 ^ (J + 1) / 2 = 2 ^ J := by rw [pow_succ]; omega
    simpa [this] using hw

/-- **Theorem 4.3, core form.** For a threshold predicate (`ok i ↔ i ≤ t`, `t < n`) and any start
`p`, `gallop ok p n` returns `t`, and the gallop stops at some `step = 2^J` with `2^J / 2 ≤ d`,
`d = |p - t|`, having made at most `2J + 2` probes in total (`2J + 1` if `J ≥ 1`). -/
theorem gallop_core (ok : ℕ → Bool) {n p t : ℕ} (ht : t < n)
    (hok : ∀ i, ok i = true ↔ i ≤ t) :
    (gallop ok p n).1 = t ∧ ∃ J, 2 ^ J / 2 ≤ Int.natAbs ((p : ℤ) - t) ∧
      (gallop ok p n).2 ≤ 2 * J + 2 ∧ (1 ≤ J → (gallop ok p n).2 ≤ 2 * J + 1) := by
  -- a probe beyond `t` is false
  have hfalse : ∀ i, t < i → ok i = false := fun i hi =>
    Bool.eq_false_iff.mpr (fun h => absurd ((hok i).mp h) (by omega))
  unfold gallop
  by_cases hpt : ok p = true
  · -- forward: `p ≤ t`; the gallop ends with `lo = p + 2^J/2 ≤ t < hi ≤ p + 2^J`
    simp only [hpt, ↓reduceIte]
    have hpt' : p ≤ t := (hok p).mp hpt
    obtain ⟨J, -, h1, h2, hlo, h3, h4⟩ := gallopUp_spec ok hok ht 0 (by simpa using hpt')
    rw [pow_zero] at h1 h2 h3 h4
    -- `last_true` on `[lo, hi)` returns `t` ...
    have hl := lastTrue_eq ok hok (lo := (gallopUp ok n p 1).1) (hi := (gallopUp ok n p 1).2.1)
      (by omega) h3
    -- ... with at most `⌈log₂ (hi - lo)⌉ ≤ J - 1` probes, as `hi - lo ≤ 2^(J-1)`
    obtain ⟨-, -, -, -, hcost⟩ := lastTrue_spec ok (lo := (gallopUp ok n p 1).1)
      (hi := (gallopUp ok n p 1).2.1) (by omega) ((hok _).mpr (by omega)) (hfalse _ h3)
    have hwin := clog_window_le (J := J)
      (w := (gallopUp ok n p 1).2.1 - (gallopUp ok n p 1).1) (by rw [h1, h2]; omega)
    have hd : Int.natAbs ((p : ℤ) - t) = t - p := by omega
    rw [hd]
    -- total: `1` (at `p`) `+ (J + 1)` (gallop) `+ (J - 1)` (`last_true`)
    exact ⟨hl, J, by omega, by omega, fun _ => by omega⟩
  · -- backward: `t < p`; the gallop ends with `p - 2^J = lo ≤ t < hi = p - 2^J/2`
    simp only [hpt, Bool.false_eq_true, ↓reduceIte]
    have hpt' : t < p := by
      by_contra h; exact hpt ((hok p).mpr (by omega))
    obtain ⟨J, -, h1, h2, hlo, h3, h4⟩ := gallopDown_spec ok hok 0 (by rw [pow_zero]; omega)
    rw [pow_zero] at h1 h2 hlo h3 h4
    have hl := lastTrue_eq ok hok (lo := (gallopDown ok p 1).1) (hi := (gallopDown ok p 1).2.1)
      hlo h3
    obtain ⟨-, -, -, -, hcost⟩ := lastTrue_spec ok (lo := (gallopDown ok p 1).1)
      (hi := (gallopDown ok p 1).2.1) (by omega) ((hok _).mpr hlo) (hfalse _ h3)
    have hwin := clog_window_le (J := J)
      (w := (gallopDown ok p 1).2.1 - (gallopDown ok p 1).1) (by rw [h1, h2]; omega)
    have hd : Int.natAbs ((p : ℤ) - t) = p - t := by omega
    rw [hd]
    exact ⟨hl, J, by omega, by omega, fun _ => by omega⟩

/-- **Theorem 4.3 (exponential search from any start).** For a threshold predicate
(`ok i ↔ i ≤ t`, `t < n`) and any start `p`, `gallop ok p n` returns `t`, using at most
`2 ⌈log₂ (|p - t| + 1)⌉ + 2` probes. Truncation at `0` and `n` is included. (PROOF.md's
`hmono`, `ok 0` and `p < n` premises are not needed: the first two follow from `hok`.) -/
theorem gallop_spec (ok : ℕ → Bool) {n p t : ℕ} (ht : t < n)
    (hok : ∀ i, ok i = true ↔ i ≤ t) :
    (gallop ok p n).1 = t ∧
      (gallop ok p n).2 ≤ 2 * Nat.clog 2 (Int.natAbs ((p : ℤ) - t) + 1) + 2 := by
  obtain ⟨h1, J, hJd, hc, -⟩ := gallop_core ok (p := p) ht hok
  -- `J ≤ ⌈log₂ (d + 1)⌉`
  have := le_clog_of_half_pow_le hJd
  exact ⟨h1, by omega⟩

/-- **Theorem 4.3 (real form of the upper bound).** The same search costs at most
`2 log₂ (|p - t| + 1) + 3` probes. -/
theorem gallop_cost_logb (ok : ℕ → Bool) {n p t : ℕ} (ht : t < n)
    (hok : ∀ i, ok i = true ↔ i ≤ t) :
    ((gallop ok p n).2 : ℝ) ≤ 2 * Real.logb 2 ((Int.natAbs ((p : ℤ) - t) : ℝ) + 1) + 3 := by
  obtain ⟨-, J, hJd, hc, hc1⟩ := gallop_core ok (p := p) ht hok
  set d := Int.natAbs ((p : ℤ) - t)
  have hd0 : (0 : ℝ) ≤ d := Nat.cast_nonneg d
  have hlog0 : 0 ≤ Real.logb 2 ((d : ℝ) + 1) := Real.logb_nonneg (by norm_num) (by linarith)
  rcases J with _ | J
  · -- `J = 0`: at most 2 probes
    have : ((gallop ok p n).2 : ℝ) ≤ 2 := by exact_mod_cast hc
    linarith
  · -- `J ≥ 1`: `2^J ≤ d`, so `J ≤ log₂ (d + 1)` and the cost `≤ 2 (J + 1) + 1`
    have hpow : 2 ^ J ≤ d := by
      have : 2 ^ (J + 1) / 2 = 2 ^ J := by rw [pow_succ]; simp
      omega
    have hpow' : (2 : ℝ) ^ J ≤ (d : ℝ) + 1 := by
      have : ((2 ^ J : ℕ) : ℝ) ≤ d := by exact_mod_cast hpow
      push_cast at this; linarith
    have hJlog : (J : ℝ) ≤ Real.logb 2 ((d : ℝ) + 1) := by
      have h := Real.logb_le_logb_of_le (b := 2) (by norm_num) (by positivity) hpow'
      rwa [Real.logb_pow, Real.logb_self_eq_one (by norm_num), mul_one] at h
    have : ((gallop ok p n).2 : ℝ) ≤ 2 * ((J + 1 : ℕ) : ℝ) + 1 := by
      exact_mod_cast hc1 (by omega)
    push_cast at this
    linarith

/-- The probe bound `2 ⌈log₂ (d + 1)⌉ + 2` is nondecreasing in the distance `d`. -/
theorem gallop_bound_mono {d d' : ℕ} (h : d ≤ d') :
    2 * Nat.clog 2 (d + 1) + 2 ≤ 2 * Nat.clog 2 (d' + 1) + 2 := by
  have := Nat.clog_mono_right 2 (show d + 1 ≤ d' + 1 by omega)
  omega

/-- **Corollary 4.4 for the repo's search.** Over `Q` lookups (start `p j`, answer `t j`), the
mean number of probes is at most `2 log₂ (1 + mean distance) + 3`. -/
theorem gallop_mean_cost {Q n : ℕ} (hQ : 0 < Q) (ok : Fin Q → ℕ → Bool) (p t : Fin Q → ℕ)
    (ht : ∀ j, t j < n) (hok : ∀ j i, ok j i = true ↔ i ≤ t j) :
    (∑ j, ((gallop (ok j) (p j) n).2 : ℝ)) / Q ≤
      2 * Real.logb 2 (1 + (∑ j, (Int.natAbs ((p j : ℤ) - t j) : ℝ)) / Q) + 3 :=
  mean_cost_le hQ (fun j => (Int.natAbs ((p j : ℤ) - t j) : ℝ))
    (fun j => ((gallop (ok j) (p j) n).2 : ℝ)) (fun _ => Nat.cast_nonneg _)
    (fun j => gallop_cost_logb (ok j) (ht j) (hok j))

/-! ### Corollary 4.8: from the Kolmogorov distance to the lookup cost -/

/-- The probe of a key lookup for `q` in sorted keys `u`: `ok j ↔ u_j ≤ q` (`false` past the end),
as in `locate_region`'s `regions_[i]->low_fence <= k`. -/
noncomputable def keyProbe {n : ℕ} (u : Fin n → ℝ) (q : ℝ) (j : ℕ) : Bool :=
  if h : j < n then decide (u ⟨j, h⟩ ≤ q) else false

/-- The clamp of `locate_from_prediction`: `y ≤ 0 ↦ 0`, `y ≥ n - 1 ↦ n - 1`, else `⌊y⌋`. -/
noncomputable def predSlot (n : ℕ) (y : ℝ) : ℕ := min (n - 1) ⌊y⌋.toNat

/-- For a key lookup in sorted keys, the probe is a threshold predicate at the key's index. -/
lemma keyProbe_iff {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) (i : Fin n) (j : ℕ) :
    keyProbe u (u i) j = true ↔ j ≤ i.val := by
  unfold keyProbe
  by_cases h : j < n
  · simp only [h, ↓reduceDIte, decide_eq_true_eq]
    rw [hu.le_iff_le, Fin.le_def]
  · simp only [h, ↓reduceDIte, Bool.false_eq_true, false_iff, not_le]
    omega

/-- **Corollary 4.8 (end to end).** Let `G` be a CDF-scale model within `D` of the empirical CDF
of the sorted keys `u` (for instance `G = U` and `D = D_KS`). A key lookup for `u i` started, as
in `locate_from_prediction`, from the floored and clamped prediction `⌊n G(u_i)⌋` returns `i` and
makes at most `2 ⌈log₂ (⌊n D⌋ + 2)⌉ + 2` probes. -/
theorem lookup_cost_of_ks {n : ℕ} {u : Fin n → ℝ} (hu : StrictMono u) {G : ℝ → ℝ} {D : ℝ}
    (hD : ∀ t, |ecdf u t - G t| ≤ D) (i : Fin n) :
    (gallop (keyProbe u (u i)) (predSlot n ((n : ℝ) * G (u i))) n).1 = i.val ∧
      (gallop (keyProbe u (u i)) (predSlot n ((n : ℝ) * G (u i))) n).2 ≤
        2 * Nat.clog 2 (⌊(n : ℝ) * D⌋₊ + 2) + 2 := by
  obtain ⟨h1, h2⟩ := gallop_spec (keyProbe u (u i)) (p := predSlot n ((n : ℝ) * G (u i)))
    i.isLt (keyProbe_iff hu i)
  refine ⟨h1, h2.trans ?_⟩
  set f : ℤ := ⌊(n : ℝ) * G (u i)⌋ with hf
  -- Theorem 3.2: the floored slot is within `n D + 1` of `i`
  have hslot : |(f : ℝ) - i.val| ≤ n * D + 1 := slot_err hu hD i
  have hD0 : 0 ≤ D := (abs_nonneg _).trans (hD 0)
  -- Lemma 3.7 in `ℤ`: clamping into `[0, n - 1]` (which contains `i`) does not increase it
  have hclamp : Int.natAbs ((predSlot n ((n : ℝ) * G (u i)) : ℤ) - i.val) ≤
      Int.natAbs (f - i.val) := by
    have := i.isLt
    unfold predSlot
    rw [← hf]
    omega
  -- so the distance `d` satisfies `d ≤ ⌊n D⌋ + 1`
  have hreal : ((Int.natAbs (f - i.val) : ℕ) : ℝ) ≤ n * D + 1 := by
    have e : ((Int.natAbs (f - i.val) : ℕ) : ℝ) = |(f : ℝ) - i.val| := by
      rw [Nat.cast_natAbs]; push_cast; rfl
    rw [e]; exact hslot
  have hfloor : Int.natAbs (f - i.val) ≤ ⌊(n : ℝ) * D⌋₊ + 1 := by
    have hnD : 0 ≤ (n : ℝ) * D := mul_nonneg (Nat.cast_nonneg n) hD0
    rw [← Nat.floor_add_one hnD]
    exact Nat.le_floor hreal
  -- and the bound of Theorem 4.3 is monotone in `d`
  exact gallop_bound_mono (by omega)

/-- The end-to-end bound is nondecreasing in `D`: lowering the Kolmogorov distance never raises
the guaranteed search cost. -/
theorem lookup_bound_mono {n : ℕ} {D D' : ℝ} (h : D ≤ D') :
    2 * Nat.clog 2 (⌊(n : ℝ) * D⌋₊ + 2) + 2 ≤ 2 * Nat.clog 2 (⌊(n : ℝ) * D'⌋₊ + 2) + 2 := by
  have hf : ⌊(n : ℝ) * D⌋₊ ≤ ⌊(n : ℝ) * D'⌋₊ :=
    Nat.floor_le_floor (mul_le_mul_of_nonneg_left h (Nat.cast_nonneg n))
  exact gallop_bound_mono (d := ⌊(n : ℝ) * D⌋₊ + 1) (d' := ⌊(n : ℝ) * D'⌋₊ + 1) (by omega)

end NFLProof
