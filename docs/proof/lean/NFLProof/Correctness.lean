module

public import NFLProof.Defs

/-!
# §2 Correctness: a strictly increasing transform preserves what the index must answer

* Theorem 2.1 `rank_comp_strictMono`, `range_comp_strictMono`, `strictMono_comp`,
  `rankLE_self` (in `Defs`): ranks, lower-bound slots and ranges are unchanged by a strictly
  increasing `T`, and the transformed keys are again sorted.
* Theorem 2.2 `ranks_preserved_iff`: the transformed keys have the right ranks **iff** they are
  strictly increasing. So monotonicity is not only sufficient but necessary.
* Proposition 2.3 `point_lookup`: point lookups only need injectivity.
-/

@[expose] public section

open Finset

namespace NFLProof

/-- The key-wise equivalence behind Theorem 2.1: `x ≤ q ↔ T x ≤ T q` for strictly increasing
`T` (both directions of PROOF.md's proof at once). -/
lemma le_iff_comp {T : ℝ → ℝ} (hT : StrictMono T) (a q : ℝ) : T a ≤ T q ↔ a ≤ q :=
  hT.le_iff_le

/-- **Theorem 2.1 (order preservation, ranks and lower bounds).** -/
theorem rank_comp_strictMono {n : ℕ} {T : ℝ → ℝ} (hT : StrictMono T) (x : Fin n → ℝ) (q : ℝ) :
    rankLE (T ∘ x) (T q) = rankLE x q ∧ rankLT (T ∘ x) (T q) = rankLT x q := by
  constructor
  · -- the two filters select the same indices, because `T (x i) ≤ T q ↔ x i ≤ q`
    unfold rankLE
    congr 1
    ext i
    simp [hT.le_iff_le]
  · unfold rankLT
    congr 1
    ext i
    simp [hT.lt_iff_lt]

/-- **Theorem 2.1 (ranges).** A range query `[q₁, q₂]` selects the same keys before and after
the transform. -/
theorem range_comp_strictMono {n : ℕ} {T : ℝ → ℝ} (hT : StrictMono T) (x : Fin n → ℝ)
    (q₁ q₂ : ℝ) :
    univ.filter (fun i => q₁ ≤ x i ∧ x i ≤ q₂) =
      univ.filter (fun i => T q₁ ≤ T (x i) ∧ T (x i) ≤ T q₂) := by
  ext i
  simp [hT.le_iff_le]

/-- **Theorem 2.1 (sortedness).** Sorted keys stay sorted, so `T` restricted to the keys is an
order-preserving bijection onto the transformed keys. -/
theorem strictMono_comp {n : ℕ} {T : ℝ → ℝ} (hT : StrictMono T) {x : Fin n → ℝ}
    (hx : StrictMono x) : StrictMono (T ∘ x) :=
  hT.comp hx

/-- **Theorem 2.1 (the inverse on the keys).** The transformed key `T (x i)` determines `i`:
its rank among the transformed keys is `i + 1`. -/
theorem rank_of_transformed {n : ℕ} {T : ℝ → ℝ} (hT : StrictMono T) {x : Fin n → ℝ}
    (hx : StrictMono x) (i : Fin n) : rankLE (T ∘ x) (T (x i)) = i.val + 1 :=
  rankLE_self (hT.comp hx) i

/-- **Theorem 2.2 (monotonicity is necessary for ranks).** For any keys `z` (for instance
`z = T ∘ x` with an arbitrary `T`): every `z i` has rank `i + 1` among the `z`'s iff
`z` is strictly increasing. -/
theorem ranks_preserved_iff {n : ℕ} (z : Fin n → ℝ) :
    (∀ i, rankLE z (z i) = i.val + 1) ↔ StrictMono z := by
  constructor
  · intro h i j hij
    -- suppose `z j ≤ z i` with `i < j`; then rank `z j ≤` rank `z i`, i.e. `j + 1 ≤ i + 1`
    by_contra hcon
    push Not at hcon
    have hsub : univ.filter (fun k => z k ≤ z j) ⊆ univ.filter (fun k => z k ≤ z i) := by
      intro k hk
      simp only [mem_filter, mem_univ, true_and] at hk ⊢
      exact hk.trans hcon
    have hcard := card_le_card hsub
    have hi := h i
    have hj := h j
    unfold rankLE at hi hj
    rw [hi, hj] at hcard
    have : j.val ≤ i.val := by omega
    exact absurd hij (not_lt.mpr (Fin.le_def.mpr this))
  · intro hz i
    exact rankLE_self hz i

/-- **Proposition 2.3 (point lookups need only injectivity).** -/
theorem point_lookup {T : ℝ → ℝ} (hT : Function.Injective T) (a q : ℝ) : T a = T q ↔ a = q :=
  hT.eq_iff

end NFLProof
