module

import Mathlib

/-! Smoke test: the toolchain and Mathlib import work. -/

example (a b : ℝ) : (a + b) ^ 2 = a ^ 2 + 2 * a * b + b ^ 2 := by ring

theorem smoke_sum (n : ℕ) : (∑ _i ∈ Finset.range n, (1 : ℝ)) = n := by simp
