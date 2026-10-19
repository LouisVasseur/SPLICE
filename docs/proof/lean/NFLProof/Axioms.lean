module

import NFLProof.Correctness
import NFLProof.Error
import NFLProof.Cost
import NFLProof.Gallop
import NFLProof.Gate
import NFLProof.Pinsker
import NFLProof.Limits

/-!
# Axiom audit

Every theorem and every supporting lemma of the formalization (81 theorems and 46 lemmas; the
smoke test in `Basic.lean` aside), with the axioms it depends on. Each `#print axioms` is wrapped
in `#guard_msgs`, which fails the build unless the printed message equals the docstring above it.
So a successful `lake build` certifies that each of these 127 declarations depends only on Lean's
standard axioms `propext`, `Classical.choice` and `Quot.sound` (three use only `propext` and
`Quot.sound`): no placeholder for an unfinished proof, and no axiom of this project.
Pinsker's inequality is a theorem (`pinsker_unif`); the DKW inequality is never an axiom, only the
explicit premise `hDKW` of `certified_error_prob`, `certified_error_prob_proved` and
`gate_false_positive`.

The unguarded block at the end prints the headline theorems in the build log.
-/

/-! ## `Defs.lean`: §1 definitions -/

/-- info: 'NFLProof.rankLE_self' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.rankLE_self

/-! ## `Correctness.lean`: §2 correctness -/

/-- info: 'NFLProof.rank_comp_strictMono' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.rank_comp_strictMono

/-- info: 'NFLProof.range_comp_strictMono' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.range_comp_strictMono

/-- info: 'NFLProof.strictMono_comp' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.strictMono_comp

/-- info: 'NFLProof.rank_of_transformed' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.rank_of_transformed

/-- info: 'NFLProof.ranks_preserved_iff' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ranks_preserved_iff

/-- info: 'NFLProof.point_lookup' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.point_lookup

/-! ## `Error.lean`: §3 error -/

/-- info: 'NFLProof.err_le_of_ks' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.err_le_of_ks

/-- info: 'NFLProof.ks_le_of_err' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ks_le_of_err

/-- info: 'NFLProof.ecdf_sub_unif_le_ksUnif' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ecdf_sub_unif_le_ksUnif

/-- info: 'NFLProof.ksUnif_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ksUnif_le

/-- info: 'NFLProof.ks_sandwich' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ks_sandwich

/-- info: 'NFLProof.slot_err' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.slot_err

/-- info: 'NFLProof.ks_ge_inv_n' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ks_ge_inv_n

/-- info: 'NFLProof.ks_ge_inv_n_of_normalized' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ks_ge_inv_n_of_normalized

/-- info: 'NFLProof.pit_ks' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.pit_ks

/-- info: 'NFLProof.pit_endpoint_err' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.pit_endpoint_err

/-- info: 'NFLProof.pit_exact' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.pit_exact

/-- info: 'NFLProof.zero_error_iff' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.zero_error_iff

/-- info: 'NFLProof.affine_invariance' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.affine_invariance

/-- info: 'NFLProof.transform_helps' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.transform_helps

/-- info: 'NFLProof.err_le_ks_add_disp' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.err_le_ks_add_disp

/-- info: 'NFLProof.disp_le_err_add_ks' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.disp_le_err_add_ks

/-- info: 'NFLProof.clamp_err' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.clamp_err

/-- info: 'NFLProof.slot_err_ksUnif' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.slot_err_ksUnif

/-- info: 'NFLProof.pit_ksUnif_eq' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.pit_ksUnif_eq

/-- info: 'NFLProof.pit_endpointErr_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.pit_endpointErr_le

/-! ## `Cost.lean`: §4 cost -/

/-- info: 'NFLProof.lastTrue_spec' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.lastTrue_spec

/-- info: 'NFLProof.lastTrue_eq' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.lastTrue_eq

/-- info: 'NFLProof.window_cost_mono' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.window_cost_mono

/-- info: 'NFLProof.DTree.card_leaves_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.DTree.card_leaves_le

/-- info: 'NFLProof.DTree.clog_le_depth' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.DTree.clog_le_depth

/-- info: 'NFLProof.mean_cost_jensen' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.mean_cost_jensen

/-- info: 'NFLProof.mean_cost_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.mean_cost_le

/-- info: 'NFLProof.beats_binary_of_mean_dist' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.beats_binary_of_mean_dist

/-- info: 'NFLProof.win_needs_small_log_dist' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.win_needs_small_log_dist

/-- info: 'NFLProof.conflict_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.conflict_le

/-- info: 'NFLProof.conflict_le_ks' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.conflict_le_ks

/-- info: 'NFLProof.conflict_ge_density' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.conflict_ge_density

/-! ## `Gallop.lean`: §4.3 and Corollary 4.8: exponential search and lookup cost -/

/-- info: 'NFLProof.gallopUp_spec' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.gallopUp_spec

/-- info: 'NFLProof.gallopDown_spec' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.gallopDown_spec

/-- info: 'NFLProof.gallop_core' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.gallop_core

/-- info: 'NFLProof.gallop_spec' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.gallop_spec

/-- info: 'NFLProof.gallop_cost_logb' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.gallop_cost_logb

/-- info: 'NFLProof.gallop_bound_mono' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.gallop_bound_mono

/-- info: 'NFLProof.gallop_mean_cost' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.gallop_mean_cost

/-- info: 'NFLProof.lookup_cost_of_ks' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.lookup_cost_of_ks

/-- info: 'NFLProof.lookup_bound_mono' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.lookup_bound_mono

/-! ## `Gate.lean`: §5 the gate -/

/-- info: 'NFLProof.prefix_le_tv' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.prefix_le_tv

/-- info: 'NFLProof.ks_le_tv_add' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ks_le_tv_add

/-- info: 'NFLProof.ksBinned_le_tv' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ksBinned_le_tv

/-- info: 'NFLProof.ksBinned_hist_le_tvUnif' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ksBinned_hist_le_tvUnif

/-- info: 'NFLProof.even_sample_count' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.even_sample_count

/-- info: 'NFLProof.even_sample_dev' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.even_sample_dev

/-- info: 'NFLProof.certified_error' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.certified_error

/-- info: 'NFLProof.certified_error_even' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.certified_error_even

/-- info: 'NFLProof.certified_error_iid' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.certified_error_iid

/-- info: 'NFLProof.certified_slot_err' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.certified_slot_err

/-- info: 'NFLProof.certified_slot_err_iid' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.certified_slot_err_iid

/-- info: 'NFLProof.certified_error_prob' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.certified_error_prob

/-- info: 'NFLProof.gate_sound' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.gate_sound

/-- info: 'NFLProof.gate_false_positive' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.gate_false_positive

/-! ## `Pinsker.lean`: Theorem P (Pinsker, proved) and Theorem 5.4 without a Pinsker premise -/

/-- info: 'NFLProof.psi_nonneg' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.psi_nonneg

/-- info: 'NFLProof.klTerm_lower' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.klTerm_lower

/-- info: 'NFLProof.pinsker_fin' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.pinsker_fin

/-- info: 'NFLProof.pinsker_tv' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.pinsker_tv

/-- info: 'NFLProof.pinsker_unif' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.pinsker_unif

/-- info: 'NFLProof.klUnif_eq_sum_klFun' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.klUnif_eq_sum_klFun

/-- info: 'NFLProof.klUnif_nonneg' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.klUnif_nonneg

/-- info: 'NFLProof.certified_error_even_proved' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.certified_error_even_proved

/-- info: 'NFLProof.certified_slot_err_even_proved' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.certified_slot_err_even_proved

/-- info: 'NFLProof.certified_error_prob_proved' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.certified_error_prob_proved

/-! ## `Limits.lean`: §6 limits -/

/-- info: 'NFLProof.piece_composite' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.piece_composite

/-- info: 'NFLProof.kpiece_composite' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.kpiece_composite

/-- info: 'NFLProof.run_diff' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.run_diff

/-- info: 'NFLProof.two_run' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.two_run

/-- info: 'NFLProof.rms_lower' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.rms_lower

/-- info: 'NFLProof.cv_floor' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.cv_floor

/-- info: 'NFLProof.span_ratio' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.span_ratio

/-- info: 'NFLProof.ratio_form' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ratio_form

/-- info: 'NFLProof.smooth_floor' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.smooth_floor

/-! ## Supporting lemmas (all files)

The lemmas that the theorems above use, audited the same way. -/

/-- info: 'NFLProof.bin_lt' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.bin_lt

/-- info: 'NFLProof.card_bins_lt' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.card_bins_lt

/-- info: 'NFLProof.chiP_monotoneOn' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.chiP_monotoneOn

/-- info: 'NFLProof.chiP_one' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.chiP_one

/-- info: 'NFLProof.clog_two_succ' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.clog_two_succ

/-- info: 'NFLProof.clog_window_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.clog_window_le

/-- info: 'NFLProof.continuous_unifCDF' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.continuous_unifCDF

/-- info: 'NFLProof.ecdf_self' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ecdf_self

/-- info: 'NFLProof.endpointErr_ge' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.endpointErr_ge

/-- info: 'NFLProof.evenIdx_first_last' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.evenIdx_first_last

/-- info: 'NFLProof.exists_bracket' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.exists_bracket

/-- info: 'NFLProof.filter_le_eq_Iic' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.filter_le_eq_Iic

/-- info: 'NFLProof.hasDerivAt_chiP' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.hasDerivAt_chiP

/-- info: 'NFLProof.hasDerivAt_psiP' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.hasDerivAt_psiP

/-- info: 'NFLProof.hist_nonneg' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.hist_nonneg

/-- info: 'NFLProof.hist_prefix' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.hist_prefix

/-- info: 'NFLProof.hist_prefix_dev' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.hist_prefix_dev

/-- info: 'NFLProof.hist_sum' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.hist_sum

/-- info: 'NFLProof.keyProbe_iff' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.keyProbe_iff

/-- info: 'NFLProof.klUnif_eq_klFin' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.klUnif_eq_klFin

/-- info: 'NFLProof.ksUnif_left_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ksUnif_left_le

/-- info: 'NFLProof.ksUnif_right_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ksUnif_right_le

/-- info: 'NFLProof.ksUnif_term_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ksUnif_term_le

/-- info: 'NFLProof.ks_chain' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.ks_chain

/-- info: 'NFLProof.le_clog_of_half_pow_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.le_clog_of_half_pow_le

/-- info: 'NFLProof.le_iff_comp' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.le_iff_comp

/-- info: 'NFLProof.le_iff_lt_rankLE' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.le_iff_lt_rankLE

/-- info: 'NFLProof.n_pos_of_fin' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.n_pos_of_fin

/-- info: 'NFLProof.pit_strictMono' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.pit_strictMono

/-- info: 'NFLProof.psiP_one' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.psiP_one

/-- info: 'NFLProof.rankLE_eq_n' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.rankLE_eq_n

/-- info: 'NFLProof.rankLE_eq_zero' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.rankLE_eq_zero

/-- info: 'NFLProof.rankLE_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.rankLE_le

/-- info: 'NFLProof.rank_scale' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.rank_scale

/-- info: 'NFLProof.span_mvt' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.span_mvt

/-- info: 'NFLProof.tvUnif_eq_tvFin' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.tvUnif_eq_tvFin

/-- info: 'NFLProof.tvUnif_nonneg' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.tvUnif_nonneg

/-- info: 'NFLProof.unifCDF_key' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.unifCDF_key

/-- info: 'NFLProof.unifCDF_le_one' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.unifCDF_le_one

/-- info: 'NFLProof.unifCDF_mono' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.unifCDF_mono

/-- info: 'NFLProof.unifCDF_nonneg' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.unifCDF_nonneg

/-- info: 'NFLProof.unifCDF_of_mem' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.unifCDF_of_mem

/-- info: 'NFLProof.unifCDF_of_nonpos' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.unifCDF_of_nonpos

/-- info: 'NFLProof.unifCDF_of_one_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.unifCDF_of_one_le

/-- info: 'NFLProof.unifCDF_sub_le' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.unifCDF_sub_le

/-- info: 'NFLProof.unif_prefix' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms NFLProof.unif_prefix
/-! ## Headline theorems, printed in the build log -/

#print axioms NFLProof.rank_comp_strictMono
#print axioms NFLProof.ranks_preserved_iff
#print axioms NFLProof.ks_sandwich
#print axioms NFLProof.slot_err
#print axioms NFLProof.pit_ksUnif_eq
#print axioms NFLProof.pit_endpoint_err
#print axioms NFLProof.window_cost_mono
#print axioms NFLProof.gallop_spec
#print axioms NFLProof.lookup_cost_of_ks
#print axioms NFLProof.conflict_le
#print axioms NFLProof.conflict_le_ks
#print axioms NFLProof.ksBinned_le_tv
#print axioms NFLProof.ks_le_tv_add
#print axioms NFLProof.even_sample_dev
#print axioms NFLProof.pinsker_unif
#print axioms NFLProof.certified_error_even_proved
#print axioms NFLProof.certified_slot_err_even_proved
#print axioms NFLProof.certified_error_prob_proved
#print axioms NFLProof.gate_sound
#print axioms NFLProof.gate_false_positive
#print axioms NFLProof.piece_composite
#print axioms NFLProof.two_run
#print axioms NFLProof.cv_floor
#print axioms NFLProof.smooth_floor
