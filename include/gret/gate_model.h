// SPDX-License-Identifier: MIT
// The gate's decision model, included by gate.h after it defines Variant, GateLeaf and GateTree (do not include it
// directly). gate/train_gate.py regenerates this whole file from the Mac labels; until it has run, this is the
// documented PLACEHOLDER: untrained, every base chooses raw (pw512 where raw is vetoed).
//
// Format, for the generator (C++17, namespace gret, both as inline constexpr):
//   kGateModelSource  one string: what trained the model (labels' results dir and date, validation), or "placeholder".
//   kGateTrees[]      one GateTree per base; the first entry whose base matches wins, "*" matches any base (put it
//                     last, and always emit one).
//     {"base", {f_root, f_true, f_false}, {thr_root, thr_true, thr_false},
//      {{pick, fallback}, {pick, fallback}, {pick, fallback}, {pick, fallback}}}
//     f_*: feature names from GateFeatures::kNames (checked at bulk_load), or nullptr for a test that always holds;
//     a test is feature <= thr; thresholds written with %.17g. Leaves: (root, child) = (holds, holds),
//     (holds, fails), (fails, holds), (fails, fails). A one-feature threshold rule is {f, nullptr, nullptr} with its
//     answers in leaves 0 (f <= thr) and 2 (f > thr). pick/fallback: Variant::raw, pw512, pw4096 or nf; fallback is
//     used when pick cannot be built (nf without a flow file, raw vetoed), so it should not be nf.
#pragma once

namespace gret {

inline constexpr const char* kGateModelSource = "placeholder: untrained, always raw (pw512 where raw is vetoed)";

inline constexpr GateTree kGateTrees[] = {
    {"*", {nullptr, nullptr, nullptr}, {0, 0, 0},
     {{Variant::raw, Variant::pw512}, {Variant::raw, Variant::pw512}, {Variant::raw, Variant::pw512},
      {Variant::raw, Variant::pw512}}},
};

}  // namespace gret
