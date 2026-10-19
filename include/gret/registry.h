// SPDX-License-Identifier: MIT
// make_index<K, P>(name): the gret index names, for GRE's get_index() (scripts/add_transforms.py inserts the call
// after GRE's own names and includes this header after GRE's competitor headers, which it needs and cannot include).
//
//   pw<K>_<base>   PwlCdf with K knots, K in [2, 65536] written without leading zeros (pw512_lipp, pw4096_btree)
//   pw_<base>      PwlCdf with K = $TRANSFORM_K (default 512); it reports itself as pw<K>_<base>
//   nf_<base>      MolFlow loaded from $TRANSFORM_FLOW
//   gate_<base>    the learned gate (gate.h, gate_model.h)
//   nflsw_<base>   NFL's switch: pw512 or raw
// Bases: lipp btree alex pgm, and artunsync where GRE's registry has it (Linux: competitor.h includes artunsync.h
// before this header; detected through the include guard of ART/Tree.h, or set GRET_HAS_ARTUNSYNC to 0 or 1).
// Returns nullptr for every other name, and for keys other than uint64_t.
#pragma once
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>

#include "gret/gate.h"
#include "gret/transform_index.h"

#ifndef GRET_HAS_ARTUNSYNC
#if defined(ARTVERSION1_TREE_H)  // src/competitor/artsync/src/ART/Tree.h, included by artunsync.h only
#define GRET_HAS_ARTUNSYNC 1
#else
#define GRET_HAS_ARTUNSYNC 0
#endif
#endif

namespace gret {

// "pw512" -> 512, "pw" -> $TRANSFORM_K (default 512); 0 for anything else.
inline unsigned pw_knots(const std::string& kind) {
  if (kind.compare(0, 2, "pw") != 0) return 0;
  std::string d = kind.substr(2);
  const bool env = d.empty();
  if (env) {
    const char* e = std::getenv("TRANSFORM_K");
    if (!e || !*e) return 512;
    d = e;
  }
  unsigned k = 0;
  if (d.size() <= 5 && d[0] != '0' && d.find_first_not_of("0123456789") == std::string::npos) k = unsigned(std::stoul(d));
  if (k >= 2 && k <= 65536) return k;
  if (env) die("TRANSFORM_K must be an integer in [2, 65536], not '" + d + "'");
  return 0;
}

namespace registry_detail {

template <class K, class P, class Base>
indexInterface<K, P>* make_on(const std::string& kind, const std::string& base, RawLimits raw) {
  if (kind == "nf") return new TransformIndex<K, P, Base, MolFlow>("nf_" + base);
  if (kind == "gate") return new GateIndex<K, P, Base>(base, GateMode::learned, raw);
  if (kind == "nflsw") return new GateIndex<K, P, Base>(base, GateMode::nflsw, raw);
  if (const unsigned k = pw_knots(kind))
    return new TransformIndex<K, P, Base, PwlCdf>("pw" + std::to_string(k) + "_" + base, TransformSpec{k, ""});
  return nullptr;
}

}  // namespace registry_detail

template <class K, class P>
indexInterface<K, P>* make_index(const std::string& name) {
  if constexpr (!std::is_same<K, uint64_t>::value) {
    (void)name;
    return nullptr;
  } else {
    using namespace registry_detail;
    const size_t us = name.find('_');
    if (us == std::string::npos) return nullptr;
    const std::string kind = name.substr(0, us), base = name.substr(us + 1);
    // the gate's raw limits (gate.h): LIPP's models run in long double, the 64-bit double on macOS arm64; ALEX and PGM
    // use 2^64 - 1 as a sentinel
    if (base == "lipp") return make_on<K, P, LIPPInterface<uint64_t, P>>(kind, base, RawLimits{LDBL_MANT_DIG < 64, false});
    if (base == "btree") return make_on<K, P, BTreeInterface<uint64_t, P>>(kind, base, RawLimits{});
    if (base == "alex") return make_on<K, P, alexInterface<uint64_t, P>>(kind, base, RawLimits{false, true});
    if (base == "pgm") return make_on<K, P, pgmInterface<uint64_t, P>>(kind, base, RawLimits{false, true});
#if GRET_HAS_ARTUNSYNC
    if (base == "artunsync") return make_on<K, P, ARTUnsynchronizedInterface<uint64_t, P>>(kind, base, RawLimits{});
#endif
    return nullptr;
  }
}

}  // namespace gret
