// SPDX-License-Identifier: MIT
// make_index<K, P>(name): the gret index names, for GRE's get_index() (scripts/add_transforms.py inserts the call
// after GRE's own names and includes this header after GRE's competitor headers, which it needs and cannot include).
//
//   pw<K>_<base>   PwlCdf with K knots, K in [2, 65536] written without leading zeros (pw512_lipp, pw4096_btree)
//   pw_<base>      PwlCdf with K = $TRANSFORM_K (default 512); it reports itself as pw<K>_<base>. Only these names
//                  read $TRANSFORM_K: pw<K>_, gate_ (pw512, pw4096) and nflsw_ (pw512) have fixed K
//   nf_<base>      MolFlow loaded from $TRANSFORM_FLOW
//   nfl_<base>     NFL's own flow (NflFlow), its weights loaded from $TRANSFORM_NFL
//   vp_<base>      virtual points (VirtualIndex) in front of the base; <t>_vp_<base> is T then V (e.g. pw512_vp_alex)
//   g_<name>       G: outlier keys peeled into an exception list (ExceptionIndex), then any other name on the bulk
//                  (g_alex, g_pw512_alex, g_vp_lipp, g_pw512_vp_alex)
//   gate_<base>    the learned gate (gate.h, gate_model.h)
//   nflsw_<base>   an NFL-style switch: pw512 or raw (gate.h)
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
#include "gret/vpoints.h"
#include "gret/exceptions.h"

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
  if (kind == "nfl") return new TransformIndex<K, P, Base, NflFlow>("nfl_" + base);
  if (kind == "gate") return new GateIndex<K, P, Base>(base, GateMode::learned, raw);
  if (kind == "nflsw") return new GateIndex<K, P, Base>(base, GateMode::nflsw, raw);
  if (const unsigned k = pw_knots(kind))
    return new TransformIndex<K, P, Base, PwlCdf>("pw" + std::to_string(k) + "_" + base, TransformSpec{k, ""});
  return nullptr;
}

}  // namespace registry_detail

template <class K, class P>
indexInterface<K, P>* raw_base(const std::string& b) {  // GRE's own index by name, for g_<base>
  if (b == "lipp") return new LIPPInterface<uint64_t, P>;
  if (b == "btree") return new BTreeInterface<uint64_t, P>;
  if (b == "alex") return new alexInterface<uint64_t, P>;
  if (b == "pgm") return new pgmInterface<uint64_t, P>;
#if GRET_HAS_ARTUNSYNC
  if (b == "artunsync") return new ARTUnsynchronizedInterface<uint64_t, P>;
#endif
  return nullptr;
}

template <class K, class P>
indexInterface<K, P>* make_index(const std::string& name) {
  if constexpr (!std::is_same<K, uint64_t>::value) {
    (void)name;
    return nullptr;
  } else {
    using namespace registry_detail;
    if (name.compare(0, 2, "g_") == 0) {  // G first: the exception list wraps whatever the rest of the name builds
      indexInterface<K, P>* inner = make_index<K, P>(name.substr(2));
      return inner ? new ExceptionIndex<K, P>(name, inner) : nullptr;
    }
    const size_t us = name.find('_');
    if (us == std::string::npos) return raw_base<K, P>(name);
    const std::string kind = name.substr(0, us), base = name.substr(us + 1);
    // the gate's raw limits (gate.h): LIPP's models run in long double, which is double on macOS arm64 (an 80-bit
    // long double holds every uint64 exactly); ALEX and PGM use 2^64 - 1 as a sentinel; ALEX predicts slots as int
    static_assert(LDBL_MANT_DIG == DBL_MANT_DIG || LDBL_MANT_DIG >= 64, "long double: neither double nor >= 64 bits");
    if (kind == "vp") {  // V alone
      if (base == "lipp") return new VirtualIndex<K, P, LIPPInterface<uint64_t, P>>(name);
      if (base == "btree") return new VirtualIndex<K, P, BTreeInterface<uint64_t, P>>(name);
      if (base == "alex") return new VirtualIndex<K, P, alexInterface<uint64_t, P>>(name);
      if (base == "pgm") return new VirtualIndex<K, P, pgmInterface<uint64_t, P>>(name);
      return nullptr;
    }
    if (base.compare(0, 3, "vp_") == 0) {  // T then V: the transform wraps the virtual-point index (raw is never vetoed:
      const std::string b = base.substr(3);  // the transformed keys stay below 2^52)
      if (b == "lipp") return make_on<K, P, VirtualIndex<K, P, LIPPInterface<uint64_t, P>>>(kind, base, RawLimits{});
      if (b == "btree") return make_on<K, P, VirtualIndex<K, P, BTreeInterface<uint64_t, P>>>(kind, base, RawLimits{});
      if (b == "alex") return make_on<K, P, VirtualIndex<K, P, alexInterface<uint64_t, P>>>(kind, base, RawLimits{});
      if (b == "pgm") return make_on<K, P, VirtualIndex<K, P, pgmInterface<uint64_t, P>>>(kind, base, RawLimits{});
      return nullptr;
    }
    if (base == "lipp")
      return make_on<K, P, LIPPInterface<uint64_t, P>>(kind, base, RawLimits{LDBL_MANT_DIG < 64, false, false});
    if (base == "btree") return make_on<K, P, BTreeInterface<uint64_t, P>>(kind, base, RawLimits{});
    if (base == "alex") return make_on<K, P, alexInterface<uint64_t, P>>(kind, base, RawLimits{false, true, true});
    if (base == "pgm") return make_on<K, P, pgmInterface<uint64_t, P>>(kind, base, RawLimits{false, true, false});
#if GRET_HAS_ARTUNSYNC
    if (base == "artunsync") return make_on<K, P, ARTUnsynchronizedInterface<uint64_t, P>>(kind, base, RawLimits{});
#endif
    return nullptr;
  }
}

}  // namespace gret
