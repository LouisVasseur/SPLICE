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
//   trace_<name>   chunk timer around any name (warm-up detection runs only); w_<name>: untimed warm-up in bulk_load
//   splice         SpliceIndex (splice.h): G + T + V; spl_t (T), spl_gt (G + T), spl_tv (T + V),
//                  splice_eq (equi-rank knots), spl_fixk (fixed granularity) its attribution ladder; 'splice' is also a base for every layer (gate_splice, pw512_splice, nfl_splice, trace_splice, w_splice)
//   sp_<parts>_<base>  SPLICE's parts on one of GRE's indexes (splice_on.h), each kept only where it pays: parts t
//                  (T), gt (G + T), gtv (G + T + V), tv (T + V), g (G), v (V), gv (G + V); e.g. sp_gtv_alex
//   hl4, hl8       HashLineIndex (hashline.h): hashed keys in buckets of 4 or 8 slots, no model, no top, no key order
//   hf4            hl4 with one multiply in place of the mix (hashline.h)
//   ab4            AdaptiveLineIndex (adaptiveline.h): rl4 with the directory refined where lookups read too many lines
//   aw4            ab4 with put (in place within two cache lines, else an overflow table that get asks only on a miss)
//   ph4            PartitionHashIndex (partitionhash.h): T, then G: a directory on the raw key, a hash inside each leaf
//   spgm16, 64, 256  SortedBoundIndex (readonly.h): static PGM over one sorted array, at those error bounds
//   gr4            RuleLineIndex (ruleline.h): ab4 over a compact arithmetic coordinate, G as one rule from $RULE_FILE; experimental
//   gf4            GapLineIndex (gapline.h): gr4 with the $GAP_K widest interior gaps shrunk to the typical gap; experimental
//   gw4            aw4 over gp4's coordinate: put, update and remove with G's period rule ($RULE_FILE), no rebuilds
//   gp4            FastRuleLineIndex (fastrule.h): gr4's coordinate (up to a constant and rounding) in fewer instructions (block
//                  table, exact reciprocal); experimental
//   gb4            AdaptiveLineIndex over BitCoord (gatedg.h): ab4 with the bulk's inner constant bits dropped (PEXT), sg4's bits
//                  candidate alone, to time it; experimental
//   sg4            GatedGIndex (gatedg.h): ab4 with G chosen at build time among identity, gp4's rule (from $RULE_FILE, or
//                  discovered on the bulk) and bit compaction
//   name@VAR=v:..  any name above, with those environment variables set first (':' or ',' between them; a value cannot hold
//                  either, nor '@'). They stay set for the rest of the process, so later names see them too. An entry without
//                  '=' or without a variable name, a second '@', or nothing before or after the '@' throws std::invalid_argument
//   rl4            RadixLineIndex (radixline.h): G, a directory of equal-width cells for T, spare slots for V; keeps key order
//   gate_<base>    the learned gate (gate.h, gate_model.h)
//   nflsw_<base>   an NFL-style switch: pw512 or raw (gate.h)
// Bases: lipp btree alex pgm, and artunsync where GRE's registry has it (Linux: competitor.h includes artunsync.h
// before this header; detected through the include guard of ART/Tree.h, or set GRET_HAS_ARTUNSYNC to 0 or 1).
// Returns nullptr for every other name, and for keys other than uint64_t.
#pragma once
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gret/gate.h"
#include "gret/transform_index.h"
#include "gret/vpoints.h"
#include "gret/exceptions.h"
#include "gret/splice_on.h"
#include "gret/warmup.h"
#include "gret/splice.h"
#include "gret/hashline.h"
#include "gret/radixline.h"
#include "gret/adaptiveline.h"
#include "gret/partitionhash.h"
#include "gret/ruleline.h"
#include "gret/gapline.h"
#include "gret/fastrule.h"
#include "gret/gatedg.h"
#include "gret/readonly.h"
#include "gret/sota/updating.h"  // alexu, btreeu: GRE's ALEX and B+tree wrappers with update implemented (ours)
// Baselines from the literature (include/gret/sota/, docs/sota/), only in a build that defines GRET_SOTA and puts their authors'
// code on the include path: RadixSpline/include, CHT/include, the parent of PLEX, and for the hash tables unordered_dense/include,
// robin-map/include and parallel-hashmap. rmi is a reimplementation and says so in its name line.
#ifdef GRET_SOTA
#include "gret/sota/radixspline.h"
#include "gret/sota/cht.h"
#include "gret/sota/plex.h"
#include "gret/sota/rmi.h"
#include "gret/sota/hashtable.h"
#include "gret/sota/fast.h"  // FAST, a third-party implementation (Leis, via TLI); x86-64 AVX2 only
#include "gret/sota/searches.h"  // bsearch, rbs<bits> (SOSD's class, by path), rbst<bits>, interp, tip (ours; they say so): needs the parent of SOSD
#include "gret/sota/csv.h"       // CSV on GRE's LIPP and ALEX: a reimplementation from the paper, no authors' code exists
#include "gret/sota/pgmpp.h"     // PGM++: the authors' builder, our lookup: needs bench_search on the include path
// Each of the next four compiles its authors' sources into this unit and needs their directories on the include path, so each
// has a define of its own (docs/sota/<name>.md says which paths). One translation unit only.
#ifdef GRET_SOTA_AFLI
#include "gret/sota/afli.h"   // AFLI, the NFL authors' index (GPL-3.0), and nfl_afli, a hybrid (our flow, their index)
#endif
#ifdef GRET_SOTA_SALI
#include "gret/sota/sali.h"   // SALI, the authors' code, one thread
#endif
#ifdef GRET_SOTA_DYTIS
#include "gret/sota/dytis.h"  // DyTIS, the authors' code; one index alive at a time
#endif
#ifdef GRET_SOTA_BLI
#include "gret/sota/bli.h"    // BLI, the authors' code; x86-64 AVX2
#endif
#ifdef GRET_SOTA_DILI  // also needs DILI/src on the include path; the header compiles the authors' sources into this unit
#include "gret/sota/dili.h"
#endif
#endif

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
    using SI = SpliceIndex<K, P>;  // SPLICE and its attribution ladder (splice.h)
    using SC = typename SI::Config;
    // base@VAR=value:VAR=value: set those variables, then build base; one gated batch can then mix settings. ':' separates them
    // (',' also works outside GRE, whose --index may split on commas). setenv: they persist for the process, and every later
    // name sees them unless it sets them again. The whole name is checked before anything is set
    if (const size_t at = name.find('@'); at != std::string::npos) {
      const std::string base = name.substr(0, at), rest = name.substr(at + 1);
      auto bad = [&](const std::string& why) { throw std::invalid_argument("gret::make_index: '" + name + "': " + why); };
      if (base.empty()) bad("no index name before '@'");
      if (rest.find('@') != std::string::npos) bad("a second '@'");
      std::vector<std::pair<std::string, std::string>> vars;
      for (size_t p = 0;;) {  // every entry, empty ones included: "a=1:" and "a=1::b=2" are malformed
        const size_t e = rest.find_first_of(":,", p);
        const std::string kv = rest.substr(p, e == std::string::npos ? std::string::npos : e - p);
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) bad("'" + kv + "' is not VAR=value");
        if (eq == 0) bad("'" + kv + "' has no variable name");
        vars.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        if (e == std::string::npos) break;
        p = e + 1;
      }
      for (const auto& [var, value] : vars)
        if (setenv(var.c_str(), value.c_str(), 1) != 0) bad("cannot set " + var);
      return make_index<K, P>(base);
    }
    if (name == "hl4") return new HashLineIndex<K, P, 4>(name);
    if (name == "hl8") return new HashLineIndex<K, P, 8>(name);
    if (name == "rl4") return new RadixLineIndex<K, P, 4>(name);
    if (name == "hf4") return new HashLineIndex<K, P, 4, mul64, true>(name);
    if (name == "ab4") return new AdaptiveLineIndex<K, P, 4>(name);
    if (name == "aw4") return new AdaptiveLineIndex<K, P, 4, IdentityCoord, true>(name);  // ab4 that takes writes: put in place, else an overflow table
    if (name == "ph4") return new PartitionHashIndex<K, P, 4>(name);
    if (name == "gr4") return new RuleLineIndex<K, P, 4>(name);
    if (name == "gp4") return new FastRuleLineIndex<K, P, 4>(name);
    if (name == "gw4") return new AdaptiveLineIndex<K, P, 4, FastPeriodRule, true>(name);  // gp4 that takes writes: aw4 over G's period rule (no rebuilds)
    if (name == "gb4") return new AdaptiveLineIndex<K, P, 4, BitCoord>(name);
    if (name == "sg4") return new GatedGIndex<K, P, 4>(name);
    if (name == "gf4") return new GapLineIndex<K, P, 4>(name);
    if (name.rfind("gf4k", 0) == 0) {  // gf4k<K>: gf4 with GAP_K = K, so one batch can hold several K
      setenv("GAP_K", name.c_str() + 4, 1);
      return new GapLineIndex<K, P, 4>(name);
    }
    if (name == "spgm16") return new SortedBoundIndex<K, P, PgmBound<K, 16>>(name);
    if (name == "spgm64") return new SortedBoundIndex<K, P, PgmBound<K, 64>>(name);
    if (name == "spgm256") return new SortedBoundIndex<K, P, PgmBound<K, 256>>(name);
    if (indexInterface<K, P>* i = sota::make_updating<K, P>(name)) return i;  // alexu, btreeu
#ifdef GRET_SOTA
    if (indexInterface<K, P>* i = sota::make_radixspline<K, P>(name)) return i;  // rs<max_error>r<radix_bits>
    if (indexInterface<K, P>* i = sota::make_cht<K, P>(name)) return i;          // cht<bins>e<max_error>[std], ccht<bins>e<max_error>
    if (indexInterface<K, P>* i = make_plex<K, P>(name)) return i;               // plex<max_error>
    if (indexInterface<K, P>* i = make_rmi<K, P>(name)) return i;                // rmi<L>[lin|spl|cub|rob|rdx|rt18|rt22][k]: a reimplementation
    if (indexInterface<K, P>* i = sota::make_fast<K, P>(name)) return i;         // fast: third-party implementation (Leis, via TLI)
    if (name.compare(0, 3, "ht_") == 0) return sota::make_hashtable<K, P>(name);  // ht_std, ht_ankerl, ht_robin, ht_robinpg, ht_phmap
    if (indexInterface<K, P>* i = sota::make_searches<K, P>(name)) return i;  // bsearch | bsearchstd | rbs<b> | rbsstd<b> | rbst<b> | interp[<k>][e] | tip[<g>][a]
    if (indexInterface<K, P>* i = make_csv<K, P>(name)) return i;             // csv_lipp[<PP>], csv_alex[<PP>]: a reimplementation of CSV
    if (indexInterface<K, P>* i = sota::make_pgmpp<K, P>(name)) return i;     // pgmpp<eps>[i<eps_i>][d<delta>][n]: authors' builder, our lookup
#ifdef GRET_SOTA_AFLI
    if (indexInterface<K, P>* i = make_afli<K, P>(name)) return i;   // afli[<B>][_cut], nfl_afli[_cut] (a hybrid): before the nfl_<base> prefix below
#endif
#ifdef GRET_SOTA_SALI
    if (indexInterface<K, P>* i = make_sali<K, P>(name)) return i;   // sali, saliu
#endif
#ifdef GRET_SOTA_DYTIS
    if (indexInterface<K, P>* i = make_dytis<K, P>(name)) return i;  // dytis, dytiss, dytisu
#endif
#ifdef GRET_SOTA_BLI
    if (indexInterface<K, P>* i = make_bli<K, P>(name)) return i;    // bli[<L>][s<S>][f<PP>][e<E>][u]
#endif
#ifdef GRET_SOTA_DILI
    if (indexInterface<K, P>* i = make_dili<K, P>(name)) return i;  // dili, dili<B>[r<PP>]: the DILI authors' code, guarded (sota/dili.h)
#endif
#endif
    if (name == "splice") return new SI(name, SC{true, true, SI::kPla, true});
    if (name == "spl_t") return new SI(name, SC{false, false, SI::kPla, true});
    if (name == "spl_gt") return new SI(name, SC{true, false, SI::kPla, true});
    if (name == "spl_tv") return new SI(name, SC{false, true, SI::kPla, true});
    if (name == "splice_eq") return new SI(name, SC{true, true, SI::kEqui, true});
    if (name == "spl_fixk") return new SI(name, SC{true, true, SI::kPla, false});
    if (name == "splice_gi") return new SI(name, SC{true, true, SI::kPla, true, true});  // + interior G (spill tails)
    // v2: the top chosen by count (locator or RankTop), prices from the machine's probed access costs, coarse-to-fine
    // granularity search; its attribution ladder
    if (name == "splice2") return new SI(name, SC{true, true, SI::kPla, true, false, true});
    if (name == "splice2_gi") return new SI(name, SC{true, true, SI::kPla, true, true, true});
    if (name == "s2_t") return new SI(name, SC{false, false, SI::kPla, true, false, true});
    if (name == "s2_gt") return new SI(name, SC{true, false, SI::kPla, true, false, true});
    if (name == "s2_tv") return new SI(name, SC{false, true, SI::kPla, true, false, true});
    if (name == "s2_eq") return new SI(name, SC{true, true, SI::kEqui, true, false, true});
    if (name == "s2_fixk") return new SI(name, SC{true, true, SI::kPla, false, false, true});
    // v3 (timed): v2's candidates, T's granularity and the top's form chosen by timing a sample of lookups
    if (name == "splice3") return new SI(name, SC{true, true, SI::kPla, true, false, true, true});
    if (name == "s3_t") return new SI(name, SC{false, false, SI::kPla, true, false, true, true});
    if (name == "s3_gt") return new SI(name, SC{true, false, SI::kPla, true, false, true, true});
    if (name == "s3_tv") return new SI(name, SC{false, true, SI::kPla, true, false, true, true});
    if (name == "s3_eq") return new SI(name, SC{true, true, SI::kEqui, true, false, true, true});
    if (name == "s3_fixk") return new SI(name, SC{true, true, SI::kPla, false, false, true, true});
    {  // trace_ / w_ / trace_w_ on a raw base: held by value and inlined, so the layers add no call to the lookup
      auto raw = [&](const std::string& n, auto tag) -> indexInterface<K, P>* {
        using B = typename decltype(tag)::type;
        if (n == "trace_" ) return new TraceIndex<K, P, B>(name);
        if (n == "w_") return new WarmIndex<K, P, B>(name);
        if (n == "trace_w_") return new TraceIndex<K, P, WarmIndex<K, P, B>>(name);
        return nullptr;
      };
      for (const char* pre : {"trace_w_", "trace_", "w_"}) {
        const std::string pr = pre;
        if (name.compare(0, pr.size(), pr) != 0) continue;
        const std::string b = name.substr(pr.size());
        if (b == "lipp") return raw(pr, std::common_type<LIPPInterface<uint64_t, P>>{});
        if (b == "btree") return raw(pr, std::common_type<BTreeInterface<uint64_t, P>>{});
        if (b == "alex") return raw(pr, std::common_type<alexInterface<uint64_t, P>>{});
        if (b == "pgm") return raw(pr, std::common_type<pgmInterface<uint64_t, P>>{});
        if (b == "splice") return raw(pr, std::common_type<SpliceIndex<K, P>>{});
        break;
      }
    }
    if (name.compare(0, 6, "trace_") == 0 || name.compare(0, 2, "w_") == 0) {  // measurement layers, outermost
      const bool tr = name[0] == 't';
      indexInterface<K, P>* inner = make_index<K, P>(name.substr(tr ? 6 : 2));
      if (!inner) return nullptr;
      if (tr) return new TraceIndex<K, P>(name, inner);
      return new WarmIndex<K, P>(name, inner);
    }
    if (name.compare(0, 3, "sp_") == 0) {  // SPLICE's parts on a GRE index: sp_<parts>_<base>
      const size_t u = name.find('_', 3);
      if (u == std::string::npos) return nullptr;
      const std::string parts = name.substr(3, u - 3), b = name.substr(u + 1);
      if (parts.empty() || parts.find_first_not_of("gtv") != std::string::npos) return nullptr;
      auto has = [&](char c) { return parts.find(c) != std::string::npos; };
      auto make = [&](auto tag, bool ties) -> indexInterface<K, P>* {
        using B = typename decltype(tag)::type;
        return new SpliceOn<K, P, B>(name, b, typename SpliceOn<K, P, B>::Config{has('g'), has('t'), has('v'), ties});
      };
      if (b == "lipp") return make(std::common_type<LIPPInterface<uint64_t, P>>{}, LDBL_MANT_DIG < 64);
      if (b == "btree") return make(std::common_type<BTreeInterface<uint64_t, P>>{}, false);
      if (b == "alex") return make(std::common_type<alexInterface<uint64_t, P>>{}, false);
      if (b == "pgm") return make(std::common_type<pgmInterface<uint64_t, P>>{}, false);
      return nullptr;
    }
    if (name.compare(0, 2, "g_") == 0) {  // G first: the exception list wraps whatever the rest of the name builds
      const std::string rest = name.substr(2);  // g_<base>: the base by value, called directly
      if (rest == "lipp") return new ExceptionIndex<K, P, LIPPInterface<uint64_t, P>>(name);
      if (rest == "btree") return new ExceptionIndex<K, P, BTreeInterface<uint64_t, P>>(name);
      if (rest == "alex") return new ExceptionIndex<K, P, alexInterface<uint64_t, P>>(name);
      if (rest == "pgm") return new ExceptionIndex<K, P, pgmInterface<uint64_t, P>>(name);
      indexInterface<K, P>* inner = make_index<K, P>(rest);
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
    if (base == "splice") return make_on<K, P, SpliceIndex<K, P>>(kind, base, RawLimits{});
    if (base == "alex") return make_on<K, P, alexInterface<uint64_t, P>>(kind, base, RawLimits{false, true, true});
    if (base == "pgm") return make_on<K, P, pgmInterface<uint64_t, P>>(kind, base, RawLimits{false, true, false});
#if GRET_HAS_ARTUNSYNC
    if (base == "artunsync") return make_on<K, P, ARTUnsynchronizedInterface<uint64_t, P>>(kind, base, RawLimits{});
#endif
    return nullptr;
  }
}

}  // namespace gret
