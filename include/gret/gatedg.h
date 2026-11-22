// SPDX-License-Identifier: MIT
// GatedGIndex (sg4): ab4 with G chosen at build time. Every candidate G that the keys admit is built in full, at the same byte
// budget, and the one with the lowest predicted lookup time is kept; identity (ab4 itself) is always a candidate.
//
//   bulk        the keys AdaptiveLineIndex indexes: all of them but the ends peel_outliers takes ($TRANSFORM_G_E, read as there)
//   candidates  identity                       always
//               the period rule (fastrule.h)   $RULE_FILE's rule; without $RULE_FILE, the rule discover_rule (discover.h) finds
//                                              on the bulk (the keys the index stores: a far key cannot pull its first block below
//                                              them, though it can change which keys are peeled),
//                                              handed over in memory ($GATE_DISCOVER=0: no discovery; $GATE_RULE_OUT=f: write it
//                                              to f.<pid>)
//               bit-field compaction            when some bits never vary between the lowest and the highest varying bit of the
//                                              bulk's keys: those bits are dropped (PEXT), which keeps stored keys in order
//   prediction  ns = base + c + f base, base = A + B (lines - 1) + C (levels - 1) from the candidate's own build (exact counts).
//               A, B, C are the selector's weights ($GATE_W, default the frozen ones). c and f are what G adds: work before the
//               directory's first read lets fewer lookups overlap, so part of it grows with the lookup's own time. The rule's,
//               $GATE_RULE_NS = 10.24 and $GATE_RULE_F = 0.104, are fitted on gp4 over layouts a rule leaves unchanged (S = 0, so
//               y = x): fb +16.4 / +27.3 ns (one block / 22), covid +13.6 / +18.0 (Jed, GRE, 2026-11-03). The bits', $GATE_BITS_NS
//               = 3 and $GATE_BITS_F = 0, are placeholders (gb4 times BitCoord alone). The 'gate:' line on stderr prints every one of
//               these values, then each candidate's prediction (or why it was not built) and the choice.
//   $GATE_FORCE=id|rule|bits keeps that candidate whatever the prediction (tests). Any other value throws, and so does a forced
//               candidate that was not built.
// The losers are freed as soon as a better candidate is built, so at most two exist at once, and once identity is built chosen_
// always points at the one kept so far (an index whose bulk_load threw is not for use, as anywhere in GRE). get jumps into the kept candidate's own compiled get(): three dependent loads (chosen_, its vtable pointer, the
// slot) and an indirect jump that always goes to the same place, then that index's lookup exactly as it is compiled for that index
// (a copy inlined here is compiled anew: under GCC 11 that copy kept at identity was 2-8% slower than ab4 itself). Measured in GRE
// (Jed, 2026-11-04, hardware counters over the timed lookups): +3 instructions and +1 branch per lookup, no added cache, TLB or
// branch misses, yet 3-7% slower where it keeps identity and 6-8% behind gp4 on genome. GRE's lookup loop is bounded by how many
// lookups overlap their misses, so every instruction a lookup adds costs time. Candidates on the heap instead of inside the gate,
// or the candidate's address taken without a load (one more branch), were no faster. scan dispatches on one or two predictable
// branches.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>
#if defined(__BMI2__)
#include <immintrin.h>
#endif

#include "gret/adaptiveline.h"
#include "gret/discover.h"
#include "gret/exceptions.h"
#include "gret/fastrule.h"

namespace gret {

class BitCoord {  // drop the bits that never vary between the lowest and the highest varying bit of the stored keys
 public:
  static constexpr bool kIdentity = false;
  template <class Pair>
  static uint64_t varying(const Pair* d, size_t m) {
    uint64_t a = ~uint64_t(0), o = 0;
    for (size_t r = 0; r < m; ++r) a &= uint64_t(d[r].first), o |= uint64_t(d[r].first);
    return a ^ o;
  }
  static bool has_inner(uint64_t v) {  // a zero bit strictly between the lowest and the highest set bit
    if (!v) return false;
    const uint64_t low = v & (0 - v);
    const unsigned hi = 63u - unsigned(__builtin_clzll(v));
    const uint64_t full = (hi == 63 ? ~uint64_t(0) : (uint64_t(1) << (hi + 1)) - 1) & ~(low - 1);  // every bit from low to hi
    return v != full;
  }
  template <class Pair>
  void load(const Pair* d, size_t m) { mask_ = varying(d, m); }
  uint64_t operator()(uint64_t x) const {
#if defined(__BMI2__)
    return _pext_u64(x, mask_);
#else
    uint64_t out = 0, bit = 1;
    for (uint64_t mk = mask_; mk; mk &= mk - 1, bit <<= 1)
      if (x & mk & (0 - mk)) out |= bit;
    return out;
#endif
  }
  size_t bytes() const { return sizeof(mask_); }

 private:
  uint64_t mask_ = ~uint64_t(0);
};

template <class K, class P, unsigned B>
class GatedGIndex final : public indexInterface<K, P> {
  using Id = AdaptiveLineIndex<K, P, B>;
  using Rule = AdaptiveLineIndex<K, P, B, FastPeriodRule>;
  using Bits = AdaptiveLineIndex<K, P, B, BitCoord>;
  static double env_d(const char* k, double def) {
    const char* e = std::getenv(k);
    return e && *e ? std::strtod(e, nullptr) : def;
  }

 public:
  explicit GatedGIndex(std::string name) : name_(std::move(name)) {}
  GatedGIndex(const GatedGIndex&) = delete;  // chosen_ points into this object's own candidates
  GatedGIndex& operator=(const GatedGIndex&) = delete;
  void init(Param* = nullptr) override {}

  void bulk_load(std::pair<K, P>* kv, size_t n, Param* p = nullptr) override {
    double A = 34.977, Bw = 46.575, C = 44.657;
    if (const char* w = std::getenv("GATE_W"); w && *w) std::sscanf(w, "%lf %lf %lf", &A, &Bw, &C);
    const double rule_ns = env_d("GATE_RULE_NS", 10.24), rule_f = env_d("GATE_RULE_F", 0.104), bits_ns = env_d("GATE_BITS_NS", 3),
                 bits_f = env_d("GATE_BITS_F", 0);
    const bool discover = env_d("GATE_DISCOVER", 1) != 0;
    const char* fe = std::getenv("GATE_FORCE");
    const std::string force = fe ? fe : "";
    const int forced = force.empty() ? -1 : force == "id" ? 0 : force == "rule" ? 1 : force == "bits" ? 2 : -2;
    if (forced == -2) throw std::invalid_argument("GatedGIndex: GATE_FORCE must be id, rule or bits, not '" + force + "'");
    const char* rf = std::getenv("RULE_FILE");
    const bool file = rf && *rf;
    auto base_ns = [&](double lines, double levels) { return A + Bw * (lines - 1) + C * (levels - 1); };
    std::string note;
    discover_detail::say(note, "W=%g,%g,%g rule_ns=%g rule_f=%g bits_ns=%g bits_f=%g discover=%d force=%s |", A, Bw, C, rule_ns, rule_f,
                         bits_ns, bits_f, int(discover), forced < 0 ? "-" : force.c_str());
    chosen_ = nullptr, use_ = 0, id_.reset(), rule_.reset(), bits_.reset();
    id_.emplace(name_ + ".id");
    id_->bulk_load(kv, n, p);
    double best = base_ns(id_->expected_lines(), id_->expected_levels());
    use_ = 0, chosen_ = &*id_;
    discover_detail::say(note, " id %.1f", best);
    auto keep = [&](int which, double ns) {  // which (1 or 2), if predicted cheaper (or forced), in place of what it replaces, freed
      if (forced >= 0 ? which != forced : !(ns < best)) return false;
      if (use_ == 0) id_.reset();
      else if (use_ == 1) rule_.reset();
      else bits_.reset();
      use_ = which, best = ns;
      chosen_ = which == 1 ? static_cast<indexInterface<K, P>*>(&*rule_) : static_cast<indexInterface<K, P>*>(&*bits_);
      return true;
    };
    size_t lo = 0, hi = n;  // the bulk, as AdaptiveLineIndex peels it
    peel_outliers(kv, n, size_t(env_d("TRANSFORM_G_E", 64)), lo, hi);

    DiscoveredRule found;
    if (!file && discover) {
      try {
        found = discover_rule(kv + lo, hi - lo);  // the bulk: its first block is at or below the bulk's first key
        if (found.found) discover_detail::say(note, " discovered %zu blocks in %.2f s", found.blocks.size(), found.seconds);
        else discover_detail::say(note, " discovery: none (%.2f s)", found.seconds);
      } catch (const std::exception& e) {  // its 16 MB of FFT buffer, say: no rule, as with none found
        found = DiscoveredRule();
        discover_detail::say(note, " discovery failed (%s)", e.what());
      }
      if (const char* o = std::getenv("GATE_RULE_OUT"); o && *o && found.found) {  // the rule it found, as a rule file, one per process
        const std::string path = std::string(o) + "." + std::to_string(long(::getpid()));
        bool written = false;
        try {
          written = write_rule_file(found, path.c_str());
        } catch (const std::exception&) {
        }
        if (!written) discover_detail::say(note, " (rule file %s not written)", path.c_str());
      }
    } else if (!file) {
      discover_detail::say(note, " rule none (no RULE_FILE, discovery off)");
    }
    if (file || found.found) {
      try {
        std::optional<FastPeriodRule::Given> given;  // the discovered rule, in place of $RULE_FILE
        if (!file) given.emplace(found.S, found.blocks);
        rule_.emplace(name_ + ".rule");
        rule_->bulk_load(kv, n, p);
        const double b = base_ns(rule_->expected_lines(), rule_->expected_levels()), ns = b + rule_ns + rule_f * b;
        discover_detail::say(note, " rule %.1f", ns);
        if (!keep(1, ns)) rule_.reset();
      } catch (const std::exception& e) {
        rule_.reset();
        discover_detail::say(note, " rule rejected (%s)", e.what());
      }
    }

    if (BitCoord::has_inner(BitCoord::varying(kv + lo, hi - lo))) {
      try {
        bits_.emplace(name_ + ".bits");
        bits_->bulk_load(kv, n, p);
        const double b = base_ns(bits_->expected_lines(), bits_->expected_levels()), ns = b + bits_ns + bits_f * b;
        discover_detail::say(note, " bits %.1f", ns);
        if (!keep(2, ns)) bits_.reset();
      } catch (const std::exception& e) {
        bits_.reset();
        discover_detail::say(note, " bits rejected (%s)", e.what());
      }
    } else {
      discover_detail::say(note, " bits none");
    }
    std::fprintf(stderr, "gret: %s gate: %s -> %s\n", name_.c_str(), note.c_str(), use_ == 0 ? "identity" : use_ == 1 ? "rule" : "bits");
    if (forced >= 0 && use_ != forced) throw std::runtime_error("GatedGIndex: GATE_FORCE=" + force + ", but no such candidate was built");
  }

  // a virtual call on purpose: the kept index's own out-of-line get(), not a copy inlined here
  bool get(K key, P& val, Param* p = nullptr) override { return chosen_->get(key, val, p); }
  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  size_t scan(K key, size_t n, std::pair<K, P>* out, Param* p = nullptr) override {
    if (use_ == 0) return id_->Id::scan(key, n, out, p);
    return use_ == 1 ? rule_->Rule::scan(key, n, out, p) : bits_->Bits::scan(key, n, out, p);
  }
  long long memory_consumption() override {
    if (use_ == 0) return id_->Id::memory_consumption();
    return use_ == 1 ? rule_->Rule::memory_consumption() : bits_->Bits::memory_consumption();
  }
  int chosen() const { return use_; }  // 0 identity, 1 rule, 2 bits

 private:
  int use_ = 0;
  indexInterface<K, P>* chosen_ = nullptr;  // the kept candidate: get() jumps into its own out-of-line get()
  std::optional<Id> id_;                    // built in place
  std::optional<Rule> rule_;
  std::optional<Bits> bits_;
  std::string name_;
};

}  // namespace gret
