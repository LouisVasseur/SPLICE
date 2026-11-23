// SPDX-License-Identifier: MIT
// CsvLipp, CsvAlex: CSV (Amarasinghe, Choudhury, Qi, Bailey: "Learned Indexes with Distribution Smoothing via Virtual Points",
// arXiv 2408.06134) in GRE's harness. The authors published no code: this is OUR REIMPLEMENTATION from the paper, on top of
// GRE's own copies of LIPP and ALEX (the two base indexes the paper uses that GRE ships), whose files are used unmodified.
// GRE's LIPP is the GRE authors' fork of the LIPP authors' code: its remove (with which the virtual keys are erased), update
// and range scan are the fork's additions.
// docs/sota/csv.md has the full account; in short:
//   - Algorithm 1 (CDF_smoothing, the paper's section 4) is csv_detail::smooth below: greedy insertion of at most
//     lambda = alpha * |K| virtual keys into a key set K, each the candidate that lowers the squared error of the refitted
//     linear model rank ~ key the most, candidates filtered per run of free integers by the sign of the loss derivative.
//     On the paper's own example (the ten keys of its Fig. 2, alpha = 0.5) it gives the paper's five points in the paper's
//     order and the paper's losses (8.327 before, 2.293 after: Table 2); tests/sota/test_csv.cpp checks that.
//   - Algorithm 2 (CSV, section 5) is the bottom-up pass in bulk_load: the base index is first built on the real keys by its
//     authors' bulk loader; then, for every node with a subtree (LIPP: levels max_level - 2 down to 2; ALEX: every model node
//     below the root), the subtree's keys are collected, smoothed, and the subtree is rebuilt from the smoothed key set if the
//     cost condition holds. LIPP: the rebuilt subtree comes from LIPP's own bulk loader and is kept if the real keys sit
//     at a lower total level than before (CSV_LIPP_RULE=loss: whenever the loss fell, the paper's wording). ALEX: the subtree
//     becomes one data node, built by ALEX's own data-node bulk load, if ALEX's own cost terms (20 x expected search
//     iterations + 20 x level) drop by more than the threshold c.
//   - A virtual point is a key that is given to the base's node builder and then erased with the base's own erase, so it ends
//     as an empty slot that shaped the model: get and scan can never return one, and lookups pay nothing for them.
// What needed access the bases' public API does not give, and how it was had without editing their files: LIPP keeps its
// root, node type and builder private, so four private members (root, build_tree_bulk, destroy_tree, and pending_two for its
// size) are reached through explicit template instantiation (which is exempt from access checks); ALEX's nodes and root are
// public (its private istats_, the key domain, is read the same way, only by CsvAlex::consistent(), which the tests call).
// See the notes.
//
// The wrappers also guard what the two bases cannot do (counted in the line bulk_load prints; a side std::map holds the keys):
//   LIPP: a lookup of a key that is not stored can exit(0) in LIPP::at (RT_ASSERT), update and remove act on whatever key
//     sits in the slot, insert of a stored key exits: get, update, remove and put first find the key with a loop over LIPP's
//     nodes that compares the key (the same slot arithmetic as LIPP::at). Two adjacent keys too close for LIPP's long double
//     models (only above 2^63 with an 80-bit long double: keys 1 apart; from 2^52 where long double is double) cannot share
//     a two-key node: the later one goes to the side store.
//   ALEX: 2^64 - 1 is its gap sentinel (side store). A key outside ALEX's key domain (the range of the bulk-loaded keys)
//     makes it expand its root, with int casts of key-distance ratios that overflow for a far key and an assert that fails
//     for a near one: such keys are never given to ALEX (side store), so the domain never moves.
// Names: csv_lipp, csv_alex (alpha = 0.1, the paper's default) and csv_lipp<PP>, csv_alex<PP> with alpha = PP / 100, PP in
// [0, 99] without leading zeros (the paper sweeps 0.05 to 0.8: 5, 10, 20, 40, 80). PP = 0 is the base behind the same guards
// with no smoothing: the control for what the wrapper itself costs. Keys and payloads are uint64.
// Environment (all read at bulk_load):
//   CSV_LAMBDA      ceil (default) or floor: how lambda = alpha * |K| becomes a count (csv_detail::lambda_of)
//   CSV_LIPP_RULE   levels (default) or loss: the cost condition on LIPP (above)
//   CSV_COST_C      default 0: the paper's cost threshold c for ALEX, <= 0, in ALEX's cost units of about 1 ns
//   CSV_SEARCH_W, CSV_LEVEL_W  default 20 and 20 (ALEX's own kExpSearchIterationsWeight and kNodeLookupsWeight): the paper's
//                   search_constant and traversal_constant of equation 22, which it says to measure and does not give
//   CSV_MAX_SUBTREE default 131072: larger subtrees are not smoothed (Algorithm 1 costs alpha * m^2 on m keys)
//   CSV_BUDGET_S    default 1200: seconds the pass is planned to take at most. The cost of every subtree is estimated
//                   (csv_detail::size_limit) and, if the sum is above the budget, the largest subtrees are left as the base
//                   built them (over_budget= in the line); a hard stop at 1.5 x the budget catches a low estimate (late=).
#pragma once

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// GRE's copies of the two base indexes, unmodified (src/competitor on the include path, as in GRE's own build). GRE's
// interface classes (lipp/lipp.h, alex/alex.h) are not used: they have no include guards, and the wrappers call the indexes
// themselves, as those classes do.
#include "lipp/src/src/core/lipp.h"
#include "alex/src/src/core/alex.h"

namespace gret {
namespace csv_detail {

// ---- access to three private members of LIPP without editing lipp.h: the names used in an explicit instantiation are
// exempt from access checking ([temp.spec]), and the friend function it defines hands the member pointer out.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend"
#endif
template <class Tag>
struct Key {
  friend constexpr auto steal(Key);
};
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
template <class Tag, auto Ptr>
struct Grant {
  friend constexpr auto steal(Key<Tag>) { return Ptr; }
};
using Lipp = ::LIPP<uint64_t, uint64_t>;
struct LippRootTag {};
struct LippBuildTag {};
struct LippDestroyTag {};
struct LippPoolTag {};
template struct Grant<LippRootTag, &Lipp::root>;                // Node* LIPP::root
template struct Grant<LippBuildTag, &Lipp::build_tree_bulk>;    // Node* LIPP::build_tree_bulk(T* keys, P* values, int size)
template struct Grant<LippDestroyTag, &Lipp::destroy_tree>;     // void LIPP::destroy_tree(Node*)
template struct Grant<LippPoolTag, &Lipp::pending_two>;         // std::stack<Node*> LIPP::pending_two (read only, for its size)
// and one of ALEX, read only, for CsvAlex::consistent() (the tests' check that ALEX is never handed a key outside its key domain)
using Alex = alex::Alex<uint64_t, uint64_t, alex::AlexCompare, std::allocator<std::pair<uint64_t, uint64_t>>, false>;  // as GRE's alexInterface instantiates it
struct AlexDomainTag {};
template struct Grant<AlexDomainTag, &Alex::istats_>;           // Alex::istats_: key_domain_min_, key_domain_max_, the two out-of-domain counters

struct Scratch {
  std::vector<double> xs, sufx;
};

// The paper's Algorithm 1 (CDF_smoothing) on one key set. reg: the keys, sorted, unique (real ones and virtual ones accepted
// earlier, which count as keys, as line 31 of the algorithm appends them to K); isv: 1 for a virtual key. Adds at most lambda
// virtual keys and returns how many.
// The loss L is the squared error of the least-squares line rank ~ key over the keys and one candidate k_v, refitted with it
// (the paper's equations 5 to 16, here in centred sums: L = Cyy - Cxy^2 / Cxx). A candidate in the run of free integers
// [lo, hi] between two adjacent keys has a fixed rank, so Cxx is a quadratic and Cxy a line in k_v. Per run, as the paper's
// section 4.2 filters them: both ends if the run has at most two integers; otherwise the sign of dL/dk_v at the two ends
// (equation 17) decides: same sign, the two ends; opposite signs, the interior minimum. The paper finds that minimum from the
// two derivative values; here it is the root of dL/dk_v = 0 in closed form (the numerator is linear in k_v), rounded to the
// two integers around it (and the two ends after all when that root is not inside the run). Each round takes the candidate with the least L over all runs and stops when lambda points are in
// or L no longer falls (lines 27 to 31). Keys are scaled to x = (k - k_0) / (k_last - k_0); ranks are 0 .. m.
// ok(v, left, right): whether the base index can hold v between its neighbours left and right.
template <class Ok>
inline size_t smooth(std::vector<uint64_t>& reg, std::vector<uint8_t>& isv, size_t lambda, Ok&& ok, Scratch& w) {
  if (reg.size() < 3 || lambda == 0 || reg.back() <= reg.front()) return 0;
  const uint64_t k0 = reg.front();
  const double span = double(reg.back() - k0), inv = 1.0 / span;
  auto X = [&](uint64_t k) { return double(k - k0) * inv; };
  size_t added = 0;
  double Lprev = -1;
  while (added < lambda) {
    const size_t m = reg.size();
    w.xs.resize(m);
    w.sufx.resize(m + 1);
    double A = 0, B = 0, S1 = 0;
    for (size_t i = 0; i < m; ++i) {
      const double x = X(reg[i]);
      w.xs[i] = x, A += x * x, B += x, S1 += x * double(i);
    }
    if (Lprev < 0) {  // L of the key set as it is
      const double N0 = double(m), Sy = N0 * (N0 - 1) / 2, Syy = (N0 - 1) * N0 * (2 * N0 - 1) / 6;
      const double cxx = A - B * B / N0, cxy = S1 - B * Sy / N0;
      Lprev = std::max(0.0, (Syy - Sy * Sy / N0) - (cxx > 0 ? cxy * cxy / cxx : 0));
    }
    w.sufx[m] = 0;
    for (size_t i = m; i-- > 0;) w.sufx[i] = w.sufx[i + 1] + w.xs[i];
    // with one more key: N keys, ranks 0 .. m; the keys after the candidate move up one rank (the suffix sum)
    const double N = double(m + 1), Sy = double(m) * double(m + 1) / 2, Syy = double(m) * double(m + 1) * double(2 * m + 1) / 6;
    const double Cyy = Syy - Sy * Sy / N, a2 = 1 - 1 / N, a1 = -2 * B / N, a0 = A - B * B / N;
    // L = Cyy - Cxy^2 / Cxx is least where fit = Cxy^2 / Cxx is most: a candidate is better than the best so far when its fit
    // is above fbest (compared without the division; fbest starts at the fit that leaves L where it is)
    double fbest = Cyy - Lprev;
    uint64_t bestk = 0;
    size_t bestg = 0;
    bool found = false;
    for (size_t g = 0; g + 1 < m; ++g) {
      if (reg[g + 1] - reg[g] < 2) continue;
      const uint64_t lo = reg[g] + 1, hi = reg[g + 1] - 1;
      const double c1 = double(g + 1) - Sy / N, c0 = (S1 + w.sufx[g + 1]) - B * Sy / N;
      // a candidate with its two sums: taken if its fit beats the best so far and the base index can hold the key
      auto take = [&](uint64_t k, double cxx, double cxy) {
        if (cxx > 0 && cxy * cxy > fbest * cxx && ok(k, reg[g], reg[g + 1])) fbest = cxy * cxy / cxx, bestk = k, bestg = g, found = true;
      };
      const double xl = X(lo), cxxl = a0 + a1 * xl + a2 * xl * xl, cxyl = c0 + c1 * xl;
      if (hi == lo) {
        take(lo, cxxl, cxyl);
        continue;
      }
      const double xh = X(hi), cxxh = a0 + a1 * xh + a2 * xh * xh, cxyh = c0 + c1 * xh;
      bool ends = true;
      if (hi - lo > 1) {
        // the sign of dL/dx = -d(Cxy^2 / Cxx)/dx at the two ends (the factor 1 / Cxx^2 > 0 is left out)
        const double dl = -(2 * cxyl * c1 * cxxl - cxyl * cxyl * (a1 + 2 * a2 * xl)), dh = -(2 * cxyh * c1 * cxxh - cxyh * cxyh * (a1 + 2 * a2 * xh));
        if ((dl < 0 && dh > 0) || (dl > 0 && dh < 0)) {
          const double den = c1 * a1 - 2 * c0 * a2;
          // as an offset from k_0 (a double holds it to 53 bits at any key magnitude; k_0 + offset in floating point would not)
          const double off = den != 0 ? (c0 * a1 - 2 * c1 * a0) / den * span : -1;
          if (off > double(lo - k0) && off < double(hi - k0)) {  // the minimum is inside the run: the two integers around it
            const uint64_t kf = std::min(std::max(k0 + uint64_t(off), lo), hi - 1);
            for (uint64_t k : {kf, kf + 1}) {
              const double x = X(k);
              take(k, a0 + a1 * x + a2 * x * x, c0 + c1 * x);
            }
            ends = false;
          }  // else the sign change is not a minimum inside the run (rounding, or the one maximum of L, where Cxy = 0)
        }
      }
      if (ends) {
        take(lo, cxxl, cxyl);
        take(hi, cxxh, cxyh);
      }
    }
    if (!found) break;  // L no longer falls
    reg.insert(reg.begin() + std::ptrdiff_t(bestg + 1), bestk);
    isv.insert(isv.begin() + std::ptrdiff_t(bestg + 1), uint8_t(1));
    Lprev = std::max(0.0, Cyy - fbest);
    ++added;
  }
  return added;
}

inline double env_number(const char* name, double dflt, double lo, double hi) {
  const char* e = std::getenv(name);
  if (!e || !*e) return dflt;
  char* end = nullptr;
  const double v = std::strtod(e, &end);
  if (!end || *end || !(v >= lo && v <= hi)) throw std::invalid_argument(std::string("gret csv: ") + name + " is not a number in range: '" + e + "'");
  return v;
}

// lambda = alpha * |K|, the most virtual points a key set of m keys may get. The paper states it both ways: Definition 1 asks
// for |V| <= lambda (the integer part, up = false), and Algorithm 1's loop "while V.size < lambda" adds a point while the
// count is below the real number alpha * |K| (rounded up, up = true). They differ where it matters for LIPP: with the integer
// part a subtree of fewer than 1 / alpha keys gets no point at all.
inline size_t lambda_of(double alpha, size_t m, bool up) {
  const double l = alpha * double(m);
  return up ? size_t(std::ceil(l - 1e-9)) : size_t(l + 1e-9);
}

// Seconds one round of smooth() spends per gap on this machine: twice the least of five timings on a made-up set of 4096
// keys (the least, so that a preempted timing does not count; twice, because large subtrees were seen to run up to 1.5 times
// as slow as this set), measured once.
inline double gap_seconds() {
  static const double s = [] {
    std::vector<uint64_t> keys;
    uint64_t k = 1000, x = 88172645463325252ull;
    for (int i = 0; i < 4096; ++i) {
      x ^= x << 13, x ^= x >> 7, x ^= x << 17;
      k += 3 + (x & 1023) * (uint64_t(1) << (2 * ((i / 512) % 4)));
      keys.push_back(k);
    }
    Scratch w;
    double least = 1;
    for (int rep = 0; rep < 5; ++rep) {
      std::vector<uint64_t> reg = keys;
      std::vector<uint8_t> isv(reg.size(), 0);
      const auto t0 = std::chrono::steady_clock::now();
      const size_t rounds = smooth(reg, isv, 16, [](uint64_t, uint64_t, uint64_t) { return true; }, w) + 1;
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      least = std::min(least, dt / (double(rounds) * 4096.0));
    }
    return std::max(2e-10, 2 * least);
  }();
  return s;
}

// The plan of the pass under CSV_BUDGET_S. Algorithm 1 on a subtree of m keys runs up to alpha * m rounds over m gaps, so the
// pass costs about gap_seconds() * alpha * sum of m^2, and the large subtrees take nearly all of it. sizes: the key counts of
// the subtrees the pass would smooth. Returns the largest size the pass takes: all of them if the estimate fits the budget,
// otherwise the size below which it does (smallest first: a subtree's gain grows with m, its cost with m^2). est: the estimate
// in seconds for the subtrees taken.
inline size_t size_limit(std::vector<size_t>& sizes, double alpha, bool up, double budget_s, size_t max_subtree, double& est) {
  std::sort(sizes.begin(), sizes.end());
  const double g = gap_seconds();
  auto cost = [&](size_t m) { return g * double(lambda_of(alpha, m, up)) * double(m); };
  est = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    const double c = cost(sizes[i]);
    if (est + c > budget_s) {
      const size_t limit = sizes[i] - 1;  // none of this size: equal sizes are taken together or not at all
      while (i > 0 && sizes[i - 1] > limit) --i, est -= cost(sizes[i]);
      return limit;
    }
    est += c;
  }
  return max_subtree;
}

// 0 if the variable is unset, empty or equal to a; 1 if it equals b
inline bool env_choice(const char* name, const char* a, const char* b) {
  const char* e = std::getenv(name);
  if (!e || !*e || std::string(e) == a) return false;
  if (std::string(e) == b) return true;
  throw std::invalid_argument(std::string("gret csv: ") + name + " must be '" + a + "' or '" + b + "', not '" + e + "'");
}

struct PassStats {
  size_t candidates = 0;      // nodes with a subtree, at the levels the pass covers
  size_t no_room = 0;         // no point to add: lambda is 0 for so few keys, or the points accepted below have used it
  size_t too_big = 0;         // more keys than CSV_MAX_SUBTREE (or than one ALEX data node holds): not smoothed
  size_t over_budget = 0;     // more keys than the size limit planned from CSV_BUDGET_S: not smoothed
  size_t late = 0;            // not smoothed because the pass had run 1.5 x CSV_BUDGET_S (the plan's estimate was too low)
  size_t smoothed = 0;        // Algorithm 1 added at least one point
  size_t accepted = 0;        // the cost condition kept the rebuilt subtree (it is tested with or without a new point)
  size_t promoted = 0;        // keys at a lower level than before, summed over the accepted rebuilds
  size_t demoted = 0;         // keys at a higher level than before in an accepted rebuild (LIPP)
  size_t nodes_removed = 0;   // nodes fewer after the accepted rebuilds
};

// the two maps of a scan merged: out holds `got` pairs of the base, ascending, all >= key
inline size_t merge_side(const std::map<uint64_t, uint64_t>& side, uint64_t key, size_t n, std::pair<uint64_t, uint64_t>* out, size_t got) {
  auto it = side.lower_bound(key);
  if (it == side.end()) return got;
  std::vector<std::pair<uint64_t, uint64_t>> base(out, out + got);
  size_t i = 0, o = 0;
  while (o < n && (i < base.size() || it != side.end())) {
    // past the base's last pair there may be base keys this scan did not fetch (it stopped at n): only side keys below it go in
    if (i < base.size() && (it == side.end() || base[i].first < it->first)) out[o++] = base[i++];
    else if (it != side.end() && (i < base.size() || got < n)) out[o++] = std::pair<uint64_t, uint64_t>(it->first, it->second), ++it;
    else break;
  }
  return o;
}

}  // namespace csv_detail

// ------------------------------------------------------------------------------------------------------------------ LIPP
class CsvLipp final : public indexInterface<uint64_t, uint64_t> {
  using K = uint64_t;
  using P = uint64_t;
  using L = csv_detail::Lipp;
  static constexpr auto kRoot = steal(csv_detail::Key<csv_detail::LippRootTag>{});
  static constexpr auto kBuild = steal(csv_detail::Key<csv_detail::LippBuildTag>{});
  static constexpr auto kDestroy = steal(csv_detail::Key<csv_detail::LippDestroyTag>{});
  static constexpr auto kPool = steal(csv_detail::Key<csv_detail::LippPoolTag>{});
  using Node = std::remove_pointer_t<std::remove_reference_t<decltype(std::declval<L&>().*kRoot)>>;
  static constexpr int kMant = LDBL_MANT_DIG >= 64 ? 64 : LDBL_MANT_DIG;  // bits LIPP's long double models keep

 public:
  static constexpr size_t kMaxKeys = (size_t(1) << 30) - 1;  // LIPP sizes its nodes in int: 2 slots a key at the root, below 2^31
  static constexpr size_t kMapNode = 4 * sizeof(void*) + 2 * sizeof(uint64_t);

  CsvLipp(std::string name, double alpha) : name_(std::move(name)), alpha_(alpha) {
    if (!(alpha >= 0 && alpha < 1)) throw std::invalid_argument(name_ + ": alpha must be in [0, 1)");
  }
  CsvLipp(const CsvLipp&) = delete;
  CsvLipp& operator=(const CsvLipp&) = delete;

  void init(Param* = nullptr) override {}

  // kv: sorted by key, keys unique (GRE's contract; checked)
  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 1; i < n; ++i)
      if (kv[i].first <= kv[i - 1].first) throw std::invalid_argument(name_ + ": bulk_load needs sorted, unique keys");
    // the settings, before anything changes: a malformed one throws
    by_loss_ = csv_detail::env_choice("CSV_LIPP_RULE", "levels", "loss");
    up_ = !csv_detail::env_choice("CSV_LAMBDA", "ceil", "floor");
    max_subtree_ = size_t(csv_detail::env_number("CSV_MAX_SUBTREE", 131072, 3, 1e9));
    budget_s_ = csv_detail::env_number("CSV_BUDGET_S", 1200, 0, 1e9);
    side_.clear(), virt_.clear();
    st_ = csv_detail::PassStats{};
    put_side_ = 0, n_virtual_ = 0;
    // keys LIPP's models cannot tell from the key kept before them go to the side store
    std::vector<std::pair<K, P>> kept;
    size_t first_close = n;
    for (size_t i = 1; i < n && first_close == n; ++i)
      if (kv[i].first - kv[i - 1].first < min_gap(kv[i].first)) first_close = i;
    const std::pair<K, P>* in = kv;
    size_t m = n;
    if (first_close < n) {
      kept.assign(kv, kv + first_close);
      for (size_t i = first_close; i < n; ++i) {
        if (kv[i].first - kept.back().first < min_gap(kv[i].first)) side_.emplace(kv[i].first, kv[i].second);
        else kept.push_back(kv[i]);
      }
      in = kept.data(), m = kept.size();
    }
    if (m > kMaxKeys) throw std::invalid_argument(name_ + ": 2^30 keys or more");
    lipp_.bulk_load(in, int(m));  // the authors' bulk load, on the real keys: the paper's "original index"
    n_lipp_ = m;
    const double base_ms = ms_since(t0);
    std::vector<size_t> before, after;
    levels(root(), 1, before);
    plan_s_ = 0;
    if (alpha_ > 0 && m >= 3) {
      // the paper's section 6.1: on LIPP the pass starts at the second level from the bottom that has subtrees and stops at
      // level 2 (the root is level 1 and is never rebuilt). Levels with subtrees are 1 .. depth - 1.
      start_level_ = std::max<int>(2, int(before.size()) - 2);
      std::vector<size_t> sizes;
      plan(root(), 1, sizes);
      limit_ = csv_detail::size_limit(sizes, alpha_, up_, budget_s_, max_subtree_, plan_s_);
      pass_t0_ = std::chrono::steady_clock::now();
      visit(root(), 1, nullptr, 0);
      n_virtual_ = virt_.size();
      if (!keep_virtual_) std::vector<K>().swap(virt_);
      // the pass's work arrays (up to CSV_MAX_SUBTREE entries each) are given back: memory_consumption does not count them
      std::vector<K>().swap(keys_), std::vector<K>().swap(nkeys_), std::vector<K>().swap(reg_);
      std::vector<P>().swap(vals_), std::vector<P>().swap(tvals_);
      std::vector<uint32_t>().swap(dep_), std::vector<uint32_t>().swap(ndep_);
      std::vector<uint8_t>().swap(isv_);
      scratch_ = csv_detail::Scratch();
      levels(root(), 1, after);
    } else {
      after = before;
    }
    const double pass_ms = ms_since(t0) - base_ms;
    // every key against the wrapper's lookup, and LIPP's own count
    if (size_t(root()->size) != m) throw std::runtime_error(name_ + ": LIPP's root counts another number of keys than were loaded");
    for (size_t i = 0; i < m; ++i) {
      const P* p = find(in[i].first);
      if (!p || *p != in[i].second) throw std::runtime_error(name_ + ": a loaded key is not found after the CSV pass");
    }
    auto mean = [](const std::vector<size_t>& h) {
      double s = 0, c = 0;
      for (size_t l = 0; l < h.size(); ++l) s += double(h[l]) * double(l + 1), c += double(h[l]);
      return c ? s / c : 0.0;
    };
    auto deep = [](const std::vector<size_t>& h) {
      size_t c = 0;
      for (size_t l = 2; l < h.size(); ++l) c += h[l];
      return c;
    };
    std::fprintf(stderr,
                 "gret: %s reimplementation of CSV (arXiv 2408.06134) on GRE's LIPP: n=%zu in_lipp=%zu side=%zu alpha=%.2f lambda=%s rule=%s levels=%zu "
                 "start_level=%d candidates=%zu no_room=%zu too_big=%zu over_budget=%zu late=%zu smoothed=%zu accepted=%zu virtual=%zu (%.2f%% of n) "
                 "promoted=%zu demoted=%zu avg_level=%.4f->%.4f keys_at_level3plus=%zu->%zu nodes_removed=%zu bytes_per_key=%.2f "
                 "(LIPP::total_size %.2f; not counted: LIPP's pool of %zu spare two-key nodes, %.0f MB) size_limit=%zu plan_s=%.1f gap_ns=%.2f base_ms=%.1f "
                 "csv_ms=%.1f verify_ms=%.1f\n",
                 name_.c_str(), n, m, side_.size(), alpha_, up_ ? "ceil" : "floor", by_loss_ ? "loss" : "levels", before.size(), start_level_, st_.candidates, st_.no_room, st_.too_big, st_.over_budget,
                 st_.late, st_.smoothed, st_.accepted, n_virtual_, n ? 100.0 * double(n_virtual_) / double(n) : 0.0, st_.promoted, st_.demoted, mean(before), mean(after),
                 deep(before), deep(after), st_.nodes_removed, n ? double(memory_consumption()) / double(n) : 0.0, n ? double(lipp_.total_size()) / double(n) : 0.0, pool_nodes(),
                 double(pool_bytes()) / 1048576.0, alpha_ > 0 && m >= 3 ? limit_ : size_t(0), plan_s_, alpha_ > 0 && m >= 3 ? 1e9 * csv_detail::gap_seconds() : 0.0, base_ms, pass_ms, ms_since(t0) - base_ms - pass_ms);
    side_reported_ = side_.size();
  }

  bool get(K key, P& val, Param* = nullptr) override {
    if (const P* p = find(key)) {
      val = *p;
      return true;
    }
    if (__builtin_expect(side_.empty(), 1)) return false;
    const auto it = side_.find(key);
    if (it == side_.end()) return false;
    val = it->second;
    return true;
  }

  // true if the key was not there and is now; false, and nothing changes, if it was
  bool put(K key, P value, Param* = nullptr) override {
    if (find(key)) return false;
    if (!side_.empty() && side_.count(key)) return false;
    if (n_lipp_ >= kMaxKeys || !can_take(key)) {
      side_.emplace(key, value);
      ++put_side_;
      return true;
    }
    lipp_.insert(key, value);  // the authors' insert
    ++n_lipp_;
    return true;
  }

  bool update(K key, P value, Param* = nullptr) override {
    if (P* p = find(key)) {
      *p = value;
      return true;
    }
    if (side_.empty()) return false;
    const auto it = side_.find(key);
    if (it == side_.end()) return false;
    it->second = value;
    return true;
  }

  bool remove(K key, Param* = nullptr) override {
    if (find(key)) {
      if (n_lipp_ == 1) lipp_.bulk_load(nullptr, 0);  // LIPP::remove dereferences the root's parent when the root empties
      else lipp_.remove(key);                         // LIPP::remove, which GRE's fork of LIPP adds (not in the LIPP authors' repository)
      --n_lipp_;
      return true;
    }
    return !side_.empty() && side_.erase(key) != 0;
  }

  size_t scan(K key, size_t n, std::pair<K, P>* out, Param* = nullptr) override {
    if (n == 0) return 0;
    const size_t got = n_lipp_ ? size_t(lipp_.range_query_len(out, key, int(std::min<size_t>(n, size_t(INT_MAX))))) : 0;  // LIPP's scan as GRE's fork has it
    return side_.empty() ? got : csv_detail::merge_side(side_, key, n, out, got);
  }

  // every byte the index holds: LIPP's nodes with all their slots and both bitmaps (LIPP::total_size(), which GRE reports for
  // LIPP, counts one byte per bitmap), the side store's tree nodes, this object. Not LIPP's pool of 10^7 spare two-key nodes,
  // which every LIPP object allocates in its constructor and GRE's LIPP number leaves out as well.
  long long memory_consumption() override {
    size_t b = sizeof(*this) + side_.size() * kMapNode + bytes(root());
    if (side_.size() != side_reported_) {
      side_reported_ = side_.size();
      std::fprintf(stderr, "gret: %s now in_lipp=%zu side=%zu (%zu puts went to the side store since the bulk load)\n", name_.c_str(), n_lipp_, side_.size(), put_side_);
    }
    return (long long)b;
  }

  // for tests and reports
  size_t in_base() const { return n_lipp_; }
  size_t in_side() const { return side_.size(); }
  size_t n_virtual() const { return n_virtual_; }
  const csv_detail::PassStats& pass() const { return st_; }
  void keep_virtual(bool on) { keep_virtual_ = on; }           // keep the accepted virtual keys after bulk_load (tests)
  const std::vector<K>& virtual_keys() const { return virt_; }
  long long authors_total_size() const { return (long long)lipp_.total_size(); }
  // LIPP's pool: every LIPP object allocates 10^7 two-key nodes in its constructor (lipp.h:83) and keeps the ones not in use
  size_t pool_nodes() const { return (lipp_.*kPool).size(); }
  // requested bytes: the node, its 8 slots, its two one-byte bitmaps, the stack's pointer to it. The allocator rounds the four
  // blocks of a node up (measured: 248 bytes a spare node on macOS against 210 here), so the pool really holds about a fifth more.
  size_t pool_bytes() const { return pool_nodes() * (sizeof(Node) + 8 * sizeof(std::declval<Node&>().items[0]) + 2 + sizeof(Node*)); }
  void verify_base() { lipp_.verify(); }  // LIPP's own size check; exits through RT_ASSERT if it fails

 private:
  // the smallest gap to the key before it at which LIPP can hold k: a two-key node puts its keys at slots 2 and 5 with slope
  // 3 / gap, and 3 k / gap must stay below 2^(mantissa + 1) for the two products to differ (docs/sota/csv.md)
  static uint64_t min_gap(uint64_t k) { return kMant >= 64 ? (k >> 63) + 1 : 4 * (k >> (kMant - 1)) + 1; }
  static constexpr uint64_t kAnyGapBelow = kMant >= 64 ? (uint64_t(1) << 63) - 1 : (uint64_t(1) << (kMant - 1)) - 8;

  Node* root() { return lipp_.*kRoot; }
  static double ms_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }

  // LIPP's PREDICT_POS (lipp.h:49), which is private: the same expressions on the same model function
  static int slot(const Node* n, K key) {
    const double v = n->model.predict_double(key);
    if (v > std::numeric_limits<int>::max() / 2) return n->num_items - 1;
    if (v < 0) return 0;
    return std::min(n->num_items - 1, static_cast<int>(v));
  }

  // LIPP::at's walk (lipp.h:114) with a comparison where at() has RT_ASSERT(key == key): the payload's address, or nullptr
  P* find(K key) {
    Node* n = root();
    for (;;) {
      const int pos = slot(n, key);
      const unsigned bit = unsigned(pos) & 7u;
      if ((n->child_bitmap[pos >> 3] >> bit) & 1) {
        n = n->items[pos].comp.child;
        continue;
      }
      if (((n->none_bitmap[pos >> 3] >> bit) & 1) || n->items[pos].comp.data.key != key) return nullptr;
      return &n->items[pos].comp.data.value;
    }
  }

  // whether LIPP can hold k (not stored) next to the keys it holds now
  bool can_take(K k) {
    if (k < kAnyGapBelow) return true;  // a gap of 1 is enough on both sides
    const uint64_t w = min_gap(k) - 1;  // a stored key in [k - w, k) is too close
    std::pair<K, P> r;
    if (n_lipp_ == 0 || lipp_.range_query_len(&r, k > w ? k - w : 0, 1) == 0) return true;
    if (r.first < k) return false;
    return r.first - k >= min_gap(r.first);  // the stored key after k
  }

  template <class F>
  static void each_slot(const Node* n, F&& f) {  // f(slot, is_child) for every slot that is not empty, ascending
    const int bytes = BITMAP_SIZE(n->num_items);
    for (int b = 0; b < bytes; ++b) {
      unsigned live = uint8_t(~n->none_bitmap[b]);
      const unsigned child = n->child_bitmap[b];
      while (live) {
        const int j = __builtin_ctz(live);
        live &= live - 1;
        const int i = b * 8 + j;
        if (i >= n->num_items) return;
        f(i, bool((child >> j) & 1));
      }
    }
  }
  static void levels(const Node* n, size_t level, std::vector<size_t>& h) {  // h[l - 1]: keys at level l
    if (h.size() < level) h.resize(level, 0);
    size_t here = 0;
    each_slot(n, [&](int i, bool child) {
      if (child) levels(n->items[i].comp.child, level + 1, h);
      else ++here;
    });
    h[level - 1] += here;
  }
  static size_t bytes(const Node* n) {
    size_t b = sizeof(Node) + size_t(n->num_items) * sizeof(n->items[0]) + 2 * size_t(BITMAP_SIZE(n->num_items));
    each_slot(n, [&](int i, bool child) {
      if (child) b += bytes(n->items[i].comp.child);
    });
    return b;
  }
  static size_t count_nodes(const Node* n) {
    size_t c = 1;
    each_slot(n, [&](int i, bool child) {
      if (child) c += count_nodes(n->items[i].comp.child);
    });
    return c;
  }
  // the keys of a subtree in ascending order, with payloads and levels (the subtree's root is level 0)
  void collect(const Node* n, uint32_t depth, std::vector<K>& keys, std::vector<P>* vals, std::vector<uint32_t>& dep) {
    each_slot(n, [&](int i, bool child) {
      if (child) return collect(n->items[i].comp.child, depth + 1, keys, vals, dep);
      keys.push_back(n->items[i].comp.data.key);
      if (vals) vals->push_back(n->items[i].comp.data.value);
      dep.push_back(depth);
    });
  }

  // the sizes of the subtrees visit() would smooth, for the plan under CSV_BUDGET_S
  void plan(const Node* n, int level, std::vector<size_t>& sizes) const {
    bool has_child = false;
    each_slot(n, [&](int i, bool child) {
      if (child) has_child = true, plan(n->items[i].comp.child, level + 1, sizes);
    });
    const size_t m = size_t(n->size);
    if (has_child && level >= 2 && level <= start_level_ && m <= max_subtree_ && csv_detail::lambda_of(alpha_, m, up_) > 0) sizes.push_back(m);
  }

  // The paper's Algorithm 2 as a post-order walk: the subtrees below a node are done before the node, which is the paper's
  // bottom-up order within every branch. virt_ holds the accepted virtual keys in key order; those of n's subtree are its tail
  // from v0 on. parent/pslot: where n hangs.
  void visit(Node* n, int level, Node* parent, int pslot) {
    const size_t v0 = virt_.size();
    bool has_child = false;
    each_slot(n, [&](int i, bool child) {
      if (!child) return;
      has_child = true;
      visit(n->items[i].comp.child, level + 1, n, i);  // may hang another node in slot i; n's bitmaps do not change
    });
    if (!has_child || level < 2 || level > start_level_) return;
    ++st_.candidates;
    const size_t m = size_t(n->size), have = virt_.size() - v0, cap = csv_detail::lambda_of(alpha_, m, up_);
    // lambda = alpha * |K| for the subtree as a whole, points accepted below included
    const size_t room = cap > have ? cap - have : 0;
    if (!room) {
      ++st_.no_room;
      // Algorithm 2 tests the cost condition whether or not Algorithm 1 added a point (lines 10 to 13). Without a new point the
      // loss has not fallen (rule = loss: no rebuild), and without any virtual key below, LIPP's builder would make the very
      // subtree it made at the bulk load. What is left: virtual keys below and none to add, where the rebuild is tried.
      if (have == 0 || by_loss_) return;
    }
    if (m > max_subtree_) { ++st_.too_big; return; }
    if (m > limit_) { ++st_.over_budget; return; }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - pass_t0_).count() > 1.5 * budget_s_) { ++st_.late; return; }
    keys_.clear(), vals_.clear(), dep_.clear();
    collect(n, 0, keys_, &vals_, dep_);
    if (keys_.size() != m) throw std::runtime_error(name_ + ": a LIPP node counts another number of keys than its subtree holds");
    // K: the real keys and the virtual keys accepted below
    reg_.resize(m + have), isv_.resize(m + have);
    for (size_t i = 0, j = v0, o = 0; o < m + have; ++o) {
      const bool v = j < virt_.size() && (i == m || virt_[j] < keys_[i]);
      reg_[o] = v ? virt_[j++] : keys_[i++], isv_[o] = v;
    }
    const size_t added = room ? csv_detail::smooth(reg_, isv_, room, [](uint64_t v, uint64_t left, uint64_t right) { return v - left >= min_gap(v) && right - v >= min_gap(right); }, scratch_) : 0;
    if (added) ++st_.smoothed;
    else if (have == 0 || by_loss_) return;  // as above
    // the subtree LIPP's own builder makes of the smoothed key set, then the virtual keys erased with LIPP::remove (GRE's fork)
    const size_t all = reg_.size();
    tvals_.assign(all, 0);
    for (size_t o = 0, i = 0; o < all; ++o)
      if (!isv_[o]) tvals_[o] = vals_[i++];
    Node* t = (lipp_.*kBuild)(reg_.data(), tvals_.data(), int(all));
    Node*& r = lipp_.*kRoot;
    Node* const whole = r;
    r = t;  // LIPP::remove starts at the root: for these calls the new subtree is the tree
    bool ok = true;
    for (size_t o = 0; o < all && ok; ++o)
      if (isv_[o]) ok = lipp_.remove(reg_[o]);
    t = r, r = whole;
    nkeys_.clear(), ndep_.clear();
    collect(t, 0, nkeys_, nullptr, ndep_);
    if (!ok || nkeys_ != keys_ || size_t(t->size) != m) {
      (lipp_.*kDestroy)(t);
      throw std::runtime_error(name_ + ": the rebuilt subtree does not hold exactly the subtree's keys");
    }
    // the cost condition. The paper: "For indexes that does not contain any searching component such as LIPP and SALI, their
    // loss function values can be taken as the cost conditions. This is because if the new model could hold more keys than
    // before, then it does not have any other component [...] that would negatively affect the performance." Algorithm 1 has
    // lowered the loss, or it would have added nothing: read to the letter (CSV_LIPP_RULE=loss), every smoothed subtree is
    // rebuilt. The default (CSV_LIPP_RULE=levels) checks the second sentence on the built nodes, because LIPP's builder fits
    // its own model (FMCD), not the least-squares line whose loss fell: the rebuilt subtree is kept if its real keys sit at
    // a lower total level than before. That can only help the lookups of the result (docs/sota/csv.md, deviation 1).
    uint64_t old_sum = 0, new_sum = 0;
    size_t up = 0, down = 0;
    for (size_t i = 0; i < m; ++i) old_sum += dep_[i], new_sum += ndep_[i], up += ndep_[i] < dep_[i], down += ndep_[i] > dep_[i];
    if (!by_loss_ && new_sum >= old_sum) {
      (lipp_.*kDestroy)(t);
      return;
    }
    const size_t nodes_old = count_nodes(n), nodes_new = count_nodes(t);
    parent->items[pslot].comp.child = t;  // n is not the root: level >= 2
    (lipp_.*kDestroy)(n);
    ++st_.accepted, st_.promoted += up, st_.demoted += down;
    st_.nodes_removed += nodes_old > nodes_new ? nodes_old - nodes_new : 0;
    virt_.resize(v0);
    for (size_t o = 0; o < all; ++o)
      if (isv_[o]) virt_.push_back(reg_[o]);
  }

  std::string name_;
  double alpha_;
  L lipp_;                      // the authors' index (its constructor allocates LIPP's pool of 10^7 two-key nodes)
  std::map<K, P> side_;         // keys LIPP cannot hold (see the head of this file)
  size_t n_lipp_ = 0, put_side_ = 0, side_reported_ = 0, n_virtual_ = 0;
  csv_detail::PassStats st_;
  bool keep_virtual_ = false, by_loss_ = false, up_ = true;
  size_t max_subtree_ = 131072, limit_ = 131072;
  double budget_s_ = 1200, plan_s_ = 0;
  int start_level_ = 2;
  std::chrono::steady_clock::time_point pass_t0_;
  std::vector<K> virt_, keys_, nkeys_, reg_;
  std::vector<P> vals_, tvals_;
  std::vector<uint32_t> dep_, ndep_;
  std::vector<uint8_t> isv_;
  csv_detail::Scratch scratch_;
};

// ------------------------------------------------------------------------------------------------------------------ ALEX
class CsvAlex final : public indexInterface<uint64_t, uint64_t> {
  using K = uint64_t;
  using P = uint64_t;
  using V = std::pair<K, P>;
  using A = csv_detail::Alex;
  static constexpr auto kDomain = steal(csv_detail::Key<csv_detail::AlexDomainTag>{});
  using ANode = alex::AlexNode<K, P>;
  using MNode = A::model_node_type;
  using DNode = A::data_node_type;
  static constexpr K kMax = ~K(0);

 public:
  static constexpr size_t kMaxKeys = size_t(1) << 30;  // ALEX counts keys in int
  static constexpr size_t kMapNode = 4 * sizeof(void*) + 2 * sizeof(uint64_t);

  CsvAlex(std::string name, double alpha) : name_(std::move(name)), alpha_(alpha), a_(new A()) {
    if (!(alpha >= 0 && alpha < 1)) throw std::invalid_argument(name_ + ": alpha must be in [0, 1)");
  }
  CsvAlex(const CsvAlex&) = delete;
  CsvAlex& operator=(const CsvAlex&) = delete;

  void init(Param* = nullptr) override {}

  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 1; i < n; ++i)
      if (kv[i].first <= kv[i - 1].first) throw std::invalid_argument(name_ + ": bulk_load needs sorted, unique keys");
    // the settings, before anything changes: a malformed one throws
    up_ = !csv_detail::env_choice("CSV_LAMBDA", "ceil", "floor");
    max_subtree_ = size_t(csv_detail::env_number("CSV_MAX_SUBTREE", 131072, 3, 1e9));
    budget_s_ = csv_detail::env_number("CSV_BUDGET_S", 1200, 0, 1e9);
    cost_c_ = csv_detail::env_number("CSV_COST_C", 0, -1e9, 0);
    w_search_ = csv_detail::env_number("CSV_SEARCH_W", alex::kExpSearchIterationsWeight, 0, 1e9);
    w_level_ = csv_detail::env_number("CSV_LEVEL_W", alex::kNodeLookupsWeight, 0, 1e9);
    side_.clear(), virt_.clear();
    st_ = csv_detail::PassStats{};
    put_side_ = 0, n_virtual_ = 0;
    a_.reset(new A());  // ALEX's bulk_load needs an empty index
    plan_s_ = 0, limit_ = 0;
    size_t m = n;
    if (n && kv[n - 1].first == kMax) side_.emplace(kMax, kv[n - 1].second), --m;  // ALEX's gap sentinel
    // more than kRun keys inside one window of a few double steps: the later ones go to the side store (see window())
    std::vector<V> kept;
    size_t first_crowded = m;
    for (size_t i = kRun; i < m && first_crowded == m; ++i)
      if (kv[i].first - kv[i - kRun].first < window(kv[i].first)) first_crowded = i;
    if (first_crowded < m) {
      kept.assign(kv, kv + first_crowded);
      for (size_t i = first_crowded; i < m; ++i) {
        if (kv[i].first - kept[kept.size() - kRun].first < window(kv[i].first)) side_.emplace(kv[i].first, kv[i].second);
        else kept.push_back(kv[i]);
      }
      kv = kept.data(), m = kept.size();
    }
    if (m > kMaxKeys) throw std::invalid_argument(name_ + ": more than 2^30 keys");
    lo_ = 1, hi_ = 0;
    if (m) a_->bulk_load(kv, int(m)), lo_ = kv[0].first, hi_ = kv[m - 1].first;  // the authors' bulk load, on the real keys
    n_alex_ = m;
    const double base_ms = ms_since(t0);
    const int nodes0 = a_->stats_.num_data_nodes + a_->stats_.num_model_nodes;
    const double level0 = mean_level();
    if (alpha_ > 0 && m >= 3 && !a_->root_node_->is_leaf_) {
      max_node_keys_ = size_t(double(a_->derived_params_.max_data_node_slots) * DNode::kInitDensity_);
      std::vector<size_t> sizes;
      plan(a_->root_node_, nullptr, sizes);
      limit_ = csv_detail::size_limit(sizes, alpha_, up_, budget_s_, max_subtree_, plan_s_);
      pass_t0_ = std::chrono::steady_clock::now();
      visit(a_->root_node_, nullptr);  // the paper: on ALEX the pass starts at the bottom level and stops below the root
      n_virtual_ = virt_.size();
      if (!keep_virtual_) std::vector<K>().swap(virt_);
      // the pass's work arrays (up to CSV_MAX_SUBTREE entries each) are given back: memory_consumption does not count them
      std::vector<DNode*>().swap(leaves_), std::vector<MNode*>().swap(models_);
      std::vector<V>().swap(kvs_), std::vector<V>().swap(vals_);
      std::vector<K>().swap(reg_);
      std::vector<uint8_t>().swap(isv_);
      scratch_ = csv_detail::Scratch();
    }
    const double pass_ms = ms_since(t0) - base_ms;
    if (size_t(a_->stats_.num_keys) != m) throw std::runtime_error(name_ + ": ALEX counts another number of keys than were loaded");
    for (size_t i = 0; i < m; ++i) {
      const P* p = a_->get_payload(kv[i].first);
      if (!p || *p != kv[i].second) throw std::runtime_error(name_ + ": a loaded key is not found after the CSV pass");
    }
    std::fprintf(stderr,
                 "gret: %s reimplementation of CSV (arXiv 2408.06134) on GRE's ALEX: n=%zu in_alex=%zu side=%zu alpha=%.2f lambda=%s c=%g w=%g,%g "
                 "candidates=%zu no_room=%zu too_big=%zu over_budget=%zu late=%zu smoothed=%zu accepted=%zu virtual=%zu (%.2f%% of n) promoted=%zu "
                 "nodes=%d->%d avg_level=%.4f->%.4f bytes_per_key=%.2f size_limit=%zu plan_s=%.1f gap_ns=%.2f base_ms=%.1f csv_ms=%.1f verify_ms=%.1f\n",
                 name_.c_str(), n, m, side_.size(), alpha_, up_ ? "ceil" : "floor", cost_c_, w_search_, w_level_, st_.candidates, st_.no_room, st_.too_big, st_.over_budget, st_.late, st_.smoothed, st_.accepted, n_virtual_,
                 n ? 100.0 * double(n_virtual_) / double(n) : 0.0, st_.promoted, nodes0, a_->stats_.num_data_nodes + a_->stats_.num_model_nodes, level0, mean_level(),
                 n ? double(memory_consumption()) / double(n) : 0.0, limit_, plan_s_, limit_ ? 1e9 * csv_detail::gap_seconds() : 0.0, base_ms, pass_ms, ms_since(t0) - base_ms - pass_ms);
    side_reported_ = side_.size();
  }

  bool get(K key, P& val, Param* = nullptr) override {
    if (__builtin_expect(key >= lo_ && key <= hi_, 1)) {
      if (const P* p = a_->get_payload(key)) {  // the authors' lookup
        val = *p;
        return true;
      }
    }
    if (__builtin_expect(side_.empty(), 1)) return false;
    const auto it = side_.find(key);
    if (it == side_.end()) return false;
    val = it->second;
    return true;
  }

  // ALEX is given a key only where its own code is sound for it (docs/sota/csv.md, "What ALEX is not given"): inside its key
  // domain when its root is a model node (a key outside makes it expand the root, whose arithmetic casts a ratio of key
  // distances to int and asserts on a key just past a wide domain), and inside the keys its root holds while the root is
  // a data node (when that node splits, ALEX fixes its key domain to the keys it holds and routes the new key by a model).
  // Everything else lives in the side store.
  bool put(K key, P value, Param* = nullptr) override {
    if (!side_.empty() && side_.count(key)) return false;
    bool take = n_alex_ > 0 && n_alex_ < kMaxKeys && key >= lo_ && key <= hi_;
    K first = 0, last = 0;
    const bool leaf_root = a_->root_node_->is_leaf_;
    if (take && leaf_root) {
      const DNode* d = static_cast<const DNode*>(a_->root_node_);
      first = d->first_key(), last = d->last_key();
      take = key >= first && key <= last;
    }
    if (take && crowded(key)) {
      if (a_->get_payload(key)) return false;
      take = false;
    }
    if (!take) {
      if (n_alex_ >= kMaxKeys && key >= lo_ && key <= hi_ && a_->get_payload(key)) return false;
      side_.emplace(key, value);
      ++put_side_;
      return true;
    }
    const bool inserted = a_->insert(key, value).second;  // the authors' insert
    // The key domain ALEX has just fixed (alex.h:1167), whether or not the key went in: a full data node is split before it
    // is searched for the key (alex_nodes.h, AlexDataNode::insert), so a put of a key that is already there can be the call
    // that turns the root into a model node.
    if (leaf_root && !a_->root_node_->is_leaf_) lo_ = first, hi_ = last;
    if (!inserted) return false;
    ++n_alex_;
    return true;
  }

  bool update(K key, P value, Param* = nullptr) override {
    if (key >= lo_ && key <= hi_) {
      if (P* p = a_->get_payload(key)) {
        *p = value;
        return true;
      }
    }
    if (side_.empty()) return false;
    const auto it = side_.find(key);
    if (it == side_.end()) return false;
    it->second = value;
    return true;
  }

  bool remove(K key, Param* = nullptr) override {
    if (key >= lo_ && key <= hi_ && a_->erase(key) > 0) {  // the authors' erase
      --n_alex_;
      return true;
    }
    return !side_.empty() && side_.erase(key) != 0;
  }

  size_t scan(K key, size_t n, std::pair<K, P>* out, Param* = nullptr) override {
    size_t got = 0;
    if (n_alex_ && key <= hi_) {
      auto it = seek(std::max(key, lo_));  // the authors' iterator
      for (; got < n && !it.is_end(); ++got) {
        out[got] = std::pair<K, P>((*it).first, (*it).second);
        it++;
      }
    }
    return side_.empty() ? got : csv_detail::merge_side(side_, key, n, out, got);
  }

  // as GRE counts ALEX (model_size() + data_size(): every node, every slot, the bitmaps), plus the side store and this object
  long long memory_consumption() override {
    if (side_.size() != side_reported_) {
      side_reported_ = side_.size();
      std::fprintf(stderr, "gret: %s now in_alex=%zu side=%zu (%zu puts went to the side store since the bulk load)\n", name_.c_str(), n_alex_, side_.size(), put_side_);
    }
    return a_->model_size() + a_->data_size() + (long long)(sizeof(*this) + sizeof(A) + side_.size() * kMapNode);
  }

  size_t in_base() const { return n_alex_; }
  size_t in_side() const { return side_.size(); }
  size_t n_virtual() const { return n_virtual_; }
  const csv_detail::PassStats& pass() const { return st_; }
  void keep_virtual(bool on) { keep_virtual_ = on; }
  const std::vector<K>& virtual_keys() const { return virt_; }
  bool root_is_model_node() const { return !a_->root_node_->is_leaf_; }  // false while ALEX is one data node
  // every data node's keys ascending and inside the linked list's order, counts consistent; and, once ALEX's root is a model
  // node, ALEX's key domain is exactly [lo_, hi_] and ALEX has never been handed a key outside it (put's contract): for tests
  bool consistent() const {
    if (!a_->root_node_->is_leaf_) {
      const auto& d = (*a_).*kDomain;
      if (d.key_domain_min_ != lo_ || d.key_domain_max_ != hi_ || d.num_keys_below_key_domain != 0 || d.num_keys_above_key_domain != 0) return false;
    }
    size_t keys = 0, leaves = 0;
    K last = 0;
    bool any = false;
    for (const DNode* d = first_leaf(); d; d = d->next_leaf_) {
      ++leaves;
      if (d->next_leaf_ && d->next_leaf_->prev_leaf_ != d) return false;
      int seen = 0;
      for (int i = 0; i < d->data_capacity_; ++i) {
        if (!d->check_exists(i)) continue;
        const K k = d->get_key(i);
        if (any && k <= last) return false;
        last = k, any = true, ++seen;
      }
      if (seen != d->num_keys_) return false;
      keys += size_t(seen);
    }
    // (not ALEX's num_data_nodes: its constructor counts the empty root that bulk_load then replaces)
    return leaves > 0 && keys == n_alex_ && size_t(a_->stats_.num_keys) == n_alex_;
  }

 private:
  static double ms_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }

  // ALEX's models run in double. Keys closer than doubles separate are told apart by the search inside a data node, which is
  // fine for a few of them (real data has such pairs) and is not for a run: a data node whose keys all predict the same
  // slots keeps failing its cost check, and Alex::insert splits it again and again without end (alex.h:1158, seen on a
  // run of 1,500 consecutive keys near 2^63). So ALEX is given at most kRun keys inside any window of about four double
  // steps; a key that would be one more goes to the side store. Keys below 2^53 are exact in double and never counted.
  static constexpr size_t kRun = 4;
  static K window(K k) { return k < (K(1) << 53) ? 0 : 4 * (k >> 52); }
  bool crowded(K key) {  // kRun keys of ALEX already lie within one window of key (key is inside [lo_, hi_])
    const K w = window(key);
    if (!w) return false;
    const K to = hi_ - key > w ? key + w : hi_;
    size_t c = 0;
    for (auto it = seek(key - lo_ > w ? key - w : lo_); !it.is_end() && (*it).first <= to; it++)
      if (++c >= kRun) return true;
    return false;
  }

  // The first key >= key as an ALEX iterator (key is inside [lo_, hi_]). Not Alex::lower_bound, which GRE's interface scans
  // with: when key is past the last key of its data node and the node's capacity is a multiple of 64,
  // AlexDataNode::find_lower reads the word after the node's bitmap (get_next_filled_position, alex_nodes.h:1525) and, if
  // that word is not zero, returns a slot past the node. Here such a key starts at the next data node that has keys.
  typename A::Iterator seek(K key) {
    DNode* leaf = a_->get_leaf(key);
    if (leaf->num_keys_ > 0 && key <= leaf->last_key()) return typename A::Iterator(leaf, leaf->find_lower(key));
    for (leaf = leaf->next_leaf_; leaf && leaf->num_keys_ == 0;) leaf = leaf->next_leaf_;
    return leaf ? typename A::Iterator(leaf, 0) : a_->end();
  }

  const DNode* first_leaf() const {  // Alex::first_data_node() is private
    const ANode* cur = a_->root_node_;
    while (!cur->is_leaf_) cur = static_cast<const MNode*>(cur)->children_[0];
    return static_cast<const DNode*>(cur);
  }

  double mean_level() const {  // keys weighted by their data node's level (the root is level 0)
    double s = 0, c = 0;
    for (const DNode* d = first_leaf(); d; d = d->next_leaf_) s += double(d->num_keys_) * double(d->level_), c += double(d->num_keys_);
    return c ? s / c : 0.0;
  }

  void gather(ANode* n) {  // the data nodes of a subtree in key order, and its model nodes
    if (n->is_leaf_) return leaves_.push_back(static_cast<DNode*>(n));
    MNode* mn = static_cast<MNode*>(n);
    models_.push_back(mn);
    for (int i = 0; i < mn->num_children_;) {
      ANode* c = mn->children_[i];
      gather(c);
      while (i < mn->num_children_ && mn->children_[i] == c) ++i;
    }
  }

  // the sizes of the subtrees visit() would smooth, for the plan under CSV_BUDGET_S; returns the keys below n
  size_t plan(const ANode* n, const MNode* parent, std::vector<size_t>& sizes) const {
    if (n->is_leaf_) return size_t(static_cast<const DNode*>(n)->num_keys_);
    const MNode* mn = static_cast<const MNode*>(n);
    size_t m = 0;
    for (int i = 0; i < mn->num_children_;) {
      const ANode* c = mn->children_[i];
      m += plan(c, mn, sizes);
      while (i < mn->num_children_ && mn->children_[i] == c) ++i;
    }
    const size_t cap = csv_detail::lambda_of(alpha_, m, up_);
    if (parent && m >= 3 && cap > 0 && m <= max_subtree_ && m + cap <= max_node_keys_) sizes.push_back(m);
    return m;
  }

  // The paper's Algorithm 2 as a post-order walk (see CsvLipp::visit). A model node with its whole subtree becomes one data
  // node holding the subtree's keys and the virtual points ("creating a new leaf-node in place of the parent node").
  void visit(ANode* n, MNode* parent) {
    if (n->is_leaf_) return;
    MNode* mn = static_cast<MNode*>(n);
    const size_t v0 = virt_.size();
    for (int i = 0; i < mn->num_children_;) {
      visit(mn->children_[i], mn);  // may put a data node in the child's slots
      ANode* now = mn->children_[i];
      while (i < mn->num_children_ && mn->children_[i] == now) ++i;
    }
    if (!parent) return;  // the root stays
    ++st_.candidates;
    leaves_.clear(), models_.clear();
    gather(mn);
    size_t m = 0;
    for (const DNode* d : leaves_) m += size_t(d->num_keys_);
    const size_t have = virt_.size() - v0, cap = csv_detail::lambda_of(alpha_, m, up_);
    if (m < 3) { ++st_.no_room; return; }
    const size_t room = cap > have ? cap - have : 0;  // lambda = alpha * |K| for the subtree as a whole, points accepted below included
    if (!room) ++st_.no_room;  // nothing to add; the cost condition is still tested (Algorithm 2, lines 10 to 13)
    if (m > max_subtree_ || m + std::max(cap, have) > max_node_keys_) { ++st_.too_big; return; }  // one ALEX data node cannot hold them
    if (m > limit_) { ++st_.over_budget; return; }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - pass_t0_).count() > 1.5 * budget_s_) { ++st_.late; return; }
    // the subtree's keys, and its cost in ALEX's own terms: per lookup, w_search (20, ALEX's weight) x the expected
    // exponential-search iterations of the data node (the value ALEX computed when it built the node) + w_level (20, ALEX's
    // weight) x the node's level
    kvs_.clear();
    double cost_old = 0;
    for (const DNode* d : leaves_) {
      for (int i = 0; i < d->data_capacity_; ++i)
        if (d->check_exists(i)) kvs_.emplace_back(d->get_key(i), d->get_payload(i));
      cost_old += double(d->num_keys_) / double(m) * (w_search_ * d->expected_avg_exp_search_iterations_ + w_level_ * double(d->level_));
    }
    if (kvs_.size() != m) throw std::runtime_error(name_ + ": an ALEX data node counts another number of keys than it holds");
    reg_.resize(m + have), isv_.resize(m + have);
    for (size_t i = 0, j = v0, o = 0; o < m + have; ++o) {
      const bool v = j < virt_.size() && (i == m || virt_[j] < kvs_[i].first);
      reg_[o] = v ? virt_[j++] : kvs_[i++].first, isv_[o] = v;
    }
    // Algorithm 2 tests the cost condition on the smoothed key set whether or not Algorithm 1 added a point to it
    if (room && csv_detail::smooth(reg_, isv_, room, [](uint64_t, uint64_t, uint64_t) { return true; }, scratch_)) ++st_.smoothed;
    const size_t all = reg_.size();
    vals_.resize(all);
    for (size_t o = 0, i = 0; o < all; ++o) vals_[o] = isv_[o] ? V(reg_[o], 0) : kvs_[i++];
    // the merged node's model and expected search iterations, by ALEX's own functions ("the inbuilt function in ALEX that
    // uses the log2 error", the paper's section 5.1), on the smoothed key set; equation 22 with ALEX's own two weights
    alex::LinearModel<K> model;
    DNode::build_model(vals_.data(), int(all), &model, false);
    alex::DataNodeStats ds;
    const double node_cost = DNode::compute_expected_cost(vals_.data(), int(all), DNode::kInitDensity_, a_->params_.expected_insert_frac, &model, false, &ds);
    const double cost_new = w_search_ * ds.num_search_iterations + w_level_ * double(mn->level_);
    if (!(cost_new - cost_old < cost_c_)) return;
    // the new data node, by ALEX's own data-node bulk load, then the virtual keys erased with its own erase
    DNode* nd = new (std::allocator<DNode>().allocate(1)) DNode(mn->level_, a_->derived_params_.max_data_node_slots, cmp_, alloc_);
    nd->bulk_load(vals_.data(), int(all), &model, false);
    const double contract = nd->contraction_threshold_;
    nd->contraction_threshold_ = 0;  // erase_one_at would repack the node below this many keys, and undo the layout
    bool ok = true;
    for (size_t o = 0; o < all && ok; ++o)
      if (isv_[o]) ok = nd->erase_one(reg_[o]) == 1;
    nd->contraction_threshold_ = contract;
    if (!ok || size_t(nd->num_keys_) != m) {
      nd->~DNode();
      std::allocator<DNode>().deallocate(nd, 1);
      throw std::runtime_error(name_ + ": the merged data node does not hold exactly the subtree's keys");
    }
    nd->duplication_factor_ = mn->duplication_factor_;
    nd->cost_ = node_cost;
    nd->expected_avg_exp_search_iterations_ = ds.num_search_iterations;
    nd->expected_avg_shifts_ = ds.num_shifts;
    for (int i = 0; i < parent->num_children_; ++i)
      if (parent->children_[i] == mn) parent->children_[i] = nd;
    nd->prev_leaf_ = leaves_.front()->prev_leaf_, nd->next_leaf_ = leaves_.back()->next_leaf_;
    if (nd->prev_leaf_) nd->prev_leaf_->next_leaf_ = nd;
    if (nd->next_leaf_) nd->next_leaf_->prev_leaf_ = nd;
    for (const DNode* d : leaves_) st_.promoted += d->level_ > mn->level_ ? size_t(d->num_keys_) : 0;
    for (DNode* d : leaves_) {
      d->~DNode();
      std::allocator<DNode>().deallocate(d, 1);
    }
    for (MNode* x : models_) {
      x->~MNode();
      std::allocator<MNode>().deallocate(x, 1);
    }
    a_->stats_.num_data_nodes -= int(leaves_.size()) - 1;
    a_->stats_.num_model_nodes -= int(models_.size());
    ++st_.accepted;
    st_.nodes_removed += leaves_.size() + models_.size() - 1;
    virt_.resize(v0);
    for (size_t o = 0; o < all; ++o)
      if (isv_[o]) virt_.push_back(reg_[o]);
  }

  std::string name_;
  double alpha_;
  alex::AlexCompare cmp_;        // the data nodes this wrapper creates refer to these two, as ALEX's own refer to its members
  std::allocator<V> alloc_;
  std::unique_ptr<A> a_;         // the authors' index
  std::map<K, P> side_;          // keys ALEX cannot be given (see the head of this file)
  K lo_ = 1, hi_ = 0;            // the keys ALEX may hold lie in [lo_, hi_]: its key domain (empty when lo_ > hi_)
  size_t n_alex_ = 0, put_side_ = 0, side_reported_ = 0, n_virtual_ = 0;
  csv_detail::PassStats st_;
  bool keep_virtual_ = false, up_ = true;
  size_t max_subtree_ = 131072, max_node_keys_ = 0, limit_ = 0;
  double budget_s_ = 1200, cost_c_ = 0, plan_s_ = 0, w_search_ = 20, w_level_ = 20;
  std::chrono::steady_clock::time_point pass_t0_;
  std::vector<DNode*> leaves_;
  std::vector<MNode*> models_;
  std::vector<V> kvs_, vals_;
  std::vector<K> virt_, reg_;
  std::vector<uint8_t> isv_;
  csv_detail::Scratch scratch_;
};

// "csv_lipp", "csv_alex" (alpha = 0.1) and "csv_lipp<PP>", "csv_alex<PP>": alpha = PP / 100, PP in [0, 99], no leading zero.
// nullptr for any other name, and for key or payload types other than uint64_t.
template <class K, class P>
indexInterface<K, P>* make_csv(const std::string& name) {
  if constexpr (!(std::is_same<K, uint64_t>::value && std::is_same<P, uint64_t>::value)) {
    (void)name;
    return nullptr;
  } else {
    const bool lipp = name.compare(0, 8, "csv_lipp") == 0, alx = name.compare(0, 8, "csv_alex") == 0;
    if (!lipp && !alx) return nullptr;
    const std::string d = name.substr(8);
    int pp = 10;
    if (!d.empty()) {
      if (d.size() > 2 || d.find_first_not_of("0123456789") != std::string::npos || (d.size() == 2 && d[0] == '0')) return nullptr;
      pp = std::stoi(d);
    }
    if (lipp) return new CsvLipp(name, pp / 100.0);
    return new CsvAlex(name, pp / 100.0);
  }
}

}  // namespace gret
