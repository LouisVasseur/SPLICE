// SPDX-License-Identifier: MIT
// SpliceOn<K, P, Base>: SPLICE's G, T and V on one of GRE's own indexes (Base, unchanged), each part used only where
// it pays for this index on these keys. Read-only, exact for every loaded key.
//
//   G  the end outliers peeled into a short sorted exception list behind one range check (peel_outliers, the rule of
//      exceptions.h and SpliceIndex). The rule is its own gate: it peels nothing unless the full span is more than twice
//      the span without the 64 keys at each end (Lean: ks_sandwich, endpointErr_ge: one far key costs the end-to-end
//      line about n G / (S + G) ranks).
//   T  a monotone transform in front of the base: a PwlCdf through the knots of an eps-PLA of the bulk (SPLICE's
//      segments, PwlCdf::from_knots), the base built on the transformed keys (TransformIndex). The granularity: for
//      eps = 4, 16, 64, 256, 1024 the endpoint error e of the transformed bulk is counted exactly (ks_sandwich: e is
//      n D_KS of the transformed keys to within one rank), priced by Lean's search bound 2 ceil(log2(e + 2)) + 2 probes
//      (lookup_cost_of_ks) plus T's own dependent accesses at the machine's price (latency.h); the cheapest eps is
//      the candidate. The gate: the base is built both ways and a fixed sample of lookups timed on each; T is kept only
//      if it is faster by more than the margin (gate_sound with measured costs: U + c + delta < L).
//   V  virtual points (VirtualIndex, CSV's Algorithm 1) among the keys the base sees, where keys pile onto one
//      predicted slot (Lean: conflict_ge_density). The same gate: built with and without, kept only if timed faster.
//
// The base is never built twice at once (LIPP at 200M holds 24 GB): a candidate that loses is destroyed and the winner
// rebuilt. One line per decision goes to stderr. SPLICE_ON_MARGIN (2, percent) is the margin; SPLICE_ON_FORCE=raw|t
// fixes T's decision and SPLICE_ON_V=0|1 V's (to reproduce a run). Names (registry.h): sp_t_<base>, sp_gt_<base>,
// sp_gtv_<base>, sp_tv_<base>; base in lipp, btree, alex, pgm.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gret/common.h"
#include "gret/exceptions.h"
#include "gret/latency.h"
#include "gret/pwlcdf.h"
#include "gret/transform_index.h"
#include "gret/vpoints.h"

namespace gret {

template <class K, class P, class Base>
class SpliceOn final : public indexInterface<K, P> {
 public:
  struct Config {
    bool g = true, t = true, v = true;
    bool raw_ties_veto = false;  // LIPP where long double is double: the raw base cannot hold keys that tie as doubles
  };
  using Tb = TransformIndex<K, P, Base, PwlCdf>;
  using Vb = VirtualIndex<K, P, Base>;
  using Tv = TransformIndex<K, P, Vb, PwlCdf>;
  enum class Form : uint8_t { raw, t, v, tv };

  SpliceOn(std::string name, std::string base, Config c) : name_(std::move(name)), base_(std::move(base)), cfg_(c) {}
  SpliceOn(const SpliceOn&) = delete;
  SpliceOn& operator=(const SpliceOn&) = delete;

  void init(Param* = nullptr) override {}

  void bulk_load(std::pair<K, P>* kv, size_t n, Param* param = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    if (n == 0) die(name_ + ": bulk_load of 0 keys");
    size_t lo = 0, hi = n;
    if (cfg_.g) peel_outliers(kv, n, env_size("TRANSFORM_G_E", 64), lo, hi);
    exc_.assign(kv, kv + lo);
    exc_.insert(exc_.end(), kv + hi, kv + n);
    std::pair<K, P>* d = kv + lo;
    const size_t m = hi - lo;
    lo_ = cfg_.g ? uint64_t(d[0].first) : 0;
    span_ = cfg_.g ? uint64_t(d[m - 1].first) - lo_ : UINT64_MAX;
    q_ = sample(d, m);
    margin_ = double(env_size("SPLICE_ON_MARGIN", 2)) / 100.0;
    // G's own effect, for the attribution: when G peeled keys, the base on every key, timed on the same sample (the
    // bulk's keys pass the range check, then reach this full build). Not a decision: G's rule decides
    double ns_plain = -1;
    if (cfg_.g && m < n && !(cfg_.raw_ties_veto && double_ties(kv, n)) && !std::getenv("SPLICE_ON_NO_PLAIN")) {
      build(Form::raw, kv, n, param);
      ns_plain = time_form();
      drop();
    }
    const char* force = std::getenv("SPLICE_ON_FORCE");
    const char* fv = std::getenv("SPLICE_ON_V");

    // T: the candidate granularity by the counted bound, then the timed gate against the raw base. A form that wins
    // stays built (no rebuild); one that loses is dropped before the next is built (one base in memory at a time)
    bool use_t = false;
    double ns_raw = -1, ns_t = -1, ns_v = -1;
    const bool raw_ok = !(cfg_.raw_ties_veto && double_ties(d, m));
    if (!cfg_.t && !raw_ok) die(name_ + ": the raw " + base_ + " cannot hold these keys (they tie as doubles); use a name with T");
    if (cfg_.t) pick_t(d, m);
    if (cfg_.t && !tr_ && !raw_ok) die(name_ + ": no PLA knots for T and the raw base cannot hold these keys");
    if (cfg_.t && tr_) {
      if (!raw_ok || (force && std::string(force) == "t")) use_t = true;
      else if (!(force && std::string(force) == "raw")) {
        build(Form::raw, d, m, param);
        ns_raw = time_form();
        drop();
        build(Form::t, d, m, param);
        ns_t = time_form();
        use_t = ns_t < ns_raw * (1 - margin_);
        if (!use_t) drop();
      }
    }
    // V: with and without, on whatever T decided
    bool use_v = false;
    const Form f0 = use_t ? Form::t : Form::raw, f1 = use_t ? Form::tv : Form::v;
    if (cfg_.v) {
      if (fv && *fv) use_v = std::string(fv) == "1";
      else {
        double ns0 = use_t ? ns_t : ns_raw;
        if (ns0 < 0) {  // T forced or off: time the form without V first
          if (!built()) build(f0, d, m, param);
          ns0 = time_form();
          (use_t ? ns_t : ns_raw) = ns0;
        }
        drop();
        build(f1, d, m, param);
        ns_v = time_form();
        use_v = ns_v < ns0 * (1 - margin_);
        if (!use_v) drop();
      }
    }
    const Form fin = use_v ? f1 : f0;
    if (!built() || form_ != fin) {
      drop();
      build(fin, d, m, param);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr,
                 "gret: %s n=%zu exceptions=%zu T=%s eps=%zu knots=%zu e_raw=%.0f e_t=%.0f probes_raw=%.0f probes_t=%.1f "
                 "V=%s ns_plain=%.1f ns_raw=%.1f ns_t=%.1f ns_v=%.1f margin=%.0f%% build_ms=%.1f\n",
                 name_.c_str(), n, exc_.size(), !cfg_.t ? "off" : use_t ? "on" : "no", t_eps_,
                 tr_ ? tr_->knots() : size_t(0), e_raw_, e_t_, probes(e_raw_), t_score_, !cfg_.v ? "off" : use_v ? "on" : "no",
                 ns_plain, ns_raw, ns_t, ns_v, 100 * margin_, ms);
  }

  // The raw base's lookup is the only one inlined here: inlining every form's lookup into one function costs the
  // common path throughput (measured on LIPP, fb at 200M); the forms with T or V are one call further
  __attribute__((flatten)) bool get(K key, P& val, Param* param = nullptr) override {
    if (__builtin_expect(uint64_t(key) - lo_ > span_, 0)) return find_exception(key, val);
    if (__builtin_expect(form_ == Form::raw, 1)) return raw_->index.Base::get(key, val, param);
    return get_other(key, val, param);
  }

  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  size_t scan(K, size_t, std::pair<K, P>*, Param* = nullptr) override { return 0; }
  long long memory_consumption() override {
    long long b = 0;
    switch (form_) {
      case Form::raw: b = raw_->index.Base::memory_consumption(); break;
      case Form::t: b = t_->memory_consumption(); break;
      case Form::v: b = v_->memory_consumption(); break;
      default: b = tv_->memory_consumption();
    }
    return b ? b + (long long)(exc_.size() * sizeof(std::pair<K, P>)) : 0;  // 0: the base does not report
  }

 private:
  __attribute__((noinline)) bool get_other(K key, P& val, Param* param) {
    switch (form_) {
      case Form::t: return t_->Tb::get(key, val, param);
      case Form::v: return v_->Vb::get(key, val, param);
      default: return tv_->Tv::get(key, val, param);
    }
  }
  struct Raw final {  // final: destroying a bare Base (polymorphic, non-virtual destructor) in optional warns
    Base index;
  };

  static size_t env_size(const char* k, size_t def) {
    const char* e = std::getenv(k);
    return e && *e ? size_t(std::strtoull(e, nullptr, 10)) : def;
  }
  static bool double_ties(const std::pair<K, P>* d, size_t m) {
    for (size_t i = 0; i + 1 < m; ++i)
      if (double(d[i].first) == double(d[i + 1].first)) return true;
    return false;
  }
  // Lean's bound on a search from a clamped linear prediction whose endpoint error is e (lookup_cost_of_ks, with
  // floor(n D) <= e + 1 by ks_sandwich): 2 ceil(log2(floor(e + 1) + 2)) + 2 probes
  static double probes(double e) { return 2 * std::ceil(std::log2(std::floor(e + 1) + 2)) + 2; }

  // T's candidate: eps-PLA knots through each segment's first key (and the last key), y = rank scaled to [0, 2^52 - 1];
  // the exact endpoint error of the transformed bulk, priced in probes, plus T's own accesses; the cheapest eps
  void pick_t(const std::pair<K, P>* d, size_t m) {
    e_raw_ = endpoint_err(d, m, nullptr);
    t_score_ = -1;
    // only maps small enough to stay in cache (SPLICE_ON_MAX_KNOTS, 65536 knots, about 2 MB): a map that misses cache
    // costs each lookup more than the probes it saves (fb at eps 4: 8.3M knots, 3.4x slower on LIPP)
    const size_t max_knots = env_size("SPLICE_ON_MAX_KNOTS", 65536);
    for (size_t eps : {size_t(4), size_t(16), size_t(64), size_t(256), size_t(1024), size_t(4096), size_t(16384)}) {
      std::vector<uint64_t> x, y;
      auto in = [&](size_t i) { return std::pair<uint64_t, size_t>(uint64_t(d[i].first), i); };
      std::vector<size_t> starts;
      pgm::internal::make_segmentation(m, eps, in, [&](const auto& sg) {
        const uint64_t k = sg.get_first_x();
        if (x.empty() || k > x.back()) x.push_back(k);
      });
      if (x.empty() || x.back() != uint64_t(d[m - 1].first)) x.push_back(uint64_t(d[m - 1].first));
      if (x.size() < 2) continue;
      if (x.size() > max_knots) {  // too big for cache: kept only as the fallback when nothing fits (raw vetoed)
        if (!fallback_ || x.size() < fallback_->knots()) {
          bool okf = true;
          std::vector<uint64_t> yf(x.size());
          for (size_t j = 0, r = 0; j < x.size(); ++j) {
            while (uint64_t(d[r].first) < x[j]) ++r;
            yf[j] = m > 1 ? uint64_t((__uint128_t(r) * kOutMax + (m - 1) / 2) / (m - 1)) : 0;
            okf &= j == 0 || yf[j] > yf[j - 1];
          }
          if (okf) fallback_ = std::make_shared<const PwlCdf>(PwlCdf::from_knots(std::vector<uint64_t>(x), std::move(yf))), fallback_eps_ = eps;
        }
        continue;
      }
      y.resize(x.size());
      size_t r = 0;
      for (size_t j = 0; j < x.size(); ++j) {  // the rank of each knot key
        while (uint64_t(d[r].first) < x[j]) ++r;
        y[j] = m > 1 ? uint64_t((__uint128_t(r) * kOutMax + (m - 1) / 2) / (m - 1)) : 0;
      }
      bool ok = true;  // y strictly increasing (ranks are distinct; 2^52 / m > 1 for m < 2^52)
      for (size_t j = 1; j < y.size() && ok; ++j) ok = y[j] > y[j - 1];
      if (!ok) continue;
      PwlCdf t = PwlCdf::from_knots(std::move(x), std::move(y));
      const double e = endpoint_err(d, m, &t);
      const unsigned st = t.search_steps();
      const double c_t = double(2 + (st > 3 ? st - 2 : 1)) * access_cost(double(t.bytes()));
      const double score = probes(e) + c_t;
      if (t_score_ < 0 || score < t_score_) {
        t_score_ = score, e_t_ = e, t_eps_ = eps;
        tr_ = std::make_shared<const PwlCdf>(std::move(t));
      }
    }
    if (!tr_ && fallback_) tr_ = fallback_, t_eps_ = fallback_eps_, e_t_ = endpoint_err(d, m, tr_.get()), t_score_ = -1;
  }
  // max over the bulk of |m u_i - (i + 1)|, u_i the key scaled to [0, 1] by the first and last key (t: after T)
  static double endpoint_err(const std::pair<K, P>* d, size_t m, const PwlCdf* t) {
    const double x0 = double(uint64_t(d[0].first)), xr = double(uint64_t(d[m - 1].first) - uint64_t(d[0].first));
    double e = 0;
    for (size_t i = 0; i < m; ++i) {
      const double u = t ? double((*t)(uint64_t(d[i].first))) / double(kOutMax)
                         : (xr > 0 ? (double(uint64_t(d[i].first)) - x0) / xr : 0);
      e = std::max(e, std::fabs(double(m) * u - double(i + 1)));
    }
    return e;
  }

  // SPLICE_TIME_N (2^20) bulk keys, uniform with a fixed seed: GRE's lookups are uniform over existing keys
  std::vector<K> sample(const std::pair<K, P>* d, size_t m) const {
    std::vector<K> q(std::min<size_t>(m, env_size("SPLICE_TIME_N", size_t(1) << 20)));
    std::mt19937_64 g(42);
    std::uniform_int_distribution<size_t> u(0, m - 1);
    for (K& k : q) k = d[u(g)].first;
    return q;
  }
  // ns per lookup over the sample through the built form, the median of 3 passes; every key must be found
  double time_form() {
    double r[3];
    for (double& t : r) {
      size_t found = 0;
      P v{};
      const auto t0 = std::chrono::steady_clock::now();
      for (const K& k : q_) found += get(k, v);
      t = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / double(q_.size());
      if (found != q_.size()) die(name_ + ": a timed form misses a bulk key");
    }
    std::sort(r, r + 3);
    return r[1];
  }
  void build(Form f, std::pair<K, P>* d, size_t m, Param* param) {
    form_ = f;
    const std::string b = base_;
    switch (f) {
      case Form::raw: raw_.emplace(); raw_->index.init(param); raw_->index.bulk_load(d, m, param); break;
      case Form::t:
        t_.emplace("sp_t_" + b, TransformSpec{0, "", nullptr, tr_});
        t_->init(param);
        t_->bulk_load(d, m, param);
        break;
      case Form::v: v_.emplace("sp_v_" + b); v_->init(param); v_->bulk_load(d, m, param); break;
      default:
        tv_.emplace("sp_tv_" + b, TransformSpec{0, "", nullptr, tr_});
        tv_->init(param);
        tv_->bulk_load(d, m, param);
    }
  }
  void drop() { raw_.reset(), t_.reset(), v_.reset(), tv_.reset(); }
  bool built() const { return raw_ || t_ || v_ || tv_; }
  bool find_exception(K key, P& val) const {
    const auto it = std::lower_bound(exc_.begin(), exc_.end(), key,
                                     [](const std::pair<K, P>& a, K k) { return a.first < k; });
    if (it == exc_.end() || it->first != key) return false;
    val = it->second;
    return true;
  }

  std::string name_, base_;
  Config cfg_;
  Form form_ = Form::raw;
  uint64_t lo_ = 0, span_ = UINT64_MAX;
  double margin_ = 0.02, e_raw_ = 0, e_t_ = 0, t_score_ = 0;
  size_t t_eps_ = 0;
  std::shared_ptr<const PwlCdf> tr_, fallback_;  // T's candidate; the smallest map over the knot cap
  size_t fallback_eps_ = 0;
  std::vector<K> q_;
  std::vector<std::pair<K, P>> exc_;
  std::optional<Raw> raw_;
  std::optional<Tb> t_;
  std::optional<Vb> v_;
  std::optional<Tv> tv_;
};

}  // namespace gret
