// SPDX-License-Identifier: MIT
// VirtualIndex<K, P, Base>: virtual points (the V of SPLICE's G/T/V; in the spirit of CSV, Amarasinghe et al., EDBT
// 2025) in front of one of GRE's own indexes, which stays unchanged. In bulk_load, the sorted keys are cut into
// regions of R keys (TRANSFORM_VP_R, default 256); in each region, CSV's Algorithm 1 adds virtual keys while they lower the
// region's linear-fit loss, up to floor(alpha * R) per region (TRANSFORM_VP_ALPHA, default 0.1, CSV's
// default budget). The base is bulk-loaded on the real and virtual keys; lookups are the base's own, unchanged, so V
// costs nothing per lookup (unlike T): a hit on a virtual key (its payload is the reserved kVirtual) reports false.
//
// The points are chosen by CSV's own Algorithm 1 (csv_smooth below, after CSVa Sec. 4). What differs from CSV, and is
// stated as such: CSV runs Algorithm 1 on the keys of each subtree inside ALEX/LIPP/SALI and keeps a subtree's points
// only if its cost rule accepts the rebuilt node (Algorithm 2); here it runs on fixed regions of R sorted keys, in front
// of an unchanged index, with no acceptance rule (a region keeps the points while they lower its loss).
//
// Exact for every loaded key: virtual keys are integers strictly inside gaps, so they never equal a real key. A real
// payload equal to kVirtual is refused at bulk_load. Read-only: put/update/remove return false, scan returns 0.
// Composes with T: TransformIndex<K, P, VirtualIndex<K, P, Base>, Tr> places V's points among the transformed keys.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace gret {

template <class K, class P, class Base>
class VirtualIndex final : public indexInterface<K, P> {
 public:
  static constexpr P kVirtual = P(~uint64_t(0) - 1);  // 2^64 - 2: PGM reserves 2^64 - 1 (its tombstone)

  explicit VirtualIndex(std::string name = "vp") : name_(std::move(name)) {}
  VirtualIndex(const VirtualIndex&) = delete;
  VirtualIndex& operator=(const VirtualIndex&) = delete;

  void init(Param* param = nullptr) override { base_.init(param); }

  void bulk_load(std::pair<K, P>* kv, size_t n, Param* param = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    const double alpha = env_double("TRANSFORM_VP_ALPHA", 0.1);
    const size_t R = std::max<size_t>(3, size_t(env_double("TRANSFORM_VP_R", 256)));
    std::vector<std::pair<K, P>> out;
    out.reserve(n + size_t(alpha * double(n)) + 16);
    std::vector<uint64_t> reg;           // the region's keys, real and virtual, sorted
    std::vector<uint8_t> isv;            // 1 for a virtual key
    std::vector<double> xs, sufx;
    size_t nvirt = 0, regions_with = 0;
    for (size_t s = 0; s < n; s += R) {
      const size_t e = std::min(n, s + R), m = e - s;
      reg.clear(); isv.assign(m, 0);
      for (size_t i = s; i < e; ++i) {
        if (kv[i].second == kVirtual) die_("a real payload equals the reserved virtual payload");
        reg.push_back(uint64_t(kv[i].first));
      }
      const size_t lambda = size_t(alpha * double(m));
      size_t added = 0;
      if (m >= 3 && lambda > 0 && reg.back() > reg.front()) added = csv_smooth(reg, isv, lambda, xs, sufx);
      if (added) ++regions_with;
      nvirt += added;
      size_t r = s;
      for (size_t j = 0; j < reg.size(); ++j) {
        if (isv[j]) out.push_back({K(reg[j]), kVirtual});
        else out.push_back(kv[r++]);
      }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "gret: %s n=%zu virtual=%zu (%.2f%%) regions_with=%zu/%zu alpha=%g R=%zu vp_ms=%.1f\n",
                 name_.c_str(), n, nvirt, n ? 100.0 * double(nvirt) / double(n) : 0.0, regions_with, (n + R - 1) / R,
                 alpha, R, ms);
    base_.bulk_load(out.data(), out.size(), param);
  }

  bool get(K key, P& val, Param* param = nullptr) override {
    if (!base_.get(key, val, param)) return false;
    return val != kVirtual;
  }
  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  size_t scan(K, size_t, std::pair<K, P>*, Param* = nullptr) override { return 0; }
  long long memory_consumption() override { return base_.memory_consumption(); }

 private:
  // CSV's Algorithm 1 (CSVa Sec. 4, Alg. 1) on one region: L = the OLS-SSE of rank on key over the region's keys and
  // one candidate kv, the line refitted. Candidates per run of free integers [lo, hi] between two keys: both ends when
  // the run holds <= 2 integers; otherwise, if dL/dkv has opposite signs at the two ends, the interior minimum
  // (a closed form: inside a run kv's rank is fixed, and the stationary point of Cxy^2 / Cxx is linear in kv), else
  // both ends. Each round adds the candidate with the least L; stop at lambda points or when L stops falling. Keys are
  // in region coordinates x = (k - k_0) / (k_last - k_0); ranks are 0 .. m. Returns the number of points added.
  static size_t csv_smooth(std::vector<uint64_t>& reg, std::vector<uint8_t>& isv, size_t lambda,
                           std::vector<double>& xs, std::vector<double>& sufx) {
    const uint64_t k0 = reg.front();
    const double span = double(reg.back() - k0);
    auto X = [&](uint64_t k) { return double(k - k0) / span; };
    size_t added = 0;
    double Lprev = -1;
    while (added < lambda) {
      const size_t m = reg.size();
      xs.resize(m);
      double A = 0, B = 0, S1 = 0;
      for (size_t i = 0; i < m; ++i) { xs[i] = X(reg[i]); A += xs[i] * xs[i]; B += xs[i]; S1 += xs[i] * double(i); }
      if (Lprev < 0) {  // L of the region as it is
        const double N0 = double(m), Sy = N0 * (N0 - 1) / 2, Syy = (N0 - 1) * N0 * (2 * N0 - 1) / 6;
        const double cxx = A - B * B / N0, cxy = S1 - B * Sy / N0;
        Lprev = (Syy - Sy * Sy / N0) - (cxx > 0 ? cxy * cxy / cxx : 0);
      }
      sufx.assign(m + 1, 0);
      for (size_t i = m; i-- > 0;) sufx[i] = sufx[i + 1] + xs[i];
      const double N = double(m + 1), Sy = double(m) * double(m + 1) / 2, Syy = double(m) * double(m + 1) * double(2 * m + 1) / 6;
      const double Cyy = Syy - Sy * Sy / N, a2 = 1 - 1 / N, a1 = -2 * B / N, a0 = A - B * B / N;
      double best = Lprev;
      uint64_t bestk = 0;
      size_t bestg = 0;
      bool found = false;
      for (size_t g = 0; g + 1 < m; ++g) {
        if (reg[g + 1] - reg[g] < 2) continue;
        const uint64_t lo = reg[g] + 1, hi = reg[g + 1] - 1;
        const double yv = double(g + 1), c1 = yv - Sy / N, c0 = (S1 + sufx[g + 1]) - B * Sy / N;
        auto L = [&](double x) {
          const double cxx = a0 + a1 * x + a2 * x * x, cxy = c0 + c1 * x;
          return Cyy - (cxx > 0 ? cxy * cxy / cxx : 0);
        };
        auto dL = [&](double x) {  // sign of dL/dx = -d(cxy^2 / cxx)/dx
          const double cxx = a0 + a1 * x + a2 * x * x, cxy = c0 + c1 * x;
          return -(2 * cxy * c1 * cxx - cxy * cxy * (a1 + 2 * a2 * x));
        };
        auto consider = [&](uint64_t k) {
          const double v = L(X(k));
          if (v < best) { best = v; bestk = k; bestg = g; found = true; }
        };
        if (hi - lo <= 1) { consider(lo); if (hi != lo) consider(hi); continue; }
        if (dL(X(lo)) * dL(X(hi)) < 0) {
          const double den = c1 * a1 - 2 * c0 * a2;
          double xstar = den != 0 ? (c0 * a1 - 2 * c1 * a0) / den : X(lo);
          long double kd = (long double)k0 + (long double)xstar * (long double)span;
          uint64_t kf = kd <= (long double)lo ? lo : kd >= (long double)hi ? hi : uint64_t(kd);
          consider(kf);
          if (kf < hi) consider(kf + 1);
        } else {
          consider(lo);
          consider(hi);
        }
      }
      if (!found) break;  // L stops falling
      reg.insert(reg.begin() + std::ptrdiff_t(bestg + 1), bestk);
      isv.insert(isv.begin() + std::ptrdiff_t(bestg + 1), uint8_t(1));
      Lprev = best;
      ++added;
    }
    return added;
  }

  static double env_double(const char* name, double dflt) {
    const char* e = std::getenv(name);
    if (!e || !*e) return dflt;
    char* end = nullptr;
    const double v = std::strtod(e, &end);
    if (!end || *end || !(v >= 0)) die_(std::string(name) + " must be a number >= 0");
    return v;
  }
  [[noreturn]] static void die_(const std::string& msg) {
    std::fprintf(stderr, "gret: %s\n", msg.c_str());
    std::exit(1);
  }

  std::string name_;
  Base base_;
};

}  // namespace gret
