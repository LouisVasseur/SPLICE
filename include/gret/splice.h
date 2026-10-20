// SPDX-License-Identifier: MIT
// SpliceIndex: SPLICE's G, T and V on one sorted array of slots. Read-only, exact.
//
//   G  REDUCE: outliers at the ends (peel_outliers, the rule of exceptions.h, exact at build) go to a small sorted
//      array behind one range check. Nothing peeled, nothing paid (the check reads two fields).
//   T  the top: the bulk is cut into segments, each with its own line rank ~ r0 + a (key - x0). Segment starts: an
//      optimal eps-PLA over (key, rank) (PGM-index's make_segmentation, eps the least that fits K segments, each
//      segment's own fitted line, |error| <= eps), or K equi-rank knots (the pw transform's, the chord between them).
//      The build keeps the one whose layout (V) touches fewer lines. A PwlCdf through the segment starts finds a key's
//      segment (PwlCdf::locate: a radix bucket and a few cache-resident steps).
//   V  EXPAND: a segment's keys go into about e times as many slots, each at its predicted slot
//      b + floor(e (r0 + a (key - x0) - f)) (f: the segment's first rank), or at the next free one when that is taken;
//      an empty slot holds a copy of the next key and its payload, so the slots stay sorted. A lookup reads from its
//      predicted slot forward, at most the segment's exact maximum displacement d further, and stops at the first key
//      >= its own: in the common case one slot of one 64-byte line. e per segment, from {1, 1.25, 1.5, 2, 3, 4},
//      minimises the exact count of lines touched by a lookup of each of the segment's keys plus mu * slots, mu the
//      least that keeps the whole index within SPLICE_MAX_BPK bytes per key (default 32); ties go to the smaller e.
//      e = 1 is the packed array: no V.
// Hierarchy: G and the choice of knots are global (one decision per index), e is per segment (about n / K keys), the
// search is per line. Every decision is an exact count at build, never a timing; E[lines] (the mean over the keys of
// the lines a lookup touches) is printed with the build. Segment bases are line-aligned, so the count is exact.
// Names (registry.h), the attribution ladder: splice = G + T + V; spl_t = T alone (no G, e = 1); spl_gt = G + T;
// spl_tv = T + V; splice_eq = splice with K equi-rank knots instead of the PLA's starts (the same K); spl_fixk = splice
// with a fixed granularity (n / 610 keys per segment) instead of the count's. Environment: SPLICE_K=K fixes K (sweeps);
// TRANSFORM_G_E (64) G's per-end cap; SPLICE_MAX_BPK (32) bytes per key; SPLICE_SPACE_PRICE (20, in 1/1000 lines).
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "gret/exceptions.h"
#include "gret/pwlcdf.h"

namespace gret {

template <class T>
struct LineAlloc {  // 64-byte aligned storage, so slot 4i starts a cache line
  using value_type = T;
  LineAlloc() = default;
  template <class U> LineAlloc(const LineAlloc<U>&) {}
  T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(64))); }
  void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t(64)); }
  template <class U> bool operator==(const LineAlloc<U>&) const { return true; }
  template <class U> bool operator!=(const LineAlloc<U>&) const { return false; }
};

template <class K, class P>
class SpliceIndex final : public indexInterface<K, P> {
 public:
  enum Knots : uint8_t { kPla = 0, kEqui = 1 };  // kEqui: the T ablation
  struct Config {
    bool g = true;   // G: peel end outliers
    bool v = true;   // V: per-segment expansion (false: e = 1)
    Knots knots = kPla;
    bool adapt = true;  // T's granularity chosen by the count (false: K = n / 610 segments, the old fixed default)
  };
  explicit SpliceIndex(std::string name = "splice", Config c = {}) : name_(std::move(name)), cfg_(c) {}

  void init(Param* = nullptr) override {}

  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    const size_t E = !cfg_.g ? 0 : env("TRANSFORM_G_E", 64);
    const unsigned Kk = unsigned(env("SPLICE_K", 16384));
    const double bpk = double(env("SPLICE_MAX_BPK", 32));
    size_t lo, hi;
    peel_outliers(kv, n, E, lo, hi);
    exc_.assign(kv, kv + lo);
    exc_.insert(exc_.end(), kv + hi, kv + n);
    n_ = hi - lo;
    const std::pair<K, P>* d = kv + lo;
    lo_ = n_ ? uint64_t(d[0].first) : 1;
    span_ = n_ ? uint64_t(d[n_ - 1].first) - lo_ : 0;
    if (!n_) {  // no bulk: only x = 1 passes the range check, and it meets one dummy line holding key 0
      lo_ = 1, span_ = 0;
      seg_.assign(1, Seg{});
      seg_[0].x0 = 1;
      top_ = PwlCdf::from_knots({1, 2}, {0, kOutMax});
      slots_.assign(4, std::pair<K, P>{});  // one whole line: lookups read lines
      return;
    }
    const double cap_bytes = bpk * double(n);
    // T's granularity, chosen by the same exact count: PLA errors eps = 1, 2, 4, ..., 256 (segments from 8 keys each
    // on average down to 64 segments), each laid out by V; the least E[lines] + price * slots/key +
    // kTopPenalty * log2(top / 1 MiB) wins (a top past 1 MiB stops being cache-resident: measured, see kTopPenalty). SPLICE_K=K instead fixes K
    // segments (eps bisected to the least that fits K, as PGM's segmentation counts them): K sweeps.
    Build best;
    std::vector<double> ka, kr;
    std::vector<uint64_t> kx;
    auto consider = [&](Build&& b) {
      b.cost = (b.lines + space_price() * double(b.total)) / double(n_) + top_penalty(b.top_bytes);
      if (!best.ok || b.cost < best.cost) std::swap(best, b);
    };
    const bool fixedK = std::getenv("SPLICE_K") && *std::getenv("SPLICE_K");
    if (n_ < 3 || cfg_.knots == kEqui) {
      size_t Ke = fixedK ? Kk : std::max<size_t>(1, n_ / 610);
      if (!fixedK && cfg_.knots == kEqui && n_ >= 3) Ke = pla_choice(d, cap_bytes);  // the same segment count as splice
      segments_equi(d, unsigned(std::min<size_t>(Ke, n_)), kx, ka, kr);
      consider(layout(d, kx, ka, kr, cap_bytes, false, 0));
    } else if (fixedK || !cfg_.adapt) {
      const size_t Kf = fixedK ? Kk : std::max<size_t>(1, n_ / 610);
      const size_t eps = eps_for(d, unsigned(std::max<size_t>(1, std::min(Kf, n_))));
      segments_pla(d, eps, kx, ka, kr);
      consider(layout(d, kx, ka, kr, cap_bytes, true, eps));
    } else {
      for (size_t eps : candidates(d)) {
        segments_pla(d, eps, kx, ka, kr);
        consider(layout(d, kx, ka, kr, cap_bytes, true, eps));
      }
    }
    top_ = std::move(best.top);
    seg_ = std::move(best.seg);
    eps_ = best.eps;
    pla_ = best.pla;
    // the slots: each key at its position, each empty slot a copy of the next key (pads after a segment: the next
    // segment's first key), so the array is sorted and every scan stops at the first key >= its own
    slots_.assign(best.total, std::pair<K, P>{});
    size_t i = 0;
    for (size_t j = 0; j < seg_.size(); ++j) {
      const Seg& s = seg_[j];
      const size_t end = best.first[j + 1];
      size_t prev = SIZE_MAX;
      for (size_t r = best.first[j]; r < end; ++r) {
        const size_t pos = place(s, uint64_t(d[r].first), prev);
        while (i < (size_t(s.b4) << 2) + pos) slots_[i++] = d[r];
        slots_[i++] = d[r];
        prev = pos;
      }
    }
    while (i < slots_.size()) slots_[i++] = d[n_ - 1];  // trailing pad (not reachable by a key's scan)
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const size_t nv = best.expanded;
    std::fprintf(stderr, "gret: %s n=%zu exceptions=%zu K=%s knots=%s eps=%zu segments=%zu expanded=%zu (%.1f%%) "
                 "slots/key=%.3f E[lines]=%.4f E[slots]=%.3f top_bytes=%zu top_steps=%u bytes_per_key=%.2f build_ms=%.1f\n",
                 name_.c_str(), n, exc_.size(), fixedK ? std::getenv("SPLICE_K") : "auto", pla_ ? "pla" : "equi", pla_ ? eps_ : 0, seg_.size(), nv,
                 100.0 * double(nv) / double(seg_.size()), double(slots_.size()) / double(n_), best.lines / double(n_),
                 best.scanned / double(n_), top_.bytes() + seg_.size() * sizeof(Seg), top_.search_steps(),
                 double(memory_consumption()) / double(n), ms);
  }

  __attribute__((flatten)) bool get(K key, P& val, Param* = nullptr) override {
    const uint64_t x = uint64_t(key);
    if (__builtin_expect(x - lo_ > span_, 0)) return find_exception(key, val);
    const Seg& s = seg_[top_.locate(x)];
    const size_t off = offset(s, x), b = size_t(s.b4) << 2;
    size_t i = b + (off & ~size_t(3));  // the predicted line
    const size_t end = b + std::min<size_t>(size_t(s.last), off + s.d);
    for (;; i += 4) {  // sorted slots, one line at a time: the first key >= x decides
      const std::pair<K, P>* l = &slots_[i];
      const size_t c = size_t(uint64_t(l[0].first) < x) + size_t(uint64_t(l[1].first) < x) +
                       size_t(uint64_t(l[2].first) < x) + size_t(uint64_t(l[3].first) < x);
      if (c < 4) {
        if (uint64_t(l[c].first) != x) return false;
        val = l[c].second;
        return true;
      }
      if (i + 4 > end) return false;
    }
  }

  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  size_t scan(K, size_t, std::pair<K, P>*, Param* = nullptr) override { return 0; }
  long long memory_consumption() override {
    return (long long)(slots_.size() * sizeof(std::pair<K, P>) + seg_.size() * sizeof(Seg) + top_.bytes() +
                       exc_.size() * sizeof(std::pair<K, P>));
  }

 private:
  struct Seg {       // 32 bytes in a 64-byte-aligned array: an entry never straddles two cache lines
    uint64_t x0;     // the segment's first key
    double a;        // predicted slot offset: floor(c + a (key - x0)), clamped to [0, last]
    float c;
    uint32_t b4;     // first slot / 4 (segments start on a line)
    uint32_t last;   // last slot offset
    uint32_t d;      // exact maximum displacement of a key after its predicted slot
  };
  static_assert(sizeof(Seg) == 32, "Seg: 32 bytes");
  static constexpr double kE[] = {1, 1.25, 1.5, 2, 3, 4};
  static constexpr int kNE = 6;

  struct Build {
    bool ok = false, pla = false;
    PwlCdf top;
    std::vector<Seg, LineAlloc<Seg>> seg;
    std::vector<size_t> first;      // first rank of each segment, and n at the end
    size_t total = 0, eps = 0, top_bytes = 0, expanded = 0;
    double lines = 0, scanned = 0, cost = 0;
  };

  static size_t env(const char* k, size_t def) {
    const char* e = std::getenv(k);
    return e && *e ? size_t(std::strtoull(e, nullptr, 10)) : def;
  }
  // the predicted slot offset, clamped to [0, last]; at build last = UINT32_MAX (not known yet): for every key the
  // clamp to the final last changes nothing, since floor(t) <= its slot <= last
  static uint32_t offset(const Seg& s, uint64_t x) {
    const double t = double(s.c) + s.a * double(x - s.x0);
    return !(t > 0) ? 0 : t >= double(s.last) ? s.last : uint32_t(t);
  }
  // build-time placement of one key: the first free slot of its predicted line or later (the slots before it in its
  // line are read by its lookup anyway), so a dense run packs its line instead of drifting right
  static size_t place_at(size_t p, size_t prev) {
    const size_t l = p & ~size_t(3);
    return prev == SIZE_MAX || l > prev ? l : prev + 1;
  }
  static size_t place(const Seg& s, uint64_t x, size_t prev) { return place_at(offset(s, x), prev); }

  // T: PLA segments at error eps, each with PGM's fitted line (slope, rank at its first key)
  void segments_pla(const std::pair<K, P>* d, size_t eps, std::vector<uint64_t>& kx, std::vector<double>& ka,
                    std::vector<double>& kr) const {
    kx.clear(); ka.clear(); kr.clear();
    auto in = [&](size_t i) { return std::pair<uint64_t, size_t>(uint64_t(d[i].first), i); };
    pgm::internal::make_segmentation(n_, eps, in, [&](const auto& sg) {
      const uint64_t k = sg.get_first_x();
      if (!kx.empty() && k <= kx.back()) return;
      const auto [slope, icpt] = sg.get_floating_point_segment(k);
      kx.push_back(k); ka.push_back(double(slope)); kr.push_back(double(icpt));
    });
  }
  // T ablation: K equi-rank knots (the pw transform's), lines = chords (filled in layout)
  void segments_equi(const std::pair<K, P>* d, unsigned Ke, std::vector<uint64_t>& kx, std::vector<double>& ka,
                     std::vector<double>& kr) const {
    kx.clear(); ka.clear(); kr.clear();
    PwlCdf eq = PwlCdf::build_by(n_, [&](size_t i) { return uint64_t(d[i].first); }, std::max(1u, Ke));
    for (size_t i = 0; i + 1 < eq.knots(); ++i) kx.push_back(eq.knot_x(i));
    if (kx.empty()) kx.push_back(uint64_t(d[0].first));
  }
  size_t pla_count(const std::pair<K, P>* d, size_t eps) const {
    auto in = [&](size_t i) { return std::pair<uint64_t, size_t>(uint64_t(d[i].first), i); };
    return pgm::internal::make_segmentation(n_, eps, in, [](auto&&) {});
  }
  size_t eps_for(const std::pair<K, P>* d, unsigned Kc) const {  // the least eps whose PLA fits Kc segments
    size_t lo = 0, hi = 1;
    while (pla_count(d, hi) > Kc) lo = hi, hi *= 2;
    while (hi - lo > 1) { const size_t mid = lo + (hi - lo) / 2; (pla_count(d, mid) > Kc ? lo : hi) = mid; }
    return hi;
  }
  std::vector<size_t> candidates(const std::pair<K, P>* d) const {  // eps = 2^i with 64 <= segments <= n / 8
    std::vector<size_t> c;
    for (size_t eps = 1; eps <= 256; eps *= 2) {
      const size_t segs = pla_count(d, eps);
      if (segs * 8 > n_) continue;
      c.push_back(eps);
      if (segs < 64) break;
    }
    if (c.empty()) c.push_back(256);
    return c;
  }
  size_t pla_choice(const std::pair<K, P>* d, double cap_bytes) const {  // splice's segment count, for splice_eq
    std::vector<uint64_t> kx; std::vector<double> ka, kr;
    double bc = 0; size_t bs = 1; bool any = false;
    for (size_t eps : candidates(d)) {
      segments_pla(d, eps, kx, ka, kr);
      Build b = layout(d, kx, ka, kr, cap_bytes, true, eps);
      const double c = (b.lines + space_price() * double(b.total)) / double(n_) + top_penalty(b.top_bytes);
      if (!any || c < bc) bc = c, bs = b.seg.size(), any = true;
    }
    return bs;
  }
  // the price of a slot in lines (SPLICE_SPACE_PRICE, in 1/1000 lines; default 20: an extra slot per key must save
  // 0.02 lines per lookup), so V expands only where it pays, and the cap binds only beyond that
  static double space_price() { return double(env("SPLICE_SPACE_PRICE", 20)) / 1000.0; }
  // a top past 1 MiB stops being cache-resident: in lines per doubling of its bytes, measured on the Mac (10M keys,
  // genome: K 16384 -> 65536 -> 262144 cost +6 and +12 ns at ~30 ns per line touched)
  static constexpr double kTopPenalty = 0.15;
  static double top_penalty(size_t bytes) {
    return bytes > (size_t(1) << 20) ? kTopPenalty * std::log2(double(bytes) / double(size_t(1) << 20)) : 0;
  }

  // V on given segments: per segment and e, the exact lines touched and slots used; mu bisected to the least that
  // fits the byte cap; then each segment's slots, displacement and clamp, and the top that locates them.
  Build layout(const std::pair<K, P>* d, std::vector<uint64_t> kx, std::vector<double> ka, std::vector<double> kr,
               double cap_bytes, bool pla, size_t eps) const {
    Build B;
    B.pla = pla;
    B.eps = eps;
    if (uint64_t(d[n_ - 1].first) > kx.back()) kx.push_back(uint64_t(d[n_ - 1].first));
    const size_t S = std::max<size_t>(1, kx.size() - 1);  // intervals; the last one includes the last key
    B.first.assign(S + 1, n_);
    {
      size_t r = 0;
      for (size_t j = 0; j < S; ++j) {
        while (r < n_ && uint64_t(d[r].first) < kx[j]) ++r;
        B.first[j] = r;
      }
      B.first[S] = n_;
    }
    if (!pla) {  // equi-rank: the chord between consecutive knots (the last interval ends at the last key)
      ka.resize(S); kr.resize(S);
      for (size_t j = 0; j < S; ++j) {
        const size_t f = B.first[j], g = j + 1 < S ? B.first[j + 1] : n_ - 1;
        const uint64_t xe = j + 1 < S ? kx[j + 1] : uint64_t(d[n_ - 1].first);
        ka[j] = xe > kx[j] ? double(g - f) / double(xe - kx[j]) : 0;
        kr[j] = double(f);
      }
    }
    // a line that extrapolates far past its segment (the last key's interval when G is off, e.g. fb's 2^64 - 1) is
    // replaced by the chord through the segment's first and last key, so offsets stay below 4 (2 nk + 256)
    for (size_t j = 0; j < S; ++j) {
      const size_t f = B.first[j], g = B.first[j + 1], nk = g - f;
      if (!nk) continue;
      const Seg s = line(kx[j], ka[j], kr[j], f, 1);
      bool bad = false;
      for (size_t r = f; r < g && !bad; ++r) bad = double(s.c) + s.a * double(uint64_t(d[r].first) - s.x0) > 2.0 * nk + 256;
      if (bad) {
        const uint64_t xe = uint64_t(d[g - 1].first);
        ka[j] = xe > kx[j] ? double(nk - 1) / double(xe - kx[j]) : 0;
        kr[j] = double(f);
      }
    }
    // V: exact lines touched and slots used per (segment, e)
    const int ne = cfg_.v ? kNE : 1;
    std::vector<double> L(S * ne), Sl(S * ne), Sc(S * ne);
    for (size_t j = 0; j < S; ++j) {
      const size_t f = B.first[j], g = B.first[j + 1], nk = g - f;
      for (int ei = 0; ei < ne; ++ei) {
        const Seg s = line(kx[j], ka[j], kr[j], f, kE[ei]);
        size_t prev = SIZE_MAX;
        double lines = 0, sc = 0;
        for (size_t r = f; r < g; ++r) {
          const size_t p = offset(s, uint64_t(d[r].first));
          const size_t pos = place_at(p, prev);
          lines += double((pos >> 2) - (p >> 2) + 1);
          sc += double(pos - (p & ~size_t(3)) + 1);
          prev = pos;
        }
        L[j * ne + ei] = lines;
        Sc[j * ne + ei] = sc;
        Sl[j * ne + ei] = double(nk ? ((prev + 1 + 3) & ~size_t(3)) : 0);
      }
    }
    const double fixed = double(S * sizeof(Seg) + (S + 1) * 16 + exc_.size() * sizeof(std::pair<K, P>));
    const double cap_slots = std::max(0.0, (cap_bytes - fixed) / double(sizeof(std::pair<K, P>)));
    auto choose = [&](double mu, std::vector<int>& pick) {
      double slots = 0;
      for (size_t j = 0; j < S; ++j) {
        int b = 0;
        for (int ei = 1; ei < ne; ++ei)
          if (L[j * ne + ei] + mu * Sl[j * ne + ei] < L[j * ne + b] + mu * Sl[j * ne + b]) b = ei;
        pick[j] = b;
        slots += Sl[j * ne + b];
      }
      return slots;
    };
    std::vector<int> pick(S, 0);
    const double mu0 = space_price();
    if (choose(mu0, pick) > cap_slots) {  // mu: the least >= mu0 that fits the cap (e = 1 everywhere if nothing fits)
      double lo = mu0, hi = std::max(1.0, 4 * mu0);
      while (choose(hi, pick) > cap_slots && hi < 1e6) lo = hi, hi *= 4;
      for (int it = 0; it < 50; ++it) { const double mid = (lo + hi) / 2; (choose(mid, pick) > cap_slots ? lo : hi) = mid; }
      choose(hi, pick);
    }
    // the segments, line-aligned
    B.seg.resize(S);
    size_t base = 0;
    for (size_t j = 0; j < S; ++j) {
      const int ei = pick[j];
      const size_t f = B.first[j], g = B.first[j + 1], nk = g - f;
      Seg s = line(kx[j], ka[j], kr[j], f, kE[ei]);
      size_t prev = SIZE_MAX, dmax = 0;
      for (size_t r = f; r < g; ++r) {
        const size_t p = offset(s, uint64_t(d[r].first));
        const size_t pos = place_at(p, prev);
        dmax = std::max(dmax, pos > p ? pos - p : 0);
        prev = pos;
      }
      if (base >> 2 > UINT32_MAX) throw std::length_error("SpliceIndex: more than 2^34 slots");
      s.b4 = uint32_t(base >> 2);
      s.last = nk ? uint32_t(prev) : 0;
      s.d = uint32_t(dmax);
      B.seg[j] = s;
      B.expanded += ei != 0;
      base += nk ? (size_t(s.last) + 1 + 3) & ~size_t(3) : 0;
      B.lines += L[j * ne + ei];
      B.scanned += Sc[j * ne + ei];
    }
    B.total = base;
    if (kx.size() == 1) {  // one key: a dummy second knot (before it at 2^64 - 1); locate gives interval 0 either way
      if (kx[0] < UINT64_MAX) kx.push_back(kx[0] + 1);
      else kx.insert(kx.begin(), kx[0] - 1);
    }
    B.top = PwlCdf::locator(std::move(kx));
    B.top_bytes = B.top.bytes() + B.seg.size() * sizeof(Seg);
    B.ok = true;
    return B;
  }
  static Seg line(uint64_t x0, double a, double r0, size_t f, double e) {  // last = UINT32_MAX: the build's clamp
    Seg s{};
    s.x0 = x0;
    s.a = e * a;
    s.c = float(e * (r0 - double(f)));
    s.last = UINT32_MAX;
    return s;
  }
  bool find_exception(K key, P& val) const {
    const auto it = std::lower_bound(exc_.begin(), exc_.end(), key,
                                     [](const std::pair<K, P>& a, K k) { return a.first < k; });
    if (it == exc_.end() || it->first != key) return false;
    val = it->second;
    return true;
  }

  std::string name_;
  Config cfg_;
  bool pla_ = false;
  size_t n_ = 0, eps_ = 0;
  uint64_t lo_ = 1, span_ = 0;
  PwlCdf top_;
  std::vector<Seg, LineAlloc<Seg>> seg_;  // 64-byte aligned: 32-byte entries never straddle a line
  std::vector<std::pair<K, P>, LineAlloc<std::pair<K, P>>> slots_;  // 64-byte aligned: 4 slots are one line
  std::vector<std::pair<K, P>> exc_;
};

}  // namespace gret
