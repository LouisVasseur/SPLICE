// SPDX-License-Identifier: MIT
// rmi<L>: a C++ REIMPLEMENTATION of the two-stage recursive model index (RMI; Kraska, Beutel, Chi, Dean, Polyzotis: "The Case for
// Learned Index Structures", SIGMOD 2018) as its reference implementation builds it (github.com/learnedsystems/RMI; Marcus, Zhang,
// Kraska: "CDFShop", SIGMOD 2020; the RMI of SOSD, Marcus et al., VLDB 2020), as a read-only index in GRE's harness. It is NOT the
// authors' code: the reference is a Rust program that writes C++ for one dataset, and it needs cargo, which this project may not
// run. Nothing of it is copied; what it does was read from its sources at commit 6f03e69 and written again here. Every index built
// from this file reports itself as "<name>(reimplementation)". docs/sota/rmi.md lists every deviation.
//
// The structure: one root model maps a key to one of N = 2^L second-stage models; that one is a line (least squares, on the keys
// the root sends to it) that predicts the key's position, stored with the largest error of that prediction, 24 bytes a leaf as in
// the reference; the position is then searched in [prediction - error, prediction + error]. Here that is a Model for
// SortedBoundIndex (readonly.h): one sorted array of (key, payload) pairs, the same branch-free binary search inside the bound as
// spgm (the static PGM) and the other read-only baselines.
//
// Names (make_rmi<K, P>(name); nullptr for every other name):
//   rmi<L>          N = 2^L leaves, L in [1, 28] without leading zeros, and the root chosen as the reference's optimizer chooses
//                   among models of one size: the one with the lowest average log2 error, among linear, linear_spline, cubic,
//                   robust_linear and radix (the roots of 16 to 32 bytes)
//   rmi<L>lin  spl  cub  rob  rdx     that root, whatever its error: linear (least squares), linear_spline (the line through the
//                   first and the last key), cubic (the reference's cubic, kept only where it beats that line), robust_linear
//                   (least squares without the lowest and the highest 0.01% of the keys), radix (the bits below the keys' common
//                   prefix)
//   rmi<L>rt18  rmi<L>rt22            the reference's radix tables as the root (2^18 or 2^22 entries of 4 bytes: 1 MiB, 16 MiB).
//                   Never chosen by rmi<L>: they are a different size, hence other points of the size/accuracy trade-off
//   <any of these>k                   (rmi20k, rmi20cubk, rmi20rt22k) the leaves of the 2018 paper rather than the reference's: each
//                   trained on its own keys alone, its error the largest over the stored keys alone. Smaller errors where the
//                   keys have wide gaps, and no promise for a key that was never stored; answers stay exact (see below)
// L is the size/accuracy knob (24 * 2^L bytes of leaves). The reference's optimizer walks 2^6 to 2^24; SOSD's configurations for
// books, fb and osm at 200M keys (its scripts/rmi_specs) are between 2^7 and 2^24 leaves, with these roots and linear leaves.
//
// What holds, and why:
//   - a stored key: its position is inside the bound, whatever the models are. The error of a leaf is measured, after training,
//     with the very functions the lookup calls (one fma a model, as in the reference's generated code, so that no compiler
//     contracts or reorders anything), over every key the root sends to that leaf.
//   - a key that was never stored, between the smallest and the largest stored key (SortedBoundIndex asks for no other): its
//     lower bound is inside the bound too, for every root but the cubic. The root is monotone over the keys' range (proved
//     for these roots, in floating point as well) and checked to be non-decreasing on the stored keys, so a query sent to leaf j
//     lies strictly between the last key before that leaf and the first key after it; a leaf's model is a line, monotone as well;
//     so the leaf's error also covers the lower bounds of the gaps inside the leaf and of the two gaps at its ends, taken at their
//     far ends as the reference does (first key of the next leaf minus one, last key of the previous leaf plus one). A leaf that
//     holds no key is a constant: the position of the next key.
//   - the cubic root (rmi<L>cub, and rmi<L> where it picked the cubic): the reference evaluates it on the raw key, by Horner's rule,
//     and so does this. Its end slopes come from one neighbour each and its 'keep it monotonic' step does not make it monotone on
//     every key set (tests/sota/test_rmi.cpp has one where it starts downwards), let alone in floating point. It is checked on the
//     stored keys (if it fails there the root falls back to linear_spline, where the reference stops with an assertion), so stored
//     keys are always found; for a key never stored the bound is then not proved, but get() answers 'absent' whatever the bound,
//     and SortedBoundIndex::scan checks both ends of the bound against the array. proved() says which case an index is in.
//   - the k names promise the bound for stored keys only, by construction. The same two facts keep get and scan exact; a scan
//     that starts at a key never stored may then cost one more binary search over the array.
//
// Needs before it: GRE's indexInterface.h and pgm/pgm.h (readonly.h names pgm::PGMIndex). Portable C++17, double only (no long
// double); std::fma is one instruction with -march=native on any x86 that has FMA3 and on arm64, a libm call otherwise.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gret/readonly.h"

// A stored key is found because the prediction measured when building is, bit for bit, the prediction made when looking up.
// That holds for IEEE arithmetic; -ffast-math lets the compiler evaluate the two differently (GRE and this project do not use it).
#if defined(__FAST_MATH__)
#error "gret/sota/rmi.h must not be compiled with -ffast-math (or -Ofast): a stored key could fall outside its bound"
#endif

namespace gret {
namespace rmi_detail {

enum RootKind : int { kAuto = -1, kLinear = 0, kLinearSpline = 1, kCubic = 2, kRobustLinear = 3, kRadix = 4, kRadix18 = 5, kRadix22 = 6 };

inline const char* root_name(int kind) {
  static const char* const names[] = {"linear", "linear_spline", "cubic", "robust_linear", "radix", "radix18", "radix22"};
  return kind >= 0 && kind <= 6 ? names[kind] : "auto";
}

// A second-stage model, laid out as the reference lays it out (two doubles, then the error): position ~ alpha + beta * key.
struct Leaf {
  double alpha, beta;
  uint64_t err;
};
static_assert(sizeof(Leaf) == 24, "rmi: a leaf is 24 bytes, as in the reference");

// The reference's FCLAMP (below 0 -> 0, above top -> top, else truncate), with NaN -> 0 where FCLAMP is undefined.
inline size_t clamp_floor(double p, double top) { return size_t(std::max(0.0, std::min(p, top))); }

// The reference's simple linear regression (models/linear.rs, slr): running means, covariance and variance in one pass.
struct Slr {
  double mean_x = 0, mean_y = 0, c = 0, m2 = 0;
  uint64_t n = 0;
  void add(double x, double y) {
    ++n;
    const double dx = x - mean_x;
    mean_x += dx / double(n);
    mean_y += (y - mean_y) / double(n);
    c += dx * (y - mean_y);
    m2 += dx * (x - mean_x);
  }
  void fit(double& alpha, double& beta) const {  // 0 points: the zero model; 1 point or no variance: the mean
    alpha = n ? mean_y : 0.0;
    beta = 0.0;
    if (n < 2) return;
    const double cov = c / double(n - 1), var = m2 / double(n - 1);
    if (!(var > 0.0)) return;
    const double slope = cov / var, icpt = mean_y - slope * mean_x;
    if (!std::isfinite(slope) || !std::isfinite(icpt)) return;
    alpha = icpt;
    beta = slope;
  }
};

struct Model {
  int kind = kLinear;            // the root in use
  bool cubic_is_line = false;    // a cubic root that kept the line (a = b = 0), as the reference does where the line fits better
  bool keys_only = false;        // the 'k' names: leaves trained on their own keys, errors over the stored keys alone
  unsigned lshift = 0, rshift = 0;  // radix: (x << lshift) >> rshift; tables: ((x << lshift) >> lshift) >> rshift
  double a = 0, b = 0, c = 0, d = 0;  // the other roots: ((a x + b) x + c) x + d, the lines with a = b = 0
  double top = 0;                // N - 1
  double nm1 = 0;                // n - 1
  size_t n = 0;
  std::vector<uint32_t> table;   // radix18, radix22
  std::vector<Leaf> leaves;
  double avg_log2 = 0;           // the reference's 'average log2 error': mean over the keys of log2(2 err + 2)
  uint64_t max_err = 0;

  // The leaf of a key. Fixed: the root when the type knows it (straight-line code, as the reference generates), kAuto when it
  // was chosen while building (one predictable branch).
  template <int Fixed>
  size_t route(uint64_t x) const {
    switch (Fixed == kAuto ? kind : Fixed) {
      case kCubic: {
        const double v = double(x);
        return clamp_floor(std::fma(std::fma(std::fma(a, v, b), v, c), v, d), top);
      }
      case kRadix:
        return size_t((x << lshift) >> rshift);
      case kRadix18:
      case kRadix22:
        return table[size_t(((x << lshift) >> lshift) >> rshift)];
      default:
        return clamp_floor(std::fma(c, double(x), d), top);
    }
  }
  size_t predict(const Leaf& lf, uint64_t x) const { return clamp_floor(std::fma(lf.beta, double(x), lf.alpha), nm1); }
  // whether the bound is proved for keys never stored (see the top of the file)
  bool proved() const { return !keys_only && (kind != kCubic || cubic_is_line); }
  size_t bytes() const { return leaves.capacity() * sizeof(Leaf) + table.capacity() * sizeof(uint32_t); }
};

// The root, trained as the reference trains it: on (key, floor(position * N / n)) over all the keys. False if this root cannot
// be built on these keys (the reference then asserts, or divides by zero). keys: sorted, unique, n >= 2; N = 2^log2n >= 2.
inline bool train_root(int kind, const uint64_t* keys, size_t n, unsigned log2n, Model& m) {
  const size_t N = size_t(1) << log2n;
  const double sf = double(N) / double(n);
  const bool use_sf = std::fabs(sf - 1.0) > std::numeric_limits<double>::epsilon();
  auto yi = [&](size_t i) { return use_sf ? uint64_t(double(i) * sf) : uint64_t(i); };
  auto y = [&](size_t i) { return double(yi(i)); };
  m.kind = kind;
  m.cubic_is_line = false;
  m.lshift = m.rshift = 0;
  m.a = m.b = m.c = m.d = 0;
  m.top = double(N - 1);
  m.nm1 = double(n - 1);
  m.n = n;
  std::vector<uint32_t>().swap(m.table);
  const double x0 = double(keys[0]), xl = double(keys[n - 1]);
  auto line = [&](double& slope, double& icpt) {  // linear_spline.rs: through the first and the last point
    slope = 0;
    icpt = y(0);
    if (x0 == xl) return;  // the keys differ, their doubles do not: one constant, as the reference has for one distinct key
    slope = (y(0) - y(n - 1)) / (x0 - xl);
    icpt = y(0) - slope * x0;
  };
  const unsigned prefix = std::min(63u, keys[0] == keys[n - 1] ? 63u : unsigned(__builtin_clzll(keys[0] ^ keys[n - 1])));
  switch (kind) {
    case kLinear:
    case kRobustLinear: {
      size_t lo = 0, hi = n;
      if (kind == kRobustLinear) {  // linear.rs, RobustLinearModel: without 0.01% of the keys (at least one) at either end
        const size_t bnd = std::max<size_t>(1, size_t(double(n) * 0.0001));
        if (2 * bnd + 1 >= n) return false;
        lo = bnd;
        hi = n - bnd;
      }
      Slr s;
      for (size_t i = lo; i < hi; ++i) s.add(double(keys[i]), y(i));
      s.fit(m.d, m.c);
      break;
    }
    case kLinearSpline:
      line(m.c, m.d);
      break;
    case kCubic: {  // cubic_spline.rs: the cubic through the two end points with the end slopes of their neighbours
      const double xmin = x0, xmax = xl, ymin = y(0), ymax = y(n - 1);
      if (!(xmax > xmin) || !(ymax > ymin)) return false;
      const double xr = xmax - xmin, yr = ymax - ymin;
      size_t i1 = 1, i2 = n - 1;  // the first key right of xmin, the last key left of xmax (as doubles)
      while (i1 < n - 1 && !((double(keys[i1]) - xmin) / xr > 0.0)) ++i1;
      while (i2 > 0 && !((double(keys[i2]) - xmin) / xr < 1.0)) --i2;
      double m1 = ((y(i1) - ymin) / yr) / ((double(keys[i1]) - xmin) / xr);
      double m2 = (1.0 - (y(i2) - ymin) / yr) / (1.0 - (double(keys[i2]) - xmin) / xr);
      if (m1 * m1 + m2 * m2 > 9.0) {  // 'keep it monotonic'
        const double tau = 3.0 / std::sqrt(m1 * m1 + m2 * m2);
        m1 *= tau;
        m2 *= tau;
      }
      const double r3 = xr * xr * xr;
      double ca = (m1 + m2 - 2.0) / r3;
      double cb = -(xmax * (2.0 * m1 + m2 - 3.0) + xmin * (m1 + 2.0 * m2 - 3.0)) / r3;
      double cc = (m1 * xmax * xmax + m2 * xmin * xmin + xmax * xmin * (2.0 * m1 + 2.0 * m2 - 6.0)) / r3;
      double cd = -xmin * (m1 * xmax * xmax + xmax * xmin * (m2 - 3.0) + xmin * xmin) / r3;
      ca *= yr, cb *= yr, cc *= yr, cd *= yr;
      cd += ymin;
      double slope, icpt;
      line(slope, icpt);
      double cubic_err = 0, line_err = 0;  // 'sometimes the slope really doesn't work out': the line, if its L1 error is lower
      for (size_t i = 0; i < n; ++i) {
        const double v = double(keys[i]), t = y(i);
        cubic_err += std::fabs(std::fma(std::fma(std::fma(ca, v, cb), v, cc), v, cd) - t);
        line_err += std::fabs(std::fma(slope, v, icpt) - t);
      }
      const bool finite = std::isfinite(ca) && std::isfinite(cb) && std::isfinite(cc) && std::isfinite(cd) && std::isfinite(cubic_err);
      if (!finite || line_err < cubic_err) {
        m.c = slope, m.d = icpt;
        m.cubic_is_line = true;
      } else {
        m.a = ca, m.b = cb, m.c = cc, m.d = cd;
      }
      break;
    }
    case kRadix: {  // radix.rs, RadixModel: the bits below the common prefix, as many as the largest leaf number needs
      const uint64_t largest = yi(n - 1);
      unsigned bits = 0;  // utils.rs, num_bits
      while (bits < 62 && ((uint64_t(1) << (bits + 1)) - 1) <= largest) ++bits;
      if (bits < 1) return false;
      m.lshift = prefix;
      m.rshift = 64 - bits;
      break;
    }
    case kRadix18:
    case kRadix22: {  // radix.rs, RadixTable: prefix of the key -> the leaf of the first key with that prefix
      const unsigned bits = kind == kRadix18 ? 18 : 22;
      m.lshift = prefix;
      m.rshift = prefix + bits > 64 ? 0 : 64 - (prefix + bits);
      m.table.assign(size_t(1) << bits, 0);
      m.table.shrink_to_fit();
      const uint32_t last_leaf = uint32_t(N - 1);
      size_t last = 0;
      for (size_t i = 0; i < n; ++i) {
        const size_t r = size_t(((keys[i] << m.lshift) >> m.lshift) >> m.rshift);
        if (r == last) continue;
        if (r < last) return false;  // cannot happen: the keys are sorted and share the prefix
        const uint32_t leaf = uint32_t(std::min<uint64_t>(yi(i), last_leaf));
        for (size_t j = last + 1; j <= r; ++j) m.table[j] = leaf;
        last = r;
      }
      for (size_t j = last + 1; j < m.table.size(); ++j) m.table[j] = last_leaf;  // the reference writes the table's size here
      break;
    }
    default:
      return false;
  }
  return std::isfinite(m.a) && std::isfinite(m.b) && std::isfinite(m.c) && std::isfinite(m.d);
}

// The leaves under m's root, and their errors. Training as in the reference (train/two_layer.rs, build_models_from): the keys
// the root sends to a leaf, plus the last key before them and the first key after them, by least squares against the position;
// a leaf no key is sent to is the constant 'position of the next key'. Errors as described at the top of the file. False if the
// root is not non-decreasing on the stored keys. keys_only: the leaves of the 2018 paper instead (their own keys, and the largest
// error over those keys: nothing is promised for a key that was never stored).
inline bool train_leaves(const uint64_t* keys, size_t n, unsigned log2n, bool keys_only, Model& m) {
  const size_t N = size_t(1) << log2n;
  m.keys_only = keys_only;
  m.leaves.clear();
  m.leaves.shrink_to_fit();
  m.leaves.resize(N, Leaf{0.0, 0.0, 0});
  m.max_err = 0;
  double sum_log2 = 0;
  auto absdiff = [](uint64_t p, uint64_t q) { return p > q ? p - q : q - p; };
  auto finish = [&](size_t j, size_t s, size_t e) {  // leaf j, its model set, was sent the keys at [s, e)
    Leaf& lf = m.leaves[j];
    uint64_t err = 0;
    if (keys_only) {
      for (size_t i = s; i < e; ++i) err = std::max(err, absdiff(m.predict(lf, keys[i]), i));
    } else if (s == e) {  // no key: queries strictly between key s - 1 and key s
      if (s > 0 && s < n && keys[s] - keys[s - 1] >= 2) err = std::max(absdiff(m.predict(lf, keys[s - 1] + 1), s), absdiff(m.predict(lf, keys[s] - 1), s));
    } else {
      uint64_t prev = 0;
      for (size_t i = s; i < e; ++i) {
        const uint64_t p = m.predict(lf, keys[i]);
        err = std::max(err, absdiff(p, i));                                              // key i itself
        if (i > s && keys[i] - keys[i - 1] >= 2) err = std::max(err, absdiff(prev, i));  // a query between key i - 1 and key i
        prev = p;
      }
      // queries below the first key of the leaf (from the key before it, plus one) and above its last key (to the next key, minus one)
      if (s > 0 && keys[s] - keys[s - 1] >= 2) err = std::max(err, absdiff(m.predict(lf, keys[s - 1] + 1), s));
      if (e < n && keys[e] - keys[e - 1] >= 2) err = std::max({err, absdiff(prev, e), absdiff(m.predict(lf, keys[e] - 1), e)});
    }
    lf.err = err;
    m.max_err = std::max(m.max_err, err);
    if (e > s) sum_log2 += double(e - s) * std::log2(2.0 * double(err) + 2.0);
  };
  auto constant = [&](size_t j, size_t at) { m.leaves[j] = Leaf{double(at), 0.0, 0}; };
  Slr acc;
  size_t last = 0, start = 0;  // the leaf being filled, and its first key
  for (size_t i = 0; i < n; ++i) {
    const size_t t = m.template route<kAuto>(keys[i]);
    if (t < last || t >= N) return false;
    if (t > last) {
      if (i > start) {
        if (!keys_only) acc.add(double(keys[i]), double(i));  // 'include the first point of the next leaf'
        acc.fit(m.leaves[last].alpha, m.leaves[last].beta);
      } else {
        constant(last, i);
      }
      finish(last, start, i);
      for (size_t j = last + 1; j < t; ++j) constant(j, i), finish(j, i, i);
      acc = Slr();
      if (i > 0 && !keys_only) acc.add(double(keys[i - 1]), double(i - 1));  // 'include the last item of this leaf in the next leaf'
      last = t;
      start = i;
    }
    acc.add(double(keys[i]), double(i));
  }
  acc.fit(m.leaves[last].alpha, m.leaves[last].beta);
  finish(last, start, n);
  for (size_t j = last + 1; j < N; ++j) constant(j, n), finish(j, n, n);
  m.avg_log2 = sum_log2 / double(n);
  return true;
}

inline bool train(int kind, const uint64_t* keys, size_t n, unsigned log2n, bool keys_only, Model& m) {
  return train_root(kind, keys, n, log2n, m) && train_leaves(keys, n, log2n, keys_only, m);
}

// make_rmi hands the number of leaves (and the 'k') to the model it is about to construct through this (SortedBoundIndex
// default-constructs its Model): set, then taken once by the constructor, on the same thread, in the same statement.
struct Pending {
  unsigned log2n = 0;
  bool keys_only = false;
};
inline Pending& pending() {
  static thread_local Pending v;
  return v;
}

}  // namespace rmi_detail

// The Model for SortedBoundIndex. Root: a root (rmi_detail::kLinear ...), or kAuto: the best of the small roots.
template <class K, int Root = rmi_detail::kAuto>
struct RmiBound {
  static_assert(std::is_same<K, uint64_t>::value, "RmiBound: uint64_t keys");
  static constexpr unsigned kMinLog2 = 1, kMaxLog2 = 28;

  RmiBound() : RmiBound(std::exchange(rmi_detail::pending(), rmi_detail::Pending{})) {}
  explicit RmiBound(unsigned log2n, bool keys_only = false) : log2n_(log2n), keys_only_(keys_only) {}

  // keys: sorted, unique, at least 2 (SortedBoundIndex calls with 64 or more). Not modified.
  void build(std::vector<K>& keys) {
    using namespace rmi_detail;
    if (log2n_ < kMinLog2 || log2n_ > kMaxLog2) throw std::logic_error("RmiBound: no number of leaves (construct it through gret::make_rmi, or with log2 N in [1, 28])");
    if (keys.size() < 2) throw std::invalid_argument("RmiBound: fewer than 2 keys");
    const uint64_t* k = keys.data();
    const size_t n = keys.size();
    std::string how = root_name(Root);
    if (Root == kAuto) {  // the lowest average log2 error among the roots of one size; ties to the earlier (cheaper) one
      bool any = false;
      Model cand;
      for (int kind : {kLinear, kLinearSpline, kCubic, kRobustLinear, kRadix}) {
        if (!train(kind, k, n, log2n_, keys_only_, cand)) continue;
        if (!any || cand.avg_log2 < m_.avg_log2) std::swap(m_, cand);
        any = true;
      }
      if (!any) throw std::runtime_error("RmiBound: no root could be built");  // cannot happen: linear_spline always builds
      how = "auto";
    } else if (!train(Root, k, n, log2n_, keys_only_, m_)) {
      // the reference stops here (an assertion in build_models_from, or no model at all); this keeps the run alive and says so.
      // Root stays the type's, and with Root == kCubic the line is evaluated as a cubic with a = b = 0: the same values
      if (!train(kLinearSpline, k, n, log2n_, keys_only_, m_)) throw std::runtime_error("RmiBound: no root could be built");
      if (Root == kCubic) m_.kind = kCubic, m_.cubic_is_line = true;
      else if (Root != kLinear && Root != kRobustLinear) throw std::runtime_error(std::string("RmiBound: the ") + root_name(Root) + " root could not be built on these keys");
      how += ":failed,linear_spline_instead";
    }
    std::fprintf(stderr, "gret: rmi(reimplementation) leaves=2^%u root=%s%s asked=%s errors=%s avg_log2_err=%.3f max_err=%llu model_bytes=%zu never_stored_bound=%s\n",
                 log2n_, root_name(m_.kind), m_.cubic_is_line ? "(the_line)" : "", how.c_str(), keys_only_ ? "stored_keys(k)" : "every_key_in_range", m_.avg_log2,
                 (unsigned long long)m_.max_err, bytes(), m_.proved() ? "proved" : keys_only_ ? "none" : "not_proved(cubic)");
  }

  // the bound of x, for x between the smallest and the largest stored key (SortedBoundIndex answers the rest itself)
  KeyBound get(K x) const {
    const rmi_detail::Leaf& lf = m_.leaves[m_.template route<Root>(uint64_t(x))];
    const size_t p = m_.predict(lf, uint64_t(x)), e = size_t(lf.err);
    return {p > e ? p - e : 0, std::min(p + e + 1, m_.n)};
  }
  size_t bytes() const { return sizeof(*this) + m_.bytes(); }  // the root's parameters are in the object

  unsigned log2_leaves() const { return log2n_; }
  bool keys_only() const { return keys_only_; }
  int root() const { return m_.kind; }  // rmi_detail::RootKind: the root in use
  bool cubic_is_line() const { return m_.cubic_is_line; }
  bool proved() const { return m_.proved(); }
  double avg_log2_error() const { return m_.avg_log2; }
  uint64_t max_error() const { return m_.max_err; }

 private:
  explicit RmiBound(const rmi_detail::Pending& p) : log2n_(p.log2n), keys_only_(p.keys_only) {}
  unsigned log2n_;
  bool keys_only_;
  rmi_detail::Model m_;
};

// rmi<L>, rmi<L>lin, spl, cub, rob, rdx, rt18, rt22, each also with a final k (see the top of the file); nullptr for any other
// name, and for other keys.
template <class K, class P>
indexInterface<K, P>* make_rmi(const std::string& name) {
  if constexpr (!std::is_same<K, uint64_t>::value) {
    (void)name;
    return nullptr;
  } else {
    using namespace rmi_detail;
    if (name.compare(0, 3, "rmi") != 0 || name.size() < 4 || name[3] < '1' || name[3] > '9') return nullptr;
    size_t p = 3;
    unsigned log2n = 0;
    for (; p < name.size() && name[p] >= '0' && name[p] <= '9' && log2n < 100; ++p) log2n = log2n * 10 + unsigned(name[p] - '0');
    if (log2n < RmiBound<K>::kMinLog2 || log2n > RmiBound<K>::kMaxLog2) return nullptr;
    std::string root = name.substr(p);
    const bool keys_only = !root.empty() && root.back() == 'k';
    if (keys_only) root.pop_back();
    const std::string shown = name + "(reimplementation)";
    auto make = [&](auto tag) -> indexInterface<K, P>* {
      pending() = Pending{log2n, keys_only};
      return new SortedBoundIndex<K, P, typename decltype(tag)::type>(shown);
    };
    if (root.empty()) return make(std::common_type<RmiBound<K, kAuto>>{});
    if (root == "lin") return make(std::common_type<RmiBound<K, kLinear>>{});
    if (root == "spl") return make(std::common_type<RmiBound<K, kLinearSpline>>{});
    if (root == "cub") return make(std::common_type<RmiBound<K, kCubic>>{});
    if (root == "rob") return make(std::common_type<RmiBound<K, kRobustLinear>>{});
    if (root == "rdx") return make(std::common_type<RmiBound<K, kRadix>>{});
    if (root == "rt18") return make(std::common_type<RmiBound<K, kRadix18>>{});
    if (root == "rt22") return make(std::common_type<RmiBound<K, kRadix22>>{});
    return nullptr;
  }
}

}  // namespace gret
