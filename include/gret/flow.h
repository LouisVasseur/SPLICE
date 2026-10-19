// SPDX-License-Identifier: MIT
// MolFlow: the learned transform. A monotone normalizing flow made of one mixture-of-logistics CDF (the elementwise
// layer of Flow++), trained by transform/train_flow.py with a flat latent on [0, 1]:
//
//   x(k) = (k - lo) / (hi - lo),   u(x) = sum_m w_m sigmoid((x - mu_m) / s_m),   T(k) = clamp(floor(u 2^52), 0, 2^52 - 1)
//
// x is computed as double(k - lo) (or -double(lo - k) for k < lo) over double(hi - lo), which train_flow.py reproduces
// bit for bit. u is monotone by construction and T was non-decreasing on all 200M keys of fb and books, but rounding
// does not guarantee it and nothing here relies on it: the wrapper sorts the outputs and sends equal ones to overflow.
//
// File (text):   molflow 1 M lo hi        lo < hi, uint64 in decimal
//                w mu s                   M lines, %.17g; w > 0, s > 0 with 1/s finite, |sum w - 1| < 1e-9
// load() rejects anything else (extra tokens or lines, non-finite values, a missing file) with std::runtime_error, then
// renormalizes w. The evaluation is a plain loop over M, padded to a multiple of 4 lanes with w = 0 terms; each lane
// accumulates in a fixed order, so the loop can be vectorized without -ffast-math and the result does not depend on
// it. exp is an inline branch-free polynomial (exp_small, relative error ~1e-16) rather than a libm call, which no
// compiler vectorizes without a vector math library. z is clamped to [-40, 40], where the sigmoid is within 4.3e-18
// of 0 or 1: any key, however far outside [lo, hi], gives a finite u in [0, 1] up to rounding, never NaN. M = 32:
// ~60 ns per key on an M-series core at -O3, against ~107 ns with libm's exp.
//
// Reproducibility is the contract: bulk_load and get() must compute the same T for a key, bit for bit, or the lookup
// misses or returns another key's payload. So T is plain IEEE arithmetic in a fixed order. Contraction of a * b + c
// into one fma is off in the evaluation (clang: the pragma; GCC: the optimize attribute, as GCC's default
// -ffp-contract=fast may fuse one inlined copy and not another); the fused steps are explicit (madd: std::fma where the
// hardware has it, else a multiply and an add); -ffast-math is refused at compile time; and operator() is kept out of
// line, so that bulk_load and get() run the same machine code even under flags that ignore the pragma.
#pragma once
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gret/common.h"

#if defined(__FAST_MATH__)
#error "gret/flow.h: do not build with -ffast-math: MolFlow's T must be reproducible bit for bit (see above)"
#endif
#if defined(__clang__)
#define GRET_FP_STRICT _Pragma("clang fp contract(off)")  // at the start of each evaluation function's body
#define GRET_FP_FN
#define GRET_OUTLINE __attribute__((noinline))
#elif defined(__GNUC__)
#define GRET_FP_STRICT
#define GRET_FP_FN __attribute__((optimize("fp-contract=off")))
#define GRET_OUTLINE __attribute__((noipa))  // noinline, and no IPA clones
#else
#define GRET_FP_STRICT
#define GRET_FP_FN
#define GRET_OUTLINE
#endif

namespace gret {

class MolFlow {
 public:
  static constexpr size_t kMaxTerms = 4096;
  static constexpr size_t kLanes = 4;
  static constexpr double kZMax = 40;

  static MolFlow load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("MolFlow::load: cannot open flow file '" + path + "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    if (f.bad()) throw std::runtime_error("MolFlow::load: cannot read flow file '" + path + "'");
    return parse(ss.str(), path);
  }

  // The same validation on the file's text; origin names it in error messages.
  static MolFlow parse(const std::string& text, const std::string& origin = "<text>") {
    MolFlow f;
    std::vector<std::string> lines = split(text, "\n"), tok;
    while (!lines.empty() && split(lines.back(), " \t\r").empty()) lines.pop_back();  // trailing blank lines only
    auto fail = [&](size_t line, const std::string& what) -> void {
      throw std::runtime_error("MolFlow::load: " + origin + ": line " + std::to_string(line + 1) + ": " + what);
    };
    if (lines.empty()) fail(0, "empty file");
    tok = split(lines[0], " \t\r");
    if (tok.size() != 5 || tok[0] != "molflow" || tok[1] != "1") fail(0, "header must be 'molflow 1 M lo hi'");
    uint64_t M = 0;
    if (!parse_u64(tok[2], M) || M < 1 || M > kMaxTerms) fail(0, "M must be an integer in [1, 4096]");
    if (!parse_u64(tok[3], f.lo_) || !parse_u64(tok[4], f.hi_)) fail(0, "lo and hi must be uint64 in decimal");
    if (f.lo_ >= f.hi_) fail(0, "lo must be below hi");
    if (lines.size() != 1 + M) fail(0, "expected " + std::to_string(M) + " component lines, found " +
                                         std::to_string(lines.size() - 1));
    f.M_ = size_t(M);
    f.Mp_ = (f.M_ + kLanes - 1) / kLanes * kLanes;
    f.w_.assign(f.Mp_, 0.0);  // padding terms: w = 0, mu = 0, 1/s = 0 add exactly 0
    f.mu_.assign(f.Mp_, 0.0);
    f.inv_s_.assign(f.Mp_, 0.0);
    f.s_.assign(f.M_, 0.0);
    double sum = 0;
    for (size_t m = 0; m < f.M_; ++m) {
      tok = split(lines[1 + m], " \t\r");
      double v[3];
      if (tok.size() != 3) fail(1 + m, "expected 'w mu s'");
      for (int i = 0; i < 3; ++i)
        if (!parse_double(tok[size_t(i)], v[i])) fail(1 + m, "not a finite number: '" + tok[size_t(i)] + "'");
      if (!(v[0] > 0)) fail(1 + m, "w must be > 0");
      if (!(v[2] > 0) || !std::isfinite(1.0 / v[2])) fail(1 + m, "s must be > 0 with a finite reciprocal");
      f.w_[m] = v[0], f.mu_[m] = v[1], f.s_[m] = v[2], f.inv_s_[m] = 1.0 / v[2];
      sum += v[0];
    }
    if (!(std::fabs(sum - 1.0) < 1e-9)) fail(0, "weights must sum to 1 (|sum - 1| < 1e-9)");
    for (size_t m = 0; m < f.M_; ++m) f.w_[m] /= sum;
    f.range_ = double(f.hi_ - f.lo_);
    return f;
  }

  // Normalized input: finite for every key, as hi - lo >= 1.
  GRET_FP_FN double x_of(uint64_t k) const {
    GRET_FP_STRICT
    return k >= lo_ ? double(k - lo_) / range_ : -double(lo_ - k) / range_;
  }

  GRET_FP_FN double u_of_x(double x) const {
    GRET_FP_STRICT
    const double* __restrict w = w_.data();
    const double* __restrict mu = mu_.data();
    const double* __restrict is = inv_s_.data();
    double acc[kLanes] = {0, 0, 0, 0};
    for (size_t m = 0; m < Mp_; m += kLanes)
      for (size_t l = 0; l < kLanes; ++l) {
        const double t = std::fmin(std::fmax((mu[m + l] - x) * is[m + l], -kZMax), kZMax);  // t = -z, clamped
        acc[l] += w[m + l] / (1.0 + exp_small(t));                                       // w sigmoid(z)
      }
    return (acc[0] + acc[1]) + (acc[2] + acc[3]);
  }

  // a * b + c: one rounding where the hardware fuses it (arm64, x86-64 with FMA), else two; one or the other for the
  // whole binary (__FP_FAST_FMA is a compile-time constant), never a mix.
  GRET_FP_FN static double madd(double a, double b, double c) {
    GRET_FP_STRICT
#if defined(__FP_FAST_FMA)
    return std::fma(a, b, c);
#else
    return a * b + c;
#endif
  }

  // e^t for |t| <= kZMax (the caller clamps): t = n ln2 + r with |r| <= ln2/2 (n rounded by the 1.5 2^52 shift, ln2 in
  // two parts with n ln2_hi exact), e^r by its degree-13 Taylor polynomial (remainder < 5e-18 relative), and 2^n
  // written into the exponent bits (|n| <= 58). Branch-free, so loops over it vectorize.
  GRET_FP_FN static double exp_small(double t) {
    GRET_FP_STRICT
    constexpr double kLog2e = 1.4426950408889634, kShift = 6755399441055744.0;  // 1.5 2^52
    constexpr double kLn2Hi = 6.93147180369123816490e-01, kLn2Lo = 1.90821492927058770002e-10;  // fdlibm's split
    const double v = madd(t, kLog2e, kShift), n = v - kShift;  // n = round(t / ln2), exact
    const double r = madd(-n, kLn2Lo, madd(-n, kLn2Hi, t));
    double p = 1.0 / 6227020800;  // 1/13!
    p = madd(p, r, 1.0 / 479001600);
    p = madd(p, r, 1.0 / 39916800);
    p = madd(p, r, 1.0 / 3628800);
    p = madd(p, r, 1.0 / 362880);
    p = madd(p, r, 1.0 / 40320);
    p = madd(p, r, 1.0 / 5040);
    p = madd(p, r, 1.0 / 720);
    p = madd(p, r, 1.0 / 120);
    p = madd(p, r, 1.0 / 24);
    p = madd(p, r, 1.0 / 6);
    p = madd(p, r, 0.5);
    p = madd(p, r, 1.0);
    p = madd(p, r, 1.0);
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof bits);  // the low mantissa bits of v hold n
    const uint64_t scale_bits = (bits - 0x4338000000000000ull + 1023) << 52;
    double scale;
    std::memcpy(&scale, &scale_bits, sizeof scale);
    return p * scale;
  }

  GRET_FP_FN double u(uint64_t k) const {
    GRET_FP_STRICT
    return u_of_x(x_of(k));
  }

  // T(k): the only entry point of the wrapper, at bulk_load and get() (out of line: see the header comment)
  GRET_OUTLINE GRET_FP_FN uint64_t operator()(uint64_t k) const {
    GRET_FP_STRICT
    // u is in [0, 1] up to rounding; fmax also maps a NaN to 0 (cannot happen: the file values are finite)
    const double t = std::fmin(std::fmax(u(k) * kOutScale, 0.0), double(kOutMax));
    return uint64_t(t);  // truncation = floor for t >= 0
  }

  size_t M() const { return M_; }
  size_t cost_terms() const { return M_; }
  uint64_t lo() const { return lo_; }
  uint64_t hi() const { return hi_; }
  double w(size_t m) const { return w_[m]; }  // renormalized
  double mu(size_t m) const { return mu_[m]; }
  double s(size_t m) const { return s_[m]; }
  size_t bytes() const {
    return sizeof(*this) + (w_.capacity() + mu_.capacity() + inv_s_.capacity() + s_.capacity()) * sizeof(double);
  }

 private:
  static std::vector<std::string> split(const std::string& s, const char* seps) {
    std::vector<std::string> out;
    const bool keep_empty = std::string(seps) == "\n";  // lines are kept even when empty, tokens are not
    size_t i = 0;
    while (i <= s.size()) {
      size_t j = s.find_first_of(seps, i);
      if (j == std::string::npos) j = s.size();
      if (keep_empty || j > i) out.push_back(s.substr(i, j - i));
      i = j + 1;
    }
    if (keep_empty && !out.empty() && out.back().empty()) out.pop_back();  // the final newline
    return out;
  }

  static bool parse_u64(const std::string& t, uint64_t& v) {
    if (t.empty() || t.size() > 20 || t.find_first_not_of("0123456789") != std::string::npos) return false;
    errno = 0;
    char* end = nullptr;
    const unsigned long long r = std::strtoull(t.c_str(), &end, 10);
    if (errno == ERANGE || end != t.c_str() + t.size()) return false;
    v = uint64_t(r);
    return true;
  }

  static bool parse_double(const std::string& t, double& v) {
    char* end = nullptr;
    v = std::strtod(t.c_str(), &end);
    return !t.empty() && end == t.c_str() + t.size() && std::isfinite(v);
  }

  size_t M_ = 0, Mp_ = 0;  // a default MolFlow has no terms and maps every key to 0
  uint64_t lo_ = 0, hi_ = 1;
  double range_ = 1;
  std::vector<double> w_, mu_, inv_s_, s_;
};

}  // namespace gret

#undef GRET_FP_STRICT
#undef GRET_FP_FN
#undef GRET_OUTLINE
