// SPDX-License-Identifier: MIT
// Dataset properties for the attribution analysis, on one GRE key file (uint64 count, then uint64 keys), after GRE's own
// preprocessing (sort, dedup) and after G (exceptions.h's peel_outliers rule). Prints one JSON line:
//   e, e_lo, e_hi   keys G peels; r = span / span after G
//   DX, DB          Kolmogorov distance of the min-max-normalised keys to uniform, before and after G (Error.lean ksUnif)
//   KL_B            KL from linear after G: sum_b p_b ln(B p_b) over B = 1000 equal-width bins (the KL gate's statistic)
//   epsU_X, epsU_B  the endpoint line's max rank error, before and after G
//   Epw512/4096     pw<K>'s max chord error over its equi-rank pieces (mpw: the mean)
//   Kg[eps]         greedy PLA segments at eps = 4, 8, 16, 32, 64, 128 (local roughness: per 1k keys = Kg / n * 1000)
//   win.*           per 256-key region: two-run floors, V's floor, chord error (medians and p90)
// The quantities and their Lean statements: docs/SPLICE.md. Build: c++ -O3 -std=c++17 dataset_bounds.cpp
// (written by the gate-bounds derivation, 2026-10-20; dedup and KL added for the 200M analysis).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
typedef long double LD;

static void peel(const std::vector<uint64_t>& k, size_t E, size_t& lo, size_t& hi) {  // exceptions.h peel_outliers
  size_t n = k.size(); lo = 0; hi = n;
  if (n > 2 * E + 2 && E > 0) {
    const uint64_t tr = k[n - 1 - E] - k[E];
    size_t pl = 0, ph = 0;
    while (hi - lo > 2 && k[hi - 1] - k[lo] - tr > tr && (pl < E || ph < E)) {
      const uint64_t gl = k[lo + 1] - k[lo], gh = k[hi - 1] - k[hi - 2];
      if (ph < E && (gh >= gl || pl >= E)) { --hi; ++ph; } else { ++lo; ++pl; }
    }
  }
}
// ksUnif of min-max normalized x[lo,hi) (Error.lean ksUnif), and the endpoint error max|n u_i - (i+1)|
static void ks(const uint64_t* x, size_t n, LD& D, LD& epsU) {
  const LD R = LD(x[n - 1] - x[0]); D = 0; epsU = 0;
  for (size_t i = 0; i < n; ++i) {
    const LD u = LD(x[i] - x[0]) / R;
    D = std::max(D, std::max(LD(i + 1) / n - u, u - LD(i) / n));
    epsU = std::max(epsU, std::fabs(LD(n) * u - LD(i + 1)));
  }
}
// pw<K> chord error: max and mean |i - chord_j(x_i)| over equi-rank pieces (pwlcdf.h knot ranks)
static void pw(const uint64_t* x, size_t n, unsigned K, LD& emax, LD& emean) {
  emax = 0; emean = 0;
  for (unsigned j = 0; j < K; ++j) {
    const size_t a = size_t((2 * (unsigned __int128)j * (n - 1) + K) / (2 * (unsigned __int128)K));
    const size_t b = size_t((2 * (unsigned __int128)(j + 1) * (n - 1) + K) / (2 * (unsigned __int128)K));
    if (b <= a) continue;
    const LD dx = LD(x[b] - x[a]), dr = LD(b - a);
    for (size_t i = (j ? a + 1 : a); i <= b; ++i) {
      const LD c = LD(a) + dr * LD(x[i] - x[a]) / dx;
      const LD e = std::fabs(LD(i) - c);
      emax = std::max(emax, e); emean += e;
    }
  }
  emean /= n;
}
static size_t greedy(const uint64_t* s, size_t m, double eps) {  // gate.h pla_segments, full resolution
  size_t segs = 0;
  for (size_t i = 0; i < m;) {
    ++segs; double lo = -INFINITY, hi = INFINITY; size_t j = i + 1;
    for (; j < m; ++j) {
      const double dx = double(s[j] - s[i]), dy = double(j - i);
      const double l = std::max(lo, (dy - eps) / dx), h = std::min(hi, (dy + eps) / dx);
      if (l > h) break;
      lo = l; hi = h;
    }
    i = j;
  }
  return segs;
}
static double ftwo(double h, double r) { return r > 1 ? h / 2 * (r - 1) / (r + 1) : 0; }
static double pct(std::vector<double> v, double q) { std::sort(v.begin(), v.end()); return v[size_t(q * (v.size() - 1))]; }

int main(int argc, char** argv) {
  FILE* f = std::fopen(argv[1], "rb"); uint64_t cnt; std::fread(&cnt, 8, 1, f);
  std::vector<uint64_t> k(cnt); std::fread(k.data(), 8, cnt, f); std::fclose(f);
  std::sort(k.begin(), k.end()); k.erase(std::unique(k.begin(), k.end()), k.end());  // as GRE's benchmark.h
  const size_t n = k.size(); size_t lo, hi; peel(k, 64, lo, hi);
  const uint64_t* b = k.data() + lo; const size_t nb = hi - lo, e = n - nb;
  LD DX, eX, DB, eB; ks(k.data(), n, DX, eX); ks(b, nb, DB, eB);
  const LD r = LD(k[n - 1] - k[0]) / LD(b[nb - 1] - b[0]);
  double KL = 0;
  {
    const size_t B = 1000; std::vector<size_t> h(B, 0); const LD R = LD(b[nb - 1] - b[0]);
    for (size_t i = 0; i < nb; ++i) { size_t j = size_t(LD(b[i] - b[0]) / R * B); h[j < B ? j : B - 1]++; }
    for (size_t j = 0; j < B; ++j) if (h[j]) { const double p = double(h[j]) / double(nb); KL += p * std::log(p * B); }
  }
  LD p512, m512, p4096, m4096, p512raw, m512raw; pw(b, nb, 512, p512, m512); pw(b, nb, 4096, p4096, m4096);
  pw(k.data(), n, 512, p512raw, m512raw);
  const double eps[] = {4, 8, 16, 32, 64, 128}; size_t Kg[6];
  for (int t = 0; t < 6; ++t) Kg[t] = greedy(b, nb, eps[t]);
  // windows of R=256 keys, h=16: two-run floor on max/min, V-limit on mean/min with nu<=alpha, chord error
  const size_t R = 256, h = 16; const double alpha = 0.1;
  std::vector<double> F2, FV, EC, GAIN; size_t vgain1 = 0, nw = 0;
  for (size_t s = 0; s + R <= nb; s += R) {
    const uint64_t* w = b + s; double smin = 1e300, smax = 0;
    for (size_t j = 0; j + h < R; j += h) { const double S = double(w[j + h] - w[j]); smin = std::min(smin, S); smax = std::max(smax, S); }
    const double W = double(w[R - 1] - w[0]), Sbar = W * h / double(R - 1);
    const double f2 = ftwo(h, smax / smin), fv = ftwo(h, (Sbar / smin) / (1 + alpha * R / (R - 1)));
    double ec = 0; for (size_t i = 0; i < R; ++i) ec = std::max(ec, std::fabs(double(i) - double(R - 1) * double(w[i] - w[0]) / W));
    F2.push_back(f2); FV.push_back(fv); EC.push_back(ec); GAIN.push_back(ec - fv); vgain1 += (ec - fv >= 1); ++nw;
  }
  std::printf("{\"file\":\"%s\",\"n\":%zu,\"e\":%zu,\"e_lo\":%zu,\"e_hi\":%zu,\"r\":%.6Lg,\"DX\":%.6Lg,\"epsU_X\":%.6Lg,"
              "\"DB\":%.6Lg,\"epsU_B\":%.6Lg,\"Epw512\":%.6Lg,\"mpw512\":%.6Lg,\"Epw4096\":%.6Lg,\"mpw4096\":%.6Lg,"
              "\"Epw512_raw\":%.6Lg,\"KL_B\":%.6g,\"Kg\":[%zu,%zu,%zu,%zu,%zu,%zu],\"win\":{\"f2_med\":%.4g,\"f2_p90\":%.4g,"
              "\"fV_med\":%.4g,\"fV_p90\":%.4g,\"ec_med\":%.4g,\"ec_p90\":%.4g,\"gainV_p90\":%.4g,\"share_gainV_ge1\":%.4g}}\n",
              argv[1], n, e, lo, n - hi, r, DX, eX, DB, eB, p512, m512, p4096, m4096, p512raw, KL, Kg[0], Kg[1], Kg[2], Kg[3],
              Kg[4], Kg[5], pct(F2, .5), pct(F2, .9), pct(FV, .5), pct(FV, .9), pct(EC, .5), pct(EC, .9), pct(GAIN, .9),
              double(vgain1) / nw);
}
