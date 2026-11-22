// SPDX-License-Identifier: MIT
// discover_rule: the blind period discovery of tools/bench/gdiscover.cpp as a function an index build can call. The same windows,
// FFT, coherence, refinement, run merging, harmonic check, block boundaries, S and row phases, in the same order and with the same
// arithmetic, so it finds the rule gdiscover writes for the same keys (byte for byte when both are built with the same compiler and
// floating-point flags). Single-threaded and deterministic, as the tool is.
//
//   DiscoveredRule r = discover_rule(keys, n);  keys sorted and unique, as gdiscover holds them once it has sorted and deduplicated
//                                               its file. gdiscover passes all of a file's keys; GatedGIndex passes the bulk (the
//                                               keys its index stores), so that a far key below it cannot pull the first block down
//                                               (and, below the phase, to 0). Either way the first block starts at or below keys[0],
//                                               which PeriodRule and FastPeriodRule require, and min(64, n / 8) keys at each end stay
//                                               out of the windows
//   r.found        gdiscover writes a rule file; false where it prints "no rule: identity"
//   r.S, r.blocks  that file: "S <S>", then "<first key> <period>" per block. rule_text(r) is its text and write_rule_file(r, path)
//                  writes it, for $RULE_FILE (ruleline.h, fastrule.h)
//   r.seconds      the discovery's wall time: gdiscover's total_with_phases_s, or its total_s when there is no rule
//   r.log          what gdiscover prints, without the "<key file>: " in front of its first line
// discover_rule(kv, n) reads the keys of a bulk_load's std::pair<K, P> array in place, with no copy.
//
// Changed from the tool, none of it in what it finds: no file and no argv (keys in, rule and report out); spread_score's buffers belong
// to the call rather than to the thread; a window under 4096 keys returns before reading its keys (gdiscover read k[b - 1] first,
// k[-1] for an empty window at rank 0, then returned on the count); n = 0 finds no rule (gdiscover read k[0]).
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace gret {

struct DiscoveredRule {
  bool found = false;  // a rule was found: gdiscover writes its rule file
  uint64_t S = 0;      // the room every row gets, as written on the file's first line
  std::vector<std::pair<uint64_t, uint64_t>> blocks;  // {first key, period}, one line of the file each, first keys increasing
  double seconds = 0;
  std::string log;
};

namespace discover_detail {

using u64 = uint64_t;
using cd = std::complex<double>;
constexpr double kPi = 3.14159265358979323846;  // M_PI's value: <cmath> need not define M_PI

__attribute__((format(printf, 2, 3))) inline void say(std::string& s, const char* fmt, ...) {
  char line[256];
  va_list ap, again;
  va_start(ap, fmt);
  va_copy(again, ap);
  const int len = std::vsnprintf(line, sizeof(line), fmt, ap);
  va_end(ap);
  if (len >= int(sizeof(line))) {
    std::vector<char> big(size_t(len) + 1);
    std::vsnprintf(big.data(), big.size(), fmt, again);
    s.append(big.data(), size_t(len));
  } else if (len > 0) {
    s.append(line, size_t(len));
  }
  va_end(again);
}

inline void fft(std::vector<cd>& a) {
  const size_t n = a.size();
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t b = n >> 1;
    for (; j & b; b >>= 1) j ^= b;
    j ^= b;
    if (i < j) std::swap(a[i], a[j]);
  }
  for (size_t len = 2; len <= n; len <<= 1) {
    const double ang = -2 * kPi / double(len);
    const cd wl(std::cos(ang), std::sin(ang));
    for (size_t i = 0; i < n; i += len) {
      cd w(1);
      for (size_t j = 0; j < len / 2; ++j) {
        const cd u = a[i + j], v = a[i + j + len / 2] * w;
        a[i + j] = u + v;
        a[i + j + len / 2] = u - v;
        w *= wl;
      }
    }
  }
}

struct Window {
  size_t a = 0, b = 0;  // ranks [a, b)
  double period = 0, score = 0, res = 0;  // res: the FFT's resolution, relative
};

struct Block {
  size_t a, b;
  double period, co, res;
};

template <class Pair>
struct PairKeys {  // a bulk_load's pairs, read as keys
  const Pair* d;
  u64 operator[](size_t i) const { return u64(d[i].first); }
};

// phase coherence of keys [a, b) at period P, on at most m keys evenly picked: 1 if all share a phase, about 1/sqrt(m) at random
template <class Keys>
double coherence(const Keys& k, size_t a, size_t b, double P, size_t m) {
  const size_t step = std::max<size_t>(1, (b - a) / m);
  double c = 0, s = 0;
  size_t used = 0;
  const u64 base = k[a];
  for (size_t i = a; i < b; i += step) {
    const double t = 2 * kPi * std::fmod(double(k[i] - base), P) / P;
    c += std::cos(t), s += std::sin(t);
    ++used;
  }
  return std::sqrt(c * c + s * s) / double(used ? used : 1);
}

// the window's strongest period from its density, 0 if nothing stands out of the noise. Harmonics of narrow rows can be as tall as the
// fundamental, so take the lowest frequency whose height is comparable to the tallest and that the tallest is a multiple of.
template <class Keys>
void propose(const Keys& k, Window& w, std::vector<cd>& buf) {
  const size_t N = buf.size(), count = w.b - w.a;
  w.period = w.score = 0;
  if (count < 4096) return;  // gdiscover's test, before the keys are read
  const u64 base = k[w.a], span = k[w.b - 1] - base;
  if (span < N / 4) return;
  std::fill(buf.begin(), buf.end(), cd(0));
  const double perbin = double(N) / (double(span) + 1);
  for (size_t i = w.a; i < w.b; ++i) buf[std::min(N - 1, size_t(double(k[i] - base) * perbin))] += 1.0;
  const double mean = double(count) / double(N);
  for (auto& x : buf) x -= mean;
  fft(buf);
  std::vector<double> mag(N / 2);
  for (size_t f = 0; f < N / 2; ++f) mag[f] = std::abs(buf[f]);
  const size_t lo = 32;  // at least 32 rows in the window
  // a smooth density has most of its energy at low frequencies, so a peak is judged against its neighbourhood: the median height per
  // quarter octave
  std::vector<double> floor_at(N / 2, 0);
  for (double e = std::log2(double(lo)); ; e += 0.25) {
    const size_t f0 = size_t(std::pow(2.0, e)), f1 = std::min(N / 2, size_t(std::pow(2.0, e + 0.25)) + 1);
    if (f0 >= N / 2) break;
    std::vector<double> part(mag.begin() + std::ptrdiff_t(f0), mag.begin() + std::ptrdiff_t(f1));
    std::nth_element(part.begin(), part.begin() + std::ptrdiff_t(part.size() / 2), part.end());
    const double m = part[part.size() / 2] + 1e-9;
    for (size_t g = f0; g < f1; ++g) floor_at[g] = m;
  }
  size_t top = 0;
  double top_score = 0;
  for (size_t f = lo; f < N / 2; ++f) {
    const double sc = mag[f] / floor_at[f];
    if (sc > top_score) top_score = sc, top = f;
  }
  if (top_score < 16) return;
  size_t fund = top;
  for (size_t d = 2; d <= 16 && top / d >= lo; ++d) {
    const size_t f = size_t(double(top) / double(d) + 0.5);
    double peak = 0;
    for (size_t g = f - 1; g <= f + 1 && g < N / 2; ++g) peak = std::max(peak, mag[g] / floor_at[g]);
    if (peak >= 0.35 * top_score) fund = f;
  }
  size_t best = fund;
  for (size_t g = fund - 1; g <= fund + 1 && g < N / 2; ++g)
    if (mag[g] > mag[best]) best = g;
  w.period = (double(span) + 1) / double(best);
  w.res = 2.0 / double(best);
  w.score = top_score;
}

// refine P over [a, b), inside the FFT's resolution: widen the span the phases must agree over, narrowing the search each time
template <class Keys>
double refine(const Keys& k, size_t a, size_t b, double P, double res) {
  const double pmin = P * (1 - res), pmax = P * (1 + res);
  const size_t mid = a + (b - a) / 2;
  size_t half = 2048;
  double best = P;
  for (;;) {
    const size_t l = mid > a + half ? mid - half : a, r = std::min(b, mid + half);
    const double rows = double(k[r - 1] - k[l]) / best;
    const double width = best / std::max(rows, 1.0);  // a period off by this much drifts one whole row across the span
    double top = -1, at = best;
    for (int i = -16; i <= 16; ++i) {
      const double c = std::min(pmax, std::max(pmin, best + double(i) * width / 8));
      const double co = coherence(k, l, r, c, 4096);
      if (co > top) top = co, at = c;
    }
    best = at;
    if (l == a && r == b) break;
    half *= 4;
  }
  return best;
}

// how evenly the rule's coordinate spreads keys [a, b) (every step-th), for a row start at phase ph: keys per occupied cell, weighted
// by key, over cells that would hold 16 keys if they were even. 16 is perfect, more is clumped. y and n are scratch.
template <class Keys>
double spread_score(const Keys& k, size_t a, size_t b, size_t step, u64 P, u64 ph, double S, std::vector<double>& y,
                    std::vector<uint32_t>& n) {
  y.clear();
  double h[66] = {};
  size_t m = 0;
  auto bits = [](u64 r) { return r ? 64 - __builtin_clzll(r) : 0; };
  auto offset = [&](u64 x) { return (x % P + P - ph) % P; };
  for (size_t i = a; i < b; i += step) ++h[bits(offset(k[i]))], ++m;
  double u[67];
  u[0] = 0;
  double cum = 0;
  for (int j = 0; j <= 64; ++j) cum += h[j], u[j + 1] = cum / double(m);
  u[0] = u[1];
  const u64 origin = k[a] - offset(k[a]);
  for (size_t i = a; i < b; i += step) {
    const u64 r = offset(k[i]), q = (k[i] - origin) / P;
    const int j = bits(r);
    const u64 half = j ? u64(1) << (j - 1) : 0;
    const double den = double(half ? half : 1);
    const double F = u[j] + (u[j + 1] - u[j]) * double(r - (half - 1)) / den;
    y.push_back(double(q) * (S + double(P)) + S * F + double(r));
  }
  const double lo = y.front(), hi = y.back();
  const size_t cells = std::max<size_t>(1, m / 16);
  n.assign(cells, 0);
  const double per = double(cells) / (hi - lo + 1);
  for (const double v : y) ++n[std::min(cells - 1, size_t((v - lo) * per))];
  double sq = 0;
  for (const uint32_t c : n) sq += double(c) * double(c);
  return sq / double(m);
}

// gdiscover's main from the sorted unique keys on: k[i] is the i-th key of n
template <class Keys>
DiscoveredRule run(const Keys& k, const size_t n) {
  using Clock = std::chrono::steady_clock;
  DiscoveredRule out;
  std::string& log = out.log;
  if (n == 0) {
    say(log, "no keys\n  no rule: identity\n");
    return out;
  }
  const auto t0 = Clock::now();

  const size_t W = 128, N = size_t(1) << 20;
  std::vector<Window> win(W);
  std::vector<cd> buf(N);
  // the two ends go first, as peel_outliers takes them: a stray key would make the end windows' spans meaningless
  const size_t trim = std::min<size_t>(64, n / 8);
  const size_t lo_rank = trim, hi_rank = n - trim;
  size_t with = 0;
  for (size_t i = 0; i < W; ++i) {
    win[i].a = lo_rank + (hi_rank - lo_rank) * i / W;
    win[i].b = lo_rank + (hi_rank - lo_rank) * (i + 1) / W;
  }
  // a first look at every eighth window: most key sets stop here, for the price of a fraction of a build
  for (size_t i = 4; i < W; i += 8) {
    propose(k, win[i], buf);
    with += win[i].period > 0;
  }
  if (with)
    for (size_t i = 0; i < W; ++i)
      if (i % 8 != 4) {
        propose(k, win[i], buf);
        with += win[i].period > 0;
      }
  const double fft_s = std::chrono::duration<double>(Clock::now() - t0).count();

  // runs of windows that agree, with windows that found nothing, or a harmonic, absorbed by the run they sit in
  std::vector<Block> blocks;
  for (size_t i = 0; i < W; ++i) {
    if (win[i].period <= 0) continue;
    const double p = win[i].period;
    if (!blocks.empty()) {
      const double q = blocks.back().period, ratio = p > q ? p / q : q / p;
      if (ratio < 1 + std::max(0.002, 2 * win[i].res) && blocks.back().b == win[i].a) {
        blocks.back().b = win[i].b;
        continue;
      }
    }
    blocks.push_back(Block{win[i].a, win[i].b, p, 0, win[i].res});
  }
  // refine each run and check that the period holds over the whole run, and that it is a period at all: many rows, not a slope
  std::vector<Block> kept;
  size_t covered = 0;
  for (auto& b : blocks) {
    b.period = refine(k, b.a, b.b, b.period, b.res);
    b.co = coherence(k, b.a, b.b, b.period, 65536);
    const double chance = 4 / std::sqrt(double(std::min<size_t>(65536, b.b - b.a))), rows = double(k[b.b - 1] - k[b.a]) / b.period;
    if (b.co > std::max(chance, 0.05) && rows >= 256) kept.push_back(b), covered += b.b - b.a;
  }
  // a run whose keys cohere nearly as well at a neighbour's period is that neighbour's: a harmonic, or a window across a boundary
  for (bool merged = true; merged;) {
    merged = false;
    for (size_t c = 0; c < kept.size() && !merged; ++c) {
      for (int side = -1; side <= 1 && !merged; side += 2) {
        if ((side < 0 && c == 0) || (side > 0 && c + 1 == kept.size())) continue;
        Block& nb = kept[size_t(std::ptrdiff_t(c) + side)];
        const size_t ka = kept[c].b - kept[c].a, na = nb.b - nb.a;
        if (ka > na) continue;  // the smaller run gives way
        if (coherence(k, kept[c].a, kept[c].b, nb.period, 65536) < 0.7 * kept[c].co) continue;
        nb.a = std::min(nb.a, kept[c].a), nb.b = std::max(nb.b, kept[c].b);
        nb.co = coherence(k, nb.a, nb.b, nb.period, 65536);
        kept.erase(kept.begin() + std::ptrdiff_t(c));
        merged = true;
      }
    }
  }
  // rows narrow enough have harmonics as strong as the fundamental: a multiple of the period that coheres better is the true one
  for (auto& b : kept)
    for (int m = 2; m <= 16;) {
      const double co = coherence(k, b.a, b.b, b.period * m, 65536);
      if (co > b.co * 1.005) {
        b.period = refine(k, b.a, b.b, b.period * m, 2.0 * b.res), b.co = coherence(k, b.a, b.b, b.period, 65536);
        m = 2;
      } else {
        ++m;
      }
    }
  // boundaries by binary search: which period do the keys around a rank cohere with? Blocks that do not touch stay apart, and the
  // keys between go to the block before, whose rule they fall under anyway
  const size_t wlen = (hi_rank - lo_rank) / W;
  for (size_t c = 0; c + 1 < kept.size(); ++c) {
    size_t l = kept[c].b > 2 * wlen && kept[c].b - 2 * wlen > kept[c].a ? kept[c].b - 2 * wlen : kept[c].a + (kept[c].b - kept[c].a) / 2;
    size_t r = kept[c + 1].a + 2 * wlen < kept[c + 1].b ? kept[c + 1].a + 2 * wlen : kept[c + 1].a + (kept[c + 1].b - kept[c + 1].a) / 2;
    while (r - l > 256) {
      const size_t m = l + (r - l) / 2, x0 = m > 8192 ? m - 8192 : 0, x1 = std::min<size_t>(n, m + 8192);
      const double ca = coherence(k, x0, x1, kept[c].period, 8192), cb = coherence(k, x0, x1, kept[c + 1].period, 8192);
      (ca >= cb ? l : r) = m;
    }
    kept[c].b = kept[c + 1].a = l + (r - l) / 2;
  }
  if (!kept.empty()) kept[0].a = lo_rank, kept.back().b = hi_rank;
  const double total_s = std::chrono::duration<double>(Clock::now() - t0).count();
  const double share = double(covered) / double(hi_rank - lo_rank);
  say(log, "windows_with_period=%zu/%zu runs=%zu kept=%zu share=%.3f fft_s=%.2f total_s=%.2f\n", with, W, blocks.size(), kept.size(),
      share, fft_s, total_s);
  for (const auto& b : kept)
    say(log, "  block ranks %zu..%zu first_key=%llu period=%.3f coherence=%.3f rows=%.0f\n", b.a, b.b, (unsigned long long)k[b.a],
        b.period, b.co, double(k[b.b - 1] - k[b.a]) / b.period);
  out.seconds = total_s;
  if (kept.empty() || share < 0.5) {
    say(log, "  no rule: identity\n");
    return out;
  }
  // S, the room every row gets: well above the widest period, so rows count and their empty parts do not, unless the coordinate
  // would not fit in 64 bits
  double total_rows = 0, widest = 0;
  for (const auto& b : kept) total_rows += double(k[b.b - 1] - k[b.a]) / b.period + 1, widest = std::max(widest, b.period);
  const double S = std::max(0.0, std::min(std::max(64 * widest, 67108864.0), std::ldexp(1.0, 63) / total_rows - widest));
  out.found = true;
  out.S = (unsigned long long)S;
  say(log, "  S=%.0f total_rows=%.0f\n", S, total_rows);
  // row starts. Rows that all begin at the same place have a sharp left edge at exact resolution: a stretch of phases with nothing,
  // then keys packed tight. Failing that, the phase whose coordinate spreads the keys most evenly.
  std::vector<double> spread_y;
  std::vector<uint32_t> spread_n;
  for (size_t c = 0; c < kept.size(); ++c) {
    const u64 P = std::max<u64>(1, u64(std::llround(kept[c].period)));
    const size_t step = std::max<size_t>(1, (kept[c].b - kept[c].a) / 2000000);
    std::vector<u64> res;
    res.reserve((kept[c].b - kept[c].a) / step + 1);
    for (size_t i = kept[c].a; i < kept[c].b; i += step) res.push_back(k[i] % P);
    std::sort(res.begin(), res.end());
    const size_t m = std::min<size_t>(32, res.size() / 4), R = res.size();
    double edge = 0;
    u64 phase = 0;
    for (size_t i = 0; i < R; ++i) {
      if (i && res[i] == res[i - 1]) continue;  // the first of equal residues is the edge
      const u64 back = res[i] >= res[(i + R - m) % R] && i >= m ? res[i] - res[i - m] : res[i] + P - res[(i + R - m) % R];
      const u64 ahead = i + m < R ? res[i + m] - res[i] : res[(i + m) % R] + P - res[i];
      const double sc = double(back) / double(ahead + 1);
      if (sc > edge) edge = sc, phase = res[i];
    }
    const char* how = "edge";
    if (edge < 8) {
      how = "spread";
      const size_t st2 = std::max<size_t>(1, (kept[c].b - kept[c].a) / 400000);
      double best_score = 1e300, best_phase = 0;
      auto consider = [&](double ph) {
        const double sc = spread_score(k, kept[c].a, kept[c].b, st2, P, u64(ph * double(P)) % P, S, spread_y, spread_n);
        if (sc < best_score) best_score = sc, best_phase = ph;
      };
      for (int i = 0; i < 64; ++i) consider(double(i) / 64);
      double radius = 1.0 / 64;
      for (int level = 0; level < 5; ++level, radius /= 4) {
        const double centre = best_phase;
        for (int i = -4; i <= 4; ++i)
          if (i) consider(std::fmod(centre + double(i) * radius / 4 + 1, 1.0));
      }
      phase = u64(best_phase * double(P)) % P;
    }
    const u64 x = c == 0 ? k[0] : k[kept[c].a];
    const u64 back = (x % P + P - phase) % P;  // first <= x with first % P == phase
    const u64 first = x >= back ? x - back : 0;
    out.blocks.emplace_back(first, P);
    say(log, "  rule %llu %llu (phase %.6f by %s, edge %.1f)\n", (unsigned long long)first, (unsigned long long)P,
        double(phase) / double(P), how, edge);
  }
  out.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  say(log, "  total_with_phases_s=%.2f\n", out.seconds);
  return out;
}

}  // namespace discover_detail

// keys: n sorted unique keys
inline DiscoveredRule discover_rule(const uint64_t* keys, size_t n) { return discover_detail::run(keys, n); }

// kv: a bulk_load's n pairs, sorted by key with unique keys
template <class K, class P>
DiscoveredRule discover_rule(const std::pair<K, P>* kv, size_t n) {
  return discover_detail::run(discover_detail::PairKeys<std::pair<K, P>>{kv}, n);
}

// the rule file gdiscover writes: "S <S>", then "<first> <period>" per block; empty when nothing was found
inline std::string rule_text(const DiscoveredRule& r) {
  std::string s;
  if (!r.found) return s;
  discover_detail::say(s, "S %llu\n", (unsigned long long)r.S);
  for (const auto& b : r.blocks) discover_detail::say(s, "%llu %llu\n", (unsigned long long)b.first, (unsigned long long)b.second);
  return s;
}

// writes rule_text(r) to path; false if nothing was found (gdiscover writes no file then) or the file cannot be written
inline bool write_rule_file(const DiscoveredRule& r, const char* path) {
  if (!r.found || !path) return false;
  FILE* f = std::fopen(path, "w");
  if (!f) return false;
  const std::string s = rule_text(r);
  const bool ok = std::fwrite(s.data(), 1, s.size(), f) == s.size();
  return std::fclose(f) == 0 && ok;
}

}  // namespace gret
