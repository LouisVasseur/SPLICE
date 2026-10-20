// SPDX-License-Identifier: MIT
// RankTop: SPLICE's segment locate for millions of segments. The common case is two dependent accesses that can
// miss: one 4-byte radix entry, then one window of segment entries (searched directly, so the entry is in hand).
//
//   v = k - x0  (value radix)  or  v = C(k)  (rank radix: C an equi-rank PwlCdf of the bulk keys, non-decreasing)
//   b = v >> s
//   R[b] sparse (top bit 0): lo = R[b], the last segment start whose bucket is below b, and every key of bucket b
//        falls in a segment of [lo, lo + W - 1]:
//        j = lo + #{c in (lo, lo + W) : entry[c].x0 <= k}      a fixed, branch-free window of W entries
//   R[b] dense (top bit 1): a second radix over the bucket, 2^q sub-buckets (q and the sub-table's offset are packed
//        in R[b]): t = (v >> (s - q)) mod 2^q, [lo, hi] = [sub[o + t], sub[o + t + 1]], then the window when
//        hi - lo < W, else a lifting search over the entries in [lo, hi] (clamped to hi).
//
// Exactness: v is non-decreasing in k, so a start whose (sub-)bucket is below that of k is below k and one whose
// (sub-)bucket is above it is above k: the answer lies in (lo, hi], and the window's entries past hi (or the W - 1
// padding entries after the last, x0 = 2^64 - 1) are above k. Keys at or above the last knot are answered before the
// search (the last interval), as in PwlCdf::locate.
// The entry array is the caller's (SpliceIndex's Seg, 32 bytes, 64-byte aligned): locate(k, entries) reads
// entries[c].x0 for c up to (m - 2) + W - 1, so the caller pads it with W - 1 entries of x0 = 2^64 - 1.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "gret/pwlcdf.h"

namespace gret {

class RankTop {
 public:
  static constexpr unsigned kW = 4;  // window: 4 entries of 32 B, one 128-byte line (Apple) or two 64-byte lines
  static constexpr uint32_t kDense = 0x80000000u;
  static constexpr unsigned kQBits = 4, kOffBits = 27;  // dense entry: 1 | q (4 bits) | sub-table offset (27 bits)

  // x: the knots (strictly increasing, m >= 2), as PwlCdf::locator takes them; rbits: log2 of the bucket count.
  // value: the radix is over k - x0 (C is ignored); else over C(k).
  static RankTop build(const std::vector<uint64_t>& x, bool last_starts, const PwlCdf& C, unsigned rbits,
                       bool value = false) {
    if (x.size() < 2) throw std::invalid_argument("RankTop: need two knots");
    if (x.size() - 2 >= kDense - kW) throw std::invalid_argument("RankTop: too many knots");
    RankTop t;
    t.m_ = x.size();
    t.x0_ = x.front();
    t.xl_ = x.back();
    t.last_ = t.m_ - 2 + size_t(last_starts);
    t.value_ = value;
    if (rbits > 30) rbits = 30;
    size_t nb;
    if (value) {  // keys reaching the search have k - x0 in [0, xl - x0 - 1]
      const uint64_t d1 = t.xl_ - t.x0_ - 1;
      const unsigned w = bit_width(d1);
      t.s_ = w > rbits ? w - rbits : 0;
      if (t.s_ > 63) t.s_ = 63;
      nb = size_t(d1 >> t.s_) + 1;
    } else {
      t.C_ = C;
      t.s_ = kOutBits - rbits;
      nb = size_t(kOutMax >> t.s_) + 1;
    }
    // lo[b] = (#starts with bucket < b) - 1, at least 0; starts 0 .. m - 2 are the search's intervals
    std::vector<uint64_t> v(t.m_ - 1);
    for (size_t j = 0; j + 1 < t.m_; ++j) {
      v[j] = t.map(x[j]);
      if (j && v[j] < v[j - 1]) throw std::logic_error("RankTop: map not monotone");
    }
    std::vector<uint32_t> lo(nb + 1, 0);
    for (size_t j = 0; j + 1 < t.m_; ++j) ++lo[size_t(v[j] >> t.s_) + 1];
    for (size_t b = 1; b <= nb; ++b) lo[b] += lo[b - 1];
    for (size_t b = 0; b <= nb; ++b) lo[b] = lo[b] ? lo[b] - 1 : 0;
    t.R_.assign(nb, 0);
    for (size_t b = 0; b < nb; ++b) {
      const size_t l = lo[b], h = lo[b + 1];  // h: the last start whose bucket is at most b
      if (h - l < kW) { t.R_[b] = uint32_t(l); continue; }
      unsigned q = bit_width(h - l) + 1;  // 2^q sub-buckets, about 2-4 per candidate
      q = std::min<unsigned>({q, (1u << kQBits) - 1, t.s_});
      const size_t off = t.sub_.size();
      if (off + (size_t(1) << q) + 1 >= (size_t(1) << kOffBits)) throw std::length_error("RankTop: sub-tables too large");
      t.R_[b] = kDense | (uint32_t(q) << kOffBits) | uint32_t(off);
      ++t.dense_;
      const unsigned ss = t.s_ - q;
      const uint64_t g0 = uint64_t(b) << q;
      size_t c = l;  // sub-entry u: the last start whose sub-bucket is below u (at least l)
      for (uint64_t u = 0; u <= (uint64_t(1) << q); ++u) {
        while (c < h && (v[c + 1] >> ss) < g0 + u) ++c;
        t.sub_.push_back(uint32_t(c));
        if (u && t.sub_[t.sub_.size() - 1] - t.sub_[t.sub_.size() - 2] >= kW) ++t.dense2_;
      }
    }
    return t;
  }

  template <class E>
  size_t locate(uint64_t k, const E* e) const {
    if (k >= xl_) return last_;
    const uint64_t v = map(k);
    const uint32_t r = R_[size_t(v >> s_)];
    size_t j;
    if (__builtin_expect(!(r & kDense), 1)) {
      j = r;
    } else {
      const unsigned q = (r >> kOffBits) & ((1u << kQBits) - 1);
      const uint32_t* sb = sub_.data() + (r & ((1u << kOffBits) - 1)) + size_t((v >> (s_ - q)) & ((uint64_t(1) << q) - 1));
      j = sb[0];
      const size_t hi = sb[1];
      if (__builtin_expect(hi - j >= kW, 0)) {  // a dense sub-bucket: lifting over the entries, clamped to hi
        for (size_t h = size_t(1) << (bit_width(hi - j) - 1); h; h >>= 1) {
          const size_t c = std::min(j + h, hi);
          j = e[c].x0 <= k ? c : j;
        }
        return j;
      }
    }
    const E* w = e + j;
    size_t a = j;
    for (unsigned c = 1; c < kW; ++c) a += size_t(w[c].x0 <= k);
    return a;
  }

  bool dense(uint64_t k) const {  // does k's bucket take the dense (sub-table) path?
    return k < xl_ && (R_[size_t(map(k) >> s_)] & kDense);
  }
  size_t c_bytes() const { return value_ ? 0 : C_.bytes(); }
  size_t bytes() const {
    return sizeof(*this) + R_.capacity() * sizeof(uint32_t) + sub_.capacity() * sizeof(uint32_t) + C_.bytes() -
           sizeof(PwlCdf);
  }
  bool value_radix() const { return value_; }
  size_t buckets() const { return R_.size(); }
  size_t dense_buckets() const { return dense_; }
  size_t dense_subbuckets() const { return dense2_; }
  size_t sub_entries() const { return sub_.size(); }
  unsigned rank_steps() const { return value_ ? 0 : C_.search_steps(); }

 private:
  static unsigned bit_width(uint64_t v) { return v ? 64 - unsigned(__builtin_clzll(v)) : 0; }
  uint64_t map(uint64_t k) const { return value_ ? k - x0_ : C_(k); }  // k in [x0, xl)
  size_t m_ = 0, last_ = 0, dense_ = 0, dense2_ = 0;
  uint64_t x0_ = 0, xl_ = 0;
  unsigned s_ = 52;
  bool value_ = false;
  PwlCdf C_;
  std::vector<uint32_t> R_, sub_;
};

}  // namespace gret
