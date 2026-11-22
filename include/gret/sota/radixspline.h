// SPDX-License-Identifier: MIT
// RadixSpline (Kipf, Marcus, van Renen, Stoian, Kemper, Kraska, Neumann: "RadixSpline: a single-pass learned index", aiDM@SIGMOD 2020) in
// GRE's harness. This is a WRAPPER around the authors' own header-only code (github.com/learnedsystems/RadixSpline, include/rs), which is
// used unmodified and included by path:  -I <RadixSpline>/include.  Nothing of it is copied here.
//
//   RadixSplineBound<K>   a Model for SortedBoundIndex (readonly.h): rs::Builder over the sorted keys, rs::RadixSpline::GetSearchBound per key
//   make_radixspline      the index by name: rs<max_error>r<radix_bits>, e.g. rs32r18 (the defaults of rs::Builder), rs2r28, rs380r12
//
// What the authors' code does, as read from it (builder.h, radix_spline.h) and pinned by tests/sota/test_radixspline.cpp:
//   - the smallest key is the first spline point and the largest is forced to be the last (Finalize), so the spline spans exactly [min, max];
//     a lookup at or below min is position 0, at or above max is position n - 1. There is no sentinel: 2^64 - 1 is an ordinary key, and unlike
//     PgmBound nothing is dropped from the model.
//   - GetSearchBound returns [begin, end), end exclusive: begin = est - max_error (not below 0), end = est + max_error + 2 (not above n),
//     est the interpolated position truncated to an integer. The position of a stored key is inside it; that is KeyBound's contract as is.
//   - the builder takes __builtin_clzl(max - min): undefined for max == min. SortedBoundIndex only builds a model from 64 keys up, and the
//     keys are unique, so max > min there; build() throws for max == min instead of running into it.
//   - duplicates would be allowed (a repeated key adds no CDF point and the bound holds for its first position); GRE's keys are unique.
//   - the radix table holds spline indexes as uint32_t: fewer than 2^32 keys (build() throws above that).
//
// Include after indexInterface.h and GRE's pgm header, like readonly.h (which this includes).
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "gret/readonly.h"
#include "rs/builder.h"  // the authors' RadixSpline: -I <RadixSpline>/include

namespace gret {
namespace sota {
struct RsPeek;  // names the specialization below, nothing else
}  // namespace sota
}  // namespace gret

namespace rs {
// rs::RadixSpline declares every rs::Serializer<T> its friend (rs/serializer.h is the authors' one). This specialization uses that to read
// how much the two vectors hold and to drop the spare capacity that the builder's push_back leaves in the spline: the bytes reported are
// then the bytes held, and equal what the authors' GetSize() counts. It changes no value and no lookup.
template <class KeyType>
class Serializer;
template <>
class Serializer<gret::sota::RsPeek> {
 public:
  template <class K>
  static void trim(RadixSpline<K>& m) {
    m.radix_table_.shrink_to_fit();
    m.spline_points_.shrink_to_fit();
  }
  template <class K>
  static size_t heap_bytes(const RadixSpline<K>& m) {
    return m.radix_table_.capacity() * sizeof(uint32_t) + m.spline_points_.capacity() * sizeof(Coord<K>);
  }
  template <class K>
  static size_t spline_points(const RadixSpline<K>& m) {
    return m.spline_points_.size();
  }
  template <class K>
  static size_t radix_entries(const RadixSpline<K>& m) {
    return m.radix_table_.size();
  }
};
}  // namespace rs

namespace gret {
namespace sota {

struct RsConfig {
  size_t max_error = 32;   // the spline's error bound in positions (rs::Builder's default)
  size_t radix_bits = 18;  // prefix bits of the radix table (rs::Builder's default)
};

inline constexpr size_t kRsMaxError = size_t(1) << 20;
inline constexpr size_t kRsMaxRadixBits = 30;  // a table of 2^30 + 1 uint32_t; the authors' tunings stop at 28

inline bool rs_valid(const RsConfig& c) {
  return c.max_error >= 1 && c.max_error <= kRsMaxError && c.radix_bits >= 1 && c.radix_bits <= kRsMaxRadixBits;
}

// "rs32r18" -> {32, 18}. The whole name must be rs<max_error>r<radix_bits>, both decimal without leading zeros, max_error in
// [1, 2^20], radix_bits in [1, 30]; anything else is not a RadixSpline name (false).
inline bool parse_rs_name(const std::string& name, RsConfig& out) {
  if (name.size() < 5 || name.compare(0, 2, "rs") != 0) return false;
  const size_t r = name.find('r', 2);
  if (r == std::string::npos || r == 2 || r + 1 >= name.size()) return false;
  auto number = [](const std::string& d, size_t max_digits, size_t& v) {
    if (d.empty() || d.size() > max_digits || d[0] == '0' || d.find_first_not_of("0123456789") != std::string::npos) return false;
    v = 0;
    for (const char ch : d) v = v * 10 + size_t(ch - '0');
    return true;
  };
  RsConfig c;
  if (!number(name.substr(2, r - 2), 7, c.max_error) || !number(name.substr(r + 1), 2, c.radix_bits) || !rs_valid(c)) return false;
  out = c;
  return true;
}

// SortedBoundIndex default-constructs its Model and takes only a name, so the parameters of the next RadixSplineBound to be constructed
// on this thread are left here by make_radixspline (and put back to the defaults by it).
inline RsConfig& rs_pending() {
  thread_local RsConfig c;
  return c;
}

template <class K>
struct RadixSplineBound {
  static_assert(std::is_same<K, uint64_t>::value || std::is_same<K, uint32_t>::value, "rs::Builder takes uint32_t or uint64_t keys");
  using Peek = rs::Serializer<RsPeek>;

  RsConfig cfg;
  rs::RadixSpline<K> m;

  RadixSplineBound() : cfg(rs_pending()) {}
  explicit RadixSplineBound(const RsConfig& c) : cfg(c) {}

  // keys: sorted ascending (as SortedBoundIndex passes them); one pass, as the authors' example and bench.cc build it
  void build(std::vector<K>& keys) {
    if (!rs_valid(cfg)) throw std::invalid_argument("RadixSplineBound: max_error must be in [1, 2^20] and radix_bits in [1, 30]");
    if (keys.empty() || keys.front() == keys.back()) throw std::invalid_argument("RadixSplineBound: needs at least two distinct keys");
    if (keys.size() > size_t(std::numeric_limits<uint32_t>::max())) throw std::length_error("RadixSplineBound: 2^32 keys or more");
    rs::Builder<K> b(keys.front(), keys.back(), cfg.radix_bits, cfg.max_error);
    for (const K k : keys) b.AddKey(k);
    m = b.Finalize();
    Peek::trim(m);
  }
  KeyBound get(K x) const {
    const rs::SearchBound b = m.GetSearchBound(x);
    return {b.begin, b.end};
  }
  // the model's bytes: this object (the authors' GetSize() counts its own) plus what the radix table and the spline hold
  size_t bytes() const { return sizeof(*this) + Peek::heap_bytes(m); }
  size_t spline_points() const { return Peek::spline_points(m); }
  size_t radix_entries() const { return Peek::radix_entries(m); }
};

// rs<max_error>r<radix_bits> -> the index; nullptr for every other name
template <class K, class P>
indexInterface<K, P>* make_radixspline(const std::string& name) {
  RsConfig c;
  if (!parse_rs_name(name, c)) return nullptr;
  struct Restore {
    ~Restore() { rs_pending() = RsConfig{}; }
  } restore;
  rs_pending() = c;
  return new SortedBoundIndex<K, P, RadixSplineBound<K>>(name);
}

}  // namespace sota
}  // namespace gret
