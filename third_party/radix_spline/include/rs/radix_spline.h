#pragma once
// Source-derived transcription; see ../../PROVENANCE.md and LICENSE.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include <utility>
#include "common.h"
namespace rs {
template <class KeyType> class RadixSpline {
 public:
  RadixSpline() = default;
  RadixSpline(KeyType min_key, KeyType max_key, size_t num_keys,
              size_t num_radix_bits, size_t num_shift_bits, size_t max_error,
              std::vector<uint32_t> radix_table,
              std::vector<rs::Coord<KeyType>> spline_points)
      : min_key_(min_key), max_key_(max_key), num_keys_(num_keys),
        num_radix_bits_(num_radix_bits), num_shift_bits_(num_shift_bits),
        max_error_(max_error), radix_table_(std::move(radix_table)),
        spline_points_(std::move(spline_points)) {}
  double GetEstimatedPosition(const KeyType key) const {
    if (key <= min_key_) return 0;
    if (key >= max_key_) return num_keys_ - 1;
    const size_t index = GetSplineSegment(key);
    const Coord<KeyType> down = spline_points_[index - 1];
    const Coord<KeyType> up = spline_points_[index];
    const double x_diff = up.x - down.x;
    const double y_diff = up.y - down.y;
    const double slope = y_diff / x_diff;
    const double key_diff = key - down.x;
    return std::fma(key_diff, slope, down.y);
  }
  SearchBound GetSearchBound(const KeyType key) const {
    const size_t estimate = GetEstimatedPosition(key);
    const size_t begin = (estimate < max_error_) ? 0 : estimate - max_error_;
    const size_t end = (estimate + max_error_ + 2 > num_keys_)
                          ? num_keys_ : estimate + max_error_ + 2;
    return SearchBound{begin, end};
  }
  size_t GetSize() const {
    return sizeof(*this) + radix_table_.size() * sizeof(uint32_t) +
           spline_points_.size() * sizeof(Coord<KeyType>);
  }
 private:
  size_t GetSplineSegment(const KeyType key) const {
    const KeyType prefix = (key - min_key_) >> num_shift_bits_;
    assert(prefix + 1 < radix_table_.size());
    const uint32_t begin = radix_table_[prefix];
    const uint32_t end = radix_table_[prefix + 1];
    if (end - begin < 32) {
      uint32_t current = begin;
      while (spline_points_[current].x < key) ++current;
      return current;
    }
    const auto lb = std::lower_bound(
      spline_points_.begin() + begin, spline_points_.begin() + end, key,
      [](const Coord<KeyType>& coord, const KeyType key) { return coord.x < key; });
    return std::distance(spline_points_.begin(), lb);
  }
  KeyType min_key_, max_key_;
  size_t num_keys_, num_radix_bits_, num_shift_bits_, max_error_;
  std::vector<uint32_t> radix_table_;
  std::vector<Coord<KeyType>> spline_points_;
  template <typename> friend class Serializer;
};
}
