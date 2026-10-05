#pragma once
// Source-derived transcription; see ../../PROVENANCE.md and LICENSE.
#include <cassert>
#include <cmath>
#include <limits>
#include "common.h"
#include "radix_spline.h"
namespace rs {
template <class KeyType> class Builder {
 public:
  Builder(KeyType min_key, KeyType max_key, size_t num_radix_bits = 18,
          size_t max_error = 32)
      : min_key_(min_key), max_key_(max_key), num_radix_bits_(num_radix_bits),
        num_shift_bits_(GetNumShiftBits(max_key - min_key, num_radix_bits)),
        max_error_(max_error), curr_num_keys_(0), curr_num_distinct_keys_(0),
        prev_key_(min_key), prev_position_(0), prev_prefix_(0) {
    const uint32_t max_prefix = (max_key - min_key) >> num_shift_bits_;
    radix_table_.resize(max_prefix + 2, 0);
  }
  void AddKey(KeyType key) {
    if (curr_num_keys_ == 0) { AddKey(key, 0); return; }
    AddKey(key, prev_position_ + 1);
  }
  RadixSpline<KeyType> Finalize() {
    assert(curr_num_keys_ == 0 || prev_key_ == max_key_);
    if (curr_num_keys_ > 0 && spline_points_.back().x != prev_key_)
      AddKeyToSpline(prev_key_, prev_position_);
    FinalizeRadixTable();
    return RadixSpline<KeyType>(min_key_, max_key_, curr_num_keys_,
      num_radix_bits_, num_shift_bits_, max_error_,
      std::move(radix_table_), std::move(spline_points_));
  }
 private:
  static size_t GetNumShiftBits(uint32_t diff, size_t num_radix_bits) {
    const uint32_t clz = __builtin_clz(diff);
    if ((32 - clz) < num_radix_bits) return 0;
    return 32 - num_radix_bits - clz;
  }
  static size_t GetNumShiftBits(uint64_t diff, size_t num_radix_bits) {
    const uint32_t clzl = __builtin_clzl(diff);
    if ((64 - clzl) < num_radix_bits) return 0;
    return 64 - num_radix_bits - clzl;
  }
  void AddKey(KeyType key, size_t position) {
    assert(key >= min_key_ && key <= max_key_);
    assert(key >= prev_key_);
    assert(position == 0 || position > prev_position_);
    PossiblyAddKeyToSpline(key, position);
    ++curr_num_keys_; prev_key_ = key; prev_position_ = position;
  }
  void AddKeyToSpline(KeyType key, double position) {
    spline_points_.push_back({key, position}); PossiblyAddKeyToRadixTable(key);
  }
  enum Orientation { Collinear, CW, CCW };
  static constexpr double precision = std::numeric_limits<double>::epsilon();
  static Orientation ComputeOrientation(double dx1, double dy1, double dx2, double dy2) {
    const double expr = std::fma(dy1, dx2, -std::fma(dy2, dx1, 0));
    if (expr > precision) return Orientation::CW;
    if (expr < -precision) return Orientation::CCW;
    return Orientation::Collinear;
  }
  void SetUpperLimit(KeyType key, double position) { upper_limit_ = {key, position}; }
  void SetLowerLimit(KeyType key, double position) { lower_limit_ = {key, position}; }
  void RememberPreviousCDFPoint(KeyType key, double position) { prev_point_ = {key, position}; }
  // Original source cites GreedySplineCorridor, Neumann & Michel, BNCOD 2008.
  void PossiblyAddKeyToSpline(KeyType key, double position) {
    if (curr_num_keys_ == 0) {
      AddKeyToSpline(key, position); ++curr_num_distinct_keys_;
      RememberPreviousCDFPoint(key, position); return;
    }
    if (key == prev_key_) return;
    ++curr_num_distinct_keys_;
    if (curr_num_distinct_keys_ == 2) {
      SetUpperLimit(key, position + max_error_);
      SetLowerLimit(key, position < max_error_ ? 0 : position - max_error_);
      RememberPreviousCDFPoint(key, position); return;
    }
    const Coord<KeyType>& last = spline_points_.back();
    const double upper_y = position + max_error_;
    const double lower_y = position < max_error_ ? 0 : position - max_error_;
    assert(upper_limit_.x >= last.x && lower_limit_.x >= last.x && key >= last.x);
    const double upper_limit_x_diff = upper_limit_.x - last.x;
    const double lower_limit_x_diff = lower_limit_.x - last.x;
    const double x_diff = key - last.x;
    assert(upper_limit_.y >= last.y && position >= last.y);
    const double upper_limit_y_diff = upper_limit_.y - last.y;
    const double lower_limit_y_diff = lower_limit_.y - last.y;
    const double y_diff = position - last.y;
    assert(prev_point_.x != last.x);
    if (ComputeOrientation(upper_limit_x_diff, upper_limit_y_diff, x_diff, y_diff) != Orientation::CW ||
        ComputeOrientation(lower_limit_x_diff, lower_limit_y_diff, x_diff, y_diff) != Orientation::CCW) {
      AddKeyToSpline(prev_point_.x, prev_point_.y);
      SetUpperLimit(key, upper_y); SetLowerLimit(key, lower_y);
    } else {
      assert(upper_y >= last.y);
      const double upper_y_diff = upper_y - last.y;
      if (ComputeOrientation(upper_limit_x_diff, upper_limit_y_diff, x_diff, upper_y_diff) == Orientation::CW)
        SetUpperLimit(key, upper_y);
      const double lower_y_diff = lower_y - last.y;
      if (ComputeOrientation(lower_limit_x_diff, lower_limit_y_diff, x_diff, lower_y_diff) == Orientation::CCW)
        SetLowerLimit(key, lower_y);
    }
    RememberPreviousCDFPoint(key, position);
  }
  void PossiblyAddKeyToRadixTable(KeyType key) {
    const KeyType curr_prefix = (key - min_key_) >> num_shift_bits_;
    if (curr_prefix != prev_prefix_) {
      const uint32_t curr_index = spline_points_.size() - 1;
      for (KeyType prefix = prev_prefix_ + 1; prefix <= curr_prefix; ++prefix)
        radix_table_[prefix] = curr_index;
      prev_prefix_ = curr_prefix;
    }
  }
  void FinalizeRadixTable() {
    ++prev_prefix_;
    const uint32_t num_spline_points = spline_points_.size();
    for (; prev_prefix_ < radix_table_.size(); ++prev_prefix_)
      radix_table_[prev_prefix_] = num_spline_points;
  }
  const KeyType min_key_, max_key_;
  const size_t num_radix_bits_, num_shift_bits_, max_error_;
  std::vector<uint32_t> radix_table_;
  std::vector<Coord<KeyType>> spline_points_;
  size_t curr_num_keys_, curr_num_distinct_keys_;
  KeyType prev_key_; size_t prev_position_; KeyType prev_prefix_;
  Coord<KeyType> upper_limit_, lower_limit_, prev_point_;
};
}
