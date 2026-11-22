// SPDX-License-Identifier: MIT
// OverflowTable<P>: where a writable index keeps the keys its slots cannot take (G, for writes): an open-addressing hash table of
// 16-byte (key, payload) pairs, one probe sequence per lookup, usually one cache line. Linear probing on mix64, a power of two
// of slots, at most 5/8 full, erase by moving the following pairs back (no tombstones). The key 2^64 - 1 marks an empty slot, so
// that one key is held apart. It has no order: the owner sorts a copy when it must scan.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include "gret/hashline.h"

namespace gret {

template <class P>
class OverflowTable {
  using Slot = std::pair<uint64_t, P>;
  static constexpr uint64_t kEmpty = ~uint64_t(0);

 public:
  size_t size() const { return n_ + size_t(has_max_); }
  bool empty() const { return n_ == 0 && !has_max_; }
  size_t bytes() const { return t_.size() * sizeof(Slot); }
  bool find(uint64_t k, P& v) const {
    if (__builtin_expect(k == kEmpty, 0)) {
      if (has_max_) v = max_;
      return has_max_;
    }
    if (t_.empty()) return false;
    for (size_t i = home(k);; i = (i + 1) & mask_) {
      if (t_[i].first == k) return v = t_[i].second, true;
      if (t_[i].first == kEmpty) return false;
    }
  }
  bool contains(uint64_t k) const {
    P v;
    return find(k, v);
  }
  bool insert(uint64_t k, P v) {  // false if k is there already
    if (__builtin_expect(k == kEmpty, 0)) {
      if (has_max_) return false;
      return has_max_ = true, max_ = v, true;
    }
    if ((n_ + 1) * 8 > t_.size() * 5) grow();
    for (size_t i = home(k);; i = (i + 1) & mask_) {
      if (t_[i].first == k) return false;
      if (t_[i].first == kEmpty) return t_[i] = Slot(k, v), ++n_, true;
    }
  }
  bool assign(uint64_t k, P v) {  // false if k is not there
    if (__builtin_expect(k == kEmpty, 0)) {
      if (has_max_) max_ = v;
      return has_max_;
    }
    if (t_.empty()) return false;
    for (size_t i = home(k);; i = (i + 1) & mask_) {
      if (t_[i].first == k) return t_[i].second = v, true;
      if (t_[i].first == kEmpty) return false;
    }
  }
  bool erase(uint64_t k) {
    if (__builtin_expect(k == kEmpty, 0)) {
      const bool was = has_max_;
      return has_max_ = false, was;
    }
    if (t_.empty()) return false;
    size_t i = home(k);
    for (;; i = (i + 1) & mask_) {
      if (t_[i].first == k) break;
      if (t_[i].first == kEmpty) return false;
    }
    for (size_t j = (i + 1) & mask_;; j = (j + 1) & mask_) {  // close the hole: move back each pair that may sit earlier
      if (t_[j].first == kEmpty) break;
      const size_t h = home(t_[j].first);
      if (((j - h) & mask_) >= ((j - i) & mask_)) t_[i] = t_[j], i = j;
    }
    t_[i].first = kEmpty;
    --n_;
    return true;
  }
  template <class F>
  void for_each(F f) const {
    for (const Slot& s : t_)
      if (s.first != kEmpty) f(s.first, s.second);
    if (has_max_) f(kEmpty, max_);
  }

 private:
  size_t home(uint64_t k) const { return size_t(mix64(k)) & mask_; }
  void grow() {
    std::vector<Slot> old;
    old.swap(t_);
    t_.assign(old.empty() ? 64 : old.size() * 2, Slot(kEmpty, P()));
    mask_ = t_.size() - 1;
    for (const Slot& s : old)
      if (s.first != kEmpty) {
        size_t i = home(s.first);
        while (t_[i].first != kEmpty) i = (i + 1) & mask_;
        t_[i] = s;
      }
  }

  std::vector<Slot> t_;
  size_t mask_ = 0, n_ = 0;
  bool has_max_ = false;
  P max_ = P();
};

}  // namespace gret
