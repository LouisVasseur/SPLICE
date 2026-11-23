// SPDX-License-Identifier: MIT
// OverflowTable<P>: where a writable index keeps the keys its slots cannot take (G, for writes): an open-addressing hash table of
// 16-byte (key, payload) pairs, one probe sequence per lookup, usually one cache line. Linear probing on mix64, any number of
// slots (the hash is scaled to it by a multiply, not masked), dense or not (set_dense: 21 to 32 or 26 to 51 bytes a key; the
// writable index uses not dense, which makes its puts 5 to 25% faster at 10M keys); erase by moving the following pairs back. The key 2^64 - 1 marks an empty slot, so
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
  // dense (the default): grown by half when 3/4 full, 21 to 32 bytes a key. Not dense: doubled when 5/8 full, 26 to 51 bytes a
  // key, but each key is moved half as often when the table grows and probe runs are shorter. Set before the first insert.
  void set_dense(bool dense) { num_ = dense ? 3 : 5, den_ = dense ? 4 : 8, half_ = dense; }
  size_t size() const { return n_ + size_t(has_max_); }
  bool empty() const { return n_ == 0 && !has_max_; }
  size_t bytes() const { return t_.size() * sizeof(Slot); }
  bool find(uint64_t k, P& v) const {
    if (__builtin_expect(k == kEmpty, 0)) {
      if (has_max_) v = max_;
      return has_max_;
    }
    if (t_.empty()) return false;
    for (size_t i = home(k);; i = next(i)) {
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
    if ((n_ + 1) * den_ > t_.size() * num_) grow();
    for (size_t i = home(k);; i = next(i)) {
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
    for (size_t i = home(k);; i = next(i)) {
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
    for (;; i = next(i)) {
      if (t_[i].first == k) break;
      if (t_[i].first == kEmpty) return false;
    }
    const size_t cap = t_.size();
    for (size_t j = next(i);; j = next(j)) {  // close the hole: move back each pair that may sit earlier
      if (t_[j].first == kEmpty) break;
      const size_t h = home(t_[j].first);
      if ((j + cap - h) % cap >= (j + cap - i) % cap) t_[i] = t_[j], i = j;
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
  __extension__ typedef unsigned __int128 u128;
  size_t home(uint64_t k) const { return size_t((u128(mix64(k)) * t_.size()) >> 64); }
  size_t next(size_t i) const { return ++i == t_.size() ? 0 : i; }
  void grow() {
    std::vector<Slot> old;
    old.swap(t_);
    t_.assign(old.empty() ? 64 : (half_ ? old.size() + old.size() / 2 : old.size() * 2), Slot(kEmpty, P()));
    for (const Slot& s : old)
      if (s.first != kEmpty) {
        size_t i = home(s.first);
        while (t_[i].first != kEmpty) i = next(i);
        t_[i] = s;
      }
  }

  std::vector<Slot> t_;
  size_t n_ = 0, num_ = 3, den_ = 4;
  bool half_ = true;
  bool has_max_ = false;
  P max_ = P();
};

}  // namespace gret
