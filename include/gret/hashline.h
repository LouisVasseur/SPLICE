// SPDX-License-Identifier: MIT
// HashLineIndex<K, P, B>: point lookups at the least cost in memory accesses and bytes: no model, no top, no key order.
// Read-only, exact for hits and misses. This is a hash table, not a learned index; it is SPLICE's slots with T removed,
// the design that hashed keys reduce it to (a single line over hashed keys beat every segmentation of them).
//
//   z = Mix(key)                       a bijection on 64 bits, mix64 (SplitMix64's finalizer) or mul64: equal z means equal key.
//                                      With Checked, the build falls back to mix64 where mul64 reads too many lines (prefer_mix64)
//   b = floor(z * buckets / 2^64)      the key's bucket, B slots, one cache line when B = 8 on Apple silicon
//                                      (hw.cachelinesize 128) and when B = 4 on x86 (64)
//
// bulk_load: the (z, payload) pairs, sorted by z, each in the first free slot of its bucket or later. An empty slot holds
// a copy of the next pair, so the slots stay sorted, as in SpliceIndex (trailing slots: the last pair). get reads its
// bucket, counts the keys below z without branching, and the first key >= z decides; all B below z moves to the next
// bucket, at most d (the build's exact maximum displacement, in buckets) further.
// Environment: HASHLINE_SPK, slots per key (default 1.25); bytes per key are 16 times that.
// Keys must be unique, as GRE's driver guarantees. put, update and remove return false, and scan returns 0: z has no
// order that means anything to a caller.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gret {

inline uint64_t mix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// A scan over slots sorted by raw key, where an empty slot holds a copy of the next pair, behind exceptions peeled from below and
// above them: exceptions below the slots, then the slots from locate(x) on (the first slot whose key is >= x), copies skipped,
// then exceptions above. locate is called only with x above the first key of the slots and at most the last.
template <class K, class P, class Slots, class Locate>
size_t scan_sorted_slots(const Slots& slots, const std::vector<std::pair<K, P>>& exc, K key, size_t n, std::pair<K, P>* out, Locate locate) {
  const uint64_t x = uint64_t(key), first = slots.front().first, last = slots.back().first;
  auto e = std::lower_bound(exc.begin(), exc.end(), key, [](const std::pair<K, P>& a, K k) { return a.first < k; });
  size_t got = 0;
  while (got < n && e != exc.end() && uint64_t(e->first) < first) out[got++] = *e++;
  if (got < n && x <= last) {
    size_t i = x <= first ? 0 : locate(x);
    bool any = false;
    uint64_t prev = 0;
    for (; got < n && i < slots.size(); ++i) {
      if (any && slots[i].first == prev) continue;
      prev = slots[i].first, any = true;
      out[got++] = std::pair<K, P>(K(prev), slots[i].second);
    }
  }
  while (got < n && e != exc.end()) out[got++] = *e++;
  return got;
}

// The same scan, taking each key's payload from the last of its copies rather than the first. For indexes that write in place
// (AdaptiveLineIndex::update): the copies a lookup can read are the last ones of a run, so those are the ones a write keeps
// current, and the copies before them may hold an older payload.
template <class K, class P, class Slots, class Locate>
size_t scan_sorted_slots_last(const Slots& slots, const std::vector<std::pair<K, P>>& exc, K key, size_t n, std::pair<K, P>* out, Locate locate) {
  const uint64_t x = uint64_t(key), first = slots.front().first, last = slots.back().first;
  auto e = std::lower_bound(exc.begin(), exc.end(), key, [](const std::pair<K, P>& a, K k) { return a.first < k; });
  size_t got = 0;
  while (got < n && e != exc.end() && uint64_t(e->first) < first) out[got++] = *e++;
  if (got < n && x <= last) {
    const size_t m = slots.size();
    for (size_t i = x <= first ? 0 : locate(x); got < n && i < m; ++i) {
      if (i + 1 < m && slots[i + 1].first == slots[i].first) continue;
      out[got++] = std::pair<K, P>(K(slots[i].first), slots[i].second);
    }
  }
  while (got < n && e != exc.end()) out[got++] = *e++;
  return got;
}

// The lines a lookup of each stored key reads, exactly, as the build places them: the mean and the tail.
class LineCount {
 public:
  void add(size_t lines) {
    sum_ += double(lines);
    ++keys_;
    worst_ = std::max(worst_, lines);
    ++bins_[std::min(lines, kBins - 1)];
  }
  double mean() const { return keys_ ? sum_ / double(keys_) : 0; }
  size_t worst() const { return worst_; }
  size_t tail(double q) const {  // the fewest lines that at least q of the keys read no more than
    const double need = q * double(keys_);
    size_t seen = 0;
    for (size_t l = 0; l < kBins; ++l)
      if (double(seen += bins_[l]) >= need) return l;
    return worst_;
  }

 private:
  static constexpr size_t kBins = 1024;
  size_t bins_[kBins] = {}, keys_ = 0, worst_ = 0;
  double sum_ = 0;
};

template <class T>
struct BucketAlloc {  // aligned to the largest cache line in use (128 bytes), so bucket boundaries are line boundaries
  using value_type = T;
  BucketAlloc() = default;
  template <class U> BucketAlloc(const BucketAlloc<U>&) {}
  T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(128))); }
  void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t(128)); }
  template <class U> bool operator==(const BucketAlloc<U>&) const { return true; }
  template <class U> bool operator!=(const BucketAlloc<U>&) const { return false; }
};

// One multiply by an odd constant: also a bijection on 64 bits, and cheaper. On the ten GRE sets it never reads more lines than
// mix64, and fewer on the dense ones (stack, libio). It mixes less, and some regular keys clump badly: keys 12345 + i * 65536
// read 15.65 lines where mix64 reads 1.38, and larger Fibonacci strides pile into one arc. See prefer_mix64.
inline uint64_t mul64(uint64_t x) { return x * 0x9E3779B97F4A7C15ull; }

// What mix64 would read on these keys, from a sample with the buckets scaled down along with it: a good hash's line count
// depends on the load alone.
template <unsigned B, class K, class P>
LineCount mix64_lines(const std::pair<K, P>* kv, size_t n, double spk) {
  __extension__ typedef unsigned __int128 u128;
  const size_t step = n >= (size_t(1) << 20) ? 64 : 1;
  std::vector<uint64_t> z;
  z.reserve(n / step + 1);
  for (size_t i = 0; i < n; i += step) z.push_back(mix64(uint64_t(kv[i].first)));
  std::sort(z.begin(), z.end());
  const size_t buckets = std::max<size_t>(1, size_t(std::ceil(double(z.size()) * spk / B)));
  LineCount lc;
  size_t prev = SIZE_MAX;
  for (const uint64_t x : z) {
    const size_t p = size_t((u128(x) * buckets) >> 64), s = p * B, pos = prev == SIZE_MAX || s > prev ? s : prev + 1;
    lc.add(pos / B - p + 1);
    prev = pos;
  }
  return lc;
}

// Whether one multiply clumps enough to pay for mix64. Its extra multiplies cost about 5 ns a lookup and a line about 33 ns
// (hl4 against hf4 at 20 bytes per key, Jed), hence 0.15 lines on average; the tail may not pass twice mix64's, whatever the average.
inline bool prefer_mix64(const LineCount& mul, const LineCount& mix) {
  return mul.mean() > mix.mean() + 0.15 || mul.worst() > 2 * mix.worst();
}

template <class K, class P, unsigned B, uint64_t (*Mix)(uint64_t) = mix64, bool Checked = false>
class HashLineIndex final : public indexInterface<K, P> {
  __extension__ typedef unsigned __int128 u128;  // __extension__: no -Wpedantic warning
  using Slot = std::pair<uint64_t, P>;

 public:
  explicit HashLineIndex(std::string name) : name_(std::move(name)) {}

  void init(Param* = nullptr) override {}

  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    if (n == 0) throw std::invalid_argument("HashLineIndex: bulk_load of 0 keys");
    const char* e = std::getenv("HASHLINE_SPK");
    const double spk = e && *e ? std::strtod(e, nullptr) : 1.25;
    buckets_ = std::max<size_t>(1, size_t(std::ceil(double(n) * spk / B)));
    std::vector<Slot> a(n);
    hash_sorted(a, kv);
    size_t prev;
    LineCount lc = place_all(a, prev);
    if constexpr (Checked) {  // Mix against mix64 on these keys, by exact line counts
      fallback_ = prefer_mix64(lc, mix64_lines<B>(kv, n, spk));
      if (fallback_) {
        hash_sorted(a, kv);
        lc = place_all(a, prev);
      }
    }
    d_ = lc.worst() - 1;
    e_lines_ = lc.mean();
    total_ = std::max(buckets_, prev / B + 1);
    slots_.resize(total_ * B);
    size_t i = 0;
    prev = SIZE_MAX;
    for (size_t r = 0; r < n; ++r) {
      const size_t pos = place(bucket(a[r].first), prev);
      while (i <= pos) slots_[i++] = a[r];
      prev = pos;
    }
    while (i < slots_.size()) slots_[i++] = a[n - 1];
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "gret: %s n=%zu B=%u slots/key=%.3f buckets=%zu d=%zu E[buckets]=%.4f lines_p99=%zu lines_p999=%zu lines_max=%zu fallback=%d bytes_per_key=%.2f build_ms=%.1f\n",
                 name_.c_str(), n, B, double(slots_.size()) / double(n), total_, d_, lc.mean(), lc.tail(0.99), lc.tail(0.999), lc.worst(), int(fallback_),
                 double(memory_consumption()) / double(n), ms);
  }

  bool get(K key, P& val, Param* = nullptr) override {
    const uint64_t z = hash(uint64_t(key));
    size_t b = bucket(z);
    const size_t b0 = b;
    for (const Slot* l = &slots_[b * B];; l += B) {
      size_t c = 0;
      for (unsigned j = 0; j < B; ++j) c += size_t(l[j].first < z);
      if (__builtin_expect(c < B, 1)) {  // the first key >= z decides
        if (l[c].first != z) return false;
        val = l[c].second;
        return true;
      }
      if (++b > b0 + d_ || b >= total_) return false;
    }
  }

  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  size_t scan(K, size_t, std::pair<K, P>*, Param* = nullptr) override { return 0; }
  long long memory_consumption() override { return (long long)(slots_.size() * sizeof(Slot)); }
  double expected_lines() const { return e_lines_; }  // lines a lookup of a stored key reads, from the build's exact count
  size_t worst_lines() const { return d_ + 1; }
  bool uses_fallback() const { return fallback_; }

 private:
  uint64_t hash(uint64_t x) const { return Checked && fallback_ ? mix64(x) : Mix(x); }
  size_t bucket(uint64_t z) const { return size_t((u128(z) * buckets_) >> 64); }
  void hash_sorted(std::vector<Slot>& a, const std::pair<K, P>* kv) const {
    for (size_t i = 0; i < a.size(); ++i) a[i] = Slot(hash(uint64_t(kv[i].first)), kv[i].second);
    std::sort(a.begin(), a.end(), [](const Slot& x, const Slot& y) { return x.first < y.first; });
  }
  LineCount place_all(const std::vector<Slot>& a, size_t& prev) const {
    LineCount lc;
    prev = SIZE_MAX;
    for (const Slot& s : a) {
      const size_t p = bucket(s.first), pos = place(p, prev);
      lc.add(pos / B - p + 1);
      prev = pos;
    }
    return lc;
  }
  static size_t place(size_t p, size_t prev) {  // the first free slot of bucket p or later
    const size_t s = p * B;
    return prev == SIZE_MAX || s > prev ? s : prev + 1;
  }

  std::string name_;
  size_t buckets_ = 1, total_ = 1, d_ = 0;
  bool fallback_ = false;
  double e_lines_ = 0;
  std::vector<Slot, BucketAlloc<Slot>> slots_;
};

}  // namespace gret
