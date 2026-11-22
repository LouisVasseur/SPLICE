// SPDX-License-Identifier: MIT
// Hist-Tree in its compact form, CHT (Crotty: "Hist-Tree: those who ignore it are doomed to learn", CIDR 2021), in GRE's harness: a read-only
// index that is NOT learned, a radix tree of histograms whose terminal bins bound the search range. This is a WRAPPER around the open
// implementation of the compact Hist-Tree by Mihail Stoian (github.com/stoianmihail/CHT, include/cht, the code that PLEX carries as ts_cht),
// used unmodified and included by path:  -I <CHT>/include.  Nothing of it is copied here. It is Stoian's implementation, not Crotty's own C
// code, which was not found published (docs/sota/cht.md).
//
//   ChtBound<K>   a Model for SortedBoundIndex (readonly.h): cht::Builder over the sorted keys, cht::CompactHistTree::GetSearchBound per key
//   make_cht      the index by name: cht<bins>e<max_error>, e.g. cht64e32 (the README's example), cht16e8, cht64e1024; ccht<bins>e<max_error>
//                 is the same tree in the builder's cache-oblivious node order (its use_cache flag). bins: a power of two in [2, 2^20];
//                 max_error in [1, 2^20]. Any other name: nullptr.
//   ChtStdIndex   the same names with "std" at the end (cht64e32std, ccht16e8std): the same array of pairs and the same tree, searched as
//                 the authors' own end-to-end benchmark searches it (bench_end_to_end.cc): std::lower_bound over GetSearchBound's
//                 [begin, end). SortedBoundIndex's search is branch-free over a window of max_error positions; a CHT bound is one-sided,
//                 the key is among the first positions of the window whenever its bin holds fewer than max_error keys, and a search
//                 with branches (std::lower_bound as g++ compiles it) predicts that. Which form is faster depends on the compiler and
//                 the data. Measured through GRE's interface on arm64, 4M to 5M keys, best setting of a sweep: with g++ 15 the authors'
//                 search is 11 and 14% faster on two skewed key sets, 9% slower on uniform keys, and up to 2x faster where the window is
//                 much wider than the bins are full; with Apple clang SortedBoundIndex's search is 17 to 19% faster on all three
//                 (docs/sota/cht.md). Run both forms and say which is reported: without "std" CHT is on the footing of the other
//                 read-only indexes here, with it CHT is searched as its authors search it.
//
// What the authors' code does, as read from it (builder.h, cht.h) and pinned by tests/sota/test_cht.cpp:
//   - every node is `bins` equal-width bins over a power-of-two range that starts at the smallest key; a bin with more than max_error keys
//     gets a child node over its range, any other is terminal and holds the position of its first key. The table is one array of 32-bit
//     entries, `bins` per node, top bit = terminal. A lookup walks it with a shift and a load per level and returns
//     [begin, begin + max_error + 1), cut at n. Keys at or below the smallest go to position 0, at or above the largest to n - 1.
//   - positions and node numbers are 31-bit: fewer than 2^31 keys (build() throws above that), and at most 2^31 nodes (a node number
//     with the top bit set would be read as a terminal bin: build() throws after Finalize if the table has more).
//   - DEFECT 1, the largest key: the builder bins the largest key like any other (initNode), and when max - min is an exact power of two
//     that bin is one past the root's last: a heap overflow, a write at builder.h:245 (ASan: heap-buffer-overflow). build() never lets
//     it happen: while the range of the keys it hands over is a power of two it leaves the top key out (at most 64 keys, since the range
//     at least halves each time), and get() sends anything above the last key handed over to the tail [built, n), as PgmBound does for
//     PGM's sentinel.
//   - DEFECT 2, a partial last level: the root's bin width is 2^shift with shift = ceil(log2(max - min)) - log2(bins), and every level takes
//     log2(bins) bits, so unless log2(bins) divides shift the deepest level has bins narrower than `bins` keys (w = shift mod log2(bins)
//     bits wide). If such a bin still holds more than max_error keys (a dense run, with max_error < 2^w <= bins / 2), the builder splits it
//     into a node whose width is negative: builder.h:233 (UBSan: shift exponent 4294967294), then :280 and cht.h:60, shift by an
//     unsigned-wrapped count. Undefined; where the machine takes the count mod 64 (x86-64, arm64) it chains about ten useless nodes per
//     such bin (50000 consecutive keys at 64 bins, max error 8: an 8.8 MB table, 13 KB at max error 16), and for some widths cannot end.
//     build() counts the keys per such bin first and throws std::invalid_argument when one is over max_error: that key set cannot be
//     indexed at that setting by this code. max_error >= bins / 2 can never throw; below that it depends on the data (and on the key
//     range, through w: with log2(bins) dividing ceil(log2(max - min)), as 16 or 256 bins on keys that span all 64 bits, w is 0).
//   - a key range narrower than one node (ceil(log2(max - min)) < log2(bins); the builder asserts against it) gets no tree: the bound is
//     the whole array, which then has at most bins / 2 + 1 keys before the ones left out for defect 1.
//   - duplicates can leave a terminal bin over max_error (the builder says so): build() throws unless the keys are strictly increasing.
//
// The bound handed to SortedBoundIndex is [begin, min(begin + max_error, built)], one position shorter than GetSearchBound's: a terminal
// bin holds at most max_error keys, so a stored key is at begin + max_error - 1 at most, and the lower bound of any key at begin +
// max_error at most, which SortedBoundIndex's search can still return. That saves the baseline a probe when max_error is a power of two.
//
// Build memory: the builder keeps its own copy of the keys and its pointer tree until Finalize returns; build() frees the caller's key
// vector (SortedBoundIndex's scratch copy) before Finalize, so the peak is the pairs, one key copy, the tree and the table.
//
// Include after indexInterface.h and GRE's pgm header, like readonly.h (which this includes).
#pragma once
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>  // cht/builder.h uses std::function and std::cerr and includes neither header
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cht/builder.h"  // Stoian's compact Hist-Tree: -I <CHT>/include
#include "cht/cht.h"
#include "gret/readonly.h"

namespace gret {
namespace sota {

struct ChtConfig {
  size_t bins = 64;       // per node, a power of two (the README's example: 64)
  size_t max_error = 32;  // most keys in a terminal bin (the README's example: 32)
  bool cache = false;     // the builder's cache-oblivious node order
  bool std_search = false;  // the index searches as the authors do (ChtStdIndex), not as SortedBoundIndex does; the tree is the same
  std::string name() const { return (cache ? "ccht" : "cht") + std::to_string(bins) + "e" + std::to_string(max_error) + (std_search ? "std" : ""); }
};

inline bool cht_config_ok(const ChtConfig& c) {
  return c.bins >= 2 && c.bins <= (size_t(1) << 20) && (c.bins & (c.bins - 1)) == 0 && c.max_error >= 1 && c.max_error <= (size_t(1) << 20);
}

// cht<bins>e<max_error> or ccht<bins>e<max_error>, decimal without leading zeros, within cht_config_ok; "std" may follow (ChtStdIndex)
inline bool parse_cht_name(const std::string& name, ChtConfig& out) {
  ChtConfig c;
  size_t p = 0;
  if (name.compare(0, 4, "ccht") == 0) c.cache = true, p = 4;
  else if (name.compare(0, 3, "cht") == 0) p = 3;
  else return false;
  auto number = [&](size_t& v) {
    const size_t b = p;
    v = 0;
    while (p < name.size() && name[p] >= '0' && name[p] <= '9' && p - b < 8) v = v * 10 + size_t(name[p++] - '0');
    return p > b && name[b] != '0';
  };
  if (!number(c.bins) || p >= name.size() || name[p++] != 'e' || !number(c.max_error)) return false;
  if (name.compare(p, std::string::npos, "std") == 0) c.std_search = true, p += 3;
  if (p != name.size() || !cht_config_ok(c)) return false;
  out = c;
  return true;
}

// SortedBoundIndex default-constructs its Model, so the settings of the next ChtBound built on this thread are left here by make_cht (and
// put back to the defaults by it).
inline ChtConfig& cht_pending() {
  static thread_local ChtConfig c;
  return c;
}

inline unsigned cht_ceil_log2(uint64_t v) {  // ceil(log2(v)) for v >= 1, as the builder's computeLog(v, true)
  return v <= 1 ? 0 : 64 - unsigned(__builtin_clzll(v - 1));
}

template <class K>
struct ChtBound {
  static_assert(std::is_same<K, uint64_t>::value, "ChtBound: uint64_t keys");
  ChtConfig cfg = cht_pending();
  cht::CompactHistTree<K> m;
  bool tree = false;      // false: the key range is narrower than one node, no tree
  K top = 0;              // the last key the tree was built on
  size_t built = 0, n = 0;  // keys in the tree, keys in all: the n - built largest are behind `top` (defect 1)

  // The most keys in one bin of the deepest level, when that level is partial and its bins could hold more than max_error keys (defect
  // 2); 0 when the builder cannot be led there. keys[0..built) strictly increasing, their range not a power of two and not below bins / 2.
  static size_t deepest_bin_load(const std::vector<K>& keys, size_t built, const ChtConfig& cfg) {
    const unsigned log = cht_ceil_log2(cfg.bins), lg = cht_ceil_log2(uint64_t(keys[built - 1]) - uint64_t(keys[0]));
    const unsigned w = (lg - log) % log;
    if (w == 0 || (uint64_t(1) << w) <= cfg.max_error) return 0;
    size_t worst = 0, run = 0;
    uint64_t bin = 0;
    for (size_t i = 0; i < built; ++i) {
      const uint64_t b = (uint64_t(keys[i]) - uint64_t(keys[0])) >> w;
      run = i && b == bin ? run + 1 : 1;
      bin = b;
      worst = std::max(worst, run);
    }
    return worst;
  }

  void build(std::vector<K>& keys) {
    const std::string who = cfg.name();
    if (!cht_config_ok(cfg)) throw std::invalid_argument(who + ": bins must be a power of two in [2, 2^20] and max_error in [1, 2^20]");
    n = keys.size();
    if (n == 0) throw std::invalid_argument(who + ": no keys");
    if (n >= (size_t(1) << 31)) throw std::invalid_argument(who + ": 2^31 keys or more, and CHT holds positions in 31 bits");
    for (size_t i = 1; i < n; ++i)
      if (!(keys[i - 1] < keys[i])) throw std::invalid_argument(who + ": keys must be sorted and unique (CHT's bound does not hold with duplicates)");
    const K lo = keys[0];
    built = n;  // defect 1: no power-of-two range reaches the builder (a range of 0 or 1 counts as one)
    while (built > 1 && ((uint64_t(keys[built - 1]) - uint64_t(lo)) & (uint64_t(keys[built - 1]) - uint64_t(lo) - 1)) == 0) --built;
    top = keys[built - 1];
    const unsigned log = cht_ceil_log2(cfg.bins);
    tree = built > 1 && cht_ceil_log2(uint64_t(top) - uint64_t(lo)) >= log;
    if (tree) {
      const size_t load = deepest_bin_load(keys, built, cfg);
      if (load > cfg.max_error)  // defect 2
        throw std::invalid_argument(who + ": CHT cannot index this key set at this setting: a bin narrower than one node holds " + std::to_string(load) +
                                    " keys, over max error " + std::to_string(cfg.max_error) + ", and its builder would split it at a negative width (max_error >= " +
                                    std::to_string(load) + " builds here; max_error >= bins / 2 always does)");
      cht::Builder<K> b(lo, top, cfg.bins, cfg.max_error, /*single_pass=*/false, /*use_cache=*/cfg.cache);
      for (size_t i = 0; i < built; ++i) b.AddKey(keys[i]);
      std::vector<K>().swap(keys);  // the builder has its own copy; the caller's is scratch (see readonly.h)
      m = b.Finalize();
      if ((m.GetSize() - sizeof(m)) / (sizeof(unsigned) * cfg.bins) > (size_t(1) << 31))
        throw std::invalid_argument(who + ": more than 2^31 nodes, and CHT holds node numbers in 31 bits");
    }
    std::fprintf(stderr, "gret: %s tree=%d nodes=%zu keys_behind_top=%zu\n", who.c_str(), int(tree), tree ? (m.GetSize() - sizeof(m)) / (sizeof(unsigned) * cfg.bins) : size_t(0),
                 n - built);
  }

  KeyBound get(K x) const {
    if (__builtin_expect(x > top, 0)) return {built, n};
    if (__builtin_expect(!tree, 0)) return {0, built};
    const size_t begin = m.GetSearchBound(x).begin;
    return {begin, std::min(begin + cfg.max_error, built)};
  }

  // The authors' own bound, [begin, end) as GetSearchBound gives it, for std::lower_bound over it (ChtStdIndex). The tail and the
  // no-tree case as in get().
  KeyBound std_bound(K x) const {
    if (__builtin_expect(x > top, 0)) return {built, n};
    if (__builtin_expect(!tree, 0)) return {0, built};
    const cht::SearchBound s = m.GetSearchBound(x);
    return {s.begin, s.end};
  }

  // the authors' GetSize() (their object and the table) plus what this wrapper adds around it
  size_t bytes() const { return sizeof(*this) - sizeof(m) + (tree ? m.GetSize() : sizeof(m)); }
};

// CHT as its authors use it (bench_end_to_end.cc, the README): one sorted array of (key, payload) pairs, GetSearchBound, then
// std::lower_bound over [begin, end). Everything else as SortedBoundIndex: fewer than 64 keys get no model, the range check first,
// exact for hits and misses (the bound holds for keys never stored too, see the top of this file), scan copies from the lower bound,
// no writes, memory = pairs + model.
template <class K, class P>
class ChtStdIndex final : public indexInterface<K, P> {
  using Pair = std::pair<K, P>;

 public:
  explicit ChtStdIndex(std::string name) : name_(std::move(name)) {}
  void init(Param* = nullptr) override {}
  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    if (n == 0) throw std::invalid_argument("ChtStdIndex: bulk_load of 0 keys");
    a_.assign(kv, kv + n);
    small_ = n < 64;
    if (!small_) {
      std::vector<K> keys(n);
      for (size_t i = 0; i < n; ++i) keys[i] = kv[i].first;
      m_.build(keys);
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "gret: %s n=%zu dir_bytes=%zu bytes_per_key=%.2f build_ms=%.1f\n", name_.c_str(), n, small_ ? size_t(0) : m_.bytes(),
                 double(memory_consumption()) / double(n), ms);
  }
  bool get(K key, P& val, Param* = nullptr) override {
    if (__builtin_expect(key < a_.front().first || key > a_.back().first, 0)) return false;
    const size_t i = lower(key);
    if (i >= a_.size() || a_[i].first != key) return false;
    val = a_[i].second;
    return true;
  }
  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  size_t scan(K key, size_t n, std::pair<K, P>* out, Param* = nullptr) override {
    if (key > a_.back().first) return 0;
    const size_t i = key <= a_.front().first ? 0 : lower(key);
    const size_t got = std::min(n, a_.size() - i);
    std::copy_n(a_.data() + i, got, out);
    return got;
  }
  long long memory_consumption() override { return (long long)(a_.size() * sizeof(Pair) + (small_ ? size_t(0) : m_.bytes())); }

 private:
  size_t lower(K x) const {  // the position of the first key >= x, for x within the key range
    const KeyBound b = small_ ? KeyBound{0, a_.size()} : m_.std_bound(x);
    const Pair* const base = a_.data();
    return size_t(std::lower_bound(base + b.lo, base + b.hi, x, [](const Pair& p, K k) { return p.first < k; }) - base);
  }

  std::string name_;
  bool small_ = false;
  std::vector<Pair> a_;
  ChtBound<K> m_;
};

// cht<bins>e<max_error> and ccht<bins>e<max_error>, and the same with "std" at the end (see the top of this file); nullptr for every other
// name and for keys other than uint64_t
template <class K, class P>
indexInterface<K, P>* make_cht(const std::string& name) {
  if constexpr (!std::is_same<K, uint64_t>::value) {
    (void)name;
    return nullptr;
  } else {
    ChtConfig c;
    if (!parse_cht_name(name, c)) return nullptr;
    struct Restore {
      ~Restore() { cht_pending() = ChtConfig(); }
    } restore;
    cht_pending() = c;
    if (c.std_search) return new ChtStdIndex<K, P>(name);
    return new SortedBoundIndex<K, P, ChtBound<K>>(name);
  }
}

}  // namespace sota
}  // namespace gret
