// SPDX-License-Identifier: MIT
// TransformIndex<K, P, Base, Tr>: one of GRE's own indexes (Base, e.g. LIPPInterface<uint64_t, P>) built on transformed
// keys t = Tr(k) in [0, 2^52) instead of the raw keys (NFL-style: Wu et al., PVLDB 2022). Tr is PwlCdf or MolFlow.
//
// Read-only point lookups (GRE's --read=1 --insert=0): get() of a bulk-loaded key returns its own payload, always;
// put/update/remove return false and scan returns 0 (a transform need not preserve order). A key that was never loaded
// may return a false positive: the payload of a loaded key with the same t. On a LIPP base such a lookup can also end
// the process, as on raw LIPP: LIPP's at() RT_ASSERTs (exit(0)) when the slot it reaches holds another key.
//
// bulk_load: t_i = Tr(k_i) (8 bytes per key); when t is not non-decreasing over the sorted keys (NF rounding), the
// ranks are sorted by (t, i). Every group of two or more equal t is a collision group: its members go to overflow_
// (raw key and payload, sorted by key) and its t to collided_ (sorted, unique). The unique (t, payload) pairs go to
// the base.
// If no t is unique, the base gets one sentinel t outside collided_ (payload P()), so that it is never empty, and the
// filter is set to all ones, so that no lookup reaches it: an unloaded key then always returns false.
// get: a one-word-per-64-collided-t Bloom filter (one hash, false-positive rate <= 1/64) in front of collided_ keeps the
// common path to a multiply, one load and one predictable branch, then base_.get(t). When nothing collided, the filter
// is a single zero word and the branch is never taken.
// memory_consumption: the base's own figure plus the transform, the overflow, collided_ and the filter; 0 (GRE's "not
// reported") when the base reports 0 (GRE's B+tree), so that the overhead alone never reads as a total. The stderr
// line gives the overhead as aux_bytes either way.
//
// Include GRE's competitor headers (indexInterface.h and the Base's interface header) before this one: they have no
// include guards, so the gret headers do not include them themselves.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "gret/common.h"
#include "gret/flow.h"
#include "gret/nflflow.h"
#include "gret/pwlcdf.h"

namespace gret {

[[noreturn]] inline void die(const std::string& msg) {
  std::fprintf(stderr, "gret: %s\n", msg.c_str());
  std::fflush(stderr);
  std::exit(1);
}

// How the wrapper obtains its transform at bulk_load: PwlCdf is built on the keys with K knots; MolFlow is flow if set
// (the gate loads it once, to check it), else loaded from flow_path (empty: the file named by $TRANSFORM_FLOW).
struct TransformSpec {
  unsigned K = 512;
  std::string flow_path;
  std::shared_ptr<const MolFlow> flow;
};

inline std::string flow_path_of(const TransformSpec& s) {
  if (!s.flow_path.empty()) return s.flow_path;
  const char* e = std::getenv("TRANSFORM_FLOW");
  return e ? e : "";
}

template <class P>
PwlCdf make_transform(PwlCdf*, const TransformSpec& s, const std::pair<uint64_t, P>* kv, size_t n) {
  try {  // reads only the K + 1 knot keys of GRE's (key, payload) pairs
    return PwlCdf::build_by(n, [kv](size_t i) { return uint64_t(kv[i].first); }, s.K);
  } catch (const std::exception& e) {
    die(e.what());
  }
}

template <class P>
MolFlow make_transform(MolFlow*, const TransformSpec& s, const std::pair<uint64_t, P>*, size_t) {
  if (s.flow) return *s.flow;
  const std::string path = flow_path_of(s);
  if (path.empty()) die("nf needs a flow file: set TRANSFORM_FLOW=<dataset>.flow (transform/train_flow.py)");
  try {
    return MolFlow::load(path);
  } catch (const std::exception& e) {
    die(e.what());
  }
}

// NflFlow: NFL's own flow, loaded from $TRANSFORM_NFL (NFL's weight file for the dataset), its z range fitted on the
// build keys.
template <class P>
NflFlow make_transform(NflFlow*, const TransformSpec& s, const std::pair<uint64_t, P>* kv, size_t n) {
  const char* e = std::getenv("TRANSFORM_NFL");
  const std::string path = !s.flow_path.empty() ? s.flow_path : e ? e : "";
  if (path.empty()) die("nfl needs NFL's weights: set TRANSFORM_NFL=<dataset>.nfl (NFL's train/numerical_flow.py)");
  try {
    NflFlow f = NflFlow::load(path);
    f.fit(n, [kv](size_t i) { return uint64_t(kv[i].first); });
    return f;
  } catch (const std::exception& ex) {
    die(ex.what());
  }
}

// Extra fields for the stderr line: MolFlow says which a * b + c its build fuses (flow.h; T differs by 1 on ~0.1% of
// the keys between the two, so collision sets and nf labels must not be mixed across them).
template <class Tr>
std::string transform_note(const Tr&) { return ""; }
inline std::string transform_note(const MolFlow&) { return MolFlow::kFusedMadd ? " fma=1" : " fma=0"; }
inline std::string transform_note(const NflFlow& f) {
  char b[80];
  std::snprintf(b, sizeof b, " zlo=%.9g zhi=%.9g", f.zlo(), f.zhi());
  return b;
}

template <class K, class P, class Base, class Tr>
class TransformIndex final : public indexInterface<K, P> {
  static_assert(std::is_same<K, uint64_t>::value, "the transforms take uint64_t keys");

 public:
  explicit TransformIndex(std::string name, TransformSpec spec = {}) : name_(std::move(name)), spec_(std::move(spec)) {}
  TransformIndex(const TransformIndex&) = delete;  // some bases own raw pointers (ART)
  TransformIndex& operator=(const TransformIndex&) = delete;

  void init(Param* param = nullptr) override { base_.init(param); }

  void bulk_load(std::pair<K, P>* kv, size_t n, Param* param = nullptr) override {
    const auto t0 = std::chrono::steady_clock::now();
    if (n == 0) die(name_ + ": bulk_load of 0 keys");
    tr_ = make_transform(static_cast<Tr*>(nullptr), spec_, kv, n);
    n_ = n;
    std::vector<uint64_t> ts(n);  // t of key i
    bool sorted = true;
    for (size_t i = 0; i < n; ++i) {
      const uint64_t t = tr_(kv[i].first);
      if (t > kOutMax) die(name_ + ": transform output above 2^52 - 1");
      ts[i] = t;
      sorted &= i == 0 || ts[i - 1] <= t;
    }
    std::vector<size_t> perm;  // rank -> key index by (t, i); only when t is not non-decreasing (rank = i otherwise)
    if (!sorted) {
      perm.resize(n);
      for (size_t i = 0; i < n; ++i) perm[i] = i;
      std::sort(perm.begin(), perm.end(),
                [&ts](size_t a, size_t b) { return ts[a] < ts[b] || (ts[a] == ts[b] && a < b); });
    }
    auto at = [&](size_t r) { return sorted ? r : perm[r]; };
    size_t unique = 0, collided = 0, groups = 0;
    for (size_t a = 0, b; a < n; a = b) {
      for (b = a + 1; b < n && ts[at(b)] == ts[at(a)]; ++b) {}
      if (b - a == 1) ++unique; else collided += b - a, ++groups;
    }
    std::vector<std::pair<uint64_t, P>> bp;
    bp.reserve(unique ? unique : 1);
    overflow_.clear(), overflow_.reserve(collided);
    collided_.clear(), collided_.reserve(groups);
    for (size_t a = 0, b; a < n; a = b) {
      const uint64_t t = ts[at(a)];
      for (b = a + 1; b < n && ts[at(b)] == t; ++b) {}
      if (b - a == 1) {
        bp.emplace_back(t, kv[at(a)].second);
      } else {
        collided_.push_back(t);
        for (size_t j = a; j < b; ++j) overflow_.push_back({kv[at(j)].first, kv[at(j)].second});
      }
    }
    std::vector<uint64_t>().swap(ts);
    std::vector<size_t>().swap(perm);
    std::sort(overflow_.begin(), overflow_.end(), [](const Entry& x, const Entry& y) { return x.key < y.key; });
    build_filter();
    if (bp.empty()) {  // every key collided: the first t not in collided_ (exists: at most n/2 groups)
      uint64_t s = 0;
      for (size_t g = 0; g < collided_.size() && collided_[g] == s; ++g) ++s;
      bp.emplace_back(s, P());
      std::fill(filter_.begin(), filter_.end(), ~uint64_t(0));  // every lookup to get_filtered, none to the sentinel
    }
    unique_ = unique, collided_keys_ = collided;
    base_.bulk_load(bp.data(), bp.size(), param);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "gret: %s n=%zu unique=%zu collided_keys=%zu groups=%zu aux_bytes=%zu%s build_ms=%.1f\n",
                 name_.c_str(), n, unique, collided, groups, aux_bytes(), transform_note(tr_).c_str(), ms);
  }

  bool get(K key, P& val, Param* param = nullptr) override {
    const uint64_t t = tr_(key);
    const uint64_t h = (t * kHashMul) >> shift_;
    if (__builtin_expect((filter_[h >> 6] >> (h & 63)) & 1, 0)) return get_filtered(key, t, val, param);
    return base_.Base::get(t, val, param);
  }

  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  size_t scan(K, size_t, std::pair<K, P>*, Param* = nullptr) override { return 0; }

  // The base's own figure plus aux_bytes(); 0 when the base reports 0 (GRE's B+tree: "not reported").
  long long memory_consumption() override {
    const long long b = base_.Base::memory_consumption();
    return b == 0 ? 0 : b + (long long)aux_bytes();
  }
  // The transform, the overflow, collided_ and the filter.
  size_t aux_bytes() const {
    return tr_.bytes() + overflow_.size() * sizeof(Entry) + (collided_.size() + filter_.size()) * sizeof(uint64_t);
  }

  const std::string& name() const { return name_; }
  const Tr& transform() const { return tr_; }
  size_t size() const { return n_; }
  size_t unique() const { return unique_; }                 // keys in the base
  size_t collided_keys() const { return collided_keys_; }  // keys in the overflow
  size_t groups() const { return collided_.size(); }

 private:
  struct Entry {
    K key;
    P val;
  };
  static constexpr uint64_t kHashMul = 0x9E3779B97F4A7C15ull;  // Fibonacci hashing: the top bits of t * kHashMul

  // 2^b >= 64 |collided_| bits, b >= 6; filter_ holds 2^b / 64 words.
  void build_filter() {
    unsigned b = 6;
    while ((uint64_t(1) << b) < 64 * uint64_t(collided_.size())) ++b;
    shift_ = 64 - b;
    filter_.assign(size_t(1) << (b - 6), 0);
    for (uint64_t t : collided_) {
      const uint64_t h = (t * kHashMul) >> shift_;
      filter_[h >> 6] |= uint64_t(1) << (h & 63);
    }
  }

  // The filter hit: t is collided (search the overflow by raw key) or a false positive (the base, as usual; none when
  // the base holds only the sentinel).
  __attribute__((noinline)) bool get_filtered(K key, uint64_t t, P& val, Param* param) {
    if (!std::binary_search(collided_.begin(), collided_.end(), t))
      return unique_ != 0 && base_.Base::get(t, val, param);
    auto it = std::lower_bound(overflow_.begin(), overflow_.end(), key, [](const Entry& e, K k) { return e.key < k; });
    if (it == overflow_.end() || it->key != key) return false;
    val = it->val;
    return true;
  }

  std::string name_;
  TransformSpec spec_;
  Tr tr_;
  Base base_;
  unsigned shift_ = 58;                            // 64 - b
  std::vector<uint64_t> filter_ = {0};             // Bloom filter over collided_
  std::vector<uint64_t> collided_;                 // t of every collision group, sorted, unique
  std::vector<Entry> overflow_;                    // members of the collision groups, sorted by raw key
  size_t n_ = 0, unique_ = 0, collided_keys_ = 0;
};

}  // namespace gret
