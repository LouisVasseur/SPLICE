// SPDX-License-Identifier: MIT
// alexu and btreeu: ALEX and the STX B+tree as GRE wraps them, with update implemented. GRE's own wrappers return false from
// update without doing anything (alex/alex.h has the call commented out, btree/btree.h has none), so an update workload cannot
// be run on "alex" or "btree" at all. Here update is the obvious one, a lookup and a store through each library's own accessor:
//   alexu   alex::Alex::get_payload(key) gives the payload's address in the data node; the new value is written there
//   btreeu  stx::btree::find(key), then the iterator's data() reference
// Every other operation is GRE's wrapper, line for line (including its B+tree scan, which counts rather than copies, and its
// memory_consumption of 0 for the B+tree), so a row for alexu or btreeu differs from alex or btree only where update is called.
// This is our addition to the baselines, not GRE's code: name it so next to any number.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace gret {
namespace sota {

template <class K, class P>
class AlexUpdating final : public indexInterface<K, P> {
 public:
  void init(Param* = nullptr) override {}
  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override { index_.bulk_load(kv, int(n)); }
  bool get(K key, P& val, Param* = nullptr) override {
    P* res = index_.get_payload(key);
    if (res != nullptr) {
      val = *res;
      return true;
    }
    return false;
  }
  bool put(K key, P val, Param* = nullptr) override { return index_.insert(key, val).second; }
  bool update(K key, P val, Param* = nullptr) override {
    P* res = index_.get_payload(key);
    if (res == nullptr) return false;
    *res = val;
    return true;
  }
  bool remove(K key, Param* = nullptr) override { return index_.erase(key) > 0; }
  size_t scan(K lo, size_t n, std::pair<K, P>* out, Param* = nullptr) override {
    auto it = index_.lower_bound(lo);
    size_t got = 0;
    for (; got < n && !it.is_end(); ++got) {
      out[got] = {(*it).first, (*it).second};
      it++;
    }
    return got;
  }
  long long memory_consumption() override { return index_.model_size() + index_.data_size(); }

 private:
  alex::Alex<K, P, alex::AlexCompare, std::allocator<std::pair<K, P>>, false> index_;
};

template <class K, class P>
class BTreeUpdating final : public indexInterface<K, P> {
 public:
  void init(Param* = nullptr) override {}
  void bulk_load(std::pair<K, P>* kv, size_t n, Param* = nullptr) override { idx_.bulk_load(kv, kv + n); }
  bool get(K key, P& val, Param* = nullptr) override {
    auto it = idx_.find(key);
    if (it == idx_.end()) return false;
    val = it.data();
    return true;
  }
  bool put(K key, P val, Param* = nullptr) override { return idx_.insert(key, val).second; }
  bool update(K key, P val, Param* = nullptr) override {
    auto it = idx_.find(key);
    if (it == idx_.end()) return false;
    it.data() = val;
    return true;
  }
  bool remove(K key, Param* = nullptr) override { return idx_.erase(key) > 0; }
  size_t scan(K lo, size_t n, std::pair<K, P>*, Param* = nullptr) override {  // as GRE's wrapper: it walks and counts
    auto it = idx_.lower_bound(lo);
    if (it == idx_.end()) return 0;
    uint64_t sum = 1;
    P acc = 0;
    for (size_t i = 0; i < n; i++) {
      ++it;
      sum++;
      if (it == idx_.end()) break;
      acc += it->second;
    }
    (void)acc;
    return sum;
  }
  long long memory_consumption() override { return 0; }

 private:
  stx::btree<K, P> idx_;
};

template <class K, class P>
indexInterface<K, P>* make_updating(const std::string& name) {
  if (name == "alexu") return new AlexUpdating<K, P>();
  if (name == "btreeu") return new BTreeUpdating<K, P>();
  return nullptr;
}

}  // namespace sota
}  // namespace gret
