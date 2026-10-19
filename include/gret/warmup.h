// SPDX-License-Identifier: MIT
// Warm-up, without touching GRE's driver or timed loop:
//   trace_<name>  times the inner index's lookups in chunks of TRACE_CHUNK calls (default 100000) and prints one line
//                 per chunk to stderr: "gret_trace <chunk> <calls_so_far> <ns_per_lookup>". One clock read per chunk.
//                 For warm-up detection runs only (scripts/warmup_detect.py), never for headline numbers.
//   w_<name>      at the end of bulk_load (which GRE does not time) runs TRANSFORM_WARM untimed lookups (default
//                 10000000) of loaded keys drawn uniformly with a fixed seed, as GRE's uniform read-only workload draws
//                 them (a separate stream: the same distribution, not the same keys). Read-only workloads only: some
//                 bases update statistics in get() (ALEX), which matters once inserts follow.
// Both wrap any registered name, raw bases included (trace_lipp, w_lipp, trace_w_g_pw512_alex).
#pragma once
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>

namespace gret {

template <class K, class P, class Inner = indexInterface<K, P>>
class TraceIndex final : public indexInterface<K, P> {
  static constexpr bool kByValue = !std::is_same<Inner, indexInterface<K, P>>::value;

 public:
  TraceIndex(std::string name = "trace", indexInterface<K, P>* inner = nullptr) : name_(std::move(name)), owned_(inner) {
    const char* e = std::getenv("TRACE_CHUNK");
    chunk_ = e && *e ? std::strtoull(e, nullptr, 10) : 100000;
    if (!chunk_) chunk_ = 100000;
  }
  void init(Param* p = nullptr) override { in().init(p); }
  void bulk_load(std::pair<K, P>* kv, size_t n, Param* p = nullptr) override { in().bulk_load(kv, n, p); }
  __attribute__((flatten)) bool get(K key, P& val, Param* p = nullptr) override {
    if (calls_ == 0) t0_ = std::chrono::steady_clock::now();
    const bool r = in().get(key, val, p);
    if (++calls_ % chunk_ == 0) {
      const auto t = std::chrono::steady_clock::now();
      const double ns = std::chrono::duration<double, std::nano>(t - t0_).count() / double(chunk_);
      std::fprintf(stderr, "gret_trace %llu %llu %.3f\n", (unsigned long long)(calls_ / chunk_ - 1),
                   (unsigned long long)calls_, ns);
      t0_ = t;
    }
    return r;
  }
  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  size_t scan(K, size_t, std::pair<K, P>*, Param* = nullptr) override { return 0; }
  long long memory_consumption() override { return in().memory_consumption(); }

 private:
  Inner& in() {
    if constexpr (kByValue) return held_;
    else return *owned_;
  }
  struct Empty {};
  std::string name_;
  std::unique_ptr<indexInterface<K, P>> owned_;
  typename std::conditional<kByValue, Inner, Empty>::type held_;
  unsigned long long chunk_ = 100000, calls_ = 0;
  std::chrono::steady_clock::time_point t0_;
};

template <class K, class P, class Inner = indexInterface<K, P>>
class WarmIndex final : public indexInterface<K, P> {
  static constexpr bool kByValue = !std::is_same<Inner, indexInterface<K, P>>::value;

 public:
  WarmIndex(std::string name = "w", indexInterface<K, P>* inner = nullptr) : name_(std::move(name)), owned_(inner) {}
  void init(Param* p = nullptr) override { in().init(p); }
  void bulk_load(std::pair<K, P>* kv, size_t n, Param* p = nullptr) override {
    in().bulk_load(kv, n, p);
    const char* e = std::getenv("TRANSFORM_WARM");
    const unsigned long long W = e && *e ? std::strtoull(e, nullptr, 10) : 10000000ULL;
    if (!n || !W) return;
    std::mt19937_64 gen(1867);  // GRE's default --seed is 1866; a separate stream
    std::uniform_int_distribution<size_t> pick(0, n - 1);
    const auto t0 = std::chrono::steady_clock::now();
    unsigned long long found = 0;
    for (unsigned long long i = 0; i < W; ++i) {
      P v;
      found += in().get(kv[pick(gen)].first, v, p);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "gret: %s warmup=%llu found=%llu warm_s=%.3f\n", name_.c_str(), W, found, s);
  }
  __attribute__((flatten)) bool get(K key, P& val, Param* p = nullptr) override { return in().get(key, val, p); }
  bool put(K, P, Param* = nullptr) override { return false; }
  bool update(K, P, Param* = nullptr) override { return false; }
  bool remove(K, Param* = nullptr) override { return false; }
  size_t scan(K, size_t, std::pair<K, P>*, Param* = nullptr) override { return 0; }
  long long memory_consumption() override { return in().memory_consumption(); }

 private:
  Inner& in() {
    if constexpr (kByValue) return held_;
    else return *owned_;
  }
  struct Empty {};
  std::string name_;
  std::unique_ptr<indexInterface<K, P>> owned_;
  typename std::conditional<kByValue, Inner, Empty>::type held_;
};

}  // namespace gret
