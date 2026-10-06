// Chunk timer for GRE (SPLICE, MIT): wraps another index and timestamps every 1,000,000th get() call, for the
// within-process warm-up transient check (tools/gre_transient.py; SERVER.md 5b). GRE names trace_splice, trace_lipp.
// GRE's --latency_sample cannot do this: print_stat sorts the samples, so their order in the run is lost, and
// changing benchmark.h would change gre_lite.sh's recorded edits. The wrapper stores to its own counter on every
// get(), so its runs are never headline numbers; the same overhead hits every wrapped index, and the check compares
// chunks within one process.
// Timestamps are taken BEFORE the call, so the boundary at the first timed call (call W after a W-lookup warm-up)
// lands after GRE's TSCNS calibration spin. Chunk k runs from the start of call k*1M to the start of call (k+1)*1M;
// the last chunk ends at memory_consumption(), after GRE's post-run work, which is why the checker drops it.
// Printed from memory_consumption() (gre_run.sh always passes --memory), then forwarded:
//   trace_chunk_size: 1000000 / trace_calls: <get calls> / trace_chunks_n: <N> / N lines 'trace_chunk: <k> <ns>'
#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include "../indexInterface.h"

template<class KEY_TYPE, class PAYLOAD_TYPE, class Inner>
class TraceInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
  typedef std::pair<KEY_TYPE, PAYLOAD_TYPE> record_t;
  static constexpr std::uint64_t kChunk = 1000000;
  static constexpr std::size_t kMax = 4096;  // 4.096e9 calls; GRE's int operations_num stays far below
  Inner *in_;
  std::uint64_t calls_ = 0, next_ = 0;
  std::size_t n_ = 0;
  long long *t_;

  static long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }

public:
  // Preallocated and touched here, so no page fault lands inside the timed loop.
  explicit TraceInterface(Inner *in) : in_(in), t_(new long long[kMax]) {
    for (std::size_t i = 0; i < kMax; ++i) t_[i] = 0;
  }
  ~TraceInterface() { delete in_; delete[] t_; }

  // Every method forwards with a qualified (non-virtual) call, so the inner get() can inline here.
  void init(Param *param = nullptr) { in_->Inner::init(param); }

  void bulk_load(record_t *key_value, size_t num, Param *param = nullptr) {
    in_->Inner::bulk_load(key_value, num, param);
  }

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) {
    if (calls_ == next_) {  // taken once per 1M calls: well predicted
      if (n_ < kMax) t_[n_++] = now_ns();
      next_ += kChunk;
    }
    ++calls_;
    return in_->Inner::get(key, val, param);
  }

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) { return in_->Inner::put(key, value, param); }

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) {
    return in_->Inner::update(key, value, param);
  }

  bool remove(KEY_TYPE key, Param *param = nullptr) { return in_->Inner::remove(key, param); }

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, record_t *result, Param *param = nullptr) {
    return in_->Inner::scan(key_low_bound, key_num, result, param);
  }

  long long memory_consumption() {
    const long long end = now_ns();
    // Past kMax timestamps the last one does not start a 1M-call chunk: drop it rather than print a merged chunk.
    const std::size_t n = (calls_ > kMax * kChunk && n_ == kMax) ? n_ - 1 : n_;
    std::printf("trace_chunk_size: %llu\n", static_cast<unsigned long long>(kChunk));
    std::printf("trace_calls: %llu\n", static_cast<unsigned long long>(calls_));
    std::printf("trace_chunks_n: %zu\n", n);
    for (std::size_t k = 0; k < n; ++k)
      std::printf("trace_chunk: %zu %lld\n", k, (k + 1 < n_ ? t_[k + 1] : end) - t_[k]);
    std::fflush(stdout);
    return in_->Inner::memory_consumption();
  }
};
