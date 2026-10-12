// Chunk timer for GRE (SPLICE, MIT): wraps another index and timestamps chunk boundaries of its get() calls, for the
// within-process warm-up analysis. GRE names trace_<index> (gre_lite.sh registry).
// GRE's --latency_sample cannot do this: print_stat sorts the samples, so their order in the run is lost, and
// changing benchmark.h would change gre_lite.sh's recorded edits. The wrapper stores to its own counter on every
// get(), so its runs are never headline numbers; the same overhead hits every wrapped index, and the analyses compare
// chunks within one process.
// Schedules (environment variable GRE_TRACE):
//   unset         every 1,000,000th call (tools/gre_transient.py, SERVER.md 5b). Output, unchanged:
//                 trace_chunk_size: 1000000 / trace_calls / trace_chunks_n: <N> / N lines 'trace_chunk: <k> <ns>'
//   geo:F:C       chunks of F, 2F, 4F, ... calls up to C, then C each (tools/warmup_detect.py): cache and TLB warm-up
//                 under uniform lookups ends within about 1e4-1e5 calls, so it needs log-time resolution early and
//                 fixed resolution later. Output: trace_schedule / trace_calls / trace_chunks_n: <N> / N lines
//                 'trace_point: <k> <call> <ns since point 0> <faults> <cpu>' / 'trace_end: <calls> <ns> <faults>'.
//   fixed:C       chunks of C calls, same output as geo.
// GRE_TRACE_FAULTS=1 adds the process's minor+major page faults at every boundary (getrusage: one syscall per
// boundary, so for diagnosis runs only; otherwise the column is -1). GRE_TRACE_CPU=1 adds the CPU the thread is on
// at every boundary (pthread_cpu_number_np on macOS, which reads a register; sched_getcpu on Linux, vDSO): it shows
// core migrations on a Mac (performance vs efficiency cores) and proves the pinning on a server. Otherwise -1.
// Timestamps are taken BEFORE the call, so the boundary at the first timed call (call W after a W-lookup warm-up)
// lands after GRE's TSCNS calibration spin. Chunk k runs from the start of its first call to the start of the next
// chunk's; the last one ends at memory_consumption(), after GRE's post-run work, which is why the analyses drop it.
#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <sys/resource.h>
#if defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sched.h>
#endif
#include "../indexInterface.h"

template<class KEY_TYPE, class PAYLOAD_TYPE, class Inner>
class TraceInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
  typedef std::pair<KEY_TYPE, PAYLOAD_TYPE> record_t;
  static constexpr std::uint64_t kDefaultChunk = 1000000;
  static constexpr std::size_t kMax = std::size_t(1) << 16;  // 65,536 boundaries: 4.3e9 calls at 64K per chunk
  Inner *in_;
  std::uint64_t calls_ = 0, next_ = 0, step_ = kDefaultChunk, cap_ = kDefaultChunk;
  std::size_t n_ = 0;
  bool legacy_ = true, faults_ = false, cpu_ = false;
  char schedule_[64] = "default";
  long long *t_;
  std::uint64_t *c_;
  long long *f_;
  int *u_;

  static long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  static int cpu_now() {
#if defined(__APPLE__)
    size_t c = 0;
    return pthread_cpu_number_np(&c) == 0 ? static_cast<int>(c) : -1;
#elif defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
  }
  static long long faults() {
    struct rusage u;
    return getrusage(RUSAGE_SELF, &u) == 0 ? static_cast<long long>(u.ru_minflt + u.ru_majflt) : -1;
  }
  [[noreturn]] static void bad(const char *s) {
    std::fprintf(stderr, "trace_error: GRE_TRACE='%s': expected geo:FIRST:CAP or fixed:SIZE, 1 <= FIRST <= CAP\n", s);
    std::exit(2);
  }
  void parse(const char *s) {
    char *e = nullptr;
    if (std::strncmp(s, "geo:", 4) == 0) {
      step_ = std::strtoull(s + 4, &e, 10);
      if (!e || *e != ':') bad(s);
      cap_ = std::strtoull(e + 1, &e, 10);
    } else if (std::strncmp(s, "fixed:", 6) == 0) {
      step_ = cap_ = std::strtoull(s + 6, &e, 10);
    } else bad(s);
    if (!e || *e || step_ < 1 || cap_ < step_) bad(s);
    legacy_ = false;
    std::snprintf(schedule_, sizeof schedule_, "%s", s);
  }

public:
  // Preallocated and touched here, so no page fault lands inside the timed loop.
  explicit TraceInterface(Inner *in) : in_(in), t_(new long long[kMax]), c_(new std::uint64_t[kMax]),
                                       f_(new long long[kMax]), u_(new int[kMax]) {
    for (std::size_t i = 0; i < kMax; ++i) { t_[i] = 0; c_[i] = 0; f_[i] = -1; u_[i] = -1; }
    if (const char *s = std::getenv("GRE_TRACE")) if (*s) parse(s);
    if (const char *s = std::getenv("GRE_TRACE_FAULTS")) faults_ = !legacy_ && s[0] == '1';
    if (const char *s = std::getenv("GRE_TRACE_CPU")) cpu_ = !legacy_ && s[0] == '1';
  }
  ~TraceInterface() { delete in_; delete[] t_; delete[] c_; delete[] f_; delete[] u_; }

  // Every method forwards with a qualified (non-virtual) call, so the inner get() can inline here.
  void init(Param *param = nullptr) { in_->Inner::init(param); }

  void bulk_load(record_t *key_value, size_t num, Param *param = nullptr) {
    in_->Inner::bulk_load(key_value, num, param);
  }

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) {
    if (calls_ == next_) {  // taken once per chunk: well predicted
      if (n_ < kMax) {
        if (faults_) f_[n_] = faults();
        if (cpu_) u_[n_] = cpu_now();
        c_[n_] = calls_;
        t_[n_++] = now_ns();
      }
      next_ = calls_ + step_;
      if (step_ < cap_) step_ = step_ * 2 < cap_ ? step_ * 2 : cap_;
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
    const long long end_faults = faults_ ? faults() : -1;
    if (legacy_) {
      // Past kMax timestamps the last one does not start a 1M-call chunk: drop it rather than print a merged chunk.
      const std::size_t n = (calls_ > kMax * kDefaultChunk && n_ == kMax) ? n_ - 1 : n_;
      std::printf("trace_chunk_size: %llu\n", static_cast<unsigned long long>(kDefaultChunk));
      std::printf("trace_calls: %llu\n", static_cast<unsigned long long>(calls_));
      std::printf("trace_chunks_n: %zu\n", n);
      for (std::size_t k = 0; k < n; ++k)
        std::printf("trace_chunk: %zu %lld\n", k, (k + 1 < n_ ? t_[k + 1] : end) - t_[k]);
    } else {
      std::printf("trace_schedule: %s\n", schedule_);
      std::printf("trace_calls: %llu\n", static_cast<unsigned long long>(calls_));
      std::printf("trace_chunks_n: %zu\n", n_);
      for (std::size_t k = 0; k < n_; ++k)
        std::printf("trace_point: %zu %llu %lld %lld %d\n", k, static_cast<unsigned long long>(c_[k]), t_[k] - t_[0],
                    f_[k], u_[k]);
      if (n_) std::printf("trace_end: %llu %lld %lld\n", static_cast<unsigned long long>(calls_), end - t_[0],
                          end_faults);
    }
    std::fflush(stdout);
    return in_->Inner::memory_consumption();
  }
};
