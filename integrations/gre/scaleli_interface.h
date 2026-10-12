// SCALE-LI in GRE (gre4index/GRE): indexInterface for the cells scaleli_B, _N, _C, _Cr, _NC, _J, _Jg0 (= _splice) (SPLICE, MIT).
// Copied by gre_lite.sh into GRE's src/competitor/scaleli/ next to scaleli_gre.hpp (INTEGRATION.md); compiled inside
// GRE's C++17 microbench TU, it only forwards to the C++20 facade (scaleli_gre.cpp, a separate TU/library).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include "../indexInterface.h"
#include "scaleli_gre.hpp"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class ScaleliInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
  static_assert(std::is_same<KEY_TYPE, std::uint64_t>::value && std::is_same<PAYLOAD_TYPE, std::uint64_t>::value,
                "SCALE-LI stores uint64 keys and payloads");
  typedef std::pair<KEY_TYPE, PAYLOAD_TYPE> record_t;
  scaleli_gre::Handle *h_;

public:
  // Config and flow-file errors surface here, at get_index, before any key is loaded.
  explicit ScaleliInterface(const char *cell) : h_(scaleli_gre::create(cell)) {}
  // GRE's indexInterface has no virtual destructor, so GRE never runs this (the index lives until exit, like the others).
  ~ScaleliInterface() { scaleli_gre::destroy(h_); }

  // Single writer: find() updates per-region write heat, and splits refit the root. GRE passes worker_num = thread_num.
  void init(Param *param = nullptr) {
    if (param && param->worker_num != 1) {
      std::fflush(stdout);
      std::fprintf(stderr, "scaleli_error: SCALE-LI is single-threaded; run GRE with --thread_num=1 (got %zu)\n",
                   static_cast<size_t>(param->worker_num));
      std::exit(2);
    }
  }

  void bulk_load(record_t *key_value, size_t num, Param *param = nullptr) {
    scaleli_gre::bulk_load(h_, key_value, num);
  }

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) { return scaleli_gre::get(h_, key, &val); }

  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) { return scaleli_gre::put(h_, key, value); }

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) { return scaleli_gre::update(h_, key, value); }

  bool remove(KEY_TYPE key, Param *param = nullptr) { return scaleli_gre::remove(h_, key); }

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, record_t *result, Param *param = nullptr) {
    return scaleli_gre::scan(h_, key_low_bound, key_num, result);
  }

  // Accounted live bytes of the index (scaleli_bench's accounted_bytes); excludes allocator overhead and the flow
  // weights. Cross-index comparisons use gre_lite's index_rss_bytes.
  long long memory_consumption() {
    long long b = scaleli_gre::accounted_bytes(h_);
    scaleli_gre::print_line(h_, "scaleli_accounted_bytes_after_run", b);
    return b;
  }
};
