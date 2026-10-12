// SPLICE-H in GRE (gre4index/GRE): indexInterface for the read-only index 'splice' and 'splice_thp' (SPLICE, MIT).
// Copied by gre_lite.sh into GRE's src/competitor/splice/ next to splice_gre.hpp; compiled inside GRE's C++17
// microbench TU. The build runs in the C++20 facade (splice_gre.cpp, its own library); after bulk_load this class
// copies the facade's splice::View and get() runs splice::get() on that copy, inlined from splice/layout.hpp: no
// cross-TU call, no try/catch, no allocation and no store except the payload of a hit.
// Not the SCALE-LI cells: scaleli_splice is SCALE-LI's joint-root cell Jg0 (../scaleli/), not this index.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include "../indexInterface.h"
#include "splice_gre.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define SPLICE_GRE_FLATTEN __attribute__((flatten))
#else
#define SPLICE_GRE_FLATTEN
#endif

template<class KEY_TYPE, class PAYLOAD_TYPE>
class SpliceInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
  static_assert(std::is_same<KEY_TYPE, std::uint64_t>::value && std::is_same<PAYLOAD_TYPE, std::uint64_t>::value,
                "SPLICE stores uint64 keys and payloads");
  typedef std::pair<KEY_TYPE, PAYLOAD_TYPE> record_t;
  static_assert(std::is_same<record_t, splice_gre::kv_t>::value, "GRE's records are bulk-loaded in place");
  splice::View v_;           // hot fields first; a copy, so get() needs neither the handle nor the other TU
  splice_gre::Handle *h_;

public:
  // SPLICE_ARGS errors surface here, at get_index, before any key is loaded.
  explicit SpliceInterface(bool thp) : v_(), h_(splice_gre::create(thp)) {}
  // GRE's indexInterface has no virtual destructor, so GRE never runs this (the index lives until exit, like the others).
  ~SpliceInterface() { splice_gre::destroy(h_); }

  // Read-only and single-threaded by contract: one timed thread, so no read is shared with a writer.
  void init(Param *param = nullptr) {
    if (param && param->worker_num != 1) {
      std::fflush(stdout);
      std::fprintf(stderr, "splice_error: SPLICE is single-threaded; run GRE with --thread_num=1 (got %zu)\n",
                   static_cast<size_t>(param->worker_num));
      std::exit(2);
    }
  }

  // Inside GRE's build_ns window: the whole build, the arena pre-touch (every page is written once) and one read of
  // the router and record lines (splice/build.hpp). Nothing is cached for the lookups beyond what the build touched.
  void bulk_load(record_t *key_value, size_t num, Param *param = nullptr) {
    splice_gre::bulk_load(h_, key_value, num);
    v_ = *splice_gre::view(h_);
  }

  // flatten: the whole lookup (get_impl and its helpers, all inline in layout.hpp) is inlined here whatever the
  // compiler's size heuristics say, so the timed path is GRE's virtual call plus straight-line SPLICE code.
  SPLICE_GRE_FLATTEN bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) {
    return splice::get(v_, key, val);
  }

  // Read-only index: writes are refused rather than half supported (GRE counts them as unsuccessful).
  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) { return false; }

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) { return false; }

  bool remove(KEY_TYPE key, Param *param = nullptr) { return false; }

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, record_t *result, Param *param = nullptr) { return 0; }

  // Self-reported bytes the index owns (arena, records, router, exceptions); cross-index comparisons use gre_lite's
  // index_rss_bytes, and gre_report.py checks the two against each other.
  long long memory_consumption() {
    long long b = splice_gre::total_bytes(h_);
    splice_gre::print_line("splice_total_bytes_after_run", b);
    return b;
  }
};
