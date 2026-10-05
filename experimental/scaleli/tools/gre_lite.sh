#!/usr/bin/env bash
# Build a trimmed GRE (gre4index/GRE, Wongkham et al., VLDB 2022) next to SPLICE, for machines without AVX2 or MKL.
#
# The full suite needs AVX2/BMI2 (HOT) and Intel MKL (XIndex, FINEdex) and cannot compile on the DIAS Atom (no AVX).
# This build keeps the indexes the NFL / CSV comparison needs: ALEX, LIPP, dynamic PGM, STX B+tree and ART (unsync),
# plus 'sortedarray', our own binary-search baseline.
# SCALE-LI (SPLICE): scaleli_b scaleli_n scaleli_c scaleli_cr scaleli_nc scaleli_j scaleli_jg0 scaleli_splice, the protocol cells of
# results/aidb_ba/run_ba.py plus the joint G+T+V root, from integrations/gre/ (README.md 'GRE cells'). They
# read SCALELI_FLOW (per-dataset flow file, cells _n _nc _j _jg0) and SCALELI_BUILD_THREADS (default 16).
# GRE ships no licence file, so it is cloned OUTSIDE this repository and only edited locally; nothing of it is committed here.
# Every edit starts from the pristine file (git checkout) and is recorded in DEST/build/gre_lite_build.txt:
#  - ALEX: on CPUs without LZCNT/BMI1 (the Atom is Goldmont) ALEX_USE_LZCNT 0, ALEX's own documented switch, which only
#    changes the gap search during inserts.
#  - competitor.h, CMakeLists.txt: trimmed to the indexes above (originals kept as *.full).
#  - sortedarray (src/competitor/sortedarray/sortedarray.h, written here): a std::vector of (key, payload) pairs reserved
#    to exactly n, searched with std::lower_bound. A binary-search throughput reference and a check of the memory
#    method: its index_rss_bytes must be about 16 B x n. Before building, its bulk_load allocates, touches and frees
#    another 16 B x n in small (slab-allocated) blocks, which stay resident unless the allocator purge works: about
#    32 B x n if it does not. The array alone is one large block that jemalloc and glibc return eagerly, so it could
#    not tell. Its build_ns includes this probe.
#  - benchmark.h, through src/benchmark/gre_lite_patch.h (written here). With the new flags left at their defaults the
#    operation list and the timed loop are those of unpatched GRE; only extra lines are printed:
#      rss_before_build_bytes / rss_after_build_bytes / index_rss_bytes: resident set (VmRSS) around get_index + init +
#        bulk_load, each read after purging jemalloc's dirty pages, so index_rss_bytes is the index's steady-state
#        footprint after build temporaries are freed. GRE's 'Memory:' is self-reported (0 for the B+tree, ART without
#        its 16-B records); this measures every index the same way.
#      rss_after_build_unpurged_bytes: VmRSS right after bulk_load, before that purge (build temporaries included).
#      build_ns: steady_clock around init + bulk_load.
#      rss_after_run_bytes: resident set after the timed run (and a purge); rss_peak_bytes: VmHWM of the process.
#      --warmup_num=W (default 0): W untimed lookups of bulk-loaded keys, single-threaded, right before the timed region.
#        Keys come from GRE's own sampler for --sample_distribution with seed + 1; the timed operation list was already
#        generated from --seed and is unchanged. GRE has no warm-up, so its first timed lookups pay cold caches and TLB.
#        Prints warmup_success_read (must equal W) and warmup_ns. For read-only workloads only: ALEX's get() updates
#        per-node lookup counters that its cost model reads when an insert expands or splits a node, so with inserts
#        a warm-up changes ALEX's structure and its numbers are not comparable with W=0.
#      --pin_core=K (default -1, not pinned as published): sched_setaffinity of the calling thread before the warm-up
#        and the timed region (Linux only; prints pinned_core). Meant for --thread_num=1; other OpenMP threads are not
#        pinned.
#    Every run also prints 'gre_lite_patch: warmup_num=W pin_core=K'. Unpatched GRE silently ignores unknown flags, so
#    a log without that line came from an unpatched binary.
#    The purges also hold at the defaults: the timed region starts with jemalloc's dirty pages released, where unpatched
#    GRE starts with them cached. A read-only run allocates nothing in the timed loop, so it is unaffected; with inserts
#    the patched and unpatched builds are comparable only approximately.
#
# Usage, inside the activated conda environment (see SERVER.md):
#   experimental/scaleli/tools/gre_lite.sh [DEST]          # default DEST=/tmp/louisvasseur/GRE
#   GRE_LITE_EDIT_ONLY=1 experimental/scaleli/tools/gre_lite.sh DEST
#       # clone and edit only: no conda, no dependency install, no build (to inspect or test the edits anywhere)
# It refuses to build while a GRE or SPLICE benchmark runs (GRE_LITE_FORCE=1 overrides).
# Then, for example (single thread, read-only, half the keys bulk-loaded as in GRE/NFL):
#   OMP_NUM_THREADS=1 DEST/build/microbench --keys_file=<SOSD file> --keys_file_type=binary --read=1 --insert=0 \
#       --operations_num=10000000 --table_size=-1 --init_table_ratio=0.5 --thread_num=1 --memory --index=lipp \
#       --output_path=out.csv [--warmup_num=10000000 --pin_core=2]
# Lookups sample bulk-loaded keys, so a correct read-only run prints success_read equal to operations_num.
# --table_size=N reads only the first N keys of the file (a prefix), for smoke tests.
# OMP_NUM_THREADS fixes the operation list: GRE's sampler seeds one generator per OpenMP thread (seed + thread id).
# With more than one OpenMP thread the warm-up stream (seed + 1) is the timed stream of thread 1, so keep it at 1.
# Run one index per process: GRE's indexInterface has no virtual destructor, so '--index=a,b' never frees index a.
set -euo pipefail
DEST=${1:-/tmp/louisvasseur/GRE}
EDIT_ONLY=${GRE_LITE_EDIT_ONLY:-0}
GRE_SHA=e807edcef51df6732f07f94d4c797fb3897519ba          # GRE master as of 2022-11-09

if [ "$EDIT_ONLY" = 1 ]; then
    echo "== 1/4 dependencies: skipped (GRE_LITE_EDIT_ONLY=1)"
else
    : "${CONDA_PREFIX:?activate the conda environment first (SERVER.md)}"
    # A build beside a timed run (an -O3 compile on 16 cores) would disturb its timings.
    BUSY='[-]-keys_file=|scaleli_bench|scaleli_hardness'
    if [ "${GRE_LITE_FORCE:-0}" != 1 ] && command -v pgrep >/dev/null 2>&1 && pgrep -f "$BUSY" >/dev/null; then
        echo "a benchmark is running (pgrep -fl '$BUSY'); build when it is done, or set GRE_LITE_FORCE=1" >&2; exit 1
    fi
    MAMBA=${MAMBA_EXE:-/tmp/louisvasseur/bin/micromamba}
    echo "== 1/4 dependencies: TBB 2020 (GRE's FindTBB reads tbb_stddef.h, gone in oneTBB 2021+) and jemalloc"
    # jemalloc pinned: the purge in gre_lite_patch.h relies on its mallctl names, and results should keep one allocator
    "$MAMBA" install -y -p "$CONDA_PREFIX" -c conda-forge "tbb-devel=2020.2" "jemalloc=5.4.0"
fi

echo "== 2/4 GRE at $GRE_SHA with only the needed submodules"
[ -d "$DEST/.git" ] || git clone https://github.com/gre4index/GRE.git "$DEST"
SCALELI_DIR=$(cd "$(dirname "$0")/.." && pwd)   # experimental/scaleli of this SPLICE checkout (SCALE-LI cells)
cd "$DEST"
git checkout -q "$GRE_SHA"
git submodule update --init src/competitor/alex/src src/competitor/lipp/src src/competitor/pgm/src \
    src/competitor/btree/src src/competitor/artsync/src

echo "== 3/4 local edits: ALEX's LZCNT switch, trimmed index registry and build file (originals kept as *.full),"
echo "        sortedarray, benchmark.h probes (RSS, build_ns, --warmup_num, --pin_core)"
# ALEX's insert-time gap search (closest_gap) uses the LZCNT/TZCNT instructions. ALEX's own setting for CPUs without them
# is ALEX_USE_LZCNT 0 (alex_nodes.h: "If your hardware does not support lzcnt/tzcnt ... set this to 0"), a bit-by-bit
# gap scan; lookups never call it. Set it from what -march=native enables, starting from the pristine file on every run.
ALEX_NODES=src/competitor/alex/src/src/core/alex_nodes.h
git -C src/competitor/alex/src checkout -q -- src/core/alex_nodes.h
MACROS=$("${CXX:-c++}" -march=native -dM -E -x c++ /dev/null)
if grep -qw __LZCNT__ <<<"$MACROS" && grep -qw __BMI__ <<<"$MACROS"; then ALEX_LZCNT=1
else ALEX_LZCNT=0; sed 's/^#define ALEX_USE_LZCNT 1$/#define ALEX_USE_LZCNT 0/' "$ALEX_NODES" > "$ALEX_NODES.tmp"
     mv "$ALEX_NODES.tmp" "$ALEX_NODES"; fi
grep -q "^#define ALEX_USE_LZCNT $ALEX_LZCNT\$" "$ALEX_NODES" || { echo "ALEX_USE_LZCNT line not found in $ALEX_NODES" >&2; exit 1; }
echo "   ALEX_USE_LZCNT=$ALEX_LZCNT (CPU $(grep -qw __LZCNT__ <<<"$MACROS" && echo has || echo lacks) LZCNT)"
[ -f src/competitor/competitor.h.full ] || cp src/competitor/competitor.h src/competitor/competitor.h.full
[ -f CMakeLists.txt.full ] || cp CMakeLists.txt CMakeLists.txt.full
cat > src/competitor/competitor.h <<'EOF'
// Trimmed by SPLICE gre_lite.sh: ALEX, LIPP, PGM, STX B+tree, ART only (no AVX2 / MKL needed), plus SPLICE's sortedarray
// and SCALE-LI (scaleli_*).
// Original: competitor.h.full
#include "./indexInterface.h"
#include "./alex/alex.h"
#include "./artsync/artunsync.h"
#include "./lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "./sortedarray/sortedarray.h"
#include "./scaleli/scaleli_interface.h"
#include "iostream"

template<class KEY_TYPE, class PAYLOAD_TYPE>
indexInterface<KEY_TYPE, PAYLOAD_TYPE> *get_index(std::string index_type) {
  indexInterface<KEY_TYPE, PAYLOAD_TYPE> *index;
  if (index_type == "alex") index = new alexInterface<KEY_TYPE, PAYLOAD_TYPE>;
  else if (index_type == "lipp") index = new LIPPInterface<KEY_TYPE, PAYLOAD_TYPE>;
  else if (index_type == "pgm") index = new pgmInterface<KEY_TYPE, PAYLOAD_TYPE>;
  else if (index_type == "btree") index = new BTreeInterface<KEY_TYPE, PAYLOAD_TYPE>;
  else if (index_type == "artunsync") index = new ARTUnsynchronizedInterface<KEY_TYPE, PAYLOAD_TYPE>;
  else if (index_type == "sortedarray") index = new SortedArrayInterface<KEY_TYPE, PAYLOAD_TYPE>;
  else if (index_type == "scaleli_b") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("B");
  else if (index_type == "scaleli_n") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("N");
  else if (index_type == "scaleli_c") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("C");
  else if (index_type == "scaleli_nc") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("NC");
  else if (index_type == "scaleli_j") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("J");
  else if (index_type == "scaleli_jg0") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("Jg0");
  else if (index_type == "scaleli_splice") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("Jg0");
  else if (index_type == "scaleli_cr") index = new ScaleliInterface<KEY_TYPE, PAYLOAD_TYPE>("Cr");
  else { std::cout << "Could not find a matching index called " << index_type << " (gre_lite: alex lipp pgm btree artunsync sortedarray scaleli_b scaleli_n scaleli_c scaleli_cr scaleli_nc scaleli_j scaleli_jg0 scaleli_splice).\n"; exit(0); }
  return index;
}
EOF
cat > CMakeLists.txt <<'EOF'
# Trimmed by SPLICE gre_lite.sh: no MKL, HOT, Masstree or Wormhole. Original: CMakeLists.txt.full
cmake_minimum_required(VERSION 3.14)
project(GRE_lite)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(OpenMP REQUIRED)
find_package(JeMalloc REQUIRED)
find_package(TBB REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${TBB_INCLUDE_DIRS} ${JEMALLOC_INCLUDE_DIR})
# -include cstdint: 2020-era headers rely on transitive <cstdint>, which GCC 13+ no longer provides
add_compile_options(-faligned-new -march=native -g -O3 -include cstdint)
add_executable(microbench ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark/microbench.cpp)
target_link_libraries(microbench PUBLIC OpenMP::OpenMP_CXX ${JEMALLOC_LIBRARIES} ${TBB_LIBRARIES})
# SPLICE SCALE-LI: C++20 in its own TU (GRE's TU is C++17 and ALEX needs that), linked into microbench.
# -ffp-contract=off: with -march=native on an FMA host, GCC would fuse multiply-adds the portable scaleli_bench does not.
set(SCALELI_INCLUDE "" CACHE PATH "SPLICE experimental/scaleli/include")
if(NOT EXISTS "${SCALELI_INCLUDE}/scaleli/index.hpp")
  message(FATAL_ERROR "SCALELI_INCLUDE must point to SPLICE experimental/scaleli/include")
endif()
find_package(Threads REQUIRED)
add_library(scaleli_gre STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/competitor/scaleli/scaleli_gre.cpp)
set_target_properties(scaleli_gre PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
target_include_directories(scaleli_gre PRIVATE ${SCALELI_INCLUDE})
target_compile_options(scaleli_gre PRIVATE -ffp-contract=off)
target_link_libraries(scaleli_gre PUBLIC Threads::Threads)
target_link_libraries(microbench PUBLIC scaleli_gre)
EOF

# Our own code (MIT, SPLICE), written whole on every run.
mkdir -p src/competitor/sortedarray
# SCALE-LI wrapper and facade (SPLICE, MIT), copied whole on every run; the core headers are read in place.
mkdir -p src/competitor/scaleli
cp "$SCALELI_DIR"/integrations/gre/scaleli_interface.h "$SCALELI_DIR"/integrations/gre/scaleli_gre.hpp \
   "$SCALELI_DIR"/integrations/gre/scaleli_gre.cpp src/competitor/scaleli/
cat > src/competitor/sortedarray/sortedarray.h <<'EOF'
// Written by SPLICE gre_lite.sh (MIT). A sorted array of (key, payload) pairs searched with std::lower_bound:
// a binary-search throughput reference, and a check of the RSS memory method (exactly n records, 16 B each for uint64).
#pragma once
#include <algorithm>
#include <cstdlib>
#include <utility>
#include <vector>
#include "../indexInterface.h"

template<class KEY_TYPE, class PAYLOAD_TYPE>
class SortedArrayInterface : public indexInterface<KEY_TYPE, PAYLOAD_TYPE> {
  typedef std::pair<KEY_TYPE, PAYLOAD_TYPE> record_t;
  typedef typename std::vector<record_t>::iterator iter_t;
  std::vector<record_t> data_;

  iter_t lower(KEY_TYPE key) {
    return std::lower_bound(data_.begin(), data_.end(), key,
                            [](const record_t &r, const KEY_TYPE &k) { return r.first < k; });
  }
  bool found(iter_t it, KEY_TYPE key) const { return it != data_.end() && it->first == key; }

public:
  void init(Param *param = nullptr) {}

  // Purge probe: 16 B/key in blocks of a small size class, touched and freed. Freed small blocks stay in the
  // allocator's slabs, so they leave the RSS taken after bulk_load only if the purge before it works: else index RSS
  // reads ~32 B/key. 1792 B is a jemalloc size class with 28-KiB slabs, the largest, so the per-slab metadata that
  // jemalloc keeps after a purge stays small; 64-B blocks (4-KiB slabs) would need 7 times as many slabs.
  static void purge_probe(size_t num) {
    const size_t block = 1792;
    std::vector<char *> blocks(num * 16 / block);
    for (size_t i = 0; i < blocks.size(); i++) {
      blocks[i] = static_cast<char *>(std::malloc(block));
      if (!blocks[i]) continue;
      volatile char *p = blocks[i];  // volatile: the compiler can drop neither the stores nor the malloc/free pair
      for (size_t j = 0; j < block; j += 256) p[j] = 1;  // every page the block spans
    }
    for (size_t i = 0; i < blocks.size(); i++) std::free(blocks[i]);
  }

  // GRE bulk-loads sorted, deduplicated keys. reserve() then assign() allocates exactly num records once.
  void bulk_load(record_t *key_value, size_t num, Param *param = nullptr) {
    std::vector<record_t>().swap(data_);
    purge_probe(num);
    data_.reserve(num);
    data_.assign(key_value, key_value + num);
  }

  bool get(KEY_TYPE key, PAYLOAD_TYPE &val, Param *param = nullptr) {
    iter_t it = lower(key);
    if (!found(it, key)) return false;
    val = it->second;
    return true;
  }

  // O(n) shifts; inserts are not what this baseline is for.
  bool put(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) {
    iter_t it = lower(key);
    if (found(it, key)) return false;
    data_.insert(it, record_t(key, value));
    return true;
  }

  bool update(KEY_TYPE key, PAYLOAD_TYPE value, Param *param = nullptr) {
    iter_t it = lower(key);
    if (!found(it, key)) return false;
    it->second = value;
    return true;
  }

  bool remove(KEY_TYPE key, Param *param = nullptr) {
    iter_t it = lower(key);
    if (!found(it, key)) return false;
    data_.erase(it);
    return true;
  }

  size_t scan(KEY_TYPE key_low_bound, size_t key_num, record_t *result, Param *param = nullptr) {
    iter_t it = lower(key_low_bound);
    size_t n = std::min(key_num, static_cast<size_t>(data_.end() - it));
    std::copy(it, it + n, result);
    return n;
  }

  long long memory_consumption() { return data_.capacity() * sizeof(record_t); }
};
EOF
cat > src/benchmark/gre_lite_patch.h <<'EOF'
// Written by SPLICE gre_lite.sh (MIT): resident-set probe, allocator purge, core pinning and an untimed lookup loop,
// called from the lines gre_lite.sh adds to GRE's benchmark.h.
#pragma once
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __linux__
#include <sched.h>
#elif defined(__APPLE__)
#include <mach/mach.h>   // local tests only
#include <malloc/malloc.h>
#include <sys/resource.h>
#endif
#if __has_include(<jemalloc/jemalloc.h>)
#include <jemalloc/jemalloc.h>
#endif

inline long long gre_lite_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

// jemalloc keeps freed pages resident until they decay (10 s by default); purge them so RSS counts live memory only.
// conda-forge builds jemalloc without a symbol prefix, so it is the process's malloc and 'mallctl' is its own name.
inline void gre_lite_purge() {
#ifdef MALLCTL_ARENAS_ALL
  char name[64];
  snprintf(name, sizeof(name), "arena.%u.purge", static_cast<unsigned>(MALLCTL_ARENAS_ALL));
  mallctl("thread.tcache.flush", nullptr, nullptr, nullptr, 0);
  mallctl(name, nullptr, nullptr, nullptr, 0);
#endif
#ifdef __APPLE__
  malloc_zone_pressure_relief(nullptr, 0);  // local tests: the system allocator, as jemalloc is not linked there
#endif
}

#ifdef __linux__
// A "<field> <n> kB" line of /proc/self/status in bytes, -1 if absent.
inline long long gre_lite_status_bytes(const char *field) {
  FILE *f = fopen("/proc/self/status", "r");
  if (!f) return -1;
  char line[256];
  long long kb = -1;
  size_t len = strlen(field);
  while (fgets(line, sizeof(line), f))
    if (strncmp(line, field, len) == 0) { kb = strtoll(line + len, nullptr, 10); break; }
  fclose(f);
  return kb < 0 ? -1 : kb * 1024;
}
#endif

// Peak resident set of the process in bytes (VmHWM), -1 if unknown.
inline long long gre_lite_peak_bytes() {
#ifdef __linux__
  return gre_lite_status_bytes("VmHWM:");
#elif defined(__APPLE__)
  struct rusage ru;
  return getrusage(RUSAGE_SELF, &ru) == 0 ? static_cast<long long>(ru.ru_maxrss) : -1;  // bytes on macOS
#else
  return -1;
#endif
}

// Resident set in bytes, -1 if unknown. Linux: VmRSS (kB granularity).
inline long long gre_lite_rss_bytes() {
#ifdef __linux__
  return gre_lite_status_bytes("VmRSS:");
#elif defined(__APPLE__)
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
    return -1;
  return static_cast<long long>(info.resident_size);
#else
  return -1;
#endif
}

// Pin the calling thread to one core.
inline bool gre_lite_pin(int core) {
#ifdef __linux__
  if (core < 0 || core >= CPU_SETSIZE) return false;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  return sched_setaffinity(0, sizeof(set), &set) == 0;
#else
  (void) core;
  return false;
#endif
}

inline volatile uint64_t gre_lite_sink;  // C++17 inline variable: one definition however often included

// Untimed lookups; the payload sum goes to a volatile so -O3 cannot drop them. Returns the number found.
template<class INDEX, class KEY, class VAL, class PARAM>
uint64_t gre_lite_lookups(INDEX *index, const KEY *keys, long long n, PARAM *param, long long &ns) {
  VAL val = 0;
  uint64_t ok = 0, sum = 0;
  long long t0 = gre_lite_now_ns();
  for (long long i = 0; i < n; i++) {
    ok += index->get(keys[i], val, param);
    sum += static_cast<uint64_t>(val);
  }
  ns = gre_lite_now_ns() - t0;
  gre_lite_sink = sum;
  return ok;
}
EOF

# benchmark.h: lines added next to unique anchor lines of the pristine file; any mismatch stops the script.
BENCH=src/benchmark/benchmark.h
git checkout -q -- "$BENCH"
add_lines() {  # add_lines before|after ANCHOR < lines; ANCHOR must be a whole line occurring exactly once in $BENCH
    local n
    n=$(grep -cxF -- "$2" "$BENCH" || true)
    [ "$n" = 1 ] || { echo "anchor found $n times in $BENCH: $2" >&2; exit 1; }
    # ENVIRON, not -v: awk -v would expand the backslash in an anchor such as printf("...\n").
    WHERE=$1 ANCHOR=$2 awk 'NR == FNR { add = add $0 "\n"; next }
        $0 == ENVIRON["ANCHOR"] && ENVIRON["WHERE"] == "before" { printf "%s", add }
        { print }
        $0 == ENVIRON["ANCHOR"] && ENVIRON["WHERE"] == "after" { printf "%s", add }' - "$BENCH" > "$BENCH.tmp"
    mv "$BENCH.tmp" "$BENCH"
}
add_lines after '#include <jemalloc/jemalloc.h>' <<'EOF'
#include "gre_lite_patch.h"  // SPLICE gre_lite.sh
EOF
add_lines after '    bool data_shift = false;' <<'EOF'
    long long warmup_num = 0;  // gre_lite: untimed lookups before the timed region
    int pin_core = -1;         // gre_lite: -1 = not pinned, as published
EOF
add_lines after '        data_shift = get_boolean_flag(flags, "data_shift");' <<'EOF'
        // gre_lite: read without get_with_default so GRE's own flag echo is unchanged
        if (flags.count("warmup_num")) warmup_num = stoll(flags["warmup_num"]);
        if (flags.count("pin_core")) pin_core = stoi(flags["pin_core"]);
        INVARIANT(warmup_num >= 0 && warmup_num <= INT_MAX);  // GRE's samplers take an int count
        printf("gre_lite_patch: warmup_num=%lld pin_core=%d\n", warmup_num, pin_core);
EOF
add_lines before '        index = get_index<KEY_TYPE, PAYLOAD_TYPE>(index_type);' <<'EOF'
        gre_lite_purge();
        long long gre_lite_rss0 = gre_lite_rss_bytes();
        printf("rss_before_build_bytes: %lld\n", gre_lite_rss0);
EOF
add_lines after '        index = get_index<KEY_TYPE, PAYLOAD_TYPE>(index_type);' <<'EOF'
        long long gre_lite_t0 = gre_lite_now_ns();
EOF
add_lines after '        index->bulk_load(init_key_values, init_keys.size(), &param);' <<'EOF'
        long long gre_lite_build_ns = gre_lite_now_ns() - gre_lite_t0;
        printf("rss_after_build_unpurged_bytes: %lld\n", gre_lite_rss_bytes());
        gre_lite_purge();
        long long gre_lite_rss1 = gre_lite_rss_bytes();
        printf("rss_after_build_bytes: %lld\n", gre_lite_rss1);
        printf("index_rss_bytes: %lld\n", gre_lite_rss1 - gre_lite_rss0);
        printf("build_ns: %lld\n", gre_lite_build_ns);
EOF
add_lines after '    void run(index_t *index) {' <<'EOF'
        // gre_lite: pinning and warm-up, before any of GRE's timing code. The operation list was generated earlier from
        // --seed; the warm-up keys come from the same sampler with seed + 1 and are freed before the timed region.
        KEY_TYPE *gre_lite_keys = nullptr;
        size_t gre_lite_seed = random_seed + 1;
        if (warmup_num > 0)
            gre_lite_keys = sample_distribution == "zipf"
                ? get_search_keys_zipf(&init_keys[0], init_table_size, warmup_num, &gre_lite_seed)
                : get_search_keys(&init_keys[0], init_table_size, warmup_num, &gre_lite_seed);
        if (pin_core >= 0) {
            if (gre_lite_pin(pin_core)) printf("pinned_core: %d\n", pin_core);
            else fprintf(stderr, "gre_lite: could not pin to core %d, running unpinned\n", pin_core);
        }
        if (warmup_num > 0) {
            Param gre_lite_param = Param(thread_num, 0);
            long long gre_lite_ns = 0;
            uint64_t gre_lite_ok = gre_lite_lookups<index_t, KEY_TYPE, PAYLOAD_TYPE>(
                index, gre_lite_keys, warmup_num, &gre_lite_param, gre_lite_ns);
            delete[] gre_lite_keys;
            printf("warmup_success_read: %llu\n", static_cast<unsigned long long>(gre_lite_ok));
            printf("warmup_ns: %lld\n", gre_lite_ns);
        }
EOF
add_lines after '        printf("Finish running\n");' <<'EOF'
        gre_lite_purge();
        printf("rss_after_run_bytes: %lld\n", gre_lite_rss_bytes());
        printf("rss_peak_bytes: %lld\n", gre_lite_peak_bytes());
EOF
grep -q '^#include "gre_lite_patch.h"' "$BENCH" || { echo "benchmark.h edit missing" >&2; exit 1; }

mkdir -p build
if [ "$EDIT_ONLY" = 1 ]; then
    echo "== 4/4 build: skipped (GRE_LITE_EDIT_ONLY=1)"
else
    echo "== 4/4 build"
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
          -DTBB_ROOT_DIR="$CONDA_PREFIX" -DJEMALLOC_ROOT_DIR="$CONDA_PREFIX" \
          -DSCALELI_INCLUDE="$SCALELI_DIR/include"
    cmake --build build -j
fi
{ echo "GRE $GRE_SHA, trimmed by SPLICE gre_lite.sh: competitor.h and CMakeLists.txt replaced (originals *.full)"
  git submodule status src/competitor/{alex,lipp,pgm,btree,artsync}/src
  echo "ALEX_USE_LZCNT=$ALEX_LZCNT (1 = as shipped)"
  echo "edits (each from the pristine file):"
  echo "  $BENCH: gre_lite_patch.h probes, --warmup_num, --pin_core; diff cksum $(git diff -- "$BENCH" | cksum)"
  echo "  src/benchmark/gre_lite_patch.h: SPLICE, cksum $(cksum < src/benchmark/gre_lite_patch.h)"
  echo "  src/competitor/sortedarray/sortedarray.h: SPLICE, cksum $(cksum < src/competitor/sortedarray/sortedarray.h)"
  for f in src/competitor/scaleli/*; do echo "  $f: SPLICE, cksum $(cksum < "$f")"; done
  echo "  SCALE-LI headers $SCALELI_DIR/include/scaleli: cksum $(cat "$SCALELI_DIR"/include/scaleli/*.hpp | cksum)"
  echo "  SPLICE $(git -C "$SCALELI_DIR" rev-parse HEAD 2>/dev/null || echo unknown) ($(git -C "$SCALELI_DIR" status --porcelain -- . 2>/dev/null | wc -l | tr -d ' ') uncommitted paths in experimental/scaleli)"
  echo "  src/competitor/competitor.h: trimmed + sortedarray, cksum $(cksum < src/competitor/competitor.h)"
  echo "  CMakeLists.txt: trimmed, cksum $(cksum < CMakeLists.txt)"
  "${CXX:-c++}" --version | sed -n 1p
  "${CXX:-c++}" -march=native -Q --help=target 2>/dev/null | grep -E '^ +-march=' | tr -s ' ' || true
  if [ "$EDIT_ONLY" = 1 ]; then echo "edit-only: not built"
  elif command -v ldd >/dev/null 2>&1; then
      # which allocator, TBB and C++ runtime the binary resolves to at run time
      ldd build/microbench | grep -E 'jemalloc|tbb|libstdc\+\+' | sed 's/^[[:space:]]*/ldd: /' || true
  fi
} > build/gre_lite_build.txt
cat build/gre_lite_build.txt
if [ "$EDIT_ONLY" = 1 ]; then echo "edited: $DEST (not built)"
else echo "built: $DEST/build/microbench   (indexes: alex lipp pgm btree artunsync sortedarray scaleli_b scaleli_n scaleli_c scaleli_cr scaleli_nc scaleli_j scaleli_jg0 scaleli_splice)"; fi
