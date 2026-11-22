// SPDX-License-Identifier: MIT
// FAST (Kim, Chhugani, Satish, Sedlar, Nguyen, Kaldewey, Lee, Brandt, Dubey: "FAST: fast architecture sensitive tree search on modern
// CPUs and GPUs", SIGMOD 2010) in GRE's harness. Intel never released code. This is a WRAPPER around a THIRD-PARTY IMPLEMENTATION: Viktor
// Leis's FAST (TUM, 2012, 32-bit keys, SSE), as generalized to 64-bit keys and AVX2 in TLI (github.com/curtis-sun/TLI,
// competitors/fast/src/fast.h, GPL-3.0). That file is used unmodified and included by path; nothing of it is copied here. It is not the
// authors' code and not our reimplementation, and the index says so when it is built:
//     gret: fast third-party implementation (Leis, via TLI): keys=... depth=... tree_mapped_bytes=... lookup=folded   (from FastBound::build)
//     gret: fast(third-party) n=... dir_bytes=... bytes_per_key=... build_ms=...                       (from SortedBoundIndex)
// ("fast_asis", "lookup=as_is" and "fast_asis(third-party)" for the other name.)
//
//   FastBound<K, Fold>  a Model for SortedBoundIndex (readonly.h): the tree over the sorted keys; lower_bound(key) is exact, for stored
//                  keys and for keys never stored, so the bound it returns is the empty range at that position and SortedBoundIndex
//                  reads the pair there without any search of its own (as TLI's competitors/fast.h does: tree, then data_[it])
//   make_fast      the index by name: "fast" (Fold = true, the one to report) and "fast_asis" (Fold = false), nothing else. FAST has
//                  no tuning knob: the block sizes (4 keys a 256-bit register, 8 a cache line, 2 MiB a page) are constants of the
//                  file, and TLI's own wrapper ignores its parameters. The two names differ only in HOW TLI's lower_bound IS CALLED:
//       fast       TLI's lower_bound, the same unmodified code, run on a FAST object that lives in the calling function's frame
//                  (fast_detail::folded_lower_bound below) and is given the built tree's three run-time fields (the mapping, the
//                  depth, the key count). TLI declares its layout constants (SIMD_DEPTH, CACHE_LINE_DEPTH, PAGE_DEPTH, ...) as
//                  per-object const fields, so on a tree that lives on the heap the compiler has to reload them and cannot unroll
//                  the three nested loops of lower_bound; on an object whose construction it sees, it folds them. Same answers
//                  (checked key by key in the test and on a sample at every build); measured on Jed (g++ 11, -O3 -march=native,
//                  4 KiB pages, the tree's lower bound alone): 141 -> 77 ns at 2M keys, 575 -> 305 ns at 150M keys. This is OUR
//                  way of calling their code, added by the independent verifier because the as-is call handicapped the baseline
//                  by almost a factor two for a reason that has nothing to do with FAST.
//       fast_asis  TLI's lower_bound called on the built object itself, exactly as TLI's own benchmark calls it. Kept so that the
//                  cost of the above can be shown and so that "FAST as TLI ships it" can be reported beside it.
//
// What TLI's fast.h does, as read from it and pinned by tests/sota/test_fast.cpp (64-bit keys):
//   - the tree is a complete binary search tree of depth d = floor(log2 n) + 1 over 2^d positions, the n keys and then the largest
//     value of the key type as padding (always at least one position of it, n more when n is a power of two). A node holds the last key
//     of its left half and a lookup goes right when key > node, so the leaf reached is the first position whose key is >= the lookup
//     key: the lower bound, n when the key is above every stored key. 2^64 - 1 is an ordinary key (it equals the padding and is found
//     at its position). Repeated keys give the first of them.
//   - layout: 2 levels (3 keys) are compared with one AVX2 instruction, 3 levels (7 keys, 8 slots) are one cache line: one SIMD
//     compare and one scalar compare a line; 15 levels (5 lines deep) are one page block, placed in its own 2 MiB of address space.
//     With depth 16 to 30 there are two page levels: the top block and 2^15 blocks of d - 15 levels, each in 2^(d - 14) slots.
//   - memory: one mmap of 8 * (2^(d + 1) + 2^18 if d > 15) bytes, then madvise(MADV_HUGEPAGE) on Linux. No MAP_HUGETLB, so huge
//     pages only come if the kernel's transparent huge pages are on for the process. Where they are off (the benchmark runs every
//     index under prctl(PR_SET_THP_DISABLE)), the madvise changes nothing and the tree lives on 4 KiB pages: the page blocks then no
//     longer save TLB misses, and the parts of the mapping that are never written stay unbacked. size_in_byte() is the whole
//     mapping; that is what bytes() and memory_consumption() count (TLI's own count), and the build line also gives what is resident.
//   - the result of mmap is not checked, and the mapping is too small from depth 31 (2^30 keys) on: the size formula counts 3 page
//     blocks where the build writes 1 + 2^15 of them. build() refuses 2^30 keys or more, and tries the mapping first so that a
//     failure is an exception and not a write through MAP_FAILED.
//   - the file is not self-contained: it calls _mm256_cmpgt_epu32 and _mm256_cmpgt_epu64, which are not Intel intrinsics; TLI defines
//     them in its util.h. That header also defines a non-inline function, a namespace util, a constant named OVERFLOW (a macro of
//     <math.h> on macOS) and much else, so by default the two compares are supplied HERE (the usual flip of the sign bit, then the
//     signed compare; namespace gret::sota::tli, below). This is the one piece of the lookup path that is ours. Building with
//     -DGRET_FAST_TLI_UTIL takes TLI's util.h instead, inside the same namespace, for checking that both give the same index.
//   - fast.h and (with GRET_FAST_TLI_UTIL) util.h are included INSIDE namespace gret::sota::tli, after every header they include has
//     been included at global scope, so that FAST, SIMD_BYTE, CACHE_LINE_BYTE and PAGE_BYTE do not become global names.
//
// x86-64 with AVX2 only. In any other build this header declares nothing of TLI's and make_fast("fast") throws std::runtime_error with
// a message that says why; every other name is still nullptr. GRET_FAST_AVAILABLE is 1 or 0 accordingly.
//
// Include after indexInterface.h and GRE's pgm header, like readonly.h (which this includes). Include path: the directory that holds
// the TLI clone (#include "TLI/competitors/fast/src/fast.h"), or the clone itself (#include "competitors/fast/src/fast.h").
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "gret/readonly.h"

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__)
#define GRET_FAST_AVAILABLE 1
#else
#define GRET_FAST_AVAILABLE 0
#endif

#if GRET_FAST_AVAILABLE

// Everything TLI's fast.h (and util.h) includes, here at global scope: inside the namespace below their #includes then find the
// include guards set and bring nothing in.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>

#include <emmintrin.h>
#include <immintrin.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif

#pragma push_macro("CACHELINE_SIZE")
#pragma push_macro("forceinline")
#pragma push_macro("OVERFLOW")
#undef OVERFLOW

namespace gret {
namespace sota {
namespace tli {

#ifdef GRET_FAST_TLI_UTIL
#undef CACHELINE_SIZE
#undef forceinline
#if defined(__has_include) && __has_include("TLI/util.h")
#include "TLI/util.h"  // TLI's own _mm256_cmpgt_epu32 / _mm256_cmpgt_epu64 (and the rest of that header, unused)
#else
#include "util.h"
#endif
#else
// a > b, lane by lane, as unsigned numbers. AVX2 only has the signed compare; flipping the top bit of both sides maps the unsigned
// order onto the signed one. Ours (see the head of this file); fast.h calls these two names.
__attribute__((always_inline)) inline __m256i _mm256_cmpgt_epu32(__m256i a, __m256i b) {
  const __m256i top = _mm256_set1_epi32(int(0x80000000u));
  return _mm256_cmpgt_epi32(_mm256_xor_si256(a, top), _mm256_xor_si256(b, top));
}
__attribute__((always_inline)) inline __m256i _mm256_cmpgt_epu64(__m256i a, __m256i b) {
  const __m256i top = _mm256_set1_epi64x((long long)0x8000000000000000ull);
  return _mm256_cmpgt_epi64(_mm256_xor_si256(a, top), _mm256_xor_si256(b, top));
}
#endif

#if defined(__has_include) && __has_include("TLI/competitors/fast/src/fast.h")
#include "TLI/competitors/fast/src/fast.h"  // the third-party FAST: -I <the directory that holds TLI>
#else
#include "competitors/fast/src/fast.h"  // -I <TLI>
#endif

}  // namespace tli
}  // namespace sota
}  // namespace gret

#pragma pop_macro("OVERFLOW")
#pragma pop_macro("forceinline")
#pragma pop_macro("CACHELINE_SIZE")

#endif  // GRET_FAST_AVAILABLE

namespace gret {
namespace sota {

inline constexpr bool kFastAvailable = GRET_FAST_AVAILABLE != 0;
inline constexpr size_t kFastMaxKeys = (size_t(1) << 30) - 1;  // depth 30; from depth 31 on TLI's mapping is smaller than what its build writes
inline constexpr const char* kFastOrigin = "third-party implementation (Leis, via TLI)";

// the depth TLI's buildFAST gives n keys: floor(log2 n) + 1, the bit length of n (n > 0)
inline unsigned fast_depth(size_t n) { return 64u - unsigned(__builtin_clzll((unsigned long long)n)); }

// The bytes TLI's buildFAST maps for n 64-bit keys, 1 <= n < 2^30, derived here from its layout and compared with its own
// size_in_byte() after every build: 2^(d + 1) slots for the last page level, and 2^18 slots (2 MiB) for the top page block when
// there are two page levels (d > 15).
inline size_t fast_mapped_bytes(size_t n) {
  const unsigned d = fast_depth(n);
  return 8 * ((size_t(1) << (d + 1)) + (d > 15 ? size_t(1) << 18 : size_t(0)));
}

inline void fast_check_len(size_t n) {
  if (n == 0) throw std::invalid_argument("FastBound: no keys (TLI's buildFAST takes log2 of the count)");
  if (n > kFastMaxKeys) throw std::length_error("FastBound: 2^30 keys or more: TLI's buildFAST maps less than it writes from depth 31 on");
}

#if GRET_FAST_AVAILABLE

namespace fast_detail {
// FAST<uint64_t>::v (the mapping), depth and len are private and the class gives no way to ask for them. They are needed to tell
// how much of the mapping is resident and on huge pages (the build line), and to hand the built tree to folded_lower_bound below.
// The arguments of an explicit instantiation are exempt from access checks ([temp.spec]), so a pointer to a member can be handed
// to a friend without touching TLI's source (as plex.h does).
template <class Tag, typename Tag::type Member>
struct Open {
  friend typename Tag::type member(Tag) { return Member; }
};
struct TreeSlots {
  using type = uint64_t* tli::FAST<uint64_t>::*;
  friend type member(TreeSlots);
};
struct TreeDepth {
  using type = unsigned tli::FAST<uint64_t>::*;
  friend type member(TreeDepth);
};
struct TreeLen {
  using type = size_t tli::FAST<uint64_t>::*;
  friend type member(TreeLen);
};
template struct Open<TreeSlots, &tli::FAST<uint64_t>::v>;
template struct Open<TreeDepth, &tli::FAST<uint64_t>::depth>;
template struct Open<TreeLen, &tli::FAST<uint64_t>::len>;

// TLI's FAST<uint64_t>::lower_bound, unmodified, on the tree (slots, depth, len) of a built FAST object, but run on a FAST object
// constructed HERE, in this frame, so that the compiler sees the constructor store the layout constants (TLI declares them as
// per-object const fields) and folds them into the lookup: fixed trip counts, the three nested loops unrolled. `flatten` makes
// the constructor and lower_bound part of this function whatever the inliner's size limits say; without that the fold would
// depend on them. The object is placed in a buffer and its destructor is never run (it would unmap the tree, which the built
// object owns); it holds nothing else. Answers are those of the built object's own lower_bound: the code is the same.
__attribute__((flatten)) inline size_t folded_lower_bound(uint64_t* slots, unsigned depth, size_t len, uint64_t x) {
  alignas(tli::FAST<uint64_t>) unsigned char frame[sizeof(tli::FAST<uint64_t>)];
  tli::FAST<uint64_t>* const f = ::new (static_cast<void*>(frame)) tli::FAST<uint64_t>();
  f->*member(TreeSlots{}) = slots;
  f->*member(TreeDepth{}) = depth;
  f->*member(TreeLen{}) = len;
  return f->lower_bound(x);
}

// bytes of [p, p + bytes) that are in memory now (mincore); 0 if the system does not say
inline size_t resident_bytes(const void* p, size_t bytes) {
  const long ps = sysconf(_SC_PAGESIZE);
  if (ps <= 0 || !p || !bytes) return 0;
  const size_t page = size_t(ps), chunk_pages = size_t(1) << 16;
#ifdef __APPLE__
  std::vector<char> vec(chunk_pages);
#else
  std::vector<unsigned char> vec(chunk_pages);
#endif
  size_t in = 0;
  for (size_t off = 0; off < bytes; off += chunk_pages * page) {
    const size_t len = std::min(bytes - off, chunk_pages * page), pages = (len + page - 1) / page;
    if (mincore(const_cast<char*>(static_cast<const char*>(p)) + off, len, vec.data()) != 0) return 0;
    for (size_t i = 0; i < pages; ++i) in += (vec[i] & 1) ? std::min(page, len - i * page) : 0;
  }
  return in;
}

// bytes of the mappings overlapping [p, p + bytes) that the kernel backs with transparent huge pages (AnonHugePages of
// /proc/self/smaps); -1 where that cannot be read (not Linux)
inline long long huge_page_bytes(const void* p, size_t bytes) {
#ifdef __linux__
  std::FILE* f = std::fopen("/proc/self/smaps", "r");
  if (!f) return -1;
  const unsigned long lo = (unsigned long)(uintptr_t)p, hi = lo + bytes;
  char line[512];
  bool inside = false;
  long long kb = 0;
  while (std::fgets(line, sizeof line, f)) {
    unsigned long a = 0, b = 0;
    long long v = 0;
    if (std::sscanf(line, "%lx-%lx ", &a, &b) == 2) inside = a < hi && b > lo;
    else if (inside && std::sscanf(line, "AnonHugePages: %lld kB", &v) == 1) kb += v;
  }
  std::fclose(f);
  return kb * 1024;
#else
  (void)p, (void)bytes;
  return -1;
#endif
}

// 1 if transparent huge pages are disabled for this process (prctl(PR_SET_THP_DISABLE), as the benchmark's wrapper sets), 0 if
// not, -1 where that cannot be asked
inline int thp_disabled() {
#if defined(__linux__) && defined(PR_GET_THP_DISABLE)
  return prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0);
#else
  return -1;
#endif
}
}  // namespace fast_detail

// Fold = true ("fast"): lookups through folded_lower_bound. Fold = false ("fast_asis"): lookups on the built object, as TLI does.
template <class K, bool Fold = true>
struct FastBound {
  static_assert(std::is_same<K, uint64_t>::value, "FastBound: checked for uint64_t keys only (TLI's FAST<uint32_t> has another layout)");

  tli::FAST<K> t;  // by value: the built tree; it owns the mapping (unmapped by its destructor)
  K* slots = nullptr;  // t's mapping and depth, read from it once after the build: what folded_lower_bound is given
  unsigned depth = 0;
  size_t n = 0;
  size_t resident = 0;     // bytes of the mapping in memory right after the build (0: unknown)
  long long huge = -1;     // bytes of it on transparent huge pages right after the build (-1: unknown)
  int thp_off = -1;        // prctl(PR_GET_THP_DISABLE) at build time (-1: unknown)

  FastBound() = default;
  FastBound(const FastBound&) = delete;  // tli::FAST unmaps in its destructor and has no copy of its own
  FastBound& operator=(const FastBound&) = delete;

  // keys: sorted ascending (as SortedBoundIndex passes them; repeats would be allowed). Once.
  void build(std::vector<K>& keys) {
    if (n != 0) throw std::logic_error("FastBound: built twice (TLI's buildFAST would leak the first mapping)");
    fast_check_len(keys.size());
    for (size_t i = 1; i < keys.size(); ++i)
      if (keys[i] < keys[i - 1]) throw std::invalid_argument("FastBound: keys not sorted");
    const size_t want = fast_mapped_bytes(keys.size());
    {  // TLI's malloc_huge does not look at what mmap returns; ask for the same mapping first, so that a refusal is an exception
      void* probe = mmap(nullptr, want, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (probe == MAP_FAILED) throw std::runtime_error("FastBound: cannot map " + std::to_string(want) + " bytes for the tree");
      munmap(probe, want);
    }
    t.buildFAST(keys.data(), keys.size());
    n = keys.size();
    if (size_t(t.size_in_byte()) != want)
      throw std::logic_error("FastBound: TLI's FAST mapped " + std::to_string(t.size_in_byte()) + " bytes where its layout as read here needs " + std::to_string(want));
    slots = t.*member(fast_detail::TreeSlots{});
    depth = t.*member(fast_detail::TreeDepth{});
    if (depth != fast_depth(n) || t.*member(fast_detail::TreeLen{}) != n || slots == nullptr)
      throw std::logic_error("FastBound: TLI's FAST has depth " + std::to_string(depth) + " for " + std::to_string(n) + " keys where its layout as read here has " + std::to_string(fast_depth(n)));
    // Up to 4096 stored keys, spread over the array, through BOTH ways of calling TLI's lower_bound: each must give the key's own
    // position (the first of its repeats). A tree that answers wrongly, or a fold that changed an answer, stops the build here.
    for (size_t j = 0, m = std::min<size_t>(n, 4096); j < m; ++j) {
      size_t i = m < n ? size_t((unsigned __int128)j * n / m) : j;
      while (i > 0 && keys[i - 1] == keys[i]) --i;
      const size_t a = t.lower_bound(keys[i]), b = fast_detail::folded_lower_bound(slots, depth, n, keys[i]);
      if (a != i || b != i)
        throw std::logic_error("FastBound: stored key at " + std::to_string(i) + " found at " + std::to_string(a) + " (TLI's lower_bound as is) and " + std::to_string(b) + " (the same on a local object)");
    }
    const auto t0 = std::chrono::steady_clock::now();
    const void* v = slots;
    resident = fast_detail::resident_bytes(v, want);
    huge = fast_detail::huge_page_bytes(v, want);
    thp_off = fast_detail::thp_disabled();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr,
                 "gret: %s %s: keys=%zu depth=%u tree_mapped_bytes=%zu tree_resident_bytes=%zu tree_huge_page_bytes=%lld thp_disabled_for_process=%d "
                 "(-1: unknown) compares=%s lookup=%s diag_ms=%.1f\n",
                 Fold ? "fast" : "fast_asis", kFastOrigin, n, fast_depth(n), want, resident, huge, thp_off,
#ifdef GRET_FAST_TLI_UTIL
                 "tli_util_h",
#else
                 "gret_shim",
#endif
                 Fold ? "folded" : "as_is", ms);
  }
  // exact: the first position whose key is >= x, n if there is none
  size_t lower_bound(K x) const { return Fold ? fast_detail::folded_lower_bound(slots, depth, n, x) : size_t(t.lower_bound(x)); }
  KeyBound get(K x) const {
    const size_t p = lower_bound(x);
    return {p, p};  // empty range: SortedBoundIndex takes lo as the answer and searches nothing
  }
  // the model's bytes: this object and the whole mapping (TLI's size_in_byte(), touched or not)
  size_t bytes() const { return sizeof(*this) + size_t(t.size_in_byte()); }
};

#endif  // GRET_FAST_AVAILABLE

// "fast" and "fast_asis" -> the index (see the head of this file for the difference); nullptr for every other name. In a build
// without AVX2 both names throw.
template <class K, class P>
indexInterface<K, P>* make_fast(const std::string& name) {
  if (name != "fast" && name != "fast_asis") return nullptr;
#if GRET_FAST_AVAILABLE
  if (name == "fast_asis") return new SortedBoundIndex<K, P, FastBound<K, false>>("fast_asis(third-party)");
  return new SortedBoundIndex<K, P, FastBound<K, true>>("fast(third-party)");
#else
  throw std::runtime_error(name + ": FAST here is a " + kFastOrigin + " that needs x86-64 with AVX2, which this build does not have (-mavx2 or -march=native, on x86-64)");
#endif
}

}  // namespace sota
}  // namespace gret
