// SPDX-License-Identifier: MIT
// plex<maxerror>: PLEX (Stoian, Kipf, Marcus, Kraska: "Towards Practical Learned Indexing", AIDB@VLDB 2021) as a read-only index in
// GRE's harness. This is the AUTHORS' code, unmodified (github.com/stoianmihail/PLEX, include/ts: ts::Builder and ts::TrieSpline),
// included by path and wrapped as a Model for SortedBoundIndex (readonly.h): one sorted array of (key, payload) pairs, PLEX bounds
// where a key is, a branch-free binary search inside the bound. Nothing of PLEX is reimplemented here.
//
// PLEX is a greedy error-bounded spline over the keys (its one parameter: the maximum error, in positions) and, on top of the
// spline points, either a radix table or a compact hist-tree; which one and at what size is chosen by its own self-tuning step
// (ts::Builder::ComputeStatistics and InferTuning) from the spline alone, under the space limit the authors fixed (no larger than
// the spline). That step is used as shipped: the wrapper passes the maximum error and nothing else.
//
// Names: plex1 plex2 plex4 plex8 plex16 plex32 plex64 plex128 plex256 plex512 plex1024 (powers of two; 32 is the value in the
// authors' example.cc), and the other values of the authors' own sweep (tune.sh): plex25 50 60 70 80 90 100 125 150 175 200 225 250
// 275 300 350 400 450 500. make_plex<K, P>(name) returns nullptr for every other name.
//
// What the wrapper adds around the authors' code, and why (docs/sota/plex.md has the details):
//   1. PLEX at this commit cannot be built when (largest key - smallest key) is exactly a power of two: its tuner and its
//      hist-tree builder take ceil(log2(range)) bits, one too few for the largest key (ts/builder.h:305,314 index out of bounds).
//      The model is then built on the keys below the largest one (repeatedly, while the range is a power of two), and a key above
//      what it was built on goes straight to the tail of the array, as PgmBound does for PGM's sentinel.
//   2. PLEX's hist-tree builder goes wrong when the tuner picks a tree whose log2(bins) does not divide the bits of the key range
//      and a bin of the last level still holds more spline points than the tree's error: the next node gets a negative bin width
//      (ts_cht/builder.h:111,117: a shift by 2^32 - something, undefined; in practice a build that never ends, or a tree that
//      works by accident). The tuner's choice cannot be seen from outside ts::Builder::Finalize(), so build() runs the statements
//      of Finalize() itself, each one the authors' own function, with one check between the tuning and the tree build, and throws
//      std::runtime_error rather than run into that. $PLEX_GUARD=0 calls the authors' Finalize() instead, with no check.
//   3. The model is copied once after it is built, so that the bytes it holds are the bytes GetSize() reports (the builder's
//      spline vector grows by doubling and GetSize() counts its size, not its capacity).
//   4. $PLEX_VERIFY=1: bulk_load checks that the bound of every stored key holds its position, and throws if one does not.
//      Off by default (one lookup per key at build time, and PLEX's build time should be PLEX's).
//
// Needs before it: GRE's indexInterface.h and pgm/pgm.h (readonly.h names pgm::PGMIndex). Include path: the directory that holds
// the PLEX clone (#include "PLEX/include/ts/builder.h"), or PLEX/include itself (#include "ts/builder.h"). One translation unit
// (as registry.h): the explicit instantiations below are definitions.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <functional>  // PLEX's ts_cht/builder.h names std::function, std::optional and std::queue without including all of them
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__has_include) && __has_include("PLEX/include/ts/builder.h")
#include "PLEX/include/ts/builder.h"
#else
#include "ts/builder.h"
#endif

#include "gret/readonly.h"

namespace gret {

// The private members of ts::Builder<uint64_t> that Finalize() uses, reached without touching PLEX's source: the arguments of an
// explicit instantiation are exempt from access checks ([temp.spec]), so a pointer to a private member can be handed to a friend.
namespace plex_detail {
using Builder = ts::Builder<uint64_t>;
template <class Tag, typename Tag::type Member>
struct Open {
  friend typename Tag::type member(Tag) { return Member; }
};
struct SplinePoints {
  using type = std::vector<ts::Coord<uint64_t>> Builder::*;
  friend type member(SplinePoints);
};
struct ChtBuilder {
  using type = ts_cht::Builder<uint64_t> Builder::*;
  friend type member(ChtBuilder);
};
struct AddKeyToSpline {
  using type = void (Builder::*)(uint64_t, double);
  friend type member(AddKeyToSpline);
};
struct ComputeStatistics {
  using type = void (Builder::*)(std::vector<ts::Statistics>&);
  friend type member(ComputeStatistics);
};
struct InferTuning {
  using type = ts::Statistics (Builder::*)(std::vector<ts::Statistics>&);
  friend type member(InferTuning);
};
template struct Open<SplinePoints, &Builder::spline_points_>;
template struct Open<ChtBuilder, &Builder::chtb_>;
template struct Open<AddKeyToSpline, &Builder::AddKeyToSpline>;
template struct Open<ComputeStatistics, &Builder::ComputeStatistics>;
template struct Open<InferTuning, &Builder::InferTuning>;

inline unsigned ceil_log2(uint64_t d) { return 63u - unsigned(__builtin_clzll(d)) + ((d & (d - 1)) != 0); }  // d > 0

// What PLEX's tuner chose, and whether ts_cht::Builder can build it (point 2 above). Empty: it can.
inline std::string unbuildable(const ts::Statistics& t, const std::vector<ts::Coord<uint64_t>>& spline, uint64_t lo, uint64_t hi) {
  const unsigned lg = ceil_log2(hi - lo), bins = ceil_log2(t.numBins);
  const std::string what = (t.treeMaxError == std::numeric_limits<unsigned>::max() ? "a radix table of 2^" + std::to_string(bins) + " cells"
                                                                                    : "a hist-tree of 2^" + std::to_string(bins) + " bins a node and error " + std::to_string(t.treeMaxError)) +
                           " over " + std::to_string(spline.size()) + " spline points and a key range of " + std::to_string(lg) + " bits";
  if (bins == 0 || bins > lg) return what + ": more bits than the range has (ts_cht/builder.h:49-50)";
  const unsigned r = lg % bins;  // the width, in bits, of a bin of the last level a node can have
  if (t.treeMaxError == std::numeric_limits<unsigned>::max() || r == 0) return "";  // one node; or last bins one key wide
  size_t run = 0, worst = 0;
  uint64_t bin = 0;
  for (size_t i = 0; i < spline.size(); ++i) {
    const uint64_t b = (spline[i].x - lo) >> r;
    run = (i && b == bin) ? run + 1 : 1;
    bin = b;
    if (run > worst) worst = run;
  }
  if (worst <= t.treeMaxError) return "";
  return what + ": a bin 2^" + std::to_string(r) + " keys wide holds " + std::to_string(worst) +
         " spline points, and splitting it gives a node of negative bin width (ts_cht/builder.h:111,117)";
}
}  // namespace plex_detail

template <class K, size_t MaxError>
struct PlexBound {
  static_assert(std::is_same<K, uint64_t>::value, "PlexBound: uint64_t keys (PLEX also has uint32_t; GRE's keys are uint64_t)");
  static_assert(MaxError >= 1, "PlexBound: the maximum error is at least 1");

  ts::TrieSpline<K> m;
  K top = 0;                // the largest key the model was built on
  size_t built = 0, n = 0;  // keys the model was built on (0: none, every key is searched in the whole array), keys in the array

  // keys: sorted, unique, at least 2 (SortedBoundIndex calls with 64 or more)
  void build(std::vector<K>& keys) {
    n = keys.size();
    size_t k = n;
    while (k > 1 && pow2(keys[k - 1] - keys[0])) --k;  // point 1: a power-of-two range leaves its largest key to the tail
    if (k < 2) {  // only {a, a + 2^i, a + 2^j, ...}: at most 65 keys, no model
      built = 0, top = keys[0];
      return;
    }
    built = k, top = keys[k - 1];
    {
      ts::Builder<K> b(keys[0], top, MaxError);
      for (size_t i = 0; i < k; ++i) b.AddKey(keys[i]);
      const char* g = std::getenv("PLEX_GUARD");
      const ts::TrieSpline<K> grown = (g && *g == '0') ? b.Finalize() : finalize(b, keys[0], top, k);
      m = grown;  // point 3: a copy, whose vectors are as large as GetSize() says
    }
    if (const char* v = std::getenv("PLEX_VERIFY"); v && *v && *v != '0') {
      for (size_t i = 0; i < n; ++i) {
        const KeyBound r = get(keys[i]);
        if (!(r.lo <= i && i < r.hi && r.hi <= n))
          throw std::runtime_error("PlexBound: plex" + std::to_string(MaxError) + " bounds key " + std::to_string(keys[i]) + " at position " +
                                   std::to_string(i) + " by [" + std::to_string(r.lo) + ", " + std::to_string(r.hi) + ")");
      }
    }
  }
  // x: between the smallest and the largest key of the array (SortedBoundIndex answers the rest itself)
  KeyBound get(K x) const {
    if (__builtin_expect((x > top) | (built == 0), 0)) return {built, n};
    const ts::SearchBound r = m.GetSearchBound(x);
    return {r.begin, r.end};
  }
  size_t bytes() const { return built ? m.GetSize() : 0; }

 private:
  static bool pow2(K d) { return d != 0 && (d & (d - 1)) == 0; }

  // ts::Builder::Finalize() (ts/builder.h:41-60), statement by statement, with the check of point 2 before the hist-tree is built.
  // b has been given `count` keys, the first `lo`, the last `hi`, at positions 0 .. count - 1.
  static ts::TrieSpline<K> finalize(ts::Builder<K>& b, K lo, K hi, size_t count) {
    using namespace plex_detail;
    std::vector<ts::Coord<K>>& spline = b.*member(SplinePoints{});
    if (spline.back().x != hi) (b.*member(AddKeyToSpline{}))(hi, double(count - 1));  // the last key ends the spline
    std::vector<ts::Statistics> statistics;
    (b.*member(ComputeStatistics{}))(statistics);
    const ts::Statistics tuning = (b.*member(InferTuning{}))(statistics);
    if (const std::string why = unbuildable(tuning, spline, lo, hi); !why.empty())
      throw std::runtime_error("PlexBound: plex" + std::to_string(MaxError) + ": PLEX's tuner chose " + why +
                               ". PLEX's builder has undefined behaviour there; not built. Another maximum error may build.");
    ts_cht::CompactHistTree<K> cht = (b.*member(ChtBuilder{})).Finalize(tuning.numBins, tuning.treeMaxError);
    return ts::TrieSpline<K>(lo, hi, count, MaxError, std::move(cht), std::move(spline));
  }
};

namespace plex_detail {
template <class K, class P, size_t... E>
indexInterface<K, P>* pick(const std::string& name, unsigned long long e) {
  indexInterface<K, P>* r = nullptr;
  (void)(((e == E) && (r = new SortedBoundIndex<K, P, PlexBound<K, E>>(name), true)) || ...);
  return r;
}
}  // namespace plex_detail

// "plex<maxerror>" with one of the values above, written without a sign or leading zeros; nullptr otherwise
template <class K, class P>
indexInterface<K, P>* make_plex(const std::string& name) {
  if constexpr (!std::is_same<K, uint64_t>::value) {
    return nullptr;
  } else {
    if (name.size() < 5 || name.size() > 8 || name.compare(0, 4, "plex") != 0 || name[4] == '0') return nullptr;
    unsigned long long e = 0;
    for (size_t i = 4; i < name.size(); ++i) {
      if (name[i] < '0' || name[i] > '9') return nullptr;
      e = e * 10 + unsigned(name[i] - '0');
    }
    return plex_detail::pick<K, P, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,  // powers of two
                             25, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500>(name, e);
  }
}

}  // namespace gret
