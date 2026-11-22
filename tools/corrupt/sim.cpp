// SPDX-License-Identifier: MIT
// sim: what writes do to ab4's layout, counted exactly, with no timing. It builds the real AdaptiveLineIndex on a base key set,
// applies a stream of inserts or deletes IN PLACE (the slots of the real index are edited, so the real get and scan are run on
// the result as a check), and at each level reports, in the currency the build optimizes (cache lines a lookup reads, at a byte
// budget):
//   lines_hit      lines a lookup of a stored key reads now (keys that found no slot: the slot walk plus OV_COST for the overflow)
//   lines_replace  the same keys placed afresh with the SAME directory and lines (V redone, T kept): what re-placing alone recovers
//   lines_fresh    a fresh build on the same keys at the same total bytes (and at the base budget per key): T and V redone
//   regret_place = hit / replace, regret_model = replace / fresh, regret = hit / fresh
// and how each insert was absorbed: Z no key moved (a free slot in reach), R / L keys shifted right / left by one slot within
// SHIFT_MAX, and three ways to fail, each a different lever's:
//   ovT   the key's predicted bucket lies past its place in sorted order (a prediction that is not monotone there: T)
//   ovV   no free slot within reach: its window of d + 1 lines is full, or a shift would push a key out of its window (V)
//   G     outside the span the directory covers: an exception (G)
// It also models the key distribution after the stream: the Kolmogorov distance between the base CDF and the new one, the total
// variation over 1024 equal-count bins of the base, the largest density ratio of a bin, and the binned CDF itself (OUT.cdf.tsv),
// and it counts how often the V and T failures fall in the same leaves (the 2 x 2 table over leaves that received inserts).
//
//   sim KEYS N STREAM [LEVELS]     KEYS: a GRE key file, or syn:uniform | syn:lognormal | syn:clusters
//                                  STREAM: holdout | hot | append | cluster | delete | delhot | churn
//                                  LEVELS: fractions of the base keys, default 0.005,0.01,0.02,0.05,0.1,0.2
//   environment: the index's own (AB_BUDGET default 20 here, AR_LEAF, AR_TAU, AB_V), SHIFT_MAX (slots, default 8), OV_COST (lines an overflow lookup adds, default 1.5),
//                SEED (1), OUT (prefix for OUT.tsv and OUT.cdf.tsv; default: stdout only), FRESH=0 skips the fresh builds
// In-place adaptations to try (each off by default):
//   D_MAX=<lines>   V: a move may leave a key up to this many lines past its bucket; the index's one walk bound d grows to fit
//   SHIFT_MAX       V: how many slots one insert may move
//   AB_SLACK_Z=<z>  V at build time: spare slots by leaf, k + z sqrt(k) (adaptiveline.h)
//   COST=1          V against G, by the loss: an insert shifts only if that adds fewer lines to the keys it moves than an overflow
//                   lookup costs (OV_COST), else it overflows (counted as ovC); W_COST prices each moved slot, in lines
//   REFIT=1|2       T and V of a leaf and its neighbours redone in place, taking the overflow keys back (see refit): 1 as soon
//                   as a key overflows, 2 once those keys' lookups have cost what the rewrite will (RW reads per write, W_LINE)
//   G_INDEX=1       G: the keys with no slot, and the exceptions, each in an index of their own; its measured lines replace OV_COST
// Inserts are not part of the index: this is the instrument that says which in-place adaptation is worth building.
#include "indexInterface.h"
#include "gret/adaptiveline.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gret {
template <class K, class P, unsigned B, class Coord>
struct AdaptiveLineProbe {
  using I = AdaptiveLineIndex<K, P, B, Coord>;
  I& x;
  auto& slots() { return x.slots_; }
  auto& exc() { return x.exc_; }
  uint64_t lo() const { return x.lo_; }
  uint64_t span() const { return x.span_; }
  size_t d() const { return x.d_; }
  size_t nodes() const { return x.nodes_.size(); }
  double tau() const { return x.tau_; }
  size_t leaf_keys() const { return x.leaf_; }
  size_t lower_slot(uint64_t k) const { return x.lower_slot(k); }
  size_t bucket(uint64_t v, size_t* leaf = nullptr) const {  // the index's own walk, also saying which leaf it ends in
    const auto* n = x.nodes_.data();
    size_t at = size_t(v >> x.shift_);
    auto e = n[at];
    while (e.a >> 31) {
      at = size_t(e.a & ~I::kInner) + size_t((v << (e.b & 255)) >> (e.b >> 8));
      e = n[at];
    }
    if (leaf) *leaf = at;
    return (size_t(e.a) + I::slot_in(v, e.b >> 24, e.b & I::kSlots)) / B;
  }
  void layout_changed() { x.removed_ = true; }  // a key may now have copies after the one get reads: update must walk them
  void set_last(uint64_t k) { x.last_key_ = k; }
  void set_d(size_t d) { x.d_ = d; }
  // a leaf's line: its first slot a, the shift that brings its range's bits to the top, its slots c
  uint32_t leaf_a(size_t at) const { return x.nodes_[at].a; }
  unsigned leaf_top(size_t at) const { return x.nodes_[at].b >> 24; }
  uint32_t leaf_c(size_t at) const { return x.nodes_[at].b & I::kSlots; }
  void set_leaf(size_t at, uint32_t a, uint32_t c) { x.nodes_[at].a = a, x.nodes_[at].b = (x.nodes_[at].b & ~I::kSlots) | c; }
  static size_t slot_on(uint32_t a, unsigned to_top, uint32_t c, uint64_t v) { return size_t(a) + I::slot_in(v, to_top, c); }
  static constexpr uint32_t kMaxSlots = I::kSlots, kMaxA = I::kInner - 1;
  void add_live(size_t n) { x.live_ += n; }
};
}  // namespace gret

namespace {
constexpr unsigned B = 4;
using Index = gret::AdaptiveLineIndex<uint64_t, uint64_t, B>;
using Probe = gret::AdaptiveLineProbe<uint64_t, uint64_t, B, gret::IdentityCoord>;
using KV = std::pair<uint64_t, uint64_t>;

uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }
double env_d(const char* k, double def) {
  const char* e = std::getenv(k);
  return e && *e ? std::strtod(e, nullptr) : def;
}

std::vector<uint64_t> load_keys(const std::string& path, size_t n, std::mt19937_64& g) {
  std::vector<uint64_t> v;
  if (path.rfind("syn:", 0) == 0) {
    const std::string kind = path.substr(4);
    v.reserve(n);
    if (kind == "uniform") {
      for (size_t i = 0; i < n; ++i) v.push_back(g() >> 4);
    } else if (kind == "lognormal") {
      std::lognormal_distribution<double> ln(0, 2);
      for (size_t i = 0; i < n; ++i) v.push_back(uint64_t(ln(g) * 1e12));
    } else if (kind == "clusters") {
      while (v.size() < n) {
        const uint64_t base = g() >> 2, step = 1 + g() % 64;
        for (size_t i = 0, m = 200 + g() % 5000; i < m && v.size() < n; ++i) v.push_back(base + i * step);
      }
    } else {
      std::fprintf(stderr, "unknown synthetic set %s\n", kind.c_str()), std::exit(2);
    }
  } else {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) std::perror(path.c_str()), std::exit(2);
    uint64_t have = 0;
    if (std::fread(&have, 8, 1, f) != 1) std::exit(2);
    v.resize(std::min<uint64_t>(have, n));
    if (std::fread(v.data(), 8, v.size(), f) != v.size()) std::exit(2);
    std::fclose(f);
  }
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

enum Ev { kZ, kR, kL, kOvT, kOvVwin, kOvVshift, kOvC, kG, kEvs };
const char* const kEvName[kEvs] = {"Z", "R", "L", "ovT", "ovVwin", "ovVshift", "ovC", "G"};

struct LeafStat {
  uint32_t ins = 0, v_fail = 0, t_fail = 0;  // cumulative: inserts routed here, V failures (shift or no slot), T failures
  uint32_t in_slots = 0;                     // its keys in the slots now (a line may change only if the window holds them all)
  uint32_t ovm = 0, next_try = 1;            // keys of this leaf now in the overflow; no refit is tried below this many
  double ov_t = 0;                           // the sum of the times they arrived, for the lines their lookups have cost so far
  uint32_t keys = 0;                         // per measurement
  uint64_t lines = 0;
};

struct Sim {
  std::unique_ptr<Index> idx;
  Probe pr;
  std::vector<uint64_t> ov;  // keys that found no slot (the overflow a real put would keep outside the slots); sync_ov fills it
  std::unordered_map<uint32_t, std::vector<uint64_t>> ovl;  // the same keys by the leaf they are routed to (kNoLeaf: T failures)
  static constexpr uint32_t kNoLeaf = 0xffffffffu;
  uint32_t last_leaf = kNoLeaf;
  int refit_mode = 0;      // REFIT: 0 never, 1 as soon as a key overflows, 2 once the overflow has cost what the refit will
  double rw = 19, w_line = 2, prior = 16;  // RW reads per write; W_LINE lines a rewritten line is worth; PRIOR keys of prior weight
  size_t refit_max = 2048, wmax = 15, base_n = 1, t_now = 0;
  size_t refits = 0, refit_tries = 0, refit_absorbed = 0, refit_slots = 0;
  size_t why[7] = {};  // refit windows: too large, leaves interleaved, no overflow key in range, no room, no gain, done on old lines, done refitted
  std::vector<LeafStat> leaf;
  size_t ev[kEvs] = {}, moved = 0, written = 0, shift_max, removed = 0;
  double ov_cost;
  double delta = 0, w_cost = 0;  // lines the move being tried adds to the keys it moves; W_COST: lines one moved slot is worth
  bool by_cost = false;          // COST=1: each insert takes the cheapest of shift right, shift left and overflow, in lines
  size_t dlim = 0, need = 0;  // D_MAX: how far the walk bound may grow, and what the move being tried needs
  bool g_index = false;      // G_INDEX=1: overflow and exceptions each in an index of their own, whose lines are measured

  Sim(std::vector<KV>& kv) : idx(new Index("ab4")), pr{*idx} {
    idx->init();
    idx->bulk_load(kv.data(), kv.size());
    leaf.resize(pr.nodes());
    shift_max = size_t(env_d("SHIFT_MAX", 8));
    ov_cost = env_d("OV_COST", 1.5);
    dlim = std::max(pr.d(), size_t(env_d("D_MAX", 0)));
    g_index = env_d("G_INDEX", 0) != 0;
    by_cost = env_d("COST", 0) != 0;
    refit_mode = int(env_d("REFIT", 0));
    rw = env_d("RW", 19), w_line = env_d("W_LINE", 2), prior = env_d("PRIOR", 16);
    refit_max = size_t(env_d("REFIT_MAX", 2048)), wmax = size_t(env_d("REFIT_LEAVES", 15));
    base_n = kv.size();
    {
      auto& S = pr.slots();
      for (size_t i = 0, n = S.size(); i < n;) {
        const uint64_t k = S[i].first;
        while (i < n && S[i].first == k) ++i;
        ++leaf[leaf_of(k)].in_slots;
      }
    }
    w_cost = env_d("W_COST", 0);
  }

  // true if key k, whose copies now start at slot f and end at slot r, is where get can find it: get starts at the first slot of
  // k's bucket and walks at most d more lines
  // fo: where its copies started before the move, for the lines the move adds to its lookups (delta)
  bool reachable(uint64_t k, size_t f, size_t r, size_t fo) {
    const size_t b = pr.bucket(k - pr.lo()), first = std::max(f, b * B);
    if (first > r || first / B > b + dlim) return false;
    need = std::max(need, first / B - b);
    delta += double(first / B) - double(std::max(fo, b * B) / B);
    return true;
  }
  void grow(size_t lines_past) {  // the walk bound is one number for the whole index: a move that needs more raises it for all
    if (lines_past > pr.d()) pr.set_d(lines_past);
  }

  Ev insert(uint64_t x) {
    auto& S = pr.slots();
    const size_t n = S.size(), d = dlim;
    need = 0, delta = 0;
    const uint64_t v = x - pr.lo();
    if (v > pr.span()) {  // G: outside the directory's span, an exception as the build makes them
      auto& e = pr.exc();
      e.insert(std::lower_bound(e.begin(), e.end(), x, [](const KV& a, uint64_t k) { return a.first < k; }), KV(x, payload(x)));
      return kG;
    }
    size_t lf = 0;
    const size_t b = pr.bucket(v, &lf), p = pr.lower_slot(x);
    LeafStat& ls = leaf[lf];
    ++ls.ins;
    last_leaf = uint32_t(lf);
    const KV me(x, payload(x));
    if (p == n) {  // above every key in the slots: the copies after the one get reads of the last key are free
      const uint64_t lastk = S[n - 1].first;
      size_t f = n - 1;
      while (f > 0 && S[f - 1].first == lastk) --f;
      const size_t bl = pr.bucket(lastk - pr.lo()), q = std::max(std::max(f, bl * B) + 1, b * B);
      if (q >= n || q / B > b + d) return ++ls.v_fail, kOvVwin;
      std::fill(S.begin() + std::ptrdiff_t(q), S.end(), me);
      written += n - q;
      pr.set_last(x);
      grow(q / B - b);
      return kZ;
    }
    const uint64_t sk = S[p].first;  // the next key, whose copies fill [p, r]
    size_t r = p;
    while (r + 1 < n && S[r + 1].first == sk) ++r;
    const size_t q = std::max(p, b * B);  // the first slot x may take: at or after its place in order and its bucket's start
    if (q > r) return ++ls.t_fail, kOvT;
    if (q / B > b + d) return ++ls.v_fail, kOvVwin;
    if (q < r) {  // free slots in reach: x takes [p, q], the next key keeps (q, r]
      if (!reachable(sk, q + 1, r, p)) return ++ls.v_fail, kOvVwin;
      if (by_cost && ov_cost < delta) return ++ls.v_fail, kOvC;  // the next key would pay more than x saves
      std::fill(S.begin() + std::ptrdiff_t(p), S.begin() + std::ptrdiff_t(q + 1), me);
      written += q + 1 - p;
      grow(std::max(need, q / B - b));
      return kZ;
    }
    // q == r: the slot x needs holds the next key itself. Shift right from r to the nearest free slot, or, when x's place is
    // past its bucket's start, shift left from r - 1. A free slot is a copy that is not the last of its run.
    size_t er = SIZE_MAX, el = SIZE_MAX, need_r = 0, need_l = 0;
    double delta_r = 0, delta_l = 0;
    need = 0, delta = 0;
    for (size_t i = r + 1; i + 1 < n && i <= r + shift_max; ++i)
      if (S[i].first == S[i + 1].first) { er = i; break; }
    if (er != SIZE_MAX) {  // every key in [r, er) moves one slot right, and the key at er loses that copy
      for (size_t i = r; i <= er && er != SIZE_MAX;) {
        const uint64_t k = S[i].first;
        size_t j = i;
        while (j + 1 < n && S[j + 1].first == k) ++j;
        const bool ok = i < er ? reachable(k, i + 1, j + 1, i == r ? p : i) : reachable(k, er + 1, j, er);
        if (!ok) er = SIZE_MAX;
        i = j + 1;
      }
    }
    need_r = need, need = 0;
    delta_r = delta, delta = 0;
    if (p == r && r >= 1 && r - 1 >= b * B) {
      for (size_t i = r - 1; i-- > 0 && r - 1 - i <= shift_max;)
        if (S[i].first == S[i + 1].first) { el = i; break; }
      if (el != SIZE_MAX) {  // every key in (el, r - 1] moves one slot left
        for (size_t i = el + 1; i <= r - 1 && el != SIZE_MAX;) {
          const uint64_t k = S[i].first;
          size_t j = i;
          while (j + 1 <= r - 1 && S[j + 1].first == k) ++j;
          size_t f = i;
          while (f > 0 && S[f - 1].first == k) --f;
          if (!reachable(k, f == i ? i - 1 : f, j - 1, f)) el = SIZE_MAX;
          i = j + 1;
        }
      }
    }
    need_l = need, delta_l = delta;
    const size_t dr = er == SIZE_MAX ? SIZE_MAX : er - r, dl = el == SIZE_MAX ? SIZE_MAX : r - 1 - el;
    if (dr == SIZE_MAX && dl == SIZE_MAX) return ++ls.v_fail, kOvVshift;
    ++ls.v_fail;  // absorbed, but only by moving keys: V was short here
    bool right = dr <= dl;
    if (by_cost) {
      // In lines, over every later lookup: x reads the same slot walk wherever it ends up, so a shift is worth it only if the
      // lines it adds to the keys it moves (plus what the write itself is priced at) stay under an overflow lookup's cost.
      const double inf = 1e300, cr = dr == SIZE_MAX ? inf : delta_r + w_cost * double(dr),
                   cl = dl == SIZE_MAX ? inf : delta_l - double(r / B - (r - 1) / B) + w_cost * double(dl);
      if (ov_cost <= std::min(cr, cl)) return kOvC;
      right = cr <= cl;
    }
    if (right) {
      std::copy_backward(S.begin() + std::ptrdiff_t(r), S.begin() + std::ptrdiff_t(er), S.begin() + std::ptrdiff_t(er + 1));
      std::fill(S.begin() + std::ptrdiff_t(p), S.begin() + std::ptrdiff_t(r + 1), me);  // the next key's copies before r are x's now
      moved += dr, written += dr + 1 + (r - p);
      grow(std::max(need_r, r / B - b));
      return kR;
    }
    std::copy(S.begin() + std::ptrdiff_t(el + 1), S.begin() + std::ptrdiff_t(r), S.begin() + std::ptrdiff_t(el));
    S[r - 1] = me;
    moved += dl, written += dl + 1;
    grow(std::max(need_l, (r - 1) / B - b));
    return kL;
  }

  size_t leaf_of(uint64_t k) {
    size_t lf = 0;
    pr.bucket(k - pr.lo(), &lf);
    return lf;
  }

  // The in-place repair: T and V of a few neighbouring leaves redone together, with G's keys taken back in.
  // The window is the leaf and, if that is not enough, its neighbours (1, 3, 5, ... leaves). Its keys are those in the slots between
  // the key before the window and the key after it, plus the overflow keys routed to its leaves. The slots between those two keys
  // are shared among the leaves by weight, keys x (1 + rate), where rate is the leaf's inserts per key seen so far, shrunk toward
  // the index's own (a gamma prior worth PRIOR keys): the slack goes where the keys are arriving, which is the new density. Each
  // leaf's line (first slot, slots) is then set so its first and last key land at the ends of its share, and the keys are placed as
  // the build places them. Kept only if every key stays within the walk bound and the window's lookups read fewer lines than
  // before (overflow lookups at OV_COST). Nothing outside those slots and those leaves' two numbers is written.
  bool refit(uint32_t lf0, uint64_t hint) {
    auto& S = pr.slots();
    const size_t n = S.size();
    ++refit_tries;
    const size_t p = pr.lower_slot(hint);
    if (p >= n) return false;
    size_t i0 = p, i1 = p, slot_keys = 0;  // the window's keys are the runs in [i0, i1); i1 starts the key after it
    std::vector<uint32_t> wl = {lf0};
    auto right = [&](uint32_t lf) {
      while (i1 < n && leaf_of(S[i1].first) == lf && slot_keys <= refit_max) {
        const uint64_t k = S[i1].first;
        while (i1 < n && S[i1].first == k) ++i1;
        ++slot_keys;
      }
    };
    auto left = [&](uint32_t lf) {
      while (i0 > 0 && leaf_of(S[i0 - 1].first) == lf && slot_keys <= refit_max) {
        const uint64_t k = S[i0 - 1].first;
        while (i0 > 0 && S[i0 - 1].first == k) --i0;
        ++slot_keys;
      }
    };
    right(lf0), left(lf0);
    struct Item { uint64_t k, pay; uint32_t lf; bool from_ov; };
    std::vector<Item> items;
    for (;;) {
      if (slot_keys > refit_max || i1 >= n) return ++why[0], false;  // too large to rewrite at once, or the last keys of the array
      const uint64_t succ = S[i1].first;
      size_t rs = i1;
      while (rs + 1 < n && S[rs + 1].first == succ) ++rs;  // the key after the window keeps its last copy, at rs
      const size_t s0 = i0, cap = rs - s0;                 // the slots to share: [s0, rs)
      const bool has_pred = i0 > 0;
      const uint64_t pred = has_pred ? S[i0 - 1].first : 0;
      items.clear();
      double before = 0;
      for (size_t i = i0; i < i1;) {
        const uint64_t k = S[i].first;
        size_t j = i;
        while (j + 1 < i1 && S[j + 1].first == k) ++j;
        size_t lf = 0;
        const size_t b = pr.bucket(k - pr.lo(), &lf);
        before += double(std::max(i, b * B) / B - b + 1);
        items.push_back(Item{k, S[j].second, uint32_t(lf), false});
        i = j + 1;
      }
      size_t taken = 0;
      for (uint32_t lf : wl) {
        const auto it = ovl.find(lf);
        if (it == ovl.end()) continue;
        for (uint64_t k : it->second)
          if ((!has_pred || k > pred) && k < succ) items.push_back(Item{k, payload(k), lf, true}), ++taken, before += 1 + ov_cost;
      }
      bool ok = taken > 0 && items.size() <= cap;
      if (!taken) ++why[2];
      else if (items.size() > cap) ++why[3];
      if (ok) {
        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.k < b.k; });
        // each leaf's keys must be one stretch, in the order of the leaves
        struct Grp { uint32_t lf; size_t k0, k1; double w; uint32_t a, c; };
        std::vector<Grp> gs;
        for (size_t i = 0; i < items.size(); ++i) {
          if (gs.empty() || gs.back().lf != items[i].lf) gs.push_back(Grp{items[i].lf, i, i, 0, 0, 0});
          gs.back().k1 = i + 1;
        }
        for (size_t a = 0; a < gs.size() && ok; ++a)
          for (size_t b = a + 1; b < gs.size(); ++b) if (gs[a].lf == gs[b].lf) ok = false;
        // a leaf's line may change only if every key it has in the slots is in this window: any other would be stranded
        bool whole = ok;
        if (whole) {
          std::unordered_map<uint32_t, uint32_t> here;
          for (const Item& it : items) if (!it.from_ov) ++here[it.lf];
          for (const Grp& g : gs) if (here[g.lf] != leaf[g.lf].in_slots) whole = false;
        }
        if (!ok) ++why[1];
        double wsum = 0;
        const double g0 = double(t_now) / double(base_n);
        for (Grp& g : gs) {
          const double cnt = double(g.k1 - g.k0), rate = (prior * g0 + double(leaf[g.lf].ins)) / (prior + cnt);
          g.w = cnt * (1 + rate), wsum += g.w;
        }
        double best = before;
        int pick = -1;
        std::vector<size_t> pos(items.size()), best_pos;
        std::vector<std::pair<uint32_t, uint32_t>> best_line;
        for (int cand = 0; cand < (whole ? 2 : 1) && ok; ++cand) {  // 0: the lines as they are (V alone redone); 1: the lines refitted (T and V)
          double acc = 0;
          bool fits = true;
          for (Grp& g : gs) {
            const unsigned top = pr.leaf_top(g.lf);
            if (cand == 0) { g.a = pr.leaf_a(g.lf), g.c = pr.leaf_c(g.lf); continue; }
            const size_t u0 = s0 + size_t(acc / wsum * double(cap)), u1 = s0 + size_t((acc + g.w) / wsum * double(cap));
            acc += g.w;
            const uint64_t v0 = items[g.k0].k - pr.lo(), v1 = items[g.k1 - 1].k - pr.lo();
            const double f0 = std::ldexp(double(v0 << top), -64), f1 = std::ldexp(double(v1 << top), -64);
            if (f1 < f0) { fits = false; break; }  // a key whose skipped bits differ: this line cannot order it
            double c = f1 > f0 ? double(u1 > u0 + 1 ? u1 - u0 - 1 : 1) / (f1 - f0) : 1;
            c = std::min(std::max(c, 1.0), double(Probe::kMaxSlots));
            uint32_t ci = uint32_t(c);
            while (ci > 1 && Probe::slot_on(0, top, ci, v0) > u0) ci /= 2;  // the first slot must not fall below 0
            const size_t off = Probe::slot_on(0, top, ci, v0);
            if (off > u0 || u0 - off > Probe::kMaxA) { fits = false; break; }
            g.a = uint32_t(u0 - off), g.c = ci;
          }
          if (!fits) continue;
          size_t prev = s0 == 0 ? SIZE_MAX : s0 - 1, worst = 0;
          double lines = 0;
          for (const Grp& g : gs) {
            const unsigned top = pr.leaf_top(g.lf);
            for (size_t i = g.k0; i < g.k1 && fits; ++i) {
              const size_t b = Probe::slot_on(g.a, top, g.c, items[i].k - pr.lo()) / B, at = prev == SIZE_MAX || b * B > prev ? b * B : prev + 1;
              if (at >= rs || at / B > b + dlim) { fits = false; break; }
              lines += double(at / B - b + 1), worst = std::max(worst, at / B - b);
              pos[i] = at, prev = at;
            }
          }
          if (!fits || lines >= best) continue;
          best = lines, pick = cand, best_pos = pos, need = worst;
          best_line.clear();
          for (const Grp& g : gs) best_line.emplace_back(g.a, g.c);
        }
        if (pick >= 0) {  // write it: the keys at their slots, each free slot a copy of the next pair
          size_t w = s0;
          for (size_t i = 0; i < items.size(); ++i) {
            const KV kvp(items[i].k, items[i].pay);
            while (w <= best_pos[i]) S[w++] = kvp;
          }
          const KV nx = S[rs];
          while (w < rs) S[w++] = nx;
          for (size_t g = 0; g < gs.size(); ++g) pr.set_leaf(gs[g].lf, best_line[g].first, best_line[g].second);
          for (const Item& it : items) if (it.from_ov) ++leaf[it.lf].in_slots;
          ++why[pick ? 6 : 5];
          grow(need);
          for (uint32_t lf : wl) {  // the overflow keys taken back
            const auto it = ovl.find(lf);
            if (it == ovl.end()) continue;
            auto& v = it->second;
            const size_t was = v.size();
            v.erase(std::remove_if(v.begin(), v.end(), [&](uint64_t k) { return (!has_pred || k > pred) && k < succ; }), v.end());
            leaf[lf].ovm = uint32_t(v.size()), leaf[lf].ov_t = double(v.size()) * double(t_now), leaf[lf].next_try = uint32_t(v.size()) + 1;
            (void)was;
          }
          pr.add_live(taken);
          ++refits, refit_absorbed += taken, refit_slots += cap, written += cap;
          return true;
        }
      }
      if (ok) ++why[4];  // room, but no fewer lines
      if (wl.size() >= wmax) return false;
      // not enough room, or no gain: take in a neighbour on each side
      const uint32_t rl = uint32_t(leaf_of(S[i1].first));
      wl.push_back(rl), right(rl);
      if (i0 > 0) {
        const uint32_t ll = uint32_t(leaf_of(S[i0 - 1].first));
        if (std::find(wl.begin(), wl.end(), ll) == wl.end()) wl.insert(wl.begin(), ll), left(ll);
      }
    }
  }

  void apply_insert(uint64_t x) {
    pr.layout_changed();
    ++t_now;
    const Ev e = insert(x);
    ++ev[e];
    if (e == kG) return;
    if (e != kOvT && e != kOvVwin && e != kOvVshift && e != kOvC) { pr.add_live(1), ++leaf[last_leaf].in_slots; return; }
    const uint32_t lf = e == kOvT ? kNoLeaf : last_leaf;
    ovl[lf].push_back(x);
    if (lf == kNoLeaf || !refit_mode) return;
    LeafStat& ls = leaf[lf];
    ++ls.ovm, ls.ov_t += double(t_now);
    if (ls.ovm < ls.next_try) return;
    if (refit_mode == 2) {
      // Rent or buy. Each overflow key has cost OV_COST lines per lookup since it arrived, at RW lookups per write spread over
      // the keys; the refit rewrites about the leaf's slots, W_LINE lines of work per line. Refit once the first has reached the
      // second: at most twice what the best choice in hindsight would have paid.
      const double paid = ov_cost * rw / double(base_n + t_now) * (double(ls.ovm) * double(t_now) - ls.ov_t);
      const double price = w_line * 1.25 * double(pr.leaf_keys() + ls.ovm) / B;
      if (paid < price) return;
    }
    if (!refit(lf, x)) ls.next_try = 2 * ls.ovm + 1;  // no room in reach: not again until the overflow has doubled
  }
  bool apply_delete(uint64_t x) {
    if (idx->remove(x)) {
      if (x - pr.lo() <= pr.span()) --leaf[leaf_of(x)].in_slots;
      return ++removed, true;
    }
    auto& v = ovl[uint32_t(leaf_of(x))];  // rare: only churn deletes keys that overflowed
    auto it = std::find(v.begin(), v.end(), x);
    if (it == v.end()) {
      auto& t = ovl[kNoLeaf];
      it = std::find(t.begin(), t.end(), x);
      if (it == t.end()) return false;
      *it = t.back(), t.pop_back();
      return ++removed, true;
    }
    *it = v.back(), v.pop_back();
    return ++removed, true;
  }
  void sync_ov() {
    ov.clear();
    for (const auto& kvp : ovl) ov.insert(ov.end(), kvp.second.begin(), kvp.second.end());
    std::sort(ov.begin(), ov.end());
  }
};

// the lines a fresh ab4 on these keys reads at this many bytes a key (NAN if they cannot hold them), and its bytes
double build_lines(const std::vector<uint64_t>& keys, double budget, double restore, double* bytes = nullptr) {
  std::vector<KV> kv;
  kv.reserve(keys.size());
  for (uint64_t k : keys) kv.emplace_back(k, payload(k));
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.4f", budget);
  setenv("AB_BUDGET", buf, 1);
  double lines = NAN;
  try {
    Index f("ab4");
    f.init();
    f.bulk_load(kv.data(), kv.size());
    lines = f.expected_lines();
    if (bytes) *bytes = double(f.memory_consumption());
  } catch (const std::exception&) {}  // under one slot per key
  std::snprintf(buf, sizeof buf, "%.4f", restore);
  setenv("AB_BUDGET", buf, 1);
  return lines;
}

struct Measure {
  size_t p99 = 0, p999 = 0;
  size_t live = 0, in_slots = 0, d_seen = 0, leaves_ins = 0, c_none = 0, c_v = 0, c_t = 0, c_vt = 0, replace_out = 0;
  double lines_hit = 0, lines_replace = 0, lines_fresh_same = NAN, lines_fresh_base = NAN, budget_same = 0, bytes_per_live = 0;
  double ks = 0, tv = 0, max_ratio = 0, miss_lines = 0, ov_extra = 0, exc_lines = 0;
  bool valid = true;
  std::vector<size_t> bins;
};

Measure measure(Sim& s, const std::vector<uint64_t>& base, double base_budget, bool fresh, std::mt19937_64& g) {
  Measure m;
  auto& S = s.pr.slots();
  const size_t n = S.size(), d = s.pr.d();
  for (LeafStat& l : s.leaf) l.keys = 0, l.lines = 0;
  s.sync_ov();
  std::vector<uint64_t> slot_keys;
  slot_keys.reserve(n);
  double sum = 0;
  std::vector<size_t> hist(66, 0);  // keys by the lines their lookup reads (65: more than 64)
  for (size_t i = 0; i < n;) {  // every key in the slots: the lines get reads for it, exactly
    const uint64_t k = S[i].first;
    size_t r = i;
    while (r + 1 < n && S[r + 1].first == k) ++r;
    size_t lf = 0;
    const size_t b = s.pr.bucket(k - s.pr.lo(), &lf), first = std::max(i, b * B);
    if (first > r || first / B > b + d) m.valid = false;  // a key get cannot find: the simulator broke an invariant
    const size_t lines = first / B - b + 1;
    sum += double(lines);
    ++hist[std::min<size_t>(lines, 65)];
    m.d_seen = std::max(m.d_seen, lines - 1);
    ++s.leaf[lf].keys, s.leaf[lf].lines += lines;
    slot_keys.push_back(k);
    i = r + 1;
  }
  m.in_slots = slot_keys.size();
  auto walk = [&](uint64_t x) {  // the lines get reads in the slots for a key that is not there
    const size_t b = s.pr.bucket(x - s.pr.lo()), p = s.pr.lower_slot(x), first = std::max(p, b * B);
    return double(std::min(d + 1, first / B - b + 1));
  };
  // What a lookup pays outside the slots. By default an assumed OV_COST for the overflow and a bisection for the exceptions; with
  // G_INDEX each set is an ab4 of its own at the base budget, rebuilt at this point, and the lines it reads are counted.
  double ov_extra = s.ov_cost, ov_bytes = 16.0 * double(s.ov.size()), exc_bytes = 16.0 * double(s.pr.exc().size());
  double exc_lines = s.pr.exc().size() > 4 ? 1 + std::ceil(std::log2(double(s.pr.exc().size()) / 4)) : 1;
  if (s.g_index) {
    if (s.ov.size() >= 4096) {
      const double l = build_lines(s.ov, base_budget, base_budget, &ov_bytes);
      if (!std::isnan(l)) ov_extra = l;
    } else {
      ov_extra = s.ov.size() > 4 ? 1 + std::ceil(std::log2(double(s.ov.size()) / 4)) : 1;
    }
    if (s.pr.exc().size() >= 4096) {
      std::vector<uint64_t> ek;
      for (const KV& e : s.pr.exc()) ek.push_back(e.first);
      const double l = build_lines(ek, base_budget, base_budget, &exc_bytes);
      if (!std::isnan(l)) exc_lines = l;
    }
  }
  m.ov_extra = ov_extra, m.exc_lines = exc_lines;
  for (uint64_t x : s.ov) {  // a key that found no slot still belongs to the leaf it is routed to
    size_t lf = 0;
    s.pr.bucket(x - s.pr.lo(), &lf);
    const double w = walk(x) + ov_extra;
    sum += w;
    ++hist[std::min<size_t>(size_t(std::ceil(w)), 65)];
    ++s.leaf[lf].keys, s.leaf[lf].lines += uint64_t(std::ceil(w));
  }
  // an exception: the range check sends it to a sorted array, searched by bisection: the lines that reads (4 pairs a line)
  sum += double(s.pr.exc().size()) * exc_lines;
  // the live keys, sorted: slots, overflow, exceptions
  std::vector<uint64_t> live(slot_keys.size() + s.ov.size());
  std::merge(slot_keys.begin(), slot_keys.end(), s.ov.begin(), s.ov.end(), live.begin());
  const size_t in_span = live.size();
  for (const KV& e : s.pr.exc()) live.push_back(e.first);
  std::inplace_merge(live.begin(), live.begin() + std::ptrdiff_t(in_span), live.end());
  m.live = live.size();
  m.lines_hit = sum / double(m.live);
  {  // the tail of what a lookup of a stored key reads: the latency side of the same count
    size_t seen = 0, tot = 0;
    for (size_t h : hist) tot += h;
    for (size_t l = 0; l < hist.size(); ++l) {
      seen += hist[l];
      if (!m.p99 && double(seen) >= 0.99 * double(tot)) m.p99 = l;
      if (!m.p999 && double(seen) >= 0.999 * double(tot)) m.p999 = l;
    }
  }
  {  // misses: keys next to stored ones that are not stored
    double ms = 0;
    size_t cnt = 0;
    for (size_t t = 0; t < 200000 && !slot_keys.empty(); ++t) {
      const uint64_t x = slot_keys[g() % slot_keys.size()] + 1;
      if (x - s.pr.lo() > s.pr.span() || std::binary_search(live.begin(), live.end(), x)) continue;
      ms += walk(x) + (s.ov.empty() ? 0 : ov_extra), ++cnt;
    }
    m.miss_lines = cnt ? ms / double(cnt) : 0;
  }
  {  // the same keys placed afresh on the same directory and lines: V redone, T kept
    size_t prev = SIZE_MAX;
    double rs = double(s.pr.exc().size()) * exc_lines;
    size_t a = 0, o = 0;
    while (a < slot_keys.size() || o < s.ov.size()) {
      const uint64_t k = o == s.ov.size() || (a < slot_keys.size() && slot_keys[a] < s.ov[o]) ? slot_keys[a++] : s.ov[o++];
      const size_t b = s.pr.bucket(k - s.pr.lo()), pos = prev == SIZE_MAX || b * B > prev ? b * B : prev + 1;
      if (pos >= n || pos / B > b + d) { ++m.replace_out, rs += walk(k) + ov_extra; continue; }
      rs += double(pos / B - b + 1);
      prev = pos;
    }
    m.lines_replace = rs / double(m.live);
  }
  const double bytes = double(s.idx->memory_consumption()) - 16.0 * double(s.pr.exc().size()) + exc_bytes + ov_bytes;
  m.bytes_per_live = bytes / double(m.live);
  m.budget_same = m.bytes_per_live;
  if (fresh) {
    m.lines_fresh_same = build_lines(live, m.budget_same, base_budget);
    m.lines_fresh_base = build_lines(live, base_budget, base_budget);
  }
  {  // the distribution now against the base: Kolmogorov distance, then 1024 equal-count bins of the base
    size_t i = 0, j = 0;
    const double n0 = double(base.size()), n1 = double(live.size());
    while (i < base.size() || j < live.size()) {
      const uint64_t x = j == live.size() || (i < base.size() && base[i] <= live[j]) ? base[i] : live[j];
      while (i < base.size() && base[i] <= x) ++i;
      while (j < live.size() && live[j] <= x) ++j;
      m.ks = std::max(m.ks, std::abs(double(i) / n0 - double(j) / n1));
    }
    const size_t nb = 1024;
    m.bins.assign(nb, 0);
    size_t at = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
      const size_t hi_rank = (bi + 1) * base.size() / nb;
      const size_t end = bi + 1 == nb ? live.size() : size_t(std::lower_bound(live.begin(), live.end(), base[hi_rank]) - live.begin());
      m.bins[bi] = end - at;
      const double p0 = double(hi_rank - bi * base.size() / nb) / n0, p1 = double(end - at) / n1;
      m.tv += 0.5 * std::abs(p1 - p0);
      if (p0 > 0) m.max_ratio = std::max(m.max_ratio, p1 / p0);
      at = end;
    }
  }
  const double tau = s.pr.tau();
  for (const LeafStat& l : s.leaf) {  // do the V and the T failures fall in the same leaves?
    if (!l.ins) continue;
    ++m.leaves_ins;
    const bool fv = l.v_fail > 0, ft = l.t_fail > 0 || (l.keys > s.pr.leaf_keys() && double(l.lines) > tau * double(l.keys));
    ++(fv ? (ft ? m.c_vt : m.c_v) : (ft ? m.c_t : m.c_none));
  }
  // the real index on the edited slots: get on stored, overflowed and absent keys, scan against the sorted keys
  uint64_t val = 0;
  for (size_t t = 0; t < 200000 && !slot_keys.empty(); ++t) {
    const uint64_t k = slot_keys[g() % slot_keys.size()];
    if (!s.idx->get(k, val) || val != payload(k)) m.valid = false;
  }
  for (size_t t = 0; t < 50000 && !s.ov.empty(); ++t)
    if (s.idx->get(s.ov[g() % s.ov.size()], val)) m.valid = false;
  for (const KV& e : s.pr.exc())
    if (!s.idx->get(e.first, val) || val != e.second) m.valid = false;
  {
    std::vector<uint64_t> expect(slot_keys);  // what scan must return: the slots' keys and the exceptions
    const size_t mid = expect.size();
    for (const KV& e : s.pr.exc()) expect.push_back(e.first);
    std::inplace_merge(expect.begin(), expect.begin() + std::ptrdiff_t(mid), expect.end());
    std::vector<KV> out(64);
    for (int t = 0; t < 2000 && !expect.empty(); ++t) {
      const uint64_t start = expect[g() % expect.size()] + (t & 1);
      const size_t got = s.idx->scan(start, out.size(), out.data());
      auto it = std::lower_bound(expect.begin(), expect.end(), start);
      const size_t want = std::min<size_t>(out.size(), size_t(expect.end() - it));
      if (got != want) m.valid = false;
      for (size_t i = 0; i < std::min(got, want); ++i)
        if (out[i].first != it[std::ptrdiff_t(i)] || out[i].second != payload(out[i].first)) m.valid = false;
    }
  }
  return m;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) return std::fprintf(stderr, "usage: sim KEYS N STREAM [LEVELS]\n"), 2;
  const std::string path = argv[1], stream = argv[3];
  const size_t want = size_t(std::strtoull(argv[2], nullptr, 10));
  std::vector<double> levels;
  for (char* t = std::strtok(argc > 4 ? argv[4] : strdup("0.005,0.01,0.02,0.05,0.1,0.2"), ","); t; t = std::strtok(nullptr, ","))
    levels.push_back(std::strtod(t, nullptr));
  std::sort(levels.begin(), levels.end());
  const double cmax = levels.back(), base_budget = env_d("AB_BUDGET", 20);
  const bool fresh = env_d("FRESH", 1) != 0, deleting = stream == "delete" || stream == "delhot";
  std::mt19937_64 g(uint64_t(env_d("SEED", 1)));
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.4f", base_budget);
  setenv("AB_BUDGET", buf, 1);

  std::vector<uint64_t> all = load_keys(path, want, g), base, ops;  // ops: the stream, in the order it is applied
  const size_t n = all.size();
  if (n < 1000) return std::fprintf(stderr, "too few keys\n"), 2;
  auto split = [&](std::vector<char>& out) {  // base = keys not marked, ops = the marked ones
    for (size_t i = 0; i < n; ++i) (out[i] ? ops : base).push_back(all[i]);
  };
  std::vector<char> mark(n, 0);
  if (stream == "holdout" || stream == "churn") {  // the stream has the base's own distribution
    const size_t s = size_t(cmax * double(n) / (1 + cmax));
    for (size_t c = 0; c < s;) { const size_t i = g() % n; if (!mark[i]) mark[i] = 1, ++c; }
    split(mark);
    std::shuffle(ops.begin(), ops.end(), g);
  } else if (stream == "hot") {  // every insert falls in one window of ranks
    const size_t s = size_t(cmax * double(n) / (1 + cmax)), w = std::min(n / 2, std::max(2 * s, n / 50)), r0 = size_t(0.6 * double(n - w));
    for (size_t c = 0; c < s;) { const size_t i = r0 + g() % w; if (!mark[i]) mark[i] = 1, ++c; }
    split(mark);
    std::shuffle(ops.begin(), ops.end(), g);
  } else if (stream == "append") {  // the largest keys arrive last, in order
    const size_t s = size_t(cmax * double(n) / (1 + cmax));
    for (size_t i = n - s; i < n; ++i) mark[i] = 1;
    split(mark);
  } else if (stream == "cluster") {  // a new dense run inside the widest gap of the middle half
    base = all;
    size_t at = n / 4;
    for (size_t i = n / 4; i + 1 < 3 * n / 4; ++i) if (all[i + 1] - all[i] > all[at + 1] - all[at]) at = i;
    const size_t s = size_t(cmax * double(n));
    const uint64_t gap = all[at + 1] - all[at], step = std::max<uint64_t>(1, gap / 4 / (s + 1));
    for (size_t i = 1; i <= s && all[at] + gap / 3 + i * step < all[at + 1]; ++i) ops.push_back(all[at] + gap / 3 + i * step);
    std::shuffle(ops.begin(), ops.end(), g);
  } else if (deleting) {
    base = all;
    const size_t s = size_t(cmax * double(n)), w = stream == "delhot" ? std::min(n / 2, std::max(2 * s, n / 50)) : n, r0 = stream == "delhot" ? size_t(0.6 * double(n - w)) : 0;
    for (size_t c = 0; c < s;) { const size_t i = r0 + g() % w; if (!mark[i]) mark[i] = 1, ops.push_back(all[i]), ++c; }
  } else {
    return std::fprintf(stderr, "unknown stream %s\n", stream.c_str()), 2;
  }
  std::vector<uint64_t>().swap(all);
  std::vector<char>().swap(mark);
  std::vector<KV> kv;
  kv.reserve(base.size());
  for (uint64_t k : base) kv.emplace_back(k, payload(k));
  Sim s(kv);
  std::vector<KV>().swap(kv);

  const char* outp = std::getenv("OUT");
  FILE* tsv = outp && *outp ? std::fopen((std::string(outp) + ".tsv").c_str(), "w") : nullptr;
  FILE* cdf = outp && *outp ? std::fopen((std::string(outp) + ".cdf.tsv").c_str(), "w") : nullptr;
  const char* head = "stream\tlevel\tbase\tlive\tops\tZ\tR\tL\tovT\tovVwin\tovVshift\tovC\tG\tremoved\tmoved_per_op\twritten_per_op\tlines_hit\tlines_replace\tlines_fresh_same\tlines_fresh_base\tregret_place\tregret_model\tregret\tmiss_lines\td\td_seen\tbytes_per_live\tks\ttv\tmax_bin_ratio\tleaves_ins\tneither\tV_only\tT_only\tV_and_T\treplace_out\tov_extra\texc_lines\tlines_p99\tlines_p999\trefits\trefit_tries\trefit_absorbed\tov_now\tvalid\n";
  std::fputs(head, stdout);
  if (tsv) std::fputs(head, tsv);
  if (cdf) std::fputs("level\tbin\tkey_lo\tbase\tlive\n", cdf);
  size_t done = 0, churn_del = 0;
  std::vector<uint64_t> churn_pool;
  if (stream == "churn") { churn_pool = base; std::shuffle(churn_pool.begin(), churn_pool.end(), g); }
  std::vector<double> lv = {0};
  lv.insert(lv.end(), levels.begin(), levels.end());
  for (double level : lv) {
    const size_t target = std::min(ops.size(), size_t(level * double(base.size())));
    for (; done < target; ++done) {
      if (deleting) s.apply_delete(ops[done]);
      else {
        if (stream == "churn" && churn_del < churn_pool.size()) s.apply_delete(churn_pool[churn_del++]);
        s.apply_insert(ops[done]);
      }
    }
    const Measure m = measure(s, base, base_budget, fresh, g);
    char line[1024];
    const double per = done ? 1.0 / double(done) : 0;
    std::snprintf(line, sizeof line, "%s\t%.4f\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%zu\t%zu\t%.3f\t%.5f\t%.5f\t%.2f\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%.3f\t%.3f\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%s\n",
                  stream.c_str(), level, base.size(), m.live, done, s.ev[kZ], s.ev[kR], s.ev[kL], s.ev[kOvT], s.ev[kOvVwin], s.ev[kOvVshift], s.ev[kOvC], s.ev[kG], s.removed,
                  double(s.moved) * per, double(s.written) * per, m.lines_hit, m.lines_replace, m.lines_fresh_same, m.lines_fresh_base,
                  m.lines_hit / m.lines_replace, m.lines_replace / m.lines_fresh_same, m.lines_hit / m.lines_fresh_same, m.miss_lines, s.pr.d(), m.d_seen,
                  m.bytes_per_live, m.ks, m.tv, m.max_ratio, m.leaves_ins, m.c_none, m.c_v, m.c_t, m.c_vt, m.replace_out, m.ov_extra, m.exc_lines, m.p99, m.p999, s.refits, s.refit_tries, s.refit_absorbed, s.ov.size(), m.valid ? "ok" : "FAIL");
    std::fputs(line, stdout);
    std::fflush(stdout);
    if (tsv) std::fputs(line, tsv), std::fflush(tsv);
    if (cdf)
      for (size_t bi = 0; bi < m.bins.size(); ++bi)
        std::fprintf(cdf, "%.4f\t%zu\t%" PRIu64 "\t%zu\t%zu\n", level, bi, base[bi * base.size() / m.bins.size()], (bi + 1) * base.size() / m.bins.size() - bi * base.size() / m.bins.size(), m.bins[bi]);
    if (!m.valid) return std::fprintf(stderr, "sim: the real get or scan disagrees after level %.4f\n", level), 1;
  }
  (void)kEvName;
  std::fprintf(stderr, "sim: refit windows: too_large=%zu interleaved=%zu none_in_range=%zu no_room=%zu no_gain=%zu kept_lines=%zu refitted=%zu\n", s.why[0], s.why[1], s.why[2], s.why[3], s.why[4], s.why[5], s.why[6]);
  return 0;
}
