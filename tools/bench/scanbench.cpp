// Range scan timing at 200M, with the same start keys for every index. Before timing, the scans are checked against the sorted keys:
// stored and absent starts, below the first key and above the last, around the peeled ends. Every returned pair is read, so nothing is
// optimized away, and the B+tree is wrapped to fill its output as the others do (GRE's scan for it only walks and adds).
//   scanbench KEYS NAME LABEL
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#ifdef WITH_ART
#include "artsync/artunsync.h"
#endif
#include "gret/registry.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
using u64 = uint64_t;
using Pair = std::pair<u64, u64>;
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }
static u64 payload(u64 k) { return k * 2654435761u + 17; }

class FilledBTree final : public indexInterface<u64, u64> {
  stx::btree<u64, u64> t;
 public:
  void init(Param* = nullptr) override {}
  void bulk_load(Pair* kv, size_t n, Param* = nullptr) override { t.bulk_load(kv, kv + n); }
  bool get(u64 k, u64& v, Param* = nullptr) override { auto i = t.find(k); if (i == t.end()) return false; v = i->second; return true; }
  bool put(u64, u64, Param* = nullptr) override { return false; }
  bool update(u64, u64, Param* = nullptr) override { return false; }
  bool remove(u64, Param* = nullptr) override { return false; }
  size_t scan(u64 k, size_t n, Pair* out, Param* = nullptr) override {
    auto i = t.lower_bound(k);
    size_t got = 0;
    for (; got < n && i != t.end(); ++got, ++i) out[got] = Pair(i->first, i->second);
    return got;
  }
  long long memory_consumption() override { const auto& s = t.get_stats(); return (long long)((s.leaves * s.leafslots + s.innernodes * s.innerslots) * 16); }
};

#ifdef WITH_ART
// GRE's wrapper for ART only counts what its range scan finds, so this one hands back the pairs, as the others do.
class FilledArt final : public indexInterface<u64, u64> {
  static void load(TID tid, Key& key) {
    u64 r = __builtin_bswap64(reinterpret_cast<Pair*>(tid)->first);
    key.set(reinterpret_cast<char*>(&r), sizeof(r));
  }
  static void make(u64 x, Key& key) {
    u64 r = __builtin_bswap64(x);
    key.set(reinterpret_cast<char*>(&r), sizeof(r));
  }
  ART_unsynchronized::Tree t{load};
  std::vector<TID> buf;
 public:
  void init(Param* = nullptr) override {}
  void bulk_load(Pair* kv, size_t n, Param* = nullptr) override {
    for (size_t i = 0; i < n; ++i) {
      Key k;
      make(kv[i].first, k);
      t.insert(k, reinterpret_cast<TID>(new Pair(kv[i])));
    }
  }
  bool get(u64 x, u64& v, Param* = nullptr) override {
    Key k;
    make(x, k);
    auto* p = reinterpret_cast<Pair*>(t.lookup(k));
    if (!p) return false;
    v = p->second;
    return true;
  }
  bool put(u64, u64, Param* = nullptr) override { return false; }
  bool update(u64, u64, Param* = nullptr) override { return false; }
  bool remove(u64, Param* = nullptr) override { return false; }
  size_t scan(u64 x, size_t n, Pair* out, Param* = nullptr) override {
    if (buf.size() < n) buf.resize(n);
    Key a, b, cont;
    make(x, a);
    make(~u64(0), b);
    size_t got = 0;
    t.lookupRange(a, b, cont, buf.data(), n, got);
    for (size_t i = 0; i < got; ++i) out[i] = *reinterpret_cast<Pair*>(buf[i]);
    return got;
  }
  long long memory_consumption() override { return (long long)t.size(); }
};
#endif

static volatile u64 g_sink;

int main(int, char** v) {
  FILE* f = std::fopen(v[1], "rb");
  u64 n = 0;
  if (!f || std::fread(&n, 8, 1, f) != 1) return 2;
  std::vector<u64> k(n);
  if (std::fread(k.data(), 8, n, f) != n) return 2;
  std::fclose(f);
  if (!std::is_sorted(k.begin(), k.end())) std::sort(k.begin(), k.end());
  k.erase(std::unique(k.begin(), k.end()), k.end());
  n = k.size();
  const std::string name = v[2], label = v[3];
  const bool sorted_output = !(std::getenv("PH_SCAN_SORTED") && std::string(std::getenv("PH_SCAN_SORTED")) == "0");
  if (k.back() == ~u64(0) && (name == "alex" || name == "pgm" || name == "artunsync")) { k.pop_back(); --n; }  // their sentinel, as in GRE's own runs on fb
  std::vector<Pair> kv(n);
  for (u64 i = 0; i < n; ++i) kv[i] = Pair(k[i], payload(k[i]));
  indexInterface<u64, u64>* idx = name == "btree" ? static_cast<indexInterface<u64, u64>*>(new FilledBTree) : gret::make_index<u64, u64>(name);
#ifdef WITH_ART
  if (name == "artunsync") idx = new FilledArt;
#endif
  if (!idx) { std::printf("%s not registered\n", v[2]); return 3; }
  Param p(1, 0);
  idx->init(&p);
  const auto b0 = Clock::now();
  idx->bulk_load(kv.data(), n, &p);
  const double build_s = secs(b0, Clock::now());
  std::vector<Pair>().swap(kv);

  // starts: the same sequence for every index on a dataset. 2^24 of them, so a timed window does not come back to the same lines
  const size_t S = size_t(1) << 24;
  std::mt19937_64 g(11);
  std::vector<u64> stored(S), absent(S);
  for (auto& x : stored) x = k[g() % n];
  for (auto& x : absent) {
    x = k[g() % n];
    for (int t = 0; t < 64; ++t) {
      const size_t r = g() % n;
      if (r + 1 < n && k[r + 1] - k[r] >= 2) { x = k[r] + (k[r + 1] - k[r]) / 2; break; }
    }
  }

  // correctness on the whole key set, not timed
  const size_t Ls[] = {16, 256, 4096, 65536};
  std::vector<Pair> out(65536);
  size_t bad = 0, checked = 0;
  auto check = [&](u64 x, size_t L) {
    if (x == ~u64(0) && name == "pgm") return;  // DynamicPGM's own sentinel: a scan from it crashes, not something the timing asks of it
    const size_t got = idx->scan(x, L, out.data(), &p);
    const size_t at = size_t(std::lower_bound(k.begin(), k.end(), x) - k.begin()), want = std::min<size_t>(L, n - at);
    if (!sorted_output) std::sort(out.begin(), out.begin() + std::ptrdiff_t(got));
    bool ok = got == want;
    for (size_t i = 0; ok && i < got; ++i) ok = out[i].first == k[at + i] && out[i].second == payload(k[at + i]);
    ++checked;
    if (!ok && ++bad <= 3) std::fprintf(stderr, "scan wrong: %s start %llu len %zu: got %zu want %zu\n", v[2], (unsigned long long)x, L, got, want);
  };
  for (size_t L : Ls) {
    const size_t m = L <= 256 ? 2000 : L == 4096 ? 300 : 40;
    for (size_t i = 0; i < m; ++i) { check(stored[i], L); check(absent[i], L); }
    for (size_t i = 0; i < std::min<u64>(80, n); ++i) { check(k[i], L); check(k[n - 1 - i], L); if (k[i]) check(k[i] - 1, L); if (k[n - 1 - i] != ~u64(0)) check(k[n - 1 - i] + 1, L); }
    check(0, L); check(~u64(0), L); check(u64(1) << 63, L);
  }
  std::vector<u64>().swap(k);

  // point lookups on the same build
  size_t hit = 0;
  u64 val = 0;
  const size_t Qp = 30000000;
  auto t0 = Clock::now();
  for (size_t i = 0; i < Qp; ++i) hit += idx->get(stored[i & (S - 1)], val, &p);
  const double point_ns = secs(t0, Clock::now()) * 1e9 / double(Qp);

  const double warm_s = 1, run_s = 3;
  u64 sink = 0;
  auto timed = [&](const std::vector<u64>& st, const char* kind, size_t L) {
    const size_t every = std::max<size_t>(1, 4096 / L);
    size_t next = 0;
    auto loop = [&](double budget, size_t& scans, u64& keys) {
      scans = keys = 0;
      const auto a = Clock::now();
      double el = 0;
      for (;;) {
        for (size_t j = 0; j < every; ++j) {
          const size_t got = idx->scan(st[next++ & (S - 1)], L, out.data(), &p);
          for (size_t i = 0; i < got; ++i) sink += out[i].first ^ out[i].second;
          keys += got;
          ++scans;
        }
        el = secs(a, Clock::now());
        if (el >= budget) return el;
      }
    };
    size_t scans;
    u64 keys;
    loop(warm_s, scans, keys);
    const double el = loop(run_s, scans, keys);
    std::printf("%s\t%s\tspk=%s\tleaf=%s\ttau=%s\tsorted=%d\t%s\tL=%zu\tns_scan=%.0f\tns_key=%.2f\tscans=%zu\tmean_got=%.1f\tpoint_ns=%.1f\thits=%zu\tBkey=%.2f\tbuild_s=%.1f\tchecked=%zu\tbad=%zu\n", label.c_str(), v[2],
                getenv("HASHLINE_SPK") ? getenv("HASHLINE_SPK") : "", getenv("AR_LEAF") ? getenv("AR_LEAF") : "", getenv("AR_TAU") ? getenv("AR_TAU") : "", int(sorted_output), kind, L,
                el * 1e9 / double(scans), el * 1e9 / double(keys ? keys : 1), scans, double(keys) / double(scans), point_ns, hit, double(idx->memory_consumption()) / double(n), build_s, checked, bad);
    std::fflush(stdout);
  };
  for (size_t L : Ls) timed(stored, "stored", L);
  timed(absent, "absent", 16);
  timed(absent, "absent", 256);
  g_sink = sink ^ val;
  return bad ? 1 : 0;
}
