// SPDX-License-Identifier: MIT
// scan on the order-keeping indexes, against a sorted array: every start key (stored, next to a stored key, between keys, below and
// above everything, either side of the peeled exceptions, around key 0 and 2^64 - 1) and length returns exactly the keys >= the start
// with their own payloads, ascending, on key sets that force single leaves, deep directories, spills and empty cells. gp4 and sg4
// (and sg4 forced to each candidate) run with AB_V in {0, 1} x AB_BUDGET in {0, 40}. Also: sg4 keeps bits on a bit field behind a far
// key, and a rule it discovers itself on rows, from the stored keys only (with tiny far keys added, the rule it writes still
// starts within one period below the first stored key); gp4 builds where rounding used to break order at an octave knot, where
// rows below the keys carried the coordinate past 2^64, and where a finer block table cuts the walks; what must throw does
// (malformed name@ settings, a bad GATE_FORCE, a forced candidate that cannot be built, bad rule files incl. S >= 2^63 and
// S + P >= 2^64, blocks closer than any block table separates), each for its own reason.
//   test_scan [KEYS...]   (GRE key files, optional: sampled starts on each as well)
#include "indexInterface.h"
#include "alex/alex.h"
#include "lipp/lipp.h"
#include "pgm/pgm.h"
#include "btree/btree.h"
#include "gret/registry.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Gated = gret::GatedGIndex<uint64_t, uint64_t, 4>;

static uint64_t payload(uint64_t k) { return k * 2654435761u + 17; }

// make_index leaves what a name@VAR=v:... sets in the environment for the process: each check puts back what its name set
class EnvRestore {
 public:
  explicit EnvRestore(const std::string& name) {
    const size_t at = name.find('@');
    if (at == std::string::npos) return;
    const std::string rest = name.substr(at + 1);
    for (size_t p = 0;;) {
      const size_t e = rest.find_first_of(":,", p);
      const std::string kv = rest.substr(p, e == std::string::npos ? std::string::npos : e - p);
      if (const size_t eq = kv.find('='); eq != std::string::npos && eq > 0) {
        const std::string var = kv.substr(0, eq);
        const char* old = std::getenv(var.c_str());
        saved_.emplace_back(var, old ? std::optional<std::string>(old) : std::nullopt);
      }
      if (e == std::string::npos) break;
      p = e + 1;
    }
  }
  ~EnvRestore() {
    for (auto it = saved_.rbegin(); it != saved_.rend(); ++it)
      if (it->second) setenv(it->first.c_str(), it->second->c_str(), 1);
      else unsetenv(it->first.c_str());
  }
  EnvRestore(const EnvRestore&) = delete;
  EnvRestore& operator=(const EnvRestore&) = delete;

 private:
  std::vector<std::pair<std::string, std::optional<std::string>>> saved_;
};

static std::vector<std::pair<uint64_t, uint64_t>> pairs(const std::vector<uint64_t>& keys) {
  std::vector<std::pair<uint64_t, uint64_t>> kv(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) kv[i] = {keys[i], payload(keys[i])};
  return kv;
}

// want: what sg4 must keep (0 identity, 1 rule, 2 bits), or -1 for anything
static size_t check(const std::vector<uint64_t>& keys, const std::string& label, const char* name, size_t max_starts, int want = -1) {
  std::vector<std::pair<uint64_t, uint64_t>> kv = pairs(keys);
  setenv("GATE_FORCE", "", 1);  // sg4 chooses for itself unless the name forces a candidate (name@GATE_FORCE=...)
  EnvRestore restore(name);
  indexInterface<uint64_t, uint64_t>* idx = nullptr;
  Param p(1, 0);
  try {
    idx = gret::make_index<uint64_t, uint64_t>(name);
    if (!idx) { std::printf("%s: not registered\n", name); return 1; }
    idx->init(&p);
    idx->bulk_load(kv.data(), kv.size(), &p);
  } catch (const std::exception& e) {
    std::printf("FAIL %s on %s: threw %s\n", name, label.c_str(), e.what());
    delete idx;
    return 1;
  }
  if (want >= 0) {
    const Gated* gate = dynamic_cast<const Gated*>(idx);
    if (!gate || gate->chosen() != want) {
      std::printf("FAIL %s on %s: sg4 kept %d, not %d\n", name, label.c_str(), gate ? gate->chosen() : -1, want);
      delete idx;
      return 1;
    }
  }
  const uint64_t M = ~uint64_t(0);
  std::vector<uint64_t> starts = {0, 1, M, M - 1, uint64_t(1) << 63};
  std::mt19937_64 g(11);
  auto around = [&](uint64_t k) { starts.push_back(k); if (k) starts.push_back(k - 1); if (k != M) starts.push_back(k + 1); };
  // both ends (where the exceptions are), the middle, and random stored keys and gaps
  for (size_t i = 0; i < std::min<size_t>(80, keys.size()); ++i) { around(keys[i]); around(keys[keys.size() - 1 - i]); }
  for (size_t i = 0; i < max_starts && !keys.empty(); ++i) {
    const size_t r = g() % keys.size();
    around(keys[r]);
    if (r + 1 < keys.size() && keys[r + 1] - keys[r] > 2) starts.push_back(keys[r] + 1 + g() % (keys[r + 1] - keys[r] - 1));
  }
  {  // point lookups too: the ends, where the peeled keys are, and a sample through the middle
    const size_t step = std::max<size_t>(1, keys.size() / 20000);
    size_t lost = 0;
    uint64_t v = 0;
    for (size_t i = 0; i < keys.size(); ++i)
      if (i < 300 || i + 300 >= keys.size() || i % step == 0) lost += !(idx->get(keys[i], v, &p) && v == payload(keys[i]));
    if (lost) std::printf("FAIL %s on %s: %zu stored keys not found\n", name, label.c_str(), lost);
    if (lost) { delete idx; return lost; }
  }
  const size_t lens[] = {1, 2, 3, 16, 256, 4096, 65536, keys.size() + 5};
  std::vector<std::pair<uint64_t, uint64_t>> out;
  size_t bad = 0;
  for (uint64_t x : starts)
    for (size_t len : lens) {
      if (len > 4096 && (g() & 63)) continue;  // long scans on a sample of the starts
      if (len > 65536 && keys.size() > 1000000) continue;
      out.assign(len, {0, 0});
      const size_t got = idx->scan(x, len, out.data(), &p);
      const size_t at = size_t(std::lower_bound(keys.begin(), keys.end(), x) - keys.begin());
      const size_t want_n = std::min(len, keys.size() - at);
      bool ok = got == want_n;
      for (size_t i = 0; ok && i < got; ++i) ok = out[i].first == keys[at + i] && out[i].second == payload(keys[at + i]);
      if (!ok) {
        if (++bad <= 3) std::printf("FAIL %s on %s: start %llu len %zu: got %zu, want %zu\n", name, label.c_str(), (unsigned long long)x, len, got, want_n);
      }
    }
  delete idx;
  return bad;
}

// a name or a setting that must not build: make_index or bulk_load throws (with must_say in its message, if given)
static size_t check_throws(const std::vector<uint64_t>& keys, const std::string& label, const char* name, const char* must_say = nullptr) {
  std::vector<std::pair<uint64_t, uint64_t>> kv = pairs(keys);
  setenv("GATE_FORCE", "", 1);
  EnvRestore restore(name);
  indexInterface<uint64_t, uint64_t>* idx = nullptr;
  try {
    idx = gret::make_index<uint64_t, uint64_t>(name);
    if (idx) {
      Param p(1, 0);
      idx->init(&p);
      idx->bulk_load(kv.data(), kv.size(), &p);
    }
  } catch (const std::exception& e) {
    delete idx;
    if (must_say && !std::strstr(e.what(), must_say)) {
      std::printf("FAIL %s on %s: threw '%s', not about '%s'\n", name, label.c_str(), e.what(), must_say);
      return 1;
    }
    return 0;
  }
  std::printf("FAIL %s on %s: %s, but should have thrown\n", name, label.c_str(), idx ? "built" : "not registered");
  delete idx;
  return 1;
}

// whether sg4 has a bits candidate: inner constant bits in the bulk that AdaptiveLineIndex keeps (TRANSFORM_G_E unset: 64 a side)
static bool bits_buildable(const std::vector<uint64_t>& keys) {
  const std::vector<std::pair<uint64_t, uint64_t>> kv = pairs(keys);
  size_t lo, hi;
  gret::peel_outliers(kv.data(), kv.size(), 64, lo, hi);
  return gret::BitCoord::has_inner(gret::BitCoord::varying(kv.data() + lo, hi - lo));
}

static bool adaptive(const std::string& nm) {  // AdaptiveLineIndex underneath: AB_V and AB_BUDGET apply
  return nm == "ab4" || nm.compare(0, 3, "gp4") == 0 || nm.compare(0, 3, "gb4") == 0 || nm.compare(0, 3, "sg4") == 0;
}

static void write_file(const std::string& path, const std::string& text) {
  FILE* o = std::fopen(path.c_str(), "w");
  if (!o) { std::printf("cannot write %s\n", path.c_str()); std::exit(2); }
  std::fputs(text.c_str(), o);
  std::fclose(o);
}

int main(int argc, char** argv) {
  const uint64_t M = ~uint64_t(0);
  unsetenv("RULE_FILE"), unsetenv("RULE_S"), unsetenv("GATE_DISCOVER"), unsetenv("TRANSFORM_G_E");
  std::mt19937_64 g(7);
  auto rnd = [&](uint64_t lo, uint64_t hi, size_t m) { std::vector<uint64_t> v; std::uniform_int_distribution<uint64_t> u(lo, hi); for (size_t i = 0; i < m; ++i) v.push_back(u(g)); return v; };
  std::vector<std::pair<std::string, std::vector<uint64_t>>> sets = {
      {"n1", {5}}, {"n1max", {M}}, {"n1zero", {0}}, {"n2", {0, M}}, {"n3", {1, 2, M}}, {"n5", {0, 1, 2, 3, M}}, {"n17", rnd(0, uint64_t(1) << 40, 17)},
      {"uniform", rnd(1, uint64_t(1) << 50, 20000)}, {"wide", rnd(0, M, 20000)}};
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 50000; ++i) v.push_back(1000 + i); sets.push_back({"run", v}); }
  { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + (uint64_t(1) << 34), 20000); for (uint64_t j = 0; j < 5; ++j) v.push_back((uint64_t(1) << 62) + j * 1000003); v.push_back(3); v.push_back(M); sets.push_back({"outliers", v}); }
  { std::vector<uint64_t> v = rnd(uint64_t(1) << 40, (uint64_t(1) << 40) + 100000, 30000); for (uint64_t j = 0; j < 100; ++j) v.push_back(M - j * 1000); sets.push_back({"over64outliers", v}); }
  { std::vector<uint64_t> v; for (int c = 0; c < 300; ++c) { const uint64_t b = g() >> 1; for (uint64_t i = 0; i < 100; ++i) v.push_back(b + i); } sets.push_back({"clusters", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 20000; ++i) v.push_back(i * 65536 + 12345); sets.push_back({"stride", v}); }
  { std::vector<uint64_t> v; for (uint64_t i = 0; i < 10000; ++i) { v.push_back(i); v.push_back(M - i); } sets.push_back({"hugegap", v}); }
  setenv("GAP_K", "16", 1);  // gf4: the 16 widest gaps shrunk; the others ignore it
  {  // bits 16-31 never vary; then the same behind 2^64 - 1, which is peeled: sg4 must find the bits in the bulk, not in all keys
    std::vector<uint64_t> v;
    for (uint64_t a = 0; a < 300; ++a) for (uint64_t b = 0; b < 200; b += 1 + (a & 3)) v.push_back((a << 32) | (b << 8) | 7);
    sets.push_back({"bitfield", v});
    v.push_back(M);
    sets.push_back({"bitfield_max", v});
  }
  const char* names[] = {"rl4", "ab4", "ph4", "gr4", "gf4", "gp4", "gb4", "sg4", "sg4@GATE_FORCE=id", "sg4@GATE_FORCE=bits", "spgm16", "spgm256"};
  const char* leaves[] = {"4", "64", "1024"};
  size_t bad = 0, cases = 0;
  for (auto& [label, keys] : sets) {
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const bool bits = bits_buildable(keys);
    for (const char* nm : names)
      for (const char* lf : leaves)
        for (int v = 0; v < (adaptive(nm) ? 4 : 1); ++v) {  // AB_V in {0, 1} x AB_BUDGET in {0, 40}: lines fitted, and on a byte budget
          setenv("AR_LEAF", lf, 1);
          setenv("AR_TAU", "1.01", 1);
          setenv("AB_V", v & 1 ? "1" : "0", 1);
          setenv("AB_BUDGET", v & 2 ? "40" : "0", 1);
          const std::string n = nm;
          size_t e;
          if (n == "sg4@GATE_FORCE=bits" && !bits) e = check_throws(keys, label, nm);  // no bits candidate to force
          else if (n == "gp4" && (v & 2) && keys.size() < 64) e = check_throws(keys, label, nm);  // 40 bytes a key: not even gp4's phase tables
          // the bits behind 2^64 - 1, at full slots (on a 40-byte budget identity reads one line at one level, and keeps it rightly)
          else if (n == "sg4" && label == "bitfield_max" && v == 0) e = check(keys, label, nm, 2000, 2);
          else e = check(keys, label, nm, 2000);
          ++cases;
          bad += e != 0;
        }
    setenv("AB_V", "0", 1);
    setenv("AB_BUDGET", "0", 1);
  }
  const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/test_scan_rule.txt";
  {  // gr4, gf4, gp4 and sg4 with a period rule that fits the keys, one that does not, periods at the edges of what the reciprocal
     // handles, blocks over the keys (one, several, many, closer than the finest block table, wider than an exact reciprocal)
    auto run_rule = [&](const std::string& label, const std::string& text, std::vector<uint64_t> keys, bool budget) {
      std::sort(keys.begin(), keys.end());
      keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
      write_file(path, text);
      setenv("RULE_FILE", path.c_str(), 1);
      setenv("AR_LEAF", "16", 1);
      setenv("AR_TAU", "1.5", 1);
      bad += check(keys, "rule_" + label, "gr4", 1000) != 0;
      bad += check(keys, "rule_gaps_" + label, "gf4", 1000) != 0;  // the rule, then the widest gaps of its coordinate
      cases += 2;
      for (int v = 0; v < (budget ? 4 : 2); ++v) {
        setenv("AB_V", v & 1 ? "1" : "0", 1);
        setenv("AB_BUDGET", v & 2 ? "40" : "0", 1);
        const std::string at = std::string(v & 1 ? "_fit" : "") + (v & 2 ? "_budget_" : "_") + label;
        bad += check(keys, "rule_fast" + at, "gp4", 1000) != 0;                                // the same rule, in fewer instructions
        bad += check(keys, "rule_fast_at" + at, "gp4@AR_LEAF=4:AR_TAU=2", 1000) != 0;          // and the name@settings form
        bad += check(keys, "rule_gate" + at, "sg4", 1000) != 0;                                // the gate's own choice
        bad += check(keys, "rule_gate_forced" + at, "sg4@GATE_FORCE=rule", 1000) != 0;         // the rule, forced
        bad += check(keys, "rule_gate_id" + at, "sg4@GATE_FORCE=id", 1000) != 0;               // identity, forced
        cases += 5;
      }
      setenv("AB_V", "0", 1);
      setenv("AB_BUDGET", "0", 1);
      setenv("RULE_FILE", "", 1);
    };
    struct Rule { const char* label; std::string text; uint64_t period; };
    std::string many = "S 67108864\n1000 100000\n";
    for (uint64_t b = 1; b < 40; ++b) many += std::to_string(1000 + b * 5000000 + b * 777) + " " + std::to_string(100000 - 13 * b) + "\n";
    const Rule rules[] = {{"fits", "S 67108864\n1000 100000\n", 100000},
                          {"wrong_period", "S 67108864\n1000 77777\n", 100000},
                          {"period_1", "0 1\n", 100000},
                          {"period_2_40", "S 1048576\n0 1099511627776\n", uint64_t(1) << 40},
                          {"wide_period", "S 1048576\n0 1099511627775\n", uint64_t(1) << 40},  // 2^40 - 1 over 2^51: rounded down
                          {"wide_two", "S 1048576\n0 1099511627775\n1099511627776000 1099511627775\n", uint64_t(1) << 40},
                          {"two_blocks", "S 67108864\n1000 100000\n2000000000 77777\n", 100000},
                          {"split_blocks", "S 67108864\n1000 100000\n100001000 100000\n150001000 77777\n", 100000},
                          {"many_blocks", many, 100000},
                          // boundaries 100 apart in a span of 2 10^8: no 2^16-bucket table separates them
                          {"close_blocks", "S 67108864\n1000 100000\n50001100 100000\n50001200 100000\n50001300 99999\n", 100000},
                          // blocks that start past every stored key (as in the S = 0 identity rules): gp4's block table must ignore them
                          {"blocks_past_keys", "S 67108864\n1000 100000\n9000000000000 77777\n9000000100000 1000\n", 100000},
                          // S + P = 2^44, so row 2^20 m lands on coordinate 0 + its offset: right inside the bulk, however far the key is
                          {"wraps", "S 17592184995840\n1000 1048576\n", 1048576}};
    for (const Rule& r : rules) {
      std::vector<uint64_t> keys;
      for (uint64_t q = 0; q < 2000; ++q)
        for (uint64_t j = 0; j < 20; ++j) keys.push_back(1000 + q * r.period + j * j * 3);
      if (std::string(r.label) == "wraps")  // stored, peeled as far outliers, and with coordinates that overflow into the bulk's range
        for (uint64_t m = 1; m <= 20; ++m) keys.push_back(1000 + (m << 40));
      run_rule(r.label, r.text, keys, true);
    }
    // the review's octave case: S, prescaled into gp4's tables and not into gr4's, rounds y(2^52) below y(2^52 - 1) in row 0, so
    // gp4 threw where gr4 builds. S is not a power of two; 63 keys, too few for a 40-byte budget to hold the phase tables
    const std::vector<uint64_t> octave = {
        4503599627370495, 4503599627370496, 36501187937803608, 72190839248974494, 108258254565442324, 145928104614934875,
        182027119641261497, 217474014001125452, 252323610697809425, 289591055867028092, 325821661080386164, 361720172904147486,
        399767181676704231, 434771490547648450, 471890361287938587, 506685418272354244, 544187808474777890, 578820547617889743,
        616671557497069610, 651815003324226646, 687296437642314332, 724072232154162565, 760791274008716033, 795780948132132928,
        832844360967719259, 867044227075442240, 904030835226987976, 940496507662504821, 976377363330598209, 1012972326214299179,
        1049039379787408237, 1084990837419414030, 1120451693696694253, 1155514991313446432, 1192660695014265356, 1228925894378189785,
        1264558137035329987, 1300382463088726173, 1337492887935078736, 1373420656471396355, 1407416723238267773, 1445526301104884363,
        1480698836329488425, 1516010712302020681, 1553739793755077197, 1589496140418667415, 1624096657591782410, 1661204011160129062,
        1700446404696757906, 1736204634439209342, 1771510850831083921, 1806565899135707712, 1843973574667172389, 1881097734051705850,
        1917857200620611099, 1952390540606886057, 1987290068280696810, 2022244776650510125, 2062140708278313166, 2098117422647036020,
        2138660456052018957, 2179281093782863877, 2207553085784652493};
    run_rule("octave", "S 242992742610752\n0 36028797018963968\n", octave, false);
  }
  {  // rule files load() refuses: gp4 throws, gr4 builds or not as before, and sg4 builds without the rule
    std::vector<uint64_t> keys;
    for (uint64_t i = 0; i < 5000; ++i) keys.push_back(1000 + i * 7);
    struct BadRule { const char* text; const char* says; };
    const BadRule bad_rules[] = {{"S 100\n1000 0\n", "period 0"},                       // a division by zero before
                                 {"S 100\n500 10\n500 20\n", "strictly increasing"},     // first keys
                                 {"S 100\n500 10\nfirst period\n", "is not '<first key> <period>'"},
                                 {"S 100\n2000 10\n", "does not cover"},                 // the first block starts above the first key
                                 {"S 9223372036854775808\n0 1000\n", "below 2^63"},      // S = 2^63: the guard keeps S F well below 2^64
                                 {"S 1\n0 18446744073709551615\n", "overflows"}};        // S + P = 2^64: the row width wraps
    setenv("AR_LEAF", "64", 1);
    setenv("AR_TAU", "1.5", 1);
    for (const BadRule& br : bad_rules) {
      write_file(path, br.text);
      setenv("RULE_FILE", path.c_str(), 1);
      bad += check_throws(keys, "bad_rule", "gp4", br.says) != 0;
      bad += check(keys, "bad_rule", "sg4", 200, 0) != 0;  // the rule rejected, identity kept
      bad += check_throws(keys, "bad_rule", "sg4@GATE_FORCE=rule") != 0;
      cases += 3;
    }
    setenv("RULE_FILE", "", 1);
    std::remove(path.c_str());
    // names: no '=', no variable, a second '@', nothing before or after '@', empty entries; GATE_FORCE neither id, rule nor bits;
    // a forced rule with no rule (no RULE_FILE, too few keys to discover one) or forced bits with no inner constant bits
    const char* broken[] = {"sg4@GATE_FORCE", "sg4@=x", "gp4@AR_LEAF=8@AR_TAU=3", "@GATE_FORCE=id", "sg4@", "sg4@AR_LEAF=4:",
                            "sg4@AR_LEAF=4::AR_TAU=2", "sg4@GATE_FORCE=Rule", "sg4@GATE_FORCE=rule", "sg4@GATE_FORCE=bits"};
    for (const char* nm : broken) bad += check_throws(keys, "broken", nm) != 0, ++cases;
    unsetenv("GRET_TEST_SCAN_VAR");  // a malformed name sets nothing, not even its well-formed entries
    try {
      gret::make_index<uint64_t, uint64_t>("sg4@GRET_TEST_SCAN_VAR=1:oops");
      std::printf("FAIL sg4@GRET_TEST_SCAN_VAR=1:oops: no exception\n"), ++bad;
    } catch (const std::invalid_argument&) {
    }
    if (std::getenv("GRET_TEST_SCAN_VAR")) std::printf("FAIL sg4@GRET_TEST_SCAN_VAR=1:oops set GRET_TEST_SCAN_VAR\n"), ++bad;
    ++cases;
  }
  {  // blocks closer than any block table separates: 2000 boundaries 25 keys apart in a dense run, and far keys in the bulk (136,
     // none peeled: their span is under twice the rest's) make every bucket at least 2^43 wide, so the stored keys would walk past
     // hundreds of blocks each in general(): gp4 refuses the rule for that, and sg4 cannot keep it (it builds, with identity or bits)
    std::vector<uint64_t> keys;
    for (uint64_t i = 0; i < 50000; ++i) keys.push_back(1000000 + i);
    for (uint64_t i = 1; i <= 136; ++i) keys.push_back(i << 52);
    std::string text = "S 0\n";
    for (uint64_t j = 0; j < 2000; ++j) text += std::to_string(1000000 + 25 * j) + " 1000\n";
    write_file(path, text);
    setenv("RULE_FILE", path.c_str(), 1);
    setenv("AR_LEAF", "64", 1);
    setenv("AR_TAU", "1.5", 1);
    bad += check_throws(keys, "close_walks", "gp4", "walk steps") != 0;
    bad += check_throws(keys, "close_walks", "sg4@GATE_FORCE=rule") != 0;
    bad += check(keys, "close_walks", "sg4", 200) != 0;
    cases += 3;
    setenv("RULE_FILE", "", 1);
    std::remove(path.c_str());
  }
  {  // rules gp4 refused needlessly before: (1) every 2^12 bucket holds at most one boundary but half the stored keys walk a step,
     // and 2^13 buckets put the boundaries on bucket starts, so none walks; (2) a wide rule whose stored keys start 270842033029
     // rows above its first block, where counting rows from the first block carried the coordinate past 2^64 among them
    std::string text = "S 67108864\n0 1000\n";
    for (uint64_t b = 1; b <= 1000; ++b) text += std::to_string(b * 16384 + 2048) + " " + std::to_string(1000 + b) + "\n";
    std::vector<uint64_t> keys;
    for (uint64_t x = 0; x < (uint64_t(1) << 24); x += 37) keys.push_back(x);
    keys.push_back((uint64_t(1) << 24) - 1);
    write_file(path, text);
    setenv("RULE_FILE", path.c_str(), 1);
    setenv("AR_LEAF", "64", 1);
    setenv("AR_TAU", "1.5", 1);
    bad += check(keys, "refine_on_walks", "gp4", 200) != 0;
    const uint64_t P = 1000003, a = 270842033029ull * P;
    keys.clear();
    for (uint64_t q = 0; q < 2000; ++q)
      for (uint64_t j = 0; j < 20; ++j) keys.push_back(a + q * P + j * j * 3 % P);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    write_file(path, "S 67108864\n0 1000003\n");
    bad += check(keys, "rows_from_lo", "gp4", 200) != 0;
    bad += check(keys, "rows_from_lo", "sg4@GATE_FORCE=rule", 200) != 0;
    // (3) S + P = 2^64 - 1 and stored keys from 2^63 to 2^64 - 1, the top one in row 1: with y(lo) near 2^63 its raw y passed 2^64,
    // although y - y(lo) increases; load() now makes y(lo) = 0
    keys.clear();
    for (uint64_t i = 0; i <= 3000; ++i) keys.push_back((uint64_t(1) << 63) + i * 3074457345617000ull);
    keys.push_back(~uint64_t(0));
    write_file(path, "S 1048576\n0 18446744073708503039\n");
    bad += check(keys, "y_lo_zero", "gp4", 200) != 0;
    cases += 4;
    setenv("RULE_FILE", "", 1);
    std::remove(path.c_str());
  }
  {  // sg4 discovers a rule itself: gsyn.cpp's 'rows' (keys in the first tenth of rows 100000 wide, 100 keys a row), RULE_FILE unset,
     // at a size discover_rule accepts (128 windows of 4096 keys at least)
    std::mt19937_64 r(9);
    const uint64_t n = 700000, P = 100000;
    std::vector<uint64_t> keys;
    for (uint64_t i = 0; i < n; ++i) keys.push_back((uint64_t(1) << 30) + r() % (n / 100) * P + r() % (P / 10));
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    setenv("RULE_FILE", "", 1);
    setenv("AR_LEAF", "64", 1);
    setenv("AR_TAU", "1.5", 1);
    bad += check(keys, "rows_discovered", "sg4", 300, 1) != 0;                  // discovered, and kept
    bad += check(keys, "rows_no_discovery", "sg4@GATE_DISCOVER=0", 300, 0) != 0;  // without discovery: identity
    cases += 2;
    // discovery reads the bulk only: with ten tiny keys added (peeled), the rule the gate writes ($GATE_RULE_OUT) still starts
    // within one period below the first stored key, not down at the tiny keys (nor at 0, where the rows would lose their phase)
    const std::string tmp = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp", pid = "." + std::to_string(long(::getpid()));
    auto slurp = [](const std::string& f) {
      std::string s;
      if (FILE* h = std::fopen(f.c_str(), "rb")) {
        char buf[4096];
        for (size_t got; (got = std::fread(buf, 1, sizeof buf, h)) > 0;) s.append(buf, got);
        std::fclose(h);
      }
      return s;
    };
    std::vector<uint64_t> tiny = keys;
    for (uint64_t i = 1; i <= 10; ++i) tiny.push_back(i);
    std::sort(tiny.begin(), tiny.end());
    const std::string a = tmp + "/test_scan_rule_a", b = tmp + "/test_scan_rule_b";
    bad += check(keys, "rows_rule_out", ("sg4@GATE_RULE_OUT=" + a).c_str(), 100, 1) != 0;
    bad += check(tiny, "rows_tiny_keys", ("sg4@GATE_RULE_OUT=" + b).c_str(), 100, 1) != 0;
    const std::string ra = slurp(a + pid), rb = slurp(b + pid);
    if (ra.compare(0, 2, "S ") != 0) std::printf("FAIL rows_rule_out: no rule file %s%s\n", a.c_str(), pid.c_str()), ++bad;
    unsigned long long s = 0, f = 0, per = 0;
    if (std::sscanf(rb.c_str(), "S %llu %llu %llu", &s, &f, &per) != 3 || per == 0 || f > keys.front() || f + per <= keys.front())
      std::printf("FAIL rows_tiny_keys: rule file '%s' does not start within a period below the first stored key %llu\n",
                  rb.substr(0, 60).c_str(), (unsigned long long)keys.front()), ++bad;
    std::remove((a + pid).c_str()), std::remove((b + pid).c_str());
    cases += 3;
  }
  for (int a = 1; a < argc; ++a) {
    FILE* f = std::fopen(argv[a], "rb");
    uint64_t n = 0;
    if (!f || std::fread(&n, 8, 1, f) != 1) return 2;
    std::vector<uint64_t> k(n);
    if (std::fread(k.data(), 8, n, f) != n) return 2;
    std::fclose(f);
    std::sort(k.begin(), k.end());
    k.erase(std::unique(k.begin(), k.end()), k.end());
    const bool bits = bits_buildable(k);
    for (const char* nm : names) {
      setenv("AR_LEAF", std::string(nm) == "ph4" ? "1024" : "16", 1);
      setenv("AR_TAU", "2", 1);
      const size_t e = std::string(nm) == "sg4@GATE_FORCE=bits" && !bits ? check_throws(k, argv[a], nm) : check(k, argv[a], nm, 3000);
      ++cases;
      bad += e != 0;
    }
  }
  std::printf("test_scan: %s (%zu cases, %zu failed)\n", bad ? "FAILED" : "ok", cases, bad);
  return bad ? 1 : 0;
}
