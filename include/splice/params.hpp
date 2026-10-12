// SPLICE-H build parameters, parsed from SPLICE_ARGS ("k=v k=v ...", later tokens win).
// Every value is an integer or a fixed list, so the canonical form round-trips exactly and the same
// args file reproduces the same layout on any machine.
#pragma once
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace splice {

enum : std::uint8_t { TIER1_PLA = 0, TIER1_HISTTREE = 1 };
enum : std::uint8_t { MODE_AUTO = 0, MODE_DIRECT = 1, MODE_FENCED = 2 };

struct SpliceParams {
    std::uint32_t k1 = 16384;
    std::uint64_t eps = 0;                  // 0 = bisect to k1
    std::uint8_t tier1 = TIER1_PLA, mode = MODE_AUTO;
    std::uint8_t alpha_mask = 0x6;          // bit q set = alpha q/4 offered (q = 1..4); default 0.25, 0.5
    std::uint8_t w_mask = 0x7;              // bit 0,1,2 = window 1,2,4 lines
    std::uint8_t entry = 2, compact = 0;
    std::uint32_t cbar = 24;
    std::uint32_t cap_tenths = 220;         // bytes per key x10
    std::uint32_t l2 = 2048;                // KB, the selection arm
    std::uint8_t router_bits = 11, router_leaf = 8;
    std::uint32_t exc = 64, outer = 4;
    std::uint64_t sim = 3000000, warm = 1000000, seed = 42;
    std::uint32_t threads = 0;              // 0 = env SPLICE_BUILD_THREADS or 16
    std::uint8_t verify = 0, hash = 1, thp = 0;
    // Cost constants in milli-D (D = one serialized DRAM access).
    std::int64_t d_tau = 50, d_walk_hit = 90, d_walk_hit_thp = 150, d_router_child = 80;
    std::int64_t d_c_direct = 300, d_c_fenced = 450, d_c_exc = 200, p0_ad = 940, p0_ae = 810, p0_b = 940;
    std::vector<unsigned> alphas() const { std::vector<unsigned> a; for (unsigned q = 1; q <= 4; ++q) if (alpha_mask >> q & 1) a.push_back(q); return a; }
    std::vector<unsigned> windows() const { std::vector<unsigned> w; for (unsigned i = 0; i < 3; ++i) if (w_mask >> i & 1) w.push_back(1u << i); return w; }
};

namespace detail {
[[noreturn]] inline void bad(const std::string& why) { throw std::invalid_argument("SPLICE_ARGS: " + why); }
inline std::uint64_t parse_u(std::string_view k, std::string_view v, std::uint64_t lo, std::uint64_t hi) {
    if (v.empty() || v.size() > 19) bad(std::string(k) + "=" + std::string(v) + " is not an integer");
    std::uint64_t r = 0;
    for (char c : v) { if (c < '0' || c > '9') bad(std::string(k) + "=" + std::string(v) + " is not an integer"); r = r * 10 + std::uint64_t(c - '0'); }
    if (r < lo || r > hi) bad(std::string(k) + "=" + std::string(v) + " out of range [" + std::to_string(lo) + "," + std::to_string(hi) + "]");
    return r;
}
// "22", "22.0" or "16.5" -> tenths.
inline std::uint32_t parse_tenths(std::string_view k, std::string_view v) {
    const auto dot = v.find('.');
    std::uint64_t whole = parse_u(k, v.substr(0, dot), 1, 1000), frac = 0;
    if (dot != std::string_view::npos) { const auto f = v.substr(dot + 1); if (f.size() != 1) bad(std::string(k) + " takes one decimal"); frac = parse_u(k, f, 0, 9); }
    return std::uint32_t(whole * 10 + frac);
}
template<class F> inline void split(std::string_view s, char sep, F f) {
    std::size_t i = 0;
    while (i <= s.size()) { const auto j = std::min(s.find(sep, i), s.size()); f(s.substr(i, j - i)); i = j + 1; }
}
} // namespace detail

inline SpliceParams parse_params(std::string_view args, bool thp_from_index) {
    using namespace detail;
    SpliceParams p; p.thp = thp_from_index;
    std::size_t i = 0;
    while (i < args.size()) {
        while (i < args.size() && (args[i] == ' ' || args[i] == '\t' || args[i] == '\n' || args[i] == '\r')) ++i;
        std::size_t j = i;
        while (j < args.size() && !(args[j] == ' ' || args[j] == '\t' || args[j] == '\n' || args[j] == '\r')) ++j;
        if (j == i) break;
        const std::string_view tok = args.substr(i, j - i);
        i = j;
        const auto eq = tok.find('=');
        if (eq == std::string_view::npos || eq == 0) bad("token '" + std::string(tok) + "' is not key=value");
        const std::string_view k = tok.substr(0, eq), v = tok.substr(eq + 1);
        for (char c : v) if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == ',')) bad("bad character in '" + std::string(tok) + "'");
        auto one_of = [&](std::initializer_list<const char*> opts) -> unsigned {
            unsigned n = 0; for (auto o : opts) { if (v == o) return n; ++n; }
            bad(std::string(k) + "=" + std::string(v) + " is not allowed");
        };
        if (k == "k1") p.k1 = std::uint32_t(parse_u(k, v, 1, 65536));
        else if (k == "eps") p.eps = parse_u(k, v, 0, 1ull << 40);
        else if (k == "tier1") p.tier1 = std::uint8_t(one_of({"pla", "histtree"}));
        else if (k == "mode") p.mode = std::uint8_t(one_of({"auto", "direct", "fenced"}));
        else if (k == "alpha") {
            std::uint8_t m = 0;
            split(v, ',', [&](std::string_view a) {
                if (a == "0.25") m |= 2; else if (a == "0.5") m |= 4; else if (a == "0.75") m |= 8; else if (a == "1") m |= 16;
                else bad("alpha value '" + std::string(a) + "' not in 0.25,0.5,0.75,1");
            });
            p.alpha_mask = m;
        } else if (k == "w") {
            std::uint8_t m = 0;
            split(v, ',', [&](std::string_view a) {
                if (a == "1") m |= 1; else if (a == "2") m |= 2; else if (a == "4") m |= 4;
                else bad("w value '" + std::string(a) + "' not in 1,2,4");
            });
            p.w_mask = m;
        } else if (k == "entry") p.entry = std::uint8_t(parse_u(k, v, 1, 2));
        else if (k == "cbar") p.cbar = std::uint32_t(parse_u(k, v, 1, 1048576));
        else if (k == "compact") p.compact = std::uint8_t(parse_u(k, v, 0, 1));
        else if (k == "cap") p.cap_tenths = parse_tenths(k, v);
        else if (k == "l2") { p.l2 = std::uint32_t(parse_u(k, v, 1024, 2048)); if (p.l2 != 1024 && p.l2 != 2048) bad("l2 must be 1024 or 2048"); }
        else if (k == "router_bits") p.router_bits = std::uint8_t(parse_u(k, v, 8, 16));
        else if (k == "router_leaf") { p.router_leaf = std::uint8_t(parse_u(k, v, 4, 8)); if (p.router_leaf != 4 && p.router_leaf != 8) bad("router_leaf must be 4 or 8"); }
        else if (k == "exc") p.exc = std::uint32_t(parse_u(k, v, 0, 64));
        else if (k == "outer") p.outer = std::uint32_t(parse_u(k, v, 1, 8));
        else if (k == "sim") p.sim = parse_u(k, v, 0, 100000000);
        else if (k == "warm") p.warm = parse_u(k, v, 0, 100000000);
        else if (k == "seed") p.seed = parse_u(k, v, 0, ~0ull >> 1);
        else if (k == "threads") p.threads = std::uint32_t(parse_u(k, v, 0, 1024));
        else if (k == "verify") p.verify = std::uint8_t(parse_u(k, v, 0, 1));
        else if (k == "hash") p.hash = std::uint8_t(parse_u(k, v, 0, 1));
        else if (k == "thp") { if (parse_u(k, v, 0, 1) != (thp_from_index ? 1u : 0u)) bad("thp=" + std::string(v) + " conflicts with the index name (splice_thp sets thp=1)"); }
        else if (k == "d_tau") p.d_tau = std::int64_t(parse_u(k, v, 0, 100000));
        else if (k == "d_walk_hit") p.d_walk_hit = std::int64_t(parse_u(k, v, 0, 100000));
        else if (k == "d_walk_hit_thp") p.d_walk_hit_thp = std::int64_t(parse_u(k, v, 0, 100000));
        else if (k == "d_router_child") p.d_router_child = std::int64_t(parse_u(k, v, 0, 100000));
        else if (k == "d_c_direct") p.d_c_direct = std::int64_t(parse_u(k, v, 0, 100000));
        else if (k == "d_c_fenced") p.d_c_fenced = std::int64_t(parse_u(k, v, 0, 100000));
        else if (k == "d_c_exc") p.d_c_exc = std::int64_t(parse_u(k, v, 0, 100000));
        else if (k == "p0_ad") p.p0_ad = std::int64_t(parse_u(k, v, 0, 1000));
        else if (k == "p0_ae") p.p0_ae = std::int64_t(parse_u(k, v, 0, 1000));
        else if (k == "p0_b") p.p0_b = std::int64_t(parse_u(k, v, 0, 1000));
        else bad("unknown key '" + std::string(k) + "'");
    }
    if (!p.alpha_mask) bad("alpha list is empty");
    if (!p.w_mask) bad("w list is empty");
    return p;
}

// Canonical one-line form, every key in a fixed order. With cell_only, the machine- and run-specific
// keys (threads, verify, hash, thp) are left out: that form is what results/splice_h/args holds.
inline std::string format_params(const SpliceParams& p, bool cell_only = false) {
    static const char* const qa[5] = {"", "0.25", "0.5", "0.75", "1"};
    std::string a, w;
    for (unsigned q = 1; q <= 4; ++q) if (p.alpha_mask >> q & 1) { if (!a.empty()) a += ','; a += qa[q]; }
    for (unsigned i = 0; i < 3; ++i) if (p.w_mask >> i & 1) { if (!w.empty()) w += ','; w += std::to_string(1u << i); }
    auto s = [](auto v) { return std::to_string(v); };
    std::string r = "k1=" + s(p.k1) + " eps=" + s(p.eps) + " tier1=" + (p.tier1 == TIER1_PLA ? "pla" : "histtree")
        + " mode=" + (p.mode == MODE_AUTO ? "auto" : p.mode == MODE_DIRECT ? "direct" : "fenced") + " alpha=" + a + " w=" + w
        + " entry=" + s(unsigned(p.entry)) + " cbar=" + s(p.cbar) + " compact=" + s(unsigned(p.compact))
        + " cap=" + s(p.cap_tenths / 10) + "." + s(p.cap_tenths % 10) + " l2=" + s(p.l2)
        + " router_bits=" + s(unsigned(p.router_bits)) + " router_leaf=" + s(unsigned(p.router_leaf)) + " exc=" + s(p.exc)
        + " outer=" + s(p.outer) + " sim=" + s(p.sim) + " warm=" + s(p.warm) + " seed=" + s(p.seed);
    if (!cell_only) r += " threads=" + s(p.threads) + " verify=" + s(unsigned(p.verify)) + " hash=" + s(unsigned(p.hash)) + " thp=" + s(unsigned(p.thp));
    r += " d_tau=" + s(p.d_tau) + " d_walk_hit=" + s(p.d_walk_hit) + " d_walk_hit_thp=" + s(p.d_walk_hit_thp)
        + " d_router_child=" + s(p.d_router_child) + " d_c_direct=" + s(p.d_c_direct) + " d_c_fenced=" + s(p.d_c_fenced)
        + " d_c_exc=" + s(p.d_c_exc) + " p0_ad=" + s(p.p0_ad) + " p0_ae=" + s(p.p0_ae) + " p0_b=" + s(p.p0_b);
    return r;
}

} // namespace splice
