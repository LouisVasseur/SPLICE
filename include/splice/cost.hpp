// SPLICE-H cost model and layout planner (C++20). Shared by build.hpp (the GRE build) and
// tools/splice_count.cpp (the offline grid), so the counts reported offline and the layout GRE builds
// come from the same code. Everything that decides the layout is integer: costs are milli-D
// (D = one serialized DRAM access), so arm64 and x86 builds choose bit-identical layouts.
#pragma once
#include "splice/layout.hpp"
#include "splice/params.hpp"
#include "scaleli/hardness.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace splice {

using u128 = unsigned __int128;
inline unsigned bitlen(std::uint64_t v) noexcept { return v ? 64u - unsigned(__builtin_clzll(v)) : 0u; }
inline std::uint64_t mix64(std::uint64_t x) noexcept {   // splitmix64 finalizer (same constants as scaleli::mix64)
    x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; return x ^ (x >> 31);
}
inline long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline unsigned resolve_threads(unsigned t) {
    if (t) return t;
    if (const char* e = std::getenv("SPLICE_BUILD_THREADS")) {
        unsigned v = 0; const char* p = e;
        for (; *p >= '0' && *p <= '9' && v < 100000; ++p) v = v * 10 + unsigned(*p - '0');
        if (*p || p == e || v == 0 || v > 1024) throw std::invalid_argument("SPLICE_BUILD_THREADS must be an integer in [1,1024]");
        return v;
    }
    return 16;
}
// Runs f(i) for i in [0, n) on `threads` workers; the first exception is rethrown here.
template<class F> void par_for(std::size_t n, unsigned threads, F&& f) {
    if (threads <= 1 || n <= 1) { for (std::size_t i = 0; i < n; ++i) f(i); return; }
    std::atomic<std::size_t> next{0};
    std::exception_ptr err; std::mutex mu;
    std::vector<std::thread> pool;
    const unsigned t = unsigned(std::min<std::size_t>(threads, n));
    for (unsigned k = 0; k < t; ++k) pool.emplace_back([&] {
        try { for (std::size_t i; (i = next++) < n;) f(i); }
        catch (...) { std::lock_guard<std::mutex> g(mu); if (!err) err = std::current_exception(); next = n; }
    });
    for (auto& th : pool) th.join();
    if (err) std::rethrow_exception(err);
}
// 64/128-B aligned owning buffer (SSE loads in the lookup are aligned).
template<class T> struct ABuf {
    T* p = nullptr; std::size_t n = 0;
    ABuf() = default;
    explicit ABuf(std::size_t count) { reset(count); }
    ABuf(const ABuf&) = delete; ABuf& operator=(const ABuf&) = delete;
    ABuf(ABuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    ABuf& operator=(ABuf&& o) noexcept { if (this != &o) { release(); p = o.p; n = o.n; o.p = nullptr; o.n = 0; } return *this; }
    ~ABuf() { release(); }
    void release() { if (p) ::operator delete(p, std::align_val_t(128)); p = nullptr; n = 0; }
    void reset(std::size_t count) {
        release();
        if (!count) return;
        p = static_cast<T*>(::operator new(count * sizeof(T), std::align_val_t(128)));
        std::memset(static_cast<void*>(p), 0, count * sizeof(T)); n = count;
    }
    T* data() const noexcept { return p; }
    std::size_t size() const noexcept { return n; }
    T& operator[](std::size_t i) const noexcept { return p[i]; }
};

// Caller's keys: key i at k[i*stride], payload at v[i*stride] (or the key itself when v is null).
struct KeySrc {
    const std::uint64_t* k = nullptr; const std::uint64_t* v = nullptr; std::size_t stride = 1, n = 0;
    std::uint64_t key(std::size_t i) const noexcept { return k[i * stride]; }
    std::uint64_t val(std::size_t i) const noexcept { return v ? v[i * stride] : k[i * stride]; }
    std::size_t upper_bound(std::size_t a, std::size_t b, std::uint64_t x) const noexcept {
        while (a < b) { const std::size_t m = a + (b - a) / 2; if (key(m) <= x) a = m + 1; else b = m; }
        return a;
    }
};

// ---------------------------------------------------------------- G: exceptions at the ends
// Peel up to E keys per end so the remaining span is within 2x of the E/E-trimmed span (fb: 21 keys).
inline void choose_exceptions(const KeySrc& ks, std::size_t E, std::size_t& bb, std::size_t& bt) {
    const std::size_t n = ks.n; bb = bt = 0;
    if (E == 0 || n < 2 * E + 2) return;
    const u128 best2 = u128(ks.key(n - 1 - E) - ks.key(E)) * 2;
    for (bt = 0; bt <= E; ++bt) if (u128(ks.key(n - 1 - bt) - ks.key(E)) <= best2) break;
    for (bb = 0; bb <= E; ++bb) if (u128(ks.key(n - 1 - bt) - ks.key(bb)) <= best2) break;
}

// ---------------------------------------------------------------- T: tier-1 knots
// Segment starts of the optimal eps-PLA over (key, local rank), through the repo's OptimalPLA port
// unchanged: a false add_point closes the segment and the point starts the next one.
inline std::size_t pla_starts(const KeySrc& ks, std::size_t a, std::size_t b, std::uint64_t eps, std::vector<std::size_t>* out) {
    scaleli::hardness::OptimalPLA<scaleli::hardness::U128> opt(static_cast<std::size_t>(eps));
    std::size_t segs = 0;
    if (out) out->clear();
    for (std::size_t i = a; i < b; ++i) {
        const auto x = static_cast<scaleli::hardness::U128>(ks.key(i));
        const bool fits = opt.add_point(x, i - a);
        if (!fits) opt.add_point(x, i - a);   // x starts the next segment
        if (!fits || i == a) { ++segs; if (out) out->push_back(i); }
    }
    return segs;
}
// Count-split dyadic trie (Hist-Tree style), the model-free ablation: knots are the first keys of the
// nonempty leaves. Returns the leaf count; starts are appended when out is given.
inline std::size_t histtree_rec(const KeySrc& ks, std::size_t a, std::size_t b, std::uint64_t blo, unsigned shift, unsigned depth,
                                std::uint64_t T, std::vector<std::size_t>* out) {
    std::size_t leaves = 0, i = a;
    while (i < b) {
        const std::uint64_t bk = (ks.key(i) - blo) >> shift;
        const u128 st = u128(blo) + (u128(bk) << shift), en128 = st + (u128(1) << shift) - 1;
        const std::uint64_t en = en128 > u128(~0ull) ? ~0ull : std::uint64_t(en128);
        const std::size_t j = ks.upper_bound(i, b, en), c = j - i;
        if (c > T && shift > 0 && depth < 4) {
            unsigned need = std::min<unsigned>(12, bitlen((c + T - 1) / T) + 1);
            need = std::min(need, shift);
            leaves += histtree_rec(ks, i, j, std::uint64_t(st), shift - need, depth + 1, T, out);
        } else { ++leaves; if (out) out->push_back(i); }
        i = j;
    }
    return leaves;
}
inline std::size_t histtree_starts(const KeySrc& ks, std::size_t a, std::size_t b, std::uint64_t T, std::vector<std::size_t>* out) {
    if (out) out->clear();
    const std::uint64_t lo = ks.key(a), span = ks.key(b - 1) - lo;
    const unsigned bl = bitlen(span), sh0 = bl > 11 ? bl - 11 : 0;
    return histtree_rec(ks, a, b, lo, sh0, 0, T, out);
}
// Integer bisection shared by both tier-1 families: the smallest parameter (within lo/32) whose
// segment count is <= K1. Larger parameters give fewer segments.
template<class F> std::uint64_t bisect_param(std::uint64_t first, std::uint64_t k1, F segs, std::vector<std::size_t>& best, std::uint32_t& passes) {
    std::vector<std::size_t> tmp;
    auto run = [&](std::uint64_t e) { ++passes; return segs(e, &tmp); };
    if (run(first) <= k1) { best.swap(tmp); return first; }
    std::uint64_t lo = first, hi = first * 4;
    while (run(hi) > k1 && hi < (1ull << 40)) { lo = hi; hi *= 4; }
    best.swap(tmp);
    while (hi > lo + std::max<std::uint64_t>(1, lo / 32)) {
        const std::uint64_t mid = lo + (hi - lo) / 2;
        if (run(mid) <= k1) { hi = mid; best.swap(tmp); } else lo = mid;
    }
    return hi;
}

// ---------------------------------------------------------------- Fit: exceptions + tier-1 + router
struct Fit {
    KeySrc ks;
    std::size_t n = 0, bb = 0, bt = 0;
    std::uint64_t lo = 0, span = 0;
    std::uint8_t tier1 = TIER1_PLA;
    std::uint64_t eps = 0;                 // PLA eps, or the Hist-Tree split threshold T
    std::uint32_t k1 = 0, passes = 0;
    long long fit_ns = 0, router_ns = 0;
    std::vector<std::size_t> start;        // K+1 absolute key indices; start[K] = n - bt
    std::vector<std::uint8_t> pre;         // per segment
    std::vector<std::uint64_t> sp;         // per segment: (last - knot) >> pre
    ABuf<Exc> exc;
    std::vector<std::uint32_t> rtop, rsub;
    std::vector<RNode> rnodes;
    std::uint8_t rshift = 0, rbits = 11, rleaf = 8;
    ABuf<Seg16> rkeys;                     // K + 9 records holding only the knot keys: routing before the fill
    std::vector<std::uint64_t> seg_rchild, seg_rlines;
    std::uint64_t rchild_total = 0, rlines_total = 0, router_depth_hist[5] = {};
    std::size_t K() const noexcept { return start.size() - 1; }
    std::size_t nseg_keys(std::size_t j) const noexcept { return start[j + 1] - start[j]; }
    std::uint64_t knot(std::size_t j) const noexcept { return ks.key(start[j]); }
    View rview() const noexcept {
        View v{}; v.lo = lo; v.span = span; v.rtop = rtop.data(); v.rnodes = rnodes.data(); v.rsub = rsub.data();
        v.seg = rkeys.data(); v.nseg = std::uint32_t(K()); v.rshift = rshift; v.exc = exc.data(); v.nexc = std::uint32_t(exc.size());
        return v;
    }
    std::uint64_t router_bytes() const noexcept { return 4 * rtop.size() + 16 * rnodes.size() + 4 * rsub.size(); }
    // Bytes outside the arena: records, router, exceptions.
    std::uint64_t meta_bytes() const noexcept { return (K() + 9) * 16 + router_bytes() + 16 * exc.size(); }
};
// Router over the knots (Hist-Tree count split): a top table of 2^R buckets; a bucket with more than
// L knots gets a child table with its own shift. Leaves name the last knot <= bucket start and how many
// knots lie inside the bucket, so the lookup ends in one masked 8-record scan.
inline void build_router(Fit& f) {
    const std::size_t K = f.K();
    std::vector<std::uint64_t> kn(K);
    for (std::size_t j = 0; j < K; ++j) kn[j] = f.knot(j);
    const unsigned R = f.rbits, L = f.rleaf, bl = bitlen(f.span);
    const unsigned sh0 = bl > R ? bl - R : 0;
    f.rshift = std::uint8_t(sh0);
    f.rtop.assign(std::size_t(1) << R, 0); f.rnodes.clear(); f.rsub.clear();
    const std::uint64_t hi = f.lo + f.span;
    auto ub = [&](std::uint64_t x) { return std::size_t(std::upper_bound(kn.begin(), kn.end(), x) - kn.begin()); };
    // Fills `count` buckets of width 2^shift from blo into table t (0 = rtop, 1 = rsub) at offset off.
    auto fill = [&](auto&& self, int t, std::size_t off, std::size_t count, std::uint64_t blo, unsigned shift) -> void {
        for (std::size_t b = 0; b < count; ++b) {
            const u128 st128 = u128(blo) + (u128(b) << shift);
            std::uint32_t e;
            if (st128 > u128(hi)) e = rleaf(std::uint32_t(K - 1), 0);   // unreachable: past the bulk range
            else {
                const std::uint64_t st = std::uint64_t(st128);
                const u128 en128 = st128 + (u128(1) << shift) - 1;
                const std::uint64_t en = en128 > u128(hi) ? hi : std::uint64_t(en128);
                const std::size_t a = ub(st), c = ub(en) - a;
                if (c <= L) e = rleaf(std::uint32_t(a - 1), std::uint32_t(c));
                else {
                    unsigned bits = std::max<unsigned>(1, bitlen((c + L - 1) / L) + 1);
                    bits = std::min({bits, 12u, shift});
                    const std::size_t ni = f.rnodes.size(), so = f.rsub.size();
                    if (ni >= 0x7FFFFFFFu || so + (std::size_t(1) << bits) > 0xFFFFFFFFu) throw std::runtime_error("router too large");
                    f.rnodes.push_back(RNode{st, std::uint32_t(so), std::uint8_t(shift - bits), std::uint8_t(bits), 0});
                    f.rsub.resize(so + (std::size_t(1) << bits), 0);
                    e = rchild_entry(std::uint32_t(ni));
                    self(self, 1, so, std::size_t(1) << bits, st, shift - bits);
                }
            }
            (t == 0 ? f.rtop : f.rsub)[off + b] = e;
        }
    };
    fill(fill, 0, 0, f.rtop.size(), f.lo, sh0);
}
// Segments of at most 2^16 keys per task keep the threads busy; chunking never changes results.
inline std::vector<std::size_t> seg_chunks(const Fit& f) {
    std::vector<std::size_t> c{0};
    const std::size_t K = f.K(), target = std::max<std::size_t>(1u << 16, (f.n - f.bb - f.bt) / 512);
    std::size_t acc = 0;
    for (std::size_t j = 0; j < K; ++j) { acc += f.nseg_keys(j); if (acc >= target && j + 1 < K) { c.push_back(j + 1); acc = 0; } }
    c.push_back(K);
    return c;
}
struct RouteCount {   // router child levels and record lines of one lookup
    static constexpr bool active = true;
    std::uint64_t rchild_n = 0, rlines = 0;
    void exc_probe(std::uint32_t) noexcept {} void exc() noexcept {} void rtab(std::uint64_t) noexcept {}
    void rchild() noexcept { ++rchild_n; }
    void recs(std::uint32_t j0, std::uint32_t j) noexcept { std::uint64_t ids[8]; rlines += record_lines(j0, j, ids); }
    void dep(const void*, int) noexcept {} void par(const void*, int) noexcept {} void level() noexcept {} void done(bool) noexcept {}
};
inline Fit make_fit(const KeySrc& ks, const SpliceParams& p, unsigned threads) {
    Fit f; f.ks = ks; f.n = ks.n; f.tier1 = p.tier1; f.k1 = p.k1; f.rbits = p.router_bits; f.rleaf = p.router_leaf;
    if (!ks.n) throw std::logic_error("make_fit needs keys");
    const long long t0 = now_ns();
    choose_exceptions(ks, p.exc, f.bb, f.bt);
    const std::size_t a = f.bb, b = f.n - f.bt;
    f.lo = ks.key(a); f.span = ks.key(b - 1) - f.lo;
    f.exc.reset(f.bb + f.bt);
    for (std::size_t i = 0, e = 0; i < f.n; ++i) if (i < a || i >= b) f.exc[e++] = Exc{ks.key(i), ks.val(i)};
    std::vector<std::size_t> st;
    if (p.tier1 == TIER1_PLA) {
        auto segs = [&](std::uint64_t e, std::vector<std::size_t>* o) { return pla_starts(ks, a, b, e, o); };
        if (p.eps) { f.eps = p.eps; f.passes = 1; segs(p.eps, &st); }
        else f.eps = bisect_param(1, p.k1, segs, st, f.passes);
    } else {
        auto segs = [&](std::uint64_t T, std::vector<std::size_t>* o) { return histtree_starts(ks, a, b, T, o); };
        if (p.eps) { f.eps = p.eps; f.passes = 1; segs(p.eps, &st); }
        else f.eps = bisect_param(16, p.k1, segs, st, f.passes);
    }
    if (st.empty() || st[0] != a) throw std::logic_error("tier-1 must start at the first bulk key");
    if (st.size() >= (1u << 27) - 16) throw std::runtime_error("too many tier-1 segments (reduce k1 or raise eps)");
    f.start = std::move(st); f.start.push_back(b);
    const std::size_t K = f.K();
    f.pre.resize(K); f.sp.resize(K);
    for (std::size_t j = 0; j < K; ++j) {
        const std::uint64_t s = ks.key(f.start[j + 1] - 1) - f.knot(j);
        const unsigned bl = bitlen(s), pr = bl > 32 ? bl - 32 : 0;
        f.pre[j] = std::uint8_t(pr); f.sp[j] = s >> pr;
    }
    f.fit_ns = now_ns() - t0;
    const long long t1 = now_ns();
    f.rkeys.reset(K + 9);
    for (std::size_t j = 0; j < K; ++j) f.rkeys[j].key = f.knot(j);
    build_router(f);
    // Router levels and record lines per stored key do not depend on the per-segment choices.
    f.seg_rchild.assign(K, 0); f.seg_rlines.assign(K, 0);
    const auto ch = seg_chunks(f);
    const View rv = f.rview();
    std::vector<std::array<std::uint64_t, 5>> dh(ch.size() - 1);
    par_for(ch.size() - 1, threads, [&](std::size_t c) {
        dh[c] = {};
        for (std::size_t j = ch[c]; j < ch[c + 1]; ++j) {
            RouteCount rc;
            for (std::size_t i = f.start[j]; i < f.start[j + 1]; ++i) {
                const std::uint64_t before = rc.rchild_n;
                if (route(rv, ks.key(i), rc) != j) throw std::logic_error("router does not reach the key's segment");
                ++dh[c][std::min<std::uint64_t>(rc.rchild_n - before, 4)];
            }
            f.seg_rchild[j] = rc.rchild_n; f.seg_rlines[j] = rc.rlines;
        }
    });
    for (std::size_t j = 0; j < K; ++j) { f.rchild_total += f.seg_rchild[j]; f.rlines_total += f.seg_rlines[j]; }
    for (auto& d : dh) for (int i = 0; i < 5; ++i) f.router_depth_hist[i] += d[std::size_t(i)];
    f.router_ns = now_ns() - t1;
    return f;
}

// ---------------------------------------------------------------- V: per-segment candidates
// The integer model of 4.5: m maps the segment's key span onto U output units (lines' slots or entries).
inline std::uint32_t slope(std::uint64_t U, std::uint64_t sp) noexcept {
    const u128 m = (u128(U) << 32) / (u128(sp) + 1);
    return m > 0xFFFFFFFFu ? 0xFFFFFFFFu : std::uint32_t(m);
}
inline std::uint64_t direct_units(std::uint64_t nj, unsigned q) noexcept { return nj + (nj * q + 3) / 4; }
inline std::uint64_t roundup(std::uint64_t x, std::uint64_t a) noexcept { return (x + a - 1) / a * a; }

// One candidate's totals for one segment. Selection-time pages use segment-local offsets.
struct Sum {
    std::uint64_t dep = 0, par = 0, pgA = 0, pgB = 0, bytes = 0, bbytes = 0;
    std::uint32_t units = 0;               // section-A units (512 B)
    std::uint64_t depth[4] = {}, ties = 0, compact_keys = 0, scan = 0;
};
// DIRECT placement: forward cumulative max f_i = max(p_i, f_{i-1} + 1); key i fills slots [c_i, f_i]
// with c_i = f_{i-1} + 1 (c_0 = 0), so a slot never precedes its prediction and a right-only scan finds it.
struct DirectPlace {
    std::uint64_t key0; unsigned pre; std::uint32_t m; std::uint64_t nextc = 0;
    // returns (p, c, f) for the next key
    void step(std::uint64_t x, std::uint64_t& p, std::uint64_t& c, std::uint64_t& f) noexcept {
        p = predict(x, key0, pre, m); c = nextc; f = p > nextc ? p : nextc; nextc = f + 1;
    }
};
inline std::uint64_t direct_lines(std::uint64_t U, std::uint64_t flast, unsigned W) noexcept {
    return roundup(std::max((U + 3) / 4, (flast >> 2) + 1) + W - 1, 8);
}
// Closed-form DIRECT evaluation for windows 1, 2, 4 in one placement pass (out[0..2]).
inline void eval_direct_seg(const Fit& f, std::size_t j, unsigned q, Sum* out) {
    const std::size_t a = f.start[j], e = f.start[j + 1], nj = e - a;
    const std::uint64_t U = direct_units(nj, q);
    DirectPlace dp{f.knot(j), f.pre[j], slope(U, f.sp[j])};
    for (int w = 0; w < 3; ++w) out[w] = Sum{};
    std::uint64_t flast = 0;
    for (std::size_t i = a; i < e; ++i) {
        std::uint64_t p, c, fi; dp.step(f.ks.key(i), p, c, fi); flast = fi;
        const std::uint64_t h = p >> 2, ai = std::max(h, c >> 2);
        for (unsigned w = 0; w < 3; ++w) {
            const std::uint64_t W = 1u << w, last = h + W - 1;
            Sum& s = out[w];
            if (ai <= last) { s.dep += 1; s.pgA += (last >> 6) - (h >> 6) + 1; }
            else { s.dep += 1 + (ai - last); s.scan += ai - last; s.pgA += (ai >> 6) - (h >> 6) + 1; }
            s.par += W - 1;
        }
    }
    for (unsigned w = 0; w < 3; ++w) {
        const std::uint64_t R = direct_lines(U, flast, 1u << w);
        out[w].bytes = R * 64; out[w].units = std::uint32_t(R / 8);
    }
}

// FENCED subtree builder. Writes one node's entry and (depth-first) its section-B subtree through Mem:
// the final arena, or a scratch buffer whose line_off values are relative to the scratch start.
// Alignment is taken on org + local offset, so scratch and arena layouts are byte-identical apart from
// line_off.
struct Scratch {
    unsigned char* p = nullptr; std::size_t cap = 0;
    Scratch() = default; Scratch(const Scratch&) = delete; Scratch& operator=(const Scratch&) = delete;
    ~Scratch() { if (p) ::operator delete(p, std::align_val_t(128)); }
    unsigned char* at(std::uint64_t off, std::size_t len) {
        if (off + len > cap) {
            const std::size_t nc = std::max<std::size_t>(std::size_t(off + len), cap * 2 + 4096);
            auto* q = static_cast<unsigned char*>(::operator new(nc, std::align_val_t(128)));
            if (p) { std::memcpy(q, p, cap); ::operator delete(p, std::align_val_t(128)); }
            p = q; cap = nc;
        }
        return p + off;
    }
    void mark(std::uint64_t, std::uint64_t, std::uint8_t) noexcept {}
};
enum : std::uint8_t { LK_NONE = 0, LK_LINE = 1, LK_CLINE = 2 };
struct ArenaMem {   // final fill: local offset 0 is arena byte `org`; lk marks payload lines for the hash
    unsigned char* arena; std::uint64_t org; std::uint8_t* lk;
    unsigned char* at(std::uint64_t off, std::size_t) const noexcept { return arena + org + off; }
    void mark(std::uint64_t off, std::uint64_t lines, std::uint8_t kind) const noexcept {
        if (lk) std::memset(lk + (org + off) / 64, kind, lines);
    }
};
template<class Mem> struct Builder {
    const KeySrc& ks; unsigned EL, F; bool compact;
    Mem& mem;
    std::uint64_t org, line_org, cur = 0, compact_keys = 0;
    unsigned EB() const noexcept { return 64 * EL; }
    void put_entry(unsigned char* ep, const EntryHdr& h, const std::int16_t* f) const noexcept {
        std::memcpy(ep, &h, sizeof h);
        std::memcpy(ep + 16, f, F * sizeof(std::int16_t));
    }
    // top != nullptr: the entry goes there (section A); else at local offset eoff (section B).
    void node(std::size_t lo, std::size_t hi, unsigned char* top, std::uint64_t eoff) {
        const std::size_t c = hi - lo;
        const std::uint64_t first = ks.key(lo), span = ks.key(hi - 1) - first;
        unsigned shift = 0;
        while ((span >> shift) > 0xFFFEu) ++shift;
        const unsigned kpl = (compact && span < (1ull << 32)) ? 5 : 4;
        const std::size_t leafcap = std::size_t(kpl) * (F + 1);
        std::int16_t f[56];
        for (auto& x : f) x = FENCE_UNUSED;
        EntryHdr h{first, 0, std::uint8_t(shift), 0, 0};
        auto q = [&](std::size_t i) { return fence_bias(fence_q(ks.key(i), first, shift)); };
        if (c <= leafcap) {
            const std::size_t nl = (c + kpl - 1) / kpl;
            h.flags = kpl == 5 ? E_COMPACT : 0; h.nlines = std::uint16_t(nl);
            h.line_off = std::uint32_t((line_org + cur) / 64);
            for (std::size_t i = 0; i + 1 < nl; ++i) f[i] = q(lo + (i + 1) * kpl);
            put_entry(top ? top : mem.at(eoff, EB()), h, f);
            unsigned char* d = mem.at(cur, nl * 64);
            for (std::size_t l = 0; l < nl; ++l) {
                if (kpl == 4) {
                    Line L;
                    for (unsigned t = 0; t < 4; ++t) { const std::size_t i = std::min(lo + l * 4 + t, hi - 1); L.k[t] = ks.key(i); L.v[t] = ks.val(i); }
                    std::memcpy(d + l * 64, &L, 64);
                } else {
                    CLine L; L.pad = 0;
                    for (unsigned t = 0; t < 5; ++t) { const std::size_t i = std::min(lo + l * 5 + t, hi - 1); L.r[t] = std::uint32_t(ks.key(i) - first); L.v[t] = ks.val(i); }
                    std::memcpy(d + l * 64, &L, 64);
                }
            }
            mem.mark(cur, nl, kpl == 5 ? LK_CLINE : LK_LINE);
            if (kpl == 5) compact_keys += c;
            cur += nl * 64;
            return;
        }
        const std::size_t k = std::min<std::size_t>(F + 1, (c + leafcap - 1) / leafcap);
        cur = roundup(org + cur, EB()) - org;
        const std::uint64_t ch0 = cur;
        cur += k * EB();
        h.flags = E_INNER; h.nlines = std::uint16_t(k); h.line_off = std::uint32_t((line_org + ch0) / 64);
        for (std::size_t i = 0; i + 1 < k; ++i) f[i] = q(lo + c * (i + 1) / k);
        put_entry(top ? top : mem.at(eoff, EB()), h, f);
        for (std::size_t i = 0; i < k; ++i) node(lo + c * i / k, lo + c * (i + 1) / k, nullptr, ch0 + i * EB());
    }
};
// Maps the walker's pointers (a top entry in a stack buffer, a subtree in scratch) to arena offsets.
template<class Sink> struct OnScratch {
    static constexpr bool active = true;
    Sink& s; const unsigned char* top; std::uint64_t top_off; const unsigned char* sb; std::uint64_t sb_off;
    std::uint64_t off(const void* p) const noexcept {
        const auto q = reinterpret_cast<std::uintptr_t>(p), t = reinterpret_cast<std::uintptr_t>(top);
        return (q >= t && q < t + 128) ? top_off + (q - t) : sb_off + (q - reinterpret_cast<std::uintptr_t>(sb));
    }
    void exc_probe(std::uint32_t) noexcept {} void exc() noexcept {} void rtab(std::uint64_t) noexcept {} void rchild() noexcept {}
    void recs(std::uint32_t, std::uint32_t) noexcept {}
    void dep(const void* p, int tag) noexcept { s.line(off(p), tag, true); }
    void par(const void* p, int tag) noexcept { s.line(off(p), tag, false); }
    void level() noexcept { s.level(); }
    void done(bool f) noexcept { s.done(f); }
};
constexpr std::uint64_t LOCAL_B = 1ull << 40;   // selection-time section-B origin, distinct from section-A pages
// Walks the groups (keys with equal predicted entry) of FENCED segment j: builds each nonempty group's
// subtree into scratch (B origin borg), descends it for every key with sink(i) and returns the
// segment's section-B bytes. aoff = the segment region's arena offset.
template<class SinkFor>
std::uint64_t fenced_walk(const Fit& f, std::size_t j, unsigned EL, std::uint32_t cbar, bool compact, std::uint64_t aoff, std::uint64_t borg,
                          Scratch& sc, std::uint64_t* compact_keys, SinkFor&& sink_for) {
    const std::size_t a = f.start[j], e = f.start[j + 1], nj = e - a;
    const std::uint64_t Q = (nj + cbar - 1) / cbar, EB = 64u * EL;
    const std::uint32_t m = slope(Q, f.sp[j]);
    const std::uint64_t k0 = f.knot(j); const unsigned pre = f.pre[j];
    alignas(128) unsigned char top[128];
    std::uint64_t segcur = 0;
    std::size_t i = a;
    while (i < e) {
        const std::uint64_t g = predict(f.ks.key(i), k0, pre, m);
        std::size_t j2 = i + 1;
        while (j2 < e && predict(f.ks.key(j2), k0, pre, m) == g) ++j2;
        if (g >= Q) throw std::logic_error("FENCED prediction out of range");
        Builder<Scratch> b{f.ks, EL, EL == 2 ? 56u : 24u, compact, sc, borg + segcur, 0};
        b.node(i, j2, top, 0);
        for (std::size_t t = i; t < j2; ++t) {
            auto& s = sink_for(t);
            OnScratch<std::remove_reference_t<decltype(s)>> ad{s, top, aoff + g * EB, sc.p, borg + segcur};
            std::uint64_t out = 0;
            const bool ok = EL == 2 ? fenced_lookup<2>(sc.p, top, f.ks.key(t), out, ad) : fenced_lookup<1>(sc.p, top, f.ks.key(t), out, ad);
            if (!ok || out != f.ks.val(t)) throw std::logic_error("FENCED walk lost a key");
        }
        if (compact_keys) *compact_keys += b.compact_keys;
        segcur += b.cur;
        i = j2;
    }
    return roundup(segcur, 128);
}
inline void eval_fenced_seg(const Fit& f, std::size_t j, unsigned EL, std::uint32_t cbar, bool compact, Scratch& sc, Sum& s) {
    const std::size_t nj = f.nseg_keys(j);
    const std::uint64_t Q = (nj + cbar - 1) / cbar, EB = 64u * EL, Qpad = roundup(Q, 512 / EB);
    Acc acc;
    s = Sum{};
    s.bbytes = fenced_walk(f, j, EL, cbar, compact, 0, LOCAL_B, sc, &s.compact_keys, [&](std::size_t) -> Acc& { return acc; });
    s.dep = acc.t.dep; s.par = acc.t.par; s.pgA = acc.t.pages4k[REG_AE]; s.pgB = acc.t.pages4k[REG_B];
    for (int d = 0; d < 4; ++d) s.depth[d] = acc.t.depth[d];
    s.ties = acc.t.ties();
    s.units = std::uint32_t(Qpad * EB / 512);
    s.bytes = Qpad * EB + s.bbytes;
}
// Per-fit candidate caches: DIRECT per alpha (windows 1,2,4), FENCED per (entry, cbar) (fast, compact).
struct DirectCache { unsigned q = 0; std::vector<Sum> s; long long ns = 0; };     // K x 3
struct FencedCache { unsigned el = 0; std::uint32_t cbar = 0; std::vector<Sum> s; long long ns = 0; unsigned variants = 0; };   // K x 2
inline DirectCache eval_direct(const Fit& f, unsigned q, unsigned threads) {
    DirectCache d; d.q = q; d.s.resize(f.K() * 3);
    const long long t0 = now_ns();
    const auto ch = seg_chunks(f);
    par_for(ch.size() - 1, threads, [&](std::size_t c) { for (std::size_t j = ch[c]; j < ch[c + 1]; ++j) eval_direct_seg(f, j, q, &d.s[j * 3]); });
    d.ns = now_ns() - t0;
    return d;
}
// variants: bit 0 fast, bit 1 compact.
inline FencedCache eval_fenced(const Fit& f, unsigned el, std::uint32_t cbar, unsigned variants, unsigned threads) {
    FencedCache d; d.el = el; d.cbar = cbar; d.variants = variants; d.s.resize(f.K() * 2);
    const long long t0 = now_ns();
    const auto ch = seg_chunks(f);
    par_for(ch.size() - 1, threads, [&](std::size_t c) {
        Scratch sc;
        for (std::size_t j = ch[c]; j < ch[c + 1]; ++j)
            for (unsigned v = 0; v < 2; ++v) if (variants >> v & 1) eval_fenced_seg(f, j, el, cbar, v == 1, sc, d.s[j * 2 + v]);
    });
    d.ns = now_ns() - t0;
    return d;
}

// ---------------------------------------------------------------- Cell: one grid point's candidate list
struct CandDesc { std::uint8_t fenced, q, w, compact; const std::vector<Sum>* src; unsigned stride, off; };
struct Cell {
    const Fit* fit = nullptr;
    SpliceParams p;
    std::vector<CandDesc> desc;
    const Sum& sum(std::size_t j, std::size_t c) const noexcept { const auto& d = desc[c]; return (*d.src)[j * d.stride + d.off]; }
};
// Candidate order is the tie-break: DIRECT (alpha asc, w asc), then FENCED.
inline Cell make_cell(const Fit& f, const SpliceParams& p, const std::vector<const DirectCache*>& dc, const FencedCache* fc) {
    Cell c; c.fit = &f; c.p = p;
    if (p.mode != MODE_FENCED)
        for (unsigned q : p.alphas()) {
            const DirectCache* d = nullptr;
            for (auto* x : dc) if (x && x->q == q) d = x;
            if (!d) throw std::logic_error("missing DIRECT candidates for an alpha");
            for (unsigned w : p.windows()) c.desc.push_back(CandDesc{0, std::uint8_t(q), std::uint8_t(w), 0, &d->s, 3, w == 1 ? 0u : w == 2 ? 1u : 2u});
        }
    if (p.mode != MODE_DIRECT) {
        if (!fc || fc->el != p.entry || fc->cbar != p.cbar || !(fc->variants >> p.compact & 1)) throw std::logic_error("missing FENCED candidates");
        c.desc.push_back(CandDesc{1, 0, 0, p.compact, &fc->s, 2, p.compact});
    }
    if (c.desc.empty() || c.desc.size() > 64) throw std::logic_error("bad candidate list");
    return c;
}

// ---------------------------------------------------------------- selection
struct PVals { std::int64_t ad, ae, b; };   // milli-probability that a 4 KiB walk misses L2, per region
inline PVals default_p(const SpliceParams& p) { return PVals{p.p0_ad, p.p0_ae, p.p0_b}; }
inline std::uint64_t cand_cost(const Cell& c, std::size_t j, std::size_t k, const PVals& P) {
    const Sum& s = c.sum(j, k); const auto& p = c.p; const bool fen = c.desc[k].fenced;
    const std::int64_t J = 1000 * std::int64_t(s.dep) + p.d_tau * std::int64_t(s.par)
        + std::int64_t(s.pgA) * ((fen ? P.ae : P.ad) + p.d_walk_hit) + std::int64_t(s.pgB) * (P.b + p.d_walk_hit)
        + (fen ? p.d_c_fenced : p.d_c_direct) * std::int64_t(c.fit->nseg_keys(j));
    return std::uint64_t(J);
}
struct Selection {
    std::vector<std::uint8_t> choice;
    std::uint64_t lambda = 0, bytes = 0, J = 0;   // J: sum of chosen candidate costs (no router/record/exception terms)
    bool cap_violated = false;
};
// Multiple-choice knapsack by a Lagrangian on bytes: the smallest integer lambda (milli-D per byte)
// whose per-segment argmin meets bytes*10 <= cap_tenths*n. bytes is the index's total_bytes exactly:
// the 512-B header plus the candidates' bytes is the arena's used end (make_plan), rounded up to the
// 2 MiB mapping, plus records, router and exceptions; the rounding is monotone, so the bisection holds.
inline Selection select_cell(const Cell& c, const PVals& P, unsigned threads) {
    const Fit& f = *c.fit; const std::size_t K = f.K(), nc = c.desc.size();
    std::vector<std::uint64_t> J(K * nc), B(K * nc);
    par_for((K + 4095) / 4096, threads, [&](std::size_t t) {
        for (std::size_t j = t * 4096; j < std::min(K, (t + 1) * 4096); ++j)
            for (std::size_t k = 0; k < nc; ++k) { J[j * nc + k] = cand_cost(c, j, k, P); B[j * nc + k] = c.sum(j, k).bytes; }
    });
    const u128 cap = u128(c.p.cap_tenths) * f.n;
    auto run = [&](std::uint64_t lam, Selection& s) {
        s.choice.resize(K); s.J = 0; s.lambda = lam;
        std::uint64_t arena = 512;
        for (std::size_t j = 0; j < K; ++j) {
            std::size_t best = 0; u128 bc = ~u128(0);
            for (std::size_t k = 0; k < nc; ++k) { const u128 v = u128(J[j * nc + k]) + u128(lam) * B[j * nc + k]; if (v < bc) { bc = v; best = k; } }
            s.choice[j] = std::uint8_t(best); arena += B[j * nc + best]; s.J += J[j * nc + best];
        }
        s.bytes = roundup(arena, 2ull << 20) + f.meta_bytes();
        return u128(s.bytes) * 10 <= cap;
    };
    Selection s;
    if (run(0, s)) return s;
    std::uint64_t lo = 0, hi = 1;
    while (!run(hi, s) && hi < (1u << 20)) { lo = hi; hi *= 2; }
    if (!run(hi, s)) { s.cap_violated = true; return s; }
    while (hi > lo + 1) { const std::uint64_t mid = lo + (hi - lo) / 2; Selection t; if (run(mid, t)) hi = mid; else lo = mid; }
    run(hi, s);
    return s;
}

// ---------------------------------------------------------------- plan: arena offsets
constexpr std::uint64_t ARENA_LIMIT = (1ull << 24) * 512;   // 24-bit base in 512-B units
struct Plan {
    std::vector<std::uint8_t> choice;
    std::vector<std::uint32_t> base;      // K+1, 512-B units; base[K] = end of section A
    std::vector<std::uint64_t> boff;      // K, absolute section-B start of FENCED segments
    std::uint64_t a_end = 0, b_end = 0, arena_bytes = 0, total_bytes = 0;
};
inline Plan make_plan(const Cell& c, const Selection& s) {
    const Fit& f = *c.fit; const std::size_t K = f.K();
    Plan pl; pl.choice = s.choice; pl.base.resize(K + 1); pl.boff.assign(K, 0);
    std::uint64_t u = 1;
    for (std::size_t j = 0; j < K; ++j) {
        pl.base[j] = std::uint32_t(u); u += c.sum(j, s.choice[j]).units;
        if (u >= (1ull << 24)) throw std::runtime_error("arena exceeds 8 GiB");
    }
    pl.base[K] = std::uint32_t(u); pl.a_end = u * 512;
    std::uint64_t b = pl.a_end;
    for (std::size_t j = 0; j < K; ++j) if (c.desc[s.choice[j]].fenced) { pl.boff[j] = b; b += c.sum(j, s.choice[j]).bbytes; }
    pl.b_end = b;
    if (pl.b_end > ARENA_LIMIT) throw std::runtime_error("arena exceeds 8 GiB");
    if (pl.b_end / 64 > 0xFFFFFFFFull) throw std::runtime_error("line_off exceeds 2^32");
    pl.arena_bytes = roundup(std::max<std::uint64_t>(pl.b_end, 1), 2ull << 20);
    pl.total_bytes = pl.arena_bytes + f.meta_bytes();
    if (pl.total_bytes != s.bytes) throw std::logic_error("plan bytes differ from the selection's");
    return pl;
}

// ---------------------------------------------------------------- plan walker (count + trace)
// Recomputes every segment's chosen layout with real offsets and accounts every stored key, so the
// totals are what get_impl<Count> must reproduce exactly (--parity). Sampled keys also get their
// line-access trace for the L2 simulation.
struct Traces {
    std::vector<std::size_t> idx;          // sorted unique key indices
    std::vector<std::uint64_t> ids;        // concatenated traces
    std::vector<std::uint64_t> off;        // idx.size()+1
    std::uint64_t truncated = 0;           // traces cut at TRACE_CAP ids (long DIRECT scans)
};
struct WalkOut { CountTotals tot; Traces tr; long long ns = 0; };
struct WSink {
    Acc* a; TraceCore* t;
    void line(std::uint64_t o, int tag, bool d) noexcept { a->line(o, tag, d); if (t) t->line(o, tag, d); }
    void level() noexcept { a->level(); }
    void done(bool f) noexcept { a->done(f); }
};
template<class Core> struct IdxOnly {   // routing / exception events only
    static constexpr bool active = true;
    Core& c;
    void exc_probe(std::uint32_t i) noexcept { c.exc_probe(i); } void exc() noexcept { c.exc(); } void rtab(std::uint64_t b) noexcept { c.rtab(b); }
    void rchild() noexcept { c.rchild(); } void recs(std::uint32_t j0, std::uint32_t j) noexcept { c.recs(j0, j); }
    void dep(const void*, int) noexcept {} void par(const void*, int) noexcept {} void level() noexcept {} void done(bool) noexcept {}
};
inline WalkOut walk_plan(const Cell& c, const Plan& pl, const std::vector<std::size_t>* samples, unsigned threads) {
    const Fit& f = *c.fit; const long long t0 = now_ns();
    WalkOut w;
    const auto ch = seg_chunks(f);
    std::vector<CountTotals> part(ch.size() - 1);
    const std::vector<std::size_t> none;
    const auto& smp = samples ? *samples : none;
    std::vector<std::vector<std::uint64_t>> tids(ch.size() - 1), toff(ch.size() - 1);
    std::vector<std::uint64_t> cuts(ch.size() - 1, 0);
    const View rv = f.rview();
    par_for(ch.size() - 1, threads, [&](std::size_t ci) {
        Scratch sc; Acc acc;
        auto& ids = tids[ci]; auto& offs = toff[ci];
        std::size_t si = std::size_t(std::lower_bound(smp.begin(), smp.end(), f.start[ch[ci]]) - smp.begin());
        const std::size_t send = std::size_t(std::lower_bound(smp.begin(), smp.end(), f.start[ch[ci + 1]]) - smp.begin());
        TraceCore tc;
        std::uint64_t cut = 0;
        auto flush = [&] { for (unsigned k = 0; k < tc.n; ++k) ids.push_back(tc.ids[k]); offs.push_back(ids.size()); cut += tc.over; tc.n = 0; tc.over = false; };
        for (std::size_t j = ch[ci]; j < ch[ci + 1]; ++j) {
            const CandDesc& d = c.desc[pl.choice[j]];
            const std::uint64_t aoff = std::uint64_t(pl.base[j]) * 512;
            const std::size_t a = f.start[j], e = f.start[j + 1];
            // Traced keys: routing events first (program order), then the arena lines.
            auto begin_key = [&](std::size_t i) -> TraceCore* {
                if (si < send && smp[si] == i) { ++si; tc.n = 0; tc.over = false; IdxOnly<TraceCore> io{tc}; route(rv, f.ks.key(i), io); return &tc; }
                return nullptr;
            };
            if (!d.fenced) {
                const std::uint64_t W = d.w, R = std::uint64_t(pl.base[j + 1] - pl.base[j]) * 8;
                DirectPlace dp{f.knot(j), f.pre[j], slope(direct_units(e - a, d.q), f.sp[j])};
                for (std::size_t i = a; i < e; ++i) {
                    std::uint64_t p, cc, fi; dp.step(f.ks.key(i), p, cc, fi);
                    const std::uint64_t h = p >> 2, ai = std::max(h, cc >> 2);
                    if (h + W > R || ai >= R) throw std::logic_error("DIRECT window outside its region");
                    TraceCore* t = begin_key(i);
                    WSink s{&acc, t};
                    s.line(aoff + h * 64, T_WINDOW, true);
                    for (std::uint64_t k = 1; k < W; ++k) s.line(aoff + (h + k) * 64, T_WINDOW, false);
                    for (std::uint64_t l = h + W; l <= ai; ++l) s.line(aoff + l * 64, T_SCAN, true);
                    s.done(true);
                    if (t) flush();
                }
            } else {
                TraceCore* cur = nullptr; WSink s{&acc, nullptr};
                fenced_walk(f, j, c.p.entry, c.p.cbar, d.compact, aoff, pl.boff[j], sc, nullptr, [&](std::size_t i) -> WSink& {
                    if (cur) { flush(); cur = nullptr; }
                    cur = begin_key(i); s.t = cur; return s;
                });
                if (cur) flush();
            }
            acc.t.rchild += f.seg_rchild[j]; acc.t.rlines += f.seg_rlines[j];
        }
        part[ci] = acc.t; cuts[ci] = cut;
    });
    for (auto x : cuts) w.tr.truncated += x;
    for (auto& p : part) w.tot += p;
    const std::uint64_t ne = f.exc.size();
    w.tot.exc += ne; w.tot.lookups += ne; w.tot.found += ne;
    // Assemble traces in key order: exceptions below the bulk, the chunks, exceptions above.
    w.tr.idx = smp; w.tr.off.push_back(0);
    auto exc_trace = [&](std::size_t i) {
        TraceCore tc; IdxOnly<TraceCore> io{tc}; std::uint64_t out;
        exc_lookup(rv, f.ks.key(i), out, io);
        for (unsigned k = 0; k < tc.n; ++k) w.tr.ids.push_back(tc.ids[k]);
        w.tr.off.push_back(w.tr.ids.size());
    };
    std::size_t s = 0;
    for (; s < smp.size() && smp[s] < f.bb; ++s) exc_trace(smp[s]);
    for (std::size_t ci = 0; ci + 1 < ch.size(); ++ci) {
        const std::uint64_t base = w.tr.ids.size();
        w.tr.ids.insert(w.tr.ids.end(), tids[ci].begin(), tids[ci].end());
        for (auto o : toff[ci]) w.tr.off.push_back(base + o);
        s += toff[ci].size();
    }
    for (; s < smp.size(); ++s) exc_trace(smp[s]);
    if (w.tr.off.size() != smp.size() + 1) throw std::logic_error("trace count mismatch");
    w.ns = now_ns() - t0;
    return w;
}

// ---------------------------------------------------------------- shared-L2 LRU residency simulation
struct Draws { std::vector<std::size_t> uniq; std::vector<std::uint32_t> pos; std::uint64_t warm = 0; };
inline std::uint64_t splitmix64(std::uint64_t& s) noexcept {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31);
}
inline Draws make_draws(std::size_t n, std::uint64_t warm, std::uint64_t sim, std::uint64_t seed) {
    Draws d; d.warm = warm;
    std::vector<std::size_t> idx(warm + sim);
    std::uint64_t s = seed;
    for (auto& x : idx) x = std::size_t((u128(splitmix64(s)) * n) >> 64);
    d.uniq = idx; std::sort(d.uniq.begin(), d.uniq.end()); d.uniq.erase(std::unique(d.uniq.begin(), d.uniq.end()), d.uniq.end());
    d.pos.resize(idx.size());
    for (std::size_t i = 0; i < idx.size(); ++i) d.pos[i] = std::uint32_t(std::lower_bound(d.uniq.begin(), d.uniq.end(), idx[i]) - d.uniq.begin());
    return d;
}
struct SimOut {
    bool thp = false; unsigned l2 = 2048;
    std::uint64_t lookups = 0, walks[REG_N] = {}, ptmiss[REG_N] = {}, router_miss = 0, record_miss = 0, exc_miss = 0, arena_miss = 0;
    std::int64_t P[REG_N] = {};
    std::int64_t router_milli = 0, record_milli = 0, exc_milli = 0;
    long long ns = 0;
};
struct LRU {
    unsigned lg; std::vector<std::uint64_t> tag, ts; std::uint64_t clk = 0;
    explicit LRU(unsigned l2kb) {
        const std::size_t sets = std::size_t(l2kb) * 1024 / 64 / 16;
        lg = bitlen(sets) - 1; tag.assign(sets * 16, 0); ts.assign(sets * 16, 0);
    }
    bool access(std::uint64_t id) noexcept {   // ts == 0 marks an empty way
        const std::size_t s = std::size_t((id * 0x9E3779B97F4A7C15ULL) >> (64 - lg)) * 16;
        std::uint64_t* T = &tag[s]; std::uint64_t* A = &ts[s];
        ++clk; unsigned v = 0;
        for (unsigned w = 0; w < 16; ++w) { if (A[w] && T[w] == id) { A[w] = clk; return true; } if (A[w] < A[v]) v = w; }
        T[v] = id; A[v] = clk; return false;
    }
};
// Replays the traces in draw order: a GRE ops line every 4th lookup, routing, records, then the arena
// lines with the page-table line of each distinct page of the lookup inserted before its first access
// (the STLB is assumed to miss: its reach is under 0.1% of the arena).
inline SimOut simulate(const Traces& tr, const Draws& d, bool thp, unsigned l2kb, const PVals& prev) {
    const long long t0 = now_ns();
    SimOut o; o.thp = thp; o.l2 = l2kb;
    LRU c(l2kb);
    std::uint64_t pages[64]; unsigned np = 0;
    const std::uint64_t LINE_MASK = (1ull << 52) - 1;
    for (std::size_t q = 0; q < d.pos.size(); ++q) {
        const bool cnt = q >= d.warm;
        if (q % 4 == 0) c.access(TR_OPS | (q / 4));
        const std::size_t u = d.pos[q];
        np = 0;
        for (std::uint64_t e = tr.off[u]; e < tr.off[u + 1]; ++e) {
            const std::uint64_t id = tr.ids[e], tg = id >> 56;
            if (tg == 4) {
                const std::uint64_t line = id & LINE_MASK; const unsigned reg = unsigned((id >> 52) & 3);
                const std::uint64_t pg = thp ? line >> 15 : line >> 6;
                bool seen = false;
                for (unsigned k = 0; k < np; ++k) if (pages[k] == pg) { seen = true; break; }
                if (!seen) {
                    if (np < 64) pages[np++] = pg;
                    const bool hit = c.access(thp ? (TR_PDE | (line >> 18)) : (TR_PTE | (line >> 9)));
                    if (cnt) { ++o.walks[reg]; o.ptmiss[reg] += !hit; }
                }
                const bool hit = c.access(TR_ARENA | line);
                if (cnt) o.arena_miss += !hit;
            } else {
                const bool hit = c.access(id);
                if (cnt && !hit) { if (tg == 1) ++o.router_miss; else if (tg == 2) ++o.record_miss; else if (tg == 3) ++o.exc_miss; }
            }
        }
        o.lookups += cnt;
    }
    const std::int64_t pv[REG_N] = {prev.ad, prev.ae, prev.b};
    for (int r = 0; r < REG_N; ++r) o.P[r] = o.walks[r] ? std::int64_t(1000 * o.ptmiss[r] / o.walks[r]) : pv[r];
    const std::uint64_t L = std::max<std::uint64_t>(o.lookups, 1);
    o.router_milli = std::int64_t(1000 * o.router_miss / L); o.record_milli = std::int64_t(1000 * o.record_miss / L); o.exc_milli = std::int64_t(1000 * o.exc_miss / L);
    o.ns = now_ns() - t0;
    return o;
}
// E[D] in milli-D for one arm: exact plan counts priced with that arm's simulated residency.
inline std::int64_t expected_d(const CountTotals& t, const SpliceParams& p, bool thp, const std::int64_t P[REG_N],
                               std::int64_t router_milli, std::int64_t record_milli, std::int64_t exc_milli) {
    if (!t.lookups) return 0;
    std::int64_t J = 1000 * std::int64_t(t.dep) + p.d_tau * std::int64_t(t.par) + p.d_router_child * std::int64_t(t.rchild)
        + p.d_c_direct * std::int64_t(t.direct_keys) + p.d_c_fenced * std::int64_t(t.fenced_keys) + p.d_c_exc * std::int64_t(t.exc);
    for (int r = 0; r < REG_N; ++r)
        J += std::int64_t(thp ? t.pages2m[r] : t.pages4k[r]) * (P[r] + (thp ? p.d_walk_hit_thp : p.d_walk_hit));
    return J / std::int64_t(t.lookups) + router_milli + record_milli + exc_milli;
}
inline std::int64_t record_miss0(const SpliceParams& p) { return std::int64_t(155) * p.k1 / 16384; }
inline std::int64_t expected_d(const CountTotals& t, const SpliceParams& p, const SimOut& s) {
    return expected_d(t, p, s.thp, s.P, s.router_milli, s.record_milli, s.exc_milli);
}
// Stage-1 E[D]: candidate summaries (segment-local pages), default residency, no simulation.
inline std::int64_t stage1_ed(const Cell& c, const Selection& s, const PVals&) {   // P is already inside s.J
    const Fit& f = *c.fit;
    const std::int64_t J = std::int64_t(s.J) + c.p.d_router_child * std::int64_t(f.rchild_total) + c.p.d_c_exc * std::int64_t(f.exc.size());
    return J / std::int64_t(f.n) + record_miss0(c.p);
}

// ---------------------------------------------------------------- outer fixed point (residency couples segments)
struct FixedPoint {
    Selection sel; Plan plan; CountTotals counts; Traces traces;
    SimOut sim;                            // selection arm (4 KiB, params.l2) of the accepted plan
    SimOut rep[4];                         // 4K/2048, 4K/1024, THP/2048, THP/1024
    std::int64_t ed[4] = {}, ed_sel = 0, stage1 = 0;
    PVals P{};                             // residency the accepted selection was made with
    std::uint32_t iters = 0;
    std::uint64_t trace_keys = 0, trace_truncated = 0;   // sampled distinct keys of the accepted plan; traces cut
    std::vector<std::int64_t> ed_hist;     // E[D] of every evaluated iterate
    long long walk_ns = 0, sim_ns = 0;
};
inline FixedPoint fixed_point(const Cell& c, const Draws* draws, unsigned threads) {
    const auto& p = c.p;
    FixedPoint fp;
    PVals P = default_p(p);
    const bool simulate_on = draws && p.sim > 0;
    auto evaluate = [&](const Selection& s, WalkOut& w, SimOut& so) {
        const Plan pl = make_plan(c, s);
        w = walk_plan(c, pl, simulate_on ? &draws->uniq : nullptr, threads); fp.walk_ns += w.ns;
        if (simulate_on) { so = simulate(w.tr, *draws, false, p.l2, P); fp.sim_ns += so.ns; return std::make_pair(pl, expected_d(w.tot, p, so)); }
        const std::int64_t P0[REG_N] = {P.ad, P.ae, P.b};
        so = SimOut{}; for (int r = 0; r < REG_N; ++r) so.P[r] = P0[r];
        so.record_milli = record_miss0(p);
        return std::make_pair(pl, expected_d(w.tot, p, false, P0, 0, record_miss0(p), 0));
    };
    fp.sel = select_cell(c, P, threads); fp.P = P;
    fp.stage1 = stage1_ed(c, fp.sel, P);
    WalkOut w; SimOut so;
    auto [pl, e] = evaluate(fp.sel, w, so);
    fp.plan = pl; fp.counts = w.tot; fp.trace_keys = w.tr.idx.size(); fp.trace_truncated = w.tr.truncated; fp.traces = std::move(w.tr); fp.sim = so; fp.ed_sel = e; fp.ed_hist.push_back(e); fp.iters = 1;
    if (simulate_on)
        for (std::uint32_t t = 1; t < p.outer; ++t) {
            const PVals Pn{fp.sim.P[REG_AD], fp.sim.P[REG_AE], fp.sim.P[REG_B]};
            Selection st = select_cell(c, Pn, threads);
            if (st.choice == fp.sel.choice) break;
            P = Pn;
            WalkOut w2; SimOut so2;
            auto [pl2, e2] = evaluate(st, w2, so2);
            fp.ed_hist.push_back(e2);
            if (!(e2 + 10 < fp.ed_sel)) break;    // accept only a drop of more than 0.01 D
            fp.sel = std::move(st); fp.plan = pl2; fp.counts = w2.tot; fp.trace_keys = w2.tr.idx.size(); fp.trace_truncated = w2.tr.truncated; fp.traces = std::move(w2.tr); fp.sim = so2; fp.ed_sel = e2; fp.P = Pn; ++fp.iters;
        }
    // Reporting arms on the accepted plan.
    const bool thp[4] = {false, false, true, true}; const unsigned l2[4] = {2048, 1024, 2048, 1024};
    if (simulate_on) {
        const PVals Pd = default_p(p);
        par_for(4, std::min(threads, 4u), [&](std::size_t k) { fp.rep[k] = simulate(fp.traces, *draws, thp[k], l2[k], Pd); });
        for (int k = 0; k < 4; ++k) { fp.ed[k] = expected_d(fp.counts, p, fp.rep[k]); fp.sim_ns += fp.rep[k].ns; }
    } else {
        const std::int64_t P0[REG_N] = {p.p0_ad, p.p0_ae, p.p0_b};
        for (int k = 0; k < 4; ++k) {
            fp.rep[k].thp = thp[k]; fp.rep[k].l2 = l2[k];
            for (int r = 0; r < REG_N; ++r) fp.rep[k].P[r] = P0[r];
            fp.rep[k].record_milli = record_miss0(p);
            fp.ed[k] = expected_d(fp.counts, p, thp[k], P0, 0, record_miss0(p), 0);
        }
    }
    return fp;
}

// ---------------------------------------------------------------- fill (write the arena)
// Each thread writes whole segments' regions and subtrees, which are disjoint, so the bytes do not
// depend on the thread count.
inline void fill_arena(const Cell& c, const Plan& pl, unsigned char* arena, std::uint8_t* lk, unsigned threads) {
    const Fit& f = *c.fit;
    const auto ch = seg_chunks(f);
    par_for(ch.size() - 1, threads, [&](std::size_t ci) {
        for (std::size_t j = ch[ci]; j < ch[ci + 1]; ++j) {
            const CandDesc& d = c.desc[pl.choice[j]];
            const std::uint64_t aoff = std::uint64_t(pl.base[j]) * 512, units = pl.base[j + 1] - pl.base[j];
            const std::size_t a = f.start[j], e = f.start[j + 1], nj = e - a;
            if (!d.fenced) {
                const std::uint64_t R = units * 8;
                Line* L = reinterpret_cast<Line*>(arena + aoff);
                DirectPlace dp{f.knot(j), f.pre[j], slope(direct_units(nj, d.q), f.sp[j])};
                std::size_t idx = a; std::uint64_t p, cc, fcur; dp.step(f.ks.key(a), p, cc, fcur);
                for (std::uint64_t s = 0; s < R * 4; ++s) {
                    while (s > fcur && idx + 1 < e) { ++idx; dp.step(f.ks.key(idx), p, cc, fcur); }
                    L[s >> 2].k[s & 3] = f.ks.key(idx); L[s >> 2].v[s & 3] = f.ks.val(idx);
                }
                if (lk) std::memset(lk + aoff / 64, LK_LINE, R);
                continue;
            }
            const unsigned EL = c.p.entry, EB = 64 * EL, F = EL == 2 ? 56 : 24;
            const std::uint64_t Q = (nj + c.p.cbar - 1) / c.p.cbar, Qpad = units * 512 / EB;
            const std::uint32_t m = slope(Q, f.sp[j]);
            const std::uint64_t k0 = f.knot(j); const unsigned pre = f.pre[j];
            std::uint64_t segcur = 0, next_e = 0;
            std::int16_t fe[56];
            for (auto& x : fe) x = FENCE_UNUSED;
            const EntryHdr eh{0, 0, 0, E_EMPTY, 0};
            auto empty = [&](std::uint64_t ei) { unsigned char* ep = arena + aoff + ei * EB; std::memcpy(ep, &eh, 16); std::memcpy(ep + 16, fe, F * 2); };
            std::size_t i = a;
            while (i < e) {
                const std::uint64_t g = predict(f.ks.key(i), k0, pre, m);
                std::size_t j2 = i + 1;
                while (j2 < e && predict(f.ks.key(j2), k0, pre, m) == g) ++j2;
                for (; next_e < g; ++next_e) empty(next_e);
                ArenaMem mem{arena, pl.boff[j] + segcur, lk};
                Builder<ArenaMem> b{f.ks, EL, F, d.compact != 0, mem, pl.boff[j] + segcur, pl.boff[j] + segcur};
                b.node(i, j2, arena + aoff + g * EB, 0);
                segcur += b.cur; next_e = g + 1; i = j2;
            }
            for (; next_e < Qpad; ++next_e) empty(next_e);
        }
    });
}
inline void fill_records(const Cell& c, const Plan& pl, Seg16* seg) {
    const Fit& f = *c.fit; const std::size_t K = f.K();
    for (std::size_t j = 0; j < K; ++j) {
        const CandDesc& d = c.desc[pl.choice[j]]; const std::size_t nj = f.nseg_keys(j);
        const std::uint64_t U = d.fenced ? (nj + c.p.cbar - 1) / c.p.cbar : direct_units(nj, d.q);
        const unsigned kind = d.fenced ? 0 : d.w == 1 ? 1 : d.w == 2 ? 2 : 3;
        seg[j] = Seg16{f.knot(j), slope(U, f.sp[j]), seg_meta(pl.base[j], f.pre[j], kind)};
    }
    seg[K] = Seg16{0, 0, seg_meta(pl.base[K], 0, 0)};
    for (std::size_t j = K + 1; j < K + 9; ++j) seg[j] = Seg16{0, 0, 0};
}

} // namespace splice
