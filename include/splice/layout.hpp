// SPLICE-H read path: POD layout and the integer-only lookup (docs/SPLICE_DESIGN.md).
// C++17 and self-contained: GRE's microbench is one C++17 TU and includes this header directly, so
// nothing here may include scaleli/*.hpp (C++20). The build side (build.hpp, cost.hpp) is C++20.
// The lookup uses no floating point, no division, no BSR/BSF/LZCNT/TZCNT (Goldmont has no LZCNT and
// BSR costs 10 cycles) and makes no store except to `out` on a hit.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#if defined(__SSE4_2__)
#include <nmmintrin.h>
#define SPLICE_SSE 1
#else
#define SPLICE_SSE 0
#endif

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "SPLICE layouts are defined little-endian");

// The read path must inline into the caller completely (no call in GRE's get(), tools/splice_asm_check.sh).
#define SPLICE_INLINE inline __attribute__((always_inline))

namespace splice {

// One merged tier-1 record per segment. meta = base:24 (512-B units from the arena start) | pre:6 | kind:2,
// kind 0 = FENCED, 1..3 = DIRECT with a window of 1 << (kind-1) lines. m is a 0.32 fixed-point slope.
struct Seg16 { std::uint64_t key; std::uint32_t m; std::uint32_t meta; };
constexpr std::uint32_t seg_base(std::uint32_t meta) noexcept { return meta & 0xFFFFFFu; }
constexpr unsigned seg_pre(std::uint32_t meta) noexcept { return (meta >> 24) & 63u; }
constexpr unsigned seg_kind(std::uint32_t meta) noexcept { return meta >> 30; }
constexpr std::uint32_t seg_meta(std::uint32_t base, unsigned pre, unsigned kind) noexcept {
    return (base & 0xFFFFFFu) | (std::uint32_t(pre & 63u) << 24) | (std::uint32_t(kind & 3u) << 30);
}
// Router entry: bit31 child (node index in 0-30), else leaf: cnt in 27-30, j0 in 0-26.
constexpr std::uint32_t rleaf(std::uint32_t j0, std::uint32_t cnt) noexcept { return (j0 & 0x07FFFFFFu) | ((cnt & 15u) << 27); }
constexpr std::uint32_t rchild_entry(std::uint32_t node) noexcept { return 0x80000000u | node; }
struct RNode { std::uint64_t lo; std::uint32_t off; std::uint8_t shift; std::uint8_t bits; std::uint16_t pad; };
// 4 keys then 4 payloads: DIRECT slot lines and FENCED fast data lines share this format.
struct alignas(64) Line { std::uint64_t k[4]; std::uint64_t v[4]; };
// Compact FENCED data line: 5 residuals key - entry.base (< 2^32), then 5 payloads.
struct alignas(64) CLine { std::uint32_t r[5]; std::uint32_t pad; std::uint64_t v[5]; };
enum : std::uint8_t { E_INNER = 1, E_COMPACT = 2, E_EMPTY = 4 };
struct EntryHdr { std::uint64_t base; std::uint32_t line_off; std::uint8_t shift; std::uint8_t flags; std::uint16_t nlines; };
struct alignas(64) Entry64 { EntryHdr h; std::int16_t f[24]; };
struct alignas(64) Entry128 { EntryHdr h; std::int16_t f[56]; };
struct Exc { std::uint64_t key; std::uint64_t val; };
struct View {
    std::uint64_t lo, span;            // bulk key range [lo, lo + span]
    const std::uint32_t* rtop;
    const RNode* rnodes;
    const std::uint32_t* rsub;
    const Seg16* seg;
    std::uint32_t nseg;
    std::uint8_t rshift, entry_lines, nfence, pad0;
    const unsigned char* arena;
    const Exc* exc;
    std::uint32_t nexc, pad1;
};

static_assert(sizeof(Seg16) == 16 && offsetof(Seg16, m) == 8 && offsetof(Seg16, meta) == 12);
static_assert(sizeof(RNode) == 16 && sizeof(Line) == 64 && sizeof(CLine) == 64 && offsetof(CLine, v) == 24);
static_assert(sizeof(EntryHdr) == 16 && sizeof(Entry64) == 64 && sizeof(Entry128) == 128);
static_assert(offsetof(Entry64, f) == 16 && offsetof(Entry128, f) == 16 && sizeof(Exc) == 16);
static_assert(sizeof(View) <= 128);
template<class T> constexpr bool pod_v = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;
static_assert(pod_v<Seg16> && pod_v<RNode> && pod_v<Line> && pod_v<CLine> && pod_v<EntryHdr> && pod_v<Entry64>);
static_assert(pod_v<Entry128> && pod_v<Exc> && pod_v<View>);

// The one predictor (build, cost model and lookup all call it): u < 2^32 and m < 2^32, so one 64-bit
// multiply suffices. The clamp only matters for absent keys far past a segment's last key.
SPLICE_INLINE std::uint64_t predict(std::uint64_t x, std::uint64_t key, unsigned pre, std::uint32_t m) noexcept {
    std::uint64_t u = (x - key) >> pre;
    u = u < 0xFFFFFFFFull ? u : 0xFFFFFFFFull;
    return (u * m) >> 32;
}
// Fence quantisation q(x) and its biased signed form (pcmpgtw is signed).
SPLICE_INLINE std::uint64_t fence_q(std::uint64_t x, std::uint64_t base, unsigned shift) noexcept {
    std::uint64_t q = (x - base) >> shift;
    q = q < 0xFFFEu ? q : 0xFFFEu;
    return x >= base ? q : 0;
}
SPLICE_INLINE std::int16_t fence_bias(std::uint64_t q) noexcept { return std::int16_t(std::uint16_t(q ^ 0x8000u)); }
constexpr std::int16_t FENCE_UNUSED = 0x7FFF;   // never <= a biased query, since q <= 0xFFFE

// #{i < 8*NV : f[i] <= qb}, branch-free over all fences.
template<unsigned NV> SPLICE_INLINE std::uint32_t fence_count(const std::int16_t* f, std::int16_t qb) noexcept {
#if SPLICE_SSE
    const __m128i Q = _mm_set1_epi16(qb);
    const __m128i* F = reinterpret_cast<const __m128i*>(f);
    std::uint32_t gt = 0;
    for (unsigned i = 0; i + 1 < NV; i += 2) {
        const __m128i a = _mm_cmpgt_epi16(_mm_load_si128(F + i), Q), b = _mm_cmpgt_epi16(_mm_load_si128(F + i + 1), Q);
        gt += std::uint32_t(__builtin_popcount(unsigned(_mm_movemask_epi8(_mm_packs_epi16(a, b)))));
    }
    if (NV & 1) {
        const __m128i a = _mm_cmpgt_epi16(_mm_load_si128(F + NV - 1), Q);
        gt += std::uint32_t(__builtin_popcount(unsigned(_mm_movemask_epi8(_mm_packs_epi16(a, _mm_setzero_si128())))));
    }
    return NV * 8 - gt;
#else
    std::uint32_t le = 0;
    for (unsigned i = 0; i < NV * 8; ++i) le += f[i] <= qb;
    return le;
#endif
}
// 4-bit mask of the line's keys equal to x.
SPLICE_INLINE std::uint32_t eq4(const Line* l, std::uint64_t x) noexcept {
#if SPLICE_SSE
    const __m128i X = _mm_set1_epi64x(static_cast<long long>(x));
    const __m128i a = _mm_cmpeq_epi64(_mm_load_si128(reinterpret_cast<const __m128i*>(l->k)), X);
    const __m128i b = _mm_cmpeq_epi64(_mm_load_si128(reinterpret_cast<const __m128i*>(l->k + 2)), X);
    return std::uint32_t(_mm_movemask_pd(_mm_castsi128_pd(a))) | (std::uint32_t(_mm_movemask_pd(_mm_castsi128_pd(b))) << 2);
#else
    return std::uint32_t(l->k[0] == x) | (std::uint32_t(l->k[1] == x) << 1) | (std::uint32_t(l->k[2] == x) << 2) | (std::uint32_t(l->k[3] == x) << 3);
#endif
}
SPLICE_INLINE std::uint32_t eq5(const CLine* l, std::uint32_t r) noexcept {
#if SPLICE_SSE
    const __m128i a = _mm_cmpeq_epi32(_mm_load_si128(reinterpret_cast<const __m128i*>(l->r)), _mm_set1_epi32(static_cast<int>(r)));
    return std::uint32_t(_mm_movemask_ps(_mm_castsi128_ps(a))) | (std::uint32_t(l->r[4] == r) << 4);
#else
    std::uint32_t m = 0;
    for (unsigned i = 0; i < 5; ++i) m |= std::uint32_t(l->r[i] == r) << i;
    return m;
#endif
}
// Lowest set bit's index through POPCNT (no TZCNT/BSF on the read path). Clang and GCC rewrite
// popcount((m & -m) - 1) into cttz, which -march=goldmont lowers to `rep bsf`; the empty asm hides the
// idiom and emits no instruction.
SPLICE_INLINE unsigned low_index(std::uint32_t m) noexcept {
    std::uint32_t t = (m & (0u - m)) - 1u;
    __asm__("" : "+r"(t));
    return unsigned(__builtin_popcount(t));
}
// Mask of the 8 record keys s[0..7] that are <= x (unsigned).
SPLICE_INLINE std::uint32_t le8(const Seg16* s, std::uint64_t x) noexcept {
#if SPLICE_SSE
    const __m128i bias = _mm_set1_epi64x(static_cast<long long>(0x8000000000000000ull));
    const __m128i X = _mm_xor_si128(_mm_set1_epi64x(static_cast<long long>(x)), bias);
    std::uint32_t gt = 0;
    for (unsigned i = 0; i < 4; ++i) {
        const __m128i a = _mm_load_si128(reinterpret_cast<const __m128i*>(s + 2 * i));
        const __m128i b = _mm_load_si128(reinterpret_cast<const __m128i*>(s + 2 * i + 1));
        const __m128i g = _mm_cmpgt_epi64(_mm_xor_si128(_mm_unpacklo_epi64(a, b), bias), X);
        gt |= std::uint32_t(_mm_movemask_pd(_mm_castsi128_pd(g))) << (2 * i);
    }
    return ~gt & 0xFFu;
#else
    std::uint32_t le = 0;
    for (unsigned i = 0; i < 8; ++i) le |= std::uint32_t(s[i].key <= x) << i;
    return le;
#endif
}

// ---------------------------------------------------------------- counter policies
// Tags classify every arena line a lookup touches; the region (DIRECT slots, section-A entries, section B)
// follows from the tag and selects the page-walk probability in the cost model.
enum : int { T_WINDOW = 0, T_SCAN, T_ENTRY, T_CHILD, T_DATA, T_TIE_ENTRY, T_TIE_LINE, T_NTAGS };
enum : int { REG_AD = 0, REG_AE = 1, REG_B = 2, REG_N = 3 };
constexpr int tag_region(int tag) noexcept { return tag <= T_SCAN ? REG_AD : tag == T_ENTRY ? REG_AE : REG_B; }
// Router lines live in one virtual byte space: rtop at 0, rnodes at 2^40, rsub at 2^41.
constexpr std::uint64_t RV_NODES = 1ull << 40, RV_SUB = 1ull << 41;

// No code, no stores: the policy GRE's get() uses.
struct NoCount {
    static constexpr bool active = false;
    void exc_probe(std::uint32_t) noexcept {}
    void exc() noexcept {}
    void rtab(std::uint64_t) noexcept {}
    void rchild() noexcept {}
    void recs(std::uint32_t, std::uint32_t) noexcept {}
    void dep(const void*, int) noexcept {}
    void par(const void*, int) noexcept {}
    void level() noexcept {}
    void done(bool) noexcept {}
};

// Record lines one lookup reads: the 8-record scan j0+1..j0+8 plus records j and j+1.
inline unsigned record_lines(std::uint32_t j0, std::uint32_t j, std::uint64_t* ids) noexcept {
    const std::uint64_t a = (std::uint64_t(j0) + 1) >> 2, b = (std::uint64_t(j0) + 8) >> 2;
    unsigned n = 0;
    for (std::uint64_t l = a; l <= b; ++l) ids[n++] = l;
    const std::uint64_t l1 = std::uint64_t(j) >> 2, l2 = (std::uint64_t(j) + 1) >> 2;
    if (l1 < a) ids[n++] = l1;
    if (l2 > b) ids[n++] = l2;
    return n;
}

// Integer totals over a set of lookups. Pages are distinct per lookup, attributed to the region of
// their first touch in that lookup.
struct CountTotals {
    std::uint64_t lookups = 0, found = 0, exc = 0, rchild = 0, rlines = 0, dep = 0, par = 0;
    std::uint64_t tagdep[T_NTAGS] = {}, pages4k[REG_N] = {}, pages2m[REG_N] = {}, depth[4] = {};
    std::uint64_t direct_keys = 0, fenced_keys = 0;
    CountTotals& operator+=(const CountTotals& o) noexcept {
        lookups += o.lookups; found += o.found; exc += o.exc; rchild += o.rchild; rlines += o.rlines; dep += o.dep; par += o.par;
        for (int i = 0; i < T_NTAGS; ++i) tagdep[i] += o.tagdep[i];
        for (int i = 0; i < REG_N; ++i) { pages4k[i] += o.pages4k[i]; pages2m[i] += o.pages2m[i]; }
        for (int i = 0; i < 4; ++i) depth[i] += o.depth[i];
        direct_keys += o.direct_keys; fenced_keys += o.fenced_keys;
        return *this;
    }
    bool operator==(const CountTotals& o) const noexcept { return std::memcmp(this, &o, sizeof o) == 0; }
    std::uint64_t ties() const noexcept { return tagdep[T_TIE_LINE] + tagdep[T_TIE_ENTRY]; }
    std::uint64_t overcap() const noexcept { return depth[1] + depth[2] + depth[3]; }
};
static_assert(std::is_trivially_copyable_v<CountTotals>);

// Offset-based accumulator shared by Count (pointers into a real arena) and the build's plan walker
// (pointers into scratch subtrees mapped to their final offsets), so both count with the same code.
struct Acc {
    CountTotals t;
    std::uint64_t pg[32], pg2[32];
    unsigned npg = 0, npg2 = 0, lvl = 0;
    bool any_arena = false, fenced = false;
    void exc_probe(std::uint32_t) noexcept {}
    void exc() noexcept { ++t.exc; }
    void rtab(std::uint64_t) noexcept {}
    void rchild() noexcept { ++t.rchild; }
    void recs(std::uint32_t j0, std::uint32_t j) noexcept { std::uint64_t ids[8]; t.rlines += record_lines(j0, j, ids); }
    static bool add(std::uint64_t* s, unsigned& n, std::uint64_t v) noexcept {
        for (unsigned i = 0; i < n; ++i) if (s[i] == v) return false;
        if (n < 32) s[n++] = v;   // more than 32 distinct pages in one lookup does not occur; extra ones count as new
        return true;
    }
    void line(std::uint64_t off, int tag, bool is_dep) noexcept {
        any_arena = true;
        if (tag >= T_ENTRY) fenced = true;
        if (is_dep) { ++t.dep; ++t.tagdep[tag]; } else ++t.par;
        const int r = tag_region(tag);
        if (add(pg, npg, off >> 12)) ++t.pages4k[r];
        if (add(pg2, npg2, off >> 21)) ++t.pages2m[r];
    }
    void level() noexcept { ++lvl; }
    void done(bool found) noexcept {
        ++t.lookups; t.found += found;
        if (lvl) ++t.depth[(lvl < 4 ? lvl : 4) - 1];
        if (any_arena) { if (fenced) ++t.fenced_keys; else ++t.direct_keys; }
        npg = npg2 = lvl = 0; any_arena = fenced = false;
    }
};

// Line ids of the shared-L2 simulation (cost.hpp): tag in bits 56-63; arena lines carry the region in 52-53.
constexpr std::uint64_t TR_ROUTER = 1ull << 56, TR_REC = 2ull << 56, TR_EXC = 3ull << 56, TR_ARENA = 4ull << 56;
constexpr std::uint64_t TR_OPS = 5ull << 56, TR_PTE = 6ull << 56, TR_PDE = 7ull << 56;
// A lookup's trace keeps its first TRACE_CAP ids; `over` records a cut, which the walker counts
// (Traces::truncated, reported per fixed point) because a cut trace under-states that lookup's misses.
constexpr unsigned TRACE_CAP = 1024;
struct TraceCore {
    std::uint64_t ids[TRACE_CAP];
    unsigned n = 0;
    bool over = false;
    void put(std::uint64_t id) noexcept { if (n < TRACE_CAP) ids[n++] = id; else over = true; }
    void exc_probe(std::uint32_t i) noexcept { put(TR_EXC | (i >> 2)); }
    void exc() noexcept {}
    void rtab(std::uint64_t vbyte) noexcept { put(TR_ROUTER | (vbyte >> 6)); }
    void rchild() noexcept {}
    void recs(std::uint32_t j0, std::uint32_t j) noexcept {
        std::uint64_t l[8]; const unsigned k = record_lines(j0, j, l);
        for (unsigned i = 0; i < k; ++i) put(TR_REC | l[i]);
    }
    void line(std::uint64_t off, int tag, bool) noexcept { put(TR_ARENA | (std::uint64_t(tag_region(tag)) << 52) | (off >> 6)); }
    void level() noexcept {}
    void done(bool) noexcept {}
};

// Pointer -> arena offset adapter: Count and Trace are this over Acc and TraceCore.
template<class Core> struct OnArena {
    static constexpr bool active = true;
    Core core;
    const unsigned char* arena = nullptr;
    explicit OnArena(const View& v) noexcept : arena(v.arena) {}
    std::uint64_t off(const void* p) const noexcept { return std::uint64_t(static_cast<const unsigned char*>(p) - arena); }
    void exc_probe(std::uint32_t i) noexcept { core.exc_probe(i); }
    void exc() noexcept { core.exc(); }
    void rtab(std::uint64_t b) noexcept { core.rtab(b); }
    void rchild() noexcept { core.rchild(); }
    void recs(std::uint32_t j0, std::uint32_t j) noexcept { core.recs(j0, j); }
    void dep(const void* p, int tag) noexcept { core.line(off(p), tag, true); }
    void par(const void* p, int tag) noexcept { core.line(off(p), tag, false); }
    void level() noexcept { core.level(); }
    void done(bool f) noexcept { core.done(f); }
};
using Count = OnArena<Acc>;
using Trace = OnArena<TraceCore>;

// ---------------------------------------------------------------- lookup pieces (shared with the build's walker)
template<class C> SPLICE_INLINE bool exc_lookup(const View& v, std::uint64_t x, std::uint64_t& out, C& c) noexcept {
    std::uint32_t lo = 0, len = v.nexc;
    while (len > 0) {
        const std::uint32_t half = len >> 1;
        c.exc_probe(lo + half);
        if (v.exc[lo + half].key < x) { lo += half + 1; len -= half + 1; } else len = half;
    }
    c.exc();
    const bool hit = lo < v.nexc && v.exc[lo].key == x;
    if (hit) out = v.exc[lo].val;
    c.done(hit);
    return hit;
}
// Segment j holding x (x inside the bulk range): radix top, count-split children, then a masked scan
// of the next 8 knots.
template<class C> SPLICE_INLINE std::uint32_t route(const View& v, std::uint64_t x, C& c) noexcept {
    const std::uint64_t ti = (x - v.lo) >> v.rshift;
    std::uint32_t t = v.rtop[ti];
    c.rtab(ti * 4);
    while (t >> 31) {
        const std::uint32_t ni = t & 0x7FFFFFFFu;
        const RNode& nd = v.rnodes[ni];
        const std::uint64_t si = nd.off + ((x - nd.lo) >> nd.shift);
        t = v.rsub[si];
        c.rtab(RV_NODES + std::uint64_t(ni) * 16); c.rtab(RV_SUB + si * 4); c.rchild();
    }
    const std::uint32_t j0 = t & 0x07FFFFFFu, cnt = (t >> 27) & 15u;
    const std::uint32_t le = le8(v.seg + j0 + 1, x) & ((1u << cnt) - 1u);
    const std::uint32_t j = j0 + std::uint32_t(__builtin_popcount(le));
    c.recs(j0, j);
    return j;
}
template<unsigned W, class C>
SPLICE_INLINE bool direct_lookup(const Line* R, std::uint64_t Lr, std::uint64_t p, std::uint64_t x, std::uint64_t& out, C& c) noexcept {
    std::uint64_t h = p >> 2;
    h = h < Lr - W ? h : Lr - W;
    const Line* L = R + h;
    std::uint32_t M = 0;
    for (unsigned i = 0; i < W; ++i) M |= eq4(L + i, x) << (4 * i);
    c.dep(L, T_WINDOW);
    for (unsigned i = 1; i < W; ++i) c.par(L + i, T_WINDOW);
    if (M) { const unsigned idx = low_index(M); out = L[idx >> 2].v[idx & 3]; c.done(true); return true; }
    if (L[W - 1].k[3] >= x) { c.done(false); return false; }
    for (std::uint64_t t = h + W; t < Lr; ++t) {
        const Line* l = R + t;
        c.dep(l, T_SCAN);
        const std::uint32_t m = eq4(l, x);
        if (m) { out = l->v[low_index(m)]; c.done(true); return true; }
        if (l->k[3] >= x) break;
    }
    c.done(false);
    return false;
}
// Descent from a top-level entry E; child entries and data lines are addressed from `arena` by
// line_off (the walker passes a scratch subtree here).
template<unsigned EL, class C>
SPLICE_INLINE bool fenced_lookup(const unsigned char* arena, const unsigned char* E, std::uint64_t x, std::uint64_t& out, C& c) noexcept {
    constexpr unsigned EB = 64 * EL, NV = EL == 2 ? 7 : 3;
    c.dep(E, T_ENTRY);
    if (EL == 2) c.par(E + 64, T_ENTRY);
    c.level();
    for (int lvl = 0; lvl < 8; ++lvl) {
        const EntryHdr* h = reinterpret_cast<const EntryHdr*>(E);
        std::uint32_t ci = fence_count<NV>(reinterpret_cast<const std::int16_t*>(E + 16), fence_bias(fence_q(x, h->base, h->shift)));
        if (!(h->flags & E_INNER)) {
            if (!(h->flags & E_COMPACT)) {
                const Line* D = reinterpret_cast<const Line*>(arena) + h->line_off + ci;
                c.dep(D, T_DATA);
                for (;;) {
                    const std::uint32_t m = eq4(D, x);
                    if (m) { out = D->v[low_index(m)]; c.done(true); return true; }
                    if (ci == 0 || D->k[0] <= x) break;
                    --ci; --D;
                    c.dep(D, T_TIE_LINE);
                }
            } else {
                if (x < h->base || x - h->base > 0xFFFFFFFFull) break;
                const std::uint32_t rr = std::uint32_t(x - h->base);
                const CLine* D = reinterpret_cast<const CLine*>(arena) + h->line_off + ci;
                c.dep(D, T_DATA);
                for (;;) {
                    const std::uint32_t m = eq5(D, rr);
                    if (m) { out = D->v[low_index(m)]; c.done(true); return true; }
                    if (ci == 0 || D->r[0] <= rr) break;
                    --ci; --D;
                    c.dep(D, T_TIE_LINE);
                }
            }
            break;
        }
        const unsigned char* ch = arena + (std::uint64_t(h->line_off) << 6) + std::uint64_t(ci) * EB;
        c.dep(ch, T_CHILD);
        if (EL == 2) c.par(ch + 64, T_CHILD);
        c.level();
        while (ci > 0 && reinterpret_cast<const EntryHdr*>(ch)->base > x) {
            --ci; ch -= EB;
            c.dep(ch, T_TIE_ENTRY);
            if (EL == 2) c.par(ch + 64, T_TIE_ENTRY);
        }
        E = ch;
    }
    c.done(false);
    return false;
}

template<class C> SPLICE_INLINE bool get_impl(const View& v, std::uint64_t x, std::uint64_t& out, C& c) noexcept {
    if (x - v.lo > v.span) return exc_lookup(v, x, out, c);   // fb: 21 of 200M keys; no expect hint, which made clang outline it as a call
    const std::uint32_t j = route(v, x, c);
    const Seg16* s = v.seg + j;
    const std::uint32_t meta = s->meta, base = seg_base(meta), end = seg_base(s[1].meta);
    const std::uint64_t p = predict(x, s->key, seg_pre(meta), s->m);
    const unsigned char* R = v.arena + (std::uint64_t(base) << 9);
    const std::uint64_t units = end - base;
    switch (seg_kind(meta)) {
    case 1: return direct_lookup<1>(reinterpret_cast<const Line*>(R), units * 8, p, x, out, c);
    case 2: return direct_lookup<2>(reinterpret_cast<const Line*>(R), units * 8, p, x, out, c);
    case 3: return direct_lookup<4>(reinterpret_cast<const Line*>(R), units * 8, p, x, out, c);
    default: break;
    }
    if (v.entry_lines == 2) {
        const std::uint64_t q = units * 4 - 1;
        return fenced_lookup<2>(v.arena, R + (p < q ? p : q) * 128, x, out, c);
    }
    const std::uint64_t q = units * 8 - 1;
    return fenced_lookup<1>(v.arena, R + (p < q ? p : q) * 64, x, out, c);
}
SPLICE_INLINE bool get(const View& v, std::uint64_t x, std::uint64_t& out) noexcept { NoCount c; return get_impl(v, x, out, c); }

} // namespace splice
