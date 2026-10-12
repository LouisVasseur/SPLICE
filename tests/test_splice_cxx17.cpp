// C++17 boundary of SPLICE-H: GRE's microbench TU sees only splice/layout.hpp. This TU is compiled as
// C++17 with nothing but include/, checks the layout's static_asserts and runs a hand-built View: one
// DIRECT segment (w = 1), then the same keys as one FENCED segment (one 2-line leaf entry).
#include "splice/layout.hpp"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <new>
static_assert(__cplusplus == 201703L, "this test must stay C++17");
static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::printf("CHECK failed: %s line %d\n", #x, __LINE__); ++failures; } } while (false)
using namespace splice;
int main() {
    const std::uint64_t keys[4] = {10, 20, 30, 40};
    auto* arena = static_cast<unsigned char*>(::operator new(4096, std::align_val_t(4096)));
    alignas(64) Seg16 seg[10] = {};
    alignas(64) std::uint32_t rtop[32];
    for (auto& t : rtop) t = rleaf(0, 0);
    // DIRECT: U = n + n/4 = 5 slots over the key span 30, m = floor(5 * 2^32 / 31); 8 lines from unit 1.
    const std::uint32_t m = std::uint32_t((static_cast<unsigned __int128>(5) << 32) / 31);
    std::memset(arena, 0, 4096);
    Line* R = reinterpret_cast<Line*>(arena + 512);
    std::uint64_t nextc = 0, f[4];
    for (int i = 0; i < 4; ++i) { const std::uint64_t p = predict(keys[i], 10, 0, m); f[i] = p > nextc ? p : nextc; nextc = f[i] + 1; }
    for (std::uint64_t s = 0, i = 0; s < 32; ++s) { while (i < 3 && s > f[i]) ++i; R[s >> 2].k[s & 3] = keys[i]; R[s >> 2].v[s & 3] = keys[i] * 7; }
    Line* L0 = reinterpret_cast<Line*>(arena);
    for (int t = 0; t < 4; ++t) { L0->k[t] = 10; L0->v[t] = 70; }
    seg[0] = Seg16{10, m, seg_meta(1, 0, 1)}; seg[1] = Seg16{0, 0, seg_meta(2, 0, 0)};
    View v{}; v.lo = 10; v.span = 30; v.rtop = rtop; v.seg = seg; v.nseg = 1; v.entry_lines = 2; v.nfence = 56; v.arena = arena;
    for (int i = 0; i < 4; ++i) { std::uint64_t o = 0; CHECK(get(v, keys[i], o) && o == keys[i] * 7); }
    for (std::uint64_t x : {0ull, 9ull, 11ull, 25ull, 39ull, 41ull, ~0ull}) { std::uint64_t o = 0; CHECK(!get(v, x, o)); }
    Count c(v); std::uint64_t o;
    for (int i = 0; i < 4; ++i) get_impl(v, keys[i], o, c);
    CHECK(c.core.t.lookups == 4 && c.core.t.found == 4 && c.core.t.dep == 4 && c.core.t.direct_keys == 4 && c.core.t.par == 0);
    // FENCED: Q = 1 entry of 128 B at unit 1 (entries 1..3 EMPTY), its leaf line at arena line 16.
    std::memset(arena + 512, 0, 512);
    for (int e = 0; e < 4; ++e) {
        Entry128* E = reinterpret_cast<Entry128*>(arena + 512) + e;
        for (auto& x : E->f) x = FENCE_UNUSED;
        E->h = e ? EntryHdr{0, 0, 0, E_EMPTY, 0} : EntryHdr{10, 16, 0, 0, 1};
    }
    Line* D = reinterpret_cast<Line*>(arena) + 16;
    for (int t = 0; t < 4; ++t) { D->k[t] = keys[t]; D->v[t] = keys[t] * 7; }
    seg[0] = Seg16{10, std::uint32_t((static_cast<unsigned __int128>(1) << 32) / 31), seg_meta(1, 0, 0)};
    for (int i = 0; i < 4; ++i) { std::uint64_t o2 = 0; CHECK(get(v, keys[i], o2) && o2 == keys[i] * 7); }
    for (std::uint64_t x : {0ull, 9ull, 11ull, 25ull, 39ull, 41ull, ~0ull}) { std::uint64_t o2 = 0; CHECK(!get(v, x, o2)); }
    Count c2(v);
    for (int i = 0; i < 4; ++i) get_impl(v, keys[i], o, c2);
    CHECK(c2.core.t.dep == 8 && c2.core.t.par == 4 && c2.core.t.fenced_keys == 4 && c2.core.t.depth[0] == 4 && c2.core.t.tagdep[T_DATA] == 4);
    ::operator delete(arena, std::align_val_t(4096));
    std::printf("splice layout C++17 (%s path): %s\n", SPLICE_SSE ? "SSE4.2" : "scalar", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
