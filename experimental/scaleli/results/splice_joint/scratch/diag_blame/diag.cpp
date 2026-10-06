// Blame diagnostics (scratch, counts only). Builds the repo Fit for one cell and measures:
//  A. FENCED group anatomy: over-cap share, group-size tail, and page co-location of entry+data
//     (k entries per 4 KiB page, data of those k groups in the same page, greedy spill).
//  B. DIRECT anatomy: cummax (linear-probe) lines vs bounded placement (window-only, overflow -> stash).
// usage: diag FILE EPS K1 EL CBAR COMPACT [TIER1=0] [THREADS=8]
#include "splice/build.hpp"
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
using namespace splice;

int main(int argc, char** argv) {
    if (argc < 7) { std::fprintf(stderr, "usage\n"); return 2; }
    const char* path = argv[1];
    SpliceParams p; p.eps = std::stoull(argv[2]); p.k1 = std::uint32_t(std::stoul(argv[3]));
    p.entry = std::uint8_t(std::stoul(argv[4])); p.cbar = std::uint32_t(std::stoul(argv[5])); p.compact = std::uint8_t(std::stoul(argv[6]));
    p.tier1 = argc > 7 ? std::uint8_t(std::stoul(argv[7])) : TIER1_PLA;
    const unsigned TH = argc > 8 ? unsigned(std::stoul(argv[8])) : 8;
    const int fd = ::open(path, O_RDONLY); struct stat st{}; ::fstat(fd, &st);
    void* base = ::mmap(nullptr, std::size_t(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    const auto* w = static_cast<const std::uint64_t*>(base);
    const KeySrc ks{w + 1, nullptr, 1, std::size_t(w[0])};
    Fit f = make_fit(ks, p, TH);
    const std::size_t K = f.K(), n = f.n;
    std::printf("{\"file\":\"%s\",\"n\":%zu,\"K\":%zu,\"eps\":%llu,\"EL\":%u,\"cbar\":%u,\"compact\":%u,\"tier1\":%u}\n", path, n, K,
                (unsigned long long)f.eps, unsigned(p.entry), p.cbar, unsigned(p.compact), unsigned(p.tier1));
    const unsigned EL = p.entry, EB = 64 * EL, F = EL == 2 ? 56 : 24;
    // ---------------- A. FENCED
    const unsigned KS[] = {0, 2, 3, 4, 6, 8, 12, 16, 24};   // 0 = today's layout (A/B sections)
    constexpr int NKS = 9;
    struct SegA { std::uint64_t keys = 0, overcap = 0, coloc[NKS] = {}, bytes[NKS] = {}; std::uint64_t ghist[8] = {}; };
    std::vector<SegA> sa(K);
    const auto ch = seg_chunks(f);
    par_for(ch.size() - 1, TH, [&](std::size_t c) {
        Scratch sc;
        std::vector<std::uint64_t> gb, gc;   // per group bytes and keys (by entry index)
        for (std::size_t j = ch[c]; j < ch[c + 1]; ++j) {
            const std::size_t a = f.start[j], e = f.start[j + 1], nj = e - a;
            const std::uint64_t Q = (nj + p.cbar - 1) / p.cbar;
            const std::uint32_t m = slope(Q, f.sp[j]);
            gb.assign(Q, 0); gc.assign(Q, 0);
            SegA& s = sa[j]; s.keys = nj;
            alignas(128) unsigned char top[128];
            std::size_t i = a;
            while (i < e) {
                const std::uint64_t g = predict(ks.key(i), f.knot(j), f.pre[j], m);
                std::size_t j2 = i + 1;
                while (j2 < e && predict(ks.key(j2), f.knot(j), f.pre[j], m) == g) ++j2;
                Builder<Scratch> b{ks, EL, F, p.compact != 0, sc, 0, 0};
                b.node(i, j2, top, 0);
                const std::size_t cnt = j2 - i;
                const std::uint64_t first = ks.key(i), span = ks.key(j2 - 1) - first;
                const unsigned kpl = (p.compact && span < (1ull << 32)) ? 5 : 4;
                if (cnt > std::size_t(kpl) * (F + 1)) s.overcap += cnt;
                gb[g] = roundup(b.cur, 64); gc[g] = cnt;
                const std::uint64_t r = cnt / p.cbar; s.ghist[r >= 7 ? 7 : bitlen(r)] += cnt;   // 0:<1x,1:1x,2:2-3x,3:4-7x...
                i = j2;
            }
            const std::uint64_t Qpad = roundup(Q, 512 / EB);
            std::uint64_t b0 = Qpad * EB; for (auto x : gb) b0 += x;
            s.bytes[0] = b0;
            for (int t = 1; t < NKS; ++t) {
                const unsigned k = KS[t];
                if (k * EB >= 4096) { s.bytes[t] = ~0ull; continue; }
                const std::uint64_t cap = 4096 - k * EB, pages = (Q + k - 1) / k;
                std::uint64_t spill = 0, col = 0;
                for (std::uint64_t pg = 0; pg < pages; ++pg) {
                    std::uint64_t used = 0;
                    for (std::uint64_t g = pg * k; g < std::min<std::uint64_t>(Q, (pg + 1) * k); ++g) {
                        if (used + gb[g] <= cap) { used += gb[g]; col += gc[g]; } else spill += gb[g];
                    }
                }
                s.coloc[t] = col; s.bytes[t] = pages * 4096 + spill;
            }
        }
    });
    {
        std::uint64_t keys = 0, oc = 0, gh[8] = {}, col[NKS] = {}, by[NKS] = {};
        for (auto& s : sa) { keys += s.keys; oc += s.overcap; for (int i = 0; i < 8; ++i) gh[i] += s.ghist[i]; for (int t = 0; t < NKS; ++t) { col[t] += s.coloc[t]; by[t] = by[t] == ~0ull || s.bytes[t] == ~0ull ? ~0ull : by[t] + s.bytes[t]; } }
        std::printf("{\"A\":\"fenced\",\"overcap_share\":%.4f,\"group_mass_by_size_over_cbar\":[", double(oc) / double(keys));
        for (int i = 0; i < 8; ++i) std::printf("%s%.4f", i ? "," : "", double(gh[i]) / double(keys));
        std::printf("],\"coloc\":[");
        for (int t = 0; t < NKS; ++t) if (by[t] != ~0ull) std::printf("%s{\"k\":%u,\"coloc_share\":%.4f,\"arena_B_per_key\":%.3f}", t ? "," : "", KS[t], double(col[t]) / double(keys), double(by[t]) / double(keys));
        std::printf("]}\n");
        // Per-segment choice of k by a Lagrangian on bytes: value = 0.99 D per co-located key (one 4 KiB walk saved).
        std::printf("{\"A\":\"coloc_frontier\",\"pts\":[");
        bool first = true;
        for (double lam : {0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 1e9}) {
            double saved = 0, bytes = 0;
            for (auto& s : sa) {
                int bt = 0; double bv = lam * double(s.bytes[0]);
                for (int t = 1; t < NKS; ++t) if (s.bytes[t] != ~0ull) { const double v = lam * double(s.bytes[t]) - 0.99 * double(s.coloc[t]); if (v < bv) { bv = v; bt = t; } }
                saved += 0.99 * double(s.coloc[bt]); bytes += double(s.bytes[bt]);
            }
            std::printf("%s{\"lam\":%g,\"saved_D\":%.4f,\"arena_B_per_key\":%.3f}", first ? "" : ",", lam, saved / double(keys), bytes / double(keys));
            first = false;
        }
        std::printf("]}\n");
    }
    // ---------------- B. DIRECT
    for (unsigned q : {1u, 2u, 4u}) {
        std::uint64_t cm_dep[3] = {}, st[3] = {}, keys = 0, bytes = 0, segok[3][3] = {};   // segok: keys in segs with stash <2%, <10%, <30%
        std::vector<std::array<std::uint64_t, 7>> part(ch.size() - 1);
        std::vector<std::array<std::uint64_t, 9>> sok(ch.size() - 1);
        par_for(ch.size() - 1, TH, [&](std::size_t c) {
            auto& P = part[c]; P = {}; auto& S = sok[c]; S = {};
            std::vector<std::uint8_t> occ;
            for (std::size_t j = ch[c]; j < ch[c + 1]; ++j) {
                Sum out[3]; eval_direct_seg(f, j, q, out);
                for (int wv = 0; wv < 3; ++wv) P[wv] += out[wv].dep;
                P[6] += out[0].bytes;
                const std::size_t a = f.start[j], e = f.start[j + 1], nj = e - a;
                const std::uint64_t U = direct_units(nj, q);
                const std::uint32_t m = slope(U, f.sp[j]);
                for (unsigned wv = 0; wv < 3; ++wv) {
                    const unsigned W = 1u << wv;
                    occ.assign((U + 3) / 4 + W + 1, 0);
                    std::uint64_t stash = 0, minl = 0;   // lines before minl are full or behind
                    for (std::size_t i = a; i < e; ++i) {
                        const std::uint64_t h = std::min<std::uint64_t>(predict(ks.key(i), f.knot(j), f.pre[j], m) >> 2, occ.size() - W);
                        std::uint64_t l = std::max(h, minl); bool ok = false;
                        for (; l < h + W; ++l) if (occ[l] < 4) { ++occ[l]; ok = true; break; }
                        if (!ok) ++stash; else minl = l;   // order kept: later keys never go left of an earlier key's line
                    }
                    P[3 + wv] += stash;
                    const double sh = double(stash) / double(nj);
                    S[wv * 3 + 0] += sh < 0.02 ? nj : 0; S[wv * 3 + 1] += sh < 0.10 ? nj : 0; S[wv * 3 + 2] += sh < 0.30 ? nj : 0;
                }
            }
        });
        for (auto& P : part) { for (int i = 0; i < 3; ++i) { cm_dep[i] += P[i]; st[i] += P[3 + i]; } bytes += P[6]; }
        for (auto& S : sok) for (int wv = 0; wv < 3; ++wv) for (int t = 0; t < 3; ++t) segok[wv][t] += S[wv * 3 + t];
        for (auto& s : sa) keys += s.keys;
        std::printf("{\"B\":\"direct\",\"alpha\":%.2f,\"bytes_per_key_w1\":%.2f", q / 4.0, double(bytes) / double(keys));
        for (int wv = 0; wv < 3; ++wv)
            std::printf(",\"w%u\":{\"cummax_dep_lines\":%.3f,\"bounded_stash_share\":%.4f,\"keys_in_segs_stash_lt_2_10_30\":[%.3f,%.3f,%.3f]}", 1u << wv,
                        double(cm_dep[wv]) / double(keys), double(st[wv]) / double(keys),
                        double(segok[wv][0]) / double(keys), double(segok[wv][1]) / double(keys), double(segok[wv][2]) / double(keys));
        std::printf("}\n");
    }
    return 0;
}
