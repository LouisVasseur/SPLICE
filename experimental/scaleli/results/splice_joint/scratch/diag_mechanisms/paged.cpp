// Scratch count (not repo code): "paged hybrid" segment where T (repo tier-1 fit, integer predictor)
// predicts a slot in a run of 4 KiB pages; each page = 2 header lines (fences over its data lines, V)
// + D data lines; keys predicted into a page stay there (cummax + backward clamp inside the page);
// a page whose predicted load exceeds its capacity keeps C keys and spills the excess (G stash).
// Lookup model: header (2 lines) and the predicted W-line window are issued together (one walk);
// a window miss costs one more line in the same page (fence-selected); a spilled key costs a spill
// walk + line. Counts are exact over all keys; D constants are the repo's defaults.
//   paged <file> <eps> [threads]
#include "splice/cost.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstdio>
using namespace splice;

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    const int fd = ::open(argv[1], O_RDONLY); struct stat st{}; ::fstat(fd, &st);
    const auto* w = static_cast<const std::uint64_t*>(::mmap(nullptr, std::size_t(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0));
    const KeySrc ks{w + 1, nullptr, 1, std::size_t(w[0])};
    SpliceParams p; p.eps = std::stoull(argv[2]); p.k1 = argc > 4 ? std::uint32_t(std::stoul(argv[4])) : (1u << 30); p.tier1 = argc > 5 ? std::uint8_t(std::stoul(argv[5])) : TIER1_PLA;
    const unsigned TH = argc > 3 ? unsigned(std::stoul(argv[3])) : 8;
    Fit f = make_fit(ks, p, TH);
    const std::size_t K = f.K(), n = f.n;
    std::printf("{\"file\":\"%s\",\"eps\":%llu,\"tier1\":%u,\"K\":%zu", argv[1], (unsigned long long)f.eps, unsigned(p.tier1), K);
    const auto ch = seg_chunks(f);
    // configs: kpl (4 fast, 5 compact), alpha quarters
    for (unsigned kpl : {4u, 5u}) for (unsigned aq : {1u, 2u, 3u, 4u}) {
        const std::uint64_t DL = 62, C = DL * kpl;   // 62 data lines after a 2-line header
        struct T { std::uint64_t keys = 0, pages = 0, empty = 0, spill = 0, hit[3] = {}, ovpages = 0, ovkeys = 0; };
        std::vector<T> part(ch.size() - 1);
        par_for(ch.size() - 1, TH, [&](std::size_t c) {
            T& t = part[c];
            std::vector<std::uint64_t> pl;   // in-page predicted slot of the page's keys
            for (std::size_t j = ch[c]; j < ch[c + 1]; ++j) {
                const std::size_t a = f.start[j], e = f.start[j + 1], nj = e - a;
                const std::uint64_t S = nj + (nj * aq + 3) / 4, P = (S + C - 1) / C;
                const std::uint32_t m = slope(P * C, f.sp[j]);
                t.keys += nj; t.pages += P;
                std::uint64_t used = 0;
                std::size_t i = a;
                while (i < e) {
                    const std::uint64_t pp = predict(ks.key(i), f.knot(j), f.pre[j], m), g = pp / C;
                    pl.clear();
                    std::size_t i2 = i;
                    while (i2 < e) { const std::uint64_t q = predict(ks.key(i2), f.knot(j), f.pre[j], m); if (q / C != g) break; pl.push_back(q - g * C); ++i2; }
                    ++used;
                    const std::uint64_t L = pl.size(), keep = std::min<std::uint64_t>(L, C);
                    if (L > C) { t.ovpages++; t.ovkeys += L; t.spill += L - C; }
                    // keep the first `keep` keys in page (contiguous key range, boundary key in header)
                    std::vector<std::uint64_t> fpos(keep);
                    std::uint64_t prev = 0;
                    for (std::uint64_t k = 0; k < keep; ++k) { const std::uint64_t q = std::max(pl[k], k ? prev + 1 : pl[k]); fpos[k] = q; prev = q; }
                    for (std::uint64_t k = keep; k-- > 0;) { const std::uint64_t lim = (k + 1 < keep ? fpos[k + 1] - 1 : C - 1); if (fpos[k] > lim) fpos[k] = lim; }
                    for (std::uint64_t k = 0; k < keep; ++k) {
                        const std::uint64_t h = pl[k] / kpl, l = fpos[k] / kpl;
                        for (unsigned wv = 0; wv < 3; ++wv) { const std::uint64_t W = 1u << wv; const std::uint64_t hh = std::min(h, DL - W); if (l >= hh && l < hh + W) t.hit[wv]++; }
                    }
                    i = i2;
                }
                t.empty += P - used;
            }
        });
        T s; for (auto& t : part) { s.keys += t.keys; s.pages += t.pages; s.empty += t.empty; s.spill += t.spill; s.ovpages += t.ovpages; s.ovkeys += t.ovkeys; for (int k = 0; k < 3; ++k) s.hit[k] += t.hit[k]; }
        const double N = double(s.keys);
        // E[D] model (milli-D -> D): walk 0.81 + 1 (header || window) + tau*(W+1) + miss*1 + spill*(0.94+1+0.5 fence) + compute 0.45 + tier-1/router 0.155*K/16384
        const double spill = s.spill / N, rec = 0.155 * double(K) / 16384.0;
        std::printf(",\n {\"kpl\":%u,\"alpha\":%.2f,\"B_per_key_arena\":%.2f,\"empty_pages\":%.4f,\"ov_page_key_share\":%.4f,\"spill_share\":%.4f", kpl, aq / 4.0,
                    double(s.pages) * 4096 / N, double(s.empty) / double(s.pages), s.ovkeys / N, spill);
        for (unsigned wv = 0; wv < 3; ++wv) {
            const double hit = s.hit[wv] / N, resident = 1 - spill, miss = resident - hit;
            const double ed = 0.81 + 1.0 + 0.05 * ((1u << wv) + 1) + miss * 1.0 + spill * 2.44 + 0.45 + rec;
            std::printf(",\"w%u\":{\"hit\":%.4f,\"miss\":%.4f,\"ED\":%.3f}", 1u << wv, hit, miss, ed);
        }
        std::printf("}");
    }
    std::printf("\n}\n");
}
