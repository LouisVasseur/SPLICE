// Diagnostic counts for G+T+V cooperation mechanisms (scratch, not repo code).
// Uses splice::make_fit (the repo's tier-1 fit) and the repo's integer predictor; everything else is
// a count over all keys of one dataset.
//   diag <sosd file> <k1> [cbar list default 16,24,32]
#include "splice/cost.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstdio>
#include <map>
using namespace splice;

static const std::uint64_t* load(const char* path, std::size_t& n) {
    int fd = ::open(path, O_RDONLY); if (fd < 0) { perror(path); exit(1); }
    struct stat st; fstat(fd, &st);
    void* p = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    const auto* u = static_cast<const std::uint64_t*>(p);
    n = u[0]; return u + 1;
}

struct GStat {   // key-weighted group-load statistics for one (cbar) FENCED grouping
    std::uint64_t keys = 0, groups = 0, nonempty = 0, le3 = 0, le7 = 0, le9 = 0, gt100 = 0, gt228 = 0, datalines = 0;
    // co-location: [el][ki] in-page keys, spilled keys, inline keys, pages, spilled lines, overcap keys
    std::uint64_t inpage[2][8] = {}, spilled[2][8] = {}, inl[2][8] = {}, pages[2][8] = {}, splines[2][8] = {}, ovc[2][8] = {};
    void add(const GStat& o) {
        keys += o.keys; groups += o.groups; nonempty += o.nonempty; le3 += o.le3; le7 += o.le7; le9 += o.le9; gt100 += o.gt100; gt228 += o.gt228; datalines += o.datalines;
        for (int e = 0; e < 2; ++e) for (int k = 0; k < 8; ++k) { inpage[e][k] += o.inpage[e][k]; spilled[e][k] += o.spilled[e][k]; inl[e][k] += o.inl[e][k]; pages[e][k] += o.pages[e][k]; splines[e][k] += o.splines[e][k]; ovc[e][k] += o.ovc[e][k]; }
    }
};
static const unsigned KS[8] = {2, 4, 6, 8, 10, 12, 16, 20};

static void group_loads(const Fit& f, std::size_t j, std::uint32_t cbar, std::vector<std::uint32_t>& L) {
    const std::size_t a = f.start[j], e = f.start[j + 1], nj = e - a;
    const std::uint64_t Q = (nj + cbar - 1) / cbar;
    const std::uint32_t m = slope(Q, f.sp[j]);
    L.assign(Q, 0);
    for (std::size_t i = a; i < e; ++i) ++L[predict(f.ks.key(i), f.knot(j), f.pre[j], m)];
}

static void gstat_seg(const std::vector<std::uint32_t>& L, GStat& s) {
    s.groups += L.size();
    for (auto c : L) {
        if (!c) continue;
        ++s.nonempty; s.keys += c;
        if (c <= 3) s.le3 += c; if (c <= 7) s.le7 += c; if (c <= 9) s.le9 += c;
        if (c > 100) s.gt100 += c; if (c > 228) s.gt228 += c;
        s.datalines += (c + 3) / 4;
    }
    for (int el = 1; el <= 2; ++el) {
        const unsigned cap = el == 2 ? 228 : 100, inl_cap = el == 2 ? 7 : 3;
        for (int ki = 0; ki < 8; ++ki) {
            const unsigned k = KS[ki];
            if (k * el >= 64) continue;
            const std::uint64_t Cd = 64 - k * el;
            for (std::size_t g0 = 0; g0 < L.size(); g0 += k) {
                ++s.pages[el - 1][ki];
                std::uint64_t used = 0;
                for (std::size_t g = g0; g < std::min(L.size(), g0 + k); ++g) {
                    const std::uint64_t c = L[g];
                    if (!c) continue;
                    if (c <= inl_cap) { s.inl[el - 1][ki] += c; continue; }
                    if (c > cap) { s.ovc[el - 1][ki] += c; s.spilled[el - 1][ki] += c; s.splines[el - 1][ki] += (c + 3) / 4 + (c / 200 + 1) * el; continue; }
                    const std::uint64_t ln = (c + 3) / 4;
                    const std::uint64_t fit = used >= Cd ? 0 : std::min(ln, Cd - used);
                    used += fit;
                    const std::uint64_t kin = std::min<std::uint64_t>(c, fit * 4);
                    s.inpage[el - 1][ki] += kin; s.spilled[el - 1][ki] += c - kin; s.splines[el - 1][ki] += ln - fit;
                }
            }
        }
    }
}

// DIRECT with a stash: a key placed outside its W-line window goes to the stash and takes no slot.
struct DStat { std::uint64_t placed[2][3] = {}, stash[2][3] = {}, lines[2][3] = {}, scanless[2][3] = {}; };
static void direct_stash_seg(const Fit& f, std::size_t j, DStat& d) {
    const std::size_t a = f.start[j], e = f.start[j + 1], nj = e - a;
    for (unsigned qi = 0; qi < 2; ++qi) {
        const unsigned q = qi + 1;
        const std::uint64_t U = direct_units(nj, q);
        const std::uint32_t m = slope(U, f.sp[j]);
        for (unsigned wi = 0; wi < 3; ++wi) {
            const std::uint64_t W = 1u << wi;
            std::uint64_t nextc = 0, placed = 0, flast = 0;
            for (std::size_t i = a; i < e; ++i) {
                const std::uint64_t p = predict(f.ks.key(i), f.knot(j), f.pre[j], m), h = p >> 2;
                const std::uint64_t slot = std::max(p, nextc);
                if ((slot >> 2) <= h + W - 1) { ++placed; nextc = slot + 1; flast = slot; }
            }
            d.placed[qi][wi] += placed; d.stash[qi][wi] += nj - placed;
            d.lines[qi][wi] += direct_lines(U, flast, unsigned(W));
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "diag <file> <k1> [cbars]\n"); return 2; }
    std::size_t n; const std::uint64_t* k = load(argv[1], n);
    SpliceParams p; p.k1 = std::uint32_t(std::atoi(argv[2])); p.threads = 16;
    std::vector<std::uint32_t> cbars{16, 24, 32};
    if (argc > 3) { cbars.clear(); std::string s = argv[3]; std::size_t i = 0; while (i < s.size()) { auto jj = s.find(',', i); if (jj == std::string::npos) jj = s.size(); cbars.push_back(std::uint32_t(std::stoul(s.substr(i, jj - i)))); i = jj + 1; } }
    const KeySrc ks{k, nullptr, 1, n};
    Fit f = make_fit(ks, p, 16);
    const std::size_t K = f.K();
    std::printf("{\"file\":\"%s\",\"n\":%zu,\"k1\":%u,\"K\":%zu,\"eps\":%llu,\"exc\":%zu", argv[1], n, p.k1, K, (unsigned long long)f.eps, f.exc.size());
    // DIRECT + stash
    {
        std::vector<DStat> per(K);
        par_for(K, 16, [&](std::size_t j) { direct_stash_seg(f, j, per[j]); });
        DStat t;
        for (auto& d : per) for (int a = 0; a < 2; ++a) for (int w = 0; w < 3; ++w) { t.placed[a][w] += d.placed[a][w]; t.stash[a][w] += d.stash[a][w]; t.lines[a][w] += d.lines[a][w]; }
        std::printf(",\"direct_stash\":[");
        const char* sep = "";
        for (int a = 0; a < 2; ++a) for (int w = 0; w < 3; ++w) {
            std::printf("%s{\"alpha\":%s,\"w\":%d,\"placed\":%.4f,\"stash\":%.4f,\"slot_B_per_key\":%.2f}", sep, a ? "0.5" : "0.25", 1 << w,
                double(t.placed[a][w]) / double(n), double(t.stash[a][w]) / double(n), double(t.lines[a][w]) * 64 / double(n));
            sep = ",";
        }
        std::printf("]");
    }
    std::printf(",\"fenced\":[");
    const char* sep = "";
    for (auto cb : cbars) {
        std::vector<GStat> per(K);
        par_for(K, 16, [&](std::size_t j) { std::vector<std::uint32_t> L; group_loads(f, j, cb, L); gstat_seg(L, per[j]); });
        GStat t; for (auto& g : per) t.add(g);
        const double N = double(n);
        std::printf("%s{\"cbar\":%u,\"groups\":%llu,\"nonempty\":%.4f,\"share_le3\":%.4f,\"share_le7\":%.4f,\"share_le9\":%.4f,\"share_gt100\":%.4f,\"share_gt228\":%.4f,"
                    "\"B_A_e1\":%.2f,\"B_A_e2\":%.2f,\"B_data\":%.2f,\"coloc\":[",
                    sep, cb, (unsigned long long)t.groups, double(t.nonempty) / double(t.groups), t.le3 / N, t.le7 / N, t.le9 / N, t.gt100 / N, t.gt228 / N,
                    double(t.groups) * 64 / N, double(t.groups) * 128 / N, double(t.datalines) * 64 / N);
        sep = ",";
        const char* s2 = "";
        for (int el = 0; el < 2; ++el) for (int ki = 0; ki < 8; ++ki) {
            if (!t.pages[el][ki]) continue;
            std::printf("%s{\"el\":%d,\"k\":%u,\"inline\":%.4f,\"inpage\":%.4f,\"spilled\":%.4f,\"overcap\":%.4f,\"B_per_key\":%.2f}", s2, el + 1, KS[ki],
                        t.inl[el][ki] / N, t.inpage[el][ki] / N, t.spilled[el][ki] / N, t.ovc[el][ki] / N,
                        (double(t.pages[el][ki]) * 4096 + double(t.splines[el][ki]) * 64) / N);
            s2 = ",";
        }
        std::printf("]}");
    }
    std::printf("]}\n");
}
