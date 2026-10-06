// Headroom map (scratch, not repo code): how much of a FENCED dataset could go DIRECT (1 line + 1 walk)
// if T refined the segments, G removed holes/clusters (knots at gaps, a bounded stash) and V placed
// slack per piece. Single-threaded counts over all keys; uses splice::make_fit and splice::predict.
//   hr <sosd file> <k1> <eps> <c2 milli-D: the dataset's FENCED E[D]> [nmax keys]
#include "splice/cost.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstdio>
#include <queue>
using namespace splice;

static const std::uint64_t* load(const char* path, std::size_t& n) {
    int fd = ::open(path, O_RDONLY); if (fd < 0) { perror(path); exit(1); }
    struct stat st; fstat(fd, &st);
    void* p = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    const auto* u = static_cast<const std::uint64_t*>(p);
    n = u[0]; return u + 1;
}
static const KeySrc* KS;
constexpr int NA = 4;                       // alpha = 1/8, 1/4, 1/2, 1
static const unsigned AQ[NA] = {1, 2, 4, 8};   // U = n + ceil(n*q/8)
struct PStat {                              // one piece [a, e), per alpha
    std::uint64_t a, e;
    std::uint64_t cm_dep[NA];               // cummax placement (today's DIRECT), W=1 dependent lines
    std::uint64_t st1[NA], st2[NA];         // bounded placement: keys that cannot sit in their W=1 / W=2 window (stash)
    std::uint64_t slots[NA];
};
static void eval_piece(std::uint64_t a, std::uint64_t e, PStat& s) {
    s.a = a; s.e = e;
    const std::uint64_t n = e - a, k0 = KS->key(a), sp0 = KS->key(e - 1) - k0;
    const unsigned bl = bitlen(sp0), pre = bl > 32 ? bl - 32 : 0;
    const std::uint64_t sp = sp0 >> pre;
    for (int q = 0; q < NA; ++q) {
        const std::uint64_t U = n + (n * AQ[q] + 7) / 8; const std::uint32_t m = slope(U, sp);
        std::uint64_t nc = 0, n1 = 0, n2 = 0, dep = 0, s1 = 0, s2 = 0;
        for (std::uint64_t i = a; i < e; ++i) {
            const std::uint64_t p = predict(KS->key(i), k0, pre, m), h = p >> 2;
            const std::uint64_t ai = std::max(h, nc >> 2);
            dep += 1 + (ai > h ? ai - h : 0);
            nc = std::max(p, nc) + 1;
            { const std::uint64_t f = std::max(p, n1); if ((f >> 2) > h) ++s1; else n1 = f + 1; }
            { const std::uint64_t f = std::max(p, n2); if ((f >> 2) > h + 1) ++s2; else n2 = f + 1; }
        }
        s.cm_dep[q] = dep; s.st1[q] = s1; s.st2[q] = s2; s.slots[q] = U;
    }
}
// Split point: median by count (T alone) or the largest gap in the central half (T placed by G's holes).
static std::uint64_t split_at(std::uint64_t a, std::uint64_t e, bool gap) {
    const std::uint64_t n = e - a;
    if (!gap) return a + n / 2;
    std::uint64_t best = a + n / 2, bg = 0;
    for (std::uint64_t i = a + n / 4 + 1; i <= a + 3 * n / 4; ++i) { const std::uint64_t g = KS->key(i) - KS->key(i - 1); if (g > bg) { bg = g; best = i; } }
    return best;
}
// Cost of a piece in milli-D summed over its keys, for alpha q, window W (1/2), under a stash policy:
// home keys pay c1 (+tau for W=2); stashed keys pay c1 + cs; the piece may instead stay FENCED at c2.
struct CostM { std::int64_t c1, cs, c2, tau; };
static std::int64_t piece_cost(const PStat& s, int q, int W, const CostM& c, bool& direct) {
    const std::int64_t n = std::int64_t(s.e - s.a), st = std::int64_t(W == 1 ? s.st1[q] : s.st2[q]);
    const std::int64_t d = n * (c.c1 + (W == 2 ? c.tau : 0)) + st * c.cs, f = n * c.c2;
    direct = d < f; return direct ? d : f;
}
// Best over (alpha, W, FENCED) per piece at Lagrangian lam (milli-D per byte); returns totals.
struct Tot { double ed, bpk, dshare, stash, fenced_share; };
static Tot choose(const std::vector<PStat>& P, std::uint64_t N, const CostM& c, double lam, double fenced_bpk) {
    double J = 0, B = 0, dk = 0, stk = 0, fk = 0;
    for (const auto& s : P) {
        const double n = double(s.e - s.a);
        double bj = 1e300, bb = 0, bst = 0; bool bd = false;
        for (int q = 0; q < NA; ++q) for (int W = 1; W <= 2; ++W) {
            bool d; std::int64_t cost = piece_cost(s, q, W, c, d);
            if (!d) continue;
            const double st = double(W == 1 ? s.st1[q] : s.st2[q]);
            const double bytes = double(s.slots[q]) * 16 + st * fenced_bpk;
            if (cost + lam * bytes < bj) { bj = cost + lam * bytes; bb = bytes; bst = st; bd = true; }
        }
        const double fc = n * c.c2, fb = n * fenced_bpk;
        if (fc + lam * fb < bj) { bj = fc + lam * fb; bb = fb; bd = false; bst = 0; }
        J += bj - lam * bb; B += bb;
        if (bd) { dk += n; stk += bst; } else fk += n;
    }
    return Tot{J / double(N) / 1000.0, B / double(N), dk / double(N), stk / double(N), fk / double(N)};
}
static Tot at_cap(const std::vector<PStat>& P, std::uint64_t N, const CostM& c, double cap_bpk, double fenced_bpk, double meta_bpk) {
    Tot t = choose(P, N, c, 0, fenced_bpk);
    if (t.bpk + meta_bpk <= cap_bpk) return t;
    double lo = 0, hi = 1e-3;
    while (choose(P, N, c, hi, fenced_bpk).bpk + meta_bpk > cap_bpk && hi < 1e6) { lo = hi; hi *= 2; }
    for (int it = 0; it < 40; ++it) { const double mid = (lo + hi) / 2; if (choose(P, N, c, mid, fenced_bpk).bpk + meta_bpk > cap_bpk) lo = mid; else hi = mid; }
    return choose(P, N, c, hi, fenced_bpk);
}
static void report(const char* tag, const std::vector<PStat>& P, std::uint64_t N, double c2d, double fenced_bpk) {
    std::uint64_t cm[NA] = {}, s1[NA] = {}, s2[NA] = {};
    for (const auto& s : P) for (int q = 0; q < NA; ++q) { cm[q] += s.cm_dep[q]; s1[q] += s.st1[q]; s2[q] += s.st2[q]; }
    const double meta = double(P.size()) * 16 / double(N);
    std::printf("{\"tag\":\"%s\",\"knots\":%zu,\"cummax_dep\":[", tag, P.size());
    for (int q = 0; q < NA; ++q) std::printf("%s%.3f", q ? "," : "", double(cm[q]) / N);
    std::printf("],\"stash_w1\":[");
    for (int q = 0; q < NA; ++q) std::printf("%s%.4f", q ? "," : "", double(s1[q]) / N);
    std::printf("],\"stash_w2\":[");
    for (int q = 0; q < NA; ++q) std::printf("%s%.4f", q ? "," : "", double(s2[q]) / N);
    std::printf("]");
    // c1 = 1 line + 1 AD walk + DIRECT compute (+ record miss 0.05); stash: sequential FENCED fallback
    // (2 lines + 2 walks + 0.15), or a page-local stash line read in parallel (tau only).
    const std::int64_t c2 = std::int64_t(c2d);
    const CostM seq{2380, 4100, c2, 50}, loc{2380, 60, c2, 50};
    for (int pol = 0; pol < 2; ++pol) {
        const CostM& c = pol ? loc : seq;
        const Tot u = choose(P, N, c, 0, fenced_bpk), t = at_cap(P, N, c, 22.0, fenced_bpk, meta);
        std::printf(",\"%s\":{\"uncapped\":{\"ed\":%.3f,\"bpk\":%.2f,\"direct\":%.4f,\"stash\":%.4f},\"cap22\":{\"ed\":%.3f,\"bpk\":%.2f,\"direct\":%.4f,\"stash\":%.4f}}",
                    pol ? "stash_pagelocal" : "stash_seq", u.ed, u.bpk + meta, u.dshare, u.stash, t.ed, t.bpk + meta, t.dshare, t.stash);
    }
    std::printf("}\n"); std::fflush(stdout);
}
int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage\n"); return 2; }
    std::size_t n; const std::uint64_t* k = load(argv[1], n);
    if (argc > 5) n = std::min<std::size_t>(n, std::strtoull(argv[5], nullptr, 10));
    KeySrc ks; ks.k = k; ks.n = n; KS = &ks;
    SpliceParams p; p.k1 = std::uint32_t(std::atoi(argv[2])); p.eps = std::strtoull(argv[3], nullptr, 10);
    const double c2 = std::atof(argv[4]);
    const double fenced_bpk = 19.3;   // today's FENCED fast-compact arena bytes per key
    const long long t0 = now_ns();
    Fit f = make_fit(ks, p, 1);
    const std::uint64_t N = f.start.back() - f.start[0];
    std::fprintf(stderr, "fit K=%zu %.1fs\n", f.K(), (now_ns() - t0) / 1e9);
    std::printf("{\"file\":\"%s\",\"n\":%zu,\"K\":%zu,\"eps\":%llu,\"c2\":%.0f}\n", argv[1], n, f.K(), (unsigned long long)f.eps, c2);
    // Uniform refinement: r equal-count pieces per segment (T alone, more knots).
    for (unsigned r : {1u, 4u, 16u}) {
        std::vector<PStat> P;
        for (std::size_t j = 0; j < f.K(); ++j) {
            const std::uint64_t a = f.start[j], e = f.start[j + 1], nj = e - a;
            for (unsigned t = 0; t < r; ++t) { const std::uint64_t x = a + nj * t / r, y = a + nj * (t + 1) / r; if (y > x) { P.emplace_back(); eval_piece(x, y, P.back()); } }
        }
        char tag[32]; std::snprintf(tag, sizeof tag, "uniform_r%u", r);
        report(tag, P, N, c2, fenced_bpk);
        std::fprintf(stderr, "%s %.1fs\n", tag, (now_ns() - t0) / 1e9);
    }
    // Adaptive refinement scored by the J excess (T trained on the count): split the piece whose
    // stash-policy cost exceeds n*c1 the most; split by count median or at the largest central gap.
    for (int gap = 0; gap < 2; ++gap) {
        std::vector<PStat> P;
        for (std::size_t j = 0; j < f.K(); ++j) { P.emplace_back(); eval_piece(f.start[j], f.start[j + 1], P.back()); }
        auto excess = [&](const PStat& s) {   // best-alpha W=2 stash count, sequential fallback, ignoring bytes
            std::uint64_t b = ~0ull; for (int q = 1; q < 3; ++q) b = std::min(b, s.st2[q]); return b; };
        std::priority_queue<std::pair<std::uint64_t, std::size_t>> pq;
        for (std::size_t i = 0; i < P.size(); ++i) pq.push({excess(P[i]), i});
        const std::size_t marks[] = {16384, 32768, 65536, 131072, 262144};
        std::size_t mi = 0;
        while (mi < 5 && !pq.empty()) {
            if (P.size() >= marks[mi]) {
                char tag[48]; std::snprintf(tag, sizeof tag, "adaptive_%s_%zu", gap ? "gap" : "median", marks[mi]);
                report(tag, P, N, c2, fenced_bpk); ++mi;
                std::fprintf(stderr, "%s %.1fs\n", tag, (now_ns() - t0) / 1e9);
                continue;
            }
            auto [ex, i] = pq.top(); pq.pop();
            if (ex == 0) { // nothing left to gain: report remaining marks as is
                while (mi < 5) { char tag[48]; std::snprintf(tag, sizeof tag, "adaptive_%s_%zu_sat%zu", gap ? "gap" : "median", marks[mi], P.size()); report(tag, P, N, c2, fenced_bpk); ++mi; }
                break;
            }
            const std::uint64_t a = P[i].a, e = P[i].e;
            if (e - a < 64) continue;
            const std::uint64_t m = split_at(a, e, gap);
            if (m <= a || m >= e) continue;
            eval_piece(a, m, P[i]); P.emplace_back(); eval_piece(m, e, P.back());
            pq.push({excess(P[i]), i}); pq.push({excess(P.back()), P.size() - 1});
        }
    }
    return 0;
}
