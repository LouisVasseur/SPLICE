#pragma once
// Joint G+T+V root ("J"): learned input compression, an order-preserving warp and CSV
// virtual fences, fitted TOGETHER at the root by guarded block coordinate descent.
//
//   G  gap removal on the input axis   u = g(x)   cut the k widest fence gaps down to the
//      median width (a k-entry table, binary-searched per lookup)
//   T  tanh-pair warp                   z = f(u)   NFL's monotone 2D2H2L shape, fitted by least
//      squares to the current slot targets (64 log-spaced gains, exact 3x3 solve, C >= 0)
//   V  CSV Algorithm 1 at the root      slot targets with a slot -> region table (smooth_cdf)
//
// Port of the Python prototypes results/aidb_threeblock/{threeblock,armA}.py (blocks and arm A's
// guarded cycle) and results/aidb_joint/joint.py (WarpBasis, GAINS). The arithmetic follows the
// prototype operation for operation (double; CPython >= 3.12's Neumaier sum() where it calls sum();
// x*(1/span) for the bare warp, x/span otherwise), so a build with -ffp-contract=off on a host whose
// long double is double (Apple arm64) replays the prototype bit for bit; V is smooth_cdf unchanged.
// Every acceptance decision is on fit_root's probe score, never on the least-squares loss.
#include "model.hpp"
#include "smoothing.hpp"
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string_view>
#include <thread>

namespace scaleli {
struct JointGap { double a = 0, w = 0, oms = 0, C = 0; };  // 32 B: gap start and raw width on the x axis, 1 - shrink factor, total shrink below
struct JointUnit { double gain = 0, coef = 0; };

// phi(k) = f(g(x(k))). Strictly increasing on every fence and fence midpoint (checked at fit time);
// exact fence search keeps lookups correct for any value anyway.
struct JointFeature {
    Key origin = 0;
    std::uint64_t span = 1;
    double scale = 1;                      // 1 - total shrink: renormalises g to [0, 1]
    std::vector<JointGap> gaps;            // sorted by a; empty = no G
    std::array<JointUnit, 2> units{};
    std::uint32_t nunits = 0;              // 0 = no T
    double x(Key k) const {
        const double d = k >= origin ? double(k - origin) : -double(origin - k);
        // The prototype's bare warp (joint.make_flow_fn) multiplies by 1/span, every other form divides.
        // Kept so that fits and probe counts replay the prototype exactly.
        return gaps.empty() && nunits ? d * (1.0 / double(span)) : d / double(span);
    }
    // Binary search over the gap starts (L1-resident table: noted as touched lines, not counted as root probes).
    double g(double x, QueryStats* s = nullptr) const {
        std::size_t lo = 0, hi = gaps.size();
        while (lo < hi) { const auto m = lo + (hi - lo) / 2; if (s) s->note(&gaps[m]); if (gaps[m].a <= x) lo = m + 1; else hi = m; }
        if (!lo) return x / scale;
        const auto& e = gaps[lo - 1];
        double t = x - e.a;
        if (t > e.w) t = e.w;
        return (x - e.C - e.oms * t) / scale;
    }
    double warp(double u) const {
        double z = 0.0;
        for (std::uint32_t h = 0; h < nunits; ++h) z += units[h].coef * std::tanh(units[h].gain * u);
        return z;
    }
    double operator()(Key k, QueryStats* s = nullptr) const {
        double u = x(k);
        if (!gaps.empty()) u = g(u, s);
        if (nunits) { if (s) ++s->transform_calls; u = warp(u); }
        return u;
    }
    // Probe-equivalents charged for the gap table: ceil(log2(K + 1)) comparisons.
    double gap_probes() const { return gaps.empty() ? 0.0 : std::ceil(std::log2(double(gaps.size() + 1))); }
};

// Owned by Index when Config::root_joint_rounds > 0: the lookup state of an adopted joint root
// (gaps are dropped when it is not adopted) plus the diagnostics of the bulk-load fit.
struct RootJoint {
    JointFeature map;
    struct PerK { std::uint32_t k = 0, k_eff = 0, rounds_run = 0, round = 0; std::uint64_t virt = 0, count = 0; double total = 0; bool candidate = false; };
    struct Round { std::uint32_t round = 0, k_eff = 0; bool accepted = false, failed = false; std::uint64_t virt = 0, count = 0; double total = 0; };
    std::vector<PerK> by_k;                // one entry per k of the grid
    std::vector<Round> trace;              // every round of the selected k
    std::vector<std::uint32_t> gap_index;  // selected gaps as fence indices (gap i = [F_i, F_i+1])
    bool candidate = false;                // some k produced a scored state
    std::uint32_t k = 0, k_eff = 0, rounds_run = 0, round = 0, rescores = 0;
    std::uint64_t virt = 0, count = 0, probe_keys = 0;
    double cost = 0, model_probes = 0, gap_probes = 0, build_ns = 0;
    double seq_total = std::numeric_limits<double>::quiet_NaN(); // k = 0 round 1: the sequential T -> V pass
    std::size_t bytes() const {
        return sizeof(RootJoint) + map.gaps.size() * sizeof(JointGap) + by_k.size() * sizeof(PerK)
             + trace.size() * sizeof(Round) + gap_index.size() * sizeof(std::uint32_t);
    }
    template<class F> void for_each_allocation(const F& f) const {
        f(static_cast<const void*>(this), sizeof(RootJoint));
        if (!map.gaps.empty()) f(static_cast<const void*>(map.gaps.data()), map.gaps.size() * sizeof(JointGap));
        if (!by_k.empty()) f(static_cast<const void*>(by_k.data()), by_k.size() * sizeof(PerK));
        if (!trace.empty()) f(static_cast<const void*>(trace.data()), trace.size() * sizeof(Round));
        if (!gap_index.empty()) f(static_cast<const void*>(gap_index.data()), gap_index.size() * sizeof(std::uint32_t));
    }
};

struct JointParams {
    unsigned rounds = 6;                   // arm A's MAX_ROUNDS
    bool tgv = false;                      // block order per round: G,T,V (false) or T,G,V
    std::size_t kmin = 0, kmax = 64;       // k grid: kmin, then every power of 4 in (kmin, kmax]
    double gap_charge = 1;                 // probe-equivalents per gap-table comparison in the score
    double alpha = 0.1;                    // V budget (Config::root_alpha)
    unsigned threads = 1;                  // k cycles are independent; results do not depend on this
};

namespace joint_detail {
// CPython >= 3.12 builtin sum() over floats (Neumaier). The prototype's numbers depend on it.
inline double pysum(const std::vector<double>& v) {
    double f = 0.0, c = 0.0;
    for (const double x : v) { const double t = f + x; c += std::abs(f) >= std::abs(x) ? (f - t) + x : (x - t) + f; f = t; }
    if (c != 0.0 && std::isfinite(c)) f += c;
    return f;
}
// joint.ols_sse: objective (*) over the real fences, plain double sums.
inline double ols_sse(const std::vector<double>& z, const std::vector<double>& s) {
    const double n = double(z.size());
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (std::size_t i = 0; i < z.size(); ++i) { const double a = z[i], b = s[i]; sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; }
    const double cxx = sxx - sx * sx / n, cxy = sxy - sx * sy / n, cyy = syy - sy * sy / n;
    const double v = cxx > 0 ? cyy - cxy * cxy / cxx : cyy;
    return v > 0 ? v : 0.0;
}
inline const std::array<double, 64>& gains() {  // joint.GAINS: log grid on [0.01, 200]
    static const std::array<double, 64> g = [] {
        std::array<double, 64> a{};
        for (int i = 0; i < 64; ++i) a[std::size_t(i)] = std::exp(std::log(0.01) + (std::log(200.0) - std::log(0.01)) * double(i) / 63.0);
        return a;
    }();
    return g;
}
// joint.WarpBasis with centre 0: tanh units tabulated over the fence coordinate u, Gram matrix precomputed.
class WarpBasis {
    std::size_t n_ = 0;
    std::vector<double> T_, S_, Q_;        // T_[a*n+i], Q_[a*64+b]
    std::vector<std::pair<std::uint16_t, std::uint16_t>> pairs_;
public:
    struct Fit { double sse = 0; std::array<JointUnit, 2> units{}; std::uint32_t nunits = 0; };
    explicit WarpBasis(const std::vector<double>& u) : n_(u.size()), T_(64 * u.size()), S_(64), Q_(64 * 64) {
        const auto& G = gains();
        std::array<bool, 64> ok{};
        std::vector<double> col(n_);
        for (std::size_t a = 0; a < 64; ++a) {
            double* t = &T_[a * n_];
            for (std::size_t i = 0; i < n_; ++i) t[i] = std::tanh(G[a] * u[i]);
            ok[a] = true;
            for (std::size_t i = 0; i + 1 < n_; ++i) if (!(t[i + 1] > t[i])) { ok[a] = false; break; }
            col.assign(t, t + n_); S_[a] = pysum(col);
        }
        for (std::size_t a = 0; a < 64; ++a) for (std::size_t b = a; b < 64; ++b) {
            const double *ta = &T_[a * n_], *tb = &T_[b * n_];
            double v = 0.0;
            for (std::size_t i = 0; i < n_; ++i) v += ta[i] * tb[i];
            Q_[a * 64 + b] = Q_[b * 64 + a] = v;
        }
        // a non-negative combination can be strictly increasing only if one of its units is
        for (std::size_t a = 0; a < 64; ++a) for (std::size_t b = a; b < 64; ++b)
            if (ok[a] || ok[b]) pairs_.emplace_back(std::uint16_t(a), std::uint16_t(b));
    }
    // argmin over (unit pair, C1, C2 >= 0, b) of sum (C1 T_a + C2 T_b + b - s)^2, walking the 400 best
    // candidates until one is strictly increasing on the fences; nullopt if none is.
    std::optional<Fit> fit(const std::vector<double>& s) const {
        const double N = double(n_), Ss = pysum(s);
        std::array<double, 64> P{};
        for (std::size_t a = 0; a < 64; ++a) { const double* ta = &T_[a * n_]; double v = 0.0; for (std::size_t i = 0; i < n_; ++i) v += ta[i] * s[i]; P[a] = v; }
        double Sss = 0.0;
        for (const double v : s) Sss += v * v;
        const double cyy = Sss - Ss * Ss / N;
        auto cov = [&](std::size_t a, std::size_t b) { return Q_[a * 64 + b] - S_[a] * S_[b] / N; };
        auto covs = [&](std::size_t a) { return P[a] - S_[a] * Ss / N; };
        auto max0 = [](double r) { return r > 0 ? r : 0.0; };
        struct Cand { double sse; std::uint16_t a, b; double C1, C2; };
        std::vector<Cand> cands; cands.reserve(pairs_.size());
        for (const auto [a, b] : pairs_) {
            const double caa = cov(a, a), cbb = cov(b, b), cab = cov(a, b), cas = covs(a), cbs = covs(b);
            bool have = false; double bs = 0, b1 = 0, b2 = 0;
            if (a != b) {
                const double det = caa * cbb - cab * cab;
                if (det > 0) {
                    const double C1 = (cbb * cas - cab * cbs) / det, C2 = (caa * cbs - cab * cas) / det;
                    if (C1 >= 0 && C2 >= 0) { bs = max0(cyy - (C1 * cas + C2 * cbs)); b1 = C1; b2 = C2; have = true; }
                }
            }
            if (!have) { // one unit active: the smaller of the two options, compared as Python tuples (sse, C1, C2)
                if (caa > 0) { const double C1 = cas / caa; if (C1 >= 0) { bs = max0(cyy - C1 * cas); b1 = C1; b2 = 0.0; have = true; } }
                if (a != b && cbb > 0) {
                    const double C2 = cbs / cbb;
                    if (C2 >= 0) {
                        const double r = max0(cyy - C2 * cbs);
                        if (!have || r < bs || (r == bs && (0.0 < b1 || (0.0 == b1 && C2 < b2)))) { bs = r; b1 = 0.0; b2 = C2; have = true; }
                    }
                }
            }
            if (have) cands.push_back({bs, a, b, b1, b2});
        }
        std::stable_sort(cands.begin(), cands.end(), [](const Cand& x, const Cand& y) { return x.sse < y.sse; });
        const auto& G = gains();
        for (std::size_t c = 0; c < std::min<std::size_t>(400, cands.size()); ++c) {
            const auto& cd = cands[c];
            if (cd.C1 <= 0 && cd.C2 <= 0) continue;
            const double *ta = &T_[cd.a * n_], *tb = &T_[cd.b * n_];
            bool good = true; double prev = cd.C1 * ta[0] + cd.C2 * tb[0];
            for (std::size_t i = 1; i < n_; ++i) { const double cur = cd.C1 * ta[i] + cd.C2 * tb[i]; if (!(cur > prev)) { good = false; break; } prev = cur; }
            if (!good) continue;
            Fit f; f.sse = cd.sse;
            if (cd.C1 > 0) f.units[f.nunits++] = {G[cd.a], cd.C1};
            if (cd.C2 > 0) f.units[f.nunits++] = {G[cd.b], cd.C2};
            return f;
        }
        return std::nullopt;
    }
};

// One state of the cycle: gap set, warp, slots (threeblock.State).
struct State {
    JointFeature f;
    std::vector<std::uint32_t> gidx;       // selected gaps (fence index), sorted
    std::vector<double> gs;                // their shrink factors s = Wmed / W
    std::vector<std::size_t> slots;        // empty: targets are ranks
    std::size_t virt = 0;
};
struct Score { std::size_t count = 0; double model_probes = 0, total = 0; LinearModel model; std::vector<std::uint32_t> table; };

// One k cycle (armA.run with the round guard). Count(model, feature, table) returns the probes fit_root's
// locate spends over the probe keys; it must be safe to call from several threads.
template<class Count> class Cycle {
    const std::vector<Key>& F_; const std::vector<Key>& P_; const JointParams& p_; const Count& count_;
    std::size_t n_; std::vector<double> xs_;
    std::vector<double> bu_; std::optional<WarpBasis> basis_;   // basis cache: T refits on an unchanged u reuse it
public:
    std::size_t k = 0;
    bool candidate = false;
    State best; Score best_score;
    std::uint32_t rounds_run = 0, round = 0;
    std::vector<RootJoint::Round> trace;
    double seq_total = std::numeric_limits<double>::quiet_NaN();

    Cycle(const std::vector<Key>& F, const std::vector<Key>& P, const JointParams& p, const Count& c, std::size_t kk)
        : F_(F), P_(P), p_(p), count_(c), n_(F.size()), xs_(F.size()), k(kk) {
        const auto s0 = empty(); for (std::size_t i = 0; i < n_; ++i) xs_[i] = s0.f.x(F_[i]); // x/span: no gaps, no warp
    }
    State empty() const {
        State s; s.f.origin = F_.front(); s.f.span = std::max<std::uint64_t>(1, F_.back() - F_.front());
        return s;
    }
    std::vector<double> feats(const State& s) const { std::vector<double> z(n_); for (std::size_t i = 0; i < n_; ++i) z[i] = s.f(F_[i]); return z; }
    // Strictly increasing on the 2n-1 probe keys; equal keys must map to equal features.
    bool monotone(const State& s) const {
        double prev = s.f(P_[0]);
        for (std::size_t i = 1; i < P_.size(); ++i) {
            const double v = s.f(P_[i]);
            if (P_[i] > P_[i - 1] ? !(v > prev) : v != prev) return false;
            prev = v;
        }
        return true;
    }
    bool set_gaps(State& s, std::vector<std::uint32_t> sel, std::vector<double> fac) const {
        s.f.gaps.clear(); double c = 0.0;
        for (std::size_t j = 0; j < sel.size(); ++j) {
            const auto i = sel[j]; JointGap e; e.a = xs_[i]; e.w = xs_[i + 1] - xs_[i]; e.oms = 1.0 - fac[j]; e.C = c;
            c += e.w * (1.0 - fac[j]); s.f.gaps.push_back(e);
        }
        s.f.scale = 1.0 - c; s.gidx = std::move(sel); s.gs = std::move(fac);
        return s.f.scale > 0.0;
    }
    // G (threeblock.apply_G, select='current'): a REPLACEMENT block, rebuilt from the raw axis each call.
    bool G(State& s) const {
        const std::size_t m = n_ - 1; std::vector<double> W(m);
        if (!s.f.nunits) { for (std::size_t i = 0; i < m; ++i) W[i] = xs_[i + 1] - xs_[i]; }
        else {
            const auto z = feats(s);
            for (std::size_t i = 0; i < m; ++i) W[i] = z[i + 1] - z[i];
            for (std::size_t j = 0; j < s.gidx.size(); ++j) W[s.gidx[j]] /= s.gs[j]; // shrunk gaps measured unshrunk, so selection cannot oscillate
        }
        std::vector<std::uint32_t> sel; std::vector<double> fac;
        if (k > 0) {
            auto w = W; std::sort(w.begin(), w.end());
            const double med = m % 2 ? w[m / 2] : 0.5 * (w[m / 2 - 1] + w[m / 2]);
            if (!(med > 0.0)) return false;
            std::vector<std::uint32_t> order(m); std::iota(order.begin(), order.end(), 0u);
            const auto top = std::min(k, m);
            std::partial_sort(order.begin(), order.begin() + std::ptrdiff_t(top), order.end(),
                              [&](std::uint32_t a, std::uint32_t b) { return W[a] > W[b] || (W[a] == W[b] && a < b); });
            for (std::size_t j = 0; j < top; ++j) if (W[order[j]] > med) sel.push_back(order[j]);
            std::sort(sel.begin(), sel.end());
            for (const auto i : sel) fac.push_back(med / W[i]);
        }
        State ns = s;
        if (!set_gaps(ns, std::move(sel), std::move(fac))) return false;
        for (const auto i : ns.gidx) if (!(ns.f.g(xs_[i + 1]) > ns.f.g(xs_[i]))) return false;
        if (!monotone(ns)) return false; // the old warp is kept; it must stay strict in the new coordinate
        s = std::move(ns); return true;
    }
    // T (threeblock.apply_T): refit the warp to the current targets in the current g coordinate; keep the
    // previous warp when no strictly increasing fit exists. Never a failure.
    void T(State& s) {
        std::vector<double> u(xs_);
        if (!s.f.gaps.empty()) for (auto& v : u) v = s.f.g(v);
        if (!basis_ || bu_ != u) { basis_.emplace(u); bu_ = std::move(u); }
        std::vector<double> t(n_);
        for (std::size_t i = 0; i < n_; ++i) t[i] = s.slots.empty() ? double(i) : double(s.slots[i]);
        const auto w = basis_->fit(t);
        if (!w) return;
        State ns = s; ns.f.units = w->units; ns.f.nunits = w->nunits;
        if (monotone(ns)) s = std::move(ns);
    }
    // V (threeblock.apply_V, guard on): CSV greedy in the current feature, previous slots kept when no worse on (*).
    bool V(State& s) const {
        const auto z = feats(s);
        for (std::size_t i = 0; i + 1 < n_; ++i) if (!(z[i + 1] > z[i])) return false;
        auto sm = smooth_cdf(z, p_.alpha);
        std::vector<double> t(n_);
        for (std::size_t i = 0; i < n_; ++i) t[i] = double(sm.slot[i]);
        std::size_t virt = sm.virtual_features.size(); auto slots = std::move(sm.slot);
        if (!s.slots.empty()) {
            const double loss = ols_sse(z, t);
            for (std::size_t i = 0; i < n_; ++i) t[i] = double(s.slots[i]);
            if (ols_sse(z, t) <= loss) { slots = s.slots; virt = slots[n_ - 1] + 1 - n_; }
        }
        s.virt = virt;
        if (virt) s.slots = std::move(slots); else s.slots.clear();
        return true;
    }
    std::optional<Score> score(const State& s) const {
        if (!monotone(s)) return std::nullopt;
        Score sc; const auto x = feats(s); std::vector<double> t(n_);
        for (std::size_t i = 0; i < n_; ++i) t[i] = s.slots.empty() ? double(i) : double(s.slots[i]);
        if (s.virt) { // as fit_root's slot table
            sc.table.assign(n_ + s.virt, 0);
            for (std::size_t j = 0; j < n_; ++j) { const auto end = j + 1 < n_ ? s.slots[j + 1] : n_ + s.virt; for (auto i = s.slots[j]; i < end; ++i) sc.table[i] = std::uint32_t(j); }
        }
        sc.model.fit_xy(F_, x, t);
        sc.count = count_(sc.model, s.f, sc.table);
        sc.model_probes = double(sc.count) / double(P_.size());
        sc.total = sc.model_probes + p_.gap_charge * s.f.gap_probes();
        return sc;
    }
    // armA.run: round 1 always accepted; round r >= 2 only if the total improves by more than 1e-6, else stop.
    // A failed block or score ends the cycle and keeps the last accepted state.
    void run() {
        std::optional<State> cur; double cur_total = 0;
        for (std::uint32_t r = 1; r <= p_.rounds; ++r) {
            State s = cur ? *cur : empty();
            bool ok = true;
            for (const char b : std::string_view(p_.tgv ? "TGV" : "GTV")) {
                if (b == 'G') ok = G(s); else if (b == 'T') T(s); else ok = V(s);
                if (!ok) break;
            }
            std::optional<Score> sc;
            if (ok) { sc = score(s); ok = bool(sc); }
            ++rounds_run;
            if (!ok) { RootJoint::Round rr; rr.round = r; rr.failed = true; trace.push_back(rr); break; }
            const bool acc = !cur || sc->total < cur_total - 1e-6;
            trace.push_back({r, std::uint32_t(s.gidx.size()), acc, false, s.virt, sc->count, sc->total});
            if (r == 1) seq_total = sc->total;
            if (!acc) break;
            cur_total = sc->total; best_score = std::move(*sc); round = r; candidate = true; cur = std::move(s);
        }
        if (cur) best = std::move(*cur);
    }
};

inline std::vector<std::size_t> k_grid(std::size_t kmin, std::size_t kmax) {
    std::vector<std::size_t> g{kmin};
    for (std::size_t k = 1; k <= kmax; k *= 4) if (k > kmin) g.push_back(k);
    return g;
}
} // namespace joint_detail

struct JointFit {
    RootJoint root;                        // diagnostics; root.map = the selected state's feature
    LinearModel model;                     // fitted on the fences in the selected feature
    std::vector<std::uint32_t> table;      // slot -> region (empty when V added nothing)
    double total = 0;                      // model probes + gap_charge * gap probes
    bool candidate = false, warp = false;
};
// Runs every k cycle (in parallel), keeps the minimum total, ties to the smaller k.
// F: fences (F[0] the first real key), P: fit_root's 2n-1 probe keys. Requires n >= 3.
template<class Count> JointFit fit_joint_root(const std::vector<Key>& F, const std::vector<Key>& P, const JointParams& p, const Count& count) {
    using C = joint_detail::Cycle<Count>;
    const auto grid = joint_detail::k_grid(p.kmin, p.kmax);
    std::vector<std::unique_ptr<C>> cyc;
    for (const auto k : grid) cyc.push_back(std::make_unique<C>(F, P, p, count, k));
    const unsigned workers = std::max(1u, std::min<unsigned>(p.threads, unsigned(cyc.size())));
    if (workers <= 1) { for (auto& c : cyc) c->run(); }
    else {
        std::vector<std::thread> pool; std::vector<std::exception_ptr> errors(workers);
        for (unsigned w = 0; w < workers; ++w) pool.emplace_back([&, w] { try { for (std::size_t j = w; j < cyc.size(); j += workers) cyc[j]->run(); } catch (...) { errors[w] = std::current_exception(); } });
        for (auto& t : pool) t.join();
        for (auto& e : errors) if (e) std::rethrow_exception(e);
    }
    JointFit out; auto& R = out.root; R.probe_keys = P.size();
    const C* best = nullptr;
    for (const auto& c : cyc) {
        RootJoint::PerK pk; pk.k = std::uint32_t(c->k); pk.rounds_run = c->rounds_run; pk.candidate = c->candidate;
        if (c->candidate) { pk.k_eff = std::uint32_t(c->best.gidx.size()); pk.round = c->round; pk.virt = c->best.virt; pk.count = c->best_score.count; pk.total = c->best_score.total; }
        R.by_k.push_back(pk);
        if (c->k == 0) R.seq_total = c->seq_total;
        if (c->candidate && (!best || c->best_score.total < best->best_score.total)) best = c.get();
    }
    if (!best) return out;
    R.candidate = out.candidate = true; R.map = best->best.f; R.gap_index = best->best.gidx; R.trace = best->trace;
    R.k = std::uint32_t(best->k); R.k_eff = std::uint32_t(best->best.gidx.size()); R.rounds_run = best->rounds_run; R.round = best->round;
    R.virt = best->best.virt; R.count = best->best_score.count; R.model_probes = best->best_score.model_probes; R.gap_probes = R.map.gap_probes();
    out.model = best->best_score.model; out.table = best->best_score.table; out.total = best->best_score.total; out.warp = R.map.nunits > 0;
    return out;
}
} // namespace scaleli
