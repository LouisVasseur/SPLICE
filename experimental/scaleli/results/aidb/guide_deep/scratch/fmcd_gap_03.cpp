// Scratch for guide section 03: LIPP's FMCD bulk-load fit and the conflict degree CD,
// re-typed from include/scaleli/hardness.hpp (fmcd_fit, FmcdModel::predict,
// conflict_degree) with ONE change: the gap count is a command-line argument instead
// of LIPP's compute_gap_count(size), so the array length L = size * (gap + 1) can be
// set to N (gap 0, "keys per rank") or 2N (gap 1, LIPP's value at 200M keys).
// Usage: fmcd_gap_03 <sosd-uint64-file> <gap> [limit]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <limits>
#include <vector>
#include <algorithm>
using U128 = unsigned __int128; using I128 = __int128;
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s file gap [limit]\n", argv[0]); return 2; }
    const int gap = std::atoi(argv[2]); const std::size_t limit = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 0;
    std::ifstream in(argv[1], std::ios::binary); if (!in) { std::fprintf(stderr, "cannot open\n"); return 2; }
    std::uint64_t n = 0; in.read(reinterpret_cast<char*>(&n), 8); if (limit && limit < n) n = limit;
    std::vector<std::uint64_t> keys(n); in.read(reinterpret_cast<char*>(keys.data()), 8 * n);
    if (!std::is_sorted(keys.begin(), keys.end())) { std::fprintf(stderr, "not sorted\n"); return 2; }
    const std::size_t size = keys.size(); const double ut_epsilon = 1e-6;
    const std::size_t capacity = size * std::size_t(gap + 1);
    auto diff = [&](std::size_t hi, std::size_t lo) -> long double { return static_cast<long double>(static_cast<I128>(keys[hi]) - static_cast<I128>(keys[lo])); };
    const long double L = static_cast<long double>(capacity);
    auto ut_for = [&](std::size_t D) { return static_cast<double>(diff(size - 1 - D, D) / (L - 2) + static_cast<long double>(ut_epsilon)); };
    auto gap_ge = [&](std::size_t i, std::size_t D, double Ut) { return static_cast<double>(static_cast<U128>(keys[i + D]) - static_cast<U128>(keys[i])) >= Ut; };
    std::size_t i = 0, D = 1; double Ut = ut_for(D);
    while (i < size - 1 - D) {
        while (i + D < size && gap_ge(i, D, Ut)) ++i;
        if (i + D >= size) break;
        ++D;
        if (D * 3 > size) break;
        Ut = ut_for(D);
    }
    bool fallback = false; long double a, base, anchor; I128 anchor2;
    if (D * 3 <= size) { a = 1.0L / static_cast<long double>(Ut); base = L / 2; anchor2 = static_cast<I128>(keys[size - 1 - D]) + static_cast<I128>(keys[D]); anchor = static_cast<long double>(anchor2) / 2; }
    else { fallback = true; const std::size_t mid1 = (size - 1) / 3, mid2 = (size - 1) * 2 / 3, g = std::size_t(gap + 1);
        const long double t1 = static_cast<long double>(mid1 * g + g / 2), t2 = static_cast<long double>(mid2 * g + g / 2);
        anchor2 = static_cast<I128>(keys[mid1]) + static_cast<I128>(keys[mid1 + 1]); anchor = static_cast<long double>(anchor2) / 2;
        const long double dk = static_cast<long double>(static_cast<I128>(keys[mid2]) + static_cast<I128>(keys[mid2 + 1]) - anchor2) / 2;
        a = (t2 - t1) / dk; base = t1; }
    auto predict = [&](std::uint64_t key) -> std::size_t {
        long double v = a * (static_cast<long double>(2 * static_cast<I128>(key) - anchor2) / 2) + base;
        const double p = static_cast<double>(v);
        if (std::isnan(p)) return 0;
        if (p > double(std::numeric_limits<int>::max() / 2)) return capacity ? capacity - 1 : 0;
        if (p < 0) return 0;
        return std::min<std::size_t>(capacity ? capacity - 1 : 0, static_cast<std::size_t>(p)); };
    std::size_t best = 1, run = 1, last = predict(keys[0]), best_slot = last, clamped_lo = 0, clamped_hi = 0;
    for (std::size_t j = 0; j < size; ++j) { const auto p = predict(keys[j]); clamped_lo += p == 0; clamped_hi += p == capacity - 1; }
    for (std::size_t j = 1; j < size; ++j) { const auto p = predict(keys[j]); if (p == last) ++run; else { if (run > best) { best = run; best_slot = last; } run = 1; last = p; } }
    if (run > best) { best = run; best_slot = last; }
    std::printf("{\"file\":\"%s\",\"n\":%zu,\"gap\":%d,\"L\":%zu,\"D\":%zu,\"U_T\":%.17g,\"slope\":%.17g,\"fallback\":%s,\"CD\":%zu,\"cd_slot\":%zu,\"keys_at_slot0\":%zu,\"keys_at_slotLm1\":%zu}\n",
        argv[1], size, gap, capacity, D, Ut, (double)a, fallback ? "true" : "false", best, best_slot, clamped_lo, clamped_hi);
    return 0;
}
