// Per-block log-gap stats of a SOSD file (scratch tool for window selection; not part of the patch).
// usage: winstats FILE BLOCK  -> lines: block off mean_lg sd_lg lag1 p99_lg  (block -1 = whole file)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <algorithm>
struct St { double s = 0, s2 = 0, sxy = 0; double prev = 0; std::uint64_t n = 0, np = 0; std::vector<float> v; };
static void out(const char* tag, long b, std::size_t off, St& t) {
    const double m = t.s / t.n, var = t.s2 / t.n - m * m;
    const double cov = t.sxy / t.np - m * m;
    std::size_t k = t.v.size() * 99 / 100; std::nth_element(t.v.begin(), t.v.begin() + k, t.v.end());
    std::printf("%s %ld %zu %.4f %.4f %.4f %.3f\n", tag, b, off, m, std::sqrt(var), var > 0 ? cov / var : 0, t.v[k]);
}
int main(int argc, char** argv) {
    int fd = open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    auto* w = (const std::uint64_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    const std::size_t n = w[0], B = std::stoull(argv[2]); const std::uint64_t* k = w + 1;
    St all; all.v.reserve(n / 8);
    for (std::size_t b = 0; b * B < n; ++b) {
        St t; t.v.reserve(B);
        const std::size_t a = b * B, e = std::min(n, a + B);
        for (std::size_t i = std::max<std::size_t>(a, 1); i < e; ++i) {
            const double x = std::log2(double(k[i] - k[i - 1]));
            t.s += x; t.s2 += x * x; if (t.n) { t.sxy += x * t.prev; ++t.np; } t.prev = x; ++t.n; t.v.push_back(float(x));
            all.s += x; all.s2 += x * x; if (all.n) { all.sxy += x * all.prev; ++all.np; } all.prev = x; ++all.n; if (i % 8 == 0) all.v.push_back(float(x));
        }
        out("block", long(b), a, t);
    }
    out("all", -1, 0, all);
}
