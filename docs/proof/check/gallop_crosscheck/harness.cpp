// Runs the repo's last_true / locate_from_prediction (copied verbatim from src/scaleli/index.hpp by
// run.sh into snippet.inc) on every start p < n and answer t < n, n <= 70, with the threshold
// predicate ok(i) = (i <= t). Prints "n p t answer probes", one line per case.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>
struct QueryStats { void note(const void*) {} };
struct Idx {
#include "snippet.inc"
};
int main() {
    Idx ix;
    for (std::size_t n = 1; n <= 70; ++n)
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t t = 0; t < n; ++t) {
                std::size_t c = 0;
                auto ok = [&](std::size_t i) { ++c; return i <= t; };
                const std::size_t r = ix.locate_from_prediction(ok, double(p), n);
                std::printf("%zu %zu %zu %zu %zu\n", n, p, t, r, c);
            }
}
