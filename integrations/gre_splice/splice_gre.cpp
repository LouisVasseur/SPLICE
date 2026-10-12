// SPLICE-H facade for GRE: the C++20 translation unit behind splice_gre.hpp (see that header and README.md). It parses
// SPLICE_ARGS with splice/params.hpp, builds splice::Index directly over GRE's pair array and prints the splice_* lines.
// get() is deliberately not here: GRE's C++17 TU runs splice::get() on its own copy of the View.
#include "splice_gre.hpp"
#include "splice/build.hpp"
#include "splice/params.hpp"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
[[noreturn]] void die(const std::string& m) {
    std::fflush(stdout);
    std::fprintf(stderr, "splice_error: %s\n", m.c_str());
    std::exit(2);
}

// SPLICE_BUILD_THREADS: plain decimal in 1..1024, default 16 (gre_run.sh --build-threads, run_ba.py's 16).
unsigned env_threads() {
    const char* t = std::getenv("SPLICE_BUILD_THREADS");
    if (!t || !*t) return 16;
    unsigned v = 0;
    for (const char* c = t; *c; ++c) {
        if (*c < '0' || *c > '9' || v > 1024) throw std::invalid_argument("SPLICE_BUILD_THREADS must be 1..1024, got '" + std::string(t) + "'");
        v = v * 10 + unsigned(*c - '0');
    }
    if (v < 1 || v > 1024) throw std::invalid_argument("SPLICE_BUILD_THREADS must be 1..1024, got '" + std::string(t) + "'");
    return v;
}
}  // namespace

struct splice_gre::Handle {
    bool thp = false;
    unsigned threads = 16;
    splice::SpliceParams params;
    std::unique_ptr<splice::Index> ix;  // non-movable after build: held by pointer, the View's arena never moves
};

// Every entry point: an exception is a failed run, reported and turned into exit status 2. Variadic: bodies hold commas.
#define SPLICE_GRE_GUARD(...) try { __VA_ARGS__ } catch (const std::exception& e) { die(e.what()); } catch (...) { die("unknown exception"); }

namespace splice_gre {
// GRE's records are read in place as two adjacent u64 words with a stride of 2 (splice::Input).
static_assert(sizeof(kv_t) == 16 && std::is_standard_layout<kv_t>::value && offsetof(kv_t, first) == 0 &&
              offsetof(kv_t, second) == 8, "GRE pairs must be two adjacent uint64 words");

Handle* create(bool thp) {
    SPLICE_GRE_GUARD(
        auto h = std::make_unique<Handle>();
        h->thp = thp;
        const char* a = std::getenv("SPLICE_ARGS");
        h->params = splice::parse_params(a ? a : "", thp);  // throws "SPLICE_ARGS: ..." before any key is loaded
        h->threads = env_threads();
        if (h->params.threads == 0) h->params.threads = static_cast<decltype(h->params.threads)>(h->threads);
        else h->threads = static_cast<unsigned>(h->params.threads);
        return h.release();)
}
void destroy(Handle* h) { delete h; }

void bulk_load(Handle* h, const kv_t* kv, std::size_t n) {
    SPLICE_GRE_GUARD(
        splice::Input in{};
        in.key = n ? &kv[0].first : nullptr;
        in.val = n ? &kv[0].second : nullptr;
        in.stride_words = 2;
        in.n = n;
        h->ix.reset();  // GRE bulk-loads once; a second call starts over
        h->ix = std::make_unique<splice::Index>();
        h->ix->build(in, h->params);
        const splice::Stats& s = h->ix->stats();
        std::printf("splice_index: %s\n", h->thp ? "splice_thp" : "splice");
        std::printf("splice_args_effective: %s\n", splice::format_params(h->params).c_str());
        std::printf("splice_build_threads: %u\n", h->threads);
        std::printf("splice_keys: %zu\n", n);
        std::printf("splice_bulk_load_ns: %lld\n", h->ix->build_ns());
        std::printf("splice_hash_ns: %lld\n", h->ix->hash_ns());
        std::printf("splice_layout_hash: 0x%016llx\n", static_cast<unsigned long long>(h->ix->layout_hash()));
        std::printf("splice_full_hash: 0x%016llx\n", static_cast<unsigned long long>(h->ix->full_hash()));
        std::printf("splice_total_bytes: %lld\n", static_cast<long long>(h->ix->total_bytes()));
        std::printf("splice_arena_bytes: %lld\n", static_cast<long long>(s.arena_bytes));
        std::printf("splice_arena_anon_huge_bytes: %lld\n", h->ix->arena_anon_huge_bytes());
        std::printf("splice_predicted_ed_milli: %lld\n", static_cast<long long>(s.ed_milli[0]));  // 4K pages, 2048 KB L2
        std::printf("splice_stats: %s\n", splice::stats_json(s).c_str());
        std::fflush(stdout);)
}
const splice::View* view(const Handle* h) {
    if (!h->ix) die("view() before bulk_load");
    return &h->ix->view();
}
long long total_bytes(const Handle* h) { return h->ix ? static_cast<long long>(h->ix->total_bytes()) : 0; }
void print_line(const char* key, long long value) {
    std::printf("%s: %lld\n", key, value);
    std::fflush(stdout);
}
}  // namespace splice_gre
