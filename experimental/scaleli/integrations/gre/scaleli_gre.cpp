// SCALE-LI facade for GRE: the C++20 translation unit behind scaleli_gre.hpp (see that header and INTEGRATION.md).
// The Config is built from a synthetic scaleli_bench argv by scaleli_bench's own parser (scaleli/cli.hpp), and the
// memory and learnability lines are cli.hpp's JSON, so a GRE index is by construction the index scaleli_bench builds
// for the same cell, and the printed scaleli_config line reproduces it with scaleli_bench.
#include "scaleli_gre.hpp"
#include "scaleli/cli.hpp"
#include "scaleli/workload.hpp"  // mix64
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace scaleli;
namespace {
using Clock = std::chrono::steady_clock;
long long since(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t).count();
}
[[noreturn]] void die(const std::string& m) {
    std::fflush(stdout);
    std::fprintf(stderr, "scaleli_error: %s\n", m.c_str());
    std::exit(2);
}

// Flags a runner may add through SCALELI_ARGS: Config only. Workload, data and the flow file stay out (GRE owns the
// workload; the flow comes from SCALELI_FLOW alone, so a log names one flow file).
const std::set<std::string> kConfigFlags = {
    "region-keys", "block-keys", "delta-limit", "restart", "routing", "policy", "codec", "min-saving", "flow-bypass",
    "flow-min-gain", "virtual-alpha", "relearn", "fusion", "flow-cost", "build-threads", "root", "root-alpha",
    "smooth-scale", "hot-enter", "hot-exit", "heat-decay", "root-joint-rounds", "root-joint-order", "root-joint-kmin",
    "root-joint-kmax", "root-joint-gap-charge"};

// run_ba.py cells(d), the protocol definitions; J and Jg0 are the joint G+T+V root on NC's regions.
// Jg0 (gap table offered free, like N's flow) is SPLICE as reported: at gap charge 1 compression is never chosen.
bool cell_flags(const std::string& cell, const std::string& flow, std::vector<std::string>& out, bool& needs_flow) {
    const std::vector<std::string> B = {"--policy", "min_bytes", "--routing", "rank", "--root", "model"};
    const std::vector<std::string> nfl = {"--flow", flow, "--flow-bypass", "1", "--flow-cost", "0"};
    const std::vector<std::string> csv = {"--virtual-alpha", "0.1", "--root-alpha", "0.1"};
    const std::vector<std::string> joint = {"--root-joint-rounds", "6", "--root-joint-order", "gtv",
                                            "--root-joint-kmin", "0", "--root-joint-kmax", "64"};
    auto cat = [&](std::initializer_list<const std::vector<std::string>*> parts) {
        out.clear();
        for (auto* p : parts) out.insert(out.end(), p->begin(), p->end());
    };
    needs_flow = cell == "N" || cell == "NC" || cell == "J" || cell == "Jg0";
    if (cell == "B") cat({&B});
    else if (cell == "N") cat({&B, &nfl});
    else if (cell == "C") cat({&B, &csv});
    else if (cell == "Cr") { cat({&B}); out.push_back("--root-alpha"); out.push_back("0.1"); }  // CSV at the root only
    else if (cell == "NC") cat({&B, &csv, &nfl});
    else if (cell == "J" || cell == "Jg0") {
        cat({&B, &csv, &nfl, &joint});
        out.push_back("--root-joint-gap-charge");
        out.push_back(cell == "J" ? "1" : "0");
    } else return false;
    return true;
}

// GRE's own argv is not ours: run the synthetic one through scaleli_bench's parser (cli.hpp Args), which also
// rejects any flag scaleli_bench would reject.
Args args_of(const std::vector<std::string>& v) {
    std::vector<std::string> store = {"scaleli_gre"};
    store.insert(store.end(), v.begin(), v.end());
    std::vector<char*> argv;
    for (auto& x : store) argv.push_back(x.data());
    return Args(int(argv.size()), argv.data());
}
std::string root_name(const Index& ix) {
    const auto l = ix.learnability();
    if (!l.root_model) return "binary";
    return std::string(l.root_joint ? "joint" : l.root_flow ? "flow" : "raw") + (l.root_vp ? "_vp" : "");
}
}  // namespace

struct scaleli_gre::Handle {
    std::string cell, flow_path;
    std::vector<std::string> argv;
    std::optional<FlowTransform> flow;  // Config::flow points here: a Handle is heap-allocated and never moves
    Config cfg;
    std::unique_ptr<Index> ix;
};

// Every entry point: an exception is a failed run, reported and turned into exit status 2. Variadic: bodies hold commas.
#define SCALELI_GRE_GUARD(...) try { __VA_ARGS__ } catch (const std::exception& e) { die(e.what()); } catch (...) { die("unknown exception"); }

namespace scaleli_gre {
Handle* create(const char* cell) {
    SCALELI_GRE_GUARD(
        auto h = std::make_unique<Handle>();
        h->cell = cell ? cell : "";
        const char* fenv = std::getenv("SCALELI_FLOW");
        bool needs_flow = false;
        if (!cell_flags(h->cell, fenv ? fenv : "", h->argv, needs_flow))
            throw std::invalid_argument("unknown cell '" + h->cell + "' (B N C Cr NC J Jg0)");
        if (needs_flow && (!fenv || !*fenv))
            throw std::invalid_argument("cell " + h->cell + " needs SCALELI_FLOW=<flows_free/<dataset>_<period>_t2000.txt>");
        h->flow_path = needs_flow ? fenv : "none";
        const char* tenv = std::getenv("SCALELI_BUILD_THREADS");
        h->argv.push_back("--build-threads");
        h->argv.push_back(tenv && *tenv ? tenv : "16");
        if (const char* extra = std::getenv("SCALELI_ARGS")) {
            std::istringstream in(extra);
            std::vector<std::string> more;
            for (std::string t; in >> t;) more.push_back(t);
            for (const auto& [k, v] : args_of(more).a)
                if (!kConfigFlags.count(k)) throw std::invalid_argument("SCALELI_ARGS: not a Config flag: --" + k);
            h->argv.insert(h->argv.end(), more.begin(), more.end());
        }
        h->cfg = config_from_args(args_of(h->argv), h->flow);
        h->ix = std::make_unique<Index>(h->cfg);
        return h.release();)
}
void destroy(Handle* h) { delete h; }

void bulk_load(Handle* h, const kv_t* kv, std::size_t n) {
    SCALELI_GRE_GUARD(
        static_assert(std::is_same_v<kv_t, Record>, "GRE pairs are bulk-loaded as scaleli::Record without conversion");
        std::vector<Record> rows(kv, kv + n);  // by value, as scaleli_bench hands w.initial to Index::bulk_load
        auto t = Clock::now();
        h->ix->bulk_load(std::move(rows));
        const long long load_ns = since(t);
        t = Clock::now();
        const auto l = h->ix->learnability();
        const auto mem = h->ix->memory();
        std::string config;
        for (const auto& x : h->argv) config += (config.empty() ? "" : " ") + x;
        std::printf("scaleli_cell: %s\n", h->cell.c_str());
        std::printf("scaleli_config: %s\n", config.c_str());
        std::printf("scaleli_flow_file: %s\n", h->flow_path.c_str());
        std::printf("scaleli_flow_bytes: %zu\n", h->cfg.flow ? h->cfg.flow->bytes() : std::size_t(0));
        std::printf("scaleli_build_threads: %u\n", h->cfg.build_threads);
        std::printf("scaleli_keys: %zu\n", h->ix->size());
        std::printf("scaleli_bulk_load_ns: %lld\n", load_ns);
        std::printf("scaleli_accounted_bytes: %llu\n", static_cast<unsigned long long>(mem.accounted_bytes()));
        std::printf("scaleli_memory: %s\n", memory_json(mem).c_str());
        std::printf("scaleli_root: %s\n", root_name(*h->ix).c_str());
        std::printf("scaleli_flow_regions: %zu\n", l.flow_regions);
        std::printf("scaleli_virtual_points: %zu\n", l.virtual_points);
        std::printf("scaleli_root_virtual: %zu\n", std::size_t(l.root_virtual));
        std::printf("scaleli_learnability: %s\n", learnability_json(*h->ix, h->cfg).c_str());
        std::printf("scaleli_stats_ns: %lld\n", since(t));  // inside GRE's build_ns window; printed so it can be removed
        std::fflush(stdout);)
}
bool get(Handle* h, std::uint64_t key, std::uint64_t* val) {
    SCALELI_GRE_GUARD(
        const auto v = h->ix->find(key);
        if (!v) return false;
        *val = *v;
        return true;)
}
bool put(Handle* h, std::uint64_t key, std::uint64_t val) { SCALELI_GRE_GUARD(return h->ix->upsert(key, val);) }
// find + upsert: two traversals, but no change to the core Index. GRE's update ops are a minority mix at most.
bool update(Handle* h, std::uint64_t key, std::uint64_t val) {
    SCALELI_GRE_GUARD(
        if (!h->ix->find(key)) return false;
        h->ix->upsert(key, val);
        return true;)
}
bool remove(Handle* h, std::uint64_t key) { SCALELI_GRE_GUARD(return h->ix->erase(key);) }
std::size_t scan(Handle* h, std::uint64_t lo, std::size_t n, kv_t* out) {
    SCALELI_GRE_GUARD(
        const auto rows = h->ix->scan(lo, n);
        std::copy(rows.begin(), rows.end(), out);
        return rows.size();)
}
long long accounted_bytes(Handle* h) {
    SCALELI_GRE_GUARD(return static_cast<long long>(h->ix->memory().accounted_bytes());)
}
void print_line(Handle*, const char* key, long long value) {
    std::printf("%s: %lld\n", key, value);
    std::fflush(stdout);
}
std::uint64_t replay_reads(Handle* h, const std::uint64_t* keys, std::size_t n, Counters* c) {
    SCALELI_GRE_GUARD(
        QueryStats s;
        std::uint64_t d = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto v = h->ix->find(keys[i], c ? &s : nullptr);
            d = mix64(d ^ (v ? mix64(*v) ^ mix64(keys[i]) : 0));
        }
        if (c) *c = {s.root_probes, s.coordinate_probes, s.fence_probes, s.delta_probes, s.key_at_calls, s.decoded_keys,
                     s.codec_bytes_examined, s.block_routes, s.correction_distance, s.max_correction_distance,
                     s.transform_calls};
        return d;)
}
std::uint64_t direct_reads(Handle* h, const std::uint64_t* keys, std::size_t n) {
    SCALELI_GRE_GUARD(
        std::uint64_t found = 0;
        for (std::size_t i = 0; i < n; ++i) found += h->ix->find(keys[i]).has_value();
        return found;)
}
}  // namespace scaleli_gre
