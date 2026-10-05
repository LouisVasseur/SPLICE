// SCALE-LI facade for GRE (gre4index/GRE). C++11-compatible, std types only: GRE compiles its whole benchmark as ONE
// C++17 translation unit that also includes ALEX, whose std::allocator usage is gone in C++20, so SCALE-LI (C++20)
// lives in its own TU (scaleli_gre.cpp) behind this opaque handle.
//
// Cells (mirror results/aidb_ba/run_ba.py cells(d); README.md "GRE cells"):
//   B   --policy min_bytes --routing rank --root model
//   N   B  + --flow $SCALELI_FLOW --flow-bypass 1 --flow-cost 0      (our NFL)
//   C   B  + --virtual-alpha 0.1 --root-alpha 0.1                      (our CSV)
//   NC  C  + N's flow flags
//   J   NC + joint G+T+V root, gap table charged (--root-joint-gap-charge 1)
//   Jg0 NC + joint G+T+V root, gap table free    (--root-joint-gap-charge 0); GRE name also scaleli_splice
//   Cr  B + CSV virtual fences at the root only  (--root-alpha 0.1)
// Environment: SCALELI_FLOW (required for N, NC, J, Jg0), SCALELI_BUILD_THREADS (default 16, as run_ba.py),
// SCALELI_ARGS (optional extra Config flags, appended last so they win).
// Every error prints "scaleli_error: ..." to stderr and exits with status 2: GRE itself exits 0 on a bad index,
// so a non-zero status is what lets a runner see the failure.
#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>

namespace scaleli_gre {
typedef std::pair<std::uint64_t, std::uint64_t> kv_t;  // same type as GRE's std::pair<KEY, PAYLOAD> and scaleli::Record
struct Handle;                                          // opaque

Handle* create(const char* cell);                       // validates the cell, the flow file and the Config up front
void destroy(Handle* h);
// Copies the n records (GRE passes them sorted and unique) and bulk-loads. Prints the scaleli_* stats lines once.
void bulk_load(Handle* h, const kv_t* kv, std::size_t n);
bool get(Handle* h, std::uint64_t key, std::uint64_t* val);
bool put(Handle* h, std::uint64_t key, std::uint64_t val);     // true iff the key was new (an existing key is overwritten)
bool update(Handle* h, std::uint64_t key, std::uint64_t val);  // true iff the key existed; absent keys are not inserted
bool remove(Handle* h, std::uint64_t key);
std::size_t scan(Handle* h, std::uint64_t lo, std::size_t n, kv_t* out);
long long accounted_bytes(Handle* h);                   // Index::memory().accounted_bytes(): live bytes the index owns
void print_line(Handle* h, const char* key, long long value);  // "key: value", flushed (for after-run lines)

// Test hooks (facade_check.cpp), not used by GRE.
struct Counters {  // scaleli::QueryStats work counters, same names as scaleli_bench's work_counters
    std::uint64_t root_probes, coordinate_probes, fence_probes, delta_probes, key_at_calls, decoded_keys,
        codec_bytes_examined, block_routes, correction_distance, max_correction_distance, transform_calls;
};
// find() over keys with counters; returns scaleli_bench's read digest chain d = mix64(d ^ (v ? mix64(v)^mix64(k) : 0)).
std::uint64_t replay_reads(Handle* h, const std::uint64_t* keys, std::size_t n, Counters* c);
// The same n finds as get(), but looped inside the facade TU (no cross-TU call): the overhead reference.
std::uint64_t direct_reads(Handle* h, const std::uint64_t* keys, std::size_t n);
}  // namespace scaleli_gre
