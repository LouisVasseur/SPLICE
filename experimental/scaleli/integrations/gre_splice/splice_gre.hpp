// SPLICE-H facade for GRE (gre4index/GRE). C++17-clean, std types plus splice/layout.hpp: GRE compiles its whole
// benchmark as ONE C++17 translation unit (ALEX's std::allocator usage is gone in C++20), while the SPLICE builder
// (splice/build.hpp, params.hpp, cost.hpp) is C++20. The build therefore lives in its own TU (splice_gre.cpp) behind
// this opaque handle, and only the read-only View crosses over: GRE's TU copies it and runs splice::get() inline, so
// a lookup makes no cross-TU call (README.md).
//
// Environment, read once in create():
//   SPLICE_ARGS           k=v tokens (splice/params.hpp), later tokens win; gre_run.sh puts the per-dataset plan
//                         file first and the global SPLICE_ARGS last.
//   SPLICE_BUILD_THREADS  decimal, default 16; used when SPLICE_ARGS leaves threads=0.
// The index name sets thp (splice 0, splice_thp 1); a conflicting thp= in SPLICE_ARGS is an error.
// Every error prints "splice_error: ..." to stderr and exits with status 2: GRE itself exits 0 on a bad index, so a
// non-zero status is what lets a runner see the failure.
#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>
#include "splice/layout.hpp"

namespace splice_gre {
typedef std::pair<std::uint64_t, std::uint64_t> kv_t;  // GRE's std::pair<KEY, PAYLOAD> for uint64 keys and payloads
struct Handle;                                          // opaque

Handle* create(bool thp);  // parses and validates SPLICE_ARGS and SPLICE_BUILD_THREADS up front
void destroy(Handle* h);
// Builds over GRE's sorted, unique pairs in place (no copy: keys and payloads are read with a stride of 2 words),
// then prints the splice_* lines once (README.md).
void bulk_load(Handle* h, const kv_t* kv, std::size_t n);
const splice::View* view(const Handle* h);  // valid from bulk_load until destroy; the arena never moves
long long total_bytes(const Handle* h);      // Index::total_bytes(): arena, records, router and exceptions
void print_line(const char* key, long long value);  // "key: value", flushed (for after-run lines)
}  // namespace splice_gre
