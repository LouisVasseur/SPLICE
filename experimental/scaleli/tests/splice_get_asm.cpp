// The GRE read path in isolation, for tools/splice_asm_check.sh: splice::get with the NoCount policy,
// compiled as GRE compiles it (C++17, -O3, -march=goldmont).
#include "splice/layout.hpp"
extern "C" bool splice_get_probe(const splice::View* v, std::uint64_t key, std::uint64_t* out) { return splice::get(*v, key, *out); }
