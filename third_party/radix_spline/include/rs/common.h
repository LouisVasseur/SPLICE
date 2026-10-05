#pragma once
#include <cstddef>
#include <cstdint>
namespace rs {
template <class KeyType> struct Coord { KeyType x; double y; };
struct SearchBound { size_t begin; size_t end; }; // Exclusive end.
}
