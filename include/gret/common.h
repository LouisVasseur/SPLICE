// SPDX-License-Identifier: MIT
// Constants shared by the transforms, the wrapper and the gate (namespace gret, C++17, standard library only).
#pragma once
#include <cstdint>

namespace gret {

// Every transform maps a uint64 key into [0, 2^kOutBits). 52 bits keep the outputs exact doubles, so the double-based
// models of LIPP and ALEX behave the same on arm64 (64-bit long double) and x86, and LIPP stays below its 2^53 limit.
constexpr unsigned kOutBits = 52;
constexpr uint64_t kOutMax = (uint64_t(1) << kOutBits) - 1;  // largest output, 2^52 - 1
constexpr double kOutScale = 4503599627370496.0;             // 2^52 as a double (exact)
static_assert(kOutScale == double(kOutMax) + 1.0, "kOutScale must be 2^kOutBits");

}  // namespace gret
