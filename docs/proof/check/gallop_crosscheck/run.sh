#!/bin/sh
# Cross-check of Theorem 4.3's transcription: NFLProof.gallop (docs/proof/lean/NFLProof/Gallop.lean)
# against the repo's C++ Index::last_true / Index::locate_from_prediction (src/scaleli/index.hpp,
# table-free branch), on all 116,795 cases with n <= 70. Prints IDENTICAL on success.
# Needs a C++20 compiler and the built Lean project (cd docs/proof/lean && lake build).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
OUT=${TMPDIR:-/tmp}/gallop_crosscheck
mkdir -p "$OUT"
# the two member functions, verbatim
awk '/std::size_t last_true\(Ok ok/ {p = 1}
     p {print}
     p && done && /^    }$/ {exit}
     p && /return last_true\(ok,lo,hi\);$/ {done = 1}' "$REPO/src/scaleli/index.hpp" > "$OUT/snippet.inc"
c++ -std=c++20 -O1 -I "$OUT" -o "$OUT/harness" "$HERE/harness.cpp"
"$OUT/harness" > "$OUT/cpp.txt"
(cd "$REPO/docs/proof/lean" && lake env lean --run "$HERE/Dump.lean") > "$OUT/lean.txt"
wc -l < "$OUT/cpp.txt"
cmp -s "$OUT/cpp.txt" "$OUT/lean.txt" && echo IDENTICAL || { echo DIFFERENT; exit 1; }
