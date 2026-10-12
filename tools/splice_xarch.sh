#!/usr/bin/env bash
# Cross-architecture identity of SPLICE-H layouts (macOS): builds splice_tests and splice_count natively
# (arm64, scalar lookup path) and as x86-64 Goldmont (SSE4.2 path, run under Rosetta), runs both on the
# 20 samples and diffs the layout-hash lists. --full also builds full fb and osm (args from
# results/splice_h/args when present) and compares their hashes.
#   tools/splice_xarch.sh [--full] [--data DIR]       (build dirs under ${SPLICE_XARCH_BUILD:-$TMPDIR/splice_xarch})
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
FULL=0; DATA="$HERE/data/external/gre"
while [ $# -gt 0 ]; do case "$1" in --full) FULL=1 ;; --data) DATA=$2; shift ;; *) echo "usage: $0 [--full] [--data DIR]"; exit 2 ;; esac; shift; done
[ "$(uname -s)" = Darwin ] || { echo "splice_xarch.sh: needs macOS with Rosetta (arm64 + x86_64 builds)"; exit 2; }
B=${SPLICE_XARCH_BUILD:-${TMPDIR:-/tmp}/splice_xarch}
SAMPLES="$HERE/data/samples"
for a in arm64 x86_64; do
  extra=""; [ $a = x86_64 ] && extra="-march=goldmont"
  cmake -S "$HERE" -B "$B/$a" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=$a -DCMAKE_CXX_FLAGS="$extra" > "$B.$a.cmake.log"
  cmake --build "$B/$a" -j 16 --target splice_tests splice_count > "$B.$a.build.log"
done
run() { if [ "$1" = x86_64 ]; then shift; arch -x86_64 "$@"; else shift; "$@"; fi; }
status=0
for a in arm64 x86_64; do
  run $a "$B/$a/splice_count" --samples "$SAMPLES" --hash-samples > "$B.$a.count.txt"
  run $a "$B/$a/splice_tests" "$SAMPLES" > "$B.$a.tests.txt" 2>&1 || { echo "splice_tests failed on $a (see $B.$a.tests.txt)"; status=1; }
  grep -E '^hash ' "$B.$a.tests.txt" > "$B.$a.tests.hash" || true
  tail -1 "$B.$a.tests.txt"
done
for k in count.txt tests.hash; do
  n=$(wc -l < "$B.arm64.$k" | tr -d ' ')
  if [ "$n" -ge 20 ] && diff "$B.arm64.$k" "$B.x86_64.$k" > /dev/null; then echo "samples ($k): $n layout hashes identical on arm64 and x86-64"
  else echo "samples ($k): DIFFER or missing"; diff "$B.arm64.$k" "$B.x86_64.$k" || true; status=1; fi
done
if [ $FULL = 1 ]; then
  for ds in fb osm; do
    args=""; f="$HERE/results/splice_h/args/$ds.args"; [ -f "$f" ] && args=$(cat "$f")
    for a in arm64 x86_64; do
      run $a "$B/$a/splice_count" --dataset $ds --data "$DATA" --grid cell --args "$args" --hash --out "$B/out_$a" | grep '^layout_hash' > "$B.$a.$ds.full"
    done
    if diff "$B.arm64.$ds.full" "$B.x86_64.$ds.full" > /dev/null; then echo "full $ds: $(cat "$B.arm64.$ds.full") identical"
    else echo "full $ds: DIFFER"; cat "$B.arm64.$ds.full" "$B.x86_64.$ds.full"; status=1; fi
  done
fi
[ $status = 0 ] && echo "splice_xarch: PASS" || echo "splice_xarch: FAIL"
exit $status
