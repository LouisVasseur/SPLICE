#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
# splice_core and splice_gre_check read the Mac-side 2M samples (scripts/prepare_data.sh); the Atom has none.
EX=()
[ -e data/samples/fb_2M_window_s42 ] || EX=(-E 'splice_core|splice_gre_check')
(cd build && ctest --output-on-failure ${EX[@]+"${EX[@]}"})
python3 -m unittest discover -s src/tests -p 'test_*.py' -v
