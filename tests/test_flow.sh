#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Unit test of MolFlow (include/gret/flow.h) against its trainer (transform/train_flow.py), on synthetic keys:
#   1. tests/flow_eval.cpp self-test: the inline exp against std::exp, u on keys far outside [lo, hi];
#   2. a tiny flow trained on clusters plus outliers up to 2^64-1, then C++ and torch evaluate the same file:
#      |du| < 1e-12, T consistent with u, u and T non-decreasing on the sorted keys, the report's fields;
#   3. the loader rejects malformed files (exit code 3, so that a sanitizer report, exit 1, is not a pass).
#
#   tests/test_flow.sh [WORK_DIR]       default: GRET_BUILD/test_flow under tests/run_tests.sh, else a temporary directory
#   GRET_CXX  compiler wrapper with the test flags, set by tests/run_tests.sh; without it, CXX (default c++) is called
#             with -O1 -g -fsanitize=address,undefined, e.g. CXX="taskpolicy -b nice -n 20 clang++" during a timed run
#   PYTHON    interpreter with numpy and torch (default: the first of python3, /usr/bin/python3 that has both)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${1:-${GRET_BUILD:+$GRET_BUILD/test_flow}}
WORK=${WORK:-$(mktemp -d)}
mkdir -p "$WORK"
if [ -n "${GRET_CXX:-}" ]; then COMPILE=("$GRET_CXX")
else COMPILE=(${CXX:-c++} -std=c++17 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all
              -fno-omit-frame-pointer -Wall -Wextra -I"$ROOT/include"); fi
if [ -z "${PYTHON:-}" ]; then
    for c in python3 /usr/bin/python3; do
        if "$c" -c 'import numpy, torch' >/dev/null 2>&1; then PYTHON=$c; break; fi
    done
    [ -n "${PYTHON:-}" ] || { echo "test_flow.sh: no python3 with numpy and torch; set PYTHON" >&2; exit 1; }
fi
echo "== test_flow: work dir $WORK, python $PYTHON"

"${COMPILE[@]}" "$ROOT/tests/flow_eval.cpp" -o "$WORK/flow_eval"
"$WORK/flow_eval" --selftest

# synthetic GRE key file (unsorted on purpose): three clusters of different widths, a uniform background, outliers
"$PYTHON" - "$WORK/keys" "$WORK/probe.txt" <<'EOF'
import sys
import numpy as np
rng = np.random.default_rng(7)
parts = [rng.normal(1e12, 1e9, 30000), rng.normal(5e12, 1e8, 20000), rng.exponential(2e11, 20000) + 8e12,
         rng.uniform(0, 1e13, 10000)]
k = np.concatenate([np.clip(p, 0, None).astype(np.uint64) for p in parts])
k = np.concatenate([k, np.array([2**63, 2**64 - 2, 2**64 - 1], dtype=np.uint64)])
k = np.unique(k)
rng.shuffle(k)
with open(sys.argv[1], 'wb') as f:
    np.array([len(k)], dtype='<u8').tofile(f)
    k.astype('<u8').tofile(f)
# probes: the keys, sorted, plus keys between them and far outside
probe = np.unique(np.concatenate([k, k[:5000] + np.uint64(1), np.array([0, 1, 2**53, 2**62], dtype=np.uint64),
                                  rng.integers(0, 2**63, 2000, dtype=np.uint64)]))
np.savetxt(sys.argv[2], probe, fmt='%d')
EOF

"$PYTHON" "$ROOT/transform/train_flow.py" "$WORK/keys" "$WORK/t.flow" --m 8 --steps 300 --batch 8192 \
    --sample 50000 --threads 1 --report "$WORK/t.json"
"$WORK/flow_eval" --check "$WORK/t.flow"
"$WORK/flow_eval" "$WORK/t.flow" < "$WORK/probe.txt" > "$WORK/cpp.txt"

"$PYTHON" - "$ROOT/transform" "$WORK" <<'EOF'
import json
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import train_flow as tf
w = sys.argv[2]
probe = np.loadtxt(w + '/probe.txt', dtype=np.uint64)
cpp = np.loadtxt(w + '/cpp.txt', dtype=np.float64, usecols=0)
T_cpp = np.loadtxt(w + '/cpp.txt', dtype=np.uint64, usecols=1)
lo, hi, wt, mu, s = tf.read_flow(w + '/t.flow')
u = tf.u_of_x(tf.to_x(probe, lo, hi), wt, mu, s)
bad = []
du = float(np.max(np.abs(cpp - u)))
if not du < 1e-12: bad.append('max |du| = %.3g' % du)
if np.any(T_cpp != tf.to_T(cpp)): bad.append('C++ T is not clamp(floor(u 2^52))')
dT = int(np.max(np.abs(T_cpp.astype(np.int64) - tf.to_T(u).astype(np.int64))))
if dT > 2 ** 52 * 1e-12 + 1: bad.append('max |dT| = %d' % dT)
dec_u, dec_T = int(np.sum(cpp[1:] < cpp[:-1])), int(np.sum(T_cpp[1:] < T_cpp[:-1]))
if dec_u or dec_T: bad.append('decreasing on sorted keys: u %d, T %d' % (dec_u, dec_T))
rep = json.load(open(w + '/t.json'))
for f in ('nll_train', 'ks_uniform', 'decreasing_pairs', 'collision_fraction', 'wall_s'):
    if f not in rep: bad.append('report lacks ' + f)
if not rep.get("ks_uniform", 1) < 0.1: bad.append('KS %.3g >= 0.1: the flow did not fit' % rep.get('ks_uniform', 1))
print('torch vs C++: %d keys, max |du| %.3g, max |dT| %d, decreasing u %d T %d, report KS %.3g collisions %.3g'
      % (len(probe), du, dT, dec_u, dec_T, rep.get('ks_uniform', -1), rep.get('collision_fraction', -1)))
if bad:
    sys.exit('FAIL: ' + '; '.join(bad))
EOF

# malformed files: each must be rejected (exit 3)
G="$WORK/t.flow"; B="$WORK/bad.flow"; fails=0; n=0
H=$(head -1 "$G"); read -r _ _ M LO HI <<<"$H"; L1=$(sed -n 2p "$G"); REST=$(sed -n '3,$p' "$G")
read -r W1 MU1 S1 <<<"$L1"
reject() {  # name, file content
    n=$((n + 1)); printf '%s' "$2" > "$B"
    set +e; "$WORK/flow_eval" --check "$B" > "$WORK/bad.out" 2>&1; rc=$?; set -e
    if [ $rc -ne 3 ]; then echo "FAIL loader accepted or crashed on: $1 (exit $rc)"; cat "$WORK/bad.out"; fails=$((fails + 1)); fi
}
reject "empty file" ""
reject "bad magic" "molfloww 1 $M $LO $HI"$'\n'"$L1"$'\n'"$REST"$'\n'
reject "version 2" "molflow 2 $M $LO $HI"$'\n'"$L1"$'\n'"$REST"$'\n'
reject "header token missing" "molflow 1 $M $LO"$'\n'"$L1"$'\n'"$REST"$'\n'
reject "header extra token" "molflow 1 $M $LO $HI 7"$'\n'"$L1"$'\n'"$REST"$'\n'
reject "M = 0" "molflow 1 0 $LO $HI"$'\n'
reject "M too large" "molflow 1 $((M + 1)) $LO $HI"$'\n'"$L1"$'\n'"$REST"$'\n'
reject "M too small" "molflow 1 $((M - 1)) $LO $HI"$'\n'"$L1"$'\n'"$REST"$'\n'
reject "M not an integer" "molflow 1 8.0 $LO $HI"$'\n'"$L1"$'\n'"$REST"$'\n'
reject "lo >= hi" "molflow 1 $M $HI $LO"$'\n'"$L1"$'\n'"$REST"$'\n'
reject "lo negative" "molflow 1 $M -5 $HI"$'\n'"$L1"$'\n'"$REST"$'\n'
reject "hi above 2^64-1" "molflow 1 $M $LO 18446744073709551616"$'\n'"$L1"$'\n'"$REST"$'\n'
reject "blank line inside" "molflow 1 $M $LO $HI"$'\n'"$L1"$'\n\n'"$REST"$'\n'
reject "extra line" "molflow 1 $M $LO $HI"$'\n'"$L1"$'\n'"$REST"$'\n'"$L1"$'\n'
reject "4 tokens" "molflow 1 $M $LO $HI"$'\n'"$L1 1"$'\n'"$REST"$'\n'
reject "2 tokens" "molflow 1 $M $LO $HI"$'\n'"$W1 $MU1"$'\n'"$REST"$'\n'
reject "trailing junk" "molflow 1 $M $LO $HI"$'\n'"$W1 $MU1 ${S1}x"$'\n'"$REST"$'\n'
reject "w negative" "molflow 1 $M $LO $HI"$'\n'"-$W1 $MU1 $S1"$'\n'"$REST"$'\n'
reject "w zero" "molflow 1 $M $LO $HI"$'\n'"0 $MU1 $S1"$'\n'"$REST"$'\n'
reject "s zero" "molflow 1 $M $LO $HI"$'\n'"$W1 $MU1 0"$'\n'"$REST"$'\n'
reject "s negative" "molflow 1 $M $LO $HI"$'\n'"$W1 $MU1 -$S1"$'\n'"$REST"$'\n'
reject "s denormal" "molflow 1 $M $LO $HI"$'\n'"$W1 $MU1 1e-310"$'\n'"$REST"$'\n'
reject "mu nan" "molflow 1 $M $LO $HI"$'\n'"$W1 nan $S1"$'\n'"$REST"$'\n'
reject "mu inf" "molflow 1 $M $LO $HI"$'\n'"$W1 inf $S1"$'\n'"$REST"$'\n'
reject "s 1e999" "molflow 1 $M $LO $HI"$'\n'"$W1 $MU1 1e999"$'\n'"$REST"$'\n'
reject "sum w != 1" "molflow 1 $M $LO $HI"$'\n'"$(awk -v w="$W1" 'BEGIN{printf "%.17g", w * 1.001}') $MU1 $S1"$'\n'"$REST"$'\n'
set +e; "$WORK/flow_eval" --check "$WORK/does_not_exist.flow" > /dev/null; rc=$?; set -e
[ $rc -eq 3 ] || { echo "FAIL loader on a missing file (exit $rc)"; fails=$((fails + 1)); }
# and still accepts the good file with trailing blank lines
printf '%s\n\n' "$(cat "$G")" > "$B"; "$WORK/flow_eval" --check "$B" > /dev/null || { echo "FAIL good file rejected"; fails=$((fails + 1)); }
echo "loader: $n malformed files, $fails failures"
[ $fails -eq 0 ] || exit 1
echo "test_flow: all ok"
