#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Run GRE's microbench, unmodified driver, one index per process, over datasets x indexes x repeats.
#
#   scripts/run_baselines.sh OUT_DIR
#
#   GRE_DIR=DIR       GRE clone built by build_gre.sh (default $HOME/gre-baselines/GRE)
#   DATA_DIR=DIR      the SOSD key files (default data/, see data/download.sh)
#   DATASETS="..."    default: covid genome libio osm books fb history planet stack wise
#   INDEXES="..."     default: the indexes in BUILD_MANIFEST.txt
#   REPEATS=N         default 3
#   OPS=N             --operations_num, default 100000000
#   TABLE=N           --table_size, default -1 (all keys); a smaller N reads only the first N keys (smoke tests)
#   RATIO=R           --init_table_ratio, default 1 (bulk-load every key)
#   CORE=K            Linux: taskset -c K (default: not pinned)
# Transform builds (EXTRA=transforms scripts/build_gre.sh; see README):
#   FLOW_DIR=DIR      TRANSFORM_FLOW=DIR/<dataset>.flow for each dataset (read by nf_* and gate_*); when unset,
#                     TRANSFORM_FLOW is cleared, so a flow never leaks from the caller's environment into a dataset
#   TRANSFORM_K=K     K of the pw_<base> indexes only (default 512 there; pw<K>_, gate_ and nflsw_ have fixed K), an
#                     integer in [2, 65536] without leading zeros, as the registry requires
# Both go into SETTINGS (only when set, so baseline SETTINGS are unchanged), and the sha256 of each dataset's flow file
# into MANIFEST.txt and OUT_DIR/FLOWS: a rerun with another flow for a dataset already run is refused.
# Provenance: the binary is hashed and copied to OUT_DIR/bin/microbench.<sha16>, and that copy is what runs (a rebuild
# during the run cannot change it; SETTINGS pins its hash). MANIFEST.txt gets DATA_DIR, the sha256 prefix of each
# dataset file (HASH_DATA=0 skips it), the process ancestry (caffeinate, the lane) and, at start and finish, the power
# state and load average.
#
# Read-only, single thread: GRE's flags --read=1 --insert=0 --thread_num=1, OMP_NUM_THREADS=1 (GRE seeds one sampler
# per OpenMP thread, so this fixes the operation list). Lookups sample bulk-loaded keys: a correct run prints
# success_read equal to operations_num, which is checked. Resumable: a (dataset, index, repeat) already in runs.tsv is
# skipped; OUT_DIR/SETTINGS pins ops/table/ratio/core/binary, and a rerun with other settings is refused.
# macOS: raw LIPP is skipped where it fails: two adjacent keys equal as doubles (long double is double there), so its
# bulk load recurses until memory runs out or RT_ASSERT exits. For GRE's ten file names this is the baselines' list,
# covid, osm and books; for any other name (the samples) scripts/double_ties.py decides (books_10M: 8 ties, skipped;
# covid_10M and osm_10M: none, run). Transform indexes on LIPP are never skipped (outputs < 2^52 are exact doubles).
# Writes OUT_DIR/{runs.tsv, logs/, MANIFEST.txt, SETTINGS}; scripts/summarize.py OUT_DIR prints the table.
set -euo pipefail
OUT=${1:?usage: run_baselines.sh OUT_DIR}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GRE_DIR=${GRE_DIR:-$HOME/gre-baselines/GRE}
DATA_DIR=${DATA_DIR:-$ROOT/data}
DATASETS=${DATASETS:-covid genome libio osm books fb history planet stack wise}
BIN=$GRE_DIR/build/microbench
[ -x "$BIN" ] || { echo "no $BIN: run scripts/build_gre.sh first" >&2; exit 1; }
INDEXES=${INDEXES:-$(sed -n 's/^indexes: //p' "$GRE_DIR/build/BUILD_MANIFEST.txt")}
REPEATS=${REPEATS:-3}; OPS=${OPS:-100000000}; TABLE=${TABLE:--1}; RATIO=${RATIO:-1}
mkdir -p "$OUT/logs"
for d in $DATASETS; do [ -f "$DATA_DIR/$d" ] || { echo "missing $DATA_DIR/$d (data/download.sh)" >&2; exit 1; }; done
sha16() { (command -v sha256sum >/dev/null && sha256sum "$1" || shasum -a 256 "$1") | cut -c1-16; }
XSET=""
if [ -n "${TRANSFORM_K:-}" ]; then
    case $TRANSFORM_K in
        0*|*[!0-9]*) false ;;
        *) [ ${#TRANSFORM_K} -le 5 ] && [ "$TRANSFORM_K" -ge 2 ] && [ "$TRANSFORM_K" -le 65536 ] ;;
    esac ||
        { echo "TRANSFORM_K=$TRANSFORM_K: must be an integer in [2, 65536] without leading zeros" >&2; exit 1; }
    case " $INDEXES " in *" pw_"*) ;; *) echo "WARNING: TRANSFORM_K affects only pw_<base>, and INDEXES has none" >&2 ;; esac
    XSET="$XSET transform_k=$TRANSFORM_K"; fi
unset TRANSFORM_FLOW TRANSFORM_NFL
if [ -n "${NFL_DIR:-}" ]; then   # NFL_DIR=DIR: TRANSFORM_NFL=DIR/<dataset>.nfl (NFL's own flow weights, read by nfl_*)
    NFL_DIR=$(cd "$NFL_DIR" && pwd) || { echo "NFL_DIR: no such directory" >&2; exit 1; }
    XSET="$XSET nfl_dir=$NFL_DIR"; fi
if [ -n "${FLOW_DIR:-}" ]; then
    FLOW_DIR=$(cd "$FLOW_DIR" && pwd) || { echo "FLOW_DIR: no such directory" >&2; exit 1; }
    XSET="$XSET flow_dir=$FLOW_DIR"; fi

host_state() {  # HOST: CPU, power source and power mode (macOS) or governor (Linux); then the load average
    if [ "$(uname -s)" = Darwin ]; then
        # pmset: 'powermode' 0 automatic, 1 low power, 2 high power (older macOS: 'lowpowermode' 0/1)
        POWER=$(pmset -g batt | sed -n 's/^Now drawing from .\(.*\).$/\1/p'); PM=$(pmset -g | awk '/lowpowermode|powermode/ {print $1 "=" $2; exit}')
        HOST="$(sysctl -n machdep.cpu.brand_string), $POWER, ${PM:-powermode=?}"
    else HOST="$(lscpu | sed -n 's/^Model name: *//p'), $(uname -r), governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo ?)"; fi
    LOAD=$(uptime | sed 's/.*load average[s]*: *//')
}
host_state
[ "$(uname -s)" != Darwin ] || { [ "$POWER" = "AC Power" ] && [ "${PM#*=}" != 1 ]; } ||
    echo "WARNING: $HOST: timings are not comparable (smoke only)" >&2
ancestry() {  # the commands this script runs under (caffeinate, mac_lane.sh, ...), innermost first
    local p=$PPID out=""
    while [ "${p:-0}" -gt 1 ]; do
        out="$out | $(ps -o args= -p "$p" 2>/dev/null | cut -c1-100)"; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    done
    echo "${out# | }"
}
BIN_SHA=$(sha16 "$BIN")
[ "$BIN_SHA" = "$(sed -n 's/^microbench sha256: //p' "$GRE_DIR/build/BUILD_MANIFEST.txt")" ] ||
    { echo "$BIN (sha256 $BIN_SHA) is not the binary its BUILD_MANIFEST.txt records: rebuild with build_gre.sh" >&2; exit 1; }
SETTINGS="ops=$OPS table=$TABLE ratio=$RATIO core=${CORE:-none} binary=$BIN_SHA$XSET"
if [ -f "$OUT/SETTINGS" ] && [ "$(cat "$OUT/SETTINGS")" != "$SETTINGS" ]; then
    echo "$OUT was run with '$(cat "$OUT/SETTINGS")', not '$SETTINGS': use another OUT_DIR" >&2; exit 1; fi
echo "$SETTINGS" > "$OUT/SETTINGS"
FLOWS=""
if [ -n "${FLOW_DIR:-}" ]; then  # OUT/FLOWS: "dataset sha256-prefix|missing", first run of each dataset wins
    touch "$OUT/FLOWS"
    for d in $DATASETS; do
        f="$FLOW_DIR/$d.flow"; if [ -f "$f" ]; then h=$(sha16 "$f"); else h=missing
            echo "WARNING: no $f: nf_* fails on $d, gate_* cannot choose nf there" >&2; fi
        old=$(awk -v d="$d" '$1 == d { print $2 }' "$OUT/FLOWS")
        [ -z "$old" ] || [ "$old" = "$h" ] || { echo "$OUT ran $d with flow $old, not $h: use another OUT_DIR" >&2; exit 1; }
        [ -n "$old" ] || echo "$d $h" >> "$OUT/FLOWS"
        FLOWS="$FLOWS $d=$h"
    done
fi
PIN=(); [ -n "${CORE:-}" ] && [ "$(uname -s)" = Linux ] && PIN=(taskset -c "$CORE")
mkdir -p "$OUT/bin"; RUN_BIN=$OUT/bin/microbench.$BIN_SHA  # the copy that runs (provenance, see above)
[ -f "$RUN_BIN" ] && [ "$(sha16 "$RUN_BIN")" = "$BIN_SHA" ] || cp -p "$BIN" "$RUN_BIN"
DATA_SHA="not hashed (HASH_DATA=0)"
[ "${HASH_DATA:-1}" = 0 ] || DATA_SHA=$(for d in $DATASETS; do printf ' %s=%s' "$d" "$(sha16 "$DATA_DIR/$d")"; done)
{ echo "started: $(date -u +%FT%TZ)"; echo "host: $HOST; load average: $LOAD"; echo "binary sha256 prefix: $BIN_SHA (runs as $RUN_BIN)"
  echo "under: $(ancestry)"; echo "data: $DATA_DIR; sha256 prefixes:$DATA_SHA"
  echo "repo: $(git -C "$ROOT" rev-parse -q --verify HEAD || echo uncommitted)$(git -C "$ROOT" diff --quiet HEAD 2>/dev/null || echo ' (dirty)')"
  echo "datasets: $DATASETS"; echo "indexes: $INDEXES"; echo "repeats=$REPEATS ops=$OPS table=$TABLE ratio=$RATIO core=${CORE:-none}"
  [ -z "$XSET" ] || echo "transforms: TRANSFORM_K=${TRANSFORM_K:-unset (512)} FLOW_DIR=${FLOW_DIR:-unset}"
  [ -z "$FLOWS" ] || echo "flow sha256 prefixes:$FLOWS"
  echo "--- BUILD_MANIFEST.txt"; cat "$GRE_DIR/build/BUILD_MANIFEST.txt"; echo "---"; } >> "$OUT/MANIFEST.txt"
[ -f "$OUT/runs.tsv" ] || printf 'dataset\tindex\trepeat\tthroughput_ops\tmemory_bytes\tsuccess_read\twall_s\tlog\n' > "$OUT/runs.tsv"

last() { awk -v k="$1" -v f="$2" 'index($0, k) == 1 { v = $f } END { print v }' "$3"; }
TIES=" "  # lipp_fails DATASET (macOS): GRE's ten names by name, as the baselines; others by double ties, cached
lipp_fails() {
    case $1 in covid|osm|books) return 0 ;; genome|libio|fb|history|planet|stack|wise) return 1 ;; esac
    case $TIES in *" $1="*) ;; *) TIES="$TIES$1=$("${PYTHON:-python3}" "$ROOT/scripts/double_ties.py" "$DATA_DIR/$1") " ;; esac
    local c=${TIES#*" $1="}; c=${c%% *}
    case $c in ''|*[!0-9]*) echo "double_ties.py failed on $DATA_DIR/$1" >&2; exit 1 ;; esac
    [ "$c" -gt 0 ]
}
n=0; total=$(( $(echo $DATASETS | wc -w) * $(echo $INDEXES | wc -w) * REPEATS ))
for r in $(seq 1 "$REPEATS"); do for d in $DATASETS; do for i in $INDEXES; do
    n=$((n + 1)); log="$OUT/logs/$d.$i.r$r.log"
    if awk -F'\t' -v d="$d" -v i="$i" -v r="$r" '$1 == d && $2 == i && $3 == r { f = 1 } END { exit !f }' "$OUT/runs.tsv"; then
        echo "[$n/$total] $d $i r$r: done"; continue; fi
    if [ "$(uname -s)" = Darwin ] && [ "$i" = lipp ] && lipp_fails "$d"; then
        echo "[$n/$total] $d $i r$r: skipped on macOS (raw LIPP fails on keys that are equal as doubles)"; continue; fi
    [ -z "${FLOW_DIR:-}" ] || export TRANSFORM_FLOW="$FLOW_DIR/$d.flow"
    [ -z "${NFL_DIR:-}" ] || export TRANSFORM_NFL="$NFL_DIR/$d.nfl"
    t0=$(date +%s)
    # GRE's own flags; no --latency_sample (it adds timer reads to the timed loop), as in GRE's throughput figures
    if ! OMP_NUM_THREADS=1 ${PIN[@]+"${PIN[@]}"} "$RUN_BIN" --keys_file="$DATA_DIR/$d" --keys_file_type=binary --read=1 --insert=0 \
        --operations_num="$OPS" --table_size="$TABLE" --init_table_ratio="$RATIO" --thread_num=1 --memory \
        --index="$i" --output_path="$OUT/logs/$d.$i.r$r.csv" > "$log" 2>&1; then
        echo "[$n/$total] $d $i r$r: FAILED ($log)"; continue; fi
    t=$(( $(date +%s) - t0 ))
    thr=$(last 'Throughput = ' 3 "$log"); mem=$(last 'Memory: ' 2 "$log"); ok=$(last 'success_read: ' 2 "$log")
    [ "$ok" = "$OPS" ] || { echo "[$n/$total] $d $i r$r: success_read=$ok != $OPS ($log)"; continue; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$d" "$i" "$r" "$thr" "$mem" "$ok" "$t" "logs/$d.$i.r$r.log" >> "$OUT/runs.tsv"
    echo "[$n/$total] $d $i r$r: $thr ops/s, mem $mem B, ${t}s"
done; done; done
host_state
{ echo "finished: $(date -u +%FT%TZ)"; echo "host at finish: $HOST; load average: $LOAD"
  b=$(sha16 "$BIN"); [ "$b" = "$BIN_SHA" ] || echo "NOTE: $BIN changed during the run (now $b); the runs used $RUN_BIN"
} >> "$OUT/MANIFEST.txt"
python3 "$ROOT/scripts/summarize.py" "$OUT"
