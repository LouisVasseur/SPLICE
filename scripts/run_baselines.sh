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
#
# Read-only, single thread: GRE's flags --read=1 --insert=0 --thread_num=1, OMP_NUM_THREADS=1 (GRE seeds one sampler
# per OpenMP thread, so this fixes the operation list). Lookups sample bulk-loaded keys: a correct run prints
# success_read equal to operations_num, which is checked. Resumable: a (dataset, index, repeat) already in runs.tsv is
# skipped; OUT_DIR/SETTINGS pins ops/table/ratio/core/binary, and a rerun with other settings is refused.
# macOS: LIPP is skipped on covid, osm and books (it asserts or grows memory until macOS kills it; see README).
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

if [ "$(uname -s)" = Darwin ]; then
    # pmset: 'powermode' 0 automatic, 1 low power, 2 high power (older macOS: 'lowpowermode' 0/1)
    POWER=$(pmset -g batt | sed -n 's/^Now drawing from .\(.*\).$/\1/p'); PM=$(pmset -g | awk '/lowpowermode|powermode/ {print $1 "=" $2; exit}')
    HOST="$(sysctl -n machdep.cpu.brand_string), $POWER, ${PM:-powermode=?}"
    [ "$POWER" = "AC Power" ] && [ "${PM#*=}" != 1 ] || echo "WARNING: $HOST: timings are not comparable (smoke only)" >&2
else HOST="$(lscpu | sed -n 's/^Model name: *//p'), $(uname -r), governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo ?)"; fi
SETTINGS="ops=$OPS table=$TABLE ratio=$RATIO core=${CORE:-none} binary=$(sed -n 's/^microbench sha256: //p' "$GRE_DIR/build/BUILD_MANIFEST.txt")"
if [ -f "$OUT/SETTINGS" ] && [ "$(cat "$OUT/SETTINGS")" != "$SETTINGS" ]; then
    echo "$OUT was run with '$(cat "$OUT/SETTINGS")', not '$SETTINGS': use another OUT_DIR" >&2; exit 1; fi
echo "$SETTINGS" > "$OUT/SETTINGS"
PIN=(); [ -n "${CORE:-}" ] && [ "$(uname -s)" = Linux ] && PIN=(taskset -c "$CORE")
{ echo "started: $(date -u +%FT%TZ)"; echo "host: $HOST"; echo "binary sha256 prefix: $(sed -n 's/^microbench sha256: //p' "$GRE_DIR/build/BUILD_MANIFEST.txt")"
  echo "repo: $(git -C "$ROOT" rev-parse -q --verify HEAD || echo uncommitted)$(git -C "$ROOT" diff --quiet HEAD 2>/dev/null || echo ' (dirty)')"
  echo "datasets: $DATASETS"; echo "indexes: $INDEXES"; echo "repeats=$REPEATS ops=$OPS table=$TABLE ratio=$RATIO core=${CORE:-none}"
  echo "--- BUILD_MANIFEST.txt"; cat "$GRE_DIR/build/BUILD_MANIFEST.txt"; echo "---"; } >> "$OUT/MANIFEST.txt"
[ -f "$OUT/runs.tsv" ] || printf 'dataset\tindex\trepeat\tthroughput_ops\tmemory_bytes\tsuccess_read\twall_s\tlog\n' > "$OUT/runs.tsv"

last() { awk -v k="$1" -v f="$2" 'index($0, k) == 1 { v = $f } END { print v }' "$3"; }
n=0; total=$(( $(echo $DATASETS | wc -w) * $(echo $INDEXES | wc -w) * REPEATS ))
for r in $(seq 1 "$REPEATS"); do for d in $DATASETS; do for i in $INDEXES; do
    n=$((n + 1)); log="$OUT/logs/$d.$i.r$r.log"
    if awk -F'\t' -v d="$d" -v i="$i" -v r="$r" '$1 == d && $2 == i && $3 == r { f = 1 } END { exit !f }' "$OUT/runs.tsv"; then
        echo "[$n/$total] $d $i r$r: done"; continue; fi
    if [ "$(uname -s)" = Darwin ] && [ "$i" = lipp ] && case $d in covid|osm|books) true;; *) false;; esac; then
        echo "[$n/$total] $d $i r$r: skipped on macOS (LIPP fails on keys above 2^53 there)"; continue; fi
    t0=$(date +%s)
    # GRE's own flags; no --latency_sample (it adds timer reads to the timed loop), as in GRE's throughput figures
    if ! OMP_NUM_THREADS=1 ${PIN[@]+"${PIN[@]}"} "$BIN" --keys_file="$DATA_DIR/$d" --keys_file_type=binary --read=1 --insert=0 \
        --operations_num="$OPS" --table_size="$TABLE" --init_table_ratio="$RATIO" --thread_num=1 --memory \
        --index="$i" --output_path="$OUT/logs/$d.$i.r$r.csv" > "$log" 2>&1; then
        echo "[$n/$total] $d $i r$r: FAILED ($log)"; continue; fi
    t=$(( $(date +%s) - t0 ))
    thr=$(last 'Throughput = ' 3 "$log"); mem=$(last 'Memory: ' 2 "$log"); ok=$(last 'success_read: ' 2 "$log")
    [ "$ok" = "$OPS" ] || { echo "[$n/$total] $d $i r$r: success_read=$ok != $OPS ($log)"; continue; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$d" "$i" "$r" "$thr" "$mem" "$ok" "$t" "logs/$d.$i.r$r.log" >> "$OUT/runs.tsv"
    echo "[$n/$total] $d $i r$r: $thr ops/s, mem $mem B, ${t}s"
done; done; done
echo "finished: $(date -u +%FT%TZ)" >> "$OUT/MANIFEST.txt"
python3 "$ROOT/scripts/summarize.py" "$OUT"
