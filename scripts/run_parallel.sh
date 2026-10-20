#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Several single-thread GRE runs at once: one scripts/run_baselines.sh per (dataset, index) cell, J cells at a time.
#
#   J=3 PASSES="1 2" scripts/run_parallel.sh OUT_DIR CELLS_FILE
#
# CELLS_FILE: one "dataset index" per line ('#' comments and blank lines ignored), in priority order. Each cell runs as
# run_baselines.sh OUT_DIR/cells/<dataset>.<index> with DATASETS=<dataset> INDEXES=<index>; every other variable
# (GRE_DIR, DATA_DIR, OPS, HASH_DATA, ...) passes through, so run_baselines.sh's resume, provenance, success_read check
# and macOS LIPP skip apply per cell. Pass p runs every cell with REPEATS=p (resuming), so the first repeat of every cell
# exists before any second one: a run cut short still has one repeat of everything. Rerunning the same command resumes.
# Progress: OUT_DIR/progress.log (one line per cell and pass, with run_baselines.sh's last line). Merged table:
# OUT_DIR/runs.tsv (every cell's runs.tsv, rewritten atomically after each cell; logs stay under cells/).
# HEAVY (an extended regex over index names, default '^lipp$'): at most one such cell at a time (a lock directory), for
# indexes whose memory at 200M keys (LIPP: up to ~150 B/key) would not fit J times.
# macOS cannot pin cores: J concurrent runs share the memory system and the last-level cache. OUT_DIR/PARALLEL.txt
# records J, so these timings are compared with each other, not with one-at-a-time numbers.
set -uo pipefail
OUT=${1:?usage: run_parallel.sh OUT_DIR CELLS_FILE}
CELLS=${2:?usage: run_parallel.sh OUT_DIR CELLS_FILE}
J=${J:-3}; PASSES=${PASSES:-"1 2"}; HEAVY=${HEAVY:-'^lipp$'}
RB=$(cd "$(dirname "$0")" && pwd)/run_baselines.sh
mkdir -p "$OUT/cells"
OUT=$(cd "$OUT" && pwd)
{ echo "started: $(date -u +%FT%TZ) host: $(hostname) J=$J passes: $PASSES"; echo "cells ($CELLS):"; grep -v '^#' "$CELLS" | awk NF
  echo "env: GRE_DIR=${GRE_DIR:-} DATA_DIR=${DATA_DIR:-} OPS=${OPS:-} HASH_DATA=${HASH_DATA:-1} HEAVY=$HEAVY"; } >> "$OUT/PARALLEL.txt"

merge() {  # every cell's runs.tsv into OUT/runs.tsv (one header), atomically
  local tmp; tmp=$(mktemp "$OUT/.runs.XXXXXX")
  printf 'dataset\tindex\trepeat\tthroughput_ops\tmemory_bytes\tsuccess_read\twall_s\tlog\n' > "$tmp"
  for f in "$OUT"/cells/*/runs.tsv; do [ -f "$f" ] && awk -v c="cells/$(basename "$(dirname "$f")")/" -F'\t' -v OFS='\t' \
      'FNR > 1 { $8 = c $8; print }' "$f"; done >> "$tmp"
  mv -f "$tmp" "$OUT/runs.tsv"
}
cell() {  # cell DATASET INDEX PASS
  local d=$1 i=$2 p=$3 o="$OUT/cells/$1.$2" t0 rc
  mkdir -p "$o"
  local heavy=0
  if [[ $i =~ $HEAVY ]]; then heavy=1; until mkdir "$OUT/.heavy.lock" 2>/dev/null; do sleep 5; done; fi
  t0=$(date +%s)
  DATASETS=$d INDEXES=$i REPEATS=$p "$RB" "$o" >> "$o/run.out" 2>&1; rc=$?
  [ $heavy = 0 ] || rmdir "$OUT/.heavy.lock"
  echo "$(date -u +%FT%TZ) pass=$p $d $i rc=$rc $(( $(date +%s) - t0 ))s | $(grep -E '^\[[0-9]+/[0-9]+\]' "$o/run.out" | tail -n 1)" \
      >> "$OUT/progress.log"
  merge
}
export -f cell merge; export OUT RB HEAVY
rmdir "$OUT/.heavy.lock" 2>/dev/null || true  # a stale lock from an interrupted run
for p in $PASSES; do
  grep -v '^#' "$CELLS" | awk 'NF >= 2 { print $1, $2 }' | sed "s/\$/ $p/" | xargs -P "$J" -L 1 bash -c 'cell "$@"' _
  echo "$(date -u +%FT%TZ) pass $p finished" >> "$OUT/progress.log"
done
merge
echo "finished: $(date -u +%FT%TZ)" >> "$OUT/PARALLEL.txt"
