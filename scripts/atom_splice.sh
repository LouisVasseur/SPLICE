#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPLICE's 200M runs on the Atom, as one detached job that waits for scripts/atom_overnight.sh to finish first:
#
#   cd /home/lvasseur && nohup setsid bash /home/lvasseur/SPLICE-gre/scripts/atom_splice.sh > /home/lvasseur/splice.out 2>&1 < /dev/null &
#   tail -f /home/lvasseur/splice/splice.log        # progress;  touch /home/lvasseur/splice/STOP  stops before a stage
#
# Stages (markers in $STATE/<stage>.done; a rerun skips finished stages, and run_parallel.sh resumes cell by cell):
#   wait     until no atom_overnight.sh and no GRE benchmark is running (the Atom's other job keeps its core)
#   build    GRE with EXTRA=transforms from this checkout into $GRE (its own directory: nothing else is rebuilt)
#   runs     the Mac's cells (results/mac200M_2026-10-20/cells.txt): J=2 single-thread runs, pinned to CORES,
#            two passes (every cell's first repeat before any second), at most one LIPP at a time
# Results: $OUT (runs.tsv, cells/, progress.log), backed up every 10 min to /home/lvasseur/backups.
set -uo pipefail
H=/home/lvasseur
REPO=${REPO:-$H/SPLICE-gre}            # git clone --branch splice-gre https://github.com/LouisVasseur/SPLICE.git
GRE=${GRE:-$H/gre-splice/GRE}
GRE_SEED=${GRE_SEED:-$H/gre-baselines/GRE}
DATA=${DATA:-$H/GRE-baselines/data}
ENV=${ENV:-$H/envs/gre}
MM=${MM:-$H/bin/micromamba}
J=${J:-2}; CORES=${CORES:-"2 3"}
STATE=${STATE:-$H/splice}
OUT=${OUT:-$REPO/results/atom200M_splice}
mkdir -p "$STATE" "$OUT"
LOG=$STATE/splice.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
done_() { [ -f "$STATE/$1.done" ]; }
mark() { date -u +%FT%TZ > "$STATE/$1.done"; log "stage $1 done"; }
stop_check() { [ -f "$STATE/STOP" ] && { log "STOP file found: stopping before $1"; exit 0; }; }

log "=== atom_splice.sh start (pid $$, host $(hostname))"
eval "$("$MM" shell hook -s bash)" && micromamba activate "$ENV" || { log "cannot activate $ENV"; exit 1; }
export MAMBA_EXE=$MM

if ! done_ wait; then
    while pgrep -f '[a]tom_overnight.sh' >/dev/null || pgrep -f '[-]-keys_file=' >/dev/null; do sleep 60; done
    mark wait
fi
stop_check build
if ! done_ build || [ ! -x "$GRE/build/microbench" ]; then
    log "building GRE with EXTRA=transforms into $GRE (log: $STATE/build.log)"
    GRE_SEED=$GRE_SEED EXTRA=transforms "$REPO/scripts/build_gre.sh" "$GRE" > "$STATE/build.log" 2>&1 && mark build ||
        { log "build FAILED (tail of build.log below)"; tail -n 20 "$STATE/build.log" | tee -a "$LOG"; exit 1; }
fi
nohup "$REPO/scripts/backup_results.sh" "$OUT" > "$STATE/backup.log" 2>&1 &
BK=$!
trap 'kill $BK 2>/dev/null; log "=== atom_splice.sh end"' EXIT
stop_check runs
if ! done_ runs; then
    cp "$REPO/results/mac200M_2026-10-20/cells.txt" "$OUT/cells.txt"
    log "stage runs: J=$J CORES=$CORES into $OUT"
    J=$J CORES="$CORES" PASSES="1 2" GRE_DIR="$GRE" DATA_DIR="$DATA" "$REPO/scripts/run_parallel.sh" "$OUT" "$OUT/cells.txt" \
        >> "$STATE/runs.out" 2>&1 && mark runs || log "runs: run_parallel.sh failed (see runs.out)"
fi
log "stages: $(ls "$STATE"/*.done 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
