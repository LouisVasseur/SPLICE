#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# One detached job for the Atom: everything below in order, surviving logout; safe to rerun (finished stages are
# skipped, and run_baselines.sh resumes inside a stage cell by cell).
#
#   cd ~ && nohup setsid ~/GRE-transform/scripts/atom_overnight.sh > ~/overnight.out 2>&1 < /dev/null &
#   tail -f ~/overnight/overnight.log          # progress;  touch ~/overnight/STOP  to stop after the current cell's stage
#
# Stages (markers in $STATE/<stage>.done):
#   0 env        micromamba env active; python3 + numpy + matplotlib in it (installed once, for the detector/sampler)
#   1 backups    scripts/backup_results.sh on every result dir of this job, every 10 min, to ~/backups
#   2 build      GRE with EXTRA=transforms into ~/gre-transform/GRE (the baselines build is never touched)
#   3 warm_cold  cold trace_lipp / trace_alex on fb and genome at 200M keys, 60M lookups, chunks of 200k
#   4 warm_w     W from scripts/warmup_detect.py --choose-w (derivative cut, 2% guard, x1.5); plot
#   5 warm_ok    trace_w_lipp / trace_w_alex on fb with TRANSFORM_WARM=W, to check the warmed series is flat; plot
#   6 baselines  the paused cold baseline run, resumed exactly as started (GRE as published: no warm-up)
#   7 layers     GRE-transform's layers on the 10M samples, all ten datasets, 2 repeats, 10M lookups
# Paths and core can be overridden by the environment (defaults below).
set -uo pipefail
H=/home/lvasseur
BASE=${BASE:-$H/GRE-baselines}                 # the paused run's repository (untouched apart from its results)
GT=${GT:-$H/GRE-transform}                     # this repository
DATA=${DATA:-$BASE/data}
GRE_BASE=${GRE_BASE:-$H/gre-baselines/GRE}     # the baselines binary: never rebuilt here
GRE_T=${GRE_T:-$H/gre-transform/GRE}
ENV=${ENV:-$H/envs/gre}
MM=${MM:-$H/bin/micromamba}
CORE=${CORE:-2}
STATE=${STATE:-$H/overnight}
FULL=$BASE/results/atom_2026-10-19_full
RW=$GT/results
mkdir -p "$STATE" "$RW"
LOG=$STATE/overnight.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
done_() { [ -f "$STATE/$1.done" ]; }
mark() { date -u +%FT%TZ > "$STATE/$1.done"; log "stage $1 done"; }
stop_check() { [ -f "$STATE/STOP" ] && { log "STOP file found: stopping before $1"; exit 0; }; }
RB_BASE=$BASE/scripts/run_baselines.sh
RB_T=$GT/scripts/run_baselines.sh

log "=== atom_overnight.sh start (pid $$, host $(hostname))"
if pgrep -f '[-]-keys_file=' >/dev/null; then log "a GRE benchmark is already running: refusing (pgrep -fl -- --keys_file=)"; exit 1; fi

# 0 env
eval "$("$MM" shell hook -s bash)" && micromamba activate "$ENV" || { log "cannot activate $ENV"; exit 1; }
export MAMBA_EXE=$MM
if ! done_ env; then
    if ! "$ENV/bin/python3" -c 'import numpy, matplotlib' 2>/dev/null; then
        log "installing python, numpy, matplotlib into $ENV"
        "$MM" install -y -q -p "$ENV" -c conda-forge python=3.11 numpy matplotlib >> "$STATE/env.log" 2>&1 || { log "python install failed (see env.log)"; }
    fi
    "$ENV/bin/python3" -c 'import numpy, matplotlib' 2>/dev/null && mark env
fi
PY=$ENV/bin/python3

# 1 backups (always restarted; it waits for directories that do not exist yet)
pkill -f 'backup_results.sh' 2>/dev/null; sleep 1
BK=$GT/scripts/backup_results.sh; [ -x "$BK" ] || BK=$H/bin/backup_results.sh
nohup "$BK" "$FULL" "$RW/atom_warm_cold" "$RW/atom_warm_verify" "$RW/atom_layers_10M" > "$STATE/backup.log" 2>&1 &
BKPID=$!
log "backups running (pid $BKPID) -> ~/backups"
finish() { kill "$BKPID" 2>/dev/null; wait "$BKPID" 2>/dev/null; log "=== atom_overnight.sh end"; }
trap finish EXIT

# 2 build
BUILT=0
if done_ build && [ -x "$GRE_T/build/microbench" ]; then BUILT=1
else
    log "building GRE with EXTRA=transforms into $GRE_T (log: $STATE/build.log)"
    if GRE_SEED=$GRE_BASE EXTRA=transforms "$GT/scripts/build_gre.sh" "$GRE_T" > "$STATE/build.log" 2>&1; then BUILT=1; mark build
    else log "build FAILED: skipping the warm-up and layer stages (tail of build.log below)"; tail -n 20 "$STATE/build.log" | tee -a "$LOG"; fi
fi

run_t() {  # OUT_DIR, then env assignments and run_baselines.sh's variables
    local out=$1; shift
    env GRE_DIR="$GRE_T" CORE="$CORE" "$@" "$RB_T" "$out" >> "$STATE/$(basename "$out").out" 2>&1
}

# 3 cold warm-up traces
if [ $BUILT = 1 ] && ! done_ warm_cold; then
    stop_check warm_cold; log "stage warm_cold: cold traces, fb and genome, 200M keys"
    run_t "$RW/atom_warm_cold" DATA_DIR="$DATA" DATASETS="fb genome" INDEXES="trace_lipp trace_alex" REPEATS=1 \
        OPS=60000000 TRACE_CHUNK=200000 && mark warm_cold || log "warm_cold: run_baselines failed (see atom_warm_cold.out)"
fi

# 4 choose W
W=$(cat "$STATE/W" 2>/dev/null || true)
if [ $BUILT = 1 ] && done_ warm_cold && ! done_ warm_w; then
    W=$("$PY" "$GT/scripts/warmup_detect.py" "$RW/atom_warm_cold" --chunk 200000 --window 10 --choose-w \
        --plot "$RW/atom_warm_cold/warmup.png" | tee "$RW/atom_warm_cold/detect.txt" | awk '$1 == "W" { print $2 }')
    if [ -n "$W" ]; then echo "$W" > "$STATE/W"; log "warm-up length W = $W lookups"; mark warm_w
    else log "warm_w: detector gave no W (see atom_warm_cold/detect.txt)"; fi
fi

# 5 verify W
if [ $BUILT = 1 ] && [ -n "${W:-}" ] && ! done_ warm_ok; then
    stop_check warm_ok; log "stage warm_ok: warmed traces with W = $W, fb, 200M keys"
    run_t "$RW/atom_warm_verify" DATA_DIR="$DATA" DATASETS="fb" INDEXES="trace_w_lipp trace_w_alex" REPEATS=1 \
        OPS=60000000 TRACE_CHUNK=200000 TRANSFORM_WARM="$W" && {
        "$PY" "$GT/scripts/warmup_detect.py" "$RW/atom_warm_verify" --chunk 200000 --window 10 \
            --plot "$RW/atom_warm_verify/warmup.png" > "$RW/atom_warm_verify/detect.txt"
        mark warm_ok; } || log "warm_ok: run_baselines failed"
fi

# 6 the paused cold baseline run, resumed with exactly its original settings (two parts, as the original chain)
if ! done_ baselines; then
    stop_check baselines; log "stage baselines: resuming $FULL"
    common=(GRE_DIR="$GRE_BASE" DATA_DIR="$DATA" CORE="$CORE")
    env "${common[@]}" DATASETS="covid genome libio books fb history wise" "$RB_BASE" "$FULL" >> "$STATE/baselines.out" 2>&1
    r1=$?
    if (cd "$DATA" && grep -E '  (osm|planet|stack)$' SHA256SUMS | sha256sum -c >> "$STATE/baselines.out" 2>&1); then
        env "${common[@]}" DATASETS="osm planet stack" "$RB_BASE" "$FULL" >> "$STATE/baselines.out" 2>&1
        r2=$?
    else r2=1; log "baselines: osm/planet/stack checksum FAILED, not run"; fi
    [ $r1 = 0 ] && [ $r2 = 0 ] && mark baselines || log "baselines: a part failed (see baselines.out)"
fi

# 7 the layers on the 10M samples
if [ $BUILT = 1 ] && ! done_ layers; then
    stop_check layers; log "stage layers: 10M samples"
    S10=$H/gre-transform/data10M
    [ -f "$S10/SAMPLES.txt" ] && [ "$(ls "$S10"/*_10M 2>/dev/null | wc -l)" -ge 10 ] || \
        "$PY" "$GT/data/sample.py" "$DATA" "$S10" --n 10000000 >> "$STATE/layers.out" 2>&1
    FL=$H/gre-transform/flows; mkdir -p "$FL"; cp -n "$GT"/results/gate_mac_2026-10-19/flows/*_10M.flow "$FL"/ 2>/dev/null
    NAMES="lipp pw512_lipp nf_lipp vp_lipp g_lipp alex pw512_alex pw4096_alex nf_alex vp_alex g_alex g_pw512_alex
           pw512_vp_alex btree pw512_btree vp_btree g_btree pgm pw512_pgm vp_pgm g_pgm artunsync pw512_artunsync"
    NFLD=""
    if [ "$(ls "$H"/gre-transform/nfl10M/*.nfl 2>/dev/null | wc -l)" -ge 10 ]; then
        NAMES="$NAMES nfl_lipp nfl_alex nfl_btree"; NFLD="NFL_DIR=$H/gre-transform/nfl10M"
    else log "layers: no NFL weights in ~/gre-transform/nfl10M (copied from the Mac): nfl_* skipped"; fi
    DS=$(cd "$S10" && ls *_10M | tr '\n' ' ')
    run_t "$RW/atom_layers_10M" DATA_DIR="$S10" DATASETS="$DS" INDEXES="$(echo $NAMES)" REPEATS=2 OPS=10000000 \
        FLOW_DIR="$FL" ${NFLD:+$NFLD} && mark layers || log "layers: run_baselines failed or some cells failed (see atom_layers_10M.out)"
fi
log "all stages processed: $(ls "$STATE"/*.done 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
