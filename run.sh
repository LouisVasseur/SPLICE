#!/usr/bin/env bash
# SPDX-License-Identifier: MIT (SPLICE)
# run.sh: the one entry point. Every command is a thin call to a script in scripts/ (or build/splice_count) with this
# machine's protocol defaults. Arguments after the command come after those defaults, so they win: gre_run.sh,
# paired_run.sh and splice_count all keep the last value of a repeated flag.
# Darwin = the Mac (counts, correctness, preview timings): GRE in ~/gre_mac/GRE built by scripts/gre_mac.sh, unpinned
# (macOS cannot pin), timed batches inside scripts/mac_lane.sh. Anything else = the DIAS Atom (the reference timings):
# GRE in /tmp/louisvasseur/GRE built by scripts/gre_lite.sh, pinned to core 2.
# Written for bash 3.2 (macOS) and bash 4.4 (Ubuntu 18.04): no associative arrays, empty arrays as ${a[@]+...}.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
S=$ROOT/scripts
DATA=$ROOT/data/external/gre
ALL_DS=fb,osm,books,covid,genome,history,libio,planet,stack,wise
if [ "$(uname -s)" = Darwin ]; then
  MACHINE=mac; GRE=${GRE_DIR:-$HOME/gre_mac/GRE}; PIN=-1
  INDEXES=splice,pgm,lipp,sortedarray,alex,btree             # no artunsync: ART needs SSE2 (docs/MAC.md)
else
  MACHINE=atom; GRE=${GRE_DIR:-/tmp/louisvasseur/GRE}; PIN=2   # core 2; keep its L2 sibling idle (docs/PROTOCOL.md 5b)
  INDEXES=splice,pgm,lipp,sortedarray,alex,btree,artunsync
fi

die() { echo "run.sh: $*" >&2; exit 2; }
usage() {
  cat <<EOF
usage: ./run.sh [--gre DIR] COMMAND [ARGS...]
machine: $MACHINE   GRE: $GRE (GRE_DIR=DIR or --gre DIR overrides)   data: data/external/gre

  setup [build|data|gre]...   CMake build in build/; fetch the datasets (present files are skipped, all are
                              sha256-checked) and, on the Mac, the .sorted copies and 2M samples; the GRE build.
                              Default: all three, in that order.
  test                        ctest + the Python test (without data/samples, skips the two tests that read them)
  smoke [ARGS]                every index of this GRE build on fb, 10M keys, 1M lookups: success_read must equal
                              the lookups. Not timed. OUTDIR under \$TMPDIR.
  bench [OUTDIR] [ARGS]       the headline: 20M untimed + 100M timed lookups, 3 rounds, on core 2 (Atom) or in
                              the Mac lane. Defaults: OUTDIR results/local/bench-DATE, --datasets fb,osm,
                              --indexes $INDEXES. --rounds N, --datasets, --indexes, any gre_run.sh flag.
  warmup --tag T [ARGS]       the paired warm-up trace batch -> results/paired/T/<machine>/ (scripts/paired_run.sh)
  analyze OUTDIR... [ARGS]    warm-up phases of trace_* runs (scripts/warmup_detect.py analyze; numpy)
  pair A B [ARGS]             one machine's trace runs against the other's (scripts/warmup_detect.py pair; numpy)
  report OUTDIR [ARGS]        checks, throughput, ratios, memory as Markdown (scripts/gre_report.py)
  counts [--datasets L] [--out D] [F]
                              SPLICE-H plans: splice_count over all keys -> D (default results/splice_h/), then
                              D/COUNTS.md (Mac)
  lane on|off|status|run -- CMD   Mac only: move heavy apps to the efficiency cores (scripts/mac_lane.sh)
  atom SUB [--out R] [--lanes K] [--dry-run] [ARGS]
                              the Atom in parallel pinned lanes (scripts/atom_batch.py, docs/ATOM_FAST.md):
                              doctor | calib | verdict | warmup | bench | run warmup,bench | counts | status |
                              merge | anchors. Real runs on Linux only; --dry-run prints the schedule anywhere.
  help                        this table

Long runs: nohup ./run.sh bench ... > bench.out 2>&1 &     Docs: README.md, docs/PROTOCOL.md (Atom), docs/MAC.md
EOF
}

# A heavy job beside a timed GRE run disturbs it: refuse, as gre_lite.sh and gre_run.sh do (RUN_FORCE=1 overrides).
# Match the executable (argv[0]), not any command line that merely mentions the name (a shell loop, tail -f, an editor).
busy_pids() {
  ps -Ao pid=,args= | awk '{ n = $2; sub(/.*\//, "", n)
    if (n == "splice_count" || n == "scaleli_hardness" || n == "scaleli_bench") { print $1; next }
    if (n == "microbench") for (i = 3; i <= NF; i++) if (index($i, "--keys_file=") == 1) { print $1; next } }'
}
need_quiet() {
  [ "${RUN_FORCE:-0}" = 1 ] && return 0
  local p; p=$(busy_pids | tr '\n' ' '); p=${p% }
  [ -z "$p" ] || die "a benchmark is running (pids $p: microbench --keys_file=, splice_count or scaleli_*); wait, or set RUN_FORCE=1"
}

if [ "${1:-}" = --gre ]; then [ $# -ge 2 ] || die "--gre needs a directory"; GRE=$2; shift 2; fi
CMD=${1:-help}; [ $# -eq 0 ] || shift

case $CMD in
  setup)
    STEPS=${*:-build data gre}
    for st in $STEPS; do case $st in build|data|gre) ;; *) die "setup: unknown step '$st' (build, data, gre)";; esac; done
    for st in $STEPS; do
      echo "== setup $st ($MACHINE)"
      case $st in
        build) need_quiet
               cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
               cmake --build "$ROOT/build" -j;;
        data)  "$DATA/download.sh"
               # the Mac also needs the .sorted copies (splice_count) and the samples (tests); the Atom's GRE sorts on load
               if [ $MACHINE = mac ]; then "$S/prepare_data.sh" "$ROOT/build"; fi;;
        gre)   need_quiet
               if [ $MACHINE = mac ]; then "$S/gre_mac.sh" "$GRE"; else "$S/gre_lite.sh" "$GRE"; fi;;
      esac
    done;;
  test)
    need_quiet
    exec "$S/check.sh";;
  smoke)
    MB=$GRE/build/microbench
    [ -x "$MB" ] || die "no GRE build at $MB (./run.sh setup gre)"
    # gre_lite.sh compiles its index list into the binary ('(gre_lite: alex lipp ...)'); gre_mac.sh drops ART from it.
    IX=$({ LC_ALL=C grep -a -o 'gre_lite: [a-z0-9_ ]*)' "$MB" || true; } | sed -n '1{s/^gre_lite: //;s/)$//;s/ /,/g;p;}')
    [ -n "$IX" ] || die "$MB has no gre_lite index list (./run.sh setup gre)"
    OUT=$(mktemp -d "${TMPDIR:-/tmp}/splice_smoke.XXXXXX")
    echo "smoke: fb, 10M keys, 1M lookups, untimed -> $OUT"
    # --splice-plan none: the build defaults, as docs/PROTOCOL.md 5b step 2 (the plans are for all 200M keys)
    exec "$S/gre_run.sh" "$OUT" --gre "$GRE" --data "$DATA" --datasets fb --indexes "$IX" --table-size 10000000 \
      --ops 1000000 --repeats 1 --warmup 0 --pin -1 --splice-plan none "$@";;
  bench)
    OUT=$ROOT/results/local/bench-$(date +%F)          # results/local*/ is ignored: copy finished runs elsewhere
    case ${1:-} in ''|-*) ;; *) OUT=$1; shift;; esac
    ARGS=()                                            # --rounds N = gre_run.sh's --repeats N (dataset x index each)
    while [ $# -gt 0 ]; do
      case $1 in --rounds) ARGS+=(--repeats);; --) ARGS+=("$@"); break;; *) ARGS+=("$1");; esac; shift
    done
    RUN=("$S/gre_run.sh" "$OUT" --gre "$GRE" --data "$DATA" --datasets fb,osm --indexes "$INDEXES"
         --warmup 20000000 --ops 100000000 --pin "$PIN" --repeats 3 ${ARGS[@]+"${ARGS[@]}"})
    if [ $MACHINE = mac ]; then exec "$S/mac_lane.sh" run -- "${RUN[@]}"; else exec "${RUN[@]}"; fi;;
  warmup)
    exec "$S/paired_run.sh" --gre "$GRE" "$@";;
  analyze|pair)
    exec python3 "$S/warmup_detect.py" "$CMD" "$@";;
  report)
    exec python3 "$S/gre_report.py" "$@";;
  counts)
    BIN=$ROOT/build/splice_count
    [ -x "$BIN" ] || die "no $BIN (./run.sh setup build)"
    need_quiet
    DSL=$ALL_DS; OUTD=$ROOT/results/splice_h
    FLAGS=()
    while [ $# -gt 0 ]; do
      case $1 in
        --datasets) [ $# -ge 2 ] || die "--datasets needs a list"; DSL=$2; shift 2;;
        --out) [ $# -ge 2 ] || die "--out needs a directory"; OUTD=$2; shift 2;;   # COUNTS.md goes there too
        *) FLAGS+=("$1"); shift;;
      esac
    done
    for ds in ${DSL//,/ }; do                          # docs/DESIGN.md section 9; unsorted sets read <ds>.sorted
      echo "== splice_count $ds"
      "$BIN" --dataset "$ds" --data "$DATA" --out "$OUTD" --threads 16 --grid pruned \
        --verify --parity --perturb ${FLAGS[@]+"${FLAGS[@]}"}
    done
    python3 "$S/splice_table.py" "$OUTD" > "$OUTD/COUNTS.md"
    echo "counts: $OUTD/<ds>.json, args/, COUNTS.md";;
  lane)
    exec "$S/mac_lane.sh" "$@";;
  atom)
    SUB=${1:-}; [ $# -eq 0 ] || shift
    case $SUB in
      doctor|calib|verdict|warmup|bench|run|counts|status|merge|anchors) ;;
      *) die "atom: SUB is one of doctor calib verdict warmup bench run counts status merge anchors (docs/ATOM_FAST.md)";;
    esac
    # this machine's GRE and data first, the caller's flags after them (argparse keeps the last value)
    exec python3 "$S/atom_batch.py" "$SUB" --gre "$GRE" --data "$DATA" "$@";;
  help|-h|--help)
    usage;;
  *)
    usage >&2; exit 2;;
esac
