#!/usr/bin/env bash
# Serial, resumable GRE microbench runs (the gre_lite.sh build) for the baseline indexes at 200M keys.
#
# Usage:
#   gre_run.sh OUTDIR --datasets fb,osm --indexes btree,alex,pgm,artunsync,lipp,sortedarray
#       [--gre DIR] [--data DIR] [--repeats 3] [--ops 100000000] [--warmup 0] [--init-ratio 1] [--pin -1]
#       [--table-size -1] [--read 1 --insert 0] [--resume] [-- extra microbench flags]
#   defaults: --gre /tmp/louisvasseur/GRE, --data /tmp/louisvasseur/SPLICE/experimental/scaleli/data/external/gre
#
# The standard conditions (read-only, single thread, every key bulk-loaded, 100M lookups of loaded keys):
#   GRE as published:  --warmup 0 --pin -1         (no warm-up, unpinned; the unpatched binary runs this too)
#   pinned:            --warmup 0 --pin 2          (separates the effect of pinning from that of the warm-up)
#   with warm-up:      --warmup 20000000 --pin 2   (AIDB-style: 20M untimed lookups, then 100M measured, on core 2)
# --warmup needs a read-only run (--read 1): with inserts it pre-trains ALEX's cost model. Runs with inserts on the
# patched build are only approximately comparable with unpatched GRE (allocator purges; gre_lite.sh header).
# Run under nohup so it survives the ssh session, one process at a time (it refuses to start beside another run):
#   nohup experimental/scaleli/tools/gre_run.sh OUTDIR --datasets ... --indexes ... > OUTDIR.out 2>&1 &
#   tail -f OUTDIR.out                         # one progress line per run
# Then: python3 experimental/scaleli/tools/gre_report.py OUTDIR > OUTDIR/report.md
# Stopped (kill, lost ssh session)? Rerun the same command with --resume: runs that passed every check are skipped,
# the others rerun (the old log is kept as *.log.prev). --resume may also raise --repeats, which is how conditions are
# interleaved in time (SERVER.md). Killing gre_run.sh also stops the microbench it started.
#
# The binary is copied to OUTDIR/bin first and only that copy runs; --resume keeps the copy. Its shared libraries
# still come from the conda environment (provenance records ldd). Loop order is repeat -> dataset -> index, so each
# repeat is a complete block. One process per run with OMP_NUM_THREADS=1 (GRE's operation sampling uses the default
# OpenMP team). GRE seeds its shuffle and sampling with --seed (default 1866), so repeats replay the same operations
# and differ only by measurement noise.
# Writes OUTDIR/provenance.txt, config.txt, runs.tsv (repeat dataset index exit_code wall_s log load1 busy: the 1-min
# load average and the number of other benchmark processes when the run started), logs/, gre_out.csv.
# GRE exits 0 on a bad file or index name, so a run counts as failed unless it exits 0, prints its throughput and,
# when read-only, finds every key (and every warm-up key, on the requested core). Exit status 1 if any run failed.
set -euo pipefail

# Flags added by gre_lite.sh's patch. They are passed only when non-default, so condition 1 needs no patch.
WARMUP_FLAG=warmup_num PIN_FLAG=pin_core
# Processes that mean the machine is not quiet: any GRE run (whatever the binary's name) and SPLICE's own benchmarks.
# '[-]-' keeps the pattern from starting with '-', which pgrep would read as an option.
BUSY=${GRE_RUN_BUSY:-'[-]-keys_file=|scaleli_bench|scaleli_hardness'}

die() { echo "gre_run.sh: $*" >&2; exit 2; }
usage() { sed -n '3,9p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }
# Decimal integers only: bash arithmetic reads a leading 0 as octal (08 is an error, 010 is 8).
int() { case $2 in 0|-1) ;; ''|-|*[!0-9-]*|?*-*|0*|-0*|???????????*) die "$1 must be an integer, got '$2'";; esac; }
range() { [ "$2" -ge "$3" ] && [ "$2" -le "$4" ] || die "$1 must be in $3..$4, got $2"; }
# A decimal in [0, 1] (GRE reads ratios with stod); $3 = 1 also requires it to be above 0.
ratio() {
  case $2 in ''|.|*[!0-9.]*|*.*.*) die "$1 must be a number in [0, 1], got '$2'";; esac
  awk -v x="$2" -v pos="$3" 'BEGIN { exit !(x <= 1 && (pos ? x > 0 : x >= 0)) }' || die "$1 must be in [0, 1], got $2"
}

CMDLINE=$(printf '%q ' "$0" "$@")
[ $# -ge 1 ] || usage
case $1 in -h|--help) usage 0;; -*) usage;; esac
OUT=$1; shift
case $OUT in /*) ;; *) OUT=$PWD/$OUT;; esac        # absolute, so the recorded commands stand alone
GRE=/tmp/louisvasseur/GRE
DATA=/tmp/louisvasseur/SPLICE/experimental/scaleli/data/external/gre
DATASETS= INDEXES= REPEATS=3 OPS=100000000 WARMUP=0 INIT_RATIO=1 PIN=-1 TABLE_SIZE=-1 READ=1 INSERT=0 RESUME=0
EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in --resume|--|-h|--help) ;; --*) [ $# -ge 2 ] && [ -n "$2" ] || die "$1 needs a value";; esac
  case $1 in
    --gre) GRE=$2; shift 2;;
    --data) DATA=$2; shift 2;;
    --datasets) DATASETS=$2; shift 2;;
    --indexes) INDEXES=$2; shift 2;;
    --repeats) REPEATS=$2; shift 2;;
    --ops) OPS=$2; shift 2;;
    --warmup) WARMUP=$2; shift 2;;
    --init-ratio) INIT_RATIO=$2; shift 2;;
    --pin) PIN=$2; shift 2;;
    --table-size) TABLE_SIZE=$2; shift 2;;
    --read) READ=$2; shift 2;;
    --insert) INSERT=$2; shift 2;;
    --resume) RESUME=1; shift;;
    -h|--help) usage 0;;
    --) shift; EXTRA=("$@"); break;;
    *) die "unknown option '$1' (microbench flags go after --)";;
  esac
done
[ -n "$DATASETS" ] || die "--datasets is required"
[ -n "$INDEXES" ] || die "--indexes is required"
int --repeats "$REPEATS"; int --ops "$OPS"; int --warmup "$WARMUP"; int --pin "$PIN"; int --table-size "$TABLE_SIZE"
range --repeats "$REPEATS" 1 1000
# stoi in GRE: operations_num and table_size must fit in an int; so must the warm-up count (the patch checks it)
range --ops "$OPS" 1 2147483647
range --warmup "$WARMUP" 0 2147483647
[ "$TABLE_SIZE" = -1 ] || range --table-size "$TABLE_SIZE" 1 2147483647
NCPU=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)
range --pin "$PIN" -1 $((NCPU - 1))
ratio --init-ratio "$INIT_RATIO" 1; ratio --read "$READ" 0; ratio --insert "$INSERT" 0
READ_ONLY=0; awk -v x="$READ" 'BEGIN { exit !(x == 1) }' && READ_ONLY=1
[ "$WARMUP" = 0 ] || [ "$READ_ONLY" = 1 ] || die "--warmup needs --read 1: with inserts it changes ALEX's structure"
IFS=, read -r -a DS <<< "$DATASETS"
IFS=, read -r -a IX <<< "$INDEXES"
[ ${#DS[@]} -gt 0 ] && [ ${#IX[@]} -gt 0 ] || die "no name in --datasets or --indexes"
for x in "${DS[@]}" "${IX[@]}"; do [ -n "$x" ] || die "empty name in --datasets or --indexes"; done

# Check everything that would otherwise fail hours in: data files, binary, patch flags, index names.
BUILD=$GRE/build
for ds in "${DS[@]}"; do
  f=$DATA/$ds
  [ -f "$f" ] && [ -r "$f" ] || die "dataset file $f not found"
  # SOSD layout: uint64 count, then count uint64 keys
  n=$(od -An -t u8 -N 8 "$f" | tr -d ' '); sz=$(wc -c < "$f" | tr -d ' ')
  [ -n "$n" ] && [ "$sz" -eq $(( 8 + 8 * n )) ] || die "$f: header says $n keys but the file has $sz bytes"
done
CONFIG="gre=$GRE
data=$DATA
datasets=$DATASETS
indexes=$INDEXES
ops=$OPS
warmup=$WARMUP
pin=$PIN
init_ratio=$INIT_RATIO
table_size=$TABLE_SIZE
read=$READ
insert=$INSERT
extra=${EXTRA[*]+${EXTRA[*]}}"
NRUNS=0
[ -f "$OUT/runs.tsv" ] && NRUNS=$(( $(wc -l < "$OUT/runs.tsv") - 1 ))
if [ "$NRUNS" -gt 0 ] && [ "$RESUME" = 0 ]; then
  die "$OUT already has $NRUNS runs; pass --resume to continue it, or choose a new OUTDIR"
fi
# A resumed run must measure the same condition with the same binary; only --repeats may grow.
if [ "$RESUME" = 1 ] && [ -f "$OUT/config.txt" ] && [ "$(grep -v '^repeats=' "$OUT/config.txt")" != "$CONFIG" ]; then
  echo "saved config ($OUT/config.txt):" >&2; grep -v '^repeats=' "$OUT/config.txt" >&2
  echo "this command:" >&2; echo "$CONFIG" >&2
  die "--resume with different settings; use a new OUTDIR"
fi
BIN=$OUT/bin/microbench
SRC=$BUILD/microbench
[ "$NRUNS" -gt 0 ] && [ -x "$BIN" ] && SRC=$BIN
[ -x "$SRC" ] || die "no $SRC (build it with gre_lite.sh)"
# String literals in the binary: GRE silently ignores a flag it does not know, and exits 0 on an unknown index.
if [ "$WARMUP" != 0 ] || [ "$PIN" != -1 ]; then
  for s in gre_lite_patch "$WARMUP_FLAG" "$PIN_FLAG"; do
    grep -a -q -F "$s" "$SRC" || die "--warmup/--pin need the gre_lite.sh patch; '$s' is not in $SRC"
  done
fi
# gre_lite.sh compiles its index list into get_index's error message: '(gre_lite: alex lipp ... sortedarray)'.
KNOWN=$(grep -a -o 'gre_lite: [a-z ]*)' "$SRC" | sed -n '1{s/^gre_lite: //;s/)$//;p;}' || true)
for idx in "${IX[@]}"; do
  case $idx in *[!a-z0-9_]*) die "index '$idx' is not a GRE index name";; esac
  if [ -n "$KNOWN" ]; then
    case " $KNOWN " in *" $idx "*) ;; *) die "index '$idx' is not one of: $KNOWN";; esac
  else  # not a gre_lite build: only a substring test is possible
    grep -a -q -F "$idx" "$SRC" || die "index '$idx' is not in $SRC"
  fi
done

LOCK=
child=
cleanup() { [ -n "$child" ] && kill "$child" 2>/dev/null; [ -n "$LOCK" ] && rm -rf "$LOCK"; return 0; }
trap cleanup EXIT
trap 'echo "gre_run.sh: interrupted, stopping the current run" >&2; exit 143' INT TERM HUP
if pgrep -f "$BUSY" > /dev/null; then die "another benchmark is running (pgrep -f '$BUSY'); wait for it"; fi
mkdir -p "$OUT/logs" "$OUT/bin"
if ! mkdir "$OUT/.lock" 2>/dev/null; then
  pid=$(cat "$OUT/.lock/pid" 2>/dev/null || echo)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && die "gre_run.sh (pid $pid) is already writing to $OUT"
  rm -rf "$OUT/.lock"; mkdir "$OUT/.lock"           # stale lock from a killed run
fi
LOCK=$OUT/.lock; echo $$ > "$LOCK/pid"
printf '%s\nrepeats=%s\n' "$CONFIG" "$REPEATS" > "$OUT/config.txt"
if [ "$SRC" = "$BIN" ]; then
  cmp -s "$BIN" "$BUILD/microbench" 2>/dev/null || echo "note: $BUILD/microbench differs from the copy this OUTDIR started with; resuming with the copy"
else
  cp "$SRC" "$BIN"
  cp "$BUILD/gre_lite_build.txt" "$OUT/bin/" 2>/dev/null || echo "(no $BUILD/gre_lite_build.txt)" > "$OUT/bin/gre_lite_build.txt"
fi

sha() { { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"; } | cut -d' ' -f1; }
section() { local name=$1 out; shift; echo "## $name"; out=$("$@" 2>/dev/null) && [ -n "$out" ] && echo "$out" || echo "(unavailable)"; }
governors() { cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c; }
loadavg() { cat /proc/loadavg 2>/dev/null || uptime; }
splice_sha() {
  local d h; d=$(cd "$(dirname "$0")" && pwd); h=$(git -C "$d" rev-parse HEAD) || return 1
  echo "$h ($(git -C "$d" status --porcelain | wc -l | tr -d ' ') uncommitted paths)"
}
datasets() { local d; for d in "${DS[@]}"; do ls -lL "$DATA/$d"; done; }
# Page and allocator policy: THP changes both the RSS method and TLB-bound throughput (SCALE-LI's protocol uses never).
thp() {
  local d=/sys/kernel/mm/transparent_hugepage f
  [ -d "$d" ] || return 1
  for f in enabled defrag khugepaged/max_ptes_none; do echo "$f: $(cat "$d/$f")"; done
}
malloc_conf() { echo "MALLOC_CONF=${MALLOC_CONF-(unset)}"; [ ! -L /etc/malloc.conf ] || echo "/etc/malloc.conf -> $(readlink /etc/malloc.conf)"; }
if [ "$NRUNS" -le 0 ] || [ ! -f "$OUT/provenance.txt" ]; then
  { section date date -u '+%Y-%m-%dT%H:%M:%SZ'
    section hostname hostname
    section uname uname -a
    section lscpu lscpu
    section governors governors
    section free free -g
    section nproc nproc
    section loadavg loadavg
    section thp thp
    section numa_balancing cat /proc/sys/kernel/numa_balancing
    section malloc_conf malloc_conf
    section splice splice_sha
    section gre_lite_build cat "$OUT/bin/gre_lite_build.txt"
    section binary sha "$BIN"
    section ldd ldd "$BIN"
    section datasets datasets
    section config cat "$OUT/config.txt"
    section command echo "$CMDLINE"
  } > "$OUT/provenance.txt"
else
  { echo; section resumed date -u '+%Y-%m-%dT%H:%M:%SZ'; section loadavg loadavg; section command echo "$CMDLINE"
  } >> "$OUT/provenance.txt"
fi
[ -f "$OUT/runs.tsv" ] || printf 'repeat\tdataset\tindex\texit_code\twall_s\tlog\tload1\tbusy\n' > "$OUT/runs.tsv"

now() { local t; t=$(date +%s.%N); case $t in *N) date +%s;; *) echo "$t";; esac; }   # BSD date has no %N
val() { awk -v k="$1" 'index($0, k) == 1 { print substr($0, length(k) + 1); exit }' "$2"; }
load1() { { cut -d' ' -f1 /proc/loadavg || sysctl -n vm.loadavg | awk '{ print $2 }'; } 2>/dev/null || echo -; }
busy() { { pgrep -f "$BUSY" || true; } | wc -l | tr -d ' '; }   # our own previous child has exited by now
# check_run EXIT_CODE LOG: sets tp sr rss and bad (empty when the run passes every check)
check_run() {
  local rc=$1 log=$2 wsr pc
  tp=$(val 'Throughput = ' "$log"); sr=$(val 'success_read: ' "$log"); rss=$(val 'index_rss_bytes: ' "$log")
  wsr=$(val 'warmup_success_read: ' "$log"); pc=$(val 'pinned_core: ' "$log")
  bad=
  [ "$rc" = 0 ] || bad="exit $rc"
  [ -n "$tp" ] || bad="${bad:+$bad, }no throughput"
  [ "$READ_ONLY" != 1 ] || [ "$sr" = "$OPS" ] || bad="${bad:+$bad, }success_read ${sr:-missing} != $OPS"
  [ "$WARMUP" = 0 ] || [ "$wsr" = "$WARMUP" ] || bad="${bad:+$bad, }warmup_success_read ${wsr:-missing} != $WARMUP"
  [ "$PIN" = -1 ] || [ "$pc" = "$PIN" ] || bad="${bad:+$bad, }not pinned to core $PIN"
  return 0
}
recorded_ok() {  # repeat dataset index: the last row for it passes the same checks as a fresh run
  local rc log=$OUT/logs/${2}__${3}__r$1.log
  rc=$(awk -F'\t' -v r="$1" -v d="$2" -v i="$3" '$1 == r && $2 == d && $3 == i { c = $4; f = 1 }
       END { if (f) print c }' "$OUT/runs.tsv")
  [ -n "$rc" ] && [ -f "$log" ] || return 1
  check_run "$rc" "$log"
  [ -z "$bad" ]
}

PATCH=()
[ "$WARMUP" != 0 ] && PATCH+=("--$WARMUP_FLAG=$WARMUP")
[ "$PIN" != -1 ] && PATCH+=("--$PIN_FLAG=$PIN")
TOTAL=$(( REPEATS * ${#DS[@]} * ${#IX[@]} ))
k=0 failed=0 skipped=0
echo "$(date '+%F %T') $TOTAL runs into $OUT (warmup $WARMUP, pin $PIN, ops $OPS, init ratio $INIT_RATIO)"
for r in $(seq 1 "$REPEATS"); do
  for ds in "${DS[@]}"; do
    for idx in "${IX[@]}"; do
      k=$((k + 1))
      rel=logs/${ds}__${idx}__r$r.log; log=$OUT/$rel
      if [ "$RESUME" = 1 ] && recorded_ok "$r" "$ds" "$idx"; then skipped=$((skipped + 1)); continue; fi
      [ -f "$log" ] && mv "$log" "$log.prev"            # keep the failed attempt next to the new one
      cmd=("$BIN" --keys_file="$DATA/$ds" --keys_file_type=binary --read="$READ" --insert="$INSERT"
           --operations_num="$OPS" --table_size="$TABLE_SIZE" --init_table_ratio="$INIT_RATIO" --thread_num=1 --memory
           --index="$idx" --output_path="$OUT/gre_out.csv" ${PATCH[@]+"${PATCH[@]}"} ${EXTRA[@]+"${EXTRA[@]}"})
      echo "cmd: OMP_NUM_THREADS=1 ${cmd[*]}" > "$log"
      l1=$(load1); nb=$(busy); bnote=
      [ "$nb" = 0 ] || bnote="  BUSY ($nb other benchmark processes at the start: pids $({ pgrep -f "$BUSY" || true; } | tr '\n' ' '))"
      t0=$(now); rc=0
      OMP_NUM_THREADS=1 "${cmd[@]}" >> "$log" 2>&1 & child=$!
      wait "$child" || rc=$?
      child=
      wall=$(awk -v a="$t0" -v b="$(now)" 'BEGIN { printf "%.1f", b - a }')
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$ds" "$idx" "$rc" "$wall" "$rel" "$l1" "$nb" >> "$OUT/runs.tsv"
      check_run "$rc" "$log"
      [ -z "$bad" ] || failed=$((failed + 1))
      mops=-; [ -z "$tp" ] || mops=$(awk -v t="$tp" 'BEGIN { printf "%.3f", t / 1e6 }')
      printf '%s [%d/%d] r%s %-8s %-12s %7ss  %s Mops/s  success_read %s%s%s\n' "$(date '+%F %T')" "$k" "$TOTAL" "$r" \
          "$ds" "$idx" "$wall" "$mops" "${sr:--}" "${rss:+  index_rss_bytes $rss}$bnote" "${bad:+  FAILED ($bad)}"
    done
  done
done
echo "$(date '+%F %T') done: $((k - skipped)) runs, $skipped skipped (already done), $failed failed"
[ "$failed" = 0 ] || exit 1
