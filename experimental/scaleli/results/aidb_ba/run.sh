#!/bin/zsh
REPO=${SPLICE_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}  # repo root (scripts live in experimental/scaleli/results/<dir>/)
# Serial 200M hardness runs. Waits for any other scaleli job to finish before each run.
H=${SCALELI_HARDNESS:-$REPO/build-fs/scaleli_hardness}
D=$REPO/experimental/scaleli/data/external/gre
F=$REPO/experimental/scaleli/results/aidb_flowv2
O=$REPO/experimental/scaleli/results/aidb_ba/out
typeset -A cfg; cfg=(books s64 fb s512 osm s512 covid s64 genome s64 history s64 libio s64 planet s512 stack s64 wise s64)
datafile() { if [[ -f $D/$1.sorted ]]; then echo $D/$1.sorted; else echo $D/$1; fi }
waitfree() { while pgrep -f "scaleli_bench|scaleli_hardness" >/dev/null; do sleep 5; done }
run() { # name scope args...
  local d=$1 s=$2; shift 2; local out=$O/${d}_${s}.json
  [[ -s $out ]] && { echo "skip $d $s"; return; }
  waitfree; local t0=$(date +%s)
  $H --data $(datafile $d) --dtype uint64 --region-keys 4096 --pla-eps 32,4096 --check-sorted 1 --threads 16 --verbose 1 "$@" > $out.tmp 2> $O/${d}_${s}.err && mv $out.tmp $out
  echo "$d $s rc=$? wall=$(( $(date +%s) - t0 ))s $(date +%T)"
}
phase=$1; shift
for d in "$@"; do
  case $phase in
    flow) run $d free --flow $F/flows_free/${d}_${cfg[$d]}_t2000.txt; run $d mono --flow $F/flows_monotone/${d}_mono.txt ;;
    csv)  run $d csv --virtual-alpha 0.1 ;;
  esac
done
echo PHASE_DONE $phase
