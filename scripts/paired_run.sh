#!/usr/bin/env bash
# SPDX-License-Identifier: MIT (SPLICE)
# Paired warm-up batch: the same GRE trace protocol on the Mac and on the Atom, compared index by index with
# `scripts/warmup_detect.py pair`. Run it with the same --tag on both machines; each writes
# results/paired/<tag>/<machine>/ (runs/ = the gre_run.sh OUTDIR, machine.txt, warmup.md and .json when numpy is
# installed). The Atom's folder comes back through git (commit and push on the Atom, pull on the Mac).
#   Protocol: GRE_TRACE=geo:64:65536 GRE_TRACE_CPU=1, --warmup 0, 120M lookups (the 20M + 100M of the headline
#   protocol, all traced), every round interleaved over datasets x indexes, the same plan files for splice.
#   Mac: the native GRE of scripts/gre_mac.sh, unpinned (macOS cannot pin), each round inside scripts/mac_lane.sh (heavy
#   apps on the efficiency cores) unless --no-lane.
#   Linux (the Atom): the scripts/gre_lite.sh build, pinned with --pin (default 2; keep the core sharing its L2 idle).
# Usage: scripts/paired_run.sh --tag NAME [--datasets fb] [--rounds 3] [--indexes LIST] [--ops 120000000]
#                            [--gre DIR] [--data DIR] [--pin N] [--machine NAME] [--no-lane]
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TAG= DATASETS=fb ROUNDS=3 OPS=120000000 INDEXES= GRE= DATA=$ROOT/data/external/gre PIN= MACHINE= LANE=1
die() { echo "paired_run.sh: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case $1 in
    --tag) TAG=$2; shift 2;; --datasets) DATASETS=$2; shift 2;; --rounds) ROUNDS=$2; shift 2;;
    --indexes) INDEXES=$2; shift 2;; --ops) OPS=$2; shift 2;; --gre) GRE=$2; shift 2;; --data) DATA=$2; shift 2;;
    --pin) PIN=$2; shift 2;; --machine) MACHINE=$2; shift 2;; --no-lane) LANE=0; shift;;
    -h|--help) sed -n '3,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) die "unknown argument $1";;
  esac
done
[ -n "$TAG" ] || die "--tag NAME is required (use the same tag on both machines)"
case $TAG in *[!A-Za-z0-9._-]*) die "--tag: letters, digits, . _ - only";; esac
BASE=trace_nullindex,trace_splice,trace_lipp,trace_pgm,trace_alex,trace_btree,trace_sortedarray
if [ "$(uname -s)" = Darwin ]; then
  OSK=mac; : "${GRE:=$HOME/gre_mac/GRE}"; : "${PIN:=-1}"; : "${INDEXES:=$BASE}"
  : "${MACHINE:=mac-$(sysctl -n machdep.cpu.brand_string | tr -cs 'A-Za-z0-9' '-' | sed 's/-$//')}"
else
  OSK=linux; LANE=0; : "${GRE:=/tmp/louisvasseur/GRE}"; : "${PIN:=2}"; : "${INDEXES:=$BASE,trace_artunsync}"
  if grep -qi 'atom' /proc/cpuinfo 2>/dev/null; then : "${MACHINE:=atom}"; else : "${MACHINE:=$(hostname -s)}"; fi
fi
OUT=$ROOT/results/paired/$TAG/$MACHINE
MB=$GRE/build/microbench
[ -x "$MB" ] || die "no GRE build at $MB: run scripts/$([ $OSK = mac ] && echo gre_mac.sh || echo gre_lite.sh) first"
for i in ${INDEXES//,/ }; do   # the registry names are string literals in the binary
  LC_ALL=C grep -aqw -- "$i" "$MB" || die "the GRE build at $GRE has no index $i: rebuild it from this checkout"
done
for d in ${DATASETS//,/ }; do
  [ -f "$DATA/$d" ] || die "dataset $DATA/$d missing (data/external/gre/download.sh; unsorted sets are <name>.sorted)"
done
mkdir -p "$OUT"

{ echo "tag: $TAG"; echo "machine: $MACHINE"; echo "date_utc: $(date -u +%FT%TZ)"; echo "uname: $(uname -a)"
  echo "commit: $(git -C "$ROOT" rev-parse HEAD) $(git -C "$ROOT" status --porcelain --untracked-files=no | grep -q . && echo dirty || echo clean)"
  echo "protocol: GRE_TRACE=geo:64:65536 GRE_TRACE_CPU=1 warmup=0 ops=$OPS rounds=$ROUNDS pin=$PIN lane=$LANE"
  echo "datasets: $DATASETS"; echo "indexes: $INDEXES"; echo "gre: $GRE"; echo "page_size: $(getconf PAGESIZE)"
  if [ $OSK = mac ]; then
    echo "cpu: $(sysctl -n machdep.cpu.brand_string); P $(sysctl -n hw.perflevel0.logicalcpu) / E $(sysctl -n hw.perflevel1.logicalcpu); P per L2 $(sysctl -n hw.perflevel0.cpusperl2)"
    echo "memory_bytes: $(sysctl -n hw.memsize)"; echo "power: $(pmset -g batt | head -1)"
  else
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)"; echo "memory: $(grep MemTotal /proc/meminfo)"
    echo "thp: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu$PIN/cpufreq/scaling_governor 2>/dev/null)"
    echo "l2_siblings_of_pin: $(cat /sys/devices/system/cpu/cpu$PIN/cache/index2/shared_cpu_list 2>/dev/null)"
  fi
  echo "load_at_start: $(uptime | sed 's/.*load average[s]*: //')"
  echo "gre_build: $(cksum < "$MB")"
} > "$OUT/machine.txt"
cat "$OUT/machine.txt"

export GRE_TRACE=geo:64:65536 GRE_TRACE_CPU=1
for k in $(seq 1 "$ROUNDS"); do
  echo "== round $k/$ROUNDS $(date)"
  cmd=("$ROOT/scripts/gre_run.sh" "$OUT/runs" --gre "$GRE" --data "$DATA" --datasets "$DATASETS" --indexes "$INDEXES"
       --ops "$OPS" --warmup 0 --pin "$PIN" --repeats "$k" --resume)
  if [ $LANE = 1 ]; then "$ROOT/scripts/mac_lane.sh" run -- "${cmd[@]}"; else "${cmd[@]}"; fi
  rc=$?; [ $rc = 2 ] && die "gre_run.sh refused (see above)"
done

if python3 -c 'import numpy' 2>/dev/null; then
  plot=; python3 -c 'import matplotlib' 2>/dev/null && plot="--plot $OUT/plots"
  python3 "$ROOT/scripts/warmup_detect.py" analyze "$OUT/runs" --json "$OUT/warmup.json" $plot > "$OUT/warmup.md" &&
    echo "analysis: $OUT/warmup.md"
else
  echo "numpy missing: the analysis runs on the Mac after the pull"
fi
echo
echo "Done. To send this machine's results (the GRE binary in runs/bin/ is ignored by .gitignore):"
echo "  git -C $ROOT add results/paired/$TAG/$MACHINE && git -C $ROOT commit -m 'paired $TAG: $MACHINE' && git -C $ROOT push"
echo "Then on the other machine: git pull && python3 scripts/warmup_detect.py pair results/paired/$TAG/<a>/runs results/paired/$TAG/<b>/runs --plot results/paired/$TAG/pair"
