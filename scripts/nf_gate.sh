#!/usr/bin/env bash
# SPDX-License-Identifier: MIT (SPLICE)
# nf_gate.sh: build/nf_gate over the datasets (full scale + 8 windows of 12.5M keys each), one dataset at a time,
# appending to OUT/cases.jsonl (stderr to OUT/<ds>.log). Counts, not timings: on the Mac it runs at background QoS
# (efficiency cores) with 4 build threads, so a timed batch beside it stays clean.
#   scripts/nf_gate.sh BIN OUT [DATASETS] [nf_gate flags...]   DATASETS default fb,osm,books,covid,...,wise
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ $# -ge 2 ] || { echo "usage: $0 BIN OUT [DATASETS] [FLAGS...]" >&2; exit 2; }
BIN=$1; OUT=$2; shift 2
DSL=fb,osm,books,covid,genome,history,libio,planet,stack,wise
[ $# -gt 0 ] && [ "${1#--}" = "$1" ] && { DSL=$1; shift; }
period() { case $1 in fb|osm|planet) echo s512;; *) echo s64;; esac; }   # gre_run.sh period()
QOS=(); [ "$(uname -s)" = Darwin ] && QOS=(taskpolicy -c background)
mkdir -p "$OUT"
IFS=, read -r -a DS <<< "$DSL"
for d in "${DS[@]}"; do
  f=$ROOT/data/external/gre/$d
  [ -f "$f.sorted" ] && f=$f.sorted     # the Mac's sorted copy of an unsorted set
  flow=$ROOT/data/flows/${d}_$(period "$d")_t2000.txt
  echo "[nf_gate.sh] $(date +%T) $d" >&2
  "${QOS[@]+"${QOS[@]}"}" "$BIN" --data "$f" --flow "$flow" --name "$d" --threads 4 --full 1 "$@" >> "$OUT/cases.jsonl" 2> "$OUT/$d.log"
done
