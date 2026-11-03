#!/bin/bash
# After the scan batch: the fixes on x86 and all ten datasets, then exact byte budgets of 20, 23 and 25, one stage at a time so that
# only the timed stages' four lanes are ever running while anything is timed.
source /scratch/lvasseur/jed_env.sh
export REPO=/scratch/lvasseur/GRE-hash GRE_DIR=/scratch/lvasseur/gre-hash/GRE
ROOT=$REPO/results/jed200M_budget_2026-11-03; RF=/scratch/lvasseur/hash_incoming/genome.rule
until [ -e "$REPO/results/jed200M_scan_2026-11-02/DONE" ]; do sleep 20; done
mkdir -p "$ROOT/reg" "$ROOT/bud" "$ROOT/rep" "$ROOT/scan" "$ROOT/cnt"; cd /scratch/lvasseur
cp hash_incoming/*.h GRE-hash/include/gret/; cp hash_incoming/test_scan.cpp hash_incoming/test_worstcase.cpp hash_incoming/test_splice_edge.cpp GRE-hash/tests/
echo "== $(date +%T) build"; bash /scratch/lvasseur/hash_build.sh 2>&1 | tail -6 | cut -c1-110
for t in "$REPO/tests/test_scan.cpp" "$REPO/tests/test_worstcase.cpp" hash_incoming/abcount.cpp hash_incoming/scanbench.cpp; do
  "$REPO/build/tests/cxx-O3" "$t" -o "$ROOT/$(basename "$t" .cpp)" 2>&1 | grep -E " error " | head -5
done
taskset -c 2 "$ROOT/test_scan" 2>/dev/null | tail -1 | tee "$ROOT/tests.txt" | cut -c1-110
taskset -c 2 "$ROOT/test_worstcase" 2000000 2>/dev/null | tee -a "$ROOT/tests.txt" | tail -1 | cut -c1-110
[ "$(grep -c ': ok' "$ROOT/tests.txt")" = 2 ] || { echo "== tests failed, no timing"; touch "$ROOT/DONE"; exit 1; }

run_arm() {  # lane root idx ds budget leaf tau v passes
  local c=$1 root=$2 idx=$3 ds=$4 b=$5 leaf=$6 tau=$7 v=$8 passes=$9 rf="" spk bud=0 p="1"
  [ "$idx" = gr4 ] && rf=$RF
  case $idx in ab4|gr4) spk=1.25; bud=$b;; *) spk=$(python3 -c "print(($b-0.005)/16)");; esac
  [ "$b" = 0 ] && { bud=0; spk=1.25; }
  [ "$passes" -ge 3 ] && p="1 2 3"; [ "$passes" = 2 ] && p="1 2"
  local tag=${idx}__b18_s${b}_t${tau}_l${leaf}_c${v}__${ds} out=$root/${idx}__b18_s${b}_t${tau}_l${leaf}_c${v}__${ds}
  mkdir -p "$out"; echo "$ds $idx" > "$out/cells.txt"
  RULE_FILE=$rf RADIX_BITS=18 HASHLINE_SPK=$spk AB_BUDGET=$bud AB_V=$v AR_TAU=$tau AR_LEAF=$leaf CORE=$c J=1 HEAVY="^$" HASH_DATA=0 PASSES="$p" /scratch/lvasseur/bin/nothp "$REPO/scripts/run_parallel.sh" "$out" "$out/cells.txt" > /dev/null 2>&1
  echo "$(date +%T) lane $c done $tag" >> "$root/progress.txt"
}
lanes() {  # file root : lines of idx ds budget leaf tau v passes
  local f=$1 root=$2
  awk -v m=4 -v o="$root/lane" '{print > (o (NR-1)%m)}' "$f"
  local j=0
  for c in 2 20 38 56; do
    ( [ -f "$root/lane$j" ] && while read -r idx ds b leaf tau v passes; do run_arm $c "$root" $idx $ds $b $leaf $tau $v $passes; done < "$root/lane$j" ) &
    j=$((j+1))
  done
  wait
}
ALL="fb genome osm planet books covid libio history stack wise"
# 1. the fixes against the earlier numbers, same settings
: > "$ROOT/reg/cells"
for d in $ALL; do
  case $d in fb|genome|osm|planet) l=8; t=2;; *) l=64; t=3;; esac
  echo "hf4 $d 0 0 0 0 3" >> "$ROOT/reg/cells"
  echo "ab4 $d 0 $l $t 0 3" >> "$ROOT/reg/cells"
  echo "ph4 $d 0 65536 0 0 1" >> "$ROOT/reg/cells"
done
lanes "$ROOT/reg/cells" "$ROOT/reg"
python3 /scratch/lvasseur/arm_agg2.py "$ROOT/reg" | tail -1
echo "== $(date +%T) REGRESSION_DONE"
# 2. scans again, ab4 with the skip and the galloping start
: > "$ROOT/scan/cells"
for d in $ALL; do
  case $d in fb|genome|osm|planet) l=8; t=2;; *) l=64; t=3;; esac
  echo "ab4 $d $l $t" >> "$ROOT/scan/cells"
done
echo "gr4 genome 16 2" >> "$ROOT/scan/cells"
awk -v m=4 -v o="$ROOT/scan/lane" '{print > (o (NR-1)%m)}' "$ROOT/scan/cells"
j=0; for c in 2 20 38 56; do
  ( while read -r idx ds l t; do [ "$idx" = gr4 ] && rf=$RF || rf=""; RULE_FILE=$rf RADIX_BITS=18 HASHLINE_SPK=1.25 AR_LEAF=$l AR_TAU=$t taskset -c $c /scratch/lvasseur/bin/nothp "$ROOT/scanbench" "$DATA_DIR/$ds" "$idx" "$ds" >> "$ROOT/scan/out_$c.tsv" 2>> "$ROOT/scan/err.txt"; done < "$ROOT/scan/lane$j" ) &
  j=$((j+1)); done; wait
cat "$ROOT"/scan/out_*.tsv > "$ROOT/scan/out.tsv"
echo "== $(date +%T) SCAN2_DONE"
# 3. exact counts, nothing timed, all cores
python3 - "$ROOT" <<'PY'
import sys
root=sys.argv[1]; c=[]
ALL="fb genome osm planet books covid libio history stack wise".split()
for ds in ALL:
    for b in (20,23,25):
        for leaf in (4,8,16,64):
            for tau in ("2","3"):
                for v in (0,1):
                    c.append(("ab4",ds,b,leaf,tau,v))
                    if ds=="genome": c.append(("gr4",ds,b,leaf,tau,v))
open(root+"/cnt/cells","w").write("".join(" ".join(map(str,x))+"\n" for x in c))
print(len(c),"count cells")
PY
count1() { read -r idx ds b leaf tau v <<< "$1"; [ "$idx" = gr4 ] && rf=$RF || rf=""
  line=$(RULE_FILE=$rf RADIX_BITS=18 AB_BUDGET=$b AB_V=$v AR_TAU=$tau AR_LEAF=$leaf "$ROOT/abcount" "$DATA_DIR/$ds" "$idx" 2>&1 | grep "^gret: " | tail -1)
  echo "$idx $ds $b $leaf $tau $v $line" >> "$ROOT/cnt/raw_$BASHPID.txt"; }
export -f count1; export ROOT RF DATA_DIR
xargs -a "$ROOT/cnt/cells" -P 32 -I{} bash -c 'count1 "{}"'
python3 - "$ROOT" <<'PY'
import glob,re,sys,csv
root=sys.argv[1]
w=csv.writer(open(root+"/counts.tsv","w"),delimiter="\t")
w.writerow("idx ds budget leaf tau v lines p999 max levels dir_bytes Bkey build_ms nodes".split())
def g(p,l):
    m=re.search(p,l); return m[1] if m else ""
n=0
for f in glob.glob(root+"/cnt/raw_*.txt"):
    for l in open(f):
        a=l.split(None,6)
        if len(a)<7 or "infeasible" in l: w.writerow(a[:6]+[""]*8); continue
        t=a[6]
        w.writerow(a[:6]+[g(r"E\[buckets\]=([\d.]+)",t),g(r"lines_p999=(\d+)",t),g(r"lines_max=(\d+)",t),g(r"E\[levels\]=([\d.]+)",t),g(r"dir_bytes=(\d+)",t),g(r"bytes_per_key=([\d.]+)",t),g(r"build_ms=([\d.]+)",t),g(r"nodes=(\d+)",t)]); n+=1
print(n,"counted")
PY
echo "== $(date +%T) COUNTS_DONE"
# 4. timed, one pass each
python3 /scratch/lvasseur/hash_incoming/budget_pick.py "$ROOT"
awk '{print $1,$2,$3,$4,$5,$6,1}' "$ROOT/arms.txt" > "$ROOT/bud/cells"
lanes "$ROOT/bud/cells" "$ROOT/bud"
python3 /scratch/lvasseur/arm_agg2.py "$ROOT/bud" | tail -1
echo "== $(date +%T) BUDGET_DONE"
# 5. two more passes on the two best order-keeping arms and the best hash per dataset and budget
python3 - "$ROOT" <<'PY'
import csv,sys,collections
root=sys.argv[1]; t=collections.defaultdict(list)
for r in csv.DictReader(open(root+"/bud/table.tsv"),delimiter="\t"):
    t[(r["dataset"],r["spk"])].append(r)
out=[]
for k,rs in t.items():
    o=sorted([r for r in rs if r["index"] in("ab4","gr4")],key=lambda r:-float(r["Mops"]))[:2]
    h=sorted([r for r in rs if r["index"] in("hf4","ph4")],key=lambda r:-float(r["Mops"]))[:2]
    for r in o+h: out.append((r["index"],r["dataset"],r["spk"],r["leaf"],r["tau"],r["cap"],2))
open(root+"/rep/cells","w").write("".join(" ".join(map(str,x))+"\n" for x in out))
print(len(out),"repeat cells")
PY
lanes "$ROOT/rep/cells" "$ROOT/rep"
python3 /scratch/lvasseur/arm_agg2.py "$ROOT/rep" | tail -1
echo "== $(date +%T) BUDGET_BATCH_DONE"; touch "$ROOT/DONE"
