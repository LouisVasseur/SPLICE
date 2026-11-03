#!/bin/bash
# On the finished code: the walk against the earlier numbers, exact counts, every distinct cell of the grid timed, more passes on the
# leaders, scans and chained latency on the winners. One stage at a time, four lanes when timing.
source /scratch/lvasseur/jed_env.sh
export REPO=/scratch/lvasseur/GRE-hash GRE_DIR=/scratch/lvasseur/gre-hash/GRE
ROOT=$REPO/results/jed200M_final_2026-11-03; IN=/scratch/lvasseur/final_incoming; G=$REPO/results/jed200M_g_2026-11-03
until [ -e "$G/DONE" ]; do sleep 20; done
echo "== $(date +%T) G batch done"
mkdir -p "$ROOT/reg" "$ROOT/cnt" "$ROOT/fin" "$ROOT/rep" "$ROOT/scan" "$ROOT/chain"; cd /scratch/lvasseur
cp $IN/*.h GRE-hash/include/gret/; cp $IN/test_scan.cpp $IN/test_worstcase.cpp $IN/test_splice_edge.cpp GRE-hash/tests/
bash /scratch/lvasseur/hash_build.sh 2>&1 | tail -3 | cut -c1-110
{ date; sha256sum "$GRE_DIR"/build/microbench* 2>/dev/null; cat GRE-hash/include/gret/*.h | cksum; cat $IN/COMMIT; } > "$ROOT/build_check.txt"
for t in "$REPO/tests/test_scan.cpp" "$REPO/tests/test_worstcase.cpp" $IN/abcount.cpp $IN/scanbench.cpp $IN/chain.cpp; do
  "$REPO/build/tests/cxx-O3" "$t" -o "$ROOT/$(basename "$t" .cpp)" 2>&1 | grep -E "error:" | head -5
done
taskset -c 2 "$ROOT/test_scan" 2>/dev/null | tail -1 | tee "$ROOT/tests.txt" | cut -c1-110
taskset -c 2 "$ROOT/test_worstcase" 2000000 2>/dev/null | tee -a "$ROOT/tests.txt" | tail -1 | cut -c1-110
AR_LEAF=4 AR_TAU=1.01 AB_V=1 taskset -c 2 "$REPO/build/tests/test_splice_edge-O3" 2>/dev/null | tail -1 | cut -c1-110
[ "$(grep -c ': ok' "$ROOT/tests.txt")" = 2 ] || { echo "== tests failed, no timing"; touch "$ROOT/DONE"; exit 1; }
RF=$G/rules/genome.rule

run_arm() {  # lane root idx ds budget leaf tau v passes
  local c=$1 root=$2 idx=$3 ds=$4 b=$5 leaf=$6 tau=$7 v=$8 passes=$9 rf="" spk bud=0 p="1"
  [ "$idx" = gr4 ] && rf=$RF
  case $idx in ab4|gr4) spk=1.25; bud=$b;; *) spk=$(python3 -c "print(($b-0.005)/16)");; esac
  [ "$b" = 0 ] && { bud=0; spk=1.25; }
  [ "$passes" -ge 3 ] && p="1 2 3"; [ "$passes" = 2 ] && p="1 2"
  local out=$root/${idx}__b18_s${b}_t${tau}_l${leaf}_c${v}__${ds}
  mkdir -p "$out"; echo "$ds $idx" > "$out/cells.txt"
  RULE_FILE=$rf RADIX_BITS=18 HASHLINE_SPK=$spk AB_BUDGET=$bud AB_V=$v AR_TAU=$tau AR_LEAF=$leaf CORE=$c J=1 HEAVY="^$" HASH_DATA=0 PASSES="$p" /scratch/lvasseur/bin/nothp "$REPO/scripts/run_parallel.sh" "$out" "$out/cells.txt" > /dev/null 2>&1
  echo "$(date +%T) lane $c done $idx $ds $b $leaf $tau $v" >> "$root/progress.txt"
}
lanes() {
  local f=$1 root=$2 j=0
  awk -v m=4 -v o="$root/lane" '{print > (o (NR-1)%m)}' "$f"
  for c in 2 20 38 56; do
    ( [ -f "$root/lane$j" ] && while read -r idx ds b leaf tau v passes; do run_arm $c "$root" $idx $ds $b $leaf $tau $v $passes; done < "$root/lane$j" ) &
    j=$((j+1))
  done
  wait
}
ALL="fb genome osm planet books covid libio history stack wise"
# 1. the rewritten walk at the settings of every earlier run
: > "$ROOT/reg/cells"
for d in $ALL; do
  case $d in fb|genome|osm|planet) l=8; t=2;; *) l=64; t=3;; esac
  echo "ab4 $d 0 $l $t 0 3" >> "$ROOT/reg/cells"
done
lanes "$ROOT/reg/cells" "$ROOT/reg"
python3 /scratch/lvasseur/arm_agg2.py "$ROOT/reg" | tail -1
echo "== $(date +%T) WALK_REGRESSION_DONE"
# 2. exact counts on this code, nothing timed
python3 - "$ROOT" <<'PY'
import sys
root=sys.argv[1]; c=[]
for ds in "fb genome osm planet books covid libio history stack wise".split():
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
# 3. every undominated cell, one pass, and an audit of the dominated ones
python3 $IN/final_pick.py "$ROOT"
lanes "$ROOT/cells.txt" "$ROOT/fin"
python3 /scratch/lvasseur/arm_agg2.py "$ROOT/fin" | tail -1
echo "== $(date +%T) GRID_DONE"
# 4. two more passes on every order-keeping cell within 2% of the fastest, at most three, and likewise any-order, at most two, per dataset and budget
python3 - "$ROOT" <<'PY'
import csv,sys,collections
root=sys.argv[1]; t=collections.defaultdict(list)
for r in csv.DictReader(open(root+"/fin/table.tsv"),delimiter="\t"):
    if r["Mops"]: t[(r["dataset"],r["spk"])].append(r)
out=[]
for k,rs in t.items():
    o=sorted([r for r in rs if r["index"] in("ab4","gr4")],key=lambda r:-float(r["Mops"]))
    h=sorted([r for r in rs if r["index"] in("hf4","ph4")],key=lambda r:-float(r["Mops"]))
    o=[r for r in o if float(r["Mops"])>=0.98*float(o[0]["Mops"])][:3] if o else []
    h=[r for r in h if float(r["Mops"])>=0.98*float(h[0]["Mops"])][:2] if h else []
    for r in o+h: out.append((r["index"],r["dataset"],r["spk"],r["leaf"],r["tau"],r["cap"],2))
open(root+"/rep/cells","w").write("".join(" ".join(map(str,x))+"\n" for x in out))
print(len(out),"repeat cells")
PY
lanes "$ROOT/rep/cells" "$ROOT/rep"
python3 /scratch/lvasseur/arm_agg2.py "$ROOT/rep" | tail -1
echo "== $(date +%T) REPEATS_DONE"
# 5. scans and chained latency on each dataset's fastest order-keeping cell at 20 bytes per key, fitted and not
python3 - "$ROOT" <<'PY'
import csv,sys,collections,statistics as st
root=sys.argv[1]; t=collections.defaultdict(list)
import os
for p in ("/fin/table.tsv","/rep/table.tsv","/../jed200M_g_2026-11-03/real/table.tsv"):  # the rule's cells at leaves 8 and 16 were timed in G
    if not os.path.exists(root+p): continue
    for r in csv.DictReader(open(root+p),delimiter="\t"):
        if r["Mops"]: t[(r["dataset"],r["spk"],r["index"],r["leaf"],r["tau"],r["cap"])].append(float(r["Mops"]))
win={}
for k,v in t.items():
    if k[1]!="20" or k[2] not in("ab4","gr4"): continue
    m=st.median(v)
    key=k[0]
    if key not in win or m>win[key][0]: win[key]=(m,k)
with open(root+"/winners.txt","w") as f:
    for ds,(m,k) in win.items():
        f.write(f"{k[2]} {k[0]} {k[3]} {k[4]} {k[5]}\n")
        if k[5]=="1": f.write(f"{k[2]} {k[0]} {k[3]} {k[4]} 0\n")
print(len(win),"winners")
PY
awk -v m=4 -v o="$ROOT/scan/lane" '{print > (o (NR-1)%m)}' "$ROOT/winners.txt"
j=0; for c in 2 20 38 56; do
  ( [ -f "$ROOT/scan/lane$j" ] && while read -r idx ds leaf tau v; do [ "$idx" = gr4 ] && rf=$RF || rf=""
      RULE_FILE=$rf RADIX_BITS=18 AB_BUDGET=20 AB_V=$v AR_LEAF=$leaf AR_TAU=$tau taskset -c $c /scratch/lvasseur/bin/nothp "$ROOT/scanbench" "$DATA_DIR/$ds" "$idx" "$ds.v$v" >> "$ROOT/scan/out_$c.tsv" 2>> "$ROOT/scan/err.txt"
    done < "$ROOT/scan/lane$j" ) &
  j=$((j+1)); done; wait
cat "$ROOT"/scan/out_*.tsv > "$ROOT/scan/out.tsv"
echo "== $(date +%T) SCANS_DONE"
j=0; for c in 2 20 38 56; do
  ( [ -f "$ROOT/scan/lane$j" ] && while read -r idx ds leaf tau v; do [ "$idx" = gr4 ] && rf=$RF || rf=""
      RULE_FILE=$rf RADIX_BITS=18 AB_BUDGET=20 AB_V=$v HASHLINE_SPK=1.25 AR_LEAF=$leaf AR_TAU=$tau taskset -c $c /scratch/lvasseur/bin/nothp "$ROOT/chain" "$DATA_DIR/$ds" "$idx" "$ds.v$v" >> "$ROOT/chain/out_$c.tsv" 2>> "$ROOT/chain/err.txt"
    done < "$ROOT/scan/lane$j" ) &
  j=$((j+1)); done; wait
cat "$ROOT"/chain/out_*.tsv > "$ROOT/chain/out.tsv"
echo "== $(date +%T) FINAL_BATCH_DONE"; touch "$ROOT/DONE"
