#!/bin/bash
# After the main job, on its own allocation: the comparison with others on point lookups and range scans, measured on the final code and one node,
# then the ablation. One stage at a time, four lanes when timing. Static PGM stands for the read-only family until RadixSpline and PLEX are approved.
source /scratch/lvasseur/jed_env.sh
export REPO=/scratch/lvasseur/GRE-hash GRE_DIR=/scratch/lvasseur/gre-hash/GRE
RES=$REPO/results; F=$RES/jed200M_final_2026-11-03; G=$RES/jed200M_g_2026-11-03; IN=/scratch/lvasseur/after_incoming
B=$RES/jed200M_base_2026-11-03; A=$RES/jed200M_ablation_2026-11-03
mkdir -p "$B"/{point,readonly,readonly_rep,pure,calib,scan} "$A"/{abl,budgets,hash,scan,cnt}; cd /scratch/lvasseur
cp $IN/*.h GRE-hash/include/gret/; cp $IN/test_scan.cpp $IN/test_splice_edge.cpp GRE-hash/tests/
bash /scratch/lvasseur/hash_build.sh 2>&1 | tail -3 | cut -c1-110
{ date; hostname; sha256sum "$GRE_DIR"/build/microbench* 2>/dev/null; cat GRE-hash/include/gret/*.h | cksum; cat $IN/COMMIT; } > "$B/build_check.txt"
grep -qF "include/gret cksum $(cat "$REPO"/include/gret/*.h | cksum))" "$GRE_DIR/build/BUILD_MANIFEST.txt" || { echo "== microbench not built from these headers, no timing"; touch "$B/DONE"; exit 1; }
"$REPO/build/tests/cxx-O3" "$REPO/tests/test_scan.cpp" -o "$B/test_scan" 2>&1 | grep -E "error:" | head -5 | cut -c1-170
"$REPO/build/tests/cxx-O3" $IN/abcount.cpp -o "$A/abcount" 2>&1 | grep -E "error:" | head -5 | cut -c1-170
# the harness allocates as GRE's binary does, since LIPP, ALEX, ART and the B+tree are made of many small nodes
"$REPO/build/tests/cxx-O3" -DWITH_ART $IN/scanbench.cpp -o "$B/scanbench" -L/scratch/lvasseur/env/lib -ljemalloc -Wl,-rpath,/scratch/lvasseur/env/lib 2>&1 | grep -E "error:" | head -5 | cut -c1-170
for x in "$B/test_scan" "$A/abcount" "$B/scanbench"; do [ -x "$x" ] || echo "== missing $x"; done
taskset -c 2 "$B/test_scan" 2>/dev/null | tail -1 | tee "$B/tests.txt" | cut -c1-110
grep -q ": ok" "$B/tests.txt" || { echo "== tests failed, no timing"; touch "$B/DONE"; exit 1; }
ALL="fb genome osm planet books covid libio history stack wise"
HIGH_OPS=100000000

run_arm() {  # lane root idx ds budget leaf tau v passes [code spk E depth noskip ops rule]
  local c=$1 root=$2 idx=$3 ds=$4 b=$5 leaf=$6 tau=$7 v=$8 passes=$9 code=${10:-18} spk=${11:-} E=${12:-64} depth=${13:-3} noskip=${14:-0} ops=${15:-$HIGH_OPS} rule=${16:-NONE}
  local bud=0 p="1"
  [ "$rule" = NONE ] && rule=""
  if [ -z "$spk" ]; then case $idx in ab4|gr4) spk=1.25; bud=$b;; *) spk=$(python3 -c "print(($b-0.005)/16)");; esac; [ "$b" = 0 ] && { bud=0; spk=1.25; }; fi
  [ "$passes" -ge 3 ] && p="1 2 3"; [ "$passes" = 2 ] && p="1 2"
  local out=$root/${idx}__b${code}_s${b}_t${tau}_l${leaf}_c${v}__${ds}
  mkdir -p "$out"; echo "$ds $idx" > "$out/cells.txt"
  OPS=$ops RULE_FILE=$rule RADIX_BITS=18 HASHLINE_SPK=$spk AB_BUDGET=$bud AB_V=$v AR_TAU=$tau AR_LEAF=$leaf TRANSFORM_G_E=$E AR_DEPTH=$depth AB_NOSKIP=$noskip CORE=$c J=1 HEAVY="^$" HASH_DATA=0 PASSES="$p" /scratch/lvasseur/bin/nothp "$REPO/scripts/run_parallel.sh" "$out" "$out/cells.txt" > /dev/null 2>&1
  echo "$(date +%T) lane $c done $idx $ds $b $leaf $tau $v code $code" >> "$root/progress.txt"
}
lanes() {
  local f=$1 root=$2 j=0
  awk -v m=4 -v o="$root/lane" '{print > (o (NR-1)%m)}' "$f"
  for c in 2 20 38 56; do
    ( [ -f "$root/lane$j" ] && while read -r line; do run_arm $c "$root" $line; done < "$root/lane$j" ) &
    j=$((j+1))
  done
  wait
  python3 /scratch/lvasseur/arm_agg2.py "$root" | tail -1
}
scans() {  # file of: label name env=... ; into a folder
  local f=$1 root=$2 j=0
  awk -v m=4 -v o="$root/lane" '{print > (o (NR-1)%m)}' "$f"
  for c in 2 20 38 56; do
    ( [ -f "$root/lane$j" ] && while read -r ds label idx envs; do
        env $envs taskset -c $c /scratch/lvasseur/bin/nothp "$B/scanbench" "$DATA_DIR/$ds" "$idx" "$label" >> "$root/out_$c.tsv" 2>> "$root/err.txt" || echo "$(date +%T) FAILED $label $idx $envs" >> "$root/failed.txt"
      done < "$root/lane$j" ) &
    j=$((j+1))
  done
  wait
  cat "$root"/out_*.tsv > "$root/out.tsv"
}
# 1. the baselines, point lookups, three passes
: > "$B/point/cells"; for d in $ALL; do for i in lipp alex pgm artunsync btree; do echo "$i $d 0 0 0 0 3" >> "$B/point/cells"; done; done
python3 -c "import random,sys;l=open(sys.argv[1]).readlines();random.seed(7);random.shuffle(l);open(sys.argv[1],'w').writelines(l)" "$B/point/cells"
lanes "$B/point/cells" "$B/point"; echo "== $(date +%T) BASE_POINT_DONE"
# 2. the read-only family: every error bound once, the fastest per dataset twice more
: > "$B/readonly/cells"; for d in $ALL; do for i in spgm16 spgm64 spgm256; do echo "$i $d 0 0 0 0 1" >> "$B/readonly/cells"; done; done
lanes "$B/readonly/cells" "$B/readonly"
python3 - "$B" <<'PY'
import csv,sys,collections
root=sys.argv[1]; t=collections.defaultdict(list)
for r in csv.DictReader(open(root+"/readonly/table.tsv"),delimiter="\t"):
    if r["Mops"]: t[r["dataset"]].append(r)
with open(root+"/readonly_rep/cells","w") as o, open(root+"/readonly_best.txt","w") as b:
    for d,rs in t.items():
        w=max(rs,key=lambda r:float(r["Mops"]))
        o.write(f"{w['index']} {d} 0 0 0 0 2\n"); b.write(f"{d} {w['index']}\n")
PY
lanes "$B/readonly_rep/cells" "$B/readonly_rep"; echo "== $(date +%T) READONLY_DONE"
# 3. range scans: every baseline, every static PGM bound, ph4 sorted and not
: > "$B/scan/cells"
for d in $ALL; do
  for i in lipp alex pgm artunsync btree spgm16 spgm64 spgm256; do echo "$d $d $i NOTHING=0" >> "$B/scan/cells"; done
  for lf in 65536 1024; do for s in 1 0; do echo "$d $d.l$lf.s$s ph4 AB_BUDGET=0 HASHLINE_SPK=1.2497 RADIX_BITS=18 AR_LEAF=$lf PH_SCAN_SORTED=$s" >> "$B/scan/cells"; done; done
done
python3 -c "import random,sys;l=open(sys.argv[1]).readlines();random.seed(8);random.shuffle(l);open(sys.argv[1],'w').writelines(l)" "$B/scan/cells"
scans "$B/scan/cells" "$B/scan"; echo "== $(date +%T) BASE_SCANS_DONE"
# 3b. misses, skew and locality on every dataset, for the comparison set; 3c. huge pages on the point numbers
"$REPO/build/tests/cxx-O3" $IN/misses.cpp -o "$B/misses" -L/scratch/lvasseur/env/lib -ljemalloc -Wl,-rpath,/scratch/lvasseur/env/lib 2>&1 | grep -E "error:" | head -5 | cut -c1-170
mkdir -p "$B/miss" "$B/thp"
python3 - "$F" "$B" <<'PY'
import sys,random
F,B=sys.argv[1:3]
win={}
for l in open(F+"/winners.txt"):
    idx,ds,leaf,tau,v=l.split()
    win.setdefault(ds,(idx,leaf,tau,v))
rows=[]
for ds,(idx,leaf,tau,v) in win.items():
    rows.append(f"{ds} {idx} AB_BUDGET=20 AB_V={v} AR_LEAF={leaf} AR_TAU={tau} RADIX_BITS=18")
    rows.append(f"{ds} hf4 HASHLINE_SPK=1.2497")
    rows.append(f"{ds} ph4 HASHLINE_SPK=1.2497 AR_LEAF=65536 RADIX_BITS=18")
    for b in ("lipp","alex","pgm","btree","spgm64"): rows.append(f"{ds} {b} NOTHING=0")
random.seed(10); random.shuffle(rows)
open(B+"/miss/cells","w").write("\n".join(rows)+"\n")
with open(B+"/thp/cells","w") as o:
    for ds,(idx,leaf,tau,v) in win.items():
        o.write(f"{ds} {idx} {leaf} {tau} {v}\n{ds} hf4 0 0 0\n{ds} ph4 65536 0 0\n{ds} lipp 0 0 0\n{ds} alex 0 0 0\n")
PY
awk -v m=4 -v o="$B/miss/lane" '{print > (o (NR-1)%m)}' "$B/miss/cells"
j=0; for c in 2 20 38 56; do
  ( [ -f "$B/miss/lane$j" ] && while read -r ds idx envs; do
      env RULE_FILE=$([ "$idx" = gr4 ] && echo "$G/rules/genome.rule") $envs taskset -c $c /scratch/lvasseur/bin/nothp "$B/misses" "$DATA_DIR/$ds" "$idx" "$ds" >> "$B/miss/out_$c.tsv" 2>> "$B/miss/err.txt" || echo "$(date +%T) FAILED $ds $idx" >> "$B/miss/failed.txt"
    done < "$B/miss/lane$j" ) &
  j=$((j+1)); done; wait
cat "$B"/miss/out_*.tsv > "$B/miss/out.tsv"; echo "== $(date +%T) MISSES_ALL_DONE"
awk -v m=4 -v o="$B/thp/lane" '{print > (o (NR-1)%m)}' "$B/thp/cells"
j=0; for c in 2 20 38 56; do
  ( [ -f "$B/thp/lane$j" ] && while read -r ds idx leaf tau v; do
      out=$B/thp/${idx}__b18_s20_t${tau}_l${leaf}_c${v}__${ds}; mkdir -p "$out"; echo "$ds $idx" > "$out/cells.txt"
      case $idx in ab4|gr4) bud=20;; *) bud=0;; esac
      [ "$idx" = gr4 ] && rf=$G/rules/genome.rule || rf=""
      RULE_FILE=$rf RADIX_BITS=18 HASHLINE_SPK=1.2497 AB_BUDGET=$bud AB_V=$v AR_TAU=$tau AR_LEAF=$leaf CORE=$c J=1 HEAVY="^$" HASH_DATA=0 PASSES="1" "$REPO/scripts/run_parallel.sh" "$out" "$out/cells.txt" > /dev/null 2>&1
    done < "$B/thp/lane$j" ) &
  j=$((j+1)); done; wait
python3 /scratch/lvasseur/arm_agg2.py "$B/thp" | tail -1; echo "== $(date +%T) HUGE_PAGES_DONE"
# 4. what the rule's arithmetic costs by itself: S=0 makes the coordinate the key, so the structure is ab4's and only the transform differs
printf 'S 0\n0 1048576\n' > "$B/pure/one.rule"
python3 -c "
print('S 0')
for k in range(22): print(k*(1<<58), 1048576)" > "$B/pure/many.rule"
: > "$B/pure/cells"
for spec in "fb 8 2" "covid 64 3"; do set -- $spec
  echo "ab4 $1 20 $2 $3 0 3 18" >> "$B/pure/cells"
  echo "gr4 $1 20 $2 $3 0 3 1 '' 64 3 0 $HIGH_OPS $B/pure/one.rule" | sed "s/''/1.25/" >> "$B/pure/cells"
  echo "gr4 $1 20 $2 $3 0 3 22 '' 64 3 0 $HIGH_OPS $B/pure/many.rule" | sed "s/''/1.25/" >> "$B/pure/cells"
done
awk -v m=4 -v o="$B/pure/lane" '{print > (o (NR-1)%m)}' "$B/pure/cells"
j=0; for c in 2 20 38 56; do
  ( [ -f "$B/pure/lane$j" ] && while read -r idx ds b leaf tau v passes code spk E depth noskip ops rule; do
      out=$B/pure/${idx}__b${code}_s${b}_t${tau}_l${leaf}_c${v}__${ds}; mkdir -p "$out"; echo "$ds $idx" > "$out/cells.txt"
      OPS=$HIGH_OPS RULE_FILE=$rule RADIX_BITS=18 HASHLINE_SPK=1.25 AB_BUDGET=$b AB_V=$v AR_TAU=$tau AR_LEAF=$leaf CORE=$c J=1 HEAVY="^$" HASH_DATA=0 PASSES="1 2 3" /scratch/lvasseur/bin/nothp "$REPO/scripts/run_parallel.sh" "$out" "$out/cells.txt" > /dev/null 2>&1
    done < "$B/pure/lane$j" ) &
  j=$((j+1)); done; wait
python3 /scratch/lvasseur/arm_agg2.py "$B/pure" | tail -1; echo "== $(date +%T) RULE_PRICE_DONE"
# 5. the same arms as the main job, on this node, to show whether the two nodes agree
: > "$B/calib/cells"; for spec in "fb 8 2" "covid 64 3"; do set -- $spec; echo "ab4 $1 20 $2 $3 1 1" >> "$B/calib/cells"; echo "hf4 $1 20 0 0 0 1" >> "$B/calib/cells"; done
lanes "$B/calib/cells" "$B/calib"; touch "$B/DONE"; echo "== $(date +%T) BASE_DONE"
# 6. ablation. Every switch on and off at exactly 20 bytes per key, around each dataset's fastest order-keeping leaf size and threshold:
#    P peeling the ends, T none (root only) / adaptive / adaptive with the prefix skip, V the fit, and R the rule on genome.
python3 - "$F" "$A" "$G" <<'PY'
import sys
F,A,G=sys.argv[1:4]
win={}
for l in open(F+"/winners.txt"):
    idx,ds,leaf,tau,v=l.split()
    win.setdefault(ds,(leaf,tau))
rows=[]
for ds,(leaf,tau) in win.items():
    for r in ((0,1) if ds=="genome" else (0,)):
        rule=G+"/rules/genome.rule" if r else "NONE"
        idx="gr4" if r else "ab4"
        for p in (1,0):
            for t in (0,1,2):
                for v in (0,1):
                    code=100*p+10*t+r
                    rows.append((idx,ds,20,leaf,tau,v,3,code,64 if p else 0,3 if t else 0,0 if t==2 else 1,rule))
open(A+"/abl/plan","w").write("".join(" ".join(map(str,x))+"\n" for x in rows))
print(len(rows),"ablation cells")
PY
count_abl() { read -r idx ds b leaf tau v passes code E depth noskip rule <<< "$1"; [ "$rule" = NONE ] && rule=""
  line=$(RULE_FILE=$rule RADIX_BITS=18 AB_BUDGET=$b AB_V=$v AR_TAU=$tau AR_LEAF=$leaf TRANSFORM_G_E=$E AR_DEPTH=$depth AB_NOSKIP=$noskip "$A/abcount" "$DATA_DIR/$ds" "$idx" 2>&1 | grep "^gret: " | tail -1)
  echo "$idx $ds $b $leaf $tau $v $passes $code $E $depth $noskip ${rule:-NONE} $line" >> "$A/cnt/raw_$BASHPID.txt"; }
export -f count_abl; export A DATA_DIR
xargs -a "$A/abl/plan" -P 32 -I{} bash -c 'count_abl "{}"'
python3 - "$A" <<'PY'
import glob,re,sys,csv
A=sys.argv[1]
w=csv.writer(open(A+"/counts.tsv","w"),delimiter="\t")
w.writerow("idx ds budget leaf tau v code E depth noskip lines p999 max levels dir_bytes Bkey build_ms".split())
def g(p,l):
    m=re.search(p,l); return m[1] if m else ""
cells=[]
for f in glob.glob(A+"/cnt/raw_*.txt"):
    for l in open(f):
        a=l.split(None,12)
        idx,ds,b,leaf,tau,v,passes,code,E,depth,noskip,rule=a[:12]
        t=a[12] if len(a)>12 else ""
        L=g(r"E\[buckets\]=([\d.]+)",t)
        w.writerow([idx,ds,b,leaf,tau,v,code,E,depth,noskip,L,g(r"lines_p999=(\d+)",t),g(r"lines_max=(\d+)",t),g(r"E\[levels\]=([\d.]+)",t),g(r"dir_bytes=(\d+)",t),g(r"bytes_per_key=([\d.]+)",t),g(r"build_ms=([\d.]+)",t)])
        if not L: continue
        # a cell reading hundreds of lines keeps its pass to a few minutes: fewer lookups, not fewer cells
        ops=int(min(100000000,max(20000,120/(33e-9*max(float(L),1.0)))))  # about two minutes at about 33 ns a line
        cells.append(f"{idx} {ds} {b} {leaf} {tau} {v} {passes} {code} 1.25 {E} {depth} {noskip} {ops} {rule}\n")
import random
random.seed(9); random.shuffle(cells)
open(A+"/abl/cells","w").writelines(cells)
print(len(cells),"cells to time")
PY
echo "== $(date +%T) ABLATION_COUNTS_DONE"
awk -v m=4 -v o="$A/abl/lane" '{print > (o (NR-1)%m)}' "$A/abl/cells"
j=0; for c in 2 20 38 56; do
  ( [ -f "$A/abl/lane$j" ] && while read -r idx ds b leaf tau v passes code spk E depth noskip ops rule; do
      [ "$rule" = NONE ] && rule=""
      out=$A/abl/${idx}__b${code}_s${b}_t${tau}_l${leaf}_c${v}__${ds}; mkdir -p "$out"; echo "$ds $idx" > "$out/cells.txt"
      OPS=$ops RULE_FILE=$rule RADIX_BITS=18 HASHLINE_SPK=1.25 AB_BUDGET=$b AB_V=$v AR_TAU=$tau AR_LEAF=$leaf TRANSFORM_G_E=$E AR_DEPTH=$depth AB_NOSKIP=$noskip CORE=$c J=1 HEAVY="^$" HASH_DATA=0 PASSES="1 2 3" /scratch/lvasseur/bin/nothp "$REPO/scripts/run_parallel.sh" "$out" "$out/cells.txt" > /dev/null 2>&1
      echo "$(date +%T) lane $c done $idx $ds code $code v $v" >> "$A/abl/progress.txt"
    done < "$A/abl/lane$j" ) &
  j=$((j+1)); done; wait
python3 /scratch/lvasseur/arm_agg2.py "$A/abl" | tail -1; echo "== $(date +%T) ABLATION_POINT_DONE"
# 7. spare slots: below the budgets of the main grid, and with none to spare
: > "$A/budgets/cells"
python3 - "$F" "$A" <<'PY'
import sys
F,A=sys.argv[1:3]
win={}
for l in open(F+"/winners.txt"):
    idx,ds,leaf,tau,v=l.split()
    win.setdefault(ds,(leaf,tau))
with open(A+"/budgets/cells","w") as o:
    for ds,(leaf,tau) in win.items():
        for b in (17,18,19):
            o.write(f"ab4 {ds} {b} {leaf} {tau} 1 1\n")
            if (leaf,tau)!=("64","3"): o.write(f"ab4 {ds} {b} 64 3 1 1\n")
            o.write(f"hf4 {ds} {b} 0 0 0 1\nph4 {ds} {b} 65536 0 0 1\n")
        o.write(f"ab4 {ds} 0 {leaf} {tau} 1 1 900 1.0\n")
PY
lanes "$A/budgets/cells" "$A/budgets"; echo "== $(date +%T) BUDGETS_DONE"
# 8. the mix against one multiply
: > "$A/hash/cells"; for d in $ALL; do echo "hl4 $d 20 0 0 0 3" >> "$A/hash/cells"; done
lanes "$A/hash/cells" "$A/hash"; echo "== $(date +%T) HASH_DONE"
# 9. every switch again for range scans: peeling, directory and fit in every combination, the rule on genome, and spare slots at each budget that fit
python3 - "$F" "$A" "$G" <<'PY'
import csv,random,sys
F,A,G=sys.argv[1:4]
win={}
for l in open(F+"/winners.txt"):
    idx,ds,leaf,tau,v=l.split()
    win.setdefault(ds,(leaf,tau))
fits=set()
try:
    for r in csv.DictReader(open(A+"/budgets/table.tsv"),delimiter="\t"):
        if r["index"]=="ab4" and r["Mops"]: fits.add((r["dataset"],r["spk"],r["leaf"],r["tau"]))
except OSError:
    pass
rows=[]
dirs=((0,1),(3,1),(3,0))
for ds,(leaf,tau) in win.items():
    base=f"RADIX_BITS=18 AR_LEAF={leaf} AR_TAU={tau}"
    for E in (64,0):
        for t,(depth,noskip) in enumerate(dirs):
            for v in (0,1):
                rows.append(f"{ds} {ds}.P{1 if E else 0}.T{t}.V{v} ab4 AB_BUDGET=20 AB_V={v} TRANSFORM_G_E={E} AR_DEPTH={depth} AB_NOSKIP={noskip} {base}")
    if ds=="genome":
        for t,(depth,noskip) in enumerate(dirs):
            for v in (0,1):
                rows.append(f"{ds} {ds}.R1.T{t}.V{v} gr4 RULE_FILE={G}/rules/genome.rule AB_BUDGET=20 AB_V={v} AR_DEPTH={depth} AB_NOSKIP={noskip} {base}")
    for b in (17,18,19):
        for lf,ta in ((leaf,tau),("64","3")):
            if (ds,str(b),lf,ta) in fits:
                rows.append(f"{ds} {ds}.B{b}.l{lf} ab4 AB_BUDGET={b} AB_V=1 RADIX_BITS=18 AR_LEAF={lf} AR_TAU={ta}")
                break
    for b in (23,25):
        rows.append(f"{ds} {ds}.B{b}.l{leaf} ab4 AB_BUDGET={b} AB_V=1 {base}")
    rows.append(f"{ds} {ds}.nospare ab4 AB_BUDGET=0 HASHLINE_SPK=1.0 AB_V=1 {base}")
random.seed(11); random.shuffle(rows)
open(A+"/scan/cells","w").write("\n".join(rows)+"\n")
print(len(rows),"scan ablation cells")
PY
scans "$A/scan/cells" "$A/scan"; echo "== $(date +%T) ABLATION_SCANS_DONE"
python3 $IN/smoke_report.py "$RES" > "$RES/jed200M_smoke_2026-11-03.txt" 2>&1; tail -n 2 "$RES/jed200M_smoke_2026-11-03.txt" | cut -c1-120
touch "$A/DONE"; echo "== $(date +%T) AFTER_BATCH_DONE"
