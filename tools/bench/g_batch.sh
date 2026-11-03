#!/bin/bash
# Step 4: blind period discovery on x86, then G against identity at exact budgets, with discovery and transform costs. Stages in turn.
source /scratch/lvasseur/jed_env.sh
export REPO=/scratch/lvasseur/GRE-hash GRE_DIR=/scratch/lvasseur/gre-hash/GRE
ROOT=$REPO/results/jed200M_g_2026-11-03; IN=/scratch/lvasseur/g_incoming
until [ -e "$REPO/results/jed200M_budget_2026-11-03/DONE" ]; do sleep 20; done
echo "== $(date +%T) budget batch done"
mkdir -p "$ROOT/rules" "$ROOT/syn" "$ROOT/disc" "$ROOT/real" "$ROOT/synt"; cd /scratch/lvasseur
cp $IN/*.h GRE-hash/include/gret/; cp $IN/test_scan.cpp GRE-hash/tests/
bash /scratch/lvasseur/hash_build.sh 2>&1 | tail -3 | cut -c1-110
"$REPO/build/tests/cxx-O3" "$REPO/tests/test_scan.cpp" -o "$ROOT/test_scan" 2>&1 | grep -E " error " | head -5
"$REPO/build/tests/cxx-O3" $IN/rulecost.cpp -o "$ROOT/rulecost" 2>&1 | grep -E " error " | head -5
"$REPO/build/tests/cxx-O3" $IN/rulecheck.cpp -o "$ROOT/rulecheck" 2>&1 | grep -E " error " | head -5
"$REPO/build/tests/cxx-O3" $IN/abcount.cpp -o "$ROOT/abcount" 2>&1 | grep -E " error " | head -5
/usr/bin/g++ -O3 -march=native -std=c++17 $IN/gdiscover.cpp -o "$ROOT/gdiscover" 2>&1 | grep -E " error " | head -5
/usr/bin/g++ -O3 -march=native -std=c++17 $IN/gsyn.cpp -o "$ROOT/gsyn" 2>&1 | grep -E " error " | head -5
taskset -c 2 "$ROOT/test_scan" 2>/dev/null | tail -1 | tee "$ROOT/tests.txt" | cut -c1-110
grep -q ": ok" "$ROOT/tests.txt" || { echo "== tests failed, no timing"; touch "$ROOT/DONE"; exit 1; }
FAM="rows three_blocks triangle"  # timed at full size; the others are discovered and counted at 20M, untimed
SCREEN="grid2 rows_and_noise soft_wave jitter stride uniform lognormal clusters"
mkdir -p "$ROOT/syn20"
ALL="fb genome osm planet books covid libio history stack wise"
for f in $FAM; do "$ROOT/gsyn" $f 200000000 "$ROOT/syn/$f" & done; wait
for f in $SCREEN; do "$ROOT/gsyn" $f 20000000 "$ROOT/syn20/$f" & done; wait
echo "== $(date +%T) generated"
# discovery, its time included: one pinned lane at a time per dataset, four lanes
: > "$ROOT/disc/cells"; for d in $ALL; do echo "$DATA_DIR/$d $d" >> "$ROOT/disc/cells"; done; for f in $FAM; do echo "$ROOT/syn/$f syn_$f" >> "$ROOT/disc/cells"; done; for f in $SCREEN; do echo "$ROOT/syn20/$f syn_$f" >> "$ROOT/disc/cells"; done
awk -v m=4 -v o="$ROOT/disc/lane" '{print > (o (NR-1)%m)}' "$ROOT/disc/cells"
j=0; for c in 2 20 38 56; do
  ( while read -r path name; do taskset -c $c "$ROOT/gdiscover" "$path" "$ROOT/rules/$name.rule" > "$ROOT/disc/$name.txt" 2>&1; done < "$ROOT/disc/lane$j" ) &
  j=$((j+1)); done; wait
echo "== $(date +%T) discovery done"
: > "$ROOT/screen.cells"
export ROOT
python3 - "$ROOT" $SCREEN <<'PY'
import os,sys
root=sys.argv[1]
with open(root+"/screen.cells","w") as o:
    for f in sys.argv[2:]:
        for v in (0,1):
            o.write(f"{f} ab4 /dev/null {v}\n")
            r=f"{root}/rules/syn_{f}.rule"
            if os.path.exists(r): o.write(f"{f} gr4 {r} {v}\n")
PY
sed -i 's# /dev/null # NONE #' "$ROOT/screen.cells"
screen2() { read -r f idx rule v <<< "$1"; [ "$rule" = NONE ] && rule=""; line=$(RULE_FILE=$rule RADIX_BITS=18 AB_BUDGET=20 AB_V=$v AR_TAU=2 AR_LEAF=16 "$ROOT/abcount" "$ROOT/syn20/$f" "$idx" 2>&1 | grep "^gret: " | tail -1); echo "$f $idx $v $line" >> "$ROOT/screen_$BASHPID.txt"; }
export -f screen2
xargs -a "$ROOT/screen.cells" -P 16 -I{} bash -c 'screen2 "{}"'
cat "$ROOT"/screen_[0-9]*.txt > "$ROOT/screen.txt"
echo "== $(date +%T) screening counted"
for d in $ALL; do [ -f "$ROOT/rules/$d.rule" ] && echo "$d: $(grep -c . "$ROOT/rules/$d.rule") lines, $(head -1 "$ROOT/rules/$d.rule")"; done | cut -c1-70
run_arm() {  # lane root data idx ds budget leaf tau v passes rule
  local c=$1 root=$2 data=$3 idx=$4 ds=$5 b=$6 leaf=$7 tau=$8 v=$9 passes=${10} rule=${11:-} p="1"
  [ "$passes" -ge 3 ] && p="1 2 3"; [ "$passes" = 2 ] && p="1 2"
  local out=$root/${idx}__b18_s${b}_t${tau}_l${leaf}_c${v}__${ds}
  mkdir -p "$out"; echo "$ds $idx" > "$out/cells.txt"
  DATA_DIR=$data RULE_FILE=$rule RADIX_BITS=18 HASHLINE_SPK=1.25 AB_BUDGET=$b AB_V=$v AR_TAU=$tau AR_LEAF=$leaf CORE=$c J=1 HEAVY="^$" HASH_DATA=0 PASSES="$p" /scratch/lvasseur/bin/nothp "$REPO/scripts/run_parallel.sh" "$out" "$out/cells.txt" > /dev/null 2>&1
  echo "$(date +%T) lane $c done $idx $ds $b $leaf $v" >> "$root/progress.txt"
}
lanes() { local f=$1 root=$2; awk -v m=4 -v o="$root/lane" '{print > (o (NR-1)%m)}' "$f"; local j=0
  for c in 2 20 38 56; do ( [ -f "$root/lane$j" ] && while read -r data idx ds b leaf tau v passes rule; do run_arm $c "$root" "$data" $idx $ds $b $leaf $tau $v $passes "$rule"; done < "$root/lane$j" ) & j=$((j+1)); done; wait; }
# genome: the discovered rule at every budget
: > "$ROOT/real/cells"
for b in 20 23 25; do for leaf in 8 16; do for v in 0 1; do
  echo "$DATA_DIR gr4 genome $b $leaf 2 $v 3 $ROOT/rules/genome.rule" >> "$ROOT/real/cells"
done; done; done
lanes "$ROOT/real/cells" "$ROOT/real"
python3 /scratch/lvasseur/arm_agg2.py "$ROOT/real" | tail -1
echo "== $(date +%T) REAL_DONE"
# the hand rule as the upper reference, same code
mkdir -p "$ROOT/hand"; : > "$ROOT/hand/cells"
for b in 20 23 25; do echo "$DATA_DIR gr4 genome $b 16 2 0 3 /scratch/lvasseur/hash_incoming/genome.rule" >> "$ROOT/hand/cells"; done
lanes "$ROOT/hand/cells" "$ROOT/hand"
python3 /scratch/lvasseur/arm_agg2.py "$ROOT/hand" | tail -1
# synthetics: identity against whatever discovery found, or against nothing where it found nothing
: > "$ROOT/synt/cells"
for f in $FAM; do for v in 0 1; do
  echo "$ROOT/syn ab4 $f 20 16 2 $v 2" >> "$ROOT/synt/cells"
  [ -f "$ROOT/rules/syn_$f.rule" ] && echo "$ROOT/syn gr4 $f 20 16 2 $v 2 $ROOT/rules/syn_$f.rule" >> "$ROOT/synt/cells"
done; done
lanes "$ROOT/synt/cells" "$ROOT/synt"
python3 /scratch/lvasseur/arm_agg2.py "$ROOT/synt" | tail -1
echo "== $(date +%T) SYN_DONE"
RULE_FILE=$ROOT/rules/genome.rule taskset -c 2 "$ROOT/rulecost" "$DATA_DIR/genome" genome | tee "$ROOT/rulecost.tsv" | cut -c1-90
RULE_FILE=$ROOT/rules/syn_three_blocks.rule taskset -c 2 "$ROOT/rulecost" "$ROOT/syn/three_blocks" three_blocks | tee -a "$ROOT/rulecost.tsv" | cut -c1-90
RULE_FILE=$ROOT/rules/genome.rule RADIX_BITS=18 AR_LEAF=16 AR_TAU=2 AB_BUDGET=20 taskset -c 2 "$ROOT/rulecheck" "$DATA_DIR/genome" gr4 2>/dev/null | tee "$ROOT/rulecheck.txt" | cut -c1-110
echo "== $(date +%T) G_BATCH_DONE"; touch "$ROOT/DONE"
