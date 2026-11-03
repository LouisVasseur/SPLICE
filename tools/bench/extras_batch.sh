#!/bin/bash
# After the final batch: the deployed gate with weights frozen before the final data, its picks timed, and misses and skew. Weights came from
# unfitted arms of six datasets, so osm, planet, stack, wise and the synthetics are held out from them.
source /scratch/lvasseur/jed_env.sh
export REPO=/scratch/lvasseur/GRE-hash GRE_DIR=/scratch/lvasseur/gre-hash/GRE
F=$REPO/results/jed200M_final_2026-11-03; G=$REPO/results/jed200M_g_2026-11-03; ROOT=$REPO/results/jed200M_extras_2026-11-03; IN=/scratch/lvasseur/final_incoming
until [ -e "$F/DONE" ]; do sleep 20; done
echo "== $(date +%T) final batch done"
mkdir -p "$ROOT/gate" "$ROOT/picks" "$ROOT/miss"; cd /scratch/lvasseur
"$REPO/build/tests/cxx-O3" $IN/gate_select.cpp -o "$ROOT/gate_select" 2>&1 | grep -E " error " | head -5
"$REPO/build/tests/cxx-O3" $IN/misses.cpp -o "$ROOT/misses" 2>&1 | grep -E " error " | head -5
W=$(cat $IN/gate_weights.txt)
RULE_NS=$(head -1 "$G/rulecost.tsv" | sed 's/.*rule_ns=\([0-9.]*\).*/\1/'); [ -n "$RULE_NS" ] || RULE_NS=12
echo "weights: $W, rule_ns: $RULE_NS" | tee "$ROOT/gate/frozen.txt"
ALL="fb genome osm planet books covid libio history stack wise"
: > "$ROOT/gate/cells"
rule_of() { local r="$G/rules/$1.rule"; [ -f "$r" ] && [ "$(head -1 "$r")" != "S 0" ] && echo "$r" || echo NONE; }
for d in $ALL; do for b in 20 23 25; do echo "$d $DATA_DIR/$d $b $(rule_of $d)" >> "$ROOT/gate/cells"; done; done
for f in rows three_blocks triangle; do echo "syn_$f $G/syn/$f 20 $(rule_of syn_$f)" >> "$ROOT/gate/cells"; done
awk -v m=4 -v o="$ROOT/gate/lane" '{print > (o (NR-1)%m)}' "$ROOT/gate/cells"
j=0; for c in 2 20 38 56; do
  ( while read -r ds path b rule; do [ "$rule" = NONE ] && rule=""
      out=$(RADIX_BITS=18 taskset -c $c "$ROOT/gate_select" "$path" "$b" "$W" "$RULE_NS" $rule 2>/dev/null)
      printf '%s\t%s\t%s\t%s\n' "$ds" "$b" "${rule:-NONE}" "$out" >> "$ROOT/gate/choices_$c.tsv"
    done < "$ROOT/gate/lane$j" ) &
  j=$((j+1)); done; wait
cat "$ROOT"/gate/choices_*.tsv > "$ROOT/gate/choices.tsv"
echo "== $(date +%T) GATE_DONE"
run_arm() {  # lane root data idx ds budget leaf tau v rule
  local c=$1 root=$2 data=$3 idx=$4 ds=$5 b=$6 leaf=$7 tau=$8 v=$9 rule=${10:-}
  local out=$root/${idx}__b18_s${b}_t${tau}_l${leaf}_c${v}__${ds}
  mkdir -p "$out"; echo "$ds $idx" > "$out/cells.txt"
  DATA_DIR=$data RULE_FILE=$rule RADIX_BITS=18 HASHLINE_SPK=1.25 AB_BUDGET=$b AB_V=$v AR_TAU=$tau AR_LEAF=$leaf CORE=$c J=1 HEAVY="^$" HASH_DATA=0 PASSES="1" /scratch/lvasseur/bin/nothp "$REPO/scripts/run_parallel.sh" "$out" "$out/cells.txt" > /dev/null 2>&1
}
awk -F'\t' 'NF>=7 && $4!="infeasible"{print $1,$2,$3,$4,$5,$6,$7}' "$ROOT/gate/choices.tsv" > "$ROOT/picks/cells"
awk -v m=4 -v o="$ROOT/picks/lane" '{print > (o (NR-1)%m)}' "$ROOT/picks/cells"
j=0; for c in 2 20 38 56; do
  ( [ -f "$ROOT/picks/lane$j" ] && while read -r ds b rule idx leaf tau v; do
      data=$DATA_DIR; name=$ds
      case $ds in syn_*) data=$G/syn; name=${ds#syn_};; esac
      [ "$rule" = NONE ] && rule=""
      [ "$idx" = ab4 ] && rule=""
      run_arm $c "$ROOT/picks" "$data" $idx $name $b $leaf $tau $v "$rule"
    done < "$ROOT/picks/lane$j" ) &
  j=$((j+1)); done; wait
python3 /scratch/lvasseur/arm_agg2.py "$ROOT/picks" | tail -1
echo "== $(date +%T) PICKS_DONE"
# misses and skew: the fastest order-keeping cell at 20 bytes per key, the hash, the partition, LIPP and ALEX
: > "$ROOT/miss/cells"
for d in fb genome covid stack; do
  grep " $d " "$F/winners.txt" | head -1 | awk '{print $1,$2,$3,$4,$5}' >> "$ROOT/miss/cells"
  echo "hf4 $d 0 0 0" >> "$ROOT/miss/cells"; echo "ph4 $d 65536 0 0" >> "$ROOT/miss/cells"; echo "lipp $d 0 0 0" >> "$ROOT/miss/cells"; echo "alex $d 0 0 0" >> "$ROOT/miss/cells"
done
awk -v m=4 -v o="$ROOT/miss/lane" '{print > (o (NR-1)%m)}' "$ROOT/miss/cells"
j=0; for c in 2 20 38 56; do
  ( while read -r idx ds leaf tau v; do
      [ "$idx" = gr4 ] && rf="$G/rules/genome.rule" || rf=""
      case $idx in ab4|gr4) bud=20; spk=1.25;; *) bud=0; spk=1.25;; esac
      RULE_FILE=$rf RADIX_BITS=18 HASHLINE_SPK=$spk AB_BUDGET=$bud AB_V=$v AR_LEAF=$leaf AR_TAU=$tau taskset -c $c /scratch/lvasseur/bin/nothp "$ROOT/misses" "$DATA_DIR/$ds" "$idx" "$ds" >> "$ROOT/miss/out_$c.tsv" 2>> "$ROOT/miss/err.txt"
    done < "$ROOT/miss/lane$j" ) &
  j=$((j+1)); done; wait
cat "$ROOT"/miss/out_*.tsv > "$ROOT/miss/out.tsv"
echo "== $(date +%T) MISSES_DONE"
python3 $IN/replay.py "$F" "$G" > "$ROOT/replay.txt" 2>&1
python3 $IN/pareto.py genome "$F" "$G" > "$ROOT/pareto_genome.txt" 2>&1
echo "== $(date +%T) EXTRAS_DONE"; touch "$ROOT/DONE"
