#!/bin/bash
#SBATCH --job-name=splice-chain
#SBATCH --partition=standard
#SBATCH --account=ai-team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=504000M
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/lvasseur/logs/splice_chain_%j.log
# Chained (dependent) lookups, which the final batch lost to a name clash between its binary and its folder: each dataset's fastest
# order-keeping cell at 20 bytes per key and its unfitted twin, hf4 and ph4, and the GRE baselines that chain.cpp can build. Four lanes.
source /scratch/lvasseur/jed_env.sh
export REPO=/scratch/lvasseur/GRE-hash
F=$REPO/results/jed200M_final_2026-11-03; G=$REPO/results/jed200M_g_2026-11-03; ROOT=$REPO/results/jed200M_chain_2026-11-03
mkdir -p "$ROOT/out"; cd /scratch/lvasseur
echo "== $(date +%T) on $(hostname)"; cat $REPO/include/gret/*.h | cksum
"$REPO/build/tests/cxx-O3" /scratch/lvasseur/final_incoming/chain.cpp -o "$ROOT/chain_bin" -L/scratch/lvasseur/env/lib -ljemalloc -Wl,-rpath,/scratch/lvasseur/env/lib 2>&1 | grep -E "error:" | head -5 | cut -c1-170
[ -x "$ROOT/chain_bin" ] || { echo "== chain did not build"; exit 1; }
: > "$ROOT/cells"
while read -r idx ds leaf tau v; do echo "$ds $idx AB_BUDGET=20 AB_V=$v AR_LEAF=$leaf AR_TAU=$tau RADIX_BITS=18" >> "$ROOT/cells"; done < "$F/winners.txt"
for ds in fb genome osm planet books covid libio history stack wise; do
  echo "$ds hf4 HASHLINE_SPK=1.2497" >> "$ROOT/cells"; echo "$ds ph4 HASHLINE_SPK=1.2497 AR_LEAF=65536 RADIX_BITS=18" >> "$ROOT/cells"
  for b in lipp alex pgm btree spgm64; do echo "$ds $b NOTHING=0" >> "$ROOT/cells"; done
done
python3 -c "import random,sys;l=open(sys.argv[1]).readlines();random.seed(12);random.shuffle(l);open(sys.argv[1],'w').writelines(l)" "$ROOT/cells"
awk -v m=4 -v o="$ROOT/lane" '{print > (o (NR-1)%m)}' "$ROOT/cells"
j=0; for c in 2 20 38 56; do
  ( while read -r ds idx envs; do
      env $envs taskset -c $c /scratch/lvasseur/bin/nothp "$ROOT/chain_bin" "$DATA_DIR/$ds" "$idx" "$ds" >> "$ROOT/out/out_$c.tsv" 2>> "$ROOT/err.txt" || echo "$(date +%T) FAILED $ds $idx $envs" >> "$ROOT/failed.txt"
    done < "$ROOT/lane$j" ) &
  j=$((j+1)); done; wait
cat "$ROOT"/out/out_*.tsv > "$ROOT/out.tsv"; echo "== $(date +%T) CHAIN_DONE $(wc -l < "$ROOT/out.tsv") rows"; touch "$ROOT/DONE"
