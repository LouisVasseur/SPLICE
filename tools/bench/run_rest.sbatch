#!/bin/bash
#SBATCH --job-name=splice-rest
#SBATCH --partition=standard
#SBATCH --account=ai-team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=504000M
#SBATCH --time=07:00:00
#SBATCH --output=/scratch/lvasseur/logs/splice_rest_%j.log
# Everything left after the interactive job's first pass, in turn, on its own node, needing nothing from a laptop: the G batch, the final
# batch on the frozen code, then the gate and the misses. They share a build tree with the first pass, so it has to be over first: it
# ends by its marker, or by the interactive job going away.
L=/scratch/lvasseur/logs; R=/scratch/lvasseur/GRE-hash/results
echo "== $(date +%T) on $(hostname), job $SLURM_JOB_ID"
while squeue -j 40859177 -h -s -o "%i" 2>/dev/null | grep -q "^40859177\.4$"; do
  grep -q BUDGET_DONE "$L/budget_batch.log" && scancel 40859177.4
  sleep 20
done
echo "== $(date +%T) first pass over: $(wc -l < "$R/jed200M_budget_2026-11-03/bud/progress.txt") arms"
touch "$R/jed200M_budget_2026-11-03/DONE"
{ echo "commit $(cat /scratch/lvasseur/final_incoming/COMMIT)"; echo "node $(hostname)"; lscpu | grep -E "Model name|^CPU\(s\)|NUMA node\(s\)"; /usr/bin/g++ --version | head -1; date; } > "$R/jed200M_manifest_2026-11-03.txt"
bash /scratch/lvasseur/g_batch.sh > "$L/g_batch_sb.log" 2>&1
sed -n 1,3p /scratch/lvasseur/GRE-hash/build/tests/cxx-O3 >> "$R/jed200M_manifest_2026-11-03.txt"
bash /scratch/lvasseur/final_batch.sh > "$L/final_batch_sb.log" 2>&1
bash /scratch/lvasseur/extras_batch.sh > "$L/extras_batch_sb.log" 2>&1
echo "== $(date +%T) ALL_DONE"
