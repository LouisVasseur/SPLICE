#!/bin/bash
#SBATCH --job-name=splice-after
#SBATCH --partition=standard
#SBATCH --account=ai-team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=504000M
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/lvasseur/logs/splice_after_%j.log
# Runs when the main job has ended, however it ended, so nothing is timed at the same time.
echo "== $(date +%T) on $(hostname), job $SLURM_JOB_ID"
bash /scratch/lvasseur/after_batch.sh > /scratch/lvasseur/logs/after_batch_sb.log 2>&1
echo "== $(date +%T) ALL_DONE"
