#!/bin/zsh
cd "$(dirname "$0")/../.."
B=../../build-fusion/experimental/scaleli
echo "$(date +%T) UNIFORM sweep start"
python3 tools/aidb_pipeline.py --binary-dir $B --steps sweep,throughput,scores,report 2>&1 | tail -3
echo "$(date +%T) UNIFORM DONE"
echo "$(date +%T) WINDOW pipeline start"
python3 tools/aidb_pipeline.py --binary-dir $B --sample-mode window --results results/aidb_window --jobs 4 2>&1 | tail -3
echo "$(date +%T) WINDOW DONE"
