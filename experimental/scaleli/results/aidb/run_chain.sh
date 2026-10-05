#!/bin/zsh
cd "$(dirname "$0")/../.."
B=../../build-fusion/experimental/scaleli
echo "$(date +%T) WINDOW start"
python3 tools/aidb_pipeline.py --binary-dir $B --sample-mode window --results results/aidb_window --jobs 4 --build-threads 16 --extra-variants fusion 2>&1 | tail -3
echo "$(date +%T) WINDOW DONE"
echo "$(date +%T) UNIFORM resume"
python3 tools/aidb_pipeline.py --binary-dir $B --steps sweep,throughput,scores,report --build-threads 16 --extra-variants fusion 2>&1 | tail -3
echo "$(date +%T) UNIFORM DONE"
echo "$(date +%T) GRANULARITY start"
python3 tools/run_suite.py --binary $B/scaleli_bench --config results/aidb_granularity/config.json --output results/aidb_granularity/sweep --timeout 3600 --resume 2>&1 | tail -2
for r in 4096 16384 32768; do python3 tools/summarize.py results/aidb_granularity/sweep/results.jsonl --baseline packed_rank_r$r --output results/aidb_granularity/summary_r$r.csv; done
echo "$(date +%T) GRANULARITY DONE"
