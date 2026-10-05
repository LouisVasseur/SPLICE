#!/bin/zsh
cd "$(dirname "$0")/../.."
until grep -q "FULLSCALE DONE" results/aidb_fullscale/run.log; do sleep 30; done
echo "$(date +%T) GRANULARITY resume"
python3 tools/run_suite.py --binary ../../build-fusion/experimental/scaleli/scaleli_bench --config results/aidb_granularity/config.json --output results/aidb_granularity/sweep --timeout 3600 --resume 2>&1 | tail -2
for r in 4096 16384 32768; do python3 tools/summarize.py results/aidb_granularity/sweep/results.jsonl --baseline packed_rank_r$r --output results/aidb_granularity/summary_r$r.csv; done
echo "$(date +%T) GRANULARITY RESUME DONE"
