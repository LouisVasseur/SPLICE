#!/bin/zsh
cd "$(dirname "$0")/../.."
echo "$(date +%T) FINAL root-ablation rerun start"
python3 tools/run_suite.py --binary ../../build-final/experimental/scaleli/scaleli_bench --config results/aidb_final/config.json --output results/aidb_final/sweep --timeout 3600 --resume 2>&1 | tail -2
python3 tools/summarize.py results/aidb_final/sweep/results.jsonl --baseline packed_rank
echo "$(date +%T) FINAL DONE"
echo "$(date +%T) FULLSCALE start"
python3 tools/run_suite.py --binary ../../build-root/experimental/scaleli/scaleli_bench --config results/aidb_fullscale/config.json --output results/aidb_fullscale/sweep --timeout 7200 --resume 2>&1 | tail -3
python3 tools/summarize.py results/aidb_fullscale/sweep/results.jsonl --baseline packed_rank
echo "$(date +%T) FULLSCALE DONE"
echo "$(date +%T) GRANULARITY resume"
python3 tools/run_suite.py --binary ../../build-fusion/experimental/scaleli/scaleli_bench --config results/aidb_granularity/config.json --output results/aidb_granularity/sweep --timeout 3600 --resume 2>&1 | tail -2
for r in 4096 16384 32768; do python3 tools/summarize.py results/aidb_granularity/sweep/results.jsonl --baseline packed_rank_r$r --output results/aidb_granularity/summary_r$r.csv; done
echo "$(date +%T) GRANULARITY RESUME DONE"
