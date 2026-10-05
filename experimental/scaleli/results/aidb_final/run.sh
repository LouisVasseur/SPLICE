#!/bin/zsh
cd "$(dirname "$0")/../.."
until grep -q "GRANULARITY DONE" results/aidb/run_chain.log; do sleep 30; done
echo "$(date +%T) FINAL start"
python3 tools/run_suite.py --binary ../../build-final/experimental/scaleli/scaleli_bench --config results/aidb_final/config.json --output results/aidb_final/sweep --timeout 3600 --resume 2>&1 | tail -2
python3 tools/summarize.py results/aidb_final/sweep/results.jsonl --baseline packed_rank
echo "$(date +%T) FINAL DONE"
