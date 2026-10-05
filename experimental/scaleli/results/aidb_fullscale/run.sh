#!/bin/zsh
cd "$(dirname "$0")/../.."
until grep -q "FINAL DONE" results/aidb_final/run.log; do sleep 30; done
echo "$(date +%T) FULLSCALE start"
python3 tools/run_suite.py --binary ../../build-root/experimental/scaleli/scaleli_bench --config results/aidb_fullscale/config.json --output results/aidb_fullscale/sweep --timeout 7200 --resume 2>&1 | tail -3
python3 tools/summarize.py results/aidb_fullscale/sweep/results.jsonl --baseline packed_rank
echo "$(date +%T) FULLSCALE DONE"
