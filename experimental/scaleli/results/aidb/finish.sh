#!/bin/zsh
cd "$(dirname "$0")/../.."
until grep -q "GRANULARITY RESUME DONE" results/aidb/run_chain2.log; do sleep 60; done
echo "$(date +%T) ASSEMBLY start"
results/aidb/assemble.sh 2>&1 | tail -12
echo "$(date +%T) ASSEMBLY DONE"
