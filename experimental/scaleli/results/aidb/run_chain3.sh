#!/bin/zsh
# Chain 3 (2026-09-21 22:20): feasible full-scale runs first, assembly, granularity remainder, assembly,
# then the planet alpha-4 root runs (195k-round smoothing) last with a 4 h cap each, assembly again.
cd "$(dirname "$0")/../.."
B=../../build-root/experimental/scaleli/scaleli_bench
log(){ echo "$(date +%T) $*"; }
fs_summarize(){
  python3 tools/summarize.py results/aidb_fullscale/sweep/results.jsonl --baseline packed_rank --output results/aidb_fullscale/sweep/summary.csv | tail -1
  python3 results/aidb/root_analysis.py results/aidb_fullscale/sweep/results.jsonl --json results/aidb_fullscale/root_analysis.json > results/aidb_fullscale/root_analysis.txt
}
log "CHAIN3 start"
log "FULLSCALE A start (fb, planet: control, root raw, root flow, root fences a=0.04, vp10 + root fences a=0.04)"
python3 tools/run_suite.py --binary $B --config results/aidb_fullscale/config_a.json --output results/aidb_fullscale/sweep --timeout 3600 --resume 2>&1 | tail -3
log "FULLSCALE B start (fb at root alpha 4)"
python3 tools/run_suite.py --binary $B --config results/aidb_fullscale/config_b.json --output results/aidb_fullscale/sweep --timeout 3600 --resume 2>&1 | tail -3
fs_summarize
log "FULLSCALE A+B DONE ($(grep -c . results/aidb_fullscale/sweep/results.jsonl) rows)"
log "ASSEMBLY 1 start"; results/aidb/assemble.sh 2>&1 | tail -12; log "ASSEMBLY 1 DONE"
log "GRANULARITY resume start"
python3 tools/run_suite.py --binary ../../build-fusion/experimental/scaleli/scaleli_bench --config results/aidb_granularity/config.json --output results/aidb_granularity/sweep --timeout 3600 --resume 2>&1 | tail -2
for r in 4096 16384 32768; do python3 tools/summarize.py results/aidb_granularity/sweep/results.jsonl --baseline packed_rank_r$r --output results/aidb_granularity/summary_r$r.csv | tail -1; done
log "GRANULARITY RESUME DONE ($(grep -c . results/aidb_granularity/sweep/results.jsonl) rows)"
log "ASSEMBLY 2 start"; results/aidb/assemble.sh 2>&1 | tail -12; log "ASSEMBLY 2 DONE"
log "FULLSCALE C start (planet at root alpha 4; 4 h cap per run)"
for c in c1 c2 c3; do
  before=$(grep -c . results/aidb_fullscale/sweep/results.jsonl)
  log "FULLSCALE $c start"
  python3 tools/run_suite.py --binary $B --config results/aidb_fullscale/config_$c.json --output results/aidb_fullscale/sweep --timeout 14400 --resume 2>&1 | tail -2
  after=$(grep -c . results/aidb_fullscale/sweep/results.jsonl)
  if [ "$after" -le "$before" ]; then log "FULLSCALE $c did not finish within 4 h; skipping the remaining planet alpha-4 runs"; break; fi
done
fs_summarize
log "FULLSCALE C DONE ($(grep -c . results/aidb_fullscale/sweep/results.jsonl) rows)"
log "ASSEMBLY 3 start"; results/aidb/assemble.sh 2>&1 | tail -12; log "ASSEMBLY 3 DONE"
log "CHAIN3 DONE"
