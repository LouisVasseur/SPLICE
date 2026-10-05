#!/bin/zsh
# Pilot: three real datasets, uniform vs contiguous-window 2M samples, six variants, one query seed.
cd "$(dirname "$0")/../.."
B=../../build-fusion/experimental/scaleli/scaleli_bench
for mode in uniform window; do
  for n in fb osm planet; do
    d=data/samples/${n}_2M_${mode}_s42
    [ -f $d ] || python3 tools/datasets.py sample data/external/gre/$n $d --n 2000000 --dtype uint64 --mode $mode --seed 42 > /dev/null
    f=results/aidb_pilot/flows/${n}_${mode}_2D2H2L.txt; mkdir -p results/aidb_pilot/flows
    [ -f $f ] || python3 tools/train_flow.py $d --output $f --sample 4096 --steps 200 --monotone > results/aidb_pilot/flows/${n}_${mode}_training.json 2>/dev/null
    echo "$(date +%T) prepared $n $mode: $(python3 -c "import json;r=json.load(open('results/aidb_pilot/flows/${n}_${mode}_training.json'));print('tail',r['tail_conflict_degree_raw'],'->',r['tail_conflict_degree_transformed'])")"
  done
  python3 - "$mode" <<'PY'
import json,sys
mode=sys.argv[1]
cfg={"description":f"pilot {mode}","datasets":[{"name":n,"data":f"data/samples/{n}_2M_{mode}_s42","format":"sosd","dtype":"uint64","flow_weights":f"results/aidb_pilot/flows/{n}_{mode}_2D2H2L.txt"} for n in ("fb","osm","planet")],
 "profiles":["read_only"],
 "variants":[{"name":"sorted_vector","index":"sorted_vector"},{"name":"packed_rank","policy":"min_bytes","routing":"rank"},
  {"name":"packed_rank_flow","policy":"min_bytes","routing":"rank","flow":"$flow","flow-bypass":1},
  {"name":"packed_rank_flow_forced","policy":"min_bytes","routing":"rank","flow":"$flow","flow-bypass":0},
  {"name":"packed_rank_vp10","policy":"min_bytes","routing":"rank","virtual-alpha":0.1},
  {"name":"packed_rank_flow_vp10","policy":"min_bytes","routing":"rank","flow":"$flow","flow-bypass":1,"virtual-alpha":0.1},
  {"name":"packed_rank_fusion_auto","policy":"min_bytes","routing":"rank","flow":"$flow","virtual-alpha":0.1,"fusion":"auto"}],
 "seeds":[11],"repeats":1,
 "common":{"load-ratio":1,"miss":0,"query-distribution":"uniform","ops":1000000,"warmup":200000,"verify":0,"instrument":1,"latency":0}}
json.dump(cfg,open(f"results/aidb_pilot/{mode}.json","w"),indent=1)
PY
  rm -rf results/aidb_pilot/$mode
  python3 tools/run_suite.py --binary $B --config results/aidb_pilot/$mode.json --output results/aidb_pilot/$mode --timeout 1800 2>&1 | tail -2
  python3 tools/summarize.py results/aidb_pilot/$mode/results.jsonl --baseline packed_rank
done
echo "$(date +%T) PILOT DONE"
