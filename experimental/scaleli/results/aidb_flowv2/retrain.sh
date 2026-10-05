#!/bin/zsh
cd "$(dirname "$0")/../.."  # experimental/scaleli
for d in books fb osm covid genome history libio planet stack wise; do
  for cfg in "512 2000" "64 2000"; do
    set -- ${=cfg}; sh=$1; st=$2
    python3 tools/train_flow.py data/samples/${d}_2M_uniform_s42 --dtype uint64 \
      --output results/aidb_flowv2/flows_free/${d}_s${sh}_t${st}.txt --sample 16384 --steps $st --shifts $sh --lr 0.05 \
      > results/aidb_flowv2/flows_free/${d}_s${sh}_t${st}.log 2>&1 &
  done
done
wait
echo RETRAIN_DONE
