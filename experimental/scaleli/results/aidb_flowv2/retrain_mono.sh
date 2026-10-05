#!/bin/zsh
cd "$(dirname "$0")/../.."  # experimental/scaleli
for d in books fb osm covid genome history libio planet stack wise; do
  python3 tools/train_flow.py data/samples/${d}_2M_uniform_s42 --dtype uint64 \
    --output results/aidb_flowv2/flows_monotone/${d}_mono.txt --sample 16384 --steps 2000 --shifts 512 --lr 0.05 --monotone \
    > results/aidb_flowv2/flows_monotone/${d}_mono.log 2>&1 &
done
wait
echo MONO_DONE
