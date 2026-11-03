# Benchmarks behind the line-read indexes

`hf4`, `ph4`, `ab4` and `gr4` at 200M keys on one Jed node (Xeon Platinum 8360Y): GRE's microbench, 4 KiB pages through `nothp`, one lane
pinned per NUMA node (cores 2, 20, 38, 56), 100M read-only lookups a pass. `run_rest.sbatch` runs `g_batch.sh`, `final_batch.sh` and
`extras_batch.sh` in turn; each writes its own `results/jed200M_*` folder, and the manifest there names the commit, node and compiler.

Fixed seeds, the same for every index on a dataset: `scanbench` start keys 11, `misses` keys 21, `gsyn` keys 9. The gate's weights
(`gate_weights.txt`) were fit on unfitted arms of six datasets before any of the final runs, which leaves osm, planet, stack, wise and the
synthetic families held out from them.

- `abcount`: one build, exact lines and bytes, no lookups. `final_pick.py`: which cells to time, by dominance, with an audit of the rest.
- `scanbench`, `misses`, `chain`, `rulecost`, `rulecheck`: range scans, misses and skew, chained latency, the rule's cost, its exactness.
- `gdiscover`, `gsyn`: blind period discovery, and synthetic key sets with a known answer.
- `gate_select`: the deterministic selector. `replay.py`, `pareto.py`: selector against oracle, memory against speed.
