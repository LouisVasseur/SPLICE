#!/usr/bin/env python3
"""Worked numeric examples for 06_protocol.md (standard library only).

Run from experimental/scaleli:  python3 results/aidb/guide_deep/scratch/protocol_examples.py
Every number printed here is quoted in 06_protocol.md; re-run to check them.
"""
import json, math, random, statistics, collections, pathlib

S = pathlib.Path(__file__).resolve().parents[4]   # experimental/scaleli

# 1. Expected root probes of the fence binary search in index.hpp:353 (one probe per loop iteration).
def binary_root_probes(n_keys, region_keys=4096):
    regions = -(-n_keys // region_keys)
    sizes = [region_keys] * (regions - 1) + [n_keys - region_keys * (regions - 1)]
    tot = 0
    for target, sz in enumerate(sizes):
        lo, hi, c = 0, regions, 0
        while lo < hi:
            m = lo + (hi - lo) // 2; c += 1
            if m <= target: lo = m + 1      # fence[m] <= k  <=>  m <= target region
            else: hi = m
        tot += c * sz
    return regions, sizes[-1], tot / n_keys
for n in (2_000_000, 200_000_000):
    print("binary root:", n, "keys ->", binary_root_probes(n))

# 2. Job counts and budgets.
print("jobs E2", 10 * 1 * 10 * 3 * 1, "E3", 20 * 9 * 3, "E4", 5 * 15 * 2, "E5 plan", 2 * 6, "E5 a+b+c", 10 + 3 + 3)
print("budgets: alpha*region", 0.1 * 4096, "root 4*489", 4 * 489, "root 0.04*48829", 0.04 * 48829, "root 4*48829", 4 * 48829)

# 3. Window sample offset (tools/datasets.py sample(), mode window).
rng = random.Random(42); print("window start index (seed 42, 200M total, n=2M):", rng.randrange(200_000_000 - 2_000_000 + 1))

# 4. The paired geometric mean and the 3-seed cluster bootstrap of tools/summarize.py on one cell.
rows = [json.loads(l) for l in open(S / "results/aidb_final/sweep/results.jsonl") if l.strip()]
by = {(r["dataset"], r["variant"], r["seed"]): r for r in rows}
def paired(dataset, variant, baseline="packed_rank", samples=2000):
    logs = [math.log(by[(dataset, variant, s)]["throughput_ops_s"] / by[(dataset, baseline, s)]["throughput_ops_s"]) for s in (11, 29, 47)]
    est = math.exp(statistics.mean(logs))
    rng = random.Random(42)
    boot = [math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(samples)]
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, max(0, math.ceil(p * len(xs)) - 1))]
    return [round(math.exp(x), 4) for x in logs], round(est, 4), round(q(boot, .025), 4), round(q(boot, .975), 4)
print("fb_uniform root_vf4 vs packed_rank:", paired("fb_uniform", "packed_rank_root_vf4"))
print("fb_uniform vp10 vs packed_rank:", paired("fb_uniform", "packed_rank_vp10"))

# 5. Noise floor: pairs whose work counters are byte-identical (identical structure, identical trace).
def same(a, b): return a["work_counters"] == b["work_counters"] and a["learnability"]["flow_regions"] == b["learnability"]["flow_regions"]
fallback = [(d, s, by[(d, "packed_rank_root_raw", s)]["throughput_ops_s"] / by[(d, "packed_rank", s)]["throughput_ops_s"])
            for d in sorted({r["dataset"] for r in rows}) for s in (11, 29, 47) if same(by[(d, "packed_rank", s)], by[(d, "packed_rank_root_raw", s)])]
print("binary-fallback identical pairs:", len(fallback), "ratio range", round(min(x for *_, x in fallback), 3), round(max(x for *_, x in fallback), 3))
spread = {d: max(by[(d, "packed_rank", s)]["throughput_ops_s"] for s in (11, 29, 47)) / min(by[(d, "packed_rank", s)]["throughput_ops_s"] for s in (11, 29, 47)) for d in sorted({r["dataset"] for r in rows})}
print("packed_rank across-seed max/min:", {d: round(x, 3) for d, x in spread.items()})
