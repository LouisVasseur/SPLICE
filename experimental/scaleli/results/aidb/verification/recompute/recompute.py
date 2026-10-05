#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Independent recompute of the root-ablation numbers from results.jsonl only.

Outputs (stdout + JSON):
  * run counts per variant, pairing check (trace_fingerprint identical per (dataset, seed))
  * per dataset x variant: median throughput, probes per lookup (root / fence / coordinate / delta / total),
    transform calls per lookup, chosen root candidate, candidate estimates
  * paired geometric-mean speedup vs packed_rank over seeds with a seed bootstrap interval
  * sorted_vector paired speedup vs every packed variant
"""
import json, math, random, statistics, sys, collections

PATH = REPO + "/experimental/scaleli/results/aidb_final/sweep/results.jsonl"
OUT = REPO + "/build-verify-recompute/scratch/recompute.json"
BASE = "packed_rank"
N_BOOT = 10000
RNG = random.Random(12345)

rows = [json.loads(l) for l in open(PATH) if l.strip()]

# ---------------------------------------------------------------- run counts
count_by_variant = collections.Counter(r["variant"] for r in rows)
count_by_variant_seed = collections.Counter((r["variant"], r["seed"]) for r in rows)
seeds_all = sorted({r["seed"] for r in rows})
datasets = sorted({r["dataset"] for r in rows})
variants = sorted(count_by_variant)

# complete grid = seeds present for every (dataset, variant) of the 9 main variants
main_variants = [v for v in variants if count_by_variant[v] >= len(datasets)]
seed_cov = {s: sum(1 for r in rows if r["seed"] == s and r["variant"] in main_variants) for s in seeds_all}
grid_seeds = sorted(s for s in seeds_all if seed_cov[s] == len(datasets) * len(main_variants))
extra_runs = [(r["dataset"], r["variant"], r["seed"]) for r in rows
              if r["seed"] not in grid_seeds or r["variant"] not in main_variants]

# ---------------------------------------------------------------- pairing check
pairing = {}
pairing_ok = True
for r in rows:
    pairing.setdefault((r["dataset"], r["seed"]), {})[r["variant"]] = (r["trace_fingerprint"], r["operations"])
pairing_report = {}
for (ds, seed), d in sorted(pairing.items()):
    fps = {fp for fp, _ in d.values()}
    ops = {op for _, op in d.values()}
    pairing_report[f"{ds}/{seed}"] = {"n_variants": len(d), "fingerprints": sorted(fps), "operations": sorted(ops)}
    if len(fps) != 1 or len(ops) != 1:
        pairing_ok = False
# checksum agreement (same lookups => same result checksum expected)
checksum_mismatch = []
for (ds, seed), d in pairing.items():
    cks = {r["result_checksum"] for r in rows if r["dataset"] == ds and r["seed"] == seed}
    if len(cks) != 1:
        checksum_mismatch.append((ds, seed, sorted(cks)))

# ---------------------------------------------------------------- per-run derived fields
def chosen_candidate(L):
    if not L.get("root_model"):
        return "binary"
    feat = "flow" if L.get("root_flow") else "raw"
    struct = "fences" if L.get("root_vp") else "ranks"
    return f"{feat}+{struct}"

def per_lookup(r):
    ops = r["operations"]
    wc = r["work_counters"]
    root = wc["root_probes"] / ops
    fence = wc["fence_probes"] / ops
    coord = wc["coordinate_probes"] / ops
    delta = wc.get("delta_probes", 0) / ops
    return {
        "root": root, "fence": fence, "coordinate": coord, "delta": delta,
        "total": root + fence + coord + delta,
        "transform_calls": wc.get("transform_calls", 0) / ops,
        "key_at_calls": wc.get("key_at_calls", 0) / ops,
        "decoded_keys": wc.get("decoded_keys", 0) / ops,
    }

by_dv = collections.defaultdict(dict)   # (dataset, variant) -> seed -> row
for r in rows:
    if r.get("learnability") is None:
        r["learnability"] = {}
    by_dv[(r["dataset"], r["variant"])][r["seed"]] = r

def median(xs):
    return statistics.median(xs) if xs else None

def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))

def boot_ci(ratios, n=N_BOOT):
    """Seed bootstrap: resample the seeds with replacement, geometric mean each time, 2.5/97.5 pct."""
    k = len(ratios)
    if k == 0:
        return None
    if k == 1:
        return [ratios[0], ratios[0]]
    gs = []
    for _ in range(n):
        samp = [ratios[RNG.randrange(k)] for _ in range(k)]
        gs.append(gmean(samp))
    gs.sort()
    lo = gs[int(0.025 * n)]
    hi = gs[min(n - 1, int(math.ceil(0.975 * n)) - 1)]
    return [lo, hi]

def paired_speedup(ds, var, ref=BASE, seeds=None):
    a = by_dv.get((ds, var), {})
    b = by_dv.get((ds, ref), {})
    common = sorted(set(a) & set(b))
    if seeds is not None:
        common = [s for s in common if s in seeds]
    ratios = [a[s]["throughput_ops_s"] / b[s]["throughput_ops_s"] for s in common]
    if not ratios:
        return None
    return {"n_seeds": len(ratios), "seeds": common, "gmean": gmean(ratios),
            "ci95": boot_ci(ratios), "per_seed": dict(zip(common, ratios)),
            "min": min(ratios), "max": max(ratios)}

summary = {}
for (ds, var), sd in sorted(by_dv.items()):
    seeds_here = sorted(sd)
    grid_here = [s for s in seeds_here if s in grid_seeds]
    def agg(field_fn, seeds):
        return median([field_fn(sd[s]) for s in seeds])
    pl = {s: per_lookup(sd[s]) for s in seeds_here}
    # learnability: check constancy across seeds
    L_fields = ["root_model", "root_flow", "root_vp", "root_virtual", "root_probes_binary", "root_probes_raw",
                "root_probes_flow", "root_probes_vp_raw", "root_probes_vp_flow", "regions", "virtual_points",
                "flow_regions", "transform_ns", "keys"]
    L_by_seed = {s: {f: sd[s]["learnability"].get(f) for f in L_fields} for s in seeds_here}
    L_const = all(L_by_seed[s] == L_by_seed[seeds_here[0]] for s in seeds_here)
    L0 = sd[seeds_here[0]]["learnability"]
    entry = {
        "seeds": seeds_here,
        "grid_seeds": grid_here,
        "throughput_by_seed": {s: sd[s]["throughput_ops_s"] for s in seeds_here},
        "median_throughput_all_seeds": agg(lambda r: r["throughput_ops_s"], seeds_here),
        "median_throughput_grid": agg(lambda r: r["throughput_ops_s"], grid_here) if grid_here else None,
        "probes_per_lookup_median_all": {k: median([pl[s][k] for s in seeds_here]) for k in pl[seeds_here[0]]},
        "probes_per_lookup_median_grid": ({k: median([pl[s][k] for s in grid_here]) for k in pl[seeds_here[0]]} if grid_here else None),
        "probes_per_lookup_by_seed": pl,
        "learnability_constant_across_seeds": L_const,
        "chosen_root": chosen_candidate(L0),
        "chosen_root_by_seed": {s: chosen_candidate(sd[s]["learnability"]) for s in seeds_here},
        "root_estimates": {k: L0.get(k) for k in ["root_probes_binary", "root_probes_raw", "root_probes_flow",
                                                   "root_probes_vp_raw", "root_probes_vp_flow", "root_virtual"]},
        "regions": L0.get("regions"),
        "metadata_bytes": sd[seeds_here[0]]["memory_before"]["metadata_bytes"],
        "build_ns_median": agg(lambda r: r["build_ns"], seeds_here),
        "speedup_vs_packed_rank_all": paired_speedup(ds, var),
        "speedup_vs_packed_rank_grid": paired_speedup(ds, var, seeds=set(grid_seeds)),
    }
    summary[f"{ds}|{var}"] = entry

# sorted_vector vs every packed variant, paired per seed
packed_variants = [v for v in variants if v.startswith("packed_rank")]
sv_vs_packed = {}
for ds in datasets:
    for pv in packed_variants:
        sp = paired_speedup(ds, "sorted_vector", ref=pv, seeds=set(grid_seeds))
        if sp:
            sv_vs_packed[f"{ds}|{pv}"] = sp

out = {
    "n_rows": len(rows),
    "runs_per_variant": dict(count_by_variant),
    "runs_per_variant_seed": {f"{v}/{s}": c for (v, s), c in sorted(count_by_variant_seed.items())},
    "seeds_all": seeds_all,
    "grid_seeds": grid_seeds,
    "main_variants": main_variants,
    "extra_runs_outside_grid": extra_runs,
    "pairing_ok": pairing_ok,
    "pairing": pairing_report,
    "result_checksum_mismatches": checksum_mismatch,
    "summary": summary,
    "sorted_vector_vs_packed": sv_vs_packed,
}
json.dump(out, open(OUT, "w"), indent=1, default=str)

# ---------------------------------------------------------------- human-readable report
def f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"

print(f"rows={len(rows)}  seeds={seeds_all}  grid_seeds={grid_seeds}")
print("runs per variant:", dict(sorted(count_by_variant.items())))
print("runs outside the complete grid:", extra_runs)
print(f"pairing_ok={pairing_ok}   checksum mismatches: {len(checksum_mismatch)}")
bad = [k for k, v in pairing_report.items() if len(v["fingerprints"]) != 1]
print("pairs with >1 fingerprint:", bad)
nonconst = [k for k, e in summary.items() if not e["learnability_constant_across_seeds"]]
print("dataset|variant with learnability varying across seeds:", nonconst)
print()
hdr = f"{'dataset':16s} {'variant':30s} {'n':>2s} {'thr_med(M/s)':>12s} {'spd_gm':>7s} {'ci_lo':>6s} {'ci_hi':>6s} {'root':>6s} {'fence':>6s} {'coord':>6s} {'total':>6s} {'xform':>6s}  chosen        est(bin/raw/flow/vpraw/vpflow)"
print(hdr)
for ds in datasets:
    for var in variants:
        e = summary.get(f"{ds}|{var}")
        if not e:
            continue
        sp = e["speedup_vs_packed_rank_grid"]
        p = e["probes_per_lookup_median_grid"] or e["probes_per_lookup_median_all"]
        est = e["root_estimates"]
        print(f"{ds:16s} {var:30s} {len(e['seeds']):2d} {e['median_throughput_grid']/1e6 if e['median_throughput_grid'] else e['median_throughput_all_seeds']/1e6:12.3f} "
              f"{f(sp['gmean'],3) if sp else '-':>7s} {f(sp['ci95'][0],3) if sp else '-':>6s} {f(sp['ci95'][1],3) if sp else '-':>6s} "
              f"{p['root']:6.2f} {p['fence']:6.2f} {p['coordinate']:6.2f} {p['total']:6.2f} {p['transform_calls']:6.3f}  {e['chosen_root']:12s} "
              f"{f(est['root_probes_binary'],1)}/{f(est['root_probes_raw'],1)}/{f(est['root_probes_flow'],1)}/{f(est['root_probes_vp_raw'],1)}/{f(est['root_probes_vp_flow'],1)}")
    print()
print("sorted_vector paired speedup vs each packed variant (grid seeds):")
for ds in datasets:
    line = f"{ds:16s}"
    for pv in packed_variants:
        sp = sv_vs_packed.get(f"{ds}|{pv}")
        line += f" {pv.replace('packed_rank','pr'):>22s}={sp['gmean']:.3f}" if sp else ""
    print(line)
allsv = [sp["gmean"] for sp in sv_vs_packed.values()]
print(f"sorted_vector vs packed variants: min={min(allsv):.3f} max={max(allsv):.3f}")
print(f"\nJSON written to {OUT}")
