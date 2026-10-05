#!/usr/bin/env python3
"""Root-fusion ablation analysis from a sweep's results.jsonl (stdlib only).

Per dataset: root probes per lookup for binary / raw model / flow / virtual fences / fusion, the
selector's choice, the candidate cost estimates, and paired throughput vs packed_rank (geometric mean
over query seeds of per-seed ratios on identical traces, with a seed cluster bootstrap).
usage: root_analysis.py results.jsonl [--json out.json]
"""
from __future__ import annotations
import argparse, collections, json, math, random, statistics, sys

VARIANTS = ["packed_rank", "packed_rank_root_raw", "packed_rank_root_flow", "packed_rank_root_vf4", "packed_rank_root_fusion", "packed_rank_root_vf004", "packed_rank_vp10", "packed_rank_vp10_root_fusion", "packed_rank_vp10_root_vf004", "raw_rank", "sorted_vector"]
SHORT = {"packed_rank": "binary root", "packed_rank_root_raw": "root raw", "packed_rank_root_flow": "root flow", "packed_rank_root_vf4": "root fences", "packed_rank_root_fusion": "root fusion", "packed_rank_root_vf004": "root fences a=0.04", "packed_rank_vp10": "region vp", "packed_rank_vp10_root_fusion": "region vp + root fusion", "packed_rank_vp10_root_vf004": "region vp + root fences a=0.04", "raw_rank": "uncompressed", "sorted_vector": "binary search"}

def paired(rows_v, rows_b):
    by_seed_b = {r["seed"]: r for r in rows_b}
    ratios = [(r["seed"], r["throughput_ops_s"] / by_seed_b[r["seed"]]["throughput_ops_s"]) for r in rows_v if r["seed"] in by_seed_b]
    if not ratios: return float("nan"), float("nan"), float("nan"), 0
    logs = [math.log(x) for _, x in ratios]; est = math.exp(statistics.mean(logs))
    if len(logs) < 2: return est, float("nan"), float("nan"), len(logs)
    rng = random.Random(42); boot = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(2000))
    return est, boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1], len(logs)

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("results"); p.add_argument("--json"); a = p.parse_args()
    rows = [json.loads(l) for l in open(a.results) if l.strip()]
    by = collections.defaultdict(list)
    for r in rows: by[(r["dataset"], r["variant"])].append(r)
    datasets = sorted({d for d, _ in by})
    out = {}
    print(f"{'dataset':<16}{'variant':<24}{'n':>2}{'Mops':>7}{'speedup':>8}{'lo':>6}{'hi':>6}{'root/op':>8}{'fence/op':>9}{'total/op':>9}{'chosen':>18}{'virt':>6}{'est bin/raw/flow/vf/fus':>26}{'build s':>8}{'meta B/key':>11}")
    for d in datasets:
        base = by.get((d, "packed_rank"), [])
        out[d] = {}
        for v in VARIANTS:
            rs = by.get((d, v), [])
            if not rs: continue
            med = lambda f: statistics.median(f(r) for r in rs)
            if v == "packed_rank": est, lo, hi, n = 1.0, 1.0, 1.0, len(rs)
            elif base: est, lo, hi, n = paired(rs, base)
            else: est, lo, hi, n = float("nan"), float("nan"), float("nan"), 0
            l = rs[0].get("learnability") or {}; w = lambda r, k: r["work_counters"][k] / r["operations"]
            chosen = "-" if not l.get("root_model") else ("flow+" if l.get("root_flow") else "raw+") + ("fences" if l.get("root_vp") else "ranks")
            if v in ("sorted_vector",): chosen = "n/a"
            elif not l.get("root_model") and v != "packed_rank" and "root" in v: chosen = "binary(fallback)"
            elif v == "packed_rank": chosen = "binary"
            rec = {"runs": len(rs), "mops": med(lambda r: r["throughput_ops_s"]) / 1e6, "speedup": est, "lo": lo, "hi": hi,
                   "root_probes": med(lambda r: w(r, "root_probes")), "fence_probes": med(lambda r: w(r, "fence_probes")),
                   "total_probes": med(lambda r: w(r, "root_probes") + w(r, "fence_probes") + w(r, "coordinate_probes")),
                   "chosen": chosen, "root_virtual": l.get("root_virtual", 0),
                   "est": [l.get("root_probes_binary", 0), l.get("root_probes_raw", 0), l.get("root_probes_flow", 0), l.get("root_probes_vp_raw", 0), l.get("root_probes_vp_flow", 0)],
                   "build_s": med(lambda r: r["build_ns"] / 1e9), "meta_bytes_per_key": med(lambda r: r["memory_before"]["metadata_bytes"] / max(1, r["initial_rows"]))}
            out[d][v] = rec
            e = rec["est"]
            print(f"{d:<16}{SHORT.get(v, v):<24}{rec['runs']:>2}{rec['mops']:7.3f}{rec['speedup']:8.3f}{rec['lo']:6.2f}{rec['hi']:6.2f}{rec['root_probes']:8.2f}{rec['fence_probes']:9.2f}{rec['total_probes']:9.2f}{rec['chosen']:>18}{rec['root_virtual']:>6}{'':>3}{e[0]:4.1f}/{e[1]:4.1f}/{e[2]:4.1f}/{e[3]:4.1f}/{e[4]:4.1f}{rec['build_s']:8.2f}{rec['meta_bytes_per_key']:11.2f}")
        print()
    # Verdicts per dataset on the fusion row's own candidate estimates and on measured probes.
    print("VERDICTS (root level): synergy = fusion candidate beats raw, flow, fences AND binary on the estimate; probes = measured root probes/op")
    verdict = {}
    for d in datasets:
        f = out[d].get("packed_rank_root_fusion"); b = out[d].get("packed_rank")
        if not f or not b: continue
        bin_, raw, flow, vf, fus = f["est"]
        singles = [x for x in (raw, flow, vf) if x > 0]
        synergy = fus > 0 and fus < min(singles) and fus < bin_
        better_than_binary = f["root_probes"] < b["root_probes"] - 1e-9
        verdict[d] = {"synergy_on_estimate": synergy, "measured_root_probes": (b["root_probes"], f["root_probes"]),
                      "total_probes": (b["total_probes"], f["total_probes"]), "speedup": (f["speedup"], f["lo"], f["hi"]), "chosen": f["chosen"]}
        print(f"  {d:<16} synergy={str(synergy):<5} root probes {b['root_probes']:.2f} -> {f['root_probes']:.2f} ({'better' if better_than_binary else 'same/fallback'}), total probes {b['total_probes']:.2f} -> {f['total_probes']:.2f}, throughput x{f['speedup']:.3f} [{f['lo']:.2f}, {f['hi']:.2f}], chosen {f['chosen']}")
    if a.json: json.dump({"per_dataset": out, "verdicts": verdict}, open(a.json, "w"), indent=1)

if __name__ == "__main__": main()
