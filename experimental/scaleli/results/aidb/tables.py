#!/usr/bin/env python3
"""Compact text tables from summarize.py CSVs for the meeting notes. usage: tables.py summary.csv [--baseline-suffix]"""
import csv, sys, collections

def table(path, variants=None, group_by_suffix=False):
    rows = list(csv.DictReader(open(path)))
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan")
    print(f"== {path}")
    print(f"{'dataset':<16}{'variant':<30}{'n':>2}{'Mops':>7}{'speedup':>8}{'lo':>6}{'hi':>6}{'fence/op':>9}{'corr/op':>8}{'root/op':>8}{'sse':>6}{'vp/key':>7}{'flow%':>6}{'auto fl/vp':>11}{'pre ns/k':>9}{'B/key':>7}")
    seen = collections.OrderedDict()
    for r in rows: seen.setdefault(r["dataset"], []).append(r)
    for d, rs in seen.items():
        rs = sorted(rs, key=lambda r: (variants.index(r["variant"]) if variants and r["variant"] in variants else 99, r["variant"]))
        for r in rs:
            if variants and r["variant"] not in variants: continue
            print(f"{d:<16}{r['variant']:<30}{int(float(r['runs'])):>2}{f(r,'throughput_ops_s_median')/1e6:7.3f}{f(r,'paired_throughput_speedup'):8.3f}{f(r,'seed_bootstrap_low'):6.2f}{f(r,'seed_bootstrap_high'):6.2f}"
                  f"{f(r,'fence_probes_per_operation'):9.2f}{f(r,'correction_distance_per_operation'):8.2f}{f(r,'root_probes_per_operation') if 'root_probes_per_operation' in r else float('nan'):8.2f}{f(r,'rank_sse_ratio'):6.2f}{f(r,'virtual_points_per_key'):7.3f}{100*f(r,'flow_region_fraction'):6.0f}"
                  f"{'':>2}{100*f(r,'fusion_flow_fraction'):4.0f}/{100*f(r,'fusion_vp_fraction'):3.0f}{f(r,'preprocess_ns_per_key'):9.0f}{f(r,'initial_bytes_per_key'):7.1f}")
        print()

if __name__ == "__main__":
    for p in sys.argv[1:]: table(p)
