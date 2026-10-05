#!/usr/bin/env python3
"""Summarize paired runs; cluster-bootstrap intervals are over workload seeds."""
from __future__ import annotations
import argparse, collections, csv, json, math, pathlib, random, statistics

def quantile(xs: list[float], q: float) -> float:
    return sorted(xs)[min(len(xs)-1, max(0, math.ceil(q*len(xs))-1))]

def paired_interval(pairs: list[tuple[int, float]], samples: int=2000) -> tuple[float, float, float]:
    """Geometric speedup, bootstrapping SEED clusters (not individual queries)."""
    groups: dict[int,list[float]] = collections.defaultdict(list)
    for seed, ratio in pairs: groups[seed].append(math.log(ratio))
    logs = [statistics.mean(v) for v in groups.values()]
    estimate = math.exp(statistics.mean(logs))
    if len(logs) < 2: return estimate, float("nan"), float("nan")
    rng = random.Random(42)
    boot = [math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(samples)]
    return estimate, quantile(boot,.025), quantile(boot,.975)

def learn(r: dict, key: str) -> float:
    l=r.get("learnability");return float(l[key]) if isinstance(l,dict) and key in l else 0.0
def choice(r: dict, key: str) -> float:
    l=r.get("learnability");c=l.get("choices") if isinstance(l,dict) else None;return float(c[key]) if isinstance(c,dict) and key in c else 0.0

def summarize(rows: list[dict], baseline: str) -> list[dict]:
    # Reject accidental pairing of different traces, data subsets or operation mixes.
    fingerprints: dict[tuple,set] = collections.defaultdict(set)
    for r in rows: fingerprints[(r["dataset"],r["profile"],r["seed"],r["repeat"])].add(r["trace_fingerprint"])
    if any(len(s)!=1 for s in fingerprints.values()): raise ValueError("unmatched trace fingerprints across variants")
    lookup={(r["dataset"],r["profile"],r["seed"],r["repeat"],r["variant"]):r for r in rows}
    grouped: dict[tuple,list[dict]] = collections.defaultdict(list)
    for r in rows: grouped[r["dataset"],r["profile"],r["variant"]].append(r)
    out=[]
    for (dataset,profile,variant), group in sorted(grouped.items()):
        pairs=[]
        for r in group:
            ref=lookup.get((dataset,profile,r["seed"],r["repeat"],baseline))
            if ref: pairs.append((r["seed"],r["throughput_ops_s"]/ref["throughput_ops_s"]))
        est,lo,hi=paired_interval(pairs) if pairs else (float("nan"),)*3
        med=lambda f:statistics.median(f(r) for r in group)
        out.append({"dataset":dataset,"profile":profile,"variant":variant,"runs":len(group),
            "all_verified":all(r["verified"] for r in group),"throughput_ops_s_median":med(lambda r:r["throughput_ops_s"]),
            "read_hit_p99_ns_median":med(lambda r:r["latency_ns"]["read_hit"]["p99"]),
            "insert_p99_ns_median":med(lambda r:r["latency_ns"]["insert"]["p99"]),
            "scan_p99_ns_median":med(lambda r:r["latency_ns"]["scan"]["p99"]),
            "initial_bytes_per_key":med(lambda r:r["memory_before"]["accounted_bytes"]/r["initial_rows"]),
            "final_bytes_per_key":med(lambda r:r["memory_after"]["accounted_bytes"]/max(1,r["final_rows"])),
            "key_arena_ratio":med(lambda r:r["memory_before"]["key_bytes"]/max(1,8*r["initial_rows"])),
            "rewrite_bytes_per_operation":med(lambda r:r["maintenance"]["bytes_rewritten"]/r["operations"]),
            "decode_work_per_operation":med(lambda r:r["work_counters"]["decoded_keys"]/r["operations"]),
            "fence_probes_per_operation":med(lambda r:r["work_counters"]["fence_probes"]/r["operations"]),
            "root_probes_per_operation":med(lambda r:r["work_counters"]["root_probes"]/r["operations"]),
            "root_model_fraction":med(lambda r:1.0 if (r.get("learnability") or {}).get("root_model") else 0.0),
            "correction_distance_per_operation":med(lambda r:r["work_counters"]["correction_distance"]/r["operations"]),
            "build_ns_per_key":med(lambda r:r["build_ns"]/max(1,r["initial_rows"])),
            "flow_region_fraction":med(lambda r:learn(r,"flow_regions")/max(1,learn(r,"regions"))),
            "virtual_points_per_key":med(lambda r:learn(r,"virtual_points")/max(1,learn(r,"keys"))),
            "rank_sse_ratio":med(lambda r:learn(r,"rank_sse_after")/learn(r,"rank_sse_before") if learn(r,"rank_sse_before")>0 else 1.0),
            "preprocess_ns_per_key":med(lambda r:(learn(r,"smoothing_ns")+learn(r,"transform_ns"))/max(1,learn(r,"keys"))),
            "fusion_both_fraction":med(lambda r:choice(r,"both")/max(1,learn(r,"regions"))),
            "fusion_flow_fraction":med(lambda r:(choice(r,"flow")+choice(r,"both"))/max(1,learn(r,"regions"))),
            "fusion_vp_fraction":med(lambda r:(choice(r,"vp")+choice(r,"both"))/max(1,learn(r,"regions"))),
            "est_probes_none":med(lambda r:learn(r,"cost_none_mean")),"est_probes_selected":med(lambda r:learn(r,"cost_selected_mean")),
            "baseline":baseline,"paired_throughput_speedup":est,"seed_bootstrap_low":lo,"seed_bootstrap_high":hi})
    return out

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("results",type=pathlib.Path);p.add_argument("--baseline",default="raw_rank");p.add_argument("--output",type=pathlib.Path)
    a=p.parse_args();rows=[json.loads(s) for s in a.results.read_text().splitlines() if s.strip()]
    if not rows:p.error("empty results")
    out=summarize(rows,a.baseline);path=a.output or a.results.with_name("summary.csv")
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
    print(path)
if __name__=="__main__":main()
