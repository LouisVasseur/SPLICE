#!/usr/bin/env python3
"""Conformance and coverage of hardness metrics against OUR control variants. Standard library only.

Clean-room implementation of the scoring protocol of Zhang, Tang, Ailamaki, "How Hard
Can Indexing Be? Principled Dataset Hardness Measurement for Learned Indexes"
(AIDB@VLDB 2026), section 3.2, equations (1)-(3). Only the protocol is taken from the
paper; the numbers this tool produces come from this repository's reduced-scale,
single-host control variants and are never a reproduction of the paper's Table 2.

Inputs
  --hardness H.json   {"<dataset>": {"<scope>": {"rmse","max_error","conflict_degree","pla_32","pla_4096"}}}
                      scope is any of full, full_flow, sample, sample_flow, sample_csv, sample_flow_csv.
  --throughput T.json {"<variant>": {"<dataset>": mops_median}}
Output
  --output S.json     {"<scope>": {"metrics": {"<name>": {"dims","coverage","conformance","per_variant",
                       "comparable_pairs","incomparable_pairs",...}}, "datasets": [...], "variants": [...],
                       "skipped": {...}}}  (one block per scope; 25 metrics per block, canonical order)

Protocol (per scope, per metric):
  * Normalization: p_hat_I(S) = p_I(S) / std_I, std over the datasets of the scope. The
    paper does not state the ddof; the corpus is the whole population being scored, so the
    default is the population std (ddof=0). --ddof 1 gives the sample std. A variant whose
    throughput is identical on every scored dataset has std 0, for which eq. (1) is
    undefined: it is SKIPPED (listed under "skipped" with that reason) and never scored,
    so an uninformative variant cannot raise Conf by scoring +1 on every metric.
  * Partial order: S_i is harder than S_j iff h_k(S_i) >= h_k(S_j) for every dimension k
    and > for at least one. Ties on every dimension are incomparable.
  * Importance: w = sigmoid(p_hat(harder) - p_hat(easier)). A comparable pair is violating
    when the harder dataset has strictly higher throughput (gap > 0), conforming otherwise
    (gap <= 0).
  * Conf_I = (R - P) / (R + P) with R = sum of w over conforming pairs, P over violating
    pairs, defined as 0 when R + P == 0. Conf = mean of Conf_I over the scored variants.
  * Cov = (|C| - |U|) / (|C| + |U|) with |C| + |U| = n choose 2.
Datasets scored in a scope are those present in the hardness block of that scope AND in
every scored variant's throughput; a variant with fewer than --min-datasets (3) overlapping
datasets is skipped and listed under "skipped". A throughput entry that is null, a string,
NaN or infinite is never scored: the dataset leaves the corpus of EVERY variant (intersection
rule) and "dropped_datasets" names the variant and the reason, because the corpus reduction
changes the std normalization of the other variants. Hardness rows with a missing or
non-numeric field are dropped the same way.
"""
from __future__ import annotations
import argparse, itertools, json, math, pathlib, statistics, sys

SCALARS = ["RMSE", "ME", "CD", "PLA-32", "PLA-4096"]
FIELDS = {"RMSE": "rmse", "ME": "max_error", "CD": "conflict_degree", "PLA-32": "pla_32", "PLA-4096": "pla_4096"}
SEP = "·"  # middle dot, as in Table 2
ALIASES = {"PLA-32" + SEP + "PLA-4096": "GRE"}
SCOPES = ["full", "full_flow", "sample", "sample_flow", "sample_csv", "sample_flow_csv"]


def metric_names() -> list[str]:
    """The 25 metrics of Table 2: 5 scalars, 10 two-dimensional, 10 three-dimensional compositions."""
    out = list(SCALARS)
    for k in (2, 3):
        out += [SEP.join(c) for c in itertools.combinations(SCALARS, k)]
    return out


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def std(values: list[float], ddof: int = 0) -> float:
    if ddof == 0:
        return statistics.pstdev(values) if len(values) > 0 else 0.0
    return statistics.stdev(values) if len(values) > 1 else 0.0


def normalize(throughput: dict[str, float], ddof: int = 0) -> dict[str, float]:
    """p_hat = p / std over datasets. A zero std yields all zeros for direct callers; score_scope()
    never reaches that case because it skips zero-std variants first (eq. 1 is undefined there)."""
    s = std(list(throughput.values()), ddof)
    if s <= 0.0:
        return {d: 0.0 for d in throughput}
    return {d: v / s for d, v in throughput.items()}


def harder(hi: dict[str, float], hj: dict[str, float], dims: list[str]) -> bool:
    """True iff dataset i is harder than dataset j: >= on every dimension and > on at least one."""
    return all(hi[k] >= hj[k] for k in dims) and any(hi[k] > hj[k] for k in dims)


def classify_pairs(hardness: dict[str, dict[str, float]], datasets: list[str], dims: list[str]):
    """Split unordered pairs into comparable [(harder, easier), ...] and incomparable [(a, b), ...]."""
    comparable, incomparable = [], []
    for a, b in itertools.combinations(datasets, 2):
        if harder(hardness[a], hardness[b], dims):
            comparable.append((a, b))
        elif harder(hardness[b], hardness[a], dims):
            comparable.append((b, a))
        else:
            incomparable.append((a, b))
    return comparable, incomparable


def coverage(n_comparable: int, n_incomparable: int) -> float | None:
    total = n_comparable + n_incomparable
    return (n_comparable - n_incomparable) / total if total else None


def conformance_of_variant(p_hat: dict[str, float], comparable: list[tuple[str, str]]) -> dict:
    """Reward/penalty accumulation of eq. (1) for one variant on the ordered comparable pairs."""
    reward = penalty = 0.0
    conforming = violating = 0
    for hard, easy in comparable:
        gap = p_hat[hard] - p_hat[easy]
        w = sigmoid(gap)
        if gap > 0:  # harder dataset is faster: a violation, weighted more the larger the gap
            penalty += w
            violating += 1
        else:  # conforming; a large negative gap is an easy pair and contributes little
            reward += w
            conforming += 1
    total = reward + penalty
    return {"conformance": (reward - penalty) / total if total > 0 else 0.0, "reward": reward, "penalty": penalty,
            "conforming": conforming, "violating": violating}


def score_metric(hardness: dict[str, dict[str, float]], normalized: dict[str, dict[str, float]], datasets: list[str], dims: list[str]) -> dict:
    comparable, incomparable = classify_pairs(hardness, datasets, dims)
    detail = {v: conformance_of_variant(p, comparable) for v, p in normalized.items()}
    per_variant = {v: d["conformance"] for v, d in detail.items()}
    conf = statistics.fmean(per_variant.values()) if per_variant else None
    return {"dims": list(dims), "coverage": coverage(len(comparable), len(incomparable)), "conformance": conf,
            "per_variant": per_variant, "comparable_pairs": len(comparable), "incomparable_pairs": len(incomparable),
            "per_variant_detail": detail,
            "pairs": {"comparable": [list(p) for p in comparable], "incomparable": [list(p) for p in incomparable]}}


def scope_hardness(hardness_json: dict, scope: str) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Per-dataset metric vectors for one scope; datasets with a missing or non-numeric field are dropped."""
    table, dropped = {}, {}
    for dataset, scopes in hardness_json.items():
        block = scopes.get(scope) if isinstance(scopes, dict) else None
        if not isinstance(block, dict):
            continue
        row = {}
        for name, field in FIELDS.items():
            v = block.get(field)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                dropped[dataset] = f"missing or non-numeric {field}"
                break
            row[name] = float(v)
        else:
            table[dataset] = row
    return table, dropped


def score_scope(hardness_json: dict, throughput_json: dict, scope: str, ddof: int = 0, min_datasets: int = 3) -> dict | None:
    table, dropped = scope_hardness(hardness_json, scope)
    if not table:
        return None
    skipped: dict[str, str] = {}
    usable: dict[str, dict[str, float]] = {}
    unusable: dict[str, dict[str, str]] = {}  # variant -> {dataset: why its throughput value cannot be scored}
    for variant, per_dataset in throughput_json.items():
        if not isinstance(per_dataset, dict):
            skipped[variant] = "throughput entry is not a dataset map"
            continue
        overlap, bad = {}, {}
        for d, v in per_dataset.items():
            if d not in table:
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                bad[d] = "null throughput" if v is None else f"non-numeric throughput ({type(v).__name__})"
            elif not math.isfinite(float(v)):
                bad[d] = "non-finite throughput"
            else:
                overlap[d] = float(v)
        if len(overlap) < min_datasets:
            skipped[variant] = (f"skipped: only {len(overlap)} numeric dataset(s) overlap the {scope} hardness scope (need {min_datasets})"
                                + (f"; {len(bad)} non-numeric or non-finite throughput value(s): " + ", ".join(f"{d} ({why})" for d, why in sorted(bad.items())) if bad else ""))
            continue
        usable[variant] = overlap
        unusable[variant] = bad
    corpus = lambda: sorted(set(table).intersection(*[set(v) for v in usable.values()])) if usable else sorted(table)
    datasets = corpus()
    for variant in list(usable):  # eq. (1) divides by the std over the scored datasets; zero std is undefined, never "conforming"
        if std([usable[variant][d] for d in datasets], ddof) <= 0.0:
            skipped[variant] = f"skipped: zero throughput std over the {len(datasets)} scored dataset(s) of scope {scope} (eq. 1 undefined); not scored"
            del usable[variant]
            del unusable[variant]
    datasets = corpus()
    for d in sorted(table):
        if d in datasets or d in dropped:
            continue
        reasons = [f"{why} for variant {v}" for v, bad in unusable.items() for dd, why in bad.items() if dd == d]
        reasons += [f"no throughput entry for variant {v}" for v in usable if d not in usable[v] and d not in unusable.get(v, {})]
        if reasons:
            dropped[d] = "; ".join(reasons) + f" (removed from every variant's corpus in scope {scope})"
    if usable and len(datasets) < min_datasets:
        for variant in list(usable):
            skipped[variant] = f"skipped: only {len(datasets)} dataset(s) shared by every variant in scope {scope} (need {min_datasets})"
        usable = {}
        datasets = sorted(table)
    normalized = {v: normalize({d: p[d] for d in datasets}, ddof) for v, p in usable.items()}
    hardness = {d: table[d] for d in datasets}
    metrics = {}
    for name in metric_names():
        m = score_metric(hardness, normalized, datasets, name.split(SEP))
        if name in ALIASES:
            m["alias"] = ALIASES[name]
        metrics[name] = m
    return {"metrics": metrics, "datasets": datasets, "variants": sorted(usable), "skipped": skipped,
            "dropped_datasets": dropped, "ddof": ddof, "normalized_throughput": normalized,
            "hardness": hardness,
            "note": "Reduced-scale, single-host, clean-room control variants scored with the AIDB 2026 protocol; not a reproduction of the paper or of NFL/CSV."}


def score_all(hardness_json: dict, throughput_json: dict, ddof: int = 0, min_datasets: int = 3, scopes: list[str] | None = None) -> dict:
    present = []
    for scopes_of in hardness_json.values():
        if isinstance(scopes_of, dict):
            present += [s for s in scopes_of if s not in present]
    ordered = [s for s in SCOPES if s in present] + sorted(s for s in present if s not in SCOPES)
    if scopes:
        ordered = [s for s in ordered if s in scopes]
    out = {}
    for scope in ordered:
        block = score_scope(hardness_json, throughput_json, scope, ddof, min_datasets)
        if block is not None:
            out[scope] = block
    return out


def fmt(x: float | None, width: int = 8) -> str:
    return f"{x:{width}.2f}" if isinstance(x, (int, float)) else f"{'n/a':>{width}}"


def render_table(scores: dict) -> str:
    """Table-2-like text: metric, per-variant Conf_I, Conf, Cov, one block per scope."""
    lines = []
    for scope, block in scores.items():
        variants = block["variants"]
        lines.append(f"scope {scope}: datasets={len(block['datasets'])} [{', '.join(block['datasets'])}] variants={len(variants)} ddof={block['ddof']}")
        for v, why in block["skipped"].items():
            lines.append(f"  {v}: {why}")
        for d, why in block.get("dropped_datasets", {}).items():
            lines.append(f"  dataset {d} dropped: {why}")
        w = max(24, max((len(n) for n in block["metrics"]), default=24) + 2)
        vw = {v: max(7, len(v) + 1) for v in variants}  # one width per variant column
        head = f"{'Metric':<{w}}" + "".join(f"{v:>{vw[v]}}" for v in variants) + f"{'Conf':>9}{'Cov':>9}"
        lines.append(head)
        lines.append("-" * len(head))
        groups = [("Scalar metrics", 1), ("Two-dimensional metrics", 2), ("Three-dimensional metrics", 3)]
        for title, k in groups:
            lines.append(title)
            for name, m in block["metrics"].items():
                if len(m["dims"]) != k:
                    continue
                label = name + (f" ({m['alias']})" if m.get("alias") else "")
                row = f"{label:<{w}}" + "".join(fmt(m["per_variant"].get(v), vw[v]) for v in variants)
                lines.append(row + fmt(m["conformance"], 9) + fmt(m["coverage"], 9))
        lines.append("")
    lines.append("Reduced-scale, single-host, clean-room controls scored with the AIDB 2026 protocol (sec. 3.2); not a reproduction of Table 2.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hardness", type=pathlib.Path, required=True, help="H.json: dataset -> scope -> five scalar metrics")
    p.add_argument("--throughput", type=pathlib.Path, required=True, help="T.json: variant -> dataset -> MOPS median")
    p.add_argument("--output", type=pathlib.Path, required=True, help="S.json destination")
    p.add_argument("--ddof", type=int, choices=[0, 1], default=0, help="std normalization: 0 population (default), 1 sample")
    p.add_argument("--min-datasets", type=int, default=3, help="variants overlapping fewer datasets are skipped")
    p.add_argument("--scope", action="append", help="restrict to these scopes (repeatable)")
    p.add_argument("--print", action="store_true", help="render a Table-2-like text table to stdout")
    a = p.parse_args(argv)
    hardness_json = json.loads(a.hardness.read_text())
    throughput_json = json.loads(a.throughput.read_text())
    if not isinstance(hardness_json, dict) or not isinstance(throughput_json, dict):
        p.error("hardness and throughput files must hold JSON objects")
    scores = score_all(hardness_json, throughput_json, a.ddof, a.min_datasets, a.scope)
    if not scores:
        p.error("no scope with usable hardness metrics found")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(scores, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.print:
        print(render_table(scores))
    else:
        print(a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
