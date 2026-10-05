import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Compare tools/aidb_scores.py against ref_scores.py on the four required cases."""
import importlib.util, json, math, random, subprocess, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
TOOL = pathlib.Path(REPO + "/experimental/scaleli/tools/aidb_scores.py")
sys.path.insert(0, str(HERE))
import ref_scores as ref

spec = importlib.util.spec_from_file_location("aidb_scores", TOOL)
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)

GRE = ["books", "fb", "osm", "covid", "genome", "history", "libio", "planet", "stack", "wise"]
VARIANTS = ["sorted_vector", "raw_rank", "packed_rank", "packed_rank_flow", "packed_rank_vp10"]
FIELDS = ["rmse", "max_error", "conflict_degree", "pla_32", "pla_4096"]
TOL = 1e-9
failures = []

def check(label, tool_block, ref_block):
    """tool_block = S.json scope block, ref_block = {metric: ref score}"""
    names_tool = list(tool_block["metrics"])
    names_ref = list(ref_block)
    if names_tool != names_ref:
        failures.append(f"{label}: metric name/order mismatch\n tool={names_tool}\n ref ={names_ref}")
    for name in names_ref:
        t = tool_block["metrics"].get(name)
        r = ref_block[name]
        if t is None:
            failures.append(f"{label}/{name}: missing in tool output"); continue
        for key, rk in (("coverage", "coverage"), ("conformance", "conformance"), ("comparable_pairs", "C"), ("incomparable_pairs", "U")):
            tv, rv = t[key], r[rk]
            if tv is None and rv is None: continue
            if tv is None or rv is None or abs(tv - rv) > TOL:
                failures.append(f"{label}/{name}: {key} tool={tv} ref={rv}")
        for v, rv in r["per_variant"].items():
            tv = t["per_variant"].get(v)
            if tv is None or abs(tv - rv) > TOL:
                failures.append(f"{label}/{name}: per_variant[{v}] tool={tv} ref={rv}")
        if set(t["per_variant"]) != set(r["per_variant"]):
            failures.append(f"{label}/{name}: variant sets differ tool={sorted(t['per_variant'])} ref={sorted(r['per_variant'])}")
        if t["dims"] != name.split("·"):
            failures.append(f"{label}/{name}: dims {t['dims']} != name split")

def run_cli(H, T, tag, extra=()):
    hp, tp, sp = HERE / f"{tag}_H.json", HERE / f"{tag}_T.json", HERE / f"{tag}_S.json"
    hp.write_text(json.dumps(H)); tp.write_text(json.dumps(T))
    r = subprocess.run([sys.executable, str(TOOL), "--hardness", str(hp), "--throughput", str(tp), "--output", str(sp), *extra],
                       capture_output=True, text=True)
    return r, sp

# ---------------- (a) random 10 datasets x 5 variants, two scopes, all 25 metrics ----------------
rng = random.Random(20260921)
H = {d: {"full": {f: rng.uniform(1, 1e6) for f in FIELDS}, "sample": {f: rng.uniform(1, 1e6) for f in FIELDS}} for d in GRE}
T = {v: {d: rng.uniform(1, 60) for d in GRE} for v in VARIANTS}
for ddof in (0, 1):
    S = tool.score_all(H, T, ddof=ddof)
    for scope in ("full", "sample"):
        check(f"(a) random ddof={ddof} scope={scope}", S[scope], ref.score_table(H, T, scope, ddof))
        # sanity: every scalar has full coverage (distinct values), each metric has 25 entries
        assert len(S[scope]["metrics"]) == 25
        for sc in ref.SCALARS:
            assert S[scope]["metrics"][sc]["coverage"] == 1.0, (scope, sc)
r, sp = run_cli(H, T, "a_random", ("--print",))
assert r.returncode == 0, r.stderr
S_cli = json.loads(sp.read_text(encoding="utf-8"))
check("(a) random via CLI scope=full", S_cli["full"], ref.score_table(H, T, "full", 0))
check("(a) random via CLI scope=sample", S_cli["sample"], ref.score_table(H, T, "sample", 0))
a_stats = {n: (round(m["coverage"], 3), round(m["conformance"], 3)) for n, m in S_cli["full"]["metrics"].items()}
print("(a) sample of results (cov, conf):", {k: a_stats[k] for k in ["RMSE", "CD", "PLA-32·PLA-4096", "RMSE·CD·PLA-32"]})
print("(a) CLI --print head:\n" + "\n".join(r.stdout.splitlines()[:6]))

# ---------------- (b) ties in one dimension (CD in {1,2,3}); plus two datasets fully tied ----------------
rng = random.Random(7)
Hb = {}
for d in GRE:
    Hb[d] = {"full": {"rmse": rng.uniform(1, 100), "max_error": rng.uniform(1, 1000), "conflict_degree": rng.choice([1, 2, 3]),
                      "pla_32": rng.randrange(10, 10000), "pla_4096": rng.randrange(1, 300)}}
Hb["wise"]["full"] = dict(Hb["books"]["full"])   # fully tied pair -> incomparable under every metric
Tb = {v: {d: rng.uniform(1, 60) for d in GRE} for v in VARIANTS}
S = tool.score_all(Hb, Tb)
check("(b) ties in CD", S["full"], ref.score_table(Hb, Tb, "full"))
cd = S["full"]["metrics"]["CD"]
print(f"(b) CD alone: comparable={cd['comparable_pairs']} incomparable={cd['incomparable_pairs']} cov={cd['coverage']:.4f} conf={cd['conformance']:.4f}")
# the fully tied pair must be incomparable everywhere
for name, m in S["full"]["metrics"].items():
    assert ["books", "wise"] in m["pairs"]["incomparable"] or ["wise", "books"] in m["pairs"]["incomparable"], name
    assert m["coverage"] < 1.0, name
print("(b) fully tied pair (books,wise) incomparable under all 25 metrics: OK")

# ---------------- (c) one variant with identical throughput on all datasets (std = 0) ----------------
Tc = dict(T); Tc["flat"] = {d: 5.0 for d in GRE}
try:
    S = tool.score_all(H, Tc)
except Exception as e:
    failures.append(f"(c) std=0 crashed: {e!r}"); S = None
if S:
    check("(c) std=0 variant", S["full"], ref.score_table(H, Tc, "full"))
    flat = {n: m["per_variant"]["flat"] for n, m in S["full"]["metrics"].items()}
    det = S["full"]["metrics"]["RMSE"]["per_variant_detail"]["flat"]
    print(f"(c) std=0 variant: per_variant Conf_I values = {sorted(set(round(x, 6) for x in flat.values()))}; RMSE detail={det}")
    print(f"(c) normalized_throughput['flat'] = {set(S['full']['normalized_throughput']['flat'].values())}")
    print(f"(c) Conf(RMSE) with flat = {S['full']['metrics']['RMSE']['conformance']:.4f}; without flat = {tool.score_all(H, T)['full']['metrics']['RMSE']['conformance']:.4f}")
    r, sp = run_cli(H, Tc, "c_flat")
    assert r.returncode == 0, r.stderr
    print("(c) CLI exit 0, output valid JSON:", bool(json.loads(sp.read_text(encoding='utf-8'))))

# ---------------- (d) two datasets, harder has higher throughput -> Conf_I < 0 ----------------
Hd = {"easy": {"full": {"rmse": 1, "max_error": 1, "conflict_degree": 1, "pla_32": 10, "pla_4096": 1}},
      "hard": {"full": {"rmse": 2, "max_error": 2, "conflict_degree": 2, "pla_32": 20, "pla_4096": 2}}}
Td = {"ix": {"easy": 1.0, "hard": 3.0}}       # harder dataset is FASTER: violation
S = tool.score_all(Hd, Td, min_datasets=2)
check("(d) two datasets", S["full"], ref.score_table(Hd, Td, "full"))
vals = {n: m["per_variant"].get("ix") for n, m in S["full"]["metrics"].items()}
print("(d) min_datasets=2: Conf_I per metric all -1?", all(v == -1.0 for v in vals.values()), "| coverage all 1?", all(m["coverage"] == 1.0 for m in S["full"]["metrics"].values()))
S_def = tool.score_all(Hd, Td)   # default min_datasets=3
print("(d) default min-datasets=3 on 2 datasets: conformance =", S_def["full"]["metrics"]["RMSE"]["conformance"], "| skipped =", S_def["full"]["skipped"])
r, sp = run_cli(Hd, Td, "d_two")
print("(d) default CLI on 2 datasets: rc", r.returncode, "| RMSE conformance:", json.loads(sp.read_text(encoding='utf-8'))["full"]["metrics"]["RMSE"]["conformance"])
r, sp = run_cli(Hd, Td, "d_two_min2", ("--min-datasets", "2"))
print("(d) CLI --min-datasets 2: rc", r.returncode, "| RMSE per_variant:", json.loads(sp.read_text(encoding='utf-8'))["full"]["metrics"]["RMSE"]["per_variant"])
# same qualitative check with 3 datasets (default min) - hardest is fastest, other pair conforming
Hd3 = dict(Hd); Hd3["mid"] = {"full": {"rmse": 1.5, "max_error": 1.5, "conflict_degree": 1.5, "pla_32": 15, "pla_4096": 1.5}}
Td3 = {"ix": {"easy": 2.0, "mid": 1.0, "hard": 3.0}}
S3 = tool.score_all(Hd3, Td3)
check("(d) three datasets", S3["full"], ref.score_table(Hd3, Td3, "full"))
print("(d) 3 datasets, hardest fastest: RMSE Conf_I =", round(S3["full"]["metrics"]["RMSE"]["per_variant"]["ix"], 4), "(negative expected)")

# ---------------- canonical names vs Table 2 ----------------
table2 = ["RMSE", "ME", "CD", "PLA-32", "PLA-4096",
          "PLA-32·PLA-4096", "RMSE·ME", "RMSE·CD", "RMSE·PLA-32", "RMSE·PLA-4096", "ME·CD", "ME·PLA-32", "ME·PLA-4096", "CD·PLA-32", "CD·PLA-4096",
          "RMSE·ME·CD", "RMSE·ME·PLA-32", "RMSE·ME·PLA-4096", "RMSE·CD·PLA-32", "RMSE·CD·PLA-4096", "RMSE·PLA-32·PLA-4096",
          "ME·CD·PLA-32", "ME·CD·PLA-4096", "ME·PLA-32·PLA-4096", "CD·PLA-32·PLA-4096"]
names = tool.metric_names()
print("canonical: 25 names?", len(names) == 25, "| set equals Table 2?", set(names) == set(table2), "| order equals Table 2 row order?", names == table2)
print("canonical: order equals Table 2 except GRE placed last among 2-dim?", names == table2[:5] + table2[6:15] + [table2[5]] + table2[15:])
print("canonical: separator is U+00B7?", all(("·" in n) == (len(n.split("·")) > 1) for n in names) and tool.SEP == "·")
print("canonical: alias GRE on", [n for n, m in S_cli["full"]["metrics"].items() if m.get("alias")])

print("\n==== RESULT ====")
if failures:
    print(f"{len(failures)} mismatches:"); print("\n".join(failures[:40]))
else:
    print("all comparisons agree with the independent implementation within", TOL)
