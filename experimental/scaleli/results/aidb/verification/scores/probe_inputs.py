import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import json, subprocess, sys, pathlib, math
TOOL = REPO + "/experimental/scaleli/tools/aidb_scores.py"
HERE = pathlib.Path(__file__).resolve().parent
GRE = ["books", "fb", "osm", "covid"]
H = {d: {"full": {"rmse": i+1, "max_error": i+1, "conflict_degree": i+1, "pla_32": i+1, "pla_4096": i+1}} for i, d in enumerate(GRE)}
def run(T, tag, extra=()):
    hp, tp, sp = HERE/f"{tag}_H.json", HERE/f"{tag}_T.json", HERE/f"{tag}_S.json"
    hp.write_text(json.dumps(H)); tp.write_text(tp_text := json.dumps(T))
    r = subprocess.run([sys.executable, TOOL, "--hardness", str(hp), "--throughput", str(tp), "--output", str(sp), *extra], capture_output=True, text=True)
    out = sp.read_text(encoding="utf-8") if sp.exists() else ""
    strict = None
    if out:
        try:
            json.loads(out, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
            strict = "strict-JSON ok"
        except ValueError as e:
            strict = f"NOT strict JSON: contains {e}"
    return r.returncode, r.stderr.strip()[-300:], strict, (json.loads(out) if out else None)
# NaN throughput
rc, err, strict, S = run({"ix": {"books": 4.0, "fb": float("nan"), "osm": 2.0, "covid": 1.0}}, "nan")
print("NaN throughput: rc", rc, "|", strict, "| RMSE conf:", S and S["full"]["metrics"]["RMSE"]["conformance"], "| per_variant:", S and S["full"]["metrics"]["RMSE"]["per_variant"], "| skipped:", S and S["full"]["skipped"], "|", err)
# inf throughput
rc, err, strict, S = run({"ix": {"books": 4.0, "fb": float("inf"), "osm": 2.0, "covid": 1.0}}, "inf")
print("inf throughput: rc", rc, "|", strict, "| RMSE conf:", S and S["full"]["metrics"]["RMSE"]["conformance"], "|", err)
# string throughput
rc, err, strict, S = run({"ix": {"books": "4.0", "fb": "3.0", "osm": "2.0", "covid": "1.0"}}, "str")
print("string throughput: rc", rc, "|", strict, "| skipped:", S and S["full"]["skipped"], "|", err)
# null throughput entries
rc, err, strict, S = run({"ix": {"books": 4.0, "fb": None, "osm": 2.0, "covid": 1.0}}, "null")
print("null throughput: rc", rc, "|", strict, "| datasets:", S and S["full"]["datasets"], "| skipped:", S and S["full"]["skipped"], "|", err)
# unknown scope filter
rc, err, strict, S = run({"ix": {"books": 4.0, "fb": 3.0, "osm": 2.0, "covid": 1.0}}, "scope", ("--scope", "bogus"))
print("--scope bogus: rc", rc, "|", err)
# negative-hardness / zero works, decreasing throughput perfectly conforming
rc, err, strict, S = run({"ix": {"books": 4.0, "fb": 3.0, "osm": 2.0, "covid": 1.0}}, "perfect")
print("perfect ordering: rc", rc, "| all Conf_I == 1:", all(m["per_variant"]["ix"] == 1.0 for m in S["full"]["metrics"].values()))
