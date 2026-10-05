import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import json, subprocess, sys, time
H = REPO + "/build-verify-pla/experimental/scaleli/scaleli_hardness"
R = REPO + "/build-verify-pla/scratch/pgm_ref"
def run_h(path, limit=0, eps="32,4096", cs=1):
    cmd = [H, "--data", path, "--dtype", "uint64", "--pla-eps", eps, "--check-sorted", str(cs)]
    if limit: cmd += ["--limit", str(limit)]
    t0 = time.time(); p = subprocess.run(cmd, capture_output=True, text=True); dt = time.time() - t0
    if p.returncode: return {"error": p.stderr.strip()}, dt
    j = json.loads(p.stdout); return {"keys": j["keys"], "sorted": j["sorted"], "dups": j["duplicates"], "pla": j["original"]["pla"], "elapsed_s": j["elapsed_ns"]/1e9, "cd": j["original"]["conflict_degree"]}, dt
def run_r(path, limit=0, eps="32,4096", shift=0, sort=0):
    cmd = [R, path, "--eps", eps, "--sort", str(sort)]
    if limit: cmd += ["--limit", str(limit)]
    if shift: cmd += ["--shift", str(shift)]
    t0 = time.time(); p = subprocess.run(cmd, capture_output=True, text=True); dt = time.time() - t0
    if p.returncode: return {"error": p.stderr.strip()}, dt
    j = json.loads(p.stdout); return {"keys": j["keys"], "sorted": j["sorted"], "dups": j["duplicates"], "pla": j["pla"]}, dt
cases = [json.loads(a) for a in sys.argv[1:]]
allok = True
for c in cases:
    h, th = run_h(c["path"], c.get("limit", 0), c.get("eps", "32,4096"), c.get("check_sorted", 1))
    r, tr = run_r(c["path"], c.get("limit", 0), c.get("eps", "32,4096"), c.get("shift", 0), c.get("sort", 0))
    ok = ("pla" in h and "pla" in r and h["pla"] == r["pla"] and h["keys"] == r["keys"])
    allok &= ok
    print(f"{'MATCH' if ok else 'MISMATCH'} {c.get('label', c['path'])} limit={c.get('limit',0)} shift={c.get('shift',0)}\n   hardness: {h} ({th:.2f}s wall)\n   pgm_ref : {r} ({tr:.2f}s wall)")
print("ALL MATCH" if allok else "SOME MISMATCH")
