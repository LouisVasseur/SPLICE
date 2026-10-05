"""Determinism + cap check for arm A: re-run selected cases with MAX_ROUNDS=12
into armA_extend/ and compare the first rounds with armA_runs/ bit-for-bit."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import armA
armA.MAX_ROUNDS = 12
armA.OUT = os.path.join(HERE, "armA_extend")
os.makedirs(armA.OUT, exist_ok=True)
cases = [("genome", "window", 64, "TGV"), ("genome", "window", 64, "TGV", True),
         ("books", "window", 16, "GTV"), ("history", "window", 4, "TGV", True)]
for c in cases:
    r = armA.run(c)
    old = json.load(open(os.path.join(HERE, "armA_runs", "%s_%s_k%d_%s%s.json" % (c[0], c[1], c[2], c[3], "_nostop" if len(c) > 4 else ""))))
    m = min(len(old["rounds"]), len(r["rounds"]), 6)
    same = all(old["rounds"][i]["total_uncharged"] == r["rounds"][i]["total_uncharged"] and old["rounds"][i]["gaps"] == r["rounds"][i]["gaps"] for i in range(m))
    print(c, "identical_first_%d=%s" % (m, same), "rounds_new=%d" % len(r["rounds"]),
          ["%.4f%s" % (x["total_uncharged"], "*" if x["accepted"] else "") for x in r["rounds"]])
