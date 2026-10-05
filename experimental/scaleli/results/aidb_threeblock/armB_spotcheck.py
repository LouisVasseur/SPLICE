"""Recompute a few cached k/indep parts with the current armB.py and compare exactly."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armB
from multiprocessing import Pool
CASES = [("k","books","uniform",16),("k","history","window",4),("k","covid","uniform",64),
         ("indep","fb","window"),("indep","stack","uniform"),("k","wise","window",1)]
def strip(r):
    if isinstance(r, dict):
        return {k: strip(v) for k, v in r.items() if k != "seconds"}
    if isinstance(r, list):
        return [strip(x) for x in r]
    return r
def go(t):
    name = "_".join(map(str, t))
    old = json.load(open(os.path.join(armB.PARTS, name + ".json")))
    new = armB.task_k(*t[1:]) if t[0] == "k" else armB.task_indep(*t[1:])
    new = json.loads(json.dumps(new))
    return name, strip(old) == strip(new)
if __name__ == "__main__":
    with Pool(3) as p:
        for name, same in p.imap_unordered(go, CASES):
            print(name, "IDENTICAL" if same else "DIFFERENT", flush=True)
