import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Instability band per dataset: how far a physically meaningless monotone
perturbation of the feature moves csv_only at the deployed budget."""
import math, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import indep as I
S = REPO + "/experimental/scaleli/data/samples"
ALPHA, EPS = 4.0, 1e-6
def csv_probes(f, feat):
    n = len(f); x = [feat(k) for k in f]
    slot, vf, _b, _a, _r = I.smooth_cdf(x, ALPHA)
    virt = len(vf)
    t = [float(s) for s in slot]
    tab = I.slot_table(slot, n, virt) if virt else None
    m = I.LM().fit_xy(f, x, t)
    return I.root_probes(f, lambda k: m.predict_x(feat(k)), tab)
out = {}
for ds in sys.argv[1:]:
    keys = I.load(os.path.join(S, "%s_2M_uniform_s42" % ds))
    f = I.fences(keys); o = f[0]; sp = float(max(1, f[-1] - o))
    nrm = lambda k: (float(k) - o) / sp
    base = csv_probes(f, nrm)
    vals = {"identity": base}
    for nm, g in (("-x^3", lambda x: x - EPS * x**3), ("+x^3", lambda x: x + EPS * x**3),
                  ("-x^2", lambda x: x - EPS * x * x), ("+x^2", lambda x: x + EPS * x * x)):
        vals[nm] = csv_probes(f, lambda k, g=g: g(nrm(k)))
    band = max(abs(v - base) for v in vals.values())
    out[ds] = {"base": base, "variants": vals, "band": band}
    print(f"{ds:<9} base {base:.4f}  band {band:.4f}  " + "  ".join(f"{k}:{v:.4f}" for k, v in vals.items()), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bands_extra.json"), "w"), indent=1)
