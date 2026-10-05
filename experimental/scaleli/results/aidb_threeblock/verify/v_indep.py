import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""VERIFIER item 2/3: independent re-scoring with verify/indep.py.
Features are recomputed with an independently written gap map and tanh warp;
V slots with indep.smooth_cdf; the probe count with indep.root_probes.
threeblock is used only to (a) replay arm A's cycle so we know the gap set /
factors / warp / guard-kept slots of the best round, and (b) nothing else."""
import json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); TB = os.path.dirname(HERE)
sys.path.insert(0, TB)
sys.path.insert(0, REPO + "/experimental/scaleli/results/aidb_joint/verify")
import indep as I
import threeblock as tb
import armA
S = REPO + "/experimental/scaleli/data/samples"


def my_gmap(xs, idx, fac):
    """independent: u(x) = (x - shrink below x) / (1 - total shrink)."""
    gaps = sorted(zip(idx, fac))
    a = [xs[i] for i, _ in gaps]; b = [xs[i + 1] for i, _ in gaps]; s = [f for _, f in gaps]
    tot = sum((b[j] - a[j]) * (1 - s[j]) for j in range(len(a)))
    def g(x):
        sh = 0.0
        for j in range(len(a)):
            if x >= b[j]:
                sh += (b[j] - a[j]) * (1 - s[j])
            elif x > a[j]:
                sh += (x - a[j]) * (1 - s[j])
            else:
                break
        return (x - sh) / (1 - tot)
    return g


def my_warp(units):
    return lambda u: sum(c * math.tanh(gg * (u - m)) for (gg, m, c) in units)


def my_factors(widths, k):
    srt = sorted(widths); n = len(srt)
    med = srt[n // 2] if n % 2 else 0.5 * (srt[n // 2 - 1] + srt[n // 2])
    order = sorted(range(len(widths)), key=lambda i: (-widths[i], i))[:k]
    sel = sorted(i for i in order if widths[i] > med)
    return sel, [med / widths[i] for i in sel]


def iscore(f, feat, slot=None, virt=0):
    n = len(f); x = [feat(k) for k in f]
    if slot is None or not virt:
        t = [float(j) for j in range(n)]; tab = None
    else:
        t = [float(s) for s in slot]; tab = I.slot_table(slot, n, virt)
    m = I.LM().fit_xy(f, x, t)
    return I.root_probes(f, lambda k: m.predict_x(feat(k)), tab)


def mono(f, feat):
    pk = I.probe_keys(f); v = [feat(k) for k in pk]
    bad = sum(1 for i in range(len(pk) - 1) if pk[i + 1] > pk[i] and not v[i + 1] > v[i])
    return bad


def charge(k_eff):
    return math.ceil(math.log2(k_eff + 1)) if k_eff else 0


def armB_cell(d, m, k, variant, stored):
    keys = I.load(os.path.join(S, "%s_2M_%s_s42" % (d, m))); f = I.fences(keys)
    o = f[0]; sp = float(max(1, f[-1] - o))
    xs = [float(q - o) / sp for q in f]
    raw = lambda q: float(q - o) / sp
    warp = my_warp(stored["warp"]) if stored["warp"] else None
    if variant == "G->V":
        sel, fac = my_factors([xs[i + 1] - xs[i] for i in range(len(xs) - 1)], k)
        g = my_gmap(xs, sel, fac); feat = lambda q: g(raw(q))
    elif variant == "T->G->V":
        z = [warp(x) for x in xs]
        sel, fac = my_factors([z[i + 1] - z[i] for i in range(len(z) - 1)], k)
        g = my_gmap(xs, sel, fac); feat = lambda q: warp(g(raw(q)))
    elif variant == "G->T->V":
        sel, fac = my_factors([xs[i + 1] - xs[i] for i in range(len(xs) - 1)], k)
        g = my_gmap(xs, sel, fac); feat = lambda q: warp(g(raw(q)))
    t0 = time.time()
    slot, vf, _, _, _ = I.smooth_cdf([feat(q) for q in f], 4.0)
    mp = iscore(f, feat, slot, len(vf))
    return {"cell": "B %s-%s k=%d %s" % (d, m, k, variant), "gaps_match": sel == stored["gaps"],
            "virt_indep": len(vf), "virt_stored": stored["virtual_count"],
            "model_indep": mp, "model_stored": stored["model_probes"],
            "charge_indep": charge(len(sel)), "charge_stored": stored["gap_table_probes"],
            "mono_violations": mono(f, feat), "V_seconds": time.time() - t0}


def armA_cell(d, m, k, order):
    # replay arm A's guarded cycle with threeblock, keeping the pre-V state of each round
    s0 = tb.initial_state(d, m); cur = None; cur_sc = None; hist = []
    for r in range(1, armA.MAX_ROUNDS + 1):
        s = s0 if cur is None else cur
        for b in order:
            if b == "G": s = tb.apply_G(s, k)
            elif b == "T": s = tb.apply_T(s)
            else:
                preV = s; s = tb.apply_V(s, alpha=tb.ALPHA, guard=True)
        sc = tb.score(s)
        acc = True if cur is None else sc["total_uncharged"] < cur_sc["total_uncharged"] - armA.TOL
        if not acc: break
        hist.append((r, preV, s, sc)); cur, cur_sc = s, sc
    r, preV, st, sc = hist[-1]
    keys = I.load(os.path.join(S, "%s_2M_%s_s42" % (d, m))); f = I.fences(keys)
    o = f[0]; sp = float(max(1, f[-1] - o)); raw = lambda q: float(q - o) / sp
    xs = [raw(q) for q in f]
    g = my_gmap(xs, st.gaps.idx, st.gaps.s) if st.gaps is not None else (lambda x: x)
    w = my_warp(st.warp.units) if st.warp is not None else (lambda u: u)
    feat = lambda q: w(g(raw(q)))
    # V independently on the pre-V features of the best round
    gp = my_gmap(xs, preV.gaps.idx, preV.gaps.s) if preV.gaps is not None else (lambda x: x)
    wp = my_warp(preV.warp.units) if preV.warp is not None else (lambda u: u)
    slot_i, vf_i, _, _, _ = I.smooth_cdf([wp(gp(raw(q))) for q in f], 4.0)
    kept = st.log[-1][3]
    slots_match = (slot_i == st.slots) if not kept else None
    mp_tbslots = iscore(f, feat, st.slots, st.virt)
    run = json.load(open(os.path.join(TB, "armA_runs", "%s_%s_k%d_%s.json" % (d, m, k, order))))
    best = [x for x in run["rounds"] if x.get("accepted")][-1]
    # also round 1 fully independent (no guard involved)
    r1, preV1, st1, sc1 = hist[0]
    g1 = my_gmap(xs, st1.gaps.idx, st1.gaps.s) if st1.gaps is not None else (lambda x: x)
    w1 = my_warp(st1.warp.units)
    feat1 = lambda q: w1(g1(raw(q)))
    s1, v1, _, _, _ = I.smooth_cdf([feat1(q) for q in f], 4.0) if order == "GTV" else (None, None, None, None, None)
    if order == "TGV":
        gp1 = my_gmap(xs, preV1.gaps.idx, preV1.gaps.s); wp1 = my_warp(preV1.warp.units)
        s1, v1, _, _, _ = I.smooth_cdf([wp1(gp1(raw(q))) for q in f], 4.0)
    return {"cell": "A %s-%s k=%d %s" % (d, m, k, order), "best_round": r, "best_round_stored": best["round"],
            "model_indep_best": mp_tbslots, "model_stored_best": best["model_probes"],
            "V_kept_previous_in_best_round": kept, "indep_smoothcdf_slots_match_best": slots_match,
            "model_indep_round1": iscore(f, feat1, s1, len(v1)), "model_stored_round1": run["rounds"][0]["model_probes"],
            "charge_indep": charge(st.gaps.k if st.gaps else 0), "charge_stored": best["gap_table_probes"],
            "mono_violations_best": mono(f, feat), "mono_violations_round1": mono(f, feat1)}


if __name__ == "__main__":
    which = sys.argv[1]
    res = []
    if which == "A":
        for c in [("books", "uniform", 64, "GTV"), ("fb", "window", 1, "TGV"), ("genome", "window", 16, "TGV"), ("planet", "uniform", 64, "GTV")]:
            res.append(armA_cell(*c)); print(json.dumps(res[-1]), flush=True)
    else:
        B = json.load(open(os.path.join(TB, "armB.json")))
        Bs = {(s["dataset"], s["mode"]): s for s in B["samples"]}
        cells = [("books", "uniform", 64, "G->V"), ("planet", "uniform", 64, "T->G->V"), ("osm", "window", 64, "G->V")]
        if which == "Bheavy":
            cells = [("osm", "uniform", 64, "G->T->V"), ("osm", "uniform", 1, "G->V")]
        for d, m, k, v in cells:
            part = json.load(open(os.path.join(TB, "armB_parts", "k_%s_%s_%d.json" % (d, m, k))))[v]
            assert part["model_probes"] == Bs[(d, m)]["per_k"][str(k)]["variants"][v]["model_probes"]
            res.append(armB_cell(d, m, k, v, part)); print(json.dumps(res[-1]), flush=True)
    json.dump(res, open(os.path.join(HERE, "v_indep_%s.json" % which), "w"), indent=1)
