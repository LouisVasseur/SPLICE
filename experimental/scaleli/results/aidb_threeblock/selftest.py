"""threeblock self-tests: G invertibility/monotonicity/no-collision, k=0 identity,
G-reselection stability, V block vs the independent smooth_cdf, composed scoring."""
import random, time, json
import threeblock as tb
import indep
import probe_metric as pm
res = {}

# 1. GapMap properties on the most gappy samples
for d, m in [("osm", "uniform"), ("planet", "uniform"), ("books", "uniform"), ("genome", "uniform"), ("osm", "window"), ("fb", "uniform")]:
    s0 = tb.initial_state(d, m)
    for k in (1, 4, 16, 64, 256, 488):
        sG = tb.apply_G(s0, k)
        gm = sG.gaps
        if gm is None:
            continue
        u = sG.u_at_fences()
        assert all(u[i + 1] > u[i] for i in range(len(u) - 1))
        assert abs(u[0]) < 1e-15 and abs(u[-1] - 1.0) < 1e-12, (u[0], u[-1])
        pk = pm.root_probe_keys(s0.f)
        xs = [float(q - s0.origin) / float(s0.span) for q in pk]
        err = max(abs(gm.inverse(gm(x)) - x) for x in xs)
        assert err < 1e-12, err
        sG.assert_monotone()
        # every shrunk gap now has width == median raw width (in pre-renormalised units)
        med = tb.median([s0.xs[i + 1] - s0.xs[i] for i in range(s0.n - 1)])
        for i in gm.idx:
            wnew = (gm(s0.xs[i + 1]) - gm(s0.xs[i])) * gm.scale
            assert abs(wnew - med) <= 1e-12 * max(1.0, med) + 1e-15, (i, wnew, med)
    res.setdefault("gapmap_ok", []).append("%s_%s" % (d, m))
print("1 GapMap: monotone, invertible (err<1e-12), shrunk widths == median, renormalised to [0,1]")

# 2. k=0 is the identity path, bit-exact
s0 = tb.initial_state("books", "uniform")
assert tb.apply_G(s0, 0).gaps is None
assert tb.score(tb.apply_G(s0, 0))["model_probes"] == tb.score(s0)["model_probes"]
print("2 k=0 -> bit-identical to baseline")

# 3. (a) with T off, G is idempotent.  (b) with T on and the warp held FIXED,
#    re-applying G is NOT idempotent by construction: G renormalises u and moves
#    fences under the fixed nonlinear warp, so the current-coordinate widths
#    change.  That churn is the G<->T coupling itself; we measure it.
for d in ("books", "planet", "osm", "wise"):
    s0 = tb.initial_state(d, "uniform")
    a0 = tb.apply_G(s0, 16); b0 = tb.apply_G(a0, 16)
    assert a0.gaps.idx == b0.gaps.idx and a0.gaps.s == b0.gaps.s, d
    s = tb.apply_T(s0)
    a = tb.apply_G(s, 16)
    b = tb.apply_G(a, 16)
    sel_in = a0.gaps.idx
    res.setdefault("k16_overlap_input_vs_afterT", {})[d] = len(set(sel_in) & set(a.gaps.idx))
    res.setdefault("k16_churn_reapply_G_fixed_warp", {})[d] = len(set(a.gaps.idx) ^ set(b.gaps.idx))
print("3 G idempotent with T off; k=16 overlap(input sel, sel after T):", res["k16_overlap_input_vs_afterT"],
      " churn when G re-applied under the same warp:", res["k16_churn_reapply_G_fixed_warp"])

# 4. V block == independent transcription, on a G-transformed coordinate
for d, m, k in [("books", "uniform", 16), ("genome", "uniform", 4), ("covid", "window", 64)]:
    st = tb.apply_G(tb.initial_state(d, m), k)
    z = st.features()
    t = time.time()
    r = tb.vfence_block(z, 4.0)
    slot, vf, sb, sa, rounds = indep.smooth_cdf(z, 4.0)
    assert slot == r["slots"] and len(vf) == r["virt"] and rounds == r["rounds"], d
    res.setdefault("V_vs_indep", []).append("%s_%s_G%d virt=%d identical" % (d, m, k, r["virt"]))
print("4 V block bit-identical to verify/indep.smooth_cdf:", res["V_vs_indep"])

# 5. composed scorer, one G+T+V state, and the probe-metric by indep on it
st = tb.apply_V(tb.apply_T(tb.apply_G(tb.initial_state("books", "uniform"), 16)))
sc = tb.score(st)
fn = st.feature_fn()
mdl = pm.LinearModel().fit_xy(st.f, st.features(), st.targets())
tab = indep.slot_table(st.slots, st.n, st.virt) if st.virt else None
ind = indep.root_probes(st.f, lambda q: mdl.predict_x(fn(q)), tab)
assert abs(ind - sc["model_probes"]) < 1e-12, (ind, sc["model_probes"])
res["books_uniform_G16_T_V"] = sc
print("5 books uniform G16->T->V:", json.dumps(sc), "indep scorer agrees")
json.dump(res, open("selftest.json", "w"), indent=1)
print("ALL OK")
