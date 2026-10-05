import json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "armA.json")))
R = J["rows"]
def grp(rs, lab):
    gains = [r["gain_rounds2plus"] for r in rs]
    nb = [r for r in rs if r["band_spread"] is not None]
    print("%-22s n=%3d  gain>1e-6: %3d  mean gain %.4f  median %.4f  max %.4f | >spread %d/%d  >lucky %d/%d  >listed %d/%d | gapset changed (accepted) %d  (any cand) %d  (nostop) %d | hit cap %d" % (
        lab, len(rs), sum(1 for g in gains if g > 1e-6), st.mean(gains), st.median(gains), max(gains),
        sum(1 for r in nb if r["exceeds_spread"]), len(nb), sum(1 for r in nb if r["exceeds_lucky"]), len(nb),
        sum(1 for r in rs if r["exceeds_listed"]), sum(1 for r in rs if r["band_listed"] is not None),
        sum(r["gapset_changed_in_accepted_round"] for r in rs), sum(r["gapset_changed_in_any_candidate"] for r in rs),
        sum(r["gapset_changed_nostop"] for r in rs), sum(r["hit_round_cap"] for r in rs)))
grp([r for r in R if r["k"] == 0], "k=0 (T<->V control)")
grp([r for r in R if r["k"] > 0], "k>0 all")
for k in (1, 4, 16, 64):
    grp([r for r in R if r["k"] == k], "k=%d" % k)
for o in ("GTV", "TGV"):
    grp([r for r in R if r["k"] > 0 and r["order"] == o], "order %s" % o)
for m in ("uniform", "window"):
    grp([r for r in R if r["k"] > 0 and r["mode"] == m], "mode %s" % m)
print()
print("cells (k>0) with gain > spread AND > lucky AND (> listed when listed):")
for r in R:
    if r["k"] > 0 and r["exceeds_spread"] and r["exceeds_lucky"] and r["exceeds_listed"] in (True, None):
        print("  %s-%s k=%d %s gain %.3f spread %.3f lucky %.3f listed %s gapset_changed %s  best %.3f vs k0best %.3f (model %+.3f)" % (
            r["dataset"], r["mode"], r["k"], r["order"], r["gain_rounds2plus"], r["band_spread"], r["band_lucky"], r["band_listed"],
            r["gapset_changed_in_accepted_round"], r["best"]["total_uncharged"], r["k0_best_total"], r["delta_best_vs_k0_model"]))
print()
print("gain when gap set changed in an accepted round vs not (k>0):")
for v in (True, False):
    rs = [r for r in R if r["k"] > 0 and r["gapset_changed_in_accepted_round"] == v]
    print("  changed=%s n=%d mean gain %.4f, >spread %d" % (v, len(rs), st.mean(r["gain_rounds2plus"] for r in rs), sum(1 for r in rs if r["exceeds_spread"])))
print()
print("Does any k>0 cell beat its sample's k=0 best on total_uncharged?", [ (r["dataset"],r["mode"],r["k"],r["order"]) for r in R if r["k"]>0 and r["delta_best_vs_k0_total"]<0])
mo = sorted([r for r in R if r["k"] > 0], key=lambda r: r["delta_best_vs_k0_model"])[:12]
print("k>0 cells with lowest MODEL probes relative to k=0 best (G table not counted):")
for r in mo:
    print("  %s-%s k=%d %s model %.3f vs k0best %.3f  delta %+.3f  spread %s" % (r["dataset"], r["mode"], r["k"], r["order"], r["best"]["model_probes"], r["k0_best_total"], r["delta_best_vs_k0_model"], r["band_spread"]))
print()
print("GTV vs TGV round-1 / best totals (k>0): TGV better r1 in", sum(1 for a,b in zip([r for r in R if r['k']>0 and r['order']=='GTV'],[r for r in R if r['k']>0 and r['order']=='TGV']) if b['round1']['total_uncharged']<a['round1']['total_uncharged']-1e-9), "of 80; better best in", sum(1 for a,b in zip([r for r in R if r['k']>0 and r['order']=='GTV'],[r for r in R if r['k']>0 and r['order']=='TGV']) if b['best']['total_uncharged']<a['best']['total_uncharged']-1e-9))
d = [abs(a['best']['total_uncharged']-b['best']['total_uncharged']) for a,b in zip([r for r in R if r['k']>0 and r['order']=='GTV'],[r for r in R if r['k']>0 and r['order']=='TGV'])]
print("  |GTV best - TGV best|: mean %.3f max %.3f" % (st.mean(d), max(d)))
print("k=0 control vs last week joint (uniform):", [(r["dataset"], round(r["best"]["total_uncharged"],6), round(r["lastweek_best_uniform"],6)) for r in R if r["k"]==0 and r["mode"]=="uniform"])
print("nostop post-selected gains > guarded gain:", sum(1 for r in R if r["gain_nostop_postselected"] > r["gain_rounds2plus"] + 1e-9), "; nostop loss-selected worse than round1:", sum(1 for r in R if r["gain_nostop_loss_selected"] < -1e-9))
