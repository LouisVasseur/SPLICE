import json, os, glob, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); TB = os.path.dirname(HERE)
A = json.load(open(os.path.join(TB, "armA.json")))
rows = {(r["dataset"], r["mode"], r["k"], r["order"]): r for r in A["rows"]}
runs = {}
for p in glob.glob(os.path.join(HERE, "band_runs2", "*.json")):
    r = json.load(open(p)); runs.setdefault((r["dataset"], r["mode"], r["k"], r["order"]), {})[r["pert"]] = r
out = []
for c, vs in sorted(runs.items(), key=lambda kv: -rows[kv[0]]["gain_rounds2plus"]):
    if len(vs) < 17 or any("error" in v for v in vs.values()): print("INCOMPLETE", c, len(vs)); continue
    def r1(v): return v["rounds"][0]["model"]
    def r1t(v): return v["rounds"][0]["total"]
    def best(v): return [x for x in v["rounds"] if x.get("accepted")][-1]["total"]
    idv = vs["id"]
    gain = r1t(idv) - best(idv)
    e6 = [k for k in vs if k == "id" or k.endswith("@1e-06")]
    b6 = max(r1(vs[k]) for k in e6) - min(r1(vs[k]) for k in e6)
    ball = max(r1(v) for v in vs.values()) - min(r1(v) for v in vs.values())
    gains = [r1t(v) - best(v) for v in vs.values()]
    bests = [best(v) for v in vs.values()]
    best_reroll_single = min(r1t(v) for v in vs.values())
    # coupling under perturbation: did gap set change in an accepted round?
    gchg = sum(1 for v in vs.values() if any(set(x["gaps"]) != set(v["rounds"][0]["gaps"]) for x in v["rounds"][1:] if x.get("accepted")))
    rr = rows[c]
    out.append(dict(cell="%s-%s k=%d %s" % (c[0], c[1][0], c[2], c[3]), gain=gain, gain_armA=rr["gain_rounds2plus"],
                    id_matches_armA=abs(r1t(idv) - rr["round1"]["total_uncharged"]) < 1e-12 and abs(best(idv) - rr["best"]["total_uncharged"]) < 1e-12,
                    band_1e6=b6, band_c4=ball, Vonly_spread_armA=rr["band_spread"],
                    gain_gt_band_1e6=gain > b6, gain_gt_band_c4=gain > ball,
                    gain_med_over_perts=st.median(gains), gain_min_over_perts=min(gains), gain_pos_frac=sum(g > 1e-6 for g in gains) / len(gains),
                    best_spread=max(bests) - min(bests), cycle_best_id=best(idv), best_single_pass_reroll=best_reroll_single,
                    cycle_beats_best_reroll=best(idv) < best_reroll_single - 1e-12, gapset_changed_accepted_perts=gchg,
                    gapset_changed_accepted_id=rr["gapset_changed_in_accepted_round"]))
json.dump(out, open(os.path.join(HERE, "v_band.json"), "w"), indent=1)
print("%-24s %6s %7s %7s %7s | %6s %6s %5s | %7s %7s %s %s" % ("cell", "gain", "b1e-6", "b_c4", "Vspr", "gmed", "gmin", "pos", "cycle", "reroll", "chgA", "chgP"))
for o in out:
    print("%-24s %6.3f %7.3f %7.3f %7.3f | %6.3f %6.3f %5.2f | %7.3f %7.3f %s %2d %s" % (o["cell"], o["gain"], o["band_1e6"], o["band_c4"], o["Vonly_spread_armA"],
          o["gain_med_over_perts"], o["gain_min_over_perts"], o["gain_pos_frac"], o["cycle_best_id"], o["best_single_pass_reroll"],
          "Y" if o["gapset_changed_accepted_id"] else "n", o["gapset_changed_accepted_perts"], "" if o["id_matches_armA"] else "ID-MISMATCH"))
G = [o for o in out if " k=0 " not in o["cell"]]
print("G cells:", len(G), "gain>band_1e6:", sum(o["gain_gt_band_1e6"] for o in G), "gain>band_c4:", sum(o["gain_gt_band_c4"] for o in G),
      "cycle beats best single-pass reroll:", sum(o["cycle_beats_best_reroll"] for o in G), "gain positive in all 17 perts:", sum(o["gain_pos_frac"] == 1 for o in G),
      "zero 1e-6 band:", sum(o["band_1e6"] == 0 for o in G))
