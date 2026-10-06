import json, glob, sys, statistics as S
J = sys.argv[1]
picks = {l.split()[0]: l.split()[5] for l in open(f"{J}/winsel/picks.txt")}
for ds in "fb osm books covid genome history libio planet stack wise".split():
    ref = json.load(open(f"{J}/ref_full/{ds}.json"))["chosen"]["fast"]["cell"]
    red = ref["ed_milli"]["4k_2048"] / 1000; rdir = ref["direct_key_ppm"] / 1e6
    rows = []
    for f in glob.glob(f"{J}/winsel/scan/{ds}_*.json"):
        r = json.load(open(f)); v = r["variants"][0]
        rows.append((r["off"], v["ed_milli"] / 1000, v["direct_key_ppm"] / 1e6, v["dep_lines_milli"] / 1000, v["pages4k_milli"] / 1000, v["bytes_per_key_milli"] / 1000, r["wall_s"]))
    rows.sort()
    eds = [x[1] for x in rows]
    pk = [x for x in rows if str(x[0]) == picks[ds]][0]
    print(f"{ds:8s} full {red:.3f} dir {rdir:.3f} | blocks ed min {min(eds):.3f} med {S.median(eds):.3f} max {max(eds):.3f} mean {S.mean(eds):.3f} | pick@{pk[0]} ed {pk[1]:.3f} (res {pk[1]-red:+.3f}) dir {pk[2]:.3f} (res {pk[2]-rdir:+.3f}) wall {pk[6]:.1f}s")
    if len(sys.argv) > 2: 
        for x in rows: print("   ", x[0], " ".join(f"{y:.3f}" for y in x[1:]))
