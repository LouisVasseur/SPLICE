import json
import threeblock as tb
out = []
print("%-8s %-7s %5s %11s %8s %8s %8s %8s %6s" % ("dataset", "mode", "n", "max/median", "top1", "top4", "top16", "top64", ">10med"))
for d in tb.DATASETS:
    for m in tb.MODES:
        f = tb.load(d, m)
        s = tb.gap_stats(f)
        s.update(dataset=d, mode=m, n_fences=len(f))
        out.append(s)
        print("%-8s %-7s %5d %11.1f %8.4f %8.4f %8.4f %8.4f %6d" % (d, m, len(f), s["max_over_median"], s["top1_frac"], s["top4_frac"], s["top16_frac"], s["top64_frac"], s["gaps_over_10x_median"]))
json.dump(out, open("gap_stats.json", "w"), indent=1)
