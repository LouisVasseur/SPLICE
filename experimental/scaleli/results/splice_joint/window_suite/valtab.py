# Window vs full-scale residuals of the chosen cell: python3 valtab.py J DIR SUFFIX...
import json, sys
J, d = sys.argv[1], sys.argv[2]
for ds in "fb osm books covid genome history libio planet stack wise".split():
    c = json.load(open(f"{J}/ref_full/{ds}.json"))["chosen"]["fast"]["cell"]
    red = c["ed_milli"]["4k_2048"]/1000; rd = c["direct_key_ppm"]/1e6; rl=c["counts"]["dep_lines_milli"]/1000; rw=sum(c["counts"]["pages4k_milli"])/1000; rb=c["bytes_per_key_milli"]/1000
    out=[]
    for suf in sys.argv[3:]:
        try: v=json.load(open(f"{J}/{d}/{ds}_{suf}.json"))["variants"][0]
        except FileNotFoundError: continue
        out.append(f"{suf}: ed {v['ed_milli']/1000-red:+.3f} dir {v['direct_key_ppm']/1e6-rd:+.3f} ln {v['dep_lines_milli']/1000-rl:+.3f} wk {v['pages4k_milli']/1000-rw:+.3f} B {v['bytes_per_key_milli']/1000-rb:+.2f} rc {v['router_child_milli']}")
    print(f"{ds:7s} {red:.3f}", " | ".join(out))
