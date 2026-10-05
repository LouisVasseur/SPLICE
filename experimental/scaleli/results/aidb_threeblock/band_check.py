"""perturbed_V_spread on the raw feature must reproduce last week's csv_only bands."""
import json
from multiprocessing import Pool
import threeblock as tb
REF = {"books": 0.361, "wise": 0.263, "genome": 0.197, "libio": 0.175, "covid": 0.086,
       "history": 0.050, "stack": 0.045, "fb": 0.0}
def r(d):
    x = tb.perturbed_V_spread(tb.initial_state(d)); return d, x
if __name__ == "__main__":
    with Pool(8) as p:
        out = dict(p.map(r, list(REF)))
    for d in REF:
        x = out[d]
        print("%-8s base %.4f max_dev %.4f spread %.4f   last week %.3f" % (d, x["base"], x["max_dev"], x["spread"], REF[d]))
    json.dump(out, open("band_check.json", "w"), indent=1)
