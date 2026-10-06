# Picks per dataset the 12.5M block whose log-gap stats (mean, sd, lag-1 autocorrelation, p99) are nearest
# the whole file's, each stat in units of its across-block spread. Pre-registered: no E[D] used.
import sys, statistics as S
J = sys.argv[1]
for ds in "fb osm books covid genome history libio planet stack wise".split():
    rows = [l.split() for l in open(f"{J}/winsel/{ds}.stats")]
    blocks = [(int(r[1]), int(r[2]), [float(x) for x in r[3:7]]) for r in rows if r[0] == "block"]
    full = [r for r in blocks if True]
    allr = [float(x) for x in [r for r in rows if r[0] == "all"][0][3:7]]
    nfull = [b for b in blocks if b[1] + 12500000 <= 200000000]
    sd = [max(S.pstdev([b[2][i] for b in nfull]), 1e-9) for i in range(4)]
    dist = sorted((sum(((b[2][i] - allr[i]) / sd[i]) ** 2 for i in range(4)) ** 0.5, b[0], b[1]) for b in nfull)
    print(ds, "pick block", dist[0][1], "off", dist[0][2], "dist %.2f" % dist[0][0], "next", [(d[1], round(d[0], 2)) for d in dist[1:4]])
