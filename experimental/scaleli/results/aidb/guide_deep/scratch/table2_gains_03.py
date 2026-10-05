#!/usr/bin/env python3
"""Recompute Figure-5-style dimension gains/drops from the printed Table 2 (AIDB 2026, p. 5) and keys-per-segment."""
import itertools, json, statistics
T2 = {  # metric: (RMI, PGM, ALEX, LIPP, XIndex, FINEdex, Conf, Cov)  [AIDB Table 2]
"RMSE":(0.34,0.19,0.41,0.09,-0.14,0.26,0.19,1.00),"ME":(0.32,0.04,0.39,0.22,-0.15,0.11,0.15,1.00),
"CD":(0.18,0.36,0.10,0.84,0.09,0.11,0.28,1.00),"PLA-32":(0.41,0.64,0.71,0.29,0.02,0.91,0.50,1.00),
"PLA-4096":(0.38,-0.03,0.17,0.03,0.03,-0.10,0.08,1.00),
"PLA-32·PLA-4096":(1.00,0.71,1.00,0.42,0.27,0.86,0.71,0.47),"RMSE·ME":(0.44,0.18,0.53,0.20,-0.09,0.24,0.25,0.82),
"RMSE·CD":(0.68,0.69,0.69,0.89,0.14,0.53,0.60,0.47),"RMSE·PLA-32":(0.69,0.73,1.00,0.34,0.05,1.00,0.64,0.60),
"RMSE·PLA-4096":(1.00,0.40,0.84,0.32,0.16,0.39,0.52,0.42),"ME·CD":(0.57,0.45,0.57,0.90,0.09,0.34,0.49,0.56),
"ME·PLA-32":(0.70,0.62,1.00,0.43,0.04,0.88,0.61,0.60),"ME·PLA-4096":(1.00,0.30,0.85,0.42,0.14,0.29,0.50,0.42),
"CD·PLA-32":(0.58,0.87,0.73,0.89,0.21,0.87,0.69,0.60),"CD·PLA-4096":(0.83,0.55,0.53,0.88,0.28,0.25,0.55,0.42),
"RMSE·ME·CD":(0.67,0.68,0.68,0.88,0.13,0.51,0.59,0.42),"RMSE·ME·PLA-32":(0.67,0.71,1.00,0.39,0.01,1.00,0.63,0.51),
"RMSE·ME·PLA-4096":(1.00,0.35,0.83,0.37,0.11,0.33,0.50,0.33),"RMSE·CD·PLA-32":(0.79,1.00,1.00,1.00,0.17,1.00,0.83,0.33),
"RMSE·CD·PLA-4096":(1.00,0.76,0.77,1.00,0.14,0.53,0.70,0.16),"RMSE·PLA-32·PLA-4096":(1.00,0.80,1.00,0.42,0.15,1.00,0.73,0.24),
"ME·CD·PLA-32":(0.81,0.83,1.00,1.00,0.18,0.84,0.78,0.38),"ME·CD·PLA-4096":(1.00,0.60,0.80,1.00,0.16,0.41,0.66,0.20),
"ME·PLA-32·PLA-4096":(1.00,0.63,1.00,0.55,0.13,0.82,0.69,0.24),"CD·PLA-32·PLA-4096":(1.00,0.80,1.00,0.85,0.20,0.81,0.78,0.24)}
S = ["RMSE","ME","CD","PLA-32","PLA-4096"]
def name(dims): return "·".join(d for d in S if d in dims)
# printed Conf vs mean of the six per-index values (rounding check)
worst = max(abs(statistics.fmean(v[:6]) - v[6]) for v in T2.values()); print(f"max |mean(Conf_I) - printed Conf| = {worst:.4f}")
for base_k in (1, 2):
    print(f"--- added dimension on {base_k}-dim bases: mean conformance gain [min,max], mean coverage drop [min,max]")
    for add in S:
        gains, drops = [], []
        for base in itertools.combinations(S, base_k):
            if add in base: continue
            b = name(base); m = name(base + (add,))
            gains.append(T2[m][6] - T2[b][6]); drops.append(T2[b][7] - T2[m][7])
        print(f"  +{add:9s}: gain {statistics.fmean(gains):+.3f} [{min(gains):+.2f},{max(gains):+.2f}]  drop {statistics.fmean(drops):.3f} [{min(drops):.2f},{max(drops):.2f}]  n={len(gains)}")
# per-index: best scalar and best overall
idx = ["RMI","PGM","ALEX","LIPP","XIndex","FINEdex"]
for j, I in enumerate(idx):
    bs = max((v[j], k) for k, v in T2.items() if "·" not in k); bo = max((v[j], k) for k, v in T2.items())
    print(f"{I:8s} best scalar {bs[1]} {bs[0]:.2f}; best overall {bo[1]} {bo[0]:.2f}; XIndex-style min over all metrics {min(v[j] for v in T2.values()):.2f}")
# keys per segment on the full files
H = json.load(open("results/aidb/hardness.json"))
print("--- keys per PLA segment (200M / segments), full files")
for d in H:
    f = H[d]["full"]; print(f"  {d:8s} PLA-32 {f['pla_32']:>10,.0f} -> {2e8/f['pla_32']:>9,.0f} keys/segment; PLA-4096 {f['pla_4096']:>6,.0f} -> {2e8/f['pla_4096']:>11,.0f} keys/segment; RMSE/N = {f['rmse']/2e8*100:.3f}% ; ME/N = {f['max_error']/2e8*100:.2f}%")
