#!/usr/bin/env python3
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Worked numeric examples for guide_deep/03_aidb.md (standard library only).

A. eight-key toy: OLS line key->rank (RMSE, ME), LIPP FMCD fit and conflict degree,
   optimal PLA segment count (port of PGM-index's OptimalPiecewiseLinearModel).
B. PLA port check on the repository fixture (expected 107 segments at eps 32, 1 at eps 4096).
C. coverage on a 4-dataset toy and on the real full-file hardness table (GRE etc.).
D. conformance on a 4-dataset toy and on the real packed_rank / CD pair (matches scores.json).
E. decoding Table 2's coverage column into (|C|, |U|) integers.
"""
import itertools, json, math, pathlib, statistics, struct

S = pathlib.Path(REPO + "/experimental/scaleli")
OUT = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.append(s)

# ---------------------------------------------------------------- A. toy
keys = [1, 2, 3, 4, 10, 20, 30, 100]
n = len(keys); ranks = list(range(n))
p("=== A. toy keys", keys, "ranks", ranks)
mx = statistics.fmean(keys); my = statistics.fmean(ranks)
sxx = sum((k - mx) ** 2 for k in keys); sxy = sum((k - mx) * (r - my) for k, r in zip(keys, ranks))
w = sxy / sxx; b = my - w * mx
p(f"OLS: mean(k)={mx}, mean(r)={my}, Sxx={sxx}, Sxy={sxy}, w=Sxy/Sxx={w:.6f}, b={b:.6f}")
errs = [r - (w * k + b) for k, r in zip(keys, ranks)]
p("residuals r_i - f(k_i):", [round(e, 4) for e in errs])
rmse = math.sqrt(sum(e * e for e in errs) / n); me = max(abs(e) for e in errs)
p(f"RMSE = sqrt(sum e^2 / n) = {rmse:.6f}   ME = max|e| = {me:.6f}")

# LIPP FMCD (port of hardness.hpp fmcd_fit / conflict_degree; LIPP build_tree_bulk_fmcd)
def lipp_gap_count(size):
    return 1 if size >= 1000000 else (2 if size >= 100000 else 5)
def fmcd_fit(keys, ut_eps=1e-6, trace=False):
    size = len(keys); gap = lipp_gap_count(size); L = size * (gap + 1)
    def ut_for(D): return (keys[size - 1 - D] - keys[D]) / (L - 2) + ut_eps
    i, D = 0, 1; Ut = ut_for(D)
    if trace: p(f"  FMCD: size={size} gap={gap} L={L}; start D=1, U_T=(k[{size-2}]-k[1])/(L-2)+1e-6={Ut:.6f}")
    while i < size - 1 - D:
        while i + D < size and keys[i + D] - keys[i] >= Ut:
            i += 1
        if i + D >= size: break
        D += 1
        if D * 3 > size: break
        Ut = ut_for(D)
        if trace: p(f"  FMCD: gap k[{i+D-1}]-k[{i}] < U_T -> D={D}, U_T=(k[{size-1-D}]-k[{D}])/(L-2)+1e-6={Ut:.6f}")
    fallback = D * 3 > size
    if not fallback:
        a = 1.0 / Ut; base = L / 2; anchor = (keys[size - 1 - D] + keys[D]) / 2
        model = lambda k: a * (k - anchor) + base
        if trace: p(f"  FMCD: final D={D}, U_T={Ut:.6f}, a=1/U_T={a:.6f}, anchor=(k[{size-1-D}]+k[{D}])/2={anchor}, base=L/2={base}")
    else:
        mid1 = (size - 1) // 3; mid2 = (size - 1) * 2 // 3; g = gap + 1
        t1 = mid1 * g + g // 2; t2 = mid2 * g + g // 2
        x1 = (keys[mid1] + keys[mid1 + 1]) / 2; x2 = (keys[mid2] + keys[mid2 + 1]) / 2
        a = (t2 - t1) / (x2 - x1); model = lambda k: a * (k - x1) + t1
        if trace: p(f"  FMCD: fallback (D*3 > size): two-point fit through ({x1},{t1}) and ({x2},{t2}), a={a:.6f}")
    def predict(k):
        v = model(k)
        if v < 0: return 0
        return min(L - 1, int(v))
    return predict, D, Ut, fallback, L
predict, D, Ut, fb_, L = fmcd_fit(keys, trace=True)
slots = [predict(k) for k in keys]
p("  predicted slots PREDICT_POS(k) = clamp(floor(a*(k-anchor)+base), 0, L-1):", slots)
run = best = 1
for i in range(1, n):
    if slots[i] == slots[i - 1]: run += 1
    else: best = max(best, run); run = 1
best = max(best, run)
p(f"  conflict degree CD = longest run of equal slots = {best}")

keys2 = [10, 11, 12, 13, 14, 60, 61, 100]
p("=== A2. second toy (dense cluster)", keys2)
predict2, D2, Ut2, fb2, L2 = fmcd_fit(keys2, trace=True)
slots2 = [predict2(k) for k in keys2]
p("  predicted slots:", slots2)
run = best2 = 1
for i in range(1, len(keys2)):
    if slots2[i] == slots2[i - 1]: run += 1
    else: best2 = max(best2, run); run = 1
best2 = max(best2, run)
p(f"  conflict degree CD = {best2}  (LIPP's own bound D = {D2})")
mx2 = statistics.fmean(keys2); sxx2 = sum((k - mx2) ** 2 for k in keys2); sxy2 = sum((k - mx2) * (r - 3.5) for k, r in zip(keys2, ranks))
w2 = sxy2 / sxx2; b2 = 3.5 - w2 * mx2; e2 = [r - (w2 * k + b2) for k, r in zip(keys2, ranks)]
p(f"  OLS: w={w2:.6f} b={b2:.6f} residuals={[round(e,3) for e in e2]} RMSE={math.sqrt(sum(e*e for e in e2)/8):.6f} ME={max(abs(e) for e in e2):.6f}")

# PLA: port of PGM-index OptimalPiecewiseLinearModel (exact integer cross products)
class OptimalPLA:
    def __init__(self, eps): self.eps = eps; self.reset()
    def reset(self): self.points = 0; self.lower = []; self.upper = []; self.ls = self.us = 0; self.rect = [None] * 4
    @staticmethod
    def sub(a, b): return (a[0] - b[0], a[1] - b[1])          # slope as (dx, dy)
    @staticmethod
    def lt(s, t): return s[1] * t[0] < s[0] * t[1]              # s < t  (dy/dx compare, dx > 0)
    @staticmethod
    def gt(s, t): return s[1] * t[0] > s[0] * t[1]
    @staticmethod
    def cross(O, A, B):
        oa = (A[0] - O[0], A[1] - O[1]); ob = (B[0] - O[0], B[1] - O[1]); return oa[0] * ob[1] - oa[1] * ob[0]
    def add_point(self, x, y):
        if self.points > 0 and x <= self.last_x: raise ValueError("points must be increasing by x")
        self.last_x = x
        p1 = (x, y + self.eps); p2 = (x, y - self.eps)
        if self.points == 0:
            self.first_x = x; self.rect[0] = p1; self.rect[1] = p2
            self.upper = [p1]; self.lower = [p2]; self.us = self.ls = 0; self.points += 1; return True
        if self.points == 1:
            self.rect[2] = p2; self.rect[3] = p1; self.upper.append(p1); self.lower.append(p2); self.points += 1; return True
        slope1 = self.sub(self.rect[2], self.rect[0]); slope2 = self.sub(self.rect[3], self.rect[1])
        outside1 = self.lt(self.sub(p1, self.rect[2]), slope1); outside2 = self.gt(self.sub(p2, self.rect[3]), slope2)
        if outside1 or outside2: self.points = 0; return False
        if self.lt(self.sub(p1, self.rect[1]), slope2):
            mn = self.sub(self.lower[self.ls], p1); mi = self.ls
            for i in range(self.ls + 1, len(self.lower)):
                val = self.sub(self.lower[i], p1)
                if self.gt(val, mn): break
                mn = val; mi = i
            self.rect[1] = self.lower[mi]; self.rect[3] = p1; self.ls = mi
            end = len(self.upper)
            while end >= self.us + 2 and self.cross(self.upper[end - 2], self.upper[end - 1], p1) <= 0: end -= 1
            del self.upper[end:]; self.upper.append(p1)
        if self.gt(self.sub(p2, self.rect[0]), slope1):
            mx_ = self.sub(self.upper[self.us], p2); mi = self.us
            for i in range(self.us + 1, len(self.upper)):
                val = self.sub(self.upper[i], p2)
                if self.lt(val, mx_): break
                mx_ = val; mi = i
            self.rect[0] = self.upper[mi]; self.rect[2] = p2; self.us = mi
            end = len(self.lower)
            while end >= self.ls + 2 and self.cross(self.lower[end - 2], self.lower[end - 1], p2) >= 0: end -= 1
            del self.lower[end:]; self.lower.append(p2)
        self.points += 1; return True

def pla_segments(keys, eps, record=None):
    """Segment count with PGM's make_segmentation semantics: sentinel (x_last+1, n) appended; keys distinct here."""
    opt = OptimalPLA(eps); c = 0; n = len(keys); starts = [0]
    def add(x, y):
        nonlocal c
        if not opt.add_point(x, y):
            c += 1; starts.append(y); opt.add_point(x, y)
    for i in range(n): add(keys[i], i)
    add(keys[-1] + 1, n)
    if record is not None: record.extend(starts)
    return c + 1

for eps in (0, 1, 2):
    starts = []; c = pla_segments(keys, eps, starts)
    p(f"  PLA-{eps}: {c} segments; segment starts at ranks {starts} (rank n={n} is PGM's sentinel point (x_last+1, n))")

for eps in (0, 1, 2):
    starts = []; c = pla_segments(keys2, eps, starts)
    p(f"  toy2 PLA-{eps}: {c} segments; segment starts at ranks {starts}")

# ---------------------------------------------------------------- B. fixture check
fx = S.parent.parent / "evidence" / "dense_sparse_fixture_uint64"
if fx.is_file():
    raw = fx.read_bytes(); cnt = struct.unpack("<Q", raw[:8])[0]
    fk = list(struct.unpack(f"<{cnt}Q", raw[8:8 + 8 * cnt]))
    p(f"=== B. fixture {fx.name}: {cnt} keys, sorted={all(fk[i] < fk[i+1] for i in range(cnt-1))}")
    for eps in (32, 4096):
        p(f"  python PLA port: eps={eps} -> {pla_segments(fk, eps)} segments (verification/pla verdict: 107 at 32, 1 at 4096)")
    # OLS on the fixture, to compare with scaleli_hardness if ever needed
else:
    p("=== B. fixture not found at", fx)

# ---------------------------------------------------------------- C. coverage
def harder(hi, hj, dims):
    return all(hi[k] >= hj[k] for k in dims) and any(hi[k] > hj[k] for k in dims)
def classify(h, dims):
    C, U = [], []
    for a, b in itertools.combinations(sorted(h), 2):
        if harder(h[a], h[b], dims): C.append((a, b))
        elif harder(h[b], h[a], dims): C.append((b, a))
        else: U.append((a, b))
    return C, U
toy = {"A": {"x": 1, "y": 1}, "B": {"x": 2, "y": 3}, "C": {"x": 3, "y": 2}, "D": {"x": 3, "y": 4}}
p("=== C. coverage toy", toy)
for dims in (["x"], ["y"], ["x", "y"]):
    C, U = classify(toy, dims)
    p(f"  dims={dims}: comparable (harder, easier) = {C}; incomparable = {U}; Cov = ({len(C)}-{len(U)})/{len(C)+len(U)} = {(len(C)-len(U))/(len(C)+len(U)):.4f}")
H = json.load(open(S / "results/aidb/hardness.json"))
FIELDS = {"RMSE": "rmse", "ME": "max_error", "CD": "conflict_degree", "PLA-32": "pla_32", "PLA-4096": "pla_4096"}
full = {d: {m: H[d]["full"][f] for m, f in FIELDS.items()} for d in H}
p("=== C. real full-file hardness table")
for d in sorted(full): p(f"  {d:8s}", {m: full[d][m] for m in FIELDS})
for dims in (["PLA-32", "PLA-4096"], ["CD", "PLA-32"], ["RMSE", "ME"]):
    C, U = classify(full, dims)
    p(f"  {'·'.join(dims)}: |C|={len(C)} |U|={len(U)} Cov={(len(C)-len(U))/45:.4f}; incomparable: {U}")
# comparable partners per dataset under GRE (paper Fig. 2 colour scale 4..8) and CD·PLA-32 (Fig. 6: 6..9)
for dims in (["PLA-32", "PLA-4096"], ["CD", "PLA-32"]):
    C, U = classify(full, dims); cnt = {d: 0 for d in full}
    for a, b in C: cnt[a] += 1; cnt[b] += 1
    p(f"  comparable partners per dataset under {'·'.join(dims)}: {dict(sorted(cnt.items()))}")
# which pairs are reversed by CD relative to all four other scalars (candidates for the one-pair difference)
cands = []
for a, b in itertools.combinations(sorted(full), 2):
    dirs = [(full[a][m] > full[b][m]) - (full[a][m] < full[b][m]) for m in ("RMSE", "ME", "PLA-32", "PLA-4096")]
    cd = (full[a]["CD"] > full[b]["CD"]) - (full[a]["CD"] < full[b]["CD"])
    if len(set(dirs)) == 1 and dirs[0] != 0 and cd != dirs[0]:
        cands.append((a, b, "CD tie" if cd == 0 else "CD reversed", full[a]["CD"], full[b]["CD"]))
p("  pairs ordered identically by RMSE, ME, PLA-32, PLA-4096 but reversed/tied by our CD:", cands)

# ---------------------------------------------------------------- D. conformance
def sigmoid(x): return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
def conformance(p_raw, C, ddof=0, trace=False):
    sd = statistics.pstdev(p_raw.values()) if ddof == 0 else statistics.stdev(p_raw.values())
    ph = {d: v / sd for d, v in p_raw.items()}
    R = P = 0.0; rows = []
    for hard, easy in C:
        gap = ph[hard] - ph[easy]; wgt = sigmoid(gap)
        if gap > 0: P += wgt; kind = "VIOLATING"
        else: R += wgt; kind = "conforming"
        rows.append((hard, easy, gap, wgt, kind))
    if trace:
        p(f"  std = {sd:.6f}; p_hat = {{{', '.join(f'{d}: {v:.4f}' for d, v in ph.items())}}}")
        for r in rows: p(f"    ({r[0]} harder than {r[1]}): delta = {r[2]:+.4f}, w = sigmoid(delta) = {r[3]:.4f}  {r[4]}")
        p(f"  R = {R:.6f}, P = {P:.6f}, Conf_I = (R-P)/(R+P) = {(R-P)/(R+P):.6f}")
    return (R - P) / (R + P), R, P, rows
p("=== D. conformance toy: hardness as in C (dims x,y), two indexes")
C, U = classify(toy, ["x", "y"])
perf = {"I1": {"A": 10.0, "B": 8.0, "C": 9.0, "D": 5.0}, "I2": {"A": 6.0, "B": 7.0, "C": 5.0, "D": 4.0}}
confs = []
for I, pr in perf.items():
    p(f"  index {I}: throughput {pr}")
    c, R, P, _ = conformance(pr, C, trace=True); confs.append(c)
p(f"  Conf = mean = {statistics.fmean(confs):.6f}; Cov(x·y) = {(len(C)-len(U))/(len(C)+len(U)):.4f}")
p("=== D. real: packed_rank throughput (median of 3 seeds, MOPS) vs scalar CD on full-file hardness")
T = json.load(open(S / "results/aidb/throughput.json"))["packed_rank"]
C, U = classify(full, ["CD"])
c, R, P, rows = conformance(T, C, trace=False)
sd = statistics.pstdev(T.values())
p(f"  std(packed_rank) = {sd:.6f} MOPS; p_hat = {{{', '.join(f'{d}: {v/sd:.3f}' for d, v in T.items())}}}")
viol = [r for r in rows if r[4] == "VIOLATING"]
p(f"  |C|={len(C)}: conforming {len(rows)-len(viol)}, violating {len(viol)}; R={R:.6f} P={P:.6f} Conf_packed_rank(CD) = {c:.6f}  (scores.json: 0.4724122609070333)")
for r in viol: p(f"    violating: {r[0]} (CD {full[r[0]]['CD']:.0f}) harder than {r[1]} (CD {full[r[1]]['CD']:.0f}) but faster: delta={r[2]:+.4f}, w={r[3]:.4f}")
smallest = sorted((r for r in rows if r[4] == "conforming"), key=lambda r: r[3])[:3]
for r in smallest: p(f"    easiest conforming (smallest w): {r[0]} > {r[1]}: delta={r[2]:+.4f}, w={r[3]:.4f}")
# the paper's history/libio anecdote: sigma of LIPP not reported; show w for illustrative sigmas
p("  paper anecdote LIPP: libio 7.1 MOPS vs history 5.6 MOPS (libio harder under GRE) -> violating pair with delta = 1.5/sigma_LIPP")
for sg in (0.5, 1.0, 1.5, 2.0): p(f"    if sigma_LIPP were {sg}: delta={1.5/sg:.3f}, w={sigmoid(1.5/sg):.4f}")

# ---------------------------------------------------------------- E. decode Table 2 coverage
p("=== E. Table 2 coverage -> (|C|, |U|) with |C|+|U| = 45")
for cov in (1.00, 0.82, 0.60, 0.56, 0.51, 0.47, 0.42, 0.38, 0.33, 0.24, 0.20, 0.16):
    sols = [(c, 45 - c) for c in range(46) if round((c - (45 - c)) / 45, 2) == cov]
    p(f"  Cov={cov:.2f}: {sols}")

pathlib.Path(__file__).with_suffix(".out.txt").write_text("\n".join(OUT) + "\n")
