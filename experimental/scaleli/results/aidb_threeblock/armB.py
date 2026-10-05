#!/usr/bin/env python3
"""ARM B -- single pass.  Imports threeblock unchanged.

Per sample (20) and k in {1,4,16,64}:
  baseline, G, T, V, G->T->V, T->G->V, G->V, T->V          (fixed orders, one pass)
  pool, pool_pwl, poolV, poolV_pwl                          (simultaneous priced greedy)

POOL (the "simultaneous allocation" version).  State = (gap set S with fixed
shrink factors, warp on/off, virtual-fence COUNT per gap c_i).  Slot ranks
s_i = i + sum_{j<i} c_j are feature independent, so a virtual fence is just
"+1 target to every fence right of gap i", whatever the coordinate.  Candidates
each step:
   V_i   add one virtual fence in gap i          cost 0 probes, uses budget int(4n)
   G_i   shrink gap i (|S| < k, W_i > W_med)     cost ceil(log2(k+1))/k  (amortised share)
   T     fit the warp to the current targets in the current G coordinate (once)
                                                 cost 6.9   (pool_pwl: 0.45)
Each candidate is evaluated EXACTLY on the fit_root probe metric (same
locate routine, same Welford fit, same slot table); its reduction is
   d = objective(before) - objective(after),
   objective = model_probes + |S| * share + T_cost * [warp on].
Ratio = d / own cost (a virtual fence has cost 0 -> ratio +inf when d > 0).
Take the max-ratio candidate with d > 1e-12; stop when none.  The amortised
share is the GENEROUS price for G (ceil(log2(k_eff+1)) >= k_eff*share for
k_eff <= k); final rows are re-scored with threeblock.score, i.e. the TRUE
table charge ceil(log2(k_eff+1)).
poolV / poolV_pwl: same greedy started from the V-only (CSV Algorithm 1) state.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import threeblock as tb  # noqa: E402

pm = tb.pm
KS = [1, 4, 16, 64]
PARTS = os.path.join(HERE, "armB_parts")
EPS = 1e-12


def row(state, extra=None):
    sc = tb.score(state)
    out = {k: sc[k] for k in ("model_probes", "gap_table_probes", "gap_k", "transform_charge_probes",
                              "total_uncharged", "total_charged", "total_charged_pwl",
                              "virtual_count", "sse")}
    out["gaps"] = list(state.gaps.idx) if state.gaps is not None else []
    out["warp"] = [list(u) for u in state.warp.units] if state.warp is not None else None
    out["log"] = [list(x) for x in state.log]
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------------------
# fixed-order variants
# ---------------------------------------------------------------------------

def task_indep(d, mode):
    t0 = time.time()
    s0 = tb.initial_state(d, mode)
    sT = tb.apply_T(s0)
    sV = tb.apply_V(s0)
    sTV = tb.apply_V(sT)
    res = {"baseline": row(s0), "T": row(sT), "V": row(sV), "T->V": row(sTV),
           "V_slots": sV.slots, "seconds": time.time() - t0}
    return res


def task_k(d, mode, k):
    t0 = time.time()
    s0 = tb.initial_state(d, mode)
    out = {}

    def safe(name, fn):
        try:
            st = fn()
            out[name] = row(st)
            return st
        except AssertionError as e:
            out[name] = {"error": "AssertionError: %s" % e}
            return None

    sG = safe("G", lambda: tb.apply_G(s0, k))
    sGT = safe("G->T", lambda: tb.apply_T(sG)) if sG is not None else None
    if sGT is not None:
        safe("G->T->V", lambda: tb.apply_V(sGT))
    if sG is not None:
        safe("G->V", lambda: tb.apply_V(sG))
    sT = tb.apply_T(s0)
    sTG = safe("T->G", lambda: tb.apply_G(sT, k))
    if sTG is not None:
        safe("T->G->V", lambda: tb.apply_V(sTG))
    # overlap of the gap sets chosen before vs after T
    try:
        a = set(out["G"]["gaps"])
        b = set(out["T->G"]["gaps"])
        out["overlap_G_vs_TG"] = [len(a & b), len(a), len(b)]
    except Exception:
        pass
    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# exact fast evaluator for the pool
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(self, f):
        self.f = f
        self.n = len(f)
        self.pk = pm.root_probe_keys(f)
        self.truth = [bisect.bisect_right(f, q) - 1 for q in self.pk]
        self.memo = {}
        self.NP = float(len(self.pk))
        n = self.n
        # C[j][p] = probes when the prediction lands on region p and the answer is j
        self.C = [[self.cost(p, j) for p in range(n)] for j in range(n)]
        self.memo = {}

    def scan_V(self, zf, zp, slots, v):
        """Fast screen of every 'add one virtual fence in gap i' candidate.
        Closed-form OLS (not Welford) and an O(1) virtual table lookup; the
        chosen candidate is always re-checked with the exact path (probes)."""
        n = self.n
        N = float(n)
        t = [float(s) for s in slots]
        Sx = sum(zf)
        Sxx = sum(z * z for z in zf)
        Sy = sum(t)
        Sxy = sum(zf[j] * t[j] for j in range(n))
        suf = [0.0] * (n + 1)
        for j in range(n - 1, -1, -1):
            suf[j] = suf[j + 1] + zf[j]
        table = pm.slot_to_region_table(slots, n, v) if v else list(range(n))
        m = n + v                     # last slot index after the insertion
        fm = float(m)
        cxx = Sxx - Sx * Sx / N
        C = self.C
        rows = [C[j] for j in self.truth]
        nq = len(zp)
        out = []
        for i in range(n - 1):
            sy = Sy + (n - 1 - i)
            sxy = Sxy + suf[i + 1]
            a = (sxy - Sx * sy / N) / cxx
            b = sy / N - a * (Sx / N)
            P = slots[i + 1]
            tot = 0
            for q in range(nq):
                y = a * zp[q] + b
                if y <= 0:
                    sl = 0
                elif y >= fm:
                    sl = m
                else:
                    sl = int(y)
                if sl < P:
                    r = table[sl]
                elif sl == P:
                    r = i
                else:
                    r = table[sl - 1]
                tot += rows[q][r]
            out.append((tot / self.NP, i))
        return out

    def cost(self, p, j):
        """#ok() calls of locate_from_prediction from start p when the answer is j
        (ok(i) <=> i <= j, f sorted).  Mirrors pm.locate_from_prediction."""
        key = (p, j)
        c = self.memo.get(key)
        if c is not None:
            return c
        n = self.n
        cnt = [0]

        def ok(i):
            cnt[0] += 1
            return i <= j

        if ok(p):
            lo, step, hi = p, 1, p + 1
            while hi < n and ok(hi):
                lo = hi
                step *= 2
                hi = min(n, p + step)
            pm._last_true(ok, lo, hi)
        else:
            hi, step = p, 1
            lo = p - step if p > step else 0
            while lo > 0 and not ok(lo):
                hi = lo
                step *= 2
                lo = p - step if p > step else 0
            pm._last_true(ok, lo, hi)
        self.memo[key] = cnt[0]
        return cnt[0]

    def probes(self, zf, zp, slots, virt):
        """== joint.score(f, fn, slots, virt) given zf = fn(fences), zp = fn(probe keys)."""
        n = self.n
        targets = [float(s) for s in slots]
        mdl = pm.LinearModel().fit_xy(self.f, zf, targets)
        a, b = mdl.slope, mdl.intercept
        tot = 0
        cost = self.cost
        truth = self.truth
        isf = math.isfinite
        if virt:
            m = n + virt - 1
            fm = float(m)
            br = bisect.bisect_right
            for q in range(len(zp)):
                y = a * zp[q] + b
                if not isf(y):
                    y = 0.0
                if y <= 0:
                    sl = 0
                elif y >= fm:
                    sl = m
                else:
                    sl = int(y)
                tot += cost(br(slots, sl) - 1, truth[q])
        else:
            fn1 = float(n - 1)
            for q in range(len(zp)):
                y = a * zp[q] + b
                if not isf(y):
                    y = 0.0
                if y <= 0:
                    p = 0
                elif y >= fn1:
                    p = n - 1
                else:
                    p = int(y)
                tot += cost(p, truth[q])
        return tot / self.NP


def slots_from_counts(c):
    s = [0]
    acc = 0
    for i, ci in enumerate(c):
        acc += ci
        s.append(i + 1 + acc)
    return s


class PoolState:
    def __init__(self, base, gaps=None, warp=None, counts=None):
        self.base = base                       # tb.State with f, xs
        self.gaps = dict(gaps or {})           # gap idx -> shrink factor
        self.warp = warp
        self.counts = list(counts) if counts is not None else [0] * (base.n - 1)

    def copy(self):
        return PoolState(self.base, self.gaps, self.warp, self.counts)

    def gapmap(self):
        if not self.gaps:
            return None
        idx = sorted(self.gaps)
        return tb.GapMap(self.base.xs, idx, [self.gaps[i] for i in idx])

    def virt(self):
        return sum(self.counts)

    def tb_state(self):
        v = self.virt()
        sl = slots_from_counts(self.counts) if v else None
        return self.base.copy(gaps=self.gapmap(), warp=self.warp, slots=sl, virt=v, log=[])


def features(ev, ps):
    st = ps.tb_state()
    fn = st.feature_fn()
    zp = [fn(q) for q in ev.pk]
    zf = zp[0::2]                              # fences sit at even positions of the probe keys
    return zf, zp


def strict_ok(ev, zp):
    pk = ev.pk
    for i in range(len(pk) - 1):
        if pk[i + 1] > pk[i]:
            if not (zp[i + 1] > zp[i]):
                return False
        elif zp[i + 1] != zp[i]:
            return False
    return True


def run_pool(ev, start, k, t_cost, budget, max_steps=100000, log_every=0):
    """The priced simultaneous greedy.  Returns (final PoolState, info)."""
    share = math.ceil(math.log2(k + 1)) / float(k)
    ps = start.copy()
    n = ev.n
    zf, zp = features(ev, ps)

    def objective(mp, p):
        return mp + len(p.gaps) * share + (t_cost if p.warp is not None else 0.0)

    v0 = ps.virt()
    slots = slots_from_counts(ps.counts)
    cur_mp = ev.probes(zf, zp, slots, v0)
    cur = objective(cur_mp, ps)
    steps = {"V": 0, "G": 0, "T": 0}
    trace = [("start", cur_mp, cur)]
    diags = []
    nstep = 0
    while nstep < max_steps:
        nstep += 1
        v = ps.virt()
        # ---- V candidates: cost 0, ratio +inf when they reduce -------------
        bestV = None
        if v < budget:
            bv = best_V(ev, zf, zp, slots, v, cur_mp)
            if bv is not None:
                bestV = (cur_mp - bv[1], bv[0], bv[1], bv[2])
        if bestV is not None:
            d, i, mp, sl = bestV
            ps.counts[i] += 1
            slots = sl
            cur_mp = mp
            cur = objective(mp, ps)
            steps["V"] += 1
            continue
        # ---- priced candidates: G_i and T ---------------------------------
        best = None                        # (ratio, d, kind, payload, mp, zf, zp)
        diag = {"best_G_d": None, "best_G_gap": None, "n_G_cands": 0, "T_d": None}
        if len(ps.gaps) < k:
            if ps.warp is None:
                xs = ps.base.xs
                W = [xs[i + 1] - xs[i] for i in range(n - 1)]
            else:
                W = [zf[i + 1] - zf[i] for i in range(n - 1)]
                for i, s in ps.gaps.items():
                    W[i] = W[i] / s
            Wmed = tb.median(W)
            for i in range(n - 1):
                if i in ps.gaps or not (W[i] > Wmed):
                    continue
                cand = ps.copy()
                cand.gaps[i] = Wmed / W[i]
                try:
                    czf, czp = features(ev, cand)
                except AssertionError:
                    continue
                if not strict_ok(ev, czp):
                    continue
                mp = ev.probes(czf, czp, slots, v)
                d = cur - objective(mp, cand)
                diag["n_G_cands"] += 1
                if diag["best_G_d"] is None or d > diag["best_G_d"]:
                    diag["best_G_d"], diag["best_G_gap"] = d, i
                if d > EPS:
                    r = d / share
                    if best is None or r > best[0]:
                        best = (r, d, "G", (i, Wmed / W[i]), mp, czf, czp)
        if ps.warp is None:
            u = ps.base.copy(gaps=ps.gapmap(), warp=None, slots=None, virt=0).u_at_fences()
            w = tb.transform_block(u, [float(s) for s in slots])
            if w is not None:
                cand = ps.copy()
                cand.warp = w
                czf, czp = features(ev, cand)
                if strict_ok(ev, czp):
                    mp = ev.probes(czf, czp, slots, v)
                    d = cur - objective(mp, cand)
                    diag["T_d"] = d
                    if d > EPS:
                        r = d / t_cost
                        if best is None or r > best[0]:
                            best = (r, d, "T", w, mp, czf, czp)
        diags.append(diag)
        if best is None:
            break
        _, d, kind, payload, mp, czf, czp = best
        if kind == "G":
            ps.gaps[payload[0]] = payload[1]
        else:
            ps.warp = payload
        zf, zp = czf, czp
        cur_mp = mp
        cur = objective(mp, ps)
        steps[kind] += 1
        trace.append((kind, cur_mp, cur, payload[0] if kind == "G" else None))
    trace.append(("end", cur_mp, cur))
    return ps, {"steps": steps, "pool_objective": cur, "share": share, "t_cost": t_cost,
                "trace_priced": [t for t in trace if t[0] != "V"], "fast_model_probes": cur_mp,
                "first_stall": diags[0] if diags else None, "final_stall": diags[-1] if diags else None}


def task_pool(d, mode, which):
    """which in {'empty', 'fromV'}; runs all k and both T prices."""
    t0 = time.time()
    base = tb.initial_state(d, mode)
    ev = Evaluator(base.f)
    n = base.n
    budget = int(tb.ALPHA * float(n))
    start = PoolState(base)
    if which == "fromV":
        sV = tb.apply_V(base)
        if sV.slots is not None:
            sl = sV.slots
            start.counts = [sl[i + 1] - sl[i] - 1 for i in range(n - 1)]
    # shared prefix: with zero-cost V having priority, the V phase until the
    # first stall does not depend on k or on the T price.  Run it once.
    vstart = v_phase(ev, start, budget)
    out = {"v_phase": {"virt": vstart.virt(), "seconds": time.time() - t0}}
    for t_name, t_cost in (("", tb.T_CHARGE), ("_pwl", tb.T_CHARGE_PWL)):
        for k in KS:
            t1 = time.time()
            ps, info = run_pool(ev, vstart, k, t_cost, budget)
            st = ps.tb_state()
            r = row(st, {"pool_info": info})
            assert abs(r["model_probes"] - info["fast_model_probes"]) < 1e-12, \
                ("fast evaluator drift", r["model_probes"], info["fast_model_probes"])
            r["seconds"] = time.time() - t1
            out["k%d%s" % (k, t_name)] = r
    out["seconds"] = time.time() - t0
    return out


def v_phase(ev, start, budget):
    """V-only greedy (ratio +inf moves) until no virtual fence reduces model probes."""
    ps = start.copy()
    n = ev.n
    zf, zp = features(ev, ps)
    slots = slots_from_counts(ps.counts)
    v = ps.virt()
    cur = ev.probes(zf, zp, slots, v)
    while v < budget:
        bv = best_V(ev, zf, zp, slots, v, cur)
        if bv is None:
            break
        ps.counts[bv[0]] += 1
        slots = bv[2]
        cur = bv[1]
        v += 1
    return ps


STATS = {"exact_fallback": 0}


def best_V(ev, zf, zp, slots, v, cur):
    """Exact argmax-reduction virtual fence, or None.  Screen all candidates
    fast; confirm the top 3 exactly; if any of them disagrees with its fast
    value, fall back to evaluating every candidate exactly."""
    sc = ev.scan_V(zf, zp, slots, v)
    sc.sort(key=lambda t: (t[0], t[1]))
    checked = []
    ok = True
    for fmp, i in sc[:3]:
        sl = slots[:i + 1] + [s + 1 for s in slots[i + 1:]]
        mp = ev.probes(zf, zp, sl, v + 1)
        if abs(mp - fmp) > 1e-12:
            ok = False
            break
        checked.append((i, mp, sl))
    if ok:
        if checked and cur - checked[0][1] > EPS:
            return checked[0]
        return None
    STATS["exact_fallback"] += 1
    best = None
    for i in range(ev.n - 1):
        sl = slots[:i + 1] + [s + 1 for s in slots[i + 1:]]
        mp = ev.probes(zf, zp, sl, v + 1)
        if cur - mp > EPS and (best is None or mp < best[1]):
            best = (i, mp, sl)
    return best


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_task(t):
    name = "_".join(str(x) for x in t)
    path = os.path.join(PARTS, name + ".json")
    if os.path.exists(path):
        return name, "cached"
    t0 = time.time()
    try:
        if t[0] == "indep":
            r = task_indep(t[1], t[2])
        elif t[0] == "k":
            r = task_k(t[1], t[2], t[3])
        else:
            r = task_pool(t[1], t[2], t[3])
    except Exception:
        r = {"error": traceback.format_exc()}
    with open(path + ".tmp", "w") as fh:
        json.dump(r, fh)
    os.replace(path + ".tmp", path)
    return name, "%.0fs%s" % (time.time() - t0, " ERROR" if "error" in r else "")


def all_tasks():
    tasks = []
    samples = [(d, m) for m in tb.MODES for d in tb.DATASETS]
    # slow ones first
    slow = {("osm", "uniform"), ("planet", "uniform")}
    samples.sort(key=lambda s: 0 if s in slow else 1)
    for d, m in samples:
        tasks.append(("pool", d, m, "empty"))
        tasks.append(("pool", d, m, "fromV"))
    for d, m in samples:
        tasks.append(("indep", d, m))
        for k in KS:
            tasks.append(("k", d, m, k))
    return tasks


if __name__ == "__main__":
    os.makedirs(PARTS, exist_ok=True)
    nproc = int(os.environ.get("NPROC", "8"))
    tasks = all_tasks()
    if len(sys.argv) > 1:
        sel = sys.argv[1:]
        tasks = [t for t in tasks if any(s == "_".join(str(x) for x in t) or s == t[1] for s in sel)]
    from multiprocessing import Pool
    with Pool(nproc) as p:
        for name, status in p.imap_unordered(run_task, tasks):
            print(time.strftime("%H:%M:%S"), name, status, flush=True)
