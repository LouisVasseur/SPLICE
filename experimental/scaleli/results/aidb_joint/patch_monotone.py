"""Apply the monotone-BCD guard to joint.py's V block.

smooth_cdf cannot warm start (smoothing.hpp:54 always rebuilds from V = empty),
so round t+1 only guarantees L(f_t, V_{t+1}) <= L(f_t, empty), NOT
L(f_t, V_{t+1}) <= L(f_t, V_t).  The iteration can therefore go uphill.

The fix is cheap and exact: the objective depends on V ONLY through the integer
slot ranks s_i, which are combinatorial and carry over unchanged when f changes.
So at every round we may simply keep the previous slot ranks whenever the freshly
re-run greedy is worse in the objective.  Both options are realisable
configurations (the slot->region table is built from slot ranks alone), so the V
block becomes non-increasing and the whole block coordinate descent is monotone.
"""
import os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint.py")
s = open(p).read()

old = """        sm = smooth_cdf_fast(z, alpha)
        v = len(sm.virtual_features)
        slots = sm.slot if v else None
        s = [float(t) for t in sm.slot]
        loss_after_V = ols_sse(z, s)
        probes = score(f, flow, slots, v) if v else score(f, flow)
        if r == 1:
            seq_probes = probes
            out["seq_virtual"] = v
            out["seq_greedy_rounds"] = sm.rounds
        history.append({"round": r, "loss": loss_after_V, "probes": probes,
                        "virtual": v, "greedy_rounds": sm.rounds,
                        "units": [list(u) for u in units]})"""

new = """        sm = smooth_cdf_fast(z, alpha)
        v = len(sm.virtual_features)
        fresh = list(sm.slot)
        loss_fresh = ols_sse(z, [float(t) for t in fresh])
        kept = False
        if cur_slots is not None:
            loss_keep = ols_sse(z, [float(t) for t in cur_slots])
            if loss_keep <= loss_fresh:
                # MONOTONE GUARD: the greedy cannot warm start, so its fresh run
                # in the new z-space may be worse than the slot ranks we already
                # hold.  Slot ranks are feature-independent, so keeping them is a
                # legal configuration and makes the V block non-increasing.
                fresh = list(cur_slots)
                v = fresh[-1] + 1 - n
                loss_fresh = loss_keep
                kept = True
        cur_slots = fresh
        slots = fresh if v else None
        s = [float(t) for t in fresh]
        loss_after_V = loss_fresh
        probes = score(f, flow, slots, v) if v else score(f, flow)
        if r == 1:
            seq_probes = probes
            out["seq_virtual"] = v
            out["seq_greedy_rounds"] = sm.rounds
        history.append({"round": r, "loss": loss_after_V, "probes": probes,
                        "virtual": v, "greedy_rounds": sm.rounds,
                        "kept_previous_slots": kept,
                        "units": [list(u) for u in units]})"""

assert old in s, "anchor not found"
s = s.replace(old, new, 1)

old2 = """    units = list(units0)
    flow = flow0
    history = []"""
new2 = """    units = list(units0)
    flow = flow0
    cur_slots = None
    history = []"""
assert old2 in s
s = s.replace(old2, new2, 1)
open(p, "w").write(s)
print("patched")
