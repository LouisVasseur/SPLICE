#!/usr/bin/env python3
"""threeblock -- the ONE shared module both experimental arms import.

Three correctors at the ROOT of the SCALE-LI index, composed on one State:

  G  gap removal on the INPUT axis    u = g(x)       (k-entry table, charged)
  T  smooth monotone tanh-pair warp   z = f(u)       (charged 6.9 probe-eq.)
  V  CSV virtual fences, OUTPUT axis  targets = slot ranks (slot->region table)

The composed feature for a key is  z = f(g(x)),  x = (key - fences[0]) / span.
The arms differ ONLY in the order and repetition of apply_G / apply_T / apply_V.

Everything that touches last week's numbers is REUSED, not rewritten:
  * T block    = joint.WarpBasis(xs, joint.GAINS).fit(targets)   (least squares,
                 exact 3x3 solve inside a 64-gain grid, centre 0, C>=0, g>0)
  * T feature  = joint.make_flow_fn when G is off (bit-identical to last week)
  * V block    = joint.smooth_cdf_fast (bit-identical to probe_metric.smooth_cdf,
                 cross-checked here against verify/indep.smooth_cdf)
  * scorer     = joint.score -> probe_metric.root_probes (fit_root replica on
                 the 2n-1 fences + floor midpoints)
  * loss       = joint.ols_sse  (objective (*) of joint.py, over REAL fences)

G (new).  Gap i is the interval between fence i and fence i+1.  gap_block
ranks gaps by their width in the CURRENT coordinate, keeps the k largest, and
shrinks each, on the input axis, linearly inside the gap by factor
s_i = W_med / W_i  (W = current-coordinate width, W_med its median over all
gaps).  With T off this is literally "shrink to the median raw gap".  With T on
it is the first-order equivalent: gap i's z-width becomes ~ the median z-width.
Only gaps with W_i > W_med are shrunk (s_i < 1), so no gap ever widens and,
because s_i > 0, no two fences collide; strictness is asserted in double on
fences AND midpoints anyway.  For a gap that is already shrunk, its current
width is measured UNSHRUNK (z-width / s_i_old), otherwise re-selection would
oscillate (a shrunk gap would look small, be released, look big again ...).
G is a REPLACEMENT block: each call recomputes the whole gap set from the raw
input axis; it never stacks on a previous G.  With T off, G is idempotent.
With T on and the warp held fixed, G is NOT idempotent: G renormalises u to
[0,1] and moves every fence under the fixed nonlinear warp, so the
current-coordinate widths change and a re-selection can differ.  That is the
G<->T coupling itself; an arm that cycles should refit T after every G.

    g(x) = x - C_j - (1 - s_j) * min(x - a_j, w_j),   j = last gap with a_j <= x
    u    = g(x) / (1 - D),     D = total shrink        (renormalised to [0, 1])

The lookup cost of g is a binary search over the k gap starts a_j:
ceil(log2(k+1)) comparisons, charged as gap_table_probes in the same probe
currency as the root search.  The min() is arithmetic on the already-loaded
entry (a_j, w_j, s_j, C_j), not a probe.

ARM USAGE (the only thing an arm should vary is the block sequence):

    import threeblock as tb
    s = tb.initial_state("books", "window")      # fences, G/T/V all off
    s = tb.apply_G(s, 16)                        # replace gap set (select="current")
    s = tb.apply_T(s)                            # refit warp to current targets in g-coord
    s = tb.apply_V(s, alpha=4.0, guard=True)     # CSV greedy in f(g(x)), keep-previous guard
    r = tb.score(s)                              # model_probes, gap_table_probes, ...
    band = tb.perturbed_V_spread(s)              # chaos band of V at this state (slow)

Low-level blocks with the brief's signatures: load, gap_block, transform_block,
vfence_block, score(State).  Regression (G off) reproduces joint_results.json
bit-for-bit: see regress.py / regress_joint.py next to this file.

Standard library only.  Nothing under experimental/scaleli is written.
"""
from __future__ import annotations
import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))

import bisect
import math
import os
import sys

AIDB = REPO + "/experimental/scaleli/results/aidb_joint"
sys.path.insert(0, AIDB)
sys.path.insert(0, os.path.join(AIDB, "verify"))
import probe_metric as pm  # noqa: E402
import joint  # noqa: E402

SAMPLES = joint.SAMPLES
DATASETS = list(joint.DATASETS)
MODES = ["uniform", "window"]
REGION_KEYS = 4096
ALPHA = 4.0                      # deployed CSV budget: lambda = 4 x #fences
T_CHARGE = 6.9                   # 28.3 ns tanh pair ~ 6.9 probe-equivalents
T_CHARGE_PWL = 0.45              # piecewise-linear table alternative (reported only)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

_FENCE_CACHE = {}


def sample_path(dataset, mode):
    return os.path.join(SAMPLES, "%s_2M_%s_s42" % (dataset, mode))


def load(dataset, mode="uniform"):
    """-> fences (keys[0], keys[4096], ...) of the 2M-key sample, as ints."""
    key = (dataset, mode)
    if key not in _FENCE_CACHE:
        keys = pm.load_sample(sample_path(dataset, mode))
        assert all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1)), "unsorted sample"
        _FENCE_CACHE[key] = pm.fences(keys, REGION_KEYS)
    return list(_FENCE_CACHE[key])


def raw_xs(f):
    origin = f[0]
    span = max(1, f[-1] - origin)
    return [float(k - origin) / float(span) for k in f], origin, span


def median(v):
    s = sorted(v)
    m = len(s)
    return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])


# ---------------------------------------------------------------------------
# G block
# ---------------------------------------------------------------------------

class GapMap:
    """Piecewise-linear, strictly increasing, exactly invertible map on the
    normalised input axis.  Identity outside the selected gaps."""

    def __init__(self, xs, idx, factors):
        # idx: sorted fence indices i of the shrunk gaps (gap i = [xs[i], xs[i+1]])
        self.idx = list(idx)
        self.s = list(factors)
        self.a = [xs[i] for i in self.idx]                  # gap starts (raw x)
        self.w = [xs[i + 1] - xs[i] for i in self.idx]      # raw widths
        self.d = [self.w[j] * (1.0 - self.s[j]) for j in range(len(self.idx))]
        self.C = []                                         # shrink strictly below gap j
        c = 0.0
        for dj in self.d:
            self.C.append(c)
            c += dj
        self.D = c
        self.scale = 1.0 - self.D
        assert self.scale > 0.0, "total shrink swallowed the key range"
        self.factor_of = dict(zip(self.idx, self.s))

    @property
    def k(self):
        return len(self.idx)

    @property
    def table_probes(self):
        return float(math.ceil(math.log2(self.k + 1))) if self.k else 0.0

    @property
    def boundaries(self):
        return list(zip(self.a, [self.a[j] + self.w[j] for j in range(self.k)]))

    def __call__(self, x):
        j = bisect.bisect_right(self.a, x) - 1
        if j < 0:
            return x / self.scale
        t = x - self.a[j]
        if t > self.w[j]:
            t = self.w[j]
        return (x - self.C[j] - (1.0 - self.s[j]) * t) / self.scale

    def inverse(self, u):
        y = u * self.scale
        # images of gap starts are a_j - C_j, increasing in j
        starts = [self.a[j] - self.C[j] for j in range(self.k)]
        j = bisect.bisect_right(starts, y) - 1
        if j < 0:
            return y
        t = y - starts[j]
        sw = self.s[j] * self.w[j]
        if t <= sw:
            return self.a[j] + t / self.s[j]
        return y + self.C[j] + self.d[j]


def gap_block(xs, k, widths=None, factor_old=None):
    """Select the k largest gaps and build the shrink map on the INPUT axis.

    xs       raw normalised fence coordinates (the input axis G acts on).
    widths   per-gap width in the CURRENT coordinate (len n-1).  Default: the
             raw widths, i.e. G alone.  Pass the z-widths to make G see T.
    factor_old  {gap index: s} of the gap set being replaced, so that its
             gaps are measured unshrunk (z-width / s).
    Returns (boundaries, gmap, table_probes); gmap is a GapMap (callable g).
    """
    n = len(xs)
    if widths is None:
        widths = [xs[i + 1] - xs[i] for i in range(n - 1)]
    W = list(widths)
    if factor_old:
        for i, s in factor_old.items():
            W[i] = W[i] / s
    if k <= 0:
        gm = GapMap(xs, [], [])
        return gm.boundaries, gm, gm.table_probes
    Wmed = median(W)
    assert Wmed > 0.0, "median current gap is zero; cannot pick a typical width"
    order = sorted(range(n - 1), key=lambda i: (-W[i], i))[:k]
    sel = sorted(i for i in order if W[i] > Wmed)
    fac = [Wmed / W[i] for i in sel]
    gm = GapMap(xs, sel, fac)
    # no collisions: every shrunk raw width stays strictly positive in double
    for j, i in enumerate(sel):
        assert gm(xs[i + 1]) > gm(xs[i]), ("gap collapsed", i)
    return gm.boundaries, gm, gm.table_probes


# ---------------------------------------------------------------------------
# T block
# ---------------------------------------------------------------------------

_BASIS_CACHE = {}


class Warp:
    def __init__(self, units, sse):
        self.units = [tuple(u) for u in units]
        self.sse = sse

    def __call__(self, u):
        z = 0.0
        for (g, m, c) in self.units:
            z += c * math.tanh(g * (u - m))
        return z


def transform_block(xs, targets):
    """Least-squares fit of z = C1 tanh(g1 u) + C2 tanh(g2 u) (C>=0, g>0, centre 0,
    joint.GAINS grid) to `targets` over the fence coordinates `xs`.
    -> Warp (callable f) or None if no strictly increasing fit exists."""
    key = tuple(xs)
    basis = _BASIS_CACHE.get(key)
    if basis is None:
        basis = joint.WarpBasis(list(xs), joint.GAINS)
        if len(_BASIS_CACHE) > 64:
            _BASIS_CACHE.clear()
        _BASIS_CACHE[key] = basis
    fit = basis.fit([float(t) for t in targets])
    if fit is None:
        return None
    sse, units = fit
    return Warp(units, sse)


# ---------------------------------------------------------------------------
# V block
# ---------------------------------------------------------------------------

def vfence_block(feature_values, alpha=ALPHA, prev_slots=None):
    """CSV Algorithm 1 (greedy insertion with refit) on the fence features.
    Budget lambda = int(alpha * n).  If prev_slots is given, the
    keep-previous-slots guard of joint.one_budget is applied: when the slots
    already held give a lower loss (*) in the new feature, they are kept.
    -> dict(slots, virt, rounds, sse_before, sse_after, kept_previous)."""
    z = [float(v) for v in feature_values]
    n = len(z)
    assert all(z[i + 1] > z[i] for i in range(n - 1)), "V needs a strictly increasing feature"
    sm = joint.smooth_cdf_fast(z, alpha)
    slots = list(sm.slot)
    virt = len(sm.virtual_features)
    loss = joint.ols_sse(z, [float(t) for t in slots])
    kept = False
    if prev_slots is not None:
        lk = joint.ols_sse(z, [float(t) for t in prev_slots])
        if lk <= loss:
            slots = list(prev_slots)
            virt = slots[-1] + 1 - n
            loss = lk
            kept = True
    return {"slots": slots, "virt": virt, "rounds": sm.rounds,
            "sse_before": sm.sse_before, "sse_after": loss, "kept_previous": kept}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State:
    """(gap set, warp, slots) over one fence set.  Immutable-ish: the apply_*
    helpers return a NEW State, so arms can branch and compare freely."""

    def __init__(self, f, gaps=None, warp=None, slots=None, virt=0, log=None):
        self.f = f
        self.n = len(f)
        self.xs, self.origin, self.span = raw_xs(f)
        self.gaps = gaps if (gaps is not None and gaps.k > 0) else None
        self.warp = warp
        self.slots = list(slots) if slots is not None else None
        self.virt = virt if self.slots is not None else 0
        self.log = list(log or [])

    def copy(self, **kw):
        d = dict(gaps=self.gaps, warp=self.warp, slots=self.slots, virt=self.virt, log=self.log)
        d.update(kw)
        return State(self.f, **d)

    # --- coordinates -------------------------------------------------------
    def u_at_fences(self):
        """the coordinate T is fitted in: g(x) (or x when G is off)."""
        if self.gaps is None:
            return list(self.xs)
        return [self.gaps(x) for x in self.xs]

    def feature_fn(self):
        o, sp = self.origin, self.span
        if self.gaps is None and self.warp is None:
            return lambda k, _o=o, _s=float(sp): float(k - _o) / _s       # == joint raw_fn
        if self.gaps is None:
            return joint.make_flow_fn(self.warp.units, o, sp)             # == last week's flow
        gm = self.gaps
        if self.warp is None:
            return lambda k, _o=o, _s=float(sp), _g=gm: _g(float(k - _o) / _s)
        w = self.warp
        return lambda k, _o=o, _s=float(sp), _g=gm, _w=w: _w(_g(float(k - _o) / _s))

    def features(self):
        fn = self.feature_fn()
        return [fn(k) for k in self.f]

    def targets(self):
        if self.slots is None:
            return [float(j) for j in range(self.n)]
        return [float(s) for s in self.slots]

    def check_monotone(self):
        """strictly increasing on fences AND on the floor midpoints (the 2n-1
        probe keys); a tie is allowed only where the keys themselves tie."""
        fn = self.feature_fn()
        pk = pm.root_probe_keys(self.f)
        v = [fn(k) for k in pk]
        for i in range(len(pk) - 1):
            if pk[i + 1] > pk[i]:
                if not (v[i + 1] > v[i]):
                    return False, i
            elif v[i + 1] != v[i]:
                return False, i
        return True, None

    def assert_monotone(self):
        ok, i = self.check_monotone()
        assert ok, "composed feature not strictly increasing at probe key #%s" % i

    # --- current-coordinate gap widths, for G after T ------------------------
    def current_widths(self):
        z = self.features()
        return [z[i + 1] - z[i] for i in range(self.n - 1)]


def initial_state(dataset, mode="uniform"):
    return State(load(dataset, mode))


def apply_G(state, k, select="current"):
    """Replace the gap set.  select='current': rank/shrink by width in the
    current composed coordinate (sees T); 'input': by raw width (ignores T)."""
    if select == "input" or state.warp is None:
        widths = [state.xs[i + 1] - state.xs[i] for i in range(state.n - 1)]
        old = None
    else:
        widths = state.current_widths()
        old = state.gaps.factor_of if state.gaps is not None else None
    _, gm, _ = gap_block(state.xs, k, widths, old)
    # the warp was fitted in the old u-coordinate; keep it (it is a function of
    # u in [0,1]) -- the arm decides whether to refit T next.
    ns = state.copy(gaps=gm if gm.k else None, log=state.log + [("G", k, gm.k)])
    ok, _ = ns.check_monotone()
    if not ok:
        raise AssertionError("G made the composed feature non-strict")
    return ns


def apply_T(state):
    """Fit the warp to the CURRENT targets in the CURRENT G coordinate.
    Falls back to keeping the previous warp (or none) if the fit is not
    strictly increasing on fences + midpoints."""
    u = state.u_at_fences()
    w = transform_block(u, state.targets())
    if w is None:
        return state.copy(log=state.log + [("T", "nofit")])
    ns = state.copy(warp=w, log=state.log + [("T", "ok")])
    ok, _ = ns.check_monotone()
    if not ok:
        return state.copy(log=state.log + [("T", "non-strict-midpoint")])
    return ns


def apply_V(state, alpha=ALPHA, guard=True):
    z = state.features()
    r = vfence_block(z, alpha, state.slots if guard else None)
    slots = r["slots"] if r["virt"] else None
    return state.copy(slots=slots, virt=r["virt"],
                      log=state.log + [("V", r["virt"], r["rounds"], r["kept_previous"])])


def drop_V(state):
    return state.copy(slots=None, virt=0)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def score(state, check=True):
    if check:
        state.assert_monotone()
    fn = state.feature_fn()
    if state.slots is not None and state.virt:
        mp = joint.score(state.f, fn, state.slots, state.virt)
    else:
        mp = joint.score(state.f, fn)
    gp = state.gaps.table_probes if state.gaps is not None else 0.0
    tc = T_CHARGE if state.warp is not None else 0.0
    z = state.features()
    return {
        "model_probes": mp,
        "gap_table_probes": gp,
        "gap_k": state.gaps.k if state.gaps is not None else 0,
        "transform_charge_probes": tc,
        "total_uncharged": mp + gp,
        "total_charged": mp + gp + tc,
        "total_charged_pwl": mp + gp + (T_CHARGE_PWL if state.warp is not None else 0.0),
        "virtual_count": state.virt,
        "sse": joint.ols_sse(z, state.targets()),
        "binary": pm.binary_probes(state.n),
    }


def perturbed_V_spread(state, alpha=ALPHA, eps_list=(1e-6, 3.33e-5, 1e-4, 1e-3)):
    """Chaos band of the V block at this state, exactly as verify/bands.py
    defines it for csv_only: the composed feature is normalised to [0,1]
    (a no-op for the raw feature) and pushed through the physically
    meaningless monotone maps u -/+ eps*u^3 and u -/+ eps*u^2; V is re-run in
    the perturbed feature and the model is fitted and scored in it.  Returns
    {"base", "variants", "max_dev", "spread"}: max_dev = max |variant - base|,
    spread = max - min over all variants incl. identity (this is the
    definition behind last week's quoted bands, books 0.361 / wise 0.263,
    which used the verify/c4.py family eps in {1e-6, 3.33e-5, 1e-4, 1e-3}).  Use it to tell a
    real G/T effect from greedy noise."""
    fn = state.feature_fn()
    z0, z1 = fn(state.f[0]), fn(state.f[-1])
    if state.gaps is None and state.warp is None:
        nrm = fn                                        # bit-identical to bands.py's nrm
    else:
        nrm = lambda k, _fn=fn, _a=z0, _w=z1 - z0: (_fn(k) - _a) / _w

    def run(p):
        pf = lambda k, _p=p: _p(nrm(k))
        x = [pf(k) for k in state.f]
        sm = joint.smooth_cdf_fast(x, alpha)
        v = len(sm.virtual_features)
        return joint.score(state.f, pf, sm.slot, v) if v else joint.score(state.f, pf)

    base = run(lambda u: u)
    variants = {"identity": base}
    for eps in eps_list:
        for nm, p in (("-x^3", lambda u, e=eps: u - e * u * u * u), ("+x^3", lambda u, e=eps: u + e * u * u * u),
                      ("-x^2", lambda u, e=eps: u - e * u * u), ("+x^2", lambda u, e=eps: u + e * u * u)):
            variants["%s eps=%g" % (nm, eps)] = run(p)
    vals = list(variants.values())
    return {"base": base, "variants": variants,
            "max_dev": max(abs(v - base) for v in vals), "spread": max(vals) - min(vals)}


# ---------------------------------------------------------------------------
# gap distribution
# ---------------------------------------------------------------------------

def gap_stats(f):
    g = [f[i + 1] - f[i] for i in range(len(f) - 1)]
    rng = float(f[-1] - f[0])
    s = sorted(g, reverse=True)
    med = median(g)
    out = {"n_gaps": len(g), "median_gap": med, "max_over_median": s[0] / med if med else float("inf")}
    for t in (1, 4, 16, 64):
        out["top%d_frac" % t] = sum(s[:t]) / rng
    out["gaps_over_10x_median"] = sum(1 for v in g if v > 10 * med)
    return out
