import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
"""Faithful Python port of the ROOT probe metric used by scaleli's C++ index.

Ported from (read-only, never modified):
  include/scaleli/index.hpp   -- Index::fit_root, Index::locate_region,
                                 Index::locate_from_prediction, Index::last_true
  include/scaleli/smoothing.hpp -- smooth_cdf, detail::Sums
  include/scaleli/model.hpp   -- LinearModel::fit / fit_xy / normalized / predict_x

Everything here is standard library only.

--------------------------------------------------------------------------
WHAT THE ROOT DOES (index.hpp)
--------------------------------------------------------------------------
Regions are cut every `region_keys` records of the sorted key array, so with a
2M-key sample and region_keys=4096 there are ceil(2e6/4096) = 489 regions.

Region j's `low_fence` is keys[j*4096] for j >= 1, and 0 (a SENTINEL) for j = 0.
fit_root() does NOT fit on the sentinel: it uses

    fences[0]   = regions_[0]->blocks.front().first   == keys[0]
    fences[j]   = regions_[j]->low_fence              == keys[j*4096]

The *predicate* used while locating, however, still reads `low_fence`, so index
0 always compares true.  Since every probe key is >= keys[0] this is identical
to comparing against fences[0], and `_ok` below encodes it either way.

A "probe" is exactly one evaluation of the comparison
`regions_[i]->low_fence <= k` (QueryStats::root_probes is incremented inside the
`ok` lambda and nowhere else at the root).  The linear model evaluation, the
flow evaluation and the slot->region table lookup are NOT counted as probes; the
flow is charged separately, as a flat `flow_cost` probe-equivalents per lookup,
only inside the candidate score.

--------------------------------------------------------------------------
TWO DIFFERENT NUMBERS.  DO NOT CONFLATE THEM.
--------------------------------------------------------------------------
(a) `root_probes(...)` -- fit_root's CANDIDATE SCORE.  In-sample, over exactly
    the 2n-1 fences and fence midpoints, plus flow_cost.  This is the objective
    the C++ actually selects on: it picks the cheapest of {raw, flow} x {ranks,
    virtual slots} and only installs the model if that beats the binary
    baseline.  A joint optimiser must minimise THIS.
    Verified against this workspace's "estimated probes" table for the plain
    linear root, all ten datasets:
       port -> fb 2.266 history 3.781 covid 4.642 wise 5.745 libio 6.520
               genome 8.635 books 11.076 osm 11.668 planet 12.595
       quoted  fb 2.3   history 3.8   covid 4.6   wise 5.7   libio 6.5
               genome 8.6   books 11.1   osm 11.7   planet 12.6

(b) `workload_probes(...)` -- QueryStats::root_probes averaged over real query
    keys drawn from the dataset.  This is what the benchmark reports and what
    this workspace's "measured root probes" table holds.  It is systematically
    a little LOWER than (a) on well-fitted roots, because the midpoints of wide
    fence gaps are probe keys that no real query ever asks for.
    Verified with root_alpha = 4 (CSV virtual fences, no flow):
       port ->  fb 2.079   books 2.600   osm 8.189
       quoted   fb 2.08    books 2.60    osm 8.18
    (the same configuration scores 2.224 / 2.728 / 8.181 under (a))
"""

import math
import os
import struct
from array import array

__all__ = [
    "load_sample",
    "fences",
    "root_probe_keys",
    "root_probes",
    "binary_probes",
    "workload_probes",
    "workload_binary_probes",
    "locate_from_prediction",
    "Sums",
    "DDSums",
    "LinearModel",
    "SmoothingResult",
    "smooth_cdf",
    "slot_to_region_table",
]


# ---------------------------------------------------------------------------
# 0. data loading
# ---------------------------------------------------------------------------

def load_sample(path):
    """Load a SOSD-format sample: 8-byte LE uint64 count, then that many LE
    uint64 keys.  Returns a plain Python list of ints, already sorted (the
    samples on disk are sorted; we do not re-sort, we only sanity check)."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        if size < 8 + 8 * n:
            raise ValueError("truncated sample: header says %d keys, file holds %d"
                             % (n, (size - 8) // 8))
        a = array("Q")
        a.fromfile(fh, n)
    if struct.pack("=Q", 1) != struct.pack("<Q", 1):  # big-endian host
        a.byteswap()
    keys = a.tolist()
    return keys


# ---------------------------------------------------------------------------
# 1. region fences
# ---------------------------------------------------------------------------

def fences(keys, region_keys=4096):
    """The keys the root model is fitted over.

    Index::bulk_load cuts a region at every multiple of region_keys, so the
    fence list is keys[0], keys[region_keys], keys[2*region_keys], ...
    (index.hpp:432 for the cut, index.hpp:370 for the fence extraction).
    """
    return [keys[i] for i in range(0, len(keys), region_keys)]


def root_probe_keys(f):
    """The 2n-1 keys fit_root scores a candidate on (index.hpp:371-372):

        for j in 0..n-1:  push fences[j]
                          if j+1 < n: push fences[j] + (fences[j+1]-fences[j])/2

    i.e. every fence, interleaved with every fence midpoint (uint64 division,
    so floor).  Order is [f0, m0, f1, m1, ..., f_{n-2}, m_{n-2}, f_{n-1}].
    """
    n = len(f)
    out = []
    for j in range(n):
        out.append(f[j])
        if j + 1 < n:
            out.append(f[j] + (f[j + 1] - f[j]) // 2)
    return out


# ---------------------------------------------------------------------------
# 2. the locate routine, with exact probe accounting
# ---------------------------------------------------------------------------

def _last_true(ok, lo, hi):
    """Index::last_true (index.hpp:337-339).  Precondition ok(lo) is true and
    every i >= hi is false (or hi == n).  Returns the last true index."""
    while hi - lo > 1:
        m = lo + (hi - lo) // 2
        if ok(m):
            lo = m
        else:
            hi = m
    return lo


def locate_from_prediction(ok, y, n, table=None):
    """Index::locate_from_prediction (index.hpp:340-347).

    `ok` is a unary predicate on a region index; every call to it is one probe.
    `y` is the raw model output (rank space without virtual fences, slot space
    with them).  `table` is root_slot_to_region_ (None/empty when no virtual
    fences).  Returns the region index.
    """
    if table:
        m = len(table) - 1
        if y <= 0:
            slot = 0
        elif y >= float(m):
            slot = m
        else:
            slot = int(y)          # std::size_t(y): truncation; y > 0 here
        p = table[slot]
    else:
        if y <= 0:
            p = 0
        elif y >= float(n - 1):
            p = n - 1
        else:
            p = int(y)

    if ok(p):
        # exponential walk to the RIGHT, then binary correction
        lo, step, hi = p, 1, p + 1
        while hi < n and ok(hi):      # short-circuit: ok not called when hi == n
            lo = hi
            step *= 2                 # NOTE: the root does NOT cap step at n
            hi = min(n, p + step)     # (Region::locate_block does: min(n, step*2))
        return _last_true(ok, lo, hi)

    # exponential walk to the LEFT, then binary correction
    hi, step = p, 1
    lo = p - step if p > step else 0
    while lo > 0 and not ok(lo):      # short-circuit: ok not called when lo == 0
        hi = lo
        step *= 2
        lo = p - step if p > step else 0
    return _last_true(ok, lo, hi)


def _ok_factory(f, k, counter):
    """The root predicate `regions_[i]->low_fence <= k`, with region 0's
    sentinel fence of 0.  Each call is one probe."""
    def ok(i):
        counter[0] += 1
        return True if i == 0 else f[i] <= k
    return ok


def root_probes(f, predict_fn, table=None, flow_cost=0.0):
    """The EXACT candidate score fit_root computes (index.hpp:389-392):

        count = 0
        for k in probes:                      # the 2n-1 fences + midpoints
            ok = lambda i: (++count, low_fence[i] <= k)
            locate_from_prediction(ok, model.predict_x(feature(k)), n, table)
        cost = count / len(probes) + (flow ? flow_cost : 0)

    `predict_fn(key) -> float` must already fold in BOTH the feature map and the
    affine model, i.e. it returns `model.predict_x(feature(k))`:
      * raw  candidate: slope * ((k - fences[0]) / span) + intercept,
                        span = max(1, fences[-1] - fences[0])
      * flow candidate: slope * flow(k) + intercept
    `table` is the slot->region table when virtual fences are used, else None.
    `flow_cost` is added flat (default Config::flow_cost is 4.0); pass 0 for a
    raw-feature candidate.
    """
    n = len(f)
    probes = root_probe_keys(f)
    counter = [0]
    for k in probes:
        ok = _ok_factory(f, k, counter)
        locate_from_prediction(ok, predict_fn(k), n, table)
    return counter[0] / float(len(probes)) + flow_cost


def workload_probes(f, predict_fn, query_keys, table=None, flow_cost=0.0):
    """QueryStats::root_probes per lookup over an actual query set
    (Index::locate_region, index.hpp:348-357).  Identical machinery to
    root_probes, but the probe keys are the caller's queries instead of the
    fences and midpoints.  This is the "measured" number the benchmark prints.
    """
    n = len(f)
    counter = [0]
    for k in query_keys:
        ok = _ok_factory(f, k, counter)
        locate_from_prediction(ok, predict_fn(k), n, table)
    return counter[0] / float(len(query_keys)) + flow_cost


def workload_binary_probes(f, query_keys):
    """The Root::Binary / !root_ready_ fallback path of locate_region over an
    actual query set: a plain lower_bound over low_fence, one probe per
    iteration."""
    n = len(f)
    total = 0
    for k in query_keys:
        lo, hi = 0, n
        while lo < hi:
            m = lo + (hi - lo) // 2
            total += 1
            if m == 0 or f[m] <= k:
                lo = m + 1
            else:
                hi = m
    return total / float(len(query_keys))


def binary_probes(n):
    """The binary-search baseline fit_root compares against (index.hpp:397):

        for k in probes:
            lo, hi = 0, n
            while lo < hi:
                m = lo + (hi-lo)//2; ++count
                if low_fence[m] <= k: lo = m+1 else: hi = m
        cost = count / len(probes)

    Depends only on n: every probe at fence j and every midpoint after fence j
    terminates at lo = j+1, so the answer multiset is
    {1,1, 2,2, ..., n-1,n-1, n} over the 2n-1 probes.
    """
    if n < 1:
        return 0.0

    def iters(target_lo):
        lo, hi, c = 0, n, 0
        while lo < hi:
            m = lo + (hi - lo) // 2
            c += 1
            if m < target_lo:      # low_fence[m] <= k  <=>  m <= j  <=>  m < lo*
                lo = m + 1
            else:
                hi = m
        return c

    total = 0
    probes = 0
    for j in range(n):
        total += iters(j + 1)
        probes += 1
        if j + 1 < n:
            total += iters(j + 1)
            probes += 1
    return total / float(probes)


# ---------------------------------------------------------------------------
# 3. least squares, exactly as model.hpp does it
# ---------------------------------------------------------------------------

class LinearModel:
    """model.hpp LinearModel.  Welford-style accumulation; slope 0 when the
    feature has no variance; predict_x returns 0 for non-finite output."""

    __slots__ = ("origin", "span", "slope", "intercept")

    def __init__(self):
        self.origin = 0
        self.span = 1
        self.slope = 0.0
        self.intercept = 0.0

    def normalized(self, k):
        if k >= self.origin:
            return float(k - self.origin) / float(self.span)
        return -float(self.origin - k) / float(self.span)

    def predict_x(self, x):
        y = self.slope * x + self.intercept       # std::fma in C++
        return y if math.isfinite(y) else 0.0

    def predict(self, k):
        return self.predict_x(self.normalized(k))

    def fit_xy(self, keys, x, targets):
        self.origin = 0
        self.span = 1
        self.slope = 0.0
        self.intercept = 0.0
        if not keys:
            return self
        self.origin = keys[0]
        self.span = max(1, keys[-1] - self.origin)
        mx = my = xx = xy = 0.0
        for i in range(len(x)):
            xi = float(x[i])
            y = float(targets[i])
            nn = float(i + 1)
            dx = xi - mx
            dy = y - my
            mx += dx / nn
            my += dy / nn
            xx += dx * (xi - mx)
            xy += dx * (y - my)
        self.slope = (xy / xx) if xx > 0 else 0.0
        self.intercept = my - self.slope * mx
        return self

    def fit(self, keys, targets):
        """LinearModel::fit -- the normalized-key fit.  fit_root uses this only
        to obtain origin/span for the raw feature (`tmp` at index.hpp:374)."""
        if not keys:
            return self.fit_xy(keys, [], [])
        origin = keys[0]
        span = max(1, keys[-1] - origin)
        x = [float(k - origin) / float(span) for k in keys]
        self.fit_xy(keys, x, targets)
        self.origin = origin
        self.span = span
        return self


# ---------------------------------------------------------------------------
# 4. CSV smoothing, exactly as smoothing.hpp does it
# ---------------------------------------------------------------------------

class Sums:
    """smoothing.hpp detail::Sums.  C++ accumulates in long double; Python has
    only double, so SSE values can differ in the last few digits."""

    __slots__ = ("n", "sx", "sxx", "sy", "syy", "sxy")

    def __init__(self, other=None):
        if other is None:
            self.n = self.sx = self.sxx = self.sy = self.syy = self.sxy = 0.0
        else:
            self.n, self.sx, self.sxx = other.n, other.sx, other.sxx
            self.sy, self.syy, self.sxy = other.sy, other.syy, other.sxy

    def add(self, x, y):
        self.n += 1.0
        self.sx += x
        self.sxx += x * x
        self.sy += y
        self.syy += y * y
        self.sxy += x * y

    def shift(self, dsy, dsyy, dsxy):
        """Apply the target-shift correction of loss_at (smoothing.hpp:65-70)."""
        self.sy += dsy
        self.syy += dsyy
        self.sxy += dsxy

    def sse(self):
        if self.n < 2:
            return 0.0
        cxx = self.sxx - self.sx * self.sx / self.n
        cxy = self.sxy - self.sx * self.sy / self.n
        cyy = self.syy - self.sy * self.sy / self.n
        s = (cyy - cxy * cxy / cxx) if cxx > 0 else cyy
        return s if s > 0 else 0.0


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


class DDSums:
    """Same interface as Sums, but each running sum is a double-double pair
    (~32 significant digits).  The C++ accumulates in `long double`; on x86 that
    is 64 mantissa bits, which Python's float (53 bits) cannot match.  That
    matters here because smooth_cdf's stopping rule compares candidates at a
    RELATIVE tolerance of 1e-12, which is finer than the cancellation noise of
    the double closed form (see the module notes).  Use DDSums when the fence
    set is close to linear and plain doubles stop the greedy insertion early."""

    __slots__ = ("n", "sx", "sxx", "sy", "syy", "sxy")

    def __init__(self, other=None):
        if other is None:
            self.n = 0.0
            self.sx = self.sxx = self.sy = self.syy = self.sxy = (0.0, 0.0)
        else:
            self.n, self.sx, self.sxx = other.n, other.sx, other.sxx
            self.sy, self.syy, self.sxy = other.sy, other.syy, other.sxy

    @staticmethod
    def _acc(p, x):
        s, e = _two_sum(p[0], x)
        e += p[1]
        return _two_sum(s, e)

    def add(self, x, y):
        self.n += 1.0
        self.sx = DDSums._acc(self.sx, x)
        self.sxx = DDSums._acc(self.sxx, x * x)
        self.sy = DDSums._acc(self.sy, y)
        self.syy = DDSums._acc(self.syy, y * y)
        self.sxy = DDSums._acc(self.sxy, x * y)

    def shift(self, dsy, dsyy, dsxy):
        self.sy = DDSums._acc(self.sy, dsy)
        self.syy = DDSums._acc(self.syy, dsyy)
        self.sxy = DDSums._acc(self.sxy, dsxy)

    def sse(self):
        if self.n < 2:
            return 0.0
        n = self.n
        sx, sxx = self.sx[0] + self.sx[1], self.sxx[0] + self.sxx[1]
        sy, syy = self.sy[0] + self.sy[1], self.syy[0] + self.syy[1]
        sxy = self.sxy[0] + self.sxy[1]
        cxx = sxx - sx * sx / n
        cxy = sxy - sx * sy / n
        cyy = syy - sy * sy / n
        s = (cyy - cxy * cxy / cxx) if cxx > 0 else cyy
        return s if s > 0 else 0.0


class SmoothingResult:
    __slots__ = ("slot", "virtual_features", "sse_before", "sse_after", "rounds")

    def __init__(self, n):
        self.slot = list(range(n))
        self.virtual_features = []
        self.sse_before = 0.0
        self.sse_after = 0.0
        self.rounds = 0


def smooth_cdf(x, alpha, max_ternary_steps=40, precision="double"):
    """smoothing.hpp smooth_cdf -- CSV Algorithm 1.

    LOSS: the OLS residual SSE of the AUGMENTED set {(feature, sequence
    position)} over real AND virtual points, in closed form from running sums
    (Sums.sse above).  Targets are sequence positions, so inserting a virtual
    point after position i shifts EVERY later target from t to t+1:
        sy  += cnt                       cnt = len(seq) - (i+1)
        syy += 2*suf_y[i+1] + cnt        (sum of (t+1)^2 - t^2 = 2t+1)
        sxy += suf_x[i+1]                (sum of x_t*(t+1) - x_t*t = x_t)
    then the new point (xv, i+1) is added.  Each candidate is therefore O(1).

    BUDGET: floor(alpha * n) insertions; stop early when no gap lowers the loss
    (Algorithm 1 line 27).  Within a gap, the loss is sampled just inside both
    ends (e = 1e-3 * gap width); if decreasing on the left AND increasing on the
    right, a ternary search finds the interior minimum; otherwise the better end
    is used.  Gaps with x[i+1] <= x[i] are skipped.

    RETURNS SmoothingResult with:
        slot[i]          -- slot rank of real key i = i + (#virtual points
                            before it).  slot[0] is always 0: insertions happen
                            at sequence index best_i+1 >= 1, never before 0.
        virtual_features -- feature value of each inserted virtual point, in
                            insertion order (NOT sorted).
        sse_before / sse_after / rounds.
    Virtual points are never stored as records; they only move rank targets.
    """
    if not (alpha >= 0) or alpha >= 64 or not math.isfinite(alpha):
        raise ValueError("smoothing alpha must be in [0, 64)")
    acc = DDSums if precision == "dd" else Sums
    n = len(x)
    out = SmoothingResult(n)

    seq_x = [float(v) for v in x]
    seq_v = [False] * n

    def total():
        s = acc()
        for i in range(len(seq_x)):
            s.add(seq_x[i], float(i))
        return s

    base = total()
    out.sse_before = base.sse()
    out.sse_after = out.sse_before
    budget = int(alpha * float(n))
    if n < 3 or not budget:
        return out

    suf_x = [0.0] * (len(seq_x) + 1)
    suf_y = [0.0] * (len(seq_x) + 1)

    def rebuild_suffix():
        m = len(seq_x)
        del suf_x[:]
        del suf_y[:]
        suf_x.extend([0.0] * (m + 1))
        suf_y.extend([0.0] * (m + 1))
        for i in range(m - 1, -1, -1):
            suf_x[i] = suf_x[i + 1] + seq_x[i]
            suf_y[i] = suf_y[i + 1] + float(i)

    def loss_at(i, xv):
        s = acc(base)
        cnt = float(len(seq_x) - (i + 1))
        s.shift(cnt, 2.0 * suf_y[i + 1] + cnt, suf_x[i + 1])
        s.add(xv, float(i + 1))
        return s.sse()

    current = base.sse()
    for _ in range(budget):
        rebuild_suffix()
        best = current
        best_i = 0
        best_x = 0.0
        found = False
        for i in range(len(seq_x) - 1):
            lo = seq_x[i]
            hi = seq_x[i + 1]
            if not (hi > lo):
                continue
            w = hi - lo
            e = w * 1e-3
            l0 = loss_at(i, lo + e)
            l1 = loss_at(i, lo + 2 * e)
            r1 = loss_at(i, hi - 2 * e)
            r0 = loss_at(i, hi - e)
            if l1 < l0 and r1 < r0:
                a, b = lo + e, hi - e
                it = 0
                while it < max_ternary_steps and b - a > e:
                    m1 = a + (b - a) / 3.0
                    m2 = b - (b - a) / 3.0
                    if loss_at(i, m1) < loss_at(i, m2):
                        b = m2
                    else:
                        a = m1
                    it += 1
                cx = (a + b) / 2.0
                cl = loss_at(i, cx)
            elif l0 <= r0:
                cx, cl = lo + e, l0
            else:
                cx, cl = hi - e, r0
            if cl < best - 1e-12 * max(1.0, best):
                best, best_i, best_x, found = cl, i, cx, True
        if not found:
            break
        seq_x.insert(best_i + 1, best_x)
        seq_v.insert(best_i + 1, True)
        out.virtual_features.append(best_x)
        base = total()
        current = base.sse()
        out.rounds += 1

    k = 0
    for i in range(len(seq_x)):
        if not seq_v[i]:
            out.slot[k] = i
            k += 1
    out.sse_after = current
    return out


def slot_to_region_table(slot, n, virt):
    """root_slot_to_region_ (index.hpp:386-387).  table[s] = the region j whose
    slot interval [slot[j], slot[j+1]) contains s; virtual slots between fence j
    and fence j+1 therefore map to region j."""
    table = [0] * (n + virt)
    for j in range(n):
        end = slot[j + 1] if j + 1 < n else n + virt
        for s in range(slot[j], end):
            table[s] = j
    return table


# ---------------------------------------------------------------------------
# 5. convenience: build the four fit_root candidates
# ---------------------------------------------------------------------------

def raw_features(f):
    """The raw root feature: tmp.normalized(fence), tmp fitted on the fences.
    normalized(k) = (k - fences[0]) / max(1, fences[-1] - fences[0])."""
    origin = f[0]
    span = max(1, f[-1] - origin)
    return [float(k - origin) / float(span) for k in f], origin, span


def candidate_cost(f, feature_fn, root_alpha=0.0, flow_cost=0.0, precision="double"):
    """Score one fit_root candidate end to end.

    feature_fn(key) -> float is the feature map (raw normalization or the flow).
    Returns (cost, n_virtual).  With root_alpha > 0 the CSV virtual-fence
    variant is built; if smooth_cdf inserts nothing the candidate is skipped in
    C++ (here we fall back to the no-vp variant and report n_virtual = 0).
    """
    n = len(f)
    x = [feature_fn(k) for k in f]
    targets = [float(j) for j in range(n)]
    table = None
    virt = 0
    if root_alpha > 0 and n > 2:
        if all(x[i] <= x[i + 1] for i in range(n - 1)):
            sm = smooth_cdf(x, root_alpha, precision=precision)
            virt = len(sm.virtual_features)
            if virt:
                targets = [float(s) for s in sm.slot]
                table = slot_to_region_table(sm.slot, n, virt)
    model = LinearModel().fit_xy(f, x, targets)
    cost = root_probes(f, lambda k: model.predict_x(feature_fn(k)), table, flow_cost)
    return cost, virt


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import random

    # ---- 1. the binary baseline the fit_root selection compares against -----
    N = 489                                   # ceil(2_000_000 / 4096)
    b = binary_probes(N)
    print("binary_probes(%d) = %.6f   (workspace baseline: 8.96)" % (N, b))
    assert abs(b - 8.96) < 0.005, b

    # ---- 2. structural checks on a synthetic, exactly linear fence set ------
    f = [1000 * j for j in range(N)]
    probes = root_probe_keys(f)
    assert len(probes) == 2 * N - 1
    # A perfect prediction costs 2 probes (ok(p) true, ok(p+1) false); the last
    # region costs 1, because hi == n short-circuits the `hi < n && ok(hi)`.
    oracle = root_probes(f, lambda k: float(min(N - 1, k // 1000)))
    assert abs(oracle - (2 * (2 * N - 2) + 1) / float(2 * N - 1)) < 1e-12
    print("floor of the metric (perfect prediction): %.6f probes" % oracle)
    x, origin, span = raw_features(f)
    m = LinearModel().fit_xy(f, x, [float(j) for j in range(N)])
    lin = root_probes(f, lambda k: m.predict_x(float(k - origin) / float(span)))
    assert abs(lin - oracle) < 1e-12
    print("raw linear root on a linear fence set:    %.6f probes" % lin)

    sm = smooth_cdf([float(i) for i in range(100)], 1.0)
    assert not sm.virtual_features and sm.slot == list(range(100))
    print("smooth_cdf on a perfect line: 0 virtual points (nothing to gain)")

    # ---- 3. real data: reproduce the workspace's linear-root table ----------
    BASE = (REPO + "/experimental/scaleli"
            "/data/samples/%s_2M_uniform_s42")
    QUOTED_LINEAR = {"fb": 2.3, "history": 3.8, "covid": 4.6, "wise": 5.7,
                     "libio": 6.5, "genome": 8.6, "books": 11.1,
                     "osm": 11.7, "planet": 12.6}
    if os.path.isdir(os.path.dirname(BASE % "fb")):
        print()
        print("%-9s %5s %9s %9s" % ("dataset", "n", "linear", "quoted"))
        random.seed(42)
        for d in ["fb", "history", "stack", "covid", "wise", "libio",
                  "genome", "books", "osm", "planet"]:
            path = BASE % d
            if not os.path.exists(path):
                continue
            keys = load_sample(path)
            fv = fences(keys)
            xv, o, sp = raw_features(fv)
            mm = LinearModel().fit_xy(fv, xv, [float(j) for j in range(len(fv))])
            c = root_probes(fv, lambda k, mm=mm, o=o, sp=sp:
                            mm.predict_x(float(k - o) / float(sp)))
            q = QUOTED_LINEAR.get(d)
            print("%-9s %5d %9.3f %9s" % (d, len(fv), c, q if q else "-"))
            if q is not None:
                assert abs(c - q) < 0.06, (d, c, q)
        print()
        print("in-sample score vs workload probes, CSV virtual fences alpha=4:")
        QUOTED_VP = {"fb": 2.08, "books": 2.60}
        for d in ["fb", "books"]:
            keys = load_sample(BASE % d)
            fv = fences(keys)
            n = len(fv)
            xs, o, sp = raw_features(fv)
            feat = lambda k, o=o, sp=sp: float(k - o) / float(sp)
            s2 = smooth_cdf(xs, 4.0)
            v = len(s2.virtual_features)
            tab = slot_to_region_table(s2.slot, n, v) if v else None
            mm = LinearModel().fit_xy(fv, xs, [float(t) for t in s2.slot])
            pf = lambda k, mm=mm, feat=feat: mm.predict_x(feat(k))
            qs = [keys[i] for i in random.sample(range(len(keys)), 200000)]
            print("  %-6s virt=%5d  in-sample=%.3f  workload=%.3f  quoted(measured)=%.2f"
                  % (d, v, root_probes(fv, pf, tab),
                     workload_probes(fv, pf, qs, tab), QUOTED_VP[d]))
    print()
    print("OK")
