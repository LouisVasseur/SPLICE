"""Independent oracles for scaleli_hardness: SOSD I/O, exact least squares, LIPP FMCD, flow transform."""
import struct, math, sys, json
from fractions import Fraction
from decimal import Decimal, getcontext
getcontext().prec = 60

def read_sosd(path, width=8, limit=0):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if limit: n = min(n, limit)
        fmt = '<%d%s' % (n, 'Q' if width == 8 else 'I')
        return list(struct.unpack(fmt, f.read(n * width)))

def write_sosd(path, keys, width=8):
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(keys)))
        f.write(struct.pack('<%d%s' % (len(keys), 'Q' if width == 8 else 'I'), *keys))

def exact_ls(keys):
    """Exact rational OLS of rank i on key x_i. Works for ints and floats (floats are exact rationals).
    Returns (slope, intercept, rmse(float), max_error(float)) with slope/intercept as Fractions."""
    n = len(keys)
    if all(isinstance(k, int) for k in keys):
        xs = keys
        sx = sum(xs); sxx = sum(x * x for x in xs); sxy = sum(i * x for i, x in enumerate(xs))
        sy = n * (n - 1) // 2
        den = n * sxx - sx * sx
        num = n * sxy - sx * sy
        # slope = num/den ; intercept = (sy - slope*sx)/n = (sy*den - num*sx)/(n*den)
        Dn = n * den
        if den == 0:
            return Fraction(0), Fraction(sy, n), 0.0, 0.0
        ic_num = sy * den - num * sx  # intercept numerator over Dn
        ss = 0; worst = 0
        for i, x in enumerate(xs):
            E = i * Dn - (n * num * x + ic_num)  # e_i * Dn
            ss += E * E
            if abs(E) > worst: worst = abs(E)
        rmse = float((Decimal(ss) / (Decimal(Dn) ** 2 * n)).sqrt())
        me = float(Fraction(worst, Dn))
        return Fraction(num, den), Fraction(ic_num, Dn), rmse, me
    xs = [Fraction(k) for k in keys]
    sx = sum(xs); sxx = sum(x * x for x in xs); sxy = sum(i * x for i, x in enumerate(xs))
    sy = Fraction(n * (n - 1), 2)
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den if den else Fraction(0)
    intercept = (sy - slope * sx) / n
    ss = Fraction(0); worst = Fraction(0)
    for i, x in enumerate(xs):
        e = i - (slope * x + intercept)
        ss += e * e
        if abs(e) > worst: worst = abs(e)
    r = ss / n
    rmse = float((Decimal(r.numerator) / Decimal(r.denominator)).sqrt())
    return slope, intercept, rmse, float(worst)

def gap_count(size):
    return 1 if size >= 1000000 else 2 if size >= 100000 else 5

def fmcd(keys, ut_eps=1e-6, mode='exact'):
    """LIPP build_tree_bulk_fmcd root fit, lipp.h @fe6ca49 (int size, L = size*(GAP+1), BUILD_LR_REMAIN=0).
    mode='exact': Ut from exact rational difference then rounded to double; positions with exact rational
    arithmetic on the double a and rational b. mode='float': plain float emulation (a*float(key)+b).
    Returns dict(a, b, L, D, Ut, fallback, pos: list of predicted item index, cd: conflict degree)."""
    size = len(keys); GAP = gap_count(size); L = size * (GAP + 1)
    if size < 3: return dict(L=L, cd=1 if size else 0, degenerate=True)
    isint = isinstance(keys[0], int)
    def ut_for(D):
        d = keys[size - 1 - D] - keys[D]
        if mode == 'exact':
            return float(Fraction(d) / (L - 2) + Fraction(ut_eps))
        return float(d) / float(L - 2) + ut_eps
    def ge(i, D, Ut):
        d = keys[i + D] - keys[i]
        return float(d) >= Ut  # LIPP: T difference (exact for uint64), converted to double for >= double
    i = 0; D = 1; Ut = ut_for(D)
    while i < size - 1 - D:
        while i + D < size and ge(i, D, Ut): i += 1
        if i + D >= size: break
        D += 1
        if D * 3 > size: break
        Ut = ut_for(D)
    fallback = not (D * 3 <= size)
    if not fallback:
        a = 1.0 / Ut
        k1, k2 = keys[size - 1 - D], keys[D]
        if mode == 'exact':
            b = (L - Fraction(a) * (Fraction(k1) + Fraction(k2))) / 2
        else:
            b = (L - a * (float(k1) + float(k2))) / 2
    else:
        mid1 = (size - 1) // 3; mid2 = (size - 1) * 2 // 3
        t1 = mid1 * (GAP + 1) + (GAP + 1) // 2; t2 = mid2 * (GAP + 1) + (GAP + 1) // 2
        if mode == 'exact':
            m1 = (Fraction(keys[mid1]) + Fraction(keys[mid1 + 1])) / 2; m2 = (Fraction(keys[mid2]) + Fraction(keys[mid2 + 1])) / 2
            a = float(Fraction(t2 - t1) / (m2 - m1)); b = t1 - Fraction(a) * m1
        else:
            m1 = (float(keys[mid1]) + float(keys[mid1 + 1])) / 2; m2 = (float(keys[mid2]) + float(keys[mid2 + 1])) / 2
            a = (t2 - t1) / (m2 - m1); b = t1 - a * m1
    counts = {}
    best = 0
    if mode == 'exact':
        af = Fraction(a)
        for k in keys:
            v = af * Fraction(k) + b
            p = math.floor(v)
            if p < 0: p = 0
            if p > L - 1: p = L - 1
            c = counts.get(p, 0) + 1; counts[p] = c
            if c > best: best = c
    else:
        for k in keys:
            v = a * float(k) + b
            p = L - 1 if v > 2147483647 // 2 else 0 if v < 0 else min(L - 1, int(v))
            c = counts.get(p, 0) + 1; counts[p] = c
            if c > best: best = c
    return dict(a=a, b=float(b), L=L, D=D, Ut=Ut, fallback=fallback, cd=best, slots=len(counts))

def load_flow(path):
    toks = open(path).read().split()
    it = iter(toks)
    in_dim, hidden, layers = int(next(it)), int(next(it)), int(next(it))
    mean, var = float(next(it)), float(next(it))
    W = []
    for l in range(layers):
        r, c = int(next(it)), int(next(it))
        W.append((r, c, [float(next(it)) for _ in range(r * c)]))
    return dict(in_dim=in_dim, hidden=hidden, layers=layers, mean=mean, var=var, W=W)

def flow_apply(f, key):
    x = (float(key) - f['mean']) / f['var']
    a = [x] + ([x - math.floor(x)] if f['in_dim'] == 2 else [])
    for l, (r, c, w) in enumerate(f['W']):
        b = [sum(a[i] * w[i * c + j] for i in range(r)) for j in range(c)]
        a = b if l + 1 == f['layers'] else [math.tanh(v) for v in b]
    z = sum(a)
    return z if math.isfinite(z) else 0.0

def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)
