"""Independent re-implementation of AIDB 2026 sec. 3.2 eq. (1)-(3), written from the paper text only.

Notation (paper): p_I(S) performance; sigma_I std of p_I over the corpus; p_hat = p/sigma;
S_i harder than S_j iff h_k(S_i) >= h_k(S_j) for all k and > for at least one;
C = ordered comparable pairs (harder, easier); U = unordered incomparable pairs;
violating for I iff p_I larger on the harder dataset (paper uses raw p_I here), conforming otherwise;
delta = p_hat(harder) - p_hat(easier); w = sigmoid(delta);
R = sum w over conforming, P = sum w over violating; Conf_I = (R-P)/(R+P); Conf = mean_I Conf_I;
Cov = (|C|-|U|)/N, N = n choose 2.
"""
import itertools, math

SCALARS = ["RMSE", "ME", "CD", "PLA-32", "PLA-4096"]
FIELD = {"RMSE": "rmse", "ME": "max_error", "CD": "conflict_degree", "PLA-32": "pla_32", "PLA-4096": "pla_4096"}


def all_metric_dims():
    """25 metrics: every non-empty subset of size 1..3 of the five scalars, components in canonical order."""
    out = []
    for d in (1, 2, 3):
        for combo in itertools.combinations(range(len(SCALARS)), d):
            out.append(tuple(SCALARS[i] for i in combo))
    return out


def stddev(xs, ddof):
    n = len(xs)
    if n - ddof <= 0:
        return 0.0
    mu = sum(xs) / n
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - ddof))


def sigmoid(x):
    # numerically plain; inputs here are O(10) at most
    return 1.0 / (1.0 + math.exp(-x))


def partial_order(hard, ds, dims):
    """hard: {dataset: {metric: value}}. Returns (C ordered list, U unordered list)."""
    C, U = [], []
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            a, b = ds[i], ds[j]
            a_ge_b = all(hard[a][k] >= hard[b][k] for k in dims)
            b_ge_a = all(hard[b][k] >= hard[a][k] for k in dims)
            # "a harder than b" == a>=b on every dim and NOT b>=a on every dim (i.e. strictly greater somewhere)
            if a_ge_b and not b_ge_a:
                C.append((a, b))
            elif b_ge_a and not a_ge_b:
                C.append((b, a))
            else:
                U.append((a, b))
    return C, U


def score(hard, thr, dims, ddof=0):
    """hard: {dataset: {metric: value}} ; thr: {variant: {dataset: p}} restricted to the same datasets."""
    ds = sorted(hard)
    n = len(ds)
    N = n * (n - 1) // 2
    C, U = partial_order(hard, ds, dims)
    assert len(C) + len(U) == N
    cov = (len(C) - len(U)) / N if N else None
    per = {}
    detail = {}
    for v, p in thr.items():
        vals = [p[d] for d in ds]
        s = stddev(vals, ddof)
        phat = {d: (p[d] / s if s > 0 else 0.0) for d in ds}
        R = P = 0.0
        nviol = 0
        for h, e in C:
            delta = phat[h] - phat[e]
            w = sigmoid(delta)
            if p[h] > p[e]:      # paper: violating when p_I is larger on the harder dataset
                P += w
                nviol += 1
            else:
                R += w
        per[v] = (R - P) / (R + P) if (R + P) > 0 else 0.0
        detail[v] = (R, P, nviol)
    conf = sum(per.values()) / len(per) if per else None
    return {"coverage": cov, "conformance": conf, "per_variant": per, "C": len(C), "U": len(U), "detail": detail}


def score_table(H, T, scope, ddof=0):
    """H: contract H.json dict, T: contract T.json dict. Returns {metric_name: score dict} for one scope."""
    hard = {}
    for d, scopes in H.items():
        blk = scopes.get(scope)
        if blk is None:
            continue
        hard[d] = {m: float(blk[FIELD[m]]) for m in SCALARS}
    ds = sorted(set(hard).intersection(*[set(T[v]) for v in T]))
    hard = {d: hard[d] for d in ds}
    thr = {v: {d: float(T[v][d]) for d in ds} for v in T}
    out = {}
    for dims in all_metric_dims():
        out["·".join(dims)] = score(hard, thr, list(dims), ddof)
    return out
