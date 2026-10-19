#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Train MolFlow (include/gret/flow.h), the monotone flow transform, on a GRE key file.

    python3 transform/train_flow.py KEYS OUT.flow [--report OUT.json] [--m 32] [--steps 2000] [--sample 2000000]

KEYS is a GRE key file (a uint64 count, then the keys, little-endian). The flow is a mixture-of-logistics CDF
u(x) = sum_m w_m sigmoid((x - mu_m) / s_m) on x = (k - lo) / (hi - lo), trained by maximum likelihood with a flat latent
on [0, 1]: Adam on -mean log du/dx, float64, CPU. lo and hi are order statistics of the sample at the 0.1% and 99.9%
quantiles, so that outliers (fb reaches 2^64-1) do not crush the bulk's resolution.

Parameters: softmax logits for w (equal at start); mu = mu0 + s0 * d, with mu0 at the sample quantiles (m + 1/2) / M
and d learned, so that Adam's steps scale with each component's initial width; s = softplus(raw) + 1e-12, with s0
from the quantile spacing. For the training only, x is clipped to [-clip, 1 + clip], so that a few far outliers
cannot dominate the likelihood; the flow file and every reported metric use the unclipped x, as the C++ side does.

The JSON report (default OUT.flow.json) gives the training NLL, the KS distance of u to uniform on the whole sample,
the decreasing pairs and the collision fraction of T = clamp(floor(u 2^52), 0, 2^52 - 1) on the sorted sample (a
sample is sparser than the key set, so the key set has more collisions), and the wall times. Same seed and --threads
give the same file. Needs numpy >= 1.17 (default_rng) and torch (1.x or 2.x, CPU build is enough): a dev tool, not
part of the benchmark. Without torch (e.g. a conda env on the Atom), train on the Mac and copy the .flow files.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

OUT_SCALE = 2.0 ** 52  # gret::kOutScale
S_MIN = 1e-12


def die(msg):
    sys.exit('train_flow.py: ' + msg)


def read_keys(path):
    """The keys of a GRE file as a read-only uint64 memmap."""
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head = f.read(8)
    if len(head) != 8:
        die('%s: not a GRE key file' % path)
    n = int(np.frombuffer(head, '<u8')[0])
    if n < 2 or size != 8 * (n + 1):
        die('%s: %d bytes for a count of %d keys' % (path, size, n))
    return np.memmap(path, dtype='<u8', mode='r', offset=8, shape=(n,))


def is_sorted(a, chunk=1 << 24):
    for i in range(0, len(a) - 1, chunk):
        b = np.asarray(a[i:i + chunk + 1])
        if np.any(b[1:] < b[:-1]):
            return False
    return True


def sample_sorted(keys, n, mode, seed):
    """n keys, sorted: positions round(i (N-1) / (n-1)) of the sorted keys (stride), or a seeded random subset."""
    N = len(keys)
    if mode == 'random':
        idx = np.sort(np.random.default_rng(seed).choice(N, size=min(n, N), replace=False))
        return np.sort(np.asarray(keys[idx], dtype=np.uint64))
    srt = keys if is_sorted(keys) else np.sort(np.asarray(keys, dtype=np.uint64))
    if n >= N:
        return np.array(srt, dtype=np.uint64)
    i = np.arange(n, dtype=np.int64)
    return np.asarray(srt[(i * (N - 1) + (n - 1) // 2) // (n - 1)], dtype=np.uint64)  # i (N-1) < 2^63


def bounds(s, q_lo, q_hi):
    """lo < hi, as Python ints: order statistics of the sorted sample s."""
    n = len(s)
    lo, hi = int(s[int(math.floor(q_lo * (n - 1)))]), int(s[int(math.ceil(q_hi * (n - 1)))])
    if hi <= lo:
        lo, hi = int(s[0]), int(s[-1])
    if hi <= lo:
        lo, hi = (lo, lo + 1) if lo < 2 ** 64 - 1 else (lo - 1, lo)
    return lo, hi


def to_x(k, lo, hi):
    """x(k) exactly as MolFlow::x_of: double(k - lo) / double(hi - lo), or -double(lo - k) / double(hi - lo)."""
    k = np.asarray(k, dtype=np.uint64)
    x = np.empty(len(k), dtype=np.float64)
    d, l = float(hi - lo), np.uint64(lo)
    ge = k >= l
    x[ge] = (k[ge] - l).astype(np.float64) / d
    x[~ge] = -(l - k[~ge]).astype(np.float64) / d
    return x


def softplus(t):
    return F.softplus(t, beta=1.0, threshold=50.0)


class Mol(torch.nn.Module):
    def __init__(self, x, M):
        super().__init__()
        edges = np.quantile(x, np.arange(M + 1) / M)
        gaps = np.diff(np.unique(x))
        floor = max(float(np.median(gaps)) if len(gaps) else 1e-6, 1e-9)
        s0 = np.maximum(np.diff(edges), floor) * 0.5
        self.register_buffer('mu0', torch.tensor(np.quantile(x, (np.arange(M) + 0.5) / M), dtype=torch.float64))
        self.register_buffer('s0', torch.tensor(s0, dtype=torch.float64))
        y = s0 - S_MIN
        self.logits = torch.nn.Parameter(torch.zeros(M, dtype=torch.float64))
        self.d = torch.nn.Parameter(torch.zeros(M, dtype=torch.float64))
        self.raw = torch.nn.Parameter(torch.tensor(y + np.log(-np.expm1(-y)), dtype=torch.float64))  # softplus^-1

    def params(self):
        return torch.softmax(self.logits, 0), self.mu0 + self.s0 * self.d, softplus(self.raw) + S_MIN

    def log_density(self, x):
        """log du/dx = logsumexp_m [log w_m + log sigmoid(z) + log sigmoid(-z) - log s_m], z = (x - mu_m) / s_m."""
        _, mu, s = self.params()
        z = (x[:, None] - mu) / s
        return torch.logsumexp(torch.log_softmax(self.logits, 0) - softplus(z) - softplus(-z) - torch.log(s), dim=1)


def write_flow(path, lo, hi, w, mu, s):
    w = np.maximum(np.asarray(w, dtype=np.float64), 1e-300)  # keep w > 0 if a softmax weight underflows
    w = w / w.sum()
    with open(path, 'w') as f:
        f.write('molflow 1 %d %d %d\n' % (len(w), lo, hi))
        for a, b, c in zip(w, mu, s):
            f.write('%.17g %.17g %.17g\n' % (a, b, c))


def read_flow(path):
    """(lo, hi, w, mu, s) from a flow file, w renormalized as MolFlow::load does (no validation: see flow.h)."""
    with open(path) as f:
        head = f.readline().split()
        rows = np.array([[float(t) for t in line.split()] for line in f if line.strip()], dtype=np.float64)
    w = rows[:, 0] / rows[:, 0].sum()
    return int(head[3]), int(head[4]), w, rows[:, 1], rows[:, 2]


def u_of_x(x, w, mu, s, chunk=1 << 18):
    """u(x) in torch float64, by chunks of keys."""
    w, mu, s = (torch.as_tensor(np.asarray(a), dtype=torch.float64) for a in (w, mu, s))
    out = []
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            xt = torch.as_tensor(x[i:i + chunk], dtype=torch.float64)
            out.append((torch.sigmoid((xt[:, None] - mu) / s) * w).sum(1).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def to_T(u):
    return np.clip(np.floor(u * OUT_SCALE), 0, OUT_SCALE - 1).astype(np.uint64)


def ks_uniform(u):
    us = np.sort(u)
    n = len(us)
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(max(np.max(i / n - us), np.max(us - (i - 1) / n)))


def collision_fraction(keys, T):
    """Share of the distinct sample keys whose T is shared with another distinct key."""
    keep = np.r_[True, keys[1:] != keys[:-1]]
    _, inv, cnt = np.unique(T[keep], return_inverse=True, return_counts=True)
    return float(np.mean(cnt[inv] >= 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('keys', help='GRE key file')
    ap.add_argument('out', help='flow file to write')
    ap.add_argument('--report', help='JSON report (default OUT + ".json", e.g. t.flow.json)')
    ap.add_argument('--m', type=int, default=32, help='mixture components')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=65536)
    ap.add_argument('--lr', type=float, default=1e-2, help='Adam, cosine decay to lr/100')
    ap.add_argument('--sample', type=int, default=2000000, help='sample size')
    ap.add_argument('--sampling', choices=('stride', 'random'), default='stride')
    ap.add_argument('--lo-q', type=float, default=0.001)
    ap.add_argument('--hi-q', type=float, default=0.999)
    ap.add_argument('--clip', type=float, default=1.0, help='training x clipped to [-clip, 1 + clip]')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    if not (1 <= a.m <= 4096 and a.steps >= 0 and a.batch >= 1 and a.sample >= 2 and a.clip >= 0
            and 0 <= a.lo_q < a.hi_q <= 1):
        die('bad arguments')
    t0 = time.time()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    keys = read_keys(a.keys)
    s = sample_sorted(keys, a.sample, a.sampling, a.seed)
    lo, hi = bounds(s, a.lo_q, a.hi_q)
    x = to_x(s, lo, hi)
    xc = np.clip(x, -a.clip, 1 + a.clip)
    t1 = time.time()

    model = Mol(xc, a.m)
    xt = torch.as_tensor(xc)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(a.steps, 1), eta_min=a.lr / 100)
    gen = torch.Generator().manual_seed(a.seed)
    for _ in range(a.steps):
        batch = xt if a.batch >= len(xt) else xt[torch.randint(0, len(xt), (a.batch,), generator=gen)]
        loss = -model.log_density(batch).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    t2 = time.time()

    with torch.no_grad():
        w, mu, sc = (p.numpy().copy() for p in model.params())
        nll = -float(torch.cat([model.log_density(xt[i:i + (1 << 18)]) for i in range(0, len(xt), 1 << 18)]).mean())
    write_flow(a.out, lo, hi, w, mu, sc)
    lo, hi, w, mu, sc = read_flow(a.out)  # the metrics use exactly what the C++ side will load
    u = u_of_x(x, w, mu, sc)
    T = to_T(u)
    rep = {
        'keys': os.path.abspath(a.keys), 'flow': os.path.abspath(a.out), 'n_keys': len(keys), 'n_sample': len(s),
        'sampling': a.sampling, 'M': a.m, 'steps': a.steps, 'batch': a.batch, 'lr': a.lr, 'seed': a.seed,
        'threads': a.threads, 'lo': lo, 'hi': hi, 'lo_q': a.lo_q, 'hi_q': a.hi_q, 'clip': a.clip,
        'clipped_fraction': float(np.mean((x < -a.clip) | (x > 1 + a.clip))),
        'nll_train': nll,  # nats per key, density of x (clipped), mean over the whole sample
        'ks_uniform': ks_uniform(u),
        'decreasing_pairs': int(np.count_nonzero(T[1:] < T[:-1])),
        'collision_fraction': collision_fraction(s, T),
        'wall_s': {'load_sample': round(t1 - t0, 3), 'train': round(t2 - t1, 3), 'total': round(time.time() - t0, 3)},
        'torch': torch.__version__, 'numpy': np.__version__,
    }
    with open(a.report or a.out + '.json', 'w') as f:
        json.dump(rep, f, indent=1)
        f.write('\n')
    print('train_flow: %s M=%d nll=%.4f ks=%.2e decreasing=%d collisions=%.2e train_s=%.1f -> %s'
          % (os.path.basename(a.keys), a.m, nll, rep['ks_uniform'], rep['decreasing_pairs'],
             rep['collision_fraction'], t2 - t1, a.out))


if __name__ == '__main__':
    main()
