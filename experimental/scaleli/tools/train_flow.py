#!/usr/bin/env python3
"""Train a small NFL-style key transformation. Python standard library only.

Reproduces the deployed shape of NFL's numerical flow (Wu et al., PVLDB 2022):
features [x, x - floor(x)] with x = (key - mean) / var, var = (max - min) / shifts,
a 2x2 tanh hidden layer, a 2x2 output layer, and a sum decoder. Weights are
written in the text format read by include/scaleli/transform.hpp, which is the
same format the official repository exports (so its author weights load too).

Training differs from the official PyTorch code: this uses the one-dimensional
change-of-variables likelihood under a standard normal, analytic gradients and
Adam, with a barrier that keeps dz/dx positive. It is a reproducible, dependency
free stand-in, not a BNAF. Training cost is minutes at most on a few thousand keys.
"""
from __future__ import annotations
import argparse, json, math, random, struct, sys, time
from pathlib import Path

def read_keys(path: Path, dtype: str) -> list[int]:
    width, fmt = (8, 'Q') if dtype == 'uint64' else (4, 'I')
    b = path.read_bytes(); n = struct.unpack_from('<Q', b)[0]
    if len(b) != 8 + width * n: raise ValueError('SOSD header count does not match file length')
    return sorted(set(struct.unpack_from(f'<{n}{fmt}', b, 8)))

def features(x: float) -> tuple[float, float]:
    return x, x - math.floor(x)

class Flow:
    def __init__(self, w0: list[float], w1: list[float], mean: float, var: float):
        self.w0, self.w1, self.mean, self.var = w0, w1, mean, var  # row-major 2x2 each
    def forward(self, key: float):
        x = (key - self.mean) / self.var; f = features(x); w0, w1 = self.w0, self.w1
        u = [f[0] * w0[0] + f[1] * w0[2], f[0] * w0[1] + f[1] * w0[3]]
        h = [math.tanh(u[0]), math.tanh(u[1])]
        v = [w1[0] + w1[1], w1[2] + w1[3]]           # sum decoder folds the two output columns
        a = [w0[0] + w0[2], w0[1] + w0[3]]           # d u_h / d x, since d f1 / d x = 1 almost everywhere
        s = [1 - h[0] ** 2, 1 - h[1] ** 2]
        z = h[0] * v[0] + h[1] * v[1]
        d = s[0] * a[0] * v[0] + s[1] * a[1] * v[1]   # dz/dx in normalized units; dz/dkey = d / var
        return z, d, f, h, s, v, a
    def transform(self, key: float) -> float: return self.forward(key)[0]
    def save(self, path: Path) -> None:
        with path.open('w') as f:
            f.write('2\t2\t2\n%.16f\t%.16f\n' % (self.mean, self.var))
            for w in (self.w0, self.w1):
                f.write('2\t2\n'); f.write('%.16f\t%.16f\t\n%.16f\t%.16f\t\n' % tuple(w))

def loss_and_grad(flow: Flow, keys: list[float], barrier: float, monotone: bool):
    # Negative log-likelihood of the keys under z ~ N(0,1) via 1-D change of variables,
    # in normalized x units (the constant log var is dropped). A barrier keeps dz/dx > 0.
    n = len(keys); total = 0.0; g0 = [0.0] * 4; g1 = [0.0] * 4
    for key in keys:
        z, d, f, h, s, v, a = flow.forward(key)
        if d > 0: total += 0.5 * z * z - math.log(d) + barrier / d; gd = -1.0 / d - barrier / (d * d)
        else: total += 0.5 * z * z + 50.0 - 10.0 * d; gd = -10.0  # linear push-back when the sign flips
        gz = z
        for hh in range(2):
            for i in range(2):
                idx = i * 2 + hh
                dz = v[hh] * s[hh] * f[i]
                dd = v[hh] * ((-2 * h[hh] * s[hh] * f[i]) * a[hh] + s[hh])
                g0[idx] += gz * dz + gd * dd
            for c in range(2):
                idx = hh * 2 + c
                dz = h[hh]; dd = s[hh] * a[hh]
                g1[idx] += gz * dz + gd * dd
    if monotone: g0[2] = g0[3] = 0.0  # fractional-feature weights stay at zero: transform is monotone in x
    return total / n, [g / n for g in g0], [g / n for g in g1]

def tail_conflict_degree(xs: list[float], amplification: float = 1.5, tail: float = 0.99) -> int:
    n = len(xs)
    if n < 2 or not xs[-1] > xs[0]: return max(0, n - 1)
    mx = sum(xs) / n; my = (n - 1) / 2
    xx = sum((x - mx) ** 2 for x in xs); xy = sum((x - mx) * (i - my) for i, x in enumerate(xs))
    slope = xy / xx if xx > 0 else 0
    if not slope > 0: slope = n / (xs[-1] - xs[0])
    intercept = -slope * xs[0] + 0.5
    max_size = max(1, min(int(n * amplification), int(max(1.0, math.floor(slope * xs[-1] + intercept) + 1))))
    pos = [min(max(int(math.floor(slope * x + intercept)), 0), max_size - 1) for x in xs]
    counts = []; run = 1
    for i in range(1, n):
        if pos[i] == pos[i - 1]: run += 1
        else: counts.append(run); run = 1
    counts.append(run); counts.sort()
    return counts[min(len(counts) - 1, max(0, math.ceil(tail * len(counts)) - 1))] - 1

def train(keys: list[int], shifts: float, steps: int, lr: float, seed: int, barrier: float, log=print, monotone: bool = False) -> tuple[Flow, dict]:
    rng = random.Random(seed)
    mean = float(keys[0]); var = max(1.0, float(keys[-1] - keys[0]) / shifts)
    # x = (key - mean) / var spans [0, shifts]. Weights on the raw x feature are
    # learned in units of 1/shifts so that Adam steps cannot saturate the tanh in
    # one move (the official code relies on BNAF weight normalisation for this).
    scale0 = [1.0 / shifts, 1.0 / shifts, 1.0, 1.0]  # w0 index i*2+h: row 0 = x feature, row 1 = fractional feature
    w0 = [abs(rng.gauss(1.5, 0.5)) * scale0[0], abs(rng.gauss(1.5, 0.5)) * scale0[1], abs(rng.gauss(0.1, 0.05)), abs(rng.gauss(0.1, 0.05))]
    w1 = [abs(rng.gauss(1.0, 0.2)) for _ in range(4)]
    if monotone: w0[2] = w0[3] = 0.0
    flow = Flow(w0, w1, mean, var); ks = [float(k) for k in keys]
    m0 = [0.0] * 4; v0 = [0.0] * 4; m1 = [0.0] * 4; v1 = [0.0] * 4; b1, b2, eps = 0.9, 0.999, 1e-8
    history = []; best = (float('inf'), list(w0), list(w1))
    for t in range(1, steps + 1):
        loss, g0, g1 = loss_and_grad(flow, ks, barrier, monotone)
        if not math.isfinite(loss): raise RuntimeError('non-finite loss; lower --lr or raise --barrier')
        if loss < best[0]: best = (loss, list(flow.w0), list(flow.w1))
        history.append(loss)
        for w, g, m, v, sc in ((flow.w0, g0, m0, v0, scale0), (flow.w1, g1, m1, v1, [1.0] * 4)):
            for i in range(4):
                gi = g[i] * sc[i]  # gradient with respect to the scaled parameter theta = w / scale
                m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi
                w[i] -= sc[i] * lr * (m[i] / (1 - b1 ** t)) / (math.sqrt(v[i] / (1 - b2 ** t)) + eps)
        if t % max(1, steps // 10) == 0 or t == 1: log(f'step {t:5d}  nll {loss:.6f}')
    flow.w0, flow.w1 = best[1], best[2]
    return flow, {'best_nll': best[0], 'history': history}

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('data', type=Path, help='SOSD-format key file (uint64 count header + keys)')
    p.add_argument('--dtype', default='uint64', choices=['uint64', 'uint32'])
    p.add_argument('--output', type=Path, required=True, help='weight file to write (NFL text format)')
    p.add_argument('--shifts', type=float, default=64, help='NFL min-max scaling divisor: x spans [0, shifts] and the fractional feature repeats every 1/shifts of the key span (author default 1e6 for 200M keys)')
    p.add_argument('--sample', type=int, default=4096, help='training keys, uniformly sampled (0 = all)')
    p.add_argument('--steps', type=int, default=200); p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--barrier', type=float, default=1e-3, help='penalty keeping dz/dx positive (monotone transform)')
    p.add_argument('--seed', type=int, default=1000000007)
    p.add_argument('--monotone', action='store_true', help='zero the fractional-feature weights so z is monotone in the key (deviation from NFL, which keeps them)')
    a = p.parse_args()
    keys = read_keys(a.data, a.dtype)
    rng = random.Random(a.seed)
    train_keys = sorted(rng.sample(keys, a.sample)) if a.sample and a.sample < len(keys) else keys
    start = time.time(); flow, info = train(train_keys, a.shifts, a.steps, a.lr, a.seed, a.barrier, log=lambda s: print(s, file=sys.stderr), monotone=a.monotone)
    elapsed = time.time() - start
    a.output.parent.mkdir(parents=True, exist_ok=True); flow.save(a.output)
    z = [flow.transform(float(k)) for k in keys]
    unordered = sum(1 for i in range(1, len(z)) if z[i] < z[i - 1])
    raw = [(k - keys[0]) / max(1, keys[-1] - keys[0]) for k in keys]
    report = {'weights': str(a.output), 'keys': len(keys), 'training_keys': len(train_keys), 'steps': a.steps, 'seed': a.seed,
              'shifts': a.shifts, 'monotone': a.monotone, 'best_nll': info['best_nll'], 'train_seconds': elapsed, 'unordered_transformed_pairs': unordered,
              'tail_conflict_degree_raw': tail_conflict_degree(raw), 'tail_conflict_degree_transformed': tail_conflict_degree(sorted(z)),
              'note': 'Stand-in trainer (1-D likelihood, Adam), not the official BNAF; format-compatible with luffy06/NFL weights.'}
    print(json.dumps(report, indent=2))
if __name__ == '__main__': main()
