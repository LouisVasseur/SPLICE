"""Power simulation for PROTOCOL_draft.md section 5.4 / 6 (reviewer 1). Light: numpy, single thread, ~seconds.
Model: y = ln throughput per run = cell/dataset effect + noise (iid per process).
Noise: 'gauss' N(0, s); 'mix' busy two-state host: N(0, 3%) + (-20% w.p. 0.3)  -> sd ~9.6%.
"""
import math, sys
import numpy as np
rng = np.random.default_rng(20261002)

def t_ppf_table(df):
    x = np.linspace(-60, 60, 600001)
    c = math.lgamma((df+1)/2) - math.lgamma(df/2) - 0.5*math.log(df*math.pi)
    pdf = np.exp(c - (df+1)/2*np.log1p(x*x/df))
    cdf = np.concatenate([[0], np.cumsum((pdf[1:]+pdf[:-1])/2*np.diff(x))]); cdf /= cdf[-1]
    return x, cdf
TT = {}
def tcdf(v, df):
    if df not in TT: TT[df] = t_ppf_table(df)
    x, c = TT[df]; return np.interp(v, x, c)
def tppf(p, df):
    if df not in TT: TT[df] = t_ppf_table(df)
    x, c = TT[df]; return float(np.interp(p, c, x))

K = 10
D = math.log(1.03)  # margin (log); lower is ln .97 ~ -0.0305, use symmetric approx
LO, HI = math.log(0.97), math.log(1.03)
cells = ['B','B2','N','Nf','C','Cr','G','GCr','SV']
def effects(scen):
    e = {c: np.zeros(K) for c in cells}
    e['Nf'] = np.log(np.linspace(0.85, 0.96, K))
    adopt = np.zeros(K); adopt[:5] = math.log(1.05)          # CSV root adopted on 5 of 10, +5%; pooled ~2.5%
    e['C'] = adopt.copy(); e['Cr'] = adopt.copy(); e['GCr'] = adopt.copy()
    e['SV'] = np.full(K, math.log(1.30))
    if scen == 'nullshift':                                    # nulls true at -1% (table cost / transforms)
        e['N'] = np.full(K, math.log(0.99)); e['G'] = np.full(K, math.log(0.99)); e['GCr'] = adopt + math.log(0.99)
    return e

def simulate(noise, sigma, n, reps, scen='base'):
    eff = effects(scen)
    Y = {}
    for c in cells:
        if noise == 'gauss': z = rng.normal(0, sigma, (reps, K, n))
        else: z = rng.normal(0, 0.03, (reps, K, n)) + np.where(rng.random((reps, K, n)) < 0.3, -0.20, 0.0)
        Y[c] = eff[c][None, :, None] + z
    # pooled within-(dataset,cell) sigma, df = cells*K*(n-1)
    ss = sum(((Y[c] - Y[c].mean(2, keepdims=True))**2).sum((1, 2)) for c in cells)
    df = len(cells)*K*(n-1); s = np.sqrt(ss/df)
    m = {c: Y[c].mean(2) for c in cells}
    base = (m['B'] + m['B2'])/2    # B u B2, 2n runs
    out = {}
    def contrast(x, y, ny_mult):   # per dataset delta and SE
        d = m[x] - (base if y == 'BB2' else m[y])
        se = np.broadcast_to(s[:, None]*math.sqrt(1/n + 1/(n*ny_mult)), d.shape)
        return d, se
    specs = {'H22 N=B': ('N','BB2',2,'eq'), 'H23 Nf<B': ('Nf','BB2',2,'sup'), 'H24 C>B': ('C','BB2',2,'sup'),
             'Cr>B (new)': ('Cr','BB2',2,'sup'), 'H25 C=Cr': ('C','Cr',1,'eq'), 'H26a G=B': ('G','BB2',2,'eq'),
             'H26b GCr=Cr': ('GCr','Cr',1,'eq'), 'H27 SV>B': ('SV','BB2',2,'sup')}
    P = {'RE': {}, 'FX': {}}
    for name, (x, y, mult, kind) in specs.items():
        d, se = contrast(x, y, mult)
        th = d.mean(1)
        for mode in ('RE', 'FX'):
            if mode == 'RE': SE = d.std(1, ddof=1)/math.sqrt(K); dfx = K-1
            else: SE = np.sqrt((se**2).sum(1))/K; dfx = df
            if kind == 'sup':   # two-sided p
                tv = th/SE; p = 2*(1 - tcdf(np.abs(tv), dfx))
            else:
                p1 = 1 - tcdf((th - LO)/SE, dfx); p2 = tcdf((th - HI)/SE, dfx); p = np.maximum(p1, p2)
            P[mode][name] = p
    # A/A live gate (draft 6.6): pooled RE CI contains 0, RE TOST +-3% at .05, <=2/10 per-dataset CIs exclude 0
    dAA = m["B2"] - m["B"]; seAA = np.broadcast_to(s[:, None]*math.sqrt(2/n), dAA.shape)
    th = dAA.mean(1); SEre = dAA.std(1, ddof=1)/math.sqrt(K)
    ci_ok = np.abs(th) < tppf(0.975, K-1)*SEre
    tost_ok = (th - tppf(0.95, K-1)*SEre > LO) & (th + tppf(0.95, K-1)*SEre < HI)
    nex = (np.abs(dAA) > tppf(0.975, df)*seAA).sum(1)
    aa = {'ci': ci_ok.mean(), 'tost': tost_ok.mean(), 'per_ds<=2': (nex <= 2).mean(), 'all3': (ci_ok & tost_ok & (nex <= 2)).mean()}
    # alternative A/A gate: pooled FX CI contains 0 and <=2/10 exclude (no TOST)
    SEfx = np.sqrt((seAA**2).sum(1))/K
    aa['alt_ci_fx&per_ds'] = ((np.abs(th) < tppf(0.975, df)*SEfx) & (nex <= 2)).mean()
    return P, aa, s

def holm(pdict, names, alpha=0.05):
    ps = np.stack([pdict[k] for k in names], 1)          # reps x m
    order = np.argsort(ps, 1); m = len(names)
    rej = np.zeros_like(ps, dtype=bool); alive = np.ones(ps.shape[0], bool)
    for j in range(m):
        idx = order[:, j]; pj = ps[np.arange(ps.shape[0]), idx]
        ok = alive & (pj <= alpha/(m - j))
        rej[np.arange(ps.shape[0]), idx] |= ok; alive &= ok
    return {k: rej[:, i].mean() for i, k in enumerate(names)}

def gate_then_holm(pdict, gate, rest, alpha=0.05):
    g = np.ones(len(pdict[gate[0]]), bool); res = {}
    for k in gate:
        g &= pdict[k] <= alpha; res[k] = g.mean()
    ps = np.stack([pdict[k] for k in rest], 1); m = len(rest)
    order = np.argsort(ps, 1); rej = np.zeros_like(ps, dtype=bool); alive = g.copy()
    for j in range(m):
        idx = order[:, j]; pj = ps[np.arange(ps.shape[0]), idx]
        ok = alive & (pj <= alpha/(m - j)); rej[np.arange(ps.shape[0]), idx] |= ok; alive &= ok
    res.update({k: rej[:, i].mean() for i, k in enumerate(rest)}); return res

draftF1 = ['H22 N=B','H23 Nf<B','H24 C>B','H25 C=Cr','H26a G=B','H26b GCr=Cr','H27 SV>B']
rest = ['H24 C>B','Cr>B (new)','H22 N=B','H25 C=Cr','H26a G=B','H26b GCr=Cr']
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
rows = []
for noise, sig in (('gauss', .04), ('gauss', .05), ('gauss', .06), ('mix', None)):
    for n in (3, 8):
        for scen in ('base', 'nullshift'):
            P, aa, s = simulate(noise, sig, n, reps, scen)
            lab = f"{noise}{'' if sig is None else ' %.0f%%' % (100*sig)} n={n} {scen} (sigma_hat med {100*np.median(s):.1f}%)"
            un_re = {k: (P['RE'][k] <= .05).mean() for k in P['RE']}
            un_fx = {k: (P['FX'][k] <= .05).mean() for k in P['FX']}
            h_re = holm(P['RE'], draftF1); g_fx = gate_then_holm(P['FX'], ['H27 SV>B','H23 Nf<B'], rest)
            g_re = gate_then_holm(P['RE'], ['H27 SV>B','H23 Nf<B'], rest)
            print('\n### ' + lab)
            print('A/A gate pass prob (true A/A):', {k: round(float(v), 3) for k, v in aa.items()})
            print('| hypothesis | RE unadj | RE Holm-7 (draft) | RE gate+Holm-6 | FX unadj | FX gate+Holm-6 |')
            for k in ['H27 SV>B','H23 Nf<B','H24 C>B','Cr>B (new)','H22 N=B','H25 C=Cr','H26a G=B','H26b GCr=Cr']:
                print(f"| {k} | {un_re[k]:.2f} | {h_re.get(k, float('nan')):.2f} | {g_re[k]:.2f} | {un_fx[k]:.2f} | {g_fx[k]:.2f} |")
# E1 gate on point estimate, df 18
print('\n### E1 gate P(sigma_hat <= 5%), df 18 (chi2)')
for st in (.03, .04, .05, .06, .07, .08):
    sh = st*np.sqrt(rng.chisquare(18, 200000)/18)
    print(f"true {100*st:.0f}%: P(<=5%) {np.mean(sh <= .05):.2f}  P(5-8%) {np.mean((sh > .05) & (sh <= .08)):.2f}  P(>8%) {np.mean(sh > .08):.2f}")
print('\n### COLD control P(mean c1 > -15% | true -21%, per-run sd 10%)')
for nn in (3, 6, 9):
    print(nn, round(float(np.mean(rng.normal(-.21, .10/math.sqrt(nn), 200000) > -.15)), 3))
for q in (0.95, 0.975, 0.99, 0.99167, 0.9875, 0.99375):
    print('t9', q, round(tppf(q, 9), 3))
