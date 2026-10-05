import sys, numpy as np
sys.argv = ['x', '10']
exec(open('power_sim.py').read().split('draftF1 =')[0])
def fam(P, mode):
    out = {}
    for f in (['H22 N=B'], ['H24 C>B', 'Cr>B (new)', 'H25 C=Cr'], ['H26a G=B', 'H26b GCr=Cr']):
        out.update(holm(P[mode], f))
    return out
print('| scenario | test | C>B | Cr>B | N=B | C=Cr | G=B | GCr=Cr |')
for noise, sig, n in (('gauss', .03, 3), ('gauss', .04, 3), ('gauss', .05, 3), ('gauss', .04, 5), ('gauss', .05, 5), ('gauss', .04, 8), ('gauss', .06, 8)):
    for scen in ('base', 'nullshift'):
        P, aa, s = simulate(noise, sig, n, 20000, scen)
        for mode in ('RE', 'FX'):
            r = fam(P, mode)
            print(f"| {100*sig:.0f}% n={n} {scen} | {mode} per-method Holm | " + ' | '.join(f"{r[k]:.2f}" for k in ['H24 C>B','Cr>B (new)','H22 N=B','H25 C=Cr','H26a G=B','H26b GCr=Cr']) + ' |')
        print(f"|  | A/A draft gate pass {aa['all3']:.2f}; alt gate {aa['alt_ci_fx&per_ds']:.2f} | | | | | | |")
