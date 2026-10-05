#!/usr/bin/env python3
"""Small exact/greedy virtual-point reference. NOT a reproduction of CSV's optimized algorithm."""
from __future__ import annotations
import argparse, bisect, itertools, json, math

def fit_loss(keys: list[int], originals: list[int]|None=None) -> tuple[float,float]:
    n=len(keys);mx=sum(keys)/n;my=(n-1)/2
    xx=sum((x-mx)**2 for x in keys);xy=sum((x-mx)*(i-my) for i,x in enumerate(keys))
    slope=xy/xx if xx else 0;intercept=my-slope*mx
    all_loss=sum((slope*x+intercept-i)**2 for i,x in enumerate(keys))
    real_loss=sum((slope*x+intercept-bisect.bisect_left(keys,x))**2 for x in (originals or keys))
    return all_loss,real_loss

def greedy(keys: list[int],budget: int) -> dict:
    originals=sorted(set(keys));aug=originals[:];virtual=[];loss=fit_loss(aug)[0]
    for _ in range(budget):
        best=None
        for k in range(aug[0]+1,aug[-1]):
            if k in aug:continue
            candidate=sorted(aug+[k]);new=fit_loss(candidate,originals)[0]
            if new<loss-1e-12 and (best is None or new<best[0]):best=(new,k,candidate)
        if best is None:break
        loss,k,aug=best;virtual.append(k)
    all_loss,real_loss=fit_loss(aug,originals)
    return {'virtual_points':virtual,'augmented_keys':aug,'sse_all':all_loss,'sse_original':real_loss}

def exhaustive(keys: list[int],budget: int,cap: int=250000) -> dict:
    originals=sorted(set(keys));pool=[k for k in range(originals[0]+1,originals[-1]) if k not in originals]
    combinations=sum(math.comb(len(pool),i) for i in range(min(budget,len(pool))+1))
    if combinations>cap:raise ValueError(f'{combinations} candidate subsets exceeds safety cap {cap}')
    best=(fit_loss(originals)[0],[],originals)
    for count in range(1,min(budget,len(pool))+1):
        for v in itertools.combinations(pool,count):
            aug=sorted(originals+list(v));loss=fit_loss(aug)[0]
            if loss<best[0]:best=(loss,list(v),aug)
    return {'virtual_points':best[1],'augmented_keys':best[2],'sse_all':best[0],
            'sse_original':fit_loss(best[2],originals)[1],'subsets_examined':combinations}

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--keys',default='1,2,3,4,5,10,20,26,27,30');p.add_argument('--budget',type=int,default=3);p.add_argument('--exact',action='store_true')
    a=p.parse_args();keys=sorted(set(map(int,a.keys.split(','))))
    if len(keys)<2 or keys[-1]-keys[0]>4096 or a.budget<0:p.error('use at least two distinct keys, span <=4096, nonnegative budget')
    result={'notice':'Illustrative toy; not the paper figure, not an optimized CSV implementation.',
            'objective':'OLS SSE over augmented keys; original-key SSE also reported.',
            'before_sse':fit_loss(keys)[0],'greedy':greedy(keys,a.budget)}
    if a.exact:result['exhaustive']=exhaustive(keys,a.budget)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
