#!/usr/bin/env python3
"""Measure logical versus physical geometry using ACTUAL encoded C++ layouts."""
from __future__ import annotations
import argparse, collections, csv, json, math, pathlib, statistics, subprocess

def percentile(xs: list[float], q: float) -> float:
    return sorted(xs)[min(len(xs)-1,max(0,math.ceil(len(xs)*q)-1))] if xs else 0.0

def analyze(path: pathlib.Path) -> dict:
    regions: dict[int,list[dict]] = collections.defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f): regions[int(r['region'])].append(r)
    rank_error=[];byte_error=[];rank_norm=[];byte_norm=[];rank_blocks=[];byte_blocks=[]
    codecs=collections.Counter();seen_blocks=set();total_bytes=0;keys=[]
    for region, rows in regions.items():
        n=len(rows);arena=int(rows[0]['region_bytes']);total_bytes+=arena
        for r in rows:
            keys.append((region,int(r['local_rank']),int(r['key']),int(r['block'])))
            er=float(r['rank_prediction'])-int(r['local_rank']);eb=float(r['byte_prediction'])-float(r['key_byte'])
            rank_error.append(abs(er));byte_error.append(abs(eb));rank_norm.append(er/max(1,n-1));byte_norm.append(eb/max(1,arena))
            rank_blocks.append(abs(int(r['rank_candidate'])-int(r['block'])));byte_blocks.append(abs(int(r['byte_candidate'])-int(r['block'])))
            block=(region,int(r['block']))
            if block not in seen_blocks:codecs[r['codec']]+=1;seen_blocks.add(block)
    n=len(keys)
    return {'keys':n,'regions':len(regions),'blocks':len(seen_blocks),'key_bytes':total_bytes,
            'bytes_per_key':total_bytes/max(1,n),'codecs':dict(codecs),
            'rank_mae':statistics.mean(rank_error) if n else 0,'rank_p99':percentile(rank_error,.99),
            'byte_mae':statistics.mean(byte_error) if n else 0,'byte_p99':percentile(byte_error,.99),
            'rank_normalized_mse':statistics.mean(x*x for x in rank_norm) if n else 0,
            'byte_normalized_mse':statistics.mean(x*x for x in byte_norm) if n else 0,
            'rank_mean_block_error':statistics.mean(rank_blocks) if n else 0,
            'byte_mean_block_error':statistics.mean(byte_blocks) if n else 0,
            'rank_p99_block_error':percentile(rank_blocks,.99),'byte_p99_block_error':percentile(byte_blocks,.99),
            '_identity':keys}

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=pathlib.Path,default=pathlib.Path('build/scaleli_bench'))
    p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('results/layout'));p.add_argument('--n',type=int,default=16384)
    p.add_argument('--distribution',default='dense_sparse');p.add_argument('--region-keys',type=int,default=4096);p.add_argument('--block-keys',type=int,default=128)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);summary=[];identity=None;rank_mse=None
    for policy in ['raw','min_bytes','smooth']:
        path=(a.output/(policy+'.csv')).resolve()
        command=[str(a.binary.resolve()),'--n',str(a.n),'--ops','1','--distribution',a.distribution,'--load-ratio','1',
                 '--region-keys',str(a.region_keys),'--block-keys',str(a.block_keys),'--policy',policy,'--dump-layout',str(path),
                 '--warmup','0','--verify','0','--instrument','0']
        result=subprocess.run(command,check=True,text=True,capture_output=True)
        row=analyze(path);current=row.pop('_identity')
        if identity is not None and current!=identity:raise AssertionError('compression changed logical keys/partitions')
        if rank_mse is not None and abs(row['rank_normalized_mse']-rank_mse)>1e-14:raise AssertionError('compression changed logical rank model')
        identity=current;rank_mse=row['rank_normalized_mse'];row['policy']=policy;row['distribution']=a.distribution;summary.append(row)
    (a.output/'summary.json').write_text(json.dumps({'logical_rank_invariance_verified':True,'results':summary},indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
