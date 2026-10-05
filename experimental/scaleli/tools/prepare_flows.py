#!/usr/bin/env python3
"""Dump each dataset of a sweep config and train its flow weights (stdlib only).

For every dataset entry with "flow_weights" (and optional "flow_train_args", e.g.
["--monotone"]), the benchmark binary writes the
dataset's full key set (load-ratio 1, first seed) in SOSD format, and
tools/train_flow.py trains the 2D2H2L transform on a sample of it. The flow is
therefore trained on keys that later inserts are drawn from, which matches
NFL's in-domain insert setting and must be stated when reporting.
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
HERE=pathlib.Path(__file__).resolve().parent

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=pathlib.Path,default=pathlib.Path('build/scaleli_bench'))
    p.add_argument('--config',type=pathlib.Path,default=pathlib.Path('configs/learnability.json'))
    p.add_argument('--sample',type=int,default=4096);p.add_argument('--steps',type=int,default=200);p.add_argument('--force',action='store_true')
    a=p.parse_args();cfg=json.loads(a.config.read_text());common=cfg.get('common',{});seed=cfg.get('seeds',[42])[0];reports=[]
    for d in cfg['datasets']:
        if isinstance(d,str) or 'flow_weights' not in d:continue
        weights=pathlib.Path(d['flow_weights']);weights.parent.mkdir(parents=True,exist_ok=True)
        if weights.exists() and not a.force:print('keep',weights,file=sys.stderr);continue
        keys=weights.with_suffix('.keys')
        cmd=[str(a.binary.resolve()),'--ops','1','--seed',str(seed),'--load-ratio','1','--verify','0','--instrument','0','--warmup','0','--dump-keys',str(keys)]
        for k,v in {**common,**{k:v for k,v in d.items() if k not in ('name','flow_weights','flow_train_args')}}.items():
            if k in ('ops','verify','instrument','warmup'):continue
            cmd+=['--'+k,str(int(v)) if isinstance(v,bool) else str(v)]
        subprocess.run(cmd,check=True,capture_output=True,text=True)
        extra=[str(x) for x in d.get('flow_train_args',[])]
        out=subprocess.run([sys.executable,str(HERE/'train_flow.py'),str(keys),'--output',str(weights),'--sample',str(a.sample),'--steps',str(a.steps),*extra],check=True,capture_output=True,text=True)
        r=json.loads(out.stdout);r['dataset']=d['name'];reports.append(r);print(d['name'],'nll',round(r['best_nll'],4),'tail conflicts',r['tail_conflict_degree_raw'],'->',r['tail_conflict_degree_transformed'],file=sys.stderr)
    (pathlib.Path(cfg['datasets'][0]['flow_weights']).parent/'training_report.json').write_text(json.dumps(reports,indent=2)) if reports else None
    print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
