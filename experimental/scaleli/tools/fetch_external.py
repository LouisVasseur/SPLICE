#!/usr/bin/env python3
"""Fetch official ALEX/PGM sources and persist exact commit IDs; never run upstream scripts."""
from __future__ import annotations
import argparse, datetime, json, pathlib, subprocess
REPOS={'ALEX':'https://github.com/microsoft/ALEX.git','PGM-index':'https://github.com/gvinciguerra/PGM-index.git'}

def git(*args: str,cwd: pathlib.Path|None=None) -> str:
    return subprocess.run(['git',*args],cwd=cwd,check=True,text=True,capture_output=True,timeout=180).stdout.strip()

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dest',type=pathlib.Path,default=pathlib.Path('external'))
    p.add_argument('--lock',type=pathlib.Path,default=pathlib.Path('external.lock.json'));p.add_argument('--locked',action='store_true',help='Require and use the existing commit lock.')
    a=p.parse_args()
    if a.lock.exists():lock=json.loads(a.lock.read_text())
    elif a.locked:p.error('missing lock file; an initial unlocked fetch is required')
    else:lock={'repositories':{}}
    a.dest.mkdir(parents=True,exist_ok=True)
    for name,url in REPOS.items():
        path=a.dest/name
        if path.exists():
            if git('remote','get-url','origin',cwd=path)!=url:raise RuntimeError(f'unexpected origin for {path}')
            if git('status','--porcelain',cwd=path):raise RuntimeError(f'refusing to modify dirty checkout {path}')
        else:git('clone','--filter=blob:none','--no-checkout',url,str(path))
        commit=lock['repositories'].get(name,{}).get('commit')
        if a.locked and not commit:p.error(f'lock has no commit for {name}')
        ref=commit or 'origin/HEAD'
        if commit:git('fetch','origin',commit,cwd=path)
        git('checkout','--detach',ref,cwd=path)
        lock['repositories'][name]={'url':url,'commit':git('rev-parse','HEAD',cwd=path)}
    lock['recorded_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    a.lock.write_text(json.dumps(lock,indent=2));print(a.lock)
if __name__=='__main__':main()
