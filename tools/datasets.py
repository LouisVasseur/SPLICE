#!/usr/bin/env python3
"""Inspect and sample SOSD integer files. No downloads (data/external/gre/download.sh)."""
from __future__ import annotations
import argparse, array, hashlib, json, mmap, pathlib, random, struct, sys

def digest(path: pathlib.Path, kind: str='sha256') -> str:
    h=hashlib.new(kind)
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def header(path: pathlib.Path,dtype: str) -> tuple[int,int]:
    width={'uint64':8,'uint32':4}[dtype]
    with path.open('rb') as f:
        b=f.read(8)
        if len(b)!=8:raise ValueError('truncated count header')
        n=struct.unpack('<Q',b)[0]
    if path.stat().st_size!=8+n*width:raise ValueError('count header does not match byte length')
    return n,width

def inspect(path: pathlib.Path,dtype: str) -> dict:
    n,w=header(path,dtype);prev=None;duplicates=0;sorted_=True;minimum=None;maximum=None
    with path.open('rb') as f:
        f.seek(8)
        while b:=f.read(65536*w):
            values=array.array('Q' if w==8 else 'I');values.frombytes(b)
            if sys.byteorder!='little':values.byteswap()
            for x in values:
                if prev is not None:duplicates+=int(x==prev);sorted_ &= x>=prev
                minimum=x if minimum is None else min(minimum,x);maximum=x if maximum is None else max(maximum,x);prev=x
    return {'path':str(path),'count':n,'dtype':dtype,'sorted':sorted_,'adjacent_duplicates':duplicates,
            'min':minimum,'max':maximum,'sha256':digest(path),'bytes':path.stat().st_size}

def sample(source: pathlib.Path,dest: pathlib.Path,n: int,dtype: str,mode: str,seed: int) -> dict:
    total,w=header(source,dtype)
    if not 0<n<=total:raise ValueError('sample size must be in [1, count]')
    if dest.exists():raise FileExistsError(f'refusing to overwrite {dest}')
    rng=random.Random(seed)
    if mode=='uniform':indices=sorted(rng.sample(range(total),n))
    elif mode=='strided':indices=[(2*i+1)*total//(2*n) for i in range(n)]
    else:
        start=rng.randrange(total-n+1);indices=range(start,start+n)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm,dest.open('wb') as out:
        out.write(struct.pack('<Q',n))
        for i in indices:out.write(mm[8+i*w:8+(i+1)*w])
    result={'source':str(source),'source_count':total,'source_sha256':digest(source),'mode':mode,'seed':seed,
            'sample_count':n,'dtype':dtype,'sample_sha256':digest(dest),
            'warning':'A derived sample is NOT the full benchmark corpus; sampling changes local hardness.'}
    dest.with_name(dest.name+'.manifest.json').write_text(json.dumps(result,indent=2));return result

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('inspect');q.add_argument('path',type=pathlib.Path);q.add_argument('--dtype',choices=['uint32','uint64'],default='uint64')
    q=sub.add_parser('sample');q.add_argument('source',type=pathlib.Path);q.add_argument('dest',type=pathlib.Path);q.add_argument('--n',type=int,required=True);q.add_argument('--dtype',choices=['uint32','uint64'],default='uint64');q.add_argument('--mode',choices=['uniform','strided','window'],default='uniform');q.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.action=='inspect':result=inspect(a.path,a.dtype)
    else:result=sample(a.source,a.dest,a.n,a.dtype,a.mode,a.seed)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
