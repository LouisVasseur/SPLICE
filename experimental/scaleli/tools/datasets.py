#!/usr/bin/env python3
"""Discover, download, validate and sample integer benchmark data. No implicit downloads."""
from __future__ import annotations
import argparse, array, datetime, hashlib, json, mmap, pathlib, random, shutil, struct, subprocess, sys, tempfile, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]

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

def fetch(entry: dict,dest: pathlib.Path,max_output: int) -> dict:
    width=8 if entry['dtype']=='uint64' else 4;expected=8+entry['count']*width
    if expected>max_output:raise ValueError(f'{expected} output bytes exceed --max-output-bytes={max_output}')
    dest.mkdir(parents=True,exist_ok=True);final=dest/entry['filename']
    if final.exists():raise FileExistsError(f'refusing to overwrite {final}')
    with tempfile.TemporaryDirectory(prefix='scaleli-download-',dir=dest) as temp:
        temp=pathlib.Path(temp);download=temp/'download';raw=temp/'raw'
        req=urllib.request.Request(entry['url'],headers={'User-Agent':'SCALE-LI-research/0.1'})
        # TLS verification remains ON, unlike an older upstream shell script.
        with urllib.request.urlopen(req,timeout=30) as response,download.open('wb') as out:
            total=0
            while b:=response.read(1024*1024):
                total+=len(b)
                if total>max_output:raise ValueError('download exceeds configured safety cap')
                out.write(b)
        if entry['compression']=='zstd':
            if not shutil.which('zstd'):raise RuntimeError('install zstd for this dataset, then retry')
            with raw.open('wb') as out:
                proc=subprocess.Popen(['zstd','-d','-c',str(download)],stdout=subprocess.PIPE)
                assert proc.stdout is not None
                total=0
                try:
                    while b:=proc.stdout.read(1024*1024):
                        total+=len(b)
                        if total>max_output:raise ValueError('decompressed data exceeds safety cap')
                        out.write(b)
                    if proc.wait()!=0:raise RuntimeError('zstd decompression failed')
                finally:
                    if proc.poll() is None:proc.kill();proc.wait()
        else:download.replace(raw)
        count,_=header(raw,entry['dtype'])
        if count!=entry['count']:raise ValueError('manifest count differs from downloaded data')
        if entry.get('md5_uncompressed') and digest(raw,'md5')!=entry['md5_uncompressed']:raise ValueError('upstream MD5 mismatch')
        result={'dataset':entry,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'sha256':digest(raw),'bytes':raw.stat().st_size,'note':'File length and checksum validated; run inspect for full sortedness audit.'}
        raw.replace(final);final.with_name(final.name+'.manifest.json').write_text(json.dumps(result,indent=2))
        return result

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
    sub.add_parser('list')
    q=sub.add_parser('fetch');q.add_argument('id');q.add_argument('--dest',type=pathlib.Path,default=pathlib.Path('data/external'));q.add_argument('--acknowledge-source',action='store_true',help='Confirm you reviewed the upstream data source and terms.');q.add_argument('--max-output-bytes',type=int,default=2_000_000_000)
    q=sub.add_parser('inspect');q.add_argument('path',type=pathlib.Path);q.add_argument('--dtype',choices=['uint32','uint64'],default='uint64')
    q=sub.add_parser('sample');q.add_argument('source',type=pathlib.Path);q.add_argument('dest',type=pathlib.Path);q.add_argument('--n',type=int,required=True);q.add_argument('--dtype',choices=['uint32','uint64'],default='uint64');q.add_argument('--mode',choices=['uniform','strided','window'],default='uniform');q.add_argument('--seed',type=int,default=42)
    a=p.parse_args();manifest=json.loads((ROOT/'configs/datasets.json').read_text())
    if a.action=='list':result=manifest
    elif a.action=='fetch':
        if not a.acknowledge_source:p.error('review the upstream source and pass --acknowledge-source to download')
        entries={x['id']:x for x in manifest['datasets']}
        if a.id not in entries:p.error('unknown dataset id; use list')
        result=fetch(entries[a.id],a.dest,a.max_output_bytes)
    elif a.action=='inspect':result=inspect(a.path,a.dtype)
    else:result=sample(a.source,a.dest,a.n,a.dtype,a.mode,a.seed)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
