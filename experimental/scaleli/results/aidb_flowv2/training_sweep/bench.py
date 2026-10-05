import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import sys, time, random
sys.path.insert(0,REPO + '/experimental/scaleli/tools')
import train_flow as T
from pathlib import Path
t=time.time(); keys=T.read_keys(Path(REPO + '/experimental/scaleli/data/samples/fb_2M_uniform_s42'),'uint64')
print('read',len(keys),time.time()-t)
rng=random.Random(1); tk=sorted(rng.sample(keys,4096))
t=time.time(); f,i=T.train(tk,64,200,0.05,1000000007,1e-3,log=lambda s:None,monotone=True)
dt=time.time()-t; print('200 steps x4096',dt,'-> key-steps/s', 4096*200/dt)
print('nll',i['best_nll'], f.w0, f.w1)
