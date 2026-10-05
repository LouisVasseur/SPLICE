import os as _os  # repo root: $SPLICE_ROOT, else the nearest ancestor holding walkthrough.py
REPO = _os.environ.get("SPLICE_ROOT") or next(p for p in (_os.path.dirname(_os.path.abspath(__file__)).rsplit("/", i)[0] for i in range(12)) if _os.path.exists(p + "/walkthrough.py"))
import math, struct
w=open(REPO + '/experimental/scaleli/results/aidb/flows/fb_2D2H2L.txt').read().split()
in_dim,hidden,layers=int(w[0]),int(w[1]),int(w[2]); mean,var=float(w[3]),float(w[4])
i=5; mats=[]
for l in range(layers):
    r,c=int(w[i]),int(w[i+1]); i+=2
    m=[float(v) for v in w[i:i+r*c]]; i+=r*c; mats.append((r,c,m))
print('header',in_dim,hidden,layers,'mean',mean,'var',var)
for l,(r,c,m) in enumerate(mats): print('W',l,r,'x',c,m)
b=open(REPO + '/experimental/scaleli/data/samples/fb_2M_uniform_s42','rb').read()
n=struct.unpack_from('<Q',b)[0]; keys=struct.unpack_from(f'<{n}Q',b,8)
print('keys',n,'min',keys[0],'max',keys[-1],'span',keys[-1]-keys[0],'span/64',(keys[-1]-keys[0])/64)
def fwd(key,trace=False):
    x=(key-mean)/var; a=[x, x-math.floor(x)]
    if trace: print('  x=(key-mean)/var =',repr(x),' frac =',repr(a[1]))
    for l,(r,c,m) in enumerate(mats):
        bb=[sum(a[rr]*m[rr*c+cc] for rr in range(r)) for cc in range(c)]
        last=(l+1==layers)
        a=[v if last else math.tanh(v) for v in bb]
        if trace: print(f'  layer {l}: pre-activation {bb!r} -> {"linear" if last else "tanh"} {a!r}')
    z=sum(a)
    if trace: print('  z =',repr(z))
    return z
for k in (keys[0], keys[1000000], keys[-1]):
    print('key',k); z=fwd(k,True)
# monotone check and z range
zs=[fwd(k) for k in keys[::1000]]
print('z range on every 1000th key',min(zs),max(zs),'monotone',all(zs[i]<zs[i+1] for i in range(len(zs)-1)))
# raw normalized
print('raw phi for k[1000000] =',(keys[1000000]-keys[0])/(keys[-1]-keys[0]))
