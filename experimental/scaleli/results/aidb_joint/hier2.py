import struct,time
exec(open('hier.py').read().split('print("%-8s')[0])

def hier_eqdepth(fx,ps,n,S):
    # equi-depth second level: bucket = floor(i*S/n) over fence RANK, top level is a linear
    # model over the S bucket-boundary keys (a 2-level tree, CSV Alg-2 shaped).
    bnd=[fx[(i*n)//S] for i in range(S)]
    import bisect
    mods=[]
    for bi in range(S):
        lo=(bi*n)//S; hi=((bi+1)*n)//S
        m=list(range(lo,hi))
        if len(m)>=2: mods.append(fit([float(fx[i]) for i in m],[float(i) for i in m]))
        elif len(m)==1: mods.append((0.0,float(m[0])))
        else: mods.append(None)
    tot=0
    for k in ps:
        bi=bisect.bisect_right(bnd,k)-1
        if bi<0: bi=0
        md=mods[bi]
        pred=float(bi*n//S) if md is None else md[0]*float(k)+md[1]
        tot+=locate_count(fx,pred,k,n)
    return tot/len(ps)

print("%-8s %8s %8s %8s %8s | %8s %8s"%("ds","h/64","h/128","h/256","h/512","eqd/64","eqd/256"))
for d in DS:
    keys=load(d); fx=fences(keys); n=len(fx); ps=probe_keys(fx)
    r=[hierarchical(fx,ps,n,S) for S in (64,128,256,512)]
    e=[hier_eqdepth(fx,ps,n,S) for S in (64,256)]
    print("%-8s "%d+" ".join("%8.3f"%v for v in r)+" | "+" ".join("%8.3f"%v for v in e))

# build cost: time the S=64 two-level fit vs one CSV greedy pass at 4x on the same fences
keys=load('planet'); fx=fences(keys); n=len(fx); ps=probe_keys(fx)
t=time.perf_counter()
for _ in range(20): hierarchical(fx,ps,n,64)
print("planet hierarchical fit+score x20: %.3f s"%(time.perf_counter()-t))
