# Replicates smoothing.hpp loss_at / Sums exactly in Python floats for x=[1,2,3,10,11,12]
import math
x=[1,2,3,10,11,12]
def sums(pts):
    n=len(pts);sx=sum(p for p,_ in pts);sxx=sum(p*p for p,_ in pts);sy=sum(y for _,y in pts);syy=sum(y*y for _,y in pts);sxy=sum(p*y for p,y in pts)
    return dict(n=n,sx=sx,sxx=sxx,sy=sy,syy=syy,sxy=sxy)
def sse(s):
    n=s['n'];cxx=s['sxx']-s['sx']**2/n;cxy=s['sxy']-s['sx']*s['sy']/n;cyy=s['syy']-s['sy']**2/n
    return max(0.0,cyy-cxy*cxy/cxx) if cxx>0 else cyy
seq=[(v,i) for i,v in enumerate(x)]
base=sums(seq);print('base sums',base,'SSE_before',sse(base))
# OLS check
n=6;mx=sum(x)/n;my=2.5;w=sum((a-mx)*(b-my) for a,b in seq)/sum((a-mx)**2 for a in x);b=my-w*mx
print('OLS w=',w,'b=',b,'SSE direct=',sum((w*a+b-y)**2 for a,y in seq))
suf_x=[0]*(len(seq)+1);suf_y=[0]*(len(seq)+1)
for i in range(len(seq)-1,-1,-1):suf_x[i]=suf_x[i+1]+seq[i][0];suf_y[i]=suf_y[i+1]+i
def loss_at(i,xv):
    s=dict(base);cnt=len(seq)-(i+1)
    s['sy']+=cnt;s['syy']+=2*suf_y[i+1]+cnt;s['sxy']+=suf_x[i+1]
    s['n']+=1;s['sx']+=xv;s['sxx']+=xv*xv;s['sy']+=i+1;s['syy']+=(i+1)**2;s['sxy']+=xv*(i+1)
    return sse(s)
def brute(i,xv):
    pts=[(v,r) for r,v in enumerate([p for p,_ in seq[:i+1]]+[xv]+[p for p,_ in seq[i+1:]])]
    return sse(sums(pts))
print('gap i (lo,hi)   e      loss(lo+e)   loss(lo+2e)  loss(hi-2e)  loss(hi-e)   brute(lo+e) interior?')
for i in range(5):
    lo,hi=seq[i][0],seq[i+1][0];w_=hi-lo;e=w_*1e-3
    l0,l1,r1,r0=loss_at(i,lo+e),loss_at(i,lo+2*e),loss_at(i,hi-2*e),loss_at(i,hi-e)
    print(f'{i} ({lo},{hi})  {e:.4f}  {l0:.6f}  {l1:.6f}  {r1:.6f}  {r0:.6f}  {brute(i,lo+e):.6f}  {l1<l0 and r1<r0}')
# ternary in gap 2
i=2;lo,hi=3,10;e=7e-3;a,b_=lo+e,hi-e
for it in range(40):
    if b_-a<=e:break
    m1=a+(b_-a)/3;m2=b_-(b_-a)/3
    if loss_at(i,m1)<loss_at(i,m2):b_=m2
    else:a=m1
cx=(a+b_)/2;print('ternary gap 2 ->',cx,'loss',loss_at(i,cx),'iterations',it)
# exact minimiser analytically by fine grid
best=min((loss_at(i,3+k*0.0001) for k in range(1,69990)));print('grid min',best)
print('midpoint 6.5 loss',loss_at(2,6.5))
# after insertion slots
seq2=[(1,0),(2,1),(3,2),(cx,3),(10,4),(11,5),(12,6)];print('SSE after',sse(sums(seq2)))
