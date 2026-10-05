#!/usr/bin/env python3
"""Check CSV equations 6-9, 15-16, 17-21 numerically on the 7-key example (stdlib only)."""
keys=[2,3,4,5,12,13,14]; n=len(keys); y0=list(range(n))
def refit_and_loss(kv):
    # insert kv, ranks after kv shift by one
    yv=sum(1 for k in keys if k<kv)          # rank of the virtual point
    ys=[y+(1 if y>=yv else 0) for y in y0]   # eq. 11/14 shift
    xs=keys+[kv]; ts=ys+[yv]
    m=len(xs); kbar=sum(xs)/m; ybar=sum(ts)/m   # eq. 8, 9 (ybar = (sum y_orig + n)/(n+1))
    w=sum((x-kbar)*(t-ybar) for x,t in zip(xs,ts))/sum((x-kbar)**2 for x in xs)  # eq. 6
    b=ybar-w*kbar                                                                # eq. 7
    # eq. 15 form
    sxy=sum(k*y for k,y in zip(keys,ys)); sxx=sum(k*k for k in keys)
    w15=(sxy+kv*yv-(n+1)*kbar*ybar)/((sxx+kv*kv)-(n+1)*kbar**2)
    loss=sum((w*x+b-t)**2 for x,t in zip(xs,ts))                                 # eq. 5
    return w,b,w15,loss,yv,ys
for kv in (6,9,11):
    w,b,w15,loss,yv,ys=refit_and_loss(kv)
    print(f"kv={kv}: y_v={yv} shifted ranks={ys}  w(eq6)={w:.6f} w(eq15)={w15:.6f} b={b:.6f} loss(eq5)={loss:.6f}")
    # eq 9 check
    print("   eq9 ybar =", (sum(y0)+n)/(n+1), " direct mean =", (sum(ys)+yv)/(n+1))

# derivative via eq. 17-21 vs finite difference, at real-valued kv
def deriv_paper(kv):
    yv=sum(1 for k in keys if k<kv); ys=[y+(1 if y>=yv else 0) for y in y0]
    S_k2=sum(k*k for k in keys); S_ky=sum(k*y for k,y in zip(keys,ys)); kbar_n=sum(keys)/n; ybar_n=sum(ys)/n
    kbar=(sum(keys)+kv)/(n+1); ybar=(sum(ys)+yv)/(n+1)
    A=(n+1)*(S_k2+kv*kv)-((n+1)*kbar)**2          # eq. 20
    B=(n+1)*(S_ky+kv*yv)-(n+1)**2*kbar*ybar        # eq. 21
    w=B/A; b=ybar-w*kbar
    wp=(A*(n*(yv-ybar_n))-B*(2*n*(kv-kbar_n)))/A**2   # eq. 18 as printed
    bp=-(w+(n+1)*kbar*wp)/(n+1)                        # eq. 19
    Lp=2*(wp*(w*S_k2+n*b*kbar_n-S_ky)+n*bp*(w*kbar_n+b-ybar_n)+(w*kv+b-yv)*(wp*kv+w+bp))  # eq. 17
    return Lp, w, b, wp, bp
def loss_cont(kv):
    yv=sum(1 for k in keys if k<kv); ys=[y+(1 if y>=yv else 0) for y in y0]
    xs=keys+[kv]; ts=ys+[yv]; m=len(xs); kbar=sum(xs)/m; ybar=sum(ts)/m
    w=sum((x-kbar)*(t-ybar) for x,t in zip(xs,ts))/sum((x-kbar)**2 for x in xs); b=ybar-w*kbar
    return sum((w*x+b-t)**2 for x,t in zip(xs,ts))
for kv in (6.0, 8.7224, 11.0):
    h=1e-5; fd=(loss_cont(kv+h)-loss_cont(kv-h))/(2*h)
    Lp,w,b,wp,bp=deriv_paper(kv)
    print(f"kv={kv}: finite-diff dL/dkv={fd:+.6f}   eq.17 as printed={Lp:+.6f}   (w'={wp:+.5f}, b'={bp:+.5f})")
# Note: eq. 18's numerator, derived properly: d w/d kv with w=B/A: (A*dB - B*dA)/A^2,
# dB/dkv = (n+1)*yv - (n+1)^2 * ybar * (1/(n+1)) = (n+1)*yv - (n+1)*ybar = (n+1)(yv - ybar)
# dA/dkv = (n+1)*2kv - 2*(n+1)*kbar = 2(n+1)(kv - kbar)
def deriv_rederived(kv):
    yv=sum(1 for k in keys if k<kv); ys=[y+(1 if y>=yv else 0) for y in y0]
    S_k2=sum(k*k for k in keys); S_ky=sum(k*y for k,y in zip(keys,ys))
    kbar=(sum(keys)+kv)/(n+1); ybar=(sum(ys)+yv)/(n+1)
    A=(n+1)*(S_k2+kv*kv)-((n+1)*kbar)**2; B=(n+1)*(S_ky+kv*yv)-(n+1)**2*kbar*ybar
    dA=2*(n+1)*(kv-kbar); dB=(n+1)*(yv-ybar)
    w=B/A; wp=(A*dB-B*dA)/A**2; b=ybar-w*kbar; bp=-(w+(n+1)*kbar*wp)/(n+1)
    # dL/dkv = sum over all m points 2 r_i (w' x_i + b') + 2 r_v * w   (r_v = w kv + b - yv)
    xs=keys+[kv]; ts=ys+[yv]
    return sum(2*(w*x+b-t)*(wp*x+bp) for x,t in zip(xs,ts)) + 2*(w*kv+b-yv)*w
for kv in (6.0, 8.7224, 11.0):
    h=1e-5; fd=(loss_cont(kv+h)-loss_cont(kv-h))/(2*h)
    print(f"kv={kv}: finite-diff={fd:+.6f}  re-derived closed form={deriv_rederived(kv):+.6f}")
