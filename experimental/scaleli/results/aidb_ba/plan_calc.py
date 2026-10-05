# Wall-clock and resolution calculator for the 200M before/after protocol (stdlib only).
import math
SIG=6.5   # % per run, within a fixed configuration (pooled A/A at 200M: 6.4%, df=7; verify/B 4.9%, df=8)
def tq(df):  # two-sided 97.5% t quantile
    T={1:12.71,2:4.30,3:3.18,4:2.78,5:2.57,6:2.45,7:2.36,8:2.31,9:2.26,10:2.23,12:2.18,15:2.13,19:2.09,20:2.09,24:2.06,26:2.06,30:2.04,39:2.02,49:2.01}
    return T.get(df, 1.96 if df>60 else min(v for k,v in T.items() if k<=df))
def halfwidth(n,sig=SIG): return tq(n-1)*sig*math.sqrt(2/n)          # paired contrast, 95% CI half-width
def mde(n,sig=SIG): return (tq(n-1)+0.84)*sig*math.sqrt(2/n)          # 80% power, alpha .05 two-sided
def pooled_hw(n,d=10,sig=SIG): return 1.96*sig*math.sqrt(2/(n*d))    # fixed-effect pooled over d datasets
print('n  CI±%   MDE80%  pooled10 CI±%')
for n in (1,2,3,4,5,6,8,10,13,20,27):
    print('%2d %6s %7s %8.1f'%(n,'%.1f'%halfwidth(n) if n>1 else 'n/a','%.1f'%mde(n) if n>1 else 'n/a',pooled_hw(n)))
print('one-sample n (brief):',[round(7.85*(SIG/d)**2,1) for d in (5,10,20)],' two-cell contrast n:',[round(2*7.85*(SIG/d)**2,1) for d in (5,10,20)])
# run time model (seconds): fixed load+workload = 31 + 1/Mops ; timed = (ops+warm)/0.75M
def run(build,ops=5e6,warm=1e6,instr=False):
    fixed=31+ops/1e6; timed=(ops+warm)/0.75e6
    return fixed+build+timed+((build+2*ops/0.75e6) if instr else 0)
B=7;N=8;Ccheap=195
Ccap=190+82; Cfull=190+2068; NCmono=5758
cheap=8; exp_=2
def plan(n,Cexp,NCexp,ops=5e6,extra_cold=True,forced=False):
    per_cheap_block=run(B,ops)*2+run(N,ops)+2*run(Ccheap,ops)+(run(N,ops) if forced else 0)
    per_exp_block=run(B,ops)*2+run(N,ops)+run(Cexp,ops)+run(NCexp,ops)+(run(N,ops) if forced else 0)
    instr=cheap*(B+N+2*Ccheap+3*2*ops/0.75e6+ (N if forced else 0))+exp_*(B+N+Cexp+NCexp+4*2*ops/0.75e6)
    cold=10*run(B,ops) if extra_cold else 0
    tot=n*(cheap*per_cheap_block+exp_*per_exp_block)+instr+cold
    return tot/3600, per_cheap_block, per_exp_block
for name,args in [('cut-down n=4, root cap 0.4',(4,Ccap,Ccap)),('cut-down n=5, root cap 0.4',(5,Ccap,Ccap)),
                  ('cut-down n=4, cap, +N_forced',(4,Ccap,Ccap,5e6,True,True)),
                  ('full n=20 cheap, alpha 4 everywhere (planet/osm C,NC n=20)',(20,Cfull,Cfull)),
                  ]:
    h,a,b=plan(*args); print('%-60s total %.1f h  (cheap block %.0f s, expensive block %.0f s)'%(name,h,a,b))
# full plan with split repeats: cheap cells n=20 everywhere, planet/osm C,NC n=5 at alpha 4
nc=20;ne=5
cheapblk=2*run(B)+run(N)+2*run(Ccheap)+run(N)          # + N_forced
exp_cheap=3*run(B)+run(N)                                # B,B',N,N_forced... (B twice + N + Nf)
exp_cheap=2*run(B)+2*run(N)
expC=run(Cfull)+run(Cfull)
instr=8*(B+N+N+2*Ccheap+5*2*5e6/0.75e6)+2*(B+2*N+2*Cfull+5*2*5e6/0.75e6)
tot=8*nc*cheapblk+2*(nc*exp_cheap+ne*expC)+instr+10*run(B)
print('FULL: cheap datasets n=20 (B,B\',N,Nf,C,NC); planet/osm cheap cells n=20, C/NC n=5 at alpha 4: %.1f h'%(tot/3600))
print('  of which planet/osm C,NC: %.1f h; per-run C at alpha4 %.0f s, cheap C %.0f s, B %.0f s'%(2*ne*expC/3600,run(Cfull),run(Ccheap),run(B)))
print('  planet NC with monotone flow (measured 5,758 s build), one structure run instrumented: %.1f h'%(run(NCmono,instr=True)/3600))
