# Count (not time) cache/TLB events of std::lower_bound over n 16-B records on a Goldmont-like hierarchy.
# L1D 24KiB 6-way 64 sets; L2 2MiB 16-way (2 MiB per 2-core module, sibling idle); uTLB 32 FA; STLB 512 4-way (4K)
# or 32-entry 2M DTLB 4-way. Page-walk loads (PTE/PDE lines) go through L1/L2 too. LRU everywhere. Physical
# frames randomised per page (L2 set index uses bits above 4K).
import random, sys
from collections import OrderedDict
class SA:
    def __init__(s, sets, ways): s.sets=[OrderedDict() for _ in range(sets)]; s.ns=sets; s.w=ways
    def access(s, tag, idx):
        st=s.sets[idx % s.ns]
        if tag in st: st.move_to_end(tag); return True
        st[tag]=1
        if len(st)>s.w: st.popitem(last=False)
        return False
def run(n=200_000_000, rec=16, huge=False, lookups=300_000, warm=200_000, seed=1):
    rnd=random.Random(seed)
    L1=SA(64,6); L2=SA(2048,16)
    pg = 21 if huge else 12
    uT=SA(1,32) if not huge else None
    ST=SA(128,4) if not huge else SA(8,4)
    frame={}
    def phys_line(vline):
        vpage = (vline*64) >> pg
        f = frame.setdefault(vpage, rnd.getrandbits(40))
        off = (vline*64) & ((1<<pg)-1)
        return (f<<pg | off) >> 6
    stats=dict(L1=0,L2=0,MEM=0,TLBmiss=0,walkMEM=0,walkL2=0,steps=0)
    def mem(pl, count, key_prefix=''):
        if L1.access(pl, pl & 63): 
            if count: stats[key_prefix+'L1' if not key_prefix else 'x']=stats.get(key_prefix+'L1',0)+1
            return 'L1'
        hit2 = L2.access(pl, (pl>>0) & 2047 if False else hash((pl*0x9E3779B97F4A7C15)&((1<<64)-1)))
        return 'L2' if hit2 else 'MEM'
    def translate(vaddr, count):
        vpage = vaddr >> pg
        if uT is not None and uT.access(vpage, 0): return
        if ST.access(vpage, vpage):
            return
        if count: stats['TLBmiss']+=1
        # walk: PTE (4K) or PDE (2M) line; table entry address in a separate region
        entry_line = (1<<45 | (vpage*8)) >> 6
        r = mem(entry_line, False)
        if count:
            if r=='MEM': stats['walkMEM']+=1
            elif r=='L2': stats['walkL2']+=1
        if not huge:  # PDE too (assume PML4/PDPT in paging-structure caches)
            pde_line = (1<<46 | ((vpage>>9)*8)) >> 6
            r2 = mem(pde_line, False)
            if count and r2=='MEM': stats['walkMEM']+=1
    base=1<<30
    for it in range(warm+lookups):
        count = it>=warm
        i = rnd.randrange(n)
        lo, length = 0, n
        last=None
        while length>0:
            half=length>>1; mid=lo+half
            va = base + mid*rec
            translate(va, count)
            vl = va>>6
            pl = phys_line(vl)
            r = mem(pl, False)
            if count:
                stats['steps']+=1; stats[r]+=1
            if mid < i: lo=mid+1; length-=half+1
            else: length=half
        # final found-check touches data_[lo], same line as last probe typically
    for k in list(stats):
        stats[k]=round(stats[k]/lookups,3)
    return stats
if __name__=='__main__':
    huge = len(sys.argv)>1 and sys.argv[1]=='huge'
    print('huge' if huge else '4K', run(huge=huge))
