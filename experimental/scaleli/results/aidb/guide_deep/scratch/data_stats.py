import mmap, struct, random, json, array, sys, os
G='data/external/gre/'; S='data/samples/'
names=['books','fb','osm','covid','genome','history','libio','planet','stack','wise']
unsorted={'covid','genome','history','libio','planet','stack','wise'}
total=200_000_000; n=2_000_000
start=random.Random(42).randrange(total-n+1)
print('window start offset (random.Random(42).randrange(198000001)) =',start, 'end =',start+n-1)
u_idx=sorted(random.Random(42).sample(range(total),n)); print('uniform first 5 indices',u_idx[:5],'last',u_idx[-3:], 'mean gap', (u_idx[-1]-u_idx[0])/(n-1))
def key_at(mm,i): return struct.unpack_from('<Q',mm,8+8*i)[0]
out={}
for d in names:
    path=G+d+('.sorted' if d in unsorted else '')
    with open(path,'rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
        cnt=struct.unpack_from('<Q',mm,0)[0]
        qs=[0,0.001,0.01,0.1,0.25,0.5,0.75,0.9,0.99,0.999,0.9999,1.0]
        pct={q:key_at(mm,min(cnt-1,int(q*(cnt-1)))) for q in qs}
        w0=key_at(mm,start); w1=key_at(mm,start+n-1)
        # last 5 keys
        tail=[key_at(mm,cnt-1-i) for i in range(5)][::-1]; head=[key_at(mm,i) for i in range(5)]
        out[d]={'count':cnt,'pct':pct,'window_first':w0,'window_last':w1,'head':head,'tail':tail}
    # window manifest check
    with open(S+f'{d}_2M_window_s42','rb') as f:
        b=f.read(16); c=struct.unpack_from('<Q',b,0)[0]; first=struct.unpack_from('<Q',b,8)[0]
        f.seek(8+8*(n-1)); last=struct.unpack('<Q',f.read(8))[0]
    out[d]['window_sample_first']=first; out[d]['window_sample_last']=last; out[d]['window_matches']=(first==w0 and last==w1)
    print(d, 'window matches offset:', out[d]['window_matches'], 'first',w0,'last',w1)
# fb and books outlier scans (full array read)
for d,thr in (('fb',[1<<40,1<<48,1<<56,1<<63]),('books',[1<<40,1<<48,1<<56,1<<62,1<<63])):
    a=array.array('Q'); 
    with open(G+d,'rb') as f: f.seek(8); a.fromfile(f,total)
    if sys.byteorder!='little': a.byteswap()
    res={}
    for t in thr:
        # binary search since sorted
        lo,hi=0,total
        while lo<hi:
            m=(lo+hi)//2
            if a[m]>=t: hi=m
            else: lo=m+1
        res[str(t)]=total-lo
    out[d]['count_ge']=res; print(d,'keys >= thresholds',res)
    # largest gaps
    top=[]
    for i in range(1,total):
        g=a[i]-a[i-1]
        if g>0 and (len(top)<5 or g>top[0][0]):
            top.append((g,i)); top.sort(); top=top[-5:]
    out[d]['top_gaps']=top; print(d,'top gaps (gap, index)',top)
    del a
json.dump(out,open('results/aidb/guide_deep/scratch/data_stats.json','w'),indent=1)
