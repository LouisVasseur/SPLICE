import math
def tcd(xs,amp=1.5,tail=0.99,trace=False):
    n=len(xs); mx=sum(xs)/n; my=(n-1)/2
    xx=sum((x-mx)**2 for x in xs); xy=sum((x-mx)*(i-my) for i,x in enumerate(xs))
    slope=xy/xx if xx>0 else 0
    if not slope>0: slope=n/(xs[-1]-xs[0])
    intercept=-slope*xs[0]+0.5
    max_size=max(1,min(int(n*amp),int(max(1.0,math.floor(slope*xs[-1]+intercept)+1))))
    pos=[min(max(int(math.floor(slope*x+intercept)),0),max_size-1) for x in xs]
    counts=[];run=1
    for i in range(1,n):
        if pos[i]==pos[i-1]:run+=1
        else:counts.append(run);run=1
    counts.append(run);counts.sort()
    idx=min(len(counts)-1,max(0,math.ceil(tail*len(counts))-1))
    if trace: print('n',n,'slope',slope,'intercept',intercept,'max_size',max_size,'\npositions',pos,'\ncounts sorted',counts,'m',len(counts),'idx',idx,'-> D99 =',counts[idx]-1)
    return counts[idx]-1
x=[0,1,2,3,4,5,6,7,8,9,9.1,9.2,9.3,9.4,9.5,20,30,40,50,60]
tcd(x,trace=True)
print('uniform 0..99 ->',tcd([float(i) for i in range(100)]))
