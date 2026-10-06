// Count-only checks of FENCE(g=32) on full 200M: last segments, bucket-size tail, depth>=3, u32-span fallback and tie cost.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
using u64=uint64_t;
struct Seg{u64 a,e;double s;};
static std::vector<Seg> pla(const u64*x,u64 n,double eps){ std::vector<Seg> S; u64 a=0;
  while(a<n){ double lo=-INFINITY,hi=INFINITY; u64 j=a+1;
    for(;j<n;j++){ u64 dxi=x[j]-x[a]; double r=double(j-a); double dx=double(dxi); double nl=std::max(lo,(r-eps)/dx), nh=std::min(hi,(r+eps)/dx); if(nl>nh) break; lo=nl;hi=nh;}
    double s=(j-a<=1)?0.0:(std::isinf(lo)?(std::isinf(hi)?0.0:hi):(std::isinf(hi)?lo:0.5*(lo+hi))); if(s<0)s=0; S.push_back({a,j,s}); a=j;} return S;}
int main(int argc,char**argv){ setvbuf(stdout,0,_IONBF,0); int fd=open(argv[1],O_RDONLY); off_t sz=lseek(fd,0,SEEK_END); const u64*m=(const u64*)mmap(0,sz,PROT_READ,MAP_PRIVATE,fd,0); u64 n=m[0]; const u64*x=m+1; double eps=atof(argv[2]); double g=32;
  auto S=pla(x,n,eps); printf("%s n=%llu segs=%zu min=%llu max=%llu\n",argv[1],(unsigned long long)n,S.size(),(unsigned long long)x[0],(unsigned long long)x[n-1]);
  for(size_t k=S.size()-3;k<S.size();k++) printf("  seg a=%llu e=%llu cnt=%llu s=%.3e x0=%llu xl=%llu\n",(unsigned long long)S[k].a,(unsigned long long)S[k].e,(unsigned long long)(S[k].e-S[k].a),S[k].s,(unsigned long long)x[S[k].a],(unsigned long long)x[S[k].e-1]);
  u64 ambK=0; u64 maxb=0, kd3=0, kd2=0, wideB=0, wideK=0, tieExtra=0, tieMax=0, emptyB=0, totB=0; std::vector<u64> hist(8,0);
  for(auto&sg:S){ u64 a=sg.a,c=sg.e-sg.a,x0=x[a]; u64 i=0;
    // keys sorted, buckets nondecreasing
    u64 prevb=0; bool firstb=true;
    while(i<c){ u64 b=(u64)(sg.s*double(x[a+i]-x0)/g); u64 e=i; while(e<c && (u64)(sg.s*double(x[a+e]-x0)/g)==b) e++;
      if(!firstb && b>prevb+1) emptyB+=b-prevb-1; if(firstb) emptyB+=b; firstb=false; prevb=b; totB++;
      u64 nb=e-i; maxb=std::max(maxb,nb); int d=1; u64 lv=(nb+55)/56; while(lv>1){lv=(lv+13)/14; d++;} if(d>=3) kd3+=nb; if(d==2) kd2+=nb; hist[std::min(d,7)]+=nb;
      // u32 span fallback at leaf level: per leaf of <=56 keys (leaves of the bucket)
      for(u64 L=i; L<e; L+=56){ u64 Le=std::min(e,L+56); u64 base=x[a+L]; u64 span=x[a+Le-1]-base; if(span>=(1ULL<<32)){ wideB++; wideK+=Le-L; int sh=64-__builtin_clzll(span)-32;
          // chunk first keys of 4-key chunks; count max run of equal shifted fences
          u64 run=1, mx=1; u64 prev=~0ULL; for(u64 q=L;q<Le;q+=4){ u64 f=(x[a+q]-base)>>sh; if(f==prev){run++; mx=std::max(mx,run);} else run=1; prev=f;}
          tieMax=std::max(tieMax,mx); if(mx>2) tieExtra+= (Le-L);
          for(u64 q=L;q<Le;q++){ u64 v=(x[a+q]-base)>>sh; u64 c=(q-L)/4; bool amb=false; for(u64 cc=1; cc*4<(Le-L); cc++){ u64 f=(x[a+L+cc*4]-base)>>sh; if(f==v && !(q==L+cc*4)) amb=true; if(f==v && cc!=c && cc!=c+1) amb=true;} if(amb) ambK++; } } }
      i=e; } }
  printf("  g=32 buckets(nonempty)=%llu empty=%llu maxkeys/bucket=%llu keys depth2=%.4f depth>=3=%.6f wide-leaves=%llu keys=%llu maxTieRun=%llu keysInLeavesWithTie>2=%llu\n",(unsigned long long)totB,(unsigned long long)emptyB,(unsigned long long)maxb,double(kd2)/n,double(kd3)/n,(unsigned long long)wideB,(unsigned long long)wideK,(unsigned long long)tieMax,(unsigned long long)tieExtra);
  printf("  ambiguous-after-shift keys=%.5f\n",double(ambK)/n); printf("  depth hist:"); for(int d=1;d<8;d++) printf(" %d:%.5f",d,double(hist[d])/n); printf("\n"); }
