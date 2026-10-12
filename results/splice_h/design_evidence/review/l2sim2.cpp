// Count-only: shared L2 LRU residency for the SPLICE-RO FENCE path (fb-like, g=32, 20.3 B/key).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
using u64=uint64_t;
struct Cache{ int sets,ways; std::vector<u64> tag; std::vector<u64> age; u64 t=0;
  Cache(u64 bytes,int w):ways(w){ sets=bytes/64/w; tag.assign((u64)sets*w,~0ULL); age.assign((u64)sets*w,0);}
  bool access(u64 line){ u64 s=(line*0x9E3779B97F4A7C15ULL>>20)%sets; u64*T=&tag[s*ways]; u64*A=&age[s*ways]; t++;
    int lru=0; for(int i=0;i<ways;i++){ if(T[i]==line){A[i]=t;return true;} if(A[i]<A[lru])lru=i;} T[lru]=line;A[lru]=t;return false;}};
int main(int argc,char**argv){
  u64 n=200000000ULL; double K1=atof(argv[1]); u64 l2=atoll(argv[2]); int hp=atoi(argv[3]); int direct=atoi(argv[4]); int merged=atoi(argv[5]);
  // regions (line ids distinct by high bits)
  const u64 R_KK=1ULL<<56, R_REC=2ULL<<56, R_BPTE=3ULL<<56, R_BKT=4ULL<<56, R_DPTE=5ULL<<56, R_DATA=6ULL<<56, R_ROUT=7ULL<<56;
  u64 nb=n/32; double dataB=18.3; // data bytes per key (20.3 total - 2 bucket)
  Cache c(l2,16); std::mt19937_64 rng(42);
  double m[8]={0}; u64 N=3000000, W=1000000;
  for(u64 q=0;q<N+W;q++){ u64 i=rng()%n; double f=double(i)/n; u64 seg=(u64)(f*K1);
    bool cnt=q>=W; int k=0; bool h;
    h=c.access(R_ROUT+(u64)(f*256)); if(cnt&&!h)m[0]++;           // 16 KB router top
    if(!merged){h=c.access(R_KK+seg/8); if(cnt&&!h)m[1]++;}                      // knot keys 8 per line
    h=c.access(R_REC+seg/4); if(cnt&&!h)m[2]++;                     // 16-B records
    u64 b=i/32; u64 bpage=b*64/4096;
    if(!hp && !direct){ h=c.access(R_BPTE+bpage/8); if(cnt&&!h)m[3]++; }       // PTE of bucket page
    if(!direct){h=c.access(R_BKT+b); if(cnt&&!h)m[4]++;}
    u64 dbyte=(u64)(double(i)*dataB); u64 dpage=dbyte/4096;
    if(!hp){ h=c.access(R_DPTE+dpage/8); if(cnt&&!h)m[5]++; } else { h=c.access(R_DPTE+dbyte/(2<<20)/8); if(cnt&&!h)m[5]++; }
    h=c.access(R_DATA+dbyte/64); if(cnt&&!h)m[6]++;
  }
  printf("K1=%g L2=%llu hp=%d miss/lookup: router %.3f knotkey %.3f record %.3f bucketPTE %.3f bucket %.3f dataPTE %.3f data %.3f\n",K1,(unsigned long long)l2,hp,m[0]/N,m[1]/N,m[2]/N,m[3]/N,m[4]/N,m[5]/N,m[6]/N);
}
