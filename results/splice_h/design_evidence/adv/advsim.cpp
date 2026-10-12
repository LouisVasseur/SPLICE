// Full-scale (200M) count of the SPLICE-MLP coarse model + directory occupancy. Counts only, no timing.
// Coarse model: exceptions (isolated extreme keys) + dyadic radix trie (L1 table 2^r1, children with own shift,
// depth<=3) whose leaves hold a fixed-point linear map [k_first,k_last] -> [rank_first, rank_last+1].
// Baseline: equal-count knots (every ks keys) with exact interpolation (no routing cost counted).
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string>
typedef unsigned __int128 u128;
static int bitlen(uint64_t v){ return v?64-__builtin_clzll(v):0; }
struct Leaf{ uint64_t kf; uint64_t slope; uint32_t base; int pre; }; // d = base + ((x-kf)*slope)>>64  (rank units scaled later)
const uint64_t* X; size_t n;
std::vector<Leaf> leaves; std::vector<int> leafdepth; size_t tables_entries=0;
size_t T; int R1=11;
std::vector<uint32_t> keyleaf; // leaf id per key
std::vector<uint8_t> keydepth;
void build(size_t a,size_t b,uint64_t lo,int shift,int bits,int depth){
  // keys X[a,b) lie in [lo, lo + 2^(shift+bits)); split into 2^bits buckets of width 2^shift
  tables_entries += (size_t)1<<bits;
  size_t i=a; uint64_t nb=(uint64_t)1<<bits;
  for(uint64_t bk=0;bk<nb && i<b;bk++){
    uint64_t blo=lo+(bk<<shift); uint64_t bhi_ex = (shift+0<64)? blo+(((uint64_t)1)<<shift)-1 : ~0ULL;
    size_t j=i; while(j<b && X[j]<=bhi_ex && X[j]>=blo) j++;
    // fast path: use upper_bound
    j = std::upper_bound(X+i,X+b,bhi_ex)-X;
    size_t c=j-i;
    if(c==0) continue;
    if(c>T && depth<3 && shift>0){
      int need=bitlen((c+T-1)/T)+1; if(need>12) need=12; if(need>shift) need=shift;
      build(i,j,blo,shift-need,need,depth+1);
    } else {
      Leaf L; L.kf=X[i]; L.base=(uint32_t)i; uint64_t span=X[j-1]-X[i];
      // slope in rank units per key unit, scaled 2^32 stored in 64 bits: (c)/(span+1)
      L.pre = bitlen(span)>32 ? bitlen(span)-32 : 0;
      L.slope = (uint64_t)(((u128)c<<32)/((u128)(span>>L.pre)+1));
      for(size_t t=i;t<j;t++){ keyleaf[t]=leaves.size(); keydepth[t]=depth; }
      leaves.push_back(L);
    }
    i=j;
  }
}
int main(int argc,char**argv){
  const char* path=argv[1]; T=atol(argv[2]);
  int fd=open(path,O_RDONLY); off_t sz=lseek(fd,0,SEEK_END);
  void* m=mmap(0,sz,PROT_READ,MAP_PRIVATE,fd,0); X=(const uint64_t*)m+1; n=(sz-8)/8;
  // exceptions: peel isolated extremes
  size_t lo=0,hi=n; int ex=0;
  { // choose up to 64 top and 64 bottom exceptions: smallest counts whose removal gets the span within 2x of the 64/64-trimmed span
    uint64_t best=X[n-65]-X[64]; size_t bt=64,bb=64;
    for(size_t t=0;t<=64;t++) if(X[n-1-t]-X[64] <= 2*best){ bt=t; break; }
    for(size_t b=0;b<=64;b++) if(X[n-1-bt]-X[b] <= 2*best){ bb=b; break; }
    lo=bb; hi=n-bt; ex=(int)(bb+bt); }
  keyleaf.assign(n,0); keydepth.assign(n,0);
  uint64_t base=X[lo]; uint64_t span=X[hi-1]-base; int sb=bitlen(span); int shift= sb>R1? sb-R1:0; int bits= sb>R1?R1:sb;
  build(lo,hi,base,shift,bits,0);
  // F per key (rank units) and directory occupancy
  double cbars[2]={24,32}; size_t Cs[3]={52,104,236};
  double d1=0,d2=0,d3=0; for(size_t t=lo;t<hi;t++){ if(keydepth[t]==0)d1++; else if(keydepth[t]==1)d2++; else d3++; }
  double nb=hi-lo;
  printf("{\"file\":\"%s\",\"T\":%zu,\"exceptions\":%d,\"leaves\":%zu,\"table_entries\":%zu,\"leaf_KB\":%.1f,\"table_KB_8B\":%.1f,\"depth_frac\":[%.4f,%.4f,%.4f]",
    path,T,ex,leaves.size(),tables_entries,leaves.size()*24/1024.0,tables_entries*8/1024.0,d1/nb,d2/nb,d3/nb);
  // max abs rank error of leaf model
  double se=0; double mx=0;
  std::vector<double> F(n);
  for(size_t t=lo;t<hi;t++){ const Leaf&L=leaves[keyleaf[t]]; double f=L.base + (double)((((X[t]-L.kf)>>L.pre)*L.slope)>>32); F[t]=f; double e=fabs(f-(double)t); se+=e; if(e>mx)mx=e; }
  for(size_t t=0;t<lo;t++) F[t]=t; for(size_t t=hi;t<n;t++) F[t]=t;
  printf(",\"leaf_rank_err_mean\":%.1f,\"leaf_rank_err_max\":%.0f",se/nb,mx);
  for(double cb: cbars){
    size_t Q=(size_t)ceil(n/cb); std::vector<uint32_t> cnt(Q,0); std::vector<uint32_t> h(n);
    for(size_t t=0;t<n;t++){ size_t d=(size_t)(F[t]*Q/n); if(d>=Q)d=Q-1; h[t]=d; cnt[d]++; }
    size_t empty=0; for(auto c:cnt) if(!c) empty++;
    printf(",\"c%.0f\":{\"empty\":%.4f",cb,(double)empty/Q);
    for(size_t C: Cs){ double heavy=0; for(size_t t=0;t<n;t++) if(cnt[h[t]]>C) heavy++; printf(",\"heavy_C%zu\":%.4f",C,heavy/n); }
    printf("}");
  }

  for(double cb: cbars){
    size_t Q=(size_t)ceil(n/cb); std::vector<uint32_t> cnt(Q,0); std::vector<uint32_t> h(n);
    for(size_t t=0;t<n;t++){ size_t d=(size_t)(F[t]*Q/n); if(d>=Q)d=Q-1; h[t]=d; cnt[d]++; }
    uint32_t mxc=0; for(auto c:cnt) if(c>mxc) mxc=c;
    double g68=0,g228=0,g13k=0; for(size_t t=0;t<n;t++){ uint32_t c=cnt[h[t]]; if(c>68)g68++; if(c>228)g228++; if(c>57*228)g13k++; }
    double bf=Q*128.0, bc=Q*128.0, b1=Q*64.0, b1c=Q*64.0;
    for(auto c:cnt){ bf+=ceil(c/4.0)*64; bc+=ceil(c/5.0)*64; b1+=ceil(c/4.0)*64; b1c+=ceil(c/5.0)*64;
      if(c>228){ bf+=ceil(c/228.0)*128; bc+=ceil(c/285.0)*128; } if(c>104){ b1+=ceil(c/104.0)*64; b1c+=ceil(c/130.0)*64; } }
    printf(",\"adv_c%.0f\":{\"max_cnt\":%u,\"keys_cnt_gt68\":%.4f,\"keys_gt228\":%.4f,\"keys_gt13224\":%.5f,\"Bkey_fast_w2\":%.2f,\"Bkey_compact_w2\":%.2f,\"Bkey_fast_w1\":%.2f,\"Bkey_compact_w1\":%.2f}",
      cb,mxc,g68/n,g228/n,g13k/n,bf/n,bc/n,b1/n,b1c/n);
  }
  printf("}\n"); return 0;
  // baseline equal-count knots with same number of segments as leaves
  size_t K=leaves.size(); size_t ks=n/K; 
  for(double cb: cbars){
    size_t Q=(size_t)ceil(n/cb); std::vector<uint32_t> cnt(Q,0); std::vector<uint32_t> h(n);
    for(size_t t=0;t<n;t++){ size_t a=(t/ks)*ks; size_t b=std::min(a+ks,n-1); double f;
      if(b==a) f=a; else f=a + (double)(X[t]-X[a])/(double)(X[b]-X[a])*(double)(b-a);
      size_t d=(size_t)(f*Q/n); if(d>=Q)d=Q-1; h[t]=d; cnt[d]++; }
    printf(",\"eq_c%.0f\":{",cb); bool first=true;
    for(size_t C: Cs){ double heavy=0; for(size_t t=0;t<n;t++) if(cnt[h[t]]>C) heavy++; printf("%s\"heavy_C%zu\":%.4f",first?"":",",C,heavy/n); first=false;}
    printf("}");
  }
  printf("}\n");
}
