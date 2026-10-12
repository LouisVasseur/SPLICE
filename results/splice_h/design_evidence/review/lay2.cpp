// Count-only simulation of the proposed SPLICE-RO layout at full 200M scale.
// Tier-1: greedy cone PLA (eps1 bisected to fit a knot budget K1) - cache resident.
// Per tier-1 segment: DIRECT (slot = (1+a)*PLA rank, order-preserving placement)
// or RADIX(m): equal-width key sub-buckets, exact slot ranges per sub-bucket (u32 directory), linear interp inside.
// Counts lines touched (16-B slots, 4/line), directory straddles, bytes. No timing.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
using u64=uint64_t;
struct Seg{u64 a,e;double s;};
static std::vector<Seg> pla(const u64*x,u64 n,double eps){
  std::vector<Seg> S; u64 a=0;
  while(a<n){ double lo=-INFINITY,hi=INFINITY; u64 j=a+1;
    for(;j<n;j++){ u64 dxi=x[j]-x[a]; double r=double(j-a);
      if(dxi==0){ if(r>eps) break; else continue;}
      double dx=double(dxi); double nl=std::max(lo,(r-eps)/dx), nh=std::min(hi,(r+eps)/dx);
      if(nl>nh) break; lo=nl;hi=nh;}
    double s = (j-a<=1)?0.0: (std::isinf(lo)? (std::isinf(hi)?0.0:hi) : (std::isinf(hi)?lo:0.5*(lo+hi)));
    if(s<0) s=0;
    S.push_back({a,j,s}); a=j;}
  return S;}
struct Res{double lines=0,straddle=0,bytes=0;};
// place keys [lo,hi) of x into w slots starting at base with predictions pred[]; returns sum lines
static double place(std::vector<int64_t>&pred,int64_t base,int64_t w,u64 cnt,std::vector<int64_t>&fin){
  fin.resize(cnt); int64_t prev=base-1;
  for(u64 i=0;i<cnt;i++){ int64_t f=std::max(pred[i],prev+1); fin[i]=f; prev=f;}
  int64_t nxt=base+w;
  for(int64_t i=int64_t(cnt)-1;i>=0;i--){ if(fin[i]>nxt-1) fin[i]=nxt-1; nxt=fin[i];}
  double L=0; for(u64 i=0;i<cnt;i++){ int64_t a=pred[i]>>2,b=fin[i]>>2; L+= double(std::llabs(a-b)+1);} return L;}

// Recursive line-node mid tier: keys x[a..a+c) with predictor rank r_i (0..), buckets of g predicted ranks,
// a bucket with <=T keys is a leaf (exact room, endpoint line); else a child node (endpoint line over its keys).
struct RecAcc{double lines=0,depth=0,entries=0,slots=0;};
static void rec_node(const u64*x,u64 a,u64 c,const std::vector<double>&r,double g,u64 T,double al,int depth,RecAcc&acc,std::vector<int64_t>&pred,std::vector<int64_t>&fin){
  // r: predicted ranks for keys a..a+c (size c)
  double rmax=0; for(u64 i=0;i<c;i++) if(r[i]>rmax) rmax=r[i];
  u64 M=(u64)(rmax/g)+1; acc.entries+=M;
  std::vector<u64> bk(c); std::vector<u64> cnt(M,0),st(M,0);
  for(u64 i=0;i<c;i++){ double v=r[i]<0?0:r[i]; u64 b=(u64)(v/g); if(b>=M)b=M-1; bk[i]=b; cnt[b]++; }
  u64 i=0;
  while(i<c){ u64 b=bk[i]; u64 e=i; while(e<c&&bk[e]==b) e++; u64 nb=e-i;
    if(nb<=T || depth>=6){ // leaf
      int64_t w=(int64_t)std::ceil((1+al)*nb); double fst=double(x[a+i]),span=double(x[a+e-1]-x[a+i])+1.0;
      pred.resize(nb); for(u64 k=0;k<nb;k++){ double f=double(x[a+i+k]-x[a+i])/span; int64_t p=(int64_t)std::floor(f*w); if(p>w-1)p=w-1; pred[k]=acc.slots>0?0:0; pred[k]=p; }
      // line alignment relative to running slot offset
      int64_t base=(int64_t)acc.slots; for(u64 k=0;k<nb;k++) pred[k]+=base;
      acc.lines+=place(pred,base,w,nb,fin); acc.slots+=w; acc.depth+=double(depth)*nb;
    } else {
      std::vector<double> r2(nb); double span=double(x[a+e-1]-x[a+i]); 
      for(u64 k=0;k<nb;k++) r2[k]= span>0? double(x[a+i+k]-x[a+i])/span*double(nb-1) : double(k);
      rec_node(x,a+i,nb,r2,g,T,al,depth+1,acc,pred,fin);
    }
    i=e; }
}
int main(int argc,char**argv){
  const char* path=argv[1]; double K1=atof(argv[2]);
  int fd=open(path,O_RDONLY); off_t sz=lseek(fd,0,SEEK_END);
  const u64* m=(const u64*)mmap(0,sz,PROT_READ,MAP_PRIVATE,fd,0); u64 n=m[0]; const u64* x=m+1;
  u64 dups=0; for(u64 i=1;i<n;i++) if(x[i]<=x[i-1]) dups++;
  // bisection on eps1 (log scale) for #segments <= K1
  double lo=16,hi=1<<18; std::vector<Seg> S;
  for(int it=0;it<9;it++){ double mid=std::sqrt(lo*hi); auto T=pla(x,n,mid); if(T.size()>K1) lo=mid; else {hi=mid;} }
  S=pla(x,n,hi); double eps1=hi;
  // max rank error check and DIRECT/RADIX eval
  const double alphas[2]={0.25,0.5}; const double fr[5]={0,1.0/32,1.0/8,1.0/2,2.0};
  // per segment costs stored for selection: [alpha][mode] -> (D4k_sum, bytes)
  size_t ns=S.size(); std::vector<double> Dk(ns*10), By(ns*10), St(ns*10), Ln(ns*10), Mb(ns*10,0.0);
  std::vector<int64_t> pred,fin; std::vector<u64> cnt; std::vector<int64_t> R;
  for(size_t j=0;j<ns;j++){ u64 a=S[j].a,e=S[j].e,c=e-a; u64 x0=x[a]; double span=double(x[e-1]-x0)+1.0;
    for(int ai=0;ai<2;ai++){ double al=alphas[ai];
      for(int mi=0;mi<5;mi++){ size_t id=j*10+ai*5+mi; pred.resize(c);
        if(mi==0){ int64_t w=(int64_t)std::ceil((1+al)*c); w=(w+3)&~3LL;
          for(u64 i=0;i<c;i++){ double r=S[j].s*double(x[a+i]-x0); int64_t p=(int64_t)std::floor((1+al)*r); if(p<0)p=0; if(p>w-1)p=w-1; pred[i]=p;}
          Ln[id]=place(pred,0,w,c,fin); St[id]=0; By[id]=16.0*w+16; Dk[id]=Ln[id]+c; }
        else if(mi==1){ u64 M=std::max<u64>(1,c/2); cnt.assign(M,0);
          double inv=double(M)/span; std::vector<u64> bk(c);
          for(u64 i=0;i<c;i++){ u64 b=(u64)(double(x[a+i]-x0)*inv); if(b>=M)b=M-1; bk[i]=b; cnt[b]++; }
          R.assign(M+1,0); for(u64 b=0;b<M;b++){ R[b+1]=R[b]+(int64_t)std::ceil((1+al)*cnt[b]); }
          int64_t W=(R[M]+3)&~3LL; double bw=span/double(M);
          for(u64 i=0;i<c;i++){ u64 b=bk[i]; double lob=double(b)*bw; double f=(double(x[a+i]-x0)-lob)/bw; int64_t w=R[b+1]-R[b];
            int64_t p=R[b]+(int64_t)std::floor(f*w); if(p>R[b+1]-1)p=R[b+1]-1; if(p<R[b])p=R[b]; pred[i]=p;}
          Ln[id]=place(pred,0,W,c,fin); double st=0; for(u64 i=0;i<c;i++) if((bk[i]&15)==15) st+=1;
          St[id]=st; Mb[id]=4.0*(M+1); By[id]=16.0*W+4.0*(M+1)+16; Dk[id]=Ln[id]+c+st+c; }
        else { // FENCE(g): bucket b=floor(r/g) addressed directly; leaf = 64-B fence line (13 u32 fences -> 14 data lines of 4 slots),
               // buckets >56 keys get inner fence lines (fanout 14). Data line = exactly the chunk line.
          const double gs[3]={16,32,48}; double g=gs[mi-2];
          u64 M=0; std::vector<u64> bk(c); for(u64 i=0;i<c;i++){ double r=S[j].s*double(x[a+i]-x0); u64 b=(u64)(r/g); bk[i]=b; if(b+1>M)M=b+1; }
          cnt.assign(M,0); for(u64 i=0;i<c;i++) cnt[bk[i]]++;
          double depth=0, lines_f=0, wide=0; double dataB=0;
          for(u64 b=0;b<M;b++){ u64 nb=cnt[b]; if(!nb){ lines_f+=1; continue; }
            u64 leaves=(nb+55)/56; u64 lev=leaves, tot=leaves, d=1; while(lev>1){ lev=(lev+13)/14; tot+=lev; d++; }
            if(leaves>1) tot-=0; // top inner node occupies the bucket's own line
            lines_f+= double(tot); depth+= double(d)*nb; dataB+= 64.0*double((nb+3)/4 + (leaves-1)); }
          // key span check for u32 fences
          for(u64 i=0,b0=~0ULL;i<c;i++){ (void)b0; }
          Ln[id]=double(c); St[id]=depth; Mb[id]=64.0*lines_f; By[id]=dataB+64.0*lines_f+16; Dk[id]=double(c)+c+depth+0.55*c; }
      }}
  }
  // router: expected knots in radix bucket, r=12 raw, r=16 raw, G-collapsed (top 64 knot gaps) r=12
  auto router=[&](int r,bool g){ std::vector<double> kk(ns); for(size_t j=0;j<ns;j++) kk[j]=0;
    std::vector<long double> c(ns); // coordinate of knot j
    std::vector<u64> gaps(ns>1?ns-1:0); for(size_t j=1;j<ns;j++) gaps[j-1]=x[S[j].a]-x[S[j-1].a];
    u64 cap=~0ULL; if(g && ns>65){ auto t=gaps; std::nth_element(t.begin(),t.end()-64,t.end()); cap=t[t.size()-64]; auto t2=gaps; std::nth_element(t2.begin(),t2.begin()+t2.size()/2,t2.end()); u64 med=t2[t2.size()/2]; 
        // collapse gaps >= cap to med
        long double acc=0; c[0]=0; for(size_t j=1;j<ns;j++){ u64 gp=gaps[j-1]; acc+= (gp>=cap? (long double)med : (long double)gp); c[j]=acc;} }
    else { for(size_t j=0;j<ns;j++) c[j]=(long double)(x[S[j].a]-x[0]); }
    long double tot=c[ns-1]+1; u64 B=1ULL<<r; std::vector<u64> bc(B,0);
    for(size_t j=0;j<ns;j++){ u64 b=(u64)(c[j]/tot*B); if(b>=B)b=B-1; bc[b]++; }
    // each key's candidate count = knots in its bucket + 1 (predecessor); weight by segment size
    double sum=0, slog=0; for(size_t j=0;j<ns;j++){ u64 b=(u64)(c[j]/tot*B); if(b>=B)b=B-1; double k=bc[b]+1; double w=double(S[j].e-S[j].a); sum+=w*k; slog+=w*std::log2(k);} 
    return std::make_pair(sum/n, slog/n);};
  auto r12=router(12,false), r16=router(16,false), g12=router(12,true);
  printf("{\"file\":\"%s\",\"n\":%llu,\"dups\":%llu,\"K1\":%zu,\"eps1\":%.1f,\"router\":{\"raw12_mean\":%.1f,\"raw12_log2\":%.2f,\"raw16_mean\":%.1f,\"raw16_log2\":%.2f,\"g12_mean\":%.1f,\"g12_log2\":%.2f},\"sel\":[",
    path,(unsigned long long)n,(unsigned long long)dups,ns,eps1,r12.first,r12.second,r16.first,r16.second,g12.first,g12.second);
  const double lams[6]={0,0.005,0.02,0.05,0.1,0.3}; bool first=true;
  for(int ai=0;ai<2;ai++) for(double lam: lams){ double D=0,B=0,L=0,mid=0,dkeys=0,midB=0; double st=0;
    for(size_t j=0;j<ns;j++){ int best=0; double bc=1e300; for(int mi=0;mi<5;mi++){ if(mi==1) continue; size_t id=j*10+ai*5+mi; double cst=Dk[id]+lam*By[id]; if(cst<bc){bc=cst;best=mi;} }
      size_t id=j*10+ai*5+best; double c=double(S[j].e-S[j].a); D+=Dk[id]; B+=By[id]; L+=Ln[id]; st+=St[id];
      if(best==0) dkeys+=c; else { mid+=c; } midB+=Mb[id]; }
    // mid bytes: recompute separately = bytes - slots; approximate by difference not tracked; report total
    printf("%s{\"alpha\":%.2f,\"lam\":%.3f,\"D4k\":%.3f,\"D2m\":%.3f,\"lines\":%.3f,\"middepth\":%.4f,\"direct_frac\":%.3f,\"Bpk\":%.2f,\"midMB\":%.1f}",first?"":",",alphas[ai],lam,D/n,(D-n)/n,L/n,st/n,dkeys/n,B/n,midB/1e6); first=false; }
  printf("],\"modes\":[");
  for(int ai=0;ai<2;ai++) for(int mi=0;mi<5;mi++){ double L=0,B=0,M=0,Q=0; for(size_t j=0;j<ns;j++){ size_t id=j*10+ai*5+mi; L+=Ln[id]; B+=By[id]; M+=Mb[id]; Q+=St[id]; }
    printf("%s{\"alpha\":%.2f,\"mode\":%d,\"lines\":%.3f,\"Bpk\":%.2f,\"midMB\":%.1f,\"depth\":%.3f}",(ai||mi)?",":"",alphas[ai],mi,L/n,B/n,M/1e6,Q/n); }
  printf("]}\n"); return 0;}
