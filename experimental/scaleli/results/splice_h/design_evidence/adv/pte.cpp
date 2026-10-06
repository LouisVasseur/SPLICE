// Count-only LRU L2 (2048 sets x 16 ways) residency of page-table lines for the SPLICE-D2 access stream, 4 KiB pages.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>
struct L2{ std::vector<uint64_t> tag; std::vector<uint32_t> age; uint32_t clk=0; L2():tag(2048*16,~0ULL),age(2048*16,0){}
  bool acc(uint64_t a){ uint64_t h=a*0x9E3779B97F4A7C15ULL; int s=(h>>40)&2047; int v=-1,o=0; uint32_t oa=~0u; clk++;
    for(int w=0;w<16;w++){ int i=s*16+w; if(tag[i]==a){ age[i]=clk; return true;} if(age[i]<oa){oa=age[i];o=i;} }
    tag[o]=a; age[o]=clk; return false; } };
int main(){ for(double dirGB: {1.07, 0.27}) for(int hot: {700, 2400}){ L2 c; std::mt19937_64 r(42); std::uniform_real_distribution<double> U(0,1);
  uint64_t ND=(uint64_t)(dirGB*1e9/32768), NDATA=(uint64_t)(3.4e9/32768); uint64_t cold=1ULL<<50; long hd=0,hdat=0,hh=0,N=3000000,W=1000000;
  for(long i=0;i<N;i++){ double u=U(r); if(i%4==0) c.acc(cold++);            // ops stream line
    bool a=c.acc((1ULL<<40)+ (uint64_t)(U(r)*hot));                           // leaf/table line
    bool b=c.acc((2ULL<<40)+(uint64_t)(u*ND));                                 // dir PTE line
    c.acc(cold++); if(U(r)<0.45) c.acc(cold++);                                // dir line(s)
    bool d=c.acc((3ULL<<40)+(uint64_t)(u*NDATA));                              // data PTE line
    c.acc(cold++);                                                             // data line
    if(i>=W){hd+=b;hdat+=d;hh+=a;} }
  printf("dirGB %.2f hot %d: leaf-hit %.3f dirPTE-hit %.3f dataPTE-hit %.3f\n",dirGB,hot,hh/(double)(N-W),hd/(double)(N-W),hdat/(double)(N-W)); } }
