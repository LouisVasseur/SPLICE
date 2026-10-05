// Static lower_bound experiment. The wrapper is ours; RadixSpline is a
// source-derived transcription of the MIT implementation (see third_party).
// No payloads/updates: do not compare these bytes directly to full-map bytes.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "rs/builder.h"
using Key = uint64_t;
using Clock = std::chrono::steady_clock;
static_assert(sizeof(unsigned long) == 8, "This source snapshot requires an LP64 platform (Linux/macOS).");
void require(bool x, const char* message) { if (!x) throw std::runtime_error(message); }
uint64_t u64le(std::istream& f) {
 uint64_t v=0; for (int i=0;i<8;++i) { const int b=f.get(); if(b==EOF) throw std::runtime_error("truncated dataset"); v|=uint64_t(uint8_t(b))<<(8*i); } return v;
}
std::vector<Key> load(const char* p) {
 std::ifstream f(p,std::ios::binary); require(bool(f),"cannot open dataset");
 const uint64_t n=u64le(f); require(n<=10000000,"demo limit is 10 million keys");
 std::vector<Key> a; a.reserve(n); for(uint64_t i=0;i<n;++i)a.push_back(u64le(f));
 require(f.peek()==EOF,"unexpected trailing bytes");
 require(std::is_sorted(a.begin(),a.end()),"keys must be sorted");
 require(std::adjacent_find(a.begin(),a.end())==a.end(),"use distinct keys in this walkthrough"); return a;
}
struct IndexedArray {
 const std::vector<Key>& a;
 rs::RadixSpline<Key> rsindex;
 bool active;
 IndexedArray(const std::vector<Key>& input,size_t epsilon=32): a(input),active(input.size()>1) {
  if(active) { rs::Builder<Key> b(a.front(),a.back(),10,epsilon); for(Key k:a)b.AddKey(k); rsindex=b.Finalize(); }
 }
 size_t lower(Key k)const {
  if(!active) return size_t(std::lower_bound(a.begin(),a.end(),k)-a.begin());
  auto b=rsindex.GetSearchBound(k);
  return size_t(std::lower_bound(a.begin()+b.begin,a.begin()+b.end,k)-a.begin());
 }
};
uint64_t self_test() {
 uint64_t checks=0;
 auto test=[&](const std::vector<Key>& a) {
  for(size_t eps:{size_t(1),size_t(8),size_t(32),size_t(128)}) {
   IndexedArray idx(a,eps); std::vector<Key> q{0,1,std::numeric_limits<Key>::max()};
   for(Key k:a) { q.push_back(k);if(k)q.push_back(k-1);if(k<std::numeric_limits<Key>::max())q.push_back(k+1); }
   std::mt19937_64 rng(9);for(int i=0;i<1000;++i)q.push_back(rng());
   for(Key k:q) {require(idx.lower(k)==size_t(std::lower_bound(a.begin(),a.end(),k)-a.begin()),"RadixSpline lower_bound mismatch");++checks;}
  }
 };
 test({}); test({0}); test({42}); test({0,1});
 for(uint64_t seed=0;seed<20;++seed){std::mt19937_64 rng(seed);std::vector<Key>a;Key k=100;
  for(size_t i=0;i<2048;++i){k+=((i/64)%2)?1+rng()%1000000:1+rng()%3;a.push_back(k);}test(a);
  std::vector<Key> high;for(Key x:a)high.push_back(std::numeric_limits<Key>::max()-a.back()+x);test(high);
 }
 return checks;
}
struct Measurement { double ns;uint64_t checksum; };
template<class F> Measurement measure(const std::vector<Key>& q,F f) {
 uint64_t c=0;auto begin=Clock::now(); for(Key k:q)c=c*0x9e3779b185ebca87ULL+f(k);
 auto end=Clock::now();return{double(std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()),c};
}
double med(std::vector<double>x){std::sort(x.begin(),x.end());return x[x.size()/2];}
int main(int argc,char**argv) {try{
 if(argc==2 && std::string(argv[1])=="--self-test") {std::cout<<"PASS radix_differential checks="<<self_test()<<"\n";return 0;}
 if(argc!=2)throw std::runtime_error("usage: radix_walkthrough DATASET or --self-test");
 auto a=load(argv[1]); require(a.size()>1,"timing demo requires two distinct keys");
 auto start=Clock::now();IndexedArray idx(a);double buildns=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
 std::mt19937_64 rng(42);std::vector<Key>q;size_t absent=0; q.reserve(100000);
 for(size_t i=0;i<100000;++i){Key k=a[rng()%a.size()];if(i%10==0 && k!=std::numeric_limits<Key>::max())++k;
  auto p=std::lower_bound(a.begin(),a.end(),k);if(p==a.end()||*p!=k)++absent;q.push_back(k);
  require(idx.lower(k)==size_t(p-a.begin()),"verification failed on timing trace");}
 uint64_t fingerprint=0;for(Key k:q)fingerprint=fingerprint*0x9e3779b185ebca87ULL+k;
 auto binary=[&](Key k){return size_t(std::lower_bound(a.begin(),a.end(),k)-a.begin());};
 auto spline=[&](Key k){return idx.lower(k);};
 auto warm1=measure(q,binary),warm2=measure(q,spline);require(warm1.checksum==warm2.checksum,"warmup mismatch");
 std::vector<double>btime,stime;uint64_t checksum=warm1.checksum;
 for(int rep=0;rep<5;++rep){Measurement b,s;if(rep%2){s=measure(q,spline);b=measure(q,binary);}else{b=measure(q,binary);s=measure(q,spline);}
  require(b.checksum==s.checksum && b.checksum==checksum,"timed checksum mismatch");btime.push_back(b.ns/q.size());stime.push_back(s.ns/q.size());}
 double widths=0;size_t maxwidth=0;for(Key k:q){auto b=idx.rsindex.GetSearchBound(k);widths+=b.end-b.begin;maxwidth=std::max(maxwidth,b.end-b.begin);}
 std::cout<<"{\n\"status\":\"locally_measured_source_transcription\",\n\"contract\":\"static uint64 lower_bound; keys only, no payload\",\n"
 <<"\"dataset\":\"bundled dense_sparse synthetic fixture\",\"keys\":"<<a.size()<<",\"queries\":"<<q.size()<<",\"absent_queries\":"<<absent<<",\"repeats_same_trace\":5,\n"
 <<"\"epsilon\":32,\"radix_bits\":10,\"key_bytes\":"<<8*a.size()<<",\"auxiliary_index_bytes\":"<<idx.rsindex.GetSize()<<",\n"
 <<"\"build_ns\":"<<buildns<<",\"binary_ns_per_query_median\":"<<med(btime)<<",\"radix_ns_per_query_median\":"<<med(stime)<<",\n"
 <<"\"mean_search_window_keys\":"<<widths/q.size()<<",\"max_search_window_keys\":"<<maxwidth<<",\"verified\":true,\n"
 <<"\"trace_fingerprint\":\""<<fingerprint<<"\",\"checksum\":\""<<checksum<<"\",\n\"caution\":\"Same-process warmed repeated trace. Not independent trials or paper reproduction.\"\n}\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}
