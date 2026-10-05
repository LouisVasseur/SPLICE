#include "scaleli/smoothing.hpp"
#include <cstdio>
int main(){
  std::vector<double> x={1,2,3,10,11,12};
  for(double a:{0.2,0.4,0.5,0.9}){
    auto r=scaleli::smooth_cdf(x,a);
    std::printf("alpha=%.2f budget=%zu rounds=%zu sse_before=%.10Lf sse_after=%.10Lf slots=",a,std::size_t(a*6),r.rounds,(long double)r.sse_before,(long double)r.sse_after);
    for(auto s:r.slot)std::printf("%zu ",s);
    std::printf(" virtual=");for(auto v:r.virtual_features)std::printf("%.10f ",v);std::printf("\n");
  }
  std::vector<double> t={1,2,3,4,5,10,20,26,27,30};auto r=scaleli::smooth_cdf(t,0.3);
  std::printf("test vector alpha 0.3: rounds=%zu before=%.6f after=%.6f virtual=",r.rounds,r.sse_before,r.sse_after);for(auto v:r.virtual_features)std::printf("%.6f ",v);std::printf("\n");
  // tail conflict degree examples
  return 0;
}
