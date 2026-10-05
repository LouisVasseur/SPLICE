// Diagnostic: how much does the trained NFL-style flow actually do?
#include "scaleli/transform.hpp"
#include "scaleli/hardness.hpp"
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
using namespace scaleli;

static std::vector<std::uint64_t> read_sosd(const std::string& p){
    std::ifstream in(p,std::ios::binary); if(!in) throw std::runtime_error("open "+p);
    std::uint64_t n=0; in.read(reinterpret_cast<char*>(&n),8);
    std::vector<std::uint64_t> k(n); in.read(reinterpret_cast<char*>(k.data()),std::streamsize(n*8));
    if(!in) throw std::runtime_error("short read "+p);
    return k;
}
struct LS { long double slope, intercept; double rmse, maxerr; };
// least squares feature -> rank, feature given as long double vector
static LS ls_fit(const std::vector<long double>& x){
    const std::size_t n=x.size(); long double mx=0,my=0,sxx=0,sxy=0;
    for(std::size_t i=0;i<n;++i){const long double k=(long double)(i+1),dx=x[i]-mx,dy=(long double)i-my;
        mx+=dx/k;my+=dy/k;sxx+=dx*(x[i]-mx);sxy+=dx*((long double)i-my);}
    LS r; r.slope=sxx>0?sxy/sxx:0; r.intercept=my-r.slope*mx;
    long double ss=0,w=0;
    for(std::size_t i=0;i<n;++i){const long double e=(long double)i-(r.slope*x[i]+r.intercept); ss+=e*e; w=std::max(w,std::fabsl(e));}
    r.rmse=(double)std::sqrt(ss/(long double)n); r.maxerr=(double)w; return r;
}
int main(int argc,char**argv){
    const std::string S=argv[1], names[10]={"books","fb","osm","covid","genome","history","libio","planet","stack","wise"};
    const std::size_t REG=4096;
    std::string json="[\n";
    printf("%-8s %10s %22s %22s | %8s %8s %8s | %9s %9s %9s %9s | %10s | %14s %14s %8s\n",
        "dataset","n","min","max","D99raw","D99flow","ratio","regRawMu","regFlwMu","regRawMx","regFlwMx","nonlin%","rmse_raw","rmse_flow","ratio");
    for(int di=0;di<10;++di){
        const std::string nm=names[di];
        auto keys=read_sosd(S+"/data/samples/"+nm+"_2M_uniform_s42");
        auto flow=FlowTransform::load(S+"/results/aidb/flows/"+nm+"_2D2H2L.txt");
        const std::size_t n=keys.size();
        const std::uint64_t kmin=keys.front(), kmax=keys.back();
        const double span=double(kmax-kmin>0?kmax-kmin:1);
        std::vector<double> raw(n), flw(n);
        for(std::size_t i=0;i<n;++i){ raw[i]=double(keys[i]-kmin)/span; flw[i]=flow.transform(double(keys[i])); }
        std::size_t unordered=0; for(std::size_t i=1;i<n;++i) unordered += flw[i]<flw[i-1];
        std::vector<double> flws=flw; if(unordered) std::sort(flws.begin(),flws.end());
        const unsigned d_raw=tail_conflict_degree(raw), d_flw=tail_conflict_degree(flws);
        // per region
        long double sr=0,sf=0; unsigned mr=0,mf=0; std::size_t regions=0;
        for(std::size_t b=0;b<n;b+=REG){ const std::size_t len=std::min(REG,n-b);
            unsigned a=tail_conflict_degree(std::span<const double>(raw.data()+b,len));
            std::vector<double> z(flw.begin()+b,flw.begin()+b+len); std::sort(z.begin(),z.end());
            unsigned c=tail_conflict_degree(std::span<const double>(z.data(),len));
            sr+=a; sf+=c; mr=std::max(mr,a); mf=std::max(mf,c); ++regions; }
        // nonlinearity of z vs key
        long double mx=0,my=0,sxx=0,sxy=0;
        for(std::size_t i=0;i<n;++i){const long double xv=(long double)(keys[i]-kmin), yv=flw[i], k=(long double)(i+1);
            const long double dx=xv-mx,dy=yv-my; mx+=dx/k;my+=dy/k; sxx+=dx*(xv-mx); sxy+=dx*(yv-my);}
        const long double aa=sxx>0?sxy/sxx:0, bb=my-aa*mx;
        long double worst=0; for(std::size_t i=0;i<n;++i) worst=std::max(worst,std::fabsl(flw[i]-(aa*(long double)(keys[i]-kmin)+bb)));
        const double zrange=*std::max_element(flw.begin(),flw.end())-*std::min_element(flw.begin(),flw.end());
        const double nonlin=zrange>0?100.0*double(worst)/zrange:0.0;
        // rmse rank fits
        std::vector<long double> xr(n),xf(n);
        for(std::size_t i=0;i<n;++i){xr[i]=(long double)(keys[i]-kmin); xf[i]=(long double)flws[i];}
        const LS lr=ls_fit(xr), lf=ls_fit(xf);
        // double-precision (non-long-double) flow fit, to probe numerical sensitivity
        printf("%-8s %10zu %22llu %22llu | %8u %8u %8.3f | %9.3f %9.3f %9u %9u | %10.5f | %14.2f %14.2f %8.3f\n",
            nm.c_str(),n,(unsigned long long)kmin,(unsigned long long)kmax,d_raw,d_flw,
            d_raw?double(d_flw)/double(d_raw):0.0, double(sr)/regions,double(sf)/regions,mr,mf,
            nonlin, lr.rmse, lf.rmse, lr.rmse>0?lf.rmse/lr.rmse:0.0);
        char buf[4096];
        snprintf(buf,sizeof buf,
          "%s{\"dataset\":\"%s\",\"n\":%zu,\"min\":%llu,\"max\":%llu,\"d99_raw\":%u,\"d99_flow\":%u,\"d99_ratio\":%.6f,"
          "\"region_keys\":%zu,\"regions\":%zu,\"region_d99_raw_mean\":%.6f,\"region_d99_flow_mean\":%.6f,"
          "\"region_d99_raw_max\":%u,\"region_d99_flow_max\":%u,\"nonlinearity_pct\":%.8f,\"z_range\":%.10g,"
          "\"flow_unordered_pairs\":%zu,\"rmse_raw\":%.10g,\"rmse_flow\":%.10g,\"rmse_ratio\":%.6f,"
          "\"maxerr_raw\":%.10g,\"maxerr_flow\":%.10g}",
          di?",\n":"",nm.c_str(),n,(unsigned long long)kmin,(unsigned long long)kmax,d_raw,d_flw,
          d_raw?double(d_flw)/double(d_raw):0.0,REG,regions,double(sr)/regions,double(sf)/regions,mr,mf,
          nonlin,zrange,unordered,lr.rmse,lf.rmse,lr.rmse>0?lf.rmse/lr.rmse:0.0,lr.maxerr,lf.maxerr);
        json+=buf;
    }
    json+="\n]\n";
    std::ofstream o{std::string(argv[2])}; o<<json;
    return 0;
}
