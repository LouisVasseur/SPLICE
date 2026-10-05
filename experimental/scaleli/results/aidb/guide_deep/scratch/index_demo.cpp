#include "scaleli/index.hpp"
#include "scaleli/hardness.hpp"
#include <cstdio>
using namespace scaleli;
int main(){
  // A: LinearModel fit_xy on 6 features vs ranks
  {std::vector<Key> keys={1,2,3,10,11,12};std::vector<double> x(6),y(6);LinearModel tmp;tmp.fit(keys,y);
   for(int i=0;i<6;++i){x[i]=double(keys[i]);y[i]=i;}
   LinearModel m;m.fit_xy(keys,x,y);std::printf("A fit_xy features=keys: slope=%.10f intercept=%.10f predict_x(10)=%.6f\n",m.slope,m.intercept,m.predict_x(10));
   for(int i=0;i<6;++i)x[i]=tmp.normalized(keys[i]);LinearModel n;n.fit_xy(keys,x,y);std::printf("A normalized: origin=%llu span=%llu phi=",(unsigned long long)n.origin,(unsigned long long)n.span);for(auto v:x)std::printf("%.6f ",v);std::printf(" slope=%.6f intercept=%.6f\n",n.slope,n.intercept);
   detail::Sums s;for(int i=0;i<6;++i)s.add(double(keys[i]),i);std::printf("A Sums: n=%.0Lf sx=%.0Lf sxx=%.0Lf sy=%.0Lf syy=%.0Lf sxy=%.0Lf sse=%.10Lf\n",s.n,s.sx,s.sxx,s.sy,s.syy,s.sxy,s.sse());}
  // B: counters on a small index; two regions of 16 keys, blocks of 4
  {Config c;c.region_keys=16;c.block_keys=4;c.delta_limit=8;c.routing=Routing::Rank;c.policy=Policy::Raw;Index ix(c);std::vector<Record> rows;
   for(unsigned i=0;i<32;++i)rows.emplace_back(i<24?Key(i)*10:Key(240)+Key(i-24)*1000,i);ix.bulk_load(rows);
   std::printf("B regions=%zu blocks=%zu\n",ix.region_count(),ix.block_count());
   for(Key q:{Key(0),Key(70),Key(230),Key(2240),Key(7240)}){QueryStats s;auto v=ix.find(q,&s);
     std::printf("B find(%llu)=%lld root_probes=%llu coordinate_probes=%llu fence_probes=%llu correction=%llu block_routes=%llu key_at=%llu\n",(unsigned long long)q,v?(long long)*v:-1,(unsigned long long)s.root_probes,(unsigned long long)s.coordinate_probes,(unsigned long long)s.fence_probes,(unsigned long long)s.correction_distance,(unsigned long long)s.block_routes,(unsigned long long)s.key_at_calls);}
   auto l=ix.learnability();std::printf("B rank_sse_before=%.4f after=%.4f\n",l.rank_sse_before,l.rank_sse_after);}
  // C: Fusion::Auto per-region selector on a clustered region
  {FlowTransform f;f.in_dim=2;f.hidden=2;f.layers=2;f.mean=0;f.var=1000;f.shapes={{2,2},{2,2}};f.weights={{0.002,0,0,0},{1,0,0,1}};
   for(double fc:{4.0,0.0}){Config c;c.region_keys=64;c.block_keys=8;c.routing=Routing::Rank;c.policy=Policy::Raw;c.fusion=Fusion::Auto;c.virtual_alpha=0.1;c.flow=&f;c.flow_cost=fc;Index ix(c);
    std::vector<Record> rows;for(unsigned i=0;i<128;++i)rows.emplace_back(i<96?Key(i)*Key(i)*7:Key(96*96*7)+(i-96)*100000,i);ix.bulk_load(rows);
    auto l=ix.learnability();std::printf("C flow_cost=%.0f regions=%zu choices none=%zu flow=%zu vp=%zu both=%zu cost_none_mean=%.4f cost_selected_mean=%.4f virtual_points=%zu tail_raw=%.2f tail_flow=%.2f\n",fc,l.regions,l.choice_none,l.choice_flow,l.choice_vp,l.choice_both,l.cost_none_mean,l.cost_selected_mean,l.virtual_points,l.tail_conflicts_raw_mean,l.tail_conflicts_flow_mean);}}
  // D: hardness metrics on a tiny set
  {std::vector<std::uint64_t> k={0,1,3};auto ls=hardness::least_squares<std::uint64_t>(k);std::printf("D ls {0,1,3}: slope=%.10Lf intercept=%.10Lf rmse=%.10f me=%.10f\n",ls.slope,ls.intercept,ls.rmse,ls.max_error);
   std::vector<std::uint64_t> cl;for(unsigned i=0;i<50;++i)cl.push_back(i);for(unsigned j=0;j<950;++j)cl.push_back(1000000+100*j);
   auto m=hardness::fmcd_fit<std::uint64_t>(cl);std::printf("D fmcd cluster: gap=%d capacity=%zu D=%zu Ut=%.9f a=%.12Lf base=%.1Lf anchor=%.1Lf fallback=%d CD=%zu\n",m.gap,m.capacity,m.d,m.ut,m.a,m.base,m.anchor,m.fallback,hardness::conflict_degree<std::uint64_t>(cl,m));
   std::printf("D predict(cl[0])=%zu predict(cl[49])=%zu predict(cl[50])=%zu predict(cl[51])=%zu predict(last)=%zu\n",m.predict(cl[0]),m.predict(cl[49]),m.predict(cl[50]),m.predict(cl[51]),m.predict(cl.back()));
   std::vector<std::uint64_t> st;for(std::uint64_t r=0;r<10;++r)for(std::uint64_t i=0;i<100;++i)st.push_back(r*1000000+i);
   std::printf("D pla stairs eps32=%zu eps4096=%zu ; line eps0=%zu\n",hardness::pla_segments<std::uint64_t>(st,32),hardness::pla_segments<std::uint64_t>(st,4096),hardness::pla_segments<std::uint64_t>(std::vector<std::uint64_t>{3,10,17,24},0));}
  // E: tail conflict degree worked example
  {std::vector<double> x={0,1,2,3,4,5,6,7,8,9,9.1,9.2,9.3,9.4,9.5,20,30,40,50,60};
   std::printf("E tail_conflict_degree(20 features)=%u ; uniform 0..99 -> %u\n",tail_conflict_degree(x),tail_conflict_degree([]{std::vector<double> u(100);for(int i=0;i<100;++i)u[i]=i;return u;}()));}
  return 0;
}
