#include "scaleli/index.hpp"
#include "scaleli/baselines.hpp"
#include <iostream>
#include <random>
#include <sstream>
#include <functional>
#include <filesystem>

using namespace scaleli;
static std::uint64_t assertions=0;
#define CHECK(x) do{++assertions;if(!(x))throw std::runtime_error(std::string("CHECK failed: ")+ #x + " line " + std::to_string(__LINE__));}while(false)
template<class F>void must_throw(F f){bool threw=false;try{f();}catch(const std::exception&){threw=true;}CHECK(threw);}
void codecs(){
    std::mt19937_64 rng(7);
    for(std::size_t n: {1u,2u,3u,7u,16u,17u,63u,64u,65u,128u,255u,1024u})
    for(int type=0;type<7;++type){
        std::vector<Key> keys(n);Key k=0;
        for(auto& x:keys){
            if(type==0)x=k++;
            else if(type==1)x=k+=1+(rng()%7);
            else if(type==2)x=rng();
            else if(type==3)x=std::numeric_limits<Key>::max()-k++;
            else if(type==4)x=42;
            else if(type==5)x=k+=rng()%2?1:100000000;
            else x=std::uint64_t((U128(std::numeric_limits<Key>::max())*k++)/std::max<std::size_t>(1,n-1));
        }
        std::sort(keys.begin(),keys.end());
        for(auto c:{Codec::Raw,Codec::For,Codec::Delta,Codec::Linear})for(unsigned r:{1,7,16,31}){
            const auto e=encode_block(keys,c,r);CHECK(decode_block(e.bytes)==keys);
            for(std::size_t i=0;i<n;++i){CHECK(key_at(e.bytes,i)==keys[i]);CHECK(encoded_position(e.bytes,i)==e.key_positions[i]);CHECK(encoded_position(e.bytes,i)<e.bytes.size());}
            must_throw([&]{key_at(e.bytes,n);});
        }
    }
    for(unsigned width=0;width<=64;++width){std::vector<std::uint8_t>b(80,0);const Key v=width==64?~Key(0):(width?((Key(1)<<width)-1):0);
        put_bits(b,3,width,v);CHECK(get_bits(b,3,width)==v);}
    std::vector<std::uint8_t> bad(10,0xff);std::size_t p=0;must_throw([&]{get_varint(bad,p);});
    std::vector<std::uint8_t> tiny(3);must_throw([&]{decode_block(tiny);});
    must_throw([&]{encode_block(std::vector<Key>{2,1},Codec::Raw);});
}
void endpoints(){
    for(auto route:{Routing::Binary,Routing::Rank,Routing::Byte})for(auto policy:{Policy::Raw,Policy::MinBytes,Policy::Smooth,Policy::Adaptive}){
        Config c;c.region_keys=8;c.block_keys=2;c.delta_limit=3;c.routing=route;c.policy=policy;Index ix(c);
        CHECK(!ix.find(0));CHECK(ix.scan(0,10).empty());CHECK(!ix.lower_bound(0));
        const auto M=std::numeric_limits<Key>::max();
        ix.bulk_load({{M,M},{0,0},{1,8},{1,9},{M-1,2},{M/2,4}});
        CHECK(ix.size()==5);CHECK(ix.find(1)==9);CHECK(ix.find(M)==M);CHECK(ix.find(0)==0);
        CHECK(ix.lower_bound(M)->first==M);CHECK(ix.scan(M,10).size()==1);CHECK(ix.scan(0,0).empty());
        CHECK(ix.erase(M));CHECK(!ix.erase(M));CHECK(!ix.lower_bound(M));CHECK(ix.upsert(M,0));CHECK(!ix.upsert(M,8));
        ix.maintain();ix.validate();CHECK(ix.find(M)==8);
        for(auto r:ix.scan(0,100))CHECK(ix.erase(r.first));
        ix.maintain();CHECK(ix.size()==0);CHECK(ix.scan(0,100).empty());CHECK(ix.upsert(M,M));ix.validate();
    }
}
void differential(){
    for(auto route:{Routing::Binary,Routing::Rank,Routing::Byte})
    for(auto policy:{Policy::Raw,Policy::MinBytes,Policy::Smooth,Policy::Adaptive,Policy::Forced})
    for(unsigned seed=0;seed<3;++seed){
        Config c;c.region_keys=64;c.block_keys=8;c.delta_limit=11;c.routing=route;c.policy=policy;c.forced_codec=Codec::Delta;
        Index ix(c);OrderedMap ref;std::mt19937_64 rng(100+seed);
        std::vector<Record> initial;for(unsigned i=0;i<256;++i)initial.emplace_back(i*10,i+10);ix.bulk_load(initial);ref.bulk_load(initial);
        for(unsigned op=0;op<3500;++op){
            const Key k=op%13==0?rng():rng()%6000;const auto type=rng()%10;
            if(type<4){const auto v=rng();CHECK(ix.upsert(k,v)==ref.upsert(k,v));}
            else if(type<6)CHECK(ix.erase(k)==ref.erase(k));
            else if(type<8)CHECK(ix.find(k)==ref.find(k));
            else{const auto n=rng()%60;CHECK(ix.scan(k,n)==ref.scan(k,n));CHECK(ix.lower_bound(k)==ref.lower_bound(k));}
            if(op%127==0){ix.maintain();ix.validate();CHECK(ix.size()==ref.size());CHECK(ix.scan(0,100000)==ref.scan(0,100000));}
        }
        ix.maintain();ix.validate();CHECK(ix.scan(0,100000)==ref.scan(0,100000));CHECK(ix.maintenance().splits>0);
    }
}
void adversarial_routes(){
    Config c;c.region_keys=128;c.block_keys=1;c.delta_limit=100;c.routing=Routing::Byte;Index ix(c);
    std::vector<Record> v;for(unsigned i=0;i<300;++i)v.emplace_back(i<250?Key(i):Key(i)*10000000,i);
    ix.bulk_load(v);for(Key q=0;q<3000000000ULL;q+=199999){SortedVector ref;ref.bulk_load(v);CHECK(ix.lower_bound(q)==ref.lower_bound(q));}
    ix.validate();
    Region r;r.rebuild(v,c); // deliberately destroy both models; correctness must not depend on accuracy
    r.rank_model.slope=0;r.rank_model.intercept=-1e15;r.byte_model.slope=0;r.byte_model.intercept=1e15;
    for(auto [k,val]:v)for(auto route:{Routing::Rank,Routing::Byte}){c.routing=route;CHECK(r.find(k,c)==val);}
}
void maintenance_and_memory(){
    Config c;c.region_keys=128;c.block_keys=16;c.delta_limit=16;c.policy=Policy::Adaptive;c.heat_decay=0.9;
    Index ix(c);std::vector<Record> rows;for(unsigned i=0;i<128;++i)rows.emplace_back(2*i,i);ix.bulk_load(rows);
    const auto before=ix.memory();for(unsigned i=0;i<40;++i)ix.upsert(2*i,i+1);ix.maintain();
    CHECK(ix.maintenance().raw_bypasses>0);for(unsigned i=0;i<100;++i)(void)ix.find(4);ix.maintain();
    CHECK(ix.memory().key_bytes<=128*8+8*16);CHECK(before.accounted_bytes()>before.key_bytes+before.value_bytes);
    std::stringstream out;ix.dump_layout(out);CHECK(out.str().find("byte_prediction")!=std::string::npos);ix.validate();
    must_throw([]{Config c;c.block_keys=0;Index bad(c);});
}
void learnability(){
    // Smoothing: SSE never increases, virtual points are never real keys, slots stay strictly increasing.
    std::vector<double> x={1,2,3,4,5,10,20,26,27,30};
    auto r=smooth_cdf(x,0.3);CHECK(r.sse_after<=r.sse_before);CHECK(r.virtual_features.size()<=3);
    for(std::size_t i=1;i<r.slot.size();++i)CHECK(r.slot[i]>r.slot[i-1]);
    CHECK(r.slot.back()==x.size()-1+r.virtual_features.size());
    for(auto v:r.virtual_features){CHECK(v>x.front()&&v<x.back());for(auto k:x)CHECK(v!=k);}
    CHECK(smooth_cdf(x,0).virtual_features.empty());must_throw([&]{smooth_cdf(x,64.0);});
    std::vector<double> lin={0,1,2,3,4,5,6,7};CHECK(smooth_cdf(lin,0.5).virtual_features.empty()); // already perfect: no insertion helps
    // Tail conflict degree: uniform features have no tail conflicts; a heavy cluster does.
    std::vector<double> u(1000);for(std::size_t i=0;i<u.size();++i)u[i]=double(i);CHECK(tail_conflict_degree(u)==0);
    std::vector<double> clustered(1000);for(std::size_t i=0;i<clustered.size();++i)clustered[i]=i<900?double(i)*1e-6:double(i);CHECK(tail_conflict_degree(clustered)>0);
    // Flow: text round trip and the hand-checked 2D2H2L forward pass.
    FlowTransform f;f.in_dim=2;f.hidden=2;f.layers=2;f.mean=0;f.var=1;f.shapes={{2,2},{2,2}};f.weights={{0.5,0,0.25,1},{1,0,0,1}};
    const double x0=2.75;const double u0=x0*0.5+0.75*0.25,u1=0.75*1;const double expect=std::tanh(u0)+std::tanh(u1);
    CHECK(std::abs(f.transform(x0)-expect)<1e-12);
    const auto path=std::filesystem::temp_directory_path()/"scaleli_flow_test.txt";f.save(path);const auto g=FlowTransform::load(path);
    CHECK(std::abs(g.transform(x0)-expect)<1e-12);std::filesystem::remove(path);
    // Differential correctness with flow + smoothing, bypass on and off, all routings.
    for(auto route:{Routing::Rank,Routing::Byte,Routing::Binary})for(bool bypass:{true,false})for(bool relearn:{false,true}){
        Config c;c.region_keys=64;c.block_keys=8;c.delta_limit=11;c.routing=route;c.flow=&f;c.flow_bypass=bypass;c.virtual_alpha=0.2;c.relearn_on_compaction=relearn;
        Index ix(c);OrderedMap ref;std::mt19937_64 rng(9);std::vector<Record> initial;for(unsigned i=0;i<300;++i)initial.emplace_back(i<200?i*3:i*1000,i);
        ix.bulk_load(initial);ref.bulk_load(initial);
        if(!bypass)CHECK(ix.learnability().flow_regions==ix.region_count());
        CHECK(ix.learnability().virtual_points>0);CHECK(ix.learnability().rank_sse_after<=ix.learnability().rank_sse_before);
        for(unsigned op=0;op<2000;++op){const Key k=rng()%400000;const auto t=rng()%10;
            if(t<4){const auto v=rng();CHECK(ix.upsert(k,v)==ref.upsert(k,v));}else if(t<6)CHECK(ix.erase(k)==ref.erase(k));
            else if(t<8)CHECK(ix.find(k)==ref.find(k));else{const auto n=rng()%40;CHECK(ix.scan(k,n)==ref.scan(k,n));}}
        ix.maintain();ix.validate();CHECK(ix.scan(0,100000)==ref.scan(0,100000));
        if(relearn)CHECK(ix.learnability().virtual_points>0); // reuse may legitimately drop stale points after heavy churn
        CHECK(ix.learnability().rank_sse_after<=ix.learnability().rank_sse_before+1e-9);
        QueryStats s;(void)ix.find(300,&s);if(!bypass && route!=Routing::Binary)CHECK(s.transform_calls>0);
    }
}
void fusion_auto(){
    FlowTransform f;f.in_dim=2;f.hidden=2;f.layers=2;f.mean=0;f.var=1000;f.shapes={{2,2},{2,2}};f.weights={{0.002,0,0,0},{1,0,0,1}};
    for(bool withflow:{true,false})for(double alpha:{0.0,0.2})for(double flowcost:{0.0,100.0}){
        Config c;c.region_keys=64;c.block_keys=8;c.delta_limit=11;c.routing=Routing::Rank;c.fusion=Fusion::Auto;c.virtual_alpha=alpha;c.flow_cost=flowcost;if(withflow)c.flow=&f;
        Index ix(c);OrderedMap ref;std::mt19937_64 rng(5);std::vector<Record> initial;
        for(unsigned i=0;i<400;++i)initial.emplace_back(i<300?Key(i)*Key(i)*7:Key(300*300*7)+(i-300)*100000,i);
        ix.bulk_load(initial);ref.bulk_load(initial);const auto l=ix.learnability();
        CHECK(l.choice_none+l.choice_flow+l.choice_vp+l.choice_both==l.regions);
        CHECK(l.cost_selected_mean<=l.cost_none_mean+1e-12);
        if(!withflow)CHECK(l.choice_flow+l.choice_both==0);
        if(alpha==0)CHECK(l.choice_vp+l.choice_both==0);
        if(flowcost>=100)CHECK(l.choice_flow+l.choice_both==0); // an expensive transform is never chosen
        for(unsigned op=0;op<1500;++op){const Key k=rng()%700000;const auto t=rng()%10;
            if(t<4){const auto v=rng();CHECK(ix.upsert(k,v)==ref.upsert(k,v));}else if(t<6)CHECK(ix.erase(k)==ref.erase(k));
            else if(t<8)CHECK(ix.find(k)==ref.find(k));else{const auto n=rng()%40;CHECK(ix.scan(k,n)==ref.scan(k,n));}}
        ix.maintain();ix.validate();CHECK(ix.scan(0,100000)==ref.scan(0,100000));
    }
    must_throw([]{Config c;c.flow_cost=-1;Index bad(c);});
    // Parallel bulk load must produce the identical structure and answers as the serial build.
    for(unsigned threads:{1u,4u}){
        Config c;c.region_keys=64;c.block_keys=8;c.routing=Routing::Rank;c.fusion=Fusion::Auto;c.virtual_alpha=0.2;c.flow=&f;c.build_threads=threads;
        Index ix(c);std::vector<Record> initial;for(unsigned i=0;i<2000;++i)initial.emplace_back(Key(i)*Key(i)*3+(i%7),i);ix.bulk_load(initial);ix.validate();
        static std::string reference;std::stringstream out;ix.dump_layout(out);
        if(threads==1)reference=out.str();else CHECK(out.str()==reference);
        for(auto [k,v]:initial)CHECK(ix.find(k)==v);
    }
}
void learned_root(){
    FlowTransform f;f.in_dim=2;f.hidden=2;f.layers=2;f.mean=0;f.var=1000;f.shapes={{2,2},{2,2}};f.weights={{0.002,0,0,0},{1,0,0,1}};
    for(bool withflow:{false,true})for(auto fusion:{Fusion::Manual,Fusion::Auto})for(double ralpha:{0.0,0.2}){
        Config c;c.region_keys=64;c.block_keys=8;c.delta_limit=11;c.routing=Routing::Rank;c.root=Root::Model;c.root_alpha=ralpha;c.fusion=fusion;c.virtual_alpha=fusion==Fusion::Auto?0.2:0;if(withflow)c.flow=&f;
        Index ix(c);OrderedMap ref;std::mt19937_64 rng(11);std::vector<Record> initial;
        for(unsigned i=0;i<3000;++i)initial.emplace_back(Key(i)*Key(i)*5+(i%3),i);ix.bulk_load(initial);ref.bulk_load(initial);
        const auto l=ix.learnability();CHECK(l.root_probes_binary>0);CHECK(l.root_probes_raw>0);
        if(l.root_model){double best=l.root_probes_raw;if(withflow)best=std::min(best,l.root_probes_flow);if(ralpha>0){best=std::min(best,l.root_probes_vp_raw>0?l.root_probes_vp_raw:best);if(withflow&&l.root_probes_vp_flow>0)best=std::min(best,l.root_probes_vp_flow);}CHECK(best<l.root_probes_binary);}
        if(ralpha>0 && l.root_model && l.root_vp)CHECK(l.root_virtual>0);
        QueryStats s;for(auto [k,v]:initial){CHECK(ix.find(k,&s)==v);}CHECK(s.root_probes>0);
        for(Key q=0;q<Key(3000)*3000*5+100;q+=9973){CHECK(ix.lower_bound(q)==ref.lower_bound(q));}
        for(unsigned op=0;op<4000;++op){const Key k=rng()%(Key(3000)*3000*5);const auto t=rng()%10;
            if(t<5){const auto v=rng();CHECK(ix.upsert(k,v)==ref.upsert(k,v));}else if(t<6)CHECK(ix.erase(k)==ref.erase(k));
            else if(t<8)CHECK(ix.find(k)==ref.find(k));else{const auto n=rng()%40;CHECK(ix.scan(k,n)==ref.scan(k,n));}}
        ix.maintain();ix.validate();CHECK(ix.maintenance().splits>0);CHECK(ix.scan(0,100000)==ref.scan(0,100000));
        for(auto [k,v]:ref.scan(0,100000))CHECK(ix.find(k)==v);
    }
    // Keys that start far from 0 (a contiguous window): the raw-feature root must not be defeated by the sentinel fence.
    {Config c;c.region_keys=64;c.block_keys=8;c.routing=Routing::Rank;c.root=Root::Model;Index ix(c);std::vector<Record> w;
     for(unsigned i=0;i<4000;++i)w.emplace_back(Key(9000000000000ULL)+Key(i)*1000+(i%7),i);ix.bulk_load(w);const auto l=ix.learnability();
     CHECK(l.root_model);CHECK(l.root_probes_raw<l.root_probes_binary);for(auto [k,v]:w)CHECK(ix.find(k)==v);ix.validate();}
    must_throw([]{Config c;(void)parse_root("tree");});must_throw([]{Config c;c.root_alpha=64.0;Index bad(c);});must_throw([]{Config c;c.virtual_alpha=1.0;Index bad(c);});
}
void joint_root(){
    // Building blocks. pysum is CPython's compensated sum(): a plain loop loses the 1.
    CHECK(joint_detail::pysum({1e100,1.0,-1e100})==1.0);
    const auto& G=joint_detail::gains();CHECK(std::abs(G[0]-0.01)<1e-15 && std::abs(G[63]-200.0)<1e-12);
    for(std::size_t i=1;i<64;++i)CHECK(G[i]>G[i-1]);
    // Fences with a few very wide gaps: G must cut them and stay strictly increasing (and invertible) on [0, 1].
    std::vector<Key> F;Key k=1000;std::mt19937_64 rng(3);
    for(unsigned i=0;i<400;++i){F.push_back(k);k+=1000+rng()%100+(i%97==0?Key(5000000):0)+(i==200?Key(90000000):0);}
    std::vector<Key> P;for(std::size_t j=0;j<F.size();++j){P.push_back(F[j]);if(j+1<F.size())P.push_back(F[j]+(F[j+1]-F[j])/2);}
    auto nocount=[](const LinearModel&,const JointFeature&,const std::vector<std::uint32_t>&){return std::size_t(0);};
    JointParams p;
    for(std::size_t kk:{1u,4u,16u}){
        joint_detail::Cycle<decltype(nocount)> cy(F,P,p,nocount,kk);auto s=cy.empty();
        CHECK(cy.G(s));CHECK(!s.gidx.empty() && s.gidx.size()<=kk);CHECK(s.f.scale>0 && s.f.scale<1);
        CHECK(s.f.g(0.0)==0.0);CHECK(std::abs(s.f.g(1.0)-1.0)<1e-12);
        double prev=-1;for(unsigned i=0;i<=20000;++i){const double u=s.f.g(double(i)/20000);CHECK(u>prev);prev=u;}
        CHECK(cy.monotone(s));
        for(std::size_t j=0;j<s.gidx.size();++j)CHECK(s.gs[j]>0 && s.gs[j]<1); // only gaps wider than the median shrink, none collapses
        // T: least-squares warp to ranks in the g coordinate, strictly increasing, non-negative, gains from the grid.
        cy.T(s);CHECK(s.f.nunits>=1 && s.f.nunits<=2);
        for(std::uint32_t h=0;h<s.f.nunits;++h){CHECK(s.f.units[h].coef>0);CHECK(std::find(G.begin(),G.end(),s.f.units[h].gain)!=G.end());}
        prev=-1e300;for(auto q:P){const double z=s.f(q);CHECK(z>prev);prev=z;}
        CHECK(cy.V(s));CHECK(s.virt<=std::size_t(p.alpha*double(F.size())));
        if(s.virt){CHECK(s.slots.size()==F.size());CHECK(s.slots.back()==F.size()-1+s.virt);for(std::size_t j=1;j<s.slots.size();++j)CHECK(s.slots[j]>s.slots[j-1]);}
    }
    // A warp alone (no G) is fitted in x = key/span and is strictly increasing on the probe keys.
    {joint_detail::Cycle<decltype(nocount)> cy(F,P,p,nocount,0);auto s=cy.empty();CHECK(cy.G(s));CHECK(s.gidx.empty());cy.T(s);CHECK(s.f.nunits>0);CHECK(cy.monotone(s));}
    // Whole index: cubic keys (a concave CDF, where the warp pays) with one wide gap (where G pays).
    std::vector<Record> rows;for(unsigned i=0;i<24000;++i){const double t=double(i)/24000;rows.emplace_back(Key(t*t*t*1e15)+Key(i)*3+1000+(i>12000?Key(1e14):0),i);}
    rows=canonicalize(rows);
    for(float charge:{1.0f,0.0f})for(unsigned threads:{1u,4u}){
        Config c;c.region_keys=64;c.block_keys=8;c.delta_limit=11;c.routing=Routing::Rank;c.root=Root::Model;c.root_alpha=0.1;c.flow_cost=0;
        c.root_joint_rounds=6;c.root_joint_kmax=64;c.root_joint_gap_charge=charge;c.build_threads=threads;
        Index ix(c);ix.bulk_load(rows);const auto l=ix.learnability();const RootJoint* j=l.joint;
        CHECK(j && j->candidate);CHECK(l.root_joint);CHECK(l.root_model && !l.root_flow);CHECK(l.root_vp==(j->virt>0));CHECK(l.root_virtual==j->virt);
        const std::size_t n=ix.region_count();CHECK(j->probe_keys==2*n-1);
        // Budget accounting: V within root_alpha, G within k, charges exactly as scored.
        CHECK(j->virt<=std::size_t(c.root_alpha*double(n)));CHECK(j->k_eff<=j->k);CHECK(j->gap_index.size()==j->k_eff);
        CHECK(j->gap_probes==(j->k_eff?std::ceil(std::log2(double(j->k_eff+1))):0.0));
        CHECK(j->model_probes==double(j->count)/double(j->probe_keys));
        CHECK(j->cost==j->model_probes+double(charge)*j->gap_probes+(j->map.nunits?c.flow_cost:0));
        if(charge==0)CHECK(j->k_eff>0); // free gap table: G is used on this key set
        // Guard: round 1 accepted, every later accepted round improves by > 1e-6, the cycle stops at the first rejection.
        CHECK(!j->trace.empty() && j->trace[0].accepted);double last=j->trace[0].total;
        for(std::size_t r=1;r<j->trace.size();++r){const auto& t=j->trace[r];
            if(t.accepted){CHECK(t.total<last-1e-6);last=t.total;}else CHECK(r+1==j->trace.size());}
        CHECK(j->trace[j->round-1].accepted);CHECK(j->model_probes+double(charge)*j->gap_probes==last);
        CHECK(j->rounds_run==j->trace.size() && j->rounds_run<=c.root_joint_rounds);
        for(const auto& b:j->by_k)if(b.candidate)CHECK(b.total>=last);           // the selected k has the minimum total
        CHECK(j->by_k.front().k==0 && j->cost<=j->seq_total+1e-12);             // never worse than the sequential T -> V pass
        CHECK(j->cost<l.root_probes_raw && j->cost<l.root_probes_binary && (l.root_probes_vp_raw==0 || j->cost<l.root_probes_vp_raw));
        // Accounting: the joint state is metadata; same regions as without it.
        Config c0=c;c0.root_joint_rounds=0;Index ref0(c0);ref0.bulk_load(rows);const auto m=ix.memory(),m0=ref0.memory();const auto l0=ref0.learnability();
        CHECK(m.key_bytes==m0.key_bytes && m.value_bytes==m0.value_bytes);CHECK(!l0.joint && !l0.root_joint);
        CHECK(m.metadata_bytes==m0.metadata_bytes+j->bytes()+4*((n+l.root_virtual)*(l.root_vp?1:0))-4*((n+l0.root_virtual)*(l0.root_vp?1:0)));
        std::uint64_t walked=0;ix.for_each_allocation([&](const void*,std::size_t b){walked+=b;});
        std::uint64_t walked0=0;ref0.for_each_allocation([&](const void*,std::size_t b){walked0+=b;});
        CHECK(walked-walked0==m.metadata_bytes-m0.metadata_bytes);
        // Lookups through the adopted joint root: every key found, misses (in gaps, below, above, uint64 max) not found.
        QueryStats s;for(auto [key,v]:rows)CHECK(ix.find(key,&s)==v);
        if(j->map.nunits)CHECK(s.transform_calls==rows.size());CHECK(s.root_probes>0);
        OrderedMap ref;ref.bulk_load(rows);
        for(std::size_t i=0;i+1<rows.size();i+=7){const Key a=rows[i].first,b=rows[i+1].first;if(b-a>1){CHECK(!ix.find(a+1));CHECK(!ix.find(a+(b-a)/2));}}
        CHECK(!ix.find(0));CHECK(!ix.find(rows.front().first-1));CHECK(!ix.find(rows.back().first+1));CHECK(!ix.find(~Key(0)));
        for(Key q=0;q<rows.back().first+(Key(1)<<30);q+=rows.back().first/4999)CHECK(ix.lower_bound(q)==ref.lower_bound(q));
        // Splits: the joint root is remapped and re-scored, never refitted; answers stay exact.
        if(threads==1){
            std::mt19937_64 r2(17);for(unsigned op=0;op<6000;++op){const Key key=r2()%(rows.back().first+1000);const auto t=r2()%10;
                if(t<6){const auto v=r2();CHECK(ix.upsert(key,v)==ref.upsert(key,v));}else if(t<7)CHECK(ix.erase(key)==ref.erase(key));
                else if(t<8){const auto v=r2();const bool had=ref.find(key).has_value();CHECK(ix.update(key,v)==had);if(had)ref.upsert(key,v);}
                else CHECK(ix.find(key)==ref.find(key));}
            ix.maintain();ix.validate();CHECK(ix.maintenance().splits>0);CHECK(ix.scan(0,1000000)==ref.scan(0,1000000));
            const auto l2=ix.learnability();CHECK(l2.root_joint && l2.joint->rescores==ix.maintenance().splits); // still adopted after every split on this key set
            for(auto [key,v]:ref.scan(0,1000000))CHECK(ix.find(key)==v);
        }
    }
    // Determinism: the k cycles run in parallel, the result must not depend on the thread count.
    {Config c;c.region_keys=64;c.block_keys=8;c.routing=Routing::Rank;c.root=Root::Model;c.root_alpha=0.1;c.flow_cost=0;c.root_joint_rounds=6;
     Index a(c);a.bulk_load(rows);c.build_threads=8;Index b(c);b.bulk_load(rows);const auto *x=a.learnability().joint,*y=b.learnability().joint;
     CHECK(x->by_k.size()==y->by_k.size());for(std::size_t i=0;i<x->by_k.size();++i)CHECK(x->by_k[i].count==y->by_k[i].count && x->by_k[i].total==y->by_k[i].total);
     CHECK(x->cost==y->cost && x->gap_index==y->gap_index && x->map.units[0].coef==y->map.units[0].coef);}
    // A charged warp must lose to the sequential candidates; ties go to them. One round with k = 0 is the sequential pass itself.
    {Config c;c.region_keys=64;c.block_keys=8;c.routing=Routing::Rank;c.root=Root::Model;c.root_alpha=0.1;c.flow_cost=1000;c.root_joint_rounds=6;
     Index ix(c);ix.bulk_load(rows);const auto l=ix.learnability();CHECK(l.joint->candidate);if(l.joint->map.nunits)CHECK(!l.root_joint);
     for(auto [key,v]:rows)CHECK(ix.find(key)==v);
     c.flow_cost=0;c.root_joint_rounds=1;c.root_joint_kmax=0;Index one(c);one.bulk_load(rows);const auto* j=one.learnability().joint;
     CHECK(j->by_k.size()==1 && j->rounds_run==1 && j->round==1 && j->cost==j->seq_total);}
    // Too few regions: no joint fit, nothing changes.
    {Config c;c.region_keys=64;c.block_keys=8;c.root=Root::Model;c.root_joint_rounds=6;Index ix(c);std::vector<Record> few;for(unsigned i=0;i<100;++i)few.emplace_back(i*i,i);
     ix.bulk_load(few);CHECK(!ix.learnability().joint);for(auto [key,v]:few)CHECK(ix.find(key)==v);}
    must_throw([]{Config c;c.root_joint_rounds=6;Index bad(c);});   // needs root model
    must_throw([]{Config c;c.root=Root::Model;c.root_joint_rounds=6;c.root_joint_kmin=65;Index bad(c);});
    must_throw([]{Config c;c.root=Root::Model;c.root_joint_rounds=6;c.root_joint_order=2;Index bad(c);});
    must_throw([]{Config c;c.root=Root::Model;c.root_joint_rounds=6;c.root_joint_gap_charge=-1;Index bad(c);});
    must_throw([]{Config c;c.root=Root::Model;c.root_joint_rounds=6;c.root_joint_gap_charge=std::numeric_limits<float>::quiet_NaN();Index bad(c);});
}
int main(){try{
    codecs();std::cout<<"codec roundtrips / full-width arithmetic: PASS\n";
    endpoints();std::cout<<"empty / duplicate / uint64 endpoint semantics: PASS\n";
    differential();std::cout<<"45 randomized differential configurations: PASS\n";
    adversarial_routes();std::cout<<"adversarial / intentionally corrupted model predictions: PASS\n";
    maintenance_and_memory();std::cout<<"maintenance / hysteresis / memory accounting: PASS\n";
    learnability();std::cout<<"flow transform / virtual-point smoothing / differential with both: PASS\n";
    fusion_auto();std::cout<<"fused per-region selection (auto) / differential: PASS\n";
    learned_root();std::cout<<"learned root routing with exact correction / splits refit / differential: PASS\n";
    joint_root();std::cout<<"joint G+T+V root: monotone blocks / guard / budgets / exact lookups / split remap: PASS\n";
    std::cout<<"ASSERTIONS="<<assertions<<" ALL TESTS PASSED\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
