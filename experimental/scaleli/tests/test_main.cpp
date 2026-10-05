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
int main(){try{
    codecs();std::cout<<"codec roundtrips / full-width arithmetic: PASS\n";
    endpoints();std::cout<<"empty / duplicate / uint64 endpoint semantics: PASS\n";
    differential();std::cout<<"45 randomized differential configurations: PASS\n";
    adversarial_routes();std::cout<<"adversarial / intentionally corrupted model predictions: PASS\n";
    maintenance_and_memory();std::cout<<"maintenance / hysteresis / memory accounting: PASS\n";
    learnability();std::cout<<"flow transform / virtual-point smoothing / differential with both: PASS\n";
    fusion_auto();std::cout<<"fused per-region selection (auto) / differential: PASS\n";
    learned_root();std::cout<<"learned root routing with exact correction / splits refit / differential: PASS\n";
    std::cout<<"ASSERTIONS="<<assertions<<" ALL TESTS PASSED\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
