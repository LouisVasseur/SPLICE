#pragma once
#include "codec.hpp"
#include "model.hpp"
#include "smoothing.hpp"
#include "transform.hpp"
#include <chrono>
#include <thread>
#include <cmath>
#include <iomanip>
#include <memory>
#include <ostream>

namespace scaleli {
enum class Routing { Binary, Rank, Byte };
enum class Policy { Raw, MinBytes, Smooth, Adaptive, Forced };
// Manual: NFL-style conflict auto-switch decides the flow, smoothing always on when alpha > 0.
// Auto:   per region, evaluate {raw, flow} x {ranks, smoothed slots} with the region's own
//         locate_block on its own keys and keep the candidate with the fewest expected probes
//         per lookup, charging flow_cost probe-equivalents per lookup for the transform.
enum class Fusion { Manual, Auto };
// Root routing: Binary = binary search over region fences (log2(regions) probes);
// Model = one linear model over the whole key range predicts the region, corrected by
// exponential + binary search on the fences. This is the one place a GLOBAL key
// transform can pay off: it straightens the whole-range CDF that the root model sees.
enum class Root { Binary, Model };
inline Root parse_root(const std::string& s){
    if(s=="binary")return Root::Binary;if(s=="model")return Root::Model;
    throw std::invalid_argument("root must be binary or model");
}
inline Fusion parse_fusion(const std::string& s){
    if(s=="manual")return Fusion::Manual;if(s=="auto")return Fusion::Auto;
    throw std::invalid_argument("fusion must be manual or auto");
}
inline Routing parse_routing(const std::string& s) {
    if(s=="binary")return Routing::Binary;if(s=="rank")return Routing::Rank;if(s=="byte")return Routing::Byte;
    throw std::invalid_argument("routing must be binary, rank or byte");
}
inline Policy parse_policy(const std::string& s) {
    if(s=="raw")return Policy::Raw;if(s=="min_bytes")return Policy::MinBytes;
    if(s=="smooth")return Policy::Smooth;if(s=="adaptive")return Policy::Adaptive;if(s=="forced")return Policy::Forced;
    throw std::invalid_argument("policy must be raw, min_bytes, smooth, adaptive or forced");
}
struct Config {
    std::size_t region_keys=4096, block_keys=128, delta_limit=64;
    unsigned restart_interval=16;
    Routing routing=Routing::Byte;
    Policy policy=Policy::MinBytes;
    Codec forced_codec=Codec::For;
    double min_saving_fraction=0.05;
    double hot_enter=0.25, hot_exit=0.05, heat_decay=0.99;
    double smooth_scale=1.0;
    // Learnability controls (NFL-style transform, CSV-style virtual points).
    const FlowTransform* flow=nullptr;   // feature = flow(key) instead of normalized key
    bool flow_bypass=true;               // NFL auto-switch: keep the flow only if tail conflicts fall by flow_min_gain
    double flow_min_gain=0.10;
    double virtual_alpha=0;              // CSV smoothing threshold; budget = alpha * region keys; 0 disables
    // Compactions reuse the bulk-load flow decision and the region's virtual points
    // (re-placed among the new keys) unless relearn_on_compaction is set. Relearning
    // repays the full O(lambda * n) smoothing and the transform on every rebuild.
    bool relearn_on_compaction=false;
    Fusion fusion=Fusion::Manual;
    Root root=Root::Binary;
    double root_alpha=0;                 // Root::Model: CSV-style virtual fences budget (fraction of regions); slots map to regions in O(1)
    double flow_cost=4.0;
    unsigned build_threads=1;            // bulk_load builds regions concurrently (regions are independent); queries stay single-threaded                // Auto mode: transform cost per lookup in probe-equivalents (measure it; default is a placeholder)
    void validate()const{
        if(!(virtual_alpha>=0 && virtual_alpha<1) || !(flow_min_gain>=0 && flow_min_gain<=1))throw std::invalid_argument("virtual_alpha in [0,1) and flow_min_gain in [0,1] required");
        if(!(flow_cost>=0) || !std::isfinite(flow_cost))throw std::invalid_argument("flow_cost must be a nonnegative finite number");
        if(!(root_alpha>=0 && root_alpha<64))throw std::invalid_argument("root_alpha must be in [0, 64)");
        if(!region_keys || region_keys>16*1024*1024 || !block_keys || block_keys>region_keys || !delta_limit)
            throw std::invalid_argument("require 1 <= block_keys <= region_keys <= 16777216 and delta_limit > 0");
        if(delta_limit>2*region_keys)throw std::invalid_argument("delta_limit exceeds 2*region_keys");
        if(!restart_interval || restart_interval>65535)throw std::invalid_argument("invalid restart interval");
        if(!(0<=min_saving_fraction && min_saving_fraction<1) || !(0<=hot_exit && hot_exit<hot_enter && hot_enter<=1)
            || !(0<=heat_decay && heat_decay<1) || !(smooth_scale>0 && std::isfinite(smooth_scale)))
            throw std::invalid_argument("invalid policy parameters");
    }
};
struct BlockDescriptor {
    Key first=0,last=0;
    std::size_t rank_begin=0, count=0, offset=0, length=0;
    Codec codec=Codec::Raw;
    std::size_t slot_begin=0; // rank_begin plus virtual points before the block (== rank_begin when no smoothing)
};
struct DeltaEntry { Key key=0; Value value=0; bool deleted=false; };

class Region {
public:
    Key low_fence=0;
    std::vector<BlockDescriptor> blocks;
    std::vector<std::uint8_t> arena;
    std::vector<Value> values;
    std::vector<DeltaEntry> delta;
    LinearModel rank_model,byte_model;
    double write_heat=0;
    bool raw_hot=false;
    // Learnability diagnostics for the last rebuild.
    bool flow_used=false;
    unsigned tail_conflicts_raw=0,tail_conflicts_flow=0;
    std::size_t virtual_points=0;
    std::vector<double> virtual_features; // persisted so compactions can re-place them without a new search
    double rank_sse_before=0,rank_sse_after=0,smoothing_ns=0,transform_ns=0;
    unsigned choice=0;                    // bit 0: flow in use, bit 1: virtual points in use (0 none, 1 flow, 2 vp, 3 both)
    double cost_none=0,cost_selected=0;   // Auto mode: expected probes per lookup of the plain candidate and of the chosen one

    double feature(Key k,const Config& c,QueryStats* s=nullptr)const{
        if(flow_used && c.flow)return (*c.flow)(k,s);
        if(s)s->note(&rank_model); // origin/span read to normalize the key
        return rank_model.normalized(k);}

    void observe(bool write,const Config& c){write_heat=c.heat_decay*write_heat+(1-c.heat_decay)*(write?1:0);}
    bool desired_hot(const Config& c)const{
        return c.policy==Policy::Adaptive && (raw_hot ? write_heat>c.hot_exit : write_heat>=c.hot_enter);
    }
    std::span<const std::uint8_t> bytes(const BlockDescriptor& b)const{return {arena.data()+b.offset,b.length};}
    std::size_t delta_lower(Key k,QueryStats* s=nullptr)const{
        std::size_t lo=0,hi=delta.size();
        while(lo<hi){auto m=lo+(hi-lo)/2;if(s){++s->delta_probes;s->note(&delta[m].key);}if(delta[m].key<k)lo=m+1;else hi=m;}
        return lo;
    }
    std::size_t predicted_block(Key k,const Config& c,QueryStats* s=nullptr)const{
        if(blocks.empty())return 0;
        const auto routing=c.routing;const double f=feature(k,c,s);
        if(s)s->note(routing==Routing::Rank?static_cast<const void*>(&rank_model):static_cast<const void*>(&byte_model));
        const double y = routing==Routing::Rank ? rank_model.predict_x(f) : byte_model.predict_x(f);
        std::size_t lo=0,hi=blocks.size();
        while(lo<hi){
            const auto m=lo+(hi-lo)/2;
            if(s){++s->coordinate_probes;s->note(routing==Routing::Rank?static_cast<const void*>(&blocks[m].slot_begin):static_cast<const void*>(&blocks[m].offset));}
            const auto x=routing==Routing::Rank ? blocks[m].slot_begin : blocks[m].offset;
            if(double(x)<=y)lo=m+1;else hi=m;
        }
        return lo ? lo-1 : 0;
    }
    // Locate the first block with max-key >= query. Query keys need not exist.
    // Predictions are hints. Exponential bracketing + exact binary correction
    // handles arbitrary model error, holes, uint64 endpoints, and distribution drift.
    std::size_t locate_block(Key k,const Config& c,QueryStats* s=nullptr)const{
        const auto n=blocks.size();if(!n)return 0;const auto routing=c.routing;
        auto less=[&](std::size_t i){if(s){++s->fence_probes;s->note(&blocks[i].last);}return blocks[i].last<k;};
        auto lower=[&](std::size_t lo,std::size_t hi){
            while(lo<hi){const auto m=lo+(hi-lo)/2;if(less(m))lo=m+1;else hi=m;}return lo;
        };
        if(routing==Routing::Binary)return lower(0,n);
        const auto p=predicted_block(k,c,s);std::size_t out;
        if(less(p)){
            std::size_t lo=p+1,hi=lo,step=1;
            while(hi<n && less(hi)) {lo=hi+1;step=std::min(n,step*2);hi=std::min(n,p+step);}
            out=lower(lo,hi<n?hi+1:n);
        }else if(p==0 || less(p-1))out=p;
        else{
            std::size_t hi=p,step=1,lo=(p>step?p-step:0);
            while(lo>0 && !less(lo)){hi=lo;step=std::min(n,step*2);lo=p>step?p-step:0;}
            if(less(lo))++lo;out=lower(lo,hi+1);
        }
        if(s){const auto d=out>p?out-p:p-out;++s->block_routes;s->correction_distance+=d;
            s->max_correction_distance=std::max<std::uint64_t>(s->max_correction_distance,d);}
        return out;
    }
    std::optional<Value> find(Key k,const Config& c,QueryStats* s=nullptr)const{
        const auto d=delta_lower(k,s);
        if(d<delta.size() && delta[d].key==k)return delta[d].deleted?std::nullopt:std::optional<Value>(delta[d].value);
        const auto j=locate_block(k,c,s);if(j==blocks.size())return std::nullopt;
        const auto& b=blocks[j];std::size_t lo=0,hi=b.count;
        // offset/length address the arena, count bounds the search, rank_begin addresses the values.
        if(s){s->note(&b.rank_begin);s->note(&b.count);s->note(&b.offset);s->note(&b.length);}
        while(lo<hi){auto m=lo+(hi-lo)/2;if(key_at(bytes(b),m,s)<k)lo=m+1;else hi=m;}
        if(lo<b.count && key_at(bytes(b),lo,s)==k){if(s)s->note(&values[b.rank_begin+lo]);return values[b.rank_begin+lo];}
        return std::nullopt;
    }
    void set_delta(Key k,Value v,bool deleted){
        const auto d=delta_lower(k);
        if(d<delta.size() && delta[d].key==k)delta[d]={k,v,deleted};
        else delta.insert(delta.begin()+std::ptrdiff_t(d),{k,v,deleted});
    }
    // Local merge iterator. A scan decodes at most one block at a time and never
    // materializes the full region merely to retrieve a short range.
    void scan_into(Key start,std::size_t limit,std::vector<Record>& out,const Config& c,QueryStats* s=nullptr)const{
        if(out.size()>=limit)return;
        std::size_t b=locate_block(start,c,s),i=0,d=delta_lower(start,s);
        std::vector<Key> decoded;
        auto load=[&](bool initial){
            if(b<blocks.size()){
                decoded=decode_block(bytes(blocks[b]),s);if(s)++s->blocks_decoded_for_scan;
                i=initial?std::size_t(std::lower_bound(decoded.begin(),decoded.end(),start)-decoded.begin()):0;
            }
        };
        load(true);
        auto advance=[&](){if(++i==decoded.size()){++b;load(false);}};
        while(out.size()<limit && (b<blocks.size() || d<delta.size())){
            const bool base=b<blocks.size();
            if(d<delta.size() && (!base || delta[d].key<=decoded[i])){
                const auto e=delta[d++];if(base && e.key==decoded[i])advance();
                if(!e.deleted)out.emplace_back(e.key,e.value);
            }else{
                out.emplace_back(decoded[i],values[blocks[b].rank_begin+i]);advance();
            }
        }
    }
    std::vector<Record> materialize(const Config& c)const{
        std::vector<Record> all;all.reserve(values.size()+delta.size());
        scan_into(0,std::numeric_limits<std::size_t>::max(),all,c);return all;
    }
    // Fixed logical boundaries intentionally remain identical across codec policies.
    // This is essential for a causal compression-only ablation.
    void rebuild(std::span<const Record> rows,const Config& c,bool hot=false,const Region* previous=nullptr){
        Region fresh;fresh.low_fence=low_fence;fresh.write_heat=write_heat;fresh.raw_hot=hot;
        const bool reuse=previous && !c.relearn_on_compaction;
        std::vector<Key> keys;keys.reserve(rows.size());
        std::vector<double> ranks,positions;ranks.reserve(rows.size());positions.reserve(rows.size());
        for(auto r:rows){keys.push_back(r.first);fresh.values.push_back(r.second);}
        const long double span=keys.size()>1?static_cast<long double>(keys.back()-keys.front()):1;
        for(std::size_t begin=0;begin<keys.size();begin+=c.block_keys){
            const auto end=std::min(keys.size(),begin+c.block_keys),n=end-begin;
            const std::span<const Key> part(keys.data()+begin,n);
            EncodedBlock selected=encode_block(part,Codec::Raw,c.restart_interval);
            if(c.policy==Policy::Forced)selected=encode_block(part,c.forced_codec,c.restart_interval);
            else if(c.policy!=Policy::Raw && !hot){
                const auto raw_size=selected.bytes.size();
                const bool smooth=c.policy==Policy::Smooth || c.policy==Policy::Adaptive;
                // Partition key span at midpoints; the controls store the same keys
                // in the same blocks. Dense intervals receive smaller byte targets.
                const long double left=begin==0?0:(static_cast<long double>(keys[begin-1]-keys[0])+static_cast<long double>(keys[begin]-keys[0]))/2;
                const long double right=end==keys.size()?span:(static_cast<long double>(keys[end-1]-keys[0])+static_cast<long double>(keys[end]-keys[0]))/2;
                const double target=double((16.0L*((keys.size()+c.block_keys-1)/c.block_keys)+8.0L*keys.size())
                                          *(right-left)/std::max(1.0L,span))*c.smooth_scale;
                auto score=[&](const EncodedBlock& e){return smooth?std::abs(double(e.bytes.size())-target):double(e.bytes.size());};
                for(auto codec:{Codec::For,Codec::Delta,Codec::Linear}){
                    auto candidate=encode_block(part,codec,c.restart_interval);
                    const bool saving=double(candidate.bytes.size())<=double(raw_size)*(1-c.min_saving_fraction);
                    if(saving && score(candidate)<score(selected))selected=std::move(candidate);
                }
            }
            BlockDescriptor desc{keys[begin],keys[end-1],begin,n,fresh.arena.size(),selected.bytes.size(),selected.codec,begin};
            for(std::size_t j=0;j<n;++j){ranks.push_back(double(begin+j));positions.push_back(double(desc.offset)+selected.key_positions[j]);}
            fresh.arena.insert(fresh.arena.end(),selected.bytes.begin(),selected.bytes.end());fresh.blocks.push_back(desc);
        }
        // Features: normalized keys, or flow-transformed keys when the NFL-style
        // auto-switch accepts the transformation for this region.
        using Clock=std::chrono::steady_clock;auto ns=[](Clock::duration d){return std::chrono::duration<double,std::nano>(d).count();};
        std::vector<double> x(keys.size());
        {LinearModel tmp;tmp.fit(keys,ranks);for(std::size_t i=0;i<keys.size();++i)x[i]=tmp.normalized(keys[i]);}
        std::vector<double> slots=ranks;bool targets_done=false;
        if(keys.size()>1 && !reuse)fresh.tail_conflicts_raw=tail_conflict_degree(x);
        else if(reuse)fresh.tail_conflicts_raw=previous->tail_conflicts_raw;
        if(c.fusion==Fusion::Auto && !reuse && keys.size()>2){
            // Fused selection. Every candidate is scored with the real locate_block on the
            // region's own keys (in-sample expected probes), so the choice minimizes the
            // work the index will actually do, not a proxy such as SSE or conflict degree.
            struct Cand{bool flow,vp;std::vector<double> feat,slots,vf;double sse_before,sse_after,cost;};
            std::vector<Cand> cands;cands.push_back({false,false,x,ranks,{},0,0,0});
            if(c.flow){
                const auto t=Clock::now();std::vector<double> z(keys.size());for(std::size_t i=0;i<keys.size();++i)z[i]=(*c.flow)(keys[i]);
                fresh.transform_ns=ns(Clock::now()-t);
                std::vector<double> zs=z;std::sort(zs.begin(),zs.end());fresh.tail_conflicts_flow=tail_conflict_degree(zs);
                cands.push_back({true,false,std::move(z),ranks,{},0,0,0});
            }
            const std::size_t base=cands.size();
            if(c.virtual_alpha>0)for(std::size_t k=0;k<base;++k){
                const auto t=Clock::now();auto sm=smooth_cdf(cands[k].feat,c.virtual_alpha);fresh.smoothing_ns+=ns(Clock::now()-t);
                Cand cd{cands[k].flow,true,cands[k].feat,{},std::move(sm.virtual_features),sm.sse_before,sm.sse_after,0};
                cd.slots.resize(keys.size());for(std::size_t i=0;i<keys.size();++i)cd.slots[i]=double(sm.slot[i]);
                cands.push_back(std::move(cd));
            }
            for(auto& cd:cands){
                if(!cd.vp){detail::Sums acc;for(std::size_t i=0;i<keys.size();++i)acc.add(cd.feat[i],ranks[i]);cd.sse_before=cd.sse_after=double(acc.sse());}
                fresh.flow_used=cd.flow;fresh.rank_model.fit_xy(keys,cd.feat,cd.slots);
                for(auto& b:fresh.blocks)b.slot_begin=std::size_t(cd.slots[b.rank_begin]);
                Config cc=c;cc.routing=Routing::Rank;QueryStats s;
                for(std::size_t i=0;i<keys.size();++i)(void)fresh.locate_block(keys[i],cc,&s);
                cd.cost=double(s.fence_probes+s.coordinate_probes)/double(keys.size())+(cd.flow?c.flow_cost:0);
            }
            std::size_t best=0;for(std::size_t k=1;k<cands.size();++k)if(cands[k].cost<cands[best].cost)best=k;
            fresh.cost_none=cands[0].cost;fresh.cost_selected=cands[best].cost;
            auto& cd=cands[best];fresh.flow_used=cd.flow;x=std::move(cd.feat);slots=std::move(cd.slots);
            fresh.virtual_features=std::move(cd.vf);fresh.virtual_points=fresh.virtual_features.size();
            fresh.rank_sse_before=cd.sse_before;fresh.rank_sse_after=cd.sse_after;
            for(auto& b:fresh.blocks)b.slot_begin=std::size_t(slots[b.rank_begin]);
            targets_done=true;
        }else if(c.flow && keys.size()>1){
            if(reuse){ // keep the earlier decision; transform only if the flow is in use
                fresh.flow_used=previous->flow_used;fresh.tail_conflicts_flow=previous->tail_conflicts_flow;
                if(fresh.flow_used){const auto t=Clock::now();for(std::size_t i=0;i<keys.size();++i)x[i]=(*c.flow)(keys[i]);fresh.transform_ns=ns(Clock::now()-t);}
            }else{
                const auto t=Clock::now();std::vector<double> z(keys.size());for(std::size_t i=0;i<keys.size();++i)z[i]=(*c.flow)(keys[i]);
                fresh.transform_ns=ns(Clock::now()-t);
                std::vector<double> zs=z;std::sort(zs.begin(),zs.end());fresh.tail_conflicts_flow=tail_conflict_degree(zs);
                const bool gain=fresh.tail_conflicts_flow<fresh.tail_conflicts_raw && double(fresh.tail_conflicts_raw-fresh.tail_conflicts_flow)>=c.flow_min_gain*double(fresh.tail_conflicts_raw);
                fresh.flow_used=!c.flow_bypass || gain;
                if(fresh.flow_used)x=std::move(z);
            }
        }
        // Targets: CSV-style virtual points move rank targets to slot ranks.
        const bool sorted_x=std::is_sorted(x.begin(),x.end());
        if(targets_done){}
        else if(c.virtual_alpha>0 && keys.size()>2 && reuse && sorted_x){
            // Re-place the previous virtual points among the new keys: O(n + v), no search.
            const auto t=Clock::now();std::vector<double> vf;
            for(auto v:previous->virtual_features)if(v>x.front() && v<x.back())vf.push_back(v);
            std::sort(vf.begin(),vf.end());
            detail::Sums before,after;std::size_t j=0;
            for(std::size_t i=0;i<keys.size();++i){
                while(j<vf.size() && vf[j]<x[i]){after.add(vf[j],double(i+j));++j;}
                slots[i]=double(i+j);before.add(x[i],ranks[i]);after.add(x[i],slots[i]);
            }
            fresh.rank_sse_before=double(before.sse());fresh.rank_sse_after=double(after.sse());
            if(after.sse()<=before.sse()){fresh.virtual_features=std::move(vf);fresh.virtual_points=fresh.virtual_features.size();}
            else{slots=ranks;fresh.rank_sse_after=fresh.rank_sse_before;} // stale virtual points no longer help: drop them, no search
            fresh.smoothing_ns=ns(Clock::now()-t);
            for(auto& b:fresh.blocks)b.slot_begin=std::size_t(slots[b.rank_begin]);
        }else if(c.virtual_alpha>0 && keys.size()>2){
            const auto t=Clock::now();auto sm=smooth_cdf(x,c.virtual_alpha);fresh.smoothing_ns=ns(Clock::now()-t);
            fresh.virtual_points=sm.virtual_features.size();fresh.virtual_features=std::move(sm.virtual_features);fresh.rank_sse_before=sm.sse_before;fresh.rank_sse_after=sm.sse_after;
            for(std::size_t i=0;i<keys.size();++i)slots[i]=double(sm.slot[i]);
            for(auto& b:fresh.blocks)b.slot_begin=sm.slot[b.rank_begin];
        }else{
            detail::Sums acc;for(std::size_t i=0;i<keys.size();++i)acc.add(x[i],ranks[i]);fresh.rank_sse_before=fresh.rank_sse_after=double(acc.sse());
        }
        fresh.rank_model.fit_xy(keys,x,slots);fresh.byte_model.fit_xy(keys,x,positions);
        fresh.choice=(fresh.flow_used?1u:0u)|(fresh.virtual_points?2u:0u);
        // shrink_to_fit is a non-binding request; memory() reports actual capacities.
        fresh.arena.shrink_to_fit();fresh.values.shrink_to_fit();fresh.blocks.shrink_to_fit();
        *this=std::move(fresh);
    }
    // Every live heap range this region owns, as (address, bytes). Used to page the
    // structure in before timing; ranges are reported at size(), not capacity(), so a
    // caller never reads the uninitialized slack past the last live byte. After
    // rebuild()'s shrink_to_fit that slack is close to zero anyway, and memory()
    // reports it separately as reserved_slack_bytes.
    template<class F> void for_each_allocation(const F& f)const{
        f(static_cast<const void*>(this),sizeof(Region));
        if(!arena.empty())f(static_cast<const void*>(arena.data()),arena.size());
        if(!values.empty())f(static_cast<const void*>(values.data()),values.size()*sizeof(Value));
        if(!blocks.empty())f(static_cast<const void*>(blocks.data()),blocks.size()*sizeof(BlockDescriptor));
        if(!delta.empty())f(static_cast<const void*>(delta.data()),delta.size()*sizeof(DeltaEntry));
        if(!virtual_features.empty())f(static_cast<const void*>(virtual_features.data()),virtual_features.size()*sizeof(double));
    }
    MemoryUsage memory()const{
        MemoryUsage m;m.key_bytes=arena.size();m.value_bytes=values.size()*sizeof(Value);
        m.metadata_bytes=sizeof(Region)+blocks.size()*sizeof(BlockDescriptor)+virtual_features.size()*sizeof(double);m.delta_bytes=delta.size()*sizeof(DeltaEntry);
        m.reserved_slack_bytes=arena.capacity()-arena.size()+(values.capacity()-values.size())*sizeof(Value)
           +(blocks.capacity()-blocks.size())*sizeof(BlockDescriptor)+(delta.capacity()-delta.size())*sizeof(DeltaEntry);
        return m;
    }
};

class Index {
    Config config_;
    std::vector<std::unique_ptr<Region>> regions_;
    std::size_t size_=0;
    MaintenanceStats maintenance_;
    LinearModel root_model_;bool root_ready_=false,root_flow_=false,root_vp_=false;double root_cost_raw_=0,root_cost_flow_=0,root_cost_binary_=0,root_cost_vp_raw_=0,root_cost_vp_flow_=0;
    std::vector<std::uint32_t> root_slot_to_region_; // virtual-fence slot -> region index (empty without virtual fences)
    std::size_t root_virtual_=0;
    double root_feature(Key k,QueryStats* s=nullptr)const{
        if(s)s->note(&root_model_);
        return root_flow_ ? (*config_.flow)(k,s) : root_model_.normalized(k);}
    // Last region whose low fence is <= k. Region 0 has fence 0, so the answer always exists.
    template<class Ok> std::size_t last_true(Ok ok,std::size_t lo,std::size_t hi)const{ // ok(lo) true; all i >= hi false or hi == n
        while(hi-lo>1){const auto m=lo+(hi-lo)/2;if(ok(m))lo=m;else hi=m;}return lo;
    }
    template<class Ok> std::size_t locate_from_prediction(Ok ok,double y,std::size_t n,const std::vector<std::uint32_t>* table=nullptr,QueryStats* s=nullptr)const{
        std::size_t p;
        if(table && !table->empty()){const auto m=table->size()-1;const auto slot=y<=0?0:(y>=double(m)?m:std::size_t(y));p=(*table)[slot];if(s)s->note(&(*table)[slot]);}
        else p=y<=0?0:(y>=double(n-1)?n-1:std::size_t(y));
        if(ok(p)){std::size_t lo=p,step=1,hi=p+1;while(hi<n && ok(hi)){lo=hi;step*=2;hi=std::min(n,p+step);}return last_true(ok,lo,hi);}
        std::size_t hi=p,step=1,lo=p>step?p-step:0;while(lo>0 && !ok(lo)){hi=lo;step*=2;lo=p>step?p-step:0;}
        return last_true(ok,lo,hi);
    }
    std::size_t locate_region(Key k,QueryStats* s=nullptr)const{
        const auto n=regions_.size();if(n<2)return 0;
        auto ok=[&](std::size_t i){if(s){++s->root_probes;s->note(&regions_[i]);s->note(&regions_[i]->low_fence);}return regions_[i]->low_fence<=k;};
        if(config_.root==Root::Binary || !root_ready_){
            std::size_t lo=0,hi=n;
            while(lo<hi){auto m=lo+(hi-lo)/2;
                if(s){++s->root_probes;s->note(&regions_[m]);s->note(&regions_[m]->low_fence);}
                if(regions_[m]->low_fence<=k)lo=m+1;else hi=m;}
            return lo?lo-1:0;
        }
        return locate_from_prediction(ok,root_model_.predict_x(root_feature(k,s)),n,&root_slot_to_region_,s);
    }
    // Fit the root model over the region fences; choose raw vs flow features by the probes the
    // real locate routine spends on the fences and the fence midpoints (charging flow_cost per lookup).
    // Candidates {raw, flow} x {ranks, virtual-fence slots}; each scored by the probes the real root
    // locate spends on the fences and fence midpoints (+ flow_cost for the flow); binary search if none wins.
    void fit_root(){
        root_ready_=false;root_flow_=false;root_vp_=false;root_virtual_=0;root_slot_to_region_.clear();
        root_cost_raw_=root_cost_flow_=root_cost_binary_=root_cost_vp_raw_=root_cost_vp_flow_=0;
        const auto n=regions_.size();if(config_.root!=Root::Model || n<2)return;
        // Fit on the regions' first real keys, not on region 0's sentinel fence (0). A fit through
        // the sentinel is meaningless whenever the keys start far from 0 (contiguous windows, ID
        // ranges) and previously made the raw-feature root lose to binary search for that reason alone.
        std::vector<Key> fences(n);std::vector<double> ranks(n);
        for(std::size_t j=0;j<n;++j){fences[j]=j==0 && !regions_[0]->blocks.empty()?regions_[0]->blocks.front().first:regions_[j]->low_fence;ranks[j]=double(j);}
        std::vector<Key> probes;probes.reserve(2*n);
        for(std::size_t j=0;j<n;++j){probes.push_back(fences[j]);if(j+1<n)probes.push_back(fences[j]+(fences[j+1]-fences[j])/2);}
        struct Cand{bool flow,vp;LinearModel model;std::vector<std::uint32_t> table;std::size_t virt=0;double cost=0;};
        std::vector<Cand> cands;LinearModel tmp;tmp.fit(fences,ranks);
        for(bool flow:{false,true}){
            if(flow && !config_.flow)continue;
            std::vector<double> x(n);for(std::size_t j=0;j<n;++j)x[j]=flow?(*config_.flow)(fences[j]):tmp.normalized(fences[j]);
            for(bool vp:{false,true}){
                if(vp && !(config_.root_alpha>0 && n>2))continue;
                Cand cd{flow,vp,{},{},0,0};std::vector<double> targets=ranks;
                if(vp){
                    if(!std::is_sorted(x.begin(),x.end()))continue; // slot table needs monotone features
                    auto sm=smooth_cdf(x,config_.root_alpha);cd.virt=sm.virtual_features.size();
                    if(!cd.virt)continue;
                    for(std::size_t j=0;j<n;++j)targets[j]=double(sm.slot[j]);
                    cd.table.assign(n+cd.virt,0);
                    for(std::size_t j=0;j<n;++j){const auto end=j+1<n?sm.slot[j+1]:n+cd.virt;for(auto sidx=sm.slot[j];sidx<end;++sidx)cd.table[sidx]=std::uint32_t(j);}
                }
                cd.model.fit_xy(fences,x,targets);std::size_t count=0;
                for(auto k:probes){auto ok=[&](std::size_t i){++count;return regions_[i]->low_fence<=k;};
                    (void)locate_from_prediction(ok,cd.model.predict_x(flow?(*config_.flow)(k):cd.model.normalized(k)),n,&cd.table);}
                cd.cost=double(count)/double(probes.size())+(flow?config_.flow_cost:0);
                (vp?(flow?root_cost_vp_flow_:root_cost_vp_raw_):(flow?root_cost_flow_:root_cost_raw_))=cd.cost;
                cands.push_back(std::move(cd));
            }
        }
        {std::size_t count=0;for(auto k:probes){std::size_t lo=0,hi=n;while(lo<hi){auto m=lo+(hi-lo)/2;++count;if(regions_[m]->low_fence<=k)lo=m+1;else hi=m;}}root_cost_binary_=double(count)/double(probes.size());}
        if(cands.empty())return;
        std::size_t best=0;for(std::size_t i=1;i<cands.size();++i)if(cands[i].cost<cands[best].cost)best=i;
        if(cands[best].cost<root_cost_binary_){auto& cd=cands[best];root_ready_=true;root_flow_=cd.flow;root_vp_=cd.vp;root_virtual_=cd.virt;root_model_=cd.model;root_slot_to_region_=std::move(cd.table);}
    }
    void compact(std::size_t r,bool force=false){
        auto& region=*regions_[r];const bool hot=region.desired_hot(config_);
        if(region.delta.empty() && !force && hot==region.raw_hot)return;
        auto rows=region.materialize(config_);const Key low=region.low_fence;
        std::uint64_t rewritten=0;
        if(rows.size()>2*config_.region_keys){
            const auto mid=rows.size()/2;
            auto left=std::make_unique<Region>();left->low_fence=low;left->write_heat=region.write_heat;
            auto right=std::make_unique<Region>();right->low_fence=rows[mid].first;right->write_heat=region.write_heat;
            left->rebuild(std::span<const Record>(rows.data(),mid),config_,hot,&region);
            right->rebuild(std::span<const Record>(rows.data()+mid,rows.size()-mid),config_,hot,&region);
            rewritten=left->arena.size()+left->values.size()*8+right->arena.size()+right->values.size()*8;
            // Grow the pointer vector before replacing the old region: allocation
            // failure leaves the original base+delta accessible.
            regions_.reserve(regions_.size()+1);
            regions_.insert(regions_.begin()+std::ptrdiff_t(r+1),std::move(right));regions_[r]=std::move(left);
            ++maintenance_.splits;fit_root();
        }else{
            region.rebuild(rows,config_,hot,&region);rewritten=region.arena.size()+region.values.size()*8;
        }
        ++maintenance_.compactions;maintenance_.bytes_rewritten+=rewritten;
        maintenance_.max_rewrite_bytes=std::max(maintenance_.max_rewrite_bytes,rewritten);
        if(hot)++maintenance_.raw_bypasses;
    }
public:
    explicit Index(Config c={}):config_(c){c.validate();bulk_load({});}
    const Config& config()const{return config_;}
    void bulk_load(std::vector<Record> rows){
        rows=canonicalize(std::move(rows));
        std::vector<std::unique_ptr<Region>> next;
        for(std::size_t i=0;i<rows.size();i+=config_.region_keys){auto r=std::make_unique<Region>();r->low_fence=i?rows[i].first:0;next.push_back(std::move(r));}
        auto build=[&](std::size_t j){const auto i=j*config_.region_keys;
            next[j]->rebuild(std::span<const Record>(rows.data()+i,std::min(config_.region_keys,rows.size()-i)),config_);};
        const unsigned workers=std::max(1u,std::min<unsigned>(config_.build_threads,unsigned(next.size())));
        if(workers<=1){for(std::size_t j=0;j<next.size();++j)build(j);}
        else{ // static interleaved partition; each region's rebuild touches only its own Region and read-only config/flow
            std::vector<std::thread> pool;std::vector<std::exception_ptr> errors(workers);
            for(unsigned w=0;w<workers;++w)pool.emplace_back([&,w]{try{for(std::size_t j=w;j<next.size();j+=workers)build(j);}catch(...){errors[w]=std::current_exception();}});
            for(auto& t:pool)t.join();for(auto& e:errors)if(e)std::rethrow_exception(e);
        }
        if(next.empty())next.push_back(std::make_unique<Region>());
        next.shrink_to_fit();regions_=std::move(next);size_=rows.size();maintenance_={};fit_root();
    }
    std::optional<Value> find(Key k,QueryStats* s=nullptr){
        const auto j=locate_region(k,s);auto& r=*regions_[j];
        // Reaching the region costs its pointer slot plus the region header fields the
        // lookup reads: the vector headers that address blocks/arena/values/delta, the
        // heat counter observe() updates, and the routing/policy config.
        if(s){s->note(&regions_[j]);s->note(&r);s->note(&r.blocks);s->note(&r.arena);s->note(&r.values);
              s->note(&r.delta);s->note(&r.write_heat);s->note(&config_);}
        r.observe(false,config_);return r.find(k,config_,s);
    }
    // Returns true on insertion, false on replacement. Values may be any uint64.
    bool upsert(Key k,Value v,QueryStats* s=nullptr){
        const auto j=locate_region(k,s);auto& r=*regions_[j];r.observe(true,config_);
        const bool exists=r.find(k,config_,s).has_value();r.set_delta(k,v,false);if(!exists)++size_;
        if(r.delta.size()>=config_.delta_limit)compact(j);
        return !exists;
    }
    bool erase(Key k,QueryStats* s=nullptr){
        const auto j=locate_region(k,s);auto& r=*regions_[j];r.observe(true,config_);
        if(!r.find(k,config_,s))return false;r.set_delta(k,0,true);--size_;
        if(r.delta.size()>=config_.delta_limit)compact(j);return true;
    }
    std::vector<Record> scan(Key lo,std::size_t limit,QueryStats* s=nullptr){
        std::vector<Record> out;out.reserve(std::min(limit,size_));if(!limit)return out;
        for(auto j=locate_region(lo,s);j<regions_.size() && out.size()<limit;++j){
            auto& r=*regions_[j];r.observe(false,config_);r.scan_into(lo,limit,out,config_,s);
        }return out;
    }
    std::optional<Record> lower_bound(Key k,QueryStats* s=nullptr){auto v=scan(k,1,s);if(v.empty())return {};return v.front();}
    void maintain(){for(std::size_t r=0;r<regions_.size();++r)compact(r);}
    std::size_t size()const{return size_;}
    std::size_t region_count()const{return regions_.size();}
    std::size_t block_count()const{std::size_t n=0;for(const auto& r:regions_)n+=r->blocks.size();return n;}
    const MaintenanceStats& maintenance()const{return maintenance_;}
    struct Learnability {
        std::size_t regions=0,flow_regions=0,virtual_points=0,keys=0;
        double rank_sse_before=0,rank_sse_after=0,smoothing_ns=0,transform_ns=0;
        double tail_conflicts_raw_mean=0,tail_conflicts_flow_mean=0;
        std::size_t choice_none=0,choice_flow=0,choice_vp=0,choice_both=0;
        double cost_none_mean=0,cost_selected_mean=0; // Auto mode only (0 otherwise)
        bool root_model=false,root_flow=false,root_vp=false;std::size_t root_virtual=0;
        double root_probes_binary=0,root_probes_raw=0,root_probes_flow=0,root_probes_vp_raw=0,root_probes_vp_flow=0; // expected root probes per lookup on fence probes
    };
    // Base-structure diagnostics of the CURRENT regions (bulk load or last compaction).
    Learnability learnability()const{
        Learnability l;l.regions=regions_.size();
        for(const auto& r:regions_){l.keys+=r->values.size();l.flow_regions+=r->flow_used;l.virtual_points+=r->virtual_points;
            l.rank_sse_before+=r->rank_sse_before;l.rank_sse_after+=r->rank_sse_after;l.smoothing_ns+=r->smoothing_ns;l.transform_ns+=r->transform_ns;
            l.tail_conflicts_raw_mean+=r->tail_conflicts_raw;l.tail_conflicts_flow_mean+=r->tail_conflicts_flow;
            l.cost_none_mean+=r->cost_none;l.cost_selected_mean+=r->cost_selected;
            switch(r->choice){case 0:++l.choice_none;break;case 1:++l.choice_flow;break;case 2:++l.choice_vp;break;default:++l.choice_both;}}
        if(l.regions){l.tail_conflicts_raw_mean/=double(l.regions);l.tail_conflicts_flow_mean/=double(l.regions);l.cost_none_mean/=double(l.regions);l.cost_selected_mean/=double(l.regions);}
        l.root_model=root_ready_;l.root_flow=root_flow_;l.root_vp=root_vp_;l.root_virtual=root_virtual_;l.root_probes_binary=root_cost_binary_;l.root_probes_raw=root_cost_raw_;l.root_probes_flow=root_cost_flow_;
        l.root_probes_vp_raw=root_cost_vp_raw_;l.root_probes_vp_flow=root_cost_vp_flow_;
        return l;
    }
    // Walk every allocation the index owns: the index object, the region pointer vector,
    // the root virtual-fence slot table, and each region's own ranges (key arena, value
    // column, block descriptors, delta, virtual features, and the Region header that
    // holds the models and fences). The set is exactly what memory() accounts for.
    template<class F> void for_each_allocation(const F& f)const{
        f(static_cast<const void*>(this),sizeof(Index));
        if(!regions_.empty())f(static_cast<const void*>(regions_.data()),regions_.size()*sizeof(std::unique_ptr<Region>));
        if(!root_slot_to_region_.empty())f(static_cast<const void*>(root_slot_to_region_.data()),root_slot_to_region_.size()*sizeof(std::uint32_t));
        for(const auto& r:regions_)r->for_each_allocation(f);
    }
    MemoryUsage memory()const{
        MemoryUsage m;m.metadata_bytes=sizeof(Index)+regions_.capacity()*sizeof(std::unique_ptr<Region>)+root_slot_to_region_.size()*sizeof(std::uint32_t);
        for(const auto& r:regions_){const auto x=r->memory();m.key_bytes+=x.key_bytes;m.value_bytes+=x.value_bytes;
            m.metadata_bytes+=x.metadata_bytes;m.delta_bytes+=x.delta_bytes;m.reserved_slack_bytes+=x.reserved_slack_bytes;}return m;
    }
    // Expensive audit, deliberately excluded from performance timings.
    void validate(){
        std::optional<Key> prev;std::size_t n=0;
        for(std::size_t r=0;r<regions_.size();++r){
            if(r && regions_[r-1]->low_fence>=regions_[r]->low_fence)throw std::logic_error("unordered fences");
            const auto rows=regions_[r]->materialize(config_);
            for(auto [k,v]:rows){
                if(prev && *prev>=k)throw std::logic_error("duplicate or unordered key");
                if(k<regions_[r]->low_fence || (r+1<regions_.size() && k>=regions_[r+1]->low_fence))throw std::logic_error("fence ownership");
                if(regions_[r]->find(k,config_)!=std::optional<Value>(v))throw std::logic_error("lookup inconsistency");
                prev=k;++n;
            }
        }if(n!=size_)throw std::logic_error("cardinality inconsistency");
    }
    // Base-only diagnostics. Delta contents are intentionally not smuggled into
    // model-error statistics; run at bulkload or after explicit maintenance.
    void dump_layout(std::ostream& out)const{
        out<<"region,local_rank,key,block,key_byte,rank_prediction,byte_prediction,rank_candidate,byte_candidate,codec,block_bytes,region_bytes,slot_rank,feature,flow_used\n";
        out<<std::setprecision(17);
        for(std::size_t r=0;r<regions_.size();++r){const auto& reg=*regions_[r];
            for(std::size_t j=0;j<reg.blocks.size();++j){const auto& b=reg.blocks[j];const auto data=reg.bytes(b);
                Config rank_cfg=config_;rank_cfg.routing=Routing::Rank;Config byte_cfg=config_;byte_cfg.routing=Routing::Byte;
                for(std::size_t i=0;i<b.count;++i){auto k=key_at(data,i);const double f=reg.feature(k,config_);
                    // slot_rank is exact only for the first key of a block; later keys report the block's slot_begin + i as a lower bound.
                    out<<r<<','<<b.rank_begin+i<<','<<k<<','<<j<<','<<double(b.offset)+encoded_position(data,i)<<','
                       <<reg.rank_model.predict_x(f)<<','<<reg.byte_model.predict_x(f)<<','
                       <<reg.predicted_block(k,rank_cfg)<<','<<reg.predicted_block(k,byte_cfg)<<','
                       <<codec_name(b.codec)<<','<<b.length<<','<<reg.arena.size()<<','<<b.slot_begin+i<<','<<f<<','<<(reg.flow_used?1:0)<<'\n';
                }
            }
        }
    }
};
} // namespace scaleli
