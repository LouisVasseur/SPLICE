#pragma once
// Command-line Config and the JSON blocks scaleli_bench prints, shared with other drivers
// (integrations/gre) so that a cell's flags build exactly the Config scaleli_bench builds.
#include "index.hpp"
#include <map>
#include <optional>
#include <set>
#include <sstream>

namespace scaleli {
struct Args {
    std::map<std::string,std::string> a;
    Args(int argc,char** argv){for(int i=1;i<argc;++i){std::string x=argv[i];if(x.rfind("--",0)!=0)throw std::invalid_argument("expected --option");x=x.substr(2);const auto eq=x.find('=');
        if(eq!=std::string::npos)a[x.substr(0,eq)]=x.substr(eq+1);else if(i+1<argc && std::string(argv[i+1]).rfind("--",0)!=0)a[x]=argv[++i];else a[x]="1";}
        const std::set<std::string> allowed={"help","index","n","ops","distribution","seed","profile","routing","policy","codec","region-keys","block-keys","delta-limit","restart","min-saving","smooth-scale","hot-enter","hot-exit","heat-decay","scan-length","load-ratio","miss","query-distribution","zipf-theta","insert-mode","bulk-sampling","insert-order","read","insert","update","erase","scan","data","format","dtype","limit","verify","instrument","dump-layout","dump-trace","warmup","flow","flow-bypass","flow-min-gain","virtual-alpha","relearn","dump-keys","latency","fusion","flow-cost","build-threads","root","qos","root-alpha","prefault","warmup-mode","chunks",
            "root-joint-rounds","root-joint-order","root-joint-kmin","root-joint-kmax","root-joint-gap-charge"};
        for(auto& [k,v]:a)if(!allowed.count(k))throw std::invalid_argument("unknown option: "+k);
    }
    std::string get(std::string k,std::string d)const{auto it=a.find(k);return it==a.end()?d:it->second;}
    std::size_t number(std::string k,std::size_t d)const{auto s=get(k,std::to_string(d));if(s.empty()||s[0]=='-')throw std::invalid_argument("nonnegative integer required: "+k);std::size_t pos;const auto v=std::stoull(s,&pos);if(pos!=s.size())throw std::invalid_argument("invalid integer: "+k);return v;}
    double real(std::string k,double d)const{auto s=get(k,std::to_string(d));std::size_t pos;auto v=std::stod(s,&pos);if(pos!=s.size()||!std::isfinite(v))throw std::invalid_argument("finite number required: "+k);return v;}
    bool flag(std::string k,bool d)const{auto s=get(k,d?"1":"0");if(s!="1"&&s!="0")throw std::invalid_argument("flag requires 0 or 1: "+k);return s=="1";}
};
inline const char* joint_order_name(unsigned o){return o?"tgv":"gtv";}
// The index Config of a command line. `flow` receives the loaded flow so that cfg.flow stays valid.
inline Config config_from_args(const Args& a,std::optional<FlowTransform>& flow){
    Config c;c.region_keys=a.number("region-keys",4096);c.block_keys=a.number("block-keys",128);c.delta_limit=a.number("delta-limit",64);
    c.restart_interval=unsigned(a.number("restart",16));c.routing=parse_routing(a.get("routing","byte"));c.policy=parse_policy(a.get("policy","min_bytes"));c.forced_codec=parse_codec(a.get("codec","for"));
    c.min_saving_fraction=a.real("min-saving",.05);c.flow_bypass=a.flag("flow-bypass",true);c.flow_min_gain=a.real("flow-min-gain",.1);c.virtual_alpha=a.real("virtual-alpha",0);c.relearn_on_compaction=a.flag("relearn",false);c.fusion=parse_fusion(a.get("fusion","manual"));c.flow_cost=a.real("flow-cost",4);c.build_threads=unsigned(a.number("build-threads",1));c.root=parse_root(a.get("root","binary"));c.root_alpha=a.real("root-alpha",0);
    {const auto r=a.number("root-joint-rounds",0),lo=a.number("root-joint-kmin",0),hi=a.number("root-joint-kmax",64);const auto o=a.get("root-joint-order","gtv");
     if(r>255 || lo>65535 || hi>65535)throw std::invalid_argument("root-joint-rounds <= 255 and root-joint-kmin/kmax <= 65535 required");
     if(o!="gtv" && o!="tgv")throw std::invalid_argument("root-joint-order must be gtv or tgv");
     c.root_joint_rounds=std::uint8_t(r);c.root_joint_kmin=std::uint16_t(lo);c.root_joint_kmax=std::uint16_t(hi);c.root_joint_order=std::uint8_t(o=="tgv"?1:0);
     c.root_joint_gap_charge=float(a.real("root-joint-gap-charge",1));}
    if(a.a.count("flow")){flow=FlowTransform::load(a.get("flow",""));c.flow=&*flow;}
    c.smooth_scale=a.real("smooth-scale",1);c.hot_enter=a.real("hot-enter",.25);c.hot_exit=a.real("hot-exit",.05);c.heat_decay=a.real("heat-decay",.99);c.validate();
    return c;
}
inline std::string memory_json(const MemoryUsage& m){std::ostringstream o;o<<"{\"key_bytes\":"<<m.key_bytes<<",\"value_bytes\":"<<m.value_bytes<<",\"metadata_bytes\":"<<m.metadata_bytes<<",\"delta_bytes\":"<<m.delta_bytes<<",\"reserved_slack_bytes\":"<<m.reserved_slack_bytes<<",\"accounted_bytes\":"<<m.accounted_bytes()<<",\"estimated\":"<<(m.estimated?"true":"false")<<'}';return o.str();}
// Joint-root keys, present only when the joint root is configured (root_joint_rounds > 0), so every other
// cell's learnability block is byte-identical to the one before the joint root existed.
inline void joint_json(std::ostream& o,const Index::Learnability& l){
    auto num=[&](double v){if(std::isfinite(v))o<<v;else o<<"null";};
    o<<",\"root_joint\":"<<(l.root_joint?"true":"false");
    const RootJoint* j=l.joint;
    if(!j){o<<",\"root_joint_status\":\"not_fitted\"";return;}
    o<<",\"root_joint_status\":\""<<(l.root_joint?"adopted":j->candidate?"lost":"no_candidate")<<'"';
    o<<",\"root_probes_joint\":";num(j->candidate?j->cost:std::numeric_limits<double>::quiet_NaN());
    o<<",\"root_joint_model_probes\":";num(j->candidate?j->model_probes:std::numeric_limits<double>::quiet_NaN());
    o<<",\"root_joint_count\":"<<j->count<<",\"root_joint_probe_keys\":"<<j->probe_keys
     <<",\"root_joint_gap_probes\":"<<j->gap_probes<<",\"root_joint_k\":"<<j->k<<",\"root_joint_k_eff\":"<<j->k_eff
     <<",\"root_joint_rounds_run\":"<<j->rounds_run<<",\"root_joint_round\":"<<j->round<<",\"root_joint_virtual\":"<<j->virt
     <<",\"root_joint_rescores\":"<<j->rescores<<",\"root_joint_seq\":";num(j->seq_total);
    o<<",\"root_joint_units\":[";
    {const auto p=o.precision(17);for(std::uint32_t h=0;h<j->map.nunits;++h)o<<(h?",":"")<<'['<<j->map.units[h].gain<<','<<j->map.units[h].coef<<']';o.precision(p);}
    o<<"],\"root_joint_gaps\":[";for(std::size_t i=0;i<j->gap_index.size();++i)o<<(i?",":"")<<j->gap_index[i];
    o<<"],\"root_joint_trace\":[";   // [round, accepted (1/0, -1 failed), probe count, total, virtual, k_eff]
    for(std::size_t i=0;i<j->trace.size();++i){const auto& r=j->trace[i];o<<(i?",":"")<<'['<<r.round<<','<<(r.failed?-1:r.accepted?1:0)<<','<<r.count<<',';num(r.failed?std::numeric_limits<double>::quiet_NaN():r.total);o<<','<<r.virt<<','<<r.k_eff<<']';}
    o<<"],\"root_joint_by_k\":{";      // k: [total, probe count, selected round, virtual, k_eff, rounds run]
    for(std::size_t i=0;i<j->by_k.size();++i){const auto& r=j->by_k[i];o<<(i?",":"")<<'"'<<r.k<<"\":[";num(r.candidate?r.total:std::numeric_limits<double>::quiet_NaN());o<<','<<r.count<<','<<r.round<<','<<r.virt<<','<<r.k_eff<<','<<r.rounds_run<<']';}
    o<<"},\"root_joint_build_ns\":"<<j->build_ns<<",\"root_joint_bytes\":"<<j->bytes();
}
inline std::string learnability_json(const Index& ix,const Config& cfg){
    const auto l=ix.learnability();std::ostringstream o;o<<std::setprecision(12)
    <<"{\"regions\":"<<l.regions<<",\"flow_regions\":"<<l.flow_regions<<",\"virtual_points\":"<<l.virtual_points<<",\"keys\":"<<l.keys
    <<",\"rank_sse_before\":"<<l.rank_sse_before<<",\"rank_sse_after\":"<<l.rank_sse_after<<",\"smoothing_ns\":"<<l.smoothing_ns<<",\"transform_ns\":"<<l.transform_ns
    <<",\"tail_conflicts_raw_mean\":"<<l.tail_conflicts_raw_mean<<",\"tail_conflicts_flow_mean\":"<<l.tail_conflicts_flow_mean
    <<",\"flow_bytes\":"<<(cfg.flow?cfg.flow->bytes():0)
    <<",\"choices\":{\"none\":"<<l.choice_none<<",\"flow\":"<<l.choice_flow<<",\"vp\":"<<l.choice_vp<<",\"both\":"<<l.choice_both<<"}"
    <<",\"cost_none_mean\":"<<l.cost_none_mean<<",\"cost_selected_mean\":"<<l.cost_selected_mean
    <<",\"root_model\":"<<(l.root_model?"true":"false")<<",\"root_flow\":"<<(l.root_flow?"true":"false")<<",\"root_probes_binary\":"<<l.root_probes_binary<<",\"root_probes_raw\":"<<l.root_probes_raw<<",\"root_probes_flow\":"<<l.root_probes_flow
    <<",\"root_vp\":"<<(l.root_vp?"true":"false")<<",\"root_virtual\":"<<l.root_virtual<<",\"root_probes_vp_raw\":"<<l.root_probes_vp_raw<<",\"root_probes_vp_flow\":"<<l.root_probes_vp_flow;
    if(cfg.root_joint_rounds)joint_json(o,l);
    o<<"}";return o.str();
}
} // namespace scaleli
