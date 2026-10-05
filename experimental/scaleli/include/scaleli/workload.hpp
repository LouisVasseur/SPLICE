#pragma once
#include "types.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <unordered_map>
#include <array>

namespace scaleli {
inline std::uint64_t mix64(std::uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
inline std::uint64_t digest_record(Record r){return mix64(r.first)^mix64(r.second+17);}
inline std::uint64_t digest_scan(const std::vector<Record>& rows){std::uint64_t h=rows.size();for(auto r:rows)h=mix64(h^digest_record(r));return h;}
inline std::vector<Record> synthetic(std::size_t n,const std::string& distribution,std::uint64_t seed){
    std::mt19937_64 rng(seed);std::normal_distribution<double> normal(0,2);std::vector<Record> out;out.reserve(n);
    Key k=1024;
    for(std::size_t i=0;i<n;++i){
        if(distribution=="linear")k=1024+4*i;
        else if(distribution=="uniform")k+=1+rng()%1024;
        else if(distribution=="lognormal"){
            const long double x=std::exp(normal(rng))*1000000000.0L;
            k=x>=static_cast<long double>(std::numeric_limits<Key>::max())?std::numeric_limits<Key>::max():Key(x);
        }else if(distribution=="dense_sparse")k+=(i/128)%2?1+rng()%1000000:1+rng()%3;
        else if(distribution=="staircase")k+=i%256?1+rng()%4:1000000000000ULL;
        else if(distribution=="locally_hard")k+=i%8==0?10000000:1;
        else if(distribution=="clustered")k+=i%1024?1+rng()%128:1000000000;
        else if(distribution=="near_u64")k=std::numeric_limits<Key>::max()-4*(n-1-i);
        else if(distribution=="duplicates")k=1024+i/8;
        else throw std::invalid_argument("unknown synthetic distribution: "+distribution);
        out.emplace_back(k,mix64(i));
    }return out;
}
inline std::uint64_t read_integer(std::istream& in,unsigned width){
    char b[8]{};in.read(b,width);if(!in)throw std::runtime_error("truncated integer file");
    std::uint64_t x=0;for(unsigned i=0;i<width;++i)x|=std::uint64_t(static_cast<unsigned char>(b[i]))<<(8*i);return x;
}
inline std::vector<Record> read_dataset(const std::filesystem::path& path,const std::string& format,unsigned width,std::size_t limit=0){
    if(width!=4 && width!=8)throw std::invalid_argument("dtype must be uint32 or uint64");
    const auto bytes=std::filesystem::file_size(path);const std::uint64_t header=format=="sosd"?8:0;
    if(format!="sosd" && format!="raw")throw std::invalid_argument("format must be sosd or raw");
    if(bytes<header || (bytes-header)%width)throw std::invalid_argument("invalid dataset byte length");
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open dataset");
    auto count=header?read_integer(in,8):(bytes/width);
    if(count!=(bytes-header)/width)throw std::invalid_argument("header count does not match file length");
    if(limit)count=std::min<std::uint64_t>(count,limit);
    std::vector<Record> out;out.reserve(count);
    for(std::uint64_t i=0;i<count;++i)out.emplace_back(read_integer(in,width),mix64(i));
    return out;
}
enum class OpKind : unsigned { ReadHit,ReadMiss,Insert,Update,Erase,Scan };
inline const char* op_name(OpKind k){static const char* names[]={"read_hit","read_miss","insert","update","erase","scan"};return names[unsigned(k)];}
struct Operation {OpKind kind;Key key;Value value=0;std::size_t length=0;};
struct WorkloadConfig {
    std::size_t operations=20000,scan_length=100;
    double read=1,insert=0,update=0,erase=0,scan=0,miss=0.1,load_ratio=0.75,zipf_theta=0.99;
    std::string query_distribution="uniform",insert_mode="random",bulk_sampling="uniform",insert_order="shuffled";
    std::uint64_t seed=42;
    void validate()const{
        if(!operations || !scan_length || !(load_ratio>0 && load_ratio<=1) || !(miss>=0 && miss<=1))throw std::invalid_argument("invalid workload sizes/ratios");
        const auto sum=read+insert+update+erase+scan;
        if(read<0||insert<0||update<0||erase<0||scan<0||std::abs(sum-1)>1e-8)throw std::invalid_argument("operation ratios must be nonnegative and sum to one");
        if(query_distribution!="uniform"&&query_distribution!="hotspot"&&query_distribution!="moving_hotspot"&&query_distribution!="zipf")throw std::invalid_argument("unknown query distribution");
        if(insert_mode!="random"&&insert_mode!="append"&&insert_mode!="hotspot"&&insert_mode!="shift")throw std::invalid_argument("unknown insert mode");
        if(bulk_sampling!="uniform"&&bulk_sampling!="prefix")throw std::invalid_argument("bulk sampling must be uniform or prefix");
        if(insert_order!="shuffled"&&insert_order!="sorted")throw std::invalid_argument("insert order must be shuffled or sorted");
        if(!(zipf_theta>0 && std::isfinite(zipf_theta)))throw std::invalid_argument("invalid Zipf theta");
    }
};
struct Workload {
    std::vector<Record> initial;
    std::vector<Operation> trace;
    std::size_t source_rows=0,unique_rows=0;
};
// The query-key sampler, factored out of make_workload so that the warm-up phase can draw
// from the SAME generator as the measured trace instead of an unrelated stride. Body is
// unchanged from the original in-line lambda, so the rng call sequence is identical.
inline std::size_t sample_query_index(std::mt19937_64& rng,std::uniform_real_distribution<double>& unit,
                                      const WorkloadConfig& c,const std::vector<double>& zipf,std::size_t n,std::size_t op){
    std::size_t i=0;
    if(c.query_distribution=="zipf")i=std::size_t(std::lower_bound(zipf.begin(),zipf.begin()+std::ptrdiff_t(n),unit(rng)*zipf[n-1])-zipf.begin());
    else if((c.query_distribution=="hotspot" || c.query_distribution=="moving_hotspot") && unit(rng)<0.8){
        const auto hot=std::max<std::size_t>(1,n/10);i=rng()%hot;
        if(c.query_distribution=="moving_hotspot" && op>=c.operations/2)i+=n-hot;
    }else i=rng()%n;
    return i;
}
// Warm-up keys drawn from the same generator as the measured trace, with a different seed.
// Warm-up runs against the bulk-loaded base (no mutation has happened yet), so the sampler
// is the static branch of make_workload's sample_live over w.initial, plus the same near-miss
// construction at the configured miss ratio. Read keys only: a warm-up must not change the
// index it is warming, and the measured trace must stay the only thing that mutates state.
// Caveat: for moving_hotspot the hot region moves at c.operations/2 of the MEASURED trace;
// a warm-up of a different length crosses that point at a different fraction of itself.
inline std::vector<Key> warmup_keys(const Workload& w,const WorkloadConfig& c,std::size_t n,std::uint64_t seed){
    std::vector<Key> out;const auto m=w.initial.size();if(!n || !m)return out;out.reserve(n);
    std::mt19937_64 rng(seed);std::uniform_real_distribution<double> unit(0,1);
    std::vector<double> zipf;
    if(c.query_distribution=="zipf"){zipf.resize(m);double s=0;for(std::size_t i=0;i<m;++i){s+=1/std::pow(double(i+1),c.zipf_theta);zipf[i]=s;}}
    auto contains=[&](Key k){auto it=std::lower_bound(w.initial.begin(),w.initial.end(),k,[](auto r,Key x){return r.first<x;});return it!=w.initial.end()&&it->first==k;};
    for(std::size_t op=0;op<n;++op){
        Key k=w.initial[sample_query_index(rng,unit,c,zipf,m,op)].first;
        if(unit(rng)<c.miss)for(unsigned attempt=0;attempt<64;++attempt){const Key cand=k+1+attempt;if(!contains(cand)){k=cand;break;}}
        out.push_back(k);
    }
    return out;
}
inline Workload make_workload(std::vector<Record> input,const WorkloadConfig& c){
    c.validate();Workload w;w.source_rows=input.size();input=canonicalize(std::move(input));w.unique_rows=input.size();
    if(input.empty())throw std::invalid_argument("empty benchmark dataset");
    std::mt19937_64 rng(c.seed);std::uniform_real_distribution<double> unit(0,1);
    if(c.bulk_sampling=="uniform")std::shuffle(input.begin(),input.end(),rng);
    const auto load_n=std::max<std::size_t>(1,std::size_t(double(input.size())*c.load_ratio));
    w.initial.assign(input.begin(),input.begin()+std::ptrdiff_t(load_n));w.initial=canonicalize(std::move(w.initial));
    std::vector<Key> heldout;heldout.reserve(input.size()-load_n);
    for(std::size_t i=load_n;i<input.size();++i)heldout.push_back(input[i].first);
    if(c.insert_order=="sorted")std::sort(heldout.begin(),heldout.end());
    else if(c.bulk_sampling=="prefix")std::shuffle(heldout.begin(),heldout.end(),rng);
    input.clear();input.shrink_to_fit();
    const bool dynamic=c.insert+c.update+c.erase>0;
    std::vector<Key> live;std::unordered_map<Key,std::size_t> pos;
    if(dynamic){live.reserve(load_n+c.operations);pos.reserve(load_n+c.operations);for(auto [k,v]:w.initial){pos.emplace(k,live.size());live.push_back(k);}}
    auto contains=[&](Key k){
        if(dynamic)return pos.find(k)!=pos.end();
        auto it=std::lower_bound(w.initial.begin(),w.initial.end(),k,[](auto r,Key x){return r.first<x;});return it!=w.initial.end()&&it->first==k;
    };
    std::vector<double> zipf;
    if(c.query_distribution=="zipf"){
        zipf.resize(load_n+(dynamic?c.operations:0));double s=0;
        for(std::size_t i=0;i<zipf.size();++i){s+=1/std::pow(double(i+1),c.zipf_theta);zipf[i]=s;}
    }
    auto sample_live=[&](std::size_t op)->Key{
        const auto n=dynamic?live.size():w.initial.size();
        const auto i=sample_query_index(rng,unit,c,zipf,n,op);
        return dynamic?live[i]:w.initial[i].first;
    };
    std::size_t insert_cursor=0;Key max_seen=w.initial.back().first;
    auto absent=[&](Key near){
        // Near misses first, falling back to broad-domain misses only when the
        // neighboring interval is full. This does not fabricate false negatives.
        for(unsigned attempt=0;attempt<64;++attempt){Key k=near+1+attempt;if(!contains(k))return k;}
        Key k;do{k=rng();}while(contains(k));return k;
    };
    w.trace.reserve(c.operations);
    for(std::size_t op=0;op<c.operations;++op){
        const auto p=unit(rng);const bool empty=dynamic&&live.empty();Operation x{OpKind::ReadHit,0};
        if(empty || (p>=c.read && p<c.read+c.insert)){
            x.kind=OpKind::Insert;Key k=0;
            if(c.insert_mode=="append" || (c.insert_mode=="shift" && op>=c.operations/2)){
                const Key gap=1+rng()%16;if(max_seen>std::numeric_limits<Key>::max()-gap)throw std::overflow_error("append exhausted uint64 domain");k=max_seen+gap;
            }else if(c.insert_mode=="hotspot" && !empty)k=absent(sample_live(op));
            else{
                while(insert_cursor<heldout.size()&&contains(heldout[insert_cursor]))++insert_cursor;
                k=insert_cursor<heldout.size()?heldout[insert_cursor++]:absent(empty?0:sample_live(op));
            }
            x.key=k;x.value=mix64(rng());max_seen=std::max(max_seen,k);
            pos.emplace(k,live.size());live.push_back(k);
        }else if(p<c.read){
            x.key=sample_live(op);if(unit(rng)<c.miss){x.kind=OpKind::ReadMiss;x.key=absent(x.key);}
        }else if(p<c.read+c.insert+c.update){x.kind=OpKind::Update;x.key=sample_live(op);x.value=mix64(rng());}
        else if(p<c.read+c.insert+c.update+c.erase){
            x.kind=OpKind::Erase;x.key=sample_live(op);const auto at=pos.at(x.key);const auto back=live.back();live[at]=back;pos[back]=at;live.pop_back();pos.erase(x.key);
        }else{x.kind=OpKind::Scan;x.key=sample_live(op);x.length=c.scan_length;}
        w.trace.push_back(x);
    }
    return w;
}
} // namespace scaleli
