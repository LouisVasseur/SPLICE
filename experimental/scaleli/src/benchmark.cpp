#include "scaleli/index.hpp"
#include "scaleli/cli.hpp"
#include "scaleli/baselines.hpp"
#include "scaleli/workload.hpp"
#ifdef SCALELI_EXTERNAL
#include "scaleli/external.hpp"
#endif
#include <chrono>
#ifdef __APPLE__
#include <pthread.h>
#include <sys/qos.h>
#endif
#include <optional>
#include <iostream>
#include <map>
#include <sstream>
#include <set>
#include <type_traits>

using namespace scaleli;
using Clock=std::chrono::steady_clock;
static double nanos(Clock::duration d){return std::chrono::duration<double,std::nano>(d).count();}
static std::string quote(const std::string& s){std::string out="\"";for(char c:s){if(c=='"'||c=='\\')out+='\\';if(c=='\n')out+="\\n";else if(c=='\r')out+="\\r";else out+=c;}return out+'"';}
struct OpResult {std::uint64_t checksum=0,rows=0;};
template<class I>OpResult execute(I& ix,const Operation& o,QueryStats* s=nullptr){
    switch(o.kind){
        case OpKind::ReadHit:case OpKind::ReadMiss:{const auto v=ix.find(o.key,s);return {v?mix64(*v)^mix64(o.key):0,0};}
        case OpKind::Insert:case OpKind::Update:return {std::uint64_t(ix.upsert(o.key,o.value,s)),0};
        case OpKind::Erase:return {std::uint64_t(ix.erase(o.key,s)),0};
        case OpKind::Scan:{const auto rows=ix.scan(o.key,o.length,s);return {digest_scan(rows),rows.size()};}
    }throw std::logic_error("invalid operation");
}
// Warm-up. With keys == nullptr this is the original strided sweep: n find() calls on the
// loaded keys strided by 997. With a key vector it replays warm-up queries drawn from the
// same generator as the measured trace (different seed), so the caches, the branch
// predictors and the TLB reach the state the measurement will actually see rather than a
// state produced by a stride that no measured workload uses.
template<class I>void warmup(I& ix,const Workload& w,std::size_t n,const std::vector<Key>* keys=nullptr){
    std::uint64_t h=0;
    if(keys){for(const auto k:*keys){auto v=ix.find(k);h^=v.value_or(0);}}
    else for(std::size_t j=0;j<std::min(n,w.initial.size());++j){auto v=ix.find(w.initial[(j*997)%w.initial.size()].first);h^=v.value_or(0);}
    // The checksum is externally consumed, preventing dead-code elimination.
    static volatile std::uint64_t sink;sink=h;(void)sink;
}
struct Prefault {std::uint64_t bytes=0;double ns=0;bool supported=false;};
// Touch one byte per page of every range the index owns, so that first-touch faults,
// and any re-faults of pages the OS reclaimed or compressed since bulk_load, are paid
// before the clock starts. READS only: writing would dirty copy-on-write pages and
// change the resident footprint that memory_before/after report. The stride is 4096,
// at or below the page size everywhere this runs (16384 on Apple silicon); a stride
// smaller than the page only means touching a page more than once. The final byte of
// each range is touched too, so a range shorter than a page is still covered.
template<class I>Prefault prefault(I& ix){
    Prefault p;
    if constexpr(requires(const I& c){c.for_each_allocation([](const void*,std::size_t){});}){
        p.supported=true;
        constexpr std::size_t stride=4096;std::uint64_t h=0;
        const auto t=Clock::now();
        ix.for_each_allocation([&](const void* addr,std::size_t n){
            if(!n)return;p.bytes+=n;
            const auto* b=static_cast<const volatile unsigned char*>(addr);
            for(std::size_t o=0;o<n;o+=stride)h+=b[o];
            h+=b[n-1];
        });
        p.ns=nanos(Clock::now()-t);
        // Same anti-elimination discipline as warmup(): the sum is externally consumed.
        static volatile std::uint64_t sink;sink=h;(void)sink;
    }
    return p;
}
static std::string json_doubles(const std::vector<double>& v){std::ostringstream o;o<<std::setprecision(12)<<'[';for(std::size_t i=0;i<v.size();++i){if(i)o<<',';o<<v[i];}o<<']';return o.str();}
static double median(std::vector<double> v){if(v.empty())return 0;std::sort(v.begin(),v.end());const auto m=v.size()/2;return v.size()%2?v[m]:(v[m-1]+v[m])/2;}
static double percentile(std::vector<double> a,double q){if(a.empty())return 0;std::sort(a.begin(),a.end());return a[std::min(a.size()-1,std::size_t(std::ceil(q*double(a.size())))-1)];}
static void print_latency(const std::vector<double>& a){
    std::cout<<"{\"count\":"<<a.size()<<",\"p50\":"<<percentile(a,.50)<<",\"p95\":"<<percentile(a,.95)<<",\"p99\":"<<percentile(a,.99)<<",\"max\":"<<(a.empty()?0:*std::max_element(a.begin(),a.end()))<<'}';
}
static void print_memory(const MemoryUsage& m){std::cout<<memory_json(m);}
static std::uint64_t fingerprint(const Workload& w){std::uint64_t h=0;for(auto r:w.initial)h=mix64(h^digest_record(r));for(auto o:w.trace)h=mix64(h^mix64(o.key)^mix64(o.value)^mix64(unsigned(o.kind))^mix64(o.length));return h;}

template<class Maker>int benchmark(Maker make,const Args& a,const Workload& w,const Config& cfg,const WorkloadConfig& wc){
    using I=typename decltype(make())::element_type;
    const bool verify=a.flag("verify",true),instrument=a.flag("instrument",true),latency=a.flag("latency",true);
    const auto warm=a.number("warmup",4096);
    const bool do_prefault=a.flag("prefault",true);
    const auto warmup_mode=a.get("warmup-mode","workload");
    if(warmup_mode!="strided" && warmup_mode!="workload")throw std::invalid_argument("warmup-mode must be strided or workload");
    const auto chunks=a.number("chunks",8);
    if(!chunks || chunks>w.trace.size())throw std::invalid_argument("chunks must be in [1, operations]");
    // Built once and shared by every pass that warms up, so all passes see the same
    // warm-up trace and the (potentially large) Zipf table is paid for only once.
    // Seed offset keeps the warm-up queries distinct from the measured ones.
    const std::vector<Key> warm_keys=warmup_mode=="workload"?warmup_keys(w,wc,std::min(warm,w.initial.size()),wc.seed^0x5741524dULL):std::vector<Key>{};
    const std::vector<Key>* const wk=warmup_mode=="workload"?&warm_keys:nullptr;
    double build_ns=0,elapsed=0,drain_ns=0;MemoryUsage before,after,drained;MaintenanceStats maint,maint_drain;
    Prefault pf;std::vector<double> chunk_ops_s;
    std::uint64_t throughput_digest=0,scan_rows=0;std::size_t final_size=0;
    std::string learn="null";
    // Pass 1: throughput. No per-op timer and no software instrumentation.
    {
        auto ix=make();auto t=Clock::now();ix->bulk_load(w.initial);build_ns=nanos(Clock::now()-t);
        // Learnability diagnostics describe the bulk-loaded base structure; they are read here,
        // after build timing and before warmup/replay, so no extra bulk load is needed.
        if constexpr(std::is_same_v<I,Index>){
            learn=learnability_json(*ix,cfg);
        }
        before=ix->memory();
        if(do_prefault)pf=prefault(*ix);
        warmup(*ix,w,warm,wk);
        // The replay is split into `chunks` equal spans purely to time them separately.
        // elapsed still spans the WHOLE replay from the first operation to the last, so
        // throughput_ops_s is computed exactly as before; the only added work inside the
        // timed region is `chunks` extra Clock::now() pairs.
        chunk_ops_s.reserve(chunks);
        t=Clock::now();auto mark=t;
        for(std::size_t ci=0;ci<chunks;++ci){
            const auto lo=w.trace.size()*ci/chunks,hi=w.trace.size()*(ci+1)/chunks;
            for(std::size_t j=lo;j<hi;++j){const auto& o=w.trace[j];const auto r=execute(*ix,o);throughput_digest=mix64(throughput_digest^r.checksum);scan_rows+=r.rows;}
            const auto now=Clock::now();const auto span=nanos(now-mark);mark=now;
            chunk_ops_s.push_back(span>0?double(hi-lo)*1e9/span:0);
        }
        elapsed=nanos(Clock::now()-t);
        after=ix->memory();maint=ix->maintenance();final_size=ix->size();
        t=Clock::now();ix->maintain();drain_ns=nanos(Clock::now()-t);drained=ix->memory();maint_drain=ix->maintenance();
    }
    // Pass 2: true per-operation timings on a fresh replay. Timer overhead is
    // reported, not subtracted. Scan materialization and result consumption count.
    // With --latency 0 the pass is skipped and every latency block is emitted empty (count 0).
    std::array<std::vector<double>,6> lat;std::array<std::vector<double>,2> phase_lat;
    std::vector<double> compact_lat;std::uint64_t latency_digest=0;
    if(latency){
        auto ix=make();ix->bulk_load(w.initial);if(do_prefault)(void)prefault(*ix);warmup(*ix,w,warm,wk);
        for(std::size_t j=0;j<w.trace.size();++j){const auto& o=w.trace[j];const auto old=ix->maintenance().compactions;
            auto t=Clock::now();const auto r=execute(*ix,o);auto d=nanos(Clock::now()-t);
            latency_digest=mix64(latency_digest^r.checksum);lat[unsigned(o.kind)].push_back(d);phase_lat[j>=w.trace.size()/2].push_back(d);
            if(ix->maintenance().compactions>old)compact_lat.push_back(d);
        }
        if(throughput_digest!=latency_digest)throw std::runtime_error("fresh replays produced different results");
    }
    // Pass 3: strict differential validation, OUTSIDE all timed sections.
    if(verify){
        auto ix=make();ix->bulk_load(w.initial);OrderedMap ref;ref.bulk_load(w.initial);
        for(std::size_t j=0;j<w.trace.size();++j){const auto o=w.trace[j];bool ok=false;
            switch(o.kind){
                case OpKind::ReadHit:case OpKind::ReadMiss:{auto x=ix->find(o.key),y=ref.find(o.key);ok=x==y && (bool(x)==(o.kind==OpKind::ReadHit));break;}
                case OpKind::Insert:case OpKind::Update:{auto x=ix->upsert(o.key,o.value),y=ref.upsert(o.key,o.value);ok=x==y && (x==(o.kind==OpKind::Insert));break;}
                case OpKind::Erase:ok=ix->erase(o.key)==ref.erase(o.key);break;
                case OpKind::Scan:ok=ix->scan(o.key,o.length)==ref.scan(o.key,o.length);break;
            }if(!ok)throw std::runtime_error("differential mismatch at operation "+std::to_string(j));
        }
        ix->validate();if(ix->scan(0,ix->size()+1)!=ref.scan(0,ref.size()+1))throw std::runtime_error("final-state mismatch");
    }
    // Pass 4: software work counters, deliberately separated from timing.
    QueryStats counters;std::uint64_t counter_digest=0;
    if(instrument){auto ix=make();ix->bulk_load(w.initial);warmup(*ix,w,warm,wk);
        // Distinct-cache-line tracking is enabled ONLY here: the per-lookup set is
        // cleared by new_operation() after every operation, so cache_lines is the sum
        // over operations of the distinct 64-byte lines that operation touched.
        counters.track_lines=true;
        for(const auto& o:w.trace){auto r=execute(*ix,o,&counters);counter_digest=mix64(counter_digest^r.checksum);counters.new_operation();}
        counters.track_lines=false;
        if(counter_digest!=throughput_digest)throw std::runtime_error("instrumentation changes results");}
    if(a.a.count("dump-layout")){
        if constexpr(requires(I& ix,std::ostream& out){ix.dump_layout(out);}){
            auto ix=make();ix->bulk_load(w.initial);std::ofstream out(a.get("dump-layout",""));if(!out)throw std::runtime_error("cannot write layout");ix->dump_layout(out);
        }else throw std::invalid_argument("layout dump only supported by scaleli");
    }
    std::vector<double> empty_clock;for(unsigned j=0;j<10000;++j){auto t=Clock::now();empty_clock.push_back(nanos(Clock::now()-t));}
    std::array<std::size_t,6> opcounts{};for(auto o:w.trace)++opcounts[unsigned(o.kind)];
    std::cout<<std::setprecision(12)<<"{\n\"schema_version\":1,\"index\":"<<quote(a.get("index","scaleli"))
        <<",\"distribution\":"<<quote(a.get("distribution","dense_sparse"))<<",\"profile\":"<<quote(a.get("profile","read_only"))
        <<",\"dataset_path\":"<<quote(a.get("data",""))<<",\"dataset_scope\":"<<quote(a.number("limit",0)?"prefix":"full_input")
        <<",\"seed\":"<<wc.seed<<",\"source_rows\":"<<w.source_rows<<",\"unique_rows\":"<<w.unique_rows
        <<",\"initial_rows\":"<<w.initial.size()<<",\"final_rows\":"<<final_size<<",\"operations\":"<<w.trace.size()
        <<",\"trace_fingerprint\":"<<quote(std::to_string(fingerprint(w)))<<",\"result_checksum\":"<<quote(std::to_string(throughput_digest))
        <<",\"verified\":"<<(verify?"true":"false")<<",\"instrumented\":"<<(instrument?"true":"false")<<",\"latency_pass\":"<<(latency?"true":"false")
        <<",\"software_counters_supported\":"<<(std::is_same_v<I,Index>?"true":"false")
        <<",\"policy\":"<<quote(a.get("policy","min_bytes"))<<",\"routing\":"<<quote(a.get("routing","byte"))
        <<",\"forced_codec\":"<<quote(a.get("codec","for"))<<",\"region_keys\":"<<cfg.region_keys<<",\"block_keys\":"<<cfg.block_keys
        <<",\"delta_limit\":"<<cfg.delta_limit<<",\"restart_interval\":"<<cfg.restart_interval
        <<",\"min_saving_fraction\":"<<cfg.min_saving_fraction<<",\"smooth_scale\":"<<cfg.smooth_scale
        <<",\"flow_weights\":"<<quote(a.get("flow",""))<<",\"flow_bypass\":"<<(cfg.flow_bypass?"true":"false")<<",\"flow_min_gain\":"<<cfg.flow_min_gain
        <<",\"virtual_alpha\":"<<cfg.virtual_alpha<<",\"relearn_on_compaction\":"<<(cfg.relearn_on_compaction?"true":"false")<<",\"fusion\":"<<quote(a.get("fusion","manual"))<<",\"flow_cost\":"<<cfg.flow_cost<<",\"build_threads\":"<<cfg.build_threads<<",\"root\":"<<quote(a.get("root","binary"))<<",\"root_alpha\":"<<cfg.root_alpha;
    if(cfg.root_joint_rounds)std::cout<<",\"root_joint_rounds\":"<<unsigned(cfg.root_joint_rounds)<<",\"root_joint_order\":"<<quote(joint_order_name(cfg.root_joint_order))
        <<",\"root_joint_kmin\":"<<cfg.root_joint_kmin<<",\"root_joint_kmax\":"<<cfg.root_joint_kmax<<",\"root_joint_gap_charge\":"<<cfg.root_joint_gap_charge;
    std::cout<<",\"learnability\":"<<learn
        <<",\"query_distribution\":"<<quote(wc.query_distribution)<<",\"insert_mode\":"<<quote(wc.insert_mode)
        <<",\"bulk_sampling\":"<<quote(wc.bulk_sampling)<<",\"insert_order\":"<<quote(wc.insert_order)<<",\"load_ratio\":"<<wc.load_ratio
        <<",\"scan_length\":"<<wc.scan_length<<",\"requested_miss_ratio\":"<<wc.miss<<",\"warmup_reads\":"<<std::min(warm,w.initial.size())
        <<",\"warmup_mode\":"<<quote(warmup_mode)
        <<",\"prefault\":"<<(do_prefault?"true":"false")<<",\"prefault_supported\":"<<(pf.supported?"true":"false")
        <<",\"prefault_bytes\":"<<pf.bytes<<",\"prefault_ns\":"<<pf.ns
        <<",\"build_ns\":"<<build_ns<<",\"elapsed_ns\":"<<elapsed<<",\"throughput_ops_s\":"<<double(w.trace.size())*1e9/elapsed
        <<",\"chunks\":"<<chunks<<",\"throughput_chunks_ops_s\":"<<json_doubles(chunk_ops_s)
        <<",\"steady_state_ratio\":"<<(median(chunk_ops_s)>0?chunk_ops_s.back()/median(chunk_ops_s):0)
        <<",\"scan_records_returned\":"<<scan_rows<<",\"scan_records_per_total_second\":"<<double(scan_rows)*1e9/elapsed
        <<",\"clock_pair_p50_ns\":"<<percentile(empty_clock,.5)<<",\"memory_before\":";print_memory(before);
    std::cout<<",\"memory_after\":";print_memory(after);std::cout<<",\"memory_after_drain\":";print_memory(drained);
    std::cout<<",\"maintenance\":{\"compactions\":"<<maint.compactions<<",\"splits\":"<<maint.splits<<",\"bytes_rewritten\":"<<maint.bytes_rewritten
        <<",\"max_rewrite_bytes\":"<<maint.max_rewrite_bytes<<",\"raw_bypasses\":"<<maint.raw_bypasses<<",\"drain_ns\":"<<drain_ns
        <<",\"drain_additional_rewrite_bytes\":"<<maint_drain.bytes_rewritten-maint.bytes_rewritten<<"},\"latency_ns\":{";
    for(unsigned j=0;j<6;++j){if(j)std::cout<<',';std::cout<<quote(op_name(OpKind(j)))<<':';print_latency(lat[j]);}
    std::cout<<"},\"phase_latency_ns\":[";print_latency(phase_lat[0]);std::cout<<',';print_latency(phase_lat[1]);std::cout<<"],\"compaction_operation_latency_ns\":";print_latency(compact_lat);
    std::cout<<",\"work_counters\":{\"root_probes\":"<<counters.root_probes<<",\"coordinate_probes\":"<<counters.coordinate_probes
        <<",\"fence_probes\":"<<counters.fence_probes<<",\"delta_probes\":"<<counters.delta_probes<<",\"key_at_calls\":"<<counters.key_at_calls
        <<",\"decoded_keys\":"<<counters.decoded_keys<<",\"codec_bytes_examined\":"<<counters.codec_bytes_examined
        <<",\"block_routes\":"<<counters.block_routes<<",\"correction_distance\":"<<counters.correction_distance
        <<",\"max_correction_distance\":"<<counters.max_correction_distance<<",\"blocks_decoded_for_scan\":"<<counters.blocks_decoded_for_scan<<",\"transform_calls\":"<<counters.transform_calls
        <<",\"cache_lines\":"<<counters.cache_lines<<",\"lines_overflow\":"<<counters.lines_overflow
        <<",\"cache_lines_per_operation\":"<<(w.trace.empty()?0.0:double(counters.cache_lines)/double(w.trace.size()))<<"}\n}\n";
    return 0;
}
int main(int argc,char** argv){try{
#ifdef __APPLE__
    // Apple Silicon schedules default-QoS work on efficiency cores at will, which halves
    // throughput for some runs. Ask for the interactive class so timed passes stay on
    // performance cores; --qos 0 keeps the default for comparison. Not a pinning guarantee.
    {bool want=true;for(int i=1;i+1<argc;++i)if(std::string(argv[i])=="--qos")want=std::string(argv[i+1])!="0";
     if(want)pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);}
#endif
    Args a(argc,argv);if(a.flag("help",false)){std::cout<<R"(SCALE-LI research benchmark (C++20)
  --index scaleli|sorted_vector|ordered_map [alex|pgm with optional upstream build]
  --distribution linear|uniform|lognormal|dense_sparse|staircase|locally_hard|clustered|near_u64|duplicates
  --n 50000 --ops 20000 --seed 42
  --profile read_only|read_heavy|scan_heavy|write_heavy|churn|append|shift
  --policy raw|min_bytes|smooth|adaptive|forced --codec raw|for|delta|linear
  --routing binary|rank|byte --region-keys 4096 --block-keys 128 --delta-limit 64 --restart 16
  --data path --format sosd|raw --dtype uint64|uint32 [--limit N (PREFIX ONLY)]
  --query-distribution uniform|hotspot|moving_hotspot|zipf --zipf-theta 0.99
  --insert-mode random|append|hotspot|shift --miss 0.1 --load-ratio 0.75 --scan-length 100
  --bulk-sampling uniform|prefix --insert-order shuffled|sorted
  --read R --insert I --update U --erase D --scan S (must sum to 1)
  --flow weights.txt (NFL-format key transform) --flow-bypass 1 --flow-min-gain 0.1
  --virtual-alpha 0.1 (CSV-style virtual-point smoothing budget per region; 0 = off)
  --relearn 0|1 (1 = rerun flow decision and smoothing search on every compaction; default reuses bulk-load results)
  --fusion manual|auto (auto = per-region cost-based choice among none/flow/virtual points/both) --flow-cost 4 (probe-equivalents per lookup charged to the flow)
  --qos 1|0 (macOS: 1 requests the user-interactive QoS class so timed passes prefer performance cores)
  --root binary|model (model = one global linear model routes to regions, raw or flow feature chosen by measured probes; binary = fence binary search)
  --root-alpha 2 (with --root model: CSV-style virtual fences, budget = alpha * regions, alpha < 64; slots map to regions in O(1); chosen only if it lowers root probes)
  --root-joint-rounds 6 (with --root model: also offer a joint G+T+V root, gap removal + tanh-pair warp + virtual fences at
     root-alpha, fitted together by up to N guarded rounds per k; adopted only if its probe score beats every other root; 0 = off)
  --root-joint-order gtv|tgv --root-joint-kmin 0 --root-joint-kmax 64 (k grid: kmin, then powers of 4 up to kmax)
  --root-joint-gap-charge 1 (probe-equivalents per gap-table comparison; the warp is charged --flow-cost)
  --build-threads 1 (bulk load builds regions concurrently; build_ns becomes wall-clock of the parallel build; queries are always single-threaded)
  --verify 1 --instrument 1 --latency 1 (0 skips the per-operation latency replay; latency blocks are emitted with count 0)
  --warmup 4096 --dump-layout layout.csv --dump-trace trace.csv --dump-keys keys.sosd
  --warmup-mode workload|strided (workload draws warm-up queries from the SAME generator as the
     measured trace with a different seed; strided is the historical 997-stride sweep)
  --prefault 1|0 (1 = read one byte per page of every allocation the index owns, after bulk_load and
     before any timing, so first-touch and reclaim faults are paid up front; reports prefault_bytes/ns)
  --chunks 8 (split the timed replay into K equal spans and report throughput_chunks_ops_s plus
     steady_state_ratio = last chunk / median chunk; a ratio far from 1 means the run was still
     warming up. throughput_ops_s is still the whole-replay number and is unaffected.)
Output: one JSON object. Timing, validation, and instrumentation use separate fresh replays;
the learnability block is read from the throughput pass's bulk-loaded index.
)";return 0;}
    std::optional<FlowTransform> flow;const Config c=config_from_args(a,flow);
    WorkloadConfig wc;const auto profile=a.get("profile","read_only");
    if(profile=="read_only"){}
    else if(profile=="read_heavy"){wc.read=.8;wc.insert=.1;wc.update=.05;wc.erase=.02;wc.scan=.03;}
    else if(profile=="scan_heavy"){wc.read=.45;wc.insert=.05;wc.scan=.5;}
    else if(profile=="write_heavy"){wc.read=.2;wc.insert=.5;wc.update=.2;wc.erase=.1;}
    else if(profile=="churn"){wc.read=.2;wc.insert=.3;wc.update=.2;wc.erase=.3;}
    else if(profile=="append"){wc.read=.5;wc.insert=.5;wc.insert_mode="append";}
    else if(profile=="shift"){wc.read=.4;wc.insert=.4;wc.update=.1;wc.scan=.1;wc.insert_mode="shift";wc.query_distribution="moving_hotspot";}
    else throw std::invalid_argument("unknown profile");
    wc.operations=a.number("ops",20000);wc.scan_length=a.number("scan-length",100);wc.seed=a.number("seed",42);
    wc.read=a.real("read",wc.read);wc.insert=a.real("insert",wc.insert);wc.update=a.real("update",wc.update);wc.erase=a.real("erase",wc.erase);wc.scan=a.real("scan",wc.scan);
    wc.miss=a.real("miss",.1);wc.load_ratio=a.real("load-ratio",.75);wc.query_distribution=a.get("query-distribution",wc.query_distribution);wc.zipf_theta=a.real("zipf-theta",.99);wc.insert_mode=a.get("insert-mode",wc.insert_mode);wc.bulk_sampling=a.get("bulk-sampling",wc.bulk_sampling);wc.insert_order=a.get("insert-order",wc.insert_order);wc.validate();
    std::vector<Record> input;
    if(a.a.count("data")){const auto dtype=a.get("dtype","uint64");if(dtype!="uint64"&&dtype!="uint32")throw std::invalid_argument("invalid dtype");input=read_dataset(a.get("data",""),a.get("format","sosd"),dtype=="uint64"?8:4,a.number("limit",0));}
    else input=synthetic(a.number("n",50000),a.get("distribution","dense_sparse"),wc.seed);
    auto w=make_workload(std::move(input),wc);
    if(a.a.count("dump-trace")){std::ofstream out(a.get("dump-trace",""));if(!out)throw std::runtime_error("cannot write trace");out<<"operation,key,value,length\n";for(auto o:w.trace)out<<op_name(o.kind)<<','<<o.key<<','<<o.value<<','<<o.length<<'\n';}
    if(a.a.count("dump-keys")){std::ofstream out(a.get("dump-keys",""),std::ios::binary);if(!out)throw std::runtime_error("cannot write keys");
        auto put=[&](std::uint64_t v){char b[8];for(unsigned i=0;i<8;++i)b[i]=char(v>>(8*i));out.write(b,8);};put(w.initial.size());for(auto r:w.initial)put(r.first);}
    const auto index=a.get("index","scaleli");
    if(index=="scaleli")return benchmark([&]{return std::make_unique<Index>(c);},a,w,c,wc);
    if(index=="sorted_vector")return benchmark([]{return std::make_unique<SortedVector>();},a,w,c,wc);
    if(index=="ordered_map")return benchmark([]{return std::make_unique<OrderedMap>();},a,w,c,wc);
#ifdef SCALELI_EXTERNAL
    if(index=="alex")return benchmark([]{return std::make_unique<AlexAdapter>();},a,w,c,wc);
    if(index=="pgm")return benchmark([]{return std::make_unique<DynamicPgmAdapter>();},a,w,c,wc);
#endif
    throw std::invalid_argument("unknown or disabled index: "+index);
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
