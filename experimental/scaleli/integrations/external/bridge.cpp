#include "scaleli/external.hpp"
#include <alex.h>
#include <pgm/pgm_index_dynamic.hpp>
#include <limits>
namespace scaleli {
struct AlexAdapter::Impl {alex::Alex<Key,Value> tree;};
AlexAdapter::AlexAdapter():impl_(std::make_unique<Impl>()){}
AlexAdapter::~AlexAdapter()=default;
void AlexAdapter::bulk_load(std::vector<Record> rows){rows=canonicalize(std::move(rows));if(rows.size()>std::size_t(std::numeric_limits<int>::max()))throw std::length_error("ALEX bulk-load API accepts int length");impl_=std::make_unique<Impl>();if(!rows.empty())impl_->tree.bulk_load(rows.data(),int(rows.size()));}
std::optional<Value> AlexAdapter::find(Key k,QueryStats*){const auto* p=impl_->tree.get_payload(k);if(!p)return {};return *p;}
bool AlexAdapter::upsert(Key k,Value v,QueryStats*){auto* p=impl_->tree.get_payload(k);if(p){*p=v;return false;}impl_->tree.insert(k,v);return true;}
bool AlexAdapter::erase(Key k,QueryStats*){return impl_->tree.erase(k)!=0;}
std::vector<Record> AlexAdapter::scan(Key k,std::size_t n,QueryStats*){std::vector<Record> out;out.reserve(std::min(n,size()));for(auto it=impl_->tree.lower_bound(k);it!=impl_->tree.end()&&out.size()<n;++it)out.emplace_back(it.key(),it.payload());return out;}
std::size_t AlexAdapter::size()const{return impl_->tree.size();}
MemoryUsage AlexAdapter::memory()const{MemoryUsage m;m.metadata_bytes=sizeof(*this)+impl_->tree.model_size()+impl_->tree.data_size();m.estimated=true;return m;}
struct DynamicPgmAdapter::Impl {
    // Arithmetic V reserves max(V) as tombstone in current upstream. A wrapper
    // selects upstream ItemB, with an explicit flag and full uint64 value domain.
    struct Payload {Value value;};
    using P=pgm::DynamicPGMIndex<Key,Payload>;
    std::unique_ptr<P> tree=std::make_unique<P>();std::size_t n=0;
};
DynamicPgmAdapter::DynamicPgmAdapter():impl_(std::make_unique<Impl>()){}
DynamicPgmAdapter::~DynamicPgmAdapter()=default;
void DynamicPgmAdapter::bulk_load(std::vector<Record> rows){rows=canonicalize(std::move(rows));std::vector<std::pair<Key,Impl::Payload>> input;input.reserve(rows.size());
    for(auto [k,v]:rows){if(k==std::numeric_limits<Key>::max())throw std::invalid_argument("upstream PGM reserves max key; use a restricted-domain comparison");input.push_back({k,{v}});}
    impl_=std::make_unique<Impl>();impl_->tree=std::make_unique<Impl::P>(input.begin(),input.end());impl_->n=rows.size();}
std::optional<Value> DynamicPgmAdapter::find(Key k,QueryStats*){if(k==std::numeric_limits<Key>::max())return {};auto it=impl_->tree->find(k);if(it==impl_->tree->end())return {};return it->second.value;}
bool DynamicPgmAdapter::upsert(Key k,Value v,QueryStats*){if(k==std::numeric_limits<Key>::max())throw std::invalid_argument("PGM sentinel key");const bool fresh=!find(k);impl_->tree->insert_or_assign(k,Impl::Payload{v});if(fresh)++impl_->n;return fresh;}
bool DynamicPgmAdapter::erase(Key k,QueryStats*){if(!find(k))return false;impl_->tree->erase(k);--impl_->n;return true;}
std::vector<Record> DynamicPgmAdapter::scan(Key k,std::size_t n,QueryStats*){std::vector<Record> out;out.reserve(std::min(n,size()));if(k==std::numeric_limits<Key>::max())return out;for(auto it=impl_->tree->lower_bound(k);it!=impl_->tree->end()&&out.size()<n;++it)out.emplace_back(it->first,it->second.value);return out;}
std::size_t DynamicPgmAdapter::size()const{return impl_->n;}
MemoryUsage DynamicPgmAdapter::memory()const{MemoryUsage m;m.metadata_bytes=sizeof(*this)+sizeof(Impl)+impl_->tree->size_in_bytes();m.estimated=true;return m;}
} // namespace scaleli
