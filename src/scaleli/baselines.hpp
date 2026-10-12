#pragma once
#include "types.hpp"
#include <map>
#include <memory>

namespace scaleli {
class SortedVector {
    std::vector<Record> rows_;
    auto lb(Key k){return std::lower_bound(rows_.begin(),rows_.end(),k,[](auto r,Key x){return r.first<x;});}
public:
    void bulk_load(std::vector<Record> rows){rows_=canonicalize(std::move(rows));rows_.shrink_to_fit();}
    // Instrumented twin of lb(): identical result, but it counts comparisons and
    // notes the cache line of every record the binary search touches. Taken ONLY when
    // a QueryStats pointer is supplied (the instrument pass), so the timed path below
    // is the untouched std::lower_bound.
    std::optional<Value> find(Key k,QueryStats* s=nullptr){
        if(s){
            std::size_t lo=0,hi=rows_.size();
            while(lo<hi){const auto m=lo+(hi-lo)/2;++s->root_probes;s->note(&rows_[m].first);
                if(rows_[m].first<k)lo=m+1;else hi=m;}
            if(lo<rows_.size()&&rows_[lo].first==k){s->note(&rows_[lo].second);return rows_[lo].second;}
            return {};
        }
        auto it=lb(k);if(it==rows_.end()||it->first!=k)return {};return it->second;}
    bool upsert(Key k,Value v,QueryStats* =nullptr){auto it=lb(k);if(it!=rows_.end()&&it->first==k){it->second=v;return false;}rows_.insert(it,{k,v});return true;}
    bool erase(Key k,QueryStats* =nullptr){auto it=lb(k);if(it==rows_.end()||it->first!=k)return false;rows_.erase(it);return true;}
    std::vector<Record> scan(Key k,std::size_t n,QueryStats* =nullptr){auto it=lb(k);return {it,it+std::ptrdiff_t(std::min<std::size_t>(n,rows_.end()-it))};}
    std::optional<Record> lower_bound(Key k,QueryStats* =nullptr){auto it=lb(k);if(it==rows_.end())return {};return *it;}
    std::size_t size()const{return rows_.size();}
    // One contiguous range: the sorted record array. Same contract as Index::for_each_allocation.
    template<class F> void for_each_allocation(const F& f)const{
        f(static_cast<const void*>(this),sizeof(*this));
        if(!rows_.empty())f(static_cast<const void*>(rows_.data()),rows_.size()*sizeof(Record));
    }
    MemoryUsage memory()const{return {rows_.size()*8,rows_.size()*8,sizeof(*this),0,(rows_.capacity()-rows_.size())*sizeof(Record),false};}
    MaintenanceStats maintenance()const{return {};}
    void maintain(){}
    void validate(){if(!std::is_sorted(rows_.begin(),rows_.end()))throw std::logic_error("unsorted vector");}
};
// Counts allocation requests, including implementation-specific map node size.
// Allocator bookkeeping and heap fragmentation remain outside this number.
struct AllocationCounter { std::size_t bytes=0; };
template<class T> struct CountingAllocator {
    using value_type=T;
    std::shared_ptr<AllocationCounter> counter;
    CountingAllocator():counter(std::make_shared<AllocationCounter>()){}
    explicit CountingAllocator(std::shared_ptr<AllocationCounter> c):counter(std::move(c)){}
    template<class U> CountingAllocator(const CountingAllocator<U>& x)noexcept:counter(x.counter){}
    T* allocate(std::size_t n){auto* p=std::allocator<T>{}.allocate(n);counter->bytes+=n*sizeof(T);return p;}
    void deallocate(T* p,std::size_t n)noexcept{counter->bytes-=n*sizeof(T);std::allocator<T>{}.deallocate(p,n);}
    template<class U> bool operator==(const CountingAllocator<U>& x)const noexcept{return counter==x.counter;}
    template<class U> bool operator!=(const CountingAllocator<U>& x)const noexcept{return !(*this==x);}
};
class OrderedMap {
    std::shared_ptr<AllocationCounter> counter_=std::make_shared<AllocationCounter>();
    using Map=std::map<Key,Value,std::less<Key>,CountingAllocator<std::pair<const Key,Value>>>;
    Map map_{std::less<Key>{},CountingAllocator<std::pair<const Key,Value>>(counter_)};
public:
    void bulk_load(std::vector<Record> rows){map_.clear();for(auto r:canonicalize(std::move(rows)))map_.emplace_hint(map_.end(),r);}
    std::optional<Value> find(Key k,QueryStats* =nullptr){auto it=map_.find(k);if(it==map_.end())return {};return it->second;}
    bool upsert(Key k,Value v,QueryStats* =nullptr){return map_.insert_or_assign(k,v).second;}
    bool erase(Key k,QueryStats* =nullptr){return map_.erase(k)!=0;}
    std::vector<Record> scan(Key k,std::size_t n,QueryStats* =nullptr){std::vector<Record> out;out.reserve(std::min(n,map_.size()));for(auto it=map_.lower_bound(k);it!=map_.end()&&out.size()<n;++it)out.push_back(*it);return out;}
    std::optional<Record> lower_bound(Key k,QueryStats* =nullptr){auto it=map_.lower_bound(k);if(it==map_.end())return {};return Record(*it);}
    std::size_t size()const{return map_.size();}
    MemoryUsage memory()const{return {map_.size()*8,map_.size()*8,sizeof(*this)+counter_->bytes-map_.size()*16+sizeof(AllocationCounter),0,0,false};}
    MaintenanceStats maintenance()const{return {};}
    void maintain(){}
    void validate(){}
};
} // namespace scaleli
