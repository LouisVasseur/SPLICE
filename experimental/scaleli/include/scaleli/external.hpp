#pragma once
// A C++17 bridge keeps official upstream headers out of the C++20 core build.
// Actual implementations are in integrations/external/bridge.cpp.
#include "types.hpp"
#include <memory>
namespace scaleli {
#define SCALELI_DECLARE_ADAPTER(NAME) \
class NAME { \
    struct Impl; std::unique_ptr<Impl> impl_; \
public: \
    NAME(); ~NAME(); \
    void bulk_load(std::vector<Record>); \
    std::optional<Value> find(Key,QueryStats* =nullptr); \
    bool upsert(Key,Value,QueryStats* =nullptr); \
    bool erase(Key,QueryStats* =nullptr); \
    std::vector<Record> scan(Key,std::size_t,QueryStats* =nullptr); \
    std::size_t size()const; MemoryUsage memory()const; \
    MaintenanceStats maintenance()const{return {};} \
    void maintain(){} void validate(){} \
};
SCALELI_DECLARE_ADAPTER(AlexAdapter)
SCALELI_DECLARE_ADAPTER(DynamicPgmAdapter)
#undef SCALELI_DECLARE_ADAPTER
} // namespace scaleli
