#include "scaleli/index.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    scaleli::Config config;
    config.policy = scaleli::Policy::MinBytes;
    config.routing = scaleli::Routing::Byte;
    scaleli::Index index(config);
    index.bulk_load({{10, 100}, {20, 200}, {30, 300}});
    if (index.upsert(20, 999)) throw std::runtime_error("replacement was an insertion");
    if (index.find(20) != std::optional<scaleli::Value>(999))
        throw std::runtime_error("incorrect value");
    const auto result = index.scan(15, 2);
    if (result != std::vector<scaleli::Record>{{20, 999}, {30, 300}})
        throw std::runtime_error("incorrect ordered scan");
    if (!index.erase(10)) throw std::runtime_error("erase failed");
    index.maintain();
    index.validate();
    for (const auto& [key, value] : result) std::cout << key << " => " << value << '\n';
    return 0;
}
