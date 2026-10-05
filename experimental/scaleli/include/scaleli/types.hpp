#pragma once
#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace scaleli {
using Key = std::uint64_t;
using Value = std::uint64_t;
using Record = std::pair<Key, Value>;

// Per-query counters are collected only when a non-null pointer is passed.
// These are software work counters, NOT hardware cache-miss measurements.
struct QueryStats {
    std::uint64_t root_probes = 0, coordinate_probes = 0, fence_probes = 0;
    std::uint64_t delta_probes = 0, key_at_calls = 0, decoded_keys = 0;
    std::uint64_t codec_bytes_examined = 0, block_routes = 0;
    std::uint64_t correction_distance = 0, max_correction_distance = 0;
    std::uint64_t blocks_decoded_for_scan = 0;
    std::uint64_t transform_calls = 0; // key-space transformations evaluated (NFL-style flow)

    // Distinct 64-byte cache lines touched by ONE lookup, accumulated over lookups.
    // Probes count comparisons; this counts the memory the lookup actually has to
    // reach, which is the quantity locality arguments are about. Still a software
    // model: it counts distinct lines touched, NOT hardware cache misses (a line
    // already resident from an earlier lookup is counted again here).
    // Opt-in: note() is a no-op unless track_lines is set, so the timed path and
    // every existing caller pay nothing.
    static constexpr std::size_t line_capacity = 128; // open-addressed, power of two
    bool track_lines = false;
    std::uint64_t cache_lines = 0;    // sum over completed lookups of distinct lines
    std::uint64_t lines_overflow = 0; // touches dropped because the per-lookup set was full
    std::uintptr_t line_set[line_capacity] = {}; // 0 = empty; stores (address >> 6) + 1
    std::uint32_t line_used = 0;                 // occupied slots in the CURRENT lookup

    void note(const void* p) {
        if (!track_lines) return;
        const std::uintptr_t tag = (reinterpret_cast<std::uintptr_t>(p) >> 6) + 1;
        std::size_t h = std::size_t((static_cast<std::uint64_t>(tag) * 0x9E3779B97F4A7C15ull) >> 57);
        for (std::size_t j = 0; j < line_capacity; ++j) {
            auto& slot = line_set[(h + j) & (line_capacity - 1)];
            if (slot == tag) return;          // already touched by this lookup
            if (slot == 0) { slot = tag; ++line_used; return; }
        }
        ++lines_overflow; // set full: report the loss instead of silently miscounting
    }
    // Note every 64-byte line spanned by [p, p + bytes). Callers that note only the two
    // endpoints of a range silently drop every line in between, which is correct only
    // when the range cannot straddle more than two lines (bytes <= 64). Use this for
    // anything read in bulk (weight matrices, descriptor arrays, decoded runs).
    void note_range(const void* p, std::size_t bytes) {
        if (!track_lines || !bytes) return;
        const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(p);
        for (std::uintptr_t line = a >> 6, last = (a + bytes - 1) >> 6; line <= last; ++line)
            note(reinterpret_cast<const void*>(line << 6));
    }
    // Close the current lookup: bank its distinct-line count and clear the set.
    void new_operation() {
        cache_lines += line_used;
        if (line_used) for (auto& x : line_set) x = 0;
        line_used = 0;
    }
};
struct MaintenanceStats {
    std::uint64_t compactions = 0, splits = 0, bytes_rewritten = 0;
    std::uint64_t max_rewrite_bytes = 0, raw_bypasses = 0;
};
struct MemoryUsage {
    std::uint64_t key_bytes = 0, value_bytes = 0, metadata_bytes = 0;
    std::uint64_t delta_bytes = 0, reserved_slack_bytes = 0;
    // Accounted live allocations; excludes malloc overhead and process RSS.
    std::uint64_t accounted_bytes() const {
        return key_bytes + value_bytes + metadata_bytes + delta_bytes + reserved_slack_bytes;
    }
    bool estimated = false;
};
inline std::vector<Record> canonicalize(std::vector<Record> rows) {
    std::stable_sort(rows.begin(), rows.end(), [](auto a, auto b) { return a.first < b.first; });
    std::size_t dst = 0;
    for (auto r : rows) {
        if (dst && rows[dst - 1].first == r.first) rows[dst - 1].second = r.second;
        else rows[dst++] = r;
    }
    rows.resize(dst);
    return rows; // stable last-input-write wins; no sentinel key/value is reserved.
}
} // namespace scaleli
