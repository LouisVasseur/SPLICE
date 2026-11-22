# ht_<table>: open-source hash tables as GRE indexes

The comparison points for the unordered index hf4 (`include/gret/hashline.h`). Header: `include/gret/sota/hashtable.h`.
Test: `tests/sota/test_hashtable.cpp`. Every table is its authors' code, unmodified, included by path. Nothing here is a
reimplementation.

## The tables

| name | table | source, tag, commit | licence | hash for uint64 | max load factor |
|---|---|---|---|---|---|
| `ht_std` | `std::unordered_map` (the floor) | the standard library | - | `std::hash`, the identity in libstdc++ and libc++ | 1.0, `<L>` 10..1000 |
| `ht_ankerl` | `ankerl::unordered_dense::map` | github.com/martinus/unordered_dense, v5.3.3, `7c58a8b7a48e790fe72f6508eb3b46317cbba9d5` | MIT | its own: one 64x64->128 multiply by 0x9E3779B97F4A7C15, low half xor high half | 0.8, `<L>` 10..100 |
| `ht_phmap` | `phmap::flat_hash_map` | github.com/greg7mdp/parallel-hashmap, v2.1.0, `2a6e2aca3b129b907d3a108ecc562ef9ddc478b4` | Apache-2.0 | its own: one multiply by 0xde5fb9d2630458e9, the halves added | fixed 7/8, no `<L>` |
| `ht_robin` | `tsl::robin_map` | github.com/Tessil/robin-map, v1.4.1, `bd14e6830a1474fed9d2d03f5c3b0683d818d540` | MIT | `MumHash` (in the header: ankerl's integer hash restated), not the library default | 0.5, `<L>` 20..95 |
| `ht_robinpg` | `tsl::robin_pg_map` | same repository | MIT | `std::hash`, the library default | 0.5, `<L>` 20..95 |

Cloned into `/Users/louisvasseur/Downloads/readonly_competitors/{unordered_dense,robin-map,parallel-hashmap}` (shallow,
at the tags above). Include paths: `-I<unordered_dense>/include -I<robin-map>/include -I<parallel-hashmap>`. A table
whose header is not on the include path is left out (`__has_include`), and its name then throws.

`<L>` is the max load factor in percent, appended to the name (`ht_robin90`, `ht_ankerl100`, `ht_std200`); without it
the table keeps its author's default. A value outside the range above throws `std::invalid_argument`, rather than being
clamped silently (tsl clamps to 0.2..0.95, ankerl to at most 1.0).

What each one is:

- **ankerl** v5 is no longer the robin-hood table of v4 that most published comparisons used. The pairs sit in one dense
  `std::vector`; the index over them is groups of 16 slots with one-byte fingerprints, compared 16 at a time (SSE2 / NEON),
  5.5 bytes per slot. A hit reads the group, then the pair.
- **phmap** is the header-only port of Abseil's Swiss table. `absl::flat_hash_map` itself was not used: it needs Abseil's
  compiled library (`raw_hash_set.cc`, `hash.cc` and others), which the rules here exclude. Groups are 16 control bytes
  with SSE2 and 8 without (arm64: phmap has no NEON group).
- **robin** is robin-hood hashing with the pair inside the bucket (24 bytes: pair, 16-bit distance, flag, 32 bits of hash
  in the padding), so a hit is usually one cache line. Its README warns that the default `std::hash` (identity) with
  power-of-two bucket counts can degenerate, and says to "use a better hash function and/or `tsl::robin_pg_map`".
  `ht_robin` takes the first advice, `ht_robinpg` the second (prime bucket counts, identity hash, a modulo by a constant
  prime reached through a function-pointer table).

## Mapping to GRE's interface

- `bulk_load`: `clear()`, `max_load_factor(L)` if `<L>` was given, `reserve(n)` once, then one `emplace` per pair in the
  order given. Peak memory is the final memory (the log line prints both). Throws if the keys are not unique.
- `get`: `find`. `put`: `try_emplace`; true if the key was absent, false and nothing changed if it is stored (as GRE's
  ALEX and B+tree wrappers do). `try_emplace` rather than `emplace` because libstdc++'s `std::unordered_map::emplace`
  (up to GCC 11 at least) allocates a node before looking the key up; for the flat tables the two are the same
  insert. `update`: overwrite a stored key's payload, false for an absent key. `remove`: `erase(key)`.
- `scan`: returns 0, writes nothing.
- `memory_consumption`: bytes the table has asked its allocator for and not returned, plus `sizeof` the table object.
  The test checks that count against each library's own layout (ankerl `index_bytes()` plus the value vector, 24-byte
  robin buckets, phmap control bytes plus slots, std nodes plus bucket pointers).
- One line on stderr per `bulk_load`: `gret: <name> n= buckets= load= max_load= bytes_per_key= peak_bytes_per_key= build_ms=`.
  `max_load` is what the table reports, except for phmap, whose `max_load_factor()` is a stub returning 1: the line
  prints the 7/8 it grows at. Both byte counts include the table object.

## Bytes per key at 200M keys (computed from each library's sizing rule, not measured)

ankerl, phmap and robin round the bucket count up to a power of two, robinpg to the next prime of a fixed list, so `<L>`
moves memory in steps, not smoothly. hf4 runs at 20 to 25.

| name | buckets | load | bytes per key |
|---|---|---|---|
| `ht_ankerl` (default 80), any `<L>` in 75..100 | 2^28 | 0.745 | 23.4 |
| `ht_ankerl<L>`, 38..74 | 2^29 | 0.373 | 30.8 |
| `ht_phmap` | 2^28 - 1 | 0.745 | 22.8 |
| `ht_robin<L>`, 75..95 | 2^28 | 0.745 | 32.2 |
| `ht_robin` (default 50), any `<L>` in 38..74 | 2^29 | 0.373 | 64.4 |
| `ht_robinpg` (default 50), any `<L>` in 50..95 | 402653189 | 0.497 | 48.3 |
| `ht_robinpg<L>`, 25..49 | 805306457 | 0.248 | 96.6 |
| `ht_std` (libstdc++: 24-byte node, 8 bytes per bucket) | about n | 1.0 | 32, and about 40 once glibc rounds each node to 32 bytes |

A sweep at 200M therefore needs only: `ht_std`, `ht_ankerl`, `ht_phmap`, `ht_robin`, `ht_robin90`, `ht_robinpg`.

## Deviations from each table's default configuration

1. The allocator is `CountingAlloc` (std::allocator plus a byte counter), not `std::allocator`, so that
   `memory_consumption` is exact. Measured effect on lookups: none (ankerl 13.4 ns with std::allocator, 12.8 with
   CountingAlloc; phmap 9.9 and 9.9; 2M random keys, called directly).
2. `ht_robin` does not use tsl's default hash, on its author's advice (above). `ht_robinpg` does.
3. Keys are inserted in ascending order, one by one. No table's bulk constructor is used (ankerl has one,
   `replace(vector&&)`; it changes build time only).
4. ankerl's author recommends `huge_page::map` (his `huge_page_allocator`, 2 MB pages) "for tables of more than a few
   MB" (his `doc/benchmarks.md`, `uint64_t` keys, one to two million entries: 0.58 of `map`'s cost on build and 0.91
   on find). It is not used here: every index in the harness, hf4 included, takes its memory from the default
   allocator, so the page size is whatever the node's transparent-huge-page setting gives all of them. `ht_ankerl` is
   his "`map` if you don't know".

Not a deviation, but a limit of GRE's interface: ankerl's batched lookup (`visit`, a chunk of keys in flight at once)
cannot be reached through a one-key `get`.

## Tests

`build/tests/cxx` (-O1, ASan + UBSan) and `build/tests/cxx-O3`, each with the three `-I` above: both print
`test_hashtable: ok (676 cases, 0 failed)`. UBSan did not fire in any of the libraries. With five 10M-key GRE samples
as arguments (libio, stack, osm, genome, wise; a 2M-key slice each) the sanitized build prints `ok (701 cases, 0 failed)`;
with all ten samples, `ok (726 cases, 0 failed)`.
Five deliberate bugs in a copy of the wrapper (update that does not write, put that overwrites, memory without the table
object, remove that reports key 0 as removed, bulk_load that does not clear) each fail it.

Also built and run, same result: clang for x86-64 (`-target x86_64-apple-macos13 -march=haswell`, under Rosetta, which
takes the SSE2 paths of phmap and ankerl) and GCC 15 with libstdc++ (`-std=gnu++17 -Wall -Wextra`, no warnings). Not
run: g++ 11 on Linux x86-64. The header and the three libraries pass `-std=c++17 -pedantic-errors` with clang and
GCC 15, and the libraries use no C++20 library facility, but ankerl 5.3.3 (October 2026) has not been built here with
a compiler as old as GCC 11. If it does not build on the cluster, leave `-I<unordered_dense>/include` out: only
`ht_ankerl` goes away (`__has_include`).

An independent verification pass (scratch file, not in `tests/`) added, on all three builds and under ASan + UBSan:
single keys and pairs at 0, 2^63 and 2^64 - 1, keys equal mod 2^32, dense runs around huge gaps, keys above 2^63,
tables exactly full (`ht_ankerl100` on 2^k keys runs at load 1.0), every `<L>` each table accepts, 5M random keys
(3M under the sanitizers), single-bit flips of stored keys as misses, and keys built to collide: every key in one bucket of robinpg's prime
(the table passes tsl's 8192-probe limit and grows during `bulk_load`: 20000 keys end in 98317 buckets, not 49157),
equal low 16 bits of MumHash for robin (ends in 262144 buckets, not 65536), one group and one fingerprint for ankerl,
one H1 group and one H2 for phmap; then 600000 random put, update, remove and get per table against a `std::map`, and
FIFO churn at a constant size. `memory_consumption` was checked against a count kept by a replaced global
`operator new` (every byte the table takes passes through `CountingAlloc`; nothing bypasses it, and destruction
returns exactly that count), with libc++ and with libstdc++. No failure, no sanitizer report.

Sanity timing from the -O3 build (Apple clang, M-series, 2M random keys, lookups in random order, ns per lookup; not a
benchmark). "direct" is the same table called without GRE's interface, with std::allocator:

| | hit | miss | bytes per key at 2M |
|---|---|---|---|
| ht_std | 38.5 | 50.6 | 40.0 |
| ht_ankerl | 29.7 | 11.0 | 27.5 |
| ht_phmap | 18.0 | 4.8 | 35.7 |
| ht_robin | 16.1 | 18.6 | 50.3 |
| ht_robin90 | 16.1 | 18.2 | 50.3 |
| ht_robinpg | 17.3 | 17.5 | 75.5 |
| ht_robinpg90 | 27.1 | 26.2 | 37.8 |
| hf4 (another run, `-DSOTA_WITH_HF4`) | 12.3 | 14.1 | 20.0 |
| ankerl direct | 18.9 | 7.3 | |
| phmap direct | 14.6 | 4.2 | |
| robin direct | 13.5 | 14.6 | |

The gap between a row and its "direct" row is the virtual call of GRE's interface, which every index in the harness
pays, hf4 included: the same wrapper called through its concrete type costs what the direct call does (13.2 against
13.4 for ankerl, 10.4 against 9.9 for phmap).

## Known limits and things to check on the cluster

- **GCC and `ht_robinpg`.** Built with GCC 15 on this Mac, a hit through the virtual call costs 55 to 63 ns against 13
  called directly (its misses cost 17 ns and `ht_robin`'s hits 15; built with clang the same hit costs 17). The `get` GCC emits is the same in both cases, so this is
  something between that code (the key is spilled around the library's indirect call to its modulo function) and this
  CPU. Whether g++ on x86-64 shows it is unknown. Compare `ht_robinpg` with the "robinpg direct" row of this test on
  Jed before using its number.
- **Identity hash on real keys** (`ht_std`, `ht_robinpg`). On the 10M samples of the ten GRE sets no table grew past
  its reserve, and `ht_robinpg` at load 0.40 costs 17 to 22 ns a hit on all of them. At load 0.79 (`ht_robinpg90` at 10M)
  the same table costs 30 to 111 ns: the keys cluster. At 200M both names sit at load 0.497, and the full sets are 20
  times denser than the samples, so this is not settled here. tsl also grows the table when a probe sequence passes
  8192 buckets; the `buckets=` of the log line shows it if it happens (expected: 402653189).
  An estimate for 200M, made by the verifier from the same 1-in-20 stride samples (keys folded mod 402653189, sample
  counts per window of 2^14 and 2^17 buckets scaled by 20; an estimate, not a measurement): seven sets stay at or
  below a local load of 0.77. **planet** (key range 2^33.1, so the fold overlaps itself) reaches 1.28 in windows of
  2^14 buckets and 1.09 in two windows of 2^17, an excess of about 11 800 keys in one window, past tsl's limit of 8192:
  expect `ht_robinpg` to rehash to 805306457 buckets there (96.6 bytes per key, local load then up to 0.94).
  **libio** (range 2^29.0) sits at a local load of 0.98 to 1.01: it probably keeps 402653189 buckets, with long probe
  sequences in those windows. **stack** (range 2^27.8, below the prime) is the opposite case: the identity is a
  perfect hash there, every key in its own bucket. So on these three sets `ht_robinpg` measures the identity hash
  as much as the table; `ht_robin` (MumHash) is the robin-hood baseline that does not depend on the key distribution.
  A table that fills one bucket was tested for exactness (it grows, and stays exact); see Tests.
- `ht_std`'s count leaves out malloc's own overhead per node (table above).
- ankerl's default index holds at most 2^32 keys.
- Not included: `absl::flat_hash_map` (needs Abseil built) and `boost::unordered_flat_map` (header-only, but needs about
  ten Boost module repositories on the include path).
- Timings here are from a Mac at 2M keys, where the tables are 40 to 150 MB. They say nothing about 200M.
