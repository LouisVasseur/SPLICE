import random, sys
from vlib import write_sosd
random.seed(20260921)
M = 2**64 - 1
# (b) 20000 distinct keys ending at 2^64-1, random gaps 1..1000
ks = [M]; 
for _ in range(19999): ks.append(ks[-1] - random.randint(1, 1000))
write_sosd('near_u64', sorted(ks))
# (c) fallback: 2000 dense consecutive keys then 1000 sparse keys 1e9 apart
ks = list(range(1_000_000, 1_002_000)) + [10_000_000_000 + i * 1_000_000_000 for i in range(1000)]
write_sosd('fallback', ks)
# (d) 50000 uniform random uint64
ks = sorted(set(random.getrandbits(64) for _ in range(50000)))
write_sosd('uniform50k', ks)
# (e) 30000 uniform random uint32
ks = sorted(set(random.getrandbits(32) for _ in range(30000)))
write_sosd('uniform32', ks, width=4)
# (f) wide-range extremes 0, 1, 2^63, 2^64-1 plus 5000 random
ks = sorted(set([0, 1, 2**63, 2**63 + 1, M - 1, M] + [random.getrandbits(64) for _ in range(5000)]))
write_sosd('wide', ks)
# (g) duplicates + unsorted for CLI checks
write_sosd('dups', [1, 2, 2, 3, 5, 5, 5, 9] + list(range(10, 5000)))
write_sosd('unsorted', [5, 3, 1, 2])
print('generated')
