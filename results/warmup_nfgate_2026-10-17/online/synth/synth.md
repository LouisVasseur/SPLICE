Synthetic validation: 40 series per scenario and noise level; true warm-up end at max(eps = 0.01, 2 x drift sd)

| noise | scenario | mser | pelt | relax | cov | hw | epoch | online | consensus |
|---|---|---|---|---|---|---|---|---|---|
| quiet | flat | 0.00 / 0.04% | 0.00 / 0.04% | 0.00 / 0.04% (82% found) | 0.00 / 0.04% | 0.00 / 0.04% | 0.00 / 0.04% | 0.00 / 0.04% | 0.00 / 0.04% |
| quiet | exp fast | 0.05 / 0.05% | 0.53 / 0.05% | 0.01 / 0.05% | 1.01 / 0.05% | 1.01 / 0.05% (92% found) | 0.05 / 0.05% | 0.05 / 0.05% | 0.05 / 0.05% |
| quiet | exp mid | 0.05 / 0.05% | 0.55 / 0.05% | 0.02 / 0.05% | 0.25 / 0.05% | 2.07 / 0.06% (95% found) | 0.05 / 0.05% | 0.05 / 0.05% | 0.03 / 0.05% |
| quiet | exp slow | 0.06 / 0.03% | 0.10 / 0.04% | 0.00 / 0.03% | 0.67 / 0.13% | 3.53 / 0.24% (98% found) | 0.09 / 0.03% | 0.21 / 0.05% | 0.04 / 0.03% |
| quiet | stretched | 0.14 / 0.04% | 0.80 / 0.04% | 0.01 / 0.04% | 0.51 / 0.04% | 2.33 / 0.04% | 0.09 / 0.04% | 0.20 / 0.04% | 0.07 / 0.04% |
| quiet | step | 0.00 / 0.05% | 0.00 / 0.06% | 0.36 / 0.04% | 3.70 / 1.07% | 0.38 / 0.05% | 0.22 / 0.05% | 3.70 / 1.07% | 0.11 / 0.05% |
| quiet | late slowdown | 0.00 / 0.05% | 0.00 / 0.06% | none | 4.78 / 3.70% | 0.00 / 0.06% (82% found) | 0.05 / 0.06% | 4.78 / 3.70% | 0.00 / 0.05% |
| noisy | flat | 0.00 / 0.81% | 0.00 / 0.72% | 0.00 / 0.80% (42% found) | 0.00 / 0.77% | 0.00 / 1.02% (57% found) | 0.00 / 0.77% | 0.00 / 0.77% | 0.00 / 0.77% |
| noisy | exp fast | 0.53 / 1.12% | 0.35 / 0.94% | 0.07 / 0.88% (48% found) | 0.82 / 0.87% | 3.26 / 1.00% (68% found) | 0.12 / 0.84% | 0.12 / 0.87% | 0.12 / 0.88% |
| noisy | exp mid | 0.31 / 1.01% | 0.28 / 0.92% | 0.11 / 0.99% (72% found) | 0.01 / 0.93% | 2.57 / 1.01% (52% found) | 0.01 / 0.94% | 0.01 / 0.93% | 0.17 / 0.93% |
| noisy | exp slow | 0.24 / 1.20% | 0.33 / 1.12% | 0.16 / 1.02% (92% found) | 0.35 / 1.11% | 3.21 / 1.19% (57% found) | 0.19 / 1.09% | 0.08 / 1.13% | 0.14 / 1.07% |
| noisy | stretched | 0.22 / 0.83% | 0.38 / 0.83% | 0.11 / 0.66% (80% found) | 0.08 / 0.67% | 2.18 / 0.70% (55% found) | 0.08 / 0.75% | 0.22 / 0.64% (98% found) | 0.14 / 0.74% |
| noisy | step | 0.03 / 1.18% | 0.00 / 1.10% | 0.21 / 1.17% | 3.70 / 1.57% | 3.70 / 1.37% (82% found) | 0.08 / 1.11% | 3.70 / 1.41% | 0.12 / 1.12% |
| noisy | late slowdown | 0.00 / 1.33% | 0.23 / 1.78% | none | 4.78 / 3.22% | 0.00 / 1.37% (38% found) | 0.25 / 1.64% | 4.78 / 3.22% | 0.05 / 1.16% |

Cells: median |log10 error| of the cut / median |bias| of the steady mean after the cut.

Online detector, lookups seen before it stopped (the offline detectors read all 120M):

| noise | scenario | true warm-up end | stopped | median seen | p90 seen | share of run | seen / true end |
|---|---|---|---|---|---|---|---|
| quiet | flat | 0 | 100% | 262K | 334K | 0.22% | - |
| quiet | exp fast | 9.21K | 100% | 721K | 990K | 0.60% | 78.3x |
| quiet | exp mid | 117K | 100% | 786K | 1.07M | 0.66% | 6.7x |
| quiet | exp slow | 3.4M | 100% | 18.4M | 21.6M | 15.35% | 5.4x |
| quiet | stretched | 212K | 100% | 1.38M | 1.7M | 1.15% | 6.5x |
| quiet | step | 5M | 100% | 197K | 275K | 0.16% | 0.0x |
| quiet | late slowdown | 60M | 100% | 262K | 393K | 0.22% | 0.0x |
| noisy | flat | 0 | 100% | 1.25M | 5.4M | 1.04% | - |
| noisy | exp fast | 5.63K | 100% | 1.93M | 5.58M | 1.61% | 343.6x |
| noisy | exp mid | 63.6K | 100% | 2.36M | 12.8M | 1.97% | 37.1x |
| noisy | exp slow | 1.61M | 100% | 15.7M | 53M | 13.11% | 9.8x |
| noisy | stretched | 79.2K | 98% | 2.56M | 15M | 2.13% | 32.3x |
| noisy | step | 5M | 100% | 524K | 6.97M | 0.44% | 0.1x |
| noisy | late slowdown | 60M | 100% | 1.9M | 5.09M | 1.58% | 0.0x |

Per method, over the cells: median of the median |log10 error| (no cut counts as 3), all scenarios and the ones a streaming detector can resolve (all but step and late slowdown):

| method | all | resolvable | mean resolvable | worst resolvable |
|---|---|---|---|---|
| mser | 0.05 | 0.10 | 0.16 | 0.53 |
| pelt | 0.26 | 0.34 | 0.33 | 0.80 |
| relax | 0.09 | 0.01 | 0.05 | 0.16 |
| cov | 0.59 | 0.30 | 0.37 | 1.01 |
| hw | 2.12 | 2.25 | 2.02 | 3.53 |
| epoch | 0.08 | 0.06 | 0.07 | 0.19 |
| online | 0.16 | 0.06 | 0.09 | 0.22 |
| consensus | 0.06 | 0.06 | 0.08 | 0.17 |
