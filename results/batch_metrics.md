# Running plates of 5 in parallel vs one experiment at a time

30 seeds, budget 100, top-1% threshold = 92.9% yield. Values are mean ± std of the best yield found after N experiments.

| method                        | best@10     | best@20    | best@50    | best@100   | found top-1% (runs)   |   median exps to top-1% |
|:------------------------------|:------------|:-----------|:-----------|:-----------|:----------------------|------------------------:|
| Sequential (q=1)              | 81.2 ± 12.1 | 89.4 ± 9.5 | 98.8 ± 2.0 | 99.9 ± 0.2 | 100%                  |                      20 |
| Batch of 5 (kriging believer) | 78.8 ± 10.8 | 87.7 ± 9.9 | 97.9 ± 2.8 | 99.5 ± 1.6 | 97%                   |                      28 |
| Random, batch of 5            | 77.1 ± 13.1 | 83.0 ± 9.3 | 90.7 ± 4.0 | 94.1 ± 2.9 | 70%                   |                      68 |
