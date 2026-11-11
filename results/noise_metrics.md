# Robustness to measurement noise

30 seeds, budget 100, top-1% threshold = 92.9% yield. Values are mean ± std of the best yield found after N experiments.

| method              | best@10     | best@20    | best@50    | best@100   | found top-1% (runs)   |   median exps to top-1% |
|:--------------------|:------------|:-----------|:-----------|:-----------|:----------------------|------------------------:|
| GP-EI, exact yields | 81.2 ± 12.1 | 89.4 ± 9.5 | 98.8 ± 2.0 | 99.9 ± 0.2 | 100%                  |                      20 |
| GP-EI, ±5% noise    | 79.7 ± 12.2 | 90.5 ± 8.9 | 98.7 ± 2.0 | 99.8 ± 0.6 | 100%                  |                      18 |
| GP-EI, ±10% noise   | 80.8 ± 11.2 | 89.4 ± 8.8 | 98.2 ± 2.3 | 99.6 ± 0.9 | 100%                  |                      26 |
| Random search       | 79.3 ± 13.6 | 85.0 ± 8.6 | 91.8 ± 4.2 | 93.7 ± 2.6 | 67%                   |                      46 |
