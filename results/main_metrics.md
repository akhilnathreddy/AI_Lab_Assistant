# Bayesian optimization vs random search (Buchwald–Hartwig, 3,955 reactions)

50 seeds, budget 100, top-1% threshold = 92.9% yield. Values are mean ± std of the best yield found after N experiments.

| method        | best@10     | best@20    | best@50    | best@100   | found top-1% (runs)   |   median exps to top-1% |
|:--------------|:------------|:-----------|:-----------|:-----------|:----------------------|------------------------:|
| Random search | 77.6 ± 14.1 | 85.1 ± 8.5 | 91.6 ± 4.3 | 93.3 ± 3.1 | 62%                   |                      59 |
| BO: GP + EI   | 80.5 ± 15.0 | 90.7 ± 9.2 | 98.8 ± 1.9 | 99.9 ± 0.2 | 100%                  |                      18 |
| BO: RF + EI   | 81.6 ± 16.9 | 94.3 ± 6.6 | 98.3 ± 2.1 | 99.2 ± 1.3 | 100%                  |                      14 |
