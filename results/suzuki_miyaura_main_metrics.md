# Bayesian optimization vs random search (Suzuki–Miyaura, 3,696 reactions)

50 seeds, budget 100, top-1% threshold = 96.2% yield. Values are mean ± std of the best yield found after N experiments.

| method        | best@10    | best@20    | best@50    | best@100   | found top-1% (runs)   |   median exps to top-1% |
|:--------------|:-----------|:-----------|:-----------|:-----------|:----------------------|------------------------:|
| Random search | 88.8 ± 9.2 | 93.1 ± 4.7 | 95.4 ± 2.0 | 96.8 ± 1.5 | 68%                   |                      61 |
| BO: GP + EI   | 90.8 ± 7.9 | 96.0 ± 3.1 | 98.6 ± 1.0 | 99.3 ± 0.5 | 100%                  |                      18 |
| BO: RF + EI   | 93.6 ± 4.8 | 96.5 ± 2.1 | 98.5 ± 1.1 | 99.1 ± 0.8 | 100%                  |                      18 |
