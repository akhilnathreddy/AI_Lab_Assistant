# Does chemistry-aware featurization speed up the search?

30 seeds, budget 100, top-1% threshold = 92.9% yield. Values are mean ± std of the best yield found after N experiments.

| method                     | best@10     | best@20     | best@50    | best@100   | found top-1% (runs)   |   median exps to top-1% |
|:---------------------------|:------------|:------------|:-----------|:-----------|:----------------------|------------------------:|
| GP-EI, one-hot             | 81.2 ± 12.1 | 89.4 ± 9.5  | 98.8 ± 2.0 | 99.9 ± 0.2 | 100%                  |                      20 |
| GP-EI, RDKit descriptors   | 79.6 ± 13.1 | 86.2 ± 10.0 | 93.0 ± 6.7 | 98.4 ± 3.4 | 93%                   |                      37 |
| GP-EI, Morgan fingerprints | 77.2 ± 12.7 | 85.5 ± 10.1 | 97.3 ± 3.5 | 99.8 ± 0.2 | 100%                  |                      30 |
