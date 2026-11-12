# Suzuki–Miyaura: Does chemistry-aware featurization speed up the search?

30 seeds, budget 100, top-1% threshold = 96.2% yield. Values are mean ± std of the best yield found after N experiments.

| method                     | best@10    | best@20    | best@50    | best@100   | found top-1% (runs)   |   median exps to top-1% |
|:---------------------------|:-----------|:-----------|:-----------|:-----------|:----------------------|------------------------:|
| GP-EI, one-hot             | 91.4 ± 7.2 | 95.9 ± 3.3 | 98.6 ± 1.1 | 99.3 ± 0.6 | 100%                  |                      18 |
| GP-EI, RDKit descriptors   | 88.8 ± 8.0 | 94.2 ± 4.0 | 97.8 ± 1.8 | 98.9 ± 1.1 | 97%                   |                      30 |
| GP-EI, Morgan fingerprints | 89.8 ± 8.2 | 94.8 ± 3.2 | 98.1 ± 1.6 | 99.4 ± 0.9 | 97%                   |                      28 |
