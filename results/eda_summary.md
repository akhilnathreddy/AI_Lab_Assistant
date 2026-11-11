# EDA summary

- Reactions: **3955** (full factorial would be 3960; 5 combinations are absent from the published data)
- Choices: 4 ligands, 3 bases, 22 additives, 15 aryl halides
- Missing values: 0; duplicate condition sets: 0
- Yield: mean 33.1%, median 28.8%, max 100.0%
- 6.0% of reactions gave 0% yield
- Share above 50%: 27.5%; above 80%: **7.1%** (280 reactions); above 90%: **1.8%** (71)
- Top-1% threshold: 92.9% yield (40 reactions)

## Variance explained by each variable alone (η²)

- Aryl halide: 42.6%
- Additive: 14.6%
- Ligand: 8.4%
- Base: 5.0%
- Sum of main effects: 70.7% → the remaining 29% comes from interactions, which is why a model is needed.

## Ten highest-yielding reactions

| ligand        | base   | additive   | aryl_halide   |   yield |
|:--------------|:-------|:-----------|:--------------|--------:|
| t-BuBrettPhos | MTBD   | A02        | 2-I-pyridine  |   100   |
| t-BuXPhos     | MTBD   | A02        | 2-I-pyridine  |   100   |
| t-BuXPhos     | MTBD   | A13        | 2-I-pyridine  |    99.6 |
| AdBrettPhos   | MTBD   | A13        | 2-I-pyridine  |    99   |
| t-BuBrettPhos | MTBD   | A15        | 2-I-pyridine  |    98.8 |
| t-BuXPhos     | MTBD   | A01        | 3-Br-pyridine |    98.7 |
| t-BuXPhos     | MTBD   | A01        | 3-I-pyridine  |    98.3 |
| t-BuBrettPhos | MTBD   | A09        | 2-I-pyridine  |    98.2 |
| t-BuBrettPhos | MTBD   | A13        | 2-I-pyridine  |    97.9 |
| t-BuXPhos     | MTBD   | A01        | 2-I-pyridine  |    97.6 |
