# EDA summary: Suzuki–Miyaura C–C coupling (Perera et al., Science 2018)

- Reactions: **3696** (full factorial would be 3696; 0 combinations are absent from the published data)
- Choices: 4 electrophiles, 3 boron reagents, 11 ligands, 7 bases, 4 solvents
- Missing values: 0; duplicate condition sets: 0
- Yield: mean 43.6%, median 36.4%, max 100.0%
- 0.4% of reactions gave 0% yield
- Share above 50%: 41.0%; above 80%: **17.8%** (659 reactions); above 90%: **8.3%** (308)
- Top-1% threshold: 96.2% yield (37 reactions)

## Variance explained by each variable alone (η²)

- Electrophile: 22.2%
- Boron reagent: 18.0%
- Ligand: 11.1%
- Solvent: 2.9%
- Base: 0.7%
- Sum of main effects: 55.0% → the remaining 45% comes from interactions, which is why a model is needed.

## Ten highest-yielding reactions

| electrophile    | nucleophile   | ligand      | base    | solvent   |   yield |
|:----------------|:--------------|:------------|:--------|:----------|--------:|
| 6-I-quinoline   | Bpin ester    | AmPhos      | LiOt-Bu | MeOH      |   100   |
| 6-I-quinoline   | Bpin ester    | P(o-Tol)3   | LiOt-Bu | MeOH      |   100   |
| 6-I-quinoline   | Bpin ester    | SPhos       | LiOt-Bu | MeOH      |   100   |
| 6-I-quinoline   | Bpin ester    | AmPhos      | LiOt-Bu | MeCN      |    99.2 |
| 6-I-quinoline   | Bpin ester    | AmPhos      | KOH     | MeCN      |    99   |
| 6-I-quinoline   | boronic acid  | PPh3        | LiOt-Bu | MeOH      |    98.7 |
| 6-OTf-quinoline | Bpin ester    | CataCXium A | CsF     | MeCN      |    98.6 |
| 6-OTf-quinoline | boronic acid  | PPh3        | NaHCO3  | MeCN      |    98.5 |
| 6-I-quinoline   | Bpin ester    | P(t-Bu)3    | LiOt-Bu | MeCN      |    98.4 |
| 6-I-quinoline   | boronic acid  | PPh3        | LiOt-Bu | MeCN      |    98.2 |
