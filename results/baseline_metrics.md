# Random-forest yield prediction

Mean ± std over 5 folds (random: 5 repeated 70/30 splits; unseen-group splits: 5-fold GroupKFold).

| Split               | Encoding            |   R² (median) | R² (mean ± std)   | RMSE (yield %)   |
|:--------------------|:--------------------|--------------:|:------------------|:-----------------|
| Random 70/30        | One-hot             |          0.88 | 0.88 ± 0.01       | 9.4 ± 0.4        |
| Unseen additives    | One-hot             |          0.48 | 0.54 ± 0.11       | 18.3 ± 2.3       |
| Unseen aryl halides | One-hot             |          0.04 | -0.74 ± 1.79      | 27.9 ± 7.0       |
| Random 70/30        | RDKit descriptors   |          0.9  | 0.90 ± 0.00       | 8.7 ± 0.2        |
| Unseen additives    | RDKit descriptors   |          0.52 | 0.50 ± 0.16       | 19.1 ± 3.4       |
| Unseen aryl halides | RDKit descriptors   |          0.66 | 0.28 ± 0.70       | 18.2 ± 7.0       |
| Random 70/30        | Morgan fingerprints |          0.92 | 0.92 ± 0.01       | 7.6 ± 0.3        |
| Unseen additives    | Morgan fingerprints |          0.72 | 0.73 ± 0.12       | 14.0 ± 3.4       |
| Unseen aryl halides | Morgan fingerprints |          0.66 | 0.39 ± 0.59       | 16.8 ± 5.0       |
