# AI Lab Assistant: Bayesian Reaction Optimization

An ML system that finds the best reaction conditions in as few experiments as possible. It uses Bayesian optimization on a published high-throughput reaction-yield dataset, so no wet lab is needed.

> **Progress (Nov 2026).** Phases 1–5 are complete, including the Suzuki second
> dataset. The Phase 6 app is built and a demo GIF is in
> `results/figures/demo.gif`. Still to do: deploy the app, add the live-demo link
> to the README, optionally record a narrated video, and publish the post (draft
> in `docs/linkedin_post.md`). Notebooks were replaced by reproducible
> scripts in `scripts/`.
>
> **Result:** BO found a top-1% reaction in a median of 14–18 experiments, against
> about 68 for random search (analytic median; mean 96). On Suzuki–Miyaura it took
> about 18, against about 69.

**Headline result to aim for:** "My optimizer found a top-yielding reaction in about 20 experiments, while random search needed 100+."

---

## How it works

A published dataset acts as a **virtual lab**: every combination of conditions already has a measured yield. The optimizer runs this loop:

1. Pick a combination of conditions (ligand, base, additive, aryl halide).
2. "Run" the experiment by looking up its measured yield.
3. Update the surrogate model with the result.
4. Use an acquisition function to choose the next most promising experiment.

Success is measured by how few lookups it takes to reach the top yields.

---

## Tech stack

| Purpose | Tool |
| --- | --- |
| Data handling | pandas, numpy |
| Chemistry features | RDKit |
| Baseline model | scikit-learn (random forest) |
| Bayesian optimization | BoTorch + GPyTorch (or scikit-learn `GaussianProcessRegressor`) |
| Plots | matplotlib / seaborn |
| Demo app | Streamlit |
| Tracking (optional) | Weights & Biases or MLflow |

---

## Repository structure

```
ai-lab-assistant/
├── data/
│   ├── raw/                 # original dataset files
│   └── processed/           # encoded feature tables
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_optimization_results.ipynb
├── src/
│   ├── data.py              # loading + cleaning
│   ├── features.py          # one-hot and RDKit encodings
│   ├── virtual_lab.py       # lookup-based experiment simulator
│   ├── optimizers.py        # random search, BO (GP + EI), batch BO
│   └── benchmark.py         # repeated runs + metrics
├── app/
│   └── streamlit_app.py
├── results/
│   ├── figures/
│   └── runs/                # saved traces from each benchmark run
├── requirements.txt
└── README.md
```

---

## Phase 1: Data and understanding

**Goal:** Get the dataset clean and understand what drives yield.

- [x] Download the Buchwald-Hartwig C–N coupling dataset (Ahneman et al., *Science* 2018). It contains about 4,000 reactions combining 15 aryl halides, 4 ligands, 3 bases, and 23 additives. It is available from the Doyle lab's public GitHub repository.
- [x] Load it into pandas and check for missing values, duplicates, and the yield distribution.
- [x] Plot yield by each variable (ligand, base, additive, aryl halide) to see which matter most.
- [x] Note how rare high yields are: what share of reactions exceed 80%? This number makes your later result more impressive.

**Done when:** you have a clean table of conditions → yield, and 3–4 EDA plots that explain the search space.

---

## Phase 2: Feature encoding and baseline model

**Goal:** Turn conditions into numbers and confirm yield is learnable.

- [x] Build **one-hot encoding** of each condition as the simple baseline representation.
- [x] Build **chemistry-aware encoding** with RDKit: Morgan fingerprints or a small set of descriptors for each component, concatenated.
- [x] Train a random forest yield predictor with each encoding.
- [x] Evaluate with a random split *and* an out-of-sample split (for example, hold out one or more additives entirely). Report R² and RMSE.

**Done when:** you have a results table comparing encodings and split types. This shows you understand that random splits can be overly optimistic.

---

## Phase 3: Virtual lab and optimizers

**Goal:** Build the core optimization loop.

- [x] Write `VirtualLab`: holds the full search space, returns the yield for a requested condition, and tracks which experiments have been run.
- [x] Implement **random search** as the baseline.
- [x] Implement **Bayesian optimization**:
  - Surrogate: Gaussian process (start with a Matérn kernel).
  - Acquisition: Expected Improvement (EI).
  - Start from 5 random initial experiments, then choose one experiment per iteration.
- [x] Optionally add a **random-forest surrogate** variant, which handles one-hot features well and is a useful comparison.

Minimal loop sketch:

```python
def run_bo(lab, X_all, n_init=5, budget=50, seed=0):
    rng = np.random.default_rng(seed)
    tried = list(rng.choice(len(X_all), n_init, replace=False))
    y = [lab.run(i) for i in tried]

    for _ in range(budget - n_init):
        gp = fit_gp(X_all[tried], np.array(y))
        candidates = [i for i in range(len(X_all)) if i not in tried]
        ei = expected_improvement(gp, X_all[candidates], best=max(y))
        nxt = candidates[int(np.argmax(ei))]
        tried.append(nxt)
        y.append(lab.run(nxt))

    return tried, y
```

**Done when:** a single BO run reliably climbs to high yields faster than a single random run.

---

## Phase 4: Benchmarking

**Goal:** Prove the result is real, not luck.

- [x] Run each method 20–50 times with different random seeds, using a fixed budget (for example, 50 experiments).
- [x] Record the best yield found so far at each step.
- [x] Compute metrics:
  - Mean best yield after 10, 20, and 50 experiments.
  - Experiments needed to reach the top 1% (or a 90%+ yield).
- [x] Create the **key chart**: best yield found vs number of experiments, one line per method with a shaded band (mean ± std or interquartile range).

**Done when:** you have the key chart and a short metrics table saved in `results/figures/`.

---

## Phase 5: Make it stronger (pick one or two)

**Goal:** Add depth that sets this apart from a tutorial.

- [x] **Chemistry-aware features in BO:** compare one-hot vs RDKit features inside the optimizer. Does chemical knowledge speed up the search?
- [x] **Batch mode:** suggest 5 experiments at once (for example, with q-EI in BoTorch or a "kriging believer" approach), as real labs run plates in parallel.
- [x] **Second dataset:** repeat the benchmark on the Suzuki–Miyaura coupling dataset (Perera et al., *Science* 2018) to show the method generalizes.
- [x] **Exploration vs exploitation:** compare EI with Upper Confidence Bound at different settings and explain the trade-off.

**Done when:** each extension has its own chart and a one-paragraph takeaway, including what didn't work.

---

## Phase 6: Interactive demo

**Goal:** Let anyone try it from a link.

- [x] Build a Streamlit app with:
  - A "Run next experiment" button.
  - The conditions the optimizer just chose and their yield.
  - A live chart of best yield so far, with random search shown for comparison.
  - A sidebar to pick method, encoding, and number of initial experiments.
- [ ] Deploy it for free on Streamlit Community Cloud or Hugging Face Spaces.
- [ ] Record a 30–60 second screen capture (GIF or video) of the app finding a high yield.

**Done when:** the app runs from a public link and you have a short recording for the post.

---

## Phase 7: Documentation and LinkedIn post

**Goal:** Present the work clearly.

**README checklist:**
- [x] One-sentence summary and the key chart at the top.
- [x] The problem, in plain language.
- [x] How the virtual lab and optimizer work (a simple diagram helps).
- [x] Results table and charts.
- [x] What didn't work and limitations (for example, a virtual lab can't capture real-world noise or new reagents).
- [ ] How to run it locally, and a link to the live demo.
- [x] References.

**LinkedIn post structure:**
1. **Hook:** "Chemists can spend weeks finding the right reaction conditions. I built an AI that does it in ~20 experiments."
2. **Problem:** one or two lines on why trial and error is slow and expensive.
3. **Approach:** Bayesian optimization on 4,000 real reactions, explained simply.
4. **Result:** the key chart plus your headline number.
5. **One honest lesson:** something surprising or something that failed.
6. **Links:** GitHub repo and live demo.
7. **Hashtags:** #MachineLearning #Chemistry #BayesianOptimization #AIforScience #DrugDiscovery

**Done when:** the repo is public and polished, and the post is published with the chart and demo recording.

---

## Ways to stand out

- Use **realistic splits** and **repeated runs**; most student projects skip these.
- Report **experiments to target**, not just final accuracy. It is the metric chemists actually care about.
- Include a **"what didn't work"** section. It signals real experimentation.
- Tie the result to **cost and time saved** in plain terms (fewer reagents, fewer days at the bench).

---

## References

- Ahneman, D. T. et al. "Predicting reaction performance in C–N cross-coupling using machine learning." *Science* (2018).
- Perera, D. et al. "A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow." *Science* (2018).
- Shields, B. J. et al. "Bayesian reaction optimization as a tool for chemical synthesis." *Nature* (2021). This is the main inspiration for the project.
- BoTorch documentation: botorch.org
