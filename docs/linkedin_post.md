# Draft LinkedIn post

Chemists can spend weeks finding the right reaction conditions. I built an AI lab assistant that finds a top-1% reaction in a median of about 18 experiments.

🧪 **The problem:** Even a "simple" C–N coupling has 4 ligands × 3 bases × 22 additives × 15 aryl halides, almost 4,000 combinations. Only 1% of them give ≥ 93% yield. Trial and error burns reagents and weeks of bench time.

🤖 **The approach:** Bayesian optimization on 3,955 real reactions from a published high-throughput screen (Ahneman et al., Science 2018). After every experiment, a Gaussian process predicts the yield of every untested reaction *and how unsure it is*. Expected Improvement then picks the experiment most likely to beat the current best.

📈 **The result** (50 repeated runs each):
• Bayesian optimization reached a top-1% reaction in a median of **14–18 experiments**
• Random search needs **~68** (median), and still missed in 38% of runs after 100 experiments
• Running plates of 5 in parallel got there in **~6 rounds**
• The same code on a second reaction (Suzuki–Miyaura, 3,696 reactions) also took **~18 experiments**, against ~69 for random search

💡 **The honest lesson:** I expected chemistry-aware features (RDKit descriptors, fingerprints) to speed up the search. They didn't: simple one-hot encoding was fastest. Those same features *did* matter when predicting yields for reagents the model had never seen (R² 0.04 → 0.66). Generalizing and searching are different problems.

🔗 Code: <GitHub link> | Live demo: <Streamlit link>
(Attach results/figures/main_benchmark.png and results/figures/demo.gif.)

#MachineLearning #Chemistry #BayesianOptimization #AIforScience #DrugDiscovery
