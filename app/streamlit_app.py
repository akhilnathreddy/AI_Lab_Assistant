"""Interactive demo: watch Bayesian optimization hunt for the best reaction.

Run locally:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.data import COMPONENTS, load_dataset  # noqa: E402
from src.features import encode  # noqa: E402
from src.optimizers import BayesOpt, RandomSearch  # noqa: E402
from src.virtual_lab import VirtualLab  # noqa: E402

st.set_page_config(page_title="AI Lab Assistant", page_icon="⚗️", layout="wide")

METHODS = {
    "Bayesian optimization: GP + Expected Improvement": lambda: BayesOpt("gp", "ei"),
    "Bayesian optimization: GP + UCB (β=4)": lambda: BayesOpt("gp", "ucb", beta=4.0),
    "Bayesian optimization: Random forest + EI": lambda: BayesOpt("rf", "ei"),
    "Random search": RandomSearch,
}
ENCODINGS = {"One-hot": "onehot", "RDKit descriptors": "descriptors",
             "Morgan fingerprints": "fingerprint"}
NICE = {"ligand": "Ligand", "base": "Base", "additive": "Additive", "aryl_halide": "Aryl halide"}


@st.cache_data
def data() -> pd.DataFrame:
    return load_dataset()


@st.cache_data
def features(encoding: str) -> np.ndarray:
    return encode(data(), encoding)[0]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
method_name = st.sidebar.selectbox("Method", list(METHODS))
encoding_name = st.sidebar.selectbox("Encoding", list(ENCODINGS),
                                     disabled=method_name == "Random search")
n_init = st.sidebar.slider("Initial random experiments", 1, 20, 5)
batch = st.sidebar.select_slider("Experiments per click (plate size)", [1, 2, 5, 10], value=1)
seed = st.sidebar.number_input("Random seed", 0, 10_000, 0)
config = (method_name, encoding_name, n_init, seed)

if st.session_state.get("config") != config or st.sidebar.button("Reset campaign"):
    st.session_state.config = config
    st.session_state.lab = VirtualLab(data())
    st.session_state.rand_lab = VirtualLab(data())
    st.session_state.rng = np.random.default_rng(seed + 10_000)
    st.session_state.rand_rng = np.random.default_rng(seed + 20_000)
    init = np.random.default_rng(seed).choice(len(data()), size=n_init, replace=False)
    st.session_state.lab.run_batch(init)
    st.session_state.rand_lab.run_batch(init)

lab: VirtualLab = st.session_state.lab
rand_lab: VirtualLab = st.session_state.rand_lab
X = features(ENCODINGS[encoding_name])
threshold = lab.quantile_threshold(0.01)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚗️ AI Lab Assistant")
st.markdown(
    "Finding the best conditions for a **Buchwald–Hartwig C–N coupling** among "
    f"**{lab.n_candidates:,}** combinations of ligand, base, additive and aryl halide. "
    "Every reaction was really run (Ahneman *et al.*, *Science* 2018); this "
    "*virtual lab* simply reveals the measured yield when the optimizer asks for it. "
    f"Only **40 reactions (1%)** reach **{threshold:.1f}%** yield or more."
)

col_a, col_b, col_c = st.columns([1, 1, 2])
step = col_a.button(f"▶ Run next {'experiment' if batch == 1 else f'{batch} experiments'}",
                    type="primary", width="stretch")
step10 = col_b.button("⏩ Run 10 rounds", width="stretch")


def advance(rounds: int) -> None:
    strategy = METHODS[method_name]()
    for _ in range(rounds):
        if lab.n_run >= lab.n_candidates:
            return
        idx, y = lab.observed()
        n = min(batch, lab.n_candidates - lab.n_run)
        with st.spinner("Fitting the model and choosing the next experiment…"):
            lab.run_batch(strategy.suggest(X, idx, y, n, st.session_state.rng))
        ridx, ry = rand_lab.observed()
        rand_lab.run_batch(RandomSearch().suggest(X, ridx, ry, n, st.session_state.rand_rng))


if step:
    advance(1)
if step10:
    advance(10)

idx, y = lab.observed()
best = lab.best_so_far()
first_hit = int(np.argmax(best >= threshold)) + 1 if (best >= threshold).any() else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Experiments run", lab.n_run)
m2.metric("Best yield found", f"{best[-1]:.1f}%",
          delta=f"{best[-1] - rand_lab.best_so_far()[-1]:+.1f} vs random")
m3.metric("Best possible", f"{lab.global_max:.1f}%")
m4.metric("Top-1% reaction found at", f"experiment {first_hit}" if first_hit else "not yet")

# ---------------------------------------------------------------------------
# Latest experiment(s) + chart
# ---------------------------------------------------------------------------
left, right = st.columns([2, 3])
with left:
    st.subheader("Latest experiment" if batch == 1 else "Latest plate")
    last = lab.history[-batch:] if lab.n_run > n_init else lab.history[-n_init:]
    for i, yi in reversed(last):
        cond = lab.conditions(i)
        star = " ⭐" if yi >= threshold else ""
        st.markdown(f"**{yi:.1f}% yield**{star}  \n" +
                    " · ".join(f"{NICE[c]}: `{cond[c]}`" for c in COMPONENTS))

with right:
    st.subheader("Best yield found so far")
    chart_df = pd.concat([
        pd.DataFrame({"Experiments": np.arange(1, len(best) + 1), "Best yield (%)": best,
                      "Method": method_name.split(": ")[-1]}),
        pd.DataFrame({"Experiments": np.arange(1, len(rand_lab.best_so_far()) + 1),
                      "Best yield (%)": rand_lab.best_so_far(), "Method": "Random search (comparison)"}),
    ])
    lines = alt.Chart(chart_df).mark_line(interpolate="step-after", strokeWidth=2.5).encode(
        x=alt.X("Experiments:Q"),
        y=alt.Y("Best yield (%):Q", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("Method:N", scale=alt.Scale(range=["#0072B2", "#999999"]),
                        legend=alt.Legend(orient="bottom", title=None, labelLimit=400)),
    )
    rule = alt.Chart(pd.DataFrame({"Best yield (%)": [threshold]})).mark_rule(
        strokeDash=[4, 4], color="#D55E00").encode(y="Best yield (%):Q")
    st.altair_chart(lines + rule, width="stretch")
    st.caption(f"Dashed line: top-1% threshold ({threshold:.1f}% yield). "
               "Random search is advanced in lock-step for comparison.")

# ---------------------------------------------------------------------------
# What the model believes
# ---------------------------------------------------------------------------
if method_name != "Random search" and lab.n_run >= 3:
    with st.expander("What does the model believe right now?"):
        strat = METHODS[method_name]()
        _, mu, sd = strat.fit_predict(X, idx, y, seed=int(seed))
        untested = lab.untested
        top = untested[np.argsort(-mu[untested])[:10]]
        table = data().iloc[top][COMPONENTS].rename(columns=NICE).copy()
        table["Predicted yield (%)"] = mu[top].round(1)
        table["Uncertainty (±1σ)"] = sd[top].round(1)
        st.markdown("Ten untested reactions the model currently rates highest:")
        st.dataframe(table, hide_index=True, width="stretch")

with st.expander("Experiment log"):
    log = data().iloc[idx][COMPONENTS].rename(columns=NICE).copy()
    log.insert(0, "#", np.arange(1, len(idx) + 1))
    log["Yield (%)"] = y.round(1)
    log["Source"] = ["initial (random)" if k < n_init else "optimizer" for k in range(len(idx))]
    st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")
