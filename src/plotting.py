"""Shared matplotlib styling so every figure looks like one set."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

# Colour-blind-safe categorical palette (Okabe–Ito order, black dropped).
PALETTE = ["#7f7f7f", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def style_axes(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)
