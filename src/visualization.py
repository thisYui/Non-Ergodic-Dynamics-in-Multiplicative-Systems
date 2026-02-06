"""
visualization.py
================

Scientific visualization utilities for non-ergodic multiplicative dynamics.

Design rules
------------
1. Pure plotting layer (no simulation or statistics).
2. Matplotlib only → reproducible, publication-friendly.
3. Functions accept precomputed data structures.
4. Suitable for:
      - research notebooks
      - academic figures
      - Monte Carlo diagnostics
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


# ---------------------------------------------------------------------
# Fan chart of wealth trajectories
# ---------------------------------------------------------------------
def plot_fan_chart(
    wealth: np.ndarray,
    max_paths: int = 200,
    title: str = "Monte Carlo Wealth Trajectories",
) -> None:
    """
    Plot subset of wealth paths to visualize divergence.

    Parameters
    ----------
    wealth : ndarray (n_paths, n_steps+1)
    max_paths : int
        Number of trajectories shown (avoid overplotting).
    """
    n_paths = wealth.shape[0]
    idx = np.linspace(0, n_paths - 1, min(max_paths, n_paths), dtype=int)

    plt.figure(figsize=(10, 6))
    plt.plot(wealth[idx].T, linewidth=0.8, alpha=0.6)

    plt.xlabel("Time")
    plt.ylabel("Wealth")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# ---------------------------------------------------------------------
# Mean vs median divergence
# ---------------------------------------------------------------------
def plot_mean_median(
    stats: Dict[str, np.ndarray],
    title: str = "Mean vs Median Wealth (Non-Ergodicity)",
) -> None:
    """
    Visualize ensemble mean vs typical trajectory.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(stats["mean"], label="Mean (ensemble)", linewidth=2)
    plt.plot(stats["median"], label="Median (typical)", linewidth=2)

    plt.fill_between(
        np.arange(len(stats["mean"])),
        stats["q05"],
        stats["q95"],
        alpha=0.2,
        label="5–95% quantile band",
    )

    plt.xlabel("Time")
    plt.ylabel("Wealth")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# ---------------------------------------------------------------------
# Terminal log-wealth distribution
# ---------------------------------------------------------------------
def plot_terminal_log_distribution(
    wealth: np.ndarray,
    bins: int = 60,
    title: str = "Terminal Log-Wealth Distribution",
) -> None:
    """
    Histogram of log terminal wealth.

    Reveals approximate log-normality and heavy right tail.
    """
    terminal = wealth[:, -1]

    if np.any(terminal <= 0):
        raise ValueError("Log undefined for non-positive wealth.")

    log_w = np.log(terminal)

    plt.figure(figsize=(9, 5))
    plt.hist(log_w, bins=bins)

    plt.xlabel("log(terminal wealth)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# ---------------------------------------------------------------------
# Survival probability curve
# ---------------------------------------------------------------------
def plot_survival_curve(
    survival: np.ndarray,
    title: str = "Survival Probability Over Time",
) -> None:
    """
    Plot probability of remaining above extinction threshold.
    """
    plt.figure(figsize=(9, 5))
    plt.plot(survival, linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# ---------------------------------------------------------------------
# CRR expectation contribution heatmap
# ---------------------------------------------------------------------
def plot_crr_contribution_heatmap(
    contrib_levels: List[np.ndarray],
    title: str = "CRR Node Contribution to Expectation",
) -> None:
    """
    Visualize expectation contributions across tree levels.

    Steps
    -----
    Convert jagged CRR levels → rectangular matrix with NaNs,
    then display as heatmap.

    Interpretation
    --------------
    Bright region drifting to extreme nodes over time
    → rare-event dominance of expectation.
    """
    n_steps = len(contrib_levels)

    # Build rectangular matrix
    mat = np.full((n_steps, n_steps), np.nan)

    for t, level in enumerate(contrib_levels):
        mat[t, : len(level)] = level

    plt.figure(figsize=(10, 6))
    plt.imshow(mat, aspect="auto", origin="lower")

    plt.colorbar(label="Contribution to expectation")
    plt.xlabel("Node index (number of up moves)")
    plt.ylabel("Time step")
    plt.title(title)
    plt.tight_layout()


# ---------------------------------------------------------------------
# Expectation trajectory from CRR tree
# ---------------------------------------------------------------------
def plot_crr_expectation(
    expectations: np.ndarray,
    title: str = "CRR Expected Value Over Time",
) -> None:
    """
    Plot deterministic expectation from binomial state space.
    """
    plt.figure(figsize=(9, 5))
    plt.plot(expectations, linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Expected value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
