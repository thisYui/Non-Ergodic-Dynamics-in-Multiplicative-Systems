"""
metrics.py
==========

Statistical analysis tools for multiplicative wealth simulations.

This module provides quantitative evidence for **non-ergodicity** in
multiplicative stochastic systems by separating:

- Ensemble statistics (mean across parallel worlds)
- Time-average growth (experienced by a single trajectory)
- Survival / extinction probabilities

The key theoretical statement demonstrated numerically:

    E[W_T] can grow
    while
    median(W_T) declines

→ Mean does NOT represent typical experience.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


# ---------------------------------------------------------------------
# Distribution statistics over time
# ---------------------------------------------------------------------
def wealth_distribution_stats(wealth: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute ensemble statistics of wealth at each time step.

    Parameters
    ----------
    wealth : ndarray, shape (n_paths, n_steps + 1)

    Returns
    -------
    dict containing:
        mean    : ensemble mean wealth
        median  : typical trajectory wealth
        q05     : 5th percentile
        q95     : 95th percentile

    Interpretation
    --------------
    Non-ergodicity appears when:

        mean ↑ over time
        median ↓ or stagnates
    """
    if wealth.ndim != 2:
        raise ValueError("`wealth` must be 2D (n_paths, n_steps+1).")

    return {
        "mean": np.mean(wealth, axis=0),
        "median": np.median(wealth, axis=0),
        "q05": np.quantile(wealth, 0.05, axis=0),
        "q95": np.quantile(wealth, 0.95, axis=0),
    }


# ---------------------------------------------------------------------
# Time-average vs ensemble growth
# ---------------------------------------------------------------------
def time_average_log_growth(returns: np.ndarray) -> float:
    """
    Estimate time-average log growth rate.

    g_time = E[ log R ]

    Governs long-run survival of multiplicative processes.
    """
    if np.any(returns <= 0):
        raise ValueError("Log-growth undefined for non-positive returns.")

    return float(np.mean(np.log(returns)))


def ensemble_arithmetic_growth(returns: np.ndarray) -> float:
    """
    Ensemble expected arithmetic return.

        g_ens = E[R] − 1

    This quantity can be positive even when
    time-average log growth is negative.
    """
    return float(np.mean(returns) - 1.0)


def compare_growth_measures(returns: np.ndarray) -> Dict[str, float]:
    """
    Convenience function returning both growth notions.

    Useful for directly demonstrating:

        ensemble growth > 0
        BUT
        log growth < 0
    """
    return {
        "time_avg_log_growth": time_average_log_growth(returns),
        "ensemble_arith_growth": ensemble_arithmetic_growth(returns),
    }


# ---------------------------------------------------------------------
# Survival / extinction analysis
# ---------------------------------------------------------------------
def survival_probability(
    wealth: np.ndarray,
    threshold: float = 1e-3,
) -> np.ndarray:
    """
    Probability that wealth stays above a threshold over time.

    Parameters
    ----------
    wealth : ndarray, shape (n_paths, n_steps+1)
    threshold : float
        Extinction boundary.

    Returns
    -------
    survival_prob : ndarray, shape (n_steps+1,)
        Fraction of surviving paths at each time.

    Interpretation
    --------------
    In non-ergodic multiplicative systems:

        survival_prob → 0
        even when mean wealth → ∞
    """
    return np.mean(wealth > threshold, axis=0)


def terminal_extinction_rate(
    wealth: np.ndarray,
    threshold: float = 1e-3,
) -> float:
    """
    Fraction of paths extinct at final time.
    """
    return float(np.mean(wealth[:, -1] <= threshold))


# ---------------------------------------------------------------------
# Log-wealth distribution diagnostics
# ---------------------------------------------------------------------
def terminal_log_wealth_stats(wealth: np.ndarray) -> Tuple[float, float]:
    """
    Mean and variance of log terminal wealth.

    Important because multiplicative systems are
    approximately log-normal under GBM assumptions.
    """
    terminal = wealth[:, -1]

    if np.any(terminal <= 0):
        raise ValueError("Log undefined for non-positive terminal wealth.")

    log_w = np.log(terminal)
    return float(np.mean(log_w)), float(np.var(log_w))


# ---------------------------------------------------------------------
# Inequality / concentration measures
# ---------------------------------------------------------------------
def gini_coefficient(x: np.ndarray) -> float:
    """
    Compute Gini coefficient of terminal wealth.

    Measures inequality caused by multiplicative compounding
    and rare extreme winners.
    """
    if np.any(x < 0):
        raise ValueError("Gini undefined for negative values.")

    x_sorted = np.sort(x)
    n = x.size
    cumulative = np.cumsum(x_sorted)

    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return float(gini)


def terminal_inequality(wealth: np.ndarray) -> Dict[str, float]:
    """
    Summary inequality metrics at final time.
    """
    terminal = wealth[:, -1]

    return {
        "gini": gini_coefficient(terminal),
        "median": float(np.median(terminal)),
        "mean": float(np.mean(terminal)),
        "p95_over_median": float(np.quantile(terminal, 0.95) / np.median(terminal)),
    }
