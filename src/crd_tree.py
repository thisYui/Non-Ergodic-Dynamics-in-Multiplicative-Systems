"""
crd_tree.py
===========

State-space representation of multiplicative dynamics using
a recombining Cox–Ross–Rubinstein (CRR) binomial tree.

Purpose
-------
This module is NOT primarily for option pricing.

Instead, it is used to:

1. Enumerate all possible multiplicative wealth states.
2. Separate:
      - probability mass
      - payoff magnitude
3. Show that expectation is dominated by
   **rare, extreme states** → core mechanism of non-ergodicity.

Mathematical structure
----------------------
At each step:

    S_{t+1} = S_t * u   with probability p
    S_{t+1} = S_t * d   with probability 1 - p

After n steps, node k has:

    value  = S0 * u^k * d^(n-k)
    prob   = C(n, k) p^k (1-p)^(n-k)

Expectation:

    E[S_n] = Σ prob_k * value_k
"""

from __future__ import annotations

import numpy as np
from math import comb
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------
# CRR parameters
# ---------------------------------------------------------------------
def crr_params(mu: float, sigma: float, dt: float) -> Tuple[float, float, float]:
    """
    Compute CRR up/down factors and risk-neutral-style probability.

    Parameters
    ----------
    mu : drift
    sigma : volatility
    dt : timestep

    Returns
    -------
    u, d, p
    """
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(mu * dt) - d) / (u - d)

    if not (0.0 <= p <= 1.0):
        raise ValueError("Invalid CRR probability. Check (mu, sigma, dt).")

    return float(u), float(d), float(p)


# ---------------------------------------------------------------------
# Build recombining value tree
# ---------------------------------------------------------------------
def build_crr_tree(S0: float, u: float, d: float, n_steps: int) -> List[np.ndarray]:
    """
    Construct recombining binomial value tree.

    Returns
    -------
    levels : list of arrays
        levels[t] contains all node values at time t.
        length(levels[t]) = t + 1
    """
    levels: List[np.ndarray] = []

    for t in range(n_steps + 1):
        k = np.arange(t + 1)  # number of up moves
        values = S0 * (u ** k) * (d ** (t - k))
        levels.append(values)

    return levels


# ---------------------------------------------------------------------
# Node probabilities
# ---------------------------------------------------------------------
def node_probabilities(p: float, n_steps: int) -> List[np.ndarray]:
    """
    Compute binomial probability mass at each tree level.

    Returns
    -------
    probs : list of arrays
        probs[t][k] = probability of k up moves at time t.
    """
    probs: List[np.ndarray] = []

    for t in range(n_steps + 1):
        k = np.arange(t + 1)
        level_probs = np.array(
            [comb(t, int(i)) * (p ** i) * ((1 - p) ** (t - i)) for i in k],
            dtype=float,
        )
        probs.append(level_probs)

    return probs


# ---------------------------------------------------------------------
# Expectation per level
# ---------------------------------------------------------------------
def level_expectations(
    values: List[np.ndarray],
    probs: List[np.ndarray],
) -> np.ndarray:
    """
    Compute expectation at each timestep.

    Returns
    -------
    ndarray shape (n_steps+1,)
    """
    if len(values) != len(probs):
        raise ValueError("values and probs must have same length.")

    return np.array(
        [np.sum(v * p) for v, p in zip(values, probs)],
        dtype=float,
    )


# ---------------------------------------------------------------------
# Contribution decomposition
# ---------------------------------------------------------------------
def expectation_contributions(
    values: List[np.ndarray],
    probs: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Contribution of each node to expectation.

        contrib_k = prob_k * value_k

    This is the **key observable** for:

        rare-event dominance of the mean.
    """
    if len(values) != len(probs):
        raise ValueError("values and probs must have same length.")

    return [v * p for v, p in zip(values, probs)]


# ---------------------------------------------------------------------
# Rare-event dominance metrics
# ---------------------------------------------------------------------
def tail_contribution_ratio(
    contrib: np.ndarray,
    quantile: float = 0.95,
) -> float:
    """
    Fraction of expectation coming from top-value tail nodes.

    Steps
    -----
    1. Sort nodes by value contribution.
    2. Keep nodes above quantile.
    3. Measure share of total expectation.

    Interpretation
    --------------
    If ratio → 1:

        mean dominated by rare extreme states.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0,1).")

    sorted_c = np.sort(contrib)
    cutoff = int(np.floor(quantile * len(sorted_c)))

    tail_sum = np.sum(sorted_c[cutoff:])
    total_sum = np.sum(sorted_c)

    return float(tail_sum / total_sum)


def terminal_tail_dominance(
    values: List[np.ndarray],
    probs: List[np.ndarray],
    quantile: float = 0.95,
) -> float:
    """
    Tail dominance measured at final timestep.
    """
    terminal_contrib = expectation_contributions(values, probs)[-1]
    return tail_contribution_ratio(terminal_contrib, quantile)


# ---------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------
def run_crr_state_space(
    S0: float,
    mu: float,
    sigma: float,
    dt: float,
    n_steps: int,
) -> Dict[str, object]:
    """
    High-level helper returning full CRR decomposition.

    Returns
    -------
    dict with:
        u, d, p
        values
        probs
        expectations
        tail_dominance
    """
    u, d, p = crr_params(mu, sigma, dt)

    values = build_crr_tree(S0, u, d, n_steps)
    probs = node_probabilities(p, n_steps)
    expectations = level_expectations(values, probs)
    tail_dom = terminal_tail_dominance(values, probs)

    return {
        "u": u,
        "d": d,
        "p": p,
        "values": values,
        "probs": probs,
        "expectations": expectations,
        "tail_dominance": tail_dom,
    }
