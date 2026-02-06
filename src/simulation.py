"""
simulation.py
==============

Core simulation engine for multiplicative stochastic wealth dynamics.

This module is intentionally **pure and dependency-light** so it can be reused
in notebooks, experiments, and unit tests without modification.

Design principles
-----------------
1. Fully vectorized Monte Carlo (no Python loops over paths).
2. Deterministic reproducibility via RNG seed.
3. Separation between:
   - return generation
   - wealth evolution
4. Numerical stability for long horizons.

Mathematical model
------------------
Multiplicative wealth process:

    W_{t+1} = W_t * R_t

Taking logs:

    log W_T = log W_0 + Σ log R_t

→ Long-run behavior governed by **expected log-return**, not mean return.
→ This is the core mechanism behind **non-ergodicity**.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional


# ---------------------------------------------------------------------
# Return generation
# ---------------------------------------------------------------------
def generate_returns(
    n_steps: int,
    n_paths: int,
    mu: float,
    sigma: float,
    dt: float = 1.0,
    seed: Optional[int] = None,
    lognormal: bool = True,
) -> np.ndarray:
    """
    Generate stochastic return paths.

    Parameters
    ----------
    n_steps : int
        Number of time steps in each path.
    n_paths : int
        Number of Monte Carlo trajectories.
    mu : float
        Drift (per unit time).
    sigma : float
        Volatility (per sqrt unit time).
    dt : float, default=1.0
        Time increment.
    seed : int or None
        RNG seed for reproducibility.
    lognormal : bool, default=True
        If True:
            R_t = exp((mu - 0.5*sigma^2) dt + sigma sqrt(dt) Z)
        Else:
            Simple arithmetic:
            R_t = 1 + mu*dt + sigma*sqrt(dt)*Z

    Returns
    -------
    returns : ndarray, shape (n_paths, n_steps)
        Multiplicative returns for each trajectory.

    Notes
    -----
    - Lognormal model ensures strictly positive returns.
    - Arithmetic model can produce negative returns (use with care).
    """
    rng = np.random.default_rng(seed)

    # Standard normal shocks
    z = rng.standard_normal(size=(n_paths, n_steps))

    if lognormal:
        # Geometric Brownian Motion discretization
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * z
        returns = np.exp(drift + diffusion)
    else:
        # Simple linear return model
        returns = 1.0 + mu * dt + sigma * np.sqrt(dt) * z

    return returns


# ---------------------------------------------------------------------
# Wealth evolution
# ---------------------------------------------------------------------
def simulate_wealth(
    returns: np.ndarray,
    w0: float = 1.0,
) -> np.ndarray:
    """
    Convert return paths into multiplicative wealth trajectories.

    Parameters
    ----------
    returns : ndarray, shape (n_paths, n_steps)
        Multiplicative returns.
    w0 : float, default=1.0
        Initial wealth.

    Returns
    -------
    wealth : ndarray, shape (n_paths, n_steps + 1)
        Wealth paths including initial value at t=0.

    Notes
    -----
    Uses cumulative product:

        W_t = W_0 * Π_{i=1..t} R_i

    This multiplicative compounding is the source of:

    - path divergence
    - skewed distributions
    - non-ergodic behavior
    """
    if returns.ndim != 2:
        raise ValueError("`returns` must be 2-dimensional (n_paths, n_steps).")

    # Cumulative product along time axis
    cumulative = np.cumprod(returns, axis=1)

    # Insert initial wealth at t = 0
    initial = np.full((returns.shape[0], 1), w0)

    wealth = np.concatenate([initial, w0 * cumulative], axis=1)
    return wealth


# ---------------------------------------------------------------------
# Volatility regimes (for experiments)
# ---------------------------------------------------------------------
def volatility_regimes() -> Dict[str, float]:
    """
    Standard volatility regimes used in non-ergodic experiments.

    Returns
    -------
    dict
        Mapping regime name → sigma value.

    Interpretation
    --------------
    Increasing volatility:

    → increases dispersion between paths
    → lowers expected log-growth
    → amplifies non-ergodicity
    """
    return {
        "low": 0.05,
        "medium": 0.20,
        "high": 0.60,
    }


# ---------------------------------------------------------------------
# Convenience wrapper for full simulation
# ---------------------------------------------------------------------
def run_monte_carlo(
    n_steps: int,
    n_paths: int,
    mu: float,
    sigma: float,
    w0: float = 1.0,
    dt: float = 1.0,
    seed: Optional[int] = None,
    lognormal: bool = True,
) -> np.ndarray:
    """
    High-level helper to run full wealth simulation in one call.

    Returns
    -------
    wealth : ndarray, shape (n_paths, n_steps + 1)

    Purpose
    -------
    Keeps notebooks clean and declarative:

        wealth = run_monte_carlo(...)
    """
    returns = generate_returns(
        n_steps=n_steps,
        n_paths=n_paths,
        mu=mu,
        sigma=sigma,
        dt=dt,
        seed=seed,
        lognormal=lognormal,
    )

    wealth = simulate_wealth(returns, w0=w0)
    return wealth
