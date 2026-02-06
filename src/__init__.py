"""
non_ergodic_dynamics.src
========================

Public API for the non-ergodic multiplicative dynamics research package.

Purpose
-------
Provide a **clean, documented import surface** so that:

- notebooks
- analysis scripts
- autonomous agents

can use the library **without reading internal source files**.

Module structure
----------------

1) simulation
   Core stochastic dynamics.

   - generate_returns(...)
       → Monte Carlo return generator (lognormal or arithmetic).

   - simulate_wealth(...)
       → Convert returns → multiplicative wealth trajectories.

   - volatility_regimes()
       → Standard σ regimes for experiments.

   - run_monte_carlo(...)
       → One-call full simulation helper.


2) metrics
   Statistical proof of non-ergodicity.

   - wealth_distribution_stats(...)
       → Mean / median / quantiles over time.

   - time_average_log_growth(...)
       → Long-run growth governing survival.

   - ensemble_arithmetic_growth(...)
       → Classical expectation (can mislead).

   - compare_growth_measures(...)
       → Direct ergodicity contrast.

   - survival_probability(...)
       → Probability wealth stays above extinction level.

   - terminal_extinction_rate(...)
       → Final ruin fraction.

   - terminal_log_wealth_stats(...)
       → Mean/variance of log terminal wealth.

   - gini_coefficient(...), terminal_inequality(...)
       → Inequality from multiplicative compounding.


3) crd_tree
   Deterministic state-space decomposition (CRR binomial tree).

   - crr_params(...)
       → Up/down factors and transition probability.

   - build_crr_tree(...)
       → Recombining wealth/value tree.

   - node_probabilities(...)
       → Binomial probability mass per node.

   - level_expectations(...)
       → Deterministic expectation trajectory.

   - expectation_contributions(...)
       → Node-level contribution to expectation.

   - terminal_tail_dominance(...)
       → Rare-event dominance metric.

   - run_crr_state_space(...)
       → Full CRR analysis in one call.


4) visualization
   Publication-quality scientific plots.

   - plot_fan_chart(...)
       → Monte Carlo path divergence.

   - plot_mean_median(...)
       → Ensemble vs typical trajectory.

   - plot_terminal_log_distribution(...)
       → Log-wealth histogram.

   - plot_survival_curve(...)
       → Survival probability over time.

   - plot_crr_contribution_heatmap(...)
       → Rare-event expectation structure.

   - plot_crr_expectation(...)
       → Deterministic CRR expectation path.
"""

# ---------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------
from .simulation import (
    generate_returns,
    simulate_wealth,
    volatility_regimes,
    run_monte_carlo,
)

# ---------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------
from .metrics import (
    wealth_distribution_stats,
    time_average_log_growth,
    ensemble_arithmetic_growth,
    compare_growth_measures,
    survival_probability,
    terminal_extinction_rate,
    terminal_log_wealth_stats,
    gini_coefficient,
    terminal_inequality,
)

# ---------------------------------------------------------------------
# crd_tree
# ---------------------------------------------------------------------
from .crd_tree import (
    crr_params,
    build_crr_tree,
    node_probabilities,
    level_expectations,
    expectation_contributions,
    terminal_tail_dominance,
    run_crr_state_space,
)

# ---------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------
from .visualization import (
    plot_fan_chart,
    plot_mean_median,
    plot_terminal_log_distribution,
    plot_survival_curve,
    plot_crr_contribution_heatmap,
    plot_crr_expectation,
)

# ---------------------------------------------------------------------
# Explicit public API
# ---------------------------------------------------------------------
__all__ = [
    # simulation
    "generate_returns",
    "simulate_wealth",
    "volatility_regimes",
    "run_monte_carlo",
    # metrics
    "wealth_distribution_stats",
    "time_average_log_growth",
    "ensemble_arithmetic_growth",
    "compare_growth_measures",
    "survival_probability",
    "terminal_extinction_rate",
    "terminal_log_wealth_stats",
    "gini_coefficient",
    "terminal_inequality",
    # crd_tree
    "crr_params",
    "build_crr_tree",
    "node_probabilities",
    "level_expectations",
    "expectation_contributions",
    "terminal_tail_dominance",
    "run_crr_state_space",
    # visualization
    "plot_fan_chart",
    "plot_mean_median",
    "plot_terminal_log_distribution",
    "plot_survival_curve",
    "plot_crr_contribution_heatmap",
    "plot_crr_expectation",
]
