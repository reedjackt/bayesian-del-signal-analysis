"""
DEL Bayesian Enrichment — `src` package.

Re-exports the main public API from submodules so callers can use
``import src`` or ``from src import summarize_enrichment`` when the
project root (the directory containing ``src/``) is on ``sys.path``.
"""

from .simulator import simulate_del_experiment
from .analyzer import (
    aggregate_enrichment_by_scaffold,
    beta_binomial_posterior,
    beta_mom_from_mean_var,
    BetaBinomialConfig,
    enrichment_posterior_mc,
    estimate_empirical_beta_prior,
    log2_enrichment_mean_digamma,
    merge_scaffold_enrichment,
    summarize_enrichment,
)
from .visualizer import (
    plot_ranked_enrichment,
    plot_enrichment_scatter,
    plot_volcano,
)
from .importer import KinDELImportConfig, LibraryScaler, import_kindel_counts, load_kindel_dataset

__all__ = [
    "simulate_del_experiment",
    "KinDELImportConfig",
    "LibraryScaler",
    "import_kindel_counts",
    "load_kindel_dataset",
    "BetaBinomialConfig",
    "beta_binomial_posterior",
    "beta_mom_from_mean_var",
    "estimate_empirical_beta_prior",
    "log2_enrichment_mean_digamma",
    "enrichment_posterior_mc",
    "summarize_enrichment",
    "aggregate_enrichment_by_scaffold",
    "merge_scaffold_enrichment",
    "plot_ranked_enrichment",
    "plot_enrichment_scatter",
    "plot_volcano",
]
