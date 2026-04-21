from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    n_compounds: int = 5000
    n_hits: int = 50
    library_size_input: int = 2_000_000
    library_size_selected: int = 200_000
    baseline_capture: float = 0.001
    hit_capture: float = 0.02
    pcr_dispersion: float = 50.0
    seed: Optional[int] = 7


def _dirichlet_library_frequencies(rng: np.random.Generator, n: int, alpha: float = 1.0) -> np.ndarray:
    conc = np.full(n, alpha, dtype=float)
    return rng.dirichlet(conc)


def _negative_binomial_overdispersed_counts(
    rng: np.random.Generator, expected: np.ndarray, dispersion: float
) -> np.ndarray:
    """
    Convert expected counts into overdispersed counts.

    Uses a Gamma-Poisson (NB) mixture:
      lambda ~ Gamma(k=dispersion, theta=mu/k)
      y ~ Poisson(lambda)
    """
    mu = np.asarray(expected, dtype=float)
    k = float(dispersion)
    if k <= 0:
        return rng.poisson(mu)
    lam = rng.gamma(shape=k, scale=np.maximum(mu, 0.0) / k)
    return rng.poisson(lam)


def simulate_del_experiment(config: SimulationConfig = SimulationConfig()) -> pd.DataFrame:
    """
    Simulate a single-round DEL experiment producing input and selected counts.

    Returns a DataFrame with:
      - compound_id
      - is_hit
      - true_capture_prob
      - input_count
      - selected_count
    """
    rng = np.random.default_rng(config.seed)
    n = int(config.n_compounds)
    n_hits = int(min(config.n_hits, n))

    compound_id = np.arange(n, dtype=int)
    is_hit = np.zeros(n, dtype=bool)
    if n_hits > 0:
        is_hit[rng.choice(n, size=n_hits, replace=False)] = True

    true_capture = np.full(n, float(config.baseline_capture), dtype=float)
    true_capture[is_hit] = float(config.hit_capture)
    true_capture = np.clip(true_capture, 1e-9, 1.0 - 1e-9)

    freqs = _dirichlet_library_frequencies(rng, n, alpha=1.0)
    expected_input = freqs * float(config.library_size_input)
    input_counts = _negative_binomial_overdispersed_counts(rng, expected_input, config.pcr_dispersion)

    expected_selected = input_counts.astype(float) * true_capture
    selected_counts = _negative_binomial_overdispersed_counts(rng, expected_selected, config.pcr_dispersion)

    if selected_counts.sum() > 0:
        selected_counts = rng.multinomial(
            n=int(config.library_size_selected),
            pvals=selected_counts / selected_counts.sum(),
        )
    else:
        selected_counts = np.zeros(n, dtype=int)

    return pd.DataFrame(
        {
            "compound_id": compound_id,
            "is_hit": is_hit,
            "true_capture_prob": true_capture,
            "input_count": input_counts.astype(int),
            "selected_count": selected_counts.astype(int),
        }
    )


def split_train_test(
    df: pd.DataFrame, test_frac: float = 0.2, seed: Optional[int] = 7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(round(len(df) * (1.0 - float(test_frac))))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

