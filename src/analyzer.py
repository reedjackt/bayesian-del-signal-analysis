from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import beta as beta_dist


@dataclass(frozen=True)
class BetaBinomialConfig:
    """Beta-Binomial enrichment inference configuration."""

    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    use_empirical_prior: bool = True
    prior_pseudocount: float = 0.5
    mad_scale_floor: float = 1e-6
    mc_samples: int = 50_000
    mc_batch_size: int = 2048
    uncertainty_mode: Literal["mc_batched", "none"] = "mc_batched"
    seed: Optional[int] = 7
    credible_interval: Tuple[float, float] = (0.025, 0.975)


def beta_mom_from_mean_var(mean: float, var: float) -> tuple[float, float]:
    """
    Method-of-moments fit of Beta(alpha, beta) from mean and variance.

    Falls back to Beta(1, 1) when variance is inconsistent with any Beta
    (e.g. var >= mean*(1-mean) or non-positive variance).
    """
    m = float(np.clip(mean, 1e-9, 1.0 - 1e-9))
    v = float(var)
    denom = m * (1.0 - m)
    if v <= 0.0 or v >= denom:
        return 1.0, 1.0
    phi = denom / v - 1.0
    if phi <= 0.0:
        return 1.0, 1.0
    alpha = m * phi
    beta = (1.0 - m) * phi
    return max(alpha, 1e-6), max(beta, 1e-6)


def smoothed_log_fold(
    x_in: np.ndarray,
    x_sel: np.ndarray,
    total_input: float,
    total_selected: float,
    pseudocount: float,
) -> np.ndarray:
    """Per-compound log-ratio of smoothed library fractions (natural log)."""
    a = float(pseudocount)
    n_in = float(total_input)
    n_sel = float(total_selected)
    xi = np.asarray(x_in, dtype=float)
    xs = np.asarray(x_sel, dtype=float)
    log_frac_sel = np.log(xs + a) - np.log(n_sel + 2.0 * a)
    log_frac_in = np.log(xi + a) - np.log(n_in + 2.0 * a)
    return log_frac_sel - log_frac_in


def _median_absolute_deviation(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def estimate_empirical_beta_prior(
    x_in: np.ndarray,
    x_sel: np.ndarray,
    total_input: float,
    total_selected: float,
    pseudocount: float = 0.5,
    mad_scale_floor: float = 1e-6,
) -> tuple[float, float]:
    """
    Empirical Bayes Beta(α,β) prior from library-wide enrichment proxy.

    Computes smoothed log fold per compound, maps to (0,1) via a logistic
    squashing u = expit(log_fold / τ) with τ = MAD(log_fold) (floored), then
    applies Beta MoM to the distribution of u. The same (α,β) is used for
    both input and selected channels, matching a single conjugate prior family.
    """
    log_r = smoothed_log_fold(x_in, x_sel, total_input, total_selected, pseudocount)
    tau = max(mad_scale_floor, _median_absolute_deviation(log_r))
    u = special.expit(log_r / tau)
    m = float(np.mean(u))
    v = float(np.var(u))
    return beta_mom_from_mean_var(m, v)


def beta_binomial_posterior(
    successes: np.ndarray | int,
    trials: np.ndarray | int,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Posterior for Binomial rate p with Beta(alpha_prior, beta_prior).

    Posterior: p | data ~ Beta(alpha_prior + successes, beta_prior + trials - successes)
    """
    s = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    if np.any(s < 0) or np.any(n < 0) or np.any(s > n):
        raise ValueError("Invalid successes/trials: require 0 <= successes <= trials.")
    return alpha_prior + s, beta_prior + (n - s)


def _posterior_beta_params(
    x_in: np.ndarray,
    x_sel: np.ndarray,
    total_input: float,
    total_selected: float,
    alpha_prior: float,
    beta_prior: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_in = float(total_input)
    n_sel = float(total_selected)
    a_in, b_in = beta_binomial_posterior(
        x_in, np.full_like(x_in, n_in, dtype=float), alpha_prior, beta_prior
    )
    a_sel, b_sel = beta_binomial_posterior(
        x_sel, np.full_like(x_sel, n_sel, dtype=float), alpha_prior, beta_prior
    )
    return a_in, b_in, a_sel, b_sel


def log2_enrichment_mean_digamma(
    a_in: np.ndarray,
    b_in: np.ndarray,
    a_sel: np.ndarray,
    b_sel: np.ndarray,
) -> np.ndarray:
    """
    E[log2(p_sel / p_in)] for independent Beta posteriors on p_in and p_sel.

    Uses E[log p] = psi(a) - psi(a+b) for Beta(a,b).
    """
    e_log_sel = special.digamma(a_sel) - special.digamma(a_sel + b_sel)
    e_log_in = special.digamma(a_in) - special.digamma(a_in + b_in)
    return (e_log_sel - e_log_in) / np.log(2.0)


def _batched_log2_enrichment_mc(
    a_in: np.ndarray,
    b_in: np.ndarray,
    a_sel: np.ndarray,
    b_sel: np.ndarray,
    mc_samples: int,
    batch_size: int,
    credible_interval: tuple[float, float],
    seed: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte Carlo quantiles and P(log2 > 0) without allocating (mc_samples, n)."""
    rng = np.random.default_rng(seed)
    n = len(a_in)
    lo = np.empty(n, dtype=float)
    hi = np.empty(n, dtype=float)
    prob = np.empty(n, dtype=float)
    lo_q, hi_q = credible_interval
    ln2 = np.log(2.0)
    eps = np.finfo(np.float64).tiny

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        ain = a_in[start:end]
        bin_ = b_in[start:end]
        asel = a_sel[start:end]
        bsel = b_sel[start:end]
        bsz = end - start
        p_in = beta_dist.rvs(ain, bin_, size=(mc_samples, bsz), random_state=rng)
        p_sel = beta_dist.rvs(asel, bsel, size=(mc_samples, bsz), random_state=rng)
        log2_enr = (np.log(np.maximum(p_sel, eps)) - np.log(np.maximum(p_in, eps))) / ln2
        lo[start:end] = np.quantile(log2_enr, lo_q, axis=0)
        hi[start:end] = np.quantile(log2_enr, hi_q, axis=0)
        prob[start:end] = np.mean(log2_enr > 0.0, axis=0)

    return lo, hi, prob


def enrichment_posterior_mc(
    input_count: np.ndarray,
    selected_count: np.ndarray,
    total_input: int,
    total_selected: int,
    config: BetaBinomialConfig = BetaBinomialConfig(),
    *,
    return_full_samples: bool = False,
    max_full_samples_n: int = 2000,
) -> dict[str, np.ndarray]:
    """
    Legacy / optional Monte Carlo over enrichment.

    By default does **not** build a (mc_samples, n) matrix (OOM-safe). Pass
    ``return_full_samples=True`` only for small ``n`` <= ``max_full_samples_n``.
    """
    x_in = np.asarray(input_count, dtype=float)
    x_sel = np.asarray(selected_count, dtype=float)
    n = len(x_in)
    alpha0, beta0 = config.alpha_prior, config.beta_prior
    if config.use_empirical_prior:
        alpha0, beta0 = estimate_empirical_beta_prior(
            x_in,
            x_sel,
            float(total_input),
            float(total_selected),
            pseudocount=config.prior_pseudocount,
            mad_scale_floor=config.mad_scale_floor,
        )
    a_in, b_in, a_sel, b_sel = _posterior_beta_params(
        x_in, x_sel, total_input, total_selected, alpha0, beta0
    )

    if return_full_samples:
        if n > max_full_samples_n:
            raise ValueError(
                f"return_full_samples requires n <= {max_full_samples_n}, got n={n}"
            )
        rng = np.random.default_rng(config.seed)
        s = int(config.mc_samples)
        eps = np.finfo(np.float64).tiny
        p_in = beta_dist.rvs(a_in, b_in, size=(s, n), random_state=rng)
        p_sel = beta_dist.rvs(a_sel, b_sel, size=(s, n), random_state=rng)
        log2_enr = (
            np.log(np.maximum(p_sel, eps)) - np.log(np.maximum(p_in, eps))
        ) / np.log(2.0)
        return {"log2_enrichment_samples": log2_enr}

    lo, hi, pr = _batched_log2_enrichment_mc(
        a_in,
        b_in,
        a_sel,
        b_sel,
        config.mc_samples,
        config.mc_batch_size,
        config.credible_interval,
        config.seed,
    )
    return {
        "log2_enrichment_ci_low": lo,
        "log2_enrichment_ci_high": hi,
        "prob_enriched": pr,
    }


def summarize_enrichment(
    df: pd.DataFrame,
    config: BetaBinomialConfig = BetaBinomialConfig(),
    input_col: str = "input_count",
    selected_col: str = "selected_count",
) -> pd.DataFrame:
    """
    Add Bayesian enrichment summaries to a DEL count table.

    Required columns: input_col, selected_col
    Adds:
      - log2_enrichment_mean (digamma / analytical expectation)
      - log2_enrichment_ci_low / ci_high (batched MC unless uncertainty_mode='none')
      - prob_enriched (batched MC unless uncertainty_mode='none')
    """
    if input_col not in df.columns or selected_col not in df.columns:
        raise ValueError(f"Missing required columns: {input_col}, {selected_col}")

    total_input = int(df[input_col].sum())
    total_selected = int(df[selected_col].sum())
    if total_input <= 0 or total_selected <= 0:
        raise ValueError("Total input/selected counts must be > 0.")

    x_in = df[input_col].to_numpy(dtype=float)
    x_sel = df[selected_col].to_numpy(dtype=float)

    alpha0, beta0 = float(config.alpha_prior), float(config.beta_prior)
    if config.use_empirical_prior:
        alpha0, beta0 = estimate_empirical_beta_prior(
            x_in,
            x_sel,
            float(total_input),
            float(total_selected),
            pseudocount=config.prior_pseudocount,
            mad_scale_floor=config.mad_scale_floor,
        )

    a_in, b_in, a_sel, b_sel = _posterior_beta_params(
        x_in, x_sel, total_input, total_selected, alpha0, beta0
    )
    mean = log2_enrichment_mean_digamma(a_in, b_in, a_sel, b_sel)

    out = df.copy()
    out["log2_enrichment_mean"] = mean

    if config.uncertainty_mode == "none":
        out["log2_enrichment_ci_low"] = np.nan
        out["log2_enrichment_ci_high"] = np.nan
        out["prob_enriched"] = np.nan
    else:
        lo, hi, prob = _batched_log2_enrichment_mc(
            a_in,
            b_in,
            a_sel,
            b_sel,
            config.mc_samples,
            config.mc_batch_size,
            config.credible_interval,
            config.seed,
        )
        out["log2_enrichment_ci_low"] = lo
        out["log2_enrichment_ci_high"] = hi
        out["prob_enriched"] = prob

    return out


def top_hits(
    df: pd.DataFrame,
    k: int = 50,
    score_col: str = "log2_enrichment_mean",
) -> pd.DataFrame:
    return df.sort_values(score_col, ascending=False).head(int(k)).reset_index(drop=True)


def aggregate_enrichment_by_scaffold(
    df: pd.DataFrame,
    scaffold_col: str = "scaffold_id",
    config: BetaBinomialConfig = BetaBinomialConfig(),
    input_col: str = "input_count",
    selected_col: str = "selected_count",
) -> pd.DataFrame:
    """
    Pool counts per scaffold and compute ``scaffold_log2_enrichment`` (digamma mean).

    Uses the same experiment totals and empirical prior estimation as compound-level
    analysis (prior fit on the full compound table, then posterior per scaffold).
    """
    if scaffold_col not in df.columns:
        raise ValueError(f"Missing scaffold column: {scaffold_col}")
    if input_col not in df.columns or selected_col not in df.columns:
        raise ValueError(f"Missing required columns: {input_col}, {selected_col}")

    total_input = int(df[input_col].sum())
    total_selected = int(df[selected_col].sum())
    if total_input <= 0 or total_selected <= 0:
        raise ValueError("Total input/selected counts must be > 0.")

    x_in_full = df[input_col].to_numpy(dtype=float)
    x_sel_full = df[selected_col].to_numpy(dtype=float)

    alpha0, beta0 = float(config.alpha_prior), float(config.beta_prior)
    if config.use_empirical_prior:
        alpha0, beta0 = estimate_empirical_beta_prior(
            x_in_full,
            x_sel_full,
            float(total_input),
            float(total_selected),
            pseudocount=config.prior_pseudocount,
            mad_scale_floor=config.mad_scale_floor,
        )

    agg = (
        df.groupby(scaffold_col, sort=False)[[input_col, selected_col]]
        .sum()
        .reset_index()
    )
    s_in = agg[input_col].to_numpy(dtype=float)
    s_sel = agg[selected_col].to_numpy(dtype=float)

    a_in, b_in, a_sel, b_sel = _posterior_beta_params(
        s_in, s_sel, total_input, total_selected, alpha0, beta0
    )
    agg["scaffold_log2_enrichment"] = log2_enrichment_mean_digamma(
        a_in, b_in, a_sel, b_sel
    )
    return agg


def merge_scaffold_enrichment(
    df: pd.DataFrame,
    scaffold_summary: pd.DataFrame,
    scaffold_col: str = "scaffold_id",
    enrichment_col: str = "scaffold_log2_enrichment",
) -> pd.DataFrame:
    """Left-merge scaffold-level log2 enrichment onto compound rows."""
    if scaffold_col not in df.columns:
        raise ValueError(f"Missing scaffold column on left: {scaffold_col}")
    if scaffold_col not in scaffold_summary.columns:
        raise ValueError(f"Missing scaffold column on summary: {scaffold_col}")
    if enrichment_col not in scaffold_summary.columns:
        raise ValueError(f"Missing enrichment column on summary: {enrichment_col}")
    right = scaffold_summary[[scaffold_col, enrichment_col]].drop_duplicates(
        subset=[scaffold_col]
    )
    return df.merge(right, on=scaffold_col, how="left")
