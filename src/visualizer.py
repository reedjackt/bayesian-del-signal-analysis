from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_outdir(outdir: Optional[str | Path]) -> Optional[Path]:
    if outdir is None:
        return None
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_ranked_enrichment(
    df: pd.DataFrame,
    score_col: str = "log2_enrichment_mean",
    hue_col: Optional[str] = "is_hit",
    outpath: Optional[str | Path] = None,
    title: str = "Ranked log2 enrichment",
) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))

    plot_df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    plot_df["rank"] = plot_df.index + 1

    if hue_col is not None and hue_col in plot_df.columns:
        sns.scatterplot(
            data=plot_df,
            x="rank",
            y=score_col,
            hue=hue_col,
            palette="Set1",
            alpha=0.8,
            s=18,
            ax=ax,
        )
        ax.legend(title=hue_col, loc="best")
    else:
        ax.scatter(plot_df["rank"], plot_df[score_col], s=18, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Rank (1 = highest enrichment)")
    ax.set_ylabel(score_col)
    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=200)
    return fig


def plot_enrichment_scatter(
    df: pd.DataFrame,
    x_col: str = "input_count",
    y_col: str = "selected_count",
    hue_col: Optional[str] = "is_hit",
    outpath: Optional[str | Path] = None,
    title: str = "Counts scatter (input vs selected)",
) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))

    plot_df = df.copy()
    if hue_col is not None and hue_col in plot_df.columns:
        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue=hue_col,
            palette="Set1",
            alpha=0.7,
            s=25,
            ax=ax,
        )
        ax.legend(title=hue_col, loc="best")
    else:
        ax.scatter(plot_df[x_col], plot_df[y_col], s=25, alpha=0.7)

    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=200)
    return fig


def plot_volcano(
    df: pd.DataFrame,
    x_col: str = "log2_enrichment_mean",
    p_col: str = "prob_enriched",
    hue_col: Optional[str] = "is_hit",
    outpath: Optional[str | Path] = None,
    title: str = "Volcano-like: log2 enrichment vs confidence",
) -> plt.Figure:
    """
    Not a classical p-value volcano; uses -log10(1 - P(enriched)).
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))

    plot_df = df.copy()
    one_minus = np.clip(1.0 - plot_df[p_col].to_numpy(dtype=float), 1e-12, 1.0)
    plot_df["neglog10_uncertainty"] = -np.log10(one_minus)

    if hue_col is not None and hue_col in plot_df.columns:
        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y="neglog10_uncertainty",
            hue=hue_col,
            palette="Set1",
            alpha=0.7,
            s=25,
            ax=ax,
        )
        ax.legend(title=hue_col, loc="best")
    else:
        ax.scatter(plot_df[x_col], plot_df["neglog10_uncertainty"], s=25, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(r"$-\log_{10}(1 - P(\mathrm{enriched}))$")
    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=200)
    return fig

