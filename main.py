from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulator import SimulationConfig, simulate_del_experiment
from src.analyzer import (
    BetaBinomialConfig,
    aggregate_enrichment_by_scaffold,
    merge_scaffold_enrichment,
    summarize_enrichment,
    top_hits,
)
from src.visualizer import plot_enrichment_scatter, plot_ranked_enrichment, plot_volcano


def run_demo(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    sim_cfg = SimulationConfig()
    df = simulate_del_experiment(sim_cfg)
    df["scaffold_id"] = (df["compound_id"] // 500).astype(np.int64)
    df.to_csv(outdir / "simulated_counts.csv", index=False)

    bayes_cfg = BetaBinomialConfig()
    df2 = summarize_enrichment(df, config=bayes_cfg)
    sc = aggregate_enrichment_by_scaffold(df2, config=bayes_cfg)
    sc.to_csv(outdir / "scaffold_enrichment.csv", index=False)
    df2 = merge_scaffold_enrichment(df2, sc)
    df2.to_csv(outdir / "enrichment_results.csv", index=False)

    hits = top_hits(df2, k=50)
    hits.to_csv(outdir / "top_hits.csv", index=False)

    plot_enrichment_scatter(df2, outpath=outdir / "counts_scatter.png")
    plot_ranked_enrichment(df2, outpath=outdir / "ranked_enrichment.png")
    plot_volcano(df2, outpath=outdir / "volcano_like.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="DEL Bayesian Enrichment demo tool")
    parser.add_argument("--demo", action="store_true", help="Run simulation + Bayesian enrichment + plots")
    parser.add_argument("--outdir", type=str, default="out", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if args.demo:
        run_demo(outdir)
        return

    parser.print_help()


if __name__ == "__main__":
    main()

