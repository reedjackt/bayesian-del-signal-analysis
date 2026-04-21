from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.simulator import SimulationConfig, simulate_del_experiment
from src.importer import KinDELImportConfig, load_kindel_dataset
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


def run_real_world_kindel(
    outdir: Path,
    input_path: Path,
    *,
    min_total_count: int = 0,
    uncertainty_mode: str | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df_raw = load_kindel_dataset(
        input_path,
        config=KinDELImportConfig(min_total_count=int(min_total_count)),
    )
    df_raw.to_csv(outdir / "imported_counts.csv", index=False)

    if uncertainty_mode is None:
        bayes_cfg = BetaBinomialConfig()
    else:
        bayes_cfg = BetaBinomialConfig(uncertainty_mode=uncertainty_mode)  # type: ignore[arg-type]
    df2 = summarize_enrichment(df_raw, config=bayes_cfg)

    if "scaffold_id" in df2.columns:
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
    parser.add_argument(
        "--demo", action="store_true", help="Run simulation + Bayesian enrichment + plots"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input dataset path (CSV/TSV/Parquet) for real-world analysis; if omitted, use --demo for simulation",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="kindel",
        choices=["kindel"],
        help="Real-world input schema (currently: kindel)",
    )
    parser.add_argument("--outdir", type=str, default="out", help="Output directory")
    parser.add_argument(
        "--min-total-count",
        type=int,
        default=0,
        help="Drop rows with (input_count + selected_count) < this threshold after import",
    )
    parser.add_argument(
        "--uncertainty-mode",
        type=str,
        default=None,
        choices=["mc_batched", "none"],
        help='Uncertainty calculation mode: "mc_batched" (default) or "none" to skip MC',
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)

    if args.input is not None:
        if args.schema != "kindel":
            raise ValueError(f"Unsupported schema: {args.schema}")
        run_real_world_kindel(
            outdir,
            Path(args.input),
            min_total_count=int(args.min_total_count),
            uncertainty_mode=args.uncertainty_mode,
        )
        return

    if args.demo:
        run_demo(outdir)
        return

    parser.print_help()


if __name__ == "__main__":
    main()

