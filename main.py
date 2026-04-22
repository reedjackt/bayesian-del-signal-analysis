from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.simulator import SimulationConfig, simulate_del_experiment
from src.importer import KinDELImportConfig, load_kindel_dataset
from src.analyzer import (
    BetaBinomialConfig,
    aggregate_enrichment_by_scaffold,
    final_triage_hits,
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

    triage = final_triage_hits(df2, k=50, prob_min=0.95)
    if triage.empty:
        triage = top_hits(df2, k=50)
    triage.to_csv(outdir / "top_hits.csv", index=False)

    plot_enrichment_scatter(df2, outpath=outdir / "counts_scatter.png")
    plot_ranked_enrichment(df2, outpath=outdir / "ranked_enrichment.png")
    plot_volcano(df2, outpath=outdir / "volcano_like.png")


def run_real_world_kindel(
    outdir: Path,
    input_path: Path,
    *,
    kindel_config: KinDELImportConfig | None = None,
    uncertainty_mode: str | None = None,
    hit_prob_min: float = 0.95,
    hit_k: int = 50,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df_raw = load_kindel_dataset(
        input_path,
        config=kindel_config
        if kindel_config is not None
        else KinDELImportConfig(),
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

    triage = final_triage_hits(df2, k=int(hit_k), prob_min=float(hit_prob_min))
    if triage.empty:
        triage = top_hits(df2, k=int(hit_k))
    triage.to_csv(outdir / "top_hits.csv", index=False)

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
        choices=["delta", "mc_batched", "none"],
        help='Uncertainty calculation mode: "delta" (fast, memory-safe), "mc_batched" (expensive), or "none"',
    )
    parser.add_argument(
        "--scaffold-col",
        type=str,
        default=None,
        metavar="COL",
        help="Use this column as scaffold_id for family-level aggregation (overrides hash prefix)",
    )
    parser.add_argument(
        "--molecule-hash-prefix-len",
        type=int,
        default=None,
        metavar="N",
        help="Set scaffold_id to the first N characters of molecule_hash (DDR1-friendly)",
    )
    parser.add_argument(
        "--max-selected-replicate-imbalance",
        type=float,
        default=None,
        metavar="RATIO",
        help="Max max/min ratio across positive selected-pool totals before failing (default: 500).",
    )
    parser.add_argument(
        "--disable-selected-replicate-imbalance-check",
        action="store_true",
        help="Skip the selected-pool imbalance guardrail (not recommended)",
    )
    parser.add_argument(
        "--hit-prob-min",
        type=float,
        default=0.95,
        help="final triage: require prob_enriched > this (default 0.95)",
    )
    parser.add_argument(
        "--hits",
        type=int,
        default=50,
        metavar="K",
        help="Number of rows to keep in top_hits after triage (default 50)",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)

    if args.input is not None:
        if args.schema != "kindel":
            raise ValueError(f"Unsupported schema: {args.schema}")
        if args.disable_selected_replicate_imbalance_check:
            max_imb: float | None = None
        elif args.max_selected_replicate_imbalance is not None:
            max_imb = float(args.max_selected_replicate_imbalance)
        else:
            max_imb = KinDELImportConfig.max_selected_replicate_imbalance

        kindel_kw: dict = {"min_total_count": int(args.min_total_count)}
        if max_imb is not None:
            kindel_kw["max_selected_replicate_imbalance"] = max_imb
        else:
            kindel_kw["max_selected_replicate_imbalance"] = None
        if args.scaffold_col is not None:
            kindel_kw["scaffold_id_col"] = args.scaffold_col
        if args.molecule_hash_prefix_len is not None:
            kindel_kw["molecule_hash_prefix_len"] = int(args.molecule_hash_prefix_len)

        run_real_world_kindel(
            outdir,
            Path(args.input),
            kindel_config=KinDELImportConfig(**kindel_kw),
            uncertainty_mode=args.uncertainty_mode,
            hit_prob_min=float(args.hit_prob_min),
            hit_k=int(args.hits),
        )
        return

    if args.demo:
        run_demo(outdir)
        return

    parser.print_help()


if __name__ == "__main__":
    main()

