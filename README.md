# DEL Bayesian Enrichment

Lightweight toolkit for simulating DNA-Encoded Library (DEL) selection experiments and analyzing enrichment using a Beta-Binomial Bayesian model.

## What’s included
- `src/simulator.py`: Standard DEL-style simulation (counts, selection, amplification/noise)
- `src/importer.py`: Real-world ingestion (KinDEL schema → `input_count`/`selected_count`, depth normalization, filters)
- `src/analyzer.py`: Beta-Binomial Bayesian enrichment inference with credible intervals
- `src/visualizer.py`: Common enrichment plots (ranked enrichment, scatter, volcano-like)
- `main.py`: Simple entry point to simulate + analyze + plot

## Quickstart

Create a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the demo pipeline:

```bash
python main.py --demo
```

Outputs (CSV + plots) are written to `out/` by default.

## Results Showcase

Ranked Bayesian enrichment (blue “Hits” are simulated high-confidence binders surfaced via Bayesian shrinkage):

![Ranked enrichment plot](assets/ranked_enrichment.png)

Volcano-like view of enrichment vs signal strength (blue “Hits” are simulated high-confidence binders surfaced via Bayesian shrinkage):

![Volcano-like plot](assets/volcano_like.png)

## How it Works

We model **input** and **selected** read counts as noisy observations of underlying per-compound rates. Using a **Beta prior** with a **Binomial likelihood**, each compound’s input and selected rates have conjugate **Beta posteriors**. Enrichment is summarized as the posterior expectation of log fold-change:

- **Analytical mean (primary point estimate)**: for $p \sim \mathrm{Beta}(a,b)$,
  $$E[\log_2(p)] = \frac{\psi(a) - \psi(a+b)}{\ln(2)}.$$
  The pipeline uses this identity to compute $E[\log_2(p_\mathrm{sel}/p_\mathrm{in})]$ as a difference of expectations (digamma), giving a stable point estimate without slow Monte Carlo.
- **Uncertainty (optional)**: credible intervals and $P(\log_2 \text{enrichment} > 0)$ can be estimated via batched Monte Carlo sampling when needed.

## Real-World Data (KinDEL)
If you have an insitro/KinDEL-style count table, you can run the same Bayesian enrichment pipeline on a CSV/TSV/Parquet file.

KinDEL column mapping (default):
- `pre-selection_counts` → `input_count`
- `target_replicate_1`, `target_replicate_2`, `target_replicate_3` → depth-normalized and aggregated into `selected_count`

Run:

```bash
python main.py --input path/to/kindel_counts.parquet --schema kindel --outdir out
```

Optional filters (recommended for sparse tables):

```bash
python main.py --input path/to/kindel_counts.csv --schema kindel --min-total-count 3 --outdir out
```

## Project layout

```
bayesian-del-signal-analysis/
  assets/             # curated plots for README/GitHub
  data/
  notebooks/
  out/                # demo outputs (ignored by git)
  main.py             # CLI entry point (demo orchestration)
  src/
    importer.py       # real-world ingestion (KinDEL)
  requirements.txt
```

## Notes
- The point estimate for enrichment uses an analytical digamma identity; Monte Carlo is only used for uncertainty summaries when enabled.
- Run `python main.py --demo` from the repo root so `src` resolves as a package.
- This is intentionally minimal boilerplate you can extend with real DEL preprocessing and multi-round models.
