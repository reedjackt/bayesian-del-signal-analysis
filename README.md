# DEL Bayesian Enrichment — Applied Data Engineering

This repository is an **applied data engineering** project for DNA-encoded library (DEL) selection readouts. The focus is **high-cardinality, sparse count tables**: millions of distinct chemical entities, most observed only a handful of times, with heavy-tailed sampling noise. The shipped pipeline normalizes heterogeneous exports, runs conjugate Beta–Binomial enrichment at library scale, and keeps memory and runtime bounded so multi-gigabyte Parquet inputs remain practical on a single machine.

## System performance

| Dimension | Implementation / outcome |
|-----------|---------------------------|
| **Dataset scale** | **912,789** unique compounds on a **~1M-row** KinDEL DDR1 extract (subset of the full **~81M-row** KinDEL library). |
| **Ingestion** | **Native Parquet** I/O plus **automated schema normalization** (column aliasing, depth normalization across replicates, optional low-count filters) in `src/importer.py`. |
| **Computational efficiency** | Full-library uncertainty no longer relies on materializing Monte Carlo sample tensors. **Analytical digamma means** for point estimates and **delta-method** Normal approximations for intervals and enrichment probability (`BetaBinomialConfig(uncertainty_mode="delta")` in `src/analyzer.py`) keep end-to-end enrichment **under ~60 seconds** on this workload class. |
| **Memory management** | **Explicit `gc.collect()`** after large frame transitions and **hexbin** density plots (1D arrays only) avoid duplicate wide tables for QC visuals. Observed **peak RAM ~1.45 GB** for the production-scale DDR1 run documented in `notebooks/real_world_exploration.ipynb`. |

## Bayesian shield (why shrinkage matters)

The **Beta–Binomial** layer is not decorative: it is the main defense against low-count volatility. An **empirical Beta prior** (method-of-moments fit on a library-wide proxy, with guarded fallbacks) pulls thousands of weak-evidence compounds toward the bulk prior instead of letting Poisson-like noise masquerade as signal.

On the DDR1 production extract, that shrinkage plus posterior **\(P(\log_2 \text{enrichment} > 0)\)** triage effectively **suppresses noise across 700k+ low-confidence rows** while surfacing **148,965** compounds above **\(P > 0.95\)** as high-confidence enrichment candidates. At scaffold level, the strongest chemical family in this run is **`c2de1253`** with **7.76** log₂ enrichment versus the library background—useful for prioritizing synthesis follow-up.

## Engineering robustness (testing)

Early development used **synthetic DEL counts** (`src/simulator.py`, `python main.py --demo`) to validate end-to-end wiring. That synthetic path is now treated explicitly as a **validation suite**: known ground truth for hit structure and count generative assumptions lets you regression-test **statistical logic** (posterior means, priors, triage, scaffold pooling) before pointing the same code at **multi-gigabyte** experimental Parquet files where labels are unknown and failures are expensive.

Run the synthetic check from the repo root:

```bash
python main.py --demo
```

Outputs land in `out/` (CSVs + plots). Extend tests or notebooks to assert properties of the enriched frame against simulated hits.

## Repository cleanup (data layout)

The pipeline expects a **binary Parquet** file, for example:

`data/kindel_ddr1_real.parquet`

If you see a file named `data/kindel_ddr1_real.parquet.htm`, that is a **misnamed or browser-saved artifact**, not a valid Parquet file—**delete it locally** or ensure it is **git-ignored** (the repo ignores `*.parquet.htm`). Replace it with the real `.parquet` export from your data provider.

## What’s included

- `src/simulator.py` — Synthetic DEL-style counts (validation suite / demos)
- `src/importer.py` — KinDEL-style ingestion → `input_count` / `selected_count`
- `src/analyzer.py` — Beta–Binomial enrichment, digamma means, delta-method (or optional batched MC) uncertainty, scaffold aggregation
- `src/visualizer.py` — Enrichment scatter, ranked, volcano-style plots
- `main.py` — CLI: `--demo` or `--input … --schema kindel`
- `notebooks/real_world_exploration.ipynb` — Production-scale KinDEL walkthrough

## Quickstart

```bash
pip install -r requirements.txt
python main.py --demo
```

Real KinDEL Parquet (after placing the file under `data/`):

```bash
python main.py --input data/kindel_ddr1_real.parquet --schema kindel --outdir out
```

Optional sparsity filters (recommended on noisy tables):

```bash
python main.py --input data/kindel_ddr1_real.parquet --schema kindel --min-total-count 3 --outdir out
```

Force uncertainty mode explicitly if needed:

```bash
python main.py --input data/kindel_ddr1_real.parquet --schema kindel --uncertainty-mode delta --outdir out
```

## How it works (compact)

Per compound, **input** and **selected** counts update independent **Beta** posteriors under a **Binomial** likelihood. The primary **log₂ enrichment** summary is the posterior expectation of the log-ratio,

\[
\mathbb{E}[\log_2(p_\mathrm{sel}/p_\mathrm{in})]
= \frac{1}{\ln 2}\Bigl(
\bigl(\psi(a_\mathrm{sel}) - \psi(a_\mathrm{sel}+b_\mathrm{sel})\bigr)
-
\bigl(\psi(a_\mathrm{in}) - \psi(a_\mathrm{in}+b_\mathrm{in})\bigr)
\Bigr),
\]

with \(\psi\) the digamma function. Uncertainty defaults to a **delta-method** Gaussian approximation over the same parameters; batched Monte Carlo remains available for comparison or small-\(n\) studies.

## Results showcase (synthetic demo)

Ranked Bayesian enrichment (blue “Hits” are simulated high-confidence binders):

![Ranked enrichment plot](assets/ranked_enrichment.png)

Volcano-like view of enrichment vs signal strength:

![Volcano-like plot](assets/volcano_like.png)

## Project layout

```
bayesian-del-signal-analysis/
  assets/             # curated plots for README
  data/               # local Parquet inputs (large files not committed)
  notebooks/          # production-scale exploration
  out/                # run outputs (gitignored)
  main.py
  src/
  requirements.txt
```

## Notes

- Run `python main.py` from the **repo root** so `from src.…` imports resolve.
- For library-scale runs, prefer **`uncertainty_mode="delta"`** (default in `BetaBinomialConfig`) unless you need batched MC for a subset.
- Deeper methods discussion: [`RESEARCH_NOTES.md`](RESEARCH_NOTES.md). Module wiring: [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).
