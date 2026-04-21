# DEL Bayesian Enrichment

Lightweight toolkit for simulating DNA-Encoded Library (DEL) selection experiments and analyzing enrichment using a Beta-Binomial Bayesian model.

## What’s included
- `src/simulator.py`: Standard DEL-style simulation (counts, selection, amplification/noise)
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

Outputs (CSV + plots) are written to `del-bayesian-enrichment/out/` by default.

## Project layout

```
del-bayesian-enrichment/
  data/
  notebooks/
  src/
  main.py
  requirements.txt
```

## Notes
- The Bayesian model treats observed counts as Binomial draws with Beta priors, then summarizes enrichment as a posterior over log2 fold-change via Monte Carlo sampling.
- This is intentionally minimal boilerplate you can extend with real DEL preprocessing and multi-round models.

