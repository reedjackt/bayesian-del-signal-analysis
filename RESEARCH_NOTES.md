# Research Notes: Analytical Digamma Enrichment and Memory-Safe Uncertainty

**Audience:** Senior bioinformatician  
**Topic:** DEL Bayesian enrichment — transition from full-matrix Monte Carlo to a digamma-based posterior mean, Empirical Bayes priors via Method of Moments (MoM), and batched Monte Carlo for credible intervals.

---

## 1. Problem statement

The initial implementation estimated per-compound log2 enrichment by drawing independent Beta posterior samples for input and selected proportions, building a matrix of shape `(mc_samples, n_compounds)`, then summarizing means and quantiles along the sample axis.

That design is statistically transparent but **does not scale**: memory and runtime grow as **O(mc_samples × n)**. Typical DEL libraries have **n** from \(10^5\) to \(10^7\) or more; even modest `mc_samples` (e.g. 10k–50k) makes the allocation infeasible on a single workstation or HPC node unless **n** is artificially small.

**Lesson:** For library-scale genomics-style problems, any workflow that materializes a full **sample × entity** tensor for downstream summaries should be treated as a code smell unless there is a strong reason (e.g. rare-event tail estimation with no closed form and strict calibration requirements).

---

## 2. Analytical mean via digamma (Log-Beta expectations)

### 2.1 Model fragment

We retain the same conjugate scaffolding used in many count-based settings:

- \(p^{(\mathrm{in})}_i \mid \text{data} \sim \mathrm{Beta}(a^{(\mathrm{in})}_i, b^{(\mathrm{in})}_i)\)
- \(p^{(\mathrm{sel})}_i \mid \text{data} \sim \mathrm{Beta}(a^{(\mathrm{sel})}_i, b^{(\mathrm{sel})}_i)\)

with independence between the two channels **conditional on the observed counts** (the same simplifying assumption as the original sampler).

### 2.2 Closed-form expectation for the log-ratio

For \(X \sim \mathrm{Beta}(a,b)\),

\[
\mathbb{E}[\log X] = \psi(a) - \psi(a+b),
\]

where \(\psi\) is the digamma function. Therefore, for independent input and selected posteriors,

\[
\mathbb{E}\left[\log\frac{p^{(\mathrm{sel})}}{p^{(\mathrm{in})}}\right]
=
\bigl(\psi(a^{(\mathrm{sel})}) - \psi(a^{(\mathrm{sel})}+b^{(\mathrm{sel})})\bigr)
-
\bigl(\psi(a^{(\mathrm{in})}) - \psi(a^{(\mathrm{in})}+b^{(\mathrm{in})})\bigr).
\]

Converting to log base 2 is a fixed scalar: divide by \(\ln 2\).

### 2.3 Why this is attractive in production pipelines

- **Exact** under the stated independence and Beta-Binomial conjugacy (up to floating-point evaluation of special functions).
- **Vectorized** over compounds: memory **O(n)** for the parameter arrays and the output vector — no dependence on `mc_samples` for the mean.
- **Numerically well behaved** relative to sampling very small probabilities and then taking ratios: we work in **log-probability space** at the expectation level, which aligns with standard practice in expression and enrichment tooling.

**Lesson:** When the inferential target is a **linear functional of log-probabilities** (here, a difference), conjugate Beta models often admit **special-function** summaries that remove an entire class of Monte Carlo error for the point estimate. Reserve MC for quantities that genuinely lack a stable closed form.

**Caveat (important for scientific leadership):** \(\mathbb{E}[\log_2(p_\mathrm{sel}/p_\mathrm{in})]\) is **not** the same as \(\log_2 \mathbb{E}[p_\mathrm{sel}/p_\mathrm{in}]\). The codebase treats the **log-ratio expectation** as the primary “analytical enrichment” summary; that choice should be documented in methods sections and, where regulatory or publication scrutiny applies, validated against simulation for a subset of compounds.

---

## 3. Empirical Bayes prior: why Method of Moments (MoM)

### 3.1 What we needed from the prior

A flat \(\mathrm{Beta}(1,1)\) prior is a reasonable default for prototyping. In DEL-like data, **background structure** (PCR noise, uneven synthesis representation, pooling skew, batch effects) often produces a **narrow, structured cloud** of apparent enrichments even when biological binding is absent. A generic flat prior can leave the posterior **over-reactive** to sampling noise at low counts.

We wanted a **library-informed** prior without introducing a heavy per-iteration optimization loop (e.g. full marginal maximum likelihood in a hierarchical model), which would complicate reproducibility, slow down exploratory cycles, and increase operational risk in pipelines.

### 3.2 MoM on a bounded proxy

Enrichment itself lives on the real line; the Beta family does not directly model log-folds. The implementation therefore uses a **documented, deterministic proxy**:

1. Compute a **smoothed empirical log fold** per compound using pseudocounts (stabilizes low-count tails).
2. Map to \((0,1)\) via a **logistic squashing** \(u = \mathrm{expit}(\mathrm{log\_fold}/\tau)\), with \(\tau\) scaled by a **robust spread** of the library (median absolute deviation, floored to avoid division by zero).
3. Fit **Beta(\(\alpha,\beta\))** to the empirical distribution of \(u\) using **MoM**: closed-form inversion from mean and variance, with **guarded fallbacks** to \(\mathrm{Beta}(1,1)\) when the empirical variance is inconsistent with any Beta (boundary or degenerate cases).

### 3.3 Why MoM here (stability / speed trade)

| Consideration | MoM rationale |
|-----------------|---------------|
| **Speed** | O(n) summary statistics; no iterative solver in the default path. |
| **Stability** | Explicit fallbacks when variance is incompatible with a Beta; avoids brittle optimizers on messy tails. |
| **Interpretability** | Priors are still **conjugate**; downstream posteriors remain Beta-Binomial with the same code paths. |
| **Operational fit** | Easy to log, diff, and regression-test: mean/variance of \(u\) are simple QC metrics. |

**Lesson:** For pipeline engineering, **MoM + conjugacy** is often the sweet spot between “too naive” (flat priors) and “too heavy” (full hierarchical EB with nested optimization). The cost is **modeling honesty**: the MoM target is a **constructed** statistic, not a first-principles generative component. That is acceptable if the mapping is **versioned, documented, and QC’d** like any other normalization rule (cf. GC-content correction, TMM, quantile normalization).

---

## 4. Uncertainty: batched Monte Carlo and OOM safety

### 4.1 What still needs sampling

Even with a closed-form mean, **nonlinear functionals** of the joint posterior over \((p^{(\mathrm{in})}, p^{(\mathrm{sel})})\) — notably **quantiles** of \(\log_2(p^{(\mathrm{sel})}/p^{(\mathrm{in})})\) and \(\Pr(\log_2(\cdot) > 0)\) — do not reduce to elementary functions in a way we rely on in production code.

### 4.2 Batched strategy

Instead of allocating `(mc_samples, n)`, the implementation:

- Processes compounds in **chunks of size `mc_batch_size`** (configurable, on the order of a few thousand).
- For each chunk, allocates at most **`(mc_samples, mc_batch_size)`**, computes quantiles and empirical probabilities **along the sample axis**, writes results into preallocated length-`n` arrays, and discards the chunk.

Peak memory is therefore **O(mc_samples × mc_batch_size)**, independent of total library size **n** (aside from storing the data itself).

### 4.3 Optional fast path

For very large runs or HPC schedules where only point estimates matter, **`uncertainty_mode = "none"`** skips batched MC entirely, emitting NaNs for interval endpoints and probability-of-enrichment fields. That is a deliberate **contract** for scale-first workflows.

**Lesson:** Treat **point estimation** and **uncertainty reporting** as separable concerns. Engineering teams should be able to run **full-library point maps** cheaply, then re-launch uncertainty on subsets (hits, scaffolds, QC windows) with tuned `mc_samples` and batch sizes.

---

## 5. Scaffold-level aggregation (brief)

Scaffold summaries pool counts within a chemical family and apply the **same** digamma-based estimator to the pooled counts, using library totals and the **same** empirical prior estimated from the full compound table. That yields a **single interpretable family score** (`scaffold_log2_enrichment`) without inflating memory.

**Lesson:** Hierarchical Bayes is ideal when time and tooling allow; **pooled conjugate summaries** are a pragmatic first layer for ranking and triage, provided the team understands they are **not** a full partial-pooling model.

---

## 6. Closing lessons (senior IC framing)

1. **Default to analytical expectations** when the target functional matches what conjugacy gives you; use MC where the **functional** demands it, not where the **implementation** defaulted to it.
2. **MoM empirical priors** are a defensible engineering choice when they preserve **conjugacy**, **determinism**, and **fast failure modes** (fallback priors), at the expense of a **constructed** sufficient statistic — document it like any assay normalization.
3. **Batched MC** is the standard pattern for **OOM-safe** uncertainty at scale; the user-facing guarantee is a **bound on peak working set**, not “Monte Carlo is free.”
4. **Reproducibility:** tie stochastic batches to explicit seeds and fixed batching order; record `scipy`/`numpy` versions for special-function behavior in edge cases.

---

*This document reflects design intent at the time of the digamma refactor; it is not a substitute for formal methods text in regulated submissions or peer-reviewed manuscripts.*
