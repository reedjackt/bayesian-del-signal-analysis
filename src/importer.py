from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KinDELImportConfig:
    """
    Import configuration for insitro / KinDEL-style count tables.

    Expected columns (default):
      - pre-selection_counts  -> input_count
      - target_replicate_1/2/3 -> selected replicate counts
    """

    input_col: str = "pre-selection_counts"
    selected_replicate_cols: tuple[str, ...] = (
        "target_replicate_1",
        "target_replicate_2",
        "target_replicate_3",
    )
    id_col_candidates: tuple[str, ...] = ("compound_id", "compound", "compoundId", "id")
    output_input_col: str = "input_count"
    output_selected_col: str = "selected_count"

    fillna_with_zero: bool = True
    min_input_count: int = 0
    min_selected_count: int = 0
    min_total_count: int = 0
    fail_on_all_zero_selected_replicates: bool = True
    warn_on_zero_selected_replicates: bool = True
    # LibraryScaler corrects unequal sequencing depth across replicates; this
    # gate catches pathological totals (mislabels / index hopping). KinDEL runs
    # often have one replicate materially deeper without implying failure — use a
    # permissive default and tune via CLI when needed.
    max_selected_replicate_imbalance: float | None = 500.0

    selected_aggregation: Literal["sum_raw", "sum_scaled_depth"] = "sum_scaled_depth"
    depth_target: Literal["median", "mean", "min", "max"] = "median"

    # Optional scaffold key for downstream ``aggregate_enrichment_by_scaffold``.
    # If ``scaffold_id_col`` is set, that column is copied to ``scaffold_id``.
    # Else if ``molecule_hash_prefix_len`` is set, ``scaffold_id`` is the first
    # ``molecule_hash_prefix_len`` characters of ``molecule_hash_col``.
    scaffold_id_col: str | None = None
    molecule_hash_col: str = "molecule_hash"
    molecule_hash_prefix_len: int | None = None


class LibraryScaler:
    """
    Sequencing-depth normalizer for per-pool count columns.

    For combining multiple selected replicates into a single `selected_count`,
    scaling each replicate to a common target depth prevents a single deep pool
    from dominating the combined library.
    """

    def __init__(self, target: Literal["median", "mean", "min", "max"] = "median") -> None:
        self.target = target
        self.totals_: dict[str, float] = {}
        self.scale_factors_: dict[str, float] = {}
        self.target_total_: float | None = None

    @staticmethod
    def _safe_total(x: pd.Series) -> float:
        s = pd.to_numeric(x, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return float(np.sum(np.clip(s, 0.0, np.inf)))

    def fit(self, df: pd.DataFrame, cols: Sequence[str]) -> "LibraryScaler":
        totals = {c: self._safe_total(df[c]) for c in cols}
        arr = np.array(list(totals.values()), dtype=float)
        if np.any(arr < 0) or np.all(arr <= 0):
            raise ValueError("All pool totals are non-positive; cannot depth-normalize.")

        if self.target == "median":
            target_total = float(np.median(arr[arr > 0]))
        elif self.target == "mean":
            target_total = float(np.mean(arr[arr > 0]))
        elif self.target == "min":
            target_total = float(np.min(arr[arr > 0]))
        elif self.target == "max":
            target_total = float(np.max(arr[arr > 0]))
        else:
            raise ValueError(f"Unknown depth target: {self.target}")

        scale = {}
        for c, tot in totals.items():
            if tot <= 0:
                scale[c] = 0.0
            else:
                scale[c] = target_total / float(tot)

        self.totals_ = totals
        self.scale_factors_ = scale
        self.target_total_ = target_total
        return self

    def transform(self, df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        if not self.scale_factors_:
            raise ValueError("LibraryScaler is not fit; call fit() first.")
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                raise ValueError(f"Missing column for scaling: {c}")
            factor = float(self.scale_factors_.get(c, 1.0))
            v = pd.to_numeric(out[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            v = np.clip(v, 0.0, np.inf) * factor
            out[c] = np.rint(v).astype(np.int64)
        return out

    def fit_transform(self, df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        return self.fit(df, cols).transform(df, cols)


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        return pd.read_csv(p, sep=sep)
    raise ValueError(f"Unsupported input extension: {p.suffix}")


def _normalize_kindel_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map public KinDEL Parquet column names onto the CSV-style contract expected
    by :class:`KinDELImportConfig` (e.g. seq_load / seq_target_* from
    kin-del-2024 Parquet dumps).
    """
    out = df
    rename: dict[str, str] = {}
    if "pre-selection_counts" not in out.columns and "seq_load" in out.columns:
        rename["seq_load"] = "pre-selection_counts"
    for i in (1, 2, 3):
        tr = f"target_replicate_{i}"
        st = f"seq_target_{i}"
        if tr not in out.columns and st in out.columns:
            rename[st] = tr
    if rename:
        out = out.rename(columns=rename)
    return out


def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _assign_scaffold_id(out: pd.DataFrame, config: KinDELImportConfig) -> pd.DataFrame:
    """Populate ``scaffold_id`` from an explicit column or a molecule-hash prefix."""
    if config.scaffold_id_col is not None:
        col = config.scaffold_id_col
        if col not in out.columns:
            raise ValueError(f"scaffold_id_col={col!r} not found in table.")
        out["scaffold_id"] = out[col].astype(str)
        return out
    if config.molecule_hash_prefix_len is not None:
        n = int(config.molecule_hash_prefix_len)
        if n <= 0:
            raise ValueError("molecule_hash_prefix_len must be a positive integer.")
        hcol = config.molecule_hash_col
        if hcol not in out.columns:
            raise ValueError(
                f"molecule_hash_prefix_len is set but column {hcol!r} is missing."
            )
        out["scaffold_id"] = out[hcol].astype(str).str.slice(0, n)
        return out
    return out


def import_kindel_counts(
    df: pd.DataFrame,
    config: KinDELImportConfig = KinDELImportConfig(),
) -> pd.DataFrame:
    """
    Convert a KinDEL-like table into the project-wide count contract:
      - `input_count`
      - `selected_count`

    The output preserves all original columns and appends/overwrites
    the standardized count columns.
    """
    if config.input_col not in df.columns:
        raise ValueError(f"Missing KinDEL input column: {config.input_col}")
    for c in config.selected_replicate_cols:
        if c not in df.columns:
            raise ValueError(f"Missing KinDEL selected replicate column: {c}")

    out = df.copy()
    count_cols = (config.input_col, *config.selected_replicate_cols)
    for c in count_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if config.fillna_with_zero:
            out[c] = out[c].fillna(0.0)
        out[c] = np.clip(out[c].to_numpy(dtype=float), 0.0, np.inf)

    # Standardize/ensure an identifier column exists when possible (non-fatal).
    id_col = _first_present(out, config.id_col_candidates)
    if id_col is None:
        out.insert(0, "compound_id", np.arange(len(out), dtype=np.int64))
    elif id_col != "compound_id":
        out = out.rename(columns={id_col: "compound_id"})

    out[config.output_input_col] = np.rint(out[config.input_col].to_numpy(dtype=float)).astype(
        np.int64
    )

    if config.selected_aggregation == "sum_raw":
        sel = np.zeros(len(out), dtype=np.int64)
        for c in config.selected_replicate_cols:
            sel += np.rint(out[c].to_numpy(dtype=float)).astype(np.int64)
        out[config.output_selected_col] = sel

    elif config.selected_aggregation == "sum_scaled_depth":
        # Dataset-level QC for real sequencing runs.
        totals = {c: LibraryScaler._safe_total(out[c]) for c in config.selected_replicate_cols}
        positive_totals = [t for t in totals.values() if t > 0]
        if len(positive_totals) == 0:
            if config.fail_on_all_zero_selected_replicates:
                raise ValueError(
                    "All selected replicate library totals are zero; cannot compute selected_count. "
                    f"Columns={list(config.selected_replicate_cols)} totals={totals}. "
                    "This usually indicates a failed selection sequencing run, a schema mismatch, "
                    "or upstream filtering that removed all selected counts."
                )
            out[config.output_selected_col] = np.zeros(len(out), dtype=np.int64)
            out.attrs["kindel_selected_pool_totals"] = totals
            out.attrs["kindel_selected_pool_scale_factors"] = {c: 0.0 for c in totals}
            out.attrs["kindel_selected_depth_target_total"] = 0.0
        else:
            zero_reps = [c for c, t in totals.items() if t <= 0]
            if zero_reps and config.warn_on_zero_selected_replicates:
                out.attrs["kindel_selected_zero_replicates"] = {
                    "columns": zero_reps,
                    "totals": {c: totals[c] for c in zero_reps},
                }

            if config.max_selected_replicate_imbalance is not None and len(positive_totals) >= 2:
                tmin = float(np.min(positive_totals))
                tmax = float(np.max(positive_totals))
                ratio = (tmax / tmin) if tmin > 0 else float("inf")
                if ratio > float(config.max_selected_replicate_imbalance):
                    raise ValueError(
                        "Selected replicate library totals are extremely imbalanced; refusing to "
                        "depth-normalize because this is usually contamination / index hopping / "
                        f"mislabeling. totals={totals} ratio={ratio:.3g} "
                        f"threshold={config.max_selected_replicate_imbalance}."
                    )

            scaler = LibraryScaler(target=config.depth_target)
            scaled = scaler.fit_transform(out, list(config.selected_replicate_cols))
            # Sum as float then round once to reduce low-count rounding artifacts.
            sel_f = np.zeros(len(scaled), dtype=float)
            for c in config.selected_replicate_cols:
                sel_f += scaled[c].to_numpy(dtype=float)
            out[config.output_selected_col] = np.rint(sel_f).astype(np.int64)
            out.attrs["kindel_selected_pool_totals"] = scaler.totals_
            out.attrs["kindel_selected_pool_scale_factors"] = scaler.scale_factors_
            out.attrs["kindel_selected_depth_target_total"] = scaler.target_total_

    else:
        raise ValueError(f"Unknown selected_aggregation: {config.selected_aggregation}")

    out = _assign_scaffold_id(out, config)

    # Low-count filters (common in real sequencing tables).
    mask = np.ones(len(out), dtype=bool)
    if config.min_input_count > 0:
        mask &= out[config.output_input_col].to_numpy(dtype=np.int64) >= int(config.min_input_count)
    if config.min_selected_count > 0:
        mask &= out[config.output_selected_col].to_numpy(dtype=np.int64) >= int(
            config.min_selected_count
        )
    if config.min_total_count > 0:
        tot = out[config.output_input_col].to_numpy(dtype=np.int64) + out[
            config.output_selected_col
        ].to_numpy(dtype=np.int64)
        mask &= tot >= int(config.min_total_count)

    out = out.loc[mask].reset_index(drop=True)
    # Guardrail: make downstream Bayesian model failures actionable.
    if len(out) == 0:
        raise ValueError(
            "All rows were filtered out during import. "
            f"Filters: min_input_count={config.min_input_count}, "
            f"min_selected_count={config.min_selected_count}, "
            f"min_total_count={config.min_total_count}."
        )
    return out


def load_kindel_dataset(
    path: str | Path,
    config: KinDELImportConfig = KinDELImportConfig(),
) -> pd.DataFrame:
    df = read_table(path)
    df = _normalize_kindel_column_aliases(df)
    return import_kindel_counts(df, config=config)

