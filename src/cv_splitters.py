"""Cross-validation splitters that enforce BOTH temporal ordering and
group (MMSI) disjointness.

Motivation
----------
The legacy ``TimeSeriesSplitWithGap`` in :mod:`src.gb_training` separates
train and validation folds by a temporal buffer but does NOT enforce
group separation. With one row per (MMSI, timestamp) the same vessel
can appear in both fold halves (earlier rows in train, later rows in
val), letting the model implicitly learn a per-vessel kinematic
fingerprint. This artificially inflates PR-AUC and biases Optuna's
hyperparameter search.

``GroupTimeSeriesSplitWithGap`` fixes this by:

1. Hashing the unique groups (MMSIs) into ``n_splits`` disjoint folds
   ONCE up front (deterministic via ``random_state``).
2. For fold k, taking the val groups = fold k's MMSI partition.
3. Choosing a temporal cutoff ``T_k`` (the median start time of the
   val groups) and requiring:
       train rows: t < T_k - gap   AND mmsi NOT in val_groups
       val   rows: T_k <= t < T_k + horizon  AND mmsi in val_groups

Both constraints together prevent leakage from either direction.
"""
from __future__ import annotations

import logging
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GroupTimeSeriesSplitWithGap:
    """Time-series CV that enforces group disjointness on top of a
    temporal gap. sklearn-style ``split(X, y, groups)`` API so it
    plugs into Optuna / ``cross_val_score`` without modifications.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    gap_hours : float
        Temporal buffer (in hours) between the end of train and the
        start of validation. Should match the prediction horizon.
    horizon_hours : float
        Validation window length (in hours) after the cutoff.
    timestamp_col : str
        Column name in X carrying the timestamp.
    random_state : Optional[int]
        Seed for the MMSI -> fold partition. Same seed -> same folds.
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap_hours: float = 24.0,
        horizon_hours: float = 24.0,
        timestamp_col: str = "Timestamp",
        random_state: Optional[int] = 42,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = int(n_splits)
        self.gap = pd.Timedelta(hours=float(gap_hours))
        self.horizon = pd.Timedelta(hours=float(horizon_hours))
        self.timestamp_col = timestamp_col
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _partition_groups(self, unique_groups: np.ndarray) -> List[np.ndarray]:
        """Deterministically partition unique groups into ``n_splits``
        folds. Returns a list of length ``n_splits``."""
        rng = np.random.default_rng(self.random_state)
        shuffled = unique_groups.copy()
        rng.shuffle(shuffled)
        return [np.array(part) for part in np.array_split(shuffled, self.n_splits)]

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            if "MMSI" not in X.columns:
                raise ValueError(
                    "`groups` is None and X has no 'MMSI' column. "
                    "Pass groups=mmsi_array or include MMSI in X."
                )
            groups = X["MMSI"].values

        if self.timestamp_col not in X.columns:
            raise ValueError(
                f"Timestamp column '{self.timestamp_col}' not found in X. "
                f"Columns: {list(X.columns)}"
            )

        groups = np.asarray(groups)
        timestamps = pd.to_datetime(X[self.timestamp_col]).values
        if len(groups) != len(X):
            raise ValueError("len(groups) != len(X)")

        unique_groups = np.unique(groups)
        if len(unique_groups) < self.n_splits:
            raise ValueError(
                f"Not enough unique groups ({len(unique_groups)}) for "
                f"n_splits={self.n_splits}. Reduce n_splits or pool more vessels."
            )

        fold_partition = self._partition_groups(unique_groups)

        # Cutoffs T_k: evenly spaced time quantiles over the full timeline.
        # Using quantiles (rather than per-fold val-group start times) makes
        # the train window comparable across folds: fold 1 sees a small
        # train window, fold k a progressively larger one — i.e. true
        # walk-forward semantics.
        ts_int = pd.to_datetime(timestamps).astype("datetime64[ns]").astype(np.int64)
        quantile_levels = np.linspace(
            1.0 / (self.n_splits + 1),
            self.n_splits / (self.n_splits + 1),
            self.n_splits,
        )
        cutoffs = [
            pd.Timestamp(int(np.quantile(ts_int, q))).to_datetime64()
            for q in quantile_levels
        ]
        cutoffs = [pd.Timestamp(c) for c in cutoffs]

        for k, val_groups in enumerate(fold_partition):
            val_groups_set = set(val_groups.tolist())
            if not val_groups_set:
                continue
            cutoff = cutoffs[k]

            train_lo = pd.Timestamp(timestamps.min())
            train_hi = cutoff - self.gap
            val_lo = cutoff
            val_hi = cutoff + self.horizon

            in_val_groups = np.isin(groups, val_groups)
            ts = pd.to_datetime(timestamps)

            train_mask = (~in_val_groups) & (ts >= train_lo) & (ts < train_hi)
            val_mask = (in_val_groups) & (ts >= val_lo) & (ts < val_hi)

            train_idx = np.flatnonzero(train_mask)
            val_idx = np.flatnonzero(val_mask)

            if len(train_idx) < 50 or len(val_idx) < 10:
                logger.debug(
                    "Fold %d skipped: |train|=%d |val|=%d (cutoff=%s)",
                    k, len(train_idx), len(val_idx), cutoff,
                )
                continue

            yield train_idx, val_idx


def holdout_split_by_mmsi(
    df: pd.DataFrame,
    holdout_ratio: float = 0.2,
    mmsi_col: str = "MMSI",
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Partition a dataframe into a CV pool and a hold-out test set
    BY MMSI (group-disjoint).

    Returns
    -------
    cv_df, holdout_df, holdout_mmsis
        ``holdout_mmsis`` is the array of MMSIs reserved for the hold-out.
        Persist it for auditability.
    """
    if mmsi_col not in df.columns:
        raise ValueError(f"Column '{mmsi_col}' not in df")
    rng = np.random.default_rng(random_state)
    unique = np.unique(df[mmsi_col].values)
    rng.shuffle(unique)
    n_hold = max(1, int(round(len(unique) * holdout_ratio)))
    holdout_mmsis = np.sort(unique[:n_hold])
    cv_mmsis = unique[n_hold:]

    cv_df = df[df[mmsi_col].isin(cv_mmsis)].copy()
    holdout_df = df[df[mmsi_col].isin(holdout_mmsis)].copy()
    return cv_df, holdout_df, holdout_mmsis
