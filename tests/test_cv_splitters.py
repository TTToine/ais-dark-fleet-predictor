"""Tests for src.cv_splitters.GroupTimeSeriesSplitWithGap.

Verifies the two invariants that ``TimeSeriesSplitWithGap`` does NOT
guarantee:

1. Group disjointness:  set(train_mmsis) & set(val_mmsis) == empty.
2. Temporal gap:        max(train_ts) + gap <= min(val_ts).

Plus reproducibility and exactly-n_splits folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cv_splitters import GroupTimeSeriesSplitWithGap, holdout_split_by_mmsi


def _synthetic_df(n_mmsi: int = 30, rows_per_mmsi: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2024-01-01")
    for m in range(n_mmsi):
        start = base + pd.Timedelta(hours=int(rng.integers(0, 24 * 30)))
        ts = pd.date_range(start, periods=rows_per_mmsi, freq="10min")
        rows.append(pd.DataFrame({"MMSI": m, "Timestamp": ts, "x": rng.normal(size=rows_per_mmsi)}))
    return pd.concat(rows, ignore_index=True)


def test_group_disjointness_per_fold():
    df = _synthetic_df()
    splitter = GroupTimeSeriesSplitWithGap(n_splits=5, gap_hours=24, horizon_hours=24, random_state=0)
    seen = 0
    for train_idx, val_idx in splitter.split(df, groups=df["MMSI"].values):
        train_mmsis = set(df.iloc[train_idx]["MMSI"].unique())
        val_mmsis = set(df.iloc[val_idx]["MMSI"].unique())
        assert train_mmsis.isdisjoint(val_mmsis), (
            f"Group leakage: {train_mmsis & val_mmsis}"
        )
        seen += 1
    assert seen >= 1


def test_temporal_gap_respected():
    df = _synthetic_df()
    gap_h = 24.0
    splitter = GroupTimeSeriesSplitWithGap(
        n_splits=4, gap_hours=gap_h, horizon_hours=24, random_state=1
    )
    gap = pd.Timedelta(hours=gap_h)
    any_fold = False
    for train_idx, val_idx in splitter.split(df, groups=df["MMSI"].values):
        train_max = pd.to_datetime(df.iloc[train_idx]["Timestamp"]).max()
        val_min = pd.to_datetime(df.iloc[val_idx]["Timestamp"]).min()
        assert train_max + gap <= val_min, (
            f"Gap violated: train_max={train_max} val_min={val_min} gap={gap}"
        )
        any_fold = True
    assert any_fold, "No fold yielded — check synthetic data"


def test_yields_exactly_n_splits_when_data_sufficient():
    # Use long-lived vessels (7 days each at 10min freq) so every
    # temporal quantile falls inside enough vessels' observation windows.
    df = _synthetic_df(n_mmsi=50, rows_per_mmsi=1000, seed=2)
    n_splits = 5
    splitter = GroupTimeSeriesSplitWithGap(
        n_splits=n_splits, gap_hours=12, horizon_hours=24, random_state=42
    )
    folds = list(splitter.split(df, groups=df["MMSI"].values))
    assert len(folds) == n_splits


def test_reproducibility_same_seed_same_partition():
    df = _synthetic_df()
    s1 = GroupTimeSeriesSplitWithGap(n_splits=4, gap_hours=6, horizon_hours=12, random_state=123)
    s2 = GroupTimeSeriesSplitWithGap(n_splits=4, gap_hours=6, horizon_hours=12, random_state=123)
    folds_1 = [(set(df.iloc[tr]["MMSI"].unique()), set(df.iloc[va]["MMSI"].unique()))
               for tr, va in s1.split(df, groups=df["MMSI"].values)]
    folds_2 = [(set(df.iloc[tr]["MMSI"].unique()), set(df.iloc[va]["MMSI"].unique()))
               for tr, va in s2.split(df, groups=df["MMSI"].values)]
    assert folds_1 == folds_2


def test_different_seed_changes_partition():
    df = _synthetic_df()
    s1 = GroupTimeSeriesSplitWithGap(n_splits=4, gap_hours=6, horizon_hours=12, random_state=1)
    s2 = GroupTimeSeriesSplitWithGap(n_splits=4, gap_hours=6, horizon_hours=12, random_state=999)
    folds_1 = [tuple(sorted(df.iloc[va]["MMSI"].unique())) for _, va in s1.split(df, groups=df["MMSI"].values)]
    folds_2 = [tuple(sorted(df.iloc[va]["MMSI"].unique())) for _, va in s2.split(df, groups=df["MMSI"].values)]
    assert folds_1 != folds_2, "Different seeds produced identical partitions"


def test_raises_without_timestamp_column():
    df = pd.DataFrame({"MMSI": [1, 2, 3], "x": [0.1, 0.2, 0.3]})
    s = GroupTimeSeriesSplitWithGap(n_splits=2)
    with pytest.raises(ValueError, match="Timestamp"):
        list(s.split(df, groups=df["MMSI"].values))


def test_holdout_split_is_group_disjoint_and_reproducible():
    df = _synthetic_df(n_mmsi=40, rows_per_mmsi=20, seed=7)
    cv1, hold1, holdout_mmsis1 = holdout_split_by_mmsi(df, holdout_ratio=0.2, random_state=42)
    cv2, hold2, holdout_mmsis2 = holdout_split_by_mmsi(df, holdout_ratio=0.2, random_state=42)

    assert set(cv1["MMSI"]).isdisjoint(set(hold1["MMSI"]))
    np.testing.assert_array_equal(holdout_mmsis1, holdout_mmsis2)
    assert len(holdout_mmsis1) == 8  # 20% of 40
