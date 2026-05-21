"""Tests for src.gb_training HPO robustness.

Verifica che la pipeline NON faccia silenziosamente "fake success" quando
il target è troppo raro per costruire fold validi.

Regressione: la run di 17h che produsse PR-AUC=0.0005 e ROC-AUC=0.4993
(letteralmente random) NON avrebbe mai dovuto salvare un modello.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("optuna")

from src.gb_training import DarkFleetPredictor


def _make_zero_positive_df(n_rows: int = 1000, n_vessels: int = 20, seed: int = 0) -> pd.DataFrame:
    """Dataset con 0 positivi: target_dark_fleet sempre 0."""
    rng = np.random.default_rng(seed)
    rows_per_vessel = n_rows // n_vessels
    rows = []
    base = pd.Timestamp("2024-01-01")
    for m in range(n_vessels):
        start = base + pd.Timedelta(hours=int(rng.integers(0, 24 * 30)))
        ts = pd.date_range(start, periods=rows_per_vessel, freq="10min")
        rows.append(pd.DataFrame({
            "MMSI": m,
            "Timestamp": ts,
            "delta_SOG": rng.normal(0, 0.5, rows_per_vessel),
            "delta_COG": rng.normal(0, 2.0, rows_per_vessel),
            "speed_acc": rng.normal(0, 0.5, rows_per_vessel),
            "turn_rate": rng.normal(0, 2.0, rows_per_vessel),
            "dt_prev_hours": np.full(rows_per_vessel, 1 / 6),
            "prob_regime_sospetto": rng.uniform(0, 1, rows_per_vessel),
            "target_dark_fleet": 0,
        }))
    return pd.concat(rows, ignore_index=True)


def test_hpo_degenerate_raises_runtime_error_with_zero_positives():
    """0 positivi → ogni trial deve essere PRUNED → optimize_and_train deve
    sollevare RuntimeError con 'HPO degenerate' nel messaggio. Nessun modello salvato."""
    df = _make_zero_positive_df(n_rows=1000, n_vessels=20)
    predictor = DarkFleetPredictor(n_trials=3, n_splits=3, gap_hours=24.0, seed=42)
    X, y = predictor.prepare_data_for_cv(df)

    assert y.sum() == 0, "fixture sanity: dataset must have 0 positives"

    with pytest.raises(RuntimeError, match="HPO degenerate"):
        predictor.optimize_and_train(X, y)

    # E nessun modello deve essere stato addestrato/assegnato.
    assert predictor.best_model is None, (
        "best_model should NOT be set when HPO is degenerate"
    )
