import pytest
import numpy as np
from src.evaluation.conformal import ConformalConfig, SplitConformal, EnbPI, create_conformal_predictor

def test_split_conformal():
    cp = SplitConformal(ConformalConfig(alpha=0.1))
    y_true = np.random.normal(0, 1, 200)
    y_pred = y_true + np.random.normal(0, 0.3, 200)
    cp.fit(y_true, y_pred)
    low, high = cp.predict_interval(y_pred)
    assert np.all(high >= low)
    assert np.mean((y_true >= low) & (y_true <= high)) >= 0.80

def test_enbpi_buffer():
    cfg = ConformalConfig(method="enbpi", sliding_window=50)
    cp = EnbPI(cfg)
    y = np.random.rand(100)
    cp.fit(y, y + 0.1)
    for _ in range(60): cp.update_residuals(np.random.rand(), np.random.rand())
    assert len(cp.residual_buffer) == 50
