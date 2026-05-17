import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

@dataclass
class ConformalConfig:
    alpha: float = 0.1
    method: str = "enbpi"
    agg_func: str = "mean"
    bootstrap_samples: int = 50
    sliding_window: int = 200
    adaptive: bool = False
    aci_gamma: float = 0.01
    aci_min_alpha: float = 0.01
    aci_max_alpha: float = 0.5

class ConformalPredictor(ABC):
    def __init__(self, config: ConformalConfig):
        self.config = config
        self.calibration_residuals: np.ndarray = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "ConformalPredictor": pass

    @abstractmethod
    def predict_interval(self, y_pred: np.ndarray, alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]: pass

    def _compute_quantile(self, residuals: np.ndarray, alpha: float) -> float:
        return float(np.quantile(np.abs(residuals), 1 - alpha))

class SplitConformal(ConformalPredictor):
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "SplitConformal":
        if len(y_true) != len(y_pred): raise ValueError("Mismatch dimensions")
        self.calibration_residuals = np.abs(y_true - y_pred)
        self.is_fitted = True
        return self
    def predict_interval(self, y_pred: np.ndarray, alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted: raise RuntimeError("Not fitted")
        q = self._compute_quantile(self.calibration_residuals, alpha or self.config.alpha)
        return y_pred - q, y_pred + q

class EnbPI(ConformalPredictor):
    def __init__(self, config: ConformalConfig):
        super().__init__(config)
        self.ensemble_predictions: List[np.ndarray] = []
        self.residual_buffer: List[float] = []
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> "EnbPI":
        if isinstance(y_pred, list) and len(y_pred) > 1:
            y_pred_agg = np.median(y_pred, axis=0) if self.config.agg_func == "median" else np.mean(y_pred, axis=0)
        else:
            y_pred_agg = y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred)
        self.ensemble_predictions = [y_pred_agg] * self.config.bootstrap_samples
        self.residual_buffer = np.abs(y_true - y_pred_agg).tolist()
        self.is_fitted = True
        return self
    def update_residuals(self, y_true_new: float, y_pred_new: float):
        self.residual_buffer.append(np.abs(y_true_new - y_pred_new))
        if len(self.residual_buffer) > self.config.sliding_window:
            self.residual_buffer.pop(0)
    def predict_interval(self, y_pred: Union[np.ndarray, List[np.ndarray]], alpha: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted: raise RuntimeError("Not fitted")
        q = self._compute_quantile(np.array(self.residual_buffer), alpha or self.config.alpha)
        pt = np.median(y_pred, axis=0) if isinstance(y_pred, list) and len(y_pred)>1 else y_pred
        return pt - q, pt + q

class AdaptiveConformalInference(EnbPI):
    def __init__(self, config): super().__init__(config); self.current_alpha = config.alpha; self.coverage_history = []
    def update_alpha(self, was_covered: bool):
        err = (1 - self.config.alpha) - int(was_covered)
        self.current_alpha = np.clip(self.current_alpha + self.config.aci_gamma * err, self.config.aci_min_alpha, self.config.aci_max_alpha)
    def predict_interval(self, y_pred, alpha=None): return super().predict_interval(y_pred, alpha or self.current_alpha)
    def post_predict_update(self, y_true, y_pred, interval):
        self.update_residuals(y_true, y_pred)
        if self.config.adaptive: self.update_alpha(interval[0] <= y_true <= interval[1])

def create_conformal_predictor(config: ConformalConfig) -> ConformalPredictor:
    methods = {"split": SplitConformal, "enbpi": EnbPI, "aci": AdaptiveConformalInference}
    if config.method not in methods: raise ValueError(f"Unknown method: {config.method}")
    if config.adaptive and config.method == "enbpi": return AdaptiveConformalInference(config)
    return methods[config.method](config)
