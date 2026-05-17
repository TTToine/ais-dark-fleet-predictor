#!/usr/bin/env python3
"""
🚀 AIS Dark Fleet Predictor - Conformal Prediction Auto-Installer
==================================================================
Questo script automatizza l'aggiunta del Conformal Prediction per
intervalli di incertezza distribution-free nel tuo progetto.
Usage:
python add_conformal.py [--dry-run] [--backup]
"""
import argparse
import shutil
import sys
import json
import logging
import textwrap
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURAZIONE
# ============================================================================
PROJECT_ROOT = Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "configs"
TESTS_DIR = PROJECT_ROOT / "tests"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
BACKUP_DIR = PROJECT_ROOT / ".conformal_backup"
DRY_RUN = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FILES_TO_CREATE = {
    SRC_DIR / "evaluation" / "conformal.py": "tmpl_conformal",
    SRC_DIR / "viz" / "uncertainty_plots.py": "tmpl_viz",
    TESTS_DIR / "test_conformal.py": "tmpl_test",
    NOTEBOOKS_DIR / "05_conformal_prediction_demo.ipynb": "tmpl_notebook",
    SRC_DIR / "models" / "conformal_wrapper.py": "tmpl_wrapper",
}

# ============================================================================
# TEMPLATE COMPATTI E CORRETTI
# ============================================================================
def tmpl_conformal():
    return textwrap.dedent('''\
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
    ''')

def tmpl_viz():
    return textwrap.dedent('''\
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px

    def create_interactive_map_with_uncertainty(df, lat_col="lat", lon_col="lon", prob_col="prob", lower_col="low", upper_col="high", title="Risk Map", zoom=3):
        df = df.copy()
        df["width"] = df[upper_col] - df[lower_col]
        df["size"] = df["width"] * 30 + 8
        fig = go.Figure(go.Scattermapbox(lat=df[lat_col], lon=df[lon_col], mode="markers",
            marker=dict(size=df["size"], color=df[prob_col], colorscale="RdYlGn_r", reversescale=True)))
        fig.update_layout(title=title, mapbox_style="open-street-map", height=600, margin=dict(l=0,r=0,t=50,b=0))
        return fig

    def generate_uncertainty_report(df, prob_col="prob", lower_col="low", upper_col="high", output_html="report.html"):
        map_fig = create_interactive_map_with_uncertainty(df, prob_col=prob_col, lower_col=lower_col, upper_col=upper_col)
        html = f"<html><body><h1>Uncertainty Report</h1>{map_fig.to_html(full_html=False)}</body></html>"
        with open(output_html, "w") as f: f.write(html)
        return html
    ''')

def tmpl_test():
    return textwrap.dedent('''\
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
    ''')

def tmpl_notebook():
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Conformal Prediction Demo\nQuantificazione incertezza distribution-free."]},
            {"cell_type": "code", "metadata": {}, "outputs": [], "source": [
                "from src.evaluation.conformal import ConformalConfig, create_conformal_predictor\n",
                "import numpy as np\n",
                "y_true = np.random.rand(100)\n",
                "y_pred = y_true + np.random.normal(0, 0.1, 100)\n",
                "cp = create_conformal_predictor(ConformalConfig(alpha=0.1, method='enbpi'))\n",
                "cp.fit(y_true, y_pred)\n",
                "low, high = cp.predict_interval(y_pred)\n",
                "print(f\"Coverage: {np.mean((y_true>=low) & (y_true<=high)):.2%}\")"
            ]}
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 4
    }
    return json.dumps(nb, indent=2)

def tmpl_wrapper():
    return textwrap.dedent('''\
    import numpy as np
    import lightgbm as lgb
    from src.evaluation.conformal import ConformalConfig, create_conformal_predictor

    class ConformalLightGBMClassifier:
        def __init__(self, conformal_alpha=0.1, conformal_method="enbpi", conformal_adaptive=True):
            self.alpha = conformal_alpha
            self.method = conformal_method
            self.adaptive = conformal_adaptive
            self.model_ = None
            self.cp_ = None
        def fit(self, X, y, X_cal=None, y_cal=None):
            self.model_ = lgb.LGBMClassifier(objective='binary')
            self.model_.fit(X, y)
            if X_cal is None: n = max(int(0.2*len(X)),50); X_cal, y_cal = X[-n:], y[-n:]
            y_cal_pred = self.model_.predict_proba(X_cal)[:,1]
            cfg = ConformalConfig(alpha=self.alpha, method=self.method, adaptive=self.adaptive)
            self.cp_ = create_conformal_predictor(cfg)
            self.cp_.fit(y_cal, y_cal_pred)
        def predict_proba_with_interval(self, X, alpha=None):
            probs = self.model_.predict_proba(X)[:,1]
            low, high = self.cp_.predict_interval(probs, alpha=alpha or self.alpha)
            return probs, np.clip(low,0,1), np.clip(high,0,1)
    ''')

# ============================================================================
# FUNZIONI DI SUPPORTO
# ============================================================================
def backup_file(filepath: Path):
    if not filepath.exists(): return
    BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"{filepath.name}.{ts}.bak"
    shutil.copy2(filepath, dest)
    logger.info(f"💾 Backup: {dest}")

def write_file(filepath: Path, content: str, dry_run: bool):
    if dry_run:
        logger.info(f"[DRY RUN] Would write {filepath}")
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    logger.info(f"✅ Created: {filepath.name}")

def append_to_file(filepath: Path, snippet: str, dry_run: bool):
    if dry_run:
        logger.info(f"[DRY RUN] Would append to {filepath}")
        return
    if not filepath.exists():
        logger.warning(f"⚠️ File not found: {filepath}")
        return
    content = filepath.read_text(encoding="utf-8")
    marker = "# [CONFORMAL_INSERT]"
    if marker in content:
        content = content.replace(marker, f"{marker}\n{snippet}")
    else:
        content += f"\n# --- Conformal Integration ---\n{snippet}\n"
    filepath.write_text(content, encoding="utf-8")
    logger.info(f"✅ Updated: {filepath.name}")

def update_pyproject(dry_run: bool):
    filepath = PROJECT_ROOT / "pyproject.toml"
    deps = ['mlconformal>=0.2.0', 'nonconformist>=2.1.0', 'plotly>=5.18.0']
    if not filepath.exists():
        minimal = "[project]\nname = \"ais-dark-fleet\"\nversion = \"0.1.0\"\nrequires-python = \">=3.10\"\n\n[project.dependencies]\n"
        for d in deps: minimal += f"{d}\n"
        write_file(filepath, minimal, dry_run)
        return
    if dry_run:
        logger.info(f"[DRY RUN] Would update {filepath}")
        return
    content = filepath.read_text(encoding="utf-8")
    modified = False
    for dep in deps:
        pkg = dep.split(">=")[0]
        if pkg not in content:
            if "[project.dependencies]" in content:
                content = content.replace("[project.dependencies]", f"[project.dependencies]\n{dep}  # Conformal")
                modified = True
            else:
                content += f"\n[project.dependencies]\n{dep}\n"
                modified = True
    if modified:
        filepath.write_text(content, encoding="utf-8")
        logger.info("✅ Updated pyproject.toml")

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Install Conformal Prediction")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()

    print("🚀 Starting Conformal Prediction Installation...")
    templates = {
        "tmpl_conformal": tmpl_conformal, "tmpl_viz": tmpl_viz,
        "tmpl_test": tmpl_test, "tmpl_notebook": tmpl_notebook, "tmpl_wrapper": tmpl_wrapper,
    }

    # 1. Crea file
    for filepath, tmpl_name in FILES_TO_CREATE.items():
        if args.backup and filepath.exists(): backup_file(filepath)
        write_file(filepath, templates[tmpl_name](), args.dry_run)

    # 2. Aggiorna pyproject.toml
    update_pyproject(args.dry_run)

    # 3. Aggiorna gb_training.py
    gb_path = SRC_DIR / "pipeline" / "gb_training.py"
    if gb_path.exists():
        if args.backup: backup_file(gb_path)
        append_to_file(gb_path, "from src.models.conformal_wrapper import ConformalLightGBMClassifier\n# Use for uncertainty-aware predictions", args.dry_run)

    # 4. Aggiorna config
    cfg_path = CONFIG_DIR / "pipeline_config.yaml"
    if cfg_path.exists():
        if args.backup: backup_file(cfg_path)
        append_to_file(cfg_path, "conformal:\n  enabled: true\n  alpha: 0.1\n  method: \"enbpi\"\n  adaptive: true", args.dry_run)
    else:
        write_file(cfg_path, "conformal:\n  enabled: true\n  alpha: 0.1\n  method: enbpi\n", args.dry_run)

    # 5. Aggiorna README
    readme = PROJECT_ROOT / "README.md"
    readme_snippet = "\n## 🎯 Conformal Prediction\nIntegrated uncertainty quantification using EnbPI/Split Conformal.\n"
    if readme.exists():
        if args.backup: backup_file(readme)
        append_to_file(readme, readme_snippet, args.dry_run)
    else:
        write_file(readme, f"# AIS Dark Fleet Predictor\n{readme_snippet}", args.dry_run)

    print("\n🎉 Installation Complete!")
    print("👉 Run: python src/pipeline/gb_training.py")

if __name__ == "__main__":
    main()