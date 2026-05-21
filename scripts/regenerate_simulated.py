"""Rigenera SOLO la fase 1 (data prep) su dati simulati.

Non chiama Fase 2 (BMM/ADVI) né Fase 3 (LightGBM/Optuna). Pensato per
ciclare velocemente sulla qualità del segnale del target senza pagare
ore di compute.

Uso::

    python scripts/regenerate_simulated.py

Output:
    data/processed/ais_enriched.parquet   (feature + target, senza prob bayesiane)
    models/simulation_ground_truth.json   (verità della simulazione)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import yaml

# Forza stdout UTF-8 (Windows cp1252 default rompe gli emoji nel log).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# run_pipeline.py importa moduli pesanti (pymc, bayesian_mixture) all'avvio.
# Per girare il SOLO simulator senza paghiamo quei costi, stubbiamo gli import
# pesanti prima di caricare run_pipeline come modulo.
import types  # noqa: E402
for stub_name in ("pymc", "arviz"):
    if stub_name not in sys.modules:
        sys.modules[stub_name] = types.ModuleType(stub_name)
# bayesian_mixture importa pymc al top-level: stub anche lui.
_bm_stub = types.ModuleType("src.bayesian_mixture")
for _attr in ("CausalBayesianMixture", "estimate_empirical_bayes_priors",
              "prior_sensitivity_analysis", "posterior_predictive_check"):
    setattr(_bm_stub, _attr, lambda *a, **k: None)
sys.modules.setdefault("src.bayesian_mixture", _bm_stub)

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("run_pipeline_mod", ROOT / "run_pipeline.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)  # carica CONFIG, _generate_simulation_study_dataset, ecc.

from src.data_prep import AISDataPreprocessor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("regenerate_simulated")


def main() -> int:
    with open(ROOT / "configs" / "pipeline_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sim = cfg.get("simulation", {})
    dp = cfg["data_prep"]
    paths = cfg["paths"]

    logger.info(
        f"Simulator: n_ships={sim.get('n_ships')} n_days={sim.get('n_days')} "
        f"dark_ratio={sim.get('dark_ratio')} seed={sim.get('random_seed')}"
    )

    mock_df, ground_truth = rp._generate_simulation_study_dataset(
        n_ships=int(sim["n_ships"]),
        n_days=int(sim["n_days"]),
        dark_ratio=float(sim["dark_ratio"]),
        bbox=dp["bounding_box"],
        seed=int(sim.get("random_seed", dp["seed"])),
    )

    import numpy as np
    assert mock_df["MMSI"].dtype == np.int64, (
        f"Simulator violated invariant: MMSI dtype is {mock_df['MMSI'].dtype}, expected int64"
    )
    logger.info(f"Simulator OK: {len(mock_df):,} ping, {mock_df['MMSI'].nunique()} navi, MMSI dtype={mock_df['MMSI'].dtype}")

    # Salva ground truth.
    gt_path = ROOT / paths["models_dir"] / "simulation_ground_truth.json"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2, default=str)
    logger.info(f"Ground truth salvato: {gt_path}")

    # Fase 1 manuale: filter → validate → downsample → features → labeling.
    pp = AISDataPreprocessor(
        bounding_box=dp["bounding_box"],
        gap_threshold_hours=float(dp["gap_threshold_hours"]),
        prediction_horizon_hours=float(dp["prediction_horizon_hours"]),
        downsample_minutes=int(dp["downsample_minutes"]),
        min_sog_knots=float(dp["min_sog"]),
        max_sog_knots=float(dp["max_sog"]),
        seed=int(dp["seed"]),
    )

    df_geo = pp.filter_geographic_area(mock_df)
    df_valid = pp.validate_physical_plausibility(df_geo)
    df_down = pp.downsample_causal(df_valid)
    assert df_down["MMSI"].dtype == np.int64, "downsample violated int64 invariant"
    df_feat = pp.engineer_causal_features(df_down)

    strategy = dp.get("labeling_strategy", "sliding_window")
    hmin = float(dp.get("labeling_horizon_minutes", 60.0))
    if strategy == "sliding_window":
        df_final = pp.label_sliding_window(df_feat, horizon_minutes=hmin)
    else:
        df_final = pp.label_last_ping(df_feat)
    logger.info(f"Labeling strategy: {strategy}")

    out_path = ROOT / paths["enriched_dataset"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(out_path, index=False)
    logger.info(f"Dataset salvato: {out_path} ({len(df_final):,} righe, "
                f"{int(df_final['target_dark_fleet'].sum())} positivi)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
