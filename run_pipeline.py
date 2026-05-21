"""
run_pipeline.py
Script di orchestrazione end-to-end per AIS Dark Fleet Predictor.
Orchestrazione: data_prep → bayesian_mixture → gb_training → contest_plots
"""
import sys
import os
import logging
import json
import traceback
import yaml
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, log_loss, brier_score_loss, roc_auc_score

# Aggiungi root progetto al PYTHONPATH
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data_prep import AISDataPreprocessor
    from src.bayesian_mixture import (
        CausalBayesianMixture,
        estimate_empirical_bayes_priors,
        prior_sensitivity_analysis,
        posterior_predictive_check,
    )
    from src.gb_training import DarkFleetPredictor, gap_threshold_sensitivity
except ImportError as e:
    print(f"❌ Import fallito. Verifica struttura cartelle e moduli patchati.")
    print(f"Dettaglio: {e}")
    sys.exit(1)

try:
    from src.viz.uncertainty_plots import plot_geographic_predictions
except ImportError:
    plot_geographic_predictions = None

try:
    from src.viz.contest_plots import (
        plot_simulation_validation,
        plot_brier_decomposition,
        plot_sensitivity_analysis,
        plot_bootstrap_ci_pr_auc,
        plot_pr_curve_with_thresholds,
        plot_executive_summary,
        plot_empirical_bayes_priors,
        plot_prior_sensitivity,
        plot_posterior_predictive_check,
    )
except ImportError:
    plot_simulation_validation = None
    plot_brier_decomposition = None
    plot_sensitivity_analysis = None
    plot_bootstrap_ci_pr_auc = None
    plot_pr_curve_with_thresholds = None
    plot_executive_summary = None
    plot_empirical_bayes_priors = None
    plot_prior_sensitivity = None
    plot_posterior_predictive_check = None


# ========================================================================
# CONFIGURAZIONE ROBUSTA
# ========================================================================
def load_config(config_path: str = "configs/pipeline_config.yaml") -> dict:
    """Carica config da YAML o fallback hardcodato SENZA spazi nelle chiavi."""
    DEFAULT_CONFIG = {
        "paths": {
            "raw_input": "data/raw/ais_sample.parquet",
            "processed_output": "data/processed/ais_v1.parquet",
            "models_dir": "models",
            "metrics_file": "models/metrics_final.json",
            "enriched_dataset": "data/processed/ais_enriched.parquet"
        },
        "data_prep": {
            "bounding_box": {"min_lat": 35.0, "max_lat": 38.0, "min_lon": 11.0, "max_lon": 15.5},
            "gap_threshold_hours": 12.0,
            "prediction_horizon_hours": 24.0,
            "downsample_minutes": 10,
            "min_sog": 0.0,
            "max_sog": 50.0,
            "seed": 42,
            "labeling_strategy": "sliding_window",
            "labeling_horizon_minutes": 60.0
        },
        "hmm": {
            "window_size": 36,
            "use_advi": True,
            "sigma_prior_speed": 1.0,
            "sigma_prior_turn": 2.0,
            "use_empirical_bayes": True,
            "eb_n_pilot_vessels": 5,
            "eb_n_advi_steps": 500,
            "run_prior_sensitivity": True,
            "ps_n_vessels": 10,
            "ps_n_advi_steps": 300,
            "run_ppc": True,
            "ppc_n_vessels": 3,
            "ppc_n_simulations": 200,
            "ppc_n_advi_steps": 500,
            "n_jobs": -1
        },
        "gb": {
            "n_trials": 30,
            "n_splits": 5,
            "gap_hours": 24.0,
            "freq_min": 10,
            "seed": 42
        },
        "pipeline": {
            "train_split_ratio": 0.8,
            "run_ablation_study": False,
            "save_enriched_data": True,
            "run_post_training_analysis": True
        },
        "simulation": {
            "n_ships": 60,
            "n_days": 7,
            "dark_ratio": 0.20
        }
    }

    def _deep_merge(base: dict, override: dict) -> dict:
        """Merge override into base recursively; base keys not in override are kept."""
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = _deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    try:
        cfg_path = Path(config_path)
        if cfg_path.exists():
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            if cfg and isinstance(cfg, dict):
                return _deep_merge(DEFAULT_CONFIG, cfg)
    except Exception as exc:
        logging.warning(f"Impossibile caricare {config_path}: {exc}. Uso DEFAULT_CONFIG.")

    return DEFAULT_CONFIG

CONFIG = load_config()

# ========================================================================
# SETUP & UTILS
# ========================================================================
def setup_logging_and_dirs():
    """Inizializza directory di output e logging."""
    dirs_to_create = [
        Path(CONFIG["paths"]["processed_output"]).parent,
        Path(CONFIG["paths"]["models_dir"]),
        Path("logs")
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        
    log_file = Path("logs") / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"📁 Directory e logging configurati. Log: {log_file}")


def save_artifacts(predictor: DarkFleetPredictor, metrics: dict, df_enriched: pd.DataFrame):
    """Salva modello, metriche e dataset arricchito in produzione."""
    models_dir = Path(CONFIG["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Salva modello LightGBM
    if predictor.best_model is not None:
        model_path = models_dir / "lgb_dark_fleet.pkl"
        joblib.dump(predictor.best_model, str(model_path))
        logging.info(f"💾 Modello LightGBM salvato in {model_path}")
        
        params_path = models_dir / "best_params.json"
        with open(params_path, 'w') as f:
            json.dump(predictor.best_params or {}, f, indent=2)
        logging.info(f"💾 Hyperparametri salvati in {params_path}")

    # Salva metriche
    metrics_path = Path(CONFIG["paths"]["metrics_file"])
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["config_summary"] = {k: v for k, v in CONFIG.items() if k != "paths"}
    
    clean_metrics = {k: (float(v) if hasattr(v, 'item') else v) for k, v in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(clean_metrics, f, indent=2)
    logging.info(f"📊 Metriche salvate in {metrics_path}")

    # Salva dataset arricchito
    if CONFIG["pipeline"]["save_enriched_data"]:
        enriched_path = Path(CONFIG["paths"]["enriched_dataset"])
        df_enriched.to_parquet(enriched_path, index=False)
        logging.info(f"💾 Dataset arricchito salvato in {enriched_path}")


# ========================================================================
# FASE 1: PRE-PROCESSING
# ========================================================================
def _generate_simulation_study_dataset(n_ships: int = 60,
                                       n_days: int = 7,
                                       dark_ratio: float = 0.20,
                                       bbox: dict = None,
                                       seed: int = 42) -> tuple:
    """
    Genera un dataset di simulazione realistico per validazione metodologica.

    Design:
        - N navi: 'dark' (con blackout intenzionali) vs 'normal' (gap rari/accidentali)
        - Traiettorie: random walk con drift verso direzione di rotta, vincolato a bbox
        - Velocità: dipendente dal tipo di nave (cargo veloce, pescherecci lenti)
        - Blackout 'dark': 8-24h di durata, frequenza 1-3 per nave/settimana
        - Blackout 'normal': 0-3h, rari, simulano guasti

    Ritorna (df, ground_truth_dict) dove ground_truth contiene il vero tipo per nave —
    utile per valutare il recovery del segnale latente (regime detection) e
    misurare la sensitività/specificità a livello di nave (non solo di punto).
    """
    rng = np.random.default_rng(seed)

    if bbox is None:
        bbox = {"min_lat": 35.0, "max_lat": 38.0, "min_lon": 11.0, "max_lon": 15.5}

    n_dark = int(round(n_ships * dark_ratio))
    n_normal = n_ships - n_dark
    is_dark_flags = [True] * n_dark + [False] * n_normal
    rng.shuffle(is_dark_flags)

    base_time = datetime(2024, 6, 1)
    total_minutes = n_days * 24 * 60
    sample_dt_min = 10  # campionamento medio (con jitter)

    mock_records = []
    ground_truth = {"ships": []}

    # Range MMSI realistico (MID 200-799 copre Europa, Africa, Asia maggiore).
    # int Python ufficiale → resta int64 nel DataFrame, niente promozioni a float.
    sim_mmsi_pool = rng.choice(
        np.arange(200_000_000, 800_000_000, dtype=np.int64),
        size=n_ships, replace=False,
    )

    for ship_idx in range(n_ships):
        mmsi = int(sim_mmsi_pool[ship_idx])  # MMSI sintetici distinti (int Python → int64)
        is_dark = is_dark_flags[ship_idx]

        # Profilo nave
        if rng.random() < 0.3:
            ship_type, base_sog, sog_noise = "fishing", rng.uniform(3, 7), 0.8
        elif rng.random() < 0.7:
            ship_type, base_sog, sog_noise = "cargo", rng.uniform(10, 16), 1.2
        else:
            ship_type, base_sog, sog_noise = "tanker", rng.uniform(8, 13), 1.0

        # Posizione iniziale random nel bbox
        lat = rng.uniform(bbox["min_lat"] + 0.2, bbox["max_lat"] - 0.2)
        lon = rng.uniform(bbox["min_lon"] + 0.2, bbox["max_lon"] - 0.2)
        heading = rng.uniform(0, 360)
        current_time = base_time

        # Programma blackout per questa nave
        blackouts = []
        if is_dark:
            n_blackouts = rng.integers(1, 4)
            for _ in range(n_blackouts):
                start_offset_min = rng.integers(60, total_minutes - 24 * 60)
                duration_h = rng.uniform(8, 24)
                blackouts.append((start_offset_min, duration_h))
        else:
            # ~30% delle navi normali hanno UN piccolo gap accidentale
            if rng.random() < 0.3:
                start_offset_min = rng.integers(60, total_minutes - 4 * 60)
                duration_h = rng.uniform(0.5, 3.0)
                blackouts.append((start_offset_min, duration_h))

        blackouts.sort()
        ground_truth["ships"].append({
            "mmsi": int(mmsi), "is_dark": bool(is_dark),
            "type": ship_type, "n_blackouts": len(blackouts),
            "blackout_durations_h": [round(b[1], 2) for b in blackouts]
        })

        # Simula traiettoria
        elapsed_min = 0
        bo_idx = 0
        while elapsed_min < total_minutes:
            # Verifica se siamo dentro un blackout (gap)
            in_blackout = False
            if bo_idx < len(blackouts):
                bo_start, bo_dur_h = blackouts[bo_idx]
                bo_end = bo_start + bo_dur_h * 60
                if bo_start <= elapsed_min < bo_end:
                    # Avanza temporalmente fino a fine blackout, senza emettere ping
                    current_time += timedelta(minutes=bo_end - elapsed_min)
                    elapsed_min = bo_end
                    bo_idx += 1
                    in_blackout = True

            if in_blackout:
                continue

            # Comportamento differente durante "pre-blackout" per le dark ships
            # (rallentano, cambiano rotta → segnale che il regime detector dovrebbe catturare)
            near_blackout = False
            if is_dark and bo_idx < len(blackouts):
                next_bo_start = blackouts[bo_idx][0]
                if 0 <= (next_bo_start - elapsed_min) < 120:  # 2h prima
                    near_blackout = True

            if near_blackout:
                sog_effective = max(0, base_sog * 0.4 + rng.normal(0, sog_noise))
                heading_drift = rng.normal(0, 8)  # rotta più erratica
            else:
                sog_effective = max(0, base_sog + rng.normal(0, sog_noise))
                heading_drift = rng.normal(0, 2)

            heading = (heading + heading_drift) % 360

            # Avanzamento spaziale ~ sog * dt (1 nodo ≈ 1.852 km/h, lat 1° ≈ 111 km)
            dt_h = sample_dt_min / 60.0
            dist_deg = (sog_effective * 1.852 * dt_h) / 111.0
            lat_new = lat + dist_deg * np.cos(np.radians(heading))
            lon_new = lon + dist_deg * np.sin(np.radians(heading)) / np.cos(np.radians(lat))

            # Rimbalza se esce dal bbox
            if not (bbox["min_lat"] < lat_new < bbox["max_lat"]):
                heading = (180 - heading) % 360
                lat_new = lat
            if not (bbox["min_lon"] < lon_new < bbox["max_lon"]):
                heading = (360 - heading) % 360
                lon_new = lon

            lat, lon = lat_new, lon_new

            mock_records.append({
                "MMSI": mmsi, "Timestamp": current_time,
                "Lat": lat, "Lon": lon, "SOG": sog_effective, "COG": heading
            })

            jitter_min = rng.normal(sample_dt_min, 1.0)
            current_time += timedelta(minutes=max(1, jitter_min))
            elapsed_min += sample_dt_min

    df = pd.DataFrame(mock_records).sort_values(["MMSI", "Timestamp"]).reset_index(drop=True)
    # Garanzia di tipo all'uscita del simulator: MMSI deve essere int64.
    from src.data_prep import _ensure_mmsi_int64
    df = _ensure_mmsi_int64(df, context="simulator")
    assert df["MMSI"].dtype == np.int64, "simulator: MMSI not int64 on exit"
    logging.info(
        f"📊 Simulation study: {n_ships} navi ({n_dark} dark, {n_normal} normal), "
        f"{len(df)} ping su {n_days} giorni"
    )
    ground_truth["meta"] = {
        "n_ships": n_ships, "n_dark": n_dark, "n_days": n_days,
        "dark_ratio": dark_ratio, "seed": seed, "total_pings": len(df),
    }
    return df, ground_truth


def run_phase1_preprocessing() -> pd.DataFrame:
    logging.info("\n" + "="*60)
    logging.info("FASE 1: Data Preprocessing & Target Labeling")
    logging.info("="*60)
    
    cfg = CONFIG["data_prep"]
    preprocessor = AISDataPreprocessor(
        bounding_box=cfg["bounding_box"],
        gap_threshold_hours=cfg["gap_threshold_hours"],
        prediction_horizon_hours=cfg["prediction_horizon_hours"],
        downsample_minutes=cfg["downsample_minutes"],
        min_sog_knots=cfg["min_sog"],
        max_sog_knots=cfg["max_sog"],
        seed=cfg["seed"]
    )

    raw_path = Path(CONFIG["paths"]["raw_input"])

    if not raw_path.exists():
        logging.warning("⚠️ File raw non trovato. Generazione SIMULATION STUDY DATASET...")
        mock_df, _ground_truth = _generate_simulation_study_dataset(
            n_ships=CONFIG.get("simulation", {}).get("n_ships", 60),
            n_days=CONFIG.get("simulation", {}).get("n_days", 7),
            dark_ratio=CONFIG.get("simulation", {}).get("dark_ratio", 0.20),
            bbox=cfg["bounding_box"],
            seed=CONFIG.get("simulation", {}).get("random_seed", cfg["seed"]),
        )
        # Salva ground truth per validazione metodologica successiva
        gt_path = Path(CONFIG["paths"]["models_dir"]) / "simulation_ground_truth.json"
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_path, "w") as f:
            json.dump(_ground_truth, f, indent=2)
        logging.info(f"📦 Ground truth simulazione salvato in {gt_path}")

        # Pipeline manuale sul mock
        df_geo = preprocessor.filter_geographic_area(mock_df)
        df_valid = preprocessor.validate_physical_plausibility(df_geo)
        df_down = preprocessor.downsample_causal(df_valid)
        df_feat = preprocessor.engineer_causal_features(df_down)
        # Etichettatura: dispatch su labeling_strategy del config.
        _strategy = CONFIG["data_prep"].get("labeling_strategy", "sliding_window")
        _hmin = float(CONFIG["data_prep"].get("labeling_horizon_minutes", 60.0))
        if _strategy == "sliding_window":
            df_processed = preprocessor.label_sliding_window(df_feat, horizon_minutes=_hmin)
        elif _strategy == "last_ping":
            df_processed = preprocessor.label_last_ping(df_feat)
        else:
            raise ValueError(f"labeling_strategy='{_strategy}' non valido")
        logging.info(
            f"   Labeling strategy: {_strategy}"
            + (f" (horizon={_hmin} min)" if _strategy == "sliding_window" else "")
        )
    else:
        df_processed = preprocessor.run_pipeline(
            input_path=CONFIG["paths"]["raw_input"],
            output_path=CONFIG["paths"]["processed_output"],
            return_df=True
        )
        
    logging.info(f"✅ Fase 1 completata. Righe: {len(df_processed)}, Target rate: {df_processed['target_dark_fleet'].mean():.3%}")
    return df_processed


# ========================================================================
# FASE 2: ESTRAZIONE FEATURE BAYESIANE (MIXTURE + MARKOV FILTER)
# ========================================================================
def run_phase2_hmm_enrichment(df_processed: pd.DataFrame) -> pd.DataFrame:
    logging.info("\n" + "="*60)
    logging.info("FASE 2: Estrazione Regimi Latenti (Bayesian Mixture + Markov Smoothing)")
    logging.info("="*60)
    
    cfg = CONFIG["hmm"]

    # === STAGE 1: Empirical Bayes hyperprior estimation ===
    sigma_speed = cfg["sigma_prior_speed"]
    sigma_turn  = cfg["sigma_prior_turn"]
    if cfg.get("use_empirical_bayes", False):
        try:
            eb_path = Path(CONFIG["paths"]["models_dir"]) / "empirical_bayes_priors.json"
            eb_result = estimate_empirical_bayes_priors(
                df_processed,
                n_pilot_vessels=cfg.get("eb_n_pilot_vessels", 5),
                n_advi_steps=cfg.get("eb_n_advi_steps", 500),
                seed=CONFIG["data_prep"]["seed"],
                save_path=str(eb_path),
            )
            sigma_speed = eb_result["sigma_prior_speed"]
            sigma_turn  = eb_result["sigma_prior_turn"]
            logging.info(
                f"🧬 Empirical Bayes attivato: "
                f"σ_speed {cfg['sigma_prior_speed']} → {sigma_speed:.3f}, "
                f"σ_turn {cfg['sigma_prior_turn']} → {sigma_turn:.3f}"
            )
        except Exception as e:
            logging.warning(f"Empirical Bayes Stage 1 fallito ({e}). Uso prior di default.")

    # === PRIOR SENSITIVITY ANALYSIS (Gelman BDA3 §6) ===
    if cfg.get("run_prior_sensitivity", False):
        try:
            ps_path = Path(CONFIG["paths"]["models_dir"]) / "prior_sensitivity.json"
            prior_sensitivity_analysis(
                df_processed,
                n_vessels=cfg.get("ps_n_vessels", 10),
                n_advi_steps=cfg.get("ps_n_advi_steps", 300),
                seed=CONFIG["data_prep"]["seed"],
                save_path=str(ps_path),
            )
        except Exception as e:
            logging.warning(f"Prior sensitivity fallita ({e}). Procedo senza.")

    # === POSTERIOR PREDICTIVE CHECK (Gelman BDA3 §6.3) ===
    if cfg.get("run_ppc", False):
        try:
            ppc_path = Path(CONFIG["paths"]["models_dir"]) / "posterior_predictive_check.json"
            posterior_predictive_check(
                df_processed,
                n_vessels=cfg.get("ppc_n_vessels", 3),
                n_simulations=cfg.get("ppc_n_simulations", 200),
                n_advi_steps=cfg.get("ppc_n_advi_steps", 500),
                sigma_prior_speed=sigma_speed,  # usa prior empirici se EB attivo
                sigma_prior_turn=sigma_turn,
                seed=CONFIG["data_prep"]["seed"],
                save_path=str(ppc_path),
            )
        except Exception as e:
            logging.warning(f"Posterior Predictive Check fallito ({e}). Procedo senza.")

    extractor = CausalBayesianMixture(
        window_size=cfg["window_size"],
        sigma_prior_speed=sigma_speed,
        sigma_prior_turn=sigma_turn,
    )

    n_ships = df_processed["MMSI"].nunique()
    logging.info(f"🚢 Processing di {n_ships} navi (iterazione interna a process_dataframe_causal)...")

    df_hmm = extractor.process_dataframe_causal(
        df_processed,
        use_advi=cfg["use_advi"],
        apply_markov=True,
        update_freq=cfg.get("update_freq", 10),
        max_advi_calls=cfg.get("max_advi_calls", None),
        adaptive_threshold=cfg.get("adaptive_threshold", 1.0),
        n_jobs=cfg.get("n_jobs", -1),
    )

    if df_hmm is None or len(df_hmm) == 0:
        raise RuntimeError("Nessuna nave processata correttamente nella Fase 2.")

    logging.info(f"✅ Fase 2 completata. Feature bayesiane estratte per {len(df_hmm)} osservazioni.")
    return df_hmm


# ========================================================================
# FASE 3: TRAINING & VALUTAZIONE GB
# ========================================================================
def run_phase3_gb_training(df_hmm: pd.DataFrame) -> tuple:
    """Ritorna (predictor, metrics) per permettere il salvataggio corretto."""
    logging.info("\n" + "="*60)
    logging.info("FASE 3: Training Gradient Boosting & Valutazione")
    logging.info("="*60)
    
    cfg = CONFIG["gb"]
    predictor = DarkFleetPredictor(
        n_trials=cfg["n_trials"], n_splits=cfg["n_splits"],
        gap_hours=cfg["gap_hours"], freq_min=cfg["freq_min"], seed=cfg["seed"]
    )

    split_idx = int(len(df_hmm) * CONFIG["pipeline"]["train_split_ratio"])
    train_df = df_hmm.iloc[:split_idx].copy()
    test_df = df_hmm.iloc[split_idx:].copy()
    logging.info(f"📅 Split: {len(train_df)} train, {len(test_df)} test")

    X_train, y_train = predictor.prepare_data_for_cv(train_df)
    X_test, y_test = predictor.prepare_data_for_cv(test_df)

    if y_test.nunique() < 2:
        logging.warning("⚠️ Test set contiene una sola classe. ROC AUC non calcolabile.")
        
    predictor.optimize_and_train(X_train, y_train)
    # predict() applica il filtro Markov con p_stay ottimo e droppa MMSI/Timestamp
    preds_proba = predictor.predict(X_test)

    metrics = {
        'pr_auc': average_precision_score(y_test, preds_proba),
        'log_loss': log_loss(y_test, preds_proba),
        'brier_score': brier_score_loss(y_test, preds_proba)
    }
    try:
        metrics['roc_auc'] = roc_auc_score(y_test, preds_proba)
    except ValueError:
        metrics['roc_auc'] = np.nan
        logging.warning("ROC AUC non definito (single class in test set)")

    logging.info("\n" + "= "*60)
    logging.info("📊 RISULTATI FINALI SUL TEST SET (FUTURO)")
    logging.info("= "*60)
    for name, value in metrics.items():
        logging.info(f"{name:15s}: {value:.4f}")
    logging.info("= "*60 + "\n")

    try:
        predictor.plot_calibration(X_test, y_test, n_bins=10)
        predictor.plot_feature_importance_shap(X_test, max_display=10)
    except Exception as e:
        logging.warning(f"⚠️ Plot generation failed: {e}")
        
    return predictor, metrics


# ========================================================================
# MAPPA GEOGRAFICA STANDALONE (no folium, solo matplotlib)
# ========================================================================
def _plot_geographic_map_standalone(df: pd.DataFrame,
                                     save_path: str = "models/06_geographic_map.png",
                                     ground_truth_path: Optional[str] = None,
                                     prob_col: str = "prob_regime_sospetto",
                                     lat_col: str = "Lat",
                                     lon_col: str = "Lon",
                                     mmsi_col: str = "MMSI") -> str:
    """
    Mappa geografica autonoma con matplotlib (zero dipendenze esterne).
    Mostra l'ultima posizione di ogni nave colorata per P(regime sospetto)
    con dimensione proporzionale alla prob, e i centroidi di traiettoria.

    Se disponibile, sovrappone i marker delle vere navi 'dark' dal ground truth.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    needed = {mmsi_col, lat_col, lon_col, prob_col}
    if not needed.issubset(df.columns):
        logging.warning(f"Mappa geografica: colonne mancanti {needed - set(df.columns)}, skip.")
        return ""

    # Ultima posizione per nave
    if "Timestamp" in df.columns:
        last_pos = (df.sort_values("Timestamp")
                      .groupby(mmsi_col, as_index=False).last())
    else:
        last_pos = df.copy()
    last_pos = last_pos.dropna(subset=[lat_col, lon_col, prob_col])

    if len(last_pos) == 0:
        logging.warning("Mappa geografica: nessuna posizione valida.")
        return ""

    # Traiettoria completa per nave (lineplot leggero)
    trajectories = df.dropna(subset=[lat_col, lon_col])

    # Ground truth (opzionale): quali navi sono dark per davvero
    dark_mmsi = set()
    if ground_truth_path and Path(ground_truth_path).exists():
        try:
            with open(ground_truth_path) as f:
                gt = json.load(f)
            dark_mmsi = {s["mmsi"] for s in gt.get("ships", []) if s.get("is_dark")}
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(11, 8))

    # 1) Traiettorie leggere in grigio per orientarsi
    for mmsi, g in trajectories.groupby(mmsi_col):
        g = g.sort_values("Timestamp") if "Timestamp" in g.columns else g
        ax.plot(g[lon_col], g[lat_col], color="#BDC3C7", lw=0.6, alpha=0.55, zorder=1)

    # 2) Ultima posizione: scatter colorato per probabilità
    scatter = ax.scatter(
        last_pos[lon_col], last_pos[lat_col],
        c=last_pos[prob_col],
        cmap="RdYlGn_r",
        vmin=0.0, vmax=1.0,
        s=80 + last_pos[prob_col] * 240,
        alpha=0.92,
        edgecolors="black", linewidths=0.7,
        zorder=3,
    )

    # 3) Cerchio rosso su navi 'dark' note (ground truth)
    if dark_mmsi:
        dark_pos = last_pos[last_pos[mmsi_col].isin(dark_mmsi)]
        if len(dark_pos) > 0:
            ax.scatter(
                dark_pos[lon_col], dark_pos[lat_col],
                s=420, facecolors="none", edgecolors="#C0392B",
                linewidths=2.2, zorder=4,
                label=f"Ground truth: dark fleet (n={len(dark_pos)})",
            )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r'$P(\mathrm{regime\ sospetto})$', fontsize=11)

    ax.set_xlabel("Longitudine", fontsize=11)
    ax.set_ylabel("Latitudine", fontsize=11)
    ax.set_title(
        f"Mappa flotta — ultime posizioni note di {len(last_pos)} navi "
        f"(traiettorie grigie, scatter = P regime sospetto)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(alpha=0.3)

    # Legenda compatta
    legend_handles = [
        Line2D([0], [0], color="#BDC3C7", lw=2, label="Traiettoria"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ECC71",
               markeredgecolor="black", markersize=10, label="P sospetto basso"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C",
               markeredgecolor="black", markersize=14, label="P sospetto alto"),
    ]
    if dark_mmsi:
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="none", markeredgecolor="#C0392B",
                   markersize=14, markeredgewidth=2.0,
                   label="Ground truth: dark"),
        )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.92)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"📊 Mappa geografica autonoma salvata: {save_path}")
    return save_path


# ========================================================================
# POST-TRAINING ANALYSIS
# ========================================================================
def _run_post_training_analysis(predictor: "DarkFleetPredictor",
                                 df_hmm: pd.DataFrame,
                                 final_metrics: dict = None):
    """Produce tutti gli artefatti di analisi post-training + grafici da presentazione.
    Controllato da pipeline.run_post_training_analysis nel YAML."""
    final_metrics = final_metrics or {}
    split_idx = int(len(df_hmm) * CONFIG["pipeline"]["train_split_ratio"])
    train_df = df_hmm.iloc[:split_idx].copy()
    test_df  = df_hmm.iloc[split_idx:].copy()

    X_test, y_test = predictor.prepare_data_for_cv(test_df)
    mmsi_test  = X_test['MMSI'].values if 'MMSI' in X_test.columns else None
    y_prob_test = predictor.predict(X_test)

    # Track soglie operative per executive summary
    f2_thr = None
    conformal_thr = None
    f2_optimal_val = None
    conformal_fpr_val = None

    # F2-ottimale threshold + confusion matrix (la funzione gestisce internamente il preprocess)
    try:
        f2_result = predictor.find_optimal_threshold(X_test, y_test)
        if isinstance(f2_result, dict):
            f2_thr = f2_result.get('optimal_threshold')
            f2_optimal_val = f2_result.get('f2.0_score') or f2_result.get('f2_score')
        logging.info("✅ F2 threshold plot salvato")
    except Exception as e:
        logging.warning(f"find_optimal_threshold fallito: {e}")

    # Conformal prediction threshold (FPR garantito)
    try:
        mid = len(X_test) // 2
        X_calib, y_calib = X_test.iloc[:mid], y_test.iloc[:mid]
        X_eval,  y_eval  = X_test.iloc[mid:], y_test.iloc[mid:]
        conformal_res = predictor.evaluate_with_conformal(
            X_calib, y_calib, X_eval, y_eval, target_fpr=0.1
        )
        if isinstance(conformal_res, dict):
            conformal_thr = conformal_res.get('conformal_threshold')
            conformal_fpr_val = conformal_res.get('empirical_fpr')
        logging.info("✅ Conformal threshold calcolato")
    except Exception as e:
        logging.warning(f"evaluate_with_conformal fallito: {e}")

    # Calibration comparison plot
    try:
        predictor.plot_calibration_comparison(X_test, y_test,
                                              save_path="models/calibration_comparison.png")
        logging.info("✅ Calibration comparison salvato in models/calibration_comparison.png")
    except Exception as e:
        logging.warning(f"plot_calibration_comparison fallito: {e}")

    # SHAP interaction plot (prime due feature per default — usa il nome post-Markov se esiste)
    try:
        # Se il filtro Markov è attivo, la feature è 'prob_regime_sospetto_markov'
        feature_names = predictor.best_model.feature_name()
        prob_feat = 'prob_regime_sospetto_markov' if 'prob_regime_sospetto_markov' in feature_names else 'prob_regime_sospetto'
        speed_feat = 'speed_acc' if 'speed_acc' in feature_names else feature_names[0]
        predictor.plot_shap_interaction(X_test, feature=prob_feat, interaction_index=speed_feat)
        logging.info(f"✅ SHAP interaction plot ({prob_feat} × {speed_feat}) salvato")
    except Exception as e:
        logging.warning(f"plot_shap_interaction fallito: {e}")

    # Gap threshold sensitivity analysis (signature aggiornata: full df, split interno)
    sensitivity_results = None
    try:
        sensitivity_results = gap_threshold_sensitivity(
            df_features=df_hmm,
            gap_thresholds_hours=[6.0, 12.0, 18.0, 24.0],
            horizon_hours=CONFIG["data_prep"].get("prediction_horizon_hours", 24.0),
            n_trials_sensitivity=10, n_splits=3,
            seed=CONFIG["gb"]["seed"]
        )
        sens_path = "models/gap_sensitivity.json"
        with open(sens_path, "w") as f:
            json.dump({str(k): v for k, v in sensitivity_results.items()}, f, indent=2)
        logging.info(f"✅ Sensitivity analysis salvata in {sens_path}")
    except Exception as e:
        logging.warning(f"gap_threshold_sensitivity fallito: {e}")

    # ============================================================
    # GRAFICI DA PRESENTAZIONE (contest_plots)
    # ============================================================
    logging.info("\n🎨 Generazione grafici per presentazione...")

    # 0a. Empirical Bayes hyperpriors (Stage 1 visualization)
    eb_path = Path(CONFIG["paths"]["models_dir"]) / "empirical_bayes_priors.json"
    if plot_empirical_bayes_priors is not None and eb_path.exists():
        try:
            plot_empirical_bayes_priors(
                str(eb_path),
                default_priors={
                    "sigma_prior_speed": CONFIG["hmm"]["sigma_prior_speed"],
                    "sigma_prior_turn":  CONFIG["hmm"]["sigma_prior_turn"],
                },
                save_path="models/00b_empirical_bayes_priors.png"
            )
        except Exception as e:
            logging.warning(f"plot_empirical_bayes_priors fallito: {e}")

    # 0b. Prior sensitivity analysis visualization
    ps_path = Path(CONFIG["paths"]["models_dir"]) / "prior_sensitivity.json"
    if plot_prior_sensitivity is not None and ps_path.exists():
        try:
            plot_prior_sensitivity(
                str(ps_path),
                save_path="models/00c_prior_sensitivity.png"
            )
        except Exception as e:
            logging.warning(f"plot_prior_sensitivity fallito: {e}")

    # 0d. Posterior Predictive Check visualization
    ppc_path = Path(CONFIG["paths"]["models_dir"]) / "posterior_predictive_check.json"
    if plot_posterior_predictive_check is not None and ppc_path.exists():
        try:
            plot_posterior_predictive_check(
                str(ppc_path),
                save_path="models/00d_posterior_predictive_check.png"
            )
        except Exception as e:
            logging.warning(f"plot_posterior_predictive_check fallito: {e}")

    # 1. Validazione su simulation study (ground truth conosciuta)
    gt_path = Path(CONFIG["paths"]["models_dir"]) / "simulation_ground_truth.json"
    if plot_simulation_validation is not None and gt_path.exists():
        try:
            plot_simulation_validation(df_hmm, str(gt_path),
                                       save_path="models/01_simulation_validation.png")
        except Exception as e:
            logging.warning(f"plot_simulation_validation fallito: {e}")

    # 2. Decomposizione di Murphy del Brier score
    if plot_brier_decomposition is not None:
        try:
            plot_brier_decomposition(y_test.values, y_prob_test,
                                     save_path="models/02_brier_decomposition.png")
        except Exception as e:
            logging.warning(f"plot_brier_decomposition fallito: {e}")

    # 3. Sensitivity analysis (line plot)
    if plot_sensitivity_analysis is not None and sensitivity_results:
        try:
            plot_sensitivity_analysis(sensitivity_results,
                                      save_path="models/03_sensitivity_analysis.png")
        except Exception as e:
            logging.warning(f"plot_sensitivity_analysis fallito: {e}")

    # 4. Bootstrap CI clustered per MMSI sul PR-AUC
    if plot_bootstrap_ci_pr_auc is not None and mmsi_test is not None:
        try:
            plot_bootstrap_ci_pr_auc(y_test.values, y_prob_test, mmsi_test,
                                     n_boot=1000, alpha=0.05,
                                     seed=CONFIG["gb"]["seed"],
                                     save_path="models/04_bootstrap_ci_pr_auc.png")
        except Exception as e:
            logging.warning(f"plot_bootstrap_ci_pr_auc fallito: {e}")

    # 5. PR curve con soglie operative (F2 + Conformal)
    if plot_pr_curve_with_thresholds is not None:
        try:
            plot_pr_curve_with_thresholds(
                y_test.values, y_prob_test,
                f2_threshold=f2_thr, conformal_threshold=conformal_thr,
                save_path="models/05_pr_curve_with_thresholds.png"
            )
        except Exception as e:
            logging.warning(f"plot_pr_curve_with_thresholds fallito: {e}")

    # 6. Executive summary (slide riassuntiva con KPI)
    if plot_executive_summary is not None:
        try:
            summary_metrics = dict(final_metrics)
            summary_metrics["f2_optimal"] = f2_optimal_val
            summary_metrics["conformal_fpr"] = conformal_fpr_val
            plot_executive_summary(summary_metrics,
                                   save_path="models/00_executive_summary.png")
        except Exception as e:
            logging.warning(f"plot_executive_summary fallito: {e}")

    # Geographic map — versione autonoma matplotlib (garantita, niente folium)
    try:
        _plot_geographic_map_standalone(
            df_hmm,
            save_path="models/06_geographic_map.png",
            ground_truth_path=str(Path(CONFIG["paths"]["models_dir"]) / "simulation_ground_truth.json"),
        )
    except Exception as e:
        logging.warning(f"Mappa geografica autonoma fallita: {e}")

    # Versione interattiva folium (best effort, opzionale)
    if plot_geographic_predictions is not None:
        try:
            if 'prob_regime_sospetto' in df_hmm.columns:
                plot_geographic_predictions(
                    df_hmm,
                    save_path="models/geographic_predictions.html"
                )
        except Exception as e:
            logging.debug(f"plot_geographic_predictions (folium) fallito: {e}")


# ========================================================================
# MAIN ORCHESTRATOR
# ========================================================================
def _parse_cli_args():
    import argparse
    p = argparse.ArgumentParser(description="AIS Dark Fleet Predictor pipeline")
    p.add_argument(
        "--labeling",
        choices=["sliding_window", "last_ping"],
        default=None,
        help=(
            "Override del labeling_strategy del YAML. "
            "'sliding_window' è deployable (default), "
            "'last_ping' è retrospettiva (audit only)."
        ),
    )
    return p.parse_args()


def main():
    args = _parse_cli_args()
    setup_logging_and_dirs()
    logging.info("🚀 AVVIO PIPELINE AIS DARK FLEET PREDICTOR")
    if args.labeling is not None:
        CONFIG["data_prep"]["labeling_strategy"] = args.labeling
        logging.info(f"⚙️  Override CLI: --labeling {args.labeling}")

    try:
        df_processed = run_phase1_preprocessing()
        df_hmm = run_phase2_hmm_enrichment(df_processed)
        try:
            predictor, final_metrics = run_phase3_gb_training(df_hmm)
        except RuntimeError as hpo_err:
            # HPO degenere (≥50% trial pruned per assenza positivi): fallisci forte.
            # NON salvare nessun modello, NON stampare il banner di successo.
            # Exit 2 → distinguibile da exit 1 (errore generico) per CI/wrapper.
            if "HPO degenerate" in str(hpo_err):
                logging.critical("=" * 70)
                logging.critical("💥 HPO DEGENERE — PIPELINE INTERROTTA SENZA SALVARE MODELLO")
                logging.critical("=" * 70)
                logging.critical(str(hpo_err))
                logging.critical("=" * 70)
                sys.exit(2)
            raise

        save_artifacts(predictor, final_metrics, df_hmm)

        # === ANALISI POST-TRAINING ===
        if CONFIG["pipeline"].get("run_post_training_analysis", True):
            _run_post_training_analysis(predictor, df_hmm, final_metrics)

        logging.info("\n" + "🎉 "*20)
        logging.info("✅ PIPELINE COMPLETATA CON SUCCESSO")
        logging.info("🎉 "*20)
        logging.info(f"📊 PR-AUC Finale: {final_metrics.get('pr_auc', 'N/A')}")
        logging.info(f"💾 Tutti gli artifact sono in: {CONFIG['paths']['models_dir']}")

    except Exception as e:
        logging.critical(f"💥 PIPELINE FALLITA: {e}")
        logging.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()