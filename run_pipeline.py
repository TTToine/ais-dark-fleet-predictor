"""
run_pipeline.py
Script di orchestrazione end-to-end per AIS Dark Fleet Predictor.
✅ Allineato alla review e ai moduli patchati (data_prep, hmm_model, gb_training)
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

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, log_loss, brier_score_loss, roc_auc_score

# Aggiungi root progetto al PYTHONPATH
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data_prep import AISDataPreprocessor
    from src.hmm_model import CausalBayesianMixture
    from src.gb_training import DarkFleetPredictor, gap_threshold_sensitivity
except ImportError as e:
    print(f"❌ Import fallito. Verifica struttura cartelle e moduli patchati.")
    print(f"Dettaglio: {e}")
    sys.exit(1)

try:
    from src.viz.uncertainty_plots import plot_geographic_predictions
except ImportError:
    plot_geographic_predictions = None


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
            "seed": 42
        },
        "hmm": {
            "window_size": 36,
            "use_advi": True,
            "sigma_prior_speed": 1.0,
            "sigma_prior_turn": 2.0
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
            "save_enriched_data": True
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
        logging.warning("⚠️ File raw non trovato. Generazione Mock Data REALISTICO con Blackout...")
        np.random.seed(cfg["seed"])
        
        n_ships = 2
        points_per_ship = 500
        base_time = datetime(2024, 6, 1)
        mock_records = []
        
        for ship_idx, mmsi in enumerate([123456789, 234567890]):
            current_time = base_time
            base_sog = np.random.uniform(8, 15)
            base_cog = np.random.uniform(0, 360)
            
            for t in range(points_per_ship):
                sog = max(0, base_sog + np.random.normal(0, 1.5))
                cog = (base_cog + np.random.normal(0, 3)) % 360
                lat = 36.5 + ship_idx * 0.3 + np.random.normal(0, 0.01)
                lon = 13.0 + np.random.normal(0, 0.02)
                
                mock_records.append({
                    'MMSI': mmsi, 'Timestamp': current_time,
                    'Lat': lat, 'Lon': lon, 'SOG': sog, 'COG': cog
                })
                
                # LOGICA BLACKOUT: Ogni ~50 punti, gap di 13h (Target=1)
                if t > 50 and t % 50 == 0:
                    current_time += timedelta(hours=13)
                else:
                    current_time += timedelta(minutes=10)
                    
        mock_df = pd.DataFrame(mock_records)
        
        # Pipeline manuale sul mock
        df_geo = preprocessor.filter_geographic_area(mock_df)
        df_valid = preprocessor.validate_physical_plausibility(df_geo)
        df_down = preprocessor.downsample_causal(df_valid)
        df_feat = preprocessor.engineer_causal_features(df_down)
        df_processed = preprocessor.create_causal_target(df_feat)
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
    extractor = CausalBayesianMixture(
        window_size=cfg["window_size"],
        sigma_prior_speed=cfg["sigma_prior_speed"],
        sigma_prior_turn=cfg["sigma_prior_turn"]
    )

    n_ships = df_processed["MMSI"].nunique()
    logging.info(f"🚢 Processing di {n_ships} navi (iterazione interna a process_dataframe_causal)...")

    df_hmm = extractor.process_dataframe_causal(
        df_processed,
        use_advi=cfg["use_advi"],
        apply_markov=True,
        update_freq=10,
        adaptive_threshold=1.0,
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
    preds_proba = predictor.best_model.predict(X_test)

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
# POST-TRAINING ANALYSIS
# ========================================================================
def _run_post_training_analysis(predictor: "DarkFleetPredictor", df_hmm: pd.DataFrame):
    """Produce tutti gli artefatti di analisi post-training.
    Controllato da pipeline.run_post_training_analysis nel YAML."""
    split_idx = int(len(df_hmm) * CONFIG["pipeline"]["train_split_ratio"])
    train_df = df_hmm.iloc[:split_idx].copy()
    test_df  = df_hmm.iloc[split_idx:].copy()

    X_test, y_test = predictor.prepare_data_for_cv(test_df)
    X_test_lgb = X_test.drop(columns=['MMSI', 'Timestamp'], errors='ignore')

    # F2-ottimale threshold + confusion matrix
    try:
        predictor.find_optimal_threshold(X_test_lgb, y_test,
                                         save_path="models/f2_threshold.png")
        logging.info("✅ F2 threshold plot salvato in models/f2_threshold.png")
    except Exception as e:
        logging.warning(f"find_optimal_threshold fallito: {e}")

    # Conformal prediction threshold (FPR garantito)
    try:
        # Usa metà del test come calibration set
        mid = len(X_test_lgb) // 2
        X_calib, y_calib = X_test_lgb.iloc[:mid], y_test.iloc[:mid]
        X_eval,  y_eval  = X_test_lgb.iloc[mid:], y_test.iloc[mid:]
        predictor.evaluate_with_conformal(X_calib, y_calib, X_eval, y_eval,
                                          target_fpr=0.1)
        logging.info("✅ Conformal threshold calcolato")
    except Exception as e:
        logging.warning(f"evaluate_with_conformal fallito: {e}")

    # Calibration comparison plot
    try:
        predictor.plot_calibration_comparison(X_test_lgb, y_test,
                                              save_path="models/calibration_comparison.png")
        logging.info("✅ Calibration comparison salvato in models/calibration_comparison.png")
    except Exception as e:
        logging.warning(f"plot_calibration_comparison fallito: {e}")

    # SHAP interaction plot (prime due feature per default)
    try:
        f1, f2 = predictor.features_full[0], predictor.features_full[1]
        predictor.plot_shap_interaction(X_test_lgb, feature1=f1, feature2=f2,
                                        save_path="models/shap_interaction.png")
        logging.info(f"✅ SHAP interaction plot ({f1} x {f2}) salvato")
    except Exception as e:
        logging.warning(f"plot_shap_interaction fallito: {e}")

    # Gap threshold sensitivity analysis
    try:
        sensitivity_results = gap_threshold_sensitivity(
            train_df=train_df, test_df=test_df,
            gap_thresholds=[6, 12, 18, 24],
            n_trials_sensitivity=10, n_splits=3,
            seed=CONFIG["gb"]["seed"]
        )
        sens_path = "models/gap_sensitivity.json"
        with open(sens_path, "w") as f:
            json.dump({str(k): v for k, v in sensitivity_results.items()}, f, indent=2)
        logging.info(f"✅ Sensitivity analysis salvata in {sens_path}")
    except Exception as e:
        logging.warning(f"gap_threshold_sensitivity fallito: {e}")

    # Geographic map (ultima posizione per nave)
    if plot_geographic_predictions is not None:
        try:
            if 'prob_regime_sospetto' in df_hmm.columns:
                plot_geographic_predictions(
                    df_hmm,
                    save_path="models/geographic_predictions.html"
                )
                logging.info("✅ Mappa geografica salvata in models/geographic_predictions.html")
        except Exception as e:
            logging.warning(f"plot_geographic_predictions fallito: {e}")


# ========================================================================
# MAIN ORCHESTRATOR
# ========================================================================
def main():
    setup_logging_and_dirs()
    logging.info("🚀 AVVIO PIPELINE AIS DARK FLEET PREDICTOR")
    
    try:
        df_processed = run_phase1_preprocessing()
        df_hmm = run_phase2_hmm_enrichment(df_processed)
        predictor, final_metrics = run_phase3_gb_training(df_hmm)

        save_artifacts(predictor, final_metrics, df_hmm)

        # === ANALISI POST-TRAINING ===
        if CONFIG["pipeline"].get("run_post_training_analysis", True):
            _run_post_training_analysis(predictor, df_hmm)

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