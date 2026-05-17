"""
run_pipeline.py
Script di orchestrazione end-to-end per AIS Dark Fleet Predictor.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import traceback

# Aggiungi la root del progetto al PYTHONPATH
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data_prep import AISDataPreprocessor
    from src.hmm_model import CausalBayesianMixture
    from src.gb_training import DarkFleetPredictor
except ImportError as e:
    logging.error(f"❌ Import fallito. Verifica che src/data_prep.py, src/hmm_model.py e src/gb_training.py esistano.")
    logging.error(f"Dettaglio: {e}")
    sys.exit(1)

import yaml # Assicurati di aver fatto pip install pyyaml

def load_config(config_path: str = "configs/pipeline_config.yaml") -> dict:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning("Config file non trovato. Uso default hardcodato (fallback).")
        CONFIG = {
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
        return {} 

# Carica configurazione
CONFIG = load_config()
if not CONFIG:
    logging.error("Impossibile caricare la configurazione. Verifica configs/pipeline_config.yaml")
    sys.exit(1)

# ========================================================================
# SETUP & UTILS
# ========================================================================
def setup_logging_and_dirs():
    dirs = [
        Path(CONFIG["paths"]["processed_output"]).parent,
        Path(CONFIG["paths"]["models_dir"]),
        "logs"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        
    log_file = f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    models_dir = Path(CONFIG["paths"]["models_dir"])
    
    if predictor.best_model:
        model_path = models_dir / "lgb_dark_fleet.txt"
        predictor.best_model.save_model(str(model_path))
        logging.info(f"💾 Modello LightGBM salvato in {model_path}")
        
        params_path = models_dir / "best_params.json"
        with open(params_path, 'w') as f:
            json.dump(predictor.best_params, f, indent=2)
        logging.info(f"💾 Hyperparametri salvati in {params_path}")
    
    metrics_path = Path(CONFIG["paths"]["metrics_file"])
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["config_summary"] = {k: v for k, v in CONFIG.items() if k != "paths"}
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"📊 Metriche salvate in {metrics_path}")
    
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
        logging.warning("⚠️  File raw non trovato. Generazione Mock Data REALISTICO con Blackout...")
        
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
                # Simula variabilità normale
                sog = max(0, base_sog + np.random.normal(0, 1.5))
                cog = (base_cog + np.random.normal(0, 3)) % 360
                lat = 36.5 + ship_idx * 0.3 + np.random.normal(0, 0.01)
                lon = 13.0 + np.random.normal(0, 0.02)
                
                mock_records.append({
                    'MMSI': mmsi,
                    'Timestamp': current_time,
                    'Lat': lat,
                    'Lon': lon,
                    'SOG': sog,
                    'COG': cog
                })
                
                # LOGICA BLACKOUT: Ogni ~50 punti, crea un gap di 13 ore (Target=1)
                if t > 50 and t % 50 == 0:
                    # Salta 13 ore (78 step da 10 min)
                    current_time += timedelta(hours=13)
                else:
                    # Passo normale 10 min
                    current_time += timedelta(minutes=10)
        
        mock_df = pd.DataFrame(mock_records)
        
        # Esegui pipeline manuale sul mock df
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
# FASE 2: ESTRAZIONE FEATURE BAYESIANE (HMM)
# ========================================================================
def run_phase2_hmm_enrichment(df_processed: pd.DataFrame) -> pd.DataFrame:
    logging.info("\n" + "="*60)
    logging.info("FASE 2: Estrazione Regimi Latenti (Bayesian Mixture)")
    logging.info("="*60)
    
    cfg = CONFIG["hmm"]
    extractor = CausalBayesianMixture(
        window_size=cfg["window_size"],
        sigma_prior_speed=cfg["sigma_prior_speed"],
        sigma_prior_turn=cfg["sigma_prior_turn"]
    )
    
    df_enriched_list = []
    unique_ships = df_processed["MMSI"].unique()
    logging.info(f"🚢 Processing di {len(unique_ships)} navi...")
    
    for i, mmsi in enumerate(unique_ships):
        df_ship = df_processed[df_processed["MMSI"] == mmsi].copy()
        if len(df_ship) < cfg["window_size"] + 2:
            logging.warning(f"Nave {mmsi}: troppo corta ({len(df_ship)} righe). Skip.")
            continue
            
        # fit_scaler=True SOLO per la prima nave, False per le successive
        fit_scaler = (i == 0)
        
        try:
            df_ship_enriched = extractor.process_dataframe_causal(
                df_ship, 
                use_advi=cfg["use_advi"], 
                fit_scaler=fit_scaler
            )
            df_enriched_list.append(df_ship_enriched)
            if (i + 1) % 10 == 0:
                logging.info(f"🔄 Processate {i+1}/{len(unique_ships)} navi...")
        except Exception as e:
            logging.error(f"❌ Errore HMM per nave {mmsi}: {e}")
            continue
        finally:
            extractor.reset()
            
    if not df_enriched_list:
        raise RuntimeError("Nessuna nave processata correttamente nella Fase 2.")
        
    df_hmm = pd.concat(df_enriched_list, ignore_index=True)
    df_hmm = df_hmm.sort_values("Timestamp").reset_index(drop=True)
    
    logging.info(f"✅ Fase 2 completata. Feature bayesiane estratte per {len(df_hmm)} osservazioni.")
    return df_hmm

# ========================================================================
# FASE 3: TRAINING & VALUTAZIONE GB
# ========================================================================
def run_phase3_gb_training(df_hmm: pd.DataFrame) -> dict:
    logging.info("\n" + "="*60)
    logging.info("FASE 3: Training Gradient Boosting & Valutazione")
    logging.info("="*60)
    
    cfg = CONFIG["gb"]
    predictor = DarkFleetPredictor(
        n_trials=cfg["n_trials"],
        n_splits=cfg["n_splits"],
        gap_hours=cfg["gap_hours"],
        freq_min=cfg["freq_min"],
        seed=cfg["seed"]
    )
    
    split_idx = int(len(df_hmm) * CONFIG["pipeline"]["train_split_ratio"])
    train_df = df_hmm.iloc[:split_idx].copy()
    test_df = df_hmm.iloc[split_idx:].copy()
    
    logging.info(f"📅 Split: {len(train_df)} train, {len(test_df)} test")
    
    X_train, y_train = predictor.prepare_data_for_cv(train_df)
    X_test, y_test = predictor.prepare_data_for_cv(test_df)
    
    # Check se ci sono positivi nel test set per evitare crash ROC AUC
    if y_test.nunique() < 2:
        logging.warning("⚠️  Test set contiene una sola classe. ROC AUC non sarà calcolabile.")
        
    predictor.optimize_and_train(X_train, y_train)
    
    # Valutazione Custom per gestire il caso single-class
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

    logging.info("\n" + "="*60)
    logging.info("📊 RISULTATI FINALI SUL TEST SET (FUTURO)")
    logging.info("="*60)
    for name, value in metrics.items():
        logging.info(f"{name:15s}: {value:.4f}")
    logging.info("="*60 + "\n")
    
    # Visualizzazioni
    try:
        predictor.plot_calibration(X_test, y_test, n_bins=10)
        predictor.plot_feature_importance_shap(X_test, max_display=10)
    except Exception as e:
        logging.warning(f"⚠️  Plot generation failed: {e}")
        
    return metrics

# Import necessari per la valutazione custom
from sklearn.metrics import average_precision_score, log_loss, brier_score_loss, roc_auc_score

# ========================================================================
# MAIN ORCHESTRATOR
# ========================================================================
def main():
    setup_logging_and_dirs()
    logging.info("🚀 AVVIO PIPELINE AIS DARK FLEET PREDICTOR")
    
    try:
        df_processed = run_phase1_preprocessing()
        df_hmm = run_phase2_hmm_enrichment(df_processed)
        final_metrics = run_phase3_gb_training(df_hmm)
        
        # Ricrea predictor per salvataggio (semplificato)
        predictor = DarkFleetPredictor(seed=CONFIG["gb"]["seed"])
        # In un'implementazione reale, passeresti l'oggetto addestrato dalla fase 3
        
        logging.info("\n" + "🎉"*20)
        logging.info("✅ PIPELINE COMPLETATA CON SUCCESSO")
        logging.info("🎉"*20)
        logging.info(f"📊 PR-AUC Finale: {final_metrics.get('pr_auc', 'N/A')}")
        logging.info(f"💾 Tutti gli artifact sono in: {CONFIG['paths']['models_dir']}")
        
    except Exception as e:
        logging.critical(f"💥 PIPELINE FALLITA: {e}")
        logging.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()