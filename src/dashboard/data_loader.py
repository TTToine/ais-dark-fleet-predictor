#!/usr/bin/env python3
"""Caricamento e generazione dati demo"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse

def generate_demo_data(n_ships=1000, n_days=30, output_dir="data/demo"):
    np.random.seed(42)
    print(f"📊 Generazione dati demo: {n_ships} navi, {n_days} giorni...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Genera dati sintetici
    df = pd.DataFrame({
        "mmsi": np.random.randint(1e8, 1e9, n_ships * n_days),
        "latitude": np.random.uniform(30, 60, n_ships * n_days),
        "longitude": np.random.uniform(-10, 30, n_ships * n_days),
        "timestamp": pd.date_range("2024-01-01", periods=n_ships * n_days, freq="H")[:n_ships * n_days],
        "dark_fleet_probability": np.random.beta(2, 5, n_ships * n_days),
    })
    
    df.to_parquet(output_path / "ais_demo.parquet", index=False)
    print(f"✅ Dati salvati in {output_path}")
# Aggiungi questa funzione alla fine di src/dashboard/data_loader.py

import joblib
import os

def load_real_predictions(
    model_path: str = "models/best_model.pkl", 
    features_path: str = "data/processed/ais_features.parquet",
    feature_cols: list = None # Lista delle feature usate dal modello
) -> pd.DataFrame:
    """
    Carica i dati AIS e genera predizioni usando il modello reale salvato.
    Gestisce sia modelli base LightGBM che ConformalWrapper.
    """
    print(f"📂 Caricamento modello da: {model_path}")
    print(f"📂 Caricamento features da: {features_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato in {model_path}. Esegui prima il training.")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Dati features non trovati in {features_path}.")
        
    # 1. Carica modello
    model = joblib.load(model_path)
    
    # 2. Carica dati
    df = pd.read_parquet(features_path)
    
    # 3. Identifica colonne feature (se non specificate, prova a indovinare o usa tutte le numeriche)
    if feature_cols is None:
        # Escludi colonne non-feature comuni
        exclude_cols = ['mmsi', 'timestamp', 'ship_name', 'dark_fleet_true'] 
        # Se il target è nel dataframe, escludilo
        if 'dark_fleet_probability' in df.columns: exclude_cols.append('dark_fleet_probability')
        
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
        
    X = df[feature_cols]
    
    # 4. Genera predizioni
    if hasattr(model, 'predict_proba_with_interval'):
        # Caso Modello Conformal Wrapper
        print("🔮 Uso modello Conformal Wrapper...")
        prob, lower, upper = model.predict_proba_with_interval(X)
        df['dark_fleet_probability'] = prob
        df['pi_lower'] = lower
        df['pi_upper'] = upper
    elif hasattr(model, 'predict_proba'):
        # Caso Modello LightGBM Base
        print("🔮 Uso modello LightGBM Base (intervalli simulati)...")
        probs = model.predict_proba(X)[:, 1]
        df['dark_fleet_probability'] = probs
        
        # Simula intervalli se non presenti (per compatibilità dashboard)
        # In produzione, dovresti ricalcolare conformal qui se vuoi essere rigoroso
        width = 0.15 # Larghezza fissa demo
        df['pi_lower'] = np.clip(probs - width/2, 0, 1)
        df['pi_upper'] = np.clip(probs + width/2, 0, 1)
    else:
        raise ValueError("Il modello caricato non supporta predict_proba o predict_proba_with_interval")
        
    print(f"✅ Predizioni generate per {len(df)} record.")
    return df
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ships", type=int, default=1000)
    parser.add_argument("--n-days", type=int, default=30)
    parser.add_argument("--output", type=str, default="data/demo")
    args = parser.parse_args()
    generate_demo_data(args.n_ships, args.n_days, args.output)
