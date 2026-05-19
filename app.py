#!/usr/bin/env python3
"""🚢 AIS Dark Fleet Predictor - Interactive Dashboard"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

# Aggiungi src al path per importare i moduli personalizzati
sys.path.append(str(Path(__file__).parent / "src"))

# Importa le funzioni necessarie
try:
    from src.dashboard.data_loader import generate_demo_data
    from src.dashboard.components import render_header, render_sidebar
except ImportError:
    st.error("❌ Impossibile importare i moduli della dashboard. Assicurati che la struttura delle cartelle sia corretta.")
    st.stop()

# Configurazione pagina
st.set_page_config(
    page_title="AIS Dark Fleet Predictor",
    page_icon="🚢",
    layout="wide"
)

def load_real_predictions(
    model_path: str = "models/lgb_dark_fleet.pkl",
    features_path: str = "data/processed/ais_enriched.parquet"
) -> pd.DataFrame:
    """
    Carica i dati AIS e genera predizioni usando il modello reale salvato.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modello non trovato in {model_path}")
    if not Path(features_path).exists():
        raise FileNotFoundError(f"Dati features non trovati in {features_path}")
        
    # 1. Carica modello
    model = joblib.load(model_path)
    
    # 2. Carica dati
    df = pd.read_parquet(features_path)
    
    # 3. Identifica colonne feature (semplificato: esclude target e ID)
    exclude_cols = ['MMSI', 'Timestamp', 'ship_name', 'target_dark_fleet']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    
    # 4. Genera predizioni
    if hasattr(model, 'predict_proba_with_interval'):
        # Caso Modello Conformal Wrapper
        prob, lower, upper = model.predict_proba_with_interval(X)
        df['dark_fleet_probability'] = prob
        df['pi_lower'] = lower
        df['pi_upper'] = upper
    elif hasattr(model, 'predict_proba'):
        # Caso Modello LightGBM Base
        probs = model.predict_proba(X)[:, 1]
        df['dark_fleet_probability'] = probs
        # Intervalli dummy se non conformal
        df['pi_lower'] = np.clip(probs - 0.1, 0, 1)
        df['pi_upper'] = np.clip(probs + 0.1, 0, 1)
    else:
        raise ValueError("Modello non supporta predict_proba")
        
    return df

def main():
    # Header
    render_header()
    
    # Sidebar
    config = render_sidebar()
    
    # --- CARICAMENTO DATI (La parte che ti interessava) ---
    try:
        # Prova a caricare predizioni REALI dal modello
        with st.spinner("Caricamento modello e predizioni reali..."):
            df = load_real_predictions(
                model_path="models/best_model.pkl",
                features_path="data/processed/ais_features.parquet"
            )
            predictions_df = df # Nel caso real, le predizioni sono nel df stesso
            st.success("✅ Dati reali caricati con successo!")
            
    except (FileNotFoundError, Exception) as e:
        st.warning(f"⚠️ Impossibile caricare modello reale ({e}). Uso dati DEMO.")
        
        # Fallback ai dati demo generati
        try:
            df_ais, df_pred = generate_demo_data(n_ships=500, n_days=14, output_dir="data/demo")
            
            # Unisci per avere formato simile al real
            df = df_ais.merge(df_pred, on=['mmsi', 'timestamp'], how='left')
            predictions_df = df_pred
            
            st.info("ℹ️ Visualizzazione basata su dati sintetici (Demo Mode)")
        except Exception as demo_err:
            st.error(f"❌ Errore anche nella generazione dati demo: {demo_err}")
            st.stop()

    # --- VISUALIZZAZIONE DEI DATI ---
    
    # KPI Rapidi
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Navi Totali", len(df))
    with col2:
        high_risk = len(df[df['dark_fleet_probability'] > 0.7])
        st.metric("Alto Rischio (>70%)", high_risk)
    with col3:
        avg_prob = df['dark_fleet_probability'].mean()
        st.metric("Probabilità Media", f"{avg_prob:.2%}")

    # Mappa Semplice (Placeholder)
    st.subheader("🗺️ Mappa Navi")
    if 'latitude' in df.columns and 'longitude' in df.columns:
        st.map(df[['latitude', 'longitude']])
    else:
        st.warning("Dati geografici non presenti per la mappa.")

    # Tabella Dati
    st.subheader("📋 Dettaglio Navi")
    st.dataframe(df.head(10))

if __name__ == "__main__":
    main()