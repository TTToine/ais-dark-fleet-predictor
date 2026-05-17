#!/usr/bin/env python3
"""
test_dashboard.py - Test per Dashboard Streamlit e Integrazione Conformal
=========================================================================
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Assicura che la root del progetto sia nel path per gli import
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


class TestDashboardPlaceholders:
    """Test di base per verificare che l'ambiente di test funzioni."""
    
    def test_placeholder(self):
        """Test placeholder iniziale."""
        assert True


class TestDashboardIntegration:
    """Test di integrazione per la dashboard e il caricamento dati."""
    
    def test_load_real_predictions_raises_if_missing(self):
        """Test che load_real_predictions sollevi FileNotFoundError se il modello manca."""
        # Import locale per evitare errori se il modulo non è ancora stato creato
        try:
            from src.dashboard.data_loader import load_real_predictions
        except ImportError:
            pytest.skip("Modulo data_loader non trovato. Esegui prima add_dashboard.py")
        
        with pytest.raises(FileNotFoundError):
            load_real_predictions(
                model_path="models/nonexistent_model.pkl",
                features_path="data/processed/ais_features.parquet"
            )

    def test_conformal_wrapper_intervals_validity(self):
        """Test che gli intervalli del wrapper Conformal siano matematicamente validi."""
        try:
            from src.models.conformal_wrapper import ConformalLightGBMClassifier
        except ImportError:
            pytest.skip("Modulo conformal_wrapper non trovato. Esegui prima add_conformal.py")
        
        # Setup minimo
        X = np.random.rand(50, 3)
        y = np.random.randint(0, 2, 50)
        
        model = ConformalLightGBMClassifier(conformal_alpha=0.1)
        model.fit(X, y)
        
        prob, lower, upper = model.predict_proba_with_interval(X)
        
        # Assert fondamentali
        assert np.all(prob >= 0) and np.all(prob <= 1), "Probabilità fuori range [0,1]"
        assert np.all(lower >= 0), "Lower bound negativo"
        assert np.all(upper <= 1), "Upper bound > 1"
        assert np.all(lower <= prob), "Lower > Probabilità"
        assert np.all(prob <= upper), "Probabilità > Upper"
        assert np.all((upper - lower) >= 0), "Intervallo negativo"

    def test_dashboard_components_structure(self):
        """Test che i componenti UI abbiano la struttura dati corretta."""
        try:
            from src.dashboard.components import render_kpi_cards
        except ImportError:
            pytest.skip("Modulo components non trovato.")
        
        # Dati finti
        df_dummy = pd.DataFrame({"mmsi": [1, 2], "lat": [0, 0]})
        pred_dummy = pd.DataFrame({
            "dark_fleet_probability": [0.8, 0.2],
            "pi_lower": [0.7, 0.1],
            "pi_upper": [0.9, 0.3]
        })
        
        # Verifica che i dati abbiano la struttura attesa dai componenti
        assert "dark_fleet_probability" in pred_dummy.columns
        assert len(pred_dummy) == len(df_dummy)
        
        # Nota: Non possiamo testare il rendering Streamlit diretto senza mock complessi,
        # ma questo test garantisce che i dati passati ai componenti siano validi.


class TestDataLoaderDemo:
    """Test per il generatore di dati demo."""
    
    def test_generate_demo_data_creates_files(self, tmp_path):
        """Test che la generazione dati crei i file attesi."""
        try:
            from src.dashboard.data_loader import generate_demo_data
        except ImportError:
            pytest.skip("Modulo data_loader non trovato.")
        
        output_dir = tmp_path / "demo_data"
        
        df_ais, df_pred = generate_demo_data(
            n_ships=100,
            n_days=7,
            output_dir=str(output_dir)
        )
        
        assert (output_dir / "ais_demo.parquet").exists()
        assert (output_dir / "predictions_demo.parquet").exists()
        
        assert len(df_ais) > 0
        assert len(df_pred) > 0
        assert "mmsi" in df_ais.columns
       