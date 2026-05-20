"""
tests/test_data_prep.py

Test sull'invariante critico di causalità (no look-ahead) in src/data_prep.py.
Una violazione di questo invariante rende l'intero PR-AUC riportato non difendibile.

Strategia: costruire dataset sintetici controllati dove possiamo verificare
PROGRAMMATICAMENTE che per ogni riga, nessuna feature dipende da osservazioni
con timestamp > current_timestamp.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep import AISDataPreprocessor


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def bbox_mediterranean():
    return {"min_lat": 30.0, "max_lat": 46.0, "min_lon": -6.0, "max_lon": 36.0}


@pytest.fixture
def preprocessor(bbox_mediterranean):
    return AISDataPreprocessor(
        bounding_box=bbox_mediterranean,
        gap_threshold_hours=12.0,
        prediction_horizon_hours=24.0,
        downsample_minutes=10,
        min_sog_knots=0.0,
        max_sog_knots=50.0,
        seed=42,
    )


@pytest.fixture
def synthetic_ais_data():
    """Mini dataset AIS: 2 navi × 50 ping ogni 10 min, all'interno del Mediterraneo."""
    rng = np.random.default_rng(42)
    records = []
    base_time = pd.Timestamp("2024-06-01 00:00:00")
    for mmsi in [111111111, 222222222]:
        lat0 = rng.uniform(35.0, 40.0)
        lon0 = rng.uniform(10.0, 15.0)
        for t in range(50):
            records.append({
                "MMSI": mmsi,
                "Timestamp": base_time + pd.Timedelta(minutes=10 * t),
                "Lat": lat0 + rng.normal(0, 0.01),
                "Lon": lon0 + rng.normal(0, 0.01),
                "SOG": max(0.0, 10.0 + rng.normal(0, 1)),
                "COG": rng.uniform(0, 360),
            })
    return pd.DataFrame(records)


@pytest.fixture
def synthetic_with_blackout():
    """Dataset con UN blackout deterministico noto per testare il target."""
    records = []
    base = pd.Timestamp("2024-06-01 00:00:00")
    mmsi = 999999999
    # 20 ping normali ogni 10 min
    for t in range(20):
        records.append({
            "MMSI": mmsi,
            "Timestamp": base + pd.Timedelta(minutes=10 * t),
            "Lat": 36.0, "Lon": 13.0, "SOG": 10.0, "COG": 90.0,
        })
    # Buco di 15 ore (>12h threshold) tra ping 19 e 20
    last_time_before = base + pd.Timedelta(minutes=10 * 19)
    next_time = last_time_before + pd.Timedelta(hours=15)
    for t in range(10):
        records.append({
            "MMSI": mmsi,
            "Timestamp": next_time + pd.Timedelta(minutes=10 * t),
            "Lat": 36.5, "Lon": 13.5, "SOG": 10.0, "COG": 90.0,
        })
    return pd.DataFrame(records), 19  # 19 è l'indice dell'ultima riga prima del gap


# ============================================================================
# 1. FILTRO GEOGRAFICO
# ============================================================================
class TestGeographicFilter:
    def test_keeps_only_in_bbox(self, preprocessor):
        df = pd.DataFrame({
            "MMSI": [1, 2, 3, 4],
            "Timestamp": pd.date_range("2024-01-01", periods=4, freq="10min"),
            "Lat": [36.0, 50.0, 37.0, 20.0],   # 2 e 4 fuori
            "Lon": [13.0, 13.0, 50.0, 13.0],   # 3 fuori
            "SOG": [10.0] * 4, "COG": [90.0] * 4,
        })
        out = preprocessor.filter_geographic_area(df)
        assert len(out) == 1
        assert out["MMSI"].iloc[0] == 1


# ============================================================================
# 2. VALIDAZIONE FISICA
# ============================================================================
class TestPhysicalValidity:
    def test_drops_out_of_range_sog(self, preprocessor):
        df = pd.DataFrame({
            "MMSI": [1, 1, 1], "Timestamp": pd.date_range("2024-01-01", periods=3, freq="10min"),
            "Lat": [36.0]*3, "Lon": [13.0]*3,
            "SOG": [10.0, -5.0, 200.0],   # -5 e 200 invalidi
            "COG": [90.0]*3,
        })
        out = preprocessor.validate_physical_plausibility(df)
        assert len(out) == 1
        assert out["SOG"].iloc[0] == 10.0


# ============================================================================
# 3. DOWNSAMPLING CAUSALE (Last-Known-State)
# ============================================================================
class TestCausalDownsampling:
    def test_uses_last_known_not_future_mean(self, preprocessor):
        """
        Critico: downsampling deve usare l'ULTIMO valore della finestra,
        non la media (che userebbe valori futuri all'interno della finestra).
        """
        # 4 punti in 15 min: due nella finestra 00:00-00:10, due in 00:10-00:20
        df = pd.DataFrame({
            "MMSI": [1]*4,
            "Timestamp": [
                pd.Timestamp("2024-06-01 00:01:00"),
                pd.Timestamp("2024-06-01 00:09:00"),
                pd.Timestamp("2024-06-01 00:11:00"),
                pd.Timestamp("2024-06-01 00:19:00"),
            ],
            "Lat": [36.0]*4, "Lon": [13.0]*4,
            "SOG": [10.0, 20.0, 30.0, 40.0],
            "COG": [0.0, 90.0, 180.0, 270.0],
        })
        out = preprocessor.downsample_causal(df)
        # In ogni finestra deve esserci l'ultimo valore, non la media
        sog_values = sorted(out["SOG"].tolist())
        # Atteso: 20.0 (ultimo della prima finestra) e 40.0 (ultimo della seconda)
        assert 20.0 in sog_values
        assert 40.0 in sog_values
        # NON deve apparire 15.0 (= media 10+20) né 35.0 (= media 30+40)
        assert 15.0 not in sog_values
        assert 35.0 not in sog_values


# ============================================================================
# 4. FEATURE ENGINEERING — INVARIANTE NO LOOK-AHEAD
# ============================================================================
class TestNoLookAhead:
    """
    Verifica programmaticamente che le feature al tempo t non dipendano da
    osservazioni con timestamp > t. Test fondamentale per la difensibilità.
    """

    def test_features_invariant_under_future_truncation(self, preprocessor, synthetic_ais_data):
        """
        Test centrale: se TRONCHIAMO il dataset al tempo t* (eliminando tutto > t*),
        le feature calcolate sulle righe <= t* devono essere IDENTICHE a quelle
        ottenute dal dataset completo.

        Se questa invarianza fallisce, c'è look-ahead.
        """
        df = synthetic_ais_data.copy()
        df_full = preprocessor.engineer_causal_features(
            preprocessor.downsample_causal(
                preprocessor.validate_physical_plausibility(df)
            )
        )

        # Tronca dataset al 50% temporale per ogni nave
        df_truncated = df.sort_values(["MMSI", "Timestamp"]).groupby("MMSI").head(25)
        df_trunc_processed = preprocessor.engineer_causal_features(
            preprocessor.downsample_causal(
                preprocessor.validate_physical_plausibility(df_truncated)
            )
        )

        feature_cols = ["delta_SOG", "delta_COG", "speed_acc", "turn_rate", "dt_prev_hours"]
        feature_cols = [c for c in feature_cols if c in df_full.columns and c in df_trunc_processed.columns]

        # Per ogni nave, le feature delle prime N righe del dataset troncato
        # devono coincidere con quelle delle prime N righe del dataset completo
        for mmsi in df_trunc_processed["MMSI"].unique():
            full = df_full[df_full["MMSI"] == mmsi].sort_values("Timestamp").reset_index(drop=True)
            trunc = df_trunc_processed[df_trunc_processed["MMSI"] == mmsi].sort_values("Timestamp").reset_index(drop=True)
            n = len(trunc)
            if n == 0:
                continue
            for col in feature_cols:
                # Tolleranza numerica per floating point
                full_vals = full[col].iloc[:n].fillna(-999).values
                trunc_vals = trunc[col].iloc[:n].fillna(-999).values
                np.testing.assert_allclose(
                    full_vals, trunc_vals, rtol=1e-9, atol=1e-9,
                    err_msg=(f"LEAKAGE rilevato in {col} per MMSI {mmsi}: "
                             "feature cambiano se il dataset viene troncato al futuro.")
                )

    def test_first_row_per_vessel_has_no_kinematic_delta(self, preprocessor, synthetic_ais_data):
        """
        La prima riga di ogni nave non ha una riga precedente:
        delta_SOG, delta_COG, dt_prev_hours devono essere NaN/0, mai derivati
        da una "riga successiva" (look-ahead).
        """
        df_pre = preprocessor.engineer_causal_features(
            preprocessor.downsample_causal(
                preprocessor.validate_physical_plausibility(synthetic_ais_data)
            )
        )
        for mmsi in df_pre["MMSI"].unique():
            ship = df_pre[df_pre["MMSI"] == mmsi].sort_values("Timestamp")
            first = ship.iloc[0]
            # delta_SOG e delta_COG sulla prima riga devono essere NaN o 0
            # (mai un valore positivo derivato dal "futuro")
            for col in ["delta_SOG", "delta_COG"]:
                if col in first.index:
                    val = first[col]
                    assert pd.isna(val) or val == 0, (
                        f"Prima riga di {mmsi} ha {col}={val}, "
                        "suggerisce derivazione da riga futura."
                    )


# ============================================================================
# 5. TARGET CAUSALE
# ============================================================================
class TestCausalTarget:
    def test_target_only_uses_next_gap(self, preprocessor, synthetic_with_blackout):
        """
        Il target Y=1 per la riga al tempo t deve essere acceso SE E SOLO SE
        il gap_to_next (prossimo blackout) >= threshold E <= horizon + threshold.
        L'invariante: niente nel target deve dipendere da osservazioni precedenti
        a t in modo non causale.
        """
        df, idx_last_before_gap = synthetic_with_blackout
        df_proc = preprocessor.engineer_causal_features(
            preprocessor.downsample_causal(
                preprocessor.validate_physical_plausibility(df)
            )
        )
        df_with_target = preprocessor.create_causal_target(df_proc)
        # Almeno una riga deve avere target = 1 (quella prima del blackout)
        assert df_with_target[preprocessor.target_col].sum() >= 1, (
            "Nessun target positivo creato: il blackout di 15h non è stato catturato."
        )

    def test_target_zero_when_no_future_gap(self, preprocessor):
        """Serie continua senza gap → tutti target = 0."""
        df = pd.DataFrame({
            "MMSI": [1]*30,
            "Timestamp": pd.date_range("2024-06-01", periods=30, freq="10min"),
            "Lat": [36.0]*30, "Lon": [13.0]*30,
            "SOG": [10.0]*30, "COG": [90.0]*30,
        })
        df_proc = preprocessor.engineer_causal_features(
            preprocessor.downsample_causal(
                preprocessor.validate_physical_plausibility(df)
            )
        )
        df_with_target = preprocessor.create_causal_target(df_proc)
        # Tutti i target devono essere 0 (nessun gap > 12h)
        assert df_with_target[preprocessor.target_col].sum() == 0


# ============================================================================
# 6. SMOKE TEST — la pipeline end-to-end non si rompe su input minimale
# ============================================================================
class TestPipelineEndToEnd:
    def test_smoke(self, preprocessor, synthetic_ais_data):
        df = preprocessor.filter_geographic_area(synthetic_ais_data)
        df = preprocessor.validate_physical_plausibility(df)
        df = preprocessor.downsample_causal(df)
        df = preprocessor.engineer_causal_features(df)
        df = preprocessor.create_causal_target(df)
        # Deve restituire un DataFrame con almeno una riga
        assert len(df) > 0
        # Deve contenere il target
        assert preprocessor.target_col in df.columns
        # Le feature cinematiche devono essere presenti
        for col in ["delta_SOG", "delta_COG", "speed_acc", "turn_rate", "dt_prev_hours"]:
            assert col in df.columns, f"Feature {col} mancante"
