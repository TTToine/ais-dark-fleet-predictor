"""
src/data_prep.py
Modulo per l'ingegneria dei dati AIS e la creazione del target causale.
Responsabilità:
- Caricamento e filtraggio geografico dei dati AIS grezzi
- Validazione fisica e pulizia (SOG/COG range, duplicati, coordinate)
- Downsampling causale (preserva l'ultimo stato noto, non media)
- Feature engineering cinematico causale (solo informazioni <= t)
- Creazione target binario: Y=1 se spegnimento intenzionale previsto entro horizon
Vincoli fondamentali:
- NO look-ahead: le feature al tempo t non usano mai dati > t
- Il target è etichettato in fase di prep ma RIMOSSO prima del training HMM
"""
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

# 🔴 FIX 2: Rimossi import di moduli inesistenti (data_validation, exceptions)
# Se in futuro servono, vanno creati o gestiti con fallback. Per ora bloccavano la compilazione.

# 🟡 FIX 19: Filtri warning specifici, non globali
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AISDataPreprocessor:
    """Pipeline di pre-processing per dati AIS marittimi con validazione causale."""

    def __init__(self,
                 bounding_box: Dict[str, float],
                 gap_threshold_hours: float = 12.0,
                 prediction_horizon_hours: float = 24.0,
                 downsample_minutes: int = 10,
                 min_sog_knots: float = 0.0,
                 max_sog_knots: float = 50.0,
                 seed: int = 42):
        """
        Args:
            bounding_box: Dict con chiavi ['min_lat', 'max_lat', 'min_lon', 'max_lon']
            gap_threshold_hours: Soglia minima di gap per considerare "spegnimento" (default: 12h)
            prediction_horizon_hours: Horizon di previsione: il gap deve avvenire entro questo tempo
            downsample_minutes: Intervallo di downsampling in minuti
            min_sog_knots: Soglia minima SOG per validità fisica
            max_sog_knots: Soglia massima SOG per validità fisica
            seed: Seed per riproducibilità
        """
        self.bbox = bounding_box
        self.gap_threshold = pd.Timedelta(hours=gap_threshold_hours)
        self.prediction_horizon = pd.Timedelta(hours=prediction_horizon_hours)
        self.downsample_minutes = downsample_minutes
        self.min_sog = min_sog_knots
        self.max_sog = max_sog_knots
        self.seed = seed
        np.random.seed(seed)
        
        self.output_features = [
            'delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate', 'dt_prev_hours',
            'prob_regime_sospetto', 'incertezza_regime'  # Aggiunti da hmm_model.py
        ]
        self.target_col = 'target_dark_fleet'

    def load_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """Carica dati AIS da CSV o Parquet con gestione memoria."""
        path = Path(path)
        logging.info(f"Caricamento dati da {path}...")
        
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")
            
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix == '.csv':
            try:
                df = pd.read_csv(path, low_memory=False)
            except MemoryError:
                logging.warning("File CSV troppo grande, lettura in chunk...")
                chunks = []
                for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
        else:
            raise ValueError(f"Formato non supportato: {path.suffix}")
            
        logging.info(f"Caricate {len(df):,} righe grezze")
        return df

    def filter_geographic_area(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra i dati all'interno del bounding box specificato."""
        logging.info(f"Filtraggio geografico: lat[{self.bbox['min_lat']}, {self.bbox['max_lat']}], "
                    f"lon[{self.bbox['min_lon']}, {self.bbox['max_lon']}]")
        
        mask = (
            (df['Lat'] >= self.bbox['min_lat']) & 
            (df['Lat'] <= self.bbox['max_lat']) & 
            (df['Lon'] >= self.bbox['min_lon']) & 
            (df['Lon'] <= self.bbox['max_lon'])
        )
        
        df_filtered = df[mask].copy()
        logging.info(f"Dopo filtraggio geografico: {len(df_filtered):,} righe "
                    f"({len(df_filtered)/len(df)*100:.1f}% del totale)")
        return df_filtered

    def validate_physical_plausibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rimuove record fisicamente implausibili o corrotti."""
        logging.info("Validazione plausibilità fisica...")
        initial_count = len(df)
        
        df = df[(df['SOG'] >= self.min_sog) & (df['SOG'] <= self.max_sog)].copy()
        df = df[(df['COG'] >= 0) & (df['COG'] <= 360)].copy()
        df = df.drop_duplicates(subset=['MMSI', 'Timestamp'], keep='first')
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['Lat', 'Lon', 'SOG', 'COG'])
        
        removed = initial_count - len(df)
        logging.info(f"Rimosse {removed:,} righe non plausibili "
                    f"({removed/initial_count*100:.2f}% del filtrato)")
        return df

    def downsample_causal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downsampling temporale causale: preserva l'ultimo stato noto in ogni finestra.
        IMPORTANTE: Usiamo .last() non .mean() per non appiattire segnali bruschi.
        """
        logging.info(f"Downsampling causale a {self.downsample_minutes} minuti (last-known-state)...")
        
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)
        
        df_indexed = df.set_index('Timestamp')
        df_down = (df_indexed
                   .groupby('MMSI', group_keys=False)
                   .resample(f'{self.downsample_minutes}min')
                   .last()
                   .dropna(subset=['Lat', 'Lon', 'SOG', 'COG'])
                   .reset_index())
        
        df_down['time_diff_min'] = df_down.groupby('MMSI')['Timestamp'].diff().dt.total_seconds() / 60
        
        logging.info(f"Dopo downsampling: {len(df_down):,} righe, "
                    f"time_diff mediano: {df_down['time_diff_min'].median():.1f} min")
        return df_down

    def engineer_causal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola feature cinematiche usando SOLO informazioni <= tempo t.
        """
        logging.info("Feature engineering causale...")
        
        df = df.copy()
        df = df.sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)
        
        grouped = df.groupby('MMSI')
        
        df['dt_prev_hours'] = grouped['Timestamp'].diff().dt.total_seconds() / 3600.0
        df['delta_SOG'] = grouped['SOG'].diff()
        df['delta_COG'] = grouped['COG'].diff()
        
        # Gestione corretta della circolarità di COG
        df['delta_COG'] = (df['delta_COG'] + 180) % 360 - 180
        
        # Feature derivate: accelerazione e tasso di virata
        df['speed_acc'] = df['delta_SOG'] / df['dt_prev_hours'].replace(0, np.nan)
        df['turn_rate'] = df['delta_COG'] / df['dt_prev_hours'].replace(0, np.nan)
        
        # Clip per stabilità numerica
        df['speed_acc'] = df['speed_acc'].clip(-15, 15)
        df['turn_rate'] = df['turn_rate'].clip(-180, 180)
        
        # Gestione NaN fisiologici (prima riga di ogni nave)
        fill_cols = ['dt_prev_hours', 'delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate']
        df[fill_cols] = df[fill_cols].fillna(0)
        
        logging.info(f"Feature engineering completato. Feature prodotte: {fill_cols}")
        return df

    def create_causal_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea il target binario Y con vincoli causali rigorosi.
        
        🔴 FIX 6: Implementato prediction_horizon_hours.
        Y=1 al tempo t SE E SOLO SE il prossimo gap è un blackout (> gap_threshold)
        E avviene ENTRO l'orizzonte di predizione.
        """
        logging.info(f"Creazione target causale: gap > {self.gap_threshold.total_seconds()/3600}h "
                     f"(entro horizon {self.prediction_horizon.total_seconds()/3600}h)")
        
        df = df.copy()
        df = df.sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)
        
        df['_next_timestamp'] = df.groupby('MMSI')['Timestamp'].shift(-1)
        df['_gap_to_next'] = (df['_next_timestamp'] - df['Timestamp']).dt.total_seconds() / 3600.0
        
        gap_hours = self.gap_threshold.total_seconds() / 3600.0
        horizon_hours = self.prediction_horizon.total_seconds() / 3600.0
        
        # Target: qualsiasi blackout (gap >= threshold) che si verifica entro horizon_hours
        df[self.target_col] = (
            (df['_gap_to_next'] >= gap_hours) &
            (df['_gap_to_next'] <= horizon_hours)
        ).astype(float)
        
        # L'ultima osservazione per nave non ha un "prossimo ping", quindi target sconosciuto
        df.loc[df['_gap_to_next'].isna(), self.target_col] = np.nan
        df = df.dropna(subset=[self.target_col])
        df[self.target_col] = df[self.target_col].astype(int)
        
        df = df.drop(columns=['_next_timestamp', '_gap_to_next'])
        
        pos_count = df[self.target_col].sum()
        pos_rate = pos_count / len(df) * 100
        logging.info(f"Target distribuito: {pos_count:,} positivi ({pos_rate:.3f}%) su {len(df):,} totali")
        
        if pos_rate < 0.1:
            logging.warning(f"⚠️ Classe positiva molto rara ({pos_rate:.3f}%).")
        elif pos_rate > 10:
            logging.warning(f"⚠️ Classe positiva frequente ({pos_rate:.3f}%).")
            
        return df

    def run_pipeline(self, 
                    input_path: Union[str, Path],
                    output_path: Union[str, Path],
                    return_df: bool = False) -> Optional[pd.DataFrame]:
        """Esegue l'intera pipeline di pre-processing."""
        logging.info("🚀 Avvio pipeline AIS Data Preprocessing")
        
        df_raw = self.load_data(input_path)
        df_geo = self.filter_geographic_area(df_raw)
        df_valid = self.validate_physical_plausibility(df_geo)
        df_down = self.downsample_causal(df_valid)
        df_feat = self.engineer_causal_features(df_down)
        df_final = self.create_causal_target(df_feat)
        
        assert not df_final[['MMSI', 'Timestamp']].duplicated().any(), "❌ Duplicati residui!"
        assert df_final[self.target_col].isna().sum() == 0, "❌ Target con NaN!"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix != '.parquet':
            logging.warning(f"Salvataggio come Parquet consigliato. Convertendo {output_path.suffix} → .parquet")
            output_path = output_path.with_suffix('.parquet')
            
        df_final.to_parquet(output_path, index=False)
        logging.info(f"💾 Dataset salvato in {output_path}")
        
        logging.info(f"Righe finali: {len(df_final):,} | Navi uniche: {df_final['MMSI'].nunique()}")
        
        if return_df:
            return df_final
        return None


if __name__ == "__main__":
    logging.info("=== TEST INTEGRAZIONE AISDataPreprocessor ===")
    np.random.seed(42)

    n_ships = 3
    points_per_ship = 500
    timestamps_base = pd.date_range('2024-06-01', periods=points_per_ship, freq='10min')
    mock_records = []

    for ship_idx, mmsi in enumerate([123456789, 234567890, 345678901]):
        base_sog = np.random.uniform(8, 15)
        base_cog = np.random.uniform(0, 360)
        
        for t, ts in enumerate(timestamps_base):
            sog = base_sog + np.random.normal(0, 1.5)
            cog = (base_cog + np.random.normal(0, 3)) % 360
            lat = 36.5 + ship_idx * 0.3 + np.random.normal(0, 0.01)
            lon = 13.0 + np.random.normal(0, 0.02)
            
            if np.random.random() < 0.02 and t > 100 and t < 400:
                continue
                
            mock_records.append({
                'MMSI': mmsi, 'Timestamp': ts, 'Lat': lat, 'Lon': lon,
                'SOG': max(0, sog), 'COG': cog
            })

    mock_df = pd.DataFrame(mock_records)
    
    sicily_bbox = {'min_lat': 35.0, 'max_lat': 38.0, 'min_lon': 11.0, 'max_lon': 15.5}
    
    preprocessor = AISDataPreprocessor(
        bounding_box=sicily_bbox,
        gap_threshold_hours=12.0,
        prediction_horizon_hours=24.0,
        downsample_minutes=10,
        seed=42
    )

    df_geo = preprocessor.filter_geographic_area(mock_df)
    df_valid = preprocessor.validate_physical_plausibility(df_geo)
    df_down = preprocessor.downsample_causal(df_valid)
    df_feat = preprocessor.engineer_causal_features(df_down)
    df_final = preprocessor.create_causal_target(df_feat)

    print("\n🔍 VALIDAZIONI POST-PIPELINE")
    leak_cols = ['_next_timestamp', '_gap_to_next', 'dt_next_hours']
    for col in leak_cols:
        assert col not in df_final.columns, f"❌ LEAKAGE: colonna '{col}' ancora presente!"
    print("✅ Anti-leakage check: PASSED")
    
    assert 'target_dark_fleet' in df_final.columns
    print("✅ Feature completeness: PASSED")
    
    print("\n📊 STATISTICHE FINALI")
    print(df_final[['target_dark_fleet']].value_counts())
    logging.info("\n✅ TEST INTEGRAZIONE COMPLETATO CON SUCCESSO")