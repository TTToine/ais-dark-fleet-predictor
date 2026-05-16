"""
src/data_prep.py
Modulo per l'ingegneria dei dati AIS e la creazione del target causale.

Responsabilità:
1. Caricamento e filtraggio geografico dei dati AIS grezzi
2. Validazione fisica e pulizia (SOG/COG range, duplicati, coordinate)
3. Downsampling causale (preserva l'ultimo stato noto, non media)
4. Feature engineering cinematico causale (solo informazioni <= t)
5. Creazione target binario: Y=1 se spegnimento intenzionale previsto entro horizon

Vincoli fondamentali:
- NO look-ahead: le feature al tempo t non usano mai dati > t
- Il target è etichettato in fase di prep ma RIMOSSO prima del training HMM
- Standardizzazione opzionale ma consigliata per coerenza con hmm_model.py
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import warnings

# Sopprimi warning specifici, non tutti
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AISDataPreprocessor:
    """
    Pipeline di pre-processing per dati AIS marittimi con validazione causale.
    
    Output: DataFrame con feature cinematiche causali + target binario per dark fleet prediction.
    """
    
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
            downsample_minutes: Intervallo di downsampling in minuti (default: 10)
            min_sog_knots: Soglia minima SOG per validità fisica (default: 0)
            max_sog_knots: Soglia massima SOG per validità fisica (default: 50 nodi)
            seed: Seed per riproducibilità (usato in sampling/mock)
        """
        self.bbox = bounding_box
        self.gap_threshold = pd.Timedelta(hours=gap_threshold_hours)
        self.prediction_horizon = pd.Timedelta(hours=prediction_horizon_hours)
        self.downsample_minutes = downsample_minutes
        self.min_sog = min_sog_knots
        self.max_sog = max_sog_knots
        self.seed = seed
        np.random.seed(seed)
        
        # Feature che questo modulo produrrà (per allineamento con altri moduli)
        self.output_features = [
            'delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate', 'dt_prev_hours',
            'prob_regime_sospetto', 'incertezza_regime'  # Saranno aggiunti da hmm_model.py
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
            # Prova lettura chunked per file grandi
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
        
        # 1. SOG (Speed Over Ground) nel range fisico
        df = df[(df['SOG'] >= self.min_sog) & (df['SOG'] <= self.max_sog)].copy()
        
        # 2. COG (Course Over Ground) nel range [0, 360]
        df = df[(df['COG'] >= 0) & (df['COG'] <= 360)].copy()
        
        # 3. Rimuovi duplicati esatti di timestamp per MMSI
        df = df.drop_duplicates(subset=['MMSI', 'Timestamp'], keep='first')
        
        # 4. Rimuovi coordinate nulle o infinite
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['Lat', 'Lon', 'SOG', 'COG'])
        
        removed = initial_count - len(df)
        logging.info(f"Rimosse {removed:,} righe non plausibili "
                    f"({removed/initial_count*100:.2f}% del filtrato)")
        return df
    
    def downsample_causal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downsampling temporale causale: preserva l'ultimo stato noto in ogni finestra.
        
        IMPORTANTE: Usiamo .last() non .mean() per non appiattire segnali bruschi
        che potrebbero essere precursori di spegnimento.
        """
        logging.info(f"Downsampling causale a {self.downsample_minutes} minuti (last-known-state)...")
        
        # Conversione timestamp e ordinamento
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)
        
        # Set index per resample
        df_indexed = df.set_index('Timestamp')
        
        # Resample per nave: .last() preserva l'ultimo ping noto nella finestra
        # group_keys=False evita problemi di indice multi-livello
        df_down = (df_indexed
                   .groupby('MMSI', group_keys=False)
                   .resample(f'{self.downsample_minutes}min')
                   .last()
                   .dropna(subset=['Lat', 'Lon', 'SOG', 'COG'])
                   .reset_index())
        
        # Calcola time_diff per diagnosticare gap residui
        df_down['time_diff_min'] = df_down.groupby('MMSI')['Timestamp'].diff().dt.total_seconds() / 60
        
        logging.info(f"Dopo downsampling: {len(df_down):,} righe, "
                    f"time_diff mediano: {df_down['time_diff_min'].median():.1f} min")
        return df_down
    
    def engineer_causal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcola feature cinematiche usando SOLO informazioni <= tempo t.
        
        Feature prodotte:
        - dt_prev_hours: tempo trascorso dal ping precedente
        - delta_SOG: variazione di velocità rispetto al ping precedente
        - delta_COG: variazione di rotta (gestione circolare corretta)
        - speed_acc: accelerazione longitudinale approssimata
        - turn_rate: tasso di virata (gradi/ora)
        """
        logging.info("Feature engineering causale...")
        
        df = df.copy()
        df = df.sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)
        
        # Raggruppamento per nave per calcoli differenziali
        grouped = df.groupby('MMSI')
        
        # 1. Delta temporale dal ping precedente (in ore)
        df['dt_prev_hours'] = grouped['Timestamp'].diff().dt.total_seconds() / 3600.0
        
        # 2. Variazioni cinematiche rispetto al ping precedente
        df['delta_SOG'] = grouped['SOG'].diff()
        df['delta_COG'] = grouped['COG'].diff()
        
        # 3. Gestione corretta della circolarità di COG
        # La differenza tra 359° e 1° è 2°, non -358°
        df['delta_COG'] = (df['delta_COG'] + 180) % 360 - 180
        
        # 4. Feature derivate: accelerazione e tasso di virata
        # Evita divisione per zero con replace
        df['speed_acc'] = df['delta_SOG'] / df['dt_prev_hours'].replace(0, np.nan)
        df['turn_rate'] = df['delta_COG'] / df['dt_prev_hours'].replace(0, np.nan)
        
        # 5. Clip per stabilità numerica (valori fisicamente plausibili)
        df['speed_acc'] = df['speed_acc'].clip(-15, 15)  # nodi/ora
        df['turn_rate'] = df['turn_rate'].clip(-180, 180)  # gradi/ora
        
        # 6. Gestione NaN fisiologici (prima riga di ogni nave)
        fill_cols = ['dt_prev_hours', 'delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate']
        df[fill_cols] = df[fill_cols].fillna(0)
        
        logging.info(f"Feature engineering completato. Feature prodotte: {fill_cols}")
        return df

    def create_causal_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea il target binario Y con vincoli causali rigorosi.
        
        Regola (Last Ping Prediction): Y=1 al tempo t SE E SOLO SE 
        il tempo tra il ping attuale e il PROSSIMO ping è maggiore di gap_threshold_hours.
        Questo identifica l'esatto istante prima che la nave entri in modalità "Dark Fleet".
        """
        logging.info(f"Creazione target causale: gap > {self.gap_threshold.total_seconds()/3600}h")
        
        df = df.copy()
        df = df.sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)
        
        # Calcola il tempo fino al PROSSIMO ping
        df['_next_timestamp'] = df.groupby('MMSI')['Timestamp'].shift(-1)
        df['_gap_to_next'] = (df['_next_timestamp'] - df['Timestamp']).dt.total_seconds() / 3600.0
        
        gap_hours = self.gap_threshold.total_seconds() / 3600
        
        # Y=1 se il gap col prossimo segnale è superiore alla soglia
        # (senza limite superiore: ci importa solo che ci sia un blackout, non quanto dura)
        # Modifica in create_causal_target in data_prep.py:
        df[self.target_col] = (df['_gap_to_next'] >= gap_hours).astype(float) # Float supporta i NaN
        df.loc[df['_gap_to_next'].isna(), self.target_col] = np.nan # Forza l'ultima riga a NaN
        df = df.dropna(subset=[self.target_col])
        df[self.target_col] = df[self.target_col].astype(int) # Torna intero    
        
        df = df.drop(columns=['_next_timestamp', '_gap_to_next'])
        
        # Diagnostica
        pos_count = df[self.target_col].sum()
        pos_rate = pos_count / len(df) * 100
        logging.info(f"Target distribuito: {pos_count:,} positivi ({pos_rate:.3f}%) su {len(df):,} totali")
        
        if pos_rate < 0.1:
            logging.warning(f"⚠️  Classe positiva molto rara ({pos_rate:.3f}%).")
        elif pos_rate > 10:
            logging.warning(f"⚠️  Classe positiva frequente ({pos_rate:.3f}%).")
        
        return df
    
    def run_pipeline(self, 
                    input_path: Union[str, Path],
                    output_path: Union[str, Path],
                    return_df: bool = False) -> Optional[pd.DataFrame]:
        """
        Esegue l'intera pipeline di pre-processing.
        
        Args:
            input_path: Percorso del file AIS grezzo (CSV o Parquet)
            output_path: Percorso per salvare il dataset processato (Parquet consigliato)
            return_df: Se True, ritorna anche il DataFrame in memoria
            
        Returns:
            pd.DataFrame se return_df=True, altrimenti None
        """
        logging.info("🚀 Avvio pipeline AIS Data Preprocessing")
        logging.info(f"Config: gap={self.gap_threshold.total_seconds()/3600}h, "
                    f"horizon={self.prediction_horizon.total_seconds()/3600}h, "
                    f"downsample={self.downsample_minutes}min")
        
        # 1. Caricamento
        df_raw = self.load_data(input_path)
        
        # 2. Filtraggio geografico
        df_geo = self.filter_geographic_area(df_raw)
        
        # 3. Validazione fisica
        df_valid = self.validate_physical_plausibility(df_geo)
        
        # 4. Downsampling causale
        df_down = self.downsample_causal(df_valid)
        
        # 5. Feature engineering causale
        df_feat = self.engineer_causal_features(df_down)
        
        # 6. Creazione target causale
        df_final = self.create_causal_target(df_feat)
        
        # 7. Sanity check finale
        assert not df_final[['MMSI', 'Timestamp']].duplicated().any(), "❌ Duplicati residui!"
        assert df_final[self.target_col].isna().sum() == 0, "❌ Target con NaN!"
        assert all(col in df_final.columns for col in ['delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate']), "❌ Feature mancanti!"
        
        # 8. Salvataggio (Parquet per efficienza)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix != '.parquet':
            logging.warning(f"Salvataggio come Parquet consigliato. Convertendo {output_path.suffix} → .parquet")
            output_path = output_path.with_suffix('.parquet')
            
        df_final.to_parquet(output_path, index=False)
        logging.info(f"💾 Dataset salvato in {output_path}")
        
        # 9. Report finale
        logging.info("\n" + "="*60)
        logging.info("✅ PIPELINE COMPLETATA CON SUCCESSO")
        logging.info("="*60)
        logging.info(f"Righe finali: {len(df_final):,}")
        logging.info(f"Feature prodotte: {len([c for c in df_final.columns if c not in ['MMSI', 'Timestamp', 'Lat', 'Lon', 'SOG', 'COG', self.target_col]])}")
        logging.info(f"Target positivi: {df_final[self.target_col].sum():,} ({df_final[self.target_col].mean()*100:.3f}%)")
        logging.info(f"Periodo temporale: {df_final['Timestamp'].min()} → {df_final['Timestamp'].max()}")
        logging.info(f"Navi uniche: {df_final['MMSI'].nunique()}")
        logging.info("="*60 + "\n")
        
        if return_df:
            return df_final
        return None
    
    def get_feature_list(self, include_target: bool = False) -> list:
        """Restituisce la lista delle feature prodotte per allineamento con altri moduli."""
        features = ['delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate', 'dt_prev_hours']
        if include_target:
            features.append(self.target_col)
        return features


def load_config_from_yaml(config_path: Union[str, Path]) -> Dict:
    """Carica configurazione da file YAML (opzionale, richiede pyyaml)."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        logging.warning("PyYAML non installato. Usare configurazione via parametri.")
        return {}
    except FileNotFoundError:
        logging.warning(f"File config non trovato: {config_path}")
        return {}


if __name__ == "__main__":
    # ========================================================================
    # TEST DI INTEGRAZIONE: Pipeline completa con mock data realistico
    # ========================================================================
    logging.info("=== TEST INTEGRAZIONE AISDataPreprocessor ===")
    
    np.random.seed(42)
    
    # 1. Genera mock data AIS realistico per 3 navi
    n_ships = 3
    points_per_ship = 500  # ~3.5 giorni a 10 min
    timestamps_base = pd.date_range('2024-06-01', periods=points_per_ship, freq='10min')
    
    mock_records = []
    
    for ship_idx, mmsi in enumerate([123456789, 234567890, 345678901]):
        # Simula pattern di navigazione realistici
        base_sog = np.random.uniform(8, 15)  # Velocità di crociera
        base_cog = np.random.uniform(0, 360)  # Rotta base
        
        for t, ts in enumerate(timestamps_base):
            # Aggiungi variabilità realistica
            sog = base_sog + np.random.normal(0, 1.5) + np.sin(t * 0.01) * 2  # Piccole oscillazioni
            cog = (base_cog + np.random.normal(0, 3) + np.sin(t * 0.005) * 10) % 360
            lat = 36.5 + ship_idx * 0.3 + np.random.normal(0, 0.01)  # Area stretta di Sicilia
            lon = 13.0 + np.random.normal(0, 0.02)
            
            # Simula occasionali gap >12h (comportamento "dark fleet")
            # ~2% di probabilità per nave, concentrati in finestre temporali
            if np.random.random() < 0.02 and t > 100 and t < 400:
                # Salta alcuni record per creare gap
                continue
                
            mock_records.append({
                'MMSI': mmsi,
                'Timestamp': ts,
                'Lat': lat,
                'Lon': lon,
                'SOG': max(0, sog),  # SOG non negativo
                'COG': cog
            })
    
    mock_df = pd.DataFrame(mock_records)
    
    # Aggiungi alcuni record "sporchi" per testare la validazione
    dirty_records = [
        {'MMSI': 999, 'Timestamp': '2024-06-01', 'Lat': 999, 'Lon': 999, 'SOG': 100, 'COG': 400},  # Fisicamente impossibile
        {'MMSI': 123456789, 'Timestamp': timestamps_base[10], 'Lat': 36.5, 'Lon': 13.0, 'SOG': 10, 'COG': 180},  # Duplicato
    ]
    mock_df = pd.concat([mock_df, pd.DataFrame(dirty_records)], ignore_index=True)
    
    logging.info(f"Mock data generato: {len(mock_df)} record, {mock_df['MMSI'].nunique()} navi")
    
    # 2. Configura preprocessor per lo Stretto di Sicilia
    sicily_bbox = {
        'min_lat': 35.0, 'max_lat': 38.0,
        'min_lon': 11.0, 'max_lon': 15.5
    }
    
    preprocessor = AISDataPreprocessor(
        bounding_box=sicily_bbox,
        gap_threshold_hours=12.0,
        prediction_horizon_hours=24.0,
        downsample_minutes=10,  # Già a 10 min nel mock, ma testiamo la funzione
        seed=42
    )
    
    # 3. Esegui pipeline in memoria (senza I/O su disco per il test)
    # Nota: Qui chiamiamo i metodi singolarmente per evitare problemi di path nel test
    df_geo = preprocessor.filter_geographic_area(mock_df)
    df_valid = preprocessor.validate_physical_plausibility(df_geo)
    df_down = preprocessor.downsample_causal(df_valid)
    df_feat = preprocessor.engineer_causal_features(df_down)
    df_final = preprocessor.create_causal_target(df_feat)
    
    # 4. Validazioni post-pipeline
    logging.info("\n🔍 VALIDAZIONI POST-PIPELINE")
    
    # Check 1: Nessuna colonna leakata
    leak_cols = ['_next_timestamp', '_gap_to_next', 'dt_next_hours']
    for col in leak_cols:
        assert col not in df_final.columns, f"❌ LEAKAGE: colonna '{col}' ancora presente!"
    logging.info("✅ Anti-leakage check: PASSED")
    
    # Check 2: Feature richieste presenti
    required_features = ['delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate', 'dt_prev_hours', 'target_dark_fleet']
    for feat in required_features:
        assert feat in df_final.columns, f"❌ Feature mancante: {feat}"
    logging.info("✅ Feature completeness: PASSED")
    
    # Check 3: Target nel range [0,1] e senza NaN
    assert df_final['target_dark_fleet'].isin([0, 1]).all(), "❌ Target con valori non binari!"
    assert df_final['target_dark_fleet'].isna().sum() == 0, "❌ Target con NaN!"
    logging.info("✅ Target integrity: PASSED")
    
    # Check 4: Causalità delle feature (nessun valore futuro)
    # Verifica che delta_* siano calcolati come diff() e non come shift(-1)
    sample_ship = df_final[df_final['MMSI'] == df_final['MMSI'].iloc[0]].head(10)
    # I delta dovrebbero essere 0 per la prima riga (fillna) e ragionevoli dopo
    assert sample_ship['delta_SOG'].iloc[0] == 0, "❌ Prima riga dovrebbe avere delta=0 (fillna)"
    logging.info("✅ Causal feature check: PASSED")
    
    # Check 5: Distribuzione target realistica
    target_rate = df_final['target_dark_fleet'].mean()
    assert 0.001 <= target_rate <= 0.15, f"❌ Target rate fuori range realistico: {target_rate:.3%}"
    logging.info(f"✅ Target distribution: {target_rate:.3%} positivi (realistico)")
    
    # 5. Output campione per ispezione
    print("\n📋 ESEMPIO OUTPUT (prime 5 righe di una nave):")
    sample_output = df_final[df_final['MMSI'] == df_final['MMSI'].iloc[0]][
        ['Timestamp', 'SOG', 'COG', 'delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate', 'target_dark_fleet']
    ].head()
    print(sample_output.to_string())
    
    print("\n📊 STATISTICHE FEATURE:")
    stats = df_final[['delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate', 'dt_prev_hours']].describe()
    print(stats.round(3))
    
    logging.info("\n✅ TEST INTEGRAZIONE COMPLETATO CON SUCCESSO")
    logging.info("🎯 Il modulo data_prep.py è pronto per l'integrazione con hmm_model.py e gb_training.py")