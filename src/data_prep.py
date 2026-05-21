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

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =============================================================================
# MMSI normalization (latent bug killer)
# =============================================================================
# Real-world AIS MMSIs are 9-digit integers. Pandas operations like
# `resample().last()` introduce NaT/NaN rows that promote int64 → float64
# silently. The promotion breaks downstream joins on MMSI (float-vs-int
# equality is type-dependent in subtle ways). This helper is the single
# choke-point: call it after every data materialization step.
# =============================================================================
def _ensure_mmsi_int64(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Rinomina ``mmsi`` → ``MMSI``, dropna su MMSI, cast a int64. Idempotente.

    Args:
        df: DataFrame in arrivo da load_data / simulator / resample.
        context: stringa libera per i log (es. "simulator", "load_data").

    Returns:
        DataFrame con ``df['MMSI'].dtype == np.int64`` garantito.
    """
    if 'mmsi' in df.columns and 'MMSI' not in df.columns:
        df = df.rename(columns={'mmsi': 'MMSI'})
    if 'MMSI' not in df.columns:
        return df
    if df['MMSI'].dtype == np.int64:
        return df
    df = df.copy()
    coerced = pd.to_numeric(df['MMSI'], errors='coerce')
    n_nan = int(coerced.isna().sum())
    if n_nan:
        logging.warning(
            f"_ensure_mmsi_int64[{context}]: scarto {n_nan} righe con MMSI non numerica."
        )
        df = df.loc[coerced.notna()].copy()
        coerced = coerced.dropna()
    df['MMSI'] = coerced.astype(np.int64)
    return df


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
            'prob_regime_sospetto', 'incertezza_regime'  # Aggiunti da bayesian_mixture.py
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
        df = _ensure_mmsi_int64(df, context="load_data")
        if 'MMSI' in df.columns:
            assert df['MMSI'].dtype == np.int64, "load_data: MMSI not int64 after coercion"
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
        # NB: usiamo `group_keys=True` (default implicito) per garantire che
        # MMSI compaia come colonna dopo `reset_index()`. `group_keys=False`
        # rimuoveva MMSI dal MultiIndex e da downstream groupby/labeling.
        df_down = (df_indexed
                   .groupby('MMSI')
                   .resample(f'{self.downsample_minutes}min')
                   .last()
                   .dropna(subset=['Lat', 'Lon', 'SOG', 'COG'])
                   .reset_index())
        
        df_down['time_diff_min'] = df_down.groupby('MMSI')['Timestamp'].diff().dt.total_seconds() / 60

        # resample().last() promuove silenziosamente int64 → float64.
        # Riportiamo MMSI a int64 prima di restituirlo, così downstream
        # label_sliding_window / Bayesian inference vedono il tipo corretto.
        df_down = _ensure_mmsi_int64(df_down, context="downsample_causal")

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
        
        # Target: gap >= threshold E che termina entro horizon_hours dal gap stesso
        df[self.target_col] = (
            (df['_gap_to_next'] >= gap_hours) &
            (df['_gap_to_next'] <= gap_hours + horizon_hours)
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

    # =====================================================================
    # ETICHETTATURA RETROSPETTIVA vs DEPLOYABLE
    # =====================================================================
    # `create_causal_target` (sopra) implementa la "Last-Ping" framing
    # retrospettiva: Y=1 SOLO per l'ultimo ping prima di un blackout. Non
    # deployable in real-time (richiede attesa di `gap_threshold_hours`
    # per sapere se un ping era "l'ultimo").
    #
    # `label_sliding_window` (qui sotto, esposta anche come funzione modulo)
    # è la formulazione deployable: Y=1 a tempo t se un blackout
    # > gap_threshold inizierà ENTRO `horizon_minutes` dopo t. Più ping
    # precedenti il blackout ricevono label positiva; il modello deve
    # distinguerli dai ping seguiti da operatività normale.
    # =====================================================================
    def label_sliding_window(self, df: pd.DataFrame,
                             horizon_minutes: float = 60.0) -> pd.DataFrame:
        """Sliding-window deployable labeler.

        Y(t)=1 se esiste un last-ping-before-blackout in (t, t+horizon].
        Il join è per MMSI; richiede ``df['MMSI'].dtype == int64`` per
        evitare mismatch silenziosi (float vs int causa join falliti).
        """
        assert df['MMSI'].dtype == np.int64, (
            f"MMSI must be int64 at labeling time, got {df['MMSI'].dtype}. "
            f"This causes silent join failures. Check load_data() / simulator."
        )
        logging.info(
            f"Sliding-window labeling: blackout>{self.gap_threshold.total_seconds()/3600}h "
            f"entro {horizon_minutes} min."
        )
        df = df.copy().sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)

        # 1) Marca il last-ping-before-blackout (criterio del Last-Ping framing).
        next_ts = df.groupby('MMSI')['Timestamp'].shift(-1)
        gap_h = (next_ts - df['Timestamp']).dt.total_seconds() / 3600.0
        gap_threshold_h = self.gap_threshold.total_seconds() / 3600.0
        is_last = (gap_h >= gap_threshold_h).fillna(False)
        df['_is_last_ping_before_blackout'] = is_last.astype(int)

        # 2) Per ogni ping (t, MMSI), label=1 se esiste un last-ping in (t, t+H].
        horizon = pd.Timedelta(minutes=horizon_minutes)
        labels = np.zeros(len(df), dtype=int)

        for _, idx_group in df.groupby('MMSI', sort=False).groups.items():
            sub = df.loc[idx_group, ['Timestamp', '_is_last_ping_before_blackout']]
            ts_arr = sub['Timestamp'].values
            last_arr = sub['_is_last_ping_before_blackout'].values
            last_positions = np.flatnonzero(last_arr == 1)
            if len(last_positions) == 0:
                continue
            last_ts = ts_arr[last_positions]
            for i, t in enumerate(ts_arr):
                # last-ping in (t, t+H] — escludo se stesso (t < t_last_ping)
                upper = pd.Timestamp(t) + horizon
                in_window = (last_ts > t) & (last_ts <= np.datetime64(upper))
                if in_window.any():
                    labels[idx_group[i]] = 1

        df[self.target_col] = labels
        df = df.drop(columns=['_is_last_ping_before_blackout'])

        pos = int(df[self.target_col].sum())
        rate = pos / max(len(df), 1) * 100
        logging.info(
            f"Sliding-window target: {pos:,} positivi ({rate:.3f}%) su {len(df):,}."
        )
        if rate < 0.1:
            logging.warning(f"⚠️ Classe positiva molto rara ({rate:.3f}%).")
        return df

    def label_last_ping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias retrocompatibile per ``create_causal_target`` (Last-Ping framing).

        Esposto per simmetria con ``label_sliding_window``. Questa etichettatura
        è RETROSPETTIVA (non deployable as-is).
        """
        assert df['MMSI'].dtype == np.int64, (
            f"MMSI must be int64 at labeling time, got {df['MMSI'].dtype}. "
            f"This causes silent join failures. Check load_data() / simulator."
        )
        return self.create_causal_target(df)

    def add_spatial_features(self, df: pd.DataFrame,
                             coastline_path: Optional[str] = None,
                             fishing_zones_path: Optional[str] = None,
                             commercial_routes_path: Optional[str] = None) -> pd.DataFrame:
        """
        Aggiunge feature spaziali al DataFrame: distanza dalla costa, da zone di pesca
        note (GFW) e da rotte commerciali principali.

        Le distanze sono calcolate in miglia nautiche (nm) tramite proiezione metrica
        (EPSG:3857) e successiva conversione. Richiede geopandas.

        Se coastline_path non è fornito, usa i dati naturalearth inclusi in geopandas
        come fallback (bassa risoluzione, sufficiente per l'analisi a scala regionale).
        I path per fishing_zones e commercial_routes sono opzionali: se assenti, le
        feature corrispondenti non vengono calcolate.

        Feature prodotte (se i dati sono disponibili):
            - dist_coast_nm
            - dist_fishing_zone_nm   (richiede fishing_zones_path)
            - dist_commercial_route_nm (richiede commercial_routes_path)
        """
        try:
            import geopandas as gpd
            from shapely.ops import unary_union
        except ImportError:
            raise ImportError(
                "geopandas e shapely sono richiesti per le feature spaziali: "
                "pip install geopandas"
            )

        logging.info("Calcolo feature spaziali (distanze in miglia nautiche)...")
        df = df.copy()

        # Crea GeoDataFrame dei punti nave in CRS metrico
        gdf_pts = gpd.GeoDataFrame(
            df[['Lat', 'Lon']].copy(),
            geometry=gpd.points_from_xy(df['Lon'], df['Lat']),
            crs='EPSG:4326'
        ).to_crs('EPSG:3857')

        def _load_and_project(path: Optional[str], fallback_fn=None):
            if path:
                gdf = gpd.read_file(path)
            elif fallback_fn is not None:
                gdf = fallback_fn()
            else:
                return None
            return unary_union(gdf.to_crs('EPSG:3857').geometry)

        def _dist_nm(geometry_union) -> np.ndarray:
            """Distanza vettorizzata dalla geometria in miglia nautiche."""
            dist_m = gdf_pts.geometry.distance(geometry_union).values
            return dist_m / 1852.0

        # ── Coastline ──────────────────────────────────────────────────────────
        def _naturalearth_coast():
            try:
                from geodatasets import get_path
                return gpd.read_file(get_path('naturalearth.land'))
            except Exception:
                return gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        if coastline_path is None:
            logging.warning("coastline_path non fornito: uso naturalearth_lowres (bassa risoluzione).")

        coast_union = _load_and_project(coastline_path, _naturalearth_coast)
        if coast_union is not None:
            logging.info("  Calcolo dist_coast_nm...")
            df['dist_coast_nm'] = _dist_nm(coast_union)

        # ── Fishing zones (GFW shapefile) ──────────────────────────────────────
        fish_union = _load_and_project(fishing_zones_path)
        if fish_union is not None:
            logging.info("  Calcolo dist_fishing_zone_nm...")
            df['dist_fishing_zone_nm'] = _dist_nm(fish_union)
        else:
            logging.info("  fishing_zones_path non fornito: dist_fishing_zone_nm non calcolata.")

        # ── Commercial routes ──────────────────────────────────────────────────
        routes_union = _load_and_project(commercial_routes_path)
        if routes_union is not None:
            logging.info("  Calcolo dist_commercial_route_nm...")
            df['dist_commercial_route_nm'] = _dist_nm(routes_union)
        else:
            logging.info("  commercial_routes_path non fornito: dist_commercial_route_nm non calcolata.")

        spatial_feats = [c for c in
                         ['dist_coast_nm', 'dist_fishing_zone_nm', 'dist_commercial_route_nm']
                         if c in df.columns]
        logging.info(f"Feature spaziali calcolate: {spatial_feats}")
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


# =============================================================================
# Module-level convenience wrappers (per `from src.data_prep import ...`)
# =============================================================================
_DEFAULT_BBOX_WORLD = {
    "min_lat": -90.0, "max_lat": 90.0, "min_lon": -180.0, "max_lon": 180.0,
}


def label_sliding_window(df: pd.DataFrame,
                         horizon_minutes: float = 60.0,
                         gap_threshold_hours: float = 12.0,
                         prediction_horizon_hours: float = 24.0,
                         downsample_minutes: int = 10) -> pd.DataFrame:
    """Module-level wrapper: identico a ``AISDataPreprocessor.label_sliding_window``."""
    pp = AISDataPreprocessor(
        bounding_box=_DEFAULT_BBOX_WORLD,
        gap_threshold_hours=gap_threshold_hours,
        prediction_horizon_hours=prediction_horizon_hours,
        downsample_minutes=downsample_minutes,
    )
    return pp.label_sliding_window(df, horizon_minutes=horizon_minutes)


def label_last_ping(df: pd.DataFrame,
                    gap_threshold_hours: float = 12.0,
                    prediction_horizon_hours: float = 24.0,
                    downsample_minutes: int = 10) -> pd.DataFrame:
    """Module-level wrapper: identico a ``AISDataPreprocessor.label_last_ping``."""
    pp = AISDataPreprocessor(
        bounding_box=_DEFAULT_BBOX_WORLD,
        gap_threshold_hours=gap_threshold_hours,
        prediction_horizon_hours=prediction_horizon_hours,
        downsample_minutes=downsample_minutes,
    )
    return pp.label_last_ping(df)


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