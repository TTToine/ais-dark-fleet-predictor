"""
src/gb_training.py
Modulo per l'addestramento, HPO e validazione del Gradient Boosting (LightGBM).
Rigorosa validazione temporale con gap causale e metriche per dataset sbilanciati.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import (
    average_precision_score, 
    log_loss, 
    roc_auc_score,
    brier_score_loss
)
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import logging
import warnings
import shap
from typing import List, Optional, Tuple

# Filtra warning specifici, non tutti
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TimeSeriesSplitWithGap:
    """
    TimeSeriesSplit personalizzato con gap temporale per prevenire leakage.
    
    Se l'horizon di previsione è H ore, il validation set deve iniziare
    almeno H ore dopo la fine del training set.
    """
    def __init__(self, n_splits: int = 5, gap_hours: float = 24.0, freq_min: int = 10):
        self.n_splits = n_splits
        self.gap_steps = int(gap_hours * 60 / freq_min)  # Converti ore in numero di step
        
    def split(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(X)
        # Dimensione approssimativa di ogni fold
        min_test_size = max(100, n_samples // (self.n_splits * 4))
        
        for i in range(self.n_splits):
            # Progressivamente espandi il training set
            train_end = int(n_samples * (i + 1) / (self.n_splits + 1))
            valid_start = train_end + self.gap_steps
            valid_end = min(valid_start + min_test_size, n_samples)
            
            if valid_end > n_samples or valid_start >= n_samples:
                continue
                
            train_idx = np.arange(0, train_end)
            valid_idx = np.arange(valid_start, valid_end)
            
            if len(valid_idx) < 10:  # Skip fold troppo piccolo
                continue
                
            yield train_idx, valid_idx


def set_global_seed(seed: int = 42):
    """Imposta seed globali per riproducibilità."""
    np.random.seed(seed)
    # LightGBM seed va nei parametri del modello
    # Optuna: usa sampler con seed se necessario


class DarkFleetPredictor:
    """
    Pipeline di addestramento LightGBM con HPO Optuna e validazione temporale causale.
    
    Supporta ablation study per dimostrare il valore aggiunto delle feature bayesiane.
    """
    
    def __init__(self, 
                 n_trials: int = 30, 
                 n_splits: int = 5,
                 gap_hours: float = 24.0,
                 freq_min: int = 10,
                 seed: int = 42):
        """
        Args:
            gap_hours: Horizon di previsione in ore (deve matchare data_prep.py)
            freq_min: Frequenza dei dati in minuti (per convertire gap in step)
        """
        set_global_seed(seed)
        
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.gap_hours = gap_hours
        self.seed = seed
        self.best_model = None
        self.best_params = None
        self.cv_splitter = TimeSeriesSplitWithGap(n_splits, gap_hours, freq_min)
        
        # Feature set completo (incluso output HMM)
        self.features_full = [
            'delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate', 'dt_prev_hours',
            'prob_regime_sospetto', 'incertezza_regime'
        ]
        # Feature set baseline (senza HMM, per ablation study)
        self.features_baseline = [
            'delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate', 'dt_prev_hours'
        ]
        self.target = 'target_dark_fleet'

    def prepare_data_for_cv(self, df: pd.DataFrame, 
                            feature_list: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara X e y con ordinamento temporale rigoroso."""
        logging.info("Preparazione dati per validazione temporale...")
        
        # Ordinamento temporale globale (fondamentale per TimeSeriesSplit)
        df_sorted = df.sort_values(by='Timestamp').reset_index(drop=True)
        
        # Drop NaN nel target (LightGBM non li accetta)
        df_sorted = df_sorted.dropna(subset=[self.target])
        
        features = feature_list if feature_list else self.features_full
        # Drop NaN nelle feature (LightGBM li gestisce, ma meglio essere espliciti)
        df_sorted = df_sorted.dropna(subset=features)
        
        X = df_sorted[features].copy()
        y = df_sorted[self.target].copy()
        
        logging.info(f"Dati pronti: {len(X)} campioni, {y.mean():.3%} positivi")
        return X, y

    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
                  feature_list: List[str]) -> float:
        """Funzione obiettivo per Optuna con validazione causale."""
        
        # Spazio di ricerca degli hyperparametri
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'seed': self.seed,  # Riproducibilità
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 128),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            # Scale pos weight: gestisce class imbalance, ottimizzato ma con baseline
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0)
        }

        scores = []
        
        # Custom CV con gap temporale
        for fold, (train_idx, valid_idx) in enumerate(self.cv_splitter.split(X)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            # Skip fold se manca la classe positiva (comune con target rari)
            if y_train.sum() == 0 or y_valid.sum() == 0:
                logging.debug(f"Fold {fold}: classe positiva assente, skip")
                continue

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

            # Training con early stopping per prevenire overfitting nel fold
            gbm = lgb.train(
                param,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=0)  # Silenzia output fold
                ]
            )

            # Predizioni e valutazione
            preds_proba = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
            
            # PR-AUC: metrica primaria per classi sbilanciate
            pr_auc = average_precision_score(y_valid, preds_proba)
            scores.append(pr_auc)
            
            # Log fold performance per debug
            logging.debug(f"Fold {fold}: PR-AUC={pr_auc:.4f}, best_iter={gbm.best_iteration}")

        if not scores:
            logging.warning("Nessun fold valido valutato")
            return 0.0

        # Optuna massimizza: ritorniamo la media PR-AUC
        return np.mean(scores)

    def optimize_and_train(self, X: pd.DataFrame, y: pd.Series,
                           feature_list: Optional[List[str]] = None) -> None:
        """Avvia HPO con Optuna e addestra il modello finale."""
        
        features = feature_list if feature_list else self.features_full
        logging.info(f"Avvio HPO con {len(features)} feature: {features}")
        
        # Ottimizzazione Optuna
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(
            lambda trial: self.objective(trial, X, y, features), 
            n_trials=self.n_trials, 
            show_progress_bar=True,
            n_jobs=4,
            catch=(RuntimeError,)
        )

        self.best_params = study.best_trial.params
        # Aggiungiamo parametri fissi non ottimizzati
        self.best_params.update({
            'objective': 'binary',
            'metric': 'binary_logloss', 
            'verbosity': -1,
            'seed': self.seed
        })

        logging.info(f"✅ Migliori parametri: {self.best_params}")
        logging.info(f"✅ Miglior PR-AUC in CV: {study.best_value:.4f}")

        # Training finale con early stopping su holdout interno
        logging.info("Addestramento modello finale con early stopping...")
        
        # Split 90/10 per early stopping (temporale: ultimi 10%)
        split_idx = int(len(X) * 0.9)
        X_train_final, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train_final, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        train_data = lgb.Dataset(X_train_final, label=y_train_final)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.best_model = lgb.train(
            self.best_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,  # Alto, fermato da early stopping
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        
        logging.info(f"✅ Modello finale: {self.best_model.best_iteration} boosting rounds")

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        if self.best_model is None:
            raise ValueError("Addestra prima il modello con optimize_and_train()")

        preds_proba = self.best_model.predict(X_test)
        
        # --- NUOVO: Conformal Prediction ---
        from src.utils import calculate_conformal_threshold
        try:
            # Calcoliamo la soglia per avere al massimo il 5% di Falsi Positivi
            # Nota: in produzione questo andrebbe calcolato su un validation set separato
            sicurezza_95_threshold = calculate_conformal_threshold(y_test.values, preds_proba, target_fpr=0.05)
            allarmi_generati = (preds_proba >= sicurezza_95_threshold).sum()
            logging.info(f"🛡️ Conformal Threshold (5% FPR): {sicurezza_95_threshold:.4f}")
            logging.info(f"🚨 Allarmi Dark Fleet scattati: {allarmi_generati} su {len(y_test)} navi")
        except Exception as e:
            logging.warning(f"Conformal prediction non calcolabile: {e}")
        # Metriche principali
        metrics = {
            'pr_auc': average_precision_score(y_test, preds_proba),
            'roc_auc': roc_auc_score(y_test, preds_proba),
            'log_loss': log_loss(y_test, preds_proba),
            'brier_score': brier_score_loss(y_test, preds_proba)
        }
        
        logging.info("\n" + "="*60)
        logging.info("📊 RISULTATI FINALI SUL TEST SET (FUTURO)")
        logging.info("="*60)
        for name, value in metrics.items():
            logging.info(f"{name:15s}: {value:.4f}")
        logging.info("="*60 + "\n")
        
        return metrics

    def plot_calibration(self, X_test: pd.DataFrame, y_test: pd.Series, n_bins: int = 10):
        """Plot di calibrazione per valutare l'affidabilità delle probabilità."""
        if self.best_model is None:
            raise ValueError("Addestra prima il modello")
            
        preds_proba = self.best_model.predict(X_test)
        
        plt.figure(figsize=(8, 6))
        CalibrationDisplay.from_predictions(y_test, preds_proba, n_bins=n_bins, ax=plt.gca())
        plt.title("Calibration Curve - Affidabilità Probabilità Predette")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("models/calibration_plot.png", dpi=300)
        logging.info("📈 Calibration plot salvato in models/calibration_plot.png")
        plt.show()

    def plot_feature_importance_shap(self, X_sample: pd.DataFrame, max_display: int = 10):
        if self.best_model is None:
            raise ValueError("Addestra prima il modello")
            
        logging.info("Calcolo SHAP values (potrebbe richiedere tempo)...")
        
        # Campiona per velocità
        if len(X_sample) > 1000:
            X_sample = X_sample.sample(n=1000, random_state=self.seed)
            
        # --- NUOVO: Filtro VIF ---
        from src.utils import filter_high_vif
        safe_features = filter_high_vif(X_sample, threshold=10.0)
        X_sample_safe = X_sample[safe_features]
        # -------------------------
        
        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(X_sample_safe)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample_safe, max_display=max_display, show=False)
        plt.title("SHAP Feature Importance (VIF Filtered)")
        plt.tight_layout()
        plt.savefig("models/shap_importance_vif.png", dpi=300)
        logging.info("📊 SHAP plot salvato in models/shap_importance_vif.png")
        plt.show()

    def ablation_study(self, test_df: pd.DataFrame, train_df: pd.DataFrame, 
                      n_trials_ablation: int = 10) -> dict:
        """
        Confronto strutturato: Logistic Regression vs GB Baseline vs GB Enhanced.
        Include test di significatività statistica tramite Grouped Bootstrap.
        """
        logging.info("🔬 Avvio Ablation Study: Logistic Baseline vs GB Baseline vs GB Enhanced")
        
        results = {}
        # Creiamo un DataFrame per salvare le predizioni sul Test Set
        df_bootstrap = test_df[['MMSI', 'target_dark_fleet']].copy()
        
        # Definizione delle configurazioni: (Nome, Feature List, Usa LogisticRegression)
        configurations = [
            ("logistic_baseline", self.features_baseline, True),
            ("gb_baseline", self.features_baseline, False),
            ("gb_enhanced", self.features_full, False)
        ]
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import average_precision_score
        
        for config_name, features, use_logistic in configurations:
            logging.info(f"\n--- Addestramento Configurazione: {config_name} ---")
            
            X_train, y_train = self.prepare_data_for_cv(train_df, feature_list=features)
            X_test, y_test = self.prepare_data_for_cv(test_df, feature_list=features)
            
            if use_logistic:
                logging.info("Addestramento Logistic Regression (Dumb Baseline)...")
                # La regressione logistica richiede dati standardizzati e senza NaN
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train.fillna(0))
                X_test_scaled = scaler.transform(X_test.fillna(0))
                
                lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=self.seed)
                lr_model.fit(X_train_scaled, y_train)
                
                # Salviamo le probabilità per la classe positiva (1)
                preds_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
                pr_auc = average_precision_score(y_test, preds_proba)
                
                # Salvataggio risultati
                df_bootstrap[f'pred_{config_name}'] = preds_proba
                results[config_name] = {'model': lr_model, 'pr_auc': pr_auc}
                
                logging.info(f"✅ {config_name} completato: PR-AUC={pr_auc:.4f}")
                continue
                
            # --- Se non è logistica, addestra Gradient Boosting (LightGBM) ---
            predictor = DarkFleetPredictor(
                n_trials=n_trials_ablation,
                n_splits=self.n_splits,
                gap_hours=self.gap_hours,
                seed=self.seed
            )
            
            predictor.optimize_and_train(X_train, y_train, feature_list=features)
            metrics = predictor.evaluate_model(X_test, y_test)
            
            # Salviamo le predizioni per il bootstrap
            preds_proba = predictor.best_model.predict(X_test)
            df_bootstrap[f'pred_{config_name}'] = preds_proba
            
            results[config_name] = {
                'model': predictor.best_model,
                'params': predictor.best_params,
                'metrics': metrics,
                'pr_auc': metrics['pr_auc']
            }
            logging.info(f"✅ {config_name} completato: PR-AUC={metrics['pr_auc']:.4f}")
        
        # ==========================================
        # STATISTICAL SIGNIFICANCE TEST (BOOTSTRAP)
        # ==========================================
        from src.utils import compare_models_grouped_bootstrap
        
        logging.info("\n📊 Calcolo Significatività Statistica (Grouped Bootstrap)...")
        # Confrontiamo il GB potenziato (A) contro il GB standard (B)
        boot_stats = compare_models_grouped_bootstrap(
            df_results=df_bootstrap,
            group_col='MMSI',
            target_col='target_dark_fleet',
            pred_col_a='pred_gb_enhanced', 
            pred_col_b='pred_gb_baseline',
            n_bootstraps=1000,
            seed=self.seed
        )
        
        logging.info(f"🎯 Delta PR-AUC Medio (Enhanced vs GB Baseline): {boot_stats['delta_mean']:+.4f}")
        logging.info(f"📈 95% Confidence Interval: [{boot_stats['ci_lower']:+.4f}, {boot_stats['ci_upper']:+.4f}]")
        
        if boot_stats['significant']:
            logging.info("✅ VITTORIA SIGNIFICATIVA: Il modello HMM migliora in modo statisticamente robusto le performance rispetto alla baseline LightGBM.")
        else:
            logging.warning("⚠️ PAREGGIO STATISTICO: Il CI include lo zero. Il miglioramento potrebbe essere casuale.")
            
        results['bootstrap_stats'] = boot_stats
        return results
        # ==========================================
        # STATISTICAL SIGNIFICANCE TEST (BOOTSTRAP)
        # ==========================================
        from src.utils import compare_models_grouped_bootstrap
        
        logging.info("\n📊 Calcolo Significatività Statistica (Grouped Bootstrap)...")
        boot_stats = compare_models_grouped_bootstrap(
            df_results=df_bootstrap,
            group_col='MMSI',
            target_col='target_dark_fleet',
            pred_col_a='pred_enhanced',
            pred_col_b='pred_baseline',
            n_bootstraps=1000,
            seed=self.seed
        )
        
        logging.info(f"🎯 Delta PR-AUC Medio: {boot_stats['delta_mean']:+.4f}")
        logging.info(f"📈 95% Confidence Interval: [{boot_stats['ci_lower']:+.4f}, {boot_stats['ci_upper']:+.4f}]")
        
        if boot_stats['significant']:
            logging.info("✅ VITTORIA SIGNIFICATIVA: Il modello HMM migliora in modo statisticamente robusto le performance.")
        else:
            logging.warning("⚠️ PAREGGIO STATISTICO: Il CI include lo zero. Il miglioramento potrebbe essere casuale.")
            
        results['bootstrap_stats'] = boot_stats
        return results


if __name__ == "__main__":
    # ========================================================================
    # TEST DI INTEGRAZIONE: Pipeline completa con mock data realistico
    # ========================================================================
    logging.info("=== TEST INTEGRAZIONE DarkFleetPredictor ===")
    
    np.random.seed(42)
    n_samples = 2000
    
    # Simulazione dati AIS con struttura temporale e correlazioni realistiche
    timestamps = pd.date_range('2024-06-01', periods=n_samples, freq='10min')
    
    # Feature cinematiche con autocorrelazione temporale (simulazione realistica)
    def generate_temporal_series(mean, std, n, autocorr=0.7):
        series = np.zeros(n)
        series[0] = np.random.normal(mean, std)
        for t in range(1, n):
            series[t] = autocorr * series[t-1] + (1-autocorr) * np.random.normal(mean, std)
        return series
    
    mock_data = pd.DataFrame({
        'Timestamp': timestamps,
        'delta_SOG': generate_temporal_series(0, 1.5, n_samples),
        'delta_COG': generate_temporal_series(0, 8, n_samples), 
        'speed_acc': generate_temporal_series(0, 0.8, n_samples),
        'turn_rate': generate_temporal_series(0, 4, n_samples),
        'dt_prev_hours': np.abs(generate_temporal_series(0.16, 0.03, n_samples, autocorr=0.9)),
    })
    
    # Feature bayesiane simulate: prob_regime_sospetto correlata con pattern di "sospetto"
    # Creiamo un segnale latente che simula il comportamento "sospetto"
    latent_suspicious = (
        (mock_data['speed_acc'].abs() < 0.3).astype(int) * 0.4 +  # Bassa accelerazione
        (mock_data['turn_rate'].abs() > 3).astype(int) * 0.4 +    # Alte virate
        np.random.normal(0, 0.1, n_samples)  # Rumore
    )
    latent_suspicious = (latent_suspicious - latent_suspicious.min()) / (latent_suspicious.max() - latent_suspicious.min())
    
    mock_data['prob_regime_sospetto'] = latent_suspicious
    mock_data['incertezza_regime'] = np.random.uniform(0.05, 0.25, n_samples)  # Incertezza variabile
    
    # Target: blackout intenzionale, debolmente correlato con regime sospetto + rumore
    # Probabilità di blackout aumenta con prob_regime_sospetto ma non è deterministico
    base_prob = 0.02  # Rarità dei blackout
    prob_blackout = np.clip(base_prob + mock_data['prob_regime_sospetto'] * 0.15, 0, 1)
    mock_data['target_dark_fleet'] = np.random.binomial(1, prob_blackout)
    
    logging.info(f"Dataset mock: {len(mock_data)} righe, target rate: {mock_data['target_dark_fleet'].mean():.3%}")
    
    # ========================================================================
    # ESECUZIONE PIPELINE
    # ========================================================================
    predictor = DarkFleetPredictor(
        n_trials=5,      # Pochi trial per test veloce
        n_splits=3,
        gap_hours=2.0,   # Gap ridotto per test (in produzione: 24h)
        freq_min=10,
        seed=42
    )
    
    # Split temporale causale: ultimi 20% come test set assoluto
    split_idx = int(len(mock_data) * 0.8)
    train_df = mock_data.iloc[:split_idx].copy()
    test_df = mock_data.iloc[split_idx:].copy()
    
    # Preparazione dati
    X_train, y_train = predictor.prepare_data_for_cv(train_df)
    X_test, y_test = predictor.prepare_data_for_cv(test_df)
    
    # HPO + Training
    predictor.optimize_and_train(X_train, y_train)
    
    # Valutazione finale
    metrics = predictor.evaluate_model(X_test, y_test)
    
    # Visualizzazioni (salvate su file)
    try:
        predictor.plot_calibration(X_test, y_test)
        predictor.plot_feature_importance_shap(X_test)
    except Exception as e:
        logging.warning(f"Plot generation skipped: {e}")
    
    logging.info("\n✅ TEST INTEGRAZIONE COMPLETATO CON SUCCESSO")
    logging.info("📁 Output salvati in: models/")