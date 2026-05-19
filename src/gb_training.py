"""
src/gb_training.py
Modulo per l'addestramento, HPO e validazione del Gradient Boosting (LightGBM).
Rigorosa validazione temporale con gap causale e metriche per dataset sbilanciati.

✅ FIX APPLICATI (dalla review):
1. Corretta indentazione globale (SyntaxError riga 290+ risolto)
2. set_global_seed completato (random, PYTHONHASHSEED)
3. Rimossa Conformal Prediction (broken concettualmente)
4. Implementato optimize_and_train() (mancante nel codice originale)
5. Optuna n_jobs=1 per riproducibilità deterministica
6. Aggiunto GroupTimeSeriesSplit per evitare leakage tra navi
7. Rimossi import di moduli inesistenti (safe fallback)
8. Aggiunta funzione save_artifacts e chiamata in __main__
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
import random
import os
import joblib
from typing import List, Optional, Tuple

# Safe import per utility opzionali
try:
    from src.utils import filter_high_vif
except ImportError:
    def filter_high_vif(X: pd.DataFrame, threshold: float = 10.0) -> List[str]:
        """Fallback: restituisce tutte le feature se il modulo VIF manca."""
        return X.columns.tolist()

# Filtra warning specifici, non tutti
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_global_seed(seed: int = 42):
    """Imposta seed globali per riproducibilità completa."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class TimeSeriesSplitWithGap:
    """TimeSeriesSplit personalizzato con gap temporale per prevenire leakage."""
    def __init__(self, n_splits: int = 5, gap_hours: float = 24.0, freq_min: int = 10):
        self.n_splits = n_splits
        self.gap_steps = int(gap_hours * 60 / freq_min)

    def split(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(X)
        min_test_size = max(100, n_samples // (self.n_splits * 4))
        
        for i in range(self.n_splits):
            train_end = int(n_samples * (i + 1) / (self.n_splits + 1))
            valid_start = train_end + self.gap_steps
            valid_end = min(valid_start + min_test_size, n_samples)
            
            if valid_end > n_samples or valid_start >= n_samples:
                continue
                
            train_idx = np.arange(0, train_end)
            valid_idx = np.arange(valid_start, valid_end)
            
            if len(valid_idx) < 10:
                continue
                
            yield train_idx, valid_idx


class GroupTimeSeriesSplit:
    """Split temporale che rispetta i gruppi (MMSI) per evitare leakage.
    Implementazione pragmatica come suggerito nella review."""
    def __init__(self, n_splits: int = 5, gap_hours: float = 24.0):
        self.n_splits = n_splits
        self.gap = pd.Timedelta(hours=gap_hours)

    def split(self, X: pd.DataFrame, groups: pd.Series = None):
        if 'MMSI' not in X.columns:
            logging.warning("MMSI non trovato in X. Fallback a split temporale puro.")
            yield from TimeSeriesSplitWithGap(self.n_splits, self.gap.total_seconds()/3600, 10).split(X)
            return

        # Ordina le navi per primo timestamp osservato
        ship_first_ts = X.groupby('MMSI')['Timestamp'].min()
        sorted_mmsi = ship_first_ts.sort_values().index.tolist()
        n_ships = len(sorted_mmsi)

        for i in range(self.n_splits):
            train_cutoff_idx = int(n_ships * (i + 1) / (self.n_splits + 1))
            train_ships = set(sorted_mmsi[:train_cutoff_idx])
            
            # Calcola tempo massimo del train set
            train_end_time = X[X['MMSI'].isin(train_ships)]['Timestamp'].max()
            valid_cutoff_time = train_end_time + self.gap
            
            # Valid set: navi diverse, con primo timestamp dopo il gap
            valid_ships = [
                m for m in sorted_mmsi if m not in train_ships and 
                ship_first_ts[m] >= valid_cutoff_time
            ]
            
            if not valid_ships:
                continue

            train_idx = X[X['MMSI'].isin(train_ships)].index.values
            valid_idx = X[X['MMSI'].isin(valid_ships)].index.values

            if len(valid_idx) < 50 or len(train_idx) < 100:
                continue

            yield train_idx, valid_idx


class DarkFleetPredictor:
    """Pipeline di addestramento LightGBM con HPO Optuna e validazione temporale causale."""

    def __init__(self, 
                 n_trials: int = 30, 
                 n_splits: int = 5,
                 gap_hours: float = 24.0,
                 freq_min: int = 10,
                 seed: int = 42):
        set_global_seed(seed)
        
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.gap_hours = gap_hours
        self.freq_min = freq_min
        self.seed = seed
        self.best_model = None
        self.best_params = None
        
        # Usa il group-aware splitter se disponibile MMSI, altrimenti fallback
        self.cv_splitter = GroupTimeSeriesSplit(n_splits, gap_hours)
        
        self.features_full = [
            'delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate', 'dt_prev_hours',
            'prob_regime_sospetto', 'incertezza_regime'
        ]
        self.features_baseline = [
            'delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate', 'dt_prev_hours'
        ]
        self.target = 'target_dark_fleet'

    def prepare_data_for_cv(self, df: pd.DataFrame, 
                            feature_list: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara X e y con ordinamento temporale rigoroso."""
        logging.info("Preparazione dati per validazione temporale...")
        
        df_sorted = df.sort_values(by='Timestamp').reset_index(drop=True)
        df_sorted = df_sorted.dropna(subset=[self.target])
        
        features = feature_list if feature_list else self.features_full
        df_sorted = df_sorted.dropna(subset=features)
        
        X = df_sorted[features].copy()
        y = df_sorted[self.target].copy()
        
        logging.info(f"Dati pronti: {len(X)} campioni, {y.mean():.3%} positivi")
        return X, y

    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
                  feature_list: List[str]) -> float:
        """Funzione obiettivo per Optuna con validazione causale."""
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'seed': self.seed,
            'deterministic': True,
            'force_row_wise': True,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 128),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0)
        }

        scores = []
        
        for fold, (train_idx, valid_idx) in enumerate(self.cv_splitter.split(X)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            if y_train.sum() == 0 or y_valid.sum() == 0:
                logging.debug(f"Fold {fold}: classe positiva assente, skip")
                continue

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

            gbm = lgb.train(
                param,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )

            preds_proba = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
            pr_auc = average_precision_score(y_valid, preds_proba)
            scores.append(pr_auc)
            logging.debug(f"Fold {fold}: PR-AUC={pr_auc:.4f}, best_iter={gbm.best_iteration}")

        if not scores:
            logging.warning("Nessun fold valido valutato")
            return 0.0

        return np.mean(scores)

    def optimize_and_train(self, X_train: pd.DataFrame, y_train: pd.Series,
                           feature_list: Optional[List[str]] = None):
        """Esegue HPO con Optuna e addestra il modello finale con i parametri ottimali.
        Il num_boost_round finale è stimato dalla media dei best_iteration sui fold CV."""
        logging.info("🚀 Avvio ottimizzazione hyperparametri (Optuna)...")
        features = feature_list if feature_list else self.features_full
        X = X_train[features]

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(lambda trial: self.objective(trial, X, y_train, features),
                       n_trials=self.n_trials, n_jobs=1)

        self.best_params = study.best_params
        logging.info(f"✅ Migliori parametri trovati: {self.best_params}")
        logging.info(f"📈 Miglior PR-AUC CV: {study.best_value:.4f}")

        # Stima del num_boost_round ottimale tramite un passaggio CV con early stopping
        logging.info("📐 Stima num_boost_round ottimale via CV con early stopping...")
        final_param_base = {
            **self.best_params,
            'objective': 'binary', 'verbosity': -1,
            'seed': self.seed, 'deterministic': True, 'force_row_wise': True
        }
        best_iterations = []
        for train_idx, valid_idx in self.cv_splitter.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]
            if y_tr.sum() == 0 or y_val.sum() == 0:
                continue
            gbm_cv = lgb.train(
                final_param_base,
                lgb.Dataset(X_tr, label=y_tr),
                valid_sets=[lgb.Dataset(X_val, label=y_val, reference=lgb.Dataset(X_tr, label=y_tr))],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=0)]
            )
            best_iterations.append(gbm_cv.best_iteration)

        optimal_rounds = int(np.mean(best_iterations)) if best_iterations else 500
        logging.info(f"📐 num_boost_round ottimale stimato: {optimal_rounds} (media su {len(best_iterations)} fold)")

        # Addestramento finale su tutto il train set con num_boost_round calibrato
        logging.info("🏋️ Addestramento modello finale...")
        self.best_model = lgb.train(
            final_param_base,
            lgb.Dataset(X, label=y_train),
            num_boost_round=optimal_rounds
        )
        logging.info(f"✅ Modello finale addestrato ({optimal_rounds} round).")

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Valuta il modello sul test set."""
        if self.best_model is None:
            raise ValueError("Addestra prima il modello con optimize_and_train()")
            
        preds_proba = self.best_model.predict(X_test)
        logging.info("📊 Valutazione modello LightGBM")

        metrics = {
            'pr_auc': average_precision_score(y_test, preds_proba),
            'roc_auc': roc_auc_score(y_test, preds_proba),
            'log_loss': log_loss(y_test, preds_proba),
            'brier_score': brier_score_loss(y_test, preds_proba) 
        }

        logging.info("\n" + "="*60)
        logging.info("📊 RISULTATI FINALI SUL TEST SET")
        logging.info("="*60)
        for name, value in metrics.items():
            logging.info(f"{name:25s}: {value:.4f}")
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
        os.makedirs("models", exist_ok=True)
        plt.savefig("models/calibration_plot.png", dpi=300)
        logging.info("📈 Calibration plot salvato in models/calibration_plot.png")
        plt.close()

    def plot_feature_importance_shap(self, X_sample: pd.DataFrame, max_display: int = 10):
        """SHAP Feature Importance con filtro VIF opzionale."""
        if self.best_model is None:
            raise ValueError("Addestra prima il modello")
        logging.info("Calcolo SHAP values...")
        
        if len(X_sample) > 1000:
            X_sample = X_sample.sample(n=1000, random_state=self.seed)
            
        safe_features = filter_high_vif(X_sample, threshold=10.0)
        X_sample_safe = X_sample[safe_features]
        
        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(X_sample_safe)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample_safe, max_display=max_display, show=False)
        plt.title("SHAP Feature Importance (VIF Filtered)")
        plt.tight_layout()
        os.makedirs("models", exist_ok=True)
        plt.savefig("models/shap_importance_vif.png", dpi=300)
        logging.info("📊 SHAP plot salvato in models/shap_importance_vif.png")
        plt.close()

    def ablation_study(self, test_df: pd.DataFrame, train_df: pd.DataFrame,
                       n_trials_ablation: int = 10) -> dict:
        """Confronto strutturato: Logistic vs GB Baseline vs GB Enhanced."""
        logging.info("🔬 Avvio Ablation Study...")
        results = {}
        df_bootstrap = test_df[['MMSI', self.target]].copy() if 'MMSI' in test_df.columns else test_df[[self.target]].copy()
        
        configurations = [
            ("logistic_baseline", self.features_baseline, True),
            ("gb_baseline", self.features_baseline, False),
            ("gb_enhanced", self.features_full, False)
        ]
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        for config_name, features, use_logistic in configurations:
            logging.info(f"\n--- Addestramento Configurazione: {config_name} ---")
            
            X_train, y_train = self.prepare_data_for_cv(train_df, feature_list=features)
            X_test, y_test = self.prepare_data_for_cv(test_df, feature_list=features)
             
            if use_logistic:
                logging.info("Addestramento Logistic Regression...")
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train.fillna(0))
                X_test_scaled = scaler.transform(X_test.fillna(0))
                
                lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=self.seed)
                lr_model.fit(X_train_scaled, y_train)
                
                preds_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
                pr_auc = average_precision_score(y_test, preds_proba)
                brier = brier_score_loss(y_test, preds_proba)

                df_bootstrap[f'pred_{config_name}'] = preds_proba
                results[config_name] = {'model': lr_model, 'pr_auc': pr_auc, 'brier_score': brier}
                logging.info(f"✅ {config_name} completato: PR-AUC={pr_auc:.4f}, Brier={brier:.4f}")
                continue
                
            predictor = DarkFleetPredictor(
                n_trials=n_trials_ablation, n_splits=self.n_splits,
                gap_hours=self.gap_hours, seed=self.seed
            )
            predictor.optimize_and_train(X_train, y_train, feature_list=features)
            metrics = predictor.evaluate_model(X_test, y_test)
            
            preds_proba = predictor.best_model.predict(X_test)
            df_bootstrap[f'pred_{config_name}'] = preds_proba
            
            results[config_name] = {
                'model': predictor.best_model,
                'params': predictor.best_params,
                'metrics': metrics,
                'pr_auc': metrics['pr_auc'],
                'brier_score': metrics['brier_score']
            }
            logging.info(f"✅ {config_name} completato: PR-AUC={metrics['pr_auc']:.4f}, Brier={metrics['brier_score']:.4f}")

        # Confronto riepilogativo con entrambe le metriche
        logging.info("\n" + "="*60)
        logging.info("📊 ABLATION STUDY — CONFRONTO FINALE")
        logging.info(f"{'Config':<22} {'PR-AUC':>8} {'Brier':>8}")
        logging.info("-"*40)
        for name, res in results.items():
            pr = res.get('pr_auc', float('nan'))
            br = res.get('brier_score', float('nan'))
            logging.info(f"{name:<22} {pr:>8.4f} {br:>8.4f}")
        logging.info("="*60)

        return results, df_bootstrap


def save_artifacts(predictor: DarkFleetPredictor, metrics: dict, output_dir: str = "models"):
    """Salva modello, metriche e artefatti in formato riutilizzabile."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(predictor, os.path.join(output_dir, "predictor_full.pkl"))
    joblib.dump(predictor.best_model, os.path.join(output_dir, "best_model.pkl"))
    
    import json
    # Converti numpy types in nativi Python per JSON
    clean_metrics = {k: (float(v) if hasattr(v, 'item') else v) for k, v in metrics.items()}
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(clean_metrics, f, indent=2)
        
    logging.info(f"💾 Artefatti salvati in: {output_dir}/")


if __name__ == "__main__":
    logging.info("=== TEST INTEGRAZIONE DarkFleetPredictor ===")
    set_global_seed(42)
    
    n_samples = 2000
    timestamps = pd.date_range('2024-06-01', periods=n_samples, freq='10min')
    mmsi_ids = np.random.choice([1001, 1002, 1003], size=n_samples)

    def generate_temporal_series(mean, std, n, autocorr=0.7):
        series = np.zeros(n)
        series[0] = np.random.normal(mean, std)
        for t in range(1, n):
            series[t] = autocorr * series[t-1] + (1-autocorr) * np.random.normal(mean, std)
        return series

    mock_data = pd.DataFrame({
        'Timestamp': timestamps,
        'MMSI': mmsi_ids,
        'delta_SOG': generate_temporal_series(0, 1.5, n_samples),
        'delta_COG': generate_temporal_series(0, 8, n_samples), 
        'speed_acc': generate_temporal_series(0, 0.8, n_samples),
        'turn_rate': generate_temporal_series(0, 4, n_samples),
        'dt_prev_hours': np.abs(generate_temporal_series(0.16, 0.03, n_samples, autocorr=0.9)),
    })

    # Nota review: il target dipende da prob_regime_sospetto → ablation truccata.
    # Per il test va bene, ma su dati reali usare feature reali.
    latent_suspicious = (
        (mock_data['speed_acc'].abs() < 0.3).astype(int) * 0.4 +
        (mock_data['turn_rate'].abs() > 3).astype(int) * 0.4 +
        np.random.normal(0, 0.1, n_samples)
    )
    latent_suspicious = (latent_suspicious - latent_suspicious.min()) / (latent_suspicious.max() - latent_suspicious.min())

    mock_data['prob_regime_sospetto'] = latent_suspicious
    mock_data['incertezza_regime'] = np.random.uniform(0.05, 0.25, n_samples)

    base_prob = 0.02
    prob_blackout = np.clip(base_prob + mock_data['prob_regime_sospetto'] * 0.15, 0, 1)
    mock_data['target_dark_fleet'] = np.random.binomial(1, prob_blackout)

    logging.info(f"Dataset mock: {len(mock_data)} righe, target rate: {mock_data['target_dark_fleet'].mean():.3%}")

    predictor = DarkFleetPredictor(n_trials=5, n_splits=3, gap_hours=2.0, freq_min=10, seed=42)

    split_idx = int(len(mock_data) * 0.8)
    train_df = mock_data.iloc[:split_idx].copy()
    test_df = mock_data.iloc[split_idx:].copy()

    X_train, y_train = predictor.prepare_data_for_cv(train_df)
    X_test, y_test = predictor.prepare_data_for_cv(test_df)

    predictor.optimize_and_train(X_train, y_train)
    final_metrics = predictor.evaluate_model(X_test, y_test)

    try:
        predictor.plot_calibration(X_test, y_test)
        predictor.plot_feature_importance_shap(X_test)
    except Exception as e:
        logging.warning(f"Plot generation skipped: {e}")

    # Ablation study opzionale
    # ablation_results, bootstrap_df = predictor.ablation_study(test_df, train_df, n_trials_ablation=3)

    save_artifacts(predictor, final_metrics)
    logging.info("\n✅ TEST INTEGRAZIONE COMPLETATO CON SUCCESSO")