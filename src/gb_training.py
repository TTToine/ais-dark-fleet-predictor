"""
src/gb_training.py
Modulo per l'addestramento, HPO e validazione del Gradient Boosting (LightGBM).
Rigorosa validazione temporale con gap causale e metriche per dataset sbilanciati.

Caratteristiche principali:
- Filtro di Markov con persistenza data-driven (Optuna trova p_stay ottimale)
- Numba JIT per il filtro Markov (fallback Python puro se numba assente)
- _preprocess_features centralizzato: stesso pipeline train / inference / SHAP
- predict() wrapper che applica le transformazioni e droppa MMSI/Timestamp
- GroupTimeSeriesSplit per evitare leakage inter-vessel
- F2-threshold e Conformal Prediction per soglie operative
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import (
    average_precision_score, log_loss, roc_auc_score, brier_score_loss,
    fbeta_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score
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
    from src.utils import filter_high_vif, calculate_conformal_threshold
except ImportError:
    try:
        from utils import filter_high_vif, calculate_conformal_threshold
    except ImportError:
        def filter_high_vif(X: pd.DataFrame, threshold: float = 10.0) -> List[str]:
            return X.columns.tolist()
        def calculate_conformal_threshold(y_calib, probs_calib, target_fpr=0.05):
            negative_probs = probs_calib[y_calib == 0]
            return float(np.quantile(negative_probs, 1 - target_fpr))

# Numba opzionale: fallback Python puro se non installato
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def njit(func=None, **kwargs):
        """Decoratore no-op se Numba non è installato (esegue come Python puro)."""
        if func is None:
            return lambda f: f
        return func
    logging.info("Numba non disponibile — fast_markov_filter girerà in Python puro (più lento).")

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_global_seed(seed: int = 42):
    """Imposta seed globali per riproducibilità completa."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# =============================================================================
# Filtro di Markov JIT-compilato (O(N), vettorializzato, isolato)
# =============================================================================
@njit
def fast_markov_filter(probs: np.ndarray, p_stay: float) -> np.ndarray:
    """
    Filtro di Markov 1D ricorsivo (forward filtering) sulle probabilità mixture.
    Assume p_00 = p_11 = p_stay per simmetria (riduzione spazio di ricerca a 1D).
    Compilato in C via Numba quando disponibile, altrimenti Python puro.
    """
    n = len(probs)
    smoothed = np.zeros(n)
    smoothed[0] = probs[0]
    p_switch = 1.0 - p_stay

    for t in range(1, n):
        pred = smoothed[t-1] * p_stay + (1 - smoothed[t-1]) * p_switch
        num = probs[t] * pred
        den = num + (1 - probs[t]) * (smoothed[t-1] * p_switch + (1 - smoothed[t-1]) * p_stay)
        smoothed[t] = num / (den + 1e-12)

    return smoothed


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
    """Split temporale che rispetta i gruppi (MMSI) per evitare leakage."""
    def __init__(self, n_splits: int = 5, gap_hours: float = 24.0):
        self.n_splits = n_splits
        self.gap = pd.Timedelta(hours=gap_hours)

    def split(self, X: pd.DataFrame, groups: pd.Series = None):
        if 'MMSI' not in X.columns:
            logging.warning("MMSI non trovato in X. Fallback a split temporale puro.")
            yield from TimeSeriesSplitWithGap(self.n_splits, self.gap.total_seconds()/3600, 10).split(X)
            return

        ship_first_ts = X.groupby('MMSI')['Timestamp'].min()
        sorted_mmsi = ship_first_ts.sort_values().index.tolist()
        n_ships = len(sorted_mmsi)

        for i in range(self.n_splits):
            train_cutoff_idx = int(n_ships * (i + 1) / (self.n_splits + 1))
            train_ships = set(sorted_mmsi[:train_cutoff_idx])

            train_end_time = X[X['MMSI'].isin(train_ships)]['Timestamp'].max()
            valid_cutoff_time = train_end_time + self.gap

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

        self.cv_splitter = GroupTimeSeriesSplit(n_splits, gap_hours)

        # incertezza_regime esclusa: è Bernoulli var di prob_regime_sospetto (ridondante)
        self.features_full = [
            'delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate', 'dt_prev_hours',
            'prob_regime_sospetto'
        ]
        self.features_baseline = [
            'delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate', 'dt_prev_hours'
        ]
        self.target = 'target_dark_fleet'

    def prepare_data_for_cv(self, df: pd.DataFrame,
                            feature_list: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara X e y con ordinamento temporale rigoroso.
        Include MMSI e Timestamp per il GroupTimeSeriesSplit; il filtro Markov per nave
        e la rimozione dei meta cols avvengono dentro objective() / predict()."""
        logging.info("Preparazione dati per validazione temporale...")

        df_sorted = df.sort_values(by='Timestamp').reset_index(drop=True)
        df_sorted = df_sorted.dropna(subset=[self.target])

        features = feature_list if feature_list else self.features_full
        df_sorted = df_sorted.dropna(subset=features)

        meta_cols = [c for c in ['MMSI', 'Timestamp'] if c in df_sorted.columns]
        X = df_sorted[features + meta_cols].copy()
        y = df_sorted[self.target].copy()

        logging.info(f"Dati pronti: {len(X)} campioni, {y.mean():.3%} positivi")
        return X, y

    # =========================================================================
    # Objective Optuna con iniezione dinamica del filtro di Markov
    # =========================================================================
    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series,
                  feature_list: List[str]) -> float:
        """Optuna ottimizza congiuntamente iperparametri LightGBM + p_stay del filtro Markov.
        Il filtro è applicato per nave (MMSI groupby) prima di costruire i fold."""

        markov_p = trial.suggest_float('markov_p', 0.70, 0.99)

        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'seed': self.seed,
            'deterministic': True,
            'force_row_wise': True,
            'num_threads': -1,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 128),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0)
        }

        X_dynamic = X.copy()
        dynamic_features = list(feature_list)

        if 'prob_regime_sospetto' in X_dynamic.columns and 'MMSI' in X_dynamic.columns:
            smoothed_probs = X_dynamic.groupby('MMSI', group_keys=False)['prob_regime_sospetto'].apply(
                lambda p_raw: pd.Series(
                    fast_markov_filter(np.ascontiguousarray(p_raw.values), markov_p),
                    index=p_raw.index
                )
            )
            X_dynamic['prob_regime_sospetto_markov'] = smoothed_probs
            dynamic_features = [
                f if f != 'prob_regime_sospetto' else 'prob_regime_sospetto_markov'
                for f in dynamic_features
            ]

        scores = []
        _meta = ['MMSI', 'Timestamp']
        pos_per_fold_train: List[int] = []
        pos_per_fold_valid: List[int] = []

        for fold, (train_idx, valid_idx) in enumerate(self.cv_splitter.split(X_dynamic)):
            X_train, X_valid = X_dynamic.iloc[train_idx], X_dynamic.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            pos_tr, pos_va = int(y_train.sum()), int(y_valid.sum())
            pos_per_fold_train.append(pos_tr)
            pos_per_fold_valid.append(pos_va)

            if pos_tr == 0 or pos_va == 0:
                logging.debug(
                    f"Fold {fold}: classe positiva assente "
                    f"(train_pos={pos_tr}, val_pos={pos_va}), skip"
                )
                continue

            X_train_lgb = X_train[dynamic_features].drop(columns=_meta, errors='ignore')
            X_valid_lgb = X_valid[dynamic_features].drop(columns=_meta, errors='ignore')
            train_data = lgb.Dataset(X_train_lgb, label=y_train)
            valid_data = lgb.Dataset(X_valid_lgb, label=y_valid, reference=train_data)

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

            preds_proba = gbm.predict(X_valid_lgb, num_iteration=gbm.best_iteration)
            pr_auc = average_precision_score(y_valid, preds_proba)
            scores.append(pr_auc)
            trial.set_user_attr(f'best_iter_fold_{fold}', gbm.best_iteration)
            logging.debug(f"Fold {fold}: PR-AUC={pr_auc:.4f}, best_iter={gbm.best_iteration}")

        # Annotazioni accessibili dall'esterno per la diagnostica HPO degenere.
        n_folds = len(pos_per_fold_valid)
        total_val_pos = int(sum(pos_per_fold_valid))
        trial.set_user_attr('n_folds_seen', n_folds)
        trial.set_user_attr('total_val_positives', total_val_pos)
        trial.set_user_attr('pos_per_fold_valid', pos_per_fold_valid)
        trial.set_user_attr('pos_per_fold_train', pos_per_fold_train)

        if not scores:
            # Distinguiamo "0.0 = modello pessimo" da "0.0 = impossibile valutare":
            # nessun fold con almeno 1 positivo in val → trial PRUNED (escluso dal best).
            logging.warning(
                f"Trial pruned: 0 positives across {n_folds} folds "
                f"(fold positive counts train={pos_per_fold_train} "
                f"val={pos_per_fold_valid})"
            )
            raise optuna.TrialPruned()

        return np.mean(scores)

    # =========================================================================
    # Consolidamento: applicazione del markov_p ottimale sul full train
    # =========================================================================
    def optimize_and_train(self, X_train: pd.DataFrame, y_train: pd.Series,
                           feature_list: Optional[List[str]] = None):
        """Esegue HPO con Optuna (LightGBM + markov_p) e addestra il modello finale."""
        logging.info("🚀 Avvio ottimizzazione hyperparametri (Optuna)...")
        features = feature_list if feature_list else self.features_full
        _meta = ['MMSI', 'Timestamp']
        avail_features = [f for f in features if f in X_train.columns]
        avail_meta = [c for c in _meta if c in X_train.columns]
        X = X_train[avail_features + avail_meta].copy()

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(lambda trial: self.objective(trial, X, y_train, avail_features),
                       n_trials=self.n_trials, n_jobs=1)

        # =============================================================
        # Guard rail: HPO degenere → fallisci rumorosamente.
        # Se >50% dei trial sono stati pruned per "nessun positivo in val"
        # NON salvare nessun modello: l'utente deve diagnosticare il target,
        # non collezionare un PR-AUC=0 fake-success dopo ore di compute.
        # =============================================================
        from optuna.trial import TrialState
        n_total = len(study.trials)
        n_complete = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        n_pruned = sum(1 for t in study.trials if t.state == TrialState.PRUNED)
        if n_pruned > n_total / 2:
            prevalence = float(y_train.mean()) if len(y_train) else 0.0
            raise RuntimeError(
                f"HPO degenerate: {n_pruned}/{n_total} trials had no "
                f"positive samples in any CV fold. This usually means:\n"
                f"  - target prevalence too low ({prevalence:.4%})\n"
                f"  - CV folds too small for the rare-event class\n"
                f"  - bug in labeling (positives not propagating to splits)\n"
                f"Run scripts/diagnose_target.py for details before retrying."
            )
        if n_complete == 0:
            raise RuntimeError(
                f"HPO degenerate: 0/{n_total} trials completed successfully. "
                f"No best_params to extract."
            )

        self.best_params = study.best_params
        logging.info(f"✅ Migliori parametri trovati: {self.best_params}")
        logging.info(f"📈 Miglior PR-AUC CV: {study.best_value:.4f}")
        if n_pruned > 0:
            logging.warning(
                f"⚠️  {n_pruned}/{n_total} trial pruned per assenza di positivi "
                f"in val (HPO operato su {n_complete} trial completi)."
            )

        best_trial_iters = [
            v for k, v in study.best_trial.user_attrs.items()
            if k.startswith('best_iter_fold_')
        ]
        optimal_rounds = int(np.median(best_trial_iters)) if best_trial_iters else 500
        logging.info(f"📐 num_boost_round ottimale: {optimal_rounds} (mediana su {len(best_trial_iters)} fold del best trial)")

        final_param_base = {
            **self.best_params,
            'objective': 'binary', 'verbosity': -1,
            'seed': self.seed, 'deterministic': True, 'force_row_wise': True,
            'num_threads': -1
        }
        final_param_base.pop('markov_p', None)

        logging.info("🔄 Applicazione filtro di Markov ottimizzato sul training set completo...")
        X_final = X.copy()
        final_features = list(avail_features)

        if 'markov_p' in self.best_params and 'prob_regime_sospetto' in X_final.columns and 'MMSI' in X_final.columns:
            opt_p = self.best_params['markov_p']
            smoothed = X_final.groupby('MMSI', group_keys=False)['prob_regime_sospetto'].apply(
                lambda p_raw: pd.Series(
                    fast_markov_filter(np.ascontiguousarray(p_raw.values), opt_p),
                    index=p_raw.index
                )
            )
            X_final['prob_regime_sospetto_markov'] = smoothed
            final_features = [
                f if f != 'prob_regime_sospetto' else 'prob_regime_sospetto_markov'
                for f in final_features
            ]
            logging.info(f"✅ Filtro Markov applicato con p_stay={opt_p:.3f}")

        X_final_lgb = X_final[final_features].drop(columns=_meta, errors='ignore')

        logging.info("🏋️ Addestramento modello finale...")
        self.best_model = lgb.train(
            final_param_base,
            lgb.Dataset(X_final_lgb, label=y_train),
            num_boost_round=optimal_rounds
        )
        logging.info(f"✅ Modello finale addestrato ({optimal_rounds} round).")

    # =========================================================================
    # Preprocessing centralizzato + Predict wrapper
    # =========================================================================
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applica le trasformazioni dinamiche (Markov filter) e allinea le colonne.
        Ritorna un DataFrame esattamente identico a quello usato in fase di training.
        Usato da predict() e dai metodi SHAP per garantire coerenza end-to-end.
        """
        if self.best_model is None:
            raise ValueError("Modello non addestrato. Impossibile preprocessare le feature.")

        X_proc = X.copy()

        # 1. Markov filter con il p_stay ottimale (se imparato)
        if self.best_params and 'markov_p' in self.best_params and 'prob_regime_sospetto' in X_proc.columns:
            opt_p = self.best_params['markov_p']

            # Ordinamento causale
            if 'MMSI' in X_proc.columns and 'Timestamp' in X_proc.columns:
                X_proc = X_proc.sort_values(by=['MMSI', 'Timestamp'])
            elif 'Timestamp' in X_proc.columns:
                X_proc = X_proc.sort_values(by='Timestamp')

            if 'MMSI' in X_proc.columns:
                smoothed = X_proc.groupby('MMSI', group_keys=False)['prob_regime_sospetto'].apply(
                    lambda p_raw: pd.Series(
                        fast_markov_filter(np.ascontiguousarray(p_raw.values), opt_p),
                        index=p_raw.index
                    )
                )
            else:
                smoothed = pd.Series(
                    fast_markov_filter(np.ascontiguousarray(X_proc['prob_regime_sospetto'].values), opt_p),
                    index=X_proc.index
                )

            X_proc['prob_regime_sospetto_markov'] = smoothed

        # 2. Allineamento rigoroso alle feature attese dal modello (droppa MMSI/Timestamp)
        expected_features = self.best_model.feature_name()
        missing = [f for f in expected_features if f not in X_proc.columns]
        if missing:
            raise ValueError(f"Feature mancanti per l'inferenza/SHAP: {missing}")

        return X_proc[expected_features]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Wrapper inferenza: garantisce allineamento perfetto tra training e inference."""
        if self.best_model is None:
            raise ValueError("Addestra prima il modello con optimize_and_train()")
        X_ready = self._preprocess_features(X)
        return self.best_model.predict(X_ready)

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Valuta il modello sul test set."""
        preds_proba = self.predict(X_test)
        logging.info("📊 Valutazione modello LightGBM")

        metrics = {
            'pr_auc':      average_precision_score(y_test, preds_proba),
            'roc_auc':     roc_auc_score(y_test, preds_proba),
            'log_loss':    log_loss(y_test, preds_proba),
            'brier_score': brier_score_loss(y_test, preds_proba)
        }

        logging.info("\n" + "="*60)
        logging.info("📊 RISULTATI FINALI SUL TEST SET")
        logging.info("="*60)
        for name, value in metrics.items():
            logging.info(f"{name:25s}: {value:.4f}")
        logging.info("="*60 + "\n")

        return metrics

    def find_optimal_threshold(self, X_test: pd.DataFrame, y_test: pd.Series,
                               beta: float = 2.0) -> dict:
        """Trova la soglia di classificazione ottimale massimizzando F-beta score."""
        probs = self.predict(X_test)
        thresholds = np.linspace(0.01, 0.99, 200)
        f_scores = [
            fbeta_score(y_test, (probs >= t).astype(int), beta=beta, zero_division=0)
            for t in thresholds
        ]

        opt_idx = int(np.argmax(f_scores))
        opt_threshold = float(thresholds[opt_idx])
        opt_f_beta    = float(f_scores[opt_idx])
        y_pred = (probs >= opt_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].plot(thresholds, f_scores, color='#2196F3', linewidth=2)
        axes[0].axvline(opt_threshold, color='red', linestyle='--',
                        label=f'Ottimale: {opt_threshold:.3f}')
        axes[0].set_xlabel('Soglia di classificazione')
        axes[0].set_ylabel(f'F{beta}-score')
        axes[0].set_title(f'F{beta}-score vs Soglia (β={beta})')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        ConfusionMatrixDisplay(cm, display_labels=['Normale', 'Dark Fleet']).plot(
            ax=axes[1], colorbar=False
        )
        axes[1].set_title(
            f'Confusion Matrix (soglia={opt_threshold:.3f}, F{beta}={opt_f_beta:.3f})'
        )

        plt.tight_layout()
        os.makedirs("models", exist_ok=True)
        plt.savefig("models/optimal_threshold_confusion_matrix.png", dpi=300)
        logging.info(
            f"📊 Soglia ottimale F{beta}: {opt_threshold:.3f}  "
            f"| Precision={precision_score(y_test, y_pred, zero_division=0):.3f}  "
            f"| Recall={recall_score(y_test, y_pred, zero_division=0):.3f}"
        )
        plt.close()

        return {
            'optimal_threshold': opt_threshold,
            f'f{beta}_score': opt_f_beta,
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
            'confusion_matrix': cm.tolist()
        }

    def evaluate_with_conformal(self, X_calib: pd.DataFrame, y_calib: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                target_fpr: float = 0.05) -> dict:
        """Applica Conformal Prediction per produrre prediction sets con garanzia di FPR."""
        probs_calib = self.predict(X_calib)
        probs_test  = self.predict(X_test)

        threshold = calculate_conformal_threshold(
            y_calib.values, probs_calib, target_fpr=target_fpr
        )

        y_pred = (probs_test >= threshold).astype(int)

        tn = int(((y_test == 0) & (y_pred == 0)).sum())
        fp = int(((y_test == 0) & (y_pred == 1)).sum())
        empirical_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        result = {
            'conformal_threshold': float(threshold),
            'target_fpr':    target_fpr,
            'empirical_fpr': empirical_fpr,
            'precision':     float(precision_score(y_test, y_pred, zero_division=0)),
            'recall':        float(recall_score(y_test, y_pred, zero_division=0)),
            'pr_auc':        float(average_precision_score(y_test, probs_test))
        }
        logging.info(
            f"🎯 Conformal soglia={threshold:.3f} FPR garantito≤{target_fpr:.0%}  "
            f"FPR empirico={empirical_fpr:.3%}  "
            f"Precision={result['precision']:.3f} Recall={result['recall']:.3f}"
        )
        return result

    def plot_calibration(self, X_test: pd.DataFrame, y_test: pd.Series, n_bins: int = 10):
        """Plot di calibrazione per valutare l'affidabilità delle probabilità."""
        preds_proba = self.predict(X_test)

        plt.figure(figsize=(8, 6))
        CalibrationDisplay.from_predictions(y_test, preds_proba, n_bins=n_bins, ax=plt.gca())
        plt.title("Calibration Curve - Affidabilità Probabilità Predette")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        os.makedirs("models", exist_ok=True)
        plt.savefig("models/calibration_plot.png", dpi=300)
        logging.info("📈 Calibration plot salvato in models/calibration_plot.png")
        plt.close()

    def plot_calibration_comparison(self, X_test: pd.DataFrame, y_test: pd.Series,
                                    baseline_model=None, n_bins: int = 10,
                                    save_path: str = "models/calibration_comparison.png"):
        """Plotta due curve di calibrazione: baseline (modello cinematico) vs enhanced (questo)."""
        preds_enhanced = self.predict(X_test)

        fig, ax = plt.subplots(figsize=(8, 7))

        if baseline_model is not None:
            try:
                preds_baseline = baseline_model.predict(X_test)
                CalibrationDisplay.from_predictions(
                    y_test, preds_baseline, n_bins=n_bins, ax=ax,
                    name="Baseline (cinematico)", color="gray", linestyle="--"
                )
            except Exception as e:
                logging.warning(f"Calibration baseline skippata: {e}")

        CalibrationDisplay.from_predictions(
            y_test, preds_enhanced, n_bins=n_bins, ax=ax,
            name="Enhanced (+Bayesian features)", color="#2196F3"
        )

        ax.set_title("Calibration Curve — Enhanced vs Baseline", fontsize=13)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logging.info(f"📈 Calibration comparison plot salvato in {save_path}")
        plt.close()

    # =========================================================================
    # SHAP allineato al preprocessing centralizzato
    # =========================================================================
    def plot_feature_importance_shap(self, X_sample: pd.DataFrame, max_display: int = 10):
        """SHAP Feature Importance con filtro VIF opzionale (post-preprocess)."""
        if self.best_model is None:
            raise ValueError("Addestra prima il modello")
        logging.info("Calcolo SHAP values...")

        X_sample_ready = self._preprocess_features(X_sample)

        if len(X_sample_ready) > 1000:
            X_sample_ready = X_sample_ready.sample(n=1000, random_state=self.seed)

        safe_features = filter_high_vif(X_sample_ready, threshold=10.0)
        X_sample_safe = X_sample_ready[safe_features]

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

    def plot_shap_interaction(self, X_sample: pd.DataFrame,
                              feature: str = 'prob_regime_sospetto_markov',
                              interaction_index: str = 'speed_acc'):
        """SHAP dependence plot per interazioni feature (post-preprocess)."""
        if self.best_model is None:
            raise ValueError("Addestra prima il modello")

        X_sample_ready = self._preprocess_features(X_sample)

        if feature not in X_sample_ready.columns or interaction_index not in X_sample_ready.columns:
            logging.warning(
                f"Feature '{feature}' o '{interaction_index}' assenti in X_sample_ready. "
                f"Colonne disponibili: {list(X_sample_ready.columns)}"
            )
            return

        if len(X_sample_ready) > 1000:
            X_sample_ready = X_sample_ready.sample(n=1000, random_state=self.seed)

        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(X_sample_ready)

        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feature, shap_values, X_sample_ready,
            interaction_index=interaction_index,
            show=False
        )
        plt.title(f"SHAP Interaction: {feature} × {interaction_index}", fontsize=12)
        plt.tight_layout()
        os.makedirs("models", exist_ok=True)
        save_name = f"models/shap_interaction_{feature}_{interaction_index}.png"
        plt.savefig(save_name, dpi=300)
        logging.info(f"📊 SHAP interaction plot salvato in {save_name}")
        plt.close()

    def ablation_study(self, test_df: pd.DataFrame, train_df: pd.DataFrame,
                       n_trials_ablation: int = 10) -> dict:
        """Confronto strutturato: Logistic vs GB Baseline vs GB Enhanced."""
        logging.info("🔬 Avvio Ablation Study...")
        results = {}
        df_bootstrap = test_df[['MMSI', self.target]].copy() if 'MMSI' in test_df.columns else test_df[[self.target]].copy()

        configurations = [
            ("logistic_baseline", self.features_baseline, True),
            ("gb_baseline",       self.features_baseline, False),
            ("gb_enhanced",       self.features_full,     False)
        ]

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        for config_name, features, use_logistic in configurations:
            logging.info(f"\n--- Addestramento Configurazione: {config_name} ---")

            X_train, y_train = self.prepare_data_for_cv(train_df, feature_list=features)
            X_test,  y_test  = self.prepare_data_for_cv(test_df,  feature_list=features)

            if use_logistic:
                logging.info("Addestramento Logistic Regression...")
                scaler = StandardScaler()
                X_train_lr = X_train.drop(columns=['MMSI', 'Timestamp'], errors='ignore')
                X_test_lr  = X_test.drop(columns=['MMSI', 'Timestamp'], errors='ignore')
                X_train_scaled = scaler.fit_transform(X_train_lr.fillna(0))
                X_test_scaled  = scaler.transform(X_test_lr.fillna(0))

                lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=self.seed)
                lr_model.fit(X_train_scaled, y_train)

                preds_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
                pr_auc = average_precision_score(y_test, preds_proba)
                brier  = brier_score_loss(y_test, preds_proba)

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

            preds_proba = predictor.predict(X_test)
            df_bootstrap[f'pred_{config_name}'] = preds_proba

            results[config_name] = {
                'model':  predictor.best_model,
                'params': predictor.best_params,
                'metrics': metrics,
                'pr_auc':  metrics['pr_auc'],
                'brier_score': metrics['brier_score']
            }
            logging.info(f"✅ {config_name} completato: PR-AUC={metrics['pr_auc']:.4f}, Brier={metrics['brier_score']:.4f}")

        logging.info("\n" + "="*60)
        logging.info("📊 ABLATION STUDY — CONFRONTO FINALE")
        logging.info(f"{'Config': <22} {'PR-AUC': >8} {'Brier': >8}")
        logging.info("-"*40)
        for name, res in results.items():
            pr = res.get('pr_auc', float('nan'))
            br = res.get('brier_score', float('nan'))
            logging.info(f"{name: <22} {pr: >8.4f} {br: >8.4f}")
        logging.info("="*60)

        return results, df_bootstrap


def gap_threshold_sensitivity(df_features: pd.DataFrame,
                              gap_thresholds_hours: Optional[List[float]] = None,
                              horizon_hours: float = 24.0,
                              n_trials_sensitivity: int = 10,
                              n_splits: int = 3,
                              seed: int = 42) -> dict:
    """
    Analisi di sensitività sulla scelta di gap_threshold_hours.
    Ricalcola il target a ogni soglia e misura PR-AUC + Brier sul GB enhanced.

    Args:
        df_features: DataFrame con MMSI, Timestamp, feature engineered, ma SENZA target fisso.
        gap_thresholds_hours: lista di soglie in ore da testare (default: [6, 12, 18, 24]).
        horizon_hours: finestra di previsione (uguale per tutte le soglie).
    """
    if gap_thresholds_hours is None:
        gap_thresholds_hours = [6.0, 12.0, 18.0, 24.0]

    logging.info(f"🔬 Sensitivity analysis — gap_threshold_hours: {gap_thresholds_hours}")
    results = {}

    features_baseline = ['delta_SOG', 'delta_COG', 'speed_acc', 'turn_rate', 'dt_prev_hours']

    for gap_h in gap_thresholds_hours:
        logging.info(f"\n--- Soglia {gap_h}h ---")

        df = df_features.drop(columns=['target_dark_fleet'], errors='ignore').copy()
        df = df.sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)
        df['_next_ts'] = df.groupby('MMSI')['Timestamp'].shift(-1)
        df['_gap']     = (df['_next_ts'] - df['Timestamp']).dt.total_seconds() / 3600.0
        df['target_dark_fleet'] = (
            (df['_gap'] >= gap_h) & (df['_gap'] <= gap_h + horizon_hours)
        ).astype(float)
        df.loc[df['_gap'].isna(), 'target_dark_fleet'] = np.nan
        df = df.dropna(subset=['target_dark_fleet']).drop(columns=['_next_ts', '_gap'])
        df['target_dark_fleet'] = df['target_dark_fleet'].astype(int)

        pos_rate = float(df['target_dark_fleet'].mean())
        logging.info(f"  Positivi: {pos_rate:.3%}")

        if pos_rate < 0.001 or pos_rate > 0.5:
            logging.warning(f"  ⚠️ Classe troppo sbilanciata a {gap_h}h ({pos_rate:.3%}), skip.")
            results[gap_h] = {'pr_auc': None, 'brier_score': None, 'pos_rate': pos_rate}
            continue

        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df  = df.iloc[split_idx:]

        predictor = DarkFleetPredictor(
            n_trials=n_trials_sensitivity, n_splits=n_splits,
            gap_hours=gap_h, seed=seed
        )

        try:
            X_tr, y_tr = predictor.prepare_data_for_cv(train_df, feature_list=features_baseline)
            X_te, y_te = predictor.prepare_data_for_cv(test_df,  feature_list=features_baseline)

            if y_tr.sum() == 0 or y_te.sum() == 0:
                logging.warning(f"  Nessun positivo in train/test per {gap_h}h, skip.")
                results[gap_h] = {'pr_auc': None, 'brier_score': None, 'pos_rate': pos_rate}
                continue

            predictor.optimize_and_train(X_tr, y_tr, feature_list=features_baseline)
            metrics = predictor.evaluate_model(X_te, y_te)

            results[gap_h] = {
                'pr_auc':      metrics['pr_auc'],
                'brier_score': metrics['brier_score'],
                'pos_rate':    pos_rate
            }
            logging.info(f"  ✅ PR-AUC={metrics['pr_auc']:.4f} Brier={metrics['brier_score']:.4f}")

        except Exception as e:
            logging.error(f"  ❌ Errore a {gap_h}h: {e}")
            results[gap_h] = {'pr_auc': None, 'brier_score': None, 'pos_rate': pos_rate}

    return results


def save_artifacts(predictor: DarkFleetPredictor, metrics: dict, df_hmm: pd.DataFrame = None,
                   output_dir: str = "models"):
    """Salva modello, metriche e artefatti in formato riutilizzabile."""
    import json
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(predictor,             os.path.join(output_dir, "predictor_full.pkl"))
    joblib.dump(predictor.best_model,  os.path.join(output_dir, "lgb_dark_fleet.pkl"))

    clean_metrics = {k: (float(v) if hasattr(v, 'item') else v) for k, v in metrics.items()}
    with open(os.path.join(output_dir, "metrics_final.json"), 'w') as f:
        json.dump(clean_metrics, f, indent=2)

    if predictor.best_params:
        with open(os.path.join(output_dir, "best_params.json"), 'w') as f:
            json.dump(predictor.best_params, f, indent=2)

    if df_hmm is not None:
        enriched_path = os.path.join("data", "processed", "ais_enriched.parquet")
        os.makedirs(os.path.dirname(enriched_path), exist_ok=True)
        df_hmm.to_parquet(enriched_path)
        logging.info(f"💾 Enriched data salvati in: {enriched_path}")

    logging.info(f"💾 Artefatti salvati in: {output_dir}/")
