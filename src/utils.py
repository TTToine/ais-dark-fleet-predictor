"""
src/utils.py
Modulo utility per funzioni statistiche avanzate, visualizzazioni e supporto alla validazione.
Include: Bootstrap Confidence Intervals, Conformal Prediction helpers, Plotting avanzato.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score
from typing import Callable, Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)

# ========================================================================
# STATISTICAL LEARNING TOOLS
# ========================================================================

def bootstrap_confidence_interval(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    metric_func: Callable, 
    n_bootstraps: int = 1000, 
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Calcola l'intervallo di confidenza bootstrap per una metrica data.
    
    Args:
        y_true: Array dei target reali.
        y_pred: Array delle probabilità predette.
        metric_func: Funzione metrica (es. average_precision_score).
        n_bootstraps: Numero di campionamenti bootstrap.
        alpha: Livello di significatività (default 0.05 per CI 95%).
        seed: Seed per riproducibilità.
        
    Returns:
        Tuple (lower_bound, point_estimate, upper_bound)
    """
    np.random.seed(seed)
    n_samples = len(y_true)
    boot_scores = []
    
    # Point estimate
    point_estimate = metric_func(y_true, y_pred)
    
    for _ in range(n_bootstraps):
        # Campionamento con replacement degli indici
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue # Skip se il campione non ha entrambe le classi
            
        score = metric_func(y_true[indices], y_pred[indices])
        boot_scores.append(score)
        
    if not boot_scores:
        return point_estimate, point_estimate, point_estimate
        
    lower = np.percentile(boot_scores, 100 * alpha / 2)
    upper = np.percentile(boot_scores, 100 * (1 - alpha / 2))
    
    return lower, point_estimate, upper

def compare_models_grouped_bootstrap(
    df_results: pd.DataFrame,
    group_col: str = 'MMSI',
    target_col: str = 'target_dark_fleet',
    pred_col_a: str = 'pred_enhanced',
    pred_col_b: str = 'pred_baseline',
    n_bootstraps: int = 1000,
    seed: int = 42
) -> dict:
    """
    Confronta due modelli tramite Grouped Bootstrap per serie temporali.
    
    Invece di campionare singole righe (distruggendo la causalità e le sequenze),
    campiona con reimmissione gli interi gruppi (le navi tramite MMSI).
    Questo garantisce che l'intervallo di confidenza sia statisticamente valido 
    per dati panel/time-series.
    """
    logging.info(f"Avvio Grouped Bootstrap ({n_bootstraps} iterazioni) su navi...")
    np.random.seed(seed)
    
    unique_groups = df_results[group_col].unique()
    n_groups = len(unique_groups)
    
    # Ottimizzazione estrema: pre-calcoliamo gli indici di ogni nave.
    # Fare df.loc[] mille volte in un loop bloccherebbe il computer.
    group_indices = df_results.groupby(group_col).indices
    
    y_true_all = df_results[target_col].values
    preds_a_all = df_results[pred_col_a].values
    preds_b_all = df_results[pred_col_b].values
    
    deltas = []
    
    for _ in range(n_bootstraps):
        # Campioniamo N navi a caso (con reimmissione)
        sampled_groups = np.random.choice(unique_groups, size=n_groups, replace=True)
        
        # Ricostruiamo gli indici del dataset bootstrapato
        boot_idx = np.concatenate([group_indices[g] for g in sampled_groups])
        
        y_boot = y_true_all[boot_idx]
        
        # Saltiamo il campione se non contiene entrambi i target (0 e 1)
        if len(np.unique(y_boot)) < 2:
            continue
            
        preds_a_boot = preds_a_all[boot_idx]
        preds_b_boot = preds_b_all[boot_idx]
        
        score_a = average_precision_score(y_boot, preds_a_boot)
        score_b = average_precision_score(y_boot, preds_b_boot)
        
        # Delta: Quanto l'Enhanced (A) è meglio del Baseline (B)
        deltas.append(score_a - score_b)
        
    if not deltas:
        raise ValueError("Bootstrap fallito: classi positive troppo rare nei campioni.")
        
    delta_mean = np.mean(deltas)
    ci_lower = np.percentile(deltas, 2.5)  # 95% CI lower bound
    ci_upper = np.percentile(deltas, 97.5) # 95% CI upper bound
    
    # Se il limite inferiore è maggiore di zero, la vittoria del modello A è statisticamente significativa
    significant = (ci_lower > 0)
    
    return {
        'delta_mean': delta_mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': significant
    }
# ========================================================================
# CONFORMAL PREDICTION HELPERS
# ========================================================================

def calculate_conformal_threshold(
    y_calib: np.ndarray, 
    probs_calib: np.ndarray, 
    target_fpr: float = 0.05
) -> float:
    """
    Calcola la soglia di probabilità per garantire un False Positive Rate (FPR) massimo.
    Utile per sistemi di allerta dove i falsi positivi sono costosi.
    """
    negative_indices = np.where(y_calib == 0)[0]
    if len(negative_indices) == 0:
        raise ValueError("Nessun negativo nel set di calibrazione.")
        
    negative_probs = probs_calib[negative_indices]
    threshold = np.quantile(negative_probs, 1 - target_fpr)
    return threshold

# ========================================================================
# PLOTTING ADVANCED
# ========================================================================

def plot_metric_comparison(
    model_names: List[str],
    metrics_dict: dict,
    metric_name: str = "PR-AUC",
    save_path: Optional[str] = None
):
    """Plotta un confronto a barre di metriche tra diversi modelli."""
    plt.figure(figsize=(10, 6))
    values = [metrics_dict[name] for name in model_names]
    
    bars = plt.bar(model_names, values, color=['#3498db', '#e74c3c', '#2ecc71'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.4f}', ha='center', va='bottom')
                 
    plt.title(f"Confronto Modelli: {metric_name}", fontsize=14)
    plt.ylabel(metric_name)
    plt.ylim(0, max(values) * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_distribution_comparison(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    title: str = "Distribuzione Feature per Classe",
    save_path: Optional[str] = None
):
    """Plotta la distribuzione di una feature separata per classe target."""
    plt.figure(figsize=(10, 6))
    
    class_0 = df[df[target_col] == 0][feature_col].dropna()
    class_1 = df[df[target_col] == 1][feature_col].dropna()
    
    sns.kdeplot(class_0, label=f"Classe 0 (Normale) - N={len(class_0)}", fill=True, alpha=0.5)
    sns.kdeplot(class_1, label=f"Classe 1 (Dark Fleet) - N={len(class_1)}", fill=True, alpha=0.5)
    
    plt.title(title)
    plt.xlabel(feature_col)
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()