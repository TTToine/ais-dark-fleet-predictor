"""
src/viz/contest_plots.py

Grafici da presentazione per un contest di statistica.
Stile: seaborn-whitegrid, palette coerente, font readable da slide.
Ogni funzione salva il PNG e ritorna il path.

Funzioni principali:
- plot_simulation_validation: confronto P(sospetto) dark vs normal con ground truth
- plot_brier_decomposition: decomposizione di Murphy (Reliability - Resolution + Uncertainty)
- plot_sensitivity_analysis: PR-AUC e Brier vs gap_threshold con CI bootstrap
- plot_bootstrap_ci_pr_auc: distribuzione bootstrap del PR-AUC clustered per MMSI
- plot_advi_vs_nuts: scatter ADVI vs NUTS con linea y=x e Spearman ρ
- plot_pr_curve_with_thresholds: PR curve con F2 e Conformal threshold marcati
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss

# ============================================================================
# STILE GLOBALE — coerenza visiva tra tutti i plot
# ============================================================================
PALETTE = {
    "dark":     "#C0392B",   # rosso scuro — dark fleet, falsi negativi
    "normal":   "#2980B9",   # blu — comportamento legittimo
    "model":    "#27AE60",   # verde — predizione modello
    "baseline": "#7F8C8D",   # grigio — baseline
    "accent":   "#E67E22",   # arancione — highlight (conformal, F2)
    "neutral":  "#34495E",   # antracite — testi/assi
}

def _apply_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.dpi': 110,
        'savefig.dpi': 220,
        'savefig.bbox': 'tight',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.edgecolor': PALETTE['neutral'],
        'axes.labelcolor': PALETTE['neutral'],
        'xtick.color': PALETTE['neutral'],
        'ytick.color': PALETTE['neutral'],
        'legend.frameon': True,
        'legend.framealpha': 0.92,
        'legend.edgecolor': '#BDC3C7',
    })


# ============================================================================
# 1) SIMULATION STUDY VALIDATION
# ============================================================================
def plot_simulation_validation(df_enriched: pd.DataFrame,
                                ground_truth_path: str,
                                save_path: str = "models/simulation_validation.png") -> str:
    """
    Confronta P(regime_sospetto) tra navi 'dark' e 'normal' secondo ground truth.
    Plot a 2 pannelli:
      (a) boxplot/violin di P media per nave, dark vs normal
      (b) ROC a livello di nave: AUC nella classificazione dark/normal usando P media
    """
    _apply_style()
    if not Path(ground_truth_path).exists():
        logging.warning(f"Ground truth non trovato in {ground_truth_path}, skip.")
        return ""

    with open(ground_truth_path) as f:
        gt = json.load(f)

    mmsi_to_dark = {s["mmsi"]: s["is_dark"] for s in gt["ships"]}

    prob_col = 'prob_regime_markov' if 'prob_regime_markov' in df_enriched.columns else 'prob_regime_sospetto'
    per_ship = (df_enriched.groupby('MMSI')[prob_col]
                .agg(['mean', 'max', 'std'])
                .reset_index())
    per_ship['is_dark'] = per_ship['MMSI'].map(mmsi_to_dark)
    per_ship = per_ship.dropna(subset=['is_dark'])

    if len(per_ship) == 0:
        logging.warning("Nessuna nave del ground truth presente nei dati enriched.")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) Violin + scatter P media per nave
    ax = axes[0]
    dark_vals = per_ship.loc[per_ship['is_dark'], 'mean'].values
    norm_vals = per_ship.loc[~per_ship['is_dark'], 'mean'].values

    parts = ax.violinplot([norm_vals, dark_vals], positions=[0, 1],
                          widths=0.7, showmeans=False, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(PALETTE['normal'] if i == 0 else PALETTE['dark'])
        pc.set_alpha(0.55)
        pc.set_edgecolor(PALETTE['neutral'])

    # Scatter individuale
    rng = np.random.default_rng(0)
    ax.scatter(rng.normal(0, 0.05, len(norm_vals)), norm_vals,
               color=PALETTE['normal'], alpha=0.8, s=40, edgecolor='white', zorder=3,
               label=f"Normal (n={len(norm_vals)})")
    ax.scatter(rng.normal(1, 0.05, len(dark_vals)), dark_vals,
               color=PALETTE['dark'], alpha=0.85, s=44, edgecolor='white', zorder=3,
               label=f"Dark (n={len(dark_vals)})")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Normal', 'Dark'])
    ax.set_ylabel(r'$\overline{P(\mathrm{regime\ sospetto})}$ per nave')
    ax.set_title("(a) Recovery del regime latente")
    ax.legend(loc='upper left')
    ax.set_ylim(-0.02, 1.02)

    # (b) ROC nave-level
    from sklearn.metrics import roc_curve, roc_auc_score
    try:
        y_ship = per_ship['is_dark'].astype(int).values
        s_ship = per_ship['mean'].values
        fpr, tpr, _ = roc_curve(y_ship, s_ship)
        auc = roc_auc_score(y_ship, s_ship)
        ax = axes[1]
        ax.plot(fpr, tpr, color=PALETTE['model'], lw=2.6,
                label=f'AUC (vessel-level) = {auc:.3f}')
        ax.plot([0, 1], [0, 1], '--', color=PALETTE['baseline'], lw=1.5, label='random')
        ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE['model'])
        ax.set_xlabel('False Positive Rate (nave)')
        ax.set_ylabel('True Positive Rate (nave)')
        ax.set_title("(b) Discriminazione dark/normal a livello di nave")
        ax.legend(loc='lower right')
    except Exception as e:
        logging.warning(f"ROC nave-level fallito: {e}")

    fig.suptitle("Validazione su Simulation Study — Ground Truth Conosciuta",
                 fontsize=15, fontweight='bold', y=1.02)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logging.info(f"📊 Simulation validation plot salvato: {save_path}")
    return save_path


# ============================================================================
# 2) MURPHY DECOMPOSITION OF BRIER SCORE
# ============================================================================
def plot_brier_decomposition(y_true: np.ndarray,
                              y_prob: np.ndarray,
                              n_bins: int = 10,
                              save_path: str = "models/brier_decomposition.png") -> str:
    """
    Decomposizione di Murphy del Brier Score:
        BS = Reliability − Resolution + Uncertainty

    - Reliability: quanto le probabilità predette sono calibrate (basso = buono)
    - Resolution: quanto il modello discrimina i bin (alto = buono)
    - Uncertainty: incertezza intrinseca del problema (= var(y))
    """
    _apply_style()
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1])

    base_rate = y_true.mean()
    uncertainty = base_rate * (1 - base_rate)

    reliability = 0.0
    resolution = 0.0
    bin_data = []
    n = len(y_true)
    for k in range(n_bins):
        mask = (bin_ids == k)
        nk = mask.sum()
        if nk == 0:
            continue
        fk = y_true[mask].mean()      # observed freq in bin
        pk = y_prob[mask].mean()      # mean predicted prob in bin
        reliability += nk * (pk - fk) ** 2
        resolution  += nk * (fk - base_rate) ** 2
        bin_data.append((pk, fk, nk))
    reliability /= n
    resolution /= n
    brier = reliability - resolution + uncertainty

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) Decomposition bars
    ax = axes[0]
    labels = ['Reliability\n(↓ meglio)', 'Resolution\n(↑ meglio)', 'Uncertainty\n(intrinseca)', 'Brier Score']
    values = [reliability, resolution, uncertainty, brier]
    colors = [PALETTE['dark'], PALETTE['model'], PALETTE['baseline'], PALETTE['accent']]
    bars = ax.bar(labels, values, color=colors, edgecolor=PALETTE['neutral'], alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Valore')
    ax.set_title('(a) Decomposizione di Murphy')
    ax.set_ylim(0, max(values) * 1.25)

    # (b) Reliability diagram
    ax = axes[1]
    if bin_data:
        pks = np.array([b[0] for b in bin_data])
        fks = np.array([b[1] for b in bin_data])
        nks = np.array([b[2] for b in bin_data])
        sizes = 60 + 400 * nks / nks.max()
        ax.plot([0, 1], [0, 1], '--', color=PALETTE['baseline'], lw=1.5, label='Perfetta calibrazione')
        ax.scatter(pks, fks, s=sizes, color=PALETTE['model'], alpha=0.75,
                   edgecolor=PALETTE['neutral'], zorder=3, label='Bin osservati')
        for pk, fk, nk in bin_data:
            ax.annotate(str(nk), (pk, fk), textcoords="offset points",
                        xytext=(0, -3), ha='center', fontsize=8)
        ax.set_xlabel('Probabilità predetta media')
        ax.set_ylabel('Frequenza osservata')
        ax.set_title('(b) Reliability diagram (dimensione = #obs)')
        ax.legend(loc='upper left')
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    fig.suptitle(f"Brier Score = {brier:.4f}  ↔  Rel ({reliability:.4f}) − Res ({resolution:.4f}) + Unc ({uncertainty:.4f})",
                 fontsize=13, y=1.02)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logging.info(f"📊 Brier decomposition salvato: {save_path}")
    return save_path


# ============================================================================
# 3) SENSITIVITY ANALYSIS
# ============================================================================
def plot_sensitivity_analysis(sensitivity_results: dict,
                              save_path: str = "models/sensitivity_analysis.png") -> str:
    """
    Plotta PR-AUC e Brier vs gap_threshold dal risultato di gap_threshold_sensitivity.
    sensitivity_results: {gap_h: {'pr_auc': ..., 'brier_score': ..., 'pos_rate': ...}}
    """
    _apply_style()
    if not sensitivity_results:
        logging.warning("Sensitivity results vuoto, skip.")
        return ""

    gap_hs = sorted([float(g) for g in sensitivity_results.keys()])
    pr_aucs = [sensitivity_results[g if g in sensitivity_results else str(g)].get('pr_auc') for g in gap_hs]
    briers  = [sensitivity_results[g if g in sensitivity_results else str(g)].get('brier_score') for g in gap_hs]
    poss    = [sensitivity_results[g if g in sensitivity_results else str(g)].get('pos_rate') for g in gap_hs]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(gap_hs, pr_aucs, marker='o', markersize=10, lw=2.5,
             color=PALETTE['model'], label='PR-AUC')
    ax1.set_xlabel('Gap threshold (ore)')
    ax1.set_ylabel('PR-AUC', color=PALETTE['model'])
    ax1.tick_params(axis='y', labelcolor=PALETTE['model'])
    ax1.set_ylim(0, max([p for p in pr_aucs if p is not None] + [0.1]) * 1.2)

    ax2 = ax1.twinx()
    ax2.plot(gap_hs, briers, marker='s', markersize=9, lw=2,
             color=PALETTE['dark'], linestyle='--', label='Brier Score')
    ax2.set_ylabel('Brier Score', color=PALETTE['dark'])
    ax2.tick_params(axis='y', labelcolor=PALETTE['dark'])
    ax2.grid(False)

    # Tasso positivi come bar
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    bar_pos = [(p * 100) if p is not None else 0 for p in poss]
    ax3.bar(gap_hs, bar_pos, alpha=0.18, color=PALETTE['accent'],
            width=1.0, label='Tasso positivi (%)')
    ax3.set_ylabel('Tasso positivi (%)', color=PALETTE['accent'])
    ax3.tick_params(axis='y', labelcolor=PALETTE['accent'])
    ax3.grid(False)

    ax1.set_title('Sensitivity Analysis: stabilità del modello al variare del gap threshold',
                  fontweight='bold')

    # Legenda combinata
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
               loc='upper left', fontsize=10)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logging.info(f"📊 Sensitivity analysis plot salvato: {save_path}")
    return save_path


# ============================================================================
# 4) BOOTSTRAP CI GROUPED BY MMSI
# ============================================================================
def plot_bootstrap_ci_pr_auc(y_true: np.ndarray,
                              y_prob: np.ndarray,
                              groups: np.ndarray,
                              n_boot: int = 1000,
                              alpha: float = 0.05,
                              seed: int = 42,
                              save_path: str = "models/bootstrap_ci_pr_auc.png") -> str:
    """
    Bootstrap CI del PR-AUC con campionamento clustered per nave (MMSI).
    Il bootstrap naive sovrastima la confidenza ignorando la correlazione intra-nave.
    """
    _apply_style()
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}

    boot_aucs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled])
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            boot_aucs.append(average_precision_score(y_true[idx], y_prob[idx]))
        except ValueError:
            continue

    boot_aucs = np.array(boot_aucs)
    if len(boot_aucs) == 0:
        logging.warning("Tutti i bootstrap hanno fallito.")
        return ""

    point = average_precision_score(y_true, y_prob)
    lo, hi = np.quantile(boot_aucs, [alpha / 2, 1 - alpha / 2])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(boot_aucs, bins=40, color=PALETTE['model'], alpha=0.75,
            edgecolor='white', label=f'{len(boot_aucs)} resample clustered per MMSI')
    ax.axvline(point, color=PALETTE['accent'], lw=3, label=f'Point estimate = {point:.3f}')
    ax.axvline(lo, color=PALETTE['dark'], lw=2, linestyle='--',
               label=f'{int((1-alpha)*100)}% CI = [{lo:.3f}, {hi:.3f}]')
    ax.axvline(hi, color=PALETTE['dark'], lw=2, linestyle='--')
    ax.fill_betweenx([0, ax.get_ylim()[1]], lo, hi, color=PALETTE['dark'], alpha=0.07)

    ax.set_xlabel('PR-AUC (bootstrap resample)')
    ax.set_ylabel('Frequenza')
    ax.set_title(f'Bootstrap CI del PR-AUC con clustering per nave (n_boot={n_boot})',
                 fontweight='bold')
    ax.legend(loc='upper right')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logging.info(f"📊 Bootstrap CI plot salvato: {save_path}")
    return save_path


# ============================================================================
# 5) PR CURVE CON SOGLIE OPERATIVE
# ============================================================================
def plot_pr_curve_with_thresholds(y_true: np.ndarray,
                                   y_prob: np.ndarray,
                                   f2_threshold: Optional[float] = None,
                                   conformal_threshold: Optional[float] = None,
                                   save_path: str = "models/pr_curve_thresholds.png") -> str:
    """
    PR curve con threshold operative marcate (F2-optimal e Conformal).
    """
    _apply_style()
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    base_rate = np.mean(y_true)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(recall, precision, color=PALETTE['model'], lw=2.6,
            label=f'PR curve (AP = {ap:.3f})')
    ax.axhline(base_rate, color=PALETTE['baseline'], lw=1.5, linestyle=':',
               label=f'Random baseline (P = {base_rate:.3f})')
    ax.fill_between(recall, precision, alpha=0.10, color=PALETTE['model'])

    def _mark(threshold, color, label):
        if threshold is None:
            return
        idx = np.searchsorted(thresholds, threshold)
        idx = min(idx, len(precision) - 1)
        ax.scatter(recall[idx], precision[idx], s=180, color=color,
                   edgecolor='white', zorder=5, linewidth=2,
                   label=f'{label} @ τ={threshold:.3f}\n  → P={precision[idx]:.2f}, R={recall[idx]:.2f}')

    _mark(f2_threshold, PALETTE['accent'], 'F2-ottimale')
    _mark(conformal_threshold, PALETTE['dark'], 'Conformal (FPR≤10%)')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PR Curve + Soglie operative (F2 vs Conformal)', fontweight='bold')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='upper right', fontsize=10)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logging.info(f"📊 PR curve plot salvato: {save_path}")
    return save_path


# ============================================================================
# 6) ADVI vs NUTS VALIDATION
# ============================================================================
def plot_advi_vs_nuts(advi_probs: np.ndarray,
                       nuts_probs: np.ndarray,
                       spearman_rho: float,
                       p_value: float = None,
                       save_path: str = "models/advi_vs_nuts.png") -> str:
    """
    Scatter ADVI vs NUTS con linea y=x. Spearman ρ > 0.85 giustifica empiricamente
    l'uso di ADVI come approssimazione adeguata di NUTS per questo problema.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(nuts_probs, advi_probs, alpha=0.55, s=55,
               color=PALETTE['model'], edgecolor='white', linewidth=0.6)
    ax.plot([0, 1], [0, 1], '--', color=PALETTE['neutral'], lw=2, label='Identità y=x')

    ax.set_xlabel(r'$P_\mathrm{NUTS}(\mathrm{regime\ sospetto})$')
    ax.set_ylabel(r'$P_\mathrm{ADVI}(\mathrm{regime\ sospetto})$')
    pval_str = f", p={p_value:.2e}" if p_value is not None else ""
    ax.set_title(f'ADVI vs NUTS — Spearman ρ = {spearman_rho:.3f}{pval_str}',
                 fontweight='bold')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.legend(loc='upper left')

    threshold_text = ("✓ ADVI giustificata (ρ > 0.85)" if spearman_rho > 0.85
                      else "⚠ ADVI marginale, valutare NUTS")
    ax.text(0.05, 0.92, threshold_text, transform=ax.transAxes,
            fontsize=11, fontweight='bold',
            color=PALETTE['model'] if spearman_rho > 0.85 else PALETTE['dark'],
            bbox=dict(facecolor='white', edgecolor=PALETTE['neutral'], alpha=0.9))

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logging.info(f"📊 ADVI vs NUTS plot salvato: {save_path}")
    return save_path


# ============================================================================
# 6b) POSTERIOR PREDICTIVE CHECK
# ============================================================================
def plot_posterior_predictive_check(ppc_path: str,
                                    save_path: str = "models/posterior_predictive_check.png") -> str:
    """
    Visualizza il Posterior Predictive Check (Gelman BDA3 §6.3).
    Grid (N navi × 4 statistiche): densità delle stat sintetiche con linea
    verticale per la stat osservata e Bayesian p-value annotato per ogni cella.
    Codifica colore: verde se 0.05 < p < 0.95 (adeguato), rosso altrimenti.
    """
    _apply_style()
    if not Path(ppc_path).exists():
        logging.warning(f"PPC result non trovato in {ppc_path}, skip.")
        return ""

    with open(ppc_path) as f:
        ppc = json.load(f)

    if ppc.get("method") != "posterior_predictive_check":
        return ""

    vessels = ppc["vessel_results"]
    stat_names = ppc["stat_names"]
    pretty_names = {
        "mean_speed":         r'$\overline{x}_\mathrm{speed}$ (location)',
        "std_speed":          r'$s_\mathrm{speed}$ (scale)',
        "autocorr_turn_lag1": r'$\rho_1(\mathrm{turn})$ (dipendenza)',
        "frac_extreme_speed": r'$P(|x_\mathrm{speed}|>1.5)$ (tail)',
    }

    n_v = len(vessels)
    n_s = len(stat_names)
    fig, axes = plt.subplots(n_v, n_s, figsize=(4.2 * n_s, 3.2 * n_v),
                             squeeze=False)

    for i, vr in enumerate(vessels):
        for j, stat in enumerate(stat_names):
            ax = axes[i, j]
            synth = np.array(vr["synthetic_stats"][stat])
            obs   = vr["observed_stats"][stat]
            p     = vr["bayesian_p_values"][stat]

            is_adequate = (p is not None and 0.05 < p < 0.95)
            dist_color = PALETTE['model'] if is_adequate else PALETTE['dark']

            if len(synth) > 0:
                ax.hist(synth, bins=30, density=True, color=dist_color,
                        alpha=0.5, edgecolor='white', label='posterior predictive')
                # Intervalli 5–95 sintetici
                lo, hi = np.quantile(synth, [0.025, 0.975])
                ax.axvspan(lo, hi, alpha=0.12, color=dist_color)

            ax.axvline(obs, color=PALETTE['accent'], lw=2.6,
                       label=f'osservato = {obs:.3f}')

            # Annotazione p-value
            p_str = f"p = {p:.3f}" if p is not None else "p = N/D"
            badge = "✓" if is_adequate else "✗"
            ax.text(0.04, 0.94, f"{badge} {p_str}", transform=ax.transAxes,
                    ha='left', va='top', fontsize=11, fontweight='bold',
                    color=dist_color,
                    bbox=dict(facecolor='white', edgecolor=dist_color,
                              alpha=0.92, boxstyle='round,pad=0.3'))

            if i == 0:
                ax.set_title(pretty_names.get(stat, stat), fontsize=11)
            if j == 0:
                ax.set_ylabel(f'MMSI {vr["mmsi"]}\n(n={vr["n_obs"]})',
                              fontsize=10, fontweight='bold')
            if i == n_v - 1:
                ax.set_xlabel('valore statistica', fontsize=10)
            ax.tick_params(labelsize=8)

    # Verdetto globale come banner sopra
    gv = ppc["global_verdict"]
    frac = gv["frac_p_in_inner_90pct"]
    adequate = gv["model_adequate"]
    verdict = "✅ MODELLO ADEGUATO" if adequate else "⚠️ DISCREPANZE SISTEMATICHE"
    verdict_color = PALETTE['model'] if adequate else PALETTE['dark']

    fig.suptitle(
        f"Posterior Predictive Check (Gelman BDA3 §6.3) — {verdict}   "
        f"[{frac:.0%} delle p-value in [0.05, 0.95], soglia 75%]",
        fontsize=13, fontweight='bold', y=1.005, color=verdict_color
    )

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logging.info(f"📊 PPC plot salvato: {save_path}")
    return save_path


# ============================================================================
# 7) PRIOR SENSITIVITY ANALYSIS
# ============================================================================
def plot_prior_sensitivity(sensitivity_path: str,
                            save_path: str = "models/prior_sensitivity.png") -> str:
    """
    Visualizza la prior sensitivity (Gelman BDA3 cap.6):
    (a) Scatter dei vessel scores sotto coppie di specificazioni con linea y=x
    (b) Banner verdetto "ROBUSTO / NON ROBUSTO" + Spearman ρ minimo
    """
    _apply_style()
    if not Path(sensitivity_path).exists():
        logging.warning(f"Prior sensitivity result non trovato in {sensitivity_path}, skip.")
        return ""

    with open(sensitivity_path) as f:
        sens = json.load(f)

    if sens.get("method") != "prior_sensitivity_analysis":
        return ""

    spec_names = list(sens["specifications"].keys())
    common_mmsi = sorted(sens["vessel_scores"][spec_names[0]].keys())
    scores = {s: np.array([sens["vessel_scores"][s][m] for m in common_mmsi])
              for s in spec_names}

    # 1 pannello per ogni coppia (max 3 coppie tipicamente)
    pairs = []
    for i, s1 in enumerate(spec_names):
        for s2 in spec_names[i + 1:]:
            pairs.append((s1, s2))

    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs + 1, figsize=(5 * (n_pairs + 1), 5.5))
    if n_pairs == 0:
        axes = [axes]
    axes = list(np.atleast_1d(axes))

    # Scatter per ogni coppia
    for ax, (s1, s2) in zip(axes[:n_pairs], pairs):
        x, y = scores[s1], scores[s2]
        rho = sens["pairwise_spearman"].get(f"{s1}_vs_{s2}", float('nan'))
        ax.scatter(x, y, s=80, alpha=0.78, color=PALETTE['model'],
                   edgecolor='white', linewidth=1.2, zorder=3)
        ax.plot([0, 1], [0, 1], '--', color=PALETTE['baseline'],
                lw=1.5, label='y = x')
        ax.set_xlabel(f'Vessel score — prior "{s1}"')
        ax.set_ylabel(f'Vessel score — prior "{s2}"')
        ax.set_title(f"ρ_Spearman = {rho:+.3f}")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=9)

    # Banner verdetto
    ax = axes[-1]
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    robust = sens.get("robust_to_prior", False)
    min_rho = sens.get("min_spearman")
    threshold = sens.get("robustness_threshold", 0.85)

    bg_color = PALETTE['model'] if robust else PALETTE['dark']
    verdict_text = "✅ ROBUSTO" if robust else "⚠️ NON ROBUSTO"

    ax.set_facecolor('#FAFBFC')
    ax.text(0.5, 0.72, verdict_text, transform=ax.transAxes,
            ha='center', va='center', fontsize=28, fontweight='bold', color=bg_color)
    ax.text(0.5, 0.50, f"min ρ_Spearman = {min_rho:.3f}",
            transform=ax.transAxes, ha='center', va='center', fontsize=15,
            color=PALETTE['neutral'])
    ax.text(0.5, 0.36, f"(soglia di robustezza: ρ > {threshold})",
            transform=ax.transAxes, ha='center', va='center', fontsize=10,
            color='#7F8C8D', style='italic')
    ax.text(0.5, 0.18, f"basato su {sens['n_vessels']} navi · "
                       f"{len(spec_names)} specificazioni di prior",
            transform=ax.transAxes, ha='center', va='center', fontsize=10,
            color=PALETTE['neutral'])
    ax.axhline(0.02, color=bg_color, lw=5, transform=ax.transAxes)

    fig.suptitle("Prior Sensitivity Analysis — robustezza del ranking alle scelte di prior "
                 "(Gelman BDA3 §6)",
                 fontsize=14, fontweight='bold', y=1.02)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logging.info(f"📊 Prior sensitivity plot salvato: {save_path}")
    return save_path


# ============================================================================
# 8) EMPIRICAL BAYES HYPERPRIOR VISUALIZATION
# ============================================================================
def plot_empirical_bayes_priors(eb_result_path: str,
                                 default_priors: dict = None,
                                 save_path: str = "models/empirical_bayes_priors.png") -> str:
    """
    Visualizza il two-stage Empirical Bayes:
    (a) Posterior means per nave pilot dei 4 parametri chiave (scatter con bande)
    (b) Confronto σ_prior default vs σ_prior empirico stimato
    """
    _apply_style()
    if not Path(eb_result_path).exists():
        logging.warning(f"EB result non trovato in {eb_result_path}, skip.")
        return ""

    with open(eb_result_path) as f:
        eb = json.load(f)

    if eb.get("method") != "two_stage_empirical_bayes":
        logging.info(f"EB result non è two_stage (method={eb.get('method')}), skip.")
        return ""

    pilots = eb["pilot_results"]
    n_pilot = len(pilots)
    if n_pilot == 0:
        return ""

    mu_sosp = np.array([p["mu_sospetto"] for p in pilots])
    gap_arr = np.array([p["gap"] for p in pilots])
    sig_sp  = np.array([(p["sigma_speed_0"] + p["sigma_speed_1"]) / 2 for p in pilots])
    sig_tr  = np.array([(p["sigma_turn_0"]  + p["sigma_turn_1"])  / 2 for p in pilots])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Posterior means per nave pilot
    ax = axes[0]
    x = np.arange(n_pilot)
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, mu_sosp, width, label=r'$\mu_\mathrm{sospetto}$',
                   color=PALETTE['dark'], edgecolor='white')
    bars2 = ax.bar(x - 0.5*width, gap_arr, width, label=r'gap $= \mu_\mathrm{transito} - \mu_\mathrm{sospetto}$',
                   color=PALETTE['accent'], edgecolor='white')
    bars3 = ax.bar(x + 0.5*width, sig_sp, width, label=r'$\bar{\sigma}_\mathrm{speed}$',
                   color=PALETTE['normal'], edgecolor='white')
    bars4 = ax.bar(x + 1.5*width, sig_tr, width, label=r'$\bar{\sigma}_\mathrm{turn}$',
                   color=PALETTE['model'], edgecolor='white')

    # Banda della media ± std cross-vessel sul mu_sospetto (illustrativa)
    cv = eb["cross_vessel_stats"]
    ax.axhline(cv["mean_mu_sospetto"], color=PALETTE['dark'],
               linestyle=':', lw=1.5, alpha=0.7,
               label=f'media flotta = {cv["mean_mu_sospetto"]:.2f}')

    ax.set_xticks(x)
    ax.set_xticklabels([f'MMSI\n{p["mmsi"]}' for p in pilots], fontsize=9)
    ax.set_ylabel('Valore posterior mean')
    ax.set_title(f'(a) Stage 1: posterior per {n_pilot} navi pilot')
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.axhline(0, color=PALETTE['neutral'], lw=0.6)

    # (b) σ_prior default vs empirico
    ax = axes[1]
    labels = [r'$\sigma_\mathrm{prior, speed}$', r'$\sigma_\mathrm{prior, turn}$']
    default_vals  = [
        (default_priors or {}).get('sigma_prior_speed', 1.0),
        (default_priors or {}).get('sigma_prior_turn', 2.0),
    ]
    empirical_vals = [eb["sigma_prior_speed"], eb["sigma_prior_turn"]]

    x = np.arange(len(labels))
    width = 0.35
    bars_d = ax.bar(x - width/2, default_vals, width,
                    label='Default (a priori)', color=PALETTE['baseline'],
                    edgecolor='white', alpha=0.85)
    bars_e = ax.bar(x + width/2, empirical_vals, width,
                    label='Empirical Bayes (Stage 1)', color=PALETTE['model'],
                    edgecolor='white')

    for bars in [bars_d, bars_e]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f'{v:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('σ del prior')
    ax.set_title('(b) Stage 2: hyperprior usato nei fit per-nave')
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(max(default_vals), max(empirical_vals)) * 1.4)

    fig.suptitle(
        f"Two-Stage Empirical Bayes — partial pooling implicito su {n_pilot} navi pilot",
        fontsize=15, fontweight='bold', y=1.02
    )
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    logging.info(f"📊 Empirical Bayes plot salvato: {save_path}")
    return save_path


# ============================================================================
# 8) DASHBOARD RIEPILOGATIVA (1 slide, 6 mini-panel)
# ============================================================================
def plot_executive_summary(metrics: dict,
                            save_path: str = "models/executive_summary.png") -> str:
    """
    Una slide riepilogativa: 6 KPI principali in card colorate.
    Da usare come 'opening slide' della presentazione.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    kpis = [
        ("PR-AUC",        metrics.get('pr_auc'),       PALETTE['model'],    '↑'),
        ("ROC-AUC",       metrics.get('roc_auc'),      PALETTE['normal'],   '↑'),
        ("Brier Score",   metrics.get('brier_score'),  PALETTE['accent'],   '↓'),
        ("Log Loss",      metrics.get('log_loss'),     PALETTE['dark'],     '↓'),
        ("F2 @ τ*",       metrics.get('f2_optimal'),   PALETTE['model'],    '↑'),
        ("Conformal FPR", metrics.get('conformal_fpr'),PALETTE['accent'],   '≤'),
    ]

    for ax, (name, val, color, arrow) in zip(axes, kpis):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('#FAFBFC')
        if val is None:
            val_str, sub = "N/D", ""
        else:
            val_str = f"{val:.3f}"
            sub = f"{arrow} meglio"
        ax.text(0.5, 0.62, val_str, transform=ax.transAxes,
                ha='center', va='center', fontsize=34, fontweight='bold', color=color)
        ax.text(0.5, 0.27, name, transform=ax.transAxes,
                ha='center', va='center', fontsize=15, color=PALETTE['neutral'])
        ax.text(0.5, 0.10, sub, transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='#7F8C8D', style='italic')
        # Cornice colorata
        for side in ['bottom']:
            ax.axhline(0.02, color=color, lw=4, transform=ax.transAxes)

    fig.suptitle("AIS Dark Fleet Predictor — Executive Summary",
                 fontsize=17, fontweight='bold', y=1.00)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, facecolor='white')
    plt.close(fig)
    logging.info(f"📊 Executive summary salvato: {save_path}")
    return save_path
