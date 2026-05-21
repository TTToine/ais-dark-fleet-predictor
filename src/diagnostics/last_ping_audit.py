"""Last-Ping diagnostic: misura quanto le "ultime ping prima del blackout"
sono distinguibili kinematicamente dalle ping NON-ultime DELLA STESSA NAVE.

Obiettivo
---------
Sotto la framing retrospettiva ``last_ping``, il modello potrebbe ottenere
PR-AUC alto semplicemente perché la nostra definizione di "positivo"
coincide con un evento (la fine osservata della traccia) che ha proprietà
kinematiche peculiari — non perché il modello sappia davvero predire un
blackout futuro.

Per separare i due effetti confrontiamo:

* **Last-pings**:  righe etichettate positive sotto Last-Ping.
* **Controlli**:   K ping NON-ultime campionate dalla stessa MMSI.

Su sole feature kinematiche addestriamo una Logistic Regression e
riportiamo:

* AUC della logistic (1 = facile distinguere; 0.5 = indistinguibili)
* Distanza Mahalanobis mediana fra i due gruppi
* Numero di vessel idonei e di campioni usati

Lettura
-------
* AUC >> 0.7 → i last-ping hanno una firma kinematica netta. Il
  modello ``last_ping`` ha valore predittivo *sull'evento osservato*,
  ma resta non deployable as-is (vedi labeling_comparison.md).
* AUC ≈ 0.5 → il modello sta imparando soprattutto artefatti del
  labeling; sotto ``last_ping`` la metrica va degradata di conseguenza.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _mark_last_ping_before_blackout(df: pd.DataFrame, gap_threshold_hours: float) -> pd.Series:
    """Ritorna una Series bool indicizzata come ``df``: True quando il
    ping è seguito da un gap >= ``gap_threshold_hours``."""
    df = df.sort_values(['MMSI', 'Timestamp'])
    next_ts = df.groupby('MMSI')['Timestamp'].shift(-1)
    gap_h = (next_ts - df['Timestamp']).dt.total_seconds() / 3600.0
    return (gap_h >= gap_threshold_hours).fillna(False)


def _mahalanobis_distance(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Distanza Mahalanobis fra centroidi di due gruppi, usando la
    covarianza pooled (regolarizzata per stabilità)."""
    pooled = np.vstack([group_a, group_b])
    cov = np.cov(pooled, rowvar=False)
    # Regolarizzazione di Tikhonov per evitare matrici singolari.
    cov_reg = cov + np.eye(cov.shape[0]) * 1e-6
    try:
        inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov_reg)
    diff = group_a.mean(axis=0) - group_b.mean(axis=0)
    return float(np.sqrt(diff @ inv @ diff))


def run_last_ping_audit(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    gap_threshold_hours: float = 12.0,
    n_controls_per_vessel: int = 5,
    output_path: Optional[str] = "results/last_ping_audit.json",
    seed: int = 42,
) -> dict:
    """Esegue la diagnostica e (opzionalmente) la salva su JSON.

    Args:
        df: DataFrame con almeno ``MMSI``, ``Timestamp`` e le ``feature_cols``.
        feature_cols: feature SOLO kinematiche (no probabilità latenti) per
            evitare di confondere "features endogene al modello" con "firma
            kinematica reale".
        gap_threshold_hours: stessa soglia del labeling.
        n_controls_per_vessel: ``K`` ping non-ultime per nave.
        output_path: dove salvare il JSON (None = non salvare).
        seed: rng seed.

    Returns:
        dict con AUC, Mahalanobis, conteggi e diagnosi testuale.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    feature_cols = list(feature_cols)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature mancanti nel df: {missing}")

    df = df.copy().sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)
    is_last = _mark_last_ping_before_blackout(df, gap_threshold_hours)
    df['_is_last'] = is_last.astype(int).values

    eligible_vessels = (
        df.groupby('MMSI')['_is_last']
        .max()
        .pipe(lambda s: s[s == 1].index.tolist())
    )
    logger.info(
        f"Last-Ping audit: {len(eligible_vessels)} vessel con almeno un blackout."
    )
    if len(eligible_vessels) < 3:
        result = {
            'auc_last_vs_controls': None,
            'mahalanobis_centroid_distance': None,
            'n_eligible_vessels': len(eligible_vessels),
            'n_last_pings': int(df['_is_last'].sum()),
            'n_controls': 0,
            'diagnosis': (
                'INSUFFICIENT_DATA: meno di 3 vessel con blackout; '
                'audit non statisticamente affidabile.'
            ),
            'feature_cols': feature_cols,
            'n_controls_per_vessel': n_controls_per_vessel,
            'gap_threshold_hours': gap_threshold_hours,
        }
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
        return result

    rng = np.random.default_rng(seed)
    last_rows: List[pd.DataFrame] = []
    ctrl_rows: List[pd.DataFrame] = []

    for mmsi in eligible_vessels:
        sub = df[df['MMSI'] == mmsi]
        lasts = sub[sub['_is_last'] == 1]
        nonlasts = sub[sub['_is_last'] == 0]
        if len(nonlasts) == 0:
            continue
        k = min(n_controls_per_vessel, len(nonlasts))
        ctrl = nonlasts.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1)))
        last_rows.append(lasts)
        ctrl_rows.append(ctrl)

    last_df = pd.concat(last_rows, ignore_index=True)
    ctrl_df = pd.concat(ctrl_rows, ignore_index=True)

    X_last = last_df[feature_cols].to_numpy(dtype=float)
    X_ctrl = ctrl_df[feature_cols].to_numpy(dtype=float)
    X = np.vstack([X_last, X_ctrl])
    y = np.concatenate([np.ones(len(X_last)), np.zeros(len(X_ctrl))]).astype(int)

    # Imputazione mediana per robustezza (la diagnostica non deve crashare
    # su NaN che il preprocessing principale ha già gestito).
    col_med = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_med, np.where(nan_mask)[1])

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    n_splits = min(5, max(2, len(np.unique(y[y == 1]))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    auc_scores = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        X_std, y, cv=cv, scoring='roc_auc',
    )
    auc = float(np.mean(auc_scores))

    maha = _mahalanobis_distance(X_std[y == 1], X_std[y == 0])

    if auc >= 0.70:
        diagnosis = (
            f"DISTINGUISHABLE (AUC={auc:.3f} ≥ 0.70): le ultime-ping prima "
            f"di un blackout hanno una firma kinematica genuinamente diversa. "
            f"Il modello Last-Ping ha valore predittivo sull'evento osservato, "
            f"ma rimane non deployable as-is."
        )
    elif auc >= 0.55:
        diagnosis = (
            f"WEAKLY_DISTINGUISHABLE (AUC={auc:.3f}): segnale presente ma "
            f"modesto. Il PR-AUC riportato sotto Last-Ping è probabilmente "
            f"in parte gonfiato dal framing retrospettivo."
        )
    else:
        diagnosis = (
            f"INDISTINGUISHABLE (AUC={auc:.3f} < 0.55): i last-ping non sono "
            f"kinematicamente diversi dai controlli. Il modello Last-Ping sta "
            f"largamente imparando artefatti del labeling — i suoi numeri "
            f"vanno degradati."
        )

    result = {
        'auc_last_vs_controls': auc,
        'auc_per_fold': [float(s) for s in auc_scores],
        'mahalanobis_centroid_distance': maha,
        'n_eligible_vessels': len(eligible_vessels),
        'n_last_pings': int(len(last_df)),
        'n_controls': int(len(ctrl_df)),
        'n_controls_per_vessel': n_controls_per_vessel,
        'gap_threshold_hours': gap_threshold_hours,
        'feature_cols': feature_cols,
        'diagnosis': diagnosis,
    }
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"📝 Last-Ping audit salvato in {output_path}")
    logger.info(f"Last-Ping audit: AUC={auc:.3f} | Mahalanobis={maha:.3f}")
    return result
