"""End-to-end smoke test su sottocampione deterministico.

Obiettivo: in <30 minuti validare l'intera pipeline (data prep → BMM → HPO
→ eval) PRIMA di lanciare il full run (~17h). Se questo fallisce, risparmi
16+ ore.

Sottocampiona 20 navi (deterministico, ordinate per MMSI), riduce ADVI/HPO
budget, e fa assert ad ogni stadio. Output: ``results/smoke_test.json``
+ verdetto GO/CAUTION/NO-GO sullo stdout.

NON sovrascrive ``models/`` (usa ``models/smoke/``) né ``.cache/hmm`` o
``.cache/bmm`` di produzione (usa ``.cache/smoke/``).

Uso::

    python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Forza UTF-8 su stdout/stderr (Windows cp1252 default).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Output paths isolati dalla produzione.
SMOKE_MODELS_DIR = ROOT / "models" / "smoke"
SMOKE_CACHE_DIR = ROOT / ".cache" / "smoke"
RESULTS_PATH = ROOT / "results" / "smoke_test.json"
DATA_PATH = ROOT / "data" / "processed" / "ais_enriched.parquet"

# Forza il cache directory ad essere quello smoke (le classi BMM lo leggono
# da costruttore se esposto, altrimenti possiamo solo disabilitare).
os.environ["BMM_CACHE_DIR"] = str(SMOKE_CACHE_DIR)

# Budget ridotto per smoke.
SMOKE_N_MMSI = 20
SMOKE_N_TRIALS = 5
SMOKE_N_SPLITS = 3
SMOKE_UPDATE_FREQ = 30        # 1 ADVI fit ogni 30 ping (vs 10-20 produzione)
SMOKE_MAX_ADVI_CALLS = 8      # cap per nave (vs 30-50 produzione)
SMOKE_N_JOBS = 1              # sequenziale: 20 navi → meno overhead di parallelismo

# Soglie verdetto.
SMOKE_MIN_POSITIVES = 10
SMOKE_PR_AUC_GO_MULTIPLE = 5.0       # GO se PR-AUC > 5× baseline
SMOKE_PR_AUC_PASS_MULTIPLE = 2.0     # PASS se PR-AUC > 2× baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("smoke")


def _fail(reason: str, results: dict) -> int:
    """Scrive il JSON e ritorna exit code 1."""
    results["verdict"] = "NO-GO"
    results["failure_reason"] = reason
    results["timestamp"] = datetime.now().isoformat()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n" + "=" * 72)
    print(f"  ❌ NO-GO: {reason}")
    print("=" * 72)
    return 1


def main() -> int:
    t_start = time.time()
    results: dict = {
        "smoke_config": {
            "n_mmsi": SMOKE_N_MMSI,
            "n_trials": SMOKE_N_TRIALS,
            "n_splits": SMOKE_N_SPLITS,
            "update_freq": SMOKE_UPDATE_FREQ,
            "max_advi_calls": SMOKE_MAX_ADVI_CALLS,
            "n_jobs": SMOKE_N_JOBS,
        },
        "phase_wall_time_seconds": {},
        "assertions": {},
    }

    # -------------------------------------------------------------------------
    # 0. Setup directory di output isolati
    # -------------------------------------------------------------------------
    SMOKE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Data prep: carica enriched, sottocampiona 20 MMSI deterministico
    # -------------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("PHASE 1: Data prep + subsample")
    logger.info("=" * 72)
    t0 = time.time()

    import numpy as np
    import pandas as pd

    if not DATA_PATH.exists():
        return _fail(
            f"Dataset arricchito non trovato: {DATA_PATH}. "
            f"Lancia prima `python scripts/regenerate_simulated.py`.",
            results,
        )

    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded: {len(df):,} pings, {df['MMSI'].nunique()} vessels")

    # Deterministic subsample: prime 20 MMSI sorted as string.
    all_mmsi = sorted(df["MMSI"].unique(), key=lambda x: str(x))
    selected_mmsi = all_mmsi[:SMOKE_N_MMSI]
    df_sub = df[df["MMSI"].isin(selected_mmsi)].copy()
    df_sub = df_sub.sort_values(["MMSI", "Timestamp"]).reset_index(drop=True)

    if df_sub["MMSI"].dtype != np.int64:
        # Fallback: cast (non dovrebbe servire dopo il fix, ma è cintura+bretelle)
        df_sub["MMSI"] = df_sub["MMSI"].astype(np.int64)

    n_pos = int(df_sub["target_dark_fleet"].sum())
    n_tot = len(df_sub)
    prevalence = n_pos / n_tot if n_tot else 0.0
    logger.info(
        f"Subsample: {n_pos} positives / {n_tot} pings "
        f"({prevalence:.4%}) on {len(selected_mmsi)} vessels"
    )

    results["data_prep"] = {
        "n_pings": n_tot,
        "n_vessels": len(selected_mmsi),
        "n_positives": n_pos,
        "prevalence": prevalence,
        "selected_mmsi": [int(m) for m in selected_mmsi],
    }
    results["phase_wall_time_seconds"]["data_prep"] = time.time() - t0

    # Assertion: smoke threshold sui positivi.
    results["assertions"]["positives_ge_min"] = n_pos >= SMOKE_MIN_POSITIVES
    if n_pos < SMOKE_MIN_POSITIVES:
        return _fail(
            f"Positivi nel subsample = {n_pos} < soglia smoke {SMOKE_MIN_POSITIVES}. "
            f"Aumenta dark_ratio o n_ships nel YAML e rigenera.",
            results,
        )

    # -------------------------------------------------------------------------
    # 2. Bayesian inference (BMM)
    # -------------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("PHASE 2: Bayesian inference (reduced budget)")
    logger.info("=" * 72)
    t0 = time.time()

    try:
        from src.bayesian_mixture import CausalBayesianMixture
    except ImportError as exc:
        return _fail(f"Import BMM fallito: {exc}", results)

    bmm = CausalBayesianMixture(window_size=36)
    try:
        df_bmm = bmm.process_dataframe_causal(
            df_sub,
            use_advi=True,
            apply_markov=True,
            update_freq=SMOKE_UPDATE_FREQ,
            max_advi_calls=SMOKE_MAX_ADVI_CALLS,
            n_jobs=SMOKE_N_JOBS,
        )
    except Exception as exc:
        return _fail(f"BMM ha sollevato: {type(exc).__name__}: {exc}", results)

    # Assertion: prob_regime_sospetto deve avere varianza > 0 e non essere all-NaN.
    if "prob_regime_sospetto" not in df_bmm.columns:
        return _fail("BMM non ha prodotto la colonna prob_regime_sospetto.", results)

    probs = df_bmm["prob_regime_sospetto"].values
    n_nan = int(np.isnan(probs).sum())
    var = float(np.nanvar(probs))
    results["bmm"] = {
        "n_nan": n_nan,
        "frac_nan": n_nan / len(probs) if len(probs) else 0.0,
        "var": var,
        "mean": float(np.nanmean(probs)),
        "min": float(np.nanmin(probs)),
        "max": float(np.nanmax(probs)),
    }
    results["phase_wall_time_seconds"]["bayesian"] = time.time() - t0

    results["assertions"]["bmm_nonzero_variance"] = var > 1e-6
    results["assertions"]["bmm_not_all_nan"] = n_nan < len(probs)
    if not results["assertions"]["bmm_nonzero_variance"]:
        return _fail(
            f"prob_regime_sospetto var={var:.6e} (≈0). "
            f"BMM è collassato su una soluzione triviale.",
            results,
        )
    if not results["assertions"]["bmm_not_all_nan"]:
        return _fail("prob_regime_sospetto è interamente NaN.", results)
    logger.info(f"BMM OK: var={var:.4f}, NaN={n_nan}/{len(probs)}")

    # -------------------------------------------------------------------------
    # 3. HPO + training (reduced budget)
    # -------------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("PHASE 3: HPO + training (reduced budget)")
    logger.info("=" * 72)
    t0 = time.time()

    try:
        from src.gb_training import DarkFleetPredictor
    except ImportError as exc:
        return _fail(f"Import gb_training fallito: {exc}", results)

    # Train/test split: 80/20 sequenziale sulle righe (smoke, no group-aware).
    split_idx = int(len(df_bmm) * 0.8)
    train_df = df_bmm.iloc[:split_idx].copy()
    test_df = df_bmm.iloc[split_idx:].copy()

    predictor = DarkFleetPredictor(
        n_trials=SMOKE_N_TRIALS,
        n_splits=SMOKE_N_SPLITS,
        gap_hours=24.0,
        freq_min=10,
        seed=42,
    )
    X_train, y_train = predictor.prepare_data_for_cv(train_df)
    X_test, y_test = predictor.prepare_data_for_cv(test_df)

    try:
        predictor.optimize_and_train(X_train, y_train)
    except RuntimeError as exc:
        # Cattura specifica per HPO degenere (guard di Prompt 1).
        if "HPO degenerate" in str(exc):
            results["hpo_degenerate_error"] = str(exc)
            return _fail(f"HPO degenere: {exc}", results)
        raise

    results["hpo"] = {
        "best_params": predictor.best_params,
    }
    results["phase_wall_time_seconds"]["hpo_and_train"] = time.time() - t0

    # Assertion: il modello è stato addestrato.
    results["assertions"]["model_trained"] = predictor.best_model is not None
    if not results["assertions"]["model_trained"]:
        return _fail("HPO non ha prodotto un best_model (best_params=None).", results)

    # -------------------------------------------------------------------------
    # 4. Final evaluation
    # -------------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("PHASE 4: Final evaluation on hold-out")
    logger.info("=" * 72)
    t0 = time.time()

    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                  log_loss, roc_auc_score)

    if y_test.nunique() < 2:
        return _fail(
            f"Test set ha un'unica classe (y.sum={int(y_test.sum())}). "
            f"Non posso valutare PR-AUC.",
            results,
        )

    probs_test = predictor.predict(X_test)
    pr_auc = float(average_precision_score(y_test, probs_test))
    roc_auc = float(roc_auc_score(y_test, probs_test))
    brier = float(brier_score_loss(y_test, probs_test))
    ll = float(log_loss(y_test, probs_test))
    baseline = float(y_test.mean())

    results["eval"] = {
        "n_test": int(len(y_test)),
        "test_positives": int(y_test.sum()),
        "baseline_prevalence": baseline,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "log_loss": ll,
        "pr_auc_over_baseline_multiple": pr_auc / baseline if baseline > 0 else None,
    }
    results["phase_wall_time_seconds"]["eval"] = time.time() - t0

    # Assertion: pr_auc > 2× baseline (modello ha imparato qualcosa).
    pr_auc_threshold = SMOKE_PR_AUC_PASS_MULTIPLE * baseline
    results["assertions"]["pr_auc_above_2x_baseline"] = pr_auc > pr_auc_threshold
    if pr_auc <= pr_auc_threshold:
        return _fail(
            f"PR-AUC={pr_auc:.4f} <= 2×baseline={pr_auc_threshold:.4f} "
            f"(baseline={baseline:.4%}). Il modello non ha imparato sopra random.",
            results,
        )

    # -------------------------------------------------------------------------
    # 5. Verdict
    # -------------------------------------------------------------------------
    pr_auc_mult = pr_auc / baseline
    if pr_auc_mult >= SMOKE_PR_AUC_GO_MULTIPLE:
        verdict = "GO"
        rec = (
            f"PR-AUC = {pr_auc:.4f} = {pr_auc_mult:.1f}× baseline "
            f"({baseline:.4%}). Modello chiaramente sopra random. "
            f"Procedi con il full run."
        )
    else:
        verdict = "CAUTION"
        rec = (
            f"PR-AUC = {pr_auc:.4f} = {pr_auc_mult:.1f}× baseline. "
            f"Sopra random ma marginale. Valuta se aumentare dark_ratio, "
            f"n_trials o ridurre prediction_horizon prima del full run."
        )

    results["verdict"] = verdict
    results["recommendation"] = rec
    results["wall_time_total_seconds"] = time.time() - t_start
    results["timestamp"] = datetime.now().isoformat()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Report a stdout.
    print()
    print("=" * 72)
    print("  SMOKE TEST REPORT")
    print("=" * 72)
    print(f"  Subsample        : {n_pos} pos / {n_tot} pings ({prevalence:.4%}) "
          f"on {len(selected_mmsi)} vessels")
    print(f"  Phase wall times :")
    for name, secs in results["phase_wall_time_seconds"].items():
        print(f"    {name:<18}: {secs:>8.1f}s")
    print(f"  Total wall time  : {results['wall_time_total_seconds']:.1f}s")
    print("-" * 72)
    print(f"  Test PR-AUC      : {pr_auc:.4f}  (baseline {baseline:.4%}, "
          f"multiple {pr_auc_mult:.1f}×)")
    print(f"  Test ROC-AUC     : {roc_auc:.4f}")
    print(f"  Test Brier       : {brier:.4f}")
    print(f"  Test log-loss    : {ll:.4f}")
    print("-" * 72)
    print(f"  Assertions       :")
    for k, v in results["assertions"].items():
        sym = "✅" if v else "❌"
        print(f"    {sym} {k}")
    print("=" * 72)
    badge = {"GO": "🟢", "CAUTION": "🟡", "NO-GO": "🔴"}[verdict]
    print(f"  {badge} VERDICT: {verdict}")
    print(f"  → {rec}")
    print(f"  📝 Saved: {RESULTS_PATH}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
