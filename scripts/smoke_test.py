"""End-to-end smoke test su sottocampione deterministico.

Obiettivo: in <30 minuti (target <15) validare l'intera pipeline (data prep
→ BMM → HPO → eval) PRIMA di lanciare il full run (~17h).

Versione 2 (post-mortem v1 morto a 7h):
- ``n_advi_iter`` ora configurabile (era hardcoded 1000 in main loop).
- Smoke usa 500: dimezza il tempo per-fit.
- Vessel selection filtra solo navi con ≥1 positivo (G).
- Pre-flight count positivi nel subsample (F).
- Hard timeout a 30min, exit 3 (D).
- Selezione vessel persistita per riproducibilità tra runs (G).
- Logging esplicito del primo ADVI fit time vs n_advi_iter (H).
- Cache: NON wirata (BMM cache dead code in pipeline). Vedi RECOVERY_LOG.

NON sovrascrive ``models/`` (usa ``models/smoke/``) né ``.cache/{hmm,bmm}/``
di produzione (usa ``.cache/smoke/`` — placeholder per quando wireremo cache).

Uso::

    python scripts/smoke_test.py

Exit codes:
    0  GO o CAUTION (smoke OK)
    1  NO-GO (assert fallita)
    3  TIMEOUT (>30 min wall time)
    4  pre-flight failure (subsample troppo sparso)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Forza UTF-8 su stdout/stderr (Windows cp1252 default rompe gli emoji).
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
VESSELS_PATH = ROOT / "results" / "smoke_test_vessels.json"
DATA_PATH = ROOT / "data" / "processed" / "ais_enriched.parquet"

# Placeholder env var (sarà onorato quando wireremo cache, vedi RECOVERY_LOG).
os.environ["BMM_CACHE_DIR"] = str(SMOKE_CACHE_DIR)

# =============================================================================
# Budget smoke (calibrato post-mortem del run 7h di v1).
# =============================================================================
SMOKE_N_MMSI = 30               # 30 navi con ≥1 positivo: con prevalenza ~0.4%
                                # 10 navi → ~77 pos totali → ogni val fold di
                                # GroupTimeSeriesSplit vede 0 positivi → HPO degenere.
                                # 30 navi triplicano i positivi assoluti e la copertura
                                # gruppi nel CV, mantenendoci entro il budget tempo.
SMOKE_N_TRIALS = 3              # HPO
SMOKE_N_SPLITS = 2              # CV folds
SMOKE_UPDATE_FREQ = 50          # 1 ADVI fit ogni 50 ping (vs 10-20 produzione)
SMOKE_MAX_ADVI_CALLS = 3        # cap per nave (vs ~30-50 produzione)
SMOKE_N_ADVI_ITER = 500         # KEY FIX: ADVI step per fit (vs 1000 produzione)
SMOKE_N_JOBS = 1

# Hard timeout (D).
HARD_TIMEOUT_SECONDS = 30 * 60

# Soglie verdetto e pre-flight.
SMOKE_MIN_POSITIVES = 10
SMOKE_PR_AUC_GO_MULTIPLE = 5.0
SMOKE_PR_AUC_PASS_MULTIPLE = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("smoke")


# =============================================================================
# Exit helpers
# =============================================================================
def _save_and_exit(results: dict, verdict: str, exit_code: int, reason: str) -> int:
    results["verdict"] = verdict
    results["failure_reason"] = reason
    results["timestamp"] = datetime.now().isoformat()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    badge = {"NO-GO": "❌", "TIMEOUT": "⏱️", "PREFLIGHT-FAIL": "🚧"}.get(verdict, "❌")
    print("\n" + "=" * 72)
    print(f"  {badge} {verdict}: {reason}")
    print(f"  📝 Saved: {RESULTS_PATH}")
    print("=" * 72)
    return exit_code


def _check_timeout(t_start: float, results: dict) -> None:
    """Raise SystemExit(3) se sforiamo il budget hard."""
    elapsed = time.time() - t_start
    if elapsed > HARD_TIMEOUT_SECONDS:
        results["wall_time_total_seconds"] = elapsed
        logger.critical(
            f"⏱️ HARD TIMEOUT raggiunto: {elapsed:.0f}s > {HARD_TIMEOUT_SECONDS}s. "
            f"Termino con exit 3."
        )
        sys.exit(_save_and_exit(
            results, "TIMEOUT", 3,
            f"Wall time {elapsed:.0f}s superato hard budget {HARD_TIMEOUT_SECONDS}s.",
        ))


# =============================================================================
# G: vessel selection con stability check
# =============================================================================
def _select_vessels(df, n_select: int) -> list[int]:
    """Filtra MMSI con ≥1 positivo, ordina deterministicamente, prendi primi n.

    Se ``results/smoke_test_vessels.json`` esiste, verifica che la selezione
    corrente coincida con quella salvata (riproducibilità tra runs).
    """
    import numpy as np
    eligible = (
        df.loc[df["target_dark_fleet"] == 1, "MMSI"]
        .dropna()
        .astype(np.int64)
        .unique()
        .tolist()
    )
    eligible_sorted = sorted(eligible)  # ordinamento numerico deterministico
    if len(eligible_sorted) < n_select:
        # Caso patologico: meno di n_select navi con positivi. Prendiamo tutte.
        logger.warning(
            f"Solo {len(eligible_sorted)} navi con positivi disponibili "
            f"(richieste {n_select}). Userò tutte le eligible."
        )
        selected = eligible_sorted
    else:
        selected = eligible_sorted[:n_select]

    # Stability check (G): se persistito, deve combaciare.
    if VESSELS_PATH.exists():
        try:
            prev = json.loads(VESSELS_PATH.read_text(encoding="utf-8"))
            prev_list = sorted(int(m) for m in prev.get("selected_mmsi", []))
            curr_list = sorted(int(m) for m in selected)
            if prev_list != curr_list:
                logger.warning(
                    f"⚠️  Selezione vessel CAMBIATA tra runs!\n"
                    f"   precedente: {prev_list}\n"
                    f"   corrente  : {curr_list}\n"
                    f"   Possibile causa: nuovo random_seed, dark_ratio diverso, "
                    f"o riordinamento non-deterministico a monte."
                )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(f"smoke_test_vessels.json non leggibile: {exc}")

    # Persisti per il prossimo run.
    VESSELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VESSELS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_mmsi": [int(m) for m in selected],
                "selection_strategy": "vessels with >=1 positive, sorted ascending, first N",
                "n_select": n_select,
                "timestamp": datetime.now().isoformat(),
            },
            f, indent=2,
        )
    return [int(m) for m in selected]


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    t_start = time.time()
    results: dict = {
        "smoke_config": {
            "n_mmsi": SMOKE_N_MMSI,
            "n_trials": SMOKE_N_TRIALS,
            "n_splits": SMOKE_N_SPLITS,
            "update_freq": SMOKE_UPDATE_FREQ,
            "max_advi_calls": SMOKE_MAX_ADVI_CALLS,
            "n_advi_iter": SMOKE_N_ADVI_ITER,
            "n_jobs": SMOKE_N_JOBS,
            "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
        },
        "phase_wall_time_seconds": {},
        "assertions": {},
    }
    SMOKE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Data prep + smart subsample (G)
    # -------------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("PHASE 1: Data prep + subsample")
    logger.info("=" * 72)
    t0 = time.time()

    import numpy as np
    import pandas as pd

    if not DATA_PATH.exists():
        return _save_and_exit(
            results, "NO-GO", 1,
            f"Dataset non trovato: {DATA_PATH}. "
            f"Lancia prima `python scripts/regenerate_simulated.py`.",
        )

    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded: {len(df):,} pings, {df['MMSI'].nunique()} vessels")

    selected_mmsi = _select_vessels(df, SMOKE_N_MMSI)
    df_sub = df[df["MMSI"].isin(selected_mmsi)].copy()
    df_sub = df_sub.sort_values(["MMSI", "Timestamp"]).reset_index(drop=True)
    if df_sub["MMSI"].dtype != np.int64:
        df_sub["MMSI"] = df_sub["MMSI"].astype(np.int64)

    n_pos = int(df_sub["target_dark_fleet"].sum())
    n_tot = len(df_sub)
    prevalence = n_pos / n_tot if n_tot else 0.0
    logger.info(
        f"Subsample: {n_pos} pos / {n_tot} pings ({prevalence:.4%}) "
        f"on {len(selected_mmsi)} vessels"
    )

    results["data_prep"] = {
        "n_pings": n_tot,
        "n_vessels": len(selected_mmsi),
        "n_positives": n_pos,
        "prevalence": prevalence,
        "selected_mmsi": selected_mmsi,
    }
    results["phase_wall_time_seconds"]["data_prep"] = time.time() - t0

    # F: pre-flight positive count → exit code 4 se troppo sparso.
    if n_pos < SMOKE_MIN_POSITIVES:
        return _save_and_exit(
            results, "PREFLIGHT-FAIL", 4,
            f"Subsample too sparse for meaningful smoke test - "
            f"choose different vessel subset or use stratified sampling "
            f"(n_pos={n_pos} < soglia {SMOKE_MIN_POSITIVES}).",
        )
    results["assertions"]["positives_ge_min"] = True
    _check_timeout(t_start, results)

    # -------------------------------------------------------------------------
    # 2. Bayesian inference (BMM) — budget ridotto
    # -------------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info(
        f"PHASE 2: Bayesian inference (n_advi_iter={SMOKE_N_ADVI_ITER}, "
        f"max_advi_calls={SMOKE_MAX_ADVI_CALLS}, update_freq={SMOKE_UPDATE_FREQ})"
    )
    logger.info("=" * 72)
    t0 = time.time()

    try:
        from src.bayesian_mixture import CausalBayesianMixture
    except ImportError as exc:
        return _save_and_exit(results, "NO-GO", 1, f"Import BMM fallito: {exc}")

    # KEY FIX: passiamo n_advi_iter al costruttore (era hardcoded 1000).
    bmm = CausalBayesianMixture(window_size=36, n_advi_iter=SMOKE_N_ADVI_ITER)

    # H: log esplicito che n_advi_iter è effettivamente onorato.
    # H (refined): log OGNI call a pm.fit con caller, n, wall time.
    # Motivo: bayesian_mixture.py ha 4 call site a pm.fit (linee 67, 446, 877, 1053).
    # Il primo call durante process_dataframe_causal NON è necessariamente quello
    # di _process_single_vessel (il main loop che n_advi_iter configura): EB
    # pilot/PS/PPC possono andare prima. Loggando solo il primo si rischia un
    # falso-positivo ("n=500 OK") mentre il main loop sta in realtà girando
    # con un altro n. Soluzione: registriamo TUTTI i call e in coda asseriamo
    # specificamente sui call provenienti da _process_single_vessel.
    import inspect
    import src.bayesian_mixture as _bmm_mod
    _original_pm_fit = _bmm_mod.pm.fit
    _advi_call_log: list[dict] = []

    def _instrumented_pm_fit(*args, **kwargs):
        # Identifica il chiamante diretto di pm.fit (= la funzione BMM che lo invoca).
        caller_name = "<unknown>"
        try:
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                caller_name = frame.f_back.f_code.co_name
        except Exception:
            pass
        n_requested = kwargs.get("n", args[0] if args else None)
        call_idx = len(_advi_call_log)
        t_fit = time.time()
        try:
            result = _original_pm_fit(*args, **kwargs)
        finally:
            elapsed = time.time() - t_fit
        record = {
            "call_idx": call_idx,
            "caller": caller_name,
            "n_steps": n_requested,
            "wall_seconds": round(elapsed, 4),
        }
        _advi_call_log.append(record)
        logger.info(
            f"H-CHECK [call {call_idx:03d}] caller={caller_name} "
            f"n={n_requested} wall={elapsed:.2f}s"
        )
        return result

    # Per-vessel loop con timeout check (fix v3):
    # Il v2 chiamava process_dataframe_causal sull'intero subsample come
    # singolo blocco. Se max_advi_calls non veniva onorato (v2 bug), il
    # _check_timeout tra fasi non poteva mai scattare perché Phase 2 non
    # ritornava. v3: chiamiamo process_dataframe_causal per ogni nave, e
    # tra una nave e l'altra controlliamo il budget. Granularità: 1 vessel.
    _bmm_mod.pm.fit = _instrumented_pm_fit
    all_vessels = sorted(int(m) for m in df_sub["MMSI"].unique())
    processed_dfs: list = []
    try:
        for i, m in enumerate(all_vessels, start=1):
            df_v = df_sub[df_sub["MMSI"] == m].copy()
            t_vessel = time.time()
            try:
                df_v_out = bmm.process_dataframe_causal(
                    df_v,
                    use_advi=True,
                    apply_markov=True,
                    update_freq=SMOKE_UPDATE_FREQ,
                    max_advi_calls=SMOKE_MAX_ADVI_CALLS,
                    n_jobs=1,  # per-vessel: sequenziale per definizione
                )
            except Exception as exc:
                return _save_and_exit(
                    results, "NO-GO", 1,
                    f"BMM ha sollevato su vessel {m}: {type(exc).__name__}: {exc}",
                )
            processed_dfs.append(df_v_out)
            elapsed_total = time.time() - t_start
            logger.info(
                f"Vessel {i}/{len(all_vessels)} (MMSI={m}) done in "
                f"{time.time()-t_vessel:.1f}s — total elapsed {elapsed_total:.0f}s "
                f"(advi_calls so far: {len(_advi_call_log)})"
            )
            # Timeout per-vessel: fires mid-Phase-2, garantisce <30 min hard.
            if elapsed_total > HARD_TIMEOUT_SECONDS:
                results["phase_wall_time_seconds"]["bayesian"] = time.time() - t0
                results["timeout_after_vessels"] = i
                results["advi_call_log"] = _advi_call_log
                results["advi_calls_by_caller"] = dict(
                    __import__("collections").Counter(c["caller"] for c in _advi_call_log)
                )
                return _save_and_exit(
                    results, "TIMEOUT", 3,
                    f"Phase 2 timeout after {i}/{len(all_vessels)} vessels, "
                    f"elapsed={elapsed_total:.0f}s > {HARD_TIMEOUT_SECONDS}s.",
                )
        df_bmm = pd.concat(processed_dfs, ignore_index=True)
    finally:
        _bmm_mod.pm.fit = _original_pm_fit  # ripristina sempre

    # Persistiamo SEMPRE il call log nel JSON, anche se vuoto, per debuggability.
    results["advi_call_log"] = _advi_call_log

    if not _advi_call_log:
        logger.warning("Nessun fit ADVI registrato (smoke con update_freq alto?)")
    else:
        avg_t = sum(c["wall_seconds"] for c in _advi_call_log) / len(_advi_call_log)
        results["advi_fit_stats"] = {
            "n_fits": len(_advi_call_log),
            "n_requested_first": _advi_call_log[0]["n_steps"],
            "avg_seconds_per_fit": avg_t,
            "total_advi_seconds": sum(c["wall_seconds"] for c in _advi_call_log),
        }
        # Aggregato per caller, utile a colpo d'occhio sul JSON.
        from collections import Counter
        caller_counts = Counter(c["caller"] for c in _advi_call_log)
        results["advi_calls_by_caller"] = dict(caller_counts)
        logger.info(
            f"ADVI: {len(_advi_call_log)} fit totali, avg {avg_t:.2f}s/fit, "
            f"totale {sum(c['wall_seconds'] for c in _advi_call_log):.1f}s. "
            f"Breakdown per caller: {dict(caller_counts)}"
        )

    # H-CHECK (refined): assert specifici sul MAIN per-vessel loop.
    # _process_single_vessel è il consumer di self.n_advi_iter. Gli EB/PS/PPC
    # call usano i loro propri n_advi_steps e NON ci interessano qui.
    main_loop_calls = [c for c in _advi_call_log if c["caller"] == "_process_single_vessel"]
    h_count_ok = len(main_loop_calls) >= 3
    h_n_steps_ok = all(c["n_steps"] == SMOKE_N_ADVI_ITER for c in main_loop_calls) \
        if main_loop_calls else False
    results["assertions"]["h_check_main_loop_count_ge_3"] = h_count_ok
    results["assertions"]["h_check_main_loop_n_steps_correct"] = h_n_steps_ok
    if not (h_count_ok and h_n_steps_ok):
        logger.critical(
            f"H-CHECK FAILED: main loop n_advi_iter not honored. "
            f"main_loop_calls={len(main_loop_calls)} (need ≥3), "
            f"n_steps observed={[c['n_steps'] for c in main_loop_calls]} "
            f"(expected all == {SMOKE_N_ADVI_ITER})."
        )
    else:
        logger.info(
            f"H-CHECK OK: {len(main_loop_calls)} call da _process_single_vessel, "
            f"tutti con n={SMOKE_N_ADVI_ITER}."
        )

    # Assert BMM output non-degenere.
    if "prob_regime_sospetto" not in df_bmm.columns:
        return _save_and_exit(
            results, "NO-GO", 1,
            "BMM non ha prodotto la colonna prob_regime_sospetto.",
        )
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
        return _save_and_exit(
            results, "NO-GO", 1,
            f"prob_regime_sospetto var={var:.6e} (≈0). BMM collassato.",
        )
    if not results["assertions"]["bmm_not_all_nan"]:
        return _save_and_exit(
            results, "NO-GO", 1, "prob_regime_sospetto interamente NaN.",
        )
    logger.info(f"BMM OK: var={var:.4f}, NaN={n_nan}/{len(probs)}")
    _check_timeout(t_start, results)

    # -------------------------------------------------------------------------
    # 3. HPO + training
    # -------------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info(f"PHASE 3: HPO (n_trials={SMOKE_N_TRIALS}, n_splits={SMOKE_N_SPLITS})")
    logger.info("=" * 72)
    t0 = time.time()

    try:
        from src.gb_training import DarkFleetPredictor
    except ImportError as exc:
        return _save_and_exit(results, "NO-GO", 1, f"Import gb_training fallito: {exc}")

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

    # SMOKE-ONLY: il cv_splitter di default di DarkFleetPredictor
    # (GroupTimeSeriesSplit in gb_training.py) richiede first_ts(val_ship) >=
    # train_end + gap. Nel simulatore tutte le navi sono attive sull'intero
    # range temporale (14gg) → la condizione non è MAI soddisfatta → 0 fold
    # validi → HPO degenere ("no positive samples in any CV fold").
    # Per lo smoke usiamo GroupTimeSeriesSplitWithGap (cv_splitters.py) con
    # horizon_hours = tutto il dataset: val = righe delle val_ships dopo il
    # cutoff temporale. Garantisce positivi nei fold val. In prod va deciso
    # quale splitter è quello "vero": vedi RECOVERY_LOG.
    from src.cv_splitters import GroupTimeSeriesSplitWithGap
    predictor.cv_splitter = GroupTimeSeriesSplitWithGap(
        n_splits=SMOKE_N_SPLITS,
        gap_hours=24.0,
        horizon_hours=24.0 * 14,   # tutto il range del simulatore
        random_state=42,
    )
    logger.info(
        "SMOKE: cv_splitter sostituito con GroupTimeSeriesSplitWithGap "
        f"(n_splits={SMOKE_N_SPLITS}, gap=24h, horizon=14d)"
    )
    X_train, y_train = predictor.prepare_data_for_cv(train_df)
    X_test, y_test = predictor.prepare_data_for_cv(test_df)

    try:
        predictor.optimize_and_train(X_train, y_train)
    except RuntimeError as exc:
        if "HPO degenerate" in str(exc):
            results["hpo_degenerate_error"] = str(exc)
            return _save_and_exit(results, "NO-GO", 1, f"HPO degenere: {exc}")
        raise

    results["hpo"] = {"best_params": predictor.best_params}
    results["phase_wall_time_seconds"]["hpo_and_train"] = time.time() - t0
    results["assertions"]["model_trained"] = predictor.best_model is not None
    if not results["assertions"]["model_trained"]:
        return _save_and_exit(results, "NO-GO", 1, "HPO senza best_model.")
    _check_timeout(t_start, results)

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
        return _save_and_exit(
            results, "NO-GO", 1,
            f"Test set ha un'unica classe (y.sum={int(y_test.sum())}).",
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

    pr_auc_threshold = SMOKE_PR_AUC_PASS_MULTIPLE * baseline
    results["assertions"]["pr_auc_above_2x_baseline"] = pr_auc > pr_auc_threshold
    if pr_auc <= pr_auc_threshold:
        return _save_and_exit(
            results, "NO-GO", 1,
            f"PR-AUC={pr_auc:.4f} <= 2×baseline={pr_auc_threshold:.4f} "
            f"(baseline={baseline:.4%}).",
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
            f"Sopra random ma marginale. Valuta dark_ratio↑ o n_trials↑ "
            f"prima del full run."
        )

    results["verdict"] = verdict
    results["recommendation"] = rec
    results["wall_time_total_seconds"] = time.time() - t_start
    results["timestamp"] = datetime.now().isoformat()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    # Stdout report.
    print()
    print("=" * 72)
    print("  SMOKE TEST REPORT")
    print("=" * 72)
    print(f"  Subsample        : {n_pos} pos / {n_tot} pings ({prevalence:.4%}) "
          f"on {len(selected_mmsi)} vessels")
    print(f"  Phase wall times :")
    for name, secs in results["phase_wall_time_seconds"].items():
        print(f"    {name:<18}: {secs:>8.1f}s")
    print(f"  Total wall time  : {results['wall_time_total_seconds']:.1f}s "
          f"(hard budget {HARD_TIMEOUT_SECONDS}s)")
    if "advi_fit_stats" in results:
        s = results["advi_fit_stats"]
        print(f"  ADVI stats       : {s['n_fits']} fits, "
              f"avg {s['avg_seconds_per_fit']:.2f}s/fit "
              f"(first n_advi_iter={s['n_requested_first']})")
    print("-" * 72)
    print(f"  Test PR-AUC      : {pr_auc:.4f}  (baseline {baseline:.4%}, "
          f"{pr_auc_mult:.1f}×)")
    print(f"  Test ROC-AUC     : {roc_auc:.4f}")
    print(f"  Test Brier       : {brier:.4f}")
    print(f"  Test log-loss    : {ll:.4f}")
    print("-" * 72)
    print(f"  Assertions       :")
    for k, v in results["assertions"].items():
        sym = "✅" if v else "❌"
        print(f"    {sym} {k}")
    print("=" * 72)
    badge = {"GO": "🟢", "CAUTION": "🟡"}[verdict]
    print(f"  {badge} VERDICT: {verdict}")
    print(f"  → {rec}")
    print(f"  📝 Saved: {RESULTS_PATH}")
    print(f"  📝 Vessels: {VESSELS_PATH}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
