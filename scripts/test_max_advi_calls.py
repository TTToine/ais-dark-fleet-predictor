"""Standalone test: verifica che max_advi_calls sia un hard cap per vessel.

Eseguito DOPO la fix in src/bayesian_mixture.py::_process_single_vessel.
Carica UNA sola nave dal dataset enriched, instanzia il BMM con cap=2 e
n_advi_iter=200 (basso per velocità), conta le invocazioni di pm.fit
provenienti dal main loop e asserisce ≤ 2.

Target wall time: <2 minuti (2 fit × ~10-30s/fit @ 200 step + setup).

Uso::

    python scripts/test_max_advi_calls.py

Exit codes:
    0 → cap rispettato (≤ 2 fit dal main loop)
    1 → cap violato (>2 fit) o errore di esecuzione
    4 → nessuna nave con dati sufficienti
"""
from __future__ import annotations

import inspect
import logging
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "processed" / "ais_enriched.parquet"

# Parametri del test (deliberatamente bassi per girare <2min).
TEST_MAX_ADVI = 2
TEST_N_ADVI_ITER = 200
TEST_MIN_PINGS = 500     # nave deve avere ≥500 ping per essere "non triviale"
TEST_UPDATE_FREQ = 5     # bassissimo: vuole forzare fit a ogni 5 step → se cap
                          # NON funziona, ne partirebbero centinaia in 500 ping

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_max_advi")


def main() -> int:
    if not DATA_PATH.exists():
        logger.error(f"Dataset non trovato: {DATA_PATH}. "
                     f"Lancia prima `python scripts/regenerate_simulated.py`.")
        return 1

    import numpy as np
    import pandas as pd

    df = pd.read_parquet(DATA_PATH)
    logger.info(f"Loaded: {len(df):,} pings, {df['MMSI'].nunique()} vessels")

    candidates = (
        df.groupby("MMSI").size().sort_index().pipe(lambda s: s[s >= TEST_MIN_PINGS])
    )
    if candidates.empty:
        logger.error(f"Nessuna nave con >={TEST_MIN_PINGS} ping.")
        return 4
    selected_mmsi = int(candidates.index[0])
    df_v = df[df["MMSI"] == selected_mmsi].copy().sort_values("Timestamp").reset_index(drop=True)
    if df_v["MMSI"].dtype != np.int64:
        df_v["MMSI"] = df_v["MMSI"].astype(np.int64)
    logger.info(f"Selected vessel MMSI={selected_mmsi} con {len(df_v)} ping.")

    try:
        from src.bayesian_mixture import CausalBayesianMixture
        import src.bayesian_mixture as _bmm_mod
    except ImportError as exc:
        logger.error(f"Import BMM fallito: {exc}")
        return 1

    _original_pm_fit = _bmm_mod.pm.fit
    main_loop_calls = []
    other_calls = []

    def _instrumented_pm_fit(*args, **kwargs):
        caller = "<unknown>"
        try:
            fr = inspect.currentframe()
            if fr is not None and fr.f_back is not None:
                caller = fr.f_back.f_code.co_name
        except Exception:
            pass
        n_req = kwargs.get("n", args[0] if args else None)
        t_fit = time.time()
        try:
            return _original_pm_fit(*args, **kwargs)
        finally:
            entry = {"caller": caller, "n": n_req, "wall": time.time() - t_fit}
            if caller == "_process_single_vessel":
                main_loop_calls.append(entry)
            else:
                other_calls.append(entry)

    bmm = CausalBayesianMixture(window_size=36, n_advi_iter=TEST_N_ADVI_ITER)
    _bmm_mod.pm.fit = _instrumented_pm_fit

    t_start = time.time()
    try:
        df_out = bmm.process_dataframe_causal(
            df_v,
            use_advi=True,
            apply_markov=False,
            update_freq=TEST_UPDATE_FREQ,
            max_advi_calls=TEST_MAX_ADVI,
            n_jobs=1,
        )
    except Exception as exc:
        logger.error(f"BMM ha sollevato: {type(exc).__name__}: {exc}")
        return 1
    finally:
        _bmm_mod.pm.fit = _original_pm_fit

    elapsed = time.time() - t_start

    print()
    print("=" * 72)
    print(f"  TEST max_advi_calls={TEST_MAX_ADVI} (n_advi_iter={TEST_N_ADVI_ITER})")
    print("=" * 72)
    print(f"  Vessel selected      : MMSI={selected_mmsi} ({len(df_v)} pings)")
    print(f"  Wall time            : {elapsed:.1f}s (target <120s)")
    print(f"  Main-loop pm.fit     : {len(main_loop_calls)} (expected <= {TEST_MAX_ADVI})")
    print(f"  Non-main pm.fit      : {len(other_calls)}")
    if main_loop_calls:
        print(f"  Per-fit walls        : {[round(c['wall'], 2) for c in main_loop_calls]}")
        print(f"  n_steps observed     : {[c['n'] for c in main_loop_calls]}")
    print("-" * 72)

    n_main = len(main_loop_calls)
    if n_main > TEST_MAX_ADVI:
        print(f"  FAIL: {n_main} fit dal main loop > cap {TEST_MAX_ADVI}.")
        print(f"        max_advi_calls NON e onorato. Fix B insufficiente.")
        print("=" * 72)
        return 1

    if "prob_regime_sospetto" not in df_out.columns:
        print(f"  FAIL: colonna prob_regime_sospetto assente.")
        print("=" * 72)
        return 1

    print(f"  PASS: {n_main} fit dal main loop <= cap {TEST_MAX_ADVI}.")
    print(f"        max_advi_calls e onorato come hard cap.")
    if elapsed > 120:
        print(f"  WARN: Wall {elapsed:.0f}s > 120s target (cap rispettato ma lento).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
