"""Diagnostica read-only del target dark-fleet.

Risponde alla domanda: la prevalenza ridicola (0.013% in train) è dovuta
a (a) carenza di blackout reali nel dataset, (b) un bug nel labeling, o
(c) ad altro (es. CV splitter)?

Esegue 6 fasi e salva un report JSON in ``results/target_diagnostic.json``.
Non modifica MAI artefatti di modello. Non chiama l'inferenza Bayesiana.

Uso::

    python scripts/diagnose_target.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Forza stdout in UTF-8 (su Windows il default è cp1252 e blocca i non-ASCII).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# Permetti l'import di src.* lanciando lo script dal root del repo.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep import AISDataPreprocessor

LOG_FMT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger("diagnose_target")


# =============================================================================
# Utility
# =============================================================================
def _detect_mmsi_col(df: pd.DataFrame) -> str:
    for c in ("MMSI", "mmsi", "Mmsi"):
        if c in df.columns:
            return c
    raise KeyError(f"Nessuna colonna MMSI trovata. Colonne: {list(df.columns)}")


def _detect_ts_col(df: pd.DataFrame) -> str:
    for c in ("Timestamp", "timestamp", "BaseDateTime", "time"):
        if c in df.columns:
            return c
    raise KeyError(f"Nessuna colonna Timestamp trovata. Colonne: {list(df.columns)}")


def _gap_buckets(gaps_hours: np.ndarray) -> dict:
    """Bin log-scale dei gap (in ore). Ritorna count per bin."""
    bins = [
        ("<1min", 0, 1 / 60),
        ("1-10min", 1 / 60, 10 / 60),
        ("10min-1h", 10 / 60, 1.0),
        ("1-6h", 1.0, 6.0),
        ("6-12h", 6.0, 12.0),
        ("12-24h", 12.0, 24.0),
        ("1-7d", 24.0, 24.0 * 7),
        (">7d", 24.0 * 7, float("inf")),
    ]
    out = {}
    g = gaps_hours[~np.isnan(gaps_hours)]
    for name, lo, hi in bins:
        out[name] = int(((g >= lo) & (g < hi)).sum())
    return out


def _load_config() -> dict:
    cfg_path = ROOT / "configs" / "pipeline_config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_data(cfg: dict) -> tuple[pd.DataFrame, str]:
    """Tenta nell'ordine: raw_input → processed_output → enriched_dataset.
    Ritorna (df, source_path_used)."""
    paths_cfg = cfg.get("paths", {})
    candidates = [
        paths_cfg.get("raw_input", "data/raw/ais_sample.parquet"),
        paths_cfg.get("processed_output", "data/processed/ais_v1.parquet"),
        paths_cfg.get("enriched_dataset", "data/processed/ais_enriched.parquet"),
    ]
    for p in candidates:
        full = ROOT / p
        if full.exists():
            if full.suffix == ".parquet":
                df = pd.read_parquet(full)
            else:
                df = pd.read_csv(full, low_memory=False)
            return df, str(full)
    raise FileNotFoundError(
        f"Nessuno dei path candidati esiste: {candidates}. "
        f"Lancia prima la simulazione o popola data/raw/."
    )


# =============================================================================
# 1. INVENTORY
# =============================================================================
def phase_inventory(df: pd.DataFrame, mmsi_col: str, ts_col: str) -> dict:
    ts = pd.to_datetime(df[ts_col])
    out = {
        "n_pings": int(len(df)),
        "n_unique_mmsi": int(df[mmsi_col].nunique()),
        "ts_min": str(ts.min()),
        "ts_max": str(ts.max()),
        "bbox": {
            "lat_min": float(df["Lat"].min()) if "Lat" in df.columns else None,
            "lat_max": float(df["Lat"].max()) if "Lat" in df.columns else None,
            "lon_min": float(df["Lon"].min()) if "Lon" in df.columns else None,
            "lon_max": float(df["Lon"].max()) if "Lon" in df.columns else None,
        },
    }
    return out


# =============================================================================
# 2. GAP DETECTION (independente dal labeling)
# =============================================================================
def phase_gaps(df: pd.DataFrame, mmsi_col: str, ts_col: str,
               gap_threshold_hours: float = 12.0) -> dict:
    df = df[[mmsi_col, ts_col]].copy()
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values([mmsi_col, ts_col])
    gaps_h = df.groupby(mmsi_col)[ts_col].diff().dt.total_seconds() / 3600.0
    gaps_arr = gaps_h.values

    is_blackout = gaps_h >= gap_threshold_hours
    blackouts_per_mmsi = (
        df.assign(_blk=is_blackout.values).groupby(mmsi_col)["_blk"].sum().astype(int)
    )
    n_unique_with_blackout = int((blackouts_per_mmsi > 0).sum())
    total_blackouts = int(blackouts_per_mmsi.sum())

    return {
        "gap_histogram_hours": _gap_buckets(gaps_arr),
        "n_gaps_ge_threshold": total_blackouts,
        "n_unique_mmsi_with_blackout": n_unique_with_blackout,
        "top10_mmsi_by_blackout_count": (
            blackouts_per_mmsi.sort_values(ascending=False).head(10)
            .to_dict()
        ),
        "gap_threshold_hours_used": float(gap_threshold_hours),
    }


# =============================================================================
# 3. LABELING SANITY CHECK
# =============================================================================
def phase_labeling(df_in: pd.DataFrame, mmsi_col: str, ts_col: str,
                   cfg: dict, n_blackouts: int) -> dict:
    """Usa AISDataPreprocessor.label_sliding_window in modalità READ-ONLY."""
    data_cfg = cfg.get("data_prep", {})
    bbox = data_cfg.get(
        "bounding_box",
        {"min_lat": -90, "max_lat": 90, "min_lon": -180, "max_lon": 180},
    )
    gap_h = float(data_cfg.get("gap_threshold_hours", 12.0))
    horizon_min = float(data_cfg.get("labeling_horizon_minutes", 60.0))
    downsample_min = int(data_cfg.get("downsample_minutes", 10))

    # Normalizza i nomi colonna a ciò che AISDataPreprocessor si aspetta.
    df = df_in.rename(columns={mmsi_col: "MMSI", ts_col: "Timestamp"}).copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    pp = AISDataPreprocessor(
        bounding_box=bbox,
        gap_threshold_hours=gap_h,
        prediction_horizon_hours=24.0,
        downsample_minutes=downsample_min,
    )

    if not hasattr(pp, "label_sliding_window"):
        return {
            "available": False,
            "reason": (
                "AISDataPreprocessor.label_sliding_window non presente — "
                "la versione corrente di src/data_prep.py potrebbe essere stata "
                "ripristinata. Diagnostica labeling saltata."
            ),
        }

    try:
        df_lab = pp.label_sliding_window(df, horizon_minutes=horizon_min)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"label_sliding_window ha sollevato: {type(exc).__name__}: {exc}",
        }

    pos = int((df_lab["target_dark_fleet"] == 1).sum())
    neg = int((df_lab["target_dark_fleet"] == 0).sum())
    total = pos + neg
    prevalence = (pos / total) if total > 0 else 0.0
    mmsi_contrib = int(
        df_lab.loc[df_lab["target_dark_fleet"] == 1, "MMSI"].nunique()
    )

    # Stima attesa: ogni blackout dovrebbe generare ~ (horizon_min / downsample_min) positivi.
    expected_per_blackout = max(1, int(horizon_min / downsample_min))
    expected_positives = n_blackouts * expected_per_blackout
    ratio_observed_vs_expected = (
        pos / expected_positives if expected_positives > 0 else None
    )

    return {
        "available": True,
        "horizon_minutes": horizon_min,
        "gap_threshold_hours": gap_h,
        "downsample_minutes": downsample_min,
        "positives": pos,
        "negatives": neg,
        "prevalence": prevalence,
        "n_unique_mmsi_with_positive": mmsi_contrib,
        "expected_positives_lower_bound": int(expected_positives),
        "ratio_observed_over_expected": ratio_observed_vs_expected,
    }


# =============================================================================
# 4. MMSI TYPE CONSISTENCY
# =============================================================================
def phase_mmsi_type(df: pd.DataFrame, mmsi_col: str) -> dict:
    dtype = str(df[mmsi_col].dtype)
    sample = df[mmsi_col].dropna().head(3).tolist()
    is_float_likely_int = (
        dtype.startswith("float") and
        all(float(x).is_integer() for x in sample if pd.notna(x))
    )
    issue = None
    if dtype.startswith("float"):
        issue = (
            "CRITICAL: MMSI è float64 ma rappresenta un identificativo intero. "
            "Confronti tra MMSI possono fallire silenziosamente se altre parti "
            "del codice castano a int o str. Forzare a int64 alla load."
            if is_float_likely_int else
            "WARNING: MMSI è float ma contiene valori non interi. Anomalo."
        )
    elif dtype == "object":
        types_seen = set(type(x).__name__ for x in sample)
        if len(types_seen) > 1:
            issue = f"CRITICAL: MMSI è object con tipi misti {types_seen}."

    out = {
        "dtype": dtype,
        "sample_values": [str(x) for x in sample],
        "consistency_issue": issue,
    }
    if issue:
        logger.critical(issue)
    return out


# =============================================================================
# 5. GEOGRAPHIC FILTERING IMPACT
# =============================================================================
def phase_geo_filter(df_in: pd.DataFrame, mmsi_col: str, ts_col: str,
                     cfg: dict, gap_threshold_hours: float = 12.0) -> dict:
    bbox = cfg.get("data_prep", {}).get("bounding_box")
    if not bbox or "Lat" not in df_in.columns or "Lon" not in df_in.columns:
        return {"applied": False, "reason": "bbox o colonne Lat/Lon assenti"}

    mask = (
        (df_in["Lat"] >= bbox["min_lat"]) & (df_in["Lat"] <= bbox["max_lat"]) &
        (df_in["Lon"] >= bbox["min_lon"]) & (df_in["Lon"] <= bbox["max_lon"])
    )
    df_filt = df_in[mask].copy()
    gaps_phase = phase_gaps(df_filt, mmsi_col, ts_col, gap_threshold_hours)
    return {
        "applied": True,
        "bbox": bbox,
        "n_pings_before": int(len(df_in)),
        "n_pings_after": int(len(df_filt)),
        "frac_kept": float(len(df_filt) / max(len(df_in), 1)),
        "gaps_after_filter": gaps_phase,
    }


# =============================================================================
# 6. VERDICT
# =============================================================================
def phase_verdict(gaps_filt: dict | None, gaps_raw: dict, labeling: dict) -> dict:
    n_blackouts_filt = (
        gaps_filt["gaps_after_filter"]["n_gaps_ge_threshold"]
        if gaps_filt and gaps_filt.get("applied") else
        gaps_raw["n_gaps_ge_threshold"]
    )

    if n_blackouts_filt < 50:
        verdict = "DATA-LIMITED"
        rec = (
            "Meno di 50 blackout ≥12h nei dati filtrati geograficamente. "
            "Il segnale è troppo raro per HPO/CV affidabile. "
            "Opzioni: (a) abbassare gap_threshold_hours; "
            "(b) espandere il bounding box; "
            "(c) ricampionare la finestra temporale; "
            "(d) riformulare il target."
        )
    elif (
        labeling.get("available")
        and n_blackouts_filt >= 200
        and labeling["positives"] < 0.1 * labeling.get("expected_positives_lower_bound", 0)
    ):
        verdict = "LABELING-BUG"
        rec = (
            f"{n_blackouts_filt} blackout esistono ma label_sliding_window ha "
            f"prodotto solo {labeling['positives']} positivi "
            f"(<10% dell'atteso {labeling['expected_positives_lower_bound']}). "
            "Debug suggerito: confronti MMSI (dtype mismatch?), "
            "sort temporale prima del shift, gestione dei NaT, "
            "edge cases sull'ultima riga per nave."
        )
    else:
        verdict = "OK"
        rec = (
            "Gap sufficienti e labeling coerente con l'atteso. "
            "Il problema PR-AUC≈0 è altrove: indagare CV splitter "
            "(positivi che non finiscono in tutti i fold), "
            "rapporto train/test stratificato per MMSI-con-blackout, "
            "scelta degli iperparametri Optuna."
        )
    return {"verdict": verdict, "recommendation": rec,
            "n_blackouts_filt": int(n_blackouts_filt)}


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    cfg = _load_config()
    df, src = _load_data(cfg)
    logger.info(f"Caricato dataset da: {src} → {len(df):,} righe")

    mmsi_col = _detect_mmsi_col(df)
    ts_col = _detect_ts_col(df)
    logger.info(f"Colonne rilevate: MMSI='{mmsi_col}', Timestamp='{ts_col}'")

    gap_h = float(cfg.get("data_prep", {}).get("gap_threshold_hours", 12.0))

    report: dict = {"source_path": src, "config_used": cfg.get("data_prep", {})}

    logger.info("[1/6] Inventory…")
    report["inventory"] = phase_inventory(df, mmsi_col, ts_col)

    logger.info("[2/6] Gap detection (raw)…")
    report["gaps_raw"] = phase_gaps(df, mmsi_col, ts_col, gap_h)

    logger.info("[4/6] MMSI type consistency…")
    report["mmsi_type"] = phase_mmsi_type(df, mmsi_col)

    logger.info("[5/6] Geographic filter impact…")
    report["geo_filter"] = phase_geo_filter(df, mmsi_col, ts_col, cfg, gap_h)

    n_blk_for_labeling = (
        report["geo_filter"]["gaps_after_filter"]["n_gaps_ge_threshold"]
        if report["geo_filter"].get("applied") else
        report["gaps_raw"]["n_gaps_ge_threshold"]
    )

    logger.info("[3/6] Labeling sanity check…")
    # Applichiamo il labeling sui dati GEO-filtrati (la pipeline reale fa così).
    if report["geo_filter"].get("applied"):
        bbox = cfg["data_prep"]["bounding_box"]
        mask = (
            (df["Lat"] >= bbox["min_lat"]) & (df["Lat"] <= bbox["max_lat"]) &
            (df["Lon"] >= bbox["min_lon"]) & (df["Lon"] <= bbox["max_lon"])
        )
        df_lab_in = df[mask].copy()
    else:
        df_lab_in = df
    report["labeling"] = phase_labeling(df_lab_in, mmsi_col, ts_col, cfg, n_blk_for_labeling)

    logger.info("[6/6] Verdict…")
    report["summary"] = phase_verdict(
        report["geo_filter"], report["gaps_raw"], report["labeling"]
    )

    # Salvataggio JSON.
    out_path = ROOT / "results" / "target_diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"📝 Report salvato in: {out_path}")

    # Stampa human-readable.
    inv = report["inventory"]
    g_raw = report["gaps_raw"]
    g_filt = report["geo_filter"].get("gaps_after_filter", {})
    lab = report["labeling"]
    summ = report["summary"]

    print()
    print("=" * 72)
    print("  TARGET DIAGNOSTIC REPORT")
    print("=" * 72)
    print(f"  Source           : {src}")
    print(f"  Pings            : {inv['n_pings']:,}")
    print(f"  Unique MMSI      : {inv['n_unique_mmsi']:,}")
    print(f"  Date range       : {inv['ts_min']} → {inv['ts_max']}")
    print(f"  BBox (lat/lon)   : "
          f"[{inv['bbox']['lat_min']:.2f}, {inv['bbox']['lat_max']:.2f}] / "
          f"[{inv['bbox']['lon_min']:.2f}, {inv['bbox']['lon_max']:.2f}]")
    print("-" * 72)
    print("  Gap histogram (raw, hours):")
    for k, v in g_raw["gap_histogram_hours"].items():
        print(f"    {k:>10} : {v:>8,}")
    print(f"  Blackouts >={g_raw['gap_threshold_hours_used']}h (raw)     : "
          f"{g_raw['n_gaps_ge_threshold']:,} "
          f"(distinct MMSI: {g_raw['n_unique_mmsi_with_blackout']})")
    if report["geo_filter"].get("applied"):
        print(f"  Blackouts after geo filter        : "
              f"{g_filt.get('n_gaps_ge_threshold', 'N/A')} "
              f"(distinct MMSI: {g_filt.get('n_unique_mmsi_with_blackout', 'N/A')})")
    print("-" * 72)
    print(f"  MMSI dtype       : {report['mmsi_type']['dtype']}")
    if report["mmsi_type"]["consistency_issue"]:
        print(f"  ⚠  {report['mmsi_type']['consistency_issue']}")
    print("-" * 72)
    if lab.get("available"):
        print(f"  Labeling (sliding_window, horizon={lab['horizon_minutes']}min):")
        print(f"    positives          : {lab['positives']:,}")
        print(f"    prevalence         : {lab['prevalence']:.4%}")
        print(f"    expected lower bound: {lab['expected_positives_lower_bound']:,}")
        print(f"    observed / expected: "
              f"{lab['ratio_observed_over_expected']!s}")
        print(f"    MMSI with ≥1 pos.  : {lab['n_unique_mmsi_with_positive']}")
    else:
        print(f"  Labeling: NOT AVAILABLE ({lab.get('reason', '?')})")
    print("=" * 72)
    print(f"  SUMMARY DIAGNOSIS: {summ['verdict']}")
    print(f"  → {summ['recommendation']}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
