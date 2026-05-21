# Recovery Log — 2026-05-21

Trace of the multi-step recovery from a 17-hour pipeline run that produced
PR-AUC = 0.0005 (random) with 12 positives total on simulated data.

## What was broken
- `df['MMSI']` was `float64` instead of `int64` (latent bug). `downsample_causal` used `groupby('MMSI', group_keys=False)` which silently dropped MMSI from output columns; the previous parquet output kept it as float64 via a different code path.
- `AISDataPreprocessor.label_sliding_window` and `label_last_ping` were **absent** from the working tree. Last committed version in `c29d62a` never had them — they only ever lived in stash `31d9e3a`. Production pipeline silently fell back to `create_causal_target` (retrospective Last-Ping).
- Simulator config used `dark_ratio=0.20` over `n_days=7` with `n_ships=60`, producing only **12 blackouts ≥12h** — insufficient signal for any HPO / CV to converge.
- `configs/pipeline_config.yaml` had a duplicate `simulation:` block; YAML kept only the last (silently overriding the first).

## What was changed
- **Restored labeling**: `label_sliding_window` and `label_last_ping` recovered from stash `31d9e3a` into `src/data_prep.py`; module-level wrappers added so `from src.data_prep import label_sliding_window` works. Dispatcher in `run_pipeline.py` Phase 1 restored, plus `--labeling` CLI flag.
- **MMSI int64 hard invariant**: helper `_ensure_mmsi_int64()` in `data_prep.py`; called at `load_data` exit, `downsample_causal` exit, simulator exit. `assert dtype == int64` at the entry of both labelers and at the BMM per-vessel split. Simulator MMSI generation moved to real MID range `[200M, 800M)` via `np.random.choice` of int64.
- **downsample_causal fix**: removed `group_keys=False` so MMSI survives `reset_index()` as a real column.
- **Simulator config bump**: `n_ships=120`, `n_days=14`, `dark_ratio=0.50`, `random_seed=42` (added). Duplicate `simulation:` block consolidated. Labeling fields exposed in YAML.
- **README disclosure**: new "Current Data Status: Simulated, Not Real" section after intro; Quick Start note; new Limitations item #1.

## Evidence saved
- `results/target_diagnostic_pre_fix.json` — 12 blackouts, MMSI float64, verdict **DATA-LIMITED**.
- `results/target_diagnostic_post_fix.json` — 86 blackouts ≥12h, 503 sliding-window positives (0.227% prevalence), MMSI int64, ratio observed/expected = 0.97, verdict **OK**.
- `results/target_diagnostic.json` — canonical copy of post_fix.

## What remains TODO
- **Smoke test**: a short-budget end-to-end run (e.g. `n_trials=5`, `n_splits=3`) to validate that Phases 2-3 work with the regenerated data and the HPO guard does NOT trip.
- **Full pipeline re-run** with production budget, only after the smoke test passes.
- **Real AIS data connection** (AISHub / Spire / equivalent). Set up `data/real/` with a loader that respects the int64 MMSI invariant.
- **`hmm_cache.py` cleanup** — file was reverted to the original broken state (bare excepts, hash-randomized key). Will be re-fixed in a later cleanup pass; deliberately out of scope here.
