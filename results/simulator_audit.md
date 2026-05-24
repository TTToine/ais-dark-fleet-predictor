# Simulator Signal Audit

**Date**: 2026-05-24
**Auditor**: read-only inspection of `run_pipeline.py::_generate_simulation_study_dataset`
**Trigger**: smoke_test 2026-05-24 02:22:43 reported PR-AUC test = 0.0057 ≈ baseline.
Need to determine if simulator generates a learnable signal at all.
**Scope**: code inspection only. No data analysis, no execution.

---

## A. Locating the simulator

Single function, defined in [run_pipeline.py:215-372](run_pipeline.py:215):

```python
def _generate_simulation_study_dataset(n_ships: int = 60,
                                       n_days: int = 7,
                                       dark_ratio: float = 0.20,
                                       bbox: dict = None,
                                       seed: int = 42) -> tuple:
```

Called from `run_phase1_preprocessing()` ([run_pipeline.py:395](run_pipeline.py:395)) only when
the raw input parquet is absent — which is the default path on this project (no
real AIS feed yet).

**Vessel-type decision** ([run_pipeline.py:239-242](run_pipeline.py:239)):
```python
n_dark = int(round(n_ships * dark_ratio))
n_normal = n_ships - n_dark
is_dark_flags = [True] * n_dark + [False] * n_normal
rng.shuffle(is_dark_flags)
```

So *which* vessel is dark is a pure random label assigned at construction time,
independent of any kinematic property.

---

## B. Dependency trace: kinematics → blackout decision

### B.1 — How is blackout *timing* chosen?

Lines [276-289](run_pipeline.py:276):
```python
blackouts = []
if is_dark:
    n_blackouts = rng.integers(1, 4)
    for _ in range(n_blackouts):
        start_offset_min = rng.integers(60, total_minutes - 24 * 60)
        duration_h = rng.uniform(8, 24)
        blackouts.append((start_offset_min, duration_h))
else:
    # ~30% delle navi normali hanno UN piccolo gap accidentale
    if rng.random() < 0.3:
        start_offset_min = rng.integers(60, total_minutes - 4 * 60)
        duration_h = rng.uniform(0.5, 3.0)
        blackouts.append((start_offset_min, duration_h))
```

**Answer: (a) + (c).** Blackout *timing* is drawn uniformly at random
(`rng.integers(60, total_minutes - 24*60)`) at the *start* of the vessel
simulation, before any kinematic state exists. The only state dependency is
on the vessel's static *identity* (`is_dark` flag, assigned at random). There
is **no coupling between blackout timing and kinematic state**.

Important implication: the choice "this vessel goes dark at minute X" is
made offline at t=0 — it is not a state-dependent decision the simulator
makes later when it "sees" the vessel slow down.

### B.2 — Is there any kinematic alteration BEFORE a blackout?

Lines [319-330](run_pipeline.py:319) — **the critical block**:
```python
# Comportamento differente durante "pre-blackout" per le dark ships
# (rallentano, cambiano rotta → segnale che il regime detector dovrebbe catturare)
near_blackout = False
if is_dark and bo_idx < len(blackouts):
    next_bo_start = blackouts[bo_idx][0]
    if 0 <= (next_bo_start - elapsed_min) < 120:  # 2h prima
        near_blackout = True

if near_blackout:
    sog_effective = max(0, base_sog * 0.4 + rng.normal(0, sog_noise))
    heading_drift = rng.normal(0, 8)  # rotta più erratica
else:
    sog_effective = max(0, base_sog + rng.normal(0, sog_noise))
    heading_drift = rng.normal(0, 2)
```

**Yes, there is a kinematic signal**, and it is non-trivial:

| feature | normal regime | pre-blackout regime (last 120 min) | magnitude |
|---|---|---|---|
| `SOG` mean | `base_sog` (3-16 kn) | `0.4 × base_sog` (1.2-6.4 kn) | **-60%** |
| `heading_drift` σ | 2° per ping | 8° per ping | **+4×** |

Both are large effects, *especially* the SOG drop. The heading-drift increase
will appear downstream as a larger `turn_rate` (one of the engineered
features) and the SOG drop as a negative `speed_acc`.

### B.3 — Is the signal absent or minimal?

**No, the signal is present and large.** A 60% drop in mean speed sustained
over 12 consecutive pings (sampling = 10 min, window = 2 h → 12 pings) is a
massive deviation from a vessel's baseline. Any classifier with access to
recent rolling-window statistics of SOG and turn-rate should detect it
trivially — *if* the labeling and feature pipeline preserve it intact.

---

## C. Labeling-vs-simulation alignment

The sliding-window labeler marks a ping at time t as positive iff a blackout
starts in `(t, t + horizon_minutes]`. Current config:
[configs/pipeline_config.yaml](configs/pipeline_config.yaml):
```yaml
data_prep:
  labeling_horizon_minutes: 120.0
```

**Alignment with the simulator's `near_blackout` window**:

| | value |
|---|---|
| Simulator pre-blackout window | `[t_blackout - 120, t_blackout)` |
| Labeling positive window | `(t_ping, t_ping + 120]` (positive ⟺ t_blackout ∈ this window) |
| Overlap | **Exactly the same 120-minute span**, by construction. |

So **every** positive-labeled ping is, by construction, a ping where the
simulator was in the `near_blackout=True` branch — meaning its `SOG` and
`heading_drift` were drawn from the suspicious distribution. The labeling
and the kinematic alteration are perfectly aligned.

Expected count of signal-bearing positives per dark vessel per blackout:
- pre-blackout window = 120 min
- sampling cadence = 10 min (with jitter ±1 min)
- ⇒ **~12 positive pings per blackout, each with altered SOG and heading**

Sanity check vs observed: smoke at 30 vessels reported 259 positives. With
~15 dark vessels × 2 average blackouts/vessel × 12 pings ≈ 360 expected.
259 is within range given geographic filtering (`bbox`) and validity
filters drop some pings.

---

## D. Verdict

### **SIGNAL-PRESENT**

Justification:
1. The simulator explicitly creates a 120-minute pre-blackout regime
   ([run_pipeline.py:319-330](run_pipeline.py:319)) where mean SOG drops by 60% and
   heading variability quadruples.
2. The labeling horizon ([configs/pipeline_config.yaml:20](configs/pipeline_config.yaml:20))
   is set to 120 minutes — exactly the same span. Positive-labeled pings
   are guaranteed to belong to the altered-kinematics regime.
3. The effect size (60% speed drop) is far above any plausible noise floor;
   `sog_noise` is on the order of 1-2 knots while the drop is 4-10 knots in
   absolute terms.

**Therefore the task is learnable in principle.** A model that fails to
distinguish positive from negative pings on this dataset is not failing
because of an absent signal in the generative process — it is failing
either:

- in *feature engineering* (signal preserved but obscured by the way
  `speed_acc`/`turn_rate` are computed),
- in *the BMM* (rolling window 36 = 6 h is **3× wider than the 2 h
  signal**, so the BMM averages 2 h of "suspicious" + 4 h of "normal" and
  may produce a weak posterior even though the underlying signal is
  strong),
- in *evaluation* (small holdout, 6 vessels in smoke; one bad fold draw
  produces near-random PR-AUC by sheer variance), or
- in *labeling* (e.g., positives lost in the geographic/physical-validity
  filters before they reach training).

Note specifically the BMM window mismatch: a 6 h causal window observing a
sudden 2 h speed drop should still show a mixture with weight ≈ 1/3 on
the "suspicious" component, but the BMM's mean over the window will be
dominated by the 4 h of normal behavior. **This is a likely partial
explanation for the modest BMM variance** observed in the smoke
(var = 0.04, mean = 0.39, range [0.09, 0.82]) — the model rarely commits
above 0.82 because no window is *fully* suspicious.

### Recommendations (out of scope for this audit, captured for triage)

1. **Verify positive-ping retention rate after the geographic/validity
   filters**: count positives before and after `filter_geographic_area` and
   `validate_physical_plausibility` to confirm the signal isn't pruned.
2. **Verify the rolling-window length vs signal length**: try
   `window_size = 12` (= 120 min, matching the signal) on a single vessel
   and inspect whether `prob_regime_sospetto` peaks more sharply just
   before a blackout.
3. **Check the holdout split** (already identified independently): with
   smoke's 6-vessel holdout, sampling variance dominates. The full run
   on 120 vessels with `holdout_split_by_mmsi` should be far more stable.

---

## E. Minimal fix proposal

**Not required.** The simulator already injects a learnable signal of
substantial magnitude. No simulator modification is recommended at this
time.

If, after the full pipeline run, the model still fails to learn, the next
audit should target the BMM window length and the feature-engineering
pipeline (`engineer_causal_features` in `src/data_prep.py`) — not the
simulator.

If a simulator strengthening were ever necessary (e.g., for a stress test
of the BMM under weaker signal), a minimal in-place sharpening would be
to make the regime change abrupt rather than uniform across the window —
e.g., a logistic ramp peaking 30 min before blackout — but this would
*decrease* learnability, not increase it. The current uniform-across-120-min
regime is already the easiest possible signal shape for a sliding-window
classifier to detect.

---

## Audit acceptance summary

- [x] Sections A-E present
- [x] Verdict in D is explicit and justified by quoted code
- [x] No simulator modification proposed (signal is sufficient)
- [x] No pipeline run launched, no data on disk inspected
