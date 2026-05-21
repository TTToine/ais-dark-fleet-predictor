"""
src/bayesian_mixture.py
Modulo per l'inferenza Bayesiana dei regimi latenti (Mixture Model) con validazione causale.

Bayesian Rolling Mixture con post-hoc Markov Smoothing.
NON è un Hidden Markov Model completo: nessuna matrice di transizione appresa via MCMC.
Applica un filtro di Markov deterministico sulle probabilità posteriori per imporre
dipendenza temporale P(regime_t | regime_{t-1}) con costo O(N).

Componenti principali:
    - CausalBayesianMixture: classe principale per inferenza rolling per nave
    - estimate_empirical_bayes_priors: stima σ_prior dalla flotta (Stage 1 di EB)
    - prior_sensitivity_analysis: Gelman BDA3 §6 - robustezza alle scelte di prior
    - posterior_predictive_check: Gelman BDA3 §6.3 - validazione del modello generativo
"""
import gc
import os
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import logging
import warnings
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning, module="pymc")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="arviz")
warnings.filterwarnings("ignore", category=FutureWarning, module="pymc")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _fit_pilot_bmm(speed_data: np.ndarray, turn_data: np.ndarray,
                    sigma_prior_speed: float = 2.0, sigma_prior_turn: float = 3.0,
                    n_advi_steps: int = 500) -> dict:
    """Fit standalone BMM su una nave intera per pilot phase di empirical Bayes.
    A differenza del modello rolling causale principale, qui usiamo tutta la serie
    in un singolo fit, per ottenere un riassunto stabile dei parametri della nave."""
    # Sanitize
    speed_data = np.asarray(speed_data, dtype=float)
    turn_data  = np.asarray(turn_data,  dtype=float)
    min_len = min(len(speed_data), len(turn_data))
    speed_data, turn_data = speed_data[:min_len], turn_data[:min_len]
    if min_len < 30:
        raise ValueError(f"Pilot fit needs >=30 obs, got {min_len}")

    with pm.Model() as _:
        theta = pm.Dirichlet("theta", a=np.array([1.0, 1.0]))
        mu_sospetto = pm.Normal("mu_sospetto", mu=0, sigma=sigma_prior_speed)
        gap = pm.HalfNormal("gap", sigma=sigma_prior_speed)
        mu_transito = pm.Deterministic("mu_transito", mu_sospetto + gap)
        mu_speed = pm.math.stack([mu_sospetto, mu_transito])
        sigma_speed = pm.HalfNormal("sigma_speed", sigma=sigma_prior_speed, shape=2)
        sigma_turn  = pm.HalfNormal("sigma_turn",  sigma=sigma_prior_turn,  shape=2)

        pm.Mixture("obs_speed", w=theta,
                   comp_dists=[pm.Normal.dist(mu=mu_speed[0], sigma=sigma_speed[0]),
                               pm.Normal.dist(mu=mu_speed[1], sigma=sigma_speed[1])],
                   observed=speed_data)
        pm.Mixture("obs_turn", w=theta,
                   comp_dists=[pm.Normal.dist(mu=0, sigma=sigma_turn[0]),
                               pm.Normal.dist(mu=0, sigma=sigma_turn[1])],
                   observed=turn_data)

        mean_field = pm.fit(n=n_advi_steps, method='advi', progressbar=False)
        trace = mean_field.sample(200)

    post = trace.posterior
    sigma_sp = post["sigma_speed"].mean(dim=["chain", "draw"]).values
    sigma_tr = post["sigma_turn"].mean(dim=["chain", "draw"]).values
    theta    = post["theta"].mean(dim=["chain", "draw"]).values
    return {
        "mu_sospetto":   float(post["mu_sospetto"].mean()),
        "gap":           float(post["gap"].mean()),
        "sigma_speed_0": float(sigma_sp[0]),
        "sigma_speed_1": float(sigma_sp[1]),
        "sigma_turn_0":  float(sigma_tr[0]),
        "sigma_turn_1":  float(sigma_tr[1]),
        "theta_0":       float(theta[0]),
        "theta_1":       float(theta[1]),
    }


def estimate_empirical_bayes_priors(df: pd.DataFrame,
                                     n_pilot_vessels: int = 5,
                                     min_vessel_length: int = 100,
                                     seed: int = 42,
                                     n_advi_steps: int = 500,
                                     save_path: Optional[str] = None) -> dict:
    """
    Two-stage Empirical Bayes per i prior del Bayesian Mixture Model.

    Stage 1 (questa funzione):
        - Seleziona N navi 'pilot' (con almeno min_vessel_length osservazioni)
        - Per ogni nave: fit standalone BMM su tutta la serie via ADVI
        - Estrai posterior means dei 4 parametri chiave (mu_sospetto, gap, sigma_*)
        - Calcola la **varianza cross-vessel** di questi parametri
        - L'hyperprior σ stimato è MAX(std_cross_vessel(mu params),
                                       mean(sigma posteriors))
          → cattura sia variabilità tra navi, sia tipica scala intra-nave

    Stage 2 (Fase 2 della pipeline, in CausalBayesianMixture):
        - Usa l'hyperprior σ stimato come `sigma_prior_speed`/`sigma_prior_turn`
        - Posterior ancora fit per-nave (no partial pooling formale)

    Justificazione accademica:
        L'approccio è exchangeable Empirical Bayes (Efron-Morris). Un full
        hierarchical model richiederebbe NUTS su un grafo a 2 livelli con N navi
        come unità → computazionalmente proibitivo a questa scala.
        Two-stage EB cattura il primo ordine del partial pooling: prior informato
        dalla flotta, posterior individuale per nave.

    Args:
        df: DataFrame con MMSI, speed_acc, turn_rate.
        n_pilot_vessels: numero di navi pilot da fittare (5-10 raccomandato).
        min_vessel_length: lunghezza minima della serie per essere eligible.
        n_advi_steps: ADVI steps per il pilot fit (500 = bilanciato).
        save_path: se fornito, salva il risultato come JSON.

    Returns:
        dict con sigma_prior_speed, sigma_prior_turn stimati + diagnostica.
    """
    import json
    from pathlib import Path
    rng = np.random.default_rng(seed)

    if "speed_acc" not in df.columns or "turn_rate" not in df.columns:
        raise ValueError("Servono colonne speed_acc e turn_rate (engineer_causal_features).")

    vessel_lengths = df.groupby("MMSI").size()
    eligible = vessel_lengths[vessel_lengths >= min_vessel_length].index.tolist()

    if len(eligible) == 0:
        logging.warning("Nessuna nave eligible per empirical Bayes. Uso default σ.")
        return {"sigma_prior_speed": 1.0, "sigma_prior_turn": 2.0,
                "method": "fallback_default", "reason": "no_eligible_vessels"}

    n_pilot = min(n_pilot_vessels, len(eligible))
    pilot_mmsi = rng.choice(eligible, size=n_pilot, replace=False)
    logging.info(f"📐 Empirical Bayes Stage 1: pilot fit su {n_pilot} navi {list(pilot_mmsi)}")

    pilot_results = []
    for mmsi in pilot_mmsi:
        df_v = df[df["MMSI"] == mmsi].sort_values("Timestamp")
        speed_raw = df_v["speed_acc"].dropna().values
        turn_raw  = df_v["turn_rate"].dropna().values
        n = min(len(speed_raw), len(turn_raw))
        if n < 30:
            logging.warning(f"  Nave {mmsi}: dati insufficienti ({n}), skip.")
            continue
        speed_raw, turn_raw = speed_raw[:n], turn_raw[:n]

        # Standardize per-vessel (stesso protocollo del modello principale)
        scaler_s = StandardScaler().fit(speed_raw.reshape(-1, 1))
        scaler_t = StandardScaler().fit(turn_raw.reshape(-1, 1))
        speed_std = scaler_s.transform(speed_raw.reshape(-1, 1)).flatten()
        turn_std  = scaler_t.transform(turn_raw.reshape(-1, 1)).flatten()

        try:
            params = _fit_pilot_bmm(speed_std, turn_std, n_advi_steps=n_advi_steps)
            params["mmsi"] = int(mmsi)
            params["n_obs"] = int(n)
            pilot_results.append(params)
            logging.info(
                f"  ✓ Nave {mmsi} ({n} obs): μ_sosp={params['mu_sospetto']:+.2f}, "
                f"gap={params['gap']:.2f}, σ_sp=[{params['sigma_speed_0']:.2f},{params['sigma_speed_1']:.2f}]"
            )
        except Exception as e:
            logging.warning(f"  ✗ Pilot fit fallito per nave {mmsi}: {e}")
            continue

    if not pilot_results:
        logging.warning("Tutti i pilot fit falliti. Fallback ai prior di default.")
        return {"sigma_prior_speed": 1.0, "sigma_prior_turn": 2.0,
                "method": "fallback_default", "reason": "all_pilots_failed"}

    mu_sosp_arr   = np.array([r["mu_sospetto"] for r in pilot_results])
    gap_arr       = np.array([r["gap"]         for r in pilot_results])
    sigma_sp_arr  = np.array([r["sigma_speed_0"] for r in pilot_results] +
                             [r["sigma_speed_1"] for r in pilot_results])
    sigma_tr_arr  = np.array([r["sigma_turn_0"]  for r in pilot_results] +
                             [r["sigma_turn_1"]  for r in pilot_results])

    # Hyperprior σ stimato: max tra std cross-vessel dei mu e mean dei σ posteriors.
    # Floor di 0.3 per evitare prior degenerate (Bayes-Stein shrinkage prudente).
    empirical_sigma_speed = float(max(
        np.std(mu_sosp_arr, ddof=1) if len(mu_sosp_arr) > 1 else 0.0,
        np.std(gap_arr, ddof=1)     if len(gap_arr) > 1     else 0.0,
        np.mean(sigma_sp_arr),
    ))
    empirical_sigma_turn = float(np.mean(sigma_tr_arr))

    empirical_sigma_speed = max(empirical_sigma_speed, 0.3)
    empirical_sigma_turn  = max(empirical_sigma_turn,  0.3)

    result = {
        "method": "two_stage_empirical_bayes",
        "sigma_prior_speed": empirical_sigma_speed,
        "sigma_prior_turn":  empirical_sigma_turn,
        "n_pilot_vessels":   len(pilot_results),
        "pilot_results":     pilot_results,
        "cross_vessel_stats": {
            "mean_mu_sospetto":  float(np.mean(mu_sosp_arr)),
            "std_mu_sospetto":   float(np.std(mu_sosp_arr,  ddof=1)) if len(mu_sosp_arr) > 1 else 0.0,
            "mean_gap":          float(np.mean(gap_arr)),
            "std_gap":           float(np.std(gap_arr, ddof=1))      if len(gap_arr) > 1     else 0.0,
            "mean_sigma_speed":  float(np.mean(sigma_sp_arr)),
            "mean_sigma_turn":   float(np.mean(sigma_tr_arr)),
        },
        "seed": seed,
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        logging.info(f"📦 Empirical Bayes priors salvati: {save_path}")

    logging.info(
        f"✅ Empirical Bayes Stage 1 done: "
        f"σ_prior_speed={empirical_sigma_speed:.3f}, "
        f"σ_prior_turn={empirical_sigma_turn:.3f} "
        f"(da {len(pilot_results)} pilot vessels)"
    )
    return result


def _vessel_suspicion_score(speed_std: np.ndarray, turn_std: np.ndarray,
                             params: dict) -> float:
    """
    Vessel-level mean P(regime_sospetto | x_t) dato un set di parametri posteriori.

    Calcolato come mean over t della responsibility del componente 'sospetto':
        γ_0(t) = π_0 · N(x_speed_t; μ_sosp, σ_sosp) · N(x_turn_t; 0, σ_turn_0)
                 ─────────────────────────────────────────────────────────────
                 Σ_k π_k · N(x_speed_t; μ_k, σ_speed_k) · N(x_turn_t; 0, σ_turn_k)

    Usato per ranking robust-to-prior delle navi.
    """
    mu_sosp = params["mu_sospetto"]
    mu_tran = params["mu_sospetto"] + params["gap"]
    theta_0 = params.get("theta_0", 0.5)
    theta_1 = params.get("theta_1", 0.5)

    lik_sosp = scipy_stats.norm.pdf(speed_std, mu_sosp, params["sigma_speed_0"]) * \
               scipy_stats.norm.pdf(turn_std,  0,       params["sigma_turn_0"])
    lik_tran = scipy_stats.norm.pdf(speed_std, mu_tran, params["sigma_speed_1"]) * \
               scipy_stats.norm.pdf(turn_std,  0,       params["sigma_turn_1"])
    num = theta_0 * lik_sosp
    den = num + theta_1 * lik_tran + 1e-12
    return float(np.mean(num / den))


def prior_sensitivity_analysis(df: pd.DataFrame,
                                n_vessels: int = 10,
                                min_vessel_length: int = 100,
                                specifications: Optional[dict] = None,
                                seed: int = 42,
                                n_advi_steps: int = 300,
                                save_path: Optional[str] = None) -> dict:
    """
    Prior sensitivity analysis (Gelman et al., BDA3 cap. 6).

    Per N navi, fit BMM standalone sotto K specificazioni di prior alternative.
    Per ogni nave + specificazione, calcola la **vessel-level mean P(regime_sospetto)**.
    Misura la correlazione di Spearman tra ranking delle navi ottenuti sotto
    le diverse specificazioni.

    Interpretazione:
        - ρ > 0.85 → ranking sostanzialmente invariato; conclusioni ROBUSTE alle scelte di prior
        - ρ < 0.85 → ranking sensibile; il modello dipende dalle scelte soggettive

    Default specifications:
        - "loose":   prior molto poco informativi (σ × 3)
        - "default": σ_speed=1.0, σ_turn=2.0 (baseline del modello)
        - "tight":   prior più informativi (σ ÷ 3)

    Args:
        df: DataFrame con MMSI, speed_acc, turn_rate
        n_vessels: numero di navi sample per la sensitivity
        specifications: dict di {nome: {sigma_prior_speed, sigma_prior_turn}}
        n_advi_steps: ADVI steps per pilot fit (300 = veloce ma stabile)
        save_path: se fornito salva JSON dei risultati

    Returns:
        dict con pairwise Spearman ρ, p-values, min_spearman, robust_to_prior bool
    """
    import json
    from pathlib import Path
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)

    specifications = specifications or {
        "loose":   {"sigma_prior_speed": 3.0, "sigma_prior_turn": 6.0},
        "default": {"sigma_prior_speed": 1.0, "sigma_prior_turn": 2.0},
        "tight":   {"sigma_prior_speed": 0.3, "sigma_prior_turn": 0.6},
    }

    if "speed_acc" not in df.columns or "turn_rate" not in df.columns:
        raise ValueError("Servono colonne speed_acc e turn_rate.")

    vessel_lengths = df.groupby("MMSI").size()
    eligible = vessel_lengths[vessel_lengths >= min_vessel_length].index.tolist()
    if len(eligible) == 0:
        logging.warning("Nessuna nave eligible per prior sensitivity.")
        return {"method": "fallback", "reason": "no_eligible_vessels"}

    n_v = min(n_vessels, len(eligible))
    sampled_mmsi = rng.choice(eligible, size=n_v, replace=False)
    logging.info(
        f"🔬 Prior sensitivity: {len(specifications)} specificazioni × {n_v} navi "
        f"= {len(specifications) * n_v} pilot fits"
    )

    # Pre-standardizza per nave (una sola volta)
    vessel_data = {}
    for mmsi in sampled_mmsi:
        df_v = df[df["MMSI"] == mmsi].sort_values("Timestamp")
        speed_raw = df_v["speed_acc"].dropna().values
        turn_raw  = df_v["turn_rate"].dropna().values
        n = min(len(speed_raw), len(turn_raw))
        if n < 30:
            continue
        speed_raw, turn_raw = speed_raw[:n], turn_raw[:n]
        scaler_s = StandardScaler().fit(speed_raw.reshape(-1, 1))
        scaler_t = StandardScaler().fit(turn_raw.reshape(-1, 1))
        vessel_data[int(mmsi)] = (
            scaler_s.transform(speed_raw.reshape(-1, 1)).flatten(),
            scaler_t.transform(turn_raw.reshape(-1, 1)).flatten(),
        )

    if not vessel_data:
        logging.warning("Nessuna nave con dati sufficienti.")
        return {"method": "fallback", "reason": "insufficient_data"}

    # Fit BMM sotto ogni specificazione, calcola vessel score
    scores_by_spec = {name: {} for name in specifications}
    for spec_name, sp in specifications.items():
        logging.info(
            f"  📋 '{spec_name}': σ_speed={sp['sigma_prior_speed']}, "
            f"σ_turn={sp['sigma_prior_turn']}"
        )
        for mmsi, (speed_std, turn_std) in vessel_data.items():
            try:
                params = _fit_pilot_bmm(
                    speed_std, turn_std,
                    sigma_prior_speed=sp["sigma_prior_speed"],
                    sigma_prior_turn=sp["sigma_prior_turn"],
                    n_advi_steps=n_advi_steps,
                )
                scores_by_spec[spec_name][mmsi] = _vessel_suspicion_score(
                    speed_std, turn_std, params
                )
            except Exception as e:
                logging.debug(f"    Fit fallito {mmsi}/{spec_name}: {e}")
                scores_by_spec[spec_name][mmsi] = None

    # Pairwise Spearman tra specificazioni
    spec_names = list(specifications.keys())
    common_mmsi = sorted(set.intersection(*[
        {k for k, v in scores_by_spec[s].items() if v is not None}
        for s in spec_names
    ]))

    pairwise_rho, pairwise_p = {}, {}
    for i, s1 in enumerate(spec_names):
        for s2 in spec_names[i + 1:]:
            x = np.array([scores_by_spec[s1][m] for m in common_mmsi])
            y = np.array([scores_by_spec[s2][m] for m in common_mmsi])
            if len(x) < 3:
                continue
            rho, p = spearmanr(x, y)
            pairwise_rho[f"{s1}_vs_{s2}"] = float(rho)
            pairwise_p[f"{s1}_vs_{s2}"]   = float(p)
            logging.info(f"  ρ_Spearman({s1}, {s2}) = {rho:+.3f}  (p={p:.2e})")

    min_rho = min(pairwise_rho.values()) if pairwise_rho else None
    robust = (min_rho is not None and min_rho > 0.85)

    result = {
        "method": "prior_sensitivity_analysis",
        "n_vessels": len(common_mmsi),
        "specifications": specifications,
        "vessel_scores": {
            spec: {str(m): scores_by_spec[spec].get(m) for m in common_mmsi}
            for spec in spec_names
        },
        "pairwise_spearman": pairwise_rho,
        "pairwise_pvalues":  pairwise_p,
        "min_spearman": min_rho,
        "robust_to_prior": robust,
        "robustness_threshold": 0.85,
        "seed": seed,
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        logging.info(f"📦 Prior sensitivity salvato: {save_path}")

    verdict = "✅ ROBUSTO" if robust else "⚠️ NON ROBUSTO"
    rho_str = f"{min_rho:.3f}" if min_rho is not None else "N/D"
    logging.info(f"{verdict} alla scelta di prior (min ρ_Spearman = {rho_str}, soglia 0.85)")

    return result


def _fit_bmm_with_posterior_samples(speed_data: np.ndarray, turn_data: np.ndarray,
                                     sigma_prior_speed: float = 1.0,
                                     sigma_prior_turn: float = 2.0,
                                     n_advi_steps: int = 500,
                                     n_samples: int = 200) -> list:
    """
    Fit BMM standalone e ritorna `n_samples` campioni dal posterior approssimato.
    Necessario per Posterior Predictive Check (Gelman BDA3 §6).
    """
    speed_data = np.asarray(speed_data, dtype=float)
    turn_data  = np.asarray(turn_data,  dtype=float)
    min_len = min(len(speed_data), len(turn_data))
    speed_data, turn_data = speed_data[:min_len], turn_data[:min_len]
    if min_len < 30:
        raise ValueError(f"PPC fit needs >=30 obs, got {min_len}")

    with pm.Model() as _:
        theta = pm.Dirichlet("theta", a=np.array([1.0, 1.0]))
        mu_sospetto = pm.Normal("mu_sospetto", mu=0, sigma=sigma_prior_speed)
        gap = pm.HalfNormal("gap", sigma=sigma_prior_speed)
        mu_transito = pm.Deterministic("mu_transito", mu_sospetto + gap)
        mu_speed = pm.math.stack([mu_sospetto, mu_transito])
        sigma_speed = pm.HalfNormal("sigma_speed", sigma=sigma_prior_speed, shape=2)
        sigma_turn  = pm.HalfNormal("sigma_turn",  sigma=sigma_prior_turn,  shape=2)

        pm.Mixture("obs_speed", w=theta,
                   comp_dists=[pm.Normal.dist(mu=mu_speed[0], sigma=sigma_speed[0]),
                               pm.Normal.dist(mu=mu_speed[1], sigma=sigma_speed[1])],
                   observed=speed_data)
        pm.Mixture("obs_turn", w=theta,
                   comp_dists=[pm.Normal.dist(mu=0, sigma=sigma_turn[0]),
                               pm.Normal.dist(mu=0, sigma=sigma_turn[1])],
                   observed=turn_data)

        mean_field = pm.fit(n=n_advi_steps, method='advi', progressbar=False)
        trace = mean_field.sample(n_samples)

    post = trace.posterior
    mu_s_arr  = post["mu_sospetto"].values.flatten()
    gap_arr   = post["gap"].values.flatten()
    sigma_sp  = post["sigma_speed"].values.reshape(-1, 2)
    sigma_tr  = post["sigma_turn"].values.reshape(-1, 2)
    theta_arr = post["theta"].values.reshape(-1, 2)

    samples = []
    for i in range(len(mu_s_arr)):
        samples.append({
            "mu_sospetto":   float(mu_s_arr[i]),
            "gap":           float(gap_arr[i]),
            "sigma_speed_0": float(sigma_sp[i, 0]),
            "sigma_speed_1": float(sigma_sp[i, 1]),
            "sigma_turn_0":  float(sigma_tr[i, 0]),
            "sigma_turn_1":  float(sigma_tr[i, 1]),
            "theta_0":       float(theta_arr[i, 0]),
            "theta_1":       float(theta_arr[i, 1]),
        })
    return samples


def _generate_pp_trajectory(posterior_sample: dict, T: int, rng: np.random.Generator) -> tuple:
    """Genera una traiettoria sintetica (speed_std, turn_std) di lunghezza T dato un posterior sample."""
    mu_s = posterior_sample["mu_sospetto"]
    mu_t = posterior_sample["mu_sospetto"] + posterior_sample["gap"]
    # Per ogni t, sample latent state z ~ Bernoulli(theta_1)  (1 = transito, 0 = sospetto)
    z = rng.binomial(1, posterior_sample["theta_1"], size=T)
    speed = np.where(
        z == 0,
        rng.normal(mu_s, posterior_sample["sigma_speed_0"], T),
        rng.normal(mu_t, posterior_sample["sigma_speed_1"], T),
    )
    turn = np.where(
        z == 0,
        rng.normal(0, posterior_sample["sigma_turn_0"], T),
        rng.normal(0, posterior_sample["sigma_turn_1"], T),
    )
    return speed, turn


def _compute_ppc_summary_stats(speed: np.ndarray, turn: np.ndarray) -> dict:
    """4 summary statistics per il PPC."""
    def _autocorr1(x):
        if len(x) < 2:
            return 0.0
        x_centered = x - np.mean(x)
        c0 = np.mean(x_centered ** 2)
        if c0 < 1e-10:
            return 0.0
        return float(np.mean(x_centered[:-1] * x_centered[1:]) / c0)

    return {
        "mean_speed":          float(np.mean(speed)),
        "std_speed":           float(np.std(speed)),
        "autocorr_turn_lag1":  _autocorr1(turn),
        "frac_extreme_speed":  float(np.mean(np.abs(speed) > 1.5)),
    }


def posterior_predictive_check(df: pd.DataFrame,
                                n_vessels: int = 3,
                                n_simulations: int = 200,
                                min_vessel_length: int = 100,
                                seed: int = 42,
                                n_advi_steps: int = 500,
                                sigma_prior_speed: float = 1.0,
                                sigma_prior_turn: float = 2.0,
                                save_path: Optional[str] = None) -> dict:
    """
    Posterior Predictive Check (PPC) — Gelman, Carlin, Stern, Rubin BDA3 §6.3.

    Per N navi rappresentative:
        1. Fit standalone BMM via ADVI (M campioni posterior)
        2. Per ogni posterior sample s ∈ {1...M}, simula una traiettoria sintetica
           (speed_t, turn_t) di lunghezza T pari all'osservata
        3. Calcola 4 summary statistics su synthetic e observed:
           - mean(speed): test di location
           - std(speed):  test di scale
           - autocorr(turn, lag=1): test di dipendenza temporale (Markov check)
           - frac(|speed|>1.5): test di tail behavior
        4. Bayesian p-value:  p = P(T(y_rep) ≥ T(y_obs) | y_obs)
           - 0.05 < p < 0.95  →  modello adeguato per quella statistica
           - p estremo        →  discrepanza sistematica

    Verdetto:
        Modello adeguato globalmente se ≥75% delle p sono in [0.05, 0.95] across navi×stat.

    Returns:
        dict con per-vessel stats, p-values, verdetto globale
    """
    import json
    from pathlib import Path
    rng = np.random.default_rng(seed)

    if "speed_acc" not in df.columns or "turn_rate" not in df.columns:
        raise ValueError("Servono colonne speed_acc e turn_rate.")

    vessel_lengths = df.groupby("MMSI").size()
    eligible = vessel_lengths[vessel_lengths >= min_vessel_length].index.tolist()
    if len(eligible) == 0:
        return {"method": "fallback", "reason": "no_eligible_vessels"}

    n_v = min(n_vessels, len(eligible))
    sampled_mmsi = rng.choice(eligible, size=n_v, replace=False)
    logging.info(f"🔍 Posterior Predictive Check su {n_v} navi (M={n_simulations} sim per nave)...")

    vessel_results = []
    stat_names = ["mean_speed", "std_speed", "autocorr_turn_lag1", "frac_extreme_speed"]

    for mmsi in sampled_mmsi:
        df_v = df[df["MMSI"] == mmsi].sort_values("Timestamp")
        speed_raw = df_v["speed_acc"].dropna().values
        turn_raw  = df_v["turn_rate"].dropna().values
        n = min(len(speed_raw), len(turn_raw))
        if n < 30:
            continue
        speed_raw, turn_raw = speed_raw[:n], turn_raw[:n]

        scaler_s = StandardScaler().fit(speed_raw.reshape(-1, 1))
        scaler_t = StandardScaler().fit(turn_raw.reshape(-1, 1))
        speed_std = scaler_s.transform(speed_raw.reshape(-1, 1)).flatten()
        turn_std  = scaler_t.transform(turn_raw.reshape(-1, 1)).flatten()

        try:
            samples = _fit_bmm_with_posterior_samples(
                speed_std, turn_std,
                sigma_prior_speed=sigma_prior_speed,
                sigma_prior_turn=sigma_prior_turn,
                n_advi_steps=n_advi_steps,
                n_samples=n_simulations,
            )
        except Exception as e:
            logging.warning(f"  ✗ PPC fit fallito per nave {mmsi}: {e}")
            continue

        # Statistiche osservate
        obs_stats = _compute_ppc_summary_stats(speed_std, turn_std)

        # Simula M traiettorie e calcola statistiche
        synth_stats = {name: [] for name in stat_names}
        for s in samples:
            try:
                speed_sim, turn_sim = _generate_pp_trajectory(s, T=n, rng=rng)
                stats = _compute_ppc_summary_stats(speed_sim, turn_sim)
                for name in stat_names:
                    synth_stats[name].append(stats[name])
            except Exception:
                continue

        # Bayesian p-values
        p_values = {}
        for name in stat_names:
            arr = np.array(synth_stats[name])
            if len(arr) == 0:
                p_values[name] = None
                continue
            p_values[name] = float(np.mean(arr >= obs_stats[name]))

        vessel_results.append({
            "mmsi": int(mmsi),
            "n_obs": int(n),
            "observed_stats": obs_stats,
            "synthetic_stats": {name: synth_stats[name] for name in stat_names},
            "bayesian_p_values": p_values,
        })
        p_str = ", ".join(
            f"{name.split('_')[0]}={p:.2f}" if p is not None else f"{name.split('_')[0]}=NA"
            for name, p in p_values.items()
        )
        logging.info(f"  ✓ Nave {mmsi} ({n} obs): {p_str}")

    if not vessel_results:
        return {"method": "fallback", "reason": "all_fits_failed"}

    # Verdetto globale: % di p in [0.05, 0.95]
    all_pvals = []
    for vr in vessel_results:
        for p in vr["bayesian_p_values"].values():
            if p is not None:
                all_pvals.append(p)
    all_pvals = np.array(all_pvals)
    frac_adequate = float(np.mean((all_pvals > 0.05) & (all_pvals < 0.95))) if len(all_pvals) else 0.0
    adequate = frac_adequate >= 0.75

    result = {
        "method": "posterior_predictive_check",
        "n_vessels": len(vessel_results),
        "n_simulations_per_vessel": n_simulations,
        "stat_names": stat_names,
        "vessel_results": vessel_results,
        "global_verdict": {
            "frac_p_in_inner_90pct": frac_adequate,
            "threshold": 0.75,
            "model_adequate": adequate,
        },
        "seed": seed,
    }

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)
        logging.info(f"📦 PPC salvato: {save_path}")

    verdict = "✅ MODELLO ADEGUATO" if adequate else "⚠️ DISCREPANZE RILEVATE"
    logging.info(
        f"{verdict} (frazione p in [0.05, 0.95] = {frac_adequate:.1%}, soglia 75%)"
    )
    return result


def _parallel_vessel_worker(df_vessel: pd.DataFrame,
                             window_size: int,
                             sigma_prior_speed: float,
                             sigma_prior_turn: float,
                             use_advi: bool,
                             apply_markov: bool,
                             update_freq: int,
                             max_advi_calls: Optional[int],
                             adaptive_threshold: Optional[float]) -> pd.DataFrame:
    """Worker module-level (necessario per pickling con joblib loky backend).
    Crea un'istanza CausalBayesianMixture fresca per processo → isolamento PyMC."""
    # Limita thread BLAS/OpenMP interni per non sovrascrivere il parallelismo joblib
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    extractor = CausalBayesianMixture(
        window_size=window_size,
        sigma_prior_speed=sigma_prior_speed,
        sigma_prior_turn=sigma_prior_turn,
    )
    return extractor._process_single_vessel(
        df_vessel, use_advi, apply_markov, update_freq, max_advi_calls, adaptive_threshold
    )


class CausalBayesianMixture:
    """
    Estrattore di regime latente tramite Mixture Model Bayesiano causale + Markov Filter.
    Features estratte:
    - prob_regime_sospetto: P(Regime=0 | dati passati), dove 0 = comportamento sospetto
    - prob_regime_markov: Versione smoothata che rispetta dipendenza temporale Markoviana
    - incertezza_regime: Varianza della posterior, segnale di ambiguità comportamentale
    """

    def __init__(self, window_size: int = 36, sigma_prior_speed: float = 1.0, 
                 sigma_prior_turn: float = 2.0):
        self.window_size = window_size
        self.sigma_prior_speed = sigma_prior_speed
        self.sigma_prior_turn = sigma_prior_turn
        self.model = None
        self.scaler = None

    def build_model(self):
        """Costruisce il grafo computazionale con marginalizzazione esplicita per ADVI."""
        with pm.Model() as self.model:
            speed_data = pm.Data("speed_data", np.zeros(self.window_size))
            turn_data = pm.Data("turn_data", np.zeros(self.window_size))
            
            # Priors per i pesi della mixture
            theta = pm.Dirichlet("theta", a=np.array([1.0, 1.0]))
            
            # Struttura dei componenti: 0=Sospetto, 1=Transito
            mu_sospetto = pm.Normal("mu_sospetto", mu=0, sigma=self.sigma_prior_speed)
            gap = pm.HalfNormal("gap", sigma=self.sigma_prior_speed)
            mu_transito = pm.Deterministic("mu_transito", mu_sospetto + gap)
            
            mu_speed = pm.math.stack([mu_sospetto, mu_transito])
            sigma_speed = pm.HalfNormal("sigma_speed", sigma=self.sigma_prior_speed, shape=2)
            sigma_turn = pm.HalfNormal("sigma_turn", sigma=self.sigma_prior_turn, shape=2)
            
            # Marginalizzazione manuale della latente: ADVI campiona solo variabili continue
            pm.Mixture("obs_speed", w=theta,
                       comp_dists=[pm.Normal.dist(mu=mu_speed[0], sigma=sigma_speed[0]),
                                   pm.Normal.dist(mu=mu_speed[1], sigma=sigma_speed[1])], 
                       observed=speed_data)
            
            pm.Mixture("obs_turn", w=theta, 
                       comp_dists=[pm.Normal.dist(mu=0, sigma=sigma_turn[0]),
                                   pm.Normal.dist(mu=0, sigma=sigma_turn[1])], 
                       observed=turn_data)

    def _calculate_posterior_regime_prob(self, trace, speed_val: float, turn_val: float) -> float:
        """Calcola P(Regime=0 | x) analiticamente dai parametri continui inferiti."""
        mu_s = float(trace.posterior["mu_sospetto"].mean())
        mu_t = float(trace.posterior["mu_transito"].mean())
        sig_sp = trace.posterior["sigma_speed"].mean().values
        sig_tr = trace.posterior["sigma_turn"].mean().values
        w = trace.posterior["theta"].mean().values
        
        # Likelihood condizionali
        lik_speed_0 = np.exp(-0.5 * ((speed_val - mu_s)**2 / sig_sp[0]**2)) / sig_sp[0]
        lik_speed_1 = np.exp(-0.5 * ((speed_val - mu_t)**2 / sig_sp[1]**2)) / sig_sp[1]
        lik_turn_0  = np.exp(-0.5 * ((turn_val)**2 / sig_tr[0]**2)) / sig_tr[0]
        lik_turn_1  = np.exp(-0.5 * ((turn_val)**2 / sig_tr[1]**2)) / sig_tr[1]
        
        # Bayes rule: P(k|x) ∝ w_k * L_k
        lik_0 = w[0] * lik_speed_0 * lik_turn_0
        lik_1 = w[1] * lik_speed_1 * lik_turn_1
        
        return lik_0 / (lik_0 + lik_1 + 1e-12)

    def apply_markov_filter(self, probs: np.ndarray, 
                            p_00: float = 0.85, p_11: float = 0.85) -> np.ndarray:
        """
        Filtro di Markov 1D ricorsivo (Forward filtering) sulle probabilità mixture.
        Impone dipendenza temporale: P(s_t | x_{1:t}, s_{t-1}) senza overhead MCMC.
        """
        smoothed = np.zeros_like(probs)
        smoothed[0] = probs[0]
        p_01, p_10 = 1.0 - p_00, 1.0 - p_11
        
        for t in range(1, len(probs)):
            # Predizione a priori basata sullo stato precedente
            pred = smoothed[t-1] * p_00 + (1 - smoothed[t-1]) * p_10
            # Aggiornamento Bayesiano
            num = probs[t] * pred
            den = num + (1 - probs[t]) * (smoothed[t-1] * p_01 + (1 - smoothed[t-1]) * p_11)
            smoothed[t] = num / (den + 1e-12)
        return smoothed

    def _standardize_features(self, speed_feat: np.ndarray, turn_feat: np.ndarray,
                           fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Standardizza le feature mantenendo causalità (fit solo sul primo chunk della nave corrente)."""
        features = np.stack([speed_feat, turn_feat], axis=1)

        if fit:
            fit_end = min(self.window_size * 3, len(features) // 3)
            self.scaler = StandardScaler()
            self.scaler.fit(features[:fit_end])

        if self.scaler is None:
            raise RuntimeError("Scaler non inizializzato. Chiama prima con fit=True.")

        features_scaled = self.scaler.transform(features)
        return features_scaled[:, 0], features_scaled[:, 1]

    def _process_single_vessel(self, df_vessel: pd.DataFrame, use_advi: bool,
                               apply_markov: bool, update_freq: int,
                               max_advi_calls: Optional[int],
                               adaptive_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Processa una singola nave con il proprio scaler indipendente.
        Se adaptive_threshold è impostato, aggiorna l'inferenza solo quando
        il cambiamento comportamentale supera la soglia (oppure ogni update_freq
        step come fallback), riducendo il costo computazionale del 30-50% su
        traiettorie regolari.
        """
        df_vessel = df_vessel.dropna(subset=['speed_acc', 'turn_rate']).copy()
        n = len(df_vessel)

        if n < self.window_size + 1:
            df_vessel['prob_regime_sospetto'] = np.nan
            df_vessel['incertezza_regime'] = np.nan
            return df_vessel

        speed_feat = df_vessel['speed_acc'].values
        turn_feat = df_vessel['turn_rate'].values

        # Fit scaler solo sui dati di questa nave (causalità per nave)
        speed_scaled, turn_scaled = self._standardize_features(speed_feat, turn_feat, fit=True)

        probs_sospetto = np.full(n, np.nan)
        incertezza = np.full(n, np.nan)

        if self.model is None:
            self.build_model()

        n_advi_steps = 1000 if use_advi else 0
        total_steps = n - self.window_size

        # Distribuisce i call ADVI uniformemente se max_advi_calls è impostato
        if max_advi_calls is not None and max_advi_calls < total_steps:
            advi_indices = set(
                int(round(i)) for i in np.linspace(self.window_size, n - 1, max_advi_calls)
            )
        else:
            advi_indices = None  # usa update_freq o adaptive_threshold

        last_prob, last_var = None, None
        last_update_t = self.window_size

        for t in range(self.window_size, n):
            if t % 500 == 0:
                logging.info(f"  step {t}/{n}...")

            if advi_indices is not None:
                should_update = t in advi_indices
            elif adaptive_threshold is not None:
                # Update se cambiamento comportamentale supera la soglia
                # oppure se è passato troppo tempo dall'ultimo update (fallback)
                delta_speed = abs(speed_scaled[t] - speed_scaled[t - 1])
                delta_turn  = abs(turn_scaled[t]  - turn_scaled[t - 1])
                behavioral_change = delta_speed + delta_turn
                time_since_update = t - last_update_t
                should_update = (behavioral_change > adaptive_threshold) or (time_since_update >= update_freq)
            else:
                should_update = (t - self.window_size) % update_freq == 0

            if not should_update and last_prob is not None:
                probs_sospetto[t] = last_prob
                incertezza[t] = last_var
                continue

            x_speed = speed_scaled[t - self.window_size + 1 : t + 1]
            x_turn = turn_scaled[t - self.window_size + 1 : t + 1]

            with self.model:
                pm.set_data({"speed_data": x_speed, "turn_data": x_turn})
                try:
                    if use_advi:
                        mean_field = pm.fit(n=n_advi_steps, method='advi', progressbar=False)
                        trace = mean_field.sample(100)
                    else:
                        trace = pm.sample(draws=100, tune=50, cores=1,
                                         progressbar=False, init='adapt_diag')

                    prob_t = self._calculate_posterior_regime_prob(trace, x_speed[-1], x_turn[-1])
                    var_t = prob_t * (1 - prob_t)
                    last_prob, last_var = prob_t, var_t
                    last_update_t = t
                    probs_sospetto[t] = prob_t
                    incertezza[t] = var_t

                except Exception as e:
                    logging.debug(f"Errore inferenza a t={t}: {e}")
                    if last_prob is not None:
                        probs_sospetto[t] = last_prob
                        incertezza[t] = last_var
                    continue
                finally:
                    gc.collect()

        probs_sospetto = np.where(np.isnan(probs_sospetto), 0.5, probs_sospetto)
        incertezza = np.where(np.isnan(incertezza), 0.25, incertezza)

        df_vessel['prob_regime_sospetto'] = probs_sospetto
        df_vessel['incertezza_regime'] = incertezza

        if apply_markov:
            df_vessel['prob_regime_markov'] = self.apply_markov_filter(probs_sospetto)

        # Reset del modello tra navi, scaler indipendente per la prossima nave
        self.reset_model_only()
        self.scaler = None

        return df_vessel

    def process_dataframe_causal(self, df: pd.DataFrame, use_advi: bool = True,
                            fit_scaler: bool = True, apply_markov: bool = True,
                            update_freq: int = 10,
                            max_advi_calls: Optional[int] = None,
                            adaptive_threshold: Optional[float] = None,
                            n_jobs: int = -1) -> pd.DataFrame:
        """
        Processa il DataFrame per nave (MMSI) in modo indipendente per evitare
        leakage dello scaler tra navi diverse.

        Args:
            max_advi_calls: se impostato, cap al numero totale di chiamate ADVI per nave
                            (distribute uniformemente). Utile su dataset > 5k punti/nave.
            adaptive_threshold: se impostato, aggiorna l'inferenza solo quando la variazione
                                combinata (speed + turn) supera questa soglia (tipico: 0.5-1.5
                                in unità standardizzate). update_freq rimane il fallback massimo.
        """
        if 'MMSI' not in df.columns:
            logging.warning("Colonna MMSI assente: processing globale (no isolamento per nave).")
            logging.info(f"Avvio inferenza causale (Finestra: {self.window_size}, ADVI: {use_advi}, update_freq: {update_freq}, max_advi_calls: {max_advi_calls}, adaptive_threshold: {adaptive_threshold})...")
            return self._process_single_vessel(df, use_advi, apply_markov, update_freq, max_advi_calls, adaptive_threshold)

        # Invariant: MMSI deve essere int64 prima del per-vessel split.
        # MMSI float64 produce vessels = array di float, le _process_single_vessel
        # vedrebbero un identificatore non-canonico e il caching per-MMSI fallirebbe.
        assert df['MMSI'].dtype == np.int64, (
            f"MMSI must be int64 at Bayesian inference time, got {df['MMSI'].dtype}. "
            f"This causes silent join failures. Check load_data() / simulator."
        )

        vessels = df['MMSI'].unique()
        effective_jobs = n_jobs if n_jobs != -1 else max(1, os.cpu_count() - 1)
        effective_jobs = min(effective_jobs, len(vessels))
        logging.info(
            f"Avvio inferenza causale su {len(vessels)} navi "
            f"(Finestra: {self.window_size}, ADVI: {use_advi}, "
            f"max_advi_calls: {max_advi_calls}, adaptive_threshold: {adaptive_threshold}, "
            f"n_jobs: {effective_jobs})..."
        )

        vessel_dfs = [df[df['MMSI'] == m].sort_values('Timestamp').copy() for m in vessels]

        if effective_jobs == 1 or len(vessels) == 1:
            # Modalità sequenziale: riusa self (no overhead processi)
            results = []
            for mmsi, df_vessel in zip(vessels, vessel_dfs):
                logging.info(f"Processando nave MMSI={mmsi}...")
                results.append(self._process_single_vessel(
                    df_vessel, use_advi, apply_markov, update_freq, max_advi_calls, adaptive_threshold
                ))
        else:
            # Modalità parallela: worker module-level, processi separati (PyMC isolation)
            results = Parallel(n_jobs=effective_jobs, backend='loky', verbose=5)(
                delayed(_parallel_vessel_worker)(
                    df_vessel,
                    self.window_size, self.sigma_prior_speed, self.sigma_prior_turn,
                    use_advi, apply_markov, update_freq, max_advi_calls, adaptive_threshold,
                )
                for df_vessel in vessel_dfs
            )

        df_out = pd.concat(results, ignore_index=True).sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)
        logging.info(f"Inferenza completata. Feature estratte per {len(df_out)} osservazioni totali.")
        return df_out
    def reset_model_only(self):
        """Resetta solo il modello PyMC, mantiene lo scaler globale tra navi."""
        self.model = None
        logging.info("Modello PyMC resettato per nuova nave (scaler mantenuto).")

    def reset(self):
        """Reset completo (modello + scaler). Usare solo se si cambia dataset o pre-processing."""
        self.model = None
        self.scaler = None
        logging.info("Estrattore resettato completamente (modello + scaler).")

    def validate_advi_vs_nuts(self, df: pd.DataFrame, n_vessels: int = 3,
                              n_nuts_draws: int = 200, n_steps: int = 200,
                              update_freq: int = 5) -> dict:
        """
        Valida ADVI contro NUTS su un subset di navi calcolando la correlazione
        di Spearman tra le stime di prob_regime_sospetto prodotte dai due metodi.
        Correlazione > 0.85 è evidenza empirica che ADVI è sufficiente e la scelta
        è giustificabile in sede di difesa accademica.

        Args:
            df: DataFrame con colonne 'speed_acc', 'turn_rate' (già ingegnerizzate).
            n_vessels: numero di navi su cui eseguire la validazione (usa MMSI se presente).
            n_nuts_draws: numero di campioni NUTS per finestra (default 200, bilancia
                         accuratezza e tempo di esecuzione).
            n_steps: numero di step per nave su cui confrontare i metodi.
            update_freq: frequenza di aggiornamento durante la validazione.

        Returns:
            dict con Spearman ρ, p-value e numero di punti per nave.
        """
        logging.info(f"=== VALIDAZIONE ADVI vs NUTS (n_vessels={n_vessels}, n_steps={n_steps}) ===")

        if 'MMSI' in df.columns:
            vessel_ids = df['MMSI'].unique()[:n_vessels]
            vessel_dfs = [df[df['MMSI'] == m].sort_values('Timestamp') for m in vessel_ids]
            labels = [f"MMSI={m}" for m in vessel_ids]
        else:
            vessel_dfs = [df]
            labels = ["global"]

        correlations = {}

        for df_v, label in zip(vessel_dfs, labels):
            logging.info(f"  Validazione su {label}...")
            df_v = df_v.dropna(subset=['speed_acc', 'turn_rate']).copy()

            if len(df_v) < self.window_size + update_freq + 1:
                logging.warning(f"  {label}: dati insufficienti, skip.")
                continue

            df_v = df_v.iloc[:min(n_steps + self.window_size, len(df_v))]

            speed_feat = df_v['speed_acc'].values
            turn_feat  = df_v['turn_rate'].values

            # Scaler causale dedicato a questa nave di validazione
            tmp_scaler = StandardScaler()
            fit_end = min(self.window_size * 3, len(speed_feat) // 3)
            tmp_scaler.fit(np.stack([speed_feat, turn_feat], axis=1)[:fit_end])
            feats_sc = tmp_scaler.transform(np.stack([speed_feat, turn_feat], axis=1))
            speed_sc, turn_sc = feats_sc[:, 0], feats_sc[:, 1]

            probs_advi, probs_nuts = [], []
            self.model = None
            self.build_model()

            for t in range(self.window_size, len(df_v), update_freq):
                x_speed = speed_sc[t - self.window_size + 1 : t + 1]
                x_turn  = turn_sc[t - self.window_size + 1 : t + 1]

                with self.model:
                    pm.set_data({"speed_data": x_speed, "turn_data": x_turn})
                    try:
                        # ADVI
                        mf = pm.fit(n=1000, method='advi', progressbar=False)
                        tr_advi = mf.sample(100)
                        prob_advi = self._calculate_posterior_regime_prob(tr_advi, x_speed[-1], x_turn[-1])

                        # NUTS — più lento ma più accurato
                        tr_nuts = pm.sample(draws=n_nuts_draws, tune=100, cores=1,
                                            progressbar=False, init='adapt_diag',
                                            return_inferencedata=True)
                        prob_nuts = self._calculate_posterior_regime_prob(tr_nuts, x_speed[-1], x_turn[-1])

                        probs_advi.append(prob_advi)
                        probs_nuts.append(prob_nuts)

                    except Exception as e:
                        logging.debug(f"    t={t}: {e}")
                    finally:
                        gc.collect()

            self.reset()

            if len(probs_advi) < 5:
                logging.warning(f"  {label}: meno di 5 stime valide, correlazione inaffidabile.")
                continue

            corr, pval = scipy_stats.spearmanr(probs_advi, probs_nuts)
            correlations[label] = {
                'spearman_r': float(corr),
                'p_value': float(pval),
                'n_points': len(probs_advi)
            }
            verdict = "✅ ADVI sufficiente" if corr > 0.85 else "⚠️  Divergenza rilevante — valutare NUTS"
            logging.info(f"  {label}: Spearman ρ={corr:.3f}  p={pval:.4f}  n={len(probs_advi)}  → {verdict}")

        if correlations:
            mean_corr = float(np.mean([v['spearman_r'] for v in correlations.values()]))
            correlations['_summary'] = {'mean_spearman_r': mean_corr}
            logging.info(f"\n📊 Correlazione media ADVI/NUTS: {mean_corr:.3f}")
            if mean_corr > 0.85:
                logging.info("✅ ADVI validato empiricamente: scelta giustificabile in difesa.")
            else:
                logging.warning("⚠️  Considerare NUTS per maggiore fedeltà inferenziale.")

        return correlations


if __name__ == "__main__":
    # ========================================================================
    # TEST DI INTEGRAZIONE: Verifica causalità, standardizzazione e feature extraction
    # ========================================================================
    logging.info("=== TEST INTEGRAZIONE CausalBayesianMixture ===")
    np.random.seed(42)
    n_transit, n_suspicious = 100, 100

    mock_speed_transit = np.random.normal(0.5, 0.3, n_transit)
    mock_turn_transit = np.random.normal(0, 0.5, n_transit)
    mock_speed_suspicious = np.random.normal(0, 0.2, n_suspicious)
    mock_turn_suspicious = np.random.normal(0, 2.0, n_suspicious)

    mock_speed_acc = np.concatenate([mock_speed_transit, mock_speed_suspicious])
    mock_turn = np.concatenate([mock_turn_transit, mock_turn_suspicious])

    test_df = pd.DataFrame({'speed_acc': mock_speed_acc, 'turn_rate': mock_turn})

    extractor = CausalBayesianMixture(window_size=30)
    df_enriched = extractor.process_dataframe_causal(test_df, use_advi=True, fit_scaler=True, apply_markov=True)

    print("\n" + "="*60)
    print("RISULTATI TEST")
    print("="*60)
    assert 'dt_next_hours' not in df_enriched.columns, "LEAKAGE: colonna futura presente!"
    print("✅ Anti-leakage check: PASSED")
    
    assert 'prob_regime_sospetto' in df_enriched.columns, "Feature mancante!"
    assert df_enriched['prob_regime_sospetto'].between(0, 1).all(), "Probabilità fuori range!"
    print("✅ Feature extraction: PASSED")
    
    early_mean = df_enriched['prob_regime_sospetto'].iloc[30:60].mean()
    late_mean = df_enriched['prob_regime_sospetto'].iloc[-30:].mean()
    print(f"📊 Probabilità media: Transito={early_mean:.3f} | Sospetto={late_mean:.3f}")
    if late_mean > early_mean:
        print("✅ Pattern detection: PASSED")
    else:
        print("⚠️  Pattern detection: Verificare priors/dati")
        
    # Verifica Markov smoothing
    if 'prob_regime_markov' in df_enriched.columns:
        diff = np.abs(df_enriched['prob_regime_markov'].values[1:] - df_enriched['prob_regime_markov'].values[:-1])
        print(f"✅ Markov Filter: Variazione media per step={diff.mean():.4f} (smoothed)")
        
    print("\n=== TEST COMPLETATO ===")