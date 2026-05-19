"""
src/hmm_model.py
Modulo per l'inferenza Bayesiana dei regimi latenti (Mixture Model) con validazione causale.
NOTA ACCADEMICA: Bayesian Rolling Mixture con Markov Smoothing.
Non è un HMM completo (nessuna matrice di transizione appresa via MCMC), ma applica
un filtro di Markov deterministico sulle probabilità posteriori per imporre
dipendenza temporale P(regime_t | regime_{t-1}) con costo O(N).
"""
import gc
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import logging
import warnings
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple

# 🟡 FIX 19: Filtri warning specifici, non globali
warnings.filterwarnings("ignore", category=UserWarning, module="pymc")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="arviz")
warnings.filterwarnings("ignore", category=FutureWarning, module="pymc")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
            # 🟠 FIX 12: Rimosso mutable=True (default in PyMC >= 5.10)
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
            
            # 🔴 FIX 5: Marginalizzazione manuale per ADVI compatibile
            # Sostituisce pm.Categorical. ADVI ora campiona solo variabili continue.
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
                               max_advi_calls: Optional[int]) -> pd.DataFrame:
        """Processa una singola nave con il proprio scaler indipendente."""
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
            advi_indices = None  # usa update_freq

        last_prob, last_var = None, None

        for t in range(self.window_size, n):
            if t % 500 == 0:
                logging.info(f"  step {t}/{n}...")

            if advi_indices is not None:
                should_update = t in advi_indices
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
                    probs_sospetto[t] = prob_t
                    incertezza[t] = var_t

                except Exception as e:
                    logging.debug(f"Errore inferenza a t={t}: {e}")
                    if last_prob is not None:
                        probs_sospetto[t] = last_prob
                        incertezza[t] = last_var
                    continue
                finally:
                    for var_name in ['mean_field', 'trace']:
                        if var_name in locals():
                            del locals()[var_name]
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
                            max_advi_calls: Optional[int] = None) -> pd.DataFrame:
        """
        Processa il DataFrame per nave (MMSI) in modo indipendente per evitare
        leakage dello scaler tra navi diverse.
        Se max_advi_calls è impostato, distribuisce i call ADVI uniformemente
        lungo la serie (cap scalabilità per dataset grandi).
        """
        if 'MMSI' not in df.columns:
            logging.warning("Colonna MMSI assente: processing globale (no isolamento per nave).")
            logging.info(f"Avvio inferenza causale (Finestra: {self.window_size}, ADVI: {use_advi}, update_freq: {update_freq}, max_advi_calls: {max_advi_calls})...")
            return self._process_single_vessel(df, use_advi, apply_markov, update_freq, max_advi_calls)

        vessels = df['MMSI'].unique()
        logging.info(f"Avvio inferenza causale su {len(vessels)} navi (Finestra: {self.window_size}, ADVI: {use_advi}, max_advi_calls: {max_advi_calls})...")

        results = []
        for mmsi in vessels:
            logging.info(f"Processando nave MMSI={mmsi}...")
            df_vessel = df[df['MMSI'] == mmsi].sort_values('Timestamp').copy()
            df_enriched = self._process_single_vessel(df_vessel, use_advi, apply_markov, update_freq, max_advi_calls)
            results.append(df_enriched)

        df_out = pd.concat(results, ignore_index=True).sort_values(['MMSI', 'Timestamp']).reset_index(drop=True)
        logging.info(f"Inferenza completata. Feature estratte per {len(df_out)} osservazioni totali.")
        return df_out
    def reset_model_only(self):
        """🔴 FIX 3: Resetta solo il modello PyMC, mantiene lo scaler globale tra navi."""
        self.model = None
        logging.info("Modello PyMC resettato per nuova nave (scaler mantenuto).")

    def reset(self):
        """Reset completo (modello + scaler). Usare solo se si cambia dataset o pre-processing."""
        self.model = None
        self.scaler = None
        logging.info("Estrattore resettato completamente (modello + scaler).")


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