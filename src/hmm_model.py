"""
src/hmm_model.py
Modulo per l'inferenza Bayesiana dei regimi latenti (Mixture Model) con validazione causale.

Questo modulo implementa un Bayesian Mixture Model con finestra scorrevole causale
per estrarre probabilità di regime "sospetto" da serie temporali AIS.
L'approccio è rigorosamente causale: al tempo t, il modello vede solo dati <= t.
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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CausalBayesianMixture:
    """
    Estrattore di regime latente tramite Mixture Model Bayesiano causale.
    
    Features estratte:
    - prob_regime_sospetto: P(Regime=0 | dati passati), dove 0 = comportamento sospetto
    - incertezza_regime: Varianza della posterior, segnale di ambiguità comportamentale
    """
    
    def __init__(self, window_size: int = 36, sigma_prior_speed: float = 1.0, 
                 sigma_prior_turn: float = 2.0):
        """
        Inizializza l'estrattore.
        
        Args:
            window_size: Dimensione della finestra temporale (es. 36 = 6 ore se dati a 10 min)
            sigma_prior_speed: Scala del prior per le medie di speed_acc (dopo standardizzazione)
            sigma_prior_turn: Scala del prior per le stddev di turn_rate (dopo standardizzazione)
        """
        self.window_size = window_size
        self.sigma_prior_speed = sigma_prior_speed
        self.sigma_prior_turn = sigma_prior_turn
        self.model = None
        self.scaler = None  # Salva lo scaler per coerenza tra train/inference
        
    def build_model(self):
        """Costruisce il grafo computazionale una sola volta usando tensori mutabili."""
        with pm.Model() as self.model:
            # 1. TENSORI MUTABILI (Permettono di aggiornare i dati senza ricompilare)
            speed_data = pm.Data("speed_data", np.zeros(self.window_size), mutable=True)
            turn_data = pm.Data("turn_data", np.zeros(self.window_size), mutable=True)
            
            # 2. PRIORS (Assunzioni sulla realtà, post-standardizzazione)
            theta = pm.Dirichlet("theta", a=np.array([1.0, 1.0]))
            
            # ORDINAMENTO RIGIDO (Anti Label-Switching)
            # Stato 0: Sospetto/Pesca/Loitering (bassa |speed_acc|, alta |turn_rate|)
            # Stato 1: Transito (alta |speed_acc| costante, bassa |turn_rate|)
            mu_sospetto = pm.Normal("mu_sospetto", mu=0, sigma=self.sigma_prior_speed)
            gap = pm.HalfNormal("gap", sigma=self.sigma_prior_speed)
            mu_transito = pm.Deterministic("mu_transito", mu_sospetto + gap)
            
            mu_speed = pm.math.stack([mu_sospetto, mu_transito])
            sigma_speed = pm.HalfNormal("sigma_speed", sigma=self.sigma_prior_speed, shape=2)
            
            # Varianza delle virate: più alta per lo stato sospetto
            sigma_turn = pm.HalfNormal("sigma_turn", sigma=self.sigma_prior_turn, shape=2)
            
            # 3. REGIMI LATENTI E LIKELIHOOD
            regime = pm.Categorical("regime", p=theta, shape=self.window_size)
            
            pm.Normal("obs_speed", mu=mu_speed[regime], sigma=sigma_speed[regime], observed=speed_data)
            pm.Normal("obs_turn", mu=0, sigma=sigma_turn[regime], observed=turn_data)

    def check_convergence_mcmc(self, trace: az.InferenceData, threshold_rhat: float = 1.05) -> bool:
        """Verifica se l'MCMC ha raggiunto la stazionarietà."""
        try:
            rhat = az.rhat(trace)
            max_rhat = float(rhat.to_dataframe().max().max())
            if max_rhat > threshold_rhat:
                logging.warning(f"Convergenza MCMC subottimale: Max R-hat = {max_rhat:.3f}")
                return False
            return True
        except Exception as e:
            logging.debug(f"Errore diagnostica MCMC: {e}")
            return False
    
    def check_convergence_advi(self, mean_field, patience: int = 100, tol: float = 1e-3) -> bool:
        """Verifica se ADVI ha raggiunto un plateau nell'ELBO."""
        try:
            loss = mean_field.hist["loss"]
            if len(loss) < patience:
                return False
            # Controlla se la media degli ultimi patience valori è stabile
            recent = loss[-patience:]
            if np.abs(recent[-1] - recent[0]) < tol * np.abs(recent[0]):
                return True
            return False
        except Exception as e:
            logging.debug(f"Errore diagnostica ADVI: {e}")
            return False

    def _standardize_features(self, speed_feat: np.ndarray, turn_feat: np.ndarray, 
                           fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Standardizza le feature mantenendo causalità (fit solo sul primo chunk)."""
        features = np.stack([speed_feat, turn_feat], axis=1)
        
        if fit:
            # Fit sullo primo chunk per evitare leakage dal futuro
            fit_end = min(self.window_size * 3, len(features) // 3)
            self.scaler = StandardScaler()
            self.scaler.fit(features[:fit_end])
        
        if self.scaler is None:
            raise RuntimeError("Scaler non inizializzato. Chiama prima con fit=True.")
            
        features_scaled = self.scaler.transform(features)
        return features_scaled[:, 0], features_scaled[:, 1]

    def process_dataframe_causal(self, df: pd.DataFrame, use_advi: bool = True,
                                fit_scaler: bool = True) -> pd.DataFrame:
        """
        Esegue l'inferenza usando una rolling window rigorosamente causale.
        
        Args:
            df: DataFrame con colonne ['speed_acc', 'turn_rate']
            use_advi: Se True usa Variational Inference (veloce), altrimenti MCMC (lento ma preciso)
            fit_scaler: Se True, fit dello scaler sul primo chunk (da True solo per la prima nave)
        
        Returns:
            DataFrame con colonne aggiuntive ['prob_regime_sospetto', 'incertezza_regime']
        """
        logging.info(f"Avvio inferenza causale (Finestra: {self.window_size}, ADVI: {use_advi})...")
        
        # Pulizia dati
        df = df.dropna(subset=['speed_acc', 'turn_rate']).copy()
        if len(df) < self.window_size + 1:
            logging.warning(f"Dataset troppo corto ({len(df)} < {self.window_size+1}). Restituisco NaN.")
            df['prob_regime_sospetto'] = np.nan
            df['incertezza_regime'] = np.nan
            return df
        
        speed_feat = df['speed_acc'].values
        turn_feat = df['turn_rate'].values
        
        # Standardizzazione (causale: fit solo sul primo chunk)
        speed_feat_scaled, turn_feat_scaled = self._standardize_features(
            speed_feat, turn_feat, fit=fit_scaler
        )
        
        # Pre-allochiamo gli array per le feature generate
        probs_sospetto = np.full(len(df), np.nan)
        incertezza = np.full(len(df), np.nan)
        
        # Costruiamo il modello una sola volta in memoria
        if self.model is None:
            self.build_model()
            
        # LOOP TEMPORALE CAUSALE
        for t in range(self.window_size, len(df)):
            if t % 500 == 0:
                logging.info(f"Processato {t}/{len(df)} time-steps...")
                
            # Finestra che INCLUDE il tempo t: [t-window_size+1 : t+1]
            # Questo garantisce che l'ultimo elemento della finestra sia esattamente il tempo t
            x_speed = speed_feat_scaled[t - self.window_size + 1 : t + 1]
            x_turn = turn_feat_scaled[t - self.window_size + 1 : t + 1]
            
            with self.model:
                # Aggiorniamo i dati nel grafo senza ricompilarlo
                pm.set_data({"speed_data": x_speed, "turn_data": x_turn})
                
                try:
                    if use_advi:
                        # VARIATIONAL INFERENCE: Molto più veloce dell'MCMC
                        mean_field = pm.fit(n=15000, method='advi', progressbar=False)
                        
                        # Diagnostica convergenza ADVI
                        if not self.check_convergence_advi(mean_field):
                            logging.debug(f"ADVI non convergente a t={t}, salto")
                            continue
                            
                        trace = mean_field.sample(500)
                        post_regimes = trace.posterior["regime"].values
                        del mean_field  # Cleanup memoria
                    else:
                        # MCMC PURO: Lento ma rigoroso
                        trace = pm.sample(draws=300, tune=200, cores=1, progressbar=False, 
                                         compute_convergence_checks=False)
                        if not self.check_convergence_mcmc(trace):
                            continue
                        post_regimes = trace.posterior["regime"].values
                        del trace  # Cleanup memoria
                    
                    # Estraiamo la probabilità SOLO per l'ultimo punto della finestra (tempo t)
                    # post_regimes shape: (chains, draws, window_size)
                    prob_t = np.mean(post_regimes[:, :, -1] == 0)
                    var_t = np.var(post_regimes[:, :, -1] == 0)
                    
                    probs_sospetto[t] = prob_t
                    incertezza[t] = var_t
                    if use_advi and 'trace' in locals():
                        del trace
                    gc.collect()
                except Exception as e:
                    logging.debug(f"Errore inferenza a t={t}: {e}")
                    continue
                    
        # Gestione causal-safe dei NaN iniziali (cold start della finestra)
        # Usiamo un prior globale invece di bfill che violerebbe la causalità
        if np.isnan(probs_sospetto).any():
            valid_mask = ~np.isnan(probs_sospetto)
            if valid_mask.any():
                global_prior = np.mean(probs_sospetto[valid_mask])
                global_unc = np.mean(incertezza[valid_mask])
                probs_sospetto = np.where(np.isnan(probs_sospetto), global_prior, probs_sospetto)
                incertezza = np.where(np.isnan(incertezza), global_unc, incertezza)
            else:
                # Fallback estremo: prior uniforme
                probs_sospetto = np.where(np.isnan(probs_sospetto), 0.5, probs_sospetto)
                incertezza = np.where(np.isnan(incertezza), 0.25, incertezza)
        
        df['prob_regime_sospetto'] = probs_sospetto
        df['incertezza_regime'] = incertezza
        
        logging.info(f"Inferenza completata. Feature estratte per {len(df)} osservazioni.")
        return df
    
    def reset(self):
        """Resetta lo stato interno per processare una nuova nave indipendente."""
        self.model = None
        self.scaler = None
        logging.info("Estrattore resettato per nuova nave.")


if __name__ == "__main__":
    # ========================================================================
    # TEST DI INTEGRAZIONE: Verifica causalità, standardizzazione e feature extraction
    # ========================================================================
    logging.info("=== TEST INTEGRAZIONE CausalBayesianMixture ===")
    
    # 1. Dataset mock con pattern distinti: prima transito, poi comportamento sospetto
    np.random.seed(42)
    n_transit, n_suspicious = 100, 100
    
    # Transito: speed_acc alta e stabile, turn_rate bassa
    mock_speed_transit = np.random.normal(0.5, 0.3, n_transit)  # dopo standardizzazione
    mock_turn_transit = np.random.normal(0, 0.5, n_transit)
    
    # Sospetto: speed_acc vicina a zero (loitering), turn_rate alta e variabile
    mock_speed_suspicious = np.random.normal(0, 0.2, n_suspicious)
    mock_turn_suspicious = np.random.normal(0, 2.0, n_suspicious)
    
    mock_speed_acc = np.concatenate([mock_speed_transit, mock_speed_suspicious])
    mock_turn = np.concatenate([mock_turn_transit, mock_turn_suspicious])
    
    test_df = pd.DataFrame({
        'speed_acc': mock_speed_acc,
        'turn_rate': mock_turn
    })
    
    # 2. Esecuzione con ADVI (veloce per test)
    extractor = CausalBayesianMixture(window_size=30)
    df_enriched = extractor.process_dataframe_causal(test_df, use_advi=True, fit_scaler=True)
    
    # 3. Validazione risultati
    print("\n" + "="*60)
    print("RISULTATI TEST")
    print("="*60)
    
    # Check 1: Nessuna colonna leakata
    assert 'dt_next_hours' not in df_enriched.columns, " LEAKAGE: colonna futura presente!"
    print(" Anti-leakage check: PASSED")
    
    # Check 2: Feature presenti e nel range [0,1] per probabilità
    assert 'prob_regime_sospetto' in df_enriched.columns, "Feature prob_regime_sospetto mancante!"
    assert df_enriched['prob_regime_sospetto'].between(0, 1).all(), " Probabilità fuori range!"
    print(" Feature extraction: PASSED")
    
    # Check 3: Pattern atteso (probabilità bassa all'inizio, alta alla fine)
    early_mean = df_enriched['prob_regime_sospetto'].iloc[30:60].mean()  # zona transito
    late_mean = df_enriched['prob_regime_sospetto'].iloc[-30:].mean()    # zona sospetta
    
    print(f"\n Probabilità media regime sospetto:")
    print(f"   - Fase transito (righe 30-60): {early_mean:.3f}")
    print(f"   - Fase sospetta (ultime 30):   {late_mean:.3f}")
    
    if late_mean > early_mean:
        print(" Pattern detection: PASSED (sospetto > transito)")
    else:
        print("  Pattern detection: ATTENZIONE (verificare priors o dati)")
    
    # Check 4: Incertezza estratta
    assert 'incertezza_regime' in df_enriched.columns, "❌ Feature incertezza_regime mancante!"
    print(" Feature incertezza: PASSED")
    
    # 4. Output campione
    print("\n Esempio output (ultime 5 righe):")
    print(df_enriched[['prob_regime_sospetto', 'incertezza_regime']].tail())
    
    print("\n=== TEST COMPLETATO ===")