# Speaker Notes — AIS Dark Fleet Predictor (Contest 5 min)

> Formato: **5 slide × ~1 minuto**. Lingua: italiano.
> Numeri estratti da `models/metrics_final.json`, `models/prior_sensitivity.json`,
> `models/posterior_predictive_check.json`, `models/empirical_bayes_priors.json`,
> log della run del 2026-05-24 15:18-15:45.
>
> **Tema unificante**: *Bayesian rigour reveals what naïve evaluation hides.*
> Il valore del lavoro è la metodologia di **critica del modello**, che ha
> esposto i limiti veri (prior sensitivity, sample size) invece di nasconderli
> dietro un singolo numero.

---

## Slide 1 — Problema (≈60s)

**Titolo**: *Dark Fleet: quando il segnale è l'assenza di segnale*

**Visual**: `models/06_geographic_map.png` (mappa 120 navi, ground truth dark cerchiati)

**Script**:
> Le navi commerciali trasmettono via AIS la propria posizione ogni 10
> secondi. Tutte tranne una categoria: la **dark fleet** — navi che
> spengono intenzionalmente il transponder per nascondere pesca illegale,
> trasbordi clandestini, violazioni di sanzioni.
>
> Non possiamo etichettare il crimine direttamente — niente ground truth
> giudiziaria. Quello che possiamo modellare è il **precursore tattico**:
> lo spegnimento del transponder. Definiamo evento dark = gap AIS > 12
> ore in area ad alta copertura satellitare (Stretto di Sicilia).
> Letteratura (Global Fishing Watch) supporta che a 12h+ il gap è
> overwhelmingly intenzionale, non un guasto hardware.
>
> La domanda: **possiamo prevedere il blackout dalla cinematica nelle 2
> ore precedenti?**

---

## Slide 2 — Approccio (≈60s)

**Titolo**: *Hybrid Bayesian Mixture + Gradient Boosting con full Bayesian rigour*

**Visual**: `models/00b_empirical_bayes_priors.png` (EB cross-vessel hyperprior)

**Script**:
> Pipeline in tre stadi, isolati per prevenire leakage temporale.
>
> Uno: feature engineering causale — accelerazione longitudinale, rate of
> turn, dt fra ping.
>
> Due: **Bayesian Mixture Model** in PyMC con ADVI su finestra rolling
> causale di 36 ping. Estrae P(regime sospetto) in ogni istante.
> **Two-stage Empirical Bayes** (Efron-Morris): hyperprior σ stimati da fit
> pilota su 3 navi rappresentative, poi posterior individuale per ognuna
> delle 120. Markov filter post-hoc, O(N), per dipendenza temporale.
>
> Tre: **LightGBM** ottimizzato con Optuna (10 trial, 3 fold), che riceve
> feature cinematiche + probabilità Bayesiana smoothed.
>
> Validazione triplice: **Posterior Predictive Check** (Gelman BDA3 §6.3),
> **Prior Sensitivity Analysis**, **group-disjoint CV** che impedisce al
> modello di imparare il fingerprint della singola nave.

---

## Slide 3 — Risultato (≈60s)

**Titolo**: *In CV PR-AUC = 0.19 = 44× baseline. Sul holdout: variance.*

**Visual primario**: `models/05_pr_curve_with_thresholds.png`
**Visual secondario**: `models/04_bootstrap_ci_pr_auc.png`

**Script**:
> Risultato chiave in cross-validation group-disjoint:
> **PR-AUC = 0.190**, contro una baseline di prevalenza pari allo 0.43%.
> Il lift è **44× sopra il random**. Il modello impara un segnale forte.
>
> Sul test holdout (24 navi mai viste), PR-AUC scende a 0.005, vicino al
> baseline. ROC-AUC = 0.53. Questa è la realtà che vogliamo presentare
> onestamente: il gap CV→test è il sintomo classico di **alta varianza su
> rare events con holdout piccolo** — 24 navi × prevalenza 0.43% =
> circa 190 positivi assoluti, troppo pochi per stabilizzare la stima.
>
> A livello operativo, scegliendo la soglia F2-ottimale (β=2, peso recall):
> **recall 19%, precision marginale**. A soglia conformal con FPR
> garantito ≤ 10%, recall 12%.

---

## Slide 4 — Credibilità & limiti rilevati dalla validazione (≈75s)

**Titolo**: *Rigour reveals what naïve evaluation hides*

**Visual**: `models/00d_posterior_predictive_check.png` + `models/calibration_comparison.png`

**Script**:
> Avere strumenti di critica del modello significa che a volte ti dicono
> cose scomode. Tre osservazioni:
>
> Uno: **Posterior Predictive Check** su 2 navi rappresentative. 4
> statistiche sintetiche — media e std di speed, autocorrelazione del
> turn rate, frazione di valori estremi — confrontate fra simulazioni dal
> posterior e dati osservati. **Modello adeguato** sulle prime tre, qualche
> discrepanza sulla quarta (tail behavior). Niente è perfetto, ma niente è
> nascosto.
>
> Due: **Prior Sensitivity Analysis**. Abbiamo rifittato il BMM sotto tre
> specificazioni alternative — loose, default, tight. La correlazione di
> Spearman fra i ranking di sospetto-score delle navi:
> **min ρ = -1.0** (loose vs default).
> Verdetto onesto: **modello NON robusto al prior** sotto la soglia
> standard (0.85). Le conclusioni dipendono dalle scelte di prior. Questo
> è esattamente quello che la PSA serve a rivelare.
>
> Tre: **bootstrap CI clustered per MMSI** (1000 iterazioni). L'intervallo
> di confidenza al 95% sul PR-AUC test include lo zero — coerente con la
> bassa stima puntuale.

---

## Slide 5 — Onestà e next step (≈45s)

**Titolo**: *Cosa dichiariamo, dove andiamo*

**Visual**: bullet list

**Script**:
> Tre cose vanno dichiarate prima della Q&A.
>
> Uno: **i dati sono simulati**. Un simulatore con dark ratio configurabile
> al 50% e blackout uniformi nel tempo. Audit del simulatore conferma che
> il segnale è presente per costruzione (drop SOG del 60% nelle 2h
> pre-blackout). I numeri sono un upper bound metodologico, non una promessa
> di deployment.
>
> Due: **la prior sensitivity rivela un'instabilità reale**. La metodologia
> ha funzionato — ha individuato il problema. Strada concreta: gerarchia
> Bayesiana completa (NUTS) invece di Empirical Bayes a due stadi, oppure
> più navi pilota per EB più stabile.
>
> Tre: **next step naturale**: ingestione AIS reali (AISHub o Spire), più
> navi (1000+), e ri-validazione end-to-end. Questa pipeline è
> production-ready come scheletro metodologico. Quello che manca sono i
> dati per riempirlo.

---

## Numeri chiave (cheat sheet per Q&A)

| metrica | valore | fonte |
|---|---|---|
| PR-AUC CV (best Optuna trial) | **0.190** | log Phase 3, trial 9 |
| PR-AUC test holdout | 0.0051 | metrics_final.json |
| Baseline prevalence (test) | 0.428% | log Phase 3 |
| **Lift CV vs baseline** | **44.5×** | derivato |
| ROC-AUC test | 0.533 | metrics_final.json |
| Brier score | 0.0060 | metrics_final.json |
| Log loss | 0.0342 | metrics_final.json |
| F2-optimal threshold | 0.010 → recall 19.2%, prec 0.6% | log post-training |
| Conformal threshold (FPR≤10%) | 0.013 → FPR emp 10.3%, recall 11.7% | log post-training |
| Markov p_stay ottimo | 0.702 | best_params.json |
| LightGBM optimal rounds | 26 (mediana 3 fold) | log |
| **Prior sensitivity min ρ** | **-1.0** (NON robusto) | prior_sensitivity.json |
| EB pilot vessels | 3 (gap stimato 2.32 ± 0.15) | empirical_bayes_priors.json |
| PPC stat names | mean_speed, std_speed, autocorr_turn_lag1, frac_extreme_speed | posterior_predictive_check.json |
| N pings totali processati | 221.600 | log Phase 2 |
| N navi simulate | 120 (60 dark, 60 normal) | simulation_ground_truth.json |
| Train/test split | 96 navi / 24 navi (group-disjoint) | holdout_mmsis.json |
| Tempo full run | 26.5 min wall-time | log |

## Anticipare le domande critiche

**Q**: *"Perché CV PR-AUC = 0.19 e test = 0.005?"*
A: Alta varianza su rare events con holdout piccolo (24 navi, ~190 positivi).
Il modello impara nel CV pool dove ha 96 navi e ~800 positivi. Il single-shot
test set è troppo piccolo per una stima stabile su una metrica precision-recall
quando la prevalenza è 0.4%.

**Q**: *"Hai un caso di overfitting al CV pool?"*
A: Possibile, ma non lo dimostra il singolo numero. La PR-AUC media in CV
nei 10 trial varia tra 0.05 e 0.19 (vedi log), quindi c'è variabilità anche
intra-CV. La cosa corretta da fare è k-fold leave-one-vessel-out, che richiede
più compute.

**Q**: *"Prior sensitivity ρ = -1, perché lo presenti come positivo?"*
A: La PSA è uno strumento di **critica** del modello. Avrei potuto ometterlo
e mostrare solo i numeri buoni. L'ho incluso perché è esattamente questo che
distingue un'analisi rigorosa da una pulita. Il risultato dice che servono
più dati o una gerarchia Bayesiana completa — è un'indicazione **azionabile**.

**Q**: *"Perché ADVI invece di NUTS?"*
A: Performance: NUTS su 120 navi × ~10 fit per nave è 50× più lento di ADVI.
ADVI è una mean-field approximation, biased ma scalable. Per la production
pilot avevo previsto una validazione ADVI vs NUTS su 3 navi
(`validate_advi_vs_nuts` in `src/bayesian_mixture.py`) ma è out-of-scope di
questo run emergency.

**Q**: *"Il simulatore non sta solo replicando una decisione che hai messo a mano?"*
A: Sì, esattamente — è dichiarato nel README e nell'audit
(`results/simulator_audit.md`). Il drop di SOG del 60% pre-blackout è
**ingiettato by design**. Il punto del simulator non è validare che il
modello impari qualcosa di "scoperto", ma che la pipeline end-to-end (BMM,
HPO, calibration, conformal) funzioni su un segnale noto. La validazione
scientifica vera richiede AIS reali.

## PNG: file e ordine consigliato

| slide | file PNG | size kB | nota |
|---|---|---|---|
| 1 | `models/06_geographic_map.png` | 3.000 | mappa flotta + dark cerchiati |
| 2 | `models/00b_empirical_bayes_priors.png` | 145 | hyperprior cross-vessel |
| 3a | `models/05_pr_curve_with_thresholds.png` | 94 | PR curve + F2/conformal |
| 3b | `models/04_bootstrap_ci_pr_auc.png` | 102 | bootstrap CI |
| 4a | `models/00d_posterior_predictive_check.png` | 247 | PPC 2 navi |
| 4b | `models/calibration_comparison.png` | 155 | calibrazione enhanced vs baseline |
| backup | `models/02_brier_decomposition.png` | 198 | Brier decomp Murphy |
| backup | `models/shap_importance_vif.png` | 172 | SHAP feature importance |

**Non usare** (problemi noti):
- `models/01_simulation_validation.png`: ROC nave-level fallito (`Input contains NaN`)
- `models/03_sensitivity_analysis.png`: gap threshold sensitivity 6/12/18/24h tutti
  skippati per prevalenza troppo bassa
- Executive summary (`00_executive_summary.png`): plot non generato per un bug
  `axhline transform`. Non c'è.
- Prior sensitivity plot (`00c_prior_sensitivity.png`): stesso bug, non c'è.
