```markdown
# AIS Dark Fleet Predictor
**A Hybrid Bayesian Mixture and Gradient Boosting Approach for Intentional Missing Data Forecasting**

This repository contains an end-to-end Statistical Learning pipeline designed to predict the intentional deactivation of Automatic Identification System (AIS) transponders by maritime vessels. This tactical behavior is frequently associated with "Dark Fleet" operations, including illegal, unreported, and unregulated (IUU) fishing and unauthorized transshipments.

Rather than treating missing data as anomalies to be imputed, this project models the absence of the signal as the primary target variable. The architecture combines the epistemic uncertainty quantification of Bayesian latent models with the non-linear predictive power of tree-based algorithms.

---

## System Architecture

The project is structured into three isolated modules, strictly designed to prevent temporal data leakage:

1. **`src/data_prep.py` (Data Preprocessing & Target Definition)**
   - Geographic spatial filtering (Strait of Sicily).
   - Causal downsampling to 10-minute intervals utilizing a Last-Known-State approach.
   - Kinematic feature engineering (e.g., longitudinal acceleration, rate of turn).
   - Labeling via **Last Ping Prediction**: The target is exclusively anchored to the final valid signal preceding a prolonged blackout.

2. **`src/hmm_model.py` (Bayesian Latent Regime Extraction)**
   - Implements a **Bayesian Mixture Model (BMM)** — *not* a Hidden Markov Model despite the filename, which is a historical artifact. There is no learned transition matrix and no MCMC/Gibbs sampling.
   - Inference is performed via **Automatic Differentiation Variational Inference (ADVI)** in PyMC over a strictly causal rolling window. A Markov smoothing filter is applied *post-hoc* (deterministically) to impose temporal continuity on the posterior probabilities.
   - The module extracts both the point estimate (`prob_regime_sospetto`: posterior probability of the suspicious regime) and epistemic uncertainty (`incertezza_regime`: posterior variance).

3. **`src/gb_training.py` (Supervised Learning & Validation)**
   - Gradient Boosting (`LightGBM`) optimized via `Optuna`.
   - Rigorous evaluation utilizing a custom **TimeSeriesSplitWithGap**, which separates training and validation folds with a temporal buffer equal to the prediction horizon.
   - Statistical significance testing executed via **Grouped Bootstrap** on vessel trajectories.

---

## Methodology: Target Abstraction

### The Proxy Problem
Due to the absence of judicial ground-truth regarding maritime crimes, this project models the tactical precursor to the illicit activity: the deactivation of the transponder.

- **Blackout Threshold:** A "Dark Fleet" event is defined as an AIS interruption exceeding 12 hours. Existing literature (e.g., Global Fishing Watch) indicates that gaps of this magnitude in areas with high coastal and satellite coverage are overwhelmingly intentional rather than technical failures.
- **Causal-Safe Formulation:** The algorithm does not attempt to predict "when" the vessel will reactivate the system; instead, it learns to identify the tactical and kinematic profile of the vessel in the exact instant immediately preceding the disappearance.

---

## Validation and Statistical Significance

Given the severe class imbalance of the target (rare events < 5%), standard metrics such as Accuracy or ROC-AUC are highly misleading. The model is evaluated using:
- **PR-AUC (Precision-Recall Area Under Curve):** The primary metric for rare event detection.
- **Brier Score:** Utilized to evaluate the probabilistic calibration of the output.

### Ablation Study
To empirically demonstrate the value of the Bayesian features, the pipeline executes a structured Ablation Study:
1. **Baseline Model:** Trained exclusively on raw kinematic features (speed, course, and their derivatives).
2. **Enhanced Model:** Trained on kinematic features combined with the Latent Regime Probability and Bayesian Uncertainty.

The statistical significance of the delta in PR-AUC is validated through a **Grouped Bootstrap** utilizing 1000 iterations. Sampling is performed by vessel identifier (`MMSI`) to ensure the temporal autocorrelation of the time series remains intact.

---

## Quick Start

### 1. Installation
Ensure Python 3.10+ is installed.
```bash
# Create virtual environment and install dependencies
bash setup.sh 
# On Windows use: .\setup.bat

```

### 2. End-to-End Execution

The primary configuration file is located at `configs/pipeline_config.yaml`.

```bash
python run_pipeline.py

```

The script autonomously executes data preparation, Bayesian inference, and HPO training. It exports the trained models alongside interpretability plots (SHAP and Calibration Curves) to the `models/` directory.

---

## Limitations and Future Work

1. **Hardware False Positives:** Although rare, catastrophic failures to a vessel's electrical infrastructure generate data gaps identical to intentional deactivations.
2. **Short Blackouts (False Negatives):** Rapid "tactical" deactivations (e.g., < 6 hours) intended to mask specific maneuvers are not captured by the current 12-hour target threshold. Sensitivity analysis regarding the `gap_threshold_hours` parameter can be configured via the YAML file.
3. **Concept Drift and Geographic Bias:** The model is calibrated to the navigational dynamics and topology of the Strait of Sicily. Deployment in Out-of-Distribution scenarios (e.g., the Pacific Ocean) will require geographic fine-tuning of the Bayesian priors and complete retraining of the discriminative model.

## 🎨 Dashboard Interattiva
Il progetto include una dashboard Streamlit completa:

### Avvio Dashboard
```bash
# Con dati demo
python src/dashboard/data_loader.py --generate-demo
streamlit run app.py

# Oppure con Docker
docker build -f Dockerfile.dashboard -t ais-dashboard .
docker run -p 8501:8501 ais-dashboard
```
