# Telco Customer Churn Analytics

An end-to-end machine learning project for predicting and explaining customer churn, featuring a professional dark-theme dashboard with business context, five predictive models with hyperparameter tuning, SHAP explainability, and actionable retention recommendations.

---

## Table of Contents

- [Overview](#overview)
- [Dashboard](#dashboard)
- [Dataset](#dataset)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Results](#model-results)
- [Technologies Used](#technologies-used)

---

## Overview

Customer churn — when a customer stops doing business with a company — is one of the most critical metrics for subscription-based businesses. This project builds a complete ML pipeline on the IBM Telco Customer Churn dataset to:

1. **Understand** churn patterns through descriptive analytics and visualizations
2. **Predict** which customers are at risk using five different ML models
3. **Explain** model decisions using SHAP (SHapley Additive exPlanations)
4. **Act** on predictions with risk-tiered retention recommendations
5. **Present** findings through a business-ready portfolio dashboard

---

## Dashboard

**Live app:** [homework1-wyu7nzfpsjq8hybmvdqm.streamlit.app](https://homework1-wyu7nzfpsjq8hybmvdqm.streamlit.app)

The dashboard features a complete dark professional UI redesign built for portfolio and stakeholder presentations:

- **Gradient hero header** with key project stats at a glance
- **Business context cards** — "What is Customer Churn?" and "Why Does This Matter?" with real revenue impact figures
- **Color-coded KPI cards** with accent borders per metric category
- **Metrics comparison table** with green highlighting for best values and red for worst, per column
- **"What Does Each Model Do?"** section explaining each algorithm in plain language
- **Interactive gauge chart** showing churn probability with color-coded risk zones
- **Risk-tiered recommendations** (Low / Moderate / High / Critical) with specific retention actions
- **SHAP waterfall explanations** written accessibly for non-technical stakeholders

---

## Dataset

**Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| File | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Rows | 7,043 customers |
| Columns | 21 features + 1 target |
| Target | `Churn` (Yes / No) — 26.5% positive rate |

**Feature categories:**

- **Demographics** — gender, senior citizen status, partner, dependents
- **Account info** — tenure, contract type, payment method, paperless billing
- **Services** — phone, internet, online security, backup, streaming TV/movies
- **Charges** — monthly charges, total charges

---

## Features

### Descriptive Analytics
- Dataset overview, shape, data types, and summary statistics
- Churn distribution (count + pie chart)
- Tenure histogram split by churn status
- Monthly charges boxplot by churn status
- Churn rate by contract type, internet service, payment method, and senior citizen status
- Full feature correlation heatmap

### Predictive Models
Five models trained on a 70/30 stratified split (`random_state=42`):

| Model | Tuning |
|---|---|
| Logistic Regression | Baseline |
| Decision Tree | GridSearchCV + 5-fold CV |
| Random Forest | GridSearchCV + 5-fold CV |
| XGBoost | GridSearchCV + 5-fold CV |
| Neural Network (MLP) | Keras, 3 hidden layers, EarlyStopping |

Each model reports: **Accuracy, Precision, Recall, F1, AUC-ROC**

### Explainability (SHAP)
- SHAP beeswarm summary plot
- SHAP mean absolute value bar plot
- SHAP waterfall plot for a single prediction with plain-language interpretation
- Applied to the best-performing tree-based model (XGBoost)

### Streamlit App — 4 Tabs

| Tab | Contents |
|---|---|
| **Executive Summary** | Business context cards, KPI cards, model leaderboard, key business insights |
| **Descriptive Analytics** | 8 interactive dark-theme Plotly charts with written interpretations |
| **Model Performance** | Color-coded metrics table, ROC curves, metric bar chart, best hyperparameters, SHAP plots with explanation |
| **Interactive Prediction** | Customer profile builder, live churn probability gauge, risk badge, actionable retention recommendation, SHAP waterfall explanation |

---

## Project Structure

```
homework1/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
├── train_models.py                          # Full ML training pipeline
├── app.py                                   # Streamlit dashboard
├── requirements.txt                         # Python dependencies
├── runtime.txt                              # Python version (3.11)
├── .streamlit/
│   └── config.toml                          # Dark theme configuration
├── models/                                  # Saved model artifacts
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── mlp.keras
│   ├── scaler.joblib
│   ├── feature_names.joblib
│   ├── metrics.joblib
│   ├── best_params.joblib
│   ├── shap_explainer.joblib
│   └── ...
└── plots/                                   # Saved plot images
    ├── 01_churn_distribution.png
    ├── 02_tenure_distribution.png
    ├── ...
    └── 13_shap_waterfall.png
```

---

## Installation

**Prerequisites:** Python 3.9 or higher

```bash
# Clone the repository
git clone https://github.com/Norahe8/homework1.git
cd homework1

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Train the Models

Run the training pipeline first. This will preprocess the data, train all five models with hyperparameter search, run SHAP analysis, and save all artifacts to `models/` and `plots/`.

```bash
python train_models.py
```

Expected output:
```
[1/7] Loading dataset ...        Rows: 7,043  |  Columns: 21
[2/7] Generating descriptive analytics plots ...
[3/7] Preprocessing & splitting data ...
[4/7] Training models ...
      [1/5] Logistic Regression ...
      [2/5] Decision Tree + GridSearchCV ...
      [3/5] Random Forest + GridSearchCV ...
      [4/5] XGBoost + GridSearchCV ...
      [5/5] Neural Network (MLP, Keras) ...
[5/7] Generating model comparison plots ...
[6/7] Running SHAP analysis ...
[7/7] Pipeline complete!
```

> Training time is approximately 5-15 minutes depending on hardware, mainly due to GridSearchCV.

### Step 2 — Launch the Streamlit App

```bash
python -m streamlit run app.py
```

Then open your browser and navigate to: **http://localhost:8501**

> If `streamlit` is not on your PATH, use `python -m streamlit run app.py`.

---

## Model Results

Results on the held-out test set (30% of data, 2,113 samples):

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| **XGBoost** | 0.8017 | 0.6723 | 0.4938 | 0.5694 | **0.8457** |
| Logistic Regression | **0.8088** | 0.6660 | 0.5615 | 0.6093 | 0.8447 |
| Random Forest | 0.7941 | **0.6810** | 0.4225 | 0.5215 | 0.8415 |
| Neural Network | 0.7918 | 0.6217 | 0.5508 | 0.5841 | 0.8295 |
| Decision Tree | 0.7946 | 0.6152 | **0.6043** | **0.6097** | 0.8290 |

**Best model by AUC-ROC: XGBoost** with best hyperparameters `learning_rate=0.05`, `max_depth=3`, `n_estimators=100`, `subsample=0.8`.

**Key SHAP findings — top drivers of churn:**
- Contract type (month-to-month = highest risk)
- Low tenure (customers in their first year)
- Fiber optic internet service
- High monthly charges
- No online security add-on

---

## Technologies Used

| Category | Library |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Machine learning | `scikit-learn`, `xgboost` |
| Deep learning | `tensorflow` / `keras` |
| Explainability | `shap` |
| Model persistence | `joblib` |
| Web app | `streamlit` |
