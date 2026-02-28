"""
Telco Customer Churn - Complete ML Pipeline
===========================================
Trains 5 models, performs SHAP analysis, saves all artifacts.
Run this script BEFORE launching the Streamlit app.

Usage:
    python train_models.py
    streamlit run app.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)

import xgboost as xgb
import shap
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR  = "models"
PLOTS_DIR   = "plots"
RANDOM_STATE = 42
TEST_SIZE    = 0.30
CV_FOLDS     = 5

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

sns.set_style("whitegrid")
sns.set_palette("husl")
COLORS = ["#2ecc71", "#e74c3c"]

# ─────────────────────────────────────────────
#  STEP 1 – LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print(" TELCO CUSTOMER CHURN – ML PIPELINE")
print("=" * 60)
print("\n[1/7] Loading dataset …")

df = pd.read_csv(DATA_PATH)
print(f"      Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(f"      Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
print(f"\n--- Basic Statistics ---")
print(df.describe(include="all").to_string())
print(f"\n--- Missing values ---")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\n--- Data types ---")
print(df.dtypes)

# ─────────────────────────────────────────────
#  PREPROCESSING HELPER (shared with app.py)
# ─────────────────────────────────────────────
CAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
BINARY_COLS   = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
NUMERIC_COLS  = ["tenure", "MonthlyCharges", "TotalCharges"]


def preprocess_dataframe(df_input: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """Clean & encode the raw dataframe.  Does NOT scale."""
    df = df_input.copy()
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Fix TotalCharges (spaces → NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Encode target
    if "Churn" in df.columns:
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Binary encodings
    df["gender"] = (df["gender"] == "Male").astype(int)
    for col in BINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    # One-hot encode multi-class categoricals
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)

    # bool → int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


# ─────────────────────────────────────────────
#  STEP 2 – DESCRIPTIVE ANALYTICS
# ─────────────────────────────────────────────
print("\n[2/7] Generating descriptive analytics plots …")


def save_fig(name: str):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()


# ── Plot 1: Churn Distribution ──────────────
fig, ax = plt.subplots(figsize=(8, 6))
counts = df["Churn"].value_counts()
bars = ax.bar(
    ["Not Churned", "Churned"], counts.values,
    color=COLORS, width=0.5, edgecolor="white", linewidth=1.5
)
for bar, cnt in zip(bars, counts.values):
    pct = cnt / len(df) * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 60,
        f"{cnt:,}\n({pct:.1f}%)",
        ha="center", va="bottom", fontweight="bold", fontsize=12,
    )
ax.set_title("Customer Churn Distribution", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Churn Status", fontsize=13)
ax.set_ylabel("Number of Customers", fontsize=13)
ax.set_ylim(0, counts.max() * 1.2)
# Interpretation annotation
ax.text(
    0.98, 0.95,
    "Class imbalance: ~73% No Churn vs ~27% Churn",
    transform=ax.transAxes, ha="right", va="top",
    fontsize=9, color="gray", style="italic",
)
save_fig("01_churn_distribution.png")

# ── Plot 2: Tenure Histogram ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, (churn_val, label, color) in enumerate(
    [("No", "Not Churned", COLORS[0]), ("Yes", "Churned", COLORS[1])]
):
    subset = df[df["Churn"] == churn_val]["tenure"]
    axes[i].hist(subset, bins=30, color=color, alpha=0.85, edgecolor="white")
    median_v = subset.median()
    axes[i].axvline(median_v, color="navy", linestyle="--", linewidth=2,
                    label=f"Median: {median_v:.0f} mo")
    axes[i].set_title(f"Tenure – {label}", fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Tenure (Months)", fontsize=11)
    axes[i].set_ylabel("Count", fontsize=11)
    axes[i].legend(fontsize=10)
fig.suptitle(
    "Tenure Distribution by Churn Status\n"
    "→ Churned customers tend to leave early (short tenure)",
    fontsize=14, fontweight="bold",
)
save_fig("02_tenure_distribution.png")

# ── Plot 3: Monthly Charges Boxplot ──────────
fig, ax = plt.subplots(figsize=(10, 7))
groups = [
    df[df["Churn"] == "No"]["MonthlyCharges"],
    df[df["Churn"] == "Yes"]["MonthlyCharges"],
]
bp = ax.boxplot(
    groups, labels=["Not Churned", "Churned"],
    patch_artist=True, notch=True,
    medianprops=dict(color="navy", linewidth=2.5),
)
bp["boxes"][0].set_facecolor(COLORS[0])
bp["boxes"][1].set_facecolor(COLORS[1])
for i, g in enumerate(groups, 1):
    ax.text(i, g.max() * 1.02, f"Median: ${g.median():.2f}",
            ha="center", fontsize=10, fontweight="bold")
ax.set_title(
    "Monthly Charges by Churn Status\n"
    "→ Churned customers pay significantly higher monthly charges",
    fontsize=14, fontweight="bold",
)
ax.set_ylabel("Monthly Charges ($)", fontsize=13)
save_fig("03_monthly_charges_boxplot.png")

# ── Plot 4: Contract Type vs Churn ───────────
fig, ax = plt.subplots(figsize=(10, 7))
ct = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct.columns = ["Not Churned", "Churned"]
ct_pct.plot(kind="bar", ax=ax, color=COLORS, edgecolor="white", width=0.6)
ax.set_title(
    "Churn Rate by Contract Type\n"
    "→ Month-to-month contracts have far higher churn rates (~43%)",
    fontsize=14, fontweight="bold",
)
ax.set_xlabel("Contract Type", fontsize=13)
ax.set_ylabel("Percentage (%)", fontsize=13)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, fontsize=11)
ax.legend(fontsize=11)
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=3)
save_fig("04_contract_churn.png")

# ── Plot 5: Internet Service vs Churn ────────
fig, ax = plt.subplots(figsize=(10, 7))
it = df.groupby(["InternetService", "Churn"]).size().unstack(fill_value=0)
it.columns = ["Not Churned", "Churned"]
it.plot(kind="bar", ax=ax, color=COLORS, edgecolor="white", width=0.6)
ax.set_title(
    "Churn by Internet Service Type\n"
    "→ Fiber optic users churn at much higher rates than DSL users",
    fontsize=14, fontweight="bold",
)
ax.set_xlabel("Internet Service", fontsize=13)
ax.set_ylabel("Number of Customers", fontsize=13)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
ax.legend(fontsize=11)
save_fig("05_internet_service_churn.png")

# ── Plot 6: Payment Method vs Churn ──────────
fig, ax = plt.subplots(figsize=(12, 7))
pm = df.groupby(["PaymentMethod", "Churn"]).size().unstack(fill_value=0)
pm_pct = pm.div(pm.sum(axis=1), axis=0) * 100
pm_pct.columns = ["Not Churned", "Churned"]
pm_pct.plot(kind="barh", ax=ax, color=COLORS, edgecolor="white")
ax.set_title(
    "Churn Rate by Payment Method\n"
    "→ Electronic check users show the highest churn rate",
    fontsize=14, fontweight="bold",
)
ax.set_xlabel("Percentage (%)", fontsize=13)
ax.set_ylabel("Payment Method", fontsize=13)
ax.legend(fontsize=11)
save_fig("06_payment_method_churn.png")

# ── Plot 7: Senior Citizen vs Churn ──────────
fig, ax = plt.subplots(figsize=(8, 6))
sc = df.groupby(["SeniorCitizen", "Churn"]).size().unstack(fill_value=0)
sc.index = ["Non-Senior", "Senior"]
sc_pct = sc.div(sc.sum(axis=1), axis=0) * 100
sc_pct.columns = ["Not Churned", "Churned"]
sc_pct.plot(kind="bar", ax=ax, color=COLORS, edgecolor="white", width=0.5)
ax.set_title(
    "Churn Rate: Senior vs Non-Senior Citizens\n"
    "→ Senior citizens are ~2× more likely to churn",
    fontsize=14, fontweight="bold",
)
ax.set_xlabel("Customer Segment", fontsize=13)
ax.set_ylabel("Percentage (%)", fontsize=13)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
ax.legend(fontsize=11)
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", fontsize=10, padding=3)
save_fig("07_senior_citizen_churn.png")

# ── Plot 8: Correlation Heatmap ───────────────
df_proc_corr = preprocess_dataframe(df)
corr = df_proc_corr.corr()
fig, ax = plt.subplots(figsize=(20, 16))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, ax=ax, cmap="RdYlBu_r",
    center=0, square=True, linewidths=0.3,
    cbar_kws={"shrink": 0.7},
)
ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)
save_fig("08_correlation_heatmap.png")

print(f"      Saved {len(os.listdir(PLOTS_DIR))} plots to '{PLOTS_DIR}/'")

# ─────────────────────────────────────────────
#  STEP 3 – FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3/7] Preprocessing & splitting data …")

df_proc = preprocess_dataframe(df)
X = df_proc.drop("Churn", axis=1)
y = df_proc["Churn"]
feature_names = X.columns.tolist()
print(f"      Features: {len(feature_names)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# Scale numeric columns only
scaler = StandardScaler()
X_train_sc = X_train.copy()
X_test_sc  = X_test.copy()
X_train_sc[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
X_test_sc[NUMERIC_COLS]  = scaler.transform(X_test[NUMERIC_COLS])

# Persist training artifacts
joblib.dump(scaler,        os.path.join(MODELS_DIR, "scaler.joblib"))
joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.joblib"))
joblib.dump(X_test_sc,     os.path.join(MODELS_DIR, "X_test_scaled.joblib"))
joblib.dump(X_test,        os.path.join(MODELS_DIR, "X_test_raw.joblib"))
joblib.dump(y_test,        os.path.join(MODELS_DIR, "y_test.joblib"))

# ─────────────────────────────────────────────
#  STEP 4 – TRAIN MODELS
# ─────────────────────────────────────────────
print("\n[4/7] Training models …")
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

metrics_all   = {}
best_params   = {}
test_probs    = {}


def record_metrics(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    m = {
        "Accuracy":  round(accuracy_score(y_te, y_pred),   4),
        "Precision": round(precision_score(y_te, y_pred),  4),
        "Recall":    round(recall_score(y_te, y_pred),     4),
        "F1":        round(f1_score(y_te, y_pred),         4),
        "AUC-ROC":   round(roc_auc_score(y_te, y_prob),   4),
    }
    metrics_all[name] = m
    test_probs[name]  = y_prob.tolist()
    print(f"      {name:25s}  Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}  AUC={m['AUC-ROC']:.4f}")
    return model, y_prob


# ─── Logistic Regression ───────────────────
print("      [1/5] Logistic Regression …")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
record_metrics("Logistic Regression", lr, X_train_sc, X_test_sc, y_train, y_test)
best_params["Logistic Regression"] = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.joblib"))

# ─── Decision Tree ─────────────────────────
print("      [2/5] Decision Tree + GridSearchCV …")
dt_param_grid = {
    "max_depth":        [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
}
dt_gs = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    dt_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1,
)
dt_gs.fit(X_train_sc, y_train)
dt_best = dt_gs.best_estimator_
best_params["Decision Tree"] = dt_gs.best_params_
print(f"             Best params: {dt_gs.best_params_}")
record_metrics("Decision Tree", dt_best, X_train_sc, X_test_sc, y_train, y_test)
joblib.dump(dt_best, os.path.join(MODELS_DIR, "decision_tree.joblib"))

# ─── Random Forest ─────────────────────────
print("      [3/5] Random Forest + GridSearchCV …")
rf_param_grid = {
    "n_estimators":    [100, 200],
    "max_depth":       [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf":  [1, 2],
}
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    rf_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1,
)
rf_gs.fit(X_train_sc, y_train)
rf_best = rf_gs.best_estimator_
best_params["Random Forest"] = rf_gs.best_params_
print(f"             Best params: {rf_gs.best_params_}")
record_metrics("Random Forest", rf_best, X_train_sc, X_test_sc, y_train, y_test)
joblib.dump(rf_best, os.path.join(MODELS_DIR, "random_forest.joblib"))

# ─── XGBoost ───────────────────────────────
print("      [4/5] XGBoost + GridSearchCV …")
xgb_param_grid = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample":     [0.8, 1.0],
}
xgb_gs = GridSearchCV(
    xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0),
    xgb_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1,
)
xgb_gs.fit(X_train_sc, y_train)
xgb_best = xgb_gs.best_estimator_
best_params["XGBoost"] = xgb_gs.best_params_
print(f"             Best params: {xgb_gs.best_params_}")
record_metrics("XGBoost", xgb_best, X_train_sc, X_test_sc, y_train, y_test)
joblib.dump(xgb_best, os.path.join(MODELS_DIR, "xgboost.joblib"))

# ─── Neural Network (Keras MLP) ────────────
print("      [5/5] Neural Network (MLP, Keras) …")
input_dim = X_train_sc.shape[1]

mlp = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.1),
    layers.Dense(1, activation="sigmoid"),
])
mlp.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

early_stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
)
mlp.fit(
    X_train_sc.values, y_train.values,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0,
)

y_prob_mlp = mlp.predict(X_test_sc.values, verbose=0).flatten()
y_pred_mlp = (y_prob_mlp > 0.5).astype(int)
mlp_metrics = {
    "Accuracy":  round(accuracy_score(y_test, y_pred_mlp),  4),
    "Precision": round(precision_score(y_test, y_pred_mlp), 4),
    "Recall":    round(recall_score(y_test, y_pred_mlp),    4),
    "F1":        round(f1_score(y_test, y_pred_mlp),        4),
    "AUC-ROC":   round(roc_auc_score(y_test, y_prob_mlp),  4),
}
metrics_all["Neural Network"] = mlp_metrics
test_probs["Neural Network"]  = y_prob_mlp.tolist()
best_params["Neural Network"] = {
    "architecture": "128→64→32→1",
    "dropout": "0.3, 0.2, 0.1",
    "optimizer": "adam",
    "batch_size": 32,
}
print(f"      {'Neural Network':25s}  Acc={mlp_metrics['Accuracy']:.4f}  "
      f"F1={mlp_metrics['F1']:.4f}  AUC={mlp_metrics['AUC-ROC']:.4f}")
mlp.save(os.path.join(MODELS_DIR, "mlp.keras"))

# Persist metrics & params
joblib.dump(metrics_all,  os.path.join(MODELS_DIR, "metrics.joblib"))
joblib.dump(best_params,  os.path.join(MODELS_DIR, "best_params.joblib"))
joblib.dump(test_probs,   os.path.join(MODELS_DIR, "test_probs.joblib"))

# ─────────────────────────────────────────────
#  STEP 5 – MODEL COMPARISON PLOTS
# ─────────────────────────────────────────────
print("\n[5/7] Generating model comparison plots …")

MODEL_COLORS = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c"]
model_names  = list(metrics_all.keys())
y_test_arr   = y_test.values

# ── ROC Curves ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
for i, name in enumerate(model_names):
    fpr, tpr, _ = roc_curve(y_test_arr, np.array(test_probs[name]))
    auc = metrics_all[name]["AUC-ROC"]
    ax.plot(fpr, tpr, color=MODEL_COLORS[i], linewidth=2.5,
            label=f"{name}  (AUC = {auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier")
ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate", fontsize=13)
ax.set_ylabel("True Positive Rate", fontsize=13)
ax.set_title("ROC Curves – All Models", fontsize=15, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
save_fig("09_roc_curves.png")

# ── Metrics Comparison Bar Chart ─────────────
metrics_df = pd.DataFrame(metrics_all).T
metric_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
x = np.arange(len(metric_cols))
width = 0.14
offsets = np.linspace(-2 * width, 2 * width, len(model_names))

fig, ax = plt.subplots(figsize=(15, 7))
for i, name in enumerate(model_names):
    vals = [metrics_all[name][m] for m in metric_cols]
    ax.bar(x + offsets[i], vals, width, label=name,
           color=MODEL_COLORS[i], alpha=0.85, edgecolor="white")
ax.set_xlabel("Metric", fontsize=13)
ax.set_ylabel("Score", fontsize=13)
ax.set_title("Model Comparison – All Metrics", fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metric_cols, fontsize=12)
ax.set_ylim(0, 1.15)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))
save_fig("10_model_comparison.png")

# ─────────────────────────────────────────────
#  STEP 6 – SHAP ANALYSIS
# ─────────────────────────────────────────────
print("\n[6/7] Running SHAP analysis …")

# Identify best tree model by AUC-ROC
tree_model_map = {
    "Decision Tree":   dt_best,
    "Random Forest":   rf_best,
    "XGBoost":         xgb_best,
}
best_tree_name  = max(tree_model_map, key=lambda k: metrics_all[k]["AUC-ROC"])
best_tree_model = tree_model_map[best_tree_name]
print(f"      Best tree model: {best_tree_name}")

# Use up to 500 test samples for SHAP (speed)
shap_n       = min(500, len(X_test_sc))
X_shap       = X_test_sc.iloc[:shap_n]
explainer    = shap.TreeExplainer(best_tree_model)
shap_values  = explainer.shap_values(X_shap)

# Handle RF (list output) vs XGB/DT (array output)
if isinstance(shap_values, list):
    sv_class1 = shap_values[1]                          # class 1 (Churn)
    ev_class1 = explainer.expected_value[1]
else:
    sv_class1 = shap_values
    ev_class1 = (explainer.expected_value[1]
                 if isinstance(explainer.expected_value, (list, np.ndarray))
                 else explainer.expected_value)

joblib.dump(explainer,  os.path.join(MODELS_DIR, "shap_explainer.joblib"))
np.save(os.path.join(MODELS_DIR, "shap_values.npy"),          sv_class1)
np.save(os.path.join(MODELS_DIR, "shap_expected_value.npy"),  np.array([ev_class1]))
joblib.dump(X_shap,     os.path.join(MODELS_DIR, "X_shap.joblib"))
joblib.dump({"best_tree_name": best_tree_name},
            os.path.join(MODELS_DIR, "shap_info.joblib"))

# ── SHAP Beeswarm Summary Plot ───────────────
fig, _ = plt.subplots(figsize=(12, 9))
shap.summary_plot(sv_class1, X_shap, feature_names=feature_names, show=False)
plt.title(f"SHAP Beeswarm Plot – {best_tree_name}", fontsize=14, fontweight="bold")
save_fig("11_shap_beeswarm.png")

# ── SHAP Bar Plot ─────────────────────────────
fig, _ = plt.subplots(figsize=(12, 9))
shap.summary_plot(sv_class1, X_shap, feature_names=feature_names,
                  plot_type="bar", show=False)
plt.title(f"SHAP Mean |SHAP Value| – {best_tree_name}", fontsize=14, fontweight="bold")
save_fig("12_shap_bar.png")

# ── SHAP Waterfall (first test sample) ───────
shap_exp = shap.Explanation(
    values=sv_class1[0],
    base_values=ev_class1,
    data=X_shap.iloc[0].values,
    feature_names=feature_names,
)
plt.figure(figsize=(12, 8))
shap.plots.waterfall(shap_exp, max_display=15, show=False)
plt.title(
    f"SHAP Waterfall – {best_tree_name} (Sample 0)",
    fontsize=14, fontweight="bold",
)
save_fig("13_shap_waterfall.png")

print(f"      SHAP plots saved.")

# ─────────────────────────────────────────────
#  STEP 7 – SUMMARY
# ─────────────────────────────────────────────
print("\n[7/7] Pipeline complete!")
print("=" * 60)
print("\nModel Performance Summary:")
print(metrics_df[metric_cols].to_string(float_format="%.4f"))
print(f"\nBest model by AUC-ROC: {metrics_df['AUC-ROC'].idxmax()}"
      f"  ({metrics_df['AUC-ROC'].max():.4f})")
print(f"\nArtifacts saved:")
print(f"  Models  -> {MODELS_DIR}/")
print(f"  Plots   -> {PLOTS_DIR}/")
print("\nRun the app:")
print("  streamlit run app.py")
print("=" * 60)
