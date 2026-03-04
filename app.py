"""
Telco Customer Churn Analytics — Portfolio Edition
===================================================
World-class dark-theme Streamlit dashboard.
Run `python train_models.py` first, then `streamlit run app.py`.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib
import streamlit as st
from sklearn.metrics import roc_curve

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from tensorflow import keras as _keras
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analytics | Portfolio",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  DESIGN SYSTEM
# ─────────────────────────────────────────────
C = {
    "bg":      "#0d1117",  "card":    "#161b27",  "card2":   "#1c2333",
    "border":  "rgba(255,255,255,0.06)",
    "purple":  "#7c3aed",  "cyan":    "#06b6d4",  "green":   "#10b981",
    "red":     "#ef4444",  "amber":   "#f59e0b",
    "text":    "#e2e8f0",  "muted":   "#64748b",
}
MODEL_PALETTE = {
    "Logistic Regression": "#06b6d4",
    "Decision Tree":       "#f59e0b",
    "Random Forest":       "#10b981",
    "XGBoost":             "#7c3aed",
    "Neural Network":      "#ef4444",
}
CHURN_COLORS = {"No": "#10b981", "Yes": "#ef4444"}
METRIC_COLS  = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLS  = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
CAT_COLS     = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
DATA_PATH  = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = "models"
PLOTS_DIR  = "plots"
MODEL_FILES = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
    "Decision Tree":       os.path.join(MODELS_DIR, "decision_tree.joblib"),
    "Random Forest":       os.path.join(MODELS_DIR, "random_forest.joblib"),
    "XGBoost":             os.path.join(MODELS_DIR, "xgboost.joblib"),
    "Neural Network":      os.path.join(MODELS_DIR, "mlp.keras"),
}
TREE_MODELS = {"Decision Tree", "Random Forest", "XGBoost"}

# ─────────────────────────────────────────────
#  CSS  (full dark design system)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
.stApp { background:#0d1117 !important; }
.stApp > header { background:transparent !important; }
.block-container { padding:2rem 3rem !important; max-width:1400px !important; }
section[data-testid="stSidebar"] { background:#161b27 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:#161b27; border-radius:12px; padding:6px; gap:4px;
    border:1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius:8px; color:#94a3b8 !important;
    font-weight:500; font-size:0.9rem; padding:8px 20px;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#7c3aed,#5b21b6) !important;
    color:#fff !important;
    box-shadow:0 4px 12px rgba(124,58,237,0.4) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top:2rem !important; }

/* ── Native Metrics ── */
[data-testid="stMetric"] {
    background:#161b27; border-radius:12px; padding:16px 20px;
    border:1px solid rgba(255,255,255,0.06);
}
[data-testid="stMetricLabel"] { color:#94a3b8 !important; font-size:0.8rem !important; }
[data-testid="stMetricValue"] { color:#e2e8f0 !important; font-size:1.5rem !important; font-weight:700 !important; }

/* ── Inputs ── */
[data-testid="stSelectbox"] label, [data-testid="stSlider"] label {
    color:#94a3b8 !important; font-size:0.82rem !important; font-weight:600 !important;
    text-transform:uppercase; letter-spacing:0.04em;
}
.stSelectbox [data-baseweb="select"] > div { background:#1c2333 !important; border-radius:8px !important; }

/* ── Button ── */
.stButton > button {
    background:linear-gradient(135deg,#7c3aed,#5b21b6) !important;
    color:#fff !important; border:none !important; border-radius:10px !important;
    font-weight:700 !important; font-size:1.05rem !important; padding:14px 28px !important;
    box-shadow:0 4px 20px rgba(124,58,237,0.45) !important; width:100% !important;
    letter-spacing:0.02em;
}
.stButton > button:hover { box-shadow:0 6px 28px rgba(124,58,237,0.6) !important; }

/* ── Expander ── */
[data-testid="stExpander"] { background:#161b27; border-radius:12px; border:1px solid rgba(255,255,255,0.06) !important; }
details summary p { color:#e2e8f0 !important; font-weight:600 !important; }

/* ── Divider ── */
hr { border-color:rgba(255,255,255,0.06) !important; margin:2rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#0d1117; }
::-webkit-scrollbar-thumb { background:#2d3748; border-radius:3px; }

/* ══ Custom Components ══ */

/* Hero */
.hero {
    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 45%,#0f3460 100%);
    border-radius:20px; padding:52px 48px;
    border:1px solid rgba(124,58,237,0.25);
    box-shadow:0 8px 40px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom:2.5rem; position:relative; overflow:hidden;
}
.hero::before {
    content:""; position:absolute; top:-60px; right:-60px;
    width:280px; height:280px; border-radius:50%;
    background:radial-gradient(circle,rgba(124,58,237,0.15),transparent 70%);
}
.hero-title { color:#fff; font-size:2.8rem; font-weight:900; margin:0 0 12px 0; letter-spacing:-0.5px; line-height:1.1; }
.hero-sub   { color:#94a3b8; font-size:1.05rem; margin:0 0 24px 0; }
.hero-badge {
    display:inline-block; background:rgba(124,58,237,0.2); color:#a78bfa;
    border:1px solid rgba(124,58,237,0.4); border-radius:20px;
    padding:5px 14px; font-size:0.8rem; font-weight:600; margin-right:8px;
}

/* KPI Cards */
.kpi-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:16px; margin:1.5rem 0; }
.kpi-card {
    background:#161b27; border-radius:16px; padding:22px 18px;
    border:1px solid rgba(255,255,255,0.06);
    border-top:3px solid var(--kpi-color,#7c3aed);
    box-shadow:0 4px 20px rgba(0,0,0,0.3);
    text-align:center; transition:transform 0.2s;
}
.kpi-card:hover { transform:translateY(-3px); box-shadow:0 8px 30px rgba(0,0,0,0.4); }
.kpi-icon  { font-size:1.8rem; margin-bottom:8px; }
.kpi-value { font-size:1.9rem; font-weight:800; color:#e2e8f0; margin:4px 0; }
.kpi-label { color:#64748b; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; }
.kpi-delta { font-size:0.82rem; margin-top:6px; font-weight:600; }

/* Context Cards */
.ctx-card {
    background:#161b27; border-radius:14px; padding:24px 26px;
    border:1px solid rgba(255,255,255,0.06);
    box-shadow:0 2px 12px rgba(0,0,0,0.25); height:100%;
}
.ctx-card h3 { color:#e2e8f0; font-size:1.05rem; font-weight:700; margin:0 0 12px 0; }
.ctx-card p  { color:#94a3b8; font-size:0.9rem; line-height:1.7; margin:0; }

/* Insight Pills */
.pill {
    background:#1c2333; border-radius:10px; padding:13px 16px;
    border-left:3px solid #7c3aed; margin-bottom:10px;
    color:#e2e8f0; font-size:0.88rem; line-height:1.55;
}
.pill.green  { border-color:#10b981; }
.pill.cyan   { border-color:#06b6d4; }
.pill.amber  { border-color:#f59e0b; }
.pill.red    { border-color:#ef4444; }
.pill-icon   { font-size:1rem; margin-right:6px; }

/* Section Header */
.sec-hdr {
    color:#e2e8f0; font-size:1.2rem; font-weight:700;
    margin:2rem 0 1.2rem 0; padding-bottom:10px;
    border-bottom:2px solid rgba(124,58,237,0.35);
    display:flex; align-items:center; gap:8px;
}

/* Chart Wrapper */
.chart-wrap { background:#161b27; border-radius:16px; padding:4px; border:1px solid rgba(255,255,255,0.05); }
.chart-caption {
    background:#1c2333; border-radius:0 0 12px 12px;
    padding:12px 18px; color:#94a3b8; font-size:0.85rem; line-height:1.55;
    border-top:1px solid rgba(255,255,255,0.04); margin-top:-6px;
}

/* Model Explain Cards */
.model-card {
    background:#161b27; border-radius:14px; padding:20px 18px;
    border:1px solid rgba(255,255,255,0.06);
    border-top:3px solid var(--mc,#7c3aed); height:100%;
}
.model-card h4 { color:#e2e8f0; font-size:0.95rem; font-weight:700; margin:0 0 6px 0; }
.model-card p  { color:#94a3b8; font-size:0.82rem; line-height:1.55; margin:0; }
.model-tag {
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-size:0.72rem; font-weight:700; margin-bottom:8px;
    border:1px solid currentColor;
}

/* Param Cards */
.param-card {
    background:#1c2333; border-radius:12px; padding:16px 18px;
    border:1px solid rgba(255,255,255,0.05); height:100%;
}
.param-card h5 { color:#a78bfa; font-size:0.88rem; font-weight:700; margin:0 0 10px 0; }
.param-row { display:flex; justify-content:space-between; padding:4px 0;
             border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.82rem; }
.param-key { color:#64748b; }
.param-val { color:#e2e8f0; font-weight:600; font-family:monospace; }

/* Prediction Result */
.pred-card {
    border-radius:20px; padding:36px 32px; text-align:center;
    border:2px solid; box-shadow:0 8px 40px rgba(0,0,0,0.45); margin-bottom:1rem;
}
.pred-card.churn    { background:rgba(239,68,68,0.08);  border-color:#ef4444; }
.pred-card.no-churn { background:rgba(16,185,129,0.08); border-color:#10b981; }
.pred-verdict { font-size:2.8rem; font-weight:900; margin:0; }
.pred-verdict.churn    { color:#ef4444; }
.pred-verdict.no-churn { color:#10b981; }
.pred-sub { color:#94a3b8; font-size:0.95rem; margin:8px 0 0 0; }

/* Risk Badge */
.risk-badge {
    display:inline-block; padding:6px 18px; border-radius:20px;
    font-size:0.88rem; font-weight:700; margin:10px 0;
}
.risk-low      { background:rgba(16,185,129,0.15);  color:#10b981; border:1px solid rgba(16,185,129,0.3); }
.risk-moderate { background:rgba(245,158,11,0.15);  color:#f59e0b; border:1px solid rgba(245,158,11,0.3); }
.risk-high     { background:rgba(239,68,68,0.15);   color:#ef4444; border:1px solid rgba(239,68,68,0.3); }
.risk-critical { background:rgba(239,68,68,0.25);   color:#ef4444; border:1px solid #ef4444; }

/* Recommendation Box */
.rec-box {
    border-radius:14px; padding:22px 24px;
    border-left:5px solid; margin-top:1.2rem;
    border:1px solid rgba(255,255,255,0.05);
    border-left-width:5px;
}
.rec-box h4 { font-size:1rem; font-weight:700; margin:0 0 8px 0; }
.rec-box p  { color:#94a3b8; font-size:0.88rem; line-height:1.6; margin:0; }

/* Feature Group Labels */
.feat-group {
    color:#7c3aed; font-size:0.75rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.08em;
    margin:1.2rem 0 0.4rem 0; padding-bottom:4px;
    border-bottom:1px solid rgba(124,58,237,0.2);
}

/* SHAP Caption */
.shap-explain {
    background:#1c2333; border-radius:12px; padding:14px 18px;
    border-left:4px solid #7c3aed; color:#94a3b8;
    font-size:0.87rem; line-height:1.6; margin-top:8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PLOTLY DARK THEME HELPER
# ─────────────────────────────────────────────
def dark_layout(title="", height=440, legend_pos=None):
    base = dict(
        title=dict(text=title, font=dict(color="#e2e8f0", size=14, family="Inter,sans-serif"), x=0.01),
        plot_bgcolor="#161b27", paper_bgcolor="#161b27",
        font=dict(color="#94a3b8", family="Inter,sans-serif", size=12),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)",
                   tickfont=dict(color="#64748b"), showline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)",
                   tickfont=dict(color="#64748b"), showline=False),
        height=height, margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                    bordercolor="rgba(255,255,255,0.05)", borderwidth=1),
        hoverlabel=dict(bgcolor="#1c2333", bordercolor="#2d3748",
                        font=dict(color="#e2e8f0", size=13)),
    )
    if legend_pos:
        base["legend"].update(legend_pos)
    return base


# ─────────────────────────────────────────────
#  MATPLOTLIB DARK THEME HELPER
# ─────────────────────────────────────────────
def apply_dark_mpl():
    plt.rcParams.update({
        "axes.facecolor":    "#161b27",
        "figure.facecolor":  "#161b27",
        "axes.edgecolor":    "#2d3748",
        "text.color":        "#e2e8f0",
        "axes.labelcolor":   "#94a3b8",
        "xtick.color":       "#94a3b8",
        "ytick.color":       "#94a3b8",
        "axes.titlecolor":   "#e2e8f0",
        "grid.color":        "#1c2333",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# ─────────────────────────────────────────────
#  PREPROCESSING (mirrors train_models.py)
# ─────────────────────────────────────────────
def preprocess_single_input(user_dict, feature_names, scaler):
    df = pd.DataFrame([user_dict])
    df["gender"] = (df["gender"] == "Male").astype(int)
    for col in BINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return df


# ─────────────────────────────────────────────
#  SHAP WATERFALL HELPER
# ─────────────────────────────────────────────
def compute_shap_waterfall(model, X_proc, feature_names, model_name, max_display=15):
    apply_dark_mpl()
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_proc)
    ev_raw = explainer.expected_value

    if isinstance(shap_vals, list):
        # Old SHAP format: list of arrays per class
        sv = shap_vals[1][0]
        ev = float(ev_raw[1])
    elif shap_vals.ndim == 3:
        # New SHAP format for multi-output RF/DT: (n_samples, n_features, n_classes)
        sv = shap_vals[0, :, 1]
        ev = float(ev_raw[1]) if hasattr(ev_raw, "__len__") else float(ev_raw)
    elif shap_vals.ndim == 2 and shap_vals.shape[0] == len(feature_names):
        # (n_features, n_classes) for a single sample — RF/DT in some SHAP versions
        sv = shap_vals[:, 1]
        ev = float(ev_raw[1]) if hasattr(ev_raw, "__len__") else float(ev_raw)
    else:
        # Single-output (XGBoost binary): (n_samples, n_features)
        sv = shap_vals[0]
        ev = float(ev_raw[0]) if hasattr(ev_raw, "__len__") else float(ev_raw)

    exp = shap.Explanation(
        values=sv, base_values=ev,
        data=X_proc.iloc[0].values, feature_names=list(feature_names),
    )
    plt.figure(figsize=(12, 7), facecolor="#161b27")
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    plt.gcf().patch.set_facecolor("#161b27")
    fig = plt.gcf()
    return fig


# ─────────────────────────────────────────────
#  RECOMMENDATION HELPER
# ─────────────────────────────────────────────
def get_recommendation(prob):
    if prob < 0.30:
        return {
            "risk": "Low Risk", "cls": "risk-low", "icon": "✅",
            "color": "#10b981", "bg": "rgba(16,185,129,0.06)",
            "title": "This customer is likely to stay.",
            "action": "Maintain the relationship through a loyalty rewards program or periodic satisfaction check-in. "
                      "Consider a small incentive (bonus data, discounted add-on) to reinforce long-term commitment.",
        }
    elif prob < 0.50:
        return {
            "risk": "Moderate Risk", "cls": "risk-moderate", "icon": "⚠️",
            "color": "#f59e0b", "bg": "rgba(245,158,11,0.06)",
            "title": "Monitor this customer closely.",
            "action": "A proactive check-in or minor discount could prevent churn. Review their current plan "
                      "for upgrade opportunities and flag the account for the retention team's next outreach cycle.",
        }
    elif prob < 0.70:
        return {
            "risk": "High Risk", "cls": "risk-high", "icon": "🚨",
            "color": "#ef4444", "bg": "rgba(239,68,68,0.06)",
            "title": "Intervention is recommended.",
            "action": "Reach out with a personalized retention offer — contract upgrade, discounted bundle, "
                      "or a free service add-on. Time-sensitive: waiting increases the likelihood of cancellation.",
        }
    else:
        return {
            "risk": "Critical Risk", "cls": "risk-critical", "icon": "🔴",
            "color": "#ef4444", "bg": "rgba(239,68,68,0.1)",
            "title": "Immediate action required!",
            "action": "Escalate to retention specialists today. Consider a significant discount (15–25%), "
                      "a free plan upgrade, dedicated account support, or a loyalty lock-in offer. "
                      "Every day of delay significantly raises the probability of losing this customer.",
        }


# ─────────────────────────────────────────────
#  LOADERS  (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_artifacts():
    a = {}
    a["metrics"]       = joblib.load(os.path.join(MODELS_DIR, "metrics.joblib"))
    a["best_params"]   = joblib.load(os.path.join(MODELS_DIR, "best_params.joblib"))
    a["test_probs"]    = joblib.load(os.path.join(MODELS_DIR, "test_probs.joblib"))
    a["feature_names"] = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    a["scaler"]        = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    a["y_test"]        = joblib.load(os.path.join(MODELS_DIR, "y_test.joblib"))
    a["X_test_scaled"] = joblib.load(os.path.join(MODELS_DIR, "X_test_scaled.joblib"))
    a["shap_values"]   = np.load(os.path.join(MODELS_DIR, "shap_values.npy"))
    a["shap_ev"]       = np.load(os.path.join(MODELS_DIR, "shap_expected_value.npy"))
    a["X_shap"]        = joblib.load(os.path.join(MODELS_DIR, "X_shap.joblib"))
    a["shap_info"]     = joblib.load(os.path.join(MODELS_DIR, "shap_info.joblib"))
    return a


@st.cache_resource
def load_sklearn_model(path):
    return joblib.load(path)


@st.cache_resource
def load_keras_model(path):
    if not KERAS_AVAILABLE:
        return None
    from tensorflow import keras
    return keras.models.load_model(path)


def load_all_models():
    out = {}
    for name, path in MODEL_FILES.items():
        try:
            out[name] = load_keras_model(path) if name == "Neural Network" else load_sklearn_model(path)
        except Exception:
            out[name] = None
    return out


# ─────────────────────────────────────────────
#  GUARD
# ─────────────────────────────────────────────
if not os.path.exists(os.path.join(MODELS_DIR, "metrics.joblib")):
    st.markdown("""
    <div style="background:rgba(239,68,68,0.1);border:1px solid #ef4444;border-radius:14px;padding:28px 32px;margin-top:2rem;">
    <h3 style="color:#ef4444;margin:0 0 8px 0;">⚠️ Models Not Found</h3>
    <p style="color:#94a3b8;margin:0;">Run the training script first:</p>
    <pre style="background:#1c2333;color:#a78bfa;padding:12px 16px;border-radius:8px;margin-top:12px;">python train_models.py</pre>
    </div>""", unsafe_allow_html=True)
    st.stop()

# Load
df_raw        = load_raw_data()
artifacts     = load_artifacts()
models        = load_all_models()
metrics_all   = artifacts["metrics"]
best_params   = artifacts["best_params"]
test_probs    = artifacts["test_probs"]
feature_names = artifacts["feature_names"]
scaler        = artifacts["scaler"]
y_test        = artifacts["y_test"]
shap_values   = artifacts["shap_values"]
shap_ev       = float(np.ravel(artifacts["shap_ev"])[0])
X_shap        = artifacts["X_shap"]
best_tree_name = artifacts["shap_info"]["best_tree_name"]
metrics_df    = pd.DataFrame(metrics_all).T

# Pre-compute derived stats
df_raw["_TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
n_customers     = len(df_raw)
n_churned       = (df_raw["Churn"] == "Yes").sum()
churn_rate      = n_churned / n_customers
avg_monthly     = df_raw["MonthlyCharges"].mean()
revenue_at_risk = df_raw.loc[df_raw["Churn"] == "Yes", "MonthlyCharges"].sum()
best_model_name = metrics_df["AUC-ROC"].idxmax()
best_auc        = metrics_df["AUC-ROC"].max()

# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <p class="hero-title">📡 Telco Customer Churn Analytics</p>
  <p class="hero-sub">End-to-end machine learning pipeline — from raw data to real-time predictions.</p>
  <span class="hero-badge">7,043 Customers</span>
  <span class="hero-badge">5 ML Models</span>
  <span class="hero-badge">Best AUC {best_auc:.4f}</span>
  <span class="hero-badge">SHAP Explainability</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠  Executive Summary",
    "📊  Descriptive Analytics",
    "🤖  Model Performance",
    "🔮  Interactive Prediction",
])


# ══════════════════════════════════════════════════
#  TAB 1 – EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════
with tab1:

    # ── What is churn? / Why it matters ────
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("""
        <div class="ctx-card">
          <h3>📖 What is Customer Churn?</h3>
          <p>
            Customer churn happens when a subscriber <strong style="color:#e2e8f0">cancels their service</strong>
            and stops being a customer. In telecom, this means a customer leaving for a competitor,
            downgrading to nothing, or simply disconnecting.<br><br>
            Each lost customer represents not just lost revenue today, but the
            <strong style="color:#e2e8f0">entire future lifetime value</strong> of that relationship —
            often worth thousands of dollars.
          </p>
        </div>""", unsafe_allow_html=True)
    with col_r:
        st.markdown(f"""
        <div class="ctx-card">
          <h3>💼 Why Does This Matter?</h3>
          <p>
            Acquiring a new customer costs <strong style="color:#f59e0b">5–7× more</strong> than retaining
            an existing one. With a churn rate of <strong style="color:#ef4444">{churn_rate:.1%}</strong>,
            this telecom provider loses approximately
            <strong style="color:#ef4444">${revenue_at_risk:,.0f}/month</strong> in recurring revenue.<br><br>
            Predicting who is <em>about to</em> churn allows the business to act
            <strong style="color:#e2e8f0">before</strong> the customer leaves — through targeted discounts,
            contract upgrades, or proactive support.
          </p>
        </div>""", unsafe_allow_html=True)

    # ── KPI Cards ──────────────────────────
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")

    def kpi(col, icon, value, label, delta, delta_up, color):
        delta_color = color if delta_up else C["red"]
        col.markdown(f"""
        <div class="kpi-card" style="--kpi-color:{color}">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-delta" style="color:{delta_color}">{delta}</div>
        </div>""", unsafe_allow_html=True)

    kpi(k1, "👥", f"{n_customers:,}",    "Total Customers",        "IBM Telco Dataset",  True,  C["cyan"])
    kpi(k2, "📉", f"{churn_rate:.1%}",   "Churn Rate",             f"{n_churned:,} left", False, C["red"])
    kpi(k3, "💵", f"${avg_monthly:.2f}", "Avg Monthly Charge",     "Per customer",        True,  C["amber"])
    kpi(k4, "🔥", f"${revenue_at_risk/1000:.0f}K", "Monthly Rev. at Risk", "From churners", False, C["red"])
    kpi(k5, "🏆", f"{best_auc:.4f}",     "Best AUC-ROC",           best_model_name,       True,  C["purple"])

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Leaderboard + Insights ─────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="sec-hdr">🥇 Model Leaderboard</div>', unsafe_allow_html=True)
        lb_df = metrics_df[METRIC_COLS].sort_values("AUC-ROC", ascending=False).reset_index()
        lb_df.columns = ["Model"] + METRIC_COLS

        # Visual bar chart leaderboard
        fig_lb = go.Figure()
        for _, row in lb_df.iterrows():
            color = MODEL_PALETTE.get(row["Model"], C["purple"])
            fig_lb.add_trace(go.Bar(
                x=[row["AUC-ROC"]], y=[row["Model"]], orientation="h",
                marker=dict(color=color, line=dict(color="rgba(0,0,0,0)")),
                text=f'{row["AUC-ROC"]:.4f}', textposition="outside",
                textfont=dict(color="#e2e8f0", size=12),
                showlegend=False, name=row["Model"],
                hovertemplate=(
                    f"<b>{row['Model']}</b><br>"
                    f"AUC-ROC: {row['AUC-ROC']:.4f}<br>"
                    f"F1: {row['F1']:.4f}<br>"
                    f"Accuracy: {row['Accuracy']:.4f}<extra></extra>"
                ),
            ))
        fig_lb.update_layout(**dark_layout(height=310))
        fig_lb.update_layout(
            xaxis=dict(range=[0.7, 0.9], gridcolor="rgba(255,255,255,0.05)",
                       tickfont=dict(color="#64748b")),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#e2e8f0", size=12)),
            bargap=0.35,
        )
        fig_lb.add_vline(x=0.5, line_dash="dot", line_color="rgba(255,255,255,0.2)")
        st.plotly_chart(fig_lb, use_container_width=True)

    with right:
        st.markdown('<div class="sec-hdr">💡 Key Business Insights</div>', unsafe_allow_html=True)
        pills = [
            ("amber", "📋",
             "<strong>Month-to-month contracts</strong> churn at ~43% vs 11% (one-year) and 3% (two-year)."),
            ("red", "💸",
             "Churners pay <strong>~$15 more/month</strong> than retained customers on average."),
            ("cyan", "🌐",
             "<strong>Fiber optic</strong> users churn at ~41% — nearly 2× higher than DSL (~19%)."),
            ("red", "💳",
             "<strong>Electronic check</strong> payers have the highest churn rate (~45%)."),
            ("amber", "👴",
             "Senior citizens are <strong>~2× more likely</strong> to churn than non-seniors."),
            ("green", "📅",
             "<strong>50% of churners</strong> leave within their first 12 months of service."),
        ]
        for cls, icon, text in pills:
            st.markdown(
                f'<div class="pill {cls}"><span class="pill-icon">{icon}</span> {text}</div>',
                unsafe_allow_html=True,
            )

    # ── Best Model Breakdown ───────────────
    st.markdown("---")
    st.markdown(f'<div class="sec-hdr">🏆 {best_model_name} — Best Model Metrics</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(METRIC_COLS))
    for col, metric in zip(cols, METRIC_COLS):
        col.metric(metric, f"{metrics_df.loc[best_model_name, metric]:.4f}")


# ══════════════════════════════════════════════════
#  TAB 2 – DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════════
with tab2:

    # ── Dataset Overview ───────────────────
    with st.expander("📋  Dataset Overview & Summary Statistics", expanded=False):
        ov1, ov2 = st.columns(2)
        with ov1:
            st.markdown("<p style='color:#94a3b8;font-size:0.85rem;font-weight:600;'>COLUMNS & TYPES</p>",
                        unsafe_allow_html=True)
            info_df = pd.DataFrame({
                "Column": df_raw.columns,
                "Type":   df_raw.dtypes.astype(str).values,
                "Non-Null": df_raw.notnull().sum().values,
            })
            st.dataframe(info_df, use_container_width=True, height=310)
        with ov2:
            st.markdown("<p style='color:#94a3b8;font-size:0.85rem;font-weight:600;'>NUMERICAL STATISTICS</p>",
                        unsafe_allow_html=True)
            df_tmp = df_raw.copy()
            df_tmp["TotalCharges"] = pd.to_numeric(df_tmp["TotalCharges"], errors="coerce")
            st.dataframe(df_tmp.describe().T.round(2), use_container_width=True, height=310)

    st.markdown("---")

    # ── 1. Churn Distribution ──────────────
    st.markdown('<div class="sec-hdr">1️⃣ &nbsp;Target Distribution — Churn vs. No Churn</div>',
                unsafe_allow_html=True)
    d1, d2 = st.columns(2, gap="medium")
    vc = df_raw["Churn"].value_counts().reset_index()
    vc.columns = ["Churn", "Count"]
    vc["Pct"] = (vc["Count"] / n_customers * 100).round(1)

    with d1:
        fig = go.Figure(go.Bar(
            x=["No Churn", "Churned"], y=vc["Count"].tolist(),
            marker_color=[C["green"], C["red"]],
            text=[f"{v:,}<br>({p:.1f}%)" for v, p in zip(vc["Count"], vc["Pct"])],
            textposition="outside", textfont=dict(color="#e2e8f0", size=12),
        ))
        fig.update_layout(**dark_layout("Customer Count by Churn Status", 360))
        fig.update_yaxes(range=[0, vc["Count"].max() * 1.22])
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        fig = go.Figure(go.Pie(
            labels=["No Churn", "Churned"], values=vc["Count"].tolist(),
            marker=dict(colors=[C["green"], C["red"]],
                        line=dict(color="#0d1117", width=3)),
            hole=0.5, textinfo="percent+label",
            textfont=dict(size=13, color="#e2e8f0"),
        ))
        fig.update_layout(**dark_layout("Churn Proportion", 360))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="chart-caption">
      <strong style="color:#f59e0b">⚠️ Class Imbalance:</strong>
      ~73% No Churn vs ~27% Churned. This imbalance means raw accuracy can be misleading —
      <strong>F1-score and AUC-ROC</strong> are the right metrics to evaluate model quality here.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. Tenure ─────────────────────────
    st.markdown('<div class="sec-hdr">2️⃣ &nbsp;Tenure Distribution by Churn Status</div>',
                unsafe_allow_html=True)
    fig = px.histogram(
        df_raw, x="tenure", color="Churn", barmode="overlay",
        nbins=40, opacity=0.8,
        color_discrete_map={"No": C["green"], "Yes": C["red"]},
        labels={"tenure": "Tenure (Months)", "count": "Number of Customers"},
    )
    fig.update_layout(**dark_layout(height=400))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="chart-caption">
      <strong style="color:#ef4444">📉 Short tenure = high churn risk.</strong>
      Churned customers have a median tenure of ~10 months vs ~38 months for retained customers.
      The first year is the highest-risk window — <strong>early onboarding and engagement programs are critical</strong>.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 3. Monthly Charges ─────────────────
    st.markdown('<div class="sec-hdr">3️⃣ &nbsp;Monthly Charges by Churn Status</div>',
                unsafe_allow_html=True)
    fig = px.box(
        df_raw, x="Churn", y="MonthlyCharges", color="Churn",
        color_discrete_map={"No": C["green"], "Yes": C["red"]},
        notched=True, points="outliers",
        labels={"MonthlyCharges": "Monthly Charges ($)", "Churn": ""},
    )
    fig.update_layout(**dark_layout(height=420), showlegend=False)
    fig.update_xaxes(ticktext=["No Churn", "Churned"], tickvals=["No", "Yes"])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="chart-caption">
      <strong style="color:#f59e0b">💰 Higher bills, higher flight risk.</strong>
      Churned customers pay ~$74/month vs ~$61 for retained ones. Customers on premium plans need
      stronger value reinforcement — loyalty discounts, bundled perks, or contract incentives.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 4. Contract Type ───────────────────
    st.markdown('<div class="sec-hdr">4️⃣ &nbsp;Churn Rate by Contract Type</div>',
                unsafe_allow_html=True)
    ct = df_raw.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
    ct_tot = ct.groupby("Contract")["Count"].transform("sum")
    ct["Pct"] = (ct["Count"] / ct_tot * 100).round(1)
    fig = px.bar(
        ct, x="Contract", y="Pct", color="Churn", barmode="group",
        color_discrete_map={"No": C["green"], "Yes": C["red"]},
        text=ct["Pct"].apply(lambda x: f"{x:.1f}%"),
        labels={"Pct": "Percentage (%)", "Contract": "Contract Type"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**dark_layout(height=420), yaxis_range=[0, 105])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="chart-caption">
      <strong style="color:#ef4444">📋 Month-to-month = highest churn (~43%).</strong>
      One-year contracts drop that to ~11%, and two-year contracts to just ~3%.
      Offering incentives to <strong>upgrade short-term subscribers to annual plans</strong> is one of the highest-ROI retention actions.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 5. Internet Service ────────────────
    st.markdown('<div class="sec-hdr">5️⃣ &nbsp;Churn by Internet Service Type</div>',
                unsafe_allow_html=True)
    it = df_raw.groupby(["InternetService", "Churn"]).size().reset_index(name="Count")
    it_tot = it.groupby("InternetService")["Count"].transform("sum")
    it["Pct"] = (it["Count"] / it_tot * 100).round(1)
    fig = px.bar(
        it, x="InternetService", y="Pct", color="Churn", barmode="group",
        color_discrete_map={"No": C["green"], "Yes": C["red"]},
        text=it["Pct"].apply(lambda x: f"{x:.1f}%"),
        labels={"Pct": "Percentage (%)", "InternetService": "Internet Service"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**dark_layout(height=420), yaxis_range=[0, 90])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="chart-caption">
      <strong style="color:#06b6d4">🌐 Fiber optic users churn at ~41%.</strong>
      Despite being a premium product, fiber has the highest churn — likely due to high pricing and fierce competition.
      <strong>Targeted retention offers for fiber subscribers</strong> could meaningfully reduce revenue loss.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 6. Payment Method ─────────────────
    st.markdown('<div class="sec-hdr">6️⃣ &nbsp;Churn Rate by Payment Method</div>',
                unsafe_allow_html=True)
    pm = df_raw.groupby(["PaymentMethod", "Churn"]).size().reset_index(name="Count")
    pm_tot = pm.groupby("PaymentMethod")["Count"].transform("sum")
    pm["Pct"] = (pm["Count"] / pm_tot * 100).round(1)
    pm_churn = pm[pm["Churn"] == "Yes"].sort_values("Pct", ascending=True)
    bar_colors = [C["red"] if p > 30 else C["amber"] for p in pm_churn["Pct"]]
    fig = go.Figure(go.Bar(
        x=pm_churn["Pct"], y=pm_churn["PaymentMethod"],
        orientation="h", marker_color=bar_colors,
        text=pm_churn["Pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside", textfont=dict(color="#e2e8f0"),
    ))
    fig.update_layout(**dark_layout(height=340))
    fig.update_xaxes(range=[0, 60])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="chart-caption">
      <strong style="color:#ef4444">💳 Electronic check → highest churn (~45%).</strong>
      Customers on auto-payment (bank transfer / credit card) churn at roughly half that rate.
      <strong>Encouraging auto-pay enrollment</strong> is a low-cost, high-impact retention lever.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 7. Senior Citizen ─────────────────
    st.markdown('<div class="sec-hdr">7️⃣ &nbsp;Senior vs. Non-Senior Churn</div>',
                unsafe_allow_html=True)
    sc = df_raw.copy()
    sc["Segment"] = sc["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior Citizen"})
    sc_grp = sc.groupby(["Segment", "Churn"]).size().reset_index(name="Count")
    sc_tot = sc_grp.groupby("Segment")["Count"].transform("sum")
    sc_grp["Pct"] = (sc_grp["Count"] / sc_tot * 100).round(1)
    fig = px.bar(
        sc_grp, x="Segment", y="Pct", color="Churn", barmode="group",
        color_discrete_map={"No": C["green"], "Yes": C["red"]},
        text=sc_grp["Pct"].apply(lambda x: f"{x:.1f}%"),
        labels={"Pct": "Percentage (%)", "Segment": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**dark_layout(height=400), yaxis_range=[0, 95])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="chart-caption">
      <strong style="color:#f59e0b">👴 Senior citizens churn at ~42% vs ~24% for non-seniors.</strong>
      This demographic may face higher price sensitivity or tech adoption barriers.
      <strong>Dedicated senior support plans or simplified billing</strong> could improve retention in this segment.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 8. Correlation Heatmap ─────────────
    st.markdown('<div class="sec-hdr">8️⃣ &nbsp;Feature Correlation Heatmap</div>',
                unsafe_allow_html=True)
    heatmap_path = os.path.join(PLOTS_DIR, "08_correlation_heatmap.png")
    if os.path.exists(heatmap_path):
        st.image(heatmap_path, use_container_width=True)
    else:
        st.info("Heatmap not found — run train_models.py to generate it.")
    st.markdown("""
    <div class="chart-caption">
      <strong style="color:#a78bfa">🔗 Key correlations:</strong>
      Tenure is <em>negatively</em> correlated with churn (longer-tenured = more loyal).
      MonthlyCharges and TotalCharges are strongly correlated with each other.
      Contract type is one of the strongest predictors overall.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  TAB 3 – MODEL PERFORMANCE
# ══════════════════════════════════════════════════
with tab3:

    # ── Model Explanations ─────────────────
    st.markdown('<div class="sec-hdr">🧠 What Does Each Model Do?</div>',
                unsafe_allow_html=True)
    _purple = C["purple"]
    model_explain = {
        "Logistic Regression": (
            C["cyan"], "Baseline · Linear",
            "Finds a straight-line decision boundary between churners and non-churners. "
            "Think of it as a weighted checklist — each feature adds or subtracts points, "
            "and you churn if the total crosses a threshold. Fast, transparent, and surprisingly competitive."
        ),
        "Decision Tree": (
            C["amber"], "Tuned · Interpretable",
            "Asks a series of yes/no questions about the customer "
            "(e.g., 'Monthly contract? → Fiber optic? → Charges > $80?'). "
            "The path through the tree leads to a prediction. Easy to visualize and explain to non-technical stakeholders."
        ),
        "Random Forest": (
            C["green"], "Tuned · Ensemble",
            "Trains hundreds of decision trees on random subsets of data, then takes a majority vote. "
            "Much more robust than a single tree — errors from individual trees cancel out, "
            "delivering strong, stable predictions."
        ),
        "XGBoost": (
            C["purple"], "Tuned · State-of-the-Art",
            "Builds trees sequentially — each new tree specifically corrects the mistakes of the previous one. "
            "This 'gradient boosting' approach delivers top-tier performance and "
            f"is the <strong style='color:{_purple}'>best model</strong> in this project (AUC {best_auc:.4f})."
        ),
        "Neural Network": (
            C["red"], "Deep Learning · Keras",
            "Inspired by the brain: layers of neurons learn increasingly abstract patterns from the data. "
            "3 hidden layers (128→64→32) with dropout regularization and batch normalization. "
            "Powerful for complex patterns, but requires more data and tuning."
        ),
    }
    mc1, mc2, mc3, mc4, mc5 = st.columns(5, gap="small")
    for col, (name, (color, tag, desc)) in zip(
            [mc1, mc2, mc3, mc4, mc5], model_explain.items()):
        col.markdown(f"""
        <div class="model-card" style="--mc:{color}">
          <span class="model-tag" style="color:{color};border-color:{color};background:rgba(0,0,0,0.2)">{tag}</span>
          <h4>{name}</h4>
          <p>{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Metrics Table ──────────────────────
    st.markdown('<div class="sec-hdr">📊 Metrics Comparison</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;font-size:0.85rem;'>🟢 Green = best per metric &nbsp;|&nbsp; 🔴 Red = worst per metric</p>",
                unsafe_allow_html=True)
    disp = metrics_df[METRIC_COLS].sort_values("AUC-ROC", ascending=False)
    styled = disp.style \
        .format("{:.4f}") \
        .highlight_max(subset=METRIC_COLS, color="#1a3a2a") \
        .highlight_min(subset=METRIC_COLS, color="#3a1a1a") \
        .set_properties(**{"background-color": "#161b27", "color": "#e2e8f0",
                           "border": "1px solid rgba(255,255,255,0.05)"})
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")

    # ── ROC Curves ─────────────────────────
    st.markdown('<div class="sec-hdr">📈 ROC Curves — All Models</div>', unsafe_allow_html=True)
    y_test_arr = np.array(y_test)
    fig_roc = go.Figure()
    for name, probs in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test_arr, np.array(probs))
        auc  = metrics_all[name]["AUC-ROC"]
        col  = MODEL_PALETTE.get(name, C["purple"])
        bold = dict(width=3.5) if name == best_model_name else dict(width=2)
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name}  (AUC = {auc:.4f})",
            line=dict(color=col, **bold),
            hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
        line=dict(color="#2d3748", width=1.5, dash="dot"), showlegend=True,
    ))
    fig_roc.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                      fillcolor="rgba(124,58,237,0.02)", line_width=0)
    fig_roc.update_layout(**dark_layout("ROC Curves — Receiver Operating Characteristic", 500,
                      legend_pos=dict(x=0.55, y=0.1)))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1.02],
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.markdown("""
    <div class="chart-caption">
      The ROC curve shows the <strong>trade-off between catching true churners (TPR) and
      falsely flagging loyal customers (FPR)</strong>. A model with AUC = 1.0 is perfect;
      AUC = 0.5 is no better than random. <strong>XGBoost leads at 0.8457</strong>.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Metrics Bar Chart ──────────────────
    st.markdown('<div class="sec-hdr">📉 Metric-by-Metric Breakdown</div>', unsafe_allow_html=True)
    fig_bar = go.Figure()
    for name in metrics_all:
        color = MODEL_PALETTE.get(name, C["purple"])
        fig_bar.add_trace(go.Bar(
            name=name, x=METRIC_COLS,
            y=[metrics_all[name][m] for m in METRIC_COLS],
            marker=dict(color=color, line=dict(color="rgba(0,0,0,0)")),
            opacity=0.9,
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig_bar.update_layout(
        **dark_layout("All Models — All Metrics", 460,
                      legend_pos=dict(orientation="h", y=-0.2)),
        barmode="group",
        yaxis_range=[0, 1.12],
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Best Hyperparameters ───────────────
    st.markdown('<div class="sec-hdr">⚙️ Best Hyperparameters (GridSearchCV)</div>',
                unsafe_allow_html=True)
    p_cols = st.columns(len(best_params), gap="small")
    for col, (name, params) in zip(p_cols, best_params.items()):
        color = MODEL_PALETTE.get(name, C["purple"])
        rows_html = "".join(
            f'<div class="param-row"><span class="param-key">{k}</span>'
            f'<span class="param-val">{v}</span></div>'
            for k, v in params.items()
        )
        col.markdown(f"""
        <div class="param-card" style="border-top:2px solid {color}">
          <h5 style="color:{color}">{name}</h5>
          {rows_html}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── SHAP Analysis ──────────────────────
    st.markdown(f'<div class="sec-hdr">🔍 SHAP Explainability — {best_tree_name}</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="ctx-card" style="margin-bottom:1.2rem">
      <h3>📖 What is SHAP?</h3>
      <p>
        SHAP (SHapley Additive exPlanations) answers the question:
        <em>"Why did the model make <strong>this specific prediction</strong>?"</em><br><br>
        It assigns each feature a contribution score — <strong style="color:{C['red']}">positive = pushes toward churn</strong>,
        <strong style="color:{C['cyan']}">negative = pushes away from churn</strong>.
        The scores are grounded in game theory and are mathematically guaranteed to be fair and consistent.
      </p>
    </div>""", unsafe_allow_html=True)

    sh1, sh2 = st.columns(2, gap="medium")
    with sh1:
        p = os.path.join(PLOTS_DIR, "11_shap_beeswarm.png")
        if os.path.exists(p):
            st.image(p, caption=f"Beeswarm: each dot = one test customer", use_container_width=True)
    with sh2:
        p = os.path.join(PLOTS_DIR, "12_shap_bar.png")
        if os.path.exists(p):
            st.image(p, caption="Mean |SHAP|: overall feature importance", use_container_width=True)

    p = os.path.join(PLOTS_DIR, "13_shap_waterfall.png")
    if os.path.exists(p):
        st.image(p, caption="Waterfall: how one prediction was built up from the base rate",
                 use_container_width=True)

    st.markdown(f"""
    <div class="chart-caption">
      <strong style="color:#a78bfa">🏆 Top churn drivers (from SHAP):</strong>
      <strong>Contract_Month-to-month</strong> (biggest risk factor) →
      <strong>low tenure</strong> → <strong>InternetService_Fiber optic</strong> →
      <strong>high MonthlyCharges</strong> → <strong>no OnlineSecurity</strong>.
      These five features alone explain the vast majority of {best_tree_name}'s predictions.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  TAB 4 – INTERACTIVE PREDICTION
# ══════════════════════════════════════════════════
with tab4:

    st.markdown("""
    <div class="ctx-card" style="margin-bottom:1.5rem">
      <h3>🔮 Real-Time Churn Prediction</h3>
      <p>
        Build a customer profile using the controls below, choose your model, and get an
        <strong>instant churn probability</strong> with a risk rating and actionable recommendation.
        For tree-based models, a <strong>SHAP waterfall</strong> explains exactly why the model
        made that prediction — feature by feature.
      </p>
    </div>""", unsafe_allow_html=True)

    # ── Model Selector ─────────────────────
    available_models = [
        n for n in MODEL_FILES if not (n == "Neural Network" and not KERAS_AVAILABLE)
    ]
    sel_model = st.selectbox(
        "🤖  Select Prediction Model",
        options=available_models,
        index=min(2, len(available_models) - 1),
        help="SHAP waterfall available for Decision Tree, Random Forest, XGBoost."
             + ("" if KERAS_AVAILABLE else "  |  Neural Network unavailable (TensorFlow not installed)."),
    )
    mc = MODEL_PALETTE.get(sel_model, C["purple"])
    st.markdown(
        f'<p style="color:{mc};font-size:0.82rem;font-weight:600;margin-top:-8px;">'
        f'Selected: {sel_model}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Feature Inputs ─────────────────────
    st.markdown('<div class="sec-hdr">⚙️ Customer Profile</div>', unsafe_allow_html=True)

    st.markdown('<p class="feat-group">👤 Demographics</p>', unsafe_allow_html=True)
    g1, g2, g3, g4 = st.columns(4)
    with g1: gender     = st.selectbox("Gender",         ["Female", "Male"])
    with g2: senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
    with g3: partner    = st.selectbox("Partner",        ["Yes", "No"])
    with g4: dependents = st.selectbox("Dependents",     ["Yes", "No"])

    st.markdown('<p class="feat-group">📞 Services</p>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1: phone_service   = st.selectbox("Phone Service",    ["Yes", "No"])
    with s2: multiple_lines  = st.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
    with s3: internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with s4: online_security  = st.selectbox("Online Security",  ["No", "Yes", "No internet service"])

    s5, s6, s7, s8 = st.columns(4)
    with s5: online_backup    = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
    with s6: device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with s7: tech_support     = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
    with s8: streaming_tv     = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])

    st.markdown('<p class="feat-group">📄 Account</p>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    with a1: streaming_movies  = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])
    with a2: contract          = st.selectbox("Contract",          ["Month-to-month", "One year", "Two year"])
    with a3: paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    with a4: payment_method    = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    st.markdown('<p class="feat-group">💰 Charges</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: tenure         = st.slider("Tenure (months)",     0,    72,   12,   step=1)
    with c2: monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    with c3: total_charges   = st.slider("Total Charges ($)",   18.0, 8700.0,
                                         float(monthly_charges * tenure if tenure else 65.0), step=1.0)

    user_input = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }

    st.markdown("---")
    predict_btn = st.button("🔮  Predict Churn Probability", type="primary", use_container_width=True)

    if predict_btn:
        try:
            X_proc = preprocess_single_input(user_input, feature_names, scaler)
            model  = models[sel_model]

            if sel_model == "Neural Network":
                if model is None:
                    st.error("Neural Network unavailable — TensorFlow not installed in this environment.")
                    st.stop()
                prob = float(model.predict(X_proc.values, verbose=0)[0][0])
            else:
                prob = float(model.predict_proba(X_proc)[0][1])

            pred = int(prob >= 0.5)
            rec  = get_recommendation(prob)
            verdict_cls = "churn" if pred == 1 else "no-churn"
            verdict_txt = "⚠️ WILL CHURN"  if pred == 1 else "✅ WILL STAY"

            st.markdown("---")

            # ── Result Cards ─────────────────
            r1, r2, r3 = st.columns([1.2, 1.8, 1.2], gap="large")

            with r1:
                st.markdown(f"""
                <div class="pred-card {verdict_cls}">
                  <p class="pred-sub">Prediction</p>
                  <p class="pred-verdict {verdict_cls}">{verdict_txt}</p>
                  <p class="pred-sub" style="margin-top:12px">Using {sel_model}</p>
                  <div class="risk-badge {rec['cls']}" style="margin-top:12px">{rec['icon']} {rec['risk']}</div>
                </div>""", unsafe_allow_html=True)

            with r2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number=dict(suffix="%", font=dict(size=36, color="#e2e8f0")),
                    title=dict(text="Churn Probability", font=dict(color="#94a3b8", size=14)),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="#2d3748",
                                  tickfont=dict(color="#64748b")),
                        bar=dict(color=C["red"] if prob >= 0.5 else C["green"],
                                 thickness=0.75),
                        bgcolor="#1c2333",
                        bordercolor="#2d3748",
                        steps=[
                            dict(range=[0,  30],  color="#0d2218"),
                            dict(range=[30, 50],  color="#1e2010"),
                            dict(range=[50, 70],  color="#2a1515"),
                            dict(range=[70, 100], color="#2a0f0f"),
                        ],
                        threshold=dict(line=dict(color="#e2e8f0", width=3), value=50),
                    ),
                ))
                fig_gauge.update_layout(
                    height=270, paper_bgcolor="#161b27",
                    margin=dict(l=30, r=30, t=40, b=10),
                    font=dict(color="#94a3b8"),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with r3:
                st.markdown(f"""
                <div class="pred-card {verdict_cls}" style="padding:24px 20px">
                  <p class="pred-sub">Churn Probability</p>
                  <p class="pred-verdict {verdict_cls}" style="font-size:2.4rem">{prob:.1%}</p>
                  <hr style="border-color:rgba(255,255,255,0.08);margin:12px 0">
                  <p class="pred-sub">Retention Probability</p>
                  <p style="font-size:2rem;font-weight:800;color:{C['green']};margin:4px 0">{1-prob:.1%}</p>
                </div>""", unsafe_allow_html=True)

            # ── Recommendation ────────────────
            st.markdown(f"""
            <div class="rec-box" style="border-left-color:{rec['color']};background:{rec['bg']};">
              <h4 style="color:{rec['color']}">{rec['icon']} {rec['title']}</h4>
              <p>{rec['action']}</p>
            </div>""", unsafe_allow_html=True)

            # ── SHAP Explanation ──────────────
            st.markdown("---")
            st.markdown('<div class="sec-hdr">🔍 Why Did the Model Predict This?</div>',
                        unsafe_allow_html=True)

            if sel_model in TREE_MODELS:
                with st.spinner("Computing SHAP explanation …"):
                    shap_fig = compute_shap_waterfall(
                        model, X_proc, feature_names, sel_model, max_display=15)
                st.pyplot(shap_fig, use_container_width=True)
                plt.close("all")
                st.markdown(f"""
                <div class="shap-explain">
                  <strong>How to read this chart:</strong>
                  The model starts from a base probability (E[f(x)] — the average churn rate across all customers).
                  Each bar shows how one feature <em>moved</em> the prediction up or down.
                  <strong style="color:{C['red']}">Red bars → push toward churn</strong> &nbsp;|&nbsp;
                  <strong style="color:{C['cyan']}">Blue bars → push away from churn</strong>.
                  Features are sorted by their absolute impact on this specific prediction.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pill amber">
                  ℹ️ SHAP waterfall is available for tree-based models
                  (Decision Tree, Random Forest, XGBoost).
                  Switch to one of those models to see a full feature explanation.
                </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)
