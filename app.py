"""
Telco Customer Churn — Streamlit Dashboard
==========================================
Tabs:
  1. Executive Summary
  2. Descriptive Analytics
  3. Model Performance
  4. Interactive Prediction

IMPORTANT: Run `python train_models.py` first to generate models & plots.
Then launch with: streamlit run app.py
"""

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # must be before pyplot import
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import joblib
import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# TensorFlow is optional — gracefully degrade if unavailable
try:
    from tensorflow import keras as _keras
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# ─────────────────────────────────────────────
#  CONSTANTS & PATHS
# ─────────────────────────────────────────────
DATA_PATH   = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR  = "models"
PLOTS_DIR   = "plots"

MODEL_FILES = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
    "Decision Tree":       os.path.join(MODELS_DIR, "decision_tree.joblib"),
    "Random Forest":       os.path.join(MODELS_DIR, "random_forest.joblib"),
    "XGBoost":             os.path.join(MODELS_DIR, "xgboost.joblib"),
    "Neural Network":      os.path.join(MODELS_DIR, "mlp.keras"),
}
TREE_MODELS = {"Decision Tree", "Random Forest", "XGBoost"}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLS  = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
CAT_COLS     = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
METRIC_COLS  = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
MODEL_COLORS = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c"]

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-title  { font-size:2.4rem; font-weight:800; color:#1a1a2e; margin-bottom:0; }
    .sub-title   { font-size:1.1rem; color:#555; margin-top:0; }
    .metric-card { background:#f8f9fa; border-radius:12px; padding:18px 22px;
                   border-left:5px solid #3498db; margin-bottom:12px; }
    .section-hdr { font-size:1.3rem; font-weight:700; color:#1a1a2e;
                   border-bottom:2px solid #e0e0e0; padding-bottom:6px; margin-top:22px; }
    .insight-box { background:#eaf4fb; border-radius:8px; padding:14px 18px;
                   border-left:4px solid #2980b9; font-size:0.95rem; margin:8px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  DATA & MODEL LOADERS  (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_artifacts():
    """Load all pre-trained artifacts from disk."""
    artifacts = {}
    artifacts["metrics"]       = joblib.load(os.path.join(MODELS_DIR, "metrics.joblib"))
    artifacts["best_params"]   = joblib.load(os.path.join(MODELS_DIR, "best_params.joblib"))
    artifacts["test_probs"]    = joblib.load(os.path.join(MODELS_DIR, "test_probs.joblib"))
    artifacts["feature_names"] = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    artifacts["scaler"]        = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    artifacts["y_test"]        = joblib.load(os.path.join(MODELS_DIR, "y_test.joblib"))
    artifacts["X_test_scaled"] = joblib.load(os.path.join(MODELS_DIR, "X_test_scaled.joblib"))
    artifacts["shap_values"]   = np.load(os.path.join(MODELS_DIR, "shap_values.npy"))
    artifacts["shap_ev"]       = np.load(os.path.join(MODELS_DIR, "shap_expected_value.npy"))
    artifacts["X_shap"]        = joblib.load(os.path.join(MODELS_DIR, "X_shap.joblib"))
    artifacts["shap_info"]     = joblib.load(os.path.join(MODELS_DIR, "shap_info.joblib"))
    return artifacts


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
    models = {}
    for name, path in MODEL_FILES.items():
        try:
            if name == "Neural Network":
                models[name] = load_keras_model(path)
            else:
                models[name] = load_sklearn_model(path)
        except Exception:
            models[name] = None
    return models


# ─────────────────────────────────────────────
#  PREPROCESSING (mirrors train_models.py)
# ─────────────────────────────────────────────
def preprocess_single_input(user_dict: dict, feature_names: list, scaler) -> pd.DataFrame:
    """Convert raw user input dict → scaled feature DataFrame."""
    df = pd.DataFrame([user_dict])

    # Binary encodings
    df["gender"] = (df["gender"] == "Male").astype(int)
    for col in BINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    # One-hot encode
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)

    # bool → int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Align columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return df


# ─────────────────────────────────────────────
#  SHAP WATERFALL HELPER
# ─────────────────────────────────────────────
def compute_shap_and_plot(model, X_proc: pd.DataFrame, feature_names: list,
                           model_name: str, max_display: int = 15):
    """Compute SHAP for a single sample & return matplotlib figure."""
    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X_proc)

    if isinstance(shap_vals, list):
        sv   = shap_vals[1][0]
        ev   = explainer.expected_value[1]
    else:
        sv   = shap_vals[0]
        ev   = (explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value)

    shap_exp = shap.Explanation(
        values=sv,
        base_values=ev,
        data=X_proc.iloc[0].values,
        feature_names=feature_names,
    )
    plt.figure(figsize=(13, 8))
    shap.plots.waterfall(shap_exp, max_display=max_display, show=False)
    plt.title(f"SHAP Waterfall – {model_name}", fontsize=13, fontweight="bold")
    fig = plt.gcf()
    return fig


# ─────────────────────────────────────────────
#  GUARD: ensure models are trained
# ─────────────────────────────────────────────
if not os.path.exists(os.path.join(MODELS_DIR, "metrics.joblib")):
    st.error(
        "⚠️  **Models not found!**  \n"
        "Please run the training script first:  \n"
        "```bash\npython train_models.py\n```"
    )
    st.stop()

# Load everything once
df_raw    = load_raw_data()
artifacts = load_artifacts()
models    = load_all_models()

metrics_all   = artifacts["metrics"]
best_params   = artifacts["best_params"]
test_probs    = artifacts["test_probs"]
feature_names = artifacts["feature_names"]
scaler        = artifacts["scaler"]
y_test        = artifacts["y_test"]
X_test_sc     = artifacts["X_test_scaled"]
shap_values   = artifacts["shap_values"]
shap_ev       = float(artifacts["shap_ev"][0])
X_shap        = artifacts["X_shap"]
best_tree_name = artifacts["shap_info"]["best_tree_name"]

metrics_df = pd.DataFrame(metrics_all).T

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(
    '<p class="main-title">📡 Telco Customer Churn Analytics</p>'
    '<p class="sub-title">End-to-end ML pipeline: Descriptive Analytics · Predictive Models · Explainability</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Executive Summary",
    "📊 Descriptive Analytics",
    "🤖 Model Performance",
    "🔮 Interactive Prediction",
])


# ══════════════════════════════════════════════
#  TAB 1 – EXECUTIVE SUMMARY
# ══════════════════════════════════════════════
with tab1:
    st.markdown("## 🏠 Executive Summary")

    churn_rate   = (df_raw["Churn"] == "Yes").mean()
    n_customers  = len(df_raw)
    n_churned    = (df_raw["Churn"] == "Yes").sum()
    best_model   = metrics_df["AUC-ROC"].idxmax()
    best_auc     = metrics_df["AUC-ROC"].max()
    best_f1      = metrics_df.loc[best_model, "F1"]
    avg_monthly  = df_raw["MonthlyCharges"].mean()
    df_raw_proc  = df_raw.copy()
    df_raw_proc["TotalCharges"] = pd.to_numeric(df_raw_proc["TotalCharges"], errors="coerce")
    revenue_at_risk = df_raw_proc.loc[df_raw_proc["Churn"] == "Yes", "MonthlyCharges"].sum()

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total Customers", f"{n_customers:,}")
    with c2:
        st.metric("Churned", f"{n_churned:,}", delta=f"{churn_rate:.1%}", delta_color="inverse")
    with c3:
        st.metric("Avg Monthly Charge", f"${avg_monthly:.2f}")
    with c4:
        st.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")
    with c5:
        st.metric("Best Model AUC-ROC", f"{best_auc:.4f}", delta=best_model)

    st.markdown("---")

    # Model leaderboard
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown('<p class="section-hdr">Model Leaderboard</p>', unsafe_allow_html=True)
        display_df = metrics_df[METRIC_COLS].sort_values("AUC-ROC", ascending=False).reset_index()
        display_df.columns = ["Model"] + METRIC_COLS
        display_df = display_df.style.format({m: "{:.4f}" for m in METRIC_COLS}) \
            .highlight_max(subset=METRIC_COLS, color="#d4edda") \
            .set_table_styles([{"selector": "th", "props": [("background", "#1a1a2e"), ("color", "white")]}])
        st.dataframe(display_df, use_container_width=True, height=230)

    with col_r:
        st.markdown('<p class="section-hdr">Key Business Insights</p>', unsafe_allow_html=True)
        insights = [
            ("📉", "Month-to-month contract customers churn at ~43% vs 11% (one-year) and 3% (two-year)."),
            ("💸", "Churned customers pay ~$15 more/month on average than retained customers."),
            ("🌐", "Fiber optic users churn at ~41% vs DSL at ~19%."),
            ("✅", "Electronic check payment method is the highest churn risk group."),
            ("👴", "Senior citizens are ~2× more likely to churn than non-seniors."),
            ("📅", "~50% of churned customers left within their first 12 months."),
        ]
        for icon, text in insights:
            st.markdown(
                f'<div class="insight-box">{icon} {text}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown('<p class="section-hdr">Best Model Performance at a Glance</p>', unsafe_allow_html=True)
    cols = st.columns(len(METRIC_COLS))
    for col, metric in zip(cols, METRIC_COLS):
        col.metric(metric, f"{metrics_df.loc[best_model, metric]:.4f}", delta=best_model)


# ══════════════════════════════════════════════
#  TAB 2 – DESCRIPTIVE ANALYTICS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Descriptive Analytics")

    # Dataset overview
    with st.expander("📋 Dataset Overview & Basic Statistics", expanded=True):
        st.write(f"**Shape:** {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Data Types & Non-Null Counts**")
            info_df = pd.DataFrame({
                "Column": df_raw.columns,
                "Dtype":  df_raw.dtypes.values,
                "Non-Null": df_raw.notnull().sum().values,
                "Null": df_raw.isnull().sum().values,
            })
            st.dataframe(info_df, use_container_width=True, height=300)
        with col_b:
            st.write("**Numerical Summary Statistics**")
            df_temp = df_raw.copy()
            df_temp["TotalCharges"] = pd.to_numeric(df_temp["TotalCharges"], errors="coerce")
            st.dataframe(df_temp.describe().T.round(2), use_container_width=True, height=300)

    st.markdown("---")

    # ── Churn Distribution ──────────────────
    st.markdown('<p class="section-hdr">1. Target Distribution (Churn: Yes vs No)</p>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        vc = df_raw["Churn"].value_counts().reset_index()
        vc.columns = ["Churn", "Count"]
        fig_cd = px.bar(
            vc, x="Churn", y="Count", color="Churn",
            color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
            text="Count", title="Churn Count",
        )
        fig_cd.update_traces(textposition="outside")
        fig_cd.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_cd, use_container_width=True)

    with col2:
        fig_pie = px.pie(
            vc, names="Churn", values="Count",
            color="Churn",
            color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
            title="Churn Proportion",
            hole=0.45,
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=14)
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown(
        '<div class="insight-box">⚠️ <b>Class Imbalance:</b> ~73% of customers did not churn vs ~27% who did. '
        "This imbalance should be considered when evaluating model performance — F1 and AUC-ROC "
        "are more informative than raw accuracy.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Tenure Distribution ─────────────────
    st.markdown('<p class="section-hdr">2. Tenure Distribution by Churn Status</p>',
                unsafe_allow_html=True)
    fig_ten = px.histogram(
        df_raw, x="tenure", color="Churn", barmode="overlay",
        nbins=40, opacity=0.75,
        color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
        labels={"tenure": "Tenure (Months)", "count": "Count"},
        title="Tenure Distribution by Churn Status",
    )
    fig_ten.update_layout(height=420)
    st.plotly_chart(fig_ten, use_container_width=True)
    st.markdown(
        '<div class="insight-box">📉 <b>Short tenure = high churn risk.</b> Churned customers have a median tenure '
        "of ~10 months vs ~38 months for retained customers. Interventions in the first year are critical.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Monthly Charges Boxplot ─────────────
    st.markdown('<p class="section-hdr">3. Monthly Charges by Churn Status</p>',
                unsafe_allow_html=True)
    fig_box = px.box(
        df_raw, x="Churn", y="MonthlyCharges", color="Churn",
        color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
        notched=True,
        points="outliers",
        title="Monthly Charges Distribution",
        labels={"MonthlyCharges": "Monthly Charges ($)"},
    )
    fig_box.update_layout(height=420, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown(
        '<div class="insight-box">💰 <b>Higher charges drive churn.</b> Customers who churned paid ~$74/month '
        "on average vs ~$61 for non-churners. Premium service tiers need stronger retention incentives.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Contract Type ──────────────────────
    st.markdown('<p class="section-hdr">4. Churn Rate by Contract Type</p>',
                unsafe_allow_html=True)
    ct = df_raw.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
    fig_ct = px.bar(
        ct, x="Contract", y="Count", color="Churn", barmode="group",
        color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
        title="Customer Count by Contract & Churn",
        text="Count",
    )
    fig_ct.update_traces(textposition="outside")
    fig_ct.update_layout(height=420)
    st.plotly_chart(fig_ct, use_container_width=True)
    st.markdown(
        '<div class="insight-box">📝 <b>Longest contracts = lowest churn.</b> Month-to-month contracts have ~43% churn rate '
        "vs ~11% (one-year) and ~3% (two-year). Encouraging annual or bi-annual plans is a key retention strategy.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Internet Service ───────────────────
    st.markdown('<p class="section-hdr">5. Churn by Internet Service Type</p>',
                unsafe_allow_html=True)
    it = df_raw.groupby(["InternetService", "Churn"]).size().reset_index(name="Count")
    fig_it = px.bar(
        it, x="InternetService", y="Count", color="Churn", barmode="group",
        color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
        title="Internet Service Type vs Churn",
        text="Count",
    )
    fig_it.update_traces(textposition="outside")
    fig_it.update_layout(height=420)
    st.plotly_chart(fig_it, use_container_width=True)
    st.markdown(
        '<div class="insight-box">🌐 <b>Fiber optic customers churn at ~41%.</b> This is likely due to high pricing, '
        "competition, or service quality issues. Targeted promotions for fiber users could significantly reduce churn.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Payment Method ─────────────────────
    st.markdown('<p class="section-hdr">6. Churn Rate by Payment Method</p>',
                unsafe_allow_html=True)
    pm_df  = df_raw.groupby(["PaymentMethod", "Churn"]).size().reset_index(name="Count")
    pm_tot = pm_df.groupby("PaymentMethod")["Count"].transform("sum")
    pm_df["Pct"] = (pm_df["Count"] / pm_tot * 100).round(1)
    fig_pm = px.bar(
        pm_df[pm_df["Churn"] == "Yes"], x="Pct", y="PaymentMethod",
        orientation="h", color="Pct",
        color_continuous_scale="Reds",
        title="Churn Rate (%) by Payment Method",
        labels={"Pct": "Churn Rate (%)", "PaymentMethod": "Payment Method"},
        text="Pct",
    )
    fig_pm.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_pm.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_pm, use_container_width=True)
    st.markdown(
        '<div class="insight-box">💳 <b>Electronic check users churn the most (~45%).</b> Customers on auto-payment '
        "(bank transfer or credit card) have significantly lower churn rates, suggesting manual payers are less engaged.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Senior Citizen ─────────────────────
    st.markdown('<p class="section-hdr">7. Senior Citizen vs Churn</p>',
                unsafe_allow_html=True)
    sc_df = df_raw.copy()
    sc_df["Segment"] = sc_df["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
    sc_grp = sc_df.groupby(["Segment", "Churn"]).size().reset_index(name="Count")
    fig_sc = px.bar(
        sc_grp, x="Segment", y="Count", color="Churn", barmode="group",
        color_discrete_map={"No": "#2ecc71", "Yes": "#e74c3c"},
        title="Senior vs Non-Senior Churn", text="Count",
    )
    fig_sc.update_traces(textposition="outside")
    fig_sc.update_layout(height=400)
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # ── Correlation Heatmap ────────────────
    st.markdown('<p class="section-hdr">8. Feature Correlation Heatmap</p>',
                unsafe_allow_html=True)
    if os.path.exists(os.path.join(PLOTS_DIR, "08_correlation_heatmap.png")):
        st.image(
            os.path.join(PLOTS_DIR, "08_correlation_heatmap.png"),
            caption="Full feature correlation heatmap (generated during training)",
            use_container_width=True,
        )
    else:
        st.info("Run train_models.py to generate the correlation heatmap.")

    st.markdown(
        '<div class="insight-box">🔗 <b>Notable correlations:</b> Tenure is negatively correlated with churn. '
        "MonthlyCharges and TotalCharges are strongly positively correlated. "
        "Contract type features are among the strongest predictors of churn.</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
#  TAB 3 – MODEL PERFORMANCE
# ══════════════════════════════════════════════
with tab3:
    st.markdown("## 🤖 Model Performance")

    # ── Metrics Table ──────────────────────
    st.markdown('<p class="section-hdr">All Models – Metrics Summary</p>', unsafe_allow_html=True)
    disp = metrics_df[METRIC_COLS].copy()
    disp = disp.style \
        .format("{:.4f}") \
        .highlight_max(color="#d4edda") \
        .highlight_min(color="#fde8e8") \
        .set_caption("Green = best per metric | Red = worst per metric")
    st.dataframe(disp, use_container_width=True)

    st.markdown("---")

    # ── ROC Curves ─────────────────────────
    st.markdown('<p class="section-hdr">ROC Curves – All Models</p>', unsafe_allow_html=True)
    y_test_arr = y_test.values
    fig_roc    = go.Figure()

    for i, (name, probs) in enumerate(test_probs.items()):
        fpr, tpr, _ = roc_curve(y_test_arr, np.array(probs))
        auc = metrics_all[name]["AUC-ROC"]
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name}  (AUC={auc:.4f})",
            line=dict(color=MODEL_COLORS[i], width=2.5),
        ))

    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Classifier",
        line=dict(color="gray", width=1.5, dash="dash"),
    ))
    fig_roc.update_layout(
        title="ROC Curves – All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=520,
        legend=dict(x=0.55, y=0.05),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # ── Metrics Bar Chart ──────────────────
    st.markdown('<p class="section-hdr">Metric Comparison – Bar Chart</p>', unsafe_allow_html=True)
    fig_bar = go.Figure()
    for i, name in enumerate(metrics_all):
        fig_bar.add_trace(go.Bar(
            name=name,
            x=METRIC_COLS,
            y=[metrics_all[name][m] for m in METRIC_COLS],
            marker_color=MODEL_COLORS[i],
            opacity=0.87,
        ))
    fig_bar.update_layout(
        barmode="group",
        title="All Models – All Metrics",
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1]),
        height=480,
        legend=dict(orientation="h", y=-0.18),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Best Hyperparameters ───────────────
    st.markdown('<p class="section-hdr">Best Hyperparameters (GridSearchCV)</p>',
                unsafe_allow_html=True)
    hp_cols = st.columns(len(best_params))
    for col, (name, params) in zip(hp_cols, best_params.items()):
        col.markdown(f"**{name}**")
        for k, v in params.items():
            col.markdown(f"- `{k}`: `{v}`")

    st.markdown("---")

    # ── SHAP Summary ──────────────────────
    st.markdown('<p class="section-hdr">SHAP Analysis – Best Tree Model</p>',
                unsafe_allow_html=True)
    st.write(f"Best tree-based model (by AUC-ROC): **{best_tree_name}**")

    shap_c1, shap_c2 = st.columns(2)
    with shap_c1:
        if os.path.exists(os.path.join(PLOTS_DIR, "11_shap_beeswarm.png")):
            st.image(
                os.path.join(PLOTS_DIR, "11_shap_beeswarm.png"),
                caption=f"SHAP Beeswarm – {best_tree_name}",
                use_container_width=True,
            )
    with shap_c2:
        if os.path.exists(os.path.join(PLOTS_DIR, "12_shap_bar.png")):
            st.image(
                os.path.join(PLOTS_DIR, "12_shap_bar.png"),
                caption=f"SHAP Mean |value| – {best_tree_name}",
                use_container_width=True,
            )

    if os.path.exists(os.path.join(PLOTS_DIR, "13_shap_waterfall.png")):
        st.image(
            os.path.join(PLOTS_DIR, "13_shap_waterfall.png"),
            caption="SHAP Waterfall – single test-set prediction",
            use_container_width=True,
        )

    st.markdown(
        '<div class="insight-box">🔍 <b>Top SHAP drivers of churn:</b> Contract type (month-to-month), '
        "tenure (low = high risk), InternetService_Fiber optic, MonthlyCharges (high = high risk), "
        "and OnlineSecurity_No are consistently the most influential features.</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
#  TAB 4 – INTERACTIVE PREDICTION
# ══════════════════════════════════════════════
with tab4:
    st.markdown("## 🔮 Interactive Prediction")
    st.write(
        "Adjust the customer profile below, choose a model, and see the predicted "
        "churn probability in real time. SHAP explanation is available for tree-based models."
    )

    # ── Model Selector ─────────────────────
    sel_model = st.selectbox(
        "🤖 Select Model",
        options=list(MODEL_FILES.keys()),
        index=2,  # default: Random Forest
        help="SHAP waterfall is available for Decision Tree, Random Forest, and XGBoost.",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Customer Feature Configuration")

    # ── Feature Inputs ─────────────────────
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    r4c1, r4c2, r4c3, r4c4 = st.columns(4)
    r5c1, r5c2, r5c3        = st.columns(3)

    with r1c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with r1c2:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    with r1c3:
        partner = st.selectbox("Partner", ["Yes", "No"])
    with r1c4:
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with r2c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with r2c2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    with r2c3:
        multiple_lines = st.selectbox("Multiple Lines",
                                      ["No", "Yes", "No phone service"])
    with r2c4:
        internet_service = st.selectbox("Internet Service",
                                        ["DSL", "Fiber optic", "No"])

    with r3c1:
        online_security = st.selectbox("Online Security",
                                       ["No", "Yes", "No internet service"])
    with r3c2:
        online_backup = st.selectbox("Online Backup",
                                     ["No", "Yes", "No internet service"])
    with r3c3:
        device_protection = st.selectbox("Device Protection",
                                         ["No", "Yes", "No internet service"])
    with r3c4:
        tech_support = st.selectbox("Tech Support",
                                    ["No", "Yes", "No internet service"])

    with r4c1:
        streaming_tv = st.selectbox("Streaming TV",
                                    ["No", "Yes", "No internet service"])
    with r4c2:
        streaming_movies = st.selectbox("Streaming Movies",
                                        ["No", "Yes", "No internet service"])
    with r4c3:
        contract = st.selectbox("Contract",
                                ["Month-to-month", "One year", "Two year"])
    with r4c4:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    with r5c1:
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )
    with r5c2:
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    with r5c3:
        total_charges = st.slider("Total Charges ($)", 18.0, 8700.0,
                                  float(monthly_charges * tenure if tenure else 65.0),
                                  step=1.0)

    # ── Build Input Dict ───────────────────
    user_input = {
        "gender":           gender,
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_service,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_protection,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod":    payment_method,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
    }

    st.markdown("---")

    # ── Prediction ─────────────────────────
    predict_btn = st.button("🔮 Predict Churn", type="primary", use_container_width=True)

    if predict_btn:
        try:
            X_proc = preprocess_single_input(user_input, feature_names, scaler)
            model  = models[sel_model]

            if sel_model == "Neural Network":
                if model is None:
                    st.error("Neural Network model unavailable in this environment.")
                    st.stop()
                prob = float(model.predict(X_proc.values, verbose=0)[0][0])
            else:
                prob = float(model.predict_proba(X_proc)[0][1])

            pred = int(prob >= 0.5)

            # Result display
            st.markdown("---")
            st.markdown("### 📊 Prediction Result")

            res_col1, res_col2, res_col3 = st.columns([1, 1.5, 1])
            with res_col1:
                st.metric(
                    label="Churn Prediction",
                    value="⚠️ WILL CHURN" if pred == 1 else "✅ WILL STAY",
                )
            with res_col2:
                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    delta={"reference": 50},
                    title={"text": "Churn Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#e74c3c" if prob >= 0.5 else "#2ecc71"},
                        "steps": [
                            {"range": [0, 30],  "color": "#d4edda"},
                            {"range": [30, 60], "color": "#fff3cd"},
                            {"range": [60, 100],"color": "#f8d7da"},
                        ],
                        "threshold": {
                            "line": {"color": "navy", "width": 4},
                            "value": 50,
                        },
                    },
                ))
                fig_gauge.update_layout(height=280)
                st.plotly_chart(fig_gauge, use_container_width=True)
            with res_col3:
                st.metric("Churn Probability", f"{prob:.2%}")
                st.metric("Retention Probability", f"{1 - prob:.2%}")
                risk = "🔴 High Risk" if prob >= 0.7 else ("🟡 Medium Risk" if prob >= 0.4 else "🟢 Low Risk")
                st.markdown(f"**Risk Level:** {risk}")

            # ── SHAP Waterfall ────────────────
            st.markdown("---")
            st.markdown("### 🔍 SHAP Explanation")

            if sel_model in TREE_MODELS:
                with st.spinner("Computing SHAP values …"):
                    shap_fig = compute_shap_and_plot(
                        model, X_proc, feature_names, sel_model, max_display=15
                    )
                st.pyplot(shap_fig, use_container_width=True)
                plt.close("all")
                st.markdown(
                    '<div class="insight-box">📌 <b>How to read this:</b> Red bars push the prediction '
                    "<b>toward churn</b>; blue bars push it <b>away from churn</b>. "
                    "The base value (E[f(x)]) is the average model output. Features are ranked by |SHAP value|.</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info(
                    f"ℹ️ SHAP waterfall is available for tree-based models "
                    f"(Decision Tree, Random Forest, XGBoost). "
                    f"Selected model: **{sel_model}**."
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)
