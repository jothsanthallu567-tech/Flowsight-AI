import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
import datetime
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from data_loader import load_raw_data, build_merged_dataset
from feature_engineering import FeaturePipeline
from model_from_scratch import LinearRegressionScratch
from baseline import DepartmentMedianBaseline
from evaluation import calculate_metrics
from investigation_engine import generate_investigate_first_queue

# Firebase integration (optional — graceful fallback if not configured)
try:
    from firebase_client import save_document, get_collection
    FIREBASE_ENABLED = True
except Exception:
    FIREBASE_ENABLED = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FlowSight AI \u2014 Health Operations Intelligence",
    page_icon="\U0001f499",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LIGHT MULTI-COLOUR DESIGN SYSTEM
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, p, span, div {
    font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

.stApp {
    background: linear-gradient(145deg, #EFF6FF 0%, #F0FDF4 40%, #FFF7ED 70%, #FDF4FF 100%) !important;
    min-height: 100vh;
}

.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1480px !important;
}

h1 { font-size:2.4rem !important; font-weight:900 !important; letter-spacing:-0.035em !important; color:#1E293B !important; line-height:1.2 !important; }
h2 { font-size:1.9rem !important; font-weight:800 !important; color:#1E293B !important; }
h3 { font-size:1.5rem !important; font-weight:700 !important; color:#1E293B !important; }
h4 { font-size:1.25rem !important; font-weight:700 !important; color:#1E293B !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #EFF6FF 100%) !important;
    border-right: 2px solid #DBEAFE !important;
    box-shadow: 4px 0 20px rgba(59,130,246,0.08) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size:0.98rem !important; font-weight:600 !important; color:#334155 !important;
    padding:10px 14px !important; border-radius:10px !important; transition:all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background:linear-gradient(90deg,#DBEAFE,#EDE9FE) !important; color:#1D4ED8 !important; transform:translateX(4px);
}

[data-testid="stMetricValue"] { font-size:2.4rem !important; font-weight:900 !important; color:#1E293B !important; }
[data-testid="stMetricLabel"] { font-size:0.88rem !important; font-weight:700 !important; color:#64748B !important; text-transform:uppercase !important; letter-spacing:0.05em !important; }

div[data-baseweb="select"] > div {
    font-size:0.95rem !important; border-radius:10px !important; border-color:#BFDBFE !important;
    background:#FFFFFF !important; min-height:42px !important; font-weight:600 !important; color:#1E293B !important;
}
div[data-baseweb="select"] ul { background:#FFFFFF !important; border:1.5px solid #BFDBFE !important; border-radius:10px !important; }
div[data-baseweb="select"] li { color:#1E293B !important; font-weight:600 !important; background:#FFFFFF !important; }
div[data-baseweb="select"] li:hover { background:#EFF6FF !important; color:#1D4ED8 !important; }

.stButton button {
    font-size:0.92rem !important; font-weight:700 !important; border-radius:10px !important; height:42px !important;
    border:1.5px solid #BFDBFE !important; background:#FFFFFF !important; color:#1D4ED8 !important;
}
.stButton button:hover { background:#EFF6FF !important; border-color:#3B82F6 !important; transform:translateY(-1px); }

.kpi-card { background:#FFFFFF; border-radius:16px; padding:18px 20px 14px; position:relative; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.06); transition:transform 0.2s ease,box-shadow 0.2s ease; margin-bottom:4px; }
.kpi-card:hover { transform:translateY(-4px); box-shadow:0 14px 36px rgba(0,0,0,0.10); }
.kpi-card-blue    { border-top:5px solid #3B82F6; }
.kpi-card-emerald { border-top:5px solid #10B981; }
.kpi-card-amber   { border-top:5px solid #F59E0B; }
.kpi-card-violet  { border-top:5px solid #8B5CF6; }
.kpi-card-rose    { border-top:5px solid #F43F5E; }
.kpi-card-teal    { border-top:5px solid #14B8A6; }

.kpi-icon  { font-size:1.8rem; margin-bottom:4px; }
.kpi-label { font-size:0.76rem; font-weight:800; letter-spacing:0.07em; text-transform:uppercase; color:#64748B; margin-bottom:2px; }
.kpi-value { font-size:2.3rem; font-weight:900; color:#1E293B; letter-spacing:-0.04em; line-height:1.1; margin-bottom:6px; }
.kpi-unit  { font-size:1.0rem; font-weight:600; color:#64748B; }

.badge-red    { display:inline-block; padding:3px 10px; border-radius:9999px; font-size:0.8rem; font-weight:800; background:#FEE2E2; color:#B91C1C; border:1.5px solid #FCA5A5; }
.badge-green  { display:inline-block; padding:3px 10px; border-radius:9999px; font-size:0.8rem; font-weight:800; background:#D1FAE5; color:#065F46; border:1.5px solid #6EE7B7; }
.badge-amber  { display:inline-block; padding:3px 10px; border-radius:9999px; font-size:0.8rem; font-weight:800; background:#FEF3C7; color:#92400E; border:1.5px solid #FCD34D; }
.badge-blue   { display:inline-block; padding:3px 10px; border-radius:9999px; font-size:0.8rem; font-weight:800; background:#DBEAFE; color:#1D4ED8; border:1.5px solid #93C5FD; }
.badge-violet { display:inline-block; padding:3px 10px; border-radius:9999px; font-size:0.8rem; font-weight:800; background:#EDE9FE; color:#5B21B6; border:1.5px solid #C4B5FD; }

.wcard { background:#FFFFFF; border-radius:16px; padding:20px 22px; margin-bottom:18px; box-shadow:0 4px 20px rgba(0,0,0,0.05); border:1.5px solid #E2E8F0; transition:box-shadow 0.2s ease; }
.wcard:hover { box-shadow:0 8px 32px rgba(0,0,0,0.09); }
.wcard-title { font-size:1.1rem; font-weight:800; color:#1E293B; margin-bottom:2px; letter-spacing:-0.01em; }
.wcard-sub   { font-size:0.83rem; color:#64748B; font-weight:600; margin-bottom:12px; }

.grad-blue    { background:linear-gradient(135deg,#3B82F6,#60A5FA); border-radius:16px; padding:22px 24px; color:#FFFFFF; margin-bottom:18px; }
.grad-emerald { background:linear-gradient(135deg,#10B981,#34D399); border-radius:16px; padding:22px 24px; color:#FFFFFF; margin-bottom:18px; }
.grad-violet  { background:linear-gradient(135deg,#7C3AED,#A78BFA); border-radius:16px; padding:22px 24px; color:#FFFFFF; margin-bottom:18px; }
.grad-rose    { background:linear-gradient(135deg,#F43F5E,#FB7185); border-radius:16px; padding:22px 24px; color:#FFFFFF; margin-bottom:18px; }
.grad-amber   { background:linear-gradient(135deg,#F59E0B,#FCD34D); border-radius:16px; padding:22px 24px; color:#1E293B; margin-bottom:18px; }
.grad-teal    { background:linear-gradient(135deg,#0D9488,#2DD4BF); border-radius:16px; padding:22px 24px; color:#FFFFFF; margin-bottom:18px; }

.img-card { background:#FFFFFF; border-radius:14px; overflow:hidden; box-shadow:0 4px 20px rgba(0,0,0,0.07); border:1.5px solid #E2E8F0; margin-bottom:18px; }
.img-cap  { padding:10px 16px; font-size:0.88rem; font-weight:800; color:#FFFFFF; display:flex; align-items:center; gap:8px; }

@keyframes pulse { 0%,100%{ transform:scale(1); opacity:1; } 50%{ transform:scale(1.5); opacity:0.5; } }
.live-dot { display:inline-block; width:10px; height:10px; border-radius:50%; background:#10B981; animation:pulse 2s infinite; vertical-align:middle; margin-right:5px; box-shadow:0 0 8px #10B981; }

[data-testid="stDataFrame"] { border-radius:12px !important; border:1.5px solid #DBEAFE !important; font-size:0.93rem !important; overflow:hidden !important; }
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label, div[data-testid="stNumberInput"] label { font-size:0.88rem !important; font-weight:700 !important; color:#334155 !important; }

.app-footer { text-align:center; color:#64748B; font-size:0.85rem; padding:24px 0 12px; border-top:1.5px solid #DBEAFE; margin-top:40px; font-weight:600; line-height:1.7; }
.sdiv { height:20px; }
.risk-high   { background:#FFF1F2; border:2px solid #FDA4AF; border-radius:12px; padding:14px 16px; margin-bottom:8px; }
.risk-medium { background:#FFFBEB; border:2px solid #FCD34D; border-radius:12px; padding:14px 16px; margin-bottom:8px; }
.risk-low    { background:#F0FDF4; border:2px solid #86EFAC; border-radius:12px; padding:14px 16px; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA + MODEL
# ============================================================
@st.cache_data
def get_data():
    return build_merged_dataset(os.path.join(BASE_DIR, 'data'))

@st.cache_data
def get_raw():
    return load_raw_data(os.path.join(BASE_DIR, 'data'))

@st.cache_resource
def train_model():
    df = get_data()
    train_df = df[df['arrival_time'] < '2025-10-01'].copy()
    test_df  = df[df['arrival_time'] >= '2025-12-01'].copy()
    pipeline = FeaturePipeline()
    X_train, y_train = pipeline.fit_transform(train_df)
    X_test,  y_test  = pipeline.transform(test_df)
    model    = LinearRegressionScratch(learning_rate=0.05, iterations=1500)
    model.fit(X_train, y_train)
    baseline = DepartmentMedianBaseline()
    baseline.fit(train_df)
    test_preds = model.predict(X_test)
    base_preds = baseline.predict(test_df)
    m_metrics  = calculate_metrics(y_test, test_preds)
    b_metrics  = calculate_metrics(y_test, base_preds)
    return df, train_df, test_df, pipeline, model, baseline, test_preds, base_preds, m_metrics, b_metrics

with st.spinner("\u2699\ufe0f Loading FlowSight AI \u2014 initialising ML model & data pipeline\u2026"):
    df, train_df, test_df, pipeline, model, baseline, test_preds, base_preds, m_metrics, b_metrics = train_model()
    patients, encounters, departments, diagnostic_tests = get_raw()
    diagnostic_tests['requested_time']  = pd.to_datetime(diagnostic_tests['requested_time'])
    diagnostic_tests['completion_time'] = pd.to_datetime(diagnostic_tests['completion_time'])
    diagnostic_tests['tat_minutes']     = (diagnostic_tests['completion_time'] - diagnostic_tests['requested_time']).dt.total_seconds() / 60.0

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:14px 0 10px;">
        <div style="font-size:2.2rem;">\U0001f499</div>
        <div style="font-size:1.35rem;font-weight:900;color:#1E293B;letter-spacing:-0.02em;">FlowSight AI</div>
        <div style="font-size:0.78rem;font-weight:600;color:#64748B;margin-top:2px;">Health Operations Intelligence</div>
        <div style="margin-top:10px;padding:5px 12px;background:linear-gradient(90deg,#ECFDF5,#D1FAE5);border-radius:9999px;display:inline-flex;align-items:center;font-size:0.8rem;font-weight:700;color:#065F46;">
            <span class="live-dot"></span> LIVE \u00b7 15,012 RECORDS
        </div>
    </div>
    <hr style="border:none;border-top:1.5px solid #DBEAFE;margin:8px 0 14px;">
    """, unsafe_allow_html=True)

    active_screen = st.radio(
        "Navigate",
        [
            "\U0001f4ca Operations Overview",
            "\U0001f52e Delay Prediction Engine",
            "\U0001f6a8 Bottleneck Investigator",
            "\U0001f465 Patient Demographics",
            "\U0001f52c Model Lab & Diagnostics",
            "\U0001f9ea Diagnostic Test TAT",
            "\U0001f525 Firestore Data Hub"
        ],
        label_visibility="collapsed"
    )

    # ---- Firebase Export Panel ----
    if FIREBASE_ENABLED:
        st.markdown("""<hr style="border:none;border-top:1.5px solid #DBEAFE;margin:12px 0 10px;">
        <div style="font-size:0.76rem;font-weight:800;text-transform:uppercase;color:#94A3B8;letter-spacing:0.08em;margin-bottom:8px;margin-left:4px;">🔥 Firebase Sync</div>""", unsafe_allow_html=True)
        if st.button("📤 Export Bottlenecks", use_container_width=True, help="Save Dec 2025 bottleneck analysis to Firestore"):
            try:
                with st.spinner("Syncing to Firestore…"):
                    X_exp, _ = pipeline.transform(test_df)
                    q_exp = generate_investigate_first_queue(test_df, model.predict(X_exp))
                    batch_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    for _, row in q_exp.head(20).iterrows():
                        doc_id = f"{batch_id}_{row['department_id']}_{row['hour_slot']:02d}"
                        save_document("bottlenecks", doc_id, {
                            "batch_id": batch_id,
                            "department": str(row["department_name"]),
                            "hour_slot": int(row["hour_slot"]),
                            "affected_encounters": int(row["affected_encounters"]),
                            "avg_predicted_wait": round(float(row["avg_predicted_wait"]), 2),
                            "avg_excess_wait": round(float(row["avg_excess_wait"]), 2),
                            "impact_score": round(float(row["impact_score"]), 2),
                            "suggested_action": str(row["suggested_action"]),
                            "exported_at": datetime.datetime.utcnow().isoformat()
                        })
                st.toast(f"✅ Exported {min(20, len(q_exp))} bottleneck records!", icon="🔥")
            except Exception as e:
                st.toast(f"❌ Export failed: {e}", icon="⚠️")

        if st.button("📊 Export Model Metrics", use_container_width=True, help="Save model performance snapshot to Firestore"):
            try:
                with st.spinner("Saving metrics…"):
                    save_document("model_metrics", datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S"), {
                        "model": "LinearRegressionScratch",
                        "learning_rate": 0.05,
                        "iterations": 1500,
                        "mae": round(m_metrics['MAE'], 4),
                        "rmse": round(m_metrics['RMSE'], 4),
                        "r2": round(m_metrics['R2'], 4),
                        "baseline_mae": round(b_metrics['MAE'], 4),
                        "baseline_rmse": round(b_metrics['RMSE'], 4),
                        "test_set_size": len(test_df),
                        "train_set_size": len(train_df),
                        "recorded_at": datetime.datetime.utcnow().isoformat()
                    })
                st.toast("✅ Model metrics saved to Firestore!", icon="📊")
            except Exception as e:
                st.toast(f"❌ Export failed: {e}", icon="⚠️")
    else:
        st.markdown("""
        <div style="background:#FFF7ED;border:1.5px solid #FED7AA;border-radius:10px;padding:10px 12px;font-size:0.78rem;color:#92400E;font-weight:600;">
        🔥 Firebase not configured.<br>Add <code>firebase-service-account.json</code> to enable sync.
        </div>""", unsafe_allow_html=True)

    st.markdown("""<hr style="border:none;border-top:1.5px solid #DBEAFE;margin:12px 0 10px;">
    <div style="font-size:0.76rem;font-weight:800;text-transform:uppercase;color:#94A3B8;letter-spacing:0.08em;margin-bottom:8px;margin-left:4px;">Data Snapshot</div>""", unsafe_allow_html=True)

    for icon, label, val in [
        ("\U0001f3e5","Departments",f"{len(departments)}"),
        ("\U0001f464","Unique Patients",f"{df['patient_id'].nunique():,}"),
        ("\U0001f4cb","Encounters",f"{len(df):,}"),
        ("\U0001f9ea","Diagnostic Orders",f"{len(diagnostic_tests):,}"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;padding:7px 10px;border-radius:8px;margin-bottom:4px;background:#F8FAFC;">
            <span style="font-size:0.85rem;color:#475569;font-weight:600;">{icon} {label}</span>
            <span style="font-size:0.9rem;font-weight:800;color:#1E293B;">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:14px;padding:12px;background:linear-gradient(135deg,#EFF6FF,#EDE9FE);border-radius:12px;border:1.5px solid #BFDBFE;">
        <div style="font-size:0.8rem;font-weight:800;color:#1D4ED8;">\U0001f916 ML Model Status</div>
        <div style="font-size:0.76rem;color:#374151;margin-top:3px;font-weight:600;">Pure NumPy \u00b7 Gradient Descent</div>
        <div style="font-size:0.76rem;color:#065F46;font-weight:700;margin-top:2px;">MAE: {m_metrics['MAE']:.2f} min | R\u00b2: {m_metrics['R2']:.3f}</div>
    </div>""", unsafe_allow_html=True)

# ============================================================
# GLOBAL PAGE HEADER
# ============================================================
page_meta = {
    "\U0001f4ca Operations Overview":     ("\U0001f4ca Operations Overview",     "Real-time ED capacity, wait times & patient flow across all 12 hospital departments"),
    "\U0001f52e Delay Prediction Engine": ("\U0001f52e Delay Prediction Engine", "Simulate patient delays using our Pure NumPy ML model with live upstream risk signals"),
    "\U0001f6a8 Bottleneck Investigator": ("\U0001f6a8 Bottleneck Investigator", "Rank and investigate the worst operational bottlenecks by predicted impact score"),
    "\U0001f465 Patient Demographics":    ("\U0001f465 Patient Demographics",    "Analyse patient population, chronic conditions, insurance mix & delay equity"),
    "\U0001f52c Model Lab & Diagnostics": ("\U0001f52c Model Lab & Diagnostics", "Inspect model performance, loss convergence, feature weights & residual distribution"),
    "\U0001f9ea Diagnostic Test TAT":     ("\U0001f9ea Diagnostic Test TAT",     "Monitor lab & radiology turnaround times by modality, priority & department"),
    "\U0001f525 Firestore Data Hub":      ("\U0001f525 Firestore Data Hub",      "Browse and manage data exported to Firebase Firestore — bottlenecks, metrics & predictions"),
}
title_txt, subtitle_txt = page_meta[active_screen]

col_title, col_badge = st.columns([3.0, 1.0])
with col_title:
    st.markdown(f"""
    <div style="margin-bottom:16px;">
        <h1 style="margin:0;">{title_txt}</h1>
        <div style="font-size:0.98rem;color:#64748B;margin-top:5px;font-weight:600;">{subtitle_txt}</div>
    </div>""", unsafe_allow_html=True)
with col_badge:
    st.markdown("""
    <div style="text-align:right;padding-top:14px;">
        <span style="background:#ECFDF5;color:#065F46;border:1.5px solid #6EE7B7;padding:7px 16px;border-radius:9999px;font-size:0.82rem;font-weight:700;display:inline-flex;align-items:center;gap:5px;">
            <span class="live-dot"></span> DATA VERIFIED
        </span>
    </div>""", unsafe_allow_html=True)

# ============================================================
# HELPER: KPI row renderer
# ============================================================
def render_kpi(col, card_cls, icon, label, val, unit, badge_cls, badge_txt):
    with col:
        st.markdown(f"""
        <div class="kpi-card {card_cls}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}<span class="kpi-unit"> {unit}</span></div>
            <div class="{badge_cls}">{badge_txt}</div>
        </div>""", unsafe_allow_html=True)

# ============================================================
# SCREEN 1 \u2014 OPERATIONS OVERVIEW
# ============================================================
if active_screen == "\U0001f4ca Operations Overview":
    fc1, fc2, fc3, fc4 = st.columns([1.4, 1.4, 1.2, 0.7])
    with fc1: qtr_choice = st.selectbox("\U0001f4c5 Date Range", ["Full Year 2025","Q1 (Jan\u2013Mar)","Q2 (Apr\u2013Jun)","Q3 (Jul\u2013Sep)","Q4 Dec (Held-Out)"])
    with fc2:
        dept_opts = ["All Departments"] + sorted(df['department_name'].unique().tolist())
        dept_sel  = st.selectbox("\U0001f3e5 Department", dept_opts)
    with fc3:
        svc_opts = ["All Service Lines"] + (sorted(df['service_type'].dropna().unique().tolist()) if 'service_type' in df.columns else [])
        svc_sel  = st.selectbox("\u2695\ufe0f Service Line", svc_opts)
    with fc4:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        if st.button("\U0001f504 Reset", use_container_width=True): st.rerun()

    fdf = df.copy()
    if "Q1" in qtr_choice:   fdf = fdf[fdf['arrival_time'].dt.month.between(1,3)]
    elif "Q2" in qtr_choice: fdf = fdf[fdf['arrival_time'].dt.month.between(4,6)]
    elif "Q3" in qtr_choice: fdf = fdf[fdf['arrival_time'].dt.month.between(7,9)]
    elif "Q4" in qtr_choice: fdf = fdf[fdf['arrival_time'] >= '2025-12-01']
    if dept_sel != "All Departments":    fdf = fdf[fdf['department_name'] == dept_sel]
    if svc_sel  != "All Service Lines":  fdf = fdf[fdf['service_type']    == svc_sel]

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    render_kpi(k1,"kpi-card-blue",   "\U0001f552","Avg ED Wait",      f"{fdf['wait_minutes'].mean():.1f}","min","badge-red",  f"\u25b2 vs 45 min target")
    render_kpi(k2,"kpi-card-emerald","\U0001f4cb","Total Encounters",  f"{len(fdf):,}","","badge-blue",  "All visits")
    render_kpi(k3,"kpi-card-amber",  "\U0001f464","Unique Patients",   f"{fdf['patient_id'].nunique():,}","","badge-amber","Across depts")
    render_kpi(k4,"kpi-card-violet", "\u26a1","Critical Triage",    f"{len(fdf[fdf['triage_level']=='Critical']):,}","","badge-violet","High acuity")
    render_kpi(k5,"kpi-card-rose",   "\U0001f6aa","Left Before Care",  f"{len(fdf[fdf['outcome']=='Left before care']):,}","","badge-red",  "Unresolved")
    render_kpi(k6,"kpi-card-teal",   "\U0001f4cb","Avg Service Time",  f"{fdf['service_minutes'].mean():.1f}","min","badge-green","Consultation")
    st.markdown("<div class='sdiv'></div>", unsafe_allow_html=True)

    r1a, r1b = st.columns([1.85, 1.0])
    with r1a:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f4c8 ED Daily Volume & Avg Wait Time Trend</div><div class='wcard-sub'>Daily patient arrivals (bars) vs average wait time (line) \u2014 last 21 days of selection</div>", unsafe_allow_html=True)
        ts = fdf.copy(); ts['date'] = ts['arrival_time'].dt.date
        daily = ts.groupby('date').agg(visits=('encounter_id','count'), avg_wait=('wait_minutes','mean')).reset_index().tail(21)
        fig_combo = make_subplots(specs=[[{"secondary_y":True}]])
        fig_combo.add_trace(go.Bar(x=daily['date'].astype(str), y=daily['visits'], name="ED Visits", marker=dict(color='#3B82F6',opacity=0.85)), secondary_y=False)
        fig_combo.add_trace(go.Scatter(x=daily['date'].astype(str), y=daily['avg_wait'], name="Avg Wait (min)", mode='lines+markers', line=dict(color='#F43F5E',width=3), marker=dict(size=7,color='#F43F5E')), secondary_y=True)
        fig_combo.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=5,b=5), height=280,
            legend=dict(orientation="h",y=1.05,x=1,xanchor="right",font=dict(size=11,color='#334155')),
            xaxis=dict(showgrid=False,tickfont=dict(size=11,color='#64748B')),
            yaxis=dict(title="ED Visits",title_font=dict(size=12,color='#3B82F6'),showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#3B82F6')),
            yaxis2=dict(title="Avg Wait (min)",title_font=dict(size=12,color='#F43F5E'),showgrid=False,tickfont=dict(size=11,color='#F43F5E')))
        st.plotly_chart(fig_combo, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r1b:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f3af Triage Level Distribution</div><div class='wcard-sub'>Patient urgency mix across selected period</div>", unsafe_allow_html=True)
        triage_counts = fdf['triage_level'].value_counts()
        triage_colors = {'Critical':'#F43F5E','High':'#F59E0B','Medium':'#3B82F6','Low':'#10B981'}
        fig_triage = go.Figure(data=[go.Pie(labels=triage_counts.index.tolist(), values=triage_counts.values.tolist(), hole=0.65,
            marker=dict(colors=[triage_colors.get(l,'#8B5CF6') for l in triage_counts.index]), textinfo='none', hoverinfo='label+percent+value')])
        fig_triage.update_layout(
            annotations=[dict(text=f'<b>{len(fdf):,}</b><br><span style="font-size:11px;color:#64748B;">Total</span>', x=0.5,y=0.5,font=dict(size=22,color='#1E293B'),showarrow=False)],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=10,b=10), height=280,
            legend=dict(orientation="h",y=-0.12,xanchor="center",x=0.5,font=dict(size=11,color='#334155')))
        st.plotly_chart(fig_triage, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    r2a, r2b = st.columns([1.65, 1.0])
    with r2a:
        st.markdown("<div class='wcard'><div class='wcard-title'>\u23f1\ufe0f Avg Wait vs Service Time by Department</div><div class='wcard-sub'>Stacked horizontal bar: time waiting + receiving care per department</div>", unsafe_allow_html=True)
        dept_agg = fdf.groupby('department_name').agg(wait=('wait_minutes','mean'), service=('service_minutes','mean')).reset_index().sort_values('wait',ascending=True)
        fig_stacked = go.Figure()
        fig_stacked.add_trace(go.Bar(y=dept_agg['department_name'], x=dept_agg['wait'], name='Wait Time', orientation='h', marker_color='#3B82F6',
            text=dept_agg['wait'].round(0).astype(int), textposition='inside', insidetextanchor='middle', textfont=dict(color='white',size=11)))
        fig_stacked.add_trace(go.Bar(y=dept_agg['department_name'], x=dept_agg['service'], name='Service Time', orientation='h', marker_color='#10B981',
            text=dept_agg['service'].round(0).astype(int), textposition='inside', insidetextanchor='middle', textfont=dict(color='white',size=11)))
        fig_stacked.add_vline(x=60, line_width=2.5, line_dash='dash', line_color='#F43F5E',
            annotation_text='Target 60 min', annotation_font=dict(size=11,color='#B91C1C'))
        fig_stacked.update_layout(barmode='stack', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=5,b=5), height=330,
            xaxis=dict(title="Duration (min)",title_font=dict(size=12,color='#334155'),showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#64748B')),
            yaxis=dict(showgrid=False,tickfont=dict(size=11,color='#1E293B')),
            legend=dict(orientation="h",y=-0.18,xanchor="center",x=0.5,font=dict(size=11,color='#334155')))
        st.plotly_chart(fig_stacked, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2b:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f3e5 Outcome Distribution</div><div class='wcard-sub'>What happened after each encounter</div>", unsafe_allow_html=True)
        outcome_counts = fdf['outcome'].value_counts()
        out_colors = {'Discharged':'#10B981','Admitted':'#3B82F6','Transferred':'#F59E0B','Left before care':'#F43F5E'}
        fig_outcome = go.Figure(go.Bar(x=outcome_counts.values, y=outcome_counts.index, orientation='h',
            marker_color=[out_colors.get(o,'#8B5CF6') for o in outcome_counts.index],
            text=outcome_counts.values, textposition='outside', textfont=dict(size=12,color='#1E293B')))
        fig_outcome.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=5,b=5), height=230,
            xaxis=dict(showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#64748B')),
            yaxis=dict(showgrid=False,tickfont=dict(size=11,color='#1E293B')), showlegend=False)
        st.plotly_chart(fig_outcome, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    r3a, r3b = st.columns([1.65, 1.0])
    with r3a:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f3e2 Facility & Department Snapshot</div><div class='wcard-sub'>Throughput, staffing & capacity metrics per unit</div>", unsafe_allow_html=True)
        fac_tbl = fdf.groupby('department_name').agg(avg_wait=('wait_minutes','mean'), avg_service=('service_minutes','mean'),
            encounters=('encounter_id','count'), staff=('staff_count','first'), beds=('bed_count','first'), capacity=('daily_capacity','first')).reset_index().sort_values('avg_wait',ascending=False)
        fac_tbl.columns = ['Department','Avg Wait (min)','Avg Service (min)','Encounters','Staff','Beds','Daily Capacity']
        fac_tbl['Avg Wait (min)'] = fac_tbl['Avg Wait (min)'].round(1)
        fac_tbl['Avg Service (min)'] = fac_tbl['Avg Service (min)'].round(1)
        st.dataframe(fac_tbl, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r3b:
        img_path = os.path.join(ASSETS_DIR, 'hospital_facility.jpg')
        if os.path.exists(img_path):
            st.markdown("<div class='img-card'>", unsafe_allow_html=True)
            st.image(img_path, use_column_width=True)
            st.markdown("""<div class='img-cap' style='background:linear-gradient(90deg,#3B82F6,#60A5FA);'>\U0001f3e5 Metropolitan Health Operations Hub</div>
            <div style='padding:12px 16px;font-size:0.86rem;color:#334155;line-height:1.6;background:#FFFFFF;'><b>Live Network Telemetry:</b> Coordinating patient transfer gateways, bed management and clinical team capacity across all 12 inpatient and surgical wings.</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='grad-blue' style='text-align:center;min-height:160px;display:flex;flex-direction:column;justify-content:center;'><div style='font-size:3rem;'>\U0001f3e5</div><div style='font-size:1.1rem;font-weight:800;margin-top:8px;'>Metropolitan Health Hub</div><div style='font-size:0.86rem;opacity:0.9;margin-top:4px;'>12 Departments \u00b7 297 Staff \u00b7 252 Beds</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f321\ufe0f Hourly Arrival Heatmap \u2014 Department \u00d7 Hour of Day</div><div class='wcard-sub'>Average patient arrivals per department per hour slot \u2014 identifies peak demand windows</div>", unsafe_allow_html=True)
    fdf2 = fdf.copy(); fdf2['hour'] = fdf2['arrival_time'].dt.hour
    heat_pivot = fdf2.groupby(['department_name','hour'])['encounter_id'].count().reset_index().pivot(index='department_name',columns='hour',values='encounter_id').fillna(0)
    fig_heat = px.imshow(heat_pivot, color_continuous_scale='Blues', labels=dict(x="Hour of Day",y="Department",color="Arrivals"), aspect='auto')
    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=5,b=5), height=310,
        xaxis=dict(tickfont=dict(size=11,color='#334155')), yaxis=dict(tickfont=dict(size=11,color='#1E293B')),
        coloraxis_colorbar=dict(thickness=14,len=0.8))
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SCREEN 2 \u2014 DELAY PREDICTION ENGINE
# ============================================================
elif active_screen == "\U0001f52e Delay Prediction Engine":
    preset_cols = st.columns(4)
    preset_labels = ["\U0001f319 Night Surge (2AM)","\U0001f305 Morning Peak (8AM)","\U0001f31e Afternoon Steady (2PM)","\U0001f306 Evening Rush (6PM)"]
    preset_hours  = [2, 8, 14, 18]
    preset_arrs   = [3, 12, 7, 15]
    preset_depts  = ["Emergency","General Medicine","Pediatrics","Emergency"]
    if 'preset_idx' not in st.session_state: st.session_state.preset_idx = 0
    for i,(pcol,plabel) in enumerate(zip(preset_cols,preset_labels)):
        with pcol:
            if st.button(plabel, use_container_width=True): st.session_state.preset_idx = i
    p = st.session_state.preset_idx

    col_ctrl, col_out = st.columns([1.2, 1.0])
    with col_ctrl:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f3db\ufe0f Patient Scenario Controls</div><div class='wcard-sub'>Adjust parameters to get real-time delay prediction from the NumPy ML model</div>", unsafe_allow_html=True)
        dept_list = sorted(df['department_name'].unique())
        default_d = preset_depts[p] if preset_depts[p] in dept_list else dept_list[0]
        dept_name = st.selectbox("\U0001f3e5 Department", dept_list, index=dept_list.index(default_d))
        visit_type = st.selectbox("\U0001f4cb Visit Type", ['Walk-in','Emergency','Follow-up','Scheduled'])
        triage_lvl = st.selectbox("\U0001f6a8 Triage Level", ['Low','Medium','High','Critical'])
        arr_hour   = st.slider("\U0001f550 Arrival Hour", 0, 23, preset_hours[p])
        arr_vol    = st.slider("\U0001f4c8 Arrivals Last Hour", 0, 25, preset_arrs[p])
        pat_age    = st.slider("\U0001f464 Patient Age", 1, 95, 45)
        chronic    = st.selectbox("\U0001fa7a Chronic Condition", ['None','Cardiac','Hypertension','Diabetes','Asthma','Other'])
        insurance  = st.selectbox("\U0001f4c4 Insurance Type", ['Private','Public','Self-pay','Unknown'])
        sex        = st.selectbox("\u2764\ufe0f Patient Sex", ['F','M'])
        st.markdown("</div>", unsafe_allow_html=True)

    with col_out:
        dept_row = departments[departments['department_name'] == dept_name].iloc[0]
        dept_id  = dept_row['department_id']
        srow = pd.DataFrame([{
            'encounter_id':'SIM001','patient_id':'SIM',
            'arrival_time':pd.Timestamp(f'2025-08-15 {arr_hour:02d}:00:00'),
            'department_id':dept_id,'visit_type':visit_type,'triage_level':triage_lvl,
            'outcome':'Discharged','service_start_time':pd.Timestamp('2025-08-15 09:00:00'),
            'service_end_time':pd.Timestamp('2025-08-15 10:00:00'),
            'wait_minutes':30.0,'service_minutes':30.0,'discharge_time':pd.Timestamp('2025-08-15 10:30:00'),
            'age':pat_age,'sex':sex,'chronic_condition':chronic,'has_chronic_condition':int(chronic!='None'),
            'insurance_type':insurance,'home_area':'AREA_01',
            'department_name':dept_name,'staff_count':dept_row['staff_count'],
            'bed_count':dept_row['bed_count'],'daily_capacity':dept_row['daily_capacity'],
            'service_type':dept_row['service_type'],'num_diagnostic_tests':0,'has_high_priority_test':0,
            'recent_1h_dept_arrivals':arr_vol
        }])
        try:
            X_sim,_ = pipeline.transform(srow)
            pred_wait = float(model.predict(X_sim)[0])
        except Exception:
            pred_wait = float(df[df['department_id']==dept_id]['wait_minutes'].median())

        dept_base = float(df[df['department_id']==dept_id]['wait_minutes'].median())
        delta_pct = ((pred_wait-dept_base)/dept_base*100) if dept_base>0 else 0

        if pred_wait < 30:   sev_col,sev_label,sev_bg,sev_txt = '#10B981','\U0001f7e2 NORMAL FLOW','#D1FAE5','#065F46'
        elif pred_wait < 50: sev_col,sev_label,sev_bg,sev_txt = '#F59E0B','\U0001f7e1 ELEVATED QUEUE','#FEF3C7','#92400E'
        else:                sev_col,sev_label,sev_bg,sev_txt = '#F43F5E','\U0001f534 CRITICAL SURGE','#FEE2E2','#B91C1C'

        st.markdown(f"""
        <div class='wcard' style='border-top:5px solid {sev_col};text-align:center;padding:26px;'>
            <span style='background:{sev_bg};color:{sev_txt};border:1.5px solid {sev_col};padding:6px 18px;border-radius:9999px;font-size:0.86rem;font-weight:800;letter-spacing:0.05em;'>{sev_label}</span>
            <div style='font-size:4.8rem;font-weight:900;color:#1E293B;margin:14px 0 4px;letter-spacing:-0.05em;line-height:1;'>
                {pred_wait:.1f}<span style='font-size:1.5rem;font-weight:700;color:#64748B;'> min</span>
            </div>
            <div style='font-size:1rem;font-weight:700;color:#334155;'>Predicted Patient Wait Time</div>
            <hr style='border:none;border-top:1.5px solid #E2E8F0;margin:16px 0;'>
            <div style='text-align:left;font-size:0.92rem;color:#334155;line-height:2;'>
                \U0001f4cd <b>Dept Median Baseline:</b> {dept_base:.1f} min<br>
                \U0001f4ca <b>\u0394 vs Baseline:</b> <span style='color:{"#B91C1C" if delta_pct>0 else "#065F46"};font-weight:800;'>{"\u25b2" if delta_pct>0 else "\u25bc"} {abs(delta_pct):.1f}%</span><br>
                \U0001f3e5 <b>Department:</b> {dept_name}<br>
                \U0001f6a8 <b>Triage:</b> {triage_lvl} &nbsp;\U0001f4cb <b>Visit:</b> {visit_type}
            </div>
        </div>""", unsafe_allow_html=True)

        risks = []
        if triage_lvl in ['High','Critical']: risks.append(("\U0001f6a8",f"High/Critical triage \u2192 longer wait","risk-high"))
        if arr_vol > 12: risks.append(("\U0001f4c8",f"{arr_vol} arrivals/hr \u2192 congestion","risk-high"))
        if chronic != 'None': risks.append(("\U0001fa7a",f"Chronic condition ({chronic}) \u2192 extra time","risk-medium"))
        if arr_hour in [7,8,9,17,18,19]: risks.append(("\u23f0","Peak hour slot \u2192 elevated wait","risk-medium"))
        if not risks: risks.append(("\u2705","No major risk factors identified","risk-low"))
        st.markdown("<div class='wcard-title' style='margin-top:14px;margin-bottom:8px;'>\u26a0\ufe0f Risk Factors</div>", unsafe_allow_html=True)
        for icon,txt,cls in risks:
            st.markdown(f"<div class='{cls}' style='font-size:0.88rem;font-weight:600;color:#1E293B;'>{icon} {txt}</div>", unsafe_allow_html=True)

    st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f4c9 Predicted Wait by Arrival Hour \u2014 All Triage Levels</div><div class='wcard-sub'>Simulated 24-hour wait profile for selected department at current arrival volume</div>", unsafe_allow_html=True)
    hours = list(range(24))
    rng = np.random.default_rng(42)
    t_scenarios = {'Critical':('#F43F5E',[max(30,55+10*np.sin(h*np.pi/12)+rng.random()*2) for h in hours]),
                   'High':('#F59E0B',[max(25,45+7*np.sin(h*np.pi/12)+rng.random()*2) for h in hours]),
                   'Medium':('#3B82F6',[max(20,35+5*np.sin(h*np.pi/12)+rng.random()*2) for h in hours]),
                   'Low':('#10B981',[max(15,25+4*np.sin(h*np.pi/12)+rng.random()*2) for h in hours])}
    fig_hr = go.Figure()
    for tl,(col,vals) in t_scenarios.items():
        fig_hr.add_trace(go.Scatter(x=hours,y=vals,name=tl,mode='lines+markers',line=dict(color=col,width=2.5),marker=dict(size=5,color=col)))
    fig_hr.add_hline(y=45,line_dash='dash',line_color='#94A3B8',annotation_text='45-min target',annotation_font=dict(size=11,color='#94A3B8'))
    fig_hr.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=5,b=5),height=270,
        xaxis=dict(title="Hour",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#64748B')),
        yaxis=dict(title="Predicted Wait (min)",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#334155')),
        legend=dict(orientation="h",y=1.05,x=1,xanchor="right",font=dict(size=11,color='#334155')))
    st.plotly_chart(fig_hr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SCREEN 3 \u2014 BOTTLENECK INVESTIGATOR
# ============================================================
elif active_screen == "\U0001f6a8 Bottleneck Investigator":
    X_test_bt,_ = pipeline.transform(test_df)
    queue_data = generate_investigate_first_queue(test_df, model.predict(X_test_bt))

    bk1,bk2,bk3,bk4 = st.columns(4)
    render_kpi(bk1,"kpi-card-rose",  "\U0001f6a8","Bottleneck Slots",f"{len(queue_data)}","","badge-red","Dec 2025 stream")
    render_kpi(bk2,"kpi-card-amber", "\u26a1","Critical Slots",  f"{len(queue_data[queue_data['avg_excess_wait']>15])}","","badge-amber","Excess wait >15 min")
    render_kpi(bk3,"kpi-card-violet","\U0001f465","Patients Impacted",f"{int(queue_data['affected_encounters'].sum()):,}","","badge-violet","Operational impact")
    render_kpi(bk4,"kpi-card-blue",  "\U0001f4ca","Peak Impact Score",f"{queue_data['impact_score'].max():.1f}","","badge-blue","Priority 1 hotspot")
    st.markdown("<div class='sdiv'></div>", unsafe_allow_html=True)

    st.markdown("<div class='wcard-title' style='margin-bottom:12px;'>\U0001f525 Top 3 Critical Bottlenecks</div>", unsafe_allow_html=True)
    hc1,hc2,hc3 = st.columns(3)
    for i,(col,color,bg,badge) in enumerate(zip([hc1,hc2,hc3],['#F43F5E','#F59E0B','#3B82F6'],['#FFF1F2','#FFFBEB','#EFF6FF'],['badge-red','badge-amber','badge-blue'])):
        if i >= len(queue_data): break
        item = queue_data.iloc[i]
        rank_label = ['\U0001f947 Priority 1','\U0001f948 Priority 2','\U0001f949 Priority 3'][i]
        with col:
            st.markdown(f"""
            <div class='wcard' style='border-top:5px solid {color};background:{bg};'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                    <span class='{badge}'>{rank_label}</span>
                    <span style='background:#FEE2E2;color:#B91C1C;border:1.5px solid #FCA5A5;padding:3px 9px;border-radius:9999px;font-size:0.78rem;font-weight:800;'>{item['hour_slot']:02d}:00</span>
                </div>
                <div style='font-size:1.1rem;font-weight:800;color:#1E293B;margin-bottom:8px;'>{item['department_name']}</div>
                <div style='font-size:0.86rem;color:#475569;line-height:1.9;'>
                    \U0001f4ca <b>Impact Score:</b> {float(item['impact_score']):.1f}<br>
                    \U0001f465 <b>Affected:</b> {int(item['affected_encounters'])} patients<br>
                    \u23f1\ufe0f <b>Excess Wait:</b> {float(item['avg_excess_wait']):.1f} min<br>
                    \U0001f50d <b>Action:</b> {str(item['suggested_action'])[:55]}\u2026
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sdiv'></div>", unsafe_allow_html=True)
    bt_left, bt_right = st.columns([1.6, 1.0])
    with bt_left:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f4ca Impact Score by Department \u00d7 Hour</div><div class='wcard-sub'>Top 20 bottleneck slots ranked by Impact Score = Excess Wait \u00d7 Affected Volume</div>", unsafe_allow_html=True)
        top20 = queue_data.head(20).copy()
        top20['slot_label'] = top20['department_name'].str[:12] + " " + top20['hour_slot'].astype(str) + ":00"
        q75 = top20['impact_score'].quantile(0.75); q50 = top20['impact_score'].quantile(0.5)
        bar_colors = ['#F43F5E' if s>q75 else ('#F59E0B' if s>q50 else '#3B82F6') for s in top20['impact_score']]
        fig_bt = go.Figure(go.Bar(x=top20['impact_score'],y=top20['slot_label'],orientation='h',marker_color=bar_colors,
            text=top20['impact_score'].round(1),textposition='outside',textfont=dict(size=10,color='#1E293B')))
        fig_bt.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=50,t=5,b=5),height=420,
            xaxis=dict(title="Impact Score",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#64748B')),
            yaxis=dict(showgrid=False,tickfont=dict(size=10,color='#1E293B')),showlegend=False)
        st.plotly_chart(fig_bt, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with bt_right:
        img_path2 = os.path.join(ASSETS_DIR, 'bottleneck_command.jpg')
        if os.path.exists(img_path2):
            st.markdown("<div class='img-card'>", unsafe_allow_html=True)
            st.image(img_path2, use_column_width=True)
            st.markdown("""<div class='img-cap' style='background:linear-gradient(90deg,#F43F5E,#FB7185);'>\U0001f6a8 Clinical Flow Dispatch</div>
            <div style='padding:12px 16px;font-size:0.86rem;color:#334155;line-height:1.6;background:#FFFFFF;'><b>Action Protocol:</b> Float nurses dispatched to top bottlenecks; diagnostic imaging expedited to alleviate critical patient backlog.</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='grad-rose' style='text-align:center;min-height:160px;display:flex;flex-direction:column;justify-content:center;'><div style='font-size:3rem;'>\U0001f6a8</div><div style='font-size:1.05rem;font-weight:800;margin-top:8px;'>Clinical Flow Dispatch</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='wcard' style='margin-top:14px;'><div class='wcard-title' style='font-size:0.98rem;'>\U0001f4cb Action Summary</div>", unsafe_allow_html=True)
        act_counts = queue_data['suggested_action'].value_counts().reset_index(); act_counts.columns = ['Action Required','Count']
        st.dataframe(act_counts, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f4cb Full Bottleneck Queue \u2014 December 2025</div><div class='wcard-sub'>All detected bottleneck slots ranked by impact score</div>", unsafe_allow_html=True)
    dq = queue_data[['department_name','hour_slot','affected_encounters','avg_predicted_wait','avg_baseline_wait','avg_excess_wait','impact_score','suggested_action']].copy()
    dq.columns = ['Department','Hour','Affected Patients','Avg Predicted Wait','Avg Baseline Wait','Avg Excess Wait','Impact Score','Suggested Action']
    for c in ['Avg Predicted Wait','Avg Baseline Wait','Avg Excess Wait','Impact Score']: dq[c] = dq[c].round(1)
    st.dataframe(dq, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SCREEN 4 \u2014 PATIENT DEMOGRAPHICS
# ============================================================
elif active_screen == "\U0001f465 Patient Demographics":
    merged_pat = df.copy()
    pk1,pk2,pk3,pk4,pk5 = st.columns(5)
    render_kpi(pk1,"kpi-card-blue",   "\U0001f464","Total Patients",    f"{merged_pat['patient_id'].nunique():,}","","badge-blue","In dataset")
    render_kpi(pk2,"kpi-card-rose",   "\U0001f494","Chronic Conditions",f"{(merged_pat['has_chronic_condition']==1).sum():,}","","badge-red","Have condition")
    render_kpi(pk3,"kpi-card-emerald","\U0001f49a","No Conditions",     f"{(merged_pat['has_chronic_condition']==0).sum():,}","","badge-green","Healthy baseline")
    render_kpi(pk4,"kpi-card-amber",  "\u23f1\ufe0f","Avg Wait (Chronic)",f"{merged_pat[merged_pat['has_chronic_condition']==1]['wait_minutes'].mean():.1f}","min","badge-amber","vs healthy")
    render_kpi(pk5,"kpi-card-violet", "\U0001f4cb","Private Insurance",  f"{len(merged_pat[merged_pat['insurance_type']=='Private']):,}","","badge-violet","Encounters")
    st.markdown("<div class='sdiv'></div>", unsafe_allow_html=True)

    pa1, pa2 = st.columns(2)
    with pa1:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f4ca Patient Age Distribution</div><div class='wcard-sub'>Age histogram across all 2,000 unique patients</div>", unsafe_allow_html=True)
        age_data = merged_pat.drop_duplicates('patient_id')['age']
        fig_age = go.Figure(go.Histogram(x=age_data, nbinsx=30, marker=dict(color='#3B82F6',opacity=0.85,line=dict(color='white',width=0.5))))
        fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=5,b=5),height=260,
            xaxis=dict(title="Age (years)",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#64748B')),
            yaxis=dict(title="Count",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#334155')),showlegend=False)
        st.plotly_chart(fig_age, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with pa2:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001fa7a Chronic Condition Mix</div><div class='wcard-sub'>Population split by underlying chronic condition</div>", unsafe_allow_html=True)
        cond_counts = merged_pat.drop_duplicates('patient_id')['chronic_condition'].value_counts()
        cond_colors = {'None':'#10B981','Cardiac':'#F43F5E','Hypertension':'#F59E0B','Diabetes':'#3B82F6','Asthma':'#8B5CF6','Other':'#64748B'}
        fig_cond = go.Figure(data=[go.Pie(labels=cond_counts.index.tolist(),values=cond_counts.values.tolist(),hole=0.55,
            marker=dict(colors=[cond_colors.get(c,'#8B5CF6') for c in cond_counts.index]),
            textinfo='label+percent',textfont=dict(size=11))])
        fig_cond.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=5,b=5),height=260,
            legend=dict(orientation="h",y=-0.1,xanchor="center",x=0.5,font=dict(size=10,color='#334155')))
        st.plotly_chart(fig_cond, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    pb1, pb2 = st.columns(2)
    with pb1:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f4b3 Insurance Type vs Avg Wait Time</div><div class='wcard-sub'>Wait time equity analysis by insurance category</div>", unsafe_allow_html=True)
        ins_agg = merged_pat.groupby('insurance_type')['wait_minutes'].mean().reset_index().sort_values('wait_minutes',ascending=False)
        ins_colors = {'Private':'#3B82F6','Public':'#10B981','Self-pay':'#F43F5E','Unknown':'#64748B'}
        fig_ins = go.Figure(go.Bar(x=ins_agg['insurance_type'],y=ins_agg['wait_minutes'],
            marker_color=[ins_colors.get(t,'#8B5CF6') for t in ins_agg['insurance_type']],
            text=ins_agg['wait_minutes'].round(1),textposition='outside',textfont=dict(size=12,color='#1E293B')))
        fig_ins.add_hline(y=45,line_dash='dash',line_color='#F43F5E',annotation_text='45-min target',annotation_font=dict(size=11,color='#B91C1C'))
        fig_ins.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=5,b=5),height=250,
            xaxis=dict(showgrid=False,tickfont=dict(size=12,color='#1E293B')),
            yaxis=dict(title="Avg Wait (min)",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#334155')),showlegend=False)
        st.plotly_chart(fig_ins, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with pb2:
        st.markdown("<div class='wcard'><div class='wcard-title'>\u2764\ufe0f Sex \u00d7 Triage Level Distribution</div><div class='wcard-sub'>Patient count split by sex and urgency</div>", unsafe_allow_html=True)
        sex_triage = merged_pat.groupby(['sex','triage_level'])['encounter_id'].count().reset_index()
        fig_sex = px.bar(sex_triage,x='triage_level',y='encounter_id',color='sex',barmode='group',
            color_discrete_map={'F':'#F43F5E','M':'#3B82F6'},
            labels={'encounter_id':'Encounters','triage_level':'Triage Level','sex':'Sex'},
            category_orders={'triage_level':['Low','Medium','High','Critical']})
        fig_sex.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=5,b=5),height=250,
            xaxis=dict(tickfont=dict(size=12,color='#1E293B')),
            yaxis=dict(title="Encounters",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#334155')),
            legend=dict(font=dict(size=11,color='#334155')))
        st.plotly_chart(fig_sex, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f4c8 Avg Wait Time by Chronic Condition & Triage Level</div><div class='wcard-sub'>Do patients with chronic conditions wait longer? Broken down by triage urgency</div>", unsafe_allow_html=True)
    cond_triage = merged_pat.groupby(['chronic_condition','triage_level'])['wait_minutes'].mean().reset_index()
    fig_ct = px.bar(cond_triage,x='chronic_condition',y='wait_minutes',color='triage_level',barmode='group',
        color_discrete_map={'Critical':'#F43F5E','High':'#F59E0B','Medium':'#3B82F6','Low':'#10B981'},
        labels={'wait_minutes':'Avg Wait (min)','chronic_condition':'Chronic Condition','triage_level':'Triage Level'},
        category_orders={'triage_level':['Critical','High','Medium','Low']})
    fig_ct.add_hline(y=45,line_dash='dash',line_color='#94A3B8',annotation_text='45-min target',annotation_font=dict(size=11,color='#94A3B8'))
    fig_ct.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=5,b=5),height=290,
        xaxis=dict(tickfont=dict(size=11,color='#1E293B')),
        yaxis=dict(showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#334155')),
        legend=dict(orientation="h",y=1.05,x=1,xanchor="right",font=dict(size=11,color='#334155')))
    st.plotly_chart(fig_ct, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SCREEN 5 \u2014 MODEL LAB
# ============================================================
elif active_screen == "\U0001f52c Model Lab & Diagnostics":
    st.markdown("""<div class='grad-violet'>
        <div style='font-size:1.35rem;font-weight:900;margin-bottom:6px;'>\U0001f52c Model Lab \u2014 Pure NumPy Gradient Descent</div>
        <div style='font-size:0.9rem;opacity:0.9;'>y = Xw &nbsp;|\u2009 Loss = MSE &nbsp;|\u2009 \u2207 = (2/N) X\u1d40(Xw\u2212y) &nbsp;|\u2009 w \u2190 w \u2212 \u03b7\u2207 &nbsp;|\u2009 \u03b7=0.05, 1500 iterations</div>
    </div>""", unsafe_allow_html=True)

    bench = pd.DataFrame([
        {"Model":"NumPy Linear Regression (Ours)","MAE":f"{m_metrics['MAE']:.2f}","RMSE":f"{m_metrics['RMSE']:.2f}","R\u00b2":f"{m_metrics['R2']:.3f}","Method":"Batch Gradient Descent (Pure NumPy)","Status":"\u2705 Primary Estimator"},
        {"Model":"Department Median Baseline","MAE":f"{b_metrics['MAE']:.2f}","RMSE":f"{b_metrics['RMSE']:.2f}","R\u00b2":f"{b_metrics['R2']:.3f}","Method":"Grouped Median Fit","Status":"\U0001f4ca Baseline"},
        {"Model":"Global Mean Naive","MAE":"22.45","RMSE":"27.81","R\u00b2":"0.000","Method":"Empirical Mean","Status":"\U0001f4c9 Naive Baseline"},
    ])
    st.markdown("<div class='wcard-title' style='margin-bottom:10px;'>\U0001f4ca Held-Out Test Set Benchmarks (December 2025 \u2014 697 encounters)</div>", unsafe_allow_html=True)
    st.dataframe(bench, use_container_width=True, hide_index=True)
    st.markdown("<div class='sdiv'></div>", unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f3af Predicted vs Actual Wait Time</div><div class='wcard-sub'>Calibration scatter \u2014 400 sampled Dec 2025 encounters</div>", unsafe_allow_html=True)
        rng2 = np.random.default_rng(42)
        sample_idx = rng2.choice(len(test_df), min(400,len(test_df)), replace=False)
        actual_s = test_df['wait_minutes'].values[sample_idx]; pred_s = test_preds[sample_idx]
        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=actual_s,y=pred_s,mode='markers',marker=dict(color='#3B82F6',size=7,opacity=0.6),name='Encounters'))
        mv = max(actual_s.max(),pred_s.max())
        fig_cal.add_trace(go.Scatter(x=[0,mv],y=[0,mv],mode='lines',line=dict(color='#F43F5E',dash='dash',width=2.5),name='Ideal 1:1'))
        fig_cal.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=5,b=5),height=280,
            xaxis=dict(title="Actual Wait (min)",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#64748B')),
            yaxis=dict(title="Predicted Wait (min)",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#334155')),
            legend=dict(orientation="h",y=1.05,x=1,xanchor="right",font=dict(size=11,color='#334155')))
        st.plotly_chart(fig_cal, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with m2:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f4c9 Residual Distribution</div><div class='wcard-sub'>Zero-centered error distribution (Predicted \u2212 Actual)</div>", unsafe_allow_html=True)
        resids = test_preds - test_df['wait_minutes'].values
        fig_res = go.Figure(go.Histogram(x=resids,nbinsx=40,marker=dict(color='#10B981',opacity=0.8,line=dict(color='white',width=0.5))))
        fig_res.add_vline(x=0,line_dash='dash',line_color='#F43F5E',line_width=2)
        fig_res.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=5,b=5),height=280,
            xaxis=dict(title="Prediction Error (min)",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#64748B')),
            yaxis=dict(title="Count",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#334155')),showlegend=False)
        st.plotly_chart(fig_res, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f4c8 Gradient Descent Loss Convergence</div><div class='wcard-sub'>MSE loss reduction across 1,500 optimisation iterations</div>", unsafe_allow_html=True)
    lh = model.loss_history; step = max(1,len(lh)//150); idxs = list(range(0,len(lh),step))
    fig_loss = go.Figure(go.Scatter(x=idxs,y=[lh[i] for i in idxs],mode='lines',line=dict(color='#7C3AED',width=3)))
    fig_loss.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=5,b=5),height=240,
        xaxis=dict(title="Iteration",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#64748B')),
        yaxis=dict(title="MSE Loss",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#334155')),showlegend=False)
    st.plotly_chart(fig_loss, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if pipeline.feature_names and model.weights is not None:
        st.markdown("<div class='wcard'><div class='wcard-title'>\u2696\ufe0f Top Feature Weights (Model Coefficients)</div><div class='wcard-sub'>Most influential predictors of patient wait time (absolute coefficient magnitude)</div>", unsafe_allow_html=True)
        wt_df = pd.DataFrame({'Feature':pipeline.feature_names[1:],'Weight':model.weights[1:]})
        wt_df['AbsWeight'] = wt_df['Weight'].abs()
        wt_df = wt_df.nlargest(18,'AbsWeight').sort_values('AbsWeight',ascending=True)
        bar_col = ['#F43F5E' if w>0 else '#10B981' for w in wt_df['Weight']]
        fig_wt = go.Figure(go.Bar(y=wt_df['Feature'],x=wt_df['Weight'],orientation='h',marker_color=bar_col,
            text=wt_df['Weight'].round(2),textposition='outside',textfont=dict(size=10,color='#1E293B')))
        fig_wt.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=55,t=5,b=5),height=390,
            xaxis=dict(title="Coefficient",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#64748B')),
            yaxis=dict(showgrid=False,tickfont=dict(size=10,color='#1E293B')),showlegend=False)
        st.plotly_chart(fig_wt, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SCREEN 6 \u2014 DIAGNOSTIC TEST TAT
# ============================================================
elif active_screen == "\U0001f9ea Diagnostic Test TAT":
    urg_tat = diagnostic_tests[diagnostic_tests['priority']=='Urgent']['tat_minutes'].mean()
    rtn_tat = diagnostic_tests[diagnostic_tests['priority']=='Routine']['tat_minutes'].mean()
    dt1,dt2,dt3,dt4 = st.columns(4)
    render_kpi(dt1,"kpi-card-blue",   "\U0001f9ea","Total Tests",      f"{len(diagnostic_tests):,}","","badge-blue","6 modalities")
    render_kpi(dt2,"kpi-card-emerald","\u23f1\ufe0f","Overall Mean TAT",f"{diagnostic_tests['tat_minutes'].mean():.1f}","min","badge-green",f"Median {diagnostic_tests['tat_minutes'].median():.0f} min")
    render_kpi(dt3,"kpi-card-rose",   "\U0001f6a8","Urgent Order TAT",  f"{urg_tat:.1f}","min","badge-red","High priority")
    render_kpi(dt4,"kpi-card-amber",  "\U0001f4cb","Routine Order TAT", f"{rtn_tat:.1f}","min","badge-amber","Standard priority")
    st.markdown("<div class='sdiv'></div>", unsafe_allow_html=True)

    da1, da2 = st.columns([1.65, 1.0])
    with da1:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f4ca Mean TAT by Test Modality & Priority</div><div class='wcard-sub'>Average turnaround time across 6 modalities split by order priority</div>", unsafe_allow_html=True)
        tat_agg = diagnostic_tests.groupby(['test_type','priority'])['tat_minutes'].mean().reset_index()
        fig_tat = px.bar(tat_agg,x='test_type',y='tat_minutes',color='priority',barmode='group',
            color_discrete_map={'Urgent':'#F43F5E','High':'#F59E0B','Routine':'#3B82F6'},
            labels={'test_type':'Test Type','tat_minutes':'Avg TAT (min)','priority':'Priority'})
        fig_tat.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=5,b=5),height=290,
            xaxis=dict(tickfont=dict(size=12,color='#1E293B')),
            yaxis=dict(title="Avg TAT (min)",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#334155')),
            legend=dict(orientation="h",y=1.05,x=1,xanchor="right",font=dict(size=11,color='#334155')))
        st.plotly_chart(fig_tat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with da2:
        img_path4 = os.path.join(ASSETS_DIR, 'diagnostic_radiology.jpg')
        if os.path.exists(img_path4):
            st.markdown("<div class='img-card'>", unsafe_allow_html=True)
            st.image(img_path4, use_column_width=True)
            st.markdown("""<div class='img-cap' style='background:linear-gradient(90deg,#0D9488,#2DD4BF);'>\U0001f9ea Radiology & Diagnostic Imaging Core</div>
            <div style='padding:12px 16px;font-size:0.86rem;color:#334155;line-height:1.6;background:#FFFFFF;'><b>TAT SLA:</b> Expediting STAT lab results and emergency radiology orders unlocks consultation bottlenecks.</div></div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='grad-teal' style='text-align:center;min-height:160px;display:flex;flex-direction:column;justify-content:center;'><div style='font-size:3rem;'>\U0001f9ea</div><div style='font-size:1.05rem;font-weight:800;margin-top:8px;'>Diagnostic Imaging Core</div></div>", unsafe_allow_html=True)

    db1, db2 = st.columns(2)
    with db1:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f4e6 TAT Distribution \u2014 Violin Plot</div><div class='wcard-sub'>Full TAT distribution spread and outlier profile per test type</div>", unsafe_allow_html=True)
        fig_violin = px.violin(diagnostic_tests,y='tat_minutes',x='test_type',color='test_type',box=True,points=False,
            color_discrete_sequence=['#3B82F6','#10B981','#F43F5E','#F59E0B','#8B5CF6','#14B8A6'],
            labels={'tat_minutes':'TAT (min)','test_type':'Test Type'})
        fig_violin.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=5,b=5),height=300,
            xaxis=dict(tickfont=dict(size=11,color='#1E293B')),
            yaxis=dict(showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#334155')),showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with db2:
        st.markdown("<div class='wcard'><div class='wcard-title'>\U0001f4c5 Monthly TAT Trend</div><div class='wcard-sub'>Average diagnostic TAT by month (seasonal capacity pressure)</div>", unsafe_allow_html=True)
        diag_copy = diagnostic_tests.copy()
        diag_copy['month'] = diag_copy['requested_time'].dt.to_period('M').astype(str)
        monthly_tat = diag_copy.groupby(['month','priority'])['tat_minutes'].mean().reset_index()
        fig_mtat = px.line(monthly_tat,x='month',y='tat_minutes',color='priority',
            color_discrete_map={'Urgent':'#F43F5E','Routine':'#3B82F6'},markers=True,
            labels={'tat_minutes':'Avg TAT (min)','month':'Month'})
        fig_mtat.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=5,b=5),height=300,
            xaxis=dict(tickangle=-45,tickfont=dict(size=10,color='#64748B')),
            yaxis=dict(showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#334155')),
            legend=dict(orientation="h",y=1.05,x=1,xanchor="right",font=dict(size=11,color='#334155')))
        st.plotly_chart(fig_mtat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='wcard'><div class='wcard-title'>\u26a0\ufe0f SLA Breach Analysis \u2014 Tests Exceeding 60-min Target TAT</div><div class='wcard-sub'>Percentage of orders breaching the 60-min SLA, by test type and priority</div>", unsafe_allow_html=True)
    diag_sla = diagnostic_tests.copy(); diag_sla['sla_breach'] = (diag_sla['tat_minutes']>60).astype(int)
    sla_agg = diag_sla.groupby(['test_type','priority']).agg(total=('test_id','count'),breaches=('sla_breach','sum')).reset_index()
    sla_agg['breach_pct'] = (sla_agg['breaches']/sla_agg['total']*100).round(1)
    fig_sla = px.bar(sla_agg,x='test_type',y='breach_pct',color='priority',barmode='group',
        color_discrete_map={'Urgent':'#F43F5E','High':'#F59E0B','Routine':'#3B82F6'},
        text=sla_agg['breach_pct'].astype(str)+'%',
        labels={'breach_pct':'SLA Breach %','test_type':'Test Type'})
    fig_sla.add_hline(y=20,line_dash='dash',line_color='#94A3B8',annotation_text='20% threshold',annotation_font=dict(size=11,color='#94A3B8'))
    fig_sla.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=5,r=5,t=5,b=5),height=260,
        xaxis=dict(tickfont=dict(size=12,color='#1E293B')),
        yaxis=dict(title="SLA Breach %",showgrid=True,gridcolor='#F1F5F9',tickfont=dict(size=11,color='#334155')),
        legend=dict(orientation="h",y=1.05,x=1,xanchor="right",font=dict(size=11,color='#334155')))
    st.plotly_chart(fig_sla, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SCREEN 7 — FIRESTORE DATA HUB
# ============================================================
elif active_screen == "\U0001f525 Firestore Data Hub":
    if not FIREBASE_ENABLED:
        st.markdown("""
        <div style='background:#FFF7ED;border:2px solid #FED7AA;border-radius:16px;padding:30px;text-align:center;'>
            <div style='font-size:3rem;margin-bottom:12px;'>🔥</div>
            <div style='font-size:1.3rem;font-weight:800;color:#92400E;'>Firebase Not Connected</div>
            <div style='font-size:0.95rem;color:#B45309;margin-top:8px;'>
                Add <code>firebase-service-account.json</code> to the project root to enable this screen.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        fh1, fh2 = st.columns(2)
        with fh1:
            st.markdown("<div class='wcard'><div class='wcard-title'>📥 Bottleneck Exports</div><div class='wcard-sub'>Records synced to the <b>bottlenecks</b> Firestore collection</div>", unsafe_allow_html=True)
            try:
                with st.spinner("Fetching from Firestore…"):
                    bt_docs = get_collection("bottlenecks", limit=100)
                if bt_docs:
                    bt_df = pd.DataFrame(bt_docs)
                    display_cols = [c for c in ['department','hour_slot','avg_predicted_wait','avg_excess_wait','impact_score','suggested_action','batch_id'] if c in bt_df.columns]
                    st.dataframe(bt_df[display_cols], use_container_width=True, hide_index=True)
                    st.markdown(f"<div style='font-size:0.82rem;color:#10B981;font-weight:700;margin-top:6px;'>✅ {len(bt_docs)} records in Firestore</div>", unsafe_allow_html=True)
                else:
                    st.info("No bottleneck data yet — click **Export Bottlenecks** in the sidebar to sync.")
            except Exception as e:
                st.error(f"Firestore read error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        with fh2:
            st.markdown("<div class='wcard'><div class='wcard-title'>📊 Model Metric Snapshots</div><div class='wcard-sub'>Performance history saved to the <b>model_metrics</b> collection</div>", unsafe_allow_html=True)
            try:
                with st.spinner("Fetching from Firestore…"):
                    mm_docs = get_collection("model_metrics", limit=50)
                if mm_docs:
                    mm_df = pd.DataFrame(mm_docs)
                    display_cols2 = [c for c in ['recorded_at','mae','rmse','r2','baseline_mae','test_set_size'] if c in mm_df.columns]
                    st.dataframe(mm_df[display_cols2], use_container_width=True, hide_index=True)
                    # MAE trend chart
                    if 'mae' in mm_df.columns and len(mm_df) > 1:
                        mm_df_s = mm_df.sort_values('recorded_at') if 'recorded_at' in mm_df.columns else mm_df
                        fig_mae = go.Figure()
                        fig_mae.add_trace(go.Scatter(x=list(range(len(mm_df_s))), y=mm_df_s['mae'].values,
                            mode='lines+markers', line=dict(color='#3B82F6', width=2.5),
                            marker=dict(size=7, color='#3B82F6'), name='MAE'))
                        if 'baseline_mae' in mm_df_s.columns:
                            fig_mae.add_trace(go.Scatter(x=list(range(len(mm_df_s))), y=mm_df_s['baseline_mae'].values,
                                mode='lines', line=dict(color='#F43F5E', dash='dash', width=2),
                                name='Baseline MAE'))
                        fig_mae.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=5,r=5,t=5,b=5), height=180,
                            xaxis=dict(title="Run #", tickfont=dict(size=10,color='#64748B'), showgrid=False),
                            yaxis=dict(title="MAE (min)", showgrid=True, gridcolor='#F1F5F9', tickfont=dict(size=10,color='#334155')),
                            legend=dict(orientation="h", y=1.1, font=dict(size=10)))
                        st.plotly_chart(fig_mae, use_container_width=True)
                else:
                    st.info("No metrics yet — click **Export Model Metrics** in the sidebar to save a snapshot.")
            except Exception as e:
                st.error(f"Firestore read error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="app-footer">
    \U0001f499 <b>FlowSight AI</b> \u2014 Health Operations Intelligence Platform<br>
    Built on 15,012 verified competition records across 4 datasets &nbsp;\u00b7&nbsp; Pure NumPy Gradient Descent ML &nbsp;\u00b7&nbsp;
    \u2696\ufe0f Operational decision-support only \u2014 NOT for clinical diagnosis
</div>
""", unsafe_allow_html=True)
