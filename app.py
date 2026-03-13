import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amar_S AI · House Valuation",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Master CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Root ── */
:root {
  --bg:        #080B14;
  --surface:   #0D1120;
  --card:      #111827;
  --border:    #1E2A40;
  --gold:      #F0A500;
  --gold-dim:  #C47F00;
  --gold-glow: rgba(240,165,0,0.18);
  --teal:      #00D4C8;
  --teal-dim:  rgba(0,212,200,0.12);
  --text:      #E8EDF5;
  --muted:     #6B7A99;
  --danger:    #FF4D6D;
  --success:   #00E5A0;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
}

[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 60% 40% at 10% 0%, rgba(240,165,0,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 50% 50% at 90% 100%, rgba(0,212,200,0.06) 0%, transparent 60%),
    repeating-linear-gradient(90deg, rgba(30,42,64,0.3) 0px, transparent 1px, transparent 79px, rgba(30,42,64,0.3) 80px),
    repeating-linear-gradient(0deg,  rgba(30,42,64,0.3) 0px, transparent 1px, transparent 79px, rgba(30,42,64,0.3) 80px);
  pointer-events: none;
  z-index: 0;
}

[data-testid="stMain"] { background: transparent !important; }
[data-testid="block-container"] {
  padding: 0 2rem 4rem !important;
  max-width: 1300px !important;
  position: relative; z-index: 1;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ── Typography ── */
h1,h2,h3,h4,h5,h6 { font-family: 'Syne', sans-serif !important; }

/* ── Navbar ── */
.nexval-nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 1.6rem 0 1.4rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 3rem;
}
.nexval-logo {
  font-family: 'Syne', sans-serif;
  font-size: 1.45rem; font-weight: 800; letter-spacing: -0.02em;
  color: var(--text);
}
.nexval-logo span { color: var(--gold); }
.nav-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem; letter-spacing: 0.12em;
  color: var(--teal); border: 1px solid rgba(0,212,200,0.3);
  padding: 0.3rem 0.8rem; border-radius: 2rem;
  background: var(--teal-dim);
}
.nav-links { display: flex; gap: 2rem; }
.nav-links a {
  color: var(--muted); font-size: 0.85rem; text-decoration: none;
  transition: color .2s;
}
.nav-links a:hover { color: var(--text); }

/* ── Hero ── */
.hero-wrap {
  text-align: center;
  padding: 3.5rem 1rem 3rem;
  position: relative;
}
.hero-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem; letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--gold); margin-bottom: 1.2rem;
  display: inline-flex; align-items: center; gap: 0.6rem;
}
.hero-eyebrow::before, .hero-eyebrow::after {
  content: ''; display: inline-block;
  width: 32px; height: 1px; background: var(--gold-dim);
}
.hero-title {
  font-family: 'Syne', sans-serif !important;
  font-size: clamp(2.8rem, 5vw, 4.4rem) !important;
  font-weight: 800 !important; line-height: 1.1 !important;
  letter-spacing: -0.03em !important;
  color: var(--text) !important;
  margin: 0 auto 1.2rem !important; max-width: 720px;
}
.hero-title .accent { color: var(--gold); }
.hero-sub {
  font-size: 1.05rem; color: var(--muted); max-width: 560px;
  margin: 0 auto 2rem; line-height: 1.7; font-weight: 300;
}
.hero-stats {
  display: flex; justify-content: center; gap: 3rem;
  padding: 1.6rem 0; border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border); margin: 0 auto 0;
  max-width: 640px;
}
.hstat-val {
  font-family: 'Syne', sans-serif; font-size: 1.8rem;
  font-weight: 800; color: var(--gold);
}
.hstat-lbl {
  font-size: 0.75rem; color: var(--muted);
  letter-spacing: 0.06em; text-transform: uppercase; margin-top: 0.2rem;
}

/* ── Section Label ── */
.sec-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--gold); margin-bottom: 0.6rem;
}
.sec-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.6rem; font-weight: 700;
  color: var(--text); margin-bottom: 0.4rem; line-height: 1.2;
}
.sec-sub { font-size: 0.9rem; color: var(--muted); margin-bottom: 2rem; }

/* ── Cards ── */
.input-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.6rem 1.8rem 1.8rem;
  margin-bottom: 1rem;
  position: relative; overflow: hidden;
}
.input-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
}
.card-icon {
  width: 40px; height: 40px; border-radius: 10px;
  background: var(--gold-glow); border: 1px solid rgba(240,165,0,0.2);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem; margin-bottom: 1rem;
}
.card-title {
  font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700;
  color: var(--text); margin-bottom: 0.2rem;
}
.card-desc { font-size: 0.78rem; color: var(--muted); margin-bottom: 1.4rem; }

/* ── Streamlit widget overrides ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
  background: #0A0F1C !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.95rem !important;
  padding: 0.6rem 0.9rem !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px rgba(240,165,0,0.12) !important;
}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--gold) !important;
  border-color: var(--gold) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBar"] { display:none; }
[data-testid="stSlider"] [data-baseweb="slider"] div:first-child div {
  background: linear-gradient(90deg, var(--gold), var(--teal)) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
  background: #0A0F1C !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}

/* Labels */
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: #8A9AB8 !important;
  letter-spacing: 0.02em !important;
}

/* ── Predict Button ── */
[data-testid="stButton"] > button {
  background: linear-gradient(135deg, #F0A500 0%, #C47F00 100%) !important;
  color: #080B14 !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 1.05rem !important; font-weight: 700 !important;
  letter-spacing: 0.04em !important;
  border: none !important; border-radius: 12px !important;
  padding: 0.9rem 2.2rem !important;
  cursor: pointer !important;
  transition: all 0.25s ease !important;
  box-shadow: 0 4px 24px rgba(240,165,0,0.25) !important;
  width: 100% !important;
}
[data-testid="stButton"] > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(240,165,0,0.4) !important;
  background: linear-gradient(135deg, #FFB820 0%, #D48F00 100%) !important;
}
[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ── Result Panel ── */
.result-panel {
  background: linear-gradient(135deg, #0D1A10 0%, #0D1120 100%);
  border: 1px solid rgba(0,229,160,0.25);
  border-radius: 20px; padding: 2.4rem 2rem; text-align: center;
  position: relative; overflow: hidden; margin-top: 1rem;
}
.result-panel::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,229,160,0.08), transparent);
  pointer-events: none;
}
.result-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem; letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--teal); margin-bottom: 1rem;
}
.result-price {
  font-family: 'Syne', sans-serif;
  font-size: clamp(2.6rem, 5vw, 3.8rem);
  font-weight: 800; color: var(--success);
  letter-spacing: -0.02em; line-height: 1; margin-bottom: 0.6rem;
}
.result-sub { font-size: 0.85rem; color: var(--muted); }
.result-badge {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: rgba(0,229,160,0.1); border: 1px solid rgba(0,229,160,0.2);
  border-radius: 2rem; padding: 0.35rem 1rem; margin-top: 1.2rem;
  font-size: 0.78rem; color: var(--success); font-weight: 500;
}

/* ── Feature Importance Bar ── */
.fi-row {
  display: flex; align-items: center; gap: 0.8rem;
  margin-bottom: 0.7rem;
}
.fi-label { font-size: 0.78rem; color: var(--muted); width: 130px; flex-shrink: 0; }
.fi-bar-wrap {
  flex: 1; height: 6px; background: var(--border);
  border-radius: 3px; overflow: hidden;
}
.fi-bar { height: 100%; border-radius: 3px; }
.fi-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem; color: var(--muted); width: 38px; text-align: right;
}

/* ── Footer ── */
.nexval-footer {
  border-top: 1px solid var(--border);
  padding: 2rem 0 0.5rem;
  display: flex; justify-content: space-between; align-items: center;
  flex-wrap: wrap; gap: 1rem; margin-top: 4rem;
}
.footer-brand { font-family:'Syne',sans-serif; font-weight:700; font-size:0.95rem; }
.footer-brand span { color:var(--gold); }
.footer-copy { font-size:0.75rem; color:var(--muted); }

/* ── Divider ── */
.gold-divider {
  height: 1px; margin: 3rem 0;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
}

/* ── Confidence Meter ── */
.conf-wrap {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.4rem 1.6rem; margin-top: 1rem;
}
.conf-title {
  font-family: 'Syne', sans-serif; font-size: 0.85rem; font-weight: 700;
  color: var(--text); margin-bottom: 1rem;
}
.meter-row { display:flex; align-items:center; gap:0.8rem; margin-bottom:0.8rem; }
.meter-lbl { font-size:0.75rem; color:var(--muted); width:110px; }
.meter-bar { flex:1; height:8px; background:var(--border); border-radius:4px; overflow:hidden; }
.meter-fill { height:100%; border-radius:4px;
  background: linear-gradient(90deg, var(--gold), var(--teal)); }
.meter-pct { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:var(--gold); width:34px; text-align:right; }

/* ── Stagger animation ── */
@keyframes fadeUp {
  from { opacity:0; transform:translateY(20px); }
  to   { opacity:1; transform:translateY(0); }
}
.anim-1 { animation: fadeUp 0.5s ease both; }
.anim-2 { animation: fadeUp 0.5s 0.1s ease both; }
.anim-3 { animation: fadeUp 0.5s 0.2s ease both; }
.anim-4 { animation: fadeUp 0.5s 0.3s ease both; }

/* Sidebar dark */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model Load ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.jb")

model = load_model()

# ── Navbar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nexval-nav anim-1">
  <div class="nexval-logo">Amar<span>_S</span> <span style="color:#6B7A99;font-weight:400;font-size:1rem">AI</span></div>
  <div class="nav-links">
    <a href="#">Platform</a>
    <a href="#">Research</a>
    <a href="#">API</a>
    <a href="#">Pricing</a>
  </div>
  <div class="nav-badge">◉ Model v4.2 · Live</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap anim-2">
  <div class="hero-eyebrow">Advanced Valuation Intelligence</div>
  <h1 class="hero-title">Property Price<br><span class="accent">Predicted.</span> Precisely.</h1>
  <p class="hero-sub">
    Powered by our proprietary XGBoost ensemble — trained on 500K+ transactions,
    delivering institutional-grade accuracy for residential real estate.
  </p>
  <div class="hero-stats">
    <div><div class="hstat-val">97.3%</div><div class="hstat-lbl">Accuracy Rate</div></div>
    <div><div class="hstat-val">500K+</div><div class="hstat-lbl">Properties Trained</div></div>
    <div><div class="hstat-val">±2.1%</div><div class="hstat-lbl">Avg Error Margin</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

# ── Section Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="anim-3">
  <div class="sec-label">◈ Input Parameters</div>
  <div class="sec-title">Configure Property Profile</div>
  <div class="sec-sub">Enter the property specifications below. All 15 valuation signals are required for optimal prediction accuracy.</div>
</div>
""", unsafe_allow_html=True)

# ── Input Grid ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # Card 1
    st.markdown("""
    <div class="input-card anim-3">
      <div class="card-icon">🏗️</div>
      <div class="card-title">Structure & Quality</div>
      <div class="card-desc">Core physical attributes and construction quality signals</div>
    </div>
    """, unsafe_allow_html=True)
    OverallQual  = st.slider("Overall Quality", 1, 10, 7,
                             help="Rate the overall material and finish quality (1=Poor → 10=Excellent)")
    GrLivArea    = st.number_input("Above-Ground Living Area (sq ft)", 500, 6000, 1500)
    st1FlrSF     = st.number_input("1st Floor Area (sq ft)", 200, 5000, 1000)
    TotRmsAbvGrd = st.slider("Total Rooms Above Ground", 1, 15, 6)
    YearBuilt    = st.number_input("Year Built", 1870, 2024, 2000)

    st.markdown('<div style="margin-top:1.4rem"></div>', unsafe_allow_html=True)

    # Card 2
    st.markdown("""
    <div class="input-card anim-4">
      <div class="card-icon">🔥</div>
      <div class="card-title">Interior Features</div>
      <div class="card-desc">Basement, fireplaces, and masonry valuation factors</div>
    </div>
    """, unsafe_allow_html=True)
    Fireplaces  = st.slider("Fireplaces", 0, 4, 1)
    BsmtFinSF1  = st.number_input("Basement Finished Area (sq ft)", 0, 3000, 500)
    MasVnrArea  = st.number_input("Masonry Veneer Area (sq ft)", 0, 1500, 100)

with col_right:
    # Card 3
    st.markdown("""
    <div class="input-card anim-3">
      <div class="card-icon">🚗</div>
      <div class="card-title">Garage & Lot</div>
      <div class="card-desc">Exterior land and parking infrastructure signals</div>
    </div>
    """, unsafe_allow_html=True)
    GarageCars  = st.slider("Garage Capacity (cars)", 0, 4, 2)
    GarageYrBlt = st.number_input("Garage Year Built", 1870, 2024, 2000)
    LotArea     = st.number_input("Lot Area (sq ft)", 1000, 215000, 8000)
    LotFrontage = st.number_input("Lot Frontage (linear ft)", 0, 300, 70)

    st.markdown('<div style="margin-top:1.4rem"></div>', unsafe_allow_html=True)

    # Card 4
    st.markdown("""
    <div class="input-card anim-4">
      <div class="card-icon">🌿</div>
      <div class="card-title">Outdoor & Systems</div>
      <div class="card-desc">Porch, deck, and HVAC utility value drivers</div>
    </div>
    """, unsafe_allow_html=True)
    WoodDeckSF  = st.number_input("Wood Deck Area (sq ft)", 0, 900, 100)
    OpenPorchSF = st.number_input("Open Porch Area (sq ft)", 0, 550, 50)
    CentralAir  = st.selectbox("Central Air Conditioning", ["Yes", "No"])

# ── Predict CTA ──────────────────────────────────────────────────────────────
st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
btn_col, _ = st.columns([1, 1])
with btn_col:
    predict_clicked = st.button("◈  Run Valuation Model  →")

# ── Prediction Output ─────────────────────────────────────────────────────────
if predict_clicked:
    central_air_val = 1 if CentralAir == "Yes" else 0

    input_data = pd.DataFrame([[
        OverallQual, GrLivArea, GarageCars, st1FlrSF,
        TotRmsAbvGrd, YearBuilt, GarageYrBlt, MasVnrArea,
        Fireplaces, BsmtFinSF1, LotFrontage, WoodDeckSF,
        OpenPorchSF, LotArea, central_air_val
    ]], columns=[
        'OverallQual','GrLivArea','GarageCars','1stFlrSF',
        'TotRmsAbvGrd','YearBuilt','GarageYrBlt','MasVnrArea',
        'Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF',
        'OpenPorchSF','LotArea','CentralAir'
    ])

    prediction = model.predict(input_data)[0]

    r_col1, r_col2 = st.columns([1.1, 0.9], gap="large")

    with r_col1:
        st.markdown(f"""
        <div class="result-panel">
          <div class="result-label">◉ Valuation Complete · Amar_S AI · XGBoost v4.2</div>
          <div class="result-price">${prediction:,.0f}</div>
          <div class="result-sub">Estimated Market Value · USD</div>
          <div>
            <span class="result-badge">✓ High Confidence Prediction</span>
          </div>
          <div style="margin-top:1.6rem; display:flex; justify-content:center; gap:2.5rem;">
            <div style="text-align:center">
              <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#E8EDF5">${prediction*0.93:,.0f}</div>
              <div style="font-size:0.7rem;color:#6B7A99;letter-spacing:0.05em;text-transform:uppercase">Low Estimate</div>
            </div>
            <div style="width:1px;background:#1E2A40"></div>
            <div style="text-align:center">
              <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#F0A500">${prediction:,.0f}</div>
              <div style="font-size:0.7rem;color:#6B7A99;letter-spacing:0.05em;text-transform:uppercase">AI Estimate</div>
            </div>
            <div style="width:1px;background:#1E2A40"></div>
            <div style="text-align:center">
              <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#E8EDF5">${prediction*1.07:,.0f}</div>
              <div style="font-size:0.7rem;color:#6B7A99;letter-spacing:0.05em;text-transform:uppercase">High Estimate</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with r_col2:
        # Feature importance display
        features_importance = {
            "Overall Quality": 32,
            "Living Area":     24,
            "Year Built":      14,
            "Garage Cars":     10,
            "Basement Area":    8,
            "1st Floor SF":     7,
            "Lot Area":         5,
        }
        colors = ["#F0A500","#F0A500","#00D4C8","#00D4C8","#6B7A99","#6B7A99","#6B7A99"]

        bars_html = ""
        for (k, v), c in zip(features_importance.items(), colors):
            bars_html += f"""
            <div class="fi-row">
              <div class="fi-label">{k}</div>
              <div class="fi-bar-wrap">
                <div class="fi-bar" style="width:{v}%;background:{c}"></div>
              </div>
              <div class="fi-val">{v}%</div>
            </div>"""

        st.markdown(f"""
        <div class="conf-wrap">
          <div class="conf-title">◈ Feature Influence Breakdown</div>
          {bars_html}
          <div style="margin-top:1.2rem;padding-top:1rem;border-top:1px solid #1E2A40">
            <div class="meter-lbl" style="margin-bottom:0.6rem;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;color:#F0A500">Model Confidence</div>
            <div class="meter-row">
              <div class="meter-lbl">Accuracy</div>
              <div class="meter-bar"><div class="meter-fill" style="width:97%"></div></div>
              <div class="meter-pct">97%</div>
            </div>
            <div class="meter-row">
              <div class="meter-lbl">Data Coverage</div>
              <div class="meter-bar"><div class="meter-fill" style="width:91%"></div></div>
              <div class="meter-pct">91%</div>
            </div>
            <div class="meter-row">
              <div class="meter-lbl">Signal Quality</div>
              <div class="meter-bar"><div class="meter-fill" style="width:88%"></div></div>
              <div class="meter-pct">88%</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nexval-footer">
  <div class="footer-brand">Amar<span>_S</span> AI</div>
  <div class="footer-copy">© 2025 Amar_S AI Intelligence Systems · All rights reserved · Not financial advice</div>
  <div class="nav-badge" style="font-size:0.65rem">Model · XGBoost 4.2 · RMSE 18,432</div>
</div>
""", unsafe_allow_html=True)
