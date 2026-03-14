"""
   NIFTY 50 — LSTM PREDICTION DASHBOARD                  
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Nifty 50 · LSTM Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #080c10;
    color: #e2e8f0;
}

.stApp { background: #080c10; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0f14 100%);
    border-right: 1px solid #1e2d3d;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8;
}

/* ── Header ── */
.dash-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #0f2338 50%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.dash-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 300px; height: 100px;
    background: radial-gradient(ellipse, rgba(16,185,129,0.07) 0%, transparent 70%);
}
.dash-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #38bdf8, #10b981, #38bdf8);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer {
    0% { background-position: 0% }
    100% { background-position: 200% }
}
.dash-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 0;
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    background: linear-gradient(135deg, #0d1b2a, #0f2033);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover {
    border-color: #38bdf8;
    transform: translateY(-2px);
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
}
.metric-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    margin-top: 4px;
}
.up   { color: #10b981; }
.down { color: #f43f5e; }
.neutral { color: #64748b; }

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px 0;
}
.section-header h3 {
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #38bdf8;
    margin: 0;
}
.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e3a5f, transparent);
}

/* ── Chart containers ── */
.chart-container {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 4px;
    margin-bottom: 20px;
}

/* ── Forecast table ── */
.forecast-row {
    display: grid;
    grid-template-columns: 100px 1fr 80px;
    gap: 12px;
    padding: 10px 16px;
    border-bottom: 1px solid #0f2033;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    align-items: center;
    transition: background 0.15s;
}
.forecast-row:hover { background: #0f2033; border-radius: 8px; }
.forecast-header {
    color: #38bdf8;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 4px;
}

/* ── Accuracy badges ── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
}
.badge-green  { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.badge-yellow { background: #1c1003; color: #fbbf24; border: 1px solid #854d0e; }
.badge-red    { background: #1c0610; color: #f87171; border: 1px solid #991b1b; }

/* ── Streamlit overrides ── */
.stSlider > div > div { background: #1e3a5f !important; }
div[data-testid="stMetric"] {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 16px;
}
.stSelectbox > div, .stMultiSelect > div {
    background: #0d1b2a !important;
    border-color: #1e3a5f !important;
}
h1, h2, h3 { color: #e2e8f0; }
.stMarkdown p { color: #94a3b8; }

/* status pill */
.status-live {
    display: inline-flex; align-items: center; gap: 6px;
    background: #052e16; color: #4ade80;
    border: 1px solid #166534;
    border-radius: 20px; padding: 4px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.08em;
}
.dot { width:7px; height:7px; border-radius:50%;
       background:#4ade80; animation: pulse 1.5s infinite; }
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model_keras():
    try:
        from tensorflow.keras.models import load_model
        return load_model('models/lstm_model.h5')
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_artifacts():
    # Artifacts are stored inside processed_data.pkl — no separate file needed
    try:
        with open('data/processed_data.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_processed():
    try:
        with open('data/processed_data.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_raw():
    try:
        df = pd.read_csv('data/nifty50_raw.csv', parse_dates=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        return df
    except Exception:
        return None

def forecast_future(model, last_seq, target_scaler, n_days=30):
    preds, seq = [], last_seq.copy()
    for _ in range(n_days):
        inp = seq.reshape(1, seq.shape[0], seq.shape[1])
        p = model.predict(inp, verbose=0)[0][0]
        preds.append(target_scaler.inverse_transform([[p]])[0][0])
        new_row = seq[-1].copy()
        new_row[0] = p
        seq = np.vstack([seq[1:], new_row])
    return preds

def add_indicators(df):
    df = df.copy()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = -delta.where(delta < 0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    df['BB_upper'] = df['MA20'] + 2 * df['Close'].rolling(20).std()
    df['BB_lower'] = df['MA20'] - 2 * df['Close'].rolling(20).std()
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    return df


# ══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════
model      = load_model_keras()
processed  = load_processed()
artifacts  = processed   # same dict — avoids loading twice
df_raw     = load_raw()

data_ok = all([model, artifacts, processed, df_raw is not None])

if data_ok:
    target_scaler  = processed['target_scaler']
    X_test         = processed['X_test']
    y_test         = processed['y_test']
    # Re-generate predictions from the loaded model so we never need a cached artifact file
    _y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred   = target_scaler.inverse_transform(_y_pred_scaled).reshape(-1)
    y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    df_ind         = add_indicators(df_raw)


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Controls")
    st.markdown("---")

    n_forecast = st.slider("📅 Forecast Days", 7, 60, 30, step=1)
    lookback   = st.slider("🔭 Chart Lookback (days)", 60, 500, 200, step=10)

    st.markdown("---")
    st.markdown("### 📊 Chart Layers")
    show_ma     = st.checkbox("Moving Averages (MA20 / MA50)", value=True)
    show_bb     = st.checkbox("Bollinger Bands", value=True)
    show_volume = st.checkbox("Volume Bars", value=True)

    st.markdown("---")
    st.markdown("### 🔮 Indicators")
    show_rsi  = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD", value=True)

    st.markdown("---")
    st.markdown(
        '<div class="status-live"><span class="dot"></span>Model Active</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-family:JetBrains Mono;font-size:0.68rem;color:#334155;"
        f"margin-top:10px'>Updated · {datetime.now().strftime('%d %b %Y %H:%M')}</p>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="dash-header">
  <p class="dash-subtitle">Deep Learning · Time-Series · NSE India</p>
  <h1 class="dash-title">Nifty 50 LSTM Predictor</h1>
  <p style="color:#475569;font-size:0.85rem;margin:0;font-family:'JetBrains Mono',monospace">
    LSTM Neural Network · 11 Features · 60-Day Window
  </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  GUARD — missing files
# ══════════════════════════════════════════════════════════
if not data_ok:
    st.error("⚠️  Model or data files not found. Please complete Steps 2–5 first.")
    missing = []
    if model     is None: missing.append("`models/lstm_model.h5`")
    if processed is None: missing.append("`data/processed_data.pkl`")
    if df_raw    is None: missing.append("`data/nifty50_raw.csv`")
    st.markdown("**Missing files:**\n" + "\n".join(f"- {m}" for m in missing))
    st.stop()


# ══════════════════════════════════════════════════════════
#  METRIC CARDS
# ══════════════════════════════════════════════════════════
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
mae  = float(mean_absolute_error(y_actual, y_pred))
mape = float(np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100)
r2   = float(1 - np.sum((y_actual - y_pred)**2) / np.sum((y_actual - np.mean(y_actual))**2))

current_price = float(y_actual[-1])
last_pred     = float(y_pred[-1])
delta_pct     = (last_pred - current_price) / current_price * 100

mape_badge = ("badge-green" if mape < 2 else "badge-yellow" if mape < 4 else "badge-red")
r2_badge   = ("badge-green" if r2 > 0.95 else "badge-yellow" if r2 > 0.85 else "badge-red")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Current Price</div>
      <div class="metric-value">₹{current_price:,.0f}</div>
      <div class="metric-delta neutral">Nifty 50 Close</div>
    </div>""", unsafe_allow_html=True)

with col2:
    arrow = "▲" if delta_pct >= 0 else "▼"
    cls   = "up" if delta_pct >= 0 else "down"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Last Prediction</div>
      <div class="metric-value">₹{last_pred:,.0f}</div>
      <div class="metric-delta {cls}">{arrow} {abs(delta_pct):.2f}% vs actual</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">MAPE · Accuracy</div>
      <div class="metric-value">{mape:.2f}%</div>
      <div class="metric-delta" style="margin-top:6px">
        <span class="badge {mape_badge}">{"Excellent" if mape<2 else "Good" if mape<4 else "Fair"}</span>
      </div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">R² Score</div>
      <div class="metric-value">{r2:.4f}</div>
      <div class="metric-delta" style="margin-top:6px">
        <span class="badge {r2_badge}">RMSE ₹{rmse:,.0f}</span>
      </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  TAB LAYOUT
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Price & Prediction",
    "🔮  Future Forecast",
    "📊  Technical Indicators",
    "🧠  Model Performance",
])


# ── TAB 1 · Price & Prediction ────────────────────────────
with tab1:
    import plotly.graph_objects as go

    st.markdown("""
    <div class="section-header">
      <h3>Historical Price · Actual vs Predicted</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    # Align dates with test predictions
    n_test   = len(y_actual)
    df_dates = df_raw['Date'].values
    test_dates = df_dates[-n_test:]

    lb = min(lookback, n_test)
    d  = test_dates[-lb:]
    a  = y_actual[-lb:]
    p  = y_pred[-lb:]

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=d, y=a, name="Actual",
        line=dict(color="#38bdf8", width=2),
        hovertemplate="<b>Actual</b><br>%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>"
    ))
    fig1.add_trace(go.Scatter(
        x=d, y=p, name="Predicted",
        line=dict(color="#10b981", width=2, dash="dot"),
        hovertemplate="<b>Predicted</b><br>%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>"
    ))
    fig1.add_trace(go.Scatter(
        x=np.concatenate([d, d[::-1]]),
        y=np.concatenate([a * 1.01, (a * 0.99)[::-1]]),
        fill='toself', fillcolor='rgba(56,189,248,0.05)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False, hoverinfo='skip', name="Band"
    ))

    if show_ma and df_ind is not None:
        df_tail = df_ind.tail(lb)
        fig1.add_trace(go.Scatter(
            x=df_tail['Date'], y=df_tail['MA20'], name="MA20",
            line=dict(color="#f59e0b", width=1.2, dash="dash"),
            hovertemplate="MA20: ₹%{y:,.0f}<extra></extra>"
        ))
        fig1.add_trace(go.Scatter(
            x=df_tail['Date'], y=df_tail['MA50'], name="MA50",
            line=dict(color="#a855f7", width=1.2, dash="dash"),
            hovertemplate="MA50: ₹%{y:,.0f}<extra></extra>"
        ))

    if show_bb and df_ind is not None:
        df_tail = df_ind.tail(lb)
        fig1.add_trace(go.Scatter(
            x=df_tail['Date'], y=df_tail['BB_upper'], name="BB Upper",
            line=dict(color="rgba(248,113,113,0.5)", width=1),
            hovertemplate="BB Upper: ₹%{y:,.0f}<extra></extra>"
        ))
        fig1.add_trace(go.Scatter(
            x=df_tail['Date'], y=df_tail['BB_lower'], name="BB Lower",
            line=dict(color="rgba(248,113,113,0.5)", width=1),
            fill='tonexty', fillcolor='rgba(248,113,113,0.04)',
            hovertemplate="BB Lower: ₹%{y:,.0f}<extra></extra>"
        ))

    fig1.update_layout(
        paper_bgcolor='#0d1b2a', plot_bgcolor='#080c10',
        font=dict(family='JetBrains Mono', color='#64748b', size=11),
        xaxis=dict(gridcolor='#0f2033', showgrid=True, zeroline=False,
                   tickfont=dict(color='#475569')),
        yaxis=dict(gridcolor='#0f2033', showgrid=True, zeroline=False,
                   tickprefix='₹', tickfont=dict(color='#475569')),
        legend=dict(bgcolor='rgba(13,27,42,0.8)', bordercolor='#1e3a5f',
                    borderwidth=1, font=dict(color='#94a3b8', size=11)),
        hovermode='x unified',
        height=440,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    st.plotly_chart(fig1, use_container_width=True)

    if show_volume and df_raw is not None:
        df_vol = df_raw.tail(lb)
        colors = ['#10b981' if c >= o else '#f43f5e'
                  for c, o in zip(df_vol['Close'], df_vol['Open'])]
        fig_vol = go.Figure(go.Bar(
            x=df_vol['Date'], y=df_vol['Volume'],
            marker_color=colors, opacity=0.7, name="Volume",
            hovertemplate="%{x|%d %b %Y}<br>Vol: %{y:,}<extra></extra>"
        ))
        fig_vol.update_layout(
            paper_bgcolor='#0d1b2a', plot_bgcolor='#080c10',
            font=dict(family='JetBrains Mono', color='#64748b', size=11),
            xaxis=dict(gridcolor='#0f2033', tickfont=dict(color='#475569')),
            yaxis=dict(gridcolor='#0f2033', tickfont=dict(color='#475569')),
            height=180, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_vol, use_container_width=True)


# ── TAB 2 · Future Forecast ───────────────────────────────
with tab2:
    st.markdown("""
    <div class="section-header">
      <h3>Future Price Forecast</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    with st.spinner(f"🔮 Computing {n_forecast}-day forecast…"):
        last_seq    = X_test[-1]
        future_pred = forecast_future(model, last_seq, target_scaler, n_forecast)

    # Build dates
    last_date    = pd.to_datetime(df_raw['Date'].iloc[-1])
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1),
                                  periods=n_forecast)

    hist_days  = 60
    hist_dates = df_raw['Date'].values[-hist_days:]
    hist_close = df_raw['Close'].values[-hist_days:]

    future_arr = np.array(future_pred)
    chg        = ((future_arr[-1] - hist_close[-1]) / hist_close[-1]) * 100
    chg_cls    = "up" if chg >= 0 else "down"
    chg_arrow  = "▲" if chg >= 0 else "▼"

    # Summary row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Forecast Start</div>
          <div class="metric-value" style="font-size:1.1rem">₹{hist_close[-1]:,.0f}</div>
          <div class="metric-delta neutral">{pd.Timestamp(hist_dates[-1]).strftime('%d %b %Y')}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{n_forecast}-Day Target</div>
          <div class="metric-value" style="font-size:1.1rem">₹{future_arr[-1]:,.0f}</div>
          <div class="metric-delta {chg_cls}">{chg_arrow} {abs(chg):.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        peak = future_arr.max(); trough = future_arr.min()
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Range</div>
          <div class="metric-value" style="font-size:1.1rem">₹{peak-trough:,.0f}</div>
          <div class="metric-delta neutral">₹{trough:,.0f} – ₹{peak:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Chart
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=hist_dates, y=hist_close, name="Historical",
        line=dict(color="#38bdf8", width=2),
        hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>"
    ))
    fig2.add_trace(go.Scatter(
        x=future_dates, y=future_arr, name=f"{n_forecast}-Day Forecast",
        line=dict(color="#10b981", width=2.5, dash="dot"),
        mode='lines+markers',
        marker=dict(size=4, color="#10b981"),
        hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>"
    ))
    # Confidence band ±1.5%
    fig2.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(future_arr * 1.015) + list((future_arr * 0.985)[::-1]),
        fill='toself', fillcolor='rgba(16,185,129,0.08)',
        line=dict(color='rgba(0,0,0,0)'), name="±1.5% Band",
        hoverinfo='skip'
    ))
    # Bridge line
    fig2.add_trace(go.Scatter(
        x=[hist_dates[-1], future_dates[0]],
        y=[hist_close[-1], future_arr[0]],
        line=dict(color="#38bdf8", width=1.5, dash="dot"),
        showlegend=False, hoverinfo='skip'
    ))
    fig2.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="#f59e0b", width=1, dash="dash")
    )
    fig2.add_annotation(
        x=last_date,
        y=1.02,
        xref="x",
        yref="paper",
        text="Forecast →",
        showarrow=False,
        font=dict(color="#f59e0b", size=11)
    )

    fig2.update_layout(
        paper_bgcolor='#0d1b2a', plot_bgcolor='#080c10',
        font=dict(family='JetBrains Mono', color='#64748b', size=11),
        xaxis=dict(gridcolor='#0f2033', tickfont=dict(color='#475569')),
        yaxis=dict(gridcolor='#0f2033', tickprefix='₹', tickfont=dict(color='#475569')),
        legend=dict(bgcolor='rgba(13,27,42,0.8)', bordercolor='#1e3a5f', borderwidth=1,
                    font=dict(color='#94a3b8', size=11)),
        hovermode='x unified', height=420,
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Forecast table
    st.markdown("""
    <div class="section-header">
      <h3>Day-by-Day Forecast Table</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    fcol1, fcol2 = st.columns([1, 1])
    half = n_forecast // 2

    def render_table(dates, prices, base):
        html = """<div class="forecast-row forecast-header">
                    <span>Date</span><span>Predicted Close</span><span>Chg</span>
                  </div>"""
        for i, (dt, pr) in enumerate(zip(dates, prices)):
            chg_d = ((pr - base) / base * 100)
            cls   = "up" if chg_d >= 0 else "down"
            arrow = "▲" if chg_d >= 0 else "▼"
            html += f"""<div class="forecast-row">
              <span style="color:#475569">{pd.Timestamp(dt).strftime('%d %b')}</span>
              <span style="color:#e2e8f0">₹{pr:,.2f}</span>
              <span class="{cls}">{arrow}{abs(chg_d):.1f}%</span>
            </div>"""
        return html

    with fcol1:
        st.markdown(render_table(future_dates[:half], future_pred[:half], hist_close[-1]),
                    unsafe_allow_html=True)
    with fcol2:
        st.markdown(render_table(future_dates[half:], future_pred[half:], hist_close[-1]),
                    unsafe_allow_html=True)

    # Download CSV
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': [round(p, 2) for p in future_pred]
    })
    csv = forecast_df.to_csv(index=False).encode()
    st.download_button(
        "⬇️  Download Forecast CSV", csv,
        file_name=f"nifty50_forecast_{n_forecast}d.csv",
        mime="text/csv"
    )


# ── TAB 3 · Technical Indicators ─────────────────────────
with tab3:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.markdown("""
    <div class="section-header">
      <h3>Technical Indicators</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    df_t = df_ind.dropna().tail(lookback)

    rows = 1 + (1 if show_rsi else 0) + (1 if show_macd else 0)
    heights = [0.55] + [0.225] * (rows - 1)

    subplot_titles = ["Price"]
    if show_rsi:  subplot_titles.append("RSI (14)")
    if show_macd: subplot_titles.append("MACD")

    fig3 = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                         vertical_spacing=0.04,
                         row_heights=heights,
                         subplot_titles=subplot_titles)

    # Candlestick
    fig3.add_trace(go.Candlestick(
        x=df_t['Date'],
        open=df_t['Open'], high=df_t['High'],
        low=df_t['Low'],   close=df_t['Close'],
        increasing_line_color='#10b981',
        decreasing_line_color='#f43f5e',
        name="OHLC"
    ), row=1, col=1)

    if show_ma:
        fig3.add_trace(go.Scatter(x=df_t['Date'], y=df_t['MA20'], name="MA20",
            line=dict(color="#f59e0b", width=1.2, dash="dash")), row=1, col=1)
        fig3.add_trace(go.Scatter(x=df_t['Date'], y=df_t['MA50'], name="MA50",
            line=dict(color="#a855f7", width=1.2, dash="dash")), row=1, col=1)

    if show_bb:
        fig3.add_trace(go.Scatter(x=df_t['Date'], y=df_t['BB_upper'], name="BB Upper",
            line=dict(color="rgba(248,113,113,0.5)", width=1)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=df_t['Date'], y=df_t['BB_lower'], name="BB Lower",
            line=dict(color="rgba(248,113,113,0.5)", width=1),
            fill='tonexty', fillcolor='rgba(248,113,113,0.04)'), row=1, col=1)

    cur_row = 2
    if show_rsi:
        fig3.add_trace(go.Scatter(x=df_t['Date'], y=df_t['RSI'], name="RSI",
            line=dict(color="#38bdf8", width=1.5)), row=cur_row, col=1)
        fig3.add_hline(y=70, line_dash="dash", line_color="#f43f5e",
                       line_width=0.8, row=cur_row, col=1)
        fig3.add_hline(y=30, line_dash="dash", line_color="#10b981",
                       line_width=0.8, row=cur_row, col=1)
        cur_row += 1

    if show_macd:
        macd_colors = ['#10b981' if v >= 0 else '#f43f5e'
                       for v in (df_t['MACD'] - df_t['MACD_signal'])]
        fig3.add_trace(go.Bar(x=df_t['Date'],
            y=df_t['MACD'] - df_t['MACD_signal'],
            marker_color=macd_colors, name="MACD Hist", opacity=0.6),
            row=cur_row, col=1)
        fig3.add_trace(go.Scatter(x=df_t['Date'], y=df_t['MACD'],
            name="MACD", line=dict(color="#38bdf8", width=1.5)),
            row=cur_row, col=1)
        fig3.add_trace(go.Scatter(x=df_t['Date'], y=df_t['MACD_signal'],
            name="Signal", line=dict(color="#f59e0b", width=1.5)),
            row=cur_row, col=1)

    fig3.update_layout(
        paper_bgcolor='#0d1b2a', plot_bgcolor='#080c10',
        font=dict(family='JetBrains Mono', color='#64748b', size=10),
        xaxis=dict(gridcolor='#0f2033', rangeslider_visible=False),
        yaxis=dict(gridcolor='#0f2033', tickprefix='₹'),
        legend=dict(bgcolor='rgba(13,27,42,0.8)', bordercolor='#1e3a5f',
                    borderwidth=1, font=dict(color='#94a3b8', size=10)),
        height=600, margin=dict(l=10, r=10, t=30, b=10),
        hovermode='x unified'
    )
    for i in range(2, rows + 1):
        fig3.update_xaxes(gridcolor='#0f2033', row=i, col=1)
        fig3.update_yaxes(gridcolor='#0f2033', row=i, col=1)

    st.plotly_chart(fig3, use_container_width=True)


# ── TAB 4 · Model Performance ─────────────────────────────
with tab4:
    st.markdown("""
    <div class="section-header">
      <h3>Model Performance Analysis</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    # Error distribution
    errors = y_actual - y_pred

    c1, c2 = st.columns(2)

    with c1:
        fig_err = go.Figure()
        fig_err.add_trace(go.Histogram(
            x=errors, nbinsx=40, name="Prediction Error",
            marker_color='#38bdf8', opacity=0.75,
            hovertemplate="Error: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>"
        ))
        fig_err.add_vline(x=0, line_dash="dash", line_color="#f59e0b", line_width=1.5)
        fig_err.update_layout(
            title=dict(text="Error Distribution", font=dict(color='#94a3b8', size=12)),
            paper_bgcolor='#0d1b2a', plot_bgcolor='#080c10',
            font=dict(family='JetBrains Mono', color='#64748b', size=10),
            xaxis=dict(gridcolor='#0f2033', tickprefix='₹', title="Error (₹)"),
            yaxis=dict(gridcolor='#0f2033', title="Frequency"),
            height=320, margin=dict(l=10, r=10, t=40, b=10), showlegend=False
        )
        st.plotly_chart(fig_err, use_container_width=True)

    with c2:
        # Scatter actual vs predicted
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=y_actual, y=y_pred, mode='markers',
            marker=dict(color='#38bdf8', size=4, opacity=0.5),
            name="Predictions",
            hovertemplate="Actual: ₹%{x:,.0f}<br>Pred: ₹%{y:,.0f}<extra></extra>"
        ))
        lim = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
        fig_sc.add_trace(go.Scatter(
            x=lim, y=lim, mode='lines',
            line=dict(color='#10b981', dash='dash', width=1.5),
            name="Perfect Fit"
        ))
        fig_sc.update_layout(
            title=dict(text="Actual vs Predicted", font=dict(color='#94a3b8', size=12)),
            paper_bgcolor='#0d1b2a', plot_bgcolor='#080c10',
            font=dict(family='JetBrains Mono', color='#64748b', size=10),
            xaxis=dict(gridcolor='#0f2033', tickprefix='₹', title="Actual (₹)"),
            yaxis=dict(gridcolor='#0f2033', tickprefix='₹', title="Predicted (₹)"),
            height=320, margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(bgcolor='rgba(13,27,42,0.8)', bordercolor='#1e3a5f',
                        font=dict(color='#94a3b8', size=10))
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    # Rolling MAPE
    st.markdown("""
    <div class="section-header">
      <h3>Rolling 30-Day MAPE</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    rolling_mape = pd.Series(
        np.abs((y_actual - y_pred) / y_actual) * 100
    ).rolling(30).mean()

    fig_rm = go.Figure()
    fig_rm.add_trace(go.Scatter(
        y=rolling_mape, mode='lines', name="Rolling MAPE",
        line=dict(color="#f59e0b", width=2),
        fill='tozeroy', fillcolor='rgba(245,158,11,0.08)',
        hovertemplate="Day %{x}<br>MAPE: %{y:.2f}%<extra></extra>"
    ))
    fig_rm.add_hline(y=2, line_dash="dash", line_color="#10b981",
                     annotation_text="2% threshold",
                     annotation_font_color="#10b981",
                     annotation_font_size=10, line_width=1)
    fig_rm.update_layout(
        paper_bgcolor='#0d1b2a', plot_bgcolor='#080c10',
        font=dict(family='JetBrains Mono', color='#64748b', size=10),
        xaxis=dict(gridcolor='#0f2033', title="Test Day"),
        yaxis=dict(gridcolor='#0f2033', ticksuffix='%', title="MAPE"),
        height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=False
    )
    st.plotly_chart(fig_rm, use_container_width=True)

    # Metrics summary table
    st.markdown("""
    <div class="section-header">
      <h3>Full Metrics Summary</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    metrics_data = {
        "Metric": ["RMSE", "MAE", "MAPE", "R² Score", "Max Error", "Min Error"],
        "Value":  [
            f"₹{rmse:,.2f}",
            f"₹{mae:,.2f}",
            f"{mape:.4f}%",
            f"{r2:.6f}",
            f"₹{errors.max():,.2f}",
            f"₹{errors.min():,.2f}",
        ],
        "Status": [
            "✅ Excellent" if rmse < 300 else "⚠️ Review",
            "✅ Excellent" if mae < 200  else "⚠️ Review",
            "✅ Excellent" if mape < 2   else ("🟡 Good" if mape < 4 else "⚠️ Review"),
            "✅ Excellent" if r2 > 0.95  else ("🟡 Good" if r2 > 0.85 else "⚠️ Review"),
            "—", "—"
        ]
    }
    st.dataframe(
        pd.DataFrame(metrics_data),
        use_container_width=True,
        hide_index=True
    )


# ── Footer ────────────────────────────────────────────────
st.markdown("""
<br>
<div style="text-align:center; padding: 20px 0;
     border-top: 1px solid #1e2d3d; margin-top: 20px;">
  <p style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
     color:#334155; letter-spacing:0.1em">
    NIFTY 50 LSTM PREDICTOR · FOR EDUCATIONAL PURPOSES ONLY
    · NOT FINANCIAL ADVICE
  </p>
</div>
""", unsafe_allow_html=True)
