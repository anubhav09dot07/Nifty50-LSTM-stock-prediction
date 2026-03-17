"""
   NIFTY 50 — LSTM PREDICTION DASHBOARD                  
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Nifty 50 · Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --apple-blue: #0071e3;
    --apple-bg: #f5f5f7;
    --apple-surface: #ffffff;
    --apple-border: #d2d2d7;
    --apple-text: #1d1d1f;
    --apple-muted: #6e6e73;
    --apple-green: #34c759;
    --apple-red: #ff3b30;
    --apple-amber: #ff9f0a;
}

html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
    color: var(--apple-text);
}

.stApp {
    background: radial-gradient(circle at 10% 0%, #ffffff 0%, #f7f7f9 38%, #f1f2f6 100%);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.82);
    backdrop-filter: blur(12px);
    border-right: 1px solid var(--apple-border);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--apple-text);
}

.dash-header {
    background: linear-gradient(130deg, #ffffff 0%, #eef5ff 50%, #f8f8fa 100%);
    border: 1px solid #d7e5fb;
    border-radius: 28px;
    padding: 46px 48px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 28px 60px rgba(15, 23, 42, 0.08);
}

.dash-header::before {
    content: '';
    position: absolute;
    width: 360px;
    height: 360px;
    top: -190px;
    right: -120px;
    background: radial-gradient(circle, rgba(0, 113, 227, 0.18), rgba(0, 113, 227, 0.01));
    border-radius: 50%;
}

.dash-title {
    font-size: clamp(2rem, 3.2vw, 3.4rem);
    line-height: 1.05;
    letter-spacing: -0.04em;
    margin: 0;
    color: var(--apple-text);
    font-weight: 700;
}

.dash-subtitle {
    margin: 0 0 12px 0;
    color: var(--apple-blue);
    font-size: 0.82rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
}

.metric-card {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid var(--apple-border);
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}

.metric-label {
    color: var(--apple-muted);
    font-size: 0.74rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
    margin-bottom: 8px;
}

.metric-value {
    color: var(--apple-text);
    font-size: 1.45rem;
    font-weight: 650;
    letter-spacing: -0.02em;
}

.metric-delta {
    margin-top: 6px;
    font-size: 0.8rem;
    font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
}

.up { color: var(--apple-green); }
.down { color: var(--apple-red); }
.neutral { color: var(--apple-muted); }

.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 30px 0 14px;
}

.section-header h3 {
    margin: 0;
    color: var(--apple-text);
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 620;
}

.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0, 113, 227, 0.35), rgba(0, 113, 227, 0));
}

.forecast-row {
    display: grid;
    grid-template-columns: 110px 1fr 90px;
    gap: 12px;
    padding: 10px 14px;
    border-bottom: 1px solid #ececf1;
    border-radius: 12px;
    color: var(--apple-text);
    font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
    font-size: 0.8rem;
}

.forecast-row:hover {
    background: #f8faff;
}

.forecast-header {
    color: var(--apple-blue);
    border-bottom: 1px solid #d7e5fb;
    margin-bottom: 5px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 0.68rem;
}

.badge {
    display: inline-flex;
    align-items: center;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.7rem;
    border: 1px solid transparent;
    font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
}

.badge-green  { background: #ecfdf3; color: #107c41; border-color: #c3f0d5; }
.badge-yellow { background: #fff6e9; color: #9a5d00; border-color: #ffe0b8; }
.badge-red    { background: #fff1f0; color: #b42318; border-color: #ffd2cf; }

.status-live {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    border-radius: 999px;
    padding: 5px 13px;
    color: #05603a;
    background: #ecfdf3;
    border: 1px solid #c5ecd5;
    font-size: 0.74rem;
    font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
}

.dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--apple-green);
    animation: pulse 1.4s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.45; transform: scale(0.76); }
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: #f2f4f8;
    border: 1px solid #d9dde7;
    border-radius: 999px;
    color: #1d1d1f;
    padding: 8px 16px;
    height: auto;
}

.stTabs [aria-selected="true"] {
    background: var(--apple-blue) !important;
    color: #ffffff !important;
    border-color: var(--apple-blue) !important;
}

.stSlider > div > div {
    background: var(--apple-blue) !important;
}

.stSelectbox > div,
.stMultiSelect > div,
.stTextInput > div > div {
    border-radius: 12px !important;
}

.stDataFrame {
    background: rgba(255, 255, 255, 0.86);
    border: 1px solid var(--apple-border);
    border-radius: 16px;
    overflow: hidden;
}

.stMarkdown p {
    color: #424245;
}

@media (max-width: 900px) {
    .dash-header {
        padding: 34px 26px;
        border-radius: 22px;
    }
}
</style>
""", unsafe_allow_html=True)


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / 'data'
MODELS_DIR = APP_DIR / 'models'
MODEL_FILES = {
    'LSTM': ['lstm_model.h5'],
    'GRU': ['gru_model.h5'],
    'Transformer': ['transformer_model.keras', 'transformer_model.h5'],
    'CNN+LSTM': ['cnn_lstm_model.keras'],
}

MODEL_COLORS = {
    'LSTM': '#0071e3',
    'GRU': '#34c759',
    'Transformer': '#5e5ce6',
    'CNN+LSTM': '#ff9f0a',
}


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        from tensorflow.keras.models import load_model
    except Exception:
        return {}

    loaded = {}
    for name, candidates in MODEL_FILES.items():
        for rel_path in candidates:
            model_path = MODELS_DIR / rel_path
            if not model_path.exists():
                continue
            try:
                loaded[name] = load_model(model_path, compile=False)
                break
            except Exception:
                continue
    return loaded

@st.cache_data(show_spinner=False)
def load_processed():
    try:
        with open(DATA_DIR / 'processed_data.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_raw():
    try:
        df = pd.read_csv(DATA_DIR / 'nifty50_raw.csv', parse_dates=['Date'])
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
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    return df


def style_plotly(fig, height=420, y_tick_prefix=None):
    fig.update_layout(
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(family='-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif', color='#1d1d1f', size=12),
        legend=dict(
            bgcolor='rgba(255,255,255,0.88)',
            bordercolor='#d2d2d7',
            borderwidth=1,
            font=dict(color='#424245', size=11),
        ),
        hovermode='x unified',
        margin=dict(l=10, r=10, t=18, b=10),
        height=height,
    )
    fig.update_xaxes(gridcolor='#ececf1', tickfont=dict(color='#6e6e73'))
    fig.update_yaxes(gridcolor='#ececf1', tickfont=dict(color='#6e6e73'))
    if y_tick_prefix:
        fig.update_yaxes(tickprefix=y_tick_prefix)


# ══════════════════════════════════════════════════════════
#  LOAD DATA
# ══════════════════════════════════════════════════════════
models = load_models()
processed = load_processed()
df_raw = load_raw()

data_ok = bool(models) and (processed is not None) and (df_raw is not None)

selected_model_name = None
selected_model = None
y_pred = None
best_model_name = None
model_predictions = {}
model_metrics_df = pd.DataFrame()
rolling_mape_df = pd.DataFrame()
pairwise_mae_df = pd.DataFrame()
improvement_df = pd.DataFrame()
baseline_model_name = None

if data_ok:
    target_scaler = processed['target_scaler']
    X_test = processed['X_test']
    y_test = processed['y_test']
    y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    model_metrics = {}
    for name, model_obj in models.items():
        pred_scaled = model_obj.predict(X_test, verbose=0)
        pred = target_scaler.inverse_transform(pred_scaled).reshape(-1)
        model_predictions[name] = pred

        model_metrics[name] = {
            'RMSE': float(np.sqrt(mean_squared_error(y_actual, pred))),
            'MAE': float(mean_absolute_error(y_actual, pred)),
            'MAPE': float(np.mean(np.abs((y_actual - pred) / np.clip(np.abs(y_actual), 1e-8, None))) * 100),
        }

    model_metrics_df = pd.DataFrame(model_metrics).T.sort_values('MAPE')
    best_model_name = model_metrics_df.index[0]
    selected_model_name = best_model_name
    selected_model = models[selected_model_name]
    y_pred = model_predictions[selected_model_name]
    df_ind = add_indicators(df_raw)

    rolling_mape_df = pd.DataFrame({
        name: pd.Series(np.abs((y_actual - pred) / np.clip(np.abs(y_actual), 1e-8, None)) * 100).rolling(30).mean()
        for name, pred in model_predictions.items()
    })

    model_order = list(model_metrics_df.index)
    pairwise_mae_df = pd.DataFrame(index=model_order, columns=model_order, dtype=float)
    for left_model in model_order:
        for right_model in model_order:
            pairwise_mae_df.loc[left_model, right_model] = float(
                np.mean(np.abs(model_predictions[left_model] - model_predictions[right_model]))
            )

    baseline_model_name = 'LSTM' if 'LSTM' in model_metrics_df.index else best_model_name
    baseline_mae = model_metrics_df.loc[baseline_model_name, 'MAE']
    baseline_mape = model_metrics_df.loc[baseline_model_name, 'MAPE']

    improvement_df = model_metrics_df.copy()
    improvement_df['MAE vs Baseline %'] = (baseline_mae - improvement_df['MAE']) / baseline_mae * 100
    improvement_df['MAPE vs Baseline %'] = (baseline_mape - improvement_df['MAPE']) / baseline_mape * 100


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Controls")
    st.markdown("---")

    n_forecast = st.slider("📅 Forecast Days", 7, 60, 30, step=1)
    lookback = st.slider("🔭 Chart Lookback (days)", 60, 500, 200, step=10)
    benchmark_window = st.slider("📐 Benchmark Window", 20, 90, 30, step=5)

    st.markdown("---")
    st.markdown("### 📊 Chart Layers")
    show_ma     = st.checkbox("Moving Averages (MA20 / MA50)", value=True)
    show_bb     = st.checkbox("Bollinger Bands", value=True)
    show_volume = st.checkbox("Volume Bars", value=True)
    show_all_models = st.checkbox("Overlay all model predictions", value=True)

    if data_ok:
        st.markdown("---")
        selected_model_name = st.selectbox(
            "🤖 Active Model",
            options=list(model_metrics_df.index),
            index=0,
            help="Models are ranked by MAPE from model_train outputs.",
        )
        selected_model = models[selected_model_name]
        y_pred = model_predictions[selected_model_name]
        st.caption(f"Best model by MAPE: {best_model_name}")

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
        f"<p style='font-family:SF Mono, Menlo, monospace;font-size:0.68rem;color:#6e6e73;"
        f"margin-top:10px'>Updated · {datetime.now().strftime('%d %b %Y %H:%M')}</p>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════
active_model_label = selected_model_name if selected_model_name else "No model loaded"

st.markdown(f"""
<div class="dash-header">
  <p class="dash-subtitle">Market Intelligence Studio · NSE India</p>
  <h1 class="dash-title">Nifty 50 Forecast Gallery</h1>
  <p style="color:#6e6e73;font-size:0.9rem;margin:12px 0 0 0;font-family:'SF Mono',Menlo,monospace">
    Active Model · {active_model_label} · 11 Features · 60-Step Window
  </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  GUARD — missing files
# ══════════════════════════════════════════════════════════
if not data_ok:
    st.error("⚠️  Model or data files not found. Please complete training/prediction steps first.")
    missing = []
    if not models: missing.append("`models/` (no loadable model files found)")
    if processed is None: missing.append("`data/processed_data.pkl`")
    if df_raw    is None: missing.append("`data/nifty50_raw.csv`")
    st.markdown("**Missing files:**\n" + "\n".join(f"- {m}" for m in missing))
    st.stop()


# ══════════════════════════════════════════════════════════
#  METRIC CARDS
# ══════════════════════════════════════════════════════════

rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
mae  = float(mean_absolute_error(y_actual, y_pred))
mape = float(np.mean(np.abs((y_actual - y_pred) / np.clip(np.abs(y_actual), 1e-8, None))) * 100)
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
            <div class="metric-label">Last Prediction · {selected_model_name}</div>
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Price & Prediction",
    "🔮  Future Forecast",
    "📊  Technical Indicators",
    "🧠  Model Performance",
    "🏁  Model Benchmarks",
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
        line=dict(color="#1d1d1f", width=2.2),
        hovertemplate="<b>Actual</b><br>%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>"
    ))
    fig1.add_trace(go.Scatter(
        x=d, y=p, name=f"Predicted ({selected_model_name})",
        line=dict(color=MODEL_COLORS.get(selected_model_name, '#0071e3'), width=2.2, dash="dot"),
        hovertemplate="<b>Predicted</b><br>%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>"
    ))

    if show_all_models:
        for model_name, pred_values in model_predictions.items():
            if model_name == selected_model_name:
                continue
            fig1.add_trace(go.Scatter(
                x=d, y=pred_values[-lb:], name=f"{model_name} (comparison)",
                line=dict(color=MODEL_COLORS.get(model_name, '#6e6e73'), width=1.4, dash='dash'),
                opacity=0.75,
                hovertemplate=f"<b>{model_name}</b><br>%{{x|%d %b %Y}}<br>₹%{{y:,.2f}}<extra></extra>"
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

    style_plotly(fig1, height=440, y_tick_prefix='₹')

    st.plotly_chart(fig1, use_container_width=True)

    if show_volume and df_raw is not None:
        df_vol = df_raw.tail(lb)
        colors = ['#34c759' if c >= o else '#ff3b30'
                  for c, o in zip(df_vol['Close'], df_vol['Open'])]
        fig_vol = go.Figure(go.Bar(
            x=df_vol['Date'], y=df_vol['Volume'],
            marker_color=colors, opacity=0.7, name="Volume",
            hovertemplate="%{x|%d %b %Y}<br>Vol: %{y:,}<extra></extra>"
        ))
        fig_vol.update_layout(showlegend=False)
        style_plotly(fig_vol, height=200)
        st.plotly_chart(fig_vol, use_container_width=True)


# ── TAB 2 · Future Forecast ───────────────────────────────
with tab2:
    st.markdown("""
    <div class="section-header">
      <h3>Future Price Forecast</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    with st.spinner(f"🔮 Computing {n_forecast}-day forecast ({selected_model_name})…"):
        last_seq    = X_test[-1]
        future_pred = forecast_future(selected_model, last_seq, target_scaler, n_forecast)

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
        line=dict(color="#1d1d1f", width=2.2),
        hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>"
    ))
    fig2.add_trace(go.Scatter(
        x=future_dates, y=future_arr, name=f"{n_forecast}-Day Forecast ({selected_model_name})",
        line=dict(color=MODEL_COLORS.get(selected_model_name, '#0071e3'), width=2.6, dash="dot"),
        mode='lines+markers',
        marker=dict(size=4, color=MODEL_COLORS.get(selected_model_name, '#0071e3')),
        hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.2f}<extra></extra>"
    ))
    # Confidence band ±1.5%
    fig2.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(future_arr * 1.015) + list((future_arr * 0.985)[::-1]),
        fill='toself', fillcolor='rgba(0,113,227,0.10)',
        line=dict(color='rgba(0,0,0,0)'), name="±1.5% Band",
        hoverinfo='skip'
    ))
    # Bridge line
    fig2.add_trace(go.Scatter(
        x=[hist_dates[-1], future_dates[0]],
        y=[hist_close[-1], future_arr[0]],
        line=dict(color="#0071e3", width=1.5, dash="dot"),
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

    style_plotly(fig2, height=420, y_tick_prefix='₹')
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
        'Predicted_Close': [round(p, 2) for p in future_pred],
        'Model': selected_model_name,
    })
    csv = forecast_df.to_csv(index=False).encode()
    safe_model_name = selected_model_name.lower().replace('+', 'plus').replace(' ', '_')
    st.download_button(
        "⬇️  Download Forecast CSV", csv,
        file_name=f"nifty50_{safe_model_name}_forecast_{n_forecast}d.csv",
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
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        font=dict(family='-apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif', color='#1d1d1f', size=11),
        xaxis=dict(gridcolor='#ececf1', rangeslider_visible=False),
        yaxis=dict(gridcolor='#ececf1', tickprefix='₹'),
        legend=dict(
            bgcolor='rgba(255,255,255,0.88)',
            bordercolor='#d2d2d7',
            borderwidth=1,
            font=dict(color='#424245', size=10),
        ),
        height=620,
        margin=dict(l=10, r=10, t=28, b=10),
        hovermode='x unified',
    )
    for i in range(2, rows + 1):
        fig3.update_xaxes(gridcolor='#ececf1', row=i, col=1)
        fig3.update_yaxes(gridcolor='#ececf1', row=i, col=1)

    st.plotly_chart(fig3, use_container_width=True)


# ── TAB 4 · Model Performance ─────────────────────────────
with tab4:
    st.markdown(f"""
        <div class="section-header">
            <h3>Model Performance Analysis · {selected_model_name}</h3>
            <div class="section-line"></div>
        </div>""", unsafe_allow_html=True)

    # Error distribution
    errors = y_actual - y_pred

    c1, c2 = st.columns(2)

    with c1:
        fig_err = go.Figure()
        fig_err.add_trace(go.Histogram(
            x=errors, nbinsx=40, name="Prediction Error",
            marker_color=MODEL_COLORS.get(selected_model_name, '#0071e3'), opacity=0.78,
            hovertemplate="Error: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>"
        ))
        fig_err.add_vline(x=0, line_dash="dash", line_color="#ff9f0a", line_width=1.4)
        fig_err.update_layout(
            title=dict(text="Error Distribution", font=dict(color='#1d1d1f', size=12)),
            xaxis=dict(title="Error (₹)"),
            yaxis=dict(title="Frequency"),
            showlegend=False,
        )
        style_plotly(fig_err, height=320)
        st.plotly_chart(fig_err, use_container_width=True)

    with c2:
        # Scatter actual vs predicted
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=y_actual, y=y_pred, mode='markers',
            marker=dict(color=MODEL_COLORS.get(selected_model_name, '#0071e3'), size=4, opacity=0.55),
            name="Predictions",
            hovertemplate="Actual: ₹%{x:,.0f}<br>Pred: ₹%{y:,.0f}<extra></extra>"
        ))
        lim = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
        fig_sc.add_trace(go.Scatter(
            x=lim, y=lim, mode='lines',
            line=dict(color='#1d1d1f', dash='dash', width=1.3),
            name="Perfect Fit"
        ))
        fig_sc.update_layout(
            title=dict(text="Actual vs Predicted", font=dict(color='#1d1d1f', size=12)),
            xaxis=dict(title="Actual (₹)"),
            yaxis=dict(title="Predicted (₹)"),
        )
        style_plotly(fig_sc, height=320)
        st.plotly_chart(fig_sc, use_container_width=True)

    # Rolling MAPE
    st.markdown("""
    <div class="section-header">
      <h3>Rolling 30-Day MAPE</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    rolling_mape = pd.Series(
        np.abs((y_actual - y_pred) / np.clip(np.abs(y_actual), 1e-8, None)) * 100
    ).rolling(30).mean()

    fig_rm = go.Figure()
    fig_rm.add_trace(go.Scatter(
        y=rolling_mape, mode='lines', name="Rolling MAPE",
        line=dict(color=MODEL_COLORS.get(selected_model_name, '#0071e3'), width=2),
        fill='tozeroy', fillcolor='rgba(0,113,227,0.1)',
        hovertemplate="Day %{x}<br>MAPE: %{y:.2f}%<extra></extra>"
    ))
    fig_rm.add_hline(y=2, line_dash="dash", line_color="#34c759",
                     annotation_text="2% threshold",
                     annotation_font_color="#34c759",
                     annotation_font_size=10, line_width=1)
    fig_rm.update_layout(
        xaxis=dict(title="Test Day"),
        yaxis=dict(ticksuffix='%', title="MAPE"),
        showlegend=False,
    )
    style_plotly(fig_rm, height=270)
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

    st.markdown("""
    <div class="section-header">
      <h3>All Model Leaderboard</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)
    st.dataframe(
        model_metrics_df.round(3).reset_index().rename(columns={'index': 'Model'}),
        use_container_width=True,
        hide_index=True,
    )


# ── TAB 5 · Model Benchmarks ──────────────────────────────
with tab5:
    import plotly.graph_objects as go

    st.markdown("""
    <div class="section-header">
      <h3>Cross-Model Benchmark Arena</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    recent_window = min(benchmark_window, len(y_actual))
    recent_rows = []
    for model_name in model_metrics_df.index:
        recent_error = np.abs(
            y_actual[-recent_window:] - model_predictions[model_name][-recent_window:]
        )
        recent_mape = float(np.mean(recent_error / np.clip(np.abs(y_actual[-recent_window:]), 1e-8, None)) * 100)
        recent_rows.append((model_name, recent_mape))

    recent_df = pd.DataFrame(recent_rows, columns=['Model', f'MAPE ({recent_window}d)']).sort_values(f'MAPE ({recent_window}d)')

    bcol1, bcol2, bcol3 = st.columns(3)
    for col, metric in zip([bcol1, bcol2, bcol3], ['RMSE', 'MAE', 'MAPE']):
        with col:
            mdf = model_metrics_df.sort_values(metric)
            fig_metric = go.Figure(go.Bar(
                x=mdf.index,
                y=mdf[metric],
                marker_color=[MODEL_COLORS.get(name, '#0071e3') for name in mdf.index],
                text=[f"{v:.2f}" for v in mdf[metric]],
                textposition='outside',
                name=metric,
            ))
            fig_metric.update_layout(
                title=dict(text=f"{metric} Comparison", font=dict(size=13, color='#1d1d1f')),
                showlegend=False,
                xaxis=dict(title='Model'),
                yaxis=dict(title=metric),
            )
            style_plotly(fig_metric, height=320, y_tick_prefix='₹' if metric in ['RMSE', 'MAE'] else None)
            if metric == 'MAPE':
                fig_metric.update_yaxes(ticksuffix='%')
            st.plotly_chart(fig_metric, use_container_width=True)

    st.markdown("""
    <div class="section-header">
      <h3>Rolling MAPE by Model</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    fig_roll = go.Figure()
    rolling_view = rolling_mape_df.tail(max(lookback, recent_window * 2))
    for model_name in model_metrics_df.index:
        fig_roll.add_trace(go.Scatter(
            y=rolling_view[model_name],
            mode='lines',
            name=model_name,
            line=dict(color=MODEL_COLORS.get(model_name, '#0071e3'), width=2),
            hovertemplate=f"{model_name}<br>Day %{{x}}<br>MAPE: %{{y:.2f}}%<extra></extra>",
        ))
    fig_roll.update_layout(
        xaxis=dict(title='Test Day'),
        yaxis=dict(title='Rolling MAPE', ticksuffix='%'),
    )
    style_plotly(fig_roll, height=360)
    st.plotly_chart(fig_roll, use_container_width=True)

    st.markdown("""
    <div class="section-header">
      <h3>Model-to-Model Distance (MAE)</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    fig_heat = go.Figure(data=go.Heatmap(
        z=pairwise_mae_df.values,
        x=pairwise_mae_df.columns,
        y=pairwise_mae_df.index,
        colorscale=[
            [0.0, '#eff4ff'],
            [0.5, '#9ec5ff'],
            [1.0, '#0071e3'],
        ],
        colorbar=dict(title='MAE'),
        hovertemplate='Model X: %{x}<br>Model Y: %{y}<br>MAE gap: ₹%{z:,.2f}<extra></extra>',
    ))
    fig_heat.update_layout(
        xaxis=dict(title='Compared Model'),
        yaxis=dict(title='Reference Model'),
    )
    style_plotly(fig_heat, height=420)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    <div class="section-header">
      <h3>Baseline Benchmark (vs LSTM)</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

    benchmark_table = improvement_df.copy()
    benchmark_table['RMSE'] = benchmark_table['RMSE'].map(lambda x: f"₹{x:,.2f}")
    benchmark_table['MAE'] = benchmark_table['MAE'].map(lambda x: f"₹{x:,.2f}")
    benchmark_table['MAPE'] = benchmark_table['MAPE'].map(lambda x: f"{x:.2f}%")
    benchmark_table['MAE vs Baseline %'] = benchmark_table['MAE vs Baseline %'].map(lambda x: f"{x:+.2f}%")
    benchmark_table['MAPE vs Baseline %'] = benchmark_table['MAPE vs Baseline %'].map(lambda x: f"{x:+.2f}%")
    benchmark_table = benchmark_table.reset_index().rename(columns={'index': 'Model'})

    st.caption(f"Baseline model: {baseline_model_name}")
    st.dataframe(
        benchmark_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("""
    <div class="section-header">
      <h3>Recent Window Leaderboard</h3>
      <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)
    st.dataframe(recent_df, use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────
st.markdown("""
<br>
<div style="text-align:center; padding: 20px 0;
     border-top: 1px solid #d2d2d7; margin-top: 20px;">
  <p style="font-family:'SF Mono',Menlo,monospace; font-size:0.68rem;
     color:#6e6e73; letter-spacing:0.08em">
    NIFTY 50 MULTI-MODEL PREDICTOR · FOR EDUCATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE
  </p>
</div>
""", unsafe_allow_html=True)
