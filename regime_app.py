"""
Market Regime Detection — Streamlit App
Recreates the TASC May 2026 framework with a Plotly UI.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Regime Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3, .metric-label {
    font-family: 'IBM Plex Mono', monospace !important;
}

.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

.metric-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 16px 20px;
    text-align: center;
}

.metric-box .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
}

.metric-box .lbl {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

.green  { color: #3fb950; }
.red    { color: #f85149; }
.yellow { color: #d29922; }
.blue   { color: #58a6ff; }
.gray   { color: #8b949e; }

.regime-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
}

.on-badge  { background:#1a3a2a; color:#3fb950; border:1px solid #3fb950; }
.off-badge { background:#3a1a1a; color:#f85149; border:1px solid #f85149; }
.cau-badge { background:#3a2e0a; color:#d29922; border:1px solid #d29922; }

.stButton > button {
    background-color: #238636;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    padding: 8px 20px;
    width: 100%;
}

.stButton > button:hover {
    background-color: #2ea043;
}

hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
BG    = "#0d1117"
PAPER = "#161b22"
GREEN = "#3fb950"
RED   = "#f85149"
AMBER = "#d29922"
BLUE  = "#58a6ff"
GRAY  = "#8b949e"

# ── Data helpers ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def download_regime_data(start: str, end: str) -> pd.DataFrame:
    tickers = ["^VIX", "^VIX3M", "^GSPC", "HYG", "IEF"]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    raw.index = raw.index.tz_localize(None)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Adj Close"] if "Adj Close" in raw.columns.levels[0] else raw["Close"]
    else:
        close = raw

    mapping = {"^GSPC": "SPX", "^VIX": "VIX", "^VIX3M": "VIX3M", "HYG": "HYG", "IEF": "IEF"}
    close = close.rename(columns=mapping)
    df = close[["SPX", "VIX", "VIX3M", "HYG", "IEF"]].dropna().sort_index()
    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False)
def download_trading_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    t = yf.Ticker(symbol)
    df = t.history(start=start, end=end, interval="1d", auto_adjust=False)
    df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    return df


# ── Quant logic ────────────────────────────────────────────────────────────────

def calculate_regimes(data: pd.DataFrame) -> pd.DataFrame:
    r = data.copy()

    r["SPX_SMA"]      = r["SPX"].rolling(200).mean()
    r["SPX_Dist_SMA"] = r["SPX"] - r["SPX_SMA"]
    r["Signal_1"]     = np.where(r["SPX_Dist_SMA"] > 0, 1, 0)

    r["TS"]       = r["VIX"] / r["VIX3M"]
    r["Signal_2"] = np.where(r["TS"] < 1, 1, 0)

    r["CREDIT_RATIO"]    = r["HYG"] / r["IEF"]
    r["CREDIT_ROLL_M"]   = r["CREDIT_RATIO"].rolling(100).mean()
    r["CREDIT_ROLL_STD"] = r["CREDIT_RATIO"].rolling(100).std()
    r["CREDIT_Z"]        = (r["CREDIT_RATIO"] - r["CREDIT_ROLL_M"]) / r["CREDIT_ROLL_STD"]
    r["Signal_3"]        = np.where(r["CREDIT_Z"] > -2, 1, 0)

    r["Regime"] = 0
    r["Regime"] = np.where((r.Signal_1==1)&(r.Signal_2==1)&(r.Signal_3==1), 1, r.Regime)
    r["Regime"] = np.where((r.Signal_1==1)&(r.Signal_2==1)&(r.Signal_3==0), 2, r.Regime)
    r["Regime"] = np.where((r.Signal_1==1)&(r.Signal_2==0)&(r.Signal_3==1), 2, r.Regime)
    r["Regime"] = np.where((r.Signal_1==0)&(r.Signal_2==1)&(r.Signal_3==1), 2, r.Regime)

    r.dropna(inplace=True)
    return r


def generate_signals(regime_data: pd.DataFrame) -> pd.DataFrame:
    signals = regime_data[["Regime"]].copy()
    signals["year"]      = signals.index.isocalendar().year
    signals["month"]     = signals.index.month
    signals["dayofweek"] = signals.index.dayofweek
    signals["week"]      = signals.index.isocalendar().week

    signals["TargetPosition"] = 0.0
    signals.loc[signals.Regime == 1, "TargetPosition"] = 1.0
    signals.loc[signals.Regime == 0, "TargetPosition"] = 0.0
    signals.loc[signals.Regime == 2, "TargetPosition"] = 0.5
    signals["TargetPosition"] = signals["TargetPosition"].shift(1).ffill()

    last_week_signal = (
        signals.groupby(["year", "week"])
        .tail(1)
        .set_index(["year", "week"])["TargetPosition"]
    )
    shifted = last_week_signal.shift(1)
    signals["WeeklySignal"] = shifted.reindex(
        signals.set_index(["year", "week"]).index
    ).values
    signals["WeeklySignal"] = (
        signals.groupby(["year", "week"])["WeeklySignal"].ffill()
    )

    signals["PositionChange"] = 0
    for i in range(1, len(signals["WeeklySignal"])):
        if signals["WeeklySignal"].iloc[i] != signals["WeeklySignal"].iloc[i - 1]:
            signals["PositionChange"].iloc[i] = 1

    signals["WeeklyChange"]   = np.where(signals.dayofweek == 0, 1, 0)
    signals["WeeklyPosition"] = np.where(signals.dayofweek == 4, 0, 1)
    return signals


def run_backtest(signals: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    bt = pd.DataFrame(index=prices_df.index)
    bt["Close"]          = prices_df["Close"].values
    bt["PositionChange"] = signals["WeeklyChange"].reindex(prices_df.index).ffill().values
    bt["TargetPosition"] = signals["WeeklySignal"].reindex(prices_df.index).ffill().values

    bt["CurrentPosition"]  = 0.0
    bt["CurrentCash"]      = 10_000.0
    bt["CurrentHoldings"]  = 0.0
    bt["CurrentPortfolio"] = 10_000.0
    bt["Shares"]           = 0.0
    bt["trade_value"]      = 0.0
    bt["SharesChange"]     = 0.0
    bt.dropna(inplace=True)

    for i in range(1, len(bt)):
        ci = bt.index[i]
        pi = bt.index[i - 1]

        bt.loc[ci, "CurrentPosition"] = bt.loc[pi, "CurrentPosition"]
        bt.loc[ci, "CurrentCash"]     = bt.loc[pi, "CurrentCash"]
        bt.loc[ci, "Shares"]          = bt.loc[pi, "Shares"]

        target_pos  = bt.loc[ci, "TargetPosition"]
        current_pos = bt.loc[pi, "CurrentPosition"]
        exec_price  = bt.loc[ci, "Close"]

        if bt.loc[pi, "PositionChange"] == 1:
            if target_pos != 0:
                pv  = bt.loc[pi, "CurrentPortfolio"]
                hv  = bt.loc[pi, "CurrentHoldings"]
                csv = bt.loc[pi, "CurrentCash"]
                trade = target_pos - current_pos
                if trade > 0:
                    tv = csv * trade if hv == 0 else min(trade * pv, csv)
                    shares = int(tv / exec_price)
                    bt.loc[ci, "Shares"]      += shares
                    bt.loc[ci, "SharesChange"] = shares
                    bt.loc[ci, "CurrentCash"] -= shares * exec_price
                    bt.loc[ci, "trade_value"]  = tv
                else:
                    tv     = min(-trade * hv, csv)
                    shares = int(tv / exec_price)
                    bt.loc[ci, "Shares"]      -= shares
                    bt.loc[ci, "SharesChange"] = -shares
                    bt.loc[ci, "CurrentCash"] += shares * exec_price
                    bt.loc[ci, "trade_value"]  = tv
                bt.loc[ci, "CurrentPosition"] = target_pos
            else:
                liq = bt.loc[ci, "Shares"]
                bt.loc[ci, "Shares"]          -= liq
                bt.loc[ci, "SharesChange"]     = liq
                bt.loc[ci, "CurrentCash"]     += liq * exec_price
                bt.loc[ci, "trade_value"]      = liq * exec_price
                bt.loc[ci, "CurrentPosition"]  = target_pos

        bt.loc[ci, "CurrentHoldings"]  = bt.loc[ci, "Shares"] * exec_price
        bt.loc[ci, "CurrentPortfolio"] = bt.loc[ci, "CurrentCash"] + bt.loc[ci, "CurrentHoldings"]

    bt["Returns"]            = bt["Close"].pct_change()
    bt["Strategy_Returns"]   = bt["CurrentPortfolio"].pct_change()
    bt["Cumulative_Returns"] = (1 + bt["Strategy_Returns"]).cumprod()
    return bt


def run_benchmark(prices_df: pd.DataFrame) -> pd.DataFrame:
    b = pd.DataFrame(index=prices_df.index)
    b["Close"]             = prices_df["Close"]
    init                   = 10_000 / b["Close"].iloc[0]
    b["Portfolio_Value"]   = init * b["Close"]
    b["Returns"]           = b["Portfolio_Value"].pct_change()
    b["Cumulative_Returns"]= (1 + b["Returns"]).cumprod()
    return b


def calculate_metrics(strat: pd.Series, bench: pd.Series) -> dict:
    def _metrics(r):
        r  = r.dropna()
        tr = (1 + r).prod() - 1
        ny = len(r) / 252
        ar = (1 + tr) ** (1 / ny) - 1
        v  = r.std() * np.sqrt(252)
        s  = ar / v if v > 0 else 0
        c  = (1 + r).cumprod()
        dd = ((c - c.expanding().max()) / c.expanding().max()).min()
        return tr, ar, v, s, dd

    s_tr, s_ar, s_v, s_s, s_dd = _metrics(strat)
    b_tr, b_ar, b_v, b_s, b_dd = _metrics(bench)
    return {
        "Total Return": s_tr, "Annual Return": s_ar, "Volatility": s_v,
        "Sharpe Ratio": s_s, "Max Drawdown": s_dd,
        "Benchmark Return": b_tr, "Benchmark Annual Return": b_ar,
        "Benchmark Volatility": b_v, "Benchmark Sharpe": b_s, "Benchmark Max DD": b_dd,
    }


# ── Plotly charts ──────────────────────────────────────────────────────────────

REGIME_COLORS = {0: RED, 1: GREEN, 2: AMBER}
REGIME_LABELS = {0: "Risk OFF", 1: "Risk ON", 2: "Caution"}

def _add_regime_shading(fig, signals: pd.DataFrame, row: int, col: int = 1):
    """Add colour bands for each regime period."""
    if signals.empty:
        return
    dates  = signals.index.tolist()
    regime = signals["Regime"].values

    start_i = 0
    for i in range(1, len(regime)):
        if regime[i] != regime[start_i]:
            fig.add_vrect(
                x0=dates[start_i], x1=dates[i - 1],
                fillcolor=REGIME_COLORS[regime[start_i]],
                opacity=0.12, line_width=0,
                row=row, col=col,
            )
            start_i = i
    fig.add_vrect(
        x0=dates[start_i], x1=dates[-1],
        fillcolor=REGIME_COLORS[regime[start_i]],
        opacity=0.12, line_width=0,
        row=row, col=col,
    )


def plot_cumulative_returns(bt: pd.DataFrame, bench: pd.DataFrame,
                             signals: pd.DataFrame, metrics: dict) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.35, 0.18, 0.18, 0.29],
        vertical_spacing=0.04,
        subplot_titles=[
            "Cumulative Returns", "Market Regime", "Position Size", "Drawdown"
        ],
    )

    # ── Row 1 : cumulative returns ──
    _add_regime_shading(fig, signals, row=1)
    fig.add_trace(go.Scatter(
        x=bt.index, y=bt["Cumulative_Returns"],
        name=f'Strategy  ({metrics["Total Return"]:+.1%})',
        line=dict(color=BLUE, width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=bench.index, y=bench["Cumulative_Returns"],
        name=f'Benchmark ({metrics["Benchmark Return"]:+.1%})',
        line=dict(color=GRAY, width=1.5, dash="dot"),
    ), row=1, col=1)

    # ── Row 2 : regime ──
    for code, label, color in [(1, "Risk ON", GREEN), (2, "Caution", AMBER), (0, "Risk OFF", RED)]:
        mask = signals["Regime"] == code
        fig.add_trace(go.Scatter(
            x=signals.index[mask], y=[0.5] * mask.sum(),
            mode="markers", marker=dict(color=color, size=3, symbol="square"),
            name=label, showlegend=True,
        ), row=2, col=1)

    # ── Row 3 : position ──
    fig.add_trace(go.Scatter(
        x=bt.index, y=bt["TargetPosition"],
        fill="tozeroy", fillcolor="rgba(63,185,80,0.18)",
        line=dict(color=GREEN, width=1.5),
        name="Position", showlegend=False,
    ), row=3, col=1)

    # ── Row 4 : drawdown ──
    sc  = bt["Cumulative_Returns"]
    sdd = (sc - sc.expanding().max()) / sc.expanding().max()
    bc  = bench["Cumulative_Returns"]
    bdd = (bc - bc.expanding().max()) / bc.expanding().max()

    fig.add_trace(go.Scatter(
        x=bt.index, y=sdd,
        fill="tozeroy", fillcolor="rgba(248,81,73,0.20)",
        line=dict(color=RED, width=1.5),
        name="Strategy DD", showlegend=False,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=bench.index, y=bdd,
        line=dict(color=GRAY, width=1.2, dash="dot"),
        name="Benchmark DD", showlegend=False,
    ), row=4, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PAPER, plot_bgcolor=BG,
        height=820,
        legend=dict(orientation="h", y=1.03, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=60, b=20),
    )
    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor="#21262d", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#21262d", row=i, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    fig.update_yaxes(tickformat=".0%", row=4, col=1)
    fig.update_yaxes(range=[0, 1.1], row=2, col=1, showticklabels=False)
    return fig


def plot_spy_sma(trading_data: pd.DataFrame) -> go.Figure:
    d = trading_data.copy()
    d["SMA200"] = d["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.index, y=d["Close"],
        name="Price", line=dict(color=BLUE, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=d.index, y=d["SMA200"],
        name="SMA 200", line=dict(color=AMBER, width=1.5, dash="dash"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PAPER, plot_bgcolor=BG,
        title="Price vs SMA 200",
        height=380,
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=50, b=20),
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d"),
    )
    return fig


def plot_vix_term_structure(regime_data: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06,
                        subplot_titles=["VIX vs VIX3M", "VIX / VIX3M Ratio (Term Structure)"])

    fig.add_trace(go.Scatter(
        x=regime_data.index, y=regime_data["VIX"],
        name="VIX", line=dict(color=RED, width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=regime_data.index, y=regime_data["VIX3M"],
        name="VIX3M", line=dict(color=BLUE, width=1.5),
    ), row=1, col=1)

    ratio = regime_data["VIX"] / regime_data["VIX3M"]
    colors = [GREEN if v < 1 else RED for v in ratio]

    fig.add_trace(go.Bar(
        x=regime_data.index, y=ratio,
        marker_color=colors, name="VIX/VIX3M", showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=1, line_color=AMBER, line_dash="dash", row=2, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PAPER, plot_bgcolor=BG,
        height=480,
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=50, b=20),
    )
    for i in (1, 2):
        fig.update_xaxes(showgrid=True, gridcolor="#21262d", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#21262d", row=i, col=1)
    return fig


def plot_credit_zscore(regime: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06,
                        subplot_titles=["HYG / IEF Ratio", "Credit Z-Score (100-day rolling)"])

    fig.add_trace(go.Scatter(
        x=regime.index, y=regime["CREDIT_RATIO"],
        name="HYG/IEF", line=dict(color=BLUE, width=1.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=regime.index, y=regime["CREDIT_ROLL_M"],
        name="100d Mean", line=dict(color=AMBER, width=1.2, dash="dash"),
    ), row=1, col=1)

    z = regime["CREDIT_Z"]
    fig.add_trace(go.Scatter(
        x=regime.index, y=z,
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.12)",
        line=dict(color=BLUE, width=1.5),
        name="Z-Score", showlegend=False,
    ), row=2, col=1)
    for level, color in [(-2, RED), (-1, AMBER), (1, AMBER), (2, GREEN)]:
        fig.add_hline(y=level, line_color=color, line_dash="dot",
                      line_width=1, row=2, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PAPER, plot_bgcolor=BG,
        height=480,
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=50, b=20),
    )
    for i in (1, 2):
        fig.update_xaxes(showgrid=True, gridcolor="#21262d", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#21262d", row=i, col=1)
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    st.markdown("---")

    symbol = st.text_input("Trading Symbol", value="SPY",
                            help="Ticker for the traded instrument (e.g. SPY, QQQ, IWM)")

    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("Start", value=datetime.date(2007, 4, 12),
                                   min_value=datetime.date(2000, 1, 1))
    with col_e:
        end_date = st.date_input("End", value=datetime.date(2025, 12, 31))

    st.markdown("---")
    st.markdown("#### Signals")
    sma_window     = st.slider("SPX SMA window", 50, 400, 200, step=10)
    ts_threshold   = st.slider("VIX/VIX3M threshold", 0.7, 1.3, 1.0, step=0.05)
    credit_window  = st.slider("Credit ratio window", 50, 200, 100, step=10)
    credit_z_floor = st.slider("Credit Z floor (Risk OFF below)", -3.0, 0.0, -2.0, step=0.25)

    st.markdown("---")
    run_btn = st.button("▶  Run Analysis")

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style="font-family:'IBM Plex Mono',monospace; font-size:1.8rem; margin-bottom:0;">
    📊 Market Regime Detector
</h1>
<p style="color:#8b949e; font-size:0.9rem; margin-top:4px;">
    Multi-signal regime classification · SPX trend · VIX term structure · Credit spread
</p>
<hr style="border-color:#30363d; margin:12px 0 20px;">
""", unsafe_allow_html=True)

# ── Main logic ─────────────────────────────────────────────────────────────────

if run_btn or "bt" not in st.session_state:
    with st.spinner("Downloading market data…"):
        try:
            regime_data  = download_regime_data(str(start_date), str(end_date))
            trading_data = download_trading_data(symbol.upper(), str(start_date), str(end_date))
        except Exception as e:
            st.error(f"Data download failed: {e}")
            st.stop()

    if regime_data.empty or trading_data.empty:
        st.error("No data returned. Check your symbol and date range.")
        st.stop()

    with st.spinner("Calculating regimes…"):
        # Recalculate with custom parameters
        r = regime_data.copy()
        r["SPX_SMA"]      = r["SPX"].rolling(sma_window).mean()
        r["SPX_Dist_SMA"] = r["SPX"] - r["SPX_SMA"]
        r["Signal_1"]     = np.where(r["SPX_Dist_SMA"] > 0, 1, 0)
        r["TS"]           = r["VIX"] / r["VIX3M"]
        r["Signal_2"]     = np.where(r["TS"] < ts_threshold, 1, 0)
        r["CREDIT_RATIO"]    = r["HYG"] / r["IEF"]
        r["CREDIT_ROLL_M"]   = r["CREDIT_RATIO"].rolling(credit_window).mean()
        r["CREDIT_ROLL_STD"] = r["CREDIT_RATIO"].rolling(credit_window).std()
        r["CREDIT_Z"]        = (r["CREDIT_RATIO"] - r["CREDIT_ROLL_M"]) / r["CREDIT_ROLL_STD"]
        r["Signal_3"]        = np.where(r["CREDIT_Z"] > credit_z_floor, 1, 0)
        r["Regime"] = 0
        r["Regime"] = np.where((r.Signal_1==1)&(r.Signal_2==1)&(r.Signal_3==1), 1, r.Regime)
        r["Regime"] = np.where((r.Signal_1==1)&(r.Signal_2==1)&(r.Signal_3==0), 2, r.Regime)
        r["Regime"] = np.where((r.Signal_1==1)&(r.Signal_2==0)&(r.Signal_3==1), 2, r.Regime)
        r["Regime"] = np.where((r.Signal_1==0)&(r.Signal_2==1)&(r.Signal_3==1), 2, r.Regime)
        r.dropna(inplace=True)

        signals = generate_signals(r)
        common  = signals.index.intersection(trading_data.index)
        signals = signals.loc[common]
        tdata   = trading_data.loc[common]

        bt    = run_backtest(signals, tdata)
        bench = run_benchmark(tdata)
        m     = calculate_metrics(bt["Strategy_Returns"], bench["Returns"])

    st.session_state.update(dict(
        bt=bt, bench=bench, signals=signals,
        regime=r, regime_data=regime_data,
        trading_data=tdata, metrics=m, symbol=symbol.upper()
    ))

# ── Display ────────────────────────────────────────────────────────────────────

if "bt" not in st.session_state:
    st.info("Configure parameters in the sidebar and press **▶ Run Analysis**.")
    st.stop()

bt      = st.session_state["bt"]
bench   = st.session_state["bench"]
signals = st.session_state["signals"]
regime  = st.session_state["regime"]
regime_data = st.session_state["regime_data"]
m       = st.session_state["metrics"]
sym     = st.session_state["symbol"]

# Current regime badge
latest_regime = signals["Regime"].iloc[-1]
badge_html = {
    1: '<span class="regime-badge on-badge">● RISK ON</span>',
    0: '<span class="regime-badge off-badge">● RISK OFF</span>',
    2: '<span class="regime-badge cau-badge">● CAUTION</span>',
}[latest_regime]

col_b, col_d = st.columns([1, 3])
with col_b:
    st.markdown(f"**Current Regime** &nbsp; {badge_html}", unsafe_allow_html=True)
with col_d:
    regime_counts = signals["Regime"].value_counts()
    total = len(signals)
    parts = []
    for code, label in [(1, "Risk ON"), (2, "Caution"), (0, "Risk OFF")]:
        pct = regime_counts.get(code, 0) / total * 100
        parts.append(f"<span style='color:{REGIME_COLORS[code]}'>{label}: {pct:.0f}%</span>")
    st.markdown("&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts), unsafe_allow_html=True)

st.markdown("")

# Metric cards
def _color(v):
    return "green" if v >= 0 else "red"

cards = [
    (f"{m['Total Return']:+.1%}",          "Total Return",         _color(m['Total Return'])),
    (f"{m['Annual Return']:+.1%}",          "Annual Return",        _color(m['Annual Return'])),
    (f"{m['Sharpe Ratio']:.2f}",            "Sharpe Ratio",         _color(m['Sharpe Ratio'])),
    (f"{m['Max Drawdown']:.1%}",            "Max Drawdown",         "red"),
    (f"{m['Volatility']:.1%}",              "Volatility",           "blue"),
    (f"{m['Benchmark Return']:+.1%}",       f"{sym} B&H Return",    _color(m['Benchmark Return'])),
    (f"{m['Benchmark Annual Return']:+.1%}",f"{sym} B&H Annual",    _color(m['Benchmark Annual Return'])),
    (f"{m['Benchmark Sharpe']:.2f}",        f"{sym} B&H Sharpe",    _color(m['Benchmark Sharpe'])),
]

cols = st.columns(4)
for idx, (val, lbl, cls) in enumerate(cards):
    with cols[idx % 4]:
        st.markdown(f"""
        <div class="metric-box">
            <div class="val {cls}">{val}</div>
            <div class="lbl">{lbl}</div>
        </div><br>
        """, unsafe_allow_html=True)

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Performance & Regimes", "🔍 Signal Details", "📋 Data"])

with tab1:
    fig_main = plot_cumulative_returns(bt, bench, signals, m)
    st.plotly_chart(fig_main, use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        fig_spy = plot_spy_sma(st.session_state["trading_data"])
        st.plotly_chart(fig_spy, use_container_width=True)

        fig_vix = plot_vix_term_structure(regime_data)
        st.plotly_chart(fig_vix, use_container_width=True)
    with c2:
        fig_credit = plot_credit_zscore(regime)
        st.plotly_chart(fig_credit, use_container_width=True)

        # Signal agreement heatmap (monthly)
        # Signal_1/2/3 live in the regime df, not signals df — join on index
        sig_monthly = regime[["Signal_1","Signal_2","Signal_3"]].copy()
        sig_monthly = sig_monthly.reindex(signals.index)  # align to trading days
        sig_monthly["month"] = sig_monthly.index.to_period("M")
        agg = sig_monthly.groupby("month")[["Signal_1","Signal_2","Signal_3"]].mean()

        hm_fig = go.Figure(go.Heatmap(
            z=agg.T.values,
            x=[str(p) for p in agg.index][::3],
            y=["SPX>SMA200", "VIX Contango", "Credit OK"],
            colorscale=[[0, RED],[0.5, AMBER],[1, GREEN]],
            zmin=0, zmax=1,
            showscale=True,
            xgap=1, ygap=1,
        ))
        hm_fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=PAPER, plot_bgcolor=BG,
            title="Monthly Signal Agreement",
            height=260,
            margin=dict(l=100, r=20, t=50, b=60),
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(hm_fig, use_container_width=True)

with tab3:
    st.subheader("Backtest Portfolio")
    disp_cols = ["Close", "TargetPosition", "CurrentPortfolio", "CurrentCash",
                 "CurrentHoldings", "Shares", "Strategy_Returns", "Cumulative_Returns"]
    available = [c for c in disp_cols if c in bt.columns]
    st.dataframe(
        bt[available].tail(100).style.format({
            "Close": "{:.2f}", "TargetPosition": "{:.1%}",
            "CurrentPortfolio": "{:,.0f}", "CurrentCash": "{:,.0f}",
            "CurrentHoldings": "{:,.0f}", "Shares": "{:,.0f}",
            "Strategy_Returns": "{:+.2%}", "Cumulative_Returns": "{:.3f}",
        }),
        use_container_width=True,
    )

    st.subheader("Regime Signals")
    sig_show = signals[["Regime", "WeeklySignal", "PositionChange"]].copy()
    sig_show = sig_show.join(regime[["Signal_1", "Signal_2", "Signal_3"]], how="left")
    sig_show = sig_show[["Signal_1", "Signal_2", "Signal_3",
                          "Regime", "WeeklySignal", "PositionChange"]].tail(60)
    st.dataframe(sig_show, use_container_width=True)

st.markdown("""
<p style="text-align:center; color:#30363d; font-size:0.75rem; margin-top:30px;">
    Market Regime Detection Framework · SPX Trend × VIX Term Structure × Credit Spread · TASC May 2026
</p>
""", unsafe_allow_html=True)
