from __future__ import annotations

import json
import math
import subprocess
from html import escape
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ASX50 = {
    "BHP.AX": "BHP Group",
    "CBA.AX": "Commonwealth Bank",
    "CSL.AX": "CSL",
    "NAB.AX": "National Australia Bank",
    "WBC.AX": "Westpac Banking",
    "ANZ.AX": "ANZ Group",
    "MQG.AX": "Macquarie Group",
    "WES.AX": "Wesfarmers",
    "WOW.AX": "Woolworths Group",
    "GMG.AX": "Goodman Group",
    "FMG.AX": "Fortescue",
    "RIO.AX": "Rio Tinto",
    "TLS.AX": "Telstra Group",
    "TCL.AX": "Transurban Group",
    "ALL.AX": "Aristocrat Leisure",
    "REA.AX": "REA Group",
    "COL.AX": "Coles Group",
    "QBE.AX": "QBE Insurance",
    "SUN.AX": "Suncorp Group",
    "JHX.AX": "James Hardie Industries",
    "RMD.AX": "ResMed",
    "WDS.AX": "Woodside Energy",
    "STO.AX": "Santos",
    "BXB.AX": "Brambles",
    "COH.AX": "Cochlear",
    "XRO.AX": "Xero",
    "CPU.AX": "Computershare",
    "NST.AX": "Northern Star Resources",
    "NEM.AX": "Newmont",
    "ORG.AX": "Origin Energy",
    "IAG.AX": "Insurance Australia Group",
    "S32.AX": "South32",
    "SCG.AX": "Scentre Group",
    "AMC.AX": "Amcor",
    "QAN.AX": "Qantas Airways",
    "ASX.AX": "ASX",
    "MPL.AX": "Medibank",
    "BSL.AX": "BlueScope Steel",
    "SOL.AX": "Washington H. Soul Pattinson",
    "CAR.AX": "CAR Group",
    "SGP.AX": "Stockland",
    "PME.AX": "Pro Medicus",
    "SHL.AX": "Sonic Healthcare",
    "EDV.AX": "Endeavour Group",
    "TWE.AX": "Treasury Wine Estates",
    "MIN.AX": "Mineral Resources",
    "APA.AX": "APA Group",
    "FPH.AX": "Fisher & Paykel Healthcare",
    "SEK.AX": "SEEK",
    "ALD.AX": "Ampol",
}


BASE_PRICES = {
    "CBA.AX": 180,
    "BHP.AX": 42,
    "CSL.AX": 240,
    "MQG.AX": 220,
    "WES.AX": 80,
    "FMG.AX": 18,
    "RIO.AX": 115,
    "NAB.AX": 40,
}


def fmt_number(value: float, prefix: str = "", suffix: str = "") -> str:
    if pd.isna(value) or not np.isfinite(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value >= 1_000_000_000:
        body = f"{value / 1_000_000_000:.2f}b"
    elif value >= 1_000_000:
        body = f"{value / 1_000_000:.2f}m"
    elif value >= 1_000:
        body = f"{value / 1_000:.1f}k"
    else:
        body = f"{value:.2f}"
    return f"{sign}{prefix}{body}{suffix}"


def simulated_ohlcv(symbol: str) -> pd.DataFrame:
    seed = sum(ord(char) for char in symbol)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=520)
    base_price = BASE_PRICES.get(symbol, 20 + seed % 180)
    base_volume = 900_000 + (seed % 9_000_000)
    market_factor = rng.normal(0.00025, 0.009, len(dates))
    idio_factor = rng.normal(0, 0.014 + (seed % 7) / 1000, len(dates))
    returns = market_factor + idio_factor
    close = base_price * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.004, len(dates)))
    ranges = np.abs(rng.normal(0.014, 0.007, len(dates))) + np.abs(returns) * 0.7
    high = np.maximum(open_, close) * (1 + ranges / 2)
    low = np.minimum(open_, close) * (1 - ranges / 2)
    volume = base_volume * (1 + np.abs(returns) * 18) * rng.lognormal(0, 0.28, len(dates))
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume.astype(int),
        }
    )


def fetch_yahoo_chart(symbol: str, years: int = 2) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years + 10)
    period1 = int(start.timestamp())
    period2 = int(end.timestamp())
    cache_path = DATA_DIR / f"{symbol.replace('.', '_')}.csv"

    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        if not cached.empty and cached["date"].max() >= pd.Timestamp(end.date() - timedelta(days=3)):
            return cached

    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}?period1={period1}&period2={period2}&interval=1d&events=history"
    )
    response = subprocess.run(
        ["curl.exe", "-L", "-A", "Mozilla/5.0", url],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    if response.returncode != 0 or not response.stdout.strip():
        raise RuntimeError(response.stderr.strip() or f"API request failed for {symbol}")
    payload = json.loads(response.stdout)

    result = payload.get("chart", {}).get("result", [])
    if not result:
        raise RuntimeError(f"No chart data returned for {symbol}")

    data = result[0]
    quote = data["indicators"]["quote"][0]
    adjclose = data.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(data["timestamp"], unit="s").date,
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adj_close": adjclose if adjclose else quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=["close", "volume"]).sort_values("date")
    frame = frame[frame["volume"] > 0].tail(520)
    frame.to_csv(cache_path, index=False)
    return frame


@st.cache_data(ttl=60 * 60)
def load_market(symbols: tuple[str, ...], use_live_api: bool) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    failures = []
    for symbol in symbols:
        if use_live_api:
            try:
                frame = fetch_yahoo_chart(symbol)
                source = "api/cache"
            except (subprocess.SubprocessError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                frame = simulated_ohlcv(symbol)
                source = "demo fallback"
                failures.append(f"{symbol}: using demo fallback because API fetch failed ({exc})")
        else:
            frame = simulated_ohlcv(symbol)
            source = "demo fallback"

        frame["symbol"] = symbol
        frame["name"] = ASX50[symbol]
        frame["source"] = source
        frames.append(frame)

    if not frames:
        return pd.DataFrame(), failures

    market = pd.concat(frames, ignore_index=True)
    market["return"] = market.groupby("symbol")["adj_close"].pct_change()
    market["abs_return"] = market["return"].abs()
    market["range_pct"] = (market["high"] - market["low"]) / market["close"]
    market["dollar_volume"] = market["close"] * market["volume"]
    market["log_volume"] = np.log1p(market["volume"])
    market["day"] = market["date"].dt.day_name()
    market["month"] = market["date"].dt.to_period("M").astype(str)
    market = market.dropna(subset=["return", "abs_return", "range_pct", "dollar_volume"])
    return market, failures


def elasticity(frame: pd.DataFrame, x_col: str = "abs_return") -> float:
    clean = frame[[x_col, "log_volume"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 10 or clean[x_col].std() == 0:
        return float("nan")
    slope = np.polyfit(clean[x_col], clean["log_volume"], 1)[0]
    return math.expm1(slope * 0.01)


def summary_by_symbol(market: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, frame in market.groupby("symbol"):
        up = frame.loc[frame["return"] > 0, "volume"].mean()
        down = frame.loc[frame["return"] < 0, "volume"].mean()
        corr = frame["volume"].corr(frame["abs_return"])
        vol_elasticity = elasticity(frame)
        avg_dollar_volume = frame["dollar_volume"].mean()
        avg_range_pct = frame["range_pct"].mean()
        realized_vol_pct = frame["return"].std() * 100
        volatile_cutoff = frame["abs_return"].quantile(0.9)
        quiet_cutoff = frame["abs_return"].quantile(0.5)
        volatile_dv = frame.loc[frame["abs_return"] >= volatile_cutoff, "dollar_volume"].mean()
        quiet_dv = frame.loc[frame["abs_return"] <= quiet_cutoff, "dollar_volume"].mean()
        high_activity = frame[
            (frame["dollar_volume"] >= frame["dollar_volume"].quantile(0.75))
            & (frame["range_pct"] >= frame["range_pct"].median())
        ]
        rows.append(
            {
                "symbol": symbol,
                "name": frame["name"].iloc[0],
                "observations": len(frame),
                "avg_volume": frame["volume"].mean(),
                "avg_dollar_volume": avg_dollar_volume,
                "avg_abs_move_pct": frame["abs_return"].mean() * 100,
                "avg_range_pct": avg_range_pct * 100,
                "realized_vol_pct": realized_vol_pct,
                "amihud_illiquidity": (frame["abs_return"] / frame["dollar_volume"]).mean() * 1_000_000_000,
                "dollar_depth_per_1pct_range": avg_dollar_volume / (avg_range_pct * 100) if avg_range_pct else np.nan,
                "impact_bps_per_10m": (avg_range_pct * 10_000) / (avg_dollar_volume / 10_000_000) if avg_dollar_volume else np.nan,
                "vol_adjusted_dollar_volume": avg_dollar_volume / realized_vol_pct if realized_vol_pct else np.nan,
                "volume_abs_return_corr": corr,
                "volume_lift_on_down_days_pct": (down / up - 1) * 100 if up else np.nan,
                "volume_lift_volatile_vs_quiet_pct": (volatile_dv / quiet_dv - 1) * 100 if quiet_dv else np.nan,
                "volume_elasticity_per_1pp_move_pct": vol_elasticity * 100,
                "l2_proxy_days_pct": len(high_activity) / len(frame) * 100 if len(frame) else np.nan,
                "latest_close": frame["close"].iloc[-1],
                "latest_date": frame["date"].max().date().isoformat(),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_dollar_volume", ascending=False)


def make_price_volume_frame(frame: pd.DataFrame) -> pd.DataFrame:
    chart = frame.set_index("date")[["close", "volume", "range_pct"]].copy()
    chart["volume_m"] = chart["volume"] / 1_000_000
    chart["range_pct"] = chart["range_pct"] * 100
    return chart


def scale(values: pd.Series, low: float, high: float, invert: bool = False) -> list[float]:
    clean = values.astype(float)
    min_v = clean.min()
    max_v = clean.max()
    if not np.isfinite(min_v) or not np.isfinite(max_v) or max_v == min_v:
        return [float((low + high) / 2)] * len(clean)
    scaled = low + (clean - min_v) / (max_v - min_v) * (high - low)
    if invert:
        scaled = high - (scaled - low)
    return scaled.tolist()


def price_volume_svg(frame: pd.DataFrame) -> str:
    recent = frame.tail(180).reset_index(drop=True)
    width, height = 760, 330
    pad_l, pad_r, pad_t, pad_b = 54, 22, 24, 42
    xs = np.linspace(pad_l, width - pad_r, len(recent))
    price_y = scale(recent["close"], pad_t, height - pad_b - 80, invert=True)
    volume_top = height - pad_b - 70
    volume_y = scale(recent["volume"], volume_top, height - pad_b, invert=True)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, price_y))
    bars = []
    bar_w = max(1.5, (width - pad_l - pad_r) / max(len(recent), 1) * 0.58)
    for x, y in zip(xs, volume_y):
        bars.append(
            f"<rect x='{x - bar_w / 2:.1f}' y='{y:.1f}' width='{bar_w:.1f}' "
            f"height='{height - pad_b - y:.1f}' fill='#0f6b5f' opacity='.28' />"
        )
    first_date = recent["date"].min().date().isoformat()
    last_date = recent["date"].max().date().isoformat()
    return f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="340" role="img">
      <rect width="{width}" height="{height}" rx="8" fill="#ffffff" stroke="#e2e6eb" />
      <line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#d8dde3" />
      <line x1="{pad_l}" y1="{volume_top}" x2="{width - pad_r}" y2="{volume_top}" stroke="#eef1f4" />
      {''.join(bars)}
      <polyline points="{points}" fill="none" stroke="#1d4f91" stroke-width="2.6" />
      <text x="{pad_l}" y="{height - 16}" fill="#667085" font-size="12">{first_date}</text>
      <text x="{width - 118}" y="{height - 16}" fill="#667085" font-size="12">{last_date}</text>
      <text x="22" y="{height - 75}" fill="#0f6b5f" font-size="12">volume</text>
      <text x="22" y="52" fill="#1d4f91" font-size="12">close</text>
    </svg>
    """


def scatter_svg(frame: pd.DataFrame) -> str:
    sample = frame.tail(260).copy()
    width, height = 560, 330
    pad_l, pad_r, pad_t, pad_b = 58, 24, 32, 44
    sample["abs_move_pct"] = sample["abs_return"] * 100
    sample["volume_m"] = sample["volume"] / 1_000_000
    xs = scale(sample["abs_move_pct"], pad_l, width - pad_r)
    ys = scale(sample["volume_m"], pad_t, height - pad_b, invert=True)
    sizes = scale(sample["range_pct"], 3, 9)
    dots = "".join(
        f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r:.1f}' fill='#b7410e' opacity='.52' />"
        for x, y, r in zip(xs, ys, sizes)
    )
    return f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="340" role="img">
      <rect width="{width}" height="{height}" rx="8" fill="#ffffff" stroke="#e2e6eb" />
      <line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#d8dde3" />
      <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" stroke="#d8dde3" />
      {dots}
      <text x="{width / 2 - 62:.0f}" y="{height - 14}" fill="#667085" font-size="12">absolute daily move %</text>
      <text x="14" y="58" fill="#667085" font-size="12" transform="rotate(-90 14,58)">volume m</text>
    </svg>
    """


def day_profile_svg(day_profile: pd.DataFrame) -> str:
    width, height = 860, 280
    pad_l, pad_r, pad_t, pad_b = 54, 24, 28, 48
    labels = list(day_profile.index)
    values = day_profile["avg_volume_m"]
    max_v = max(values.max(), 1)
    slot = (width - pad_l - pad_r) / max(len(values), 1)
    bars = []
    for idx, (label, value) in enumerate(zip(labels, values)):
        bar_h = (value / max_v) * (height - pad_t - pad_b)
        x = pad_l + idx * slot + slot * 0.2
        y = height - pad_b - bar_h
        bars.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{slot * 0.6:.1f}' height='{bar_h:.1f}' fill='#1d4f91' opacity='.82' />"
            f"<text x='{x + slot * 0.3:.1f}' y='{height - 20}' text-anchor='middle' fill='#667085' font-size='12'>{escape(label[:3])}</text>"
        )
    return f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="280" role="img">
      <rect width="{width}" height="{height}" rx="8" fill="#ffffff" stroke="#e2e6eb" />
      <line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#d8dde3" />
      {''.join(bars)}
    </svg>
    """


def html_table(frame: pd.DataFrame) -> str:
    header = "".join(f"<th>{escape(str(col))}</th>" for col in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for col, value in row.items():
            if isinstance(value, (float, np.floating)):
                value = f"{value:,.2f}"
            elif isinstance(value, (int, np.integer)):
                value = f"{value:,}"
            cells.append(f"<td>{escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def style() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f8fa; color: #171a1f; }
        section[data-testid="stSidebar"] { background: #111820; color: #f8fafc; }
        h1, h2, h3 { letter-spacing: 0; }
        .hero {
            background: linear-gradient(135deg, #101820 0%, #203241 42%, #0f6b5f 100%);
            color: #fff;
            padding: 28px 30px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .hero p { color: #dce7e4; max-width: 960px; margin: 8px 0 0 0; }
        .answer {
            background: #fff;
            border: 1px solid #e2e6eb;
            border-left: 4px solid #0f6b5f;
            border-radius: 8px;
            padding: 16px 18px;
            min-height: 148px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
        }
        .answer b { color: #101820; }
        .answer ul { margin: 8px 0 0 18px; padding: 0; }
        .answer li { margin: 5px 0; }
        .note {
            background: #fff8e6;
            border: 1px solid #efd48b;
            padding: 12px 14px;
            border-radius: 8px;
            color: #463600;
        }
        .chart-title {
            color: #101820;
            font-size: 15px;
            font-weight: 700;
            line-height: 1.25;
            margin: 0 0 8px 0;
            min-height: 20px;
        }
        div[data-testid="stMetric"] {
            background: #fff;
            border: 1px solid #e2e6eb;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
        }
        .table-wrap {
            overflow-x: auto;
            background: #fff;
            border: 1px solid #e2e6eb;
            border-radius: 8px;
        }
        table { border-collapse: collapse; width: 100%; font-size: 13px; }
        th, td { padding: 10px 12px; border-bottom: 1px solid #edf0f3; text-align: left; white-space: nowrap; }
        th { background: #f3f5f7; color: #344054; font-weight: 700; }
        td { color: #1f2933; }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="ASX 50 Liquidity Dashboard", page_icon="26", layout="wide")
style()

st.markdown(
    """
    <div class="hero">
      <h1>ASX 50 Liquidity Dashboard</h1>
      <p>Two years of ASX 50 trading history converted into liquidity diagnostics: turnover depth, impact proxies, volatility response, price sensitivity, and where volume concentrates during stressed sessions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

symbols = tuple(ASX50.keys())
use_live_api = False

with st.spinner("Fetching ASX 50 market data and calculating liquidity diagnostics..."):
    market, failures = load_market(symbols, use_live_api)

if market.empty:
    st.error("No market data could be fetched. Check the internet connection and try again.")
    if use_live_api and failures:
        st.code("\n".join(failures[:20]))
    st.stop()

filter_options = ["All ASX 50"] + list(ASX50.keys())
stock_filter = st.multiselect(
    "Filter by stock",
    options=filter_options,
    default=["All ASX 50"],
    format_func=lambda s: "All ASX 50" if s == "All ASX 50" else f"{s.replace('.AX', '')} - {ASX50[s]}",
)
selected_filters = [symbol for symbol in stock_filter if symbol != "All ASX 50"]
if not stock_filter or "All ASX 50" in stock_filter:
    selected_filters = list(symbols)

market_view = market[market["symbol"].isin(selected_filters)].copy()
if market_view.empty:
    st.warning("Choose at least one stock to analyze.")
    st.stop()

summary = summary_by_symbol(market_view)
leader = summary.iloc[0]
selected_symbol = leader["symbol"]
focus = market_view[market_view["symbol"] == selected_symbol].copy()
focus_summary = leader
market_corr = market_view["volume"].corr(market_view["abs_return"])
overall_elasticity = elasticity(market_view) * 100
volatile_days = market_view[market_view["abs_return"] >= market_view["abs_return"].quantile(0.9)]
quiet_days = market_view[market_view["abs_return"] <= market_view["abs_return"].quantile(0.5)]
volatile_lift = (volatile_days["dollar_volume"].mean() / quiet_days["dollar_volume"].mean() - 1) * 100
scenario_notional = st.slider(
    "Scenario order notional",
    min_value=1_000_000,
    max_value=100_000_000,
    value=10_000_000,
    step=1_000_000,
    format="$%d",
)
summary["scenario_participation_pct"] = scenario_notional / summary["avg_dollar_volume"] * 100
summary["scenario_impact_bps"] = summary["impact_bps_per_10m"] * (scenario_notional / 10_000_000)
top3_turnover_share = summary.head(3)["avg_dollar_volume"].sum() / summary["avg_dollar_volume"].sum() * 100
best_scenario = summary.sort_values("scenario_impact_bps", ascending=True).iloc[0]
fragile_scenario = summary.sort_values("scenario_impact_bps", ascending=False).iloc[0]
avg_daily_turnover = summary["avg_dollar_volume"].mean()
median_depth = summary["dollar_depth_per_1pct_range"].median()
median_impact = summary["impact_bps_per_10m"].median()
median_amihud = summary["amihud_illiquidity"].median()

metric_cols = st.columns(5)
metric_cols[0].metric("Stocks analyzed", f"{summary['symbol'].nunique()} / {len(symbols)}")
metric_cols[1].metric("Avg daily turnover", fmt_number(avg_daily_turnover, prefix="$"))
metric_cols[2].metric("$ depth per 1% range", fmt_number(median_depth, prefix="$"))
metric_cols[3].metric("Impact proxy / $10m", f"{median_impact:.1f} bps")
metric_cols[4].metric("Amihud illiquidity", f"{median_amihud:.3f}")
st.caption("Metrics use the selected stock set; charts use the highest average dollar-volume stock inside that filter.")

st.subheader("Execution Scenario")
scenario_cols = st.columns(3)
with scenario_cols[0]:
    st.markdown(
        f"""
        <div class="answer">
        <b>Best capacity candidate</b><br>
        For a <b>{fmt_number(scenario_notional, prefix='$')}</b> order, <b>{best_scenario['symbol'].replace('.AX', '')}</b> has the lowest modeled impact at <b>{best_scenario['scenario_impact_bps']:.1f} bps</b>, with participation of <b>{best_scenario['scenario_participation_pct']:.1f}%</b> of average daily turnover.
        </div>
        """,
        unsafe_allow_html=True,
    )

with scenario_cols[1]:
    st.markdown(
        f"""
        <div class="answer">
        <b>Main execution risk</b><br>
        <b>{fragile_scenario['symbol'].replace('.AX', '')}</b> screens as the most fragile for the same notional: <b>{fragile_scenario['scenario_impact_bps']:.1f} bps</b> modeled impact and <b>{fragile_scenario['scenario_participation_pct']:.1f}%</b> participation. That is the name to slow down, slice, or monitor more tightly.
        </div>
        """,
        unsafe_allow_html=True,
    )

with scenario_cols[2]:
    st.markdown(
        f"""
        <div class="answer">
        <b>Portfolio-level read</b><br>
        Liquidity is concentrated: the top three selected names represent <b>{top3_turnover_share:.1f}%</b> of average turnover. That means a basket is not equally executable name-by-name; capacity is carried by a small part of the list.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("Liquidity Readout")
answer_cols = st.columns(2)

with answer_cols[0]:
    st.markdown(
        f"""
        <div class="answer">
        <b>Participation and urgency</b><br>
        <b>{selected_symbol.replace('.AX', '')}</b> is the deepest name in the selected set with average daily turnover of <b>{fmt_number(focus_summary['avg_dollar_volume'], prefix='$')}</b>. Down days trade <b>{focus_summary['volume_lift_on_down_days_pct']:.1f}%</b> more volume than up days, which points to stronger sell-side urgency when prices weaken.
        </div>
        """,
        unsafe_allow_html=True,
    )

with answer_cols[1]:
    st.markdown(
        f"""
        <div class="answer">
        <b>Depth under stress</b><br>
        Volatile sessions trade <b>{volatile_lift:.1f}%</b> more dollar volume than quiet sessions. For <b>{selected_symbol.replace('.AX', '')}</b>, high-turnover and above-normal range conditions occur on <b>{focus_summary['l2_proxy_days_pct']:.1f}%</b> of days, marking the sessions where order-book depth matters most.
        </div>
        """,
        unsafe_allow_html=True,
    )

answer_cols = st.columns(2)
with answer_cols[0]:
    st.markdown(
        f"""
        <div class="answer">
        <b>Price sensitivity</b><br>
        <b>{selected_symbol.replace('.AX', '')}</b> has a volume-to-absolute-move correlation of <b>{focus_summary['volume_abs_return_corr']:.2f}</b>. A 1 percentage point larger daily move is associated with about <b>{focus_summary['volume_elasticity_per_1pp_move_pct']:.1f}%</b> more volume, so activity rises materially as price discovery becomes more intense.
        </div>
        """,
        unsafe_allow_html=True,
    )

with answer_cols[1]:
    best = summary.sort_values("impact_bps_per_10m", ascending=True).iloc[0]
    st.markdown(
        f"""
        <div class="answer">
        <b>Execution cost proxy</b><br>
        The tightest impact profile is <b>{best['symbol'].replace('.AX', '')}</b> at <b>{best['impact_bps_per_10m']:.1f} bps per $10m</b>. The selected set median is <b>{median_impact:.1f} bps per $10m</b>, while median dollar depth is <b>{fmt_number(median_depth, prefix='$')}</b> per 1% daily range.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.subheader(f"Volume Interacting With Price: {selected_symbol.replace('.AX', '')}")

chart_cols = st.columns((1.4, 1))
with chart_cols[0]:
    st.markdown("<div class='chart-title'>Close price and volume</div>", unsafe_allow_html=True)
    st.markdown(price_volume_svg(focus), unsafe_allow_html=True)
with chart_cols[1]:
    st.markdown("<div class='chart-title'>Volume response to price movement</div>", unsafe_allow_html=True)
    st.markdown(scatter_svg(focus), unsafe_allow_html=True)

st.subheader("Ranked Liquidity Sensitivity")
st.caption("Use this table as a triage list: prioritize low-impact, high-depth names for larger notional flow; treat high-impact names as candidates for smaller slices or slower execution.")
display = summary[
    [
        "symbol",
        "name",
        "avg_dollar_volume",
        "scenario_participation_pct",
        "scenario_impact_bps",
        "dollar_depth_per_1pct_range",
        "impact_bps_per_10m",
        "amihud_illiquidity",
        "vol_adjusted_dollar_volume",
        "avg_abs_move_pct",
        "volume_abs_return_corr",
        "volume_elasticity_per_1pp_move_pct",
        "volume_lift_volatile_vs_quiet_pct",
    ]
].copy()
display["symbol"] = display["symbol"].str.replace(".AX", "", regex=False)
display = display.rename(
    columns={
        "symbol": "Ticker",
        "name": "Name",
        "avg_dollar_volume": "Avg $ volume",
        "scenario_participation_pct": "Scenario ADV %",
        "scenario_impact_bps": "Scenario impact bps",
        "dollar_depth_per_1pct_range": "$ depth / 1% range",
        "impact_bps_per_10m": "Impact bps / $10m",
        "amihud_illiquidity": "Amihud illiquidity",
        "vol_adjusted_dollar_volume": "Vol-adjusted $ volume",
        "avg_abs_move_pct": "Avg abs move %",
        "volume_abs_return_corr": "Vol/price corr",
        "volume_elasticity_per_1pp_move_pct": "Vol lift per 1pp move %",
        "volume_lift_volatile_vs_quiet_pct": "Volatile vs quiet $vol lift %",
    }
)
st.markdown(html_table(display), unsafe_allow_html=True)

st.subheader("Daily Pattern Proxy")
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
day_profile = (
    focus.groupby("day")
    .agg(avg_volume=("volume", "mean"), avg_abs_move=("abs_return", "mean"), avg_range=("range_pct", "mean"))
    .reindex(day_order)
    .dropna()
)
day_profile["avg_volume_m"] = day_profile["avg_volume"] / 1_000_000
day_profile["avg_abs_move_pct"] = day_profile["avg_abs_move"] * 100
st.markdown("<div class='chart-title'>Average volume by weekday</div>", unsafe_allow_html=True)
st.markdown(day_profile_svg(day_profile), unsafe_allow_html=True)

st.markdown(
    """
    <div class="note">
    <b>Decision rule:</b> move the scenario notional, then watch participation and modeled impact. High depth plus low impact suggests capacity; high impact plus high volatility response suggests fragility. The useful question is not "which stock traded most?" but "where does extra size begin to move the price?"
    </div>
    """,
    unsafe_allow_html=True,
)
