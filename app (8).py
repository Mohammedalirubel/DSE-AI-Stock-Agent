"""DSE Price Forecast: a phone-first Streamlit app built from predictive_analytics_LSTM.ipynb.

Same method as the notebook: one small LSTM per share, 20-session window, closing
price only, time-based 80/20 split, recursive multi-step forecast over Sun-Thu sessions.
"""
from __future__ import annotations

import html
import inspect
import re
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Settings (match the notebook)
# --------------------------------------------------------------------------------------
WINDOW = 20
TRAIN_RATIO = 0.80
EPOCHS = 15
BATCH_SIZE = 16
LOOKBACK_DAYS = 150
WEEKMASK = "Sun Mon Tue Wed Thu"  # DSE trading week (public holidays are not modelled)
HORIZONS = {"Next day": 1, "1 week": 5, "2 weeks": 10, "1 month": 22}

INK, TEAL, AMBER, UP, DOWN, GRID = "#12292C", "#0E7C86", "#D98E04", "#1B7F4B", "#C23B22", "#D5E2E2"

COLUMNS = ["DATE", "TRADING CODE", "LTP*", "HIGH", "LOW", "OPENP*", "CLOSEP*", "YCP", "TRADE", "VALUE (mn)", "VOLUME"]
ALIASES = {
    "date": "DATE", "tradingcode": "TRADING CODE", "symbol": "TRADING CODE", "code": "TRADING CODE",
    "ltp": "LTP*", "high": "HIGH", "low": "LOW", "openp": "OPENP*", "open": "OPENP*",
    "closep": "CLOSEP*", "close": "CLOSEP*", "ycp": "YCP", "trade": "TRADE",
    "value": "VALUE (mn)", "valuemn": "VALUE (mn)", "volume": "VOLUME",
}

# Mutual funds and treasury instruments are not ordinary shares, so they are left out.
EXCLUDED_CODES = {
    "1JANATAMF", "1STPRIMFMF", "AIBL1STIMF", "ATCSLGF", "CAPMBDBLMF", "CAPMIBBLMF", "DBH1STMF",
    "EBL1STMF", "EBLNRBMF", "EXIM1STMF", "FBFIF", "IC3RDNRB", "ICBAGRANI1", "ICBAMCL2ND",
    "ICBEPMF1S1", "ICBSONALI1", "IFILISLMF1", "LANKABA1MF", "LRGLOBMF1", "MBL1STMF", "NCCBLMF1",
    "NLI1STMF", "PHPMF1", "POPULAR1MF", "PRIME1ICBA", "RELIANCE1", "SEMLIBBLSF", "SEMLLECMF",
    "TRUSTB1MF", "VAMLBDG1", "VAMLRBBF",
}
EXCLUDE_PATTERN = r"MF$|MUTUAL|BOND|DEB|TB\d|TNOTE|TBILL"

st.set_page_config(
    page_title="DSE Price Forecast",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Newer Streamlit uses width="stretch"; older versions use use_container_width=True.
if "width" in inspect.signature(st.button).parameters:
    STRETCH = {"width": "stretch"}
else:
    STRETCH = {"use_container_width": True}

# --------------------------------------------------------------------------------------
# Styling: a single phone-width column, one bold forecast card, quiet everything else
# --------------------------------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700;800&display=swap');
.stApp {{ font-family: 'Figtree', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
.block-container {{ max-width: 620px; padding: 1.1rem 1rem 4rem; }}
footer, #MainMenu {{ visibility: hidden; }}
h1 {{ font-size: 1.65rem !important; font-weight: 800 !important; letter-spacing: -0.02em; padding: 0 0 .15rem !important; }}
h3 {{ font-size: 1.05rem !important; font-weight: 700 !important; }}
.sub {{ color: #4B6568; font-size: .95rem; margin: 0 0 .9rem; line-height: 1.35; }}
.stButton > button, .stDownloadButton > button {{ min-height: 3rem; border-radius: 12px; font-weight: 700; }}

.fc {{ border-radius: 18px; overflow: hidden; background: #fff; border: 1px solid {GRID}; margin: .4rem 0 1rem; }}
.fc-main {{ background: linear-gradient(160deg, #0E7C86 0%, #0A565D 100%); color: #fff; padding: 1.1rem 1.2rem 1.15rem; }}
.fc-label {{ font-size: .85rem; opacity: .85; font-weight: 500; }}
.fc-price {{ font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; margin: .2rem 0 .15rem; }}
.fc-delta {{ font-size: 1.05rem; font-weight: 700; }}
.fc-delta.up {{ color: #A6EBC0; }}
.fc-delta.down {{ color: #FFC2B6; }}
.fc-row {{ display: grid; grid-template-columns: repeat(3, 1fr); }}
.fc-row > div {{ padding: .8rem .9rem; border-right: 1px solid {GRID}; }}
.fc-row > div:last-child {{ border-right: 0; }}
.k {{ display: block; font-size: .72rem; color: #4B6568; font-weight: 500; }}
.v {{ display: block; font-size: 1.1rem; font-weight: 700; color: {INK}; }}
.s {{ display: block; font-size: .72rem; color: #6F8587; }}
.stats {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: .6rem; margin: .3rem 0 .8rem; }}
.stats > div {{ background: #fff; border: 1px solid {GRID}; border-radius: 12px; padding: .65rem .8rem; }}
.note {{ background: #FFF6E0; border-left: 4px solid {AMBER}; border-radius: 8px; padding: .65rem .8rem; font-size: .9rem; margin: .5rem 0; }}
.demo {{ background: #E8F1FF; border-left: 4px solid #3B6FD1; border-radius: 8px; padding: .65rem .8rem; font-size: .9rem; margin: .3rem 0 .7rem; }}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Tidy a DSE response into the standard columns (same logic as the notebook's API)."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = raw.reset_index().copy()
    df.columns = [ALIASES.get(re.sub(r"[^a-z0-9]", "", str(c).lower()), str(c)) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    if not {"DATE", "TRADING CODE", "CLOSEP*"}.issubset(df.columns):
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["TRADING CODE"] = df["TRADING CODE"].astype(str).str.upper().str.strip()
    for col in COLUMNS[2:]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    df = df.dropna(subset=["DATE"]).drop_duplicates(subset=["DATE", "TRADING CODE"])
    return df[COLUMNS].sort_values(["DATE", "TRADING CODE"]).reset_index(drop=True)


def to_frame(norm: pd.DataFrame) -> pd.DataFrame:
    """Keep ordinary shares with a real closing price (DSE reports 0 on days with no trades)."""
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(norm["DATE"]),
            "code": norm["TRADING CODE"],
            "close": norm["CLOSEP*"],
            "volume": norm["VOLUME"],
        }
    ).dropna(subset=["date", "code", "close"])
    out = out[out["close"] > 0]
    bad = out["code"].isin(EXCLUDED_CODES) | out["code"].str.contains(EXCLUDE_PATTERN, regex=True, na=False)
    bad |= out["code"].str.fullmatch(r"TB\d+Y\d+", na=False)
    return out[~bad].sort_values(["code", "date"]).reset_index(drop=True)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_dse(today_iso: str) -> pd.DataFrame:
    from bdshare import get_historical_data

    end = date.fromisoformat(today_iso)
    start = end - timedelta(days=LOOKBACK_DAYS)
    frame = to_frame(normalize(get_historical_data(start.isoformat(), end.isoformat())))
    if frame.empty:
        raise ValueError("The exchange returned no price rows for this period.")
    return frame


def demo_frame() -> pd.DataFrame:
    """Synthetic random-walk prices so the app can be explored when DSE is unreachable."""
    rng = np.random.default_rng(7)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=110, freq="C", weekmask=WEEKMASK)
    starts = {"BRACBANK": 41, "GP": 285, "SQURPHARMA": 210, "BATBC": 390, "ROBI": 26, "RENATA": 760,
              "CITYBANK": 24, "ISLAMIBANK": 33, "LHB": 54, "WALTONHIL": 880}
    rows = []
    for code, p0 in starts.items():
        price = p0 * np.exp(np.cumsum(rng.normal(0.0003, 0.013, len(days))))
        vol = rng.lognormal(12, 0.5, len(days))
        rows.append(pd.DataFrame({"date": days, "code": code, "close": price.round(1), "volume": vol.round()}))
    return pd.concat(rows, ignore_index=True)


def load_data() -> tuple[pd.DataFrame, str]:
    source = st.session_state.get("source", "dse")
    if source == "demo":
        return demo_frame(), "demo"
    if source == "csv":
        return st.session_state["csv_frame"], "csv"
    try:
        with st.spinner("Loading latest prices from DSE…"):
            return fetch_dse(date.today().isoformat()), "dse"
    except Exception as exc:  # network, DSE downtime, library change
        st.markdown("### Prices didn't load")
        st.write("The Dhaka Stock Exchange feed didn't respond. You can try again or use another source.")
        st.caption(f"Reason ({type(exc).__name__}): {exc}")
        if isinstance(exc, ModuleNotFoundError):
            st.warning(
                f"The package '{exc.name}' is not installed on this server. Add it to requirements.txt "
                "(in the repo root, next to app.py), push, then reboot the app."
            )
        if st.button("Try again", type="primary", **STRETCH):
            st.rerun()
        up = st.file_uploader("Or upload a CSV with DATE, TRADING CODE and CLOSEP* columns", type="csv")
        if up is not None:
            try:
                st.session_state["csv_frame"] = to_frame(normalize(pd.read_csv(up)))
                st.session_state["source"] = "csv"
                st.rerun()
            except Exception as err:
                st.error(f"That file couldn't be read: {err}")
        if st.button("Explore with demo data", **STRETCH):
            st.session_state["source"] = "demo"
            st.rerun()
        st.stop()


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=10)
def train_lstm(code: str, last_date: str, prices: tuple):
    """Train one LSTM for one share. The scaler is fitted on training rows only and kept
    with the model, so forecasting uses exactly the scaling the model learned."""
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras

    tf.keras.utils.set_random_seed(42)
    arr = np.asarray(prices, dtype="float64").reshape(-1, 1)
    n = len(arr)
    train_size = int(n * TRAIN_RATIO)
    if n < WINDOW + 10 or train_size <= WINDOW + 10:
        raise ValueError(f"{code} has only {n} trading days of data; at least {int((WINDOW + 11) / TRAIN_RATIO)} are needed.")

    scaler = MinMaxScaler().fit(arr[:train_size])
    scaled = scaler.transform(arr).ravel()
    X = np.stack([scaled[i - WINDOW : i] for i in range(WINDOW, n)]).astype("float32")[..., None]
    y = scaled[WINDOW:].astype("float32")
    target_idx = np.arange(WINDOW, n)
    train_mask, test_mask = target_idx < train_size, target_idx >= train_size

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(WINDOW, 1)),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    hist = model.fit(X[train_mask], y[train_mask], epochs=EPOCHS, batch_size=BATCH_SIZE,
                     validation_split=0.2, shuffle=False, verbose=0)

    # Honest check on the untouched last 20%: one-step-ahead predictions vs reality,
    # compared with the naive guess "tomorrow = today".
    pred = scaler.inverse_transform(model(X[test_mask], training=False).numpy()).ravel()
    actual = arr[WINDOW:, 0][test_mask]
    previous = arr[WINDOW - 1 : -1, 0][test_mask]
    moved = actual != previous
    if moved.any():
        direction = float(np.mean(np.sign(pred - previous)[moved] == np.sign(actual - previous)[moved])) * 100
    else:
        direction = float("nan")
    metrics = {
        "mae": float(np.mean(np.abs(pred - actual))),
        "naive_mae": float(np.mean(np.abs(previous - actual))),
        "mape": float(np.mean(np.abs(pred - actual) / actual) * 100),
        "direction": direction,
        "test_days": int(test_mask.sum()),
        "train_loss": float(hist.history["loss"][-1]),
        "val_loss": float(hist.history["val_loss"][-1]),
    }
    return model, scaler, metrics


def forecast_prices(model, scaler, prices: np.ndarray, steps: int) -> list[float]:
    window = scaler.transform(prices[-WINDOW:].reshape(-1, 1)).ravel().astype("float32")
    out = []
    for _ in range(steps):
        nxt = float(model(window.reshape(1, WINDOW, 1), training=False).numpy()[0, 0])
        window = np.append(window[1:], np.float32(nxt))
        out.append(float(scaler.inverse_transform([[nxt]])[0, 0]))
    return out


def trading_sessions(last: pd.Timestamp, n: int | None = None, end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    start = last + pd.Timedelta(days=1)
    if n is not None:
        return pd.bdate_range(start=start, periods=n, freq="C", weekmask=WEEKMASK)
    return pd.bdate_range(start=start, end=end, freq="C", weekmask=WEEKMASK)


# --------------------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------------------
def money(x: float) -> str:
    return f"{x:,.2f}"


def forecast_chart(history: pd.DataFrame, fc: pd.DataFrame) -> alt.Chart:
    data = pd.concat([history.assign(series="Actual"), fc.assign(series="Forecast")], ignore_index=True)
    domain = ["Actual", "Forecast"]
    base = alt.Chart(data).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0, tickCount=4, grid=False)),
        y=alt.Y("close:Q", title=None, scale=alt.Scale(zero=False), axis=alt.Axis(gridColor=GRID, tickCount=5)),
        color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=[INK, TEAL]),
                        legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=domain, range=[[1, 0], [6, 4]]), legend=None),
        tooltip=[alt.Tooltip("date:T", format="%a %d %b"), alt.Tooltip("close:Q", format=",.2f", title="Close")],
    )
    end = alt.Chart(fc.tail(1)).mark_point(filled=True, size=100, color=TEAL, opacity=1).encode(x="date:T", y="close:Q")
    return (base.mark_line(strokeWidth=2.4) + end).properties(height=270).configure_view(strokeWidth=0)


def history_chart(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_line(color=INK, strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%d %b", labelAngle=0, tickCount=4, grid=False)),
            y=alt.Y("close:Q", title=None, scale=alt.Scale(zero=False), axis=alt.Axis(gridColor=GRID, tickCount=5)),
            tooltip=[alt.Tooltip("date:T", format="%a %d %b %Y"), alt.Tooltip("close:Q", format=",.2f", title="Close")],
        )
        .properties(height=240)
        .configure_view(strokeWidth=0)
    )


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
st.title("DSE Price Forecast")
st.markdown('<p class="sub">LSTM closing-price forecasts for shares on the Dhaka Stock Exchange.</p>', unsafe_allow_html=True)

data, source = load_data()

if source != "dse":
    label = "uploaded CSV" if source == "csv" else "demo data (synthetic prices, not real shares)"
    st.markdown(f'<div class="demo">Using {label}.</div>', unsafe_allow_html=True)
    if st.button("Switch back to live DSE data", **STRETCH):
        st.session_state["source"] = "dse"
        st.rerun()

counts = data.groupby("code")["close"].count()
codes = sorted(counts[counts >= int((WINDOW + 11) / TRAIN_RATIO)].index)
if not codes:
    st.error("No share has enough price history to train a model. Try a longer date range.")
    st.stop()

code = st.selectbox(
    "Share", codes,
    index=codes.index("BRACBANK") if "BRACBANK" in codes else 0,
    help="Tap and type to search trading codes.",
)
horizon = st.segmented_control("How far ahead", [*HORIZONS, "Pick date"], default="1 week") or "1 week"

series = data[data["code"] == code].drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
last_date, last_close = series["date"].iloc[-1], float(series["close"].iloc[-1])
age = (pd.Timestamp.today().normalize() - last_date).days
if source == "dse" and age > 7:
    st.markdown(
        f'<div class="note">The latest price for {html.escape(code)} is from {last_date:%d %b %Y}, {age} days ago. It may not be trading.</div>',
        unsafe_allow_html=True,
    )

custom_target = None
if horizon == "Pick date":
    custom_target = pd.Timestamp(
        st.date_input(
            "Target date",
            value=(last_date + pd.Timedelta(days=7)).date(),
            min_value=(last_date + pd.Timedelta(days=1)).date(),
            max_value=(last_date + pd.Timedelta(days=120)).date(),
        )
    )

tab_forecast, tab_history, tab_about = st.tabs(["Forecast", "History", "About"])

with tab_forecast:
    if st.button("Forecast price", type="primary", **STRETCH):
        try:
            if custom_target is None:
                sessions = trading_sessions(last_date, n=HORIZONS[horizon])
            else:
                sessions = trading_sessions(last_date, end=custom_target)
            if len(sessions) == 0:
                raise ValueError("There is no trading session between the last price and that date (DSE trades Sunday to Thursday).")
            with st.spinner(f"Training the {code} model. The first run takes 10–20 seconds."):
                model, scaler, metrics = train_lstm(code, str(last_date.date()), tuple(np.round(series["close"].to_numpy(), 4)))
                preds = forecast_prices(model, scaler, series["close"].to_numpy(), len(sessions))
            st.session_state["result"] = {
                "key": (code, horizon, str(custom_target), source),
                "dates": sessions, "preds": preds, "metrics": metrics,
            }
        except Exception as exc:
            st.session_state.pop("result", None)
            st.error(f"Forecast failed: {exc}")

    res = st.session_state.get("result")
    if res and res["key"] == (code, horizon, str(custom_target), source):
        dates, preds, m = res["dates"], res["preds"], res["metrics"]
        target_pred = preds[-1]
        diff, pct = target_pred - last_close, (target_pred - last_close) / last_close * 100
        up = diff >= 0
        card = (
            '<div class="fc"><div class="fc-main">'
            f'<div class="fc-label">{html.escape(code)} predicted close on {dates[-1]:%a %d %b %Y}</div>'
            f'<div class="fc-price">৳ {money(target_pred)}</div>'
            f'<div class="fc-delta {"up" if up else "down"}">{"▲" if up else "▼"} {pct:+.2f}% ({diff:+,.2f})</div>'
            '</div><div class="fc-row">'
            f'<div><span class="k">Last close</span><span class="v">{money(last_close)}</span><span class="s">{last_date:%d %b}</span></div>'
            f'<div><span class="k">Sessions ahead</span><span class="v">{len(dates)}</span><span class="s">Sun–Thu</span></div>'
            f'<div><span class="k">Typical 1-day error</span><span class="v">±{m["mape"]:.1f}%</span><span class="s">held-out days</span></div>'
            "</div></div>"
        )
        st.markdown(card, unsafe_allow_html=True)

        hist = series.tail(60)[["date", "close"]]
        fc_df = pd.DataFrame({"date": [last_date, *dates], "close": [last_close, *preds]})
        st.altair_chart(forecast_chart(hist, fc_df), **STRETCH)

        if len(dates) > 30:
            st.markdown(
                '<div class="note">Forecasts this far out are repeated one-step guesses, so errors pile up. Treat the number as a rough direction at best.</div>',
                unsafe_allow_html=True,
            )
        if target_pred <= 0:
            st.markdown(
                '<div class="note">The model predicted a price at or below zero, which can\'t happen. Ignore this forecast.</div>',
                unsafe_allow_html=True,
            )

        with st.expander("How reliable is this share's model?"):
            ratio = m["mae"] / m["naive_mae"] if m["naive_mae"] else float("nan")
            if ratio < 0.95:
                verdict = "On days it hadn't seen, the model beat the simple guess \"tomorrow = today\"."
            elif ratio <= 1.05:
                verdict = "On days it hadn't seen, the model did about as well as the simple guess \"tomorrow = today\", so it adds little."
            else:
                verdict = "On days it hadn't seen, the model did worse than the simple guess \"tomorrow = today\". Treat this forecast with extra caution."
            st.write(verdict)
            st.dataframe(
                pd.DataFrame(
                    {
                        "Measure": ["Model error (BDT)", "\"Tomorrow = today\" error (BDT)", "Model error (%)", "Right direction", "Held-out days"],
                        "Value": [
                            f"{m['mae']:.2f}",
                            f"{m['naive_mae']:.2f}",
                            f"{m['mape']:.2f}%",
                            "n/a" if np.isnan(m["direction"]) else f"{m['direction']:.0f}% of moves",
                            str(m["test_days"]),
                        ],
                    }
                ),
                hide_index=True,
                **STRETCH,
            )
            st.caption("Errors come from predicting one day ahead on the last 20% of the data, which the model never trained on.")

        table = pd.DataFrame({"Date": dates.strftime("%a %d %b %Y"), "Predicted Close (BDT)": np.round(preds, 2)})
        table["Change vs last close"] = [f"{(p - last_close) / last_close * 100:+.2f}%" for p in preds]
        with st.expander("Day-by-day forecast"):
            st.dataframe(table, hide_index=True, **STRETCH)
        csv = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Predicted Close (BDT)": np.round(preds, 2)}).to_csv(index=False)
        st.download_button("Download forecast (CSV)", csv, file_name=f"DSE_{code}_LSTM_Forecast.csv", mime="text/csv", **STRETCH)
    elif not res:
        st.caption("Choose a share and how far ahead, then tap Forecast price.")
    else:
        st.caption("You changed the share or horizon. Tap Forecast price to update.")

with tab_history:
    st.markdown(f"### {html.escape(code)} closing price")
    st.altair_chart(history_chart(series[["date", "close"]]), **STRETCH)
    recent = series.tail(21)
    chg = (recent["close"].iloc[-1] / recent["close"].iloc[0] - 1) * 100 if len(recent) > 1 else 0.0
    vol = series["volume"].tail(20).mean()
    stats = [
        ("Last close", money(last_close), f"{last_date:%d %b %Y}"),
        ("Change over 20 sessions", f"{chg:+.2f}%", ""),
        ("Period high / low", f"{money(series['close'].max())} / {money(series['close'].min())}", f"{len(series)} sessions"),
        ("Average volume", "n/a" if pd.isna(vol) else f"{vol:,.0f}", "last 20 sessions"),
    ]
    st.markdown(
        '<div class="stats">'
        + "".join(f'<div><span class="k">{k}</span><span class="v">{v}</span><span class="s">{s}</span></div>' for k, v, s in stats)
        + "</div>",
        unsafe_allow_html=True,
    )

with tab_about:
    st.markdown(
        f"""
**How it works.** Each share gets its own small LSTM neural network. It reads the last {WINDOW} closing prices and predicts the next one.
To look further ahead, it feeds each prediction back in as if it had happened.

**Held-out check.** The model trains on the first 80% of the price history. The last 20% is kept aside, and the
"How reliable" panel scores the model on those days against the simple guess that tomorrow's close equals today's.

**Limits to keep in mind**
- It only sees closing prices. News, volume, and market-wide moves aren't in the model.
- The further ahead you look, the less the number means.
- DSE holidays aren't modelled, so session counts can be off by a day or two.
- Prices come from the DSE via the `bdshare` library for the last {LOOKBACK_DAYS} days and refresh every 6 hours.
"""
    )
