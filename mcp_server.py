
"""DSE MCP tools for the Streamlit AI Analyst (stdio transport).

Reads snapshots produced by the Streamlit application,
without importing Streamlit or retraining models.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "agent_data.csv"
FORECAST_FILE = ROOT / "agent_forecast.json"

mcp = FastMCP("DSE Analysis Tools")


def _series(code: str):
    code = str(code).strip().upper()

    if not code:
        return None, "Please provide a trading code."

    if not DATA_FILE.is_file():
        return None, (
            "No shared market snapshot (agent_data.csv). "
            "Load data in the Forecast tab first."
        )

    try:
        df = pd.read_csv(DATA_FILE)

        required = {"date", "close"}
        if not required.issubset(df.columns):
            return None, (
                "Snapshot must contain date and close columns."
            )

        code_col = (
            "code" if "code" in df.columns
            else "trading_code" if "trading_code" in df.columns
            else None
        )

        if code_col is None:
            return None, (
                "Snapshot must contain code or trading_code."
            )

        df = df.loc[
            df[code_col].astype(str).str.strip()
            .str.upper().eq(code)
        ].copy()

        df["date"] = pd.to_datetime(
            df["date"], errors="coerce"
        )
        df["close"] = pd.to_numeric(
            df["close"], errors="coerce"
        )

        df = df.dropna(subset=["date", "close"])
        df = (
            df.loc[df["close"] > 0]
            .sort_values("date")
            .drop_duplicates("date", keep="last")
        )

        if df.empty:
            return None, (
                f"No valid snapshot prices found for {code}."
            )

        return df, None

    except Exception as exc:
        logging.exception("Failed to read DSE snapshot")
        return None, (
            f"Could not read snapshot: "
            f"{type(exc).__name__}: {exc}"
        )


@mcp.tool()
def server_status() -> dict:
    """Check MCP server readiness and snapshot availability."""
    return {
        "status": "ready",
        "transport": "stdio",
        "snapshot_available": DATA_FILE.is_file(),
        "forecast_available": FORECAST_FILE.is_file(),
        "note": (
            "Snapshot files are local to this app instance, "
            "not live market feeds."
        ),
    }


@mcp.tool()
def share_history(code: str) -> dict:
    """Get the last 20 historical closing prices."""
    df, error = _series(code)

    if error:
        return {"error": error}

    recent = df[["date", "close"]].tail(20).copy()
    recent["date"] = recent["date"].dt.strftime(
        "%Y-%m-%d"
    )

    return {
        "code": code.strip().upper(),
        "source": "app snapshot (not live)",
        "last_date": df["date"].iloc[-1].strftime(
            "%Y-%m-%d"
        ),
        "last_close": float(df["close"].iloc[-1]),
        "observations": int(len(df)),
        "last_20": recent.to_dict(orient="records"),
    }


@mcp.tool()
def technical_analysis(code: str) -> dict:
    """Calculate SMA20, SMA50, RSI14 and volatility."""
    df, error = _series(code)

    if error:
        return {"error": error}

    prices = df["close"].astype(float)

    if len(prices) < 20:
        return {
            "error": (
                "At least 20 closing prices are required."
            )
        }

    delta = prices.diff()

    gains = delta.clip(lower=0).ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    losses = (-delta.clip(upper=0)).ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    gain = float(gains.iloc[-1])
    loss = float(losses.iloc[-1])

    if gain == 0 and loss == 0:
        rsi = 50.0
    elif loss == 0:
        rsi = 100.0
    else:
        rsi = 100 - 100 / (1 + gain / loss)

    returns = np.log(
        prices / prices.shift(1)
    ).dropna()

    volatility = (
        float(returns.std() * np.sqrt(252))
        if len(returns) > 1
        else None
    )

    return {
        "code": code.strip().upper(),
        "source": "app snapshot (not live)",
        "last_date": df["date"].iloc[-1].strftime(
            "%Y-%m-%d"
        ),
        "close": float(prices.iloc[-1]),
        "sma20": float(prices.tail(20).mean()),
        "sma50": (
            float(prices.tail(50).mean())
            if len(prices) >= 50
            else None
        ),
        "rsi14": float(rsi),
        "annualized_volatility_approx": volatility,
    }


@mcp.tool()
def forecast_explanation(code: str) -> dict:
    """Retrieve a saved LSTM forecast without recalculating."""
    if not FORECAST_FILE.is_file():
        return {
            "error": (
                "No saved LSTM forecast. "
                "Click Forecast price in the main app first."
            )
        }

    try:
        payload = json.loads(
            FORECAST_FILE.read_text(encoding="utf-8")
        )

        saved_code = str(
            payload.get(
                "code",
                payload.get("trading_code", "")
            )
        ).upper()

        if saved_code != code.strip().upper():
            return {
                "error": (
                    "The saved forecast belongs to "
                    "another trading code. "
                    "Forecast this code first."
                )
            }

        return payload

    except Exception as exc:
        logging.exception("Could not read saved forecast")
        return {
            "error": (
                f"Could not read forecast: "
                f"{type(exc).__name__}: {exc}"
            )
        }


if __name__ == "__main__":
    # stdout is reserved for the MCP protocol.
    mcp.run(transport="stdio")
