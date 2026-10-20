
    recent = df[["date", "close"]].tail(20).copy()
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
    return {
        "code": code.strip().upper(), "source": "app snapshot (not live)",
        "last_date": df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "last_close": float(df["close"].iloc[-1]),
        "observations": int(len(df)),
        "last_20": recent.to_dict(orient="records"),
    }


@mcp.tool()
def technical_analysis(code: str) -> dict:
    """Calculate SMA20, SMA50, RSI14, and approximate annualized volatility from saved closes."""
    df, error = _series(code)
    if error:
        return {"error": error}
    prices = df["close"].astype(float)
    if len(prices) < 20:
        return {"error": "At least 20 closing prices are required."}
    delta = prices.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    gain, loss = float(gains.iloc[-1]), float(losses.iloc[-1])
    rsi = 50.0 if gain == 0 and loss == 0 else 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    returns = np.log(prices / prices.shift(1)).dropna()
    volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else None
    return {
        "code": code.strip().upper(), "source": "app snapshot (not live)",
        "last_date": df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "close": float(prices.iloc[-1]),
        "sma20": float(prices.tail(20).mean()),
        "sma50": float(prices.tail(50).mean()) if len(prices) >= 50 else None,
        "rsi14": float(rsi),
        "annualized_volatility_approx": volatility,
    }


@mcp.tool()
def forecast_explanation(code: str) -> dict:
    """Retrieve an already saved LSTM forecast for a trading code; never create a forecast."""
    if not FORECAST_FILE.is_file():
        return {"error": "No saved LSTM forecast. Click Forecast price in the main app first."}
    try:
        payload = json.loads(FORECAST_FILE.read_text(encoding="utf-8"))
        saved_code = str(payload.get("code", payload.get("trading_code", ""))).upper()
        if saved_code != code.strip().upper():
            return {"error": "The saved forecast belongs to another trading code. Forecast this code first."}
        return payload
    except Exception as exc:
        logging.exception("Could not read saved forecast")
        return {"error": f"Could not read forecast: {type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    # MCP protocol uses stdout. Do not print anything here.
    mcp.run(transport="stdio")
