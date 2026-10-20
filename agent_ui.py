"""Streamlit UI for the DSE AI stock agent.

Reconstructed file: the MCP test section and imports were rebuilt around the
code that was pasted. Adjust MCP_SERVER_CANDIDATES if your server file has a
different name.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

# File names to look for, in order. Change/add yours if needed.
MCP_SERVER_CANDIDATES = [
    "mcp_server.py",
    "dse_mcp_server.py",
    "server.py",
    "mcp_servers/dse_server.py",
]

# Used only when GEMINI_MODEL is missing in Streamlit Secrets.
# Verify this against Google's current model list.
DEFAULT_MODEL = "gemini-3.8-flash"

PLACEHOLDER_MODELS = {
    "YOUR_AVAILABLE_MODEL_ID",
    "YOUR_MODEL_ID",
    "YOUR_GEMINI_MODEL",
}


def _find_server() -> Path | None:
    """Return the first MCP server file that exists, or None."""
    for name in MCP_SERVER_CANDIDATES:
        path = ROOT / name
        if path.exists():
            return path
    return None


DATE_COLUMNS = ["date", "datetime", "trading_date", "time"]
CLOSE_COLUMNS = ["close", "closep", "close_price", "adj close", "adj_close", "ltp"]
CODE_COLUMNS = ["code", "symbol", "trading_code", "scrip"]


def _pick(columns: dict, candidates: list):
    """Return the original column name for the first matching candidate."""
    for name in candidates:
        if name in columns:
            return columns[name]
    return None


def _json_default(obj):
    """Make numpy/pandas values JSON-serializable."""
    if isinstance(obj, pd.DataFrame):
        return obj.tail(60).to_dict("records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _export_snapshot(data, code: str, current_result, root: Path = ROOT) -> dict:
    """Write agent_data.csv (and agent_forecast.json) for the MCP server.

    Returns a dict describing what was written, or {"error": "..."}.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        return {"error": "No price data is available to share with the agent."}

    df = data.copy()
    if not any(str(c).lower() in DATE_COLUMNS for c in df.columns):
        df = df.reset_index()  # date may be stored in the index

    columns = {str(c).lower(): c for c in df.columns}
    date_col = _pick(columns, DATE_COLUMNS)
    close_col = _pick(columns, CLOSE_COLUMNS)
    code_col = _pick(columns, CODE_COLUMNS)
    if date_col is None or close_col is None:
        return {
            "error": (
                "Could not find date/close columns in the app data. "
                f"Columns found: {list(map(str, data.columns))}"
            )
        }

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "close": pd.to_numeric(df[close_col], errors="coerce"),
    })
    if code_col is not None:
        codes = df[code_col].astype(str).str.upper()
        if (codes == code.upper()).any():
            out["code"] = codes
            out = out[out["code"] == code.upper()]
        else:
            out["code"] = code.upper()
    else:
        out["code"] = code.upper()
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    if out.empty:
        return {"error": "The app data has no usable date/close rows."}
    out[["date", "code", "close"]].to_csv(root / "agent_data.csv", index=False)

    info = {
        "rows": int(len(out)),
        "last_date": str(out["date"].iloc[-1].date()),
        "last_close": float(out["close"].iloc[-1]),
        "forecast_written": False,
        "forecast_preview": None,
    }

    forecast_path = root / "agent_forecast.json"
    has_result = current_result is not None and not (
        hasattr(current_result, "empty") and current_result.empty
    )
    if has_result:
        payload = {"source": "latest result passed from the main app"}
        if isinstance(current_result, dict):
            payload.update(current_result)
        else:
            payload["result"] = current_result
        payload["code"] = code.upper()  # set last so the server's check passes
        text = json.dumps(payload, default=_json_default, indent=2)
        forecast_path.write_text(text, encoding="utf-8")
        info["forecast_written"] = True
        info["forecast_preview"] = text[:3000]
    elif forecast_path.exists():
        forecast_path.unlink()  # drop a stale forecast from another share
    return info


def _probe_server(server: Path) -> None:
    """Start the MCP server once and show its exit code and stderr."""
    try:
        proc = subprocess.run(
            [sys.executable, str(server)],
            input="",
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(ROOT),
        )
        st.write("Exit code:", proc.returncode)
        st.code(proc.stderr[-5000:] or "(No stderr output)")
        if proc.stdout.strip():
            st.warning(
                "The server wrote to stdout. A stdio MCP server must not "
                "print(); send logs to sys.stderr instead."
            )
            st.code(proc.stdout[-2000:])
        st.caption(
            "A stdio server may exit normally when given empty input. "
            "This checks startup errors, not the complete MCP handshake."
        )
    except subprocess.TimeoutExpired:
        st.info("MCP subprocess stayed active during the test (this is good).")
    except Exception as exc:
        st.exception(exc)


def render_agent(
    data,
    source,
    code,
    last_date,
    last_close,
    current_result,
) -> None:
    """Render the AI agent panel for the selected share.

    Matches the call in app.py:
    render_agent(data, source, code, last_date, last_close, current_result)
    """
    st.subheader("AI Agent")

    code = str(code or "").strip()
    if not code:
        st.info("Select a share code to use the agent.")
        return

    st.caption(f"Selected share: {code} | Last date: {last_date} | Last close: {last_close}")

    # ---- MCP diagnostics -------------------------------------------------
    with st.expander("MCP server diagnostics"):
        server = _find_server()
        if server is None:
            st.warning(
                "No MCP server file found. Looked for: "
                + ", ".join(MCP_SERVER_CANDIDATES)
            )
        else:
            st.caption(f"MCP server file: {server.relative_to(ROOT)}")
            if st.button("Test MCP server startup", key="test_mcp_server"):
                _probe_server(server)

    # ---- API key and model -----------------------------------------------
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.info("Add GOOGLE_API_KEY in Streamlit Secrets to enable the agent.")
        return
    os.environ["GOOGLE_API_KEY"] = str(api_key)

    configured_model = str(st.secrets.get("GEMINI_MODEL", "")).strip()
    if not configured_model or configured_model.upper() in PLACEHOLDER_MODELS:
        st.warning(
            "GEMINI_MODEL is missing or contains an example placeholder. "
            f"Using {DEFAULT_MODEL} temporarily. Set a supported model "
            "in Streamlit Secrets."
        )
        configured_model = DEFAULT_MODEL
    os.environ["GEMINI_MODEL"] = configured_model
    st.caption(f"Active Gemini model: {configured_model}")

    # ---- Question and answer ---------------------------------------------
    question = st.text_input(
        "Ask about the selected share",
        value=(
            f"Analyze {code}: recent price, technical indicators and "
            "available LSTM forecast. Mention limitations."
        ),
    )

    if st.button("Ask AI Agent", key="ask_dse_ai_agent"):
        if not question.strip():
            st.warning("Enter a question first.")
            return
        try:
            exported = _export_snapshot(data, code, current_result)
            if "error" in exported:
                st.warning(exported["error"])
                return
            with st.expander("Data shared with the agent"):
                st.write(
                    f"{exported['rows']} price rows for {code} "
                    f"(last {exported['last_date']}, close {exported['last_close']})."
                )
                if exported["forecast_written"]:
                    st.caption("Latest app result shared as the forecast:")
                    st.code(exported["forecast_preview"], language="json")
                else:
                    st.caption("No forecast result available yet for this share.")

            from agent_service import ask_agent

            with st.spinner("Analyzing via MCP tools..."):
                response = asyncio.run(ask_agent(question))
            st.write(response or "No response returned.")
        except Exception as exc:
            status = getattr(exc, "code", None)
            if status == 503 or "503 UNAVAILABLE" in str(exc):
                st.warning(
                    "Gemini is temporarily overloaded (503). Please retry later."
                )
            else:
                st.error(f"AI Agent error: {exc}")
                text = f"{exc!r} {exc.__cause__!r}"
                if any(
                    key in text
                    for key in ("MCP", "Connection closed", "TaskGroup")
                ):
                    st.markdown("**MCP server startup output (likely cause):**")
                    server = _find_server()
                    if server is None:
                        st.warning(
                            "No MCP server file found. Looked for: "
                            + ", ".join(MCP_SERVER_CANDIDATES)
                        )
                    else:
                        _probe_server(server)
            st.caption("See Manage app → Logs for further details.")
