
"""Streamlit UI for the DSE Google ADK + MCP agent."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent


def render_agent(data, source, code, last_date, last_close, result):
    st.subheader("AI Analyst (Google ADK + MCP)")
    st.caption(
        "Uses the same price snapshot as the existing predictor. "
        "Forecast tool reads the last calculated result."
    )

    if source == "demo":
        st.warning("Synthetic demo prices: not real market data.")

    # Save the price snapshot for the MCP subprocess.
    snapshot = ROOT / "agent_data.csv"

    try:
        data[["date", "code", "close", "volume"]].to_csv(
            snapshot, index=False
        )
    except Exception as exc:
        st.error(f"Could not save price snapshot: {exc}")

    # Save the latest LSTM forecast.
    forecast_path = ROOT / "agent_forecast.json"

    if result:
        try:
            payload = {
                "code": str(code).upper(),
                "last_date": str(last_date.date()),
                "last_close": float(last_close),
                "target_date": str(result["dates"][-1].date()),
                "predicted_close": float(result["preds"][-1]),
                "metrics": result.get("metrics", {}),
                "source": source,
            }

            forecast_path.write_text(
                json.dumps(payload, default=str),
                encoding="utf-8",
            )

        except Exception as exc:
            st.warning(f"Could not save forecast: {exc}")

    # MCP diagnostics.
    with st.expander("MCP connection diagnostics"):
        st.write("Python version:", sys.version.split()[0])
        st.write(
            "MCP server exists:",
            (ROOT / "mcp_server.py").is_file(),
        )
        st.write(
            "Price snapshot exists:",
            snapshot.is_file(),
        )
        st.write(
            "Forecast snapshot exists:",
            forecast_path.is_file(),
        )

        if st.button(
            "Diagnose MCP Server",
            key="diagnose_mcp_server",
        ):
            server = ROOT / "mcp_server.py"

            if not server.is_file():
                st.error("mcp_server.py is missing.")
            else:
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
                    st.code(
                        proc.stderr[-5000:]
                        or "(No stderr output)"
                    )
                    st.caption(
                        "A stdio server may exit normally "
                        "when given empty input. This test "
                        "does not verify the full MCP handshake."
                    )

                except subprocess.TimeoutExpired:
                    st.info(
                        "MCP subprocess stayed active "
                        "during the test."
                    )
                except Exception as exc:
                    st.exception(exc)

    # Google API configuration.
    api_key = (
        st.secrets.get("GOOGLE_API_KEY")
        or st.secrets.get("GEMINI_API_KEY")
    )

    if not api_key:
        st.info(
            "Add GOOGLE_API_KEY in Streamlit Secrets "
            "to enable the agent."
        )
        return

    os.environ["GOOGLE_API_KEY"] = api_key

    question = st.text_input(
        "Ask about the selected share",
        value=(
            f"Analyze {code}: recent price, "
            "technical indicators and available "
            "LSTM forecast. Mention limitations."
        ),
    )

    if st.button(
        "Ask AI Agent",
        key="ask_dse_ai_agent",
    ):
        if not question.strip():
            st.warning("Enter a question first.")
            return

        try:
            from agent_service import ask_agent

            with st.spinner("Analyzing via MCP tools..."):
                response = asyncio.run(
                    ask_agent(question)
                )

            st.write(
                response or "No response returned."
            )

        except Exception as exc:
            st.error(f"AI Agent error: {exc}")
            st.caption(
                "See Manage app → Logs "
                "for the full traceback."
            )
