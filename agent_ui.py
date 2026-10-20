"""Streamlit UI for the DSE AI stock agent.

Reconstructed file: the MCP test section and imports were rebuilt around the
code that was pasted. Adjust MCP_SERVER_CANDIDATES if your server file has a
different name.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

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
                    st.caption(
                        "A stdio server may exit normally when given empty input. "
                        "This checks startup errors, not the complete MCP handshake."
                    )
                except subprocess.TimeoutExpired:
                    st.info("MCP subprocess stayed active during the test.")
                except Exception as exc:
                    st.exception(exc)

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
            st.caption("See Manage app → Logs for further details.")
