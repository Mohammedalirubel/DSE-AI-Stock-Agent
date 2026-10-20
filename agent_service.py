"""Google ADK connects to local MCP server over stdio."""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams
from mcp import StdioServerParameters
from google.genai import types

ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "mcp_server.py"


def _get_model() -> str:
    model = os.getenv("GEMINI_MODEL", "").strip()
    if not model or model.upper() in {
        "YOUR_AVAILABLE_MODEL_ID", "YOUR_MODEL_ID", "YOUR_GEMINI_MODEL"
    }:
        return "gemini-3.8-flash"
    return model


def _preflight_server() -> None:
    """Start the MCP server once and fail with its real error message.

    Without this, a server that crashes at startup only shows up as
    'Connection closed', which hides the actual cause.
    """
    init = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "preflight", "version": "0"},
        },
    })
    try:
        proc = subprocess.run(
            [sys.executable, str(SERVER)],
            input=init + "\n",
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return  # server stayed alive, which is fine
    if '"result"' not in proc.stdout:
        raise RuntimeError(
            f"MCP server failed to start (exit code {proc.returncode}).\n"
            f"stderr:\n{proc.stderr[-3000:] or '(empty)'}\n"
            f"stdout:\n{proc.stdout[-1000:] or '(empty)'}"
        )


def _make_connection() -> StdioConnectionParams:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    try:
        # Longer startup allowance for cold starts (pandas/numpy imports).
        return StdioConnectionParams(server_params=params, timeout=30.0)
    except Exception:
        return StdioConnectionParams(server_params=params)


async def ask_agent(question: str) -> str:
    await asyncio.to_thread(_preflight_server)

    toolset = MCPToolset(connection_params=_make_connection())
    agent = LlmAgent(
        name='dse_stock_analyst',
        model=_get_model(),
        instruction=(
            'You are a DSE stock research assistant. Use MCP tools for data, '
            'indicators and saved LSTM predictions. Never invent prices or '
            'forecasts. Distinguish synthetic demo data from actual DSE data. '
            'Forecasts are uncertain and not investment advice.'
        ),
        tools=[toolset],
    )
    runner = InMemoryRunner(agent=agent, app_name='dse_mcp_app')
    session = await runner.session_service.create_session(
        app_name='dse_mcp_app', user_id='streamlit_user'
    )
    answer = []
    try:
        async for event in runner.run_async(
            user_id='streamlit_user',
            session_id=session.id,
            new_message=types.Content(
                role='user', parts=[types.Part(text=question)]
            ),
        ):
            if event.is_final_response() and event.content:
                answer.extend(
                    p.text for p in event.content.parts if getattr(p, 'text', None)
                )
    finally:
        await toolset.close()
    return '\n'.join(answer) or 'No response returned.'
