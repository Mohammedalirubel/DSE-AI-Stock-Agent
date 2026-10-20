"""Google ADK connects to local MCP server over stdio."""
import os
import sys
from pathlib import Path
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams
from mcp import StdioServerParameters
from google.genai import types

def _get_model() -> str:
    model = os.getenv("GEMINI_MODEL", "").strip()
    if not model or model.upper() in {
        "YOUR_AVAILABLE_MODEL_ID", "YOUR_MODEL_ID", "YOUR_GEMINI_MODEL"
    }:
        return "gemini-3.8-flash"
    return model


async def ask_agent(question: str) -> str:
    server = str(Path(__file__).with_name('mcp_server.py'))
    toolset = MCPToolset(connection_params=StdioConnectionParams(server_params=StdioServerParameters(command=sys.executable,args=[server])))
    agent = LlmAgent(name='dse_stock_analyst', model=_get_model(), instruction=('You are a DSE stock research assistant. Use MCP tools for data, indicators and saved LSTM predictions. Never invent prices or forecasts. Distinguish synthetic demo data from actual DSE data. Forecasts are uncertain and not investment advice.'), tools=[toolset])
    runner = InMemoryRunner(agent=agent, app_name='dse_mcp_app')
    session = await runner.session_service.create_session(app_name='dse_mcp_app', user_id='streamlit_user')
    answer=[]
    try:
        async for event in runner.run_async(user_id='streamlit_user',session_id=session.id,new_message=types.Content(role='user',parts=[types.Part(text=question)])):
            if event.is_final_response() and event.content:
                answer.extend(p.text for p in event.content.parts if getattr(p,'text',None))
    finally:
        await toolset.close()
    return '\n'.join(answer) or 'No response returned.'
