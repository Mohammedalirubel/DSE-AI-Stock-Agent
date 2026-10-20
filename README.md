# DSE LSTM + Google ADK + MCP

This is an additive integration of the uploaded `app (8).py` (not a pretrained-model loader). Its LSTM training, 80/20 validation, UI and forecasting functions are preserved. It trains per share on first request, as the original does.

## Deployment
1. Fork or duplicate the original GitHub repo to a new repository.
2. Upload these files, including the updated `app.py`, to the duplicate repository.
3. In Streamlit Community Cloud create a **new app** from the duplicate repository, with `app.py` as entry point. Choose Python 3.11 if available and ensure enough memory for TensorFlow.
4. In the new app's Secrets add `GOOGLE_API_KEY = "..."`. Never commit actual secrets.
5. Reboot. First run a forecast in the Forecast tab; then visit **AI Agent + MCP**.

## Architecture
- `app.py` is the user's original app plus an extra tab.
- `agent_ui.py` exports a price snapshot and last computed forecast to local files.
- `mcp_server.py` serves three tools over **stdio** using FastMCP.
- `agent_service.py` runs a Google ADK agent connected to those MCP tools.

## Important caveats
- The uploaded app trains LSTM models on demand; it does not load persisted `.keras` and scaler files. To use saved pretrained models, their path and loader must be supplied.
- Snapshot files are shared at the deployment filesystem level, **not per user**. Do not use this prototype for simultaneous private user sessions or multi-user access. For production, replace the files with authenticated session-scoped storage and enforce isolation.
- DSE data retrieval depends on `bdshare`, which may be unavailable. The app also supports uploaded CSV and clearly marked demo data.
- Streamlit Community Cloud has limited RAM; TensorFlow and ADK may exceed limits.
- Forecast tool only explains the last generated prediction; it does not run training from within MCP.
- LSTM forecasts can be negative or inaccurate, especially for long horizons; the original warning remains.
- The Sunday–Thursday calendar omits DSE public holidays.
- The ADK SDK is fast-moving: API compatibility should be verified on deployment. Syntax checked only, not integration tested against API credentials.
  ## requirements.txt
  streamlit
tensorflow==2.20.0
numpy
pandas
scikit-learn
altair
bdshare
google-adk
mcp
google-genai
joblib
plotly
requests
