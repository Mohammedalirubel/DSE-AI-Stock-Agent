# DSE LSTM Price Forecast + Google ADK + MCP

This deployment bundle is based on the supplied DSE LSTM Streamlit app and earlier ADK/MCP integration. The original LSTM architecture, on-demand per-share training, 20-session window, 80/20 time split, forecast controls and charts are retained.

## Included files

- `app.py` — Streamlit DSE forecasting app and AI Agent tab
- `requirements.txt` — application dependencies
- `agent_ui.py` — Streamlit agent interface and local snapshot export
- `agent_service.py` — Google ADK runner connecting to MCP
- `mcp_server.py` — local stdio MCP tools
- `.streamlit/secrets.toml.example` — example secrets only; do not commit real API keys

## Deploy on Streamlit Community Cloud

1. Create a **new GitHub repository** (or use your duplicate). Do not overwrite your original app.
2. Upload **all five `.py` / `.txt` files** from this folder to the repository root, along with this README. The `agent_*.py` and `mcp_server.py` files are required for the agent tab.
3. Create a **new Streamlit app**, selecting that repository, `main` branch, and `app.py` as the entrypoint.
4. Under **Advanced settings**, choose **Python 3.12**. The Python version cannot be changed by adding a line to `requirements.txt`; for an existing app, redeploy as a new app to change Python.
5. Add your `GOOGLE_API_KEY` in the Streamlit Cloud app's **Secrets** interface. Never put your actual key in GitHub.
6. Deploy. Check the **About → Deployment diagnostics** expander for Python and installed dependency versions.
7. Run a forecast using DSE data or a CSV, then open **AI Agent + MCP**.

## Troubleshooting

If installation fails, inspect the **first ERROR line** in the deployment logs. The trailing `installer returned a non-zero exit code` line is not diagnostic. Earlier logs showed `tensorflow==2.20.0` unavailable in the then-selected environment; choosing Python 3.12 is the recommended first test, not a guarantee. If the build still fails, confirm the actual Python version and package index/architecture in the build logs.

## Limitations and security

- Models train on demand and are **not** loaded from previously saved `.keras` files. Training can be slow or exceed Streamlit Community Cloud memory.
- The MCP snapshot implementation stores data in shared local files, not per-user isolated storage. **Do not deploy for multi-user private financial data** without redesigning storage and authentication.
- Live DSE retrieval depends on `bdshare`; uploaded CSV and labeled demo data are alternatives.
- ADK and MCP integration has not been live-tested against your credentials or Cloud deployment.
- Forecasts are uncertain, especially at long horizons; no financial advice is implied.
