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

    api_key = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.info("Add GOOGLE_API_KEY in Streamlit Secrets to enable the agent.")
        return
    os.environ["GOOGLE_API_KEY"] = str(api_key)
    configured_model = str(st.secrets.get("GEMINI_MODEL", "")).strip()
    if not configured_model or configured_model.upper() in {
        "YOUR_AVAILABLE_MODEL_ID", "YOUR_MODEL_ID", "YOUR_GEMINI_MODEL"
    }:
        st.warning(
            "GEMINI_MODEL is missing or contains an example placeholder. "
            "Using gemini-3.8-flash temporarily. Set a supported model "
            "in Streamlit Secrets."
        )
        configured_model = "gemini-3.8-flash"
    os.environ["GEMINI_MODEL"] = configured_model
    st.caption(f"Active Gemini model: {configured_model}")

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
                st.warning("Gemini is temporarily overloaded (503). Please retry later.")
            else:
                st.error(f"AI Agent error: {exc}")
            st.caption("See Manage app → Logs for further details.")
