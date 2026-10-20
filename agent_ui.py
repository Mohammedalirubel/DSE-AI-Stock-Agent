import asyncio
import json
from pathlib import Path
import streamlit as st

def render_agent(data, source, code, last_date, last_close, result):
    st.subheader('AI Analyst (Google ADK + MCP)')
    st.caption('Uses the same price snapshot as the existing predictor. Forecast tool reads the last calculated result.')
    if source=='demo': st.warning('Synthetic demo prices: not real market data.')
    # Local snapshot is shared with the MCP subprocess; never include private credentials.
    snapshot=Path(__file__).with_name('agent_data.csv')
    data[['date','code','close','volume']].to_csv(snapshot,index=False)
    saved=Path(__file__).with_name('agent_forecast.json')
    if result:
        try:
            saved.write_text(json.dumps({'code':code.upper(),'last_date':str(last_date.date()),'last_close':last_close,'target_date':str(result['dates'][-1].date()),'predicted_close':float(result['preds'][-1]),'metrics':result['metrics'],'source':source},default=str),encoding='utf-8')
        except Exception as e: st.warning(f'Could not share forecast with agent: {e}')
    if not (st.secrets.get('GOOGLE_API_KEY') or st.secrets.get('GEMINI_API_KEY')):
        st.info('Add GOOGLE_API_KEY in Streamlit Secrets to enable the agent.')
        return
    import os
    os.environ['GOOGLE_API_KEY']=st.secrets.get('GOOGLE_API_KEY',st.secrets.get('GEMINI_API_KEY',''))
    q=st.text_input('Ask about the selected share',value=f'Analyze {code}: recent price, technical indicators and available LSTM forecast. Mention limitations.')
    if st.button('Ask AI Agent'):
        try:
            from agent_service import ask_agent
            with st.spinner('Analyzing via MCP tools...'):
                st.write(asyncio.run(ask_agent(q)))
        except Exception as e:
            st.error(f'AI Agent error: {e}')
