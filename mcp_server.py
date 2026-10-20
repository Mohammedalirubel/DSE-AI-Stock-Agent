try:
    from mcp.server.fastmcp import FastMCP  # mcp 1.x
except ModuleNotFoundError:
    from mcp.server.mcpserver import MCPServer as FastMCP  # mcp 2.x (renamed)
import pandas as pd
import numpy as np
from pathlib import Path

mcp = FastMCP('DSE Analysis Tools')
DATA_FILE = Path(__file__).with_name('agent_data.csv')

def get_series(code: str):
    if not DATA_FILE.exists():
        return None, 'No shared market snapshot. Use the main app to export data first.'
    df = pd.read_csv(DATA_FILE, parse_dates=['date'])
    sub = df[df.code.astype(str).str.upper() == code.upper()].sort_values('date')
    if sub.empty:
        return None, 'Trading code not found in the available snapshot.'
    return sub, None

@mcp.tool()
def share_history(code: str) -> dict:
    """Return recent closing prices and date of a DSE share from the app's data snapshot."""
    df, err = get_series(code)
    if err: return {'error':err}
    return {'code':code.upper(), 'source':'app snapshot', 'last_date':str(df.date.iloc[-1].date()),'last_close':float(df.close.iloc[-1]),'observations':len(df),'last_20':df[['date','close']].tail(20).assign(date=lambda x:x.date.dt.strftime('%Y-%m-%d')).to_dict('records')}

@mcp.tool()
def technical_analysis(code: str) -> dict:
    """Compute SMA20, SMA50, RSI14 and historical volatility for a share."""
    df, err = get_series(code)
    if err: return {'error':err}
    p = df.close.astype(float)
    if len(p)<20: return {'error':'Need at least 20 price observations'}
    delta=p.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain.iloc[-1]/loss.iloc[-1] if loss.iloc[-1]>0 else np.inf
    returns=np.log(p/p.shift(1)).dropna()
    return {'code':code.upper(),'last_date':str(df.date.iloc[-1].date()),'close':float(p.iloc[-1]),'sma20':float(p.tail(20).mean()),'sma50':float(p.tail(50).mean()) if len(p)>=50 else None,'rsi14':float(100-100/(1+rs)),'annualized_volatility_approx':float(returns.std()*np.sqrt(252)) if len(returns)>1 else None,'source':'app snapshot'}

@mcp.tool()
def forecast_explanation(code: str) -> dict:
    """Return latest LSTM prediction saved by the existing app, without retraining."""
    f=Path(__file__).with_name('agent_forecast.json')
    if not f.exists(): return {'error':'First click Forecast price in the main Streamlit app.'}
    import json
    d=json.loads(f.read_text(encoding='utf-8'))
    if d.get('code')!=code.upper(): return {'error':'Saved forecast belongs to a different share. Forecast this code first.'}
    return d

if __name__=='__main__':
    mcp.run(transport='stdio')
