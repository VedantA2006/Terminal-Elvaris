import json
import pandas as pd
from strategy_executor import execute_strategy

def run():
    with open('data/leaderboard.json', 'r', encoding='utf-8') as f:
        lb = json.load(f)
        
    target_strat = next(s for s in lb if s.get('name') == 'Tick and Volume Confluence (Round #117)')
    
    df = pd.read_csv('data/XAUUSD_5m.csv', index_col=0, parse_dates=True)
    res = execute_strategy(target_strat['code'], df)
    
    trades = res.get('trades', [])
    monthly_returns = {}
    
    for t in trades:
        if 'exit_time' not in t: continue
        try:
            dt = pd.to_datetime(t['exit_time'])
            m_key = dt.strftime('%Y-%m')
            
            r_gain = 0.0
            if 'r_return' in t:
                r_gain = t['r_return']
            elif 'risk_usd' in t and t['risk_usd']:
                r_gain = t['pnl'] / t['risk_usd']
            else:
                r_gain = t['pnl'] / 1000.0
                
            monthly_returns[m_key] = monthly_returns.get(m_key, 0.0) + r_gain
        except:
            pass
            
    for k in sorted(monthly_returns.keys()):
        print(f"{k}: {monthly_returns[k]:.2f}")

if __name__ == '__main__':
    run()
