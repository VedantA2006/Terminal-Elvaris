import json
import pandas as pd
from download_data import load_or_download
from strategy_executor import execute_strategy

def compute_monthly_r_breakdown(trades, lot_size=100.0):
    records = []
    for t in trades:
        en_time = t.get('entry_time')
        if not en_time: continue
        try:
            dt = pd.to_datetime(en_time)
            if dt.year < 2026: continue
            month_key = dt.strftime('%b %Y')
            month_sort = dt.strftime('%Y-%m')
        except Exception:
            continue
        pnl = t.get('pnl', 0.0)
        if 'r_return' in t and t['r_return'] is not None:
            r_val = float(t['r_return'])
        else:
            entry = t.get('entry_price', 0.0)
            sl = t.get('sl', None)
            if sl is not None and entry > 0 and abs(entry - sl) > 0:
                risk_usd = abs(entry - sl) * lot_size
                r_val = pnl / risk_usd if risk_usd > 0 else (1.0 if pnl > 0 else -1.0)
            else:
                r_val = pnl / 1000.0
        records.append({'month': month_key, 'month_sort': month_sort, 'r': r_val})
        
    if not records:
        return {}
        
    df_trades = pd.DataFrame(records)
    monthly = df_trades.groupby(['month_sort', 'month'])['r'].sum().reset_index()
    monthly = monthly.sort_values('month_sort')
    
    return {row['month']: round(float(row['r']), 1) for _, row in monthly.iterrows()}


def run():
    print("Loading full dataset...")
    full_df = load_or_download()
    
    with open('strat_temp.py', 'r', encoding='utf-8') as f:
        code = f.read()

    print("Evaluating strategy on full dataset...")
    res = execute_strategy(code, full_df)
    trades = res.get('trades', [])
    
    monthly_r = compute_monthly_r_breakdown(trades)
    
    print("\nMonthly Returns (R) for 2026:")
    total_r = 0.0
    for month, r in monthly_r.items():
        print(f"{month}: {'+' if r > 0 else ''}{r}R")
        total_r += r
        
    print(f"\nTotal Return (YTD 2026): {total_r:.1f}R")

if __name__ == '__main__':
    run()
