import json
import pandas as pd
from download_data import load_or_download
from strategy_executor import execute_strategy

def run():
    print("Loading full dataset...")
    full_df = load_or_download()
    
    with open('strat_temp.py', 'r', encoding='utf-8') as f:
        code = f.read()

    print("Evaluating strategy on full dataset to preserve indicator warmup...")
    res = execute_strategy(code, full_df)
    trades = res.get('trades', [])
    stats = res.get('stats', {})
    
    print(f"Total Trades overall: {len(trades)}")
    
    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades['exit_time_pd'] = pd.to_datetime(df_trades['exit_time'])
        
        # Filter trades that exited in 2026
        df_trades_2026 = df_trades[df_trades['exit_time_pd'].dt.year == 2026].copy()
        
        df_trades_2026['month'] = df_trades_2026['exit_time_pd'].dt.to_period('M')
        
        monthly_r = (df_trades_2026.groupby('month')['pnl'].sum() / 1000.0)
        
        print("\nMonthly Returns (R) for 2026:")
        for month, r in monthly_r.items():
            print(f"{month.strftime('%b')}: {'+' if r > 0 else ''}{r:.2f}R")
            
        total_2026_r = monthly_r.sum()
        print(f"\nTotal Return (YTD 2026): {total_2026_r:.2f}R")
    else:
        print("No trades found.")

if __name__ == '__main__':
    run()
