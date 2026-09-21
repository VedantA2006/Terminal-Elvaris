import json
import pandas as pd
from download_data import load_or_download
from strategy_executor import execute_strategy
from backtest import run_backtest

def run():
    print("Loading full 2-year dataset...")
    full_df = load_or_download()
    
    with open('data/leaderboard.json', 'r', encoding='utf-8') as f:
        leaderboard = json.load(f)
        
    top_30 = leaderboard[:30]
    results = []
    
    print(f"Testing {len(top_30)} strategies over {len(full_df)} candles (Full 2-Year Dataset)...")
    
    for i, s in enumerate(top_30):
        code = s.get('code')
        name = s.get('name')
        if not code:
            continue
            
        print(f"[{i+1}/{len(top_30)}] Evaluating {name}...")
        try:
            res = execute_strategy(code, full_df)
            trades = res.get('trades', [])
            stats = res.get('stats', {})
            
            if not trades:
                continue
                
            total_r = stats.get('total_pnl', 0) / 1000.0  # 1000 USD = 1R
            win_rate = stats.get('win_rate', 0)
            profit_factor = stats.get('profit_factor', 0)
            max_dd = stats.get('max_drawdown', 0) / 1000.0
            total_trades = len(trades)
            
            # Check monthly consistency
            df_trades = pd.DataFrame(trades)
            if not df_trades.empty:
                df_trades['exit_time_pd'] = pd.to_datetime(df_trades['exit_time'])
                df_trades['month'] = df_trades['exit_time_pd'].dt.to_period('M')
                
                # Monthly R calculation
                monthly_r = (df_trades.groupby('month')['pnl'].sum() / 1000.0)
                profitable_months = (monthly_r > 0).sum()
                total_months = len(monthly_r)
                consistency_pct = (profitable_months / total_months) * 100 if total_months > 0 else 0
                avg_monthly_r = monthly_r.mean() if total_months > 0 else 0
                
                # Average Monthly Drawdown calculation
                # Sort globally by time just in case
                df_trades = df_trades.sort_values('exit_time_pd')
                df_trades['cum_pnl'] = df_trades['pnl'].cumsum() / 1000.0
                df_trades['peak'] = df_trades['cum_pnl'].cummax()
                df_trades['dd'] = df_trades['peak'] - df_trades['cum_pnl']
                
                # Max drawdown within each month
                monthly_max_dd = df_trades.groupby('month')['dd'].max()
                avg_monthly_dd = monthly_max_dd.mean() if not monthly_max_dd.empty else 0
                avg_monthly_dd = monthly_max_dd.mean() if not monthly_max_dd.empty else 0
                
                # Daily calculations
                df_trades['day'] = df_trades['exit_time_pd'].dt.to_period('D')
                daily_r = (df_trades.groupby('day')['pnl'].sum() / 1000.0)
                avg_daily_r = daily_r.mean() if len(daily_r) > 0 else 0
                
                daily_max_dd = df_trades.groupby('day')['dd'].max()
                avg_daily_dd = daily_max_dd.mean() if not daily_max_dd.empty else 0
                
                results.append({
                    'Rank': s.get('rank', i+1),
                    'Name': name,
                    'Total R': round(total_r, 1),
                    'Avg Monthly R': round(avg_monthly_r, 1),
                    'Avg Monthly DD (R)': round(avg_monthly_dd, 1),
                    'Avg Daily R': round(avg_daily_r, 2),
                    'Avg Daily DD (R)': round(avg_daily_dd, 2),
                    'Max DD (R)': round(max_dd, 1),
                    'Win Rate %': round(win_rate, 1),
                    'Consistency %': round(consistency_pct, 1),
                    'Total Trades': total_trades,
                    'Profit Factor': round(profit_factor, 2)
                })
        except Exception as e:
            print(f"Error on {name}: {e}")
            
    # Sort by Avg Monthly R (or Consistency)
    results.sort(key=lambda x: (x['Consistency %'], x['Avg Monthly R']), reverse=True)
    
    # Generate Markdown Table
    md = "# Full 2-Year Backtest Analysis (Consistency & Profitability)\n\n"
    md += "This table ranks the top strategies across the *entire* continuous dataset (approx 2 years), focusing on monthly consistency, drawdowns, and real-world robustness out-of-sample.\n\n"
    md += "| Rank | Name | Avg Monthly Return | Avg Monthly DD | Avg Daily Return | Avg Daily DD | Consistency % | Total Return | Max Drawdown | Win Rate | Profit Factor | Trades |\n"
    md += "|------|------|--------------------|----------------|------------------|--------------|---------------|--------------|--------------|----------|---------------|--------|\n"
    
    for r in results:
        md += f"| {r['Rank']} | {r['Name']} | +{r['Avg Monthly R']}% | -{r['Avg Monthly DD (R)']}% | +{r['Avg Daily R']}% | -{r['Avg Daily DD (R)']}% | {r['Consistency %']}% | +{r['Total R']}% | -{r['Max DD (R)']}% | {r['Win Rate %']}% | {r['Profit Factor']} | {r['Total Trades']} |\n"
        
    with open('c:/Users/vedan/.gemini/antigravity-ide/brain/bc238e03-423d-456f-b72b-7de9a1f06bbd/consistency_ranking.md', 'w', encoding='utf-8') as f:
        f.write(md)
        
    print("\nAnalysis complete! Results saved to consistency_ranking.md")

if __name__ == "__main__":
    run()
