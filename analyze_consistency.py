import json
import pandas as pd
import numpy as np
import time
from strategy_executor import execute_strategy

def calculate_consistency_score(prof_month_pct, vol, max_dd):
    # Score out of 100
    # 40% for prof_month_pct (e.g., 100% = 40 points)
    p_score = (prof_month_pct / 100.0) * 40.0
    
    # 30% for volatility. If vol is 0, 30 points. Vol of 10R means 0 points.
    v_score = max(0, 30.0 - (vol * 3.0)) 
    
    # 30% for drawdown. 0% DD = 30 points, 30% DD = 0 points.
    d_score = max(0, 30.0 - max_dd)
    
    return min(100.0, max(0.0, p_score + v_score + d_score))

def run():
    start_time = time.time()
    
    with open('data/leaderboard.json', 'r', encoding='utf-8') as f:
        lb = json.load(f)
        
    df = pd.read_csv('data/XAUUSD_5m.csv', index_col=0, parse_dates=True)
    start_period = df.index[0].to_period('M')
    end_period = df.index[-1].to_period('M')
    all_months = pd.period_range(start_period, end_period, freq='M')
    
    top_strats = lb[:100]
    results = []
    
    for i, strat in enumerate(top_strats):
        print(f"Testing {i+1}/{len(top_strats)}: {strat.get('name')}", flush=True)
        res = execute_strategy(strat['code'], df)
        if not res.get('success'):
            continue
            
        trades = res.get('trades', [])
        stats = res.get('stats', {})
        
        monthly_returns = {m.strftime('%Y-%m'): 0.0 for m in all_months}
        total_r = 0.0
        
        for t in trades:
            if 'exit_time' not in t: continue
            try:
                dt = pd.to_datetime(t['exit_time'])
                month_key = dt.strftime('%Y-%m')
            except:
                continue
                
            r_gain = 0.0
            if 'r_return' in t:
                r_gain = t['r_return']
            elif 'risk_usd' in t and t['risk_usd']:
                r_gain = t['pnl'] / t['risk_usd']
            else:
                r_gain = t['pnl'] / 1000.0
                
            if month_key in monthly_returns:
                monthly_returns[month_key] += r_gain
            total_r += r_gain
            
        months = [monthly_returns[m.strftime('%Y-%m')] for m in all_months]
        
        prof_months = sum(1 for m in months if m > 0)
        loss_months = sum(1 for m in months if m < 0) # 0 is breakeven, count strictly <0 as loss
        
        prof_month_pct = (prof_months / len(months) * 100) if len(months) > 0 else 0
        avg_monthly = np.mean(months)
        best_month = max(months)
        worst_month = min(months)
        monthly_vol = np.std(months) if len(months) > 1 else 0
        max_dd = stats.get('max_drawdown_pct', 0.0)
        
        longest_streak = 0
        current_streak = 0
        for m in months:
            if m < 0:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 0
                
        score = calculate_consistency_score(prof_month_pct, monthly_vol, max_dd)
        
        results.append({
            'name': strat.get('name', 'Unknown'),
            'total_r': total_r,
            'max_dd': max_dd,
            'prof_month_pct': prof_month_pct,
            'avg_monthly': avg_monthly,
            'best_month': best_month,
            'worst_month': worst_month,
            'longest_streak': longest_streak,
            'monthly_vol': monthly_vol,
            'pf': stats.get('profit_factor', 0.0),
            'score': score
        })
        
    results.sort(key=lambda x: x['score'], reverse=True)
    
    with open('consistency_leaderboard.md', 'w', encoding='utf-8') as f:
        f.write("# Consistency Leaderboard (Top 10)\n\n")
        f.write("| Rank | Strategy | Total Return (R) | Max Drawdown (%) | Profitable Months (%) | Avg Monthly (R) | Best Month (R) | Worst Month (R) | Longest Losing Streak | Monthly Volatility | PF | Consistency Score /100 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        
        for i, r in enumerate(results[:10]):
            f.write(f"| {i+1} | {r['name']} | {r['total_r']:.2f} | {r['max_dd']:.2f}% | {r['prof_month_pct']:.1f}% | {r['avg_monthly']:.2f} | {r['best_month']:.2f} | {r['worst_month']:.2f} | {r['longest_streak']} | {r['monthly_vol']:.2f} | {r['pf']:.2f} | **{r['score']:.1f}** |\n")
            
    print("Done")

if __name__ == '__main__':
    run()
