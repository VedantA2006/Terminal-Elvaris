import json
import sys

# Ensure clean UTF-8 console output
sys.stdout.reconfigure(encoding='utf-8')

with open('data/leaderboard.json', 'r', encoding='utf-8') as f:
    lb = json.load(f)

indices = [0, 6, 12] # Ranks 1, 7, 13

for idx in indices:
    if idx < len(lb):
        s = lb[idx]
        rank = idx + 1
        print(f"\n=======================================================")
        print(f"RANK #{rank}: {s.get('name')}")
        print(f"=======================================================")
        print(f"Strategy ID:     {s.get('id')}")
        print(f"Rank Score:      {s.get('rank_score')}  <--- RANKING METRIC")
        print(f"Total 6M Return: {s.get('total_r')} R")
        print(f"Total PnL:       ${s.get('total_pnl', 0):,.2f}")
        print(f"Trades Taken:    {s.get('total_trades')} trades ({s.get('winning_trades')}W / {s.get('losing_trades')}L)")
        print(f"Win Rate:        {s.get('win_rate')}%")
        print(f"Profit Factor:   {s.get('profit_factor')}")
        print(f"Sharpe Ratio:    {s.get('sharpe_ratio')}")
        print(f"Max Drawdown:    {s.get('max_drawdown_r')} R ({s.get('max_drawdown_pct')}%)")
        print(f"Consistency:     {s.get('months_ge_10r')} months with >= +10R")
        print(f"Monte Carlo VaR: {s.get('monte_carlo', {}).get('var_95_max_dd_r')} R (95% worst DD)")
        print(f"Data Splits:     Train = {s.get('train_r')}R | Val = {s.get('val_r')}R | Test = {s.get('test_r')}R")
        print(f"Monthly Breakdown: {s.get('monthly_r')}")
        print("\nFull Python Code:")
        print(s.get('code'))
