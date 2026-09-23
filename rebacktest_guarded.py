import json
import pandas as pd
from datetime import datetime

from autonomous_research_loop import ResearchLoopManager
from strategy_executor import execute_strategy
from leaderboard import compute_monthly_r_breakdown, compute_rank_score, run_monte_carlo

def run():
    print("Loading data...")
    train, val, test = ResearchLoopManager()._ensure_data('XAUUSD')
    df = pd.concat([train, val, test])
    
    print("Loading leaderboard...")
    with open('data/leaderboard.json', 'r', encoding='utf-8') as f:
        strats = json.load(f)
        
    # Sort and take top 30
    strats.sort(key=lambda x: -x.get('rank_score', 0))
    top_30 = strats[:30]
    
    # Evaluate with news guard active
    for i, s in enumerate(top_30):
        name = s.get('name', 'Unknown')
        code = s.get('code', '')
        
        safe_name = name[:40].encode('ascii', 'ignore').decode('ascii')
        print(f"[{i+1}/30] Evaluating {safe_name}...")
        
        # This will natively trigger the is_embargo logic inside _run_backtest_numba
        res = execute_strategy(code, df, spread=0.20, slippage=0.05, lot_size=100.0)
        
        if not res.get('success'):
            safe_name2 = name.encode('ascii', 'ignore').decode('ascii')
            print(f"  Warning: Failed to execute {safe_name2}")
            continue
            
        stats = res.get('stats', {})
        trades = res.get('trades', [])
        
        monthly_r = compute_monthly_r_breakdown(trades)
        months_ge_10 = sum(1 for v in monthly_r.values() if v >= 10.0)
        
        total_pnl = float(stats.get('total_pnl', 0.0))
        total_r = round(float(sum(monthly_r.values())), 1) if monthly_r else round(total_pnl / 1000.0, 1)
        max_dd = float(stats.get('max_drawdown', 0.0))
        max_dd_r = round(max_dd / 1000.0, 1)
        pf = float(stats.get('profit_factor', 0.0))
        wr = float(stats.get('win_rate', 0.0))
        total_trades = int(stats.get('total_trades', len(trades)))
        
        # Monte Carlo
        mc_results = run_monte_carlo(trades, num_simulations=1000)
        rank_score = compute_rank_score(total_r, months_ge_10, max_dd_r, pf, wr, total_trades)
        
        # Update strategy dictionary
        s['total_r'] = total_r
        s['total_trades'] = total_trades
        s['winning_trades'] = stats.get('winning_trades', 0)
        s['losing_trades'] = stats.get('losing_trades', 0)
        s['win_rate'] = wr
        s['profit_factor'] = pf
        s['total_pnl'] = total_pnl
        s['max_drawdown_r'] = max_dd_r
        s['max_drawdown_pct'] = float(stats.get('max_drawdown_pct', 0.0))
        s['sharpe_ratio'] = float(stats.get('sharpe_ratio', 0.0))
        s['monthly_r'] = monthly_r
        s['months_ge_10r'] = months_ge_10
        s['monte_carlo'] = {
            'expected_max_dd_r': mc_results.get('expected_max_dd_r'),
            'var_95_max_dd_r': mc_results.get('var_95_max_dd_r'),
            'risk_of_ruin_10r': mc_results.get('risk_of_ruin_10r'),
            'risk_of_ruin_20r': mc_results.get('risk_of_ruin_20r'),
            'probability_of_profit': mc_results.get('probability_of_profit'),
            'median_final_r': mc_results.get('median_final_r'),
        }
        s['rank_score'] = rank_score
        s['data_split'] = 'guarded_full'
        s['status'] = 'guarded'
        
        print(f"  -> New Score: {rank_score:.1f} | Total R: {total_r:+.1f} | DD: {max_dd_r}R | PF: {pf:.2f}")

    # Re-sort all strats (including the top 30 we just updated)
    strats.sort(key=lambda x: -x.get('rank_score', 0))
    
    # Save back
    print("\nSaving leaderboard.json...")
    with open('data/leaderboard.json', 'w', encoding='utf-8') as f:
        json.dump(strats, f, indent=2)
        
    print("Done!")

if __name__ == '__main__':
    run()
