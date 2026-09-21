import json
import warnings
import pandas as pd
from download_data import download_from_mt5
from leaderboard import LEADERBOARD_FILE, save_leaderboard, compute_monthly_r_breakdown, compute_rank_score
from strategy_executor import execute_strategy
from monte_carlo import run_monte_carlo
import sys
import codecs

# Fix Windows console encoding for printing weird characters (like non-breaking hyphens)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

warnings.filterwarnings('ignore')

def run():
    print("Fetching latest XAUUSD data from MT5...")
    full_df = download_from_mt5('XAUUSD', '5m', max_bars=100000)
    
    if full_df is None or len(full_df) == 0:
        print("Failed to download data from MT5.")
        return
        
    print(f"Successfully downloaded {len(full_df)} bars. Latest bar: {full_df.index[-1]}")
    
    if isinstance(full_df.index, pd.DatetimeIndex):
        df_for_calc = full_df[full_df.index >= '2026-01-01'].copy()
    else:
        df_for_calc = full_df
        
    with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f:
        leaderboard = json.load(f)
        
    leaderboard.sort(key=lambda x: x.get('total_r', 0.0), reverse=True)
    top_30 = leaderboard[:30]
    updated_count = 0
    
    print("\nRe-evaluating Top 30 Strategies with Latest Data...\n")
    
    for i, entry in enumerate(top_30):
        # Clean up the name string to avoid CP1252 charmap errors
        name = entry.get('name', 'Unknown').encode('ascii', 'replace').decode('ascii')
        old_r = entry.get('total_r', 0.0)
        code = entry.get('code', '')
        
        if not code:
            continue
            
        print(f"[{i+1}/30] Evaluating: {name} (Old R: {old_r})...")
        
        try:
            res = execute_strategy(code, df_for_calc)
            if res.get('success'):
                stats = res.get('stats', {})
                trades = res.get('trades', [])
                
                monthly_r = compute_monthly_r_breakdown(trades)
                months_ge_10 = sum(1 for v in monthly_r.values() if v >= 10.0)
                new_total_r = round(float(sum(monthly_r.values())), 1) if monthly_r else round(float(stats.get('total_pnl', 0.0) / 1000.0), 1)
                max_dd_r = round(float(stats.get('max_drawdown', 0.0) / 1000.0), 1)
                mc_results = run_monte_carlo(trades, num_simulations=100)
                pf = float(stats.get('profit_factor', 1.0))
                win_rate = float(stats.get('win_rate', 50.0))
                total_trades = int(stats.get('total_trades', len(trades)))
                
                rank_score = compute_rank_score(new_total_r, months_ge_10, max_dd_r, pf, win_rate, total_trades)
                
                diff = new_total_r - old_r
                diff_str = f"+{diff:.1f}" if diff >= 0 else f"{diff:.1f}"
                print(f"   => New R: {new_total_r}R (Diff: {diff_str}R)\n")
                
                entry.update({
                    'total_r': new_total_r,
                    'total_trades': total_trades,
                    'winning_trades': stats.get('winning_trades', 0),
                    'losing_trades': stats.get('losing_trades', 0),
                    'win_rate': win_rate,
                    'profit_factor': pf,
                    'total_pnl': float(stats.get('total_pnl', 0.0)),
                    'max_drawdown_r': max_dd_r,
                    'max_drawdown_pct': float(stats.get('max_drawdown_pct', 0.0)),
                    'sharpe_ratio': float(stats.get('sharpe_ratio', 0.0)),
                    'monthly_r': monthly_r,
                    'months_ge_10r': months_ge_10,
                    'monte_carlo': {
                        'expected_max_dd_r': mc_results.get('expected_max_dd_r'),
                        'var_95_max_dd_r': mc_results.get('var_95_max_dd_r'),
                        'risk_of_ruin_10r': mc_results.get('risk_of_ruin_10r'),
                        'risk_of_ruin_20r': mc_results.get('risk_of_ruin_20r'),
                        'probability_of_profit': mc_results.get('probability_of_profit'),
                        'median_final_r': mc_results.get('median_final_r'),
                    },
                    'rank_score': rank_score,
                })
                updated_count += 1
            else:
                print(f"   => Failed execution: {res.get('error', 'Unknown error')}\n")
        except Exception as e:
            print(f"   => Error evaluating: {e}\n")
            
    print(f"Successfully evaluated {updated_count} out of 30 strategies.")
    save_leaderboard(leaderboard)
    print("Leaderboard updated and saved.")

if __name__ == '__main__':
    run()
