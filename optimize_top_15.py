import json
import pandas as pd
from ai_research_agents import ParameterGridSweeper
from leaderboard import load_leaderboard, add_strategy_to_leaderboard
from autonomous_research_loop import _compute_code_hash
from download_data import load_or_download

def run_optimization():
    print("Loading data...")
    full_df = load_or_download()
    df = full_df[full_df.index >= '2026-01-01'].copy()

    # Load top 15 strategies
    top_15 = load_leaderboard(df, range_key='2026_ytd')[:15]
    print(f"Loaded {len(top_15)} top strategies for optimization.")

    for i, strat in enumerate(top_15):
        print(f"\n[{i+1}/15] Optimizing '{strat['name']}'...")
        base_code = strat['code']
        old_net_r = strat.get('total_r', 0)
        
        try:
            # Optimize parameters
            opt_code, opt_stats, opt_trades, opt_monthly = ParameterGridSweeper.sweep_and_optimize(base_code, df)
            
            # Check improvement
            new_net_r = sum(opt_monthly.values()) if opt_monthly else opt_stats.get('total_pnl', 0.0) / 1000.0
            new_net_r = round(new_net_r, 1)
            
            if new_net_r > old_net_r and _compute_code_hash(opt_code) != _compute_code_hash(base_code):
                print(f"  --> Improvement found! {old_net_r}R -> {new_net_r}R")
                add_strategy_to_leaderboard(
                    name=strat['name'],
                    concept=strat['concept'],
                    code=opt_code,
                    stats=opt_stats,
                    trades=opt_trades,
                    author=strat['author'],
                    data_split='mt5_ecn_2026'
                )
            else:
                print(f"  --> No improvement found (Peak was {old_net_r}R, best sweep {new_net_r}R)")
        except Exception as e:
            print(f"  --> Error during optimization: {e}")

    print("\nOptimization complete! Leaderboard updated.")

if __name__ == '__main__':
    run_optimization()
