import json
import pandas as pd
from autonomous_research_loop import ResearchLoopManager
from ai_research_agents import ParameterGridSweeper

def run():
    print("Loading data...")
    loop = ResearchLoopManager()
    train_df, _, _ = loop._ensure_data()
    
    with open('data/leaderboard.json', 'r', encoding='utf-8') as f:
        leaderboard = json.load(f)

    # We want top 30, and only those without "Optimized" in the name
    top_30 = leaderboard[:30]
    changed = False
    
    print("Scanning Top 30 for unoptimized strategies...")
    for s in top_30:
        name = s.get('name', '')
        if '(Optimized)' not in name and 'Peak Optimized' not in name and 'Absolute Peak' not in name:
            print(f"\n[+] Optimizing: {name}")
            code = s.get('code')
            
            # Using ParameterGridSweeper directly
            best_code, best_stats, best_trades, best_monthly = ParameterGridSweeper.sweep_and_optimize(code, train_df)
            
            # If the sweeper returns valid stats and Net PnL improved
            if best_stats and best_stats.get('total_pnl', 0) > s.get('stats', {}).get('total_pnl', 0):
                old_pnl = s.get('stats', {}).get('total_pnl', 0)
                new_pnl = best_stats.get('total_pnl', 0)
                print(f"  -> Improved! Net PnL: ${old_pnl:,.2f} -> ${new_pnl:,.2f}")
                
                s['code'] = best_code
                s['stats'] = best_stats
                s['name'] = name + " (Optimized)"
                
                net_r = sum(best_monthly.values()) if best_monthly else new_pnl / 1000.0
                s['total_r'] = round(net_r, 1)
                s['win_rate'] = round(best_stats.get('win_rate', 0), 1)
                s['profit_factor'] = round(best_stats.get('profit_factor', 1), 2)
                s['months_ge_10r'] = sum(1 for v in best_monthly.values() if v >= 10.0)
                
                if best_stats.get('max_drawdown', 0) > 0:
                    s['max_drawdown_r'] = round(best_stats.get('max_drawdown', 0) / 1000.0, 1)
                    
                changed = True
            else:
                print(f"  -> No mathematical improvement found using Grid Sweeper. Labeling as Absolute Peak.")
                s['name'] = name + " (Absolute Peak)"
                changed = True

    if changed:
        print("\nRe-sorting and saving leaderboard...")
        leaderboard.sort(key=lambda x: x.get('total_r', 0), reverse=True)
        # Update ranks
        for i, st in enumerate(leaderboard):
            st['rank'] = i + 1
            
        with open('data/leaderboard.json', 'w', encoding='utf-8') as f:
            json.dump(leaderboard, f, indent=4)
        print("Leaderboard updated and saved!")
    else:
        print("No strategies in the Top 30 needed optimization.")

if __name__ == "__main__":
    run()
