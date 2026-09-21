import json
import pandas as pd
from ai_research_agents import execute_strategy, ParameterGridSweeper, compute_monthly_r_breakdown
from download_data import load_or_download
import hashlib

def run_deep_optimize():
    print("Loading data...")
    df = load_or_download()
    
    with open('data/leaderboard.json', 'r') as f:
        data = json.load(f)
        
    strats = list(data.values()) if isinstance(data, dict) else data
    target_name = 'Tick and Volume Confluence (Round #117)'
    
    target_strat = None
    for s in strats:
        if s['name'] == target_name:
            target_strat = s
            break
            
    if not target_strat:
        print(f"Could not find {target_name}")
        return
        
    base_code = target_strat['code']
    base_score = target_strat.get('total_r', 0.0)
    print(f"Base Total R: {base_score:.1f}")
    
    atr_mults = [1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5]
    rr_ratios = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    
    best_code = base_code
    best_net_r = base_score
    best_stats = {}
    best_trades = []
    
    # Grid Sweep
    total = len(atr_mults) * len(rr_ratios)
    count = 0
    for am in atr_mults:
        for rr in rr_ratios:
            count += 1
            mutated = ParameterGridSweeper._mutate_code_params(base_code, am, rr)
            if mutated == base_code:
                continue
                
            res = execute_strategy(mutated, df)
            if res.get('success'):
                trades = res.get('trades', [])
                stats = res.get('stats', {})
                if len(trades) < 20:
                    continue
                    
                monthly = compute_monthly_r_breakdown(trades)
                net_r = sum(monthly.values()) if monthly else stats.get('total_pnl', 0.0) / 1000.0
                
                print(f"[{count}/{total}] ATR: {am}, RR: {rr} -> Net R: {net_r:.1f}")
                
                if net_r > best_net_r:
                    print(f"  >>> NEW BEST! {best_net_r:.1f} -> {net_r:.1f}")
                    best_net_r = net_r
                    best_code = mutated
                    best_stats = stats
                    best_trades = trades
                    
    if best_net_r > base_score:
        print(f"\nOptimization successful: {base_score:.1f} -> {best_net_r:.1f}R")
        
        code_hash = hashlib.md5(best_code.encode()).hexdigest()
        new_strat = {
            "name": f"{target_name} (Peak Optimized)",
            "code": best_code,
            "code_hash": code_hash,
            "total_r": best_net_r,
            "profit_factor": best_stats.get('profit_factor', 0.0),
            "max_drawdown": best_stats.get('max_drawdown', 0.0),
            "win_rate": best_stats.get('win_rate', 0.0),
            "trades": best_trades
        }
        
        # Append to leaderboard
        if isinstance(data, dict):
            data[code_hash] = new_strat
        else:
            data.append(new_strat)
            
        with open('data/leaderboard.json', 'w') as f:
            json.dump(data, f, indent=4)
        print("Leaderboard updated.")
    else:
        print("\nCould not find a better parameter set.")

if __name__ == '__main__':
    run_deep_optimize()
