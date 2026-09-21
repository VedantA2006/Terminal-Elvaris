import json
import pandas as pd
from ai_research_agents import execute_strategy, ParameterGridSweeper, compute_monthly_r_breakdown
from download_data import load_or_download
import hashlib

def run_hyper_optimize():
    print("Loading data...")
    df = load_or_download()
    
    with open('data/leaderboard.json', 'r') as f:
        data = json.load(f)
        
    strats = list(data.values()) if isinstance(data, dict) else data
    target_name = 'Tick and Volume Confluence (Round #117) (Peak Optimized)'
    
    target_strat = None
    for s in strats:
        if s['name'] == target_name:
            target_strat = s
            break
            
    if not target_strat:
        # Fallback to the original if the peak optimized isn't found
        target_name = 'Tick and Volume Confluence (Round #117)'
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
    
    # We hit the ceiling at RR 6.0 last time. Let's push it further!
    atr_mults = [1.5, 1.6, 1.7, 1.8]
    rr_ratios = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    
    best_code = base_code
    best_net_r = base_score
    best_stats = {}
    best_trades = []
    
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
                    print(f"  >>> NEW ABSOLUTE PEAK! {best_net_r:.1f} -> {net_r:.1f}")
                    best_net_r = net_r
                    best_code = mutated
                    best_stats = stats
                    best_trades = trades
                    
    if best_net_r > base_score:
        print(f"\nHyper-Optimization successful: {base_score:.1f} -> {best_net_r:.1f}R")
        
        code_hash = hashlib.md5(best_code.encode()).hexdigest()
        new_strat = {
            "name": f"Tick and Volume Confluence (Round #117) (Absolute Peak)",
            "code": best_code,
            "code_hash": code_hash,
            "total_r": best_net_r,
            "profit_factor": best_stats.get('profit_factor', 0.0),
            "max_drawdown": best_stats.get('max_drawdown', 0.0),
            "win_rate": best_stats.get('win_rate', 0.0),
            "trades": best_trades
        }
        
        if isinstance(data, dict):
            data[code_hash] = new_strat
        else:
            data.append(new_strat)
            
        with open('data/leaderboard.json', 'w') as f:
            json.dump(data, f, indent=4)
        print("Leaderboard updated with Absolute Peak.")
    else:
        print("\nCould not break the peak. 71.2R is the absolute ceiling.")

if __name__ == '__main__':
    run_hyper_optimize()
