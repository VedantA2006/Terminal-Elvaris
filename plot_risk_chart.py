import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from ai_research_agents import execute_strategy
from download_data import load_or_download

INITIAL_CAPITAL = 50000

def get_strategy_data(target_name):
    with open('data/leaderboard.json', 'r') as f:
        data = json.load(f)
    strats = list(data.values()) if isinstance(data, dict) else data
    for s in strats:
        if s['name'] == target_name:
            return s
    return None

def plot_risk():
    print("Loading data...")
    df = load_or_download()
    
    strat = get_strategy_data("Tick and Volume Confluence (Round #117) (Absolute Peak)")
    if not strat:
        print("Strategy not found!")
        return
        
    print("Executing Strat...")
    res = execute_strategy(strat['code'], df)
    trades = res.get('trades', [])
    
    risk_pcts = np.arange(0.10, 1.55, 0.05)
    profits = []
    drawdowns = []
    
    for r in risk_pcts:
        balance = INITIAL_CAPITAL
        high_water = INITIAL_CAPITAL
        max_dd = 0.0
        
        for t in trades:
            trade_r = t['pnl'] / 1000.0
            risk_amount = balance * (r / 100.0)
            balance += trade_r * risk_amount
            
            if balance > high_water:
                high_water = balance
                
            dd = (high_water - balance) / high_water * 100
            if dd > max_dd:
                max_dd = dd
                
        profits.append(balance - INITIAL_CAPITAL)
        drawdowns.append(max_dd)
        
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    ax1.set_xlabel('Risk Per Trade (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Profit ($)', color='#00ffcc', fontsize=12, fontweight='bold')
    ax1.plot(risk_pcts, profits, color='#00ffcc', linewidth=2.5, marker='o')
    ax1.tick_params(axis='y', labelcolor='#00ffcc')
    ax1.grid(True, alpha=0.2)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Max Trailing Drawdown (%)', color='#ff3366', fontsize=12, fontweight='bold')
    ax2.plot(risk_pcts, drawdowns, color='#ff3366', linewidth=2.5, marker='s')
    ax2.tick_params(axis='y', labelcolor='#ff3366')
    
    # Draw the 5% prop firm red line
    ax2.axhline(5.0, color='red', linestyle='--', linewidth=2, label='5% Prop Limit')
    
    # Highlight the 0.65% sweet spot
    sweet_spot_idx = np.abs(risk_pcts - 0.65).argmin()
    ax1.axvline(0.65, color='yellow', linestyle=':', linewidth=2, alpha=0.8)
    ax1.text(0.65, profits[sweet_spot_idx], ' Sweet Spot (0.65%) ', color='yellow', 
             fontsize=11, fontweight='bold', ha='left', va='bottom', bbox=dict(facecolor='black', alpha=0.5))

    ax2.legend(loc='upper left')
    plt.title('Risk vs. Reward Profile (Tick & Volume Confluence - Absolute Peak)', fontsize=15, pad=15)
    
    os.makedirs('.gemini/artifacts', exist_ok=True)
    plt.savefig('risk_chart.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Chart saved.")

if __name__ == '__main__':
    plot_risk()
