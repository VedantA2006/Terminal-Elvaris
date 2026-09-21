import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from ai_research_agents import execute_strategy
from download_data import load_or_download
from matplotlib.gridspec import GridSpec

INITIAL_CAPITAL = 50000

def get_strategy_data(target_name):
    with open('data/leaderboard.json', 'r') as f:
        data = json.load(f)
    strats = list(data.values()) if isinstance(data, dict) else data
    for s in strats:
        if s['name'] == target_name:
            return s
    return None

def calculate_curve(trades, risk_pct):
    balance = INITIAL_CAPITAL
    high_water_mark = INITIAL_CAPITAL
    
    dates = []
    balances = []
    drawdowns = []
    
    for t in trades:
        date_str = str(t['entry_time'])
        trade_r = t['pnl'] / 1000.0
        risk_amount = balance * (risk_pct / 100.0)
        
        balance += (trade_r * risk_amount)
        
        if balance > high_water_mark:
            high_water_mark = balance
            
        dd_pct = (high_water_mark - balance) / high_water_mark * 100
        
        dates.append(pd.to_datetime(date_str))
        balances.append(balance)
        drawdowns.append(-dd_pct) # negative for plot
        
    return dates, balances, drawdowns

def plot_curves():
    print("Loading data...")
    df = load_or_download()
    
    strat1 = get_strategy_data("Tick and Volume Confluence (Round #117) (Absolute Peak)")
    if not strat1:
        strat1 = get_strategy_data("Tick and Volume Confluence (Round #117)")
        
    strat2 = get_strategy_data("Multi-Timeframe Macro Trend Confluence (Round #111)")
    
    print("Executing Strat 1...")
    res1 = execute_strategy(strat1['code'], df)
    
    print("Executing Strat 2...")
    res2 = execute_strategy(strat2['code'], df)
    
    dates1, bal1, dd1 = calculate_curve(res1.get('trades', []), 0.65)
    dates2, bal2, dd2 = calculate_curve(res2.get('trades', []), 0.40)
    
    plt.style.use('dark_background')
    
    # --- Plot Strategy 1 ---
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates1, bal1, color='#00ffcc', linewidth=2)
    ax1.set_title(f"Equity Curve - {strat1['name']} (0.65% Risk)", fontsize=14, pad=15)
    ax1.set_ylabel("Account Balance ($)", fontsize=12)
    ax1.grid(True, alpha=0.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(dates1, dd1, 0, color='#ff3366', alpha=0.5)
    ax2.plot(dates1, dd1, color='#ff3366', linewidth=1)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.set_ylim(-6, 0.5)
    ax2.grid(True, alpha=0.2)
    ax2.axhline(-5, color='red', linestyle='--', alpha=0.7, label='5% Prop Limit')
    ax2.legend(loc='lower left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    os.makedirs('.gemini/artifacts', exist_ok=True)
    plt.savefig('strat1_curve.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # --- Plot Strategy 2 ---
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.1)
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates2, bal2, color='#00ccff', linewidth=2)
    ax1.set_title(f"Equity Curve - {strat2['name']} (0.40% Risk)", fontsize=14, pad=15)
    ax1.set_ylabel("Account Balance ($)", fontsize=12)
    ax1.grid(True, alpha=0.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(dates2, dd2, 0, color='#ff9933', alpha=0.5)
    ax2.plot(dates2, dd2, color='#ff9933', linewidth=1)
    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.set_ylim(-6, 0.5)
    ax2.grid(True, alpha=0.2)
    ax2.axhline(-5, color='red', linestyle='--', alpha=0.7, label='5% Prop Limit')
    ax2.legend(loc='lower left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.savefig('strat2_curve.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Plots saved.")

if __name__ == '__main__':
    plot_curves()
