import json
import pandas as pd
from ai_research_agents import execute_strategy
from download_data import load_or_download

INITIAL_CAPITAL = 50000
MAX_DAILY_DD_PERCENT = 2.9
MAX_TRAIL_DD_PERCENT = 4.8

def run_prop_test():
    full_df = load_or_download()
    
    # We will test over the full dataset to ensure long-term safety, and also calculate the Jan 2026 profit.
    # Actually, let's just do exactly what they asked: "test this satergy with the 50 k account give details of risk taken"
    # We'll use the entire dataset for risk calculation to be absolutely safe, 
    # but also report the profit from Jan 2026.
    
    df = full_df.copy()
    
    with open('data/leaderboard.json', 'r') as f:
        data = json.load(f)
    strats = list(data.values()) if isinstance(data, dict) else data
    
    target_name = "Tick and Volume Confluence (Round #117) (Absolute Peak)"
    
    target_strat = None
    for s in strats:
        if target_name in s['name']:
            target_strat = s
            break
            
    if not target_strat:
        print("Strategy not found!")
        return
        
    print(f"Simulating '{target_strat['name']}' over full dataset...")
    res = execute_strategy(target_strat['code'], df)
    trades = res.get('trades', [])
    
    if not trades:
        print("No trades found.")
        return
        
    best_risk = 0.0
    max_profit = 0.0
    best_daily_dd = 0.0
    best_trail_dd = 0.0
    jan_2026_profit = 0.0
    
    for risk in range(100, 4, -5):
        risk_pct = risk / 100.0
        balance = INITIAL_CAPITAL
        high_water_mark = INITIAL_CAPITAL
        max_trail = 0.0
        daily_balances = {}
        breached = False
        
        jan_balance = None
        
        for t in trades:
            date_str = str(t['entry_time']).split(' ')[0]
            
            if date_str >= '2026-01-01' and jan_balance is None:
                jan_balance = balance
                
            trade_r = t['pnl'] / 1000.0
            risk_amount = balance * (risk_pct / 100.0)
            trade_profit = trade_r * risk_amount
            
            if date_str not in daily_balances:
                daily_balances[date_str] = balance
                
            balance += trade_profit
            
            if balance > high_water_mark:
                high_water_mark = balance
            
            trail_dd_pct = (high_water_mark - balance) / high_water_mark * 100
            if trail_dd_pct > max_trail:
                max_trail = trail_dd_pct
                
            start_of_day_balance = daily_balances[date_str]
            daily_dd_pct = (start_of_day_balance - balance) / start_of_day_balance * 100
            
            if trail_dd_pct >= MAX_TRAIL_DD_PERCENT or daily_dd_pct >= MAX_DAILY_DD_PERCENT:
                breached = True
                break
                
        if not breached and balance > INITIAL_CAPITAL:
            actual_max_daily = 0
            temp_bals = {}
            temp_bal = INITIAL_CAPITAL
            for t in trades:
                tr_r = t['pnl'] / 1000.0
                tr_profit = tr_r * (temp_bal * (risk_pct / 100.0))
                ds = str(t['entry_time']).split(' ')[0]
                if ds not in temp_bals:
                    temp_bals[ds] = temp_bal
                temp_bal += tr_profit
                d_dd = (temp_bals[ds] - temp_bal) / temp_bals[ds] * 100
                if d_dd > actual_max_daily:
                    actual_max_daily = d_dd

            best_risk = risk_pct
            max_profit = balance - INITIAL_CAPITAL
            best_daily_dd = actual_max_daily
            best_trail_dd = max_trail
            if jan_balance is not None:
                jan_2026_profit = balance - jan_balance
            else:
                jan_2026_profit = balance - INITIAL_CAPITAL
            break 
            
    if best_risk > 0:
        print("\n--- EQUITY EDGE 50K RESULTS ---")
        print(f"Strategy: {target_strat['name']}")
        print(f"Safe Risk/Trade: {best_risk}%")
        print(f"Est. Profit (YTD from Jan 2026): ${jan_2026_profit:,.2f}")
        print(f"Est. Profit (Total 1.5 Years): ${max_profit:,.2f}")
        print(f"Max Daily DD: {best_daily_dd:.2f}% (Limit 3%)")
        print(f"Max Trail DD: {best_trail_dd:.2f}% (Limit 5%)")
        print(f"Total Trades: {len(trades)}")
    else:
        print("Could not find a safe risk level that doesn't breach the limits!")

if __name__ == '__main__':
    run_prop_test()
