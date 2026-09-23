import json
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

from autonomous_research_loop import ResearchLoopManager
from strategy_executor import execute_strategy

def check_conditions():
    print("Loading data...")
    # Load MT5 data
    train, val, test = ResearchLoopManager()._ensure_data('XAUUSD')
    df = pd.concat([train, val, test])
    # Ensure dt column exists as in executor
    if 'dt' not in df.columns:
        df['dt'] = df.index
    
    # Load News Data
    news_df = pd.read_csv('XAUUSD_News_FOMC_NFP_CPI_PPI_PMI_2Y.csv', encoding='utf-16le')
    # Parse Date and Time into a single datetime
    # Format: Date: 2024.09.18, Time: 19:00
    news_df['datetime_str'] = news_df['Date'] + ' ' + news_df['Time']
    news_df['dt'] = pd.to_datetime(news_df['datetime_str'], format='%Y.%m.%d %H:%M')
    news_df['dt'] = news_df['dt'].dt.tz_localize('UTC')
    
    news_times = news_df['dt'].tolist()
    
    print("Loading strategies...")
    with open('data/leaderboard_full.json', 'r', encoding='utf-8') as f:
        strats = json.load(f)
        
    strats.sort(key=lambda x: -x.get('rank_score', 0))
    top_30 = strats[:30]
    
    results = []
    
    print("Evaluating Top 30 Strategies...")
    for i, s in enumerate(top_30):
        name = s.get('name', 'Unknown')
        code = s.get('code', '')
        
        # Execute strategy to get trades
        res = execute_strategy(code, df, spread=0.20, slippage=0.05, lot_size=100.0)
        trades = res.get('trades', [])
        
        weekend_trades = 0
        news_trades = 0
        total_trades = len(trades)
        
        for t in trades:
            if not t.get('entry_time') or not t.get('exit_time'):
                continue
            
            en = pd.to_datetime(t['entry_time'])
            if en.tzinfo is None: en = en.tz_localize('UTC')
            
            ex = pd.to_datetime(t['exit_time'])
            if ex.tzinfo is None: ex = ex.tz_localize('UTC')
            
            # Weekend Check
            # If the trade spans across a Saturday/Sunday
            # A simple check: if days between entry and exit >= 2, and the span includes weekend days.
            # Or just check if the date range includes a weekday > 4 (0=Mon, 4=Fri, 5=Sat, 6=Sun)
            trade_days = pd.date_range(start=en.floor('D'), end=ex.floor('D'))
            if any(d.weekday() >= 5 for d in trade_days):
                weekend_trades += 1
            
            # News Check
            # If any news time falls within [entry, exit]
            is_news = False
            for nt in news_times:
                if en <= nt <= ex:
                    is_news = True
                    break
            
            if is_news:
                news_trades += 1
                
        results.append({
            'name': name,
            'total': total_trades,
            'weekend': weekend_trades,
            'news': news_trades
        })
        
        print(f"[{i+1}/30] {name[:40]} | Total: {total_trades} | Weekend: {weekend_trades} | News: {news_trades}")

    print("\n--- SUMMARY ---")
    safe_strats = [r for r in results if r['weekend'] == 0 and r['news'] == 0]
    print(f"Strategies completely avoiding News AND Weekends: {len(safe_strats)}/30")
    for r in safe_strats:
        print(f"  -> {r['name'][:60]}")

if __name__ == '__main__':
    check_conditions()
