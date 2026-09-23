import json
import pandas as pd
from strategy_executor import execute_strategy
from autonomous_research_loop import ResearchLoopManager
import sys

sys.stdout.reconfigure(encoding='utf-8')
print('Loading data...')
rlm = ResearchLoopManager()
train, val, test = rlm._ensure_data('XAUUSD')
df = pd.concat([train, val, test])

strats = json.load(open('data/leaderboard.json'))[:20]

for i, s in enumerate(strats):
    res = execute_strategy(s['code'], df, lot_size=100.0, spread=0.20, slippage=0.05)
    
    if res['success']:
        trades = res['trades']
        news_exits = sum(1 for t in trades if t.get('exit_reason') == 'News Embargo')
        weekend_exits = sum(1 for t in trades if t.get('exit_reason') == 'Weekend Close')
        print(f'[{i+1}/20] {s["name"][:30]}... | Total Trades: {len(trades)} | News Exits: {news_exits} | Weekend Exits: {weekend_exits}')
    else:
        print(f'[{i+1}/20] {s["name"][:30]}... | FAILED')
