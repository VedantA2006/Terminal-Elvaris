import json
from pathlib import Path

data_dir = Path('data')
best_strats = []

for file in data_dir.glob('leaderboard*.json'):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            strats = json.load(f)
            
        for s in strats:
            inst = file.stem.replace('leaderboard_', '') if 'leaderboard_' in file.stem else 'XAUUSD'
            total_r = s.get('total_r', 0)
            win_rate = s.get('win_rate', 0)
            max_dd = s.get('max_drawdown_r', 0)
            pf = s.get('profit_factor', 0)
            score = s.get('rank_score', 0)
            trades = s.get('total_trades', 0)
            name = s.get('name', 'Unknown')
            
            if trades > 20 and pf > 1.1:
                best_strats.append({
                    'instrument': inst,
                    'name': name,
                    'score': score,
                    'total_r': total_r,
                    'max_dd_r': max_dd,
                    'win_rate': win_rate,
                    'pf': pf,
                    'trades': trades
                })
    except Exception as e:
        pass

# Sort by lowest max_dd_r, then highest total_r
best_strats.sort(key=lambda x: (x['max_dd_r'], -x['total_r']))

print(f'Found {len(best_strats)} strategies.')
print("\n--- TOP 10 SAFEST (LOWEST DRAWDOWN) STRATEGIES ---")
for s in best_strats[:10]:
    # Calculate recommended risk % to keep drawdown strictly <= 4%
    # If max_dd is 0, default to something safe. 
    safe_dd = max(s['max_dd_r'], 1.0)
    # We want max_dd_r * risk_pct <= 3.5% (leaving 1.5% buffer for 5% trailing limit)
    rec_risk = 3.5 / safe_dd
    # Cap risk at 1.0% since prop firms usually frown on massive single-trade risk for consistency rules
    rec_risk = min(rec_risk, 1.0)
    
    print(f"[{s['instrument']}] {s['name'][:40]}... | R: +{s['total_r']} | Max DD: -{s['max_dd_r']}R | PF: {s['pf']} | Trades: {s['trades']} | REC RISK: {rec_risk:.2f}%")

print("\n--- TOP 10 HIGHEST SCORING STRATEGIES ---")
best_strats.sort(key=lambda x: -x['score'])
for s in best_strats[:10]:
    safe_dd = max(s['max_dd_r'], 1.0)
    rec_risk = 3.5 / safe_dd
    rec_risk = min(rec_risk, 1.0)
    print(f"[{s['instrument']}] {s['name'][:40]}... | R: +{s['total_r']} | Max DD: -{s['max_dd_r']}R | PF: {s['pf']} | Trades: {s['trades']} | REC RISK: {rec_risk:.2f}%")
