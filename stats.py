import json
import numpy as np

def analyze(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        strats = json.load(f)
    for s in strats:
        name = s.get('name', '')
        if '15-Minute' in name or 'Champion' in name or 'NY Midnight' in name:
            monthly_r = s.get('monthly_r', {})
            vals = list(monthly_r.values())
            if vals:
                print(f"[{file_path}] {name}")
                print(f"  Avg Monthly: +{np.mean(vals):.2f}R | Worst: {np.min(vals):.2f}R | Max DD: {s.get('max_drawdown_r')}R")

analyze('data/leaderboard_full.json')
analyze('data/leaderboard_1y.json')
