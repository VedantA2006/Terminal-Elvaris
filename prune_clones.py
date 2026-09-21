import json
import pandas as pd
import numpy as np
import math
from download_data import load_or_download
import warnings
warnings.filterwarnings('ignore')

print('Loading full dataset...')
df = load_or_download()

with open('data/leaderboard.json', 'r') as f:
    leaderboard = json.load(f)

# Sort leaderboard by rank so we prioritize keeping higher ranked strategies
leaderboard.sort(key=lambda x: x.get('rank', 999))
top_30 = leaderboard[:30]

print('Extracting signals...')
df_1h = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

def _memoize_indicator(name, func, *args, **kwargs):
    return func(*args, **kwargs)

position_dict = {}
valid_indices = []

for i, s in enumerate(top_30):
    code_str = s['code']
    local_env = {'df': df.copy(), 'df_1h': df_1h.copy(), 'df_4h': df_4h.copy(), 'pd': pd, 'np': np, 'math': math, '_memoize_indicator': _memoize_indicator}
    
    try:
        exec(code_str, globals(), local_env)
        df_sig = local_env['df']
        
        # Strategies output 'bull_signal' and 'bear_signal' instead of Buy_Signal/Sell_Signal
        pos_series = pd.Series(0, index=df_sig.index)
        if 'bull_signal' in df_sig.columns:
            pos_series.loc[df_sig['bull_signal'] == True] = 1
        if 'bear_signal' in df_sig.columns:
            pos_series.loc[df_sig['bear_signal'] == True] = -1
            
        position_dict[i] = pos_series
        valid_indices.append(i)
    except Exception as e:
        print(f"Failed to process strategy {i}: {e}")

df_pos = pd.DataFrame(position_dict)
print('Calculating correlation matrix...')
corr_matrix = df_pos.corr()

to_remove = set()
threshold = 0.85

for i in valid_indices:
    if i in to_remove: continue
    for j in valid_indices:
        if i >= j: continue
        if j in to_remove: continue
        
        # If correlation is above threshold, mark the lower-ranked one for removal
        # Since indices are sorted by rank, j is always the lower ranked (higher index)
        if corr_matrix.loc[i, j] > threshold:
            print(f"Removing #{j+1} due to high correlation ({corr_matrix.loc[i,j]:.2f}) with #{i+1}")
            to_remove.add(j)

# Also remove ALL top 30 strategies that don't take any long trades (enforce 20% rule on existing)
for i in valid_indices:
    if i in to_remove: continue
    pos = position_dict[i]
    longs = (pos == 1).sum()
    shorts = (pos == -1).sum()
    total = longs + shorts
    if total > 0:
        if longs / total < 0.20 or shorts / total < 0.20:
            print(f"Removing #{i+1} due to directional bias (L:{longs} S:{shorts})")
            to_remove.add(i)

# Rebuild leaderboard
new_leaderboard = []
for i, s in enumerate(leaderboard):
    if i < 30 and i in to_remove:
        continue
    new_leaderboard.append(s)
    
# Re-rank
for i, s in enumerate(new_leaderboard):
    s['rank'] = i + 1

with open('data/leaderboard.json', 'w') as f:
    json.dump(new_leaderboard, f, indent=4)
    
print(f"Pruned {len(to_remove)} highly correlated or biased strategies. Leaderboard went from {len(leaderboard)} to {len(new_leaderboard)}.")
