import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from download_data import load_or_download

import math
import warnings
warnings.filterwarnings('ignore')

print('Loading full dataset...')
df = load_or_download()

with open('data/leaderboard.json', 'r') as f:
    leaderboard = json.load(f)

top_30 = sorted(leaderboard, key=lambda x: x.get('rank', 999))[:30]

print('Extracting signals...')
position_dict = {}

df_1h = df.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
df_4h = df.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()

def _memoize_indicator(name, func, *args, **kwargs):
    return func(*args, **kwargs)

for i, s in enumerate(top_30):
    code_str = s['code']
    name = s['name']
    
    local_env = {'df': df.copy(), 'df_1h': df_1h.copy(), 'df_4h': df_4h.copy(), 'pd': pd, 'np': np, 'math': math, '_memoize_indicator': _memoize_indicator}
    
    try:
        exec(code_str, globals(), local_env)
        df_sig = local_env['df']
        
        pos_series = pd.Series(0, index=df_sig.index)
        if 'Buy_Signal' in df_sig.columns:
            pos_series.loc[df_sig['Buy_Signal']] = 1
        if 'Sell_Signal' in df_sig.columns:
            pos_series.loc[df_sig['Sell_Signal']] = -1
            
        short_name = f"#{s.get('rank', i+1)} {name.split('(')[0].strip()[:15]}"
        position_dict[short_name] = pos_series
    except Exception as e:
        pass

df_pos = pd.DataFrame(position_dict)
df_pos.fillna(0, inplace=True)

print('Calculating correlation matrix...')
corr_matrix = df_pos.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, fmt='.2f', linewidths=0.5)
plt.title('Strategy Signal Correlation Heatmap (Top 30)')
plt.tight_layout()
plt.savefig('C:/Users/vedan/.gemini/antigravity-ide/brain/bc238e03-423d-456f-b72b-7de9a1f06bbd/correlation_heatmap.png', dpi=300)

total_candles = len(df_pos)
buy_signals = (df_pos == 1).sum(axis=1)
sell_signals = (df_pos == -1).sum(axis=1)

md = '# Strategy Signal Correlation & Overlap Analysis\n\n'
md += '![Correlation Heatmap](file:///C:/Users/vedan/.gemini/antigravity-ide/brain/bc238e03-423d-456f-b72b-7de9a1f06bbd/correlation_heatmap.png)\n\n'
md += '## Signal Overlap Deep Dive\n'
md += 'This analyzes how often multiple strategies in the Top 30 agree on a direction at the exact same 5-minute candle.\n\n'

md += '### Buy Signal Confluence (Max 30)\n'
md += '| Concurrent Buy Signals | Frequency (Candles) | % of Total Time |\n'
md += '|------------------------|---------------------|-----------------|\n'
for i in range(2, 31, 2):
    freq = (buy_signals >= i).sum()
    pct = (freq / total_candles) * 100
    md += f'| {i}+ Strategies Buying | {freq} | {pct:.4f}% |\n'
    
md += '\n### Sell Signal Confluence (Max 30)\n'
md += '| Concurrent Sell Signals | Frequency (Candles) | % of Total Time |\n'
md += '|-------------------------|---------------------|-----------------|\n'
for i in range(2, 31, 2):
    freq = (sell_signals >= i).sum()
    pct = (freq / total_candles) * 100
    md += f'| {i}+ Strategies Selling | {freq} | {pct:.4f}% |\n'
    
md += '\n## Highly Correlated Pairs (>0.50)\n'
pairs_found = False
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        val = corr_matrix.iloc[i, j]
        if val > 0.50:
            pairs_found = True
            md += f'- **{corr_matrix.columns[i]}** & **{corr_matrix.columns[j]}** (Correlation: {val:.2f})\n'
            
if not pairs_found:
    md += 'No pairs were found with > 0.50 correlation. This indicates an extremely diversified portfolio ensemble!\n'
    
with open('C:/Users/vedan/.gemini/antigravity-ide/brain/bc238e03-423d-456f-b72b-7de9a1f06bbd/correlation_report.md', 'w') as f:
    f.write(md)

print('Done!')
