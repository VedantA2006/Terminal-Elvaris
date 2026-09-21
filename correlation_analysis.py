import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from strategy_executor import execute_strategy
from download_data import load_or_download
import warnings
warnings.filterwarnings('ignore')

def run_correlation():
    print("Loading dataset...")
    df = load_or_download()
    
    with open('data/leaderboard.json', 'r') as f:
        leaderboard = json.load(f)
        
    top_30 = sorted(leaderboard, key=lambda x: x.get('rank', 999))[:30]
    
    print("Evaluating strategies to extract position vectors...")
    position_dict = {}
    signal_dict = {}
    
    for i, s in enumerate(top_30):
        name = s['name']
        print(f"[{i+1}/30] Processing {name[:30]}...")
        res = execute_strategy(s['code'], df.copy())
        
        # Create empty position and signal series
        pos_series = pd.Series(0, index=df.index)
        sig_series = pd.Series(0, index=df.index)
        
        if res.get('success'):
            trades = res.get('trades', [])
            for t in trades:
                try:
                    entry_dt = pd.to_datetime(t['entry_time'])
                    exit_dt = pd.to_datetime(t['exit_time'])
                    direction = 1 if t.get('direction', '') == 'LONG' else -1
                    
                    # Position vector (holding the trade)
                    mask = (df.index >= entry_dt) & (df.index <= exit_dt)
                    pos_series.loc[mask] = direction
                    
                    # Signal vector (the exact entry candle)
                    # We match the entry_dt which is in UTC.
                    sig_series.loc[entry_dt] = direction
                except Exception as e:
                    pass
                    
        short_name = f"#{s.get('rank', i+1)} {name.split('(')[0].strip()[:15]}"
        position_dict[short_name] = pos_series
        signal_dict[short_name] = sig_series

    df_pos = pd.DataFrame(position_dict)
    df_sig = pd.DataFrame(signal_dict)
    
    # Calculate Correlation Matrix (based on holding positions)
    corr_matrix = df_pos.corr()
    
    # Plot Heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, fmt=".2f", linewidths=0.5)
    plt.title('Strategy Position Correlation Heatmap (Top 30)')
    plt.tight_layout()
    artifact_dir = "C:/Users/vedan/.gemini/antigravity-ide/brain/bc238e03-423d-456f-b72b-7de9a1f06bbd"
    plt.savefig(f'{artifact_dir}/correlation_heatmap.png', dpi=300)
    
    # Calculate overlap stats based on POSITIONS (holding a trade)
    total_candles = len(df_pos)
    pos_longs = (df_pos == 1).sum(axis=1)
    pos_shorts = (df_pos == -1).sum(axis=1)
    
    # Calculate overlap stats based on SIGNALS (exact entry candle)
    sig_buys = (df_sig == 1).sum(axis=1)
    sig_sells = (df_sig == -1).sum(axis=1)
    
    # Generate Report
    md = "# Strategy Correlation & Overlap Analysis\n\n"
    md += "![Correlation Heatmap](file:///C:/Users/vedan/.gemini/antigravity-ide/brain/bc238e03-423d-456f-b72b-7de9a1f06bbd/correlation_heatmap.png)\n\n"
    
    md += "## 1. Trade Duration Overlap (Position Confluence)\n"
    md += "This analyzes how often multiple strategies are *holding a position* in the same direction at the same time.\n\n"
    
    md += "### Concurrent LONG Positions\n"
    md += "| Concurrent Longs | Frequency (Candles) | % of Total Time |\n"
    md += "|------------------|---------------------|-----------------|\n"
    for i in range(2, 31, 2):
        freq = (pos_longs >= i).sum()
        pct = (freq / total_candles) * 100
        md += f"| {i}+ Strategies Long | {freq} | {pct:.4f}% |\n"
        
    md += "\n### Concurrent SHORT Positions\n"
    md += "| Concurrent Shorts | Frequency (Candles) | % of Total Time |\n"
    md += "|-------------------|---------------------|-----------------|\n"
    for i in range(2, 31, 2):
        freq = (pos_shorts >= i).sum()
        pct = (freq / total_candles) * 100
        md += f"| {i}+ Strategies Short | {freq} | {pct:.4f}% |\n"
        
    md += "\n## 2. Instantaneous Signal Overlap\n"
    md += "This analyzes how often multiple strategies trigger an *entry signal* on the exact same 5-minute candle.\n\n"
    
    md += "### Concurrent BUY Signals\n"
    md += "| Concurrent Buys | Frequency (Candles) |\n"
    md += "|-----------------|---------------------|\n"
    for i in range(2, 11):
        freq = (sig_buys >= i).sum()
        md += f"| {i}+ Strategies Buying | {freq} |\n"
        
    md += "\n### Concurrent SELL Signals\n"
    md += "| Concurrent Sells | Frequency (Candles) |\n"
    md += "|------------------|---------------------|\n"
    for i in range(2, 11):
        freq = (sig_sells >= i).sum()
        md += f"| {i}+ Strategies Selling | {freq} |\n"
        
    md += "\n## Highly Correlated Pairs (>0.50)\n"
    pairs_found = False
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if val > 0.50:
                pairs_found = True
                md += f"- **{corr_matrix.columns[i]}** & **{corr_matrix.columns[j]}** (Correlation: {val:.2f})\n"
                
    if not pairs_found:
        md += "No pairs were found with > 0.50 correlation. This indicates an extremely diversified portfolio ensemble!\n"
        
    import os
    with open(os.path.join(artifact_dir, 'correlation_report.md'), 'w') as f:
        f.write(md)
        
    print("Done!")

if __name__ == '__main__':
    run_correlation()
