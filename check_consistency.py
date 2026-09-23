import json
import numpy as np
from pathlib import Path

data_dir = Path('data')
all_strats = []

# Load all strategies
for file in data_dir.glob('leaderboard*.json'):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            strats = json.load(f)
            
        for s in strats:
            inst = file.stem.replace('leaderboard_', '') if 'leaderboard_' in file.stem else 'XAUUSD'
            s['instrument'] = inst
            all_strats.append(s)
    except Exception as e:
        pass

# Sort by rank_score to get the top 20
all_strats.sort(key=lambda x: -x.get('rank_score', 0))
top_20 = all_strats[:20]

print("--- TOP 20 STRATEGIES CONSISTENCY ANALYSIS ---")
for s in top_20:
    name = s.get('name', 'Unknown')
    monthly_r_dict = s.get('monthly_r', {})
    
    if not monthly_r_dict:
        continue
        
    months = list(monthly_r_dict.keys())
    returns = list(monthly_r_dict.values())
    
    # Calculate consistency metrics
    total_months = len(returns)
    winning_months = sum(1 for r in returns if r > 0)
    losing_months = sum(1 for r in returns if r < 0)
    
    mean_return = np.mean(returns) if returns else 0
    std_return = np.std(returns) if returns else 1
    
    # Simple annualized/monthly sharpe-like metric for consistency
    consistency_score = (mean_return / std_return) if std_return > 0 else 0
    
    # Calculate max consecutive losing months
    max_consecutive_losses = 0
    current_losses = 0
    for r in returns:
        if r < 0:
            current_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_losses = 0
            
    print(f"\n[{s['instrument']}] {name[:40]}...")
    print(f"Total R: +{s.get('total_r', 0)} | Score: {s.get('rank_score', 0)} | PF: {s.get('profit_factor', 0)}")
    print(f"Months Traded: {total_months} (Win: {winning_months}, Loss: {losing_months})")
    print(f"Max Consecutive Losing Months: {max_consecutive_losses}")
    print(f"Consistency Ratio (Mean/Std): {consistency_score:.2f}")
    
    # Print the timeline string
    timeline = []
    for m, r in monthly_r_dict.items():
        if r > 5: timeline.append("🟩")
        elif r > 0: timeline.append("🟨")
        else: timeline.append("🟥")
    
    print("Timeline:", "".join(timeline))
