import os

def generate():
    with open(r'C:\Users\vedan\.gemini\antigravity-ide\brain\aae4ba5f-b364-4073-8e38-98aae5c70b01\xauusd_consistency_leaderboard.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    out = [
        "# Prop Firm Projections ($50k Funded Account)\n\n",
        "Assuming a strict prop firm rule of **5% Max Trailing Drawdown**, we scale the risk per trade of each strategy so its historical maximum drawdown equals a very safe **2.5%**.\n\n",
        "| Rank | Strategy | Base Max DD | Safe Risk/Trade | Avg Monthly (R) | Monthly Return (%) | Expected Monthly Profit ($50k) |\n",
        "|---|---|---|---|---|---|---|\n"
    ]
    
    for line in lines:
        if line.startswith('|') and 'Rank' not in line and '---' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) > 6:
                try:
                    rank = parts[1]
                    name = parts[2]
                    max_dd = float(parts[4].replace('%', '').strip())
                    avg_m_r = float(parts[6].replace('R', '').strip())
                    
                    if max_dd > 0:
                        risk_per_trade = 2.5 / max_dd
                        risk_per_trade = min(1.0, risk_per_trade)
                        
                        monthly_pct = avg_m_r * risk_per_trade
                        expected_money = (monthly_pct / 100) * 50000
                        
                        out.append(f"| {rank} | {name} | {max_dd}% | {risk_per_trade:.2f}% | {avg_m_r} R | {monthly_pct:.2f}% | **${expected_money:,.2f}** |\n")
                except Exception as e:
                    pass
                    
    with open('prop_firm_temp.md', 'w', encoding='utf-8') as f:
        f.writelines(out)

if __name__ == '__main__':
    generate()
