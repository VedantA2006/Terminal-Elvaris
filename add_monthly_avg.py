import os

def update_table():
    with open('top100_results.md', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    out = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split('|')
        if len(parts) < 8:
            out.append(line)
            continue
        
        if "Return (R)" in line:
            parts.insert(4, " Avg Monthly (R) ")
            parts.insert(6, " Avg Monthly (%) ")
        elif "---" in line:
            parts.insert(4, "---")
            parts.insert(6, "---")
        else:
            try:
                # Avg Monthly R
                total_r = float(parts[3].replace('R', '').strip())
                avg_r = total_r / 17.0
                parts.insert(4, f" {avg_r:.2f} R ")
                
                # Avg Monthly %
                total_pct = float(parts[5].replace('%', '').strip())
                avg_pct = total_pct / 17.0
                parts.insert(6, f" {avg_pct:.2f}% ")
            except Exception as e:
                parts.insert(4, " N/A ")
                parts.insert(6, " N/A ")
        out.append("|".join(parts))
        
    with open('top100_results_avg.md', 'w', encoding='utf-8') as f:
        f.writelines(out)

if __name__ == '__main__':
    update_table()
    print("Done")
