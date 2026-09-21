import json
import os

files = ['data/leaderboard.json', 'data/leaderboard_full.json']

for f in files:
    if not os.path.exists(f):
        continue
        
    with open(f, 'r') as file:
        data = json.load(file)
        
    updated = False
    for item in data:
        inst = item.get('instrument', 'XAUUSD')
        if 'id' in item and not item['id'].startswith(f"{inst}-"):
            item['id'] = f"{inst}-{item['id']}"
            updated = True
            
    if updated:
        with open(f, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Updated {f}")
