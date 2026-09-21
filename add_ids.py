import json
import uuid

def add_ids_to_leaderboard():
    file_path = 'data/leaderboard.json'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    is_dict = isinstance(data, dict)
    strategies = list(data.values()) if is_dict else data
    
    updated_count = 0
    for strat in strategies:
        if 'id' not in strat:
            strat['id'] = uuid.uuid4().hex[:8]
            updated_count += 1
            
    if is_dict:
        # Re-save as dict if it was a dict
        data = {s['name']: s for s in strategies}
    else:
        data = strategies
        
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"Migration complete! Assigned unique IDs to {updated_count} strategies.")

if __name__ == '__main__':
    add_ids_to_leaderboard()
