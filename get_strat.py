import json

def run():
    with open('data/leaderboard.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for s in data:
        if s.get('name') == 'Tick and Volume Confluence (Round #117) (Absolute Peak)':
            with open('strat_temp.py', 'w', encoding='utf-8') as out:
                out.write(s['code'])
            print("Successfully extracted Absolute Peak strategy.")
            return

    print("Strategy not found.")

if __name__ == '__main__':
    run()
