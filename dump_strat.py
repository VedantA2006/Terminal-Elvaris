import json

def get_strategy():
    with open('data/leaderboard.json', 'r', encoding='utf-8') as f:
        lb = json.load(f)
        
    for strat in lb:
        if strat.get('name') == 'Tick and Volume Confluence (Round #117)':
            print("FOUND STRATEGY!")
            with open('strategy_117_dump.json', 'w', encoding='utf-8') as out:
                json.dump(strat, out, indent=4)
            break

if __name__ == '__main__':
    get_strategy()
