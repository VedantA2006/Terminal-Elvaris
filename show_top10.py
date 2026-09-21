import json

data = json.load(open('data/leaderboard.json'))
print(f"Total strategies on leaderboard: {len(data)}")
print()
print(f"{'#':>3} | {'Strategy Name':55s} | {'6M R':>7} | {'Trades':>6} | {'WR%':>6} | {'PF':>5} | {'Train R':>8} | {'Val R':>6} | {'Test R':>7}")
print("-" * 130)
for i, s in enumerate(data[:10]):
    name = s.get('name', '?')[:55]
    total_r = s.get('total_r', 0)
    trades = s.get('total_trades', 0)
    wr = s.get('win_rate', 0)
    pf = s.get('profit_factor', 0)
    train_r = s.get('train_r', 0)
    val_r = s.get('val_r', 0)
    test_r = s.get('test_r', 0)
    print(f"{i+1:>3} | {name:55s} | {total_r:+7.1f} | {trades:6d} | {wr:5.1f}% | {pf:5.2f} | {train_r:+7.1f}R | {val_r:+5.1f}R | {test_r:+6.1f}R")
