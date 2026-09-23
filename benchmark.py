import time
from strategy_executor import execute_strategy
from download_data import load_or_download

instruments = ["XAUUSD", "US100", "SPX500", "EURUSD", "GBPUSD"]

code = """
def generate_signals(df):
    df['bull_signal'] = df['close'] > df['open']
    df['bear_signal'] = df['close'] < df['open']
    df['sl_long'] = df['close'] * 0.99
    df['sl_short'] = df['close'] * 1.01
    df['tp_long'] = df['close'] * 1.02
    df['tp_short'] = df['close'] * 0.98
    return df
"""

output_lines = ["Starting benchmark...\n"]

for inst in instruments:
    try:
        df = load_or_download(symbol=inst, timeframe='5m')
        raw_df = df[df.index >= '2026-01-01'].copy()
        raw_df.attrs['symbol'] = inst
        
        t0 = time.time()
        for _ in range(5):
            res = execute_strategy(code, raw_df)
        t1 = time.time()
        
        avg_time = (t1 - t0) / 5.0
        line = f"[{inst}] Avg Backtest Time: {avg_time:.5f} seconds (Rows: {len(raw_df)})\n"
        output_lines.append(line)
        print(line.strip())
    except Exception as e:
        line = f"[{inst}] Failed: {str(e)}\n"
        output_lines.append(line)
        print(line.strip())

with open("benchmark_result.txt", "w", encoding="utf-8") as f:
    f.writelines(output_lines)
