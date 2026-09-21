"""
Default Strategy Template for the Python Strategy Editor.
Implements a simple Asset-Agnostic Moving Average Crossover.
"""

DEFAULT_STRATEGY_CODE = '''# =============================================================================
# TRADINGVIEW PYTHON STRATEGY — BASIC ASSET AGNOSTIC
# =============================================================================
import numpy as np
import pandas as pd

def sma(series, period):
    return series.rolling(period).mean()

def atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr0 = abs(high - low)
    tr1 = abs(high - close.shift())
    tr2 = abs(low - close.shift())
    tr = pd.concat((tr0, tr1, tr2), axis=1).max(axis=1)
    return tr.rolling(period).mean()

fast_len = 10
slow_len = 50

close = df['close']
df['sma_fast'] = sma(close, fast_len)
df['sma_slow'] = sma(close, slow_len)
df['atr_val'] = atr(df)

# Signals
bull = (df['sma_fast'] > df['sma_slow']) & (df['sma_fast'].shift(1) <= df['sma_slow'].shift(1))
bear = (df['sma_fast'] < df['sma_slow']) & (df['sma_fast'].shift(1) >= df['sma_slow'].shift(1))

df['bull_signal'] = bull
df['bear_signal'] = bear

# Stop Loss / Take Profit using dynamic ATR (asset agnostic)
df['sl_long'] = df['close'] - (df['atr_val'] * 2.0)
df['tp1_long'] = df['close'] + (df['atr_val'] * 3.0)

df['sl_short'] = df['close'] + (df['atr_val'] * 2.0)
df['tp1_short'] = df['close'] - (df['atr_val'] * 3.0)

print(f"Computed {bull.sum()} BUY signals and {bear.sum()} SELL signals.")
'''
