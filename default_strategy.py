"""
Default Strategy Template for the Python Strategy Editor.
Implements the Elvaris River Strategy V2.
"""

DEFAULT_STRATEGY_CODE = '''# =============================================================================
# TRADINGVIEW PYTHON STRATEGY — ELVARIS RIVER V2 (XAUUSD 5m)
# =============================================================================
# Available globals: df (['open','high','low','close','volume']), pd, np, math
# Built-in indicator functions: sma(), ema(), smma(), atr(), rsi(), macd(), bollinger_bands()
# Rules enforced: ZERO Lookahead Bias (Strict Causality)
# =============================================================================

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Strategy Parameters
# -----------------------------------------------------------------------------
bb_length = 55
bb_mult = 0.2

smrng1_input = 200
smrng1_sens = 13.0
smrng2_input = 32
sensitivity = 2.7

percent_stop = 1.0     # SL distance divisor
percent_take1 = 0.2    # TP1 %
percent_take2 = 0.4    # TP2 %
percent_take3 = 0.6    # TP3 %
percent_take4 = 0.8    # TP4 %
percent_take5 = 1.0    # TP5 %

# -----------------------------------------------------------------------------
# 2. Smooth Range Filter Functions
# -----------------------------------------------------------------------------
def smoothrng(x, t, m):
    wper = t * 2 - 1
    avrng = ema(x.diff().abs(), t)
    return ema(avrng, wper) * m

def rngfilt(x, r):
    n = len(x)
    filt = np.zeros(n)
    filt[0] = x.iloc[0]
    x_vals = x.values
    r_vals = r.values

    for i in range(1, n):
        prev = filt[i - 1]
        xi = x_vals[i]
        ri = r_vals[i] if not np.isnan(r_vals[i]) else 0.0
        if xi > prev:
            filt[i] = prev if (xi - ri) < prev else (xi - ri)
        else:
            filt[i] = prev if (xi + ri) > prev else (xi + ri)
    return pd.Series(filt, index=x.index)

# -----------------------------------------------------------------------------
# 3. Indicator Computations
# -----------------------------------------------------------------------------
close = df['close']
high = df['high']
low = df['low']
n = len(df)

# Bollinger Bands
bb_basis = sma(close, bb_length)
bb_dev = bb_mult * close.rolling(bb_length).std(ddof=0)
df['bb_upper'] = bb_basis + bb_dev
df['bb_lower'] = bb_basis - bb_dev
df['bb_basis'] = bb_basis

# Smooth Range Filter
smrng1 = smoothrng(close, smrng1_input, smrng1_sens)
smrng2 = smoothrng(close, smrng2_input, sensitivity)
smrng_combined = (smrng1 + smrng2) / 2
filt = rngfilt(close, smrng_combined)
df['range_filter'] = filt

# Up / Down streaks
up_count = np.zeros(n)
dn_count = np.zeros(n)
filt_vals = filt.values

for i in range(1, n):
    if filt_vals[i] > filt_vals[i - 1]:
        up_count[i] = up_count[i - 1] + 1
        dn_count[i] = 0
    elif filt_vals[i] < filt_vals[i - 1]:
        dn_count[i] = dn_count[i - 1] + 1
        up_count[i] = 0
    else:
        up_count[i] = up_count[i - 1]
        dn_count[i] = dn_count[i - 1]

bull_cond = (close.values > filt_vals) & (up_count > 0)
bear_cond = (close.values < filt_vals) & (dn_count > 0)

last_cond = np.zeros(n)
for i in range(n):
    if bull_cond[i]:
        last_cond[i] = 1
    elif bear_cond[i]:
        last_cond[i] = -1
    elif i > 0:
        last_cond[i] = last_cond[i - 1]

# -----------------------------------------------------------------------------
# 4. Signal Generation (Causal: Evaluated on Bar Close)
# -----------------------------------------------------------------------------
bull_signal = np.zeros(n, dtype=bool)
bear_signal = np.zeros(n, dtype=bool)

for i in range(1, n):
    bull_signal[i] = bull_cond[i] and last_cond[i - 1] == -1
    bear_signal[i] = bear_cond[i] and last_cond[i - 1] == 1

df['bull_signal'] = bull_signal
df['bear_signal'] = bear_signal

# -----------------------------------------------------------------------------
# 5. Stop Loss & 5 Take Profit Levels
# -----------------------------------------------------------------------------
df['sl_long'] = close * (1.0 - percent_stop / 700.0)
df['sl_short'] = close * (1.0 + percent_stop / 700.0)

df['tp1_long'] = close * (1.0 + percent_take1 / 100.0)
df['tp2_long'] = close * (1.0 + percent_take2 / 100.0)
df['tp3_long'] = close * (1.0 + percent_take3 / 100.0)
df['tp4_long'] = close * (1.0 + percent_take4 / 100.0)
df['tp5_long'] = close * (1.0 + percent_take5 / 100.0)

df['tp1_short'] = close * (1.0 - percent_take1 / 100.0)
df['tp2_short'] = close * (1.0 - percent_take2 / 100.0)
df['tp3_short'] = close * (1.0 - percent_take3 / 100.0)
df['tp4_short'] = close * (1.0 - percent_take4 / 100.0)
df['tp5_short'] = close * (1.0 - percent_take5 / 100.0)

# Optional SMMA overlays
df['smma33_h'] = smma(high, 33)
df['smma33_l'] = smma(low, 33)
df['smma144_h'] = smma(high, 144)
df['smma144_l'] = smma(low, 144)

print(f"Computed {bull_signal.sum()} BUY signals and {bear_signal.sum()} SELL signals.")
'''
