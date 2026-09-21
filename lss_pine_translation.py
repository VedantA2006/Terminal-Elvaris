# ==============================================================================
# LIQUIDITY SWEEP SIGNALS (LSS) — PINE SCRIPT V6 EXACT PYTHON TRANSLATION
# Author: EVLabs | Converted for Python Strategy Terminal Backtesting
# ==============================================================================

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 1. Strategy Inputs (Exact Pine Script Defaults)
# ------------------------------------------------------------------------------
swing_len   = 10      # Swing Length
max_levels  = 5       # Max Liq Levels
atr_len     = 14      # ATR Length
atr_mult    = 0.5     # SL ATR Buffer
rr_ratio    = 2.0     # Risk-to-Reward Ratio
expire_bars = 30      # Expire setup after N bars
max_fvg     = 20      # Max FVGs stored

def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # Session filter: London (07:00-12:00 Berlin) & NY (13:30-20:00 NY)
    # Dukascopy data is UTC (London: 06:00-11:00 UTC, NY: 17:30-24:00 UTC)
    in_session = np.zeros(n, dtype=bool)
    times = df.index
    for i in range(n):
        mins = times[i].hour * 60 + times[i].minute
        in_lon = (6 * 60 <= mins < 11 * 60)
        in_ny = (17 * 60 + 30 <= mins < 24 * 60)
        in_session[i] = in_lon or in_ny

    # ta.atr(14) — Wilder RMA / SMMA of True Range
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    atr_vals = np.zeros(n)
    atr_vals[0] = tr[0]
    alpha = 1.0 / atr_len
    for i in range(1, n):
        atr_vals[i] = alpha * tr[i] + (1.0 - alpha) * atr_vals[i - 1]

    # Causal Pivots: confirmed strictly after swing_len bars
    swing_highs = [None] * n
    swing_lows = [None] * n
    for i in range(swing_len * 2, n):
        mid = i - swing_len
        if highs[mid] == max(highs[i - 2 * swing_len : i + 1]):
            swing_highs[i] = highs[mid]
        if lows[mid] == min(lows[i - 2 * swing_len : i + 1]):
            swing_lows[i] = lows[mid]

    bsl_lvls, bsl_swept = [], []
    ssl_lvls, ssl_swept = [], []
    b_fvg_top, b_fvg_bot, b_fvg_bar = [], [], []
    s_fvg_top, s_fvg_bot, s_fvg_bar = [], [], []

    look_long, look_short = False, False
    setup_bar_l, setup_bar_s = 0, 0
    setup_sl_l, setup_sl_s = 0.0, 0.0

    bull_signal = np.zeros(n, dtype=bool)
    bear_signal = np.zeros(n, dtype=bool)
    sl_long = np.full(n, np.nan)
    tp1_long = np.full(n, np.nan)
    sl_short = np.full(n, np.nan)
    tp1_short = np.full(n, np.nan)

    for i in range(2, n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        cur_atr = atr_vals[i]
        sess = in_session[i]

        # Liquidity levels
        if swing_highs[i] is not None:
            bsl_lvls.append(swing_highs[i])
            bsl_swept.append(False)
            if len(bsl_lvls) > max_levels:
                bsl_lvls.pop(0)
                bsl_swept.pop(0)

        if swing_lows[i] is not None:
            ssl_lvls.append(swing_lows[i])
            ssl_swept.append(False)
            if len(ssl_lvls) > max_levels:
                ssl_lvls.pop(0)
                ssl_swept.pop(0)

        # FVGs
        if l > highs[i - 2]:
            b_fvg_top.append(l)
            b_fvg_bot.append(highs[i - 2])
            b_fvg_bar.append(i)
            if len(b_fvg_top) > max_fvg:
                b_fvg_top.pop(0)
                b_fvg_bot.pop(0)
                b_fvg_bar.pop(0)

        if h < lows[i - 2]:
            s_fvg_top.append(lows[i - 2])
            s_fvg_bot.append(h)
            s_fvg_bar.append(i)
            if len(s_fvg_top) > max_fvg:
                s_fvg_top.pop(0)
                s_fvg_bot.pop(0)
                s_fvg_bar.pop(0)

        # Sweeps
        ssl_sweep, bsl_sweep = False, False
        sweep_low, sweep_high = 0.0, 0.0

        for s_idx in range(len(ssl_lvls)):
            if not ssl_swept[s_idx]:
                lvl = ssl_lvls[s_idx]
                if l < lvl and c > lvl:
                    ssl_swept[s_idx] = True
                    ssl_sweep = True
                    sweep_low = l

        for b_idx in range(len(bsl_lvls)):
            if not bsl_swept[b_idx]:
                lvl = bsl_lvls[b_idx]
                if h > lvl and c < lvl:
                    bsl_swept[b_idx] = True
                    bsl_sweep = True
                    sweep_high = h

        # Setups
        if ssl_sweep and sess:
            look_long = True
            look_short = False
            setup_bar_l = i
            setup_sl_l = sweep_low - cur_atr * atr_mult

        if bsl_sweep and sess:
            look_short = True
            look_long = False
            setup_bar_s = i
            setup_sl_s = sweep_high + cur_atr * atr_mult

        if not sess:
            look_long = False
            look_short = False

        if look_long and (i - setup_bar_l) > expire_bars:
            look_long = False
        if look_short and (i - setup_bar_s) > expire_bars:
            look_short = False

        # Entry triggers on FVG Mitigation
        if look_long:
            for f_idx in range(len(b_fvg_bar) - 1, -1, -1):
                if b_fvg_bar[f_idx] > setup_bar_l:
                    f_top = b_fvg_top[f_idx]
                    f_bot = b_fvg_bot[f_idx]
                    mid = (f_top + f_bot) / 2.0
                    if l <= f_top and c > f_bot:
                        entry_p = mid
                        sl_p = setup_sl_l
                        sl_dist = entry_p - sl_p
                        if sl_dist > 0:
                            bull_signal[i] = True
                            sl_long[i] = sl_p
                            tp1_long[i] = entry_p + sl_dist * rr_ratio
                            look_long = False
                            break

        if look_short:
            for f_idx in range(len(s_fvg_bar) - 1, -1, -1):
                if s_fvg_bar[f_idx] > setup_bar_s:
                    f_top = s_fvg_top[f_idx]
                    f_bot = s_fvg_bot[f_idx]
                    mid = (f_top + f_bot) / 2.0
                    if h >= f_bot and c < f_top:
                        entry_p = mid
                        sl_p = setup_sl_s
                        sl_dist = sl_p - entry_p
                        if sl_dist > 0:
                            bear_signal[i] = True
                            sl_short[i] = sl_p
                            tp1_short[i] = entry_p - sl_dist * rr_ratio
                            look_short = False
                            break

    df['bull_signal'] = bull_signal
    df['bear_signal'] = bear_signal
    df['sl_long'] = sl_long
    df['tp1_long'] = tp1_long
    df['sl_short'] = sl_short
    df['tp1_short'] = tp1_short
    return df
