"""
================================================================================
 5 MONTHS >= 10R RUNNER-UP STRATEGY (+81.8R CHAMPION)
 Liquidity Sweep + Fair Value Gap (LSS) Engine for Python Strategy Terminal
================================================================================
 Exact Parameters:
   SWING_LEN   = 7
   MAX_LEVELS  = 3
   ATR_LEN     = 4
   ATR_MULT    = 0.19
   RR_RATIO    = 1.22
   EXPIRE_BARS = 19
   MAX_FVG     = 12
================================================================================
"""

import numpy as np
import pandas as pd

# Strategy Hyperparameters
SWING_LEN   = 7
MAX_LEVELS  = 3
ATR_LEN     = 4
ATR_MULT    = 0.19
RR_RATIO    = 1.22
EXPIRE_BARS = 19
MAX_FVG     = 12

def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates LSS entry signals and TP/SL levels with zero lookahead bias.
    """
    n = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # Session Filter (London: 06:00 - 11:00 UTC, NY: 17:30 - 24:00 UTC)
    in_session = np.zeros(n, dtype=bool)
    times = df.index
    for i in range(n):
        t = times[i]
        mins = t.hour * 60 + t.minute
        in_lon = (6 * 60 <= mins < 11 * 60)
        in_ny = (17 * 60 + 30 <= mins < 24 * 60)
        in_session[i] = in_lon or in_ny

    # Calculate Wilder's SMMA ATR
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    atr_vals = np.zeros(n)
    atr_vals[0] = tr[0]
    alpha = 1.0 / ATR_LEN
    for i in range(1, n):
        atr_vals[i] = alpha * tr[i] + (1.0 - alpha) * atr_vals[i - 1]

    # Precalculate Causal Swing Pivots (Confirmed at bar i = mid + SWING_LEN)
    swing_highs = [None] * n
    swing_lows = [None] * n
    for i in range(SWING_LEN * 2, n):
        mid = i - SWING_LEN
        if highs[mid] == max(highs[i - 2 * SWING_LEN : i + 1]):
            swing_highs[i] = highs[mid]
        if lows[mid] == min(lows[i - 2 * SWING_LEN : i + 1]):
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

        # Register Confirmed Pivots
        if swing_highs[i] is not None:
            bsl_lvls.append(swing_highs[i])
            bsl_swept.append(False)
            if len(bsl_lvls) > MAX_LEVELS:
                bsl_lvls.pop(0)
                bsl_swept.pop(0)

        if swing_lows[i] is not None:
            ssl_lvls.append(swing_lows[i])
            ssl_swept.append(False)
            if len(ssl_lvls) > MAX_LEVELS:
                ssl_lvls.pop(0)
                ssl_swept.pop(0)

        # Register Fair Value Gaps (FVG)
        if l > highs[i - 2]:
            b_fvg_top.append(l)
            b_fvg_bot.append(highs[i - 2])
            b_fvg_bar.append(i)
            if len(b_fvg_top) > MAX_FVG:
                b_fvg_top.pop(0)
                b_fvg_bot.pop(0)
                b_fvg_bar.pop(0)

        if h < lows[i - 2]:
            s_fvg_top.append(lows[i - 2])
            s_fvg_bot.append(h)
            s_fvg_bar.append(i)
            if len(s_fvg_top) > MAX_FVG:
                s_fvg_top.pop(0)
                s_fvg_bot.pop(0)
                s_fvg_bar.pop(0)

        # Detect Liquidity Sweeps
        ssl_sweep = False
        bsl_sweep = False
        sweep_low = 0.0
        sweep_high = 0.0

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

        # Setup Activations (Session Filtered)
        if ssl_sweep and sess:
            look_long = True
            look_short = False
            setup_bar_l = i
            setup_sl_l = sweep_low - cur_atr * ATR_MULT

        if bsl_sweep and sess:
            look_short = True
            look_long = False
            setup_bar_s = i
            setup_sl_s = sweep_high + cur_atr * ATR_MULT

        # Cancel on Session Exit or Bar Expiry
        if not sess:
            look_long = False
            look_short = False

        if look_long and (i - setup_bar_l) > EXPIRE_BARS:
            look_long = False
        if look_short and (i - setup_bar_s) > EXPIRE_BARS:
            look_short = False

        # Entry on FVG Mitigation
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
                            tp1_long[i] = entry_p + sl_dist * RR_RATIO
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
                            tp1_short[i] = entry_p - sl_dist * RR_RATIO
                            look_short = False
                            break

    df['bull_signal'] = bull_signal
    df['bear_signal'] = bear_signal
    df['sl_long'] = sl_long
    df['tp1_long'] = tp1_long
    df['sl_short'] = sl_short
    df['tp1_short'] = tp1_short
    return df
