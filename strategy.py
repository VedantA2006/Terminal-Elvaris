"""
Elvaris Strategy Engine - Python translation of the Pine Script indicator
"River Strategy Modified" / "Slaya-Wolf-SuperScalper BY LEO"

Faithfully translates the core signal logic:
  - Smooth Range Filter (smoothrng + rngfilt)
  - Bull/Bear signal triggers
  - Strong Bull/Bear signals
  - SL/TP percentage-based levels
  - Bollinger Bands (55, 0.2)
  - SMMA 33/144 H/L
  - RSI, EMA 144
"""

import numpy as np
import pandas as pd


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ema(series, period):
    """Exponential Moving Average (matches ta.ema in Pine Script)."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series, period):
    """Simple Moving Average (matches ta.sma in Pine Script)."""
    return series.rolling(window=period, min_periods=period).mean()


def stdev(series, period):
    """Rolling standard deviation (matches ta.stdev in Pine Script)."""
    return series.rolling(window=period, min_periods=period).std(ddof=0)


def smma(src, period):
    """
    Smoothed Moving Average (SMMA) - matches Pine Script's manual SMMA.
    smma := na(smma[1]) ? sma : (smma[1] * (len - 1) + src) / len
    """
    n = len(src)
    result = np.full(n, np.nan)
    # First valid SMMA = SMA of first `period` values
    first_valid = src.iloc[:period].mean()
    result[period - 1] = first_valid
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + src.iloc[i]) / period
    return pd.Series(result, index=src.index)


# =============================================================================
# CORE STRATEGY FUNCTIONS (Pine Script translation)
# =============================================================================

def smoothrng(x, t, m):
    """
    Pine Script:
        smoothrng(x, t, m) =>
            wper = t * 2 - 1
            avrng = ta.ema(math.abs(x - x[1]), t)
            smoothrng = ta.ema(avrng, wper) * m
    """
    wper = t * 2 - 1
    avrng = ema(x.diff().abs(), t)
    return ema(avrng, wper) * m


def rngfilt(x, r):
    """
    Pine Script (self-referencing, must be bar-by-bar):
        rngfilt(x, r) =>
            rngfilt = x
            rngfilt := x > nz(rngfilt[1]) ?
                        (x - r < nz(rngfilt[1]) ? nz(rngfilt[1]) : x - r) :
                        (x + r > nz(rngfilt[1]) ? nz(rngfilt[1]) : x + r)
    """
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


# =============================================================================
# MAIN STRATEGY CALCULATION
# =============================================================================

def calculate_signals(df,
                      # Smoothing parameters
                      smrng1_input=200, smrng1_sens=13.0,
                      smrng2_input=32, sensitivity=2.7,
                      # Strong signal sensitivity
                      sensitivity2=13.0,
                      # SL/TP percentages (from Pine Script defaults)
                      percent_stop=1.0,
                      percent_take1=0.2, percent_take2=0.4,
                      percent_take3=0.6, percent_take4=0.8,
                      percent_take5=1.0,
                      # Bollinger Bands
                      bb_length=55, bb_stdev_mult=0.2):
    """
    Calculate all strategy indicators and signals.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume
        Index should be DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Original df with all indicator columns and signal columns added.
    """
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    n = len(df)

    # =================================================================
    # 1. BOLLINGER BANDS (55, 0.2) — Pine Script lines 3-11
    # =================================================================
    bb_basis = sma(close, bb_length)
    bb_dev = bb_stdev_mult * stdev(close, bb_length)
    df['bb_upper'] = bb_basis + bb_dev
    df['bb_lower'] = bb_basis - bb_dev
    df['bb_basis'] = bb_basis

    # =================================================================
    # 2. SMOOTH RANGE FILTER — Pine Script lines 203-269
    # =================================================================
    source = close.copy()

    # smrng1 = smoothrng(source, 200, 13.0)
    smrng1 = smoothrng(source, smrng1_input, smrng1_sens)
    # smrng2 = smoothrng(source, 32, sensitivity=2.7)
    smrng2 = smoothrng(source, smrng2_input, sensitivity)
    # smrng = (smrng1 + smrng2) / 2
    smrng_combined = (smrng1 + smrng2) / 2

    # filt = rngfilt(source, smrng)
    filt = rngfilt(source, smrng_combined)
    df['range_filter'] = filt

    # up/dn counters (bar-by-bar, self-referencing)
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

    # bullCond / bearCond
    # Pine: bullCond := source > filt and source > source[1] and up > 0
    #                 or source > filt and source < source[1] and up > 0
    # Simplifies to: (source > filt) and (up > 0)
    src_vals = source.values
    bull_cond = (src_vals > filt_vals) & (up_count > 0)
    bear_cond = (src_vals < filt_vals) & (dn_count > 0)

    # lastCond (bar-by-bar)
    last_cond = np.zeros(n)
    for i in range(n):
        if bull_cond[i]:
            last_cond[i] = 1
        elif bear_cond[i]:
            last_cond[i] = -1
        elif i > 0:
            last_cond[i] = last_cond[i - 1]

    # bull = bullCond and lastCond[1] == -1
    # bear = bearCond and lastCond[1] == 1
    bull_signal = np.zeros(n, dtype=bool)
    bear_signal = np.zeros(n, dtype=bool)
    for i in range(1, n):
        bull_signal[i] = bull_cond[i] and last_cond[i - 1] == -1
        bear_signal[i] = bear_cond[i] and last_cond[i - 1] == 1

    df['bull_signal'] = bull_signal
    df['bear_signal'] = bear_signal
    df['trend'] = pd.Series(last_cond, index=df.index)

    # =================================================================
    # 3. STRONG SIGNALS — Pine Script lines 270-279
    # =================================================================
    smrng3 = smoothrng(source, smrng2_input, sensitivity2)
    smrng_strong = (smrng1 + smrng3) / 2
    filt2 = rngfilt(source, smrng_strong)

    up2 = np.zeros(n)
    dn2 = np.zeros(n)
    filt2_vals = filt2.values
    for i in range(1, n):
        if filt2_vals[i] > filt2_vals[i - 1]:
            up2[i] = up2[i - 1] + 1
            dn2[i] = 0
        elif filt2_vals[i] < filt2_vals[i - 1]:
            dn2[i] = dn2[i - 1] + 1
            up2[i] = 0
        else:
            up2[i] = up2[i - 1]
            dn2[i] = dn2[i - 1]

    strong_bull_cond = (src_vals > filt2_vals) & (up2 > 0)
    strong_bear_cond = (src_vals < filt2_vals) & (dn2 > 0)

    last_cond2 = np.zeros(n)
    for i in range(n):
        if strong_bull_cond[i]:
            last_cond2[i] = 1
        elif strong_bear_cond[i]:
            last_cond2[i] = -1
        elif i > 0:
            last_cond2[i] = last_cond2[i - 1]

    strong_bull = np.zeros(n, dtype=bool)
    strong_bear = np.zeros(n, dtype=bool)
    for i in range(1, n):
        strong_bull[i] = strong_bull_cond[i] and last_cond2[i - 1] == -1
        strong_bear[i] = strong_bear_cond[i] and last_cond2[i - 1] == 1

    df['strong_bull_signal'] = strong_bull
    df['strong_bear_signal'] = strong_bear

    # =================================================================
    # 4. SL / TP LEVELS — Pine Script lines 342-362
    # =================================================================
    # SL: atrBand = srcStop * (percentStop / 700)
    # TP: atrBandN = srcStop * (percentTakeN / 100)
    # For long: SL below, TPs above
    # For short: SL above, TPs below
    df['sl_long'] = close * (1 - percent_stop / 700)
    df['sl_short'] = close * (1 + percent_stop / 700)
    df['tp1_long'] = close * (1 + percent_take1 / 100)
    df['tp2_long'] = close * (1 + percent_take2 / 100)
    df['tp3_long'] = close * (1 + percent_take3 / 100)
    df['tp4_long'] = close * (1 + percent_take4 / 100)
    df['tp5_long'] = close * (1 + percent_take5 / 100)
    df['tp1_short'] = close * (1 - percent_take1 / 100)
    df['tp2_short'] = close * (1 - percent_take2 / 100)
    df['tp3_short'] = close * (1 - percent_take3 / 100)
    df['tp4_short'] = close * (1 - percent_take4 / 100)
    df['tp5_short'] = close * (1 - percent_take5 / 100)

    # =================================================================
    # 5. SMMA 33/144 HIGH/LOW — Pine Script lines 1606-1651
    # =================================================================
    df['smma_33_high'] = smma(high, 33)
    df['smma_33_low'] = smma(low, 33)
    df['smma_144_high'] = smma(high, 144)
    df['smma_144_low'] = smma(low, 144)

    # =================================================================
    # 6. RSI (28) — Pine Script lines 285-287
    # =================================================================
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 28, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 28, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # =================================================================
    # 7. EMA 144 — Pine Script line 291
    # =================================================================
    df['ema_144'] = ema(close, 144)

    return df
