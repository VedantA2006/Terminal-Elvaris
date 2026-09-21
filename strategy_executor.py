"""
Dynamic Strategy Executor.

Safely executes user-written or AI-generated Python trading strategies:
1. Passes code through the AST Lookahead Bias Validator (hardcoded guard)
2. Executes in an isolated context with technical indicator helpers
3. Extracts signals, SL/TP levels, and custom indicator series
4. Simulates trades using backtest.py
5. Captures print statements and returns complete TradingView strategy tester data
"""

import io
import sys
import traceback
import time
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from lookahead_guard import assert_no_lookahead, LookaheadBiasError
from security_guard import assert_no_dangerous_code, SecurityViolationError, SAFE_BUILTINS
from backtest import run_backtest
from stats_utils import compute_sharpe_sortino


import functools

# ---------------------------------------------------------------------------
# INDICATOR MEMOIZATION CACHE (Task 4 — grid-sweep speedup)
#
# train_df/val_df/test_df/raw_df are loaded once per process and reused for
# an entire tournament session (see ResearchLoopManager._ensure_data()), and
# execute_strategy() always works on a *copy* of the same underlying data.
# The indicator helpers below are pure functions of (data, params), so their
# results are safe to cache keyed on a cheap content fingerprint rather than
# object identity — this avoids re-running expensive Python-loop indicators
# (find_swings, supertrend) and groupby-based ones (vwap, daily_levels) on
# every one of the ~13 backtests inside a single parameter grid sweep.
# ---------------------------------------------------------------------------
_INDICATOR_CACHE: Dict[Any, Any] = {}
_INDICATOR_CACHE_MAX_ENTRIES = 500


def _fingerprint(obj) -> tuple:
    """Cheap, stable fingerprint for a Series or DataFrame. Stable across a
    fresh .copy() of the same underlying data (execute_strategy always
    passes a copy), so repeated calls on 'the same data' correctly hit
    the cache even though object identity differs each time."""
    if isinstance(obj, pd.DataFrame):
        ref = obj['close'] if 'close' in obj.columns else obj.iloc[:, 0]
    else:
        ref = obj
    n = len(ref)
    if n == 0:
        return ('empty',)
    first, last = ref.iloc[0], ref.iloc[-1]
    return (
        n,
        str(ref.index[0]),
        str(ref.index[-1]),
        round(float(first), 5) if pd.notna(first) else None,
        round(float(last), 5) if pd.notna(last) else None,
    )


def _memoize_indicator(fn):
    """Decorator for pure indicator helpers: fn(data, *args, **kwargs) -> Series | DataFrame | tuple thereof."""
    @functools.wraps(fn)
    def wrapper(data, *args, **kwargs):
        key = (fn.__name__, _fingerprint(data), args, tuple(sorted(kwargs.items())))
        cached = _INDICATOR_CACHE.get(key)
        if cached is not None:
            if isinstance(cached, tuple):
                return tuple(c.copy() if hasattr(c, 'copy') else c for c in cached)
            return cached.copy() if hasattr(cached, 'copy') else cached
        result = fn(data, *args, **kwargs)
        if len(_INDICATOR_CACHE) >= _INDICATOR_CACHE_MAX_ENTRIES:
            _INDICATOR_CACHE.clear()
        _INDICATOR_CACHE[key] = result
        return result
    return wrapper


# Standard Indicator Helpers available to any strategy code
@_memoize_indicator
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

@_memoize_indicator
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

@_memoize_indicator
def smma(series: pd.Series, period: int) -> pd.Series:
    """Smoothed Moving Average (Wilder's MA)"""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()

@_memoize_indicator
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return smma(tr, period)

@_memoize_indicator
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = smma(gain, period)
    avg_loss = smma(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))

@_memoize_indicator
def bollinger_bands(series: pd.Series, period: int = 20, mult: float = 2.0):
    basis = series.rolling(period).mean()
    dev = series.rolling(period).std() * mult
    return basis + dev, basis - dev, basis

@_memoize_indicator
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

@_memoize_indicator
def vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday Volume-Weighted Average Price resetting daily."""
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    vol = df['volume'] if 'volume' in df.columns and (df['volume'] > 0).any() else pd.Series(1.0, index=df.index)
    date_col = df.index.date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('dt', df.get('datetime', df.index))).dt.date
    pv_cum = (tp * vol).groupby(date_col).cumsum()
    v_cum = vol.groupby(date_col).cumsum()
    return pv_cum / v_cum.replace(0, np.nan)

@_memoize_indicator
def adx(df: pd.DataFrame, period: int = 14):
    """Average Directional Index (ADX, +DI, -DI)."""
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    up = high - high.shift(1)
    down = low.shift(1) - low
    p_dm = np.where((up > down) & (up > 0), up, 0.0)
    m_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_smooth = smma(tr, period)
    p_di = 100.0 * smma(pd.Series(p_dm, index=df.index), period) / atr_smooth.replace(0, np.nan)
    m_di = 100.0 * smma(pd.Series(m_dm, index=df.index), period) / atr_smooth.replace(0, np.nan)
    dx = 100.0 * ((p_di - m_di).abs() / (p_di + m_di).replace(0, np.nan))
    return smma(dx, period), p_di, m_di

@_memoize_indicator
def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """Causal Supertrend: returns (supertrend_line, direction_series [+1/-1])."""
    atr_val = atr(df, period)
    hl2 = (df['high'] + df['low']) / 2.0
    ub = hl2 + (mult * atr_val)
    lb = hl2 - (mult * atr_val)
    n = len(df)
    st = np.zeros(n)
    direction = np.ones(n)
    c = df['close'].values
    ub_v, lb_v = ub.values, lb.values
    f_ub, f_lb = ub_v.copy(), lb_v.copy()
    for i in range(1, n):
        f_ub[i] = ub_v[i] if ub_v[i] < f_ub[i - 1] or c[i - 1] > f_ub[i - 1] else f_ub[i - 1]
        f_lb[i] = lb_v[i] if lb_v[i] > f_lb[i - 1] or c[i - 1] < f_lb[i - 1] else f_lb[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if c[i] < f_lb[i] else 1
            st[i] = f_ub[i] if direction[i] == -1 else f_lb[i]
        else:
            direction[i] = 1 if c[i] > f_ub[i] else -1
            st[i] = f_lb[i] if direction[i] == 1 else f_ub[i]
    return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)

@_memoize_indicator
def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling statistical Z-score: (price - mean) / std."""
    m = series.rolling(period).mean()
    s = series.rolling(period).std().replace(0, np.nan)
    return (series - m) / s

@_memoize_indicator
def keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, mult: float = 2.0):
    mid = ema(df['close'], ema_period)
    atr_v = atr(df, atr_period)
    return mid + mult * atr_v, mid - mult * atr_v, mid

@_memoize_indicator
def donchian_channels(df: pd.DataFrame, period: int = 20):
    up = df['high'].rolling(period).max()
    lo = df['low'].rolling(period).min()
    return up, lo, (up + lo) / 2.0

@_memoize_indicator
def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    lo = df['low'].rolling(k_period).min()
    hi = df['high'].rolling(k_period).max()
    k = 100.0 * (df['close'] - lo) / (hi - lo).replace(0, np.nan)
    d = sma(k, d_period)
    return k, d

@_memoize_indicator
def find_swings(df: pd.DataFrame, swing_len: int = 7):
    n = len(df)
    h_vals, l_vals = df['high'].values, df['low'].values
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(swing_len * 2, n):
        mid = i - swing_len
        if h_vals[mid] == max(h_vals[i - 2 * swing_len : i + 1]):
            sh[i] = h_vals[mid]
        if l_vals[mid] == min(l_vals[i - 2 * swing_len : i + 1]):
            sl[i] = l_vals[mid]
    s_h = pd.Series(sh, index=df.index).ffill()
    s_l = pd.Series(sl, index=df.index).ffill()
    return s_h, s_l

@_memoize_indicator
def find_fvgs(df: pd.DataFrame):
    h, l = df['high'], df['low']
    b_mask = l > h.shift(2)
    s_mask = h < l.shift(2)
    b_top = l.where(b_mask, np.nan).ffill()
    b_bot = h.shift(2).where(b_mask, np.nan).ffill()
    s_top = l.shift(2).where(s_mask, np.nan).ffill()
    s_bot = h.where(s_mask, np.nan).ffill()
    return b_top, b_bot, s_top, s_bot

def session_mask(df: pd.DataFrame, session: str = 'london_ny') -> pd.Series:
    """
    Returns a boolean mask for institutional trading sessions (UTC hours).
    - 'london':                 06:00 - 11:00 UTC  (London open through European morning)
    - 'ny':                     12:20 - 17:30 UTC  (COMEX Gold floor / US institutional hours)
    - 'asia':                   00:00 - 06:00 UTC  (Asian session)
    - 'london_fix':             14:30 - 15:30 UTC  (Official LBMA London Gold PM Fix window)
    - 'london_ny' / 'continuous': 06:00 - 18:00 UTC (Continuous full institutional window - NO midday blackout)
    """
    times = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('dt', df.get('datetime', df.index)))
    mins = times.hour * 60 + times.minute
    if session == 'london':
        m = (6 * 60 <= mins) & (mins < 11 * 60)
    elif session == 'ny':
        m = (12 * 60 + 20 <= mins) & (mins < 17 * 60 + 30)
    elif session == 'asia':
        m = (0 <= mins) & (mins < 6 * 60)
    elif session == 'london_fix':
        m = (14 * 60 + 30 <= mins) & (mins < 15 * 60 + 30)
    else:
        # london_ny / continuous / institutional: continuous 06:00 to 18:00 UTC window
        m = (6 * 60 <= mins) & (mins < 18 * 60)
    return pd.Series(m, index=df.index)

@_memoize_indicator
def daily_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns causal Previous Day High (PDH), Low (PDL), Close (PDC),
    and Floor Pivots (PP, R1, S1, R2, S2).
    Strictly shifted by 1 day so bar t only sees yesterday's completed day.
    """
    date_col = df.index.date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('dt', df.get('datetime', df.index))).dt.date
    daily = df.groupby(date_col).agg({'high': 'max', 'low': 'min', 'close': 'last'})
    daily_shifted = daily.shift(1)
    pp = (daily_shifted['high'] + daily_shifted['low'] + daily_shifted['close']) / 3.0
    r1 = (2.0 * pp) - daily_shifted['low']
    s1 = (2.0 * pp) - daily_shifted['high']
    r2 = pp + (daily_shifted['high'] - daily_shifted['low'])
    s2 = pp - (daily_shifted['high'] - daily_shifted['low'])
    levels_df = pd.DataFrame({
        'pdh': daily_shifted['high'],
        'pdl': daily_shifted['low'],
        'pdc': daily_shifted['close'],
        'pp': pp, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2
    })
    date_series = pd.Series(date_col, index=df.index)
    mapped = levels_df.reindex(date_series.values)
    mapped.index = df.index
    return mapped

@_memoize_indicator
def rvol(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Relative Volume (volume / rolling mean volume)."""
    vol = df['volume'] if 'volume' in df.columns and (df['volume'] > 0).any() else pd.Series(1.0, index=df.index)
    avg_vol = vol.rolling(period).mean().replace(0, np.nan)
    return vol / avg_vol

@_memoize_indicator
def linear_regression_slope(series: pd.Series, period: int = 20) -> pd.Series:
    """Vectorized linear regression slope over rolling window."""
    x = np.arange(period)
    x_mean = x.mean()
    weights = (x - x_mean) / ((x - x_mean) ** 2).sum()
    conv = np.convolve(series.values, weights[::-1], mode='valid')
    return pd.Series(np.concatenate([np.full(period - 1, np.nan), conv]), index=series.index)

@_memoize_indicator
def efficiency_ratio(series: pd.Series, period: int = 20) -> pd.Series:
    """Kaufman Efficiency Ratio (1.0 = clean trend, ~0.0 = pure noise/chop)."""
    direction = (series - series.shift(period)).abs()
    volatility = (series - series.shift(1)).abs().rolling(period).sum()
    return direction / volatility.replace(0, np.nan)

@_memoize_indicator
def chandelier_exit(df: pd.DataFrame, period: int = 22, mult: float = 3.0):
    """Chandelier Exit trailing stop line for longs and shorts."""
    atr_val = atr(df, period)
    highest_high = df['high'].rolling(period).max()
    lowest_low = df['low'].rolling(period).min()
    long_stop = highest_high - (atr_val * mult)
    short_stop = lowest_low + (atr_val * mult)
    return long_stop, short_stop

@_memoize_indicator
def htf_ema(series: pd.Series, period: int = 200, timeframe: str = '1h') -> pd.Series:
    """
    Computes strictly causal Higher Timeframe (HTF) Exponential Moving Average.
    Aggregates 5m bars to HTF scale (1h = 12 bars, 4h = 48 bars).
    """
    tf = str(timeframe).lower()
    ratio = 48 if '4h' in tf else (12 if '1h' in tf or '60' in tf else 12)
    return ema(series, period * ratio)


@_memoize_indicator
def htf_trend_filter(df: pd.DataFrame, timeframe: str = '1h') -> pd.Series:
    """
    Computes strictly causal Higher Timeframe (HTF) Trend Regime Filter:
    +1: Bullish macro regime (Close > HTF Fast EMA and HTF Fast EMA > HTF Slow EMA)
    -1: Bearish macro regime (Close < HTF Fast EMA and HTF Fast EMA < HTF Slow EMA)
     0: Neutral / Chop regime (consolidation or transition)
    Timeframe options: '1h' (default, 12 bars on 5m) or '4h' (48 bars on 5m).
    """
    tf = str(timeframe).lower()
    ratio = 48 if '4h' in tf else 12
    fast_period = 20 * ratio
    slow_period = 50 * ratio
    fast = ema(df['close'], fast_period)
    slow = ema(df['close'], slow_period)
    close = df['close']

    bull = (close > fast) & (fast > slow)
    bear = (close < fast) & (fast < slow)
    regime = np.where(bull, 1, np.where(bear, -1, 0))
    return pd.Series(regime, index=df.index)


@_memoize_indicator
def volatility_ratio(df: pd.DataFrame, fast_period: int = 5, slow_period: int = 30) -> pd.Series:
    """Computes dynamic volatility ratio: ATR(fast) / ATR(slow). Value < 0.65 = extreme compression, > 1.10 = explosive expansion."""
    fast_atr = atr(df, fast_period)
    slow_atr = atr(df, slow_period).replace(0, np.nan)
    return fast_atr / slow_atr


@_memoize_indicator
def volume_profile_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns causal Previous Day Volume Profile Value Area:
    POC (Point of Control), VAH (Value Area High - 70%), and VAL (Value Area Low - 70%).
    Strictly shifted by 1 day so today's bars only see yesterday's completed profile levels.
    """
    date_col = df.index.date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('dt', df.get('datetime', df.index))).dt.date
    daily_groups = df.groupby(date_col)

    profiles = {}
    for d, g in daily_groups:
        vol = g['volume'] if 'volume' in g.columns and (g['volume'] > 0).any() else pd.Series(1.0, index=g.index)
        p_min, p_max = g['low'].min(), g['high'].max()
        if p_max - p_min < 0.5:
            p_max = p_min + 1.0
        bins = np.linspace(p_min, p_max, 25)
        typ = (g['high'] + g['low'] + g['close']) / 3.0
        binned = pd.cut(typ, bins=bins, labels=bins[:-1])
        vol_per_bin = vol.groupby(binned, observed=False).sum()

        if len(vol_per_bin) > 0 and vol_per_bin.max() > 0:
            poc_level = float(vol_per_bin.idxmax())
            total_vol = vol_per_bin.sum()
            sorted_bins = vol_per_bin.sort_values(ascending=False)
            cum_vol = sorted_bins.cumsum()
            val_bins = sorted_bins[cum_vol <= total_vol * 0.70].index
            if len(val_bins) > 0:
                vah_level = float(max(val_bins))
                val_level = float(min(val_bins))
            else:
                vah_level = poc_level + 2.0
                val_level = poc_level - 2.0
        else:
            poc_level = (p_min + p_max) / 2.0
            vah_level = poc_level + 2.0
            val_level = poc_level - 2.0

        profiles[d] = {'poc': poc_level, 'vah': vah_level, 'val': val_level}

    prof_df = pd.DataFrame(profiles).T
    prof_df_shifted = prof_df.shift(1)
    df_date = pd.Series(date_col, index=df.index)
    res = df_date.map(prof_df_shifted.to_dict(orient='index'))
    return pd.DataFrame(res.tolist(), index=df.index)


@_memoize_indicator
def htf_swings(df: pd.DataFrame, swing_len: int = 5, timeframe: str = '15min'):
    """
    Causal Multi-Timeframe Swings:
    Resamples 5m bars to Higher Timeframe (e.g. 15min), detects swing highs/lows on
    completed HTF bars, and projects them forward to 5m index causally.
    """
    resampled = df.resample(timeframe).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()

    n = len(resampled)
    h_vals, l_vals = resampled['high'].values, resampled['low'].values
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(swing_len * 2, n):
        mid = i - swing_len
        if h_vals[mid] == max(h_vals[i - 2 * swing_len : i + 1]):
            sh[i] = h_vals[mid]
        if l_vals[mid] == min(l_vals[i - 2 * swing_len : i + 1]):
            sl[i] = l_vals[mid]

    res_h = pd.Series(sh, index=resampled.index).ffill()
    res_l = pd.Series(sl, index=resampled.index).ffill()

    s_h_5m = res_h.reindex(df.index, method='ffill')
    s_l_5m = res_l.reindex(df.index, method='ffill')
    return s_h_5m, s_l_5m


@_memoize_indicator
def session_compression(df: pd.DataFrame, session: str = 'asia', lookback: int = 10) -> pd.Series:
    """
    Computes session volatility compression ratio:
    Today's session range / 10-day rolling median session range.
    Value < 0.70 indicates extreme institutional accumulation / coiled expansion setup.
    """
    times = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('dt', df.get('datetime', df.index)))
    mins = times.hour * 60 + times.minute
    if session == 'asia':
        s_mask = (0 <= mins) & (mins < 6 * 60)
    elif session == 'london':
        s_mask = (6 * 60 <= mins) & (mins < 11 * 60)
    else:
        s_mask = (12 * 60 <= mins) & (mins < 17 * 60)

    s_df = df[s_mask]
    if len(s_df) == 0:
        return pd.Series(1.0, index=df.index)

    date_col = s_df.index.date
    s_range = s_df.groupby(date_col).apply(lambda g: g['high'].max() - g['low'].min())
    rolling_med = s_range.rolling(lookback, min_periods=3).median()
    ratio_by_date = (s_range / rolling_med).shift(1)  # Causal: shifted by 1 day or completed session

    full_date = pd.Series(times.date, index=df.index)
    return full_date.map(ratio_by_date).fillna(1.0)


@_memoize_indicator
def absorption_volume(df: pd.DataFrame, rvol_len: int = 20, spread_len: int = 20) -> pd.Series:
    """
    Institutional Volume Absorption (Stopping Volume):
    High relative volume (RVOL > 1.3) compressed into narrow price spread (< 0.90x avg spread),
    indicating institutional iceberg limit order absorption at structural levels.
    """
    vol = df['volume'] if 'volume' in df.columns and (df['volume'] > 0).any() else pd.Series(1.0, index=df.index)
    rvol_val = vol / vol.rolling(rvol_len, min_periods=5).mean().replace(0, np.nan)
    spread = df['high'] - df['low']
    avg_spread = spread.rolling(spread_len, min_periods=5).mean().replace(0, np.nan)
    spread_ratio = spread / avg_spread
    is_abs = (rvol_val > 1.3) & (spread_ratio < 0.90)
    return is_abs.fillna(False)



@_memoize_indicator
def htf_order_blocks(df: pd.DataFrame, timeframe: str = '1h'):
    """
    Detects institutional order blocks on Higher Timeframe (e.g. 1h or 4h).
    Returns (bullish_ob_level, bearish_ob_level).
    """
    tf = str(timeframe).lower()
    resampled = df.resample(tf).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    
    bullish = (resampled['close'] > resampled['open']) & \
              (resampled['close'].shift(1) < resampled['open'].shift(1)) & \
              (resampled['close'] > resampled['high'].shift(1))
              
    bearish = (resampled['close'] < resampled['open']) & \
              (resampled['close'].shift(1) > resampled['open'].shift(1)) & \
              (resampled['close'] < resampled['low'].shift(1))
              
    bull_ob = resampled['low'].shift(1).where(bullish, np.nan).ffill()
    bear_ob = resampled['high'].shift(1).where(bearish, np.nan).ffill()
    
    s_bull_ob = bull_ob.reindex(df.index, method='ffill')
    s_bear_ob = bear_ob.reindex(df.index, method='ffill')
    return s_bull_ob, s_bear_ob

@_memoize_indicator
def macro_dxy(df: pd.DataFrame) -> pd.Series:
    """Synthetic causal Dollar Index (DXY) proxy for cross-market divergence."""
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, len(df))
    dxy = 104.0 - (df['close'] - 2000) * 0.01 + noise
    return pd.Series(dxy, index=df.index)

@_memoize_indicator
def macro_us10y(df: pd.DataFrame) -> pd.Series:
    """Synthetic causal US 10-Year Yield proxy for cross-market divergence."""
    np.random.seed(43)
    noise = np.random.normal(0, 0.02, len(df))
    us10y = 4.2 - (df['close'] - 2000) * 0.002 + noise
    return pd.Series(us10y, index=df.index)

@_memoize_indicator
def order_flow_imbalance(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Tick-level order flow imbalance proxy. Positive = Buying pressure, Negative = Selling pressure."""
    spread = df['high'] - df['low']
    spread = spread.replace(0, 0.01)
    body = df['close'] - df['open']
    vol = df['volume'] if 'volume' in df.columns else pd.Series(1.0, index=df.index)
    imbalance = (body / spread) * vol
    return imbalance.rolling(period).mean()

@_memoize_indicator
def realized_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Annualized rolling realized volatility."""
    log_ret = np.log(df['close'] / df['close'].shift(1).replace(0, np.nan))
    return log_ret.rolling(period).std() * np.sqrt(252 * 288) * 100.0


class RuntimeLookaheadTrap:
    """
    Guarantees at runtime that no strategy can sneak past the AST validator
    using variable-passed shifts (e.g. k = -1; df.shift(k)) or indirect calls.
    """
    def __enter__(self):
        self._orig_s_shift = pd.Series.shift
        self._orig_df_shift = pd.DataFrame.shift
        self._orig_s_diff = pd.Series.diff
        self._orig_df_diff = pd.DataFrame.diff
        self._orig_s_pct = pd.Series.pct_change
        self._orig_df_pct = pd.DataFrame.pct_change

        def guarded_s_shift(series_self, periods=1, *args, **kwargs):
            if isinstance(periods, (int, float, np.integer, np.floating)) and periods < 0:
                raise LookaheadBiasError(f"Runtime Lookahead Violation: Prohibited negative shift({periods})!")
            return self._orig_s_shift(series_self, periods, *args, **kwargs)

        def guarded_df_shift(df_self, periods=1, *args, **kwargs):
            if isinstance(periods, (int, float, np.integer, np.floating)) and periods < 0:
                raise LookaheadBiasError(f"Runtime Lookahead Violation: Prohibited negative shift({periods})!")
            return self._orig_df_shift(df_self, periods, *args, **kwargs)

        def guarded_s_diff(series_self, periods=1, *args, **kwargs):
            if isinstance(periods, (int, float, np.integer, np.floating)) and periods < 0:
                raise LookaheadBiasError(f"Runtime Lookahead Violation: Prohibited negative diff({periods})!")
            return self._orig_s_diff(series_self, periods, *args, **kwargs)

        def guarded_df_diff(df_self, periods=1, *args, **kwargs):
            if isinstance(periods, (int, float, np.integer, np.floating)) and periods < 0:
                raise LookaheadBiasError(f"Runtime Lookahead Violation: Prohibited negative diff({periods})!")
            return self._orig_df_diff(df_self, periods, *args, **kwargs)

        def guarded_s_pct(series_self, periods=1, *args, **kwargs):
            if isinstance(periods, (int, float, np.integer, np.floating)) and periods < 0:
                raise LookaheadBiasError(f"Runtime Lookahead Violation: Prohibited negative pct_change({periods})!")
            return self._orig_s_pct(series_self, periods, *args, **kwargs)

        def guarded_df_pct(df_self, periods=1, *args, **kwargs):
            if isinstance(periods, (int, float, np.integer, np.floating)) and periods < 0:
                raise LookaheadBiasError(f"Runtime Lookahead Violation: Prohibited negative pct_change({periods})!")
            return self._orig_df_pct(df_self, periods, *args, **kwargs)

        pd.Series.shift = guarded_s_shift
        pd.DataFrame.shift = guarded_df_shift
        pd.Series.diff = guarded_s_diff
        pd.DataFrame.diff = guarded_df_diff
        pd.Series.pct_change = guarded_s_pct
        pd.DataFrame.pct_change = guarded_df_pct
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pd.Series.shift = self._orig_s_shift
        pd.DataFrame.shift = self._orig_df_shift
        pd.Series.diff = self._orig_s_diff
        pd.DataFrame.diff = self._orig_df_diff
        pd.Series.pct_change = self._orig_s_pct
        pd.DataFrame.pct_change = self._orig_df_pct


def _format_native_sim_trades(trades_df: pd.DataFrame, initial_capital: float = 100000.0, df_copy: pd.DataFrame = None):
    """Formats raw simulation trades into TradingView terminal format with 1R risk sizing ($1,000 / trade)."""
    risk_per_trade_usd = initial_capital * 0.01  # 1% risk = $1,000 per 1R
    formatted_trades = []
    equity_curve = [{'time': 'Start', 'time_ts': 0, 'equity': initial_capital, 'pnl': 0.0}]
    cum_pnl = 0.0
    peak_equity = initial_capital
    max_dd_dollars = 0.0

    res_df = df_copy.copy() if df_copy is not None else pd.DataFrame()
    res_df['bull_signal'] = False
    res_df['bear_signal'] = False
    res_df['sl_long'] = np.nan
    res_df['tp1_long'] = np.nan
    res_df['sl_short'] = np.nan
    res_df['tp1_short'] = np.nan
    for _, t in trades_df.iterrows():
        is_long = (t.get('direction') == 1)
        direction = 'long' if is_long else 'short'
        b_idx = int(t.get('entry_bar', 0))
        # Strictly enforce entry is taken from the candle closing
        if 0 <= b_idx < len(res_df):
            en_p = float(res_df.iloc[b_idx]['close'])
        else:
            en_p = float(t.get('entry_price', 0.0))

        sl_p = float(t.get('sl', 0.0))
        risk_dist = abs(en_p - sl_p)
        if is_long:
            tp_p = en_p + (risk_dist * 1.22)
        else:
            tp_p = en_p - (risk_dist * 1.22)

        size = risk_per_trade_usd / risk_dist if risk_dist > 0 else 100.0
        # Realistic broker friction: $0.20 spread + $0.05 slippage = $0.25 per oz round-turn
        friction_cost_usd = round(size * 0.25, 2)
        r_ret = float(t.get('r_return', 0.0))
        # PnL net of spread and execution friction
        gross_pnl = r_ret * risk_per_trade_usd
        pnl = gross_pnl - friction_cost_usd
        net_r_ret = round(pnl / risk_per_trade_usd, 3)

        cum_pnl += pnl
        current_eq = initial_capital + cum_pnl
        if current_eq > peak_equity:
            peak_equity = current_eq
        dd = peak_equity - current_eq
        if dd > max_dd_dollars:
            max_dd_dollars = dd

        en_time = str(t.get('entry_time', ''))
        ex_time = str(t.get('exit_time', en_time))
        try:
            en_ts = int(pd.to_datetime(en_time).timestamp())
        except Exception:
            en_ts = 0
        try:
            ex_ts = int(pd.to_datetime(ex_time).timestamp())
        except Exception:
            ex_ts = 0

        b_idx = int(t.get('entry_bar', 0))

        if 0 <= b_idx < len(res_df):
            if is_long:
                res_df.iloc[b_idx, res_df.columns.get_loc('bull_signal')] = True
                res_df.iloc[b_idx, res_df.columns.get_loc('sl_long')] = sl_p
                res_df.iloc[b_idx, res_df.columns.get_loc('tp1_long')] = tp_p
            else:
                res_df.iloc[b_idx, res_df.columns.get_loc('bear_signal')] = True
                res_df.iloc[b_idx, res_df.columns.get_loc('sl_short')] = sl_p
                res_df.iloc[b_idx, res_df.columns.get_loc('tp1_short')] = tp_p

        formatted_trades.append({
            'id': int(t.get('id', len(formatted_trades) + 1)),
            'direction': direction,
            'entry_price': round(en_p, 2),
            'entry_time': en_time,
            'entry_time_ts': en_ts,
            'entry_bar_idx': b_idx,
            'sl': round(sl_p, 2),
            'tps': [round(tp_p, 2)],
            'exit_price': round(float(t.get('exit_price', en_p)), 2),
            'exit_time': ex_time,
            'exit_time_ts': ex_ts,
            'exit_reason': str(t.get('exit_reason', 'TP' if r_ret > 0 else 'SL')),
            'initial_size': round(size, 2),
            'remaining_size': 0.0,
            'pnl': round(pnl, 2),
            'cum_pnl': round(cum_pnl, 2),
            'equity': round(current_eq, 2),
            'return_pct': round((pnl / initial_capital) * 100.0, 3),
            'r_return': net_r_ret
        })

        equity_curve.append({
            'time': ex_time,
            'time_ts': ex_ts,
            'equity': round(current_eq, 2),
            'pnl': round(cum_pnl, 2)
        })

    total_trades = len(formatted_trades)
    winning_trades = [tr for tr in formatted_trades if tr['pnl'] > 0]
    losing_trades = [tr for tr in formatted_trades if tr['pnl'] < 0]
    long_trades = [tr for tr in formatted_trades if tr['direction'] == 'long']
    short_trades = [tr for tr in formatted_trades if tr['direction'] == 'short']
    win_rate = (len(winning_trades) / total_trades * 100.0) if total_trades > 0 else 0.0
    tot_win_usd = sum(tr['pnl'] for tr in winning_trades)
    tot_loss_usd = abs(sum(tr['pnl'] for tr in losing_trades))
    pf = (tot_win_usd / tot_loss_usd) if tot_loss_usd > 0 else 99.9

    cur_w = max_w = cur_l = max_l = 0
    pnls = [tr['pnl'] for tr in formatted_trades]
    for p in pnls:
        if p > 0:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)

    avg_win = round(tot_win_usd / len(winning_trades), 2) if winning_trades else 0.0
    avg_loss = round(tot_loss_usd / len(losing_trades), 2) if losing_trades else 0.0
    win_loss_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

    stats = {
        'initial_capital': initial_capital,
        'final_equity': round(initial_capital + cum_pnl, 2),
        'total_pnl': round(cum_pnl, 2),
        'net_profit_pct': round((cum_pnl / initial_capital) * 100.0, 2),
        'gross_profit': round(tot_win_usd, 2),
        'gross_loss': round(tot_loss_usd, 2),
        'total_trades': total_trades,
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': round(win_rate, 1),
        'profit_factor': round(pf, 2),
        'max_drawdown': round(max_dd_dollars, 2),
        'max_drawdown_pct': round((max_dd_dollars / peak_equity) * 100.0, 2) if peak_equity > 0 else 0.0,
        'avg_pnl': round(cum_pnl / total_trades, 2) if total_trades > 0 else 0.0,
        'avg_trade_pnl': round(cum_pnl / total_trades, 2) if total_trades > 0 else 0.0,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'ratio_win_loss': win_loss_ratio,
        'best_trade': round(max(pnls), 2) if pnls else 0.0,
        'worst_trade': round(min(pnls), 2) if pnls else 0.0,
        'max_consecutive_wins': max_w,
        'max_consecutive_losses': max_l,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
    }

    # Compute real Sharpe/Sortino from actual trade data (not hardcoded)
    sharpe, sortino = compute_sharpe_sortino(formatted_trades, equity_curve, initial_capital)
    stats['sharpe_ratio'] = sharpe
    stats['sortino_ratio'] = sortino

    return formatted_trades, equity_curve, stats, res_df


def execute_strategy(code_str: str, raw_df: pd.DataFrame, initial_capital: float = 100000.0,
                     lot_size: float = 100.0, spread: float = 0.20, slippage: float = 0.05) -> Dict[str, Any]:
    """
    Executes a custom Python strategy and returns backtest results.
    """
    # 1. HARDCODED ANTI-LOOKAHEAD BIAS AST CHECK
    try:
        assert_no_lookahead(code_str)
    except LookaheadBiasError as e:
        return {
            'success': False,
            'error_type': 'LOOKAHEAD_BIAS',
            'message': str(e),
            'logs': 'Strategy rejected by Hardcoded Anti-Lookahead Bias Validator.'
        }
    except Exception as e:
        return {
            'success': False,
            'error_type': 'SYNTAX_ERROR',
            'message': f"Code parsing error: {str(e)}",
            'logs': ''
        }

    # 1b. SECURITY CHECK — block imports, eval, exec, open, etc.
    try:
        assert_no_dangerous_code(code_str)
    except SecurityViolationError as e:
        return {
            'success': False,
            'error_type': 'SECURITY_VIOLATION',
            'message': str(e),
            'logs': 'Strategy rejected by Security Guard: dangerous code patterns detected.'
        }

    # 2. Prepare execution context
    df_copy = raw_df.copy()
    if 'dt' not in df_copy.columns:
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy['dt'] = df_copy.index
        elif 'datetime' in df_copy.columns:
            df_copy['dt'] = pd.to_datetime(df_copy['datetime'])
        else:
            df_copy['dt'] = pd.date_range(start='2026-01-01', periods=len(df_copy), freq='5min')

    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture
    
    # Generate True MTF DataFrames
    df_1h = df_copy.resample('1h', on='dt').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).ffill()
    df_1h['dt'] = df_1h.index
    df_4h = df_copy.resample('4h', on='dt').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).ffill()
    df_4h['dt'] = df_4h.index

    env = {
        '__builtins__': SAFE_BUILTINS,
        'pd': pd,
        'np': np,
        'math': math,
        'df': df_copy,
        'df_1h': df_1h,
        'df_4h': df_4h,
        'sma': sma,
        'ema': ema,
        'smma': smma,
        'atr': atr,
        'rsi': rsi,
        'bollinger_bands': bollinger_bands,
        'macd': macd,
        'vwap': vwap,
        'adx': adx,
        'supertrend': supertrend,
        'zscore': zscore,
        'keltner_channels': keltner_channels,
        'donchian_channels': donchian_channels,
        'stochastic': stochastic,
        'find_swings': find_swings,
        'find_fvgs': find_fvgs,
        'session_mask': session_mask,
        'daily_levels': daily_levels,
        'rvol': rvol,
        'linear_regression_slope': linear_regression_slope,
        'efficiency_ratio': efficiency_ratio,
        'chandelier_exit': chandelier_exit,
        'htf_ema': htf_ema,
        'htf_trend_filter': htf_trend_filter,
        'volatility_ratio': volatility_ratio,
        'volume_profile_levels': volume_profile_levels,
        'htf_swings': htf_swings,
        'session_compression': session_compression,
        'absorption_volume': absorption_volume,
        'htf_order_blocks': htf_order_blocks,
        'macro_dxy': macro_dxy,
        'macro_us10y': macro_us10y,
        'order_flow_imbalance': order_flow_imbalance,
        'realized_volatility': realized_volatility,
    }

    try:
        import time
        class ExecutionTimeoutError(Exception):
            pass

        def _timeout_tracer(start_time, timeout):
            def tracer(frame, event, arg):
                if time.time() - start_time > timeout:
                    raise ExecutionTimeoutError(f"Strategy execution timed out after {timeout} seconds")
                return tracer
            return tracer

        with RuntimeLookaheadTrap():
            # Execute strategy code with a strict 30 second timeout
            start_time = time.time()
            old_trace = sys.gettrace()
            sys.settrace(_timeout_tracer(start_time, 30.0))
            try:
                exec(code_str, env)
            finally:
                sys.settrace(old_trace)

            # Check if user defined calculate_signals, strategy, run_simulation, or simulate
            res_df = None
            native_sim_handled = False
            for func_name in ['calculate_signals', 'generate_signals', 'compute_signals', 'get_signals', 'strategy']:
                if func_name in env and callable(env[func_name]):
                    res_df = env[func_name](df_copy)
                    if res_df is None:
                        res_df = env.get('df', df_copy)
                    break

            if res_df is None:
                if 'run_simulation' in env and callable(env['run_simulation']):
                    sim_res = env['run_simulation'](df_copy)
                    if isinstance(sim_res, tuple) and len(sim_res) > 0 and isinstance(sim_res[0], pd.DataFrame):
                        res_df = sim_res[0]
                    elif isinstance(sim_res, pd.DataFrame):
                        res_df = sim_res
                elif 'simulate' in env and callable(env['simulate']):
                    sim_res = env['simulate'](df_copy)
                    if isinstance(sim_res, tuple) and len(sim_res) > 0 and isinstance(sim_res[0], pd.DataFrame):
                        first_item = sim_res[0]
                        if 'bull_signal' in first_item.columns:
                            res_df = first_item
                        elif not first_item.empty and ('result' in first_item.columns or 'r_return' in first_item.columns):
                            trades, equity_curve, stats, res_df = _format_native_sim_trades(first_item, initial_capital, df_copy)
                            native_sim_handled = True
                        else:
                            # first_item is trades_df, map trades into signals
                            res_df = df_copy.copy()
                            res_df['bull_signal'] = False
                            res_df['bear_signal'] = False
                            res_df['sl_long'] = np.nan
                            res_df['tp1_long'] = np.nan
                            res_df['sl_short'] = np.nan
                            res_df['tp1_short'] = np.nan
                            for _, tr in first_item.iterrows():
                                b_idx = tr.get('entry_bar')
                                if b_idx is not None and not pd.isna(b_idx):
                                    b_idx = int(b_idx)
                                    if 0 <= b_idx < len(res_df):
                                        if tr.get('direction') == 1:
                                            res_df.iloc[b_idx, res_df.columns.get_loc('bull_signal')] = True
                                            res_df.iloc[b_idx, res_df.columns.get_loc('sl_long')] = tr.get('sl')
                                            res_df.iloc[b_idx, res_df.columns.get_loc('tp1_long')] = tr.get('tp')
                                        else:
                                            res_df.iloc[b_idx, res_df.columns.get_loc('bear_signal')] = True
                                            res_df.iloc[b_idx, res_df.columns.get_loc('sl_short')] = tr.get('sl')
                                            res_df.iloc[b_idx, res_df.columns.get_loc('tp1_short')] = tr.get('tp')
                else:
                    # Assumed df was mutated in-place
                    res_df = env.get('df', df_copy)

        if res_df is None or not isinstance(res_df, pd.DataFrame):
            raise ValueError("Strategy must return or modify a pandas DataFrame named 'df'.")

        # Normalize signal columns
        if 'bull_signal' not in res_df.columns and 'buy_signal' in res_df.columns:
            res_df['bull_signal'] = res_df['buy_signal']
        if 'bear_signal' not in res_df.columns and 'sell_signal' in res_df.columns:
            res_df['bear_signal'] = res_df['sell_signal']

        if 'bull_signal' not in res_df.columns:
            res_df['bull_signal'] = False
        if 'bear_signal' not in res_df.columns:
            res_df['bear_signal'] = False

        # Fill NaNs in signals
        res_df['bull_signal'] = res_df['bull_signal'].fillna(False).astype(bool)
        res_df['bear_signal'] = res_df['bear_signal'].fillna(False).astype(bool)

        if hasattr(raw_df, 'attrs') and 'eval_start_time' in raw_df.attrs:
            res_df.attrs['eval_start_time'] = raw_df.attrs['eval_start_time']

        if not native_sim_handled:
            # Run bar-by-bar backtest
            trades, equity_curve, stats = run_backtest(
                res_df,
                initial_capital=initial_capital,
                lot_size=lot_size,
                spread=spread,
                slippage=slippage
            )

        # Extract chart markers strictly for trades taken (clean chart)
        markers = []
        for trade in trades:
            try:
                en_ts = int(pd.to_datetime(trade['entry_time']).timestamp())
                is_long = trade['direction'] == 'long'
                markers.append({
                    'time': en_ts,
                    'position': 'belowBar' if is_long else 'aboveBar',
                    'color': '#089981' if is_long else '#f23645',
                    'shape': 'arrowUp' if is_long else 'arrowDown',
                    'text': f"#{trade['id']} {'BUY' if is_long else 'SELL'}",
                })
                if trade.get('exit_time') and trade.get('exit_reason'):
                    exit_ts = int(pd.to_datetime(trade['exit_time']).timestamp())
                    is_win = trade['pnl'] > 0
                    markers.append({
                        'time': exit_ts,
                        'position': 'aboveBar' if is_long else 'belowBar',
                        'color': '#089981' if is_win else '#f23645',
                        'shape': 'circle',
                        'text': f"{trade['exit_reason']} ${trade['pnl']:+,.0f}",
                    })
            except Exception:
                pass

        markers.sort(key=lambda x: x['time'])

        # Clean chart: do not plot indicator lines, only show trades taken
        custom_indicators = {}

        logs = stdout_capture.getvalue()
        return {
            'success': True,
            'stats': stats,
            'trades': trades,
            'equity': equity_curve,
            'markers': markers,
            'indicators': custom_indicators,
            'logs': logs,
        }

    except LookaheadBiasError as e:
        return {
            'success': False,
            'error_type': 'LOOKAHEAD_BIAS',
            'message': str(e),
            'logs': 'Strategy rejected: Runtime lookahead violation detected.'
        }
    except SecurityViolationError as e:
        return {
            'success': False,
            'error_type': 'SECURITY_VIOLATION',
            'message': str(e),
            'logs': 'Strategy rejected: Security violation detected at runtime.'
        }
    except Exception as e:
        err_msg = traceback.format_exc()
        return {
            'success': False,
            'error_type': 'EXECUTION_ERROR',
            'message': str(e),
            'traceback': err_msg,
            'logs': stdout_capture.getvalue()
        }
    finally:
        sys.stdout = old_stdout
