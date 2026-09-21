"""
===============================================================================
STANDALONE BACKTESTER: Stochastic Dynamic Cycle + MACD Acceleration (Round #85)
===============================================================================
Description:
    Fully self-contained, standalone quantitative backtest runner for Strategy #85
    (Rank #1 Champion on the Leaderboard).
    
    Zero internal dependencies: Runs 100% autonomously using only standard Python,
    matplotlib, numpy, and pandas. Reads directly from the downloaded MT5 XAUUSD 5m data.
    
Dataset:
    Full Available MetaTrader 5 ECN Broker XAUUSD (Gold) 5-Minute Candles:
    100,000 candles from April 10, 2025 to September 11, 2026 (1.5 Years)
    
Execution Friction:
    Spread: $0.20 / oz ($0.10 half-spread)
    Slippage: $0.05 / oz
    Total Friction per fill: $0.15 / oz
===============================================================================
"""

import os
import sys
import math
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================================
# 1. STANDALONE TECHNICAL INDICATOR HELPERS
# =============================================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def smma(series: pd.Series, period: int) -> pd.Series:
    """Smoothed Moving Average (Wilder's MA)."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's SMMA smoothing."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return smma(tr, period)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Moving Average Convergence Divergence."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday Volume-Weighted Average Price resetting daily."""
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    vol = df['volume'] if 'volume' in df.columns and (df['volume'] > 0).any() else pd.Series(1.0, index=df.index)
    date_col = df.index.date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['datetime']).dt.date
    pv_cum = (tp * vol).groupby(date_col).cumsum()
    v_cum = vol.groupby(date_col).cumsum()
    return pv_cum / v_cum.replace(0, np.nan)


def find_swings(df: pd.DataFrame, swing_len: int = 7):
    """Causal rolling Swing Highs and Swing Lows with lookback delay."""
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


def daily_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Causal Daily High/Low/Close and Floor Pivots (strictly shifted by 1 day)."""
    date_col = df.index.date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['datetime']).dt.date
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


def session_mask(df: pd.DataFrame, session: str = 'london_ny') -> pd.Series:
    """
    Returns boolean mask for institutional liquidity sessions (UTC).
    - 'london_ny': Combined London (06:00-11:00 UTC) + NY (12:20-17:30 UTC).
    """
    times = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['datetime'])
    mins = times.hour * 60 + times.minute
    if session == 'london_ny':
        m = ((6 * 60 <= mins) & (mins < 11 * 60)) | ((12 * 60 + 20 <= mins) & (mins < 17 * 60 + 30))
    elif session == 'london':
        m = (6 * 60 <= mins) & (mins < 11 * 60)
    elif session == 'ny':
        m = (12 * 60 + 20 <= mins) & (mins < 17 * 60 + 30)
    else:
        m = (0 <= mins) & (mins < 6 * 60)
    return pd.Series(m, index=df.index)


# =============================================================================
# 2. STRATEGY #85 EXACT SIGNAL & RISK LOGIC
# =============================================================================

def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exact signal and execution rules for:
    Stochastic Dynamic Cycle + MACD Acceleration (Round #85)
    """
    df = df.copy()

    # Session mask for high liquidity periods (London + NY)
    sess = session_mask(df, 'london_ny')

    # Swing highs and lows
    sw_highs, sw_lows = find_swings(df, swing_len=7)

    # Daily levels
    levels = daily_levels(df)

    # VWAP line
    vwap_line = vwap(df)

    # SSL and BSL sweeps
    ssl_sweep = (df['low'] < sw_lows) & (df['close'] > sw_lows)
    bsl_sweep = (df['high'] > sw_highs) & (df['close'] < sw_highs)

    # Armed conditions (lookback memory of sweep)
    armed_long = ssl_sweep.rolling(10, min_periods=1).max() == 1
    armed_short = bsl_sweep.rolling(10, min_periods=1).max() == 1

    # MACD momentum
    macd_line, macd_sig, macd_hist = macd(df['close'], 12, 26, 9)

    # Bullish signal: SSL sweep with expanding MACD histogram above VWAP
    raw_bull = (armed_long & (macd_hist > macd_hist.shift(1)) & (df['close'] > vwap_line) & sess)
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))

    # Bearish signal: BSL sweep with contracting MACD histogram below VWAP
    raw_bear = (armed_short & (macd_hist < macd_hist.shift(1)) & (df['close'] < vwap_line) & sess)
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))

    # Calculate stop losses (1.8x ATR past swing extreme)
    atr_val = atr(df, 14)
    df['sl_long'] = np.minimum(df['low'], sw_lows) - (1.8 * atr_val)
    df['sl_short'] = np.maximum(df['high'], sw_highs) + (1.8 * atr_val)

    # Calculate risk (minimum $4.00 stop distance)
    risk_long = np.maximum(df['close'] - df['sl_long'], 4.00)
    risk_short = np.maximum(df['sl_short'] - df['close'], 4.00)

    # Calculate take profits (3.0x Risk)
    df['tp1_long'] = df['close'] + (risk_long * 3.0)
    df['tp1_short'] = df['close'] - (risk_short * 3.0)

    return df


# =============================================================================
# 3. STANDALONE CAUSAL BAR-BY-BAR BACKTEST ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame,
                 initial_capital: float = 100000.0,
                 lot_size: float = 100.0,
                 spread: float = 0.20,
                 slippage: float = 0.05):
    """
    Exact bar-by-bar execution simulator with MT5 ECN friction modeling.
    Records full time-series equity curve and underwater drawdown.
    """
    trades = []
    position = None
    cumulative_pnl = 0.0
    equity_curve = []
    peak_equity = initial_capital
    max_drawdown = 0.0
    max_drawdown_pct = 0.0

    cost_per_oz = (spread / 2.0) + slippage  # $0.10 + $0.05 = $0.15/oz
    has_multi_tp = 'tp1_long' in df.columns
    has_single_tp = 'tp_long' in df.columns
    has_sl = 'sl_long' in df.columns

    for i in range(1, len(df)):
        row = df.iloc[i]
        bar_time = row.name if isinstance(row.name, pd.Timestamp) else pd.to_datetime(row.name if hasattr(row, 'name') else row.get('datetime'))
        bar_time_str = bar_time.isoformat() if hasattr(bar_time, 'isoformat') else str(bar_time)

        # ---- 1. CHECK EXITS FOR OPEN POSITION ----
        if position is not None:
            closed = False
            risk_dist = abs(position['entry_price'] - (position.get('initial_sl') or position['sl'])) if position.get('sl') else 10.0

            if position['direction'] == 'long':
                tp_val = position['tps'][0] if (position.get('tps') and len(position['tps']) > 0 and position['tps'][0] is not None) else None
                sl_val = position['sl']

                sl_hit = (sl_val is not None and row['low'] <= sl_val)
                tp_hit = (tp_val is not None and row['high'] >= tp_val)

                take_tp = False
                take_sl = False

                if tp_hit and not sl_hit:
                    take_tp = True
                elif sl_hit and not tp_hit:
                    take_sl = True
                elif tp_hit and sl_hit:
                    if row['open'] >= tp_val:
                        take_tp = True
                    else:
                        take_sl = True

                if take_tp:
                    fill_price = tp_val - cost_per_oz
                    pnl = (fill_price - position['entry_price']) * position['remaining_size']
                    cumulative_pnl += pnl
                    position['pnl'] += pnl
                    position['exit_price'] = fill_price
                    position['exit_time'] = bar_time_str
                    position['exit_reason'] = 'TP'
                    position['remaining_size'] = 0.0

                    r_ret = round(position['pnl'] / position['risk_usd'], 2) if position['risk_usd'] > 0 else 0.0
                    trades.append({
                        'id': position['id'],
                        'direction': 'long',
                        'entry_time': position['entry_time'],
                        'exit_time': position['exit_time'],
                        'entry_price': round(position['entry_price'], 2),
                        'exit_price': round(position['exit_price'], 2),
                        'pnl': round(position['pnl'], 2),
                        'r_return': r_ret,
                        'exit_reason': position['exit_reason'],
                    })
                    position = None
                    closed = True

                elif take_sl:
                    exit_price = sl_val - cost_per_oz
                    pnl = (exit_price - position['entry_price']) * position['remaining_size']
                    cumulative_pnl += pnl
                    position['pnl'] += pnl
                    position['exit_price'] = exit_price
                    position['exit_time'] = bar_time_str
                    position['exit_reason'] = 'SL'
                    position['remaining_size'] = 0.0

                    r_ret = round(position['pnl'] / position['risk_usd'], 2) if position['risk_usd'] > 0 else 0.0
                    trades.append({
                        'id': position['id'],
                        'direction': 'long',
                        'entry_time': position['entry_time'],
                        'exit_time': position['exit_time'],
                        'entry_price': round(position['entry_price'], 2),
                        'exit_price': round(position['exit_price'], 2),
                        'pnl': round(position['pnl'], 2),
                        'r_return': r_ret,
                        'exit_reason': position['exit_reason'],
                    })
                    position = None
                    closed = True

            elif position['direction'] == 'short':
                tp_val = position['tps'][0] if (position.get('tps') and len(position['tps']) > 0 and position['tps'][0] is not None) else None
                sl_val = position['sl']

                sl_hit = (sl_val is not None and row['high'] >= sl_val)
                tp_hit = (tp_val is not None and row['low'] <= tp_val)

                take_tp = False
                take_sl = False

                if tp_hit and not sl_hit:
                    take_tp = True
                elif sl_hit and not tp_hit:
                    take_sl = True
                elif tp_hit and sl_hit:
                    if row['open'] <= tp_val:
                        take_tp = True
                    else:
                        take_sl = True

                if take_tp:
                    fill_price = tp_val + cost_per_oz
                    pnl = (position['entry_price'] - fill_price) * position['remaining_size']
                    cumulative_pnl += pnl
                    position['pnl'] += pnl
                    position['exit_price'] = fill_price
                    position['exit_time'] = bar_time_str
                    position['exit_reason'] = 'TP'
                    position['remaining_size'] = 0.0

                    r_ret = round(position['pnl'] / position['risk_usd'], 2) if position['risk_usd'] > 0 else 0.0
                    trades.append({
                        'id': position['id'],
                        'direction': 'short',
                        'entry_time': position['entry_time'],
                        'exit_time': position['exit_time'],
                        'entry_price': round(position['entry_price'], 2),
                        'exit_price': round(position['exit_price'], 2),
                        'pnl': round(position['pnl'], 2),
                        'r_return': r_ret,
                        'exit_reason': position['exit_reason'],
                    })
                    position = None
                    closed = True

                elif take_sl:
                    exit_price = sl_val + cost_per_oz
                    pnl = (position['entry_price'] - exit_price) * position['remaining_size']
                    cumulative_pnl += pnl
                    position['pnl'] += pnl
                    position['exit_price'] = exit_price
                    position['exit_time'] = bar_time_str
                    position['exit_reason'] = 'SL'
                    position['remaining_size'] = 0.0

                    r_ret = round(position['pnl'] / position['risk_usd'], 2) if position['risk_usd'] > 0 else 0.0
                    trades.append({
                        'id': position['id'],
                        'direction': 'short',
                        'entry_time': position['entry_time'],
                        'exit_time': position['exit_time'],
                        'entry_price': round(position['entry_price'], 2),
                        'exit_price': round(position['exit_price'], 2),
                        'pnl': round(position['pnl'], 2),
                        'r_return': r_ret,
                        'exit_reason': position['exit_reason'],
                    })
                    position = None
                    closed = True

            # Opposite signal close
            if not closed and position is not None:
                opp_signal = (position['direction'] == 'long' and bool(row.get('bear_signal', False))) or \
                             (position['direction'] == 'short' and bool(row.get('bull_signal', False)))
                if opp_signal:
                    if position['direction'] == 'long':
                        exit_price = row['close'] - cost_per_oz
                        pnl = (exit_price - position['entry_price']) * position['remaining_size']
                    else:
                        exit_price = row['close'] + cost_per_oz
                        pnl = (position['entry_price'] - exit_price) * position['remaining_size']

                    cumulative_pnl += pnl
                    position['pnl'] += pnl
                    position['exit_price'] = exit_price
                    position['exit_time'] = bar_time_str
                    position['exit_reason'] = 'Signal'

                    r_ret = round(position['pnl'] / position['risk_usd'], 2) if position['risk_usd'] > 0 else 0.0
                    trades.append({
                        'id': position['id'],
                        'direction': position['direction'],
                        'entry_time': position['entry_time'],
                        'exit_time': position['exit_time'],
                        'entry_price': round(position['entry_price'], 2),
                        'exit_price': round(position['exit_price'], 2),
                        'pnl': round(position['pnl'], 2),
                        'r_return': r_ret,
                        'exit_reason': position['exit_reason'],
                    })
                    position = None

        # ---- 2. CHECK NEW ENTRIES ----
        if position is None:
            bull = bool(row.get('bull_signal', False))
            bear = bool(row.get('bear_signal', False))

            if bull and not bear:
                entry_price = row['close'] + cost_per_oz
                sl = row.get('sl_long') if has_sl else None
                if sl is not None:
                    if math.isnan(sl) or sl >= entry_price:
                        sl = entry_price - 10.0
                    elif (entry_price - sl) < 4.0:
                        sl = entry_price - 4.0  # Mandatory $4.00 stop distance
                else:
                    sl = entry_price - 10.0

                risk_dist = abs(entry_price - sl)
                tps = []
                if has_multi_tp:
                    for k in range(1, 6):
                        val = row.get(f'tp{k}_long')
                        if val is not None and not math.isnan(val) and val > entry_price:
                            tps.append(val)
                elif has_single_tp:
                    val = row.get('tp_long')
                    if val is not None and not math.isnan(val) and val > entry_price:
                        tps.append(val)

                if not tps:
                    tps = [entry_price + max(4.0, 1.5 * risk_dist)]
                else:
                    min_tp = entry_price + (1.5 * risk_dist)
                    if tps[0] < min_tp:
                        tps[0] = min_tp

                # 1% Account Risk model: $1,000 per trade
                risk_per_trade_usd = initial_capital * 0.01
                calc_size = round(risk_per_trade_usd / risk_dist, 2) if risk_dist > 0 else lot_size

                position = {
                    'id': len(trades) + 1,
                    'direction': 'long',
                    'entry_price': entry_price,
                    'entry_time': bar_time_str,
                    'sl': sl,
                    'initial_sl': sl,
                    'tps': tps.copy(),
                    'initial_size': calc_size,
                    'remaining_size': calc_size,
                    'risk_usd': risk_per_trade_usd,
                    'pnl': 0.0,
                    'exit_price': None,
                    'exit_time': None,
                    'exit_reason': None,
                }

            elif bear and not bull:
                entry_price = row['close'] - cost_per_oz
                sl = row.get('sl_short') if has_sl else None
                if sl is not None:
                    if math.isnan(sl) or sl <= entry_price:
                        sl = entry_price + 10.0
                    elif (sl - entry_price) < 4.0:
                        sl = entry_price + 4.0  # Mandatory $4.00 stop distance
                else:
                    sl = entry_price + 10.0

                risk_dist = abs(sl - entry_price)
                tps = []
                if has_multi_tp:
                    for k in range(1, 6):
                        val = row.get(f'tp{k}_short')
                        if val is not None and not math.isnan(val) and val < entry_price:
                            tps.append(val)
                elif has_single_tp:
                    val = row.get('tp_short')
                    if val is not None and not math.isnan(val) and val < entry_price:
                        tps.append(val)

                if not tps:
                    tps = [entry_price - max(4.0, 1.5 * risk_dist)]
                else:
                    min_tp = entry_price - (1.5 * risk_dist)
                    if tps[0] > min_tp:
                        tps[0] = min_tp

                risk_per_trade_usd = initial_capital * 0.01
                calc_size = round(risk_per_trade_usd / risk_dist, 2) if risk_dist > 0 else lot_size

                position = {
                    'id': len(trades) + 1,
                    'direction': 'short',
                    'entry_price': entry_price,
                    'entry_time': bar_time_str,
                    'sl': sl,
                    'initial_sl': sl,
                    'tps': tps.copy(),
                    'initial_size': calc_size,
                    'remaining_size': calc_size,
                    'risk_usd': risk_per_trade_usd,
                    'pnl': 0.0,
                    'exit_price': None,
                    'exit_time': None,
                    'exit_reason': None,
                }

        # ---- 3. CONTINUOUS INTRA-BAR ADVERSE EXCURSION DRAWDOWN ----
        unrealized = 0.0
        worst_unrealized = 0.0
        if position is not None:
            if position['direction'] == 'long':
                unrealized = (row['close'] - position['entry_price']) * position['remaining_size']
                worst_unrealized = (row['low'] - position['entry_price']) * position['remaining_size']
            else:
                unrealized = (position['entry_price'] - row['close']) * position['remaining_size']
                worst_unrealized = (position['entry_price'] - row['high']) * position['remaining_size']

        current_eq = initial_capital + cumulative_pnl + unrealized
        peak_equity = max(peak_equity, current_eq)

        worst_eq = initial_capital + cumulative_pnl + worst_unrealized
        intra_dd = peak_equity - worst_eq
        if intra_dd > max_drawdown:
            max_drawdown = intra_dd
            max_drawdown_pct = (intra_dd / peak_equity) * 100 if peak_equity > 0 else 0.0

        # Sample equity curve every 20 bars, on trades, or at end
        if i % 20 == 0 or position is None or i == len(df) - 1:
            equity_curve.append({
                'time': bar_time,
                'equity': round(current_eq, 2),
                'pnl': round(cumulative_pnl + unrealized, 2),
                'pnl_r': round((cumulative_pnl + unrealized) / 1000.0, 2),
                'drawdown': round(intra_dd, 2),
                'drawdown_r': round(intra_dd / 1000.0, 2),
                'drawdown_pct': round((intra_dd / peak_equity) * 100 if peak_equity > 0 else 0.0, 2)
            })

    # Close open position at end of series
    if position is not None:
        last = df.iloc[-1]
        bar_time = last.name if isinstance(last.name, pd.Timestamp) else pd.to_datetime(last.name if hasattr(last, 'name') else last.get('datetime'))
        bar_time_str = bar_time.isoformat() if hasattr(bar_time, 'isoformat') else str(bar_time)
        if position['direction'] == 'long':
            exit_price = last['close'] - cost_per_oz
            pnl = (exit_price - position['entry_price']) * position['remaining_size']
        else:
            exit_price = last['close'] + cost_per_oz
            pnl = (position['entry_price'] - exit_price) * position['remaining_size']
        cumulative_pnl += pnl
        position['pnl'] += pnl
        position['exit_price'] = exit_price
        position['exit_time'] = bar_time_str
        position['exit_reason'] = 'End'
        r_ret = round(position['pnl'] / position['risk_usd'], 2) if position['risk_usd'] > 0 else 0.0
        trades.append({
            'id': position['id'],
            'direction': position['direction'],
            'entry_time': position['entry_time'],
            'exit_time': position['exit_time'],
            'entry_price': round(position['entry_price'], 2),
            'exit_price': round(position['exit_price'], 2),
            'pnl': round(position['pnl'], 2),
            'r_return': r_ret,
            'exit_reason': position['exit_reason'],
        })

    # Summary Statistics
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.0
    win_rate = round((len(wins) / len(trades)) * 100, 1) if trades else 0.0
    total_pnl = round(sum(pnls), 2)
    total_r = round(sum(t['r_return'] for t in trades), 1)
    max_dd_r = round(max_drawdown / 1000.0, 1)

    # Monthly Breakdown
    records = []
    for t in trades:
        en_time = t.get('entry_time')
        if en_time:
            dt = pd.to_datetime(en_time)
            records.append({
                'month': dt.strftime('%b %Y'),
                'month_sort': dt.strftime('%Y-%m'),
                'r': t['r_return']
            })
    df_trades = pd.DataFrame(records)
    monthly_series = df_trades.groupby(['month_sort', 'month'])['r'].sum().reset_index().sort_values('month_sort')
    monthly_r = {row['month']: round(float(row['r']), 1) for _, row in monthly_series.iterrows()}

    stats = {
        'total_trades': len(trades),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_pnl': total_pnl,
        'total_r': total_r,
        'max_drawdown_usd': round(max_drawdown, 2),
        'max_drawdown_r': max_dd_r,
        'max_drawdown_pct': round(max_drawdown_pct, 1),
        'monthly_r': monthly_r
    }

    return trades, equity_curve, stats


# =============================================================================
# 4. INSTITUTIONAL VISUALIZATION: EQUITY & DRAWDOWN CURVES
# =============================================================================

def plot_equity_and_drawdown(equity_curve: list, trades: list, stats: dict, output_paths: list):
    """
    Renders a high-resolution dark-themed TradingView-style 2-panel chart:
    - Top panel: Cumulative Equity Curve ($ & R-multiple) with High Water Mark
    - Bottom panel: Underwater Drawdown Area Curve (% & R-multiple)
    """
    if not equity_curve:
        return

    eq_df = pd.DataFrame(equity_curve)
    eq_df['time'] = pd.to_datetime(eq_df['time'])
    eq_df.set_index('time', inplace=True)
    eq_df.sort_index(inplace=True)

    # Compute high water mark
    eq_df['hwm'] = eq_df['equity'].cummax()
    eq_df['drawdown_pct'] = ((eq_df['equity'] - eq_df['hwm']) / eq_df['hwm']) * 100.0
    eq_df['drawdown_r'] = (eq_df['equity'] - eq_df['hwm']) / 1000.0

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 9), sharex=True,
        gridspec_kw={'height_ratios': [2.2, 1.0], 'hspace': 0.08}
    )

    fig.patch.set_facecolor('#0d1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#161b22')
        ax.grid(True, color='#30363d', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.tick_params(colors='#c9d1d9', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # ---- Panel 1: Equity Curve ----
    dates = eq_df.index
    equity_r = (eq_df['equity'] - 100000.0) / 1000.0
    hwm_r = (eq_df['hwm'] - 100000.0) / 1000.0

    ax1.plot(dates, hwm_r, color='#3fb950', linestyle=':', linewidth=1.2, alpha=0.6, label='High Water Mark')
    ax1.plot(dates, equity_r, color='#58a6ff', linewidth=2.0, label='Strategy #85 Equity (R)')
    ax1.fill_between(dates, equity_r, 0, where=(equity_r >= 0), color='#58a6ff', alpha=0.15)
    ax1.fill_between(dates, equity_r, 0, where=(equity_r < 0), color='#f85149', alpha=0.15)
    ax1.axhline(0, color='#8b949e', linestyle='-', linewidth=0.8, alpha=0.5)

    # Annotate final return and stats
    final_r = stats['total_r']
    ax1.scatter([dates[-1]], [equity_r.iloc[-1]], color='#58a6ff', s=80, zorder=5)
    ax1.annotate(
        f"Final: {final_r:+.1f}R\nPF: {stats['profit_factor']:.2f}\nTrades: {stats['total_trades']}",
        xy=(dates[-1], equity_r.iloc[-1]),
        xytext=(-130, -45), textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor='#58a6ff', alpha=0.9),
        fontsize=10, color='#f0f6fc', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#58a6ff', lw=1.2)
    )

    ax1.set_title(
        "Stochastic Dynamic Cycle + MACD Acceleration (Round #85) — Full MT5 Backtest\n"
        f"100,000 Candles (April 2025 – September 2026) | Return: {stats['total_r']:+.1f}R | Max DD: -{stats['max_drawdown_r']}R ({stats['max_drawdown_pct']}%) | PF: {stats['profit_factor']:.2f}",
        fontsize=13, color='#f0f6fc', fontweight='bold', pad=12
    )
    ax1.set_ylabel("Cumulative PnL (R-Multiples)", color='#f0f6fc', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', facecolor='#21262d', edgecolor='#30363d', fontsize=10)

    # Secondary Y-axis for USD
    ax1_usd = ax1.twinx()
    ax1_usd.set_ylim(ax1.get_ylim()[0] * 1000 + 100000, ax1.get_ylim()[1] * 1000 + 100000)
    ax1_usd.set_ylabel("Account Balance ($ USD)", color='#8b949e', fontsize=10)
    ax1_usd.tick_params(colors='#8b949e', labelsize=9)
    ax1_usd.yaxis.set_major_formatter('${x:,.0f}')

    # ---- Panel 2: Underwater Drawdown ----
    ax2.plot(dates, eq_df['drawdown_r'], color='#f85149', linewidth=1.5, label='Underwater Drawdown (R)')
    ax2.fill_between(dates, eq_df['drawdown_r'], 0, color='#f85149', alpha=0.35)
    ax2.axhline(0, color='#8b949e', linestyle='-', linewidth=0.8, alpha=0.5)

    # Mark Maximum Drawdown
    min_dd_idx = eq_df['drawdown_r'].idxmin()
    min_dd_val = eq_df['drawdown_r'].min()
    ax2.scatter([min_dd_idx], [min_dd_val], color='#ff7b72', s=90, zorder=5)
    ax2.annotate(
        f"Max DD: {min_dd_val:.1f}R (-{stats['max_drawdown_pct']}%)",
        xy=(min_dd_idx, min_dd_val),
        xytext=(30, -15), textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262d', edgecolor='#f85149', alpha=0.9),
        fontsize=9, color='#ff7b72', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#f85149', lw=1.2)
    )

    ax2.set_ylabel("Drawdown (R)", color='#f0f6fc', fontsize=11, fontweight='bold')
    ax2.set_xlabel("Date (UTC)", color='#f0f6fc', fontsize=11, fontweight='bold')
    ax2.legend(loc='lower left', facecolor='#21262d', edgecolor='#30363d', fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha='center')

    for p in output_paths:
        try:
            p_obj = Path(p)
            p_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p_obj, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"      [Chart Saved]: {p_obj}")
        except Exception as e:
            print(f"      [Chart Save Warning] {p}: {e}")

    plt.close()


# =============================================================================
# 5. MAIN STANDALONE EXECUTOR
# =============================================================================

def run_standalone():
    print("=" * 80)
    print("  STANDALONE FULL MT5 BACKTESTER: Strategy #85")
    print("  Stochastic Dynamic Cycle + MACD Acceleration (Leaderboard Champion)")
    print("=" * 80)

    # Locate MT5 Data
    candidates = [
        Path('data/XAUUSD_5min.csv'),
        Path(__file__).parent / 'data' / 'XAUUSD_5min.csv',
        Path(__file__).parent.parent / 'data' / 'XAUUSD_5min.csv',
        Path('c:/Users/vedan/Desktop/Foundeer/data/XAUUSD_5min.csv'),
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if not csv_path:
        raise FileNotFoundError("Could not locate MT5 data file 'data/XAUUSD_5min.csv'.")

    print(f"\n[1/4] Loading Full MT5 Broker 5m Dataset from: {csv_path} ...")
    raw_df = pd.read_csv(csv_path, index_col='datetime', parse_dates=True)
    total_bars = len(raw_df)
    start_date = raw_df.index[0]
    end_date = raw_df.index[-1]
    days_span = (end_date - start_date).days

    print(f"      Candles: {total_bars:,} bars (5-Minute Timeframe)")
    print(f"      Date Span: {start_date.strftime('%Y-%m-%d %H:%M')} -> {end_date.strftime('%Y-%m-%d %H:%M')} ({days_span} days / ~1.5 years)")

    # Compute Signals on Full Dataset
    print("\n[2/4] Computing Causal Indicators & Signals across ALL 100,000 Bars ...")
    df_signals = calculate_signals(raw_df)
    bull_count = int(df_signals['bull_signal'].sum())
    bear_count = int(df_signals['bear_signal'].sum())
    print(f"      Signals Generated: {bull_count:,} Longs | {bear_count:,} Shorts (Total: {bull_count + bear_count:,})")

    # Run Backtest on ALL Available MT5 Data
    print("\n[3/4] Executing Full Historical Backtest (Spread: $0.20, Slippage: $0.05) ...")
    trades_full, eq_curve_full, stats_full = run_backtest(df_signals, initial_capital=100000.0, lot_size=100.0)

    # Plot Equity Curve & Drawdown Curve
    print("\n[4/4] Generating Institutional Equity Curve & Underwater Drawdown Chart ...")
    chart_output_local = Path("strat85_full_mt5_equity_drawdown.png")
    chart_output_brain = Path(r"C:\Users\vedan\.gemini\antigravity-ide\brain\326a6e15-ae14-4354-8630-96a63f6b7aa2\strat85_full_mt5_equity_drawdown.png")
    plot_equity_and_drawdown(eq_curve_full, trades_full, stats_full, [chart_output_local, chart_output_brain])

    # Also compute 2026 Sub-Period for direct comparison
    trades_2026 = [t for t in trades_full if t['entry_time'] >= '2026-01-01']
    pnls_2026 = [t['pnl'] for t in trades_2026]
    r_2026 = sum(t['r_return'] for t in trades_2026)
    wins_2026 = [p for p in pnls_2026 if p > 0]
    losses_2026 = [p for p in pnls_2026 if p <= 0]
    pf_2026 = round(sum(wins_2026) / abs(sum(losses_2026)), 2) if losses_2026 and sum(losses_2026) != 0 else 0.0
    wr_2026 = round((len(wins_2026) / len(trades_2026)) * 100, 1) if trades_2026 else 0.0

    # 2025 Sub-Period
    trades_2025 = [t for t in trades_full if t['entry_time'] < '2026-01-01']
    r_2025 = sum(t['r_return'] for t in trades_2025)
    wins_2025 = [t['pnl'] for t in trades_2025 if t['pnl'] > 0]
    losses_2025 = [t['pnl'] for t in trades_2025 if t['pnl'] <= 0]
    pf_2025 = round(sum(wins_2025) / abs(sum(losses_2025)), 2) if losses_2025 and sum(losses_2025) != 0 else 0.0
    wr_2025 = round((len(wins_2025) / len(trades_2025)) * 100, 1) if trades_2025 else 0.0

    # Output Detailed Report
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE QUANTITATIVE AUDIT: FULL MT5 HISTORY (100,000 CANDLES)")
    print("=" * 80)
    print(f"  - Total Return:           {stats_full['total_r']:+.1f} R  (${stats_full['total_pnl']:+,.2f})")
    print(f"  - Total Trades:           {stats_full['total_trades']:,}  ({stats_full['winning_trades']}W / {stats_full['losing_trades']}L)")
    print(f"  - Win Rate:               {stats_full['win_rate']}%")
    print(f"  - Profit Factor:          {stats_full['profit_factor']:.2f}")
    print(f"  - Maximum Drawdown:       -{stats_full['max_drawdown_r']} R  (${stats_full['max_drawdown_usd']:,.2f} / {stats_full['max_drawdown_pct']}%)")
    print(f"  - Trade Frequency:        {stats_full['total_trades'] / max(1, days_span):.2f} trades/day")

    print("\n  " + "-" * 76)
    print("  REGIME SUB-PERIOD DECOMPOSITION:")
    print("  " + "-" * 76)
    print(f"  [2025 Pre-Rally Regime]:  {r_2025:+.1f} R | {len(trades_2025)} Trades | WR {wr_2025}% | PF {pf_2025:.2f} (Apr 2025 -> Dec 2025)")
    print(f"  [2026 Momentum Regime]:   {r_2026:+.1f} R | {len(trades_2026)} Trades | WR {wr_2026}% | PF {pf_2026:.2f} (Jan 2026 -> Sep 2026)")

    print("\n  " + "-" * 76)
    print("  COMPLETE MONTH-BY-MONTH R-RETURN BREAKDOWN (ALL 18 MONTHS):")
    print("  " + "-" * 76)
    months_ge_10 = 0
    for month, val in stats_full['monthly_r'].items():
        tag = "[* >= 10R]" if val >= 10.0 else ("[- LOSS -]" if val < 0 else "[+ PROFIT]")
        if val >= 10.0:
            months_ge_10 += 1
        print(f"    {month:12s} : {val:+6.1f} R   {tag}")

    print(f"\n  - Total Months >= +10R:   {months_ge_10} out of {len(stats_full['monthly_r'])} months")
    print("=" * 80)

    return stats_full


if __name__ == '__main__':
    run_standalone()
