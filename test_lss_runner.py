#!/usr/bin/env python3
"""
Test script implementing the exact +81.8R Runner-Up LSS Strategy logic.
"""

import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime
try:
    import pytz
    has_pytz = True
except ImportError:
    has_pytz = False

# Strategy Default Parameters
SWING_LEN   = 7
MAX_LEVELS  = 3
ATR_LEN     = 4
ATR_MULT    = 0.19
RR_RATIO    = 1.22
EXPIRE_BARS = 19
MAX_FVG     = 12

def calculate_atr(df, period=4):
    """Wilder's SMMA ATR"""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    tr0 = high - low
    tr1 = (high - prev_close).abs()
    tr2 = (low - prev_close).abs()
    tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    
    # Wilder's RMA / SMMA: alpha = 1 / period
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr

def is_in_session(dt_val):
    """
    Session filter:
    London: 07:00 - 12:00 Europe/Berlin
    NY:     13:30 - 20:00 America/New_York
    """
    if not isinstance(dt_val, (datetime, pd.Timestamp)):
        try:
            dt_val = pd.to_datetime(dt_val)
        except Exception:
            return True

    # If datetime is timezone-naive, treat as UTC
    if dt_val.tzinfo is None:
        dt_utc = dt_val.tz_localize('UTC')
    else:
        dt_utc = dt_val.tz_convert('UTC')

    if has_pytz:
        try:
            # London Session in Europe/Berlin (UTC+1 / UTC+2 DST)
            tz_berlin = pytz.timezone('Europe/Berlin')
            berlin_dt = dt_utc.astimezone(tz_berlin)
            berlin_mins = berlin_dt.hour * 60 + berlin_dt.minute
            in_london = (7 * 60 <= berlin_mins < 12 * 60)

            # NY Session in America/New_York
            tz_ny = pytz.timezone('America/New_York')
            ny_dt = dt_utc.astimezone(tz_ny)
            ny_mins = ny_dt.hour * 60 + ny_dt.minute
            in_ny = (13 * 60 + 30 <= ny_mins < 20 * 60)

            return in_london or in_ny
        except Exception:
            pass

    # Fallback UTC approximation
    # London approx 06:00 - 11:00 UTC
    # NY approx 18:30 - 01:00 UTC
    h = dt_val.hour
    m = dt_val.minute
    total_m = h * 60 + m
    in_lon = (6 * 60 <= total_m < 11 * 60)
    in_ny = (18 * 60 + 30 <= total_m <= 24 * 60) or (0 <= total_m < 1 * 60)
    return in_lon or in_ny

def run_simulation(df, swing_len=7, max_levels=3, atr_len=4, atr_mult=0.19, rr_ratio=1.22, expire_bars=19, max_fvg=12, filter_sessions=True):
    """
    Executes the exact LSS state-machine simulation on the input dataframe.
    """
    df = df.copy()
    if 'datetime' in df.columns:
        df['dt'] = pd.to_datetime(df['datetime'])
    elif isinstance(df.index, pd.DatetimeIndex):
        df['dt'] = df.index
    else:
        df['dt'] = pd.date_range(start='2026-01-01', periods=len(df), freq='5min')

    atr_series = calculate_atr(df, period=atr_len).values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    times = df['dt'].values

    n = len(df)
    
    # Precompute swing highs and swing lows causality:
    # A pivot high with left=L, right=R is confirmed ONLY at bar i = pivot_idx + R.
    # At bar i, we check if bar (i - R) was the highest in [i - 2R, i].
    swing_highs_confirmed = [None] * n
    swing_lows_confirmed = [None] * n

    for i in range(swing_len * 2, n):
        mid = i - swing_len
        mid_h = highs[mid]
        mid_l = lows[mid]
        
        # Check pivot high: mid_h > all in [i - 2*swing_len, i] (except mid)
        is_pivot_h = True
        for k in range(i - 2 * swing_len, i + 1):
            if k != mid and highs[k] >= mid_h:
                is_pivot_h = False
                break
        if is_pivot_h:
            swing_highs_confirmed[i] = mid_h

        # Check pivot low: mid_l < all in [i - 2*swing_len, i] (except mid)
        is_pivot_l = True
        for k in range(i - 2 * swing_len, i + 1):
            if k != mid and lows[k] <= mid_l:
                is_pivot_l = False
                break
        if is_pivot_l:
            swing_lows_confirmed[i] = mid_l

    # Liquidity queues
    bsl_lvls = []
    bsl_swept = []
    ssl_lvls = []
    ssl_swept = []

    # FVG queues
    b_fvg_top = []
    b_fvg_bot = []
    b_fvg_bar = []

    s_fvg_top = []
    s_fvg_bot = []
    s_fvg_bar = []

    # Setup state
    look_long = False
    look_short = False
    setup_bar_l = 0
    setup_bar_s = 0
    setup_sl_l = 0.0
    setup_sl_s = 0.0

    # Signal/Trade tracking
    trades = []
    active_trade = None # One trade at a time

    # Signal columns for backtester integration
    bull_signals = np.zeros(n, dtype=bool)
    bear_signals = np.zeros(n, dtype=bool)
    sl_long = np.full(n, np.nan)
    tp_long = np.full(n, np.nan)
    sl_short = np.full(n, np.nan)
    tp_short = np.full(n, np.nan)

    for i in range(2, n):
        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]
        current_time = times[i]
        current_atr = atr_series[i] if not np.isnan(atr_series[i]) else 1.0

        in_session = is_in_session(current_time) if filter_sessions else True

        # 1. Update Active Trade Exits
        if active_trade is not None:
            trade_dir = active_trade['direction']
            tp_val = active_trade['tp']
            sl_val = active_trade['sl']
            entry_bar = active_trade['entry_bar']
            age = i - entry_bar
            resolved = False

            if age >= expire_bars * 3:
                # Expired
                active_trade['exit_bar'] = i
                active_trade['exit_time'] = current_time
                active_trade['exit_price'] = c
                active_trade['result'] = 2 # Expired
                active_trade['exit_reason'] = 'Expired'
                active_trade['r_return'] = 0.0
                trades.append(active_trade)
                active_trade = None
                resolved = True
            elif trade_dir == 1: # Long
                if h >= tp_val and l <= sl_val:
                    # Intra-bar priority: check open
                    if o >= tp_val or abs(o - tp_val) <= abs(o - sl_val):
                        win = True
                    else:
                        win = False
                    active_trade['result'] = 1 if win else -1
                    active_trade['exit_price'] = tp_val if win else sl_val
                    active_trade['exit_reason'] = 'TP' if win else 'SL'
                    active_trade['r_return'] = rr_ratio if win else -1.0
                    active_trade['exit_bar'] = i
                    active_trade['exit_time'] = current_time
                    trades.append(active_trade)
                    active_trade = None
                    resolved = True
                elif h >= tp_val:
                    active_trade['result'] = 1
                    active_trade['exit_price'] = tp_val
                    active_trade['exit_reason'] = 'TP'
                    active_trade['r_return'] = rr_ratio
                    active_trade['exit_bar'] = i
                    active_trade['exit_time'] = current_time
                    trades.append(active_trade)
                    active_trade = None
                    resolved = True
                elif l <= sl_val:
                    active_trade['result'] = -1
                    active_trade['exit_price'] = sl_val
                    active_trade['exit_reason'] = 'SL'
                    active_trade['r_return'] = -1.0
                    active_trade['exit_bar'] = i
                    active_trade['exit_time'] = current_time
                    trades.append(active_trade)
                    active_trade = None
                    resolved = True

            elif trade_dir == -1: # Short
                if l <= tp_val and h >= sl_val:
                    if o <= tp_val or abs(o - tp_val) <= abs(o - sl_val):
                        win = True
                    else:
                        win = False
                    active_trade['result'] = 1 if win else -1
                    active_trade['exit_price'] = tp_val if win else sl_val
                    active_trade['exit_reason'] = 'TP' if win else 'SL'
                    active_trade['r_return'] = rr_ratio if win else -1.0
                    active_trade['exit_bar'] = i
                    active_trade['exit_time'] = current_time
                    trades.append(active_trade)
                    active_trade = None
                    resolved = True
                elif l <= tp_val:
                    active_trade['result'] = 1
                    active_trade['exit_price'] = tp_val
                    active_trade['exit_reason'] = 'TP'
                    active_trade['r_return'] = rr_ratio
                    active_trade['exit_bar'] = i
                    active_trade['exit_time'] = current_time
                    trades.append(active_trade)
                    active_trade = None
                    resolved = True
                elif h >= sl_val:
                    active_trade['result'] = -1
                    active_trade['exit_price'] = sl_val
                    active_trade['exit_reason'] = 'SL'
                    active_trade['r_return'] = -1.0
                    active_trade['exit_bar'] = i
                    active_trade['exit_time'] = current_time
                    trades.append(active_trade)
                    active_trade = None
                    resolved = True

        has_open = (active_trade is not None)

        # 2. Register newly confirmed swing highs/lows
        if swing_highs_confirmed[i] is not None:
            bsl_lvls.append(swing_highs_confirmed[i])
            bsl_swept.append(False)
            if len(bsl_lvls) > max_levels:
                bsl_lvls.pop(0)
                bsl_swept.pop(0)

        if swing_lows_confirmed[i] is not None:
            ssl_lvls.append(swing_lows_confirmed[i])
            ssl_swept.append(False)
            if len(ssl_lvls) > max_levels:
                ssl_lvls.pop(0)
                ssl_swept.pop(0)

        # 3. Register FVGs
        # Bullish FVG: low[i] > high[i-2]
        if l > highs[i - 2]:
            b_fvg_top.append(l)
            b_fvg_bot.append(highs[i - 2])
            b_fvg_bar.append(i)
            if len(b_fvg_top) > max_fvg:
                b_fvg_top.pop(0)
                b_fvg_bot.pop(0)
                b_fvg_bar.pop(0)

        # Bearish FVG: high[i] < low[i-2]
        if h < lows[i - 2]:
            s_fvg_top.append(lows[i - 2])
            s_fvg_bot.append(h)
            s_fvg_bar.append(i)
            if len(s_fvg_top) > max_fvg:
                s_fvg_top.pop(0)
                s_fvg_bot.pop(0)
                s_fvg_bar.pop(0)

        # 4. Check Liquidity Sweeps
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

        # 5. Setup Trigger
        if ssl_sweep and in_session and not has_open:
            look_long = True
            look_short = False
            setup_bar_l = i
            setup_sl_l = sweep_low - current_atr * atr_mult

        if bsl_sweep and in_session and not has_open:
            look_short = True
            look_long = False
            setup_bar_s = i
            setup_sl_s = sweep_high + current_atr * atr_mult

        # Cancel setup if outside session or expired
        if not in_session:
            look_long = False
            look_short = False

        if look_long and (i - setup_bar_l) > expire_bars:
            look_long = False
        if look_short and (i - setup_bar_s) > expire_bars:
            look_short = False

        # 6. Entry Trigger (Mitigate qualifying FVG)
        if look_long and not has_open:
            nf = len(b_fvg_bar)
            if nf > 0:
                for f_idx in range(nf - 1, -1, -1):
                    if b_fvg_bar[f_idx] > setup_bar_l:
                        f_top = b_fvg_top[f_idx]
                        f_bot = b_fvg_bot[f_idx]
                        mid = (f_top + f_bot) / 2.0
                        if l <= f_top and c > f_bot:
                            entry_p = mid
                            sl_p = setup_sl_l
                            sl_dist = entry_p - sl_p
                            if sl_dist > 0:
                                tp_p = entry_p + sl_dist * rr_ratio
                                
                                # Register trade
                                active_trade = {
                                    'id': len(trades) + 1,
                                    'direction': 1,
                                    'entry_price': entry_p,
                                    'entry_bar': i,
                                    'entry_time': current_time,
                                    'signal_time': int(pd.to_datetime(current_time).timestamp() * 1000),
                                    'sl': sl_p,
                                    'tp': tp_p,
                                    'risk_dist': sl_dist,
                                    'result': 0, # Open
                                }
                                bull_signals[i] = True
                                sl_long[i] = sl_p
                                tp_long[i] = tp_p

                                look_long = False
                                break

        if look_short and not has_open:
            nf = len(s_fvg_bar)
            if nf > 0:
                for f_idx in range(nf - 1, -1, -1):
                    if s_fvg_bar[f_idx] > setup_bar_s:
                        f_top = s_fvg_top[f_idx]
                        f_bot = s_fvg_bot[f_idx]
                        mid = (f_top + f_bot) / 2.0
                        if h >= f_bot and c < f_top:
                            entry_p = mid
                            sl_p = setup_sl_s
                            sl_dist = sl_p - entry_p
                            if sl_dist > 0:
                                tp_p = entry_p - sl_dist * rr_ratio
                                
                                # Register trade
                                active_trade = {
                                    'id': len(trades) + 1,
                                    'direction': -1,
                                    'entry_price': entry_p,
                                    'entry_bar': i,
                                    'entry_time': current_time,
                                    'signal_time': int(pd.to_datetime(current_time).timestamp() * 1000),
                                    'sl': sl_p,
                                    'tp': tp_p,
                                    'risk_dist': sl_dist,
                                    'result': 0, # Open
                                }
                                bear_signals[i] = True
                                sl_short[i] = sl_p
                                tp_short[i] = tp_p

                                look_short = False
                                break

    # Add signal columns to df
    df['bull_signal'] = bull_signals
    df['bear_signal'] = bear_signals
    df['sl_long'] = sl_long
    df['tp1_long'] = tp_long
    df['sl_short'] = sl_short
    df['tp1_short'] = tp_short

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # Calculate performance metrics
    if not trades_df.empty and 'result' in trades_df.columns:
        wins = int((trades_df['result'] == 1).sum())
        losses = int((trades_df['result'] == -1).sum())
        expired = int((trades_df['result'] == 2).sum())
        resolved = wins + losses
        win_rate = (wins / resolved * 100.0) if resolved > 0 else 0.0
        net_r = (wins * rr_ratio) - losses
        
        # Drawdown calculation in R
        r_history = [t['r_return'] for t in trades if t['result'] in (1, -1)]
        cum_r = 0.0
        peak = 0.0
        max_dd = 0.0
        for r_val in r_history:
            cum_r += r_val
            if cum_r > peak:
                peak = cum_r
            dd = peak - cum_r
            if dd > max_dd:
                max_dd = dd

        summary = {
            'retained_signals': len(trades),
            'resolved': resolved,
            'wins': wins,
            'losses': losses,
            'expired': expired,
            'win_rate_percent': round(win_rate, 1),
            'net_r_all_retained_completed_R': round(net_r, 1),
            'max_drawdown_R': round(max_dd, 1),
        }
    else:
        summary = {
            'retained_signals': 0, 'resolved': 0, 'wins': 0, 'losses': 0, 'expired': 0,
            'win_rate_percent': 0.0, 'net_r_all_retained_completed_R': 0.0, 'max_drawdown_R': 0.0
        }

    return df, trades_df, summary

if __name__ == '__main__':
    df = pd.read_csv('data/XAUUSD_5min.csv')
    processed_df, trades_df, summary = run_simulation(df)
    print("Backtest Results on XAUUSD_5min.csv:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
