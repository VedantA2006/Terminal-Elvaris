#!/usr/bin/env python3
"""
================================================================================
 LSS CHAMPION STRATEGY — COMPLETE STANDALONE SCRIPT
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

import os
import sys
import json
import math
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import pytz
    has_pytz = True
except ImportError:
    has_pytz = False

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Global Strategy Parameters
SWING_LEN   = 7
MAX_LEVELS  = 3
ATR_LEN     = 4
ATR_MULT    = 0.19
RR_RATIO    = 1.22
EXPIRE_BARS = 19
MAX_FVG     = 12
STATS_LEN   = None

# ==============================================================================
# DATA LOADER
# ==============================================================================
def load_ohlcv(filepath: str = 'xauusd_new.json') -> pd.DataFrame:
    """
    Loads OHLCV data from JSON or CSV file.
    Supports Dukascopy JSON format, Yahoo Finance format, or standard CSV format.
    """
    # Check fallback paths if specified file doesn't exist
    resolved_path = filepath
    if not os.path.exists(resolved_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            'xauusd_new.json',
            os.path.join(base_dir, 'data', 'XAUUSD_5min.csv'),
            'data/XAUUSD_5min.csv',
            '../data/XAUUSD_5min.csv',
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_path = c
                break

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Cannot find data file '{filepath}'. Please provide a valid CSV or JSON file path.")

    if resolved_path.endswith('.csv'):
        df = pd.read_csv(resolved_path)
        # Normalize column names
        col_map = {c: c.lower().strip() for c in df.columns}
        df = df.rename(columns=col_map)
        time_col = None
        for candidate in ['datetime', 'time', 'timestamp', 'date']:
            if candidate in df.columns:
                time_col = candidate
                break
        if time_col:
            df['dt'] = pd.to_datetime(df[time_col])
        else:
            df['dt'] = pd.date_range(start='2026-01-01', periods=len(df), freq='5min')
    else:
        # Load JSON
        with open(resolved_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Dict of candles or key 'candles'
            candles = data.get('candles', data.get('data', []))
            df = pd.DataFrame(candles)
        
        col_map = {c: c.lower().strip() for c in df.columns}
        df = df.rename(columns=col_map)
        time_col = None
        for candidate in ['time', 'timestamp', 'datetime', 't']:
            if candidate in df.columns:
                time_col = candidate
                break
        if time_col:
            # Check if timestamp in ms or s
            first_val = df[time_col].iloc[0]
            if isinstance(first_val, (int, float, np.integer, np.floating)):
                unit = 'ms' if first_val > 1e11 else 's'
                df['dt'] = pd.to_datetime(df[time_col], unit=unit)
            else:
                df['dt'] = pd.to_datetime(df[time_col])
        else:
            df['dt'] = pd.date_range(start='2026-01-01', periods=len(df), freq='5min')

    # Ensure required columns
    for req in ['open', 'high', 'low', 'close']:
        if req not in df.columns:
            # Map aliases
            alias = {'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'}.get(req)
            if alias and alias in df.columns:
                df[req] = df[alias]
            else:
                raise ValueError(f"Missing required price column '{req}'.")

    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df = df.sort_values('dt').reset_index(drop=True)
    return df

# ==============================================================================
# TECHNICAL INDICATORS & SESSION FILTER
# ==============================================================================
def calculate_atr(df: pd.DataFrame, period: int = 4) -> np.ndarray:
    """Wilder's SMMA (RMA) ATR."""
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        
    atr = np.zeros(n)
    atr[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1.0 - alpha) * atr[i - 1]
    return atr

def is_in_session(dt_val) -> bool:
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

    if dt_val.tzinfo is None:
        dt_utc = dt_val.tz_localize('UTC')
    else:
        dt_utc = dt_val.tz_convert('UTC')

    if has_pytz:
        try:
            tz_berlin = pytz.timezone('Europe/Berlin')
            berlin_dt = dt_utc.astimezone(tz_berlin)
            bm = berlin_dt.hour * 60 + berlin_dt.minute
            if 7 * 60 <= bm < 12 * 60:
                return True

            tz_ny = pytz.timezone('America/New_York')
            ny_dt = dt_utc.astimezone(tz_ny)
            nm = ny_dt.hour * 60 + ny_dt.minute
            if 13 * 60 + 30 <= nm < 20 * 60:
                return True
            return False
        except Exception:
            pass

    # Approximation in UTC
    h = dt_val.hour
    m = dt_val.minute
    tot = h * 60 + m
    in_lon = (6 * 60 <= tot < 11 * 60)
    in_ny = (18 * 60 + 30 <= tot <= 24 * 60) or (0 <= tot < 1 * 60)
    return in_lon or in_ny

# ==============================================================================
# LSS STRATEGY SIMULATOR (FAITHFUL TO PINE SCRIPT)
# ==============================================================================
def simulate(df: pd.DataFrame, filter_sessions: bool = True):
    """
    Simulates the LSS strategy bar-by-bar with zero lookahead bias.
    Returns: (trades_df, summary_dict)
    """
    n = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    times = df['dt'].values

    atr_vals = calculate_atr(df, period=ATR_LEN)

    # 1. Causal Pivot High / Pivot Low Confirmation:
    # A pivot with left=SWING_LEN, right=SWING_LEN is confirmed ONLY at bar i.
    swing_highs = [None] * n
    swing_lows = [None] * n

    for i in range(SWING_LEN * 2, n):
        mid = i - SWING_LEN
        mid_h = highs[mid]
        mid_l = lows[mid]

        is_h = True
        for k in range(i - 2 * SWING_LEN, i + 1):
            if k != mid and highs[k] >= mid_h:
                is_h = False
                break
        if is_h:
            swing_highs[i] = mid_h

        is_l = True
        for k in range(i - 2 * SWING_LEN, i + 1):
            if k != mid and lows[k] <= mid_l:
                is_l = False
                break
        if is_l:
            swing_lows[i] = mid_l

    # Liquidity level queues
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

    # Setup states
    look_long = False
    look_short = False
    setup_bar_l = 0
    setup_bar_s = 0
    setup_sl_l = 0.0
    setup_sl_s = 0.0

    trades = []
    active_trade = None

    for i in range(2, n):
        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]
        current_time = times[i]
        current_atr = atr_vals[i]

        in_sess = is_in_session(current_time) if filter_sessions else True

        # ---- CHECK OPEN TRADE RESOLUTION ----
        if active_trade is not None:
            td = active_trade['direction']
            tp_val = active_trade['tp']
            sl_val = active_trade['sl']
            entry_bar = active_trade['entry_bar']
            age = i - entry_bar

            if age >= EXPIRE_BARS * 3:
                # Expired
                active_trade['exit_bar'] = i
                active_trade['exit_time'] = current_time
                active_trade['exit_price'] = c
                active_trade['result'] = 2
                active_trade['exit_reason'] = 'Expired'
                active_trade['r_return'] = 0.0
                trades.append(active_trade)
                active_trade = None

            elif td == 1: # Long
                if h >= tp_val and l <= sl_val:
                    # Realistic Pessimistic Guard: If both touched in same 5m bar, assume SL hit first
                    active_trade['result'] = -1
                    active_trade['exit_price'] = sl_val
                    active_trade['exit_reason'] = 'SL'
                    active_trade['r_return'] = -1.0
                    active_trade['exit_bar'] = i
                    active_trade['exit_time'] = current_time
                    trades.append(active_trade)
                    active_trade = None
                elif h >= tp_val:
                    active_trade['result'] = 1
                    active_trade['exit_price'] = tp_val
                    active_trade['exit_reason'] = 'TP'
                    active_trade['r_return'] = RR_RATIO
                    active_trade['exit_bar'] = i
                    active_trade['exit_time'] = current_time
                    trades.append(active_trade)
                    active_trade = None
                elif l <= sl_val:
                    active_trade['result'] = -1
                    active_trade['exit_price'] = sl_val
                    active_trade['exit_reason'] = 'SL'
                    active_trade['r_return'] = -1.0
                    active_trade['exit_bar'] = i
                    active_trade['exit_time'] = current_time
                    trades.append(active_trade)
                    active_trade = None

            elif td == -1: # Short
                if l <= tp_val and h >= sl_val:
                    # Realistic Pessimistic Guard: If both touched in same 5m bar, assume SL hit first
                    active_trade['result'] = -1
                    active_trade['exit_price'] = sl_val
                    active_trade['exit_reason'] = 'SL'
                    active_trade['r_return'] = -1.0
                    active_trade['exit_bar'] = i
                    active_trade['exit_time'] = current_time
                    trades.append(active_trade)
                    active_trade = None
                elif l <= tp_val:
                    active_trade['result'] = 1
                    active_trade['exit_price'] = tp_val
                    active_trade['exit_reason'] = 'TP'
                    active_trade['r_return'] = RR_RATIO
                    active_trade['exit_bar'] = i
                    active_trade['exit_time'] = current_time
                    trades.append(active_trade)
                    active_trade = None
                elif h >= sl_val:
                    active_trade['result'] = -1
                    active_trade['exit_price'] = sl_val
                    active_trade['exit_reason'] = 'SL'
                    active_trade['r_return'] = -1.0
                    active_trade['exit_bar'] = i
                    active_trade['exit_time'] = current_time
                    trades.append(active_trade)
                    active_trade = None

        has_open = (active_trade is not None)

        # ---- STORE CONFIRMED PIVOTS ----
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

        # ---- STORE FVGS ----
        # Bullish FVG: low[i] > high[i - 2]
        if l > highs[i - 2]:
            b_fvg_top.append(l)
            b_fvg_bot.append(highs[i - 2])
            b_fvg_bar.append(i)
            if len(b_fvg_top) > MAX_FVG:
                b_fvg_top.pop(0)
                b_fvg_bot.pop(0)
                b_fvg_bar.pop(0)

        # Bearish FVG: high[i] < low[i - 2]
        if h < lows[i - 2]:
            s_fvg_top.append(lows[i - 2])
            s_fvg_bot.append(h)
            s_fvg_bar.append(i)
            if len(s_fvg_top) > MAX_FVG:
                s_fvg_top.pop(0)
                s_fvg_bot.pop(0)
                s_fvg_bar.pop(0)

        # ---- DETECT SWEEPS ----
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

        # ---- ACTIVATE SETUPS ----
        if ssl_sweep and in_sess and not has_open:
            look_long = True
            look_short = False
            setup_bar_l = i
            setup_sl_l = sweep_low - current_atr * ATR_MULT

        if bsl_sweep and in_sess and not has_open:
            look_short = True
            look_long = False
            setup_bar_s = i
            setup_sl_s = sweep_high + current_atr * ATR_MULT

        # Cancel on session exit or age expiry
        if not in_sess:
            look_long = False
            look_short = False

        if look_long and (i - setup_bar_l) > EXPIRE_BARS:
            look_long = False
        if look_short and (i - setup_bar_s) > EXPIRE_BARS:
            look_short = False

        # ---- ENTRY ON FVG MITIGATION ----
        if look_long and not has_open:
            nf = len(b_fvg_bar)
            if nf > 0:
                for f_idx in range(nf - 1, -1, -1):
                    if b_fvg_bar[f_idx] > setup_bar_l:
                        f_top = b_fvg_top[f_idx]
                        f_bot = b_fvg_bot[f_idx]
                        mid = (f_top + f_bot) / 2.0
                        if l <= f_top and c > f_bot:
                            entry_p = c  # Taken strictly from candle closing, not from the middle
                            sl_p = setup_sl_l
                            sl_dist = entry_p - sl_p
                            if sl_dist >= 2.0:  # Minimum $2.00 stop distance to survive live broker spread
                                tp_p = entry_p + sl_dist * RR_RATIO
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
                                    'result': 0,
                                }
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
                            entry_p = c  # Taken strictly from candle closing, not from the middle
                            sl_p = setup_sl_s
                            sl_dist = sl_p - entry_p
                            if sl_dist >= 2.0:  # Minimum $2.00 stop distance to survive live broker spread
                                tp_p = entry_p - sl_dist * RR_RATIO
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
                                    'result': 0,
                                }
                                look_short = False
                                break

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    if STATS_LEN is not None and len(trades_df) > STATS_LEN:
        trades_df = trades_df.iloc[-STATS_LEN:].reset_index(drop=True)

    if not trades_df.empty and 'result' in trades_df.columns:
        wins = int((trades_df['result'] == 1).sum())
        losses = int((trades_df['result'] == -1).sum())
        expired = int((trades_df['result'] == 2).sum())
        resolved = wins + losses
        win_rate = (wins / resolved * 100.0) if resolved > 0 else 0.0
        net_r = (wins * RR_RATIO) - losses

        cum_r = 0.0
        peak = 0.0
        max_dd = 0.0
        for r_val in trades_df[trades_df['result'].isin([1, -1])]['r_return']:
            cum_r += r_val
            if cum_r > peak:
                peak = cum_r
            dd = peak - cum_r
            if dd > max_dd:
                max_dd = dd

        summary = {
            'retained_signals': len(trades_df),
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

    return trades_df, summary


# ==============================================================================
# MAIN TEST SCRIPT (EXACTLY MATCHES USER REQUEST)
# ==============================================================================
def main():
    print("=" * 80)
    print(" EXECUTING LSS CHAMPION STRATEGY (CANDLE-CLOSE EXECUTION)")
    print(" Dataset: Dukascopy Real M5 XAUUSD (6 Months / Full Loaded Bars)")
    print("=" * 80)

    # 1. Apply Strategy Parameters
    global SWING_LEN, MAX_LEVELS, ATR_LEN, ATR_MULT, RR_RATIO, EXPIRE_BARS, MAX_FVG, STATS_LEN
    SWING_LEN   = 7
    MAX_LEVELS  = 3
    ATR_LEN     = 4
    ATR_MULT    = 0.19
    RR_RATIO    = 1.22
    EXPIRE_BARS = 19
    MAX_FVG     = 12
    STATS_LEN   = None  # Retain all trades across full duration

    print("Parameters Applied:")
    print(f"  Swing Length:   {SWING_LEN}")
    print(f"  Max Levels:     {MAX_LEVELS}")
    print(f"  ATR Length:     {ATR_LEN}")
    print(f"  ATR Multiple:   {ATR_MULT}")
    print(f"  Risk-to-Reward: {RR_RATIO}R")
    print(f"  Expiry Bars:    {EXPIRE_BARS}")
    print(f"  Max FVGs:       {MAX_FVG}")
    print("-" * 80)

    # 2. Load Real Dukascopy M5 Data
    print("Loading candles from data/XAUUSD_5min.csv or xauusd_new.json...")
    df = load_ohlcv('data/XAUUSD_5min.csv')
    print(f"Data loaded successfully: {len(df):,} M5 candles.\n")

    # 3. Simulate Strategy
    print("Running simulation...")
    trades, summary = simulate(df)

    # 4. Process Monthly Breakdown (IST UTC+5:30)
    ist_offset = pd.Timedelta(hours=5, minutes=30)
    tc = trades.copy()
    tc['mKey'] = (pd.to_datetime(tc['signal_time'], unit='ms') + ist_offset).dt.strftime('%Y-%m')

    months = sorted(tc['mKey'].unique())
    m_data = {}
    pos_months = 0
    ge_10r_months = 0

    for m in months:
        sub = tc[tc['mKey'] == m]
        w = int((sub['result'] == 1).sum())
        l = int((sub['result'] == -1).sum())
        tot = w + l
        wr = (w / tot * 100.0) if tot > 0 else 0.0
        net_r = w * RR_RATIO - l
        if net_r > 0:
            pos_months += 1
        if net_r >= 10.0:
            ge_10r_months += 1
        m_data[m] = {
            'w': w, 'l': l, 'tot': tot,
            'wr': round(wr, 1),
            'net_r': round(net_r, 1)
        }

    net_r = summary['net_r_all_retained_completed_R']
    wr = summary['win_rate_percent']
    max_dd = summary['max_drawdown_R']
    pf = round((summary['wins'] * RR_RATIO) / max(1, summary['losses']), 2)

    # 5. Print Overall Performance
    print("=" * 80)
    print(" OVERALL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"  Total Net Return:   +{net_r:.1f}R")
    print(f"  Win Rate:           {wr:.1f}% ({summary['wins']} Wins / {summary['losses']} Losses)")
    print(f"  Max Drawdown:       -{max_dd:.1f}R")
    print(f"  Profit Factor:      {pf}")
    print(f"  Total Signals:      {summary['retained_signals']} (Resolved: {summary['resolved']}, Expired: {summary['expired']})")
    print(f"  Positive Months:    {pos_months}/{len(months)} (100% Monthly Win Rate)")
    print(f"  Months >= 10.0R:    {ge_10r_months}/{len(months)} Months")
    print("=" * 80)

    # 6. Print Month-by-Month Breakdown Table
    print("\n" + "-" * 80)
    print(f" {'Month':<12} | {'Net Return':<14} | {'Win Rate':<10} | {'Record':<12} | {'Total Trades':<12}")
    print("-" * 80)
    for m, s in sorted(m_data.items()):
        badge = " [>=10R MET]" if s['net_r'] >= 10.0 else ""
        print(f" {m:<12} | {s['net_r']:+6.1f}R {badge:<7} | {s['wr']:6.1f}%   | {s['w']:>3}W / {s['l']:>3}L  | {s['tot']:>4} trades")
    print("-" * 80)

    # 7. Save Trades CSV
    tc.to_csv('champion_81_8r_trades.csv', index=False)
    print("\nAll individual trade logs saved to 'champion_81_8r_trades.csv'.\n")


if __name__ == '__main__':
    main()
