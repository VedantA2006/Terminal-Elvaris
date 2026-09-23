"""
Elvaris & Custom Strategy Backtester — Bar-by-bar simulation engine.

Features:
  - Bar-by-bar causal simulation (strictly NO future-peeking)
  - Realistic spread and slippage modeling (prevents fake zero-friction results)
  - Supports both multi-level partial TPs (Elvaris style) and standard single TP/SL
  - Comprehensive TradingView Strategy Tester statistics (Sharpe, Sortino, Drawdown %, Win/Loss ratios)
"""

import numpy as np
import pandas as pd
import math

from stats_utils import compute_sharpe_sortino

def _compute_stats(trades, equity_curve, initial_capital, max_drawdown, max_drawdown_pct):
    if not trades:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0.0, 'total_pnl': 0.0, 'net_profit_pct': 0.0,
            'gross_profit': 0.0, 'gross_loss': 0.0, 'profit_factor': 0.0,
            'max_drawdown': 0.0, 'max_drawdown_pct': 0.0,
            'avg_pnl': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
            'max_consecutive_wins': 0, 'max_consecutive_losses': 0,
            'long_trades': 0, 'short_trades': 0,
        }

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    longs = [t for t in trades if t['direction'] == 'long']
    shorts = [t for t in trades if t['direction'] == 'short']

    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    total_pnl = sum(pnls)

    # Consecutive wins & losses
    cur_w = max_w = cur_l = max_l = 0
    for p in pnls:
        if p > 0:
            cur_w += 1
            cur_l = 0
            max_w = max(max_w, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_l = max(max_l, cur_l)

    # Sharpe & Sortino via shared single-source-of-truth utility
    sharpe, sortino = compute_sharpe_sortino(trades, equity_curve, initial_capital)

    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(abs(np.mean(losses))) if losses else 0.0
    win_loss_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

    return {
        'total_trades': len(trades),
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': round(len(wins) / len(trades) * 100, 1),
        'total_pnl': round(total_pnl, 2),
        'net_profit_pct': round((total_pnl / initial_capital) * 100, 2),
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
        'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.0,
        'max_drawdown': round(max_drawdown, 2),
        'max_drawdown_pct': round(max_drawdown_pct, 2),
        'avg_pnl': round(float(np.mean(pnls)), 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'ratio_win_loss': win_loss_ratio,
        'best_trade': round(max(pnls), 2),
        'worst_trade': round(min(pnls), 2),
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_consecutive_wins': max_w,
        'max_consecutive_losses': max_l,
        'long_trades': len(longs),
        'short_trades': len(shorts),
    }


from numba import njit






@njit(cache=True)
def _run_backtest_numba(
    open_arr, high_arr, low_arr, close_arr,
    bull_signal, bear_signal,
    sl_long, sl_short,
    tp_longs, tp_shorts, # shape (N, 5)
    risk_pct, use_breakeven_arr, use_trailing_arr,
    is_weekend_arr, is_news_arr,
    initial_capital, lot_size, partial_tp, cost_per_oz,
    fallback_sl_dist, min_stop_dist, min_tp_dist, be_threshold,
    has_sl, has_multi_tp, has_single_tp,
    in_warmup_arr
):
    N = len(open_arr)
    MAX_TRADES = N // 2 + 10
    
    # Preallocate output arrays
    tr_id = np.zeros(MAX_TRADES, dtype=np.int32)
    tr_dir = np.zeros(MAX_TRADES, dtype=np.int32) # 1=long, -1=short
    tr_entry_idx = np.zeros(MAX_TRADES, dtype=np.int32)
    tr_exit_idx = np.zeros(MAX_TRADES, dtype=np.int32)
    tr_entry_price = np.zeros(MAX_TRADES, dtype=np.float64)
    tr_exit_price = np.zeros(MAX_TRADES, dtype=np.float64)
    tr_sl = np.zeros(MAX_TRADES, dtype=np.float64)
    tr_initial_sl = np.zeros(MAX_TRADES, dtype=np.float64)
    tr_tps = np.zeros((MAX_TRADES, 5), dtype=np.float64)
    tr_initial_size = np.zeros(MAX_TRADES, dtype=np.float64)
    tr_pnl = np.zeros(MAX_TRADES, dtype=np.float64)
    tr_exit_reason = np.zeros(MAX_TRADES, dtype=np.int32)
    tr_tp_hits = np.zeros(MAX_TRADES, dtype=np.int32)
    tr_risk_usd = np.zeros(MAX_TRADES, dtype=np.float64)
    
    eq_idx = np.zeros(N, dtype=np.int32)
    eq_val = np.zeros(N, dtype=np.float64)
    eq_pnl = np.zeros(N, dtype=np.float64)
    eq_count = 0
    
    trade_count = 0
    pos_active = False
    
    # Position state
    p_dir = 0
    p_entry_price = 0.0
    p_entry_idx = 0
    p_sl = 0.0
    p_initial_sl = 0.0
    p_tps = np.zeros(5, dtype=np.float64)
    p_tps_orig = np.zeros(5, dtype=np.float64)
    p_initial_size = 0.0
    p_remaining_size = 0.0
    p_risk_usd = 0.0
    p_tp_hits = 0
    p_pnl = 0.0
    p_be_activated = False
    p_use_breakeven = False
    p_use_trailing = False
    p_peak_price = 0.0
    p_trough_price = 0.0
    
    cumulative_pnl = 0.0
    peak_equity = initial_capital
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    
    for i in range(1, N):
        row_open = open_arr[i]
        row_high = high_arr[i]
        row_low = low_arr[i]
        row_close = close_arr[i]
        
        # 1. CHECK EXITS FOR OPEN POSITION
        if pos_active:
            closed = False
            risk_dist = abs(p_entry_price - p_initial_sl) if p_initial_sl > 0 else fallback_sl_dist
            
            # Weekend & News Force Close
            if is_weekend_arr[i] or is_news_arr[i]:
                exit_price = row_open - cost_per_oz if p_dir == 1 else row_open + cost_per_oz
                pnl = (exit_price - p_entry_price) * p_remaining_size if p_dir == 1 else (p_entry_price - exit_price) * p_remaining_size
                cumulative_pnl += pnl
                p_pnl += pnl
                
                tr_id[trade_count] = trade_count + 1
                tr_dir[trade_count] = p_dir
                tr_entry_idx[trade_count] = p_entry_idx
                tr_exit_idx[trade_count] = i
                tr_entry_price[trade_count] = p_entry_price
                tr_exit_price[trade_count] = exit_price
                tr_sl[trade_count] = p_sl
                tr_initial_sl[trade_count] = p_initial_sl
                for k in range(5): tr_tps[trade_count, k] = p_tps_orig[k]
                tr_initial_size[trade_count] = p_initial_size
                tr_pnl[trade_count] = p_pnl
                tr_exit_reason[trade_count] = 1 if is_weekend_arr[i] else 2 
                tr_tp_hits[trade_count] = p_tp_hits
                tr_risk_usd[trade_count] = p_risk_usd
                
                trade_count += 1
                pos_active = False
                closed = True
                
            if not closed and p_dir == 1:
                # Dynamic Breakeven Stop
                if p_use_breakeven and not p_be_activated and (row_high >= p_entry_price + (1.8 * risk_dist)):
                    be_level = p_entry_price + cost_per_oz
                    if p_sl == 0.0 or p_sl < be_level:
                        p_sl = be_level
                        p_be_activated = True
                        
                tp_val = p_tps[p_tp_hits] if (p_tp_hits < 5 and p_tps[p_tp_hits] > 0) else 0.0
                sl_val = p_sl
                
                sl_hit = (sl_val > 0.0 and row_low <= sl_val)
                tp_hit = (tp_val > 0.0 and row_high >= tp_val)
                
                take_tp = False
                take_sl = False
                
                if tp_hit and not sl_hit:
                    take_tp = True
                elif sl_hit and not tp_hit:
                    take_sl = True
                elif tp_hit and sl_hit:
                    if row_open >= tp_val:
                        take_tp = True
                    else:
                        take_sl = True
                        
                if take_tp:
                    fill_price = tp_val - cost_per_oz
                    has_runner = (p_tps[p_tp_hits+1] > 0.0 if p_tp_hits+1 < 5 else False) and (p_remaining_size > np.round(p_initial_size * 0.45, 2))
                    if has_runner:
                        scale_size = np.round(p_initial_size * 0.5, 2)
                        pnl = (fill_price - p_entry_price) * scale_size
                        cumulative_pnl += pnl
                        p_pnl += pnl
                        p_remaining_size -= scale_size
                        p_tp_hits += 1
                        
                        p_sl = max(p_sl, p_entry_price + cost_per_oz)
                        p_be_activated = True
                    else:
                        pnl = (fill_price - p_entry_price) * p_remaining_size
                        cumulative_pnl += pnl
                        p_pnl += pnl
                        
                        tr_id[trade_count] = trade_count + 1
                        tr_dir[trade_count] = p_dir
                        tr_entry_idx[trade_count] = p_entry_idx
                        tr_exit_idx[trade_count] = i
                        tr_entry_price[trade_count] = p_entry_price
                        tr_exit_price[trade_count] = fill_price
                        tr_sl[trade_count] = p_sl
                        tr_initial_sl[trade_count] = p_initial_sl
                        for k in range(5): tr_tps[trade_count, k] = p_tps_orig[k]
                        tr_initial_size[trade_count] = p_initial_size
                        tr_pnl[trade_count] = p_pnl
                        tr_exit_reason[trade_count] = 3 if p_tp_hits == 0 else 4 # 3=TP, 4=TP_Runner
                        tr_tp_hits[trade_count] = p_tp_hits
                        tr_risk_usd[trade_count] = p_risk_usd
                        
                        trade_count += 1
                        pos_active = False
                        closed = True
                elif take_sl:
                    exit_price = sl_val - cost_per_oz
                    pnl = (exit_price - p_entry_price) * p_remaining_size
                    cumulative_pnl += pnl
                    p_pnl += pnl
                    
                    reason = 8 # SL
                    if p_tp_hits > 0 and pnl > 0: reason = 5 # Trail_SL
                    elif p_be_activated and abs(pnl) < be_threshold: reason = 6 # BE
                    elif p_tp_hits > 0: reason = 7 # TP1_SL
                    
                    tr_id[trade_count] = trade_count + 1
                    tr_dir[trade_count] = p_dir
                    tr_entry_idx[trade_count] = p_entry_idx
                    tr_exit_idx[trade_count] = i
                    tr_entry_price[trade_count] = p_entry_price
                    tr_exit_price[trade_count] = exit_price
                    tr_sl[trade_count] = p_sl
                    tr_initial_sl[trade_count] = p_initial_sl
                    for k in range(5): tr_tps[trade_count, k] = p_tps_orig[k]
                    tr_initial_size[trade_count] = p_initial_size
                    tr_pnl[trade_count] = p_pnl
                    tr_exit_reason[trade_count] = reason
                    tr_tp_hits[trade_count] = p_tp_hits
                    tr_risk_usd[trade_count] = p_risk_usd
                    
                    trade_count += 1
                    pos_active = False
                    closed = True
                    
            elif not closed and p_dir == -1:
                # Dynamic Breakeven Stop
                if p_use_breakeven and not p_be_activated and (row_low <= p_entry_price - (1.8 * risk_dist)):
                    be_level = p_entry_price - cost_per_oz
                    if p_sl == 0.0 or p_sl > be_level:
                        p_sl = be_level
                        p_be_activated = True
                        
                tp_val = p_tps[p_tp_hits] if (p_tp_hits < 5 and p_tps[p_tp_hits] > 0) else 0.0
                sl_val = p_sl
                
                sl_hit = (sl_val > 0.0 and row_high >= sl_val)
                tp_hit = (tp_val > 0.0 and row_low <= tp_val)
                
                take_tp = False
                take_sl = False
                
                if tp_hit and not sl_hit:
                    take_tp = True
                elif sl_hit and not tp_hit:
                    take_sl = True
                elif tp_hit and sl_hit:
                    if row_open <= tp_val:
                        take_tp = True
                    else:
                        take_sl = True
                        
                if take_tp:
                    fill_price = tp_val + cost_per_oz
                    has_runner = (p_tps[p_tp_hits+1] > 0.0 if p_tp_hits+1 < 5 else False) and (p_remaining_size > np.round(p_initial_size * 0.45, 2))
                    if has_runner:
                        scale_size = np.round(p_initial_size * 0.5, 2)
                        pnl = (p_entry_price - fill_price) * scale_size
                        cumulative_pnl += pnl
                        p_pnl += pnl
                        p_remaining_size -= scale_size
                        p_tp_hits += 1
                        
                        p_sl = min(p_sl if p_sl > 0 else 1e9, p_entry_price - cost_per_oz)
                        p_be_activated = True
                    else:
                        pnl = (p_entry_price - fill_price) * p_remaining_size
                        cumulative_pnl += pnl
                        p_pnl += pnl
                        
                        tr_id[trade_count] = trade_count + 1
                        tr_dir[trade_count] = p_dir
                        tr_entry_idx[trade_count] = p_entry_idx
                        tr_exit_idx[trade_count] = i
                        tr_entry_price[trade_count] = p_entry_price
                        tr_exit_price[trade_count] = fill_price
                        tr_sl[trade_count] = p_sl
                        tr_initial_sl[trade_count] = p_initial_sl
                        for k in range(5): tr_tps[trade_count, k] = p_tps_orig[k]
                        tr_initial_size[trade_count] = p_initial_size
                        tr_pnl[trade_count] = p_pnl
                        tr_exit_reason[trade_count] = 3 if p_tp_hits == 0 else 4 # TP
                        tr_tp_hits[trade_count] = p_tp_hits
                        tr_risk_usd[trade_count] = p_risk_usd
                        
                        trade_count += 1
                        pos_active = False
                        closed = True
                elif take_sl:
                    exit_price = sl_val + cost_per_oz
                    pnl = (p_entry_price - exit_price) * p_remaining_size
                    cumulative_pnl += pnl
                    p_pnl += pnl
                    
                    reason = 8 # SL
                    if p_tp_hits > 0 and pnl > 0: reason = 5 # Trail_SL
                    elif p_be_activated and abs(pnl) < be_threshold: reason = 6 # BE
                    elif p_tp_hits > 0: reason = 7 # TP1_SL
                    
                    tr_id[trade_count] = trade_count + 1
                    tr_dir[trade_count] = p_dir
                    tr_entry_idx[trade_count] = p_entry_idx
                    tr_exit_idx[trade_count] = i
                    tr_entry_price[trade_count] = p_entry_price
                    tr_exit_price[trade_count] = exit_price
                    tr_sl[trade_count] = p_sl
                    tr_initial_sl[trade_count] = p_initial_sl
                    for k in range(5): tr_tps[trade_count, k] = p_tps_orig[k]
                    tr_initial_size[trade_count] = p_initial_size
                    tr_pnl[trade_count] = p_pnl
                    tr_exit_reason[trade_count] = reason
                    tr_tp_hits[trade_count] = p_tp_hits
                    tr_risk_usd[trade_count] = p_risk_usd
                    
                    trade_count += 1
                    pos_active = False
                    closed = True
                    
            # 3. Close on opposite signal
            if not closed and pos_active:
                opp_signal = (p_dir == 1 and bear_signal[i]) or (p_dir == -1 and bull_signal[i])
                if opp_signal:
                    unrealized_r = 0.0
                    if p_dir == 1:
                        unrealized_r = (row_close - p_entry_price) / risk_dist if risk_dist > 0 else 0.0
                    else:
                        unrealized_r = (p_entry_price - row_close) / risk_dist if risk_dist > 0 else 0.0
                        
                    is_protected_runner = (p_tp_hits > 0) or (unrealized_r >= 1.2 and p_be_activated)
                    if not is_protected_runner:
                        if p_dir == 1:
                            exit_price = row_close - cost_per_oz
                            pnl = (exit_price - p_entry_price) * p_remaining_size
                        else:
                            exit_price = row_close + cost_per_oz
                            pnl = (p_entry_price - exit_price) * p_remaining_size
                            
                        cumulative_pnl += pnl
                        p_pnl += pnl
                        
                        tr_id[trade_count] = trade_count + 1
                        tr_dir[trade_count] = p_dir
                        tr_entry_idx[trade_count] = p_entry_idx
                        tr_exit_idx[trade_count] = i
                        tr_entry_price[trade_count] = p_entry_price
                        tr_exit_price[trade_count] = exit_price
                        tr_sl[trade_count] = p_sl
                        tr_initial_sl[trade_count] = p_initial_sl
                        for k in range(5): tr_tps[trade_count, k] = p_tps_orig[k]
                        tr_initial_size[trade_count] = p_initial_size
                        tr_pnl[trade_count] = p_pnl
                        tr_exit_reason[trade_count] = 9 # Signal
                        tr_tp_hits[trade_count] = p_tp_hits
                        tr_risk_usd[trade_count] = p_risk_usd
                        
                        trade_count += 1
                        pos_active = False
                        closed = True
                        
            # 4. Asymmetric Dynamic Trailing Stop
            if not closed and pos_active:
                if p_dir == 1:
                    p_peak_price = max(p_peak_price, row_high)
                    if p_tp_hits > 0 or p_use_trailing:
                        trail_stop = p_peak_price - (1.6 * risk_dist)
                        if p_sl == 0.0 or trail_stop > p_sl:
                            p_sl = trail_stop
                else:
                    p_trough_price = min(p_trough_price, row_low)
                    if p_tp_hits > 0 or p_use_trailing:
                        trail_stop = p_trough_price + (1.6 * risk_dist)
                        if p_sl == 0.0 or trail_stop < p_sl:
                            p_sl = trail_stop
                            
        # ---- CHECK NEW ENTRIES ----
        if not pos_active and not in_warmup_arr[i]:
            bull = bull_signal[i]
            bear = bear_signal[i]
            
            if bull or bear:
                if is_weekend_arr[i] or is_news_arr[i]:
                    bull = False
                    bear = False
                    
            if bull and not bear:
                entry_price = row_close + cost_per_oz
                sl = sl_long[i] if has_sl else np.nan
                if not np.isnan(sl):
                    if sl >= entry_price:
                        sl = entry_price - fallback_sl_dist
                    elif (entry_price - sl) < min_stop_dist:
                        sl = entry_price - min_stop_dist
                else:
                    sl = entry_price - fallback_sl_dist
                    
                risk_dist = abs(entry_price - sl)
                
                cur_tps = np.zeros(5, dtype=np.float64)
                tp_count = 0
                if has_multi_tp:
                    for k in range(5):
                        val = tp_longs[i, k]
                        if not np.isnan(val) and val > entry_price:
                            cur_tps[tp_count] = val
                            tp_count += 1
                elif has_single_tp:
                    val = tp_longs[i, 0]
                    if not np.isnan(val) and val > entry_price:
                        cur_tps[tp_count] = val
                        tp_count += 1
                        
                if tp_count == 0:
                    cur_tps[0] = entry_price + max(min_tp_dist, 1.5 * risk_dist)
                else:
                    min_tp = entry_price + (1.5 * risk_dist)
                    if cur_tps[0] < min_tp:
                        cur_tps[0] = min_tp
                        
                r_pct = risk_pct[i]
                if np.isnan(r_pct): r_pct = 0.01
                risk_per_trade_usd = initial_capital * r_pct
                calc_size = np.round(risk_per_trade_usd / risk_dist, 2) if risk_dist > 0 else lot_size
                
                pos_active = True
                p_dir = 1
                p_entry_price = entry_price
                p_peak_price = entry_price
                p_trough_price = entry_price
                p_entry_idx = i
                p_sl = sl
                p_initial_sl = sl
                p_be_activated = False
                p_use_breakeven = (use_breakeven_arr[i] or tp_count > 1 or partial_tp)
                p_use_trailing = use_trailing_arr[i]
                for k in range(5): 
                    p_tps[k] = cur_tps[k]
                    p_tps_orig[k] = cur_tps[k]
                p_initial_size = calc_size
                p_remaining_size = calc_size
                p_risk_usd = risk_per_trade_usd
                p_tp_hits = 0
                p_pnl = 0.0
                
            elif bear and not bull:
                entry_price = row_close - cost_per_oz
                sl = sl_short[i] if has_sl else np.nan
                if not np.isnan(sl):
                    if sl <= entry_price:
                        sl = entry_price + fallback_sl_dist
                    elif (sl - entry_price) < min_stop_dist:
                        sl = entry_price + min_stop_dist
                else:
                    sl = entry_price + fallback_sl_dist
                    
                risk_dist = abs(sl - entry_price)
                
                cur_tps = np.zeros(5, dtype=np.float64)
                tp_count = 0
                if has_multi_tp:
                    for k in range(5):
                        val = tp_shorts[i, k]
                        if not np.isnan(val) and val < entry_price:
                            cur_tps[tp_count] = val
                            tp_count += 1
                elif has_single_tp:
                    val = tp_shorts[i, 0]
                    if not np.isnan(val) and val < entry_price:
                        cur_tps[tp_count] = val
                        tp_count += 1
                        
                if tp_count == 0:
                    cur_tps[0] = entry_price - max(min_tp_dist, 1.5 * risk_dist)
                else:
                    min_tp = entry_price - (1.5 * risk_dist)
                    if cur_tps[0] > min_tp:
                        cur_tps[0] = min_tp
                        
                r_pct = risk_pct[i]
                if np.isnan(r_pct): r_pct = 0.01
                risk_per_trade_usd = initial_capital * r_pct
                calc_size = np.round(risk_per_trade_usd / risk_dist, 2) if risk_dist > 0 else lot_size
                
                pos_active = True
                p_dir = -1
                p_entry_price = entry_price
                p_peak_price = entry_price
                p_trough_price = entry_price
                p_entry_idx = i
                p_sl = sl
                p_initial_sl = sl
                p_be_activated = False
                p_use_breakeven = (use_breakeven_arr[i] or tp_count > 1 or partial_tp)
                p_use_trailing = use_trailing_arr[i]
                for k in range(5): 
                    p_tps[k] = cur_tps[k]
                    p_tps_orig[k] = cur_tps[k]
                p_initial_size = calc_size
                p_remaining_size = calc_size
                p_risk_usd = risk_per_trade_usd
                p_tp_hits = 0
                p_pnl = 0.0
                
        # ---- EQUITY CURVE ----
        unrealized = 0.0
        worst_unrealized = 0.0
        if pos_active:
            if p_dir == 1:
                unrealized = (row_close - p_entry_price) * p_remaining_size
                worst_unrealized = (row_low - p_entry_price) * p_remaining_size
            else:
                unrealized = (p_entry_price - row_close) * p_remaining_size
                worst_unrealized = (p_entry_price - row_high) * p_remaining_size
                
        current_eq = initial_capital + cumulative_pnl + unrealized
        peak_equity = max(peak_equity, current_eq)
        
        worst_eq = initial_capital + cumulative_pnl + worst_unrealized
        intra_dd = peak_equity - worst_eq
        if intra_dd > max_drawdown:
            max_drawdown = intra_dd
            max_drawdown_pct = (intra_dd / peak_equity) * 100 if peak_equity > 0 else 0.0
            
        if i % 20 == 0 or not pos_active or i == N - 1:
            eq_idx[eq_count] = i
            eq_val[eq_count] = np.round(current_eq, 2)
            eq_pnl[eq_count] = np.round(cumulative_pnl + unrealized, 2)
            eq_count += 1
            
    # CLOSE AT END
    if pos_active:
        last_close = close_arr[N-1]
        if p_dir == 1:
            exit_price = last_close - cost_per_oz
            pnl = (exit_price - p_entry_price) * p_remaining_size
        else:
            exit_price = last_close + cost_per_oz
            pnl = (p_entry_price - exit_price) * p_remaining_size
            
        cumulative_pnl += pnl
        p_pnl += pnl
        
        tr_id[trade_count] = trade_count + 1
        tr_dir[trade_count] = p_dir
        tr_entry_idx[trade_count] = p_entry_idx
        tr_exit_idx[trade_count] = N - 1
        tr_entry_price[trade_count] = p_entry_price
        tr_exit_price[trade_count] = exit_price
        tr_sl[trade_count] = p_sl
        tr_initial_sl[trade_count] = p_initial_sl
        for k in range(5): tr_tps[trade_count, k] = p_tps_orig[k]
        tr_initial_size[trade_count] = p_initial_size
        tr_pnl[trade_count] = p_pnl
        tr_exit_reason[trade_count] = 10 # End
        tr_tp_hits[trade_count] = p_tp_hits
        tr_risk_usd[trade_count] = p_risk_usd
        trade_count += 1
        
    return (
        trade_count, tr_id, tr_dir, tr_entry_idx, tr_exit_idx, tr_entry_price, tr_exit_price,
        tr_sl, tr_initial_sl, tr_tps, tr_initial_size, tr_pnl, tr_exit_reason, tr_tp_hits, tr_risk_usd,
        eq_count, eq_idx, eq_val, eq_pnl,
        max_drawdown, max_drawdown_pct
    )



def run_backtest(df, initial_capital=100000.0, lot_size=100.0, partial_tp=False, spread=0.20, slippage=0.05):
    import numpy as np
    import pandas as pd
    from stats_utils import compute_sharpe_sortino
    from news_calendar import is_weekend_embargo, is_news_embargo
    
    
    cost_per_oz = (spread / 2.0) + slippage
    
    median_price = float(df['close'].median()) if len(df) > 0 else 1.0
    fallback_sl_dist = median_price * 0.0025
    min_stop_dist    = median_price * 0.001
    min_tp_dist      = median_price * 0.001
    be_threshold     = median_price * 0.025
    
    eval_start_time = df.attrs.get('eval_start_time') if hasattr(df, 'attrs') else None
    symbol = df.attrs.get('symbol', 'XAUUSD')
    
    has_multi_tp = 'tp1_long' in df.columns
    has_single_tp = 'tp_long' in df.columns
    has_sl = 'sl_long' in df.columns
    
    dt_index = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('datetime', df.get('time', df.get('dt', df.index))))
    
    is_weekend_vec = np.vectorize(is_weekend_embargo)
    is_news_vec = np.vectorize(lambda d: is_news_embargo(d, symbol=symbol))
    
    is_weekend_arr = is_weekend_vec(dt_index.to_pydatetime())
    is_news_arr = is_news_vec(dt_index.to_pydatetime())
    
    in_warmup_arr = np.zeros(len(df), dtype=np.bool_)
    if eval_start_time is not None:
        in_warmup_arr = (dt_index < eval_start_time).to_numpy()
        
    open_arr = df['open'].to_numpy(dtype=np.float64)
    high_arr = df['high'].to_numpy(dtype=np.float64)
    low_arr = df['low'].to_numpy(dtype=np.float64)
    close_arr = df['close'].to_numpy(dtype=np.float64)
    
    bull_signal = df.get('bull_signal', pd.Series(False, index=df.index)).to_numpy(dtype=np.bool_)
    bear_signal = df.get('bear_signal', pd.Series(False, index=df.index)).to_numpy(dtype=np.bool_)
    
    sl_long = df.get('sl_long', pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64)
    sl_short = df.get('sl_short', pd.Series(np.nan, index=df.index)).to_numpy(dtype=np.float64)
    
    tp_longs = np.full((len(df), 5), np.nan, dtype=np.float64)
    tp_shorts = np.full((len(df), 5), np.nan, dtype=np.float64)
    
    if has_multi_tp:
        for k in range(5):
            col_l = f'tp{k+1}_long'
            col_s = f'tp{k+1}_short'
            if col_l in df.columns: tp_longs[:, k] = df[col_l].to_numpy()
            if col_s in df.columns: tp_shorts[:, k] = df[col_s].to_numpy()
    elif has_single_tp:
        if 'tp_long' in df.columns: tp_longs[:, 0] = df['tp_long'].to_numpy()
        if 'tp_short' in df.columns: tp_shorts[:, 0] = df['tp_short'].to_numpy()
        
    risk_pct = df.get('risk_pct', pd.Series(0.01, index=df.index)).to_numpy(dtype=np.float64)
    use_breakeven_arr = df.get('use_breakeven', pd.Series(False, index=df.index)).to_numpy(dtype=np.bool_)
    use_trailing_arr = df.get('use_trailing', pd.Series(False, index=df.index)).to_numpy(dtype=np.bool_)
    
    res = _run_backtest_numba(
        open_arr, high_arr, low_arr, close_arr,
        bull_signal, bear_signal,
        sl_long, sl_short,
        tp_longs, tp_shorts,
        risk_pct, use_breakeven_arr, use_trailing_arr,
        is_weekend_arr, is_news_arr,
        float(initial_capital), float(lot_size), bool(partial_tp), float(cost_per_oz),
        float(fallback_sl_dist), float(min_stop_dist), float(min_tp_dist), float(be_threshold),
        bool(has_sl), bool(has_multi_tp), bool(has_single_tp),
        in_warmup_arr
    )
    
    (trade_count, tr_id, tr_dir, tr_entry_idx, tr_exit_idx, tr_entry_price, tr_exit_price,
    tr_sl, tr_initial_sl, tr_tps, tr_initial_size, tr_pnl, tr_exit_reason, tr_tp_hits, tr_risk_usd,
    eq_count, eq_idx, eq_val, eq_pnl,
    max_drawdown, max_drawdown_pct) = res
    
    trades = []
    reasons = {
        0: None, 1: 'Weekend Close', 2: 'News Embargo', 3: 'TP', 4: 'TP_Runner',
        5: 'Trail_SL', 6: 'BE', 7: 'TP1_SL', 8: 'SL', 9: 'Signal', 10: 'End'
    }
    
    str_times = df.index.astype(str).values if isinstance(df.index, pd.DatetimeIndex) else df.get('datetime', df.get('time', df.get('dt', df.index))).astype(str).values
    try:
        ts_times = (pd.to_datetime(str_times).view('int64') // 10**9)
    except:
        ts_times = np.zeros(len(df), dtype=np.int64)
    
    cumulative_pnl = 0.0
    for t in range(trade_count):
        entry_i = tr_entry_idx[t]
        exit_i = tr_exit_idx[t]
        cumulative_pnl += tr_pnl[t]
        
        r_ret = round(tr_pnl[t] / tr_risk_usd[t], 2) if tr_risk_usd[t] > 0 else round(tr_pnl[t] / 1000.0, 2)
        
        trades.append({
            'id': int(tr_id[t]),
            'direction': 'long' if tr_dir[t] == 1 else 'short',
            'entry_price': round(tr_entry_price[t], 2),
            'entry_time': str_times[entry_i],
            'entry_time_ts': int(ts_times[entry_i]) if ts_times[entry_i] > 0 else None,
            'exit_price': round(tr_exit_price[t], 2) if tr_exit_price[t] > 0 else None,
            'exit_time': str_times[exit_i],
            'exit_time_ts': int(ts_times[exit_i]) if ts_times[exit_i] > 0 else None,
            'exit_reason': reasons.get(tr_exit_reason[t]),
            'sl': round(tr_sl[t], 2) if tr_sl[t] > 0 else None,
            'tps': [round(x, 2) if x > 0 else None for x in tr_tps[t]],
            'initial_size': float(tr_initial_size[t]),
            'pnl': round(tr_pnl[t], 2),
            'pnl_percent': round((tr_pnl[t] / (tr_entry_price[t] * tr_initial_size[t])) * 100, 2) if tr_entry_price[t] > 0 else 0.0,
            'cumulative_pnl': round(cumulative_pnl, 2),
            'tp_hits': [1] * tr_tp_hits[t],
            'r_return': r_ret
        })
        
    equity_curve = []
    for e in range(eq_count):
        i = eq_idx[e]
        equity_curve.append({
            'time': int(ts_times[i]),
            'equity': float(eq_val[e]),
            'pnl': float(eq_pnl[e])
        })
        
    from backtest import _compute_stats
    stats = _compute_stats(trades, equity_curve, initial_capital, max_drawdown, max_drawdown_pct)
    
    return trades, equity_curve, stats
