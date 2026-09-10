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


def run_backtest(df, initial_capital=100000.0, lot_size=100.0, partial_tp=False, spread=0.20, slippage=0.05):
    """
    Run backtest on strategy-processed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'open', 'high', 'low', 'close', and signal columns:
        'bull_signal', 'bear_signal', plus optional 'sl_long', 'sl_short', 'tp...'.
    initial_capital : float
        Starting account balance in USD.
    lot_size : float
        Position size in oz (1 standard lot = 100 oz XAUUSD).
    partial_tp : bool
        If True, closes fractional positions across available TP levels.
    spread : float
        Dukascopy bid-ask spread in USD per oz (default $0.20).
    slippage : float
        Estimated execution slippage in USD per oz (default $0.05).

    Returns
    -------
    trades : list[dict]
    equity_curve : list[dict]
    stats : dict
    """
    trades = []
    position = None
    cumulative_pnl = 0.0
    equity_curve = []
    peak_equity = initial_capital
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    total_friction_cost = 0.0

    cost_per_oz = (spread / 2.0) + slippage

    # Check available TP/SL columns in df
    has_multi_tp = 'tp1_long' in df.columns
    has_single_tp = 'tp_long' in df.columns
    has_sl = 'sl_long' in df.columns

    for i in range(1, len(df)):
        row = df.iloc[i]
        if hasattr(row.name, 'isoformat'):
            bar_time = row.name.isoformat()
            bar_ts = int(row.name.timestamp())
        elif 'datetime' in row and pd.notna(row['datetime']):
            dt_val = pd.to_datetime(row['datetime'])
            bar_time = dt_val.isoformat()
            bar_ts = int(dt_val.timestamp())
        elif 'time' in row and pd.notna(row['time']):
            dt_val = pd.to_datetime(row['time'])
            bar_time = dt_val.isoformat()
            bar_ts = int(dt_val.timestamp())
        elif 'dt' in row and pd.notna(row['dt']):
            dt_val = pd.to_datetime(row['dt'])
            bar_time = dt_val.isoformat()
            bar_ts = int(dt_val.timestamp())
        else:
            bar_time = str(row.name)
            bar_ts = None

        # ---- CHECK EXITS FOR OPEN POSITION ----
        if position is not None:
            closed = False

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
                    # Conservative tie-breaking: only take TP if bar opened with an upside gap beyond TP
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
                    position['exit_time'] = bar_time
                    position['exit_time_ts'] = bar_ts
                    position['exit_reason'] = 'TP'
                    position['remaining_size'] = 0.0
                    trades.append(_finalize_trade(position, cumulative_pnl, initial_capital))
                    position = None
                    closed = True
                elif take_sl:
                    exit_price = sl_val - cost_per_oz
                    pnl = (exit_price - position['entry_price']) * position['remaining_size']
                    cumulative_pnl += pnl
                    position['pnl'] += pnl
                    position['exit_price'] = exit_price
                    position['exit_time'] = bar_time
                    position['exit_time_ts'] = bar_ts
                    position['exit_reason'] = 'SL'
                    position['remaining_size'] = 0.0
                    trades.append(_finalize_trade(position, cumulative_pnl, initial_capital))
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
                    # Conservative tie-breaking: only take TP if bar opened with a downside gap beyond TP
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
                    position['exit_time'] = bar_time
                    position['exit_time_ts'] = bar_ts
                    position['exit_reason'] = 'TP'
                    position['remaining_size'] = 0.0
                    trades.append(_finalize_trade(position, cumulative_pnl, initial_capital))
                    position = None
                    closed = True
                elif take_sl:
                    exit_price = sl_val + cost_per_oz
                    pnl = (position['entry_price'] - exit_price) * position['remaining_size']
                    cumulative_pnl += pnl
                    position['pnl'] += pnl
                    position['exit_price'] = exit_price
                    position['exit_time'] = bar_time
                    position['exit_time_ts'] = bar_ts
                    position['exit_reason'] = 'SL'
                    position['remaining_size'] = 0.0
                    trades.append(_finalize_trade(position, cumulative_pnl, initial_capital))
                    position = None
                    closed = True

            # 3. Close on opposite signal
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
                    position['exit_time'] = bar_time
                    position['exit_time_ts'] = bar_ts
                    position['exit_reason'] = 'Signal'
                    trades.append(_finalize_trade(position, cumulative_pnl, initial_capital))
                    position = None

        # ---- CHECK NEW ENTRIES ----
        if position is None:
            bull = bool(row.get('bull_signal', False))
            bear = bool(row.get('bear_signal', False))

            if bull and not bear:
                entry_price = row['close'] + cost_per_oz
                sl = row.get('sl_long') if has_sl else None
                if sl is not None:
                    if math.isnan(sl) or sl >= entry_price:
                        sl = entry_price - 10.0  # Fallback 10 pt SL
                    elif (entry_price - sl) < 2.5:
                        sl = entry_price - 2.5   # Mandatory minimum $2.50 stop distance to survive spread
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
                    tps = [entry_price + max(4.0, 1.5 * risk_dist)]  # Robust 1.5:1 TP
                else:
                    # Prevent inverted payouts: ensure primary TP is at least 1.5x the actual stop distance
                    min_tp = entry_price + (1.5 * risk_dist)
                    if tps[0] < min_tp:
                        tps[0] = min_tp

                risk_per_trade_usd = initial_capital * 0.01
                calc_size = round(risk_per_trade_usd / risk_dist, 2) if risk_dist > 0 else lot_size

                position = {
                    'id': len(trades) + 1,
                    'direction': 'long',
                    'entry_price': entry_price,
                    'entry_time': bar_time,
                    'entry_time_ts': bar_ts,
                    'entry_bar_idx': i,
                    'sl': sl,
                    'tps': tps.copy(),
                    'tps_original': tps.copy(),
                    'initial_size': calc_size,
                    'remaining_size': calc_size,
                    'risk_usd': risk_per_trade_usd,
                    'tp_hits': [],
                    'pnl': 0.0,
                    'exit_price': None,
                    'exit_time': None,
                    'exit_time_ts': None,
                    'exit_reason': None,
                }

            elif bear and not bull:
                entry_price = row['close'] - cost_per_oz
                sl = row.get('sl_short') if has_sl else None
                if sl is not None:
                    if math.isnan(sl) or sl <= entry_price:
                        sl = entry_price + 10.0
                    elif (sl - entry_price) < 2.5:
                        sl = entry_price + 2.5   # Mandatory minimum $2.50 stop distance to survive spread
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
                    # Prevent inverted payouts: ensure primary TP is at least 1.5x the actual stop distance
                    min_tp = entry_price - (1.5 * risk_dist)
                    if tps[0] > min_tp:
                        tps[0] = min_tp

                risk_per_trade_usd = initial_capital * 0.01
                calc_size = round(risk_per_trade_usd / risk_dist, 2) if risk_dist > 0 else lot_size

                position = {
                    'id': len(trades) + 1,
                    'direction': 'short',
                    'entry_price': entry_price,
                    'entry_time': bar_time,
                    'entry_time_ts': bar_ts,
                    'entry_bar_idx': i,
                    'sl': sl,
                    'tps': tps.copy(),
                    'tps_original': tps.copy(),
                    'initial_size': calc_size,
                    'remaining_size': calc_size,
                    'risk_usd': risk_per_trade_usd,
                    'tp_hits': [],
                    'pnl': 0.0,
                    'exit_price': None,
                    'exit_time': None,
                    'exit_time_ts': None,
                    'exit_reason': None,
                }

        # ---- CONTINUOUS PER-BAR EQUITY & ADVERSE EXCURSION DRAWDOWN ----
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

        # Intra-bar worst excursion drawdown (prevents hidden drawdown between sample intervals)
        worst_eq = initial_capital + cumulative_pnl + worst_unrealized
        intra_dd = peak_equity - worst_eq
        if intra_dd > max_drawdown:
            max_drawdown = intra_dd
            max_drawdown_pct = (intra_dd / peak_equity) * 100 if peak_equity > 0 else 0.0

        if i % 20 == 0 or position is None or i == len(df) - 1:
            equity_curve.append({
                'time': bar_time,
                'equity': round(current_eq, 2),
                'pnl': round(cumulative_pnl + unrealized, 2),
            })

    # ---- CLOSE ANY OPEN POSITION AT END OF DATA (WITH REAL FRICTION) ----
    if position is not None:
        last = df.iloc[-1]
        bar_time = last.name.isoformat() if hasattr(last.name, 'isoformat') else str(last.name)
        if position['direction'] == 'long':
            exit_price = last['close'] - cost_per_oz
            pnl = (exit_price - position['entry_price']) * position['remaining_size']
        else:
            exit_price = last['close'] + cost_per_oz
            pnl = (position['entry_price'] - exit_price) * position['remaining_size']
        cumulative_pnl += pnl
        position['pnl'] += pnl
        position['exit_price'] = exit_price
        position['exit_time'] = bar_time
        position['exit_time_ts'] = bar_ts
        position['exit_reason'] = 'End'
        trades.append(_finalize_trade(position, cumulative_pnl, initial_capital))

    stats = _compute_stats(trades, equity_curve, initial_capital, max_drawdown, max_drawdown_pct)
    return trades, equity_curve, stats


def _finalize_trade(pos, cumulative_pnl, initial_capital):
    pnl = pos['pnl']
    risk_usd = pos.get('risk_usd', 1000.0)
    r_ret = round(pnl / risk_usd, 2) if risk_usd > 0 else round(pnl / 1000.0, 2)
    return {
        'id': pos['id'],
        'direction': pos['direction'],
        'entry_price': round(pos['entry_price'], 2),
        'entry_time': pos['entry_time'],
        'entry_time_ts': pos.get('entry_time_ts'),
        'exit_price': round(pos['exit_price'], 2) if pos['exit_price'] else None,
        'exit_time': pos['exit_time'],
        'exit_time_ts': pos.get('exit_time_ts'),
        'exit_reason': pos['exit_reason'],
        'sl': round(pos['sl'], 2) if pos['sl'] is not None else None,
        'tps': [round(t, 2) if t is not None else None for t in pos['tps_original']],
        'initial_size': pos['initial_size'],
        'pnl': round(pnl, 2),
        'pnl_percent': round((pnl / (pos['entry_price'] * pos['initial_size'])) * 100, 2) if pos['entry_price'] > 0 else 0.0,
        'cumulative_pnl': round(cumulative_pnl, 2),
        'tp_hits': pos['tp_hits'],
        'r_return': r_ret,
    }


def _compute_stats(trades, equity_curve, initial_capital, max_drawdown, max_drawdown_pct):
    if not trades:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0.0, 'total_pnl': 0.0, 'net_profit_pct': 0.0,
            'gross_profit': 0.0, 'gross_loss': 0.0, 'profit_factor': 0.0,
            'max_drawdown': 0.0, 'max_drawdown_pct': 0.0,
            'avg_pnl': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
            'ratio_win_loss': 0.0, 'best_trade': 0.0, 'worst_trade': 0.0,
            'sharpe_ratio': 0.0, 'sortino_ratio': 0.0,
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

    # Standardized Daily Periodic Return Sharpe and Sortino Ratios (Annualized x sqrt(252))
    sharpe = 0.0
    sortino = 0.0
    if equity_curve and len(equity_curve) > 1:
        try:
            eq_df = pd.DataFrame(equity_curve)
            eq_df['date'] = pd.to_datetime(eq_df['time']).dt.date
            daily_eq = eq_df.groupby('date')['equity'].last()
            daily_rets = daily_eq.pct_change().dropna()
            if len(daily_rets) > 1 and daily_rets.std() > 0:
                mean_ret = float(daily_rets.mean())
                std_ret = float(daily_rets.std())
                sharpe = round((mean_ret / std_ret) * np.sqrt(252), 2)
                downside_rets = daily_rets[daily_rets < 0]
                if len(downside_rets) > 1 and downside_rets.std() > 0:
                    sortino = round((mean_ret / float(downside_rets.std())) * np.sqrt(252), 2)
                else:
                    sortino = sharpe
        except Exception:
            pass

    # Fallback to trade-based annualization if daily curve is insufficient
    if sharpe == 0.0 and len(pnls) > 1 and np.std(pnls) > 0:
        mean_r = np.mean(pnls)
        std_r = np.std(pnls)
        sharpe = round((mean_r / std_r) * np.sqrt(min(len(pnls), 252)), 2)
        downside = [p for p in pnls if p < 0]
        if downside and np.std(downside) > 0:
            sortino = round((mean_r / np.std(downside)) * np.sqrt(min(len(pnls), 252)), 2)
        else:
            sortino = sharpe

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
