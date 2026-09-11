"""
Strategy Leaderboard & Persistence Engine.

Tracks, ranks, and stress-tests all generated and baseline algorithmic trading strategies.
Ranks strategies by:
  - Total Net Return in R (+R gained)
  - Monthly R performance breakdown & Count of High-Yield Months (>= +10.0R)
  - Max Drawdown in R and percentage
  - Profit Factor & Win Rate
  - Monte Carlo 95% Value at Risk (VaR) & Risk of Ruin
"""

import os
import re
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from monte_carlo import run_monte_carlo
from strategy_executor import execute_strategy as _execute_strategy

LEADERBOARD_FILE = Path(__file__).parent / 'data' / 'leaderboard.json'


def compute_monthly_r_breakdown(trades: List[Dict[str, Any]], lot_size: float = 100.0) -> Dict[str, float]:
    """
    Groups trades by month (e.g. 'Jan 2026', 'Mar 2026') and computes cumulative R gained.
    """
    if not trades:
        return {}

    records = []
    for t in trades:
        en_time = t.get('entry_time')
        if not en_time:
            continue
        try:
            dt = pd.to_datetime(en_time)
            month_key = dt.strftime('%b %Y')
            month_sort = dt.strftime('%Y-%m')
        except Exception:
            continue

        pnl = t.get('pnl', 0.0)
        if 'r_return' in t and t['r_return'] is not None:
            r_val = float(t['r_return'])
        else:
            entry = t.get('entry_price', 0.0)
            sl = t.get('sl', None)
            if sl is not None and entry > 0 and abs(entry - sl) > 0:
                risk_usd = abs(entry - sl) * lot_size
                r_val = pnl / risk_usd if risk_usd > 0 else (1.0 if pnl > 0 else -1.0)
            else:
                r_val = pnl / 1000.0

        records.append({'month': month_key, 'month_sort': month_sort, 'r': r_val})

    if not records:
        return {}

    df_trades = pd.DataFrame(records)
    monthly = df_trades.groupby(['month_sort', 'month'])['r'].sum().reset_index()
    monthly = monthly.sort_values('month_sort')

    res = {}
    for _, row in monthly.iterrows():
        res[row['month']] = round(float(row['r']), 1)
    return res


def compute_rank_score(total_r: float, months_ge_10r: int, max_dd_r: float, profit_factor: float, win_rate: float, total_trades: int = 100) -> float:
    """
    Composite Quantitative Alpha Score:
    Rewards total R-yield, consistency (months >= 10R), and high profit factor,
    while heavily penalizing large drawdowns in R.
    Applies a confidence penalty for strategies with < 30 trades (statistically unreliable).
    """
    pf = min(profit_factor, 5.0)
    score = (total_r * 1.0) + (months_ge_10r * 8.0) - (max_dd_r * 2.0) + (pf * 15.0) + (win_rate * 0.2)
    # Confidence penalty: strategies with < 30 trades get proportionally reduced scores
    if total_trades < 30:
        confidence = max(0.1, total_trades / 30.0)
        score *= confidence
    return round(score, 1)


def _get_champion_lss_code() -> str:
    """Returns the code for the Champion LSS strategy."""
    return """# =============================================================================
# CHAMPION LSS STRATEGY — Candle-Close Execution
# =============================================================================
# Parameters: SWING_LEN = 7, ATR_LEN = 4, ATR_MULT = 1.5, RR_RATIO = 1.22
# SL/TP computed relative to df['close'] (matches backtest.py fill price)
# =============================================================================
import numpy as np
import pandas as pd

def calculate_signals(df):
    df = df.copy()
    sw_h, sw_l = find_swings(df, swing_len=7)
    b_fvg_t, b_fvg_b, s_fvg_t, s_fvg_b = find_fvgs(df)
    m = session_mask(df, 'london_ny')
    a = atr(df, 14)
    
    # BSL / SSL Sweeps
    bsl_sweep = (df['high'] > sw_h) & (df['close'] < sw_h)
    ssl_sweep = (df['low'] < sw_l) & (df['close'] > sw_l)
    
    # Anti-bleed transition signals
    raw_bull = ssl_sweep & m
    raw_bear = bsl_sweep & m
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))
    
    # SL/TP relative to candle close (matches backtest.py entry fill price)
    df['sl_long'] = np.minimum(df['low'], sw_l) - (1.5 * a)
    risk_long = np.maximum(df['close'] - df['sl_long'], 2.50)
    df['tp1_long'] = df['close'] + (risk_long * 1.22)
    
    df['sl_short'] = np.maximum(df['high'], sw_h) + (1.5 * a)
    risk_short = np.maximum(df['sl_short'] - df['close'], 2.50)
    df['tp1_short'] = df['close'] - (risk_short * 1.22)
    return df
"""


def _get_default_seed_strategies(full_df=None, train_df=None) -> List[Dict[str, Any]]:
    """Seed leaderboard with Champion and Baseline strategies using REAL computed stats across full 6-month data."""
    seeds = []
    exec_df = full_df if full_df is not None and len(full_df) > 100 else train_df

    # Helper to compute a seed entry from real backtest
    def _build_seed(seed_id, name, concept, author, code):
        if not code or not code.strip():
            return None

        entry = {
            'id': seed_id,
            'name': name,
            'concept': concept,
            'author': author,
            'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'code': code,
        }

        if exec_df is not None:
            try:
                result = _execute_strategy(code, exec_df)
                if result.get('success'):
                    stats = result.get('stats', {})
                    trades = result.get('trades', [])
                    monthly_r = compute_monthly_r_breakdown(trades)
                    months_ge_10 = sum(1 for v in monthly_r.values() if v >= 10.0)
                    total_r = round(float(sum(monthly_r.values())), 1) if monthly_r else round(float(stats.get('total_pnl', 0.0) / 1000.0), 1)
                    max_dd_r = round(float(stats.get('max_drawdown', 0.0) / 1000.0), 1)

                    mc_results = run_monte_carlo(trades, num_simulations=1000)

                    entry.update({
                        'total_r': total_r,
                        'total_trades': stats.get('total_trades', len(trades)),
                        'winning_trades': stats.get('winning_trades', 0),
                        'losing_trades': stats.get('losing_trades', 0),
                        'win_rate': float(stats.get('win_rate', 0.0)),
                        'profit_factor': float(stats.get('profit_factor', 0.0)),
                        'total_pnl': float(stats.get('total_pnl', 0.0)),
                        'max_drawdown_r': max_dd_r,
                        'max_drawdown_pct': float(stats.get('max_drawdown_pct', 0.0)),
                        'sharpe_ratio': float(stats.get('sharpe_ratio', 0.0)),
                        'monthly_r': monthly_r,
                        'months_ge_10r': months_ge_10,
                        'monte_carlo': {
                            'expected_max_dd_r': mc_results.get('expected_max_dd_r'),
                            'var_95_max_dd_r': mc_results.get('var_95_max_dd_r'),
                            'risk_of_ruin_10r': mc_results.get('risk_of_ruin_10r'),
                            'risk_of_ruin_20r': mc_results.get('risk_of_ruin_20r'),
                            'probability_of_profit': mc_results.get('probability_of_profit'),
                            'median_final_r': mc_results.get('median_final_r'),
                        },
                        'rank_score': compute_rank_score(total_r, months_ge_10, max_dd_r, float(stats.get('profit_factor', 1.0)), float(stats.get('win_rate', 50.0)), int(stats.get('total_trades', 0))),
                        'data_split': 'full_6m',
                    })
                    print(f"  [Leaderboard Seed] {name}: {total_r:+.1f}R, {stats.get('total_trades')} trades, PF={stats.get('profit_factor')} (computed from real backtest)")
                    return entry
            except Exception as e:
                print(f"  [Leaderboard Seed] Warning: Failed to compute real stats for {name}: {e}")

        # Fallback: minimal entry if no data available
        entry.update({
            'total_r': 0.0, 'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0,
            'max_drawdown_r': 0.0, 'max_drawdown_pct': 0.0, 'sharpe_ratio': 0.0,
            'monthly_r': {}, 'months_ge_10r': 0,
            'monte_carlo': {}, 'rank_score': 0.0, 'data_split': 'full_6m',
        })
        return entry

    # Champion LSS Strategy
    champ_code = _get_champion_lss_code()
    champ = _build_seed(
        'champion_lss', 'Champion LSS Strategy (Candle-Close Execution)',
        'SSL/BSL Sweep + Anti-Bleed Transition + ATR Structural Stop',
        'Verified Dukascopy Champion', champ_code
    )
    if champ:
        seeds.append(champ)

    # Baseline Elvaris V2
    try:
        from default_strategy import DEFAULT_STRATEGY_CODE
        base = _build_seed(
            'elvaris_v2_baseline', 'Elvaris River Strategy V2 (Baseline)',
            'Dual Smooth Range Filter + Bollinger Squeeze (55, 0.2)',
            'Leo / TradingView Community', DEFAULT_STRATEGY_CODE
        )
        if base:
            seeds.append(base)
    except Exception:
        pass

    return seeds


def upgrade_leaderboard_to_full_6m(full_df) -> int:
    """
    Re-runs backtest across the full 6-month historical dataset for all
    existing strategies on the leaderboard, ensuring all stats (total_r, trades,
    win_rate, profit_factor, max_dd, monthly breakdown, monte carlo) reflect
    the complete 6-month period.
    """
    if full_df is None or len(full_df) < 100:
        return 0
    if not LEADERBOARD_FILE.exists():
        return 0

    try:
        with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f:
            items = json.load(f)
    except Exception:
        return 0

    if not isinstance(items, list) or len(items) == 0:
        return 0

    updated_count = 0
    for entry in items:
        if entry.get('data_split') == 'full_6m':
            continue
        code = entry.get('code', '')
        if not code or not code.strip():
            continue
        try:
            res = _execute_strategy(code, full_df)
            if res.get('success'):
                stats = res.get('stats', {})
                trades = res.get('trades', [])
                monthly_r = compute_monthly_r_breakdown(trades)
                months_ge_10 = sum(1 for v in monthly_r.values() if v >= 10.0)
                total_r = round(float(sum(monthly_r.values())), 1) if monthly_r else round(float(stats.get('total_pnl', 0.0) / 1000.0), 1)
                max_dd_r = round(float(stats.get('max_drawdown', 0.0) / 1000.0), 1)
                mc_results = run_monte_carlo(trades, num_simulations=1000)
                pf = float(stats.get('profit_factor', 1.0))
                win_rate = float(stats.get('win_rate', 50.0))
                rank_score = compute_rank_score(total_r, months_ge_10, max_dd_r, pf, win_rate, int(stats.get('total_trades', len(trades))))

                # Preserve old train_r if not set
                if 'train_r' not in entry and entry.get('total_r') is not None:
                    entry['train_r'] = entry.get('total_r')
                    entry['train_pf'] = entry.get('profit_factor')
                    entry['train_trades'] = entry.get('total_trades')

                entry.update({
                    'total_r': total_r,
                    'total_trades': stats.get('total_trades', len(trades)),
                    'winning_trades': stats.get('winning_trades', 0),
                    'losing_trades': stats.get('losing_trades', 0),
                    'win_rate': win_rate,
                    'profit_factor': pf,
                    'total_pnl': float(stats.get('total_pnl', 0.0)),
                    'max_drawdown_r': max_dd_r,
                    'max_drawdown_pct': float(stats.get('max_drawdown_pct', 0.0)),
                    'sharpe_ratio': float(stats.get('sharpe_ratio', 0.0)),
                    'monthly_r': monthly_r,
                    'months_ge_10r': months_ge_10,
                    'monte_carlo': {
                        'expected_max_dd_r': mc_results.get('expected_max_dd_r'),
                        'var_95_max_dd_r': mc_results.get('var_95_max_dd_r'),
                        'risk_of_ruin_10r': mc_results.get('risk_of_ruin_10r'),
                        'risk_of_ruin_20r': mc_results.get('risk_of_ruin_20r'),
                        'probability_of_profit': mc_results.get('probability_of_profit'),
                        'median_final_r': mc_results.get('median_final_r'),
                    },
                    'rank_score': rank_score,
                    'data_split': 'full_6m'
                })
                updated_count += 1
        except Exception as err:
            print(f"  [Upgrade Error] {entry.get('name')}: {err}")

    if updated_count > 0:
        save_leaderboard(items)
        print(f"  [Leaderboard] Successfully upgraded {updated_count} strategies to full 6-month backtest metrics!")
    return updated_count


def load_leaderboard(full_df=None, train_df=None) -> List[Dict[str, Any]]:
    """Loads leaderboard list from disk, ensuring seed champions exist and reflect full 6-month backtests."""
    LEADERBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_for_calc = full_df if full_df is not None and len(full_df) > 100 else train_df

    if not LEADERBOARD_FILE.exists():
        seeds = _get_default_seed_strategies(full_df=df_for_calc)
        save_leaderboard(seeds)
        return seeds

    try:
        with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0:
            # Upgrade any non-full_6m entries if data is provided
            if df_for_calc is not None and len(df_for_calc) > 100:
                has_old = any(item.get('data_split') != 'full_6m' for item in data)
                if has_old:
                    upgrade_leaderboard_to_full_6m(df_for_calc)
                    with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f2:
                        data = json.load(f2)
            # Sort by rank_score descending
            data.sort(key=lambda x: x.get('rank_score', 0.0), reverse=True)
            return data
    except Exception:
        pass

    seeds = _get_default_seed_strategies(full_df=df_for_calc)
    save_leaderboard(seeds)
    return seeds


def save_leaderboard(items: List[Dict[str, Any]]):
    """Persists leaderboard list to disk sorted by rank_score atomically."""
    LEADERBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    items.sort(key=lambda x: x.get('rank_score', 0.0), reverse=True)
    tmp_path = LEADERBOARD_FILE.with_suffix('.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2)
    tmp_path.replace(LEADERBOARD_FILE)


def add_strategy_to_leaderboard(name: str,
                                concept: str,
                                code: str,
                                stats: Dict[str, Any],
                                trades: List[Dict[str, Any]],
                                author: str = 'Autonomous AI Generator',
                                val_stats: Dict[str, Any] = None,
                                test_stats: Dict[str, Any] = None,
                                train_stats: Dict[str, Any] = None,
                                data_split: str = 'full_6m') -> Dict[str, Any]:
    """
    Evaluates a newly discovered strategy with Monte Carlo stress test and monthly R breakdown,
    and inserts it into the persistent ranked leaderboard.
    All primary stats reflect the full 6-month backtest.
    """
    current_board = load_leaderboard()

    # Deduplication Guard: Check if an identical strategy already exists in current_board
    def _norm_code(c: str) -> str:
        if not c:
            return ""
        no_doc = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '', c)
        no_comm = re.sub(r'#.*', '', no_doc)
        return re.sub(r'\s+', '', no_comm)

    norm_new = _norm_code(code)
    new_trades_count = stats.get('total_trades', len(trades))
    new_pnl = round(float(stats.get('total_pnl', 0.0)), 2)

    # 1. Run Monte Carlo simulation (1,000 iterations)
    mc_results = run_monte_carlo(trades, num_simulations=1000)

    # 2. Compute Monthly R breakdown
    monthly_r = compute_monthly_r_breakdown(trades)
    months_ge_10 = sum(1 for v in monthly_r.values() if v >= 10.0)

    # 3. Calculate total R-return
    total_r = round(float(sum(monthly_r.values())), 1) if monthly_r else round(float(stats.get('total_pnl', 0.0) / 1000.0), 1)

    max_dd_r = round(float(stats.get('max_drawdown', 0.0) / 1000.0), 1)
    if max_dd_r <= 0:
        max_dd_r = round(float(mc_results.get('expected_max_dd_r', 5.0)), 1)

    pf = float(stats.get('profit_factor', 1.0))
    win_rate = float(stats.get('win_rate', 50.0))

    rank_score = compute_rank_score(total_r, months_ge_10, max_dd_r, pf, win_rate, int(stats.get('total_trades', 0)))

    entry = {
        'id': f"strat_{int(datetime.utcnow().timestamp())}",
        'name': name,
        'concept': concept,
        'author': author,
        'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'code': code,
        'total_r': total_r,
        'total_trades': stats.get('total_trades', len(trades)),
        'winning_trades': stats.get('winning_trades', 0),
        'losing_trades': stats.get('losing_trades', 0),
        'win_rate': win_rate,
        'profit_factor': pf,
        'total_pnl': float(stats.get('total_pnl', 0.0)),
        'max_drawdown_r': max_dd_r,
        'max_drawdown_pct': float(stats.get('max_drawdown_pct', 0.0)),
        'sharpe_ratio': float(stats.get('sharpe_ratio', 0.0)),
        'monthly_r': monthly_r,
        'months_ge_10r': months_ge_10,
        'monte_carlo': {
            'expected_max_dd_r': mc_results.get('expected_max_dd_r'),
            'var_95_max_dd_r': mc_results.get('var_95_max_dd_r'),
            'risk_of_ruin_10r': mc_results.get('risk_of_ruin_10r'),
            'risk_of_ruin_20r': mc_results.get('risk_of_ruin_20r'),
            'probability_of_profit': mc_results.get('probability_of_profit'),
            'median_final_r': mc_results.get('median_final_r'),
        },
        'rank_score': rank_score,
        'data_split': data_split,
    }

    # Attach train, validation and test set stats if provided
    if train_stats:
        entry['train_r'] = round(float(train_stats.get('total_r', 0.0)), 1)
        entry['train_pf'] = float(train_stats.get('profit_factor', 0.0))
        entry['train_trades'] = int(train_stats.get('total_trades', 0))
    if val_stats:
        entry['val_r'] = round(float(val_stats.get('total_r', 0.0)), 1)
        entry['val_pf'] = float(val_stats.get('profit_factor', 0.0))
        entry['val_trades'] = int(val_stats.get('total_trades', 0))
    if test_stats:
        entry['test_r'] = round(float(test_stats.get('total_r', 0.0)), 1)
        entry['test_pf'] = float(test_stats.get('profit_factor', 0.0))
        entry['test_trades'] = int(test_stats.get('total_trades', 0))

    for idx, existing in enumerate(current_board):
        same_code = bool(norm_new and _norm_code(existing.get('code', '')) == norm_new)
        same_stats = (existing.get('total_trades') == new_trades_count and
                      abs(round(float(existing.get('total_pnl', 0.0)), 2) - new_pnl) < 0.05)
        if same_code:
            # Upgrade existing if new submission is full_6m
            if data_split == 'full_6m' and existing.get('data_split') != 'full_6m':
                existing.update({
                    'total_r': entry['total_r'],
                    'total_trades': entry['total_trades'],
                    'winning_trades': entry['winning_trades'],
                    'losing_trades': entry['losing_trades'],
                    'win_rate': entry['win_rate'],
                    'profit_factor': entry['profit_factor'],
                    'total_pnl': entry['total_pnl'],
                    'max_drawdown_r': entry['max_drawdown_r'],
                    'max_drawdown_pct': entry['max_drawdown_pct'],
                    'sharpe_ratio': entry['sharpe_ratio'],
                    'monthly_r': entry['monthly_r'],
                    'months_ge_10r': entry['months_ge_10r'],
                    'monte_carlo': entry['monte_carlo'],
                    'rank_score': entry['rank_score'],
                    'data_split': 'full_6m',
                })
                if 'train_r' in entry:
                    existing['train_r'] = entry['train_r']
                    existing['train_pf'] = entry['train_pf']
                    existing['train_trades'] = entry['train_trades']
                if 'val_r' in entry:
                    existing['val_r'] = entry['val_r']
                    existing['val_pf'] = entry['val_pf']
                    existing['val_trades'] = entry['val_trades']
                if 'test_r' in entry:
                    existing['test_r'] = entry['test_r']
                    existing['test_pf'] = entry['test_pf']
                    existing['test_trades'] = entry['test_trades']
                save_leaderboard(current_board)
                ranked = load_leaderboard()
                rank = next((i + 1 for i, item in enumerate(ranked) if item['id'] == existing['id']), len(ranked))
                existing['rank'] = rank
                return existing
            existing['rank'] = idx + 1
            return existing
        elif same_stats:
            existing['rank'] = idx + 1
            return existing

    # Add to list and re-sort
    current_board.append(entry)
    save_leaderboard(current_board)

    # Find position of this entry
    ranked = load_leaderboard()
    rank = next((i + 1 for i, item in enumerate(ranked) if item['id'] == entry['id']), len(ranked))
    entry['rank'] = rank
    return entry


def get_strategy_by_id(strat_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves strategy record by ID."""
    items = load_leaderboard()
    for item in items:
        if item.get('id') == strat_id:
            return item
    return None
