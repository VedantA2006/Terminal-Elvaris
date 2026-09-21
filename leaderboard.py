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

asset_name = os.environ.get('ASSET', 'XAUUSD').upper()
file_name = 'leaderboard.json' if asset_name == 'XAUUSD' else f'leaderboard_{asset_name}.json'
LEADERBOARD_FILE = Path(__file__).parent / 'data' / file_name


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
            # Enforce 2026-only trades
            if dt.year < 2026:
                continue
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
    score = (total_r * 2.5) + (months_ge_10r * 10.0) - (max_dd_r * 2.0) + (pf * 8.0) + (win_rate * 0.2)
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
                        'data_split': 'mt5_ecn',
                    })
                    print(f"  [Leaderboard Seed] {name}: {total_r:+.1f}R, {stats.get('total_trades')} trades, PF={stats.get('profit_factor')} (computed from MT5 broker backtest)")
                    return entry
            except Exception as e:
                print(f"  [Leaderboard Seed] Warning: Failed to compute real stats for {name}: {e}")

        # Fallback: minimal entry if no data available
        entry.update({
            'total_r': 0.0, 'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0.0, 'profit_factor': 0.0, 'total_pnl': 0.0,
            'max_drawdown_r': 0.0, 'max_drawdown_pct': 0.0, 'sharpe_ratio': 0.0,
            'monthly_r': {}, 'months_ge_10r': 0,
            'monte_carlo': {}, 'rank_score': 0.0, 'data_split': 'mt5_ecn',
        })
        return entry

    # Champion LSS Strategy
    champ_code = _get_champion_lss_code()
    champ = _build_seed(
        'champion_lss', 'Champion LSS Strategy (Candle-Close Execution)',
        'SSL/BSL Sweep + Anti-Bleed Transition + ATR Structural Stop',
        'Verified MT5 ECN Champion', champ_code
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


def upgrade_leaderboard_to_mt5_data(full_df, train_df=None, val_df=None, test_df=None, force=False) -> int:
    """
    Re-runs backtest across the full MetaTrader 5 ECN broker dataset (100,000 candles) for all
    existing strategies on the leaderboard, ensuring all stats (total_r, trades, win_rate,
    profit_factor, max_dd, monthly breakdown across 17 months, monte carlo, and train/val/test splits)
    reflect the MT5 dataset.
    """
    if full_df is None or len(full_df) < 100:
        return 0
    # Enforce strictly 2026-only data for leaderboard backtests
    if isinstance(full_df.index, pd.DatetimeIndex):
        full_df = full_df[full_df.index >= '2026-01-01'].copy()
    if not LEADERBOARD_FILE.exists():
        return 0

    try:
        with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f:
            items = json.load(f)
    except Exception:
        return 0

    if not isinstance(items, list) or len(items) == 0:
        return 0

    if train_df is None or val_df is None:
        from data_split import split_data
        train_df, val_df, test_df = split_data(full_df)

    train_end = train_df.index[-1]
    val_end = val_df.index[-1]

    updated_count = 0
    for entry in items:
        if not force and entry.get('data_split') in ('mt5_ecn', 'mt5_ecn_2026', 'equityedge_mt5_2026'):
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
                mc_results = run_monte_carlo(trades, num_simulations=400)
                pf = float(stats.get('profit_factor', 1.0))
                win_rate = float(stats.get('win_rate', 50.0))
                total_trades = int(stats.get('total_trades', len(trades)))
                rank_score = compute_rank_score(total_r, months_ge_10, max_dd_r, pf, win_rate, total_trades)

                tr_trades = [t for t in trades if pd.to_datetime(t['entry_time']) <= train_end]
                v_trades = [t for t in trades if train_end < pd.to_datetime(t['entry_time']) <= val_end]
                te_trades = [t for t in trades if pd.to_datetime(t['entry_time']) > val_end]

                def _calc_split(sub_t):
                    if not sub_t:
                        return 0.0, 0.0, 0, 0.0
                    r_val = round(float(sum(compute_monthly_r_breakdown(sub_t).values())), 1)
                    wins = len([t for t in sub_t if t.get('pnl', 0) > 0])
                    wr = round(wins / len(sub_t) * 100.0, 1)
                    gw = sum(t.get('pnl', 0) for t in sub_t if t.get('pnl', 0) > 0)
                    gl = abs(sum(t.get('pnl', 0) for t in sub_t if t.get('pnl', 0) < 0))
                    sub_pf = round(float(gw / max(0.01, gl)), 2)
                    return r_val, sub_pf, len(sub_t), wr

                tr_r, tr_pf, tr_tr, tr_wr = _calc_split(tr_trades)
                v_r, v_pf, v_tr, v_wr = _calc_split(v_trades)
                te_r, te_pf, te_tr, te_wr = _calc_split(te_trades)

                entry.update({
                    'total_r': total_r,
                    'total_trades': total_trades,
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
                    'data_split': 'mt5_ecn',
                    'train_r': tr_r,
                    'train_pf': tr_pf,
                    'train_trades': tr_tr,
                    'train_win_rate': tr_wr,
                    'val_r': v_r,
                    'val_pf': v_pf,
                    'val_trades': v_tr,
                    'test_r': te_r,
                    'test_pf': te_pf,
                    'test_trades': te_tr,
                })
                updated_count += 1
        except Exception as err:
            print(f"  [Upgrade Error] {entry.get('name')}: {err}")

    if updated_count > 0:
        save_leaderboard(items)
        print(f"  [Leaderboard] Successfully upgraded {updated_count} strategies to MT5 broker metrics!")
    return updated_count

# Backwards compatibility alias
upgrade_leaderboard_to_full_6m = upgrade_leaderboard_to_mt5_data


def get_date_slice_for_range(df: pd.DataFrame, range_key: str = '2026_ytd') -> pd.DataFrame:
    """Returns the subset of DataFrame matching the requested backtest range."""
    if df is None or len(df) == 0:
        return df
    times = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('dt', df.get('datetime', df.index)))
    latest_dt = times.max()
    
    if range_key == 'full':
        return df.copy()
    elif range_key == '1y':
        start_dt = latest_dt - pd.Timedelta(days=365)
        return df[times >= start_dt].copy()
    elif range_key == '6m':
        start_dt = latest_dt - pd.Timedelta(days=180)
        return df[times >= start_dt].copy()
    elif range_key == '3m':
        start_dt = latest_dt - pd.Timedelta(days=90)
        return df[times >= start_dt].copy()
    else:
        # Default: 2026 YTD
        return df[times >= '2026-01-01'].copy()


def load_leaderboard(full_df=None, train_df=None, range_key: str = '2026_ytd') -> List[Dict[str, Any]]:
    """
    Loads leaderboard list from disk for the specified backtest range.
    Uses precomputed range caches (leaderboard_full.json, leaderboard_1y.json, etc.)
    for instantaneous (<50ms) retrieval, falling back dynamically if not precomputed.
    """
    LEADERBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Check for range-specific precomputed cache
    if range_key and range_key != '2026_ytd':
        range_file = LEADERBOARD_FILE.parent / f"leaderboard_{range_key}.json"
        if range_file.exists():
            try:
                with open(range_file, 'r', encoding='utf-8') as f:
                    range_data = json.load(f)
                if isinstance(range_data, list) and len(range_data) > 0:
                    range_data.sort(key=lambda x: x.get('total_r', 0.0), reverse=True)
                    for i, item in enumerate(range_data):
                        item['rank'] = i + 1
                    return range_data
            except Exception:
                pass

    df_for_calc = full_df if full_df is not None and len(full_df) > 100 else train_df
    if df_for_calc is not None and isinstance(df_for_calc.index, pd.DatetimeIndex):
        df_for_calc = get_date_slice_for_range(df_for_calc, range_key)

    if not LEADERBOARD_FILE.exists():
        if asset_name != 'XAUUSD':
            return []
        seeds = _get_default_seed_strategies(full_df=df_for_calc)
        save_leaderboard(seeds)
        return seeds

    try:
        with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            if len(data) == 0:
                if asset_name != 'XAUUSD':
                    return []
                seeds = _get_default_seed_strategies(full_df=df_for_calc)
                save_leaderboard(seeds)
                return seeds
            # Upgrade any non-mt5_ecn_2026 entries if data is provided and range is 2026
            if df_for_calc is not None and len(df_for_calc) > 100 and range_key == '2026_ytd':
                has_old = any(item.get('data_split') != 'mt5_ecn_2026' for item in data)
                if has_old:
                    upgrade_leaderboard_to_mt5_data(df_for_calc)
                    with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f2:
                        data = json.load(f2)
            # Sort strictly by total_r descending
            data.sort(key=lambda x: x.get('total_r', 0.0), reverse=True)
            for i, item in enumerate(data):
                item['rank'] = i + 1
            return data
    except Exception:
        pass

    if asset_name != 'XAUUSD':
        return []
    seeds = _get_default_seed_strategies(full_df=df_for_calc)
    save_leaderboard(seeds)
    return seeds


def get_research_candidates(train_df=None, min_train_trades: int = 10, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Returns leaderboard entries ranked by TRAIN-SPLIT performance only, for use
    by the autonomous research loop (few-shot LLM context, genetic breeding
    parent selection). Deliberately excludes validation/test performance so
    strategy generation never sees out-of-sample results — this prevents the
    held-out data from leaking back into the discovery process.

    Entries without a recorded train_r (legacy/seed entries added before
    train/val/test tracking existed) are excluded, since their reported
    performance can't be verified as leak-free.
    """
    all_entries = load_leaderboard(train_df)
    candidates = [
        e for e in all_entries
        if e.get('train_r') is not None
        and e.get('train_trades', 0) >= min_train_trades
        and e.get('code')
    ]
    candidates.sort(key=lambda e: e.get('train_r', -999), reverse=True)
    return candidates[:limit]


def save_leaderboard(items: List[Dict[str, Any]]):
    """Persists leaderboard list to disk sorted by rank_score atomically with exact integer ranks."""
    LEADERBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    items.sort(key=lambda x: x.get('rank_score', 0.0), reverse=True)
    for i, item in enumerate(items):
        item['rank'] = i + 1
    tmp_path = LEADERBOARD_FILE.with_suffix('.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2)
    tmp_path.replace(LEADERBOARD_FILE)


def _backfill_validation_metadata(existing: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    """
    Copies train/val/test provenance from a newly-evaluated `entry` into an
    already-registered `existing` leaderboard record, for any split where
    `existing` doesn't already have it. Never overwrites data that's already
    present. Returns True if anything changed (caller should persist).
    """
    changed = False
    for prefix in ('train', 'val', 'test'):
        r_key, pf_key, tr_key, wr_key = f'{prefix}_r', f'{prefix}_pf', f'{prefix}_trades', f'{prefix}_win_rate'
        if r_key in entry and r_key not in existing:
            existing[r_key] = entry[r_key]
            if pf_key in entry:
                existing[pf_key] = entry[pf_key]
            if tr_key in entry:
                existing[tr_key] = entry[tr_key]
            if wr_key in entry:
                existing[wr_key] = entry[wr_key]
            changed = True
    return changed


def add_strategy_to_leaderboard(name: str,
                                concept: str,
                                code: str,
                                stats: Dict[str, Any],
                                trades: List[Dict[str, Any]],
                                author: str = 'Autonomous AI Generator',
                                val_stats: Dict[str, Any] = None,
                                test_stats: Dict[str, Any] = None,
                                train_stats: Dict[str, Any] = None,
                                data_split: str = 'mt5_ecn_2026',
                                instrument: str = 'XAUUSD',
                                strategy_id: str = None) -> Dict[str, Any]:
    """
    Evaluates a newly discovered strategy with Monte Carlo stress test and monthly R breakdown,
    and inserts it into the persistent ranked leaderboard.
    All primary stats reflect the MT5 broker dataset.
    """
    current_board = load_leaderboard()

    import uuid
    if strategy_id is None:
        strategy_id = uuid.uuid4().hex[:8]

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
        'id': strategy_id,
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
        entry['train_win_rate'] = float(train_stats.get('win_rate', 0.0))
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
        same_strat_identity = (existing.get('name') == name and existing.get('author') == author)
        same_stats = (same_strat_identity and existing.get('total_trades') == new_trades_count and
                      abs(round(float(existing.get('total_pnl', 0.0)), 2) - new_pnl) < 0.05)

        if same_code or same_stats:
            backfilled = _backfill_validation_metadata(existing, entry)

            upgraded_to_full = (
                same_code and data_split == 'mt5_ecn' and existing.get('data_split') != 'mt5_ecn'
            )
            if upgraded_to_full:
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
                    'data_split': 'mt5_ecn',
                })

            if backfilled or upgraded_to_full:
                save_leaderboard(current_board)
                ranked = load_leaderboard()
                rank = next((i + 1 for i, item in enumerate(ranked) if item.get('id') == existing.get('id')), len(ranked))
                existing['rank'] = rank
                return existing

            existing['rank'] = idx + 1
            return existing

    # Add to list and re-sort
    current_board.append(entry)
    save_leaderboard(current_board)

    # Find position of this entry
    ranked = load_leaderboard()
    rank = next((i + 1 for i, item in enumerate(ranked) if item.get('id') == entry.get('id')), len(ranked))
    entry['rank'] = rank
    return entry


def get_strategy_by_id(strat_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves strategy record by ID."""
    items = load_leaderboard()
    for item in items:
        if item.get('id') == strat_id:
            return item
    return None
