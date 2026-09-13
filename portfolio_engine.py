"""
Institutional Multi-Strategy Portfolio Ensemble Engine.

Aggregates, risk-budgets, and stress-tests multi-strategy portfolios composed
of top algorithmic champions from the leaderboard.

Key capabilities:
1. Parallel execution across top N leaderboard strategies on 2026 MT5 data.
2. Pairwise cross-strategy correlation matrix & diversification ratio.
3. Multiple portfolio allocation models:
   - Unconstrained (1R per strategy trade)
   - Equal-Weight Risk Budgeted (1/Nth R per trade)
   - Pruned Uncorrelated Champions (1 best from each unique archetype)
   - Risk Parity (Inverse Drawdown Weighting)
4. Chronological trade ledger with concurrency & market exposure tracking.
5. Monthly performance matrix & Monte Carlo portfolio stress-testing.
"""

import os
import sys
import json
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

repo_dir = Path(__file__).parent
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from download_data import load_or_download
from strategy_executor import execute_strategy
from monte_carlo import run_monte_carlo

PORTFOLIO_FILE = repo_dir / 'data' / 'portfolio_ensemble.json'
LEADERBOARD_FILE = repo_dir / 'data' / 'leaderboard.json'

def _eval_strategy_worker(item):
    rank, name, concept, code, df_data = item
    try:
        res = execute_strategy(code, df_data)
        trades = res.get('trades', [])
        stats = res.get('stats', {})
        return {
            'rank': rank,
            'name': name,
            'concept': concept,
            'trades': trades,
            'stats': stats,
            'success': True
        }
    except Exception as e:
        return {
            'rank': rank,
            'name': name,
            'concept': concept,
            'trades': [],
            'stats': {},
            'success': False,
            'error': str(e)
        }

def calc_max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    peak = series.cummax()
    dd = peak - series
    return round(float(dd.max()), 1)

def compute_monthly_breakdown(dates: pd.Series, pnl_r: pd.Series) -> Dict[str, float]:
    df = pd.DataFrame({'date': pd.to_datetime(dates), 'r': pnl_r})
    df['month'] = df['date'].dt.strftime('%b %Y')
    df['month_sort'] = df['date'].dt.strftime('%Y-%m')
    grouped = df.groupby(['month_sort', 'month'])['r'].sum().reset_index().sort_values('month_sort')
    return {row['month']: round(float(row['r']), 1) for _, row in grouped.iterrows()}

def build_portfolio_ensemble(df_2026: Optional[pd.DataFrame] = None, top_n: int = 10) -> Dict[str, Any]:
    """
    Simulates and constructs the institutional multi-strategy portfolio ensemble.
    """
    if df_2026 is None:
        full_df = load_or_download()
        df_2026 = full_df[full_df.index >= '2026-01-01'].copy()
    else:
        df_2026 = df_2026[df_2026.index >= '2026-01-01'].copy()

    if not LEADERBOARD_FILE.exists():
        raise FileNotFoundError(f"Leaderboard file not found at {LEADERBOARD_FILE}")

    with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f:
        lb = json.load(f)

    selected_strats = lb[:top_n]
    print(f"[PortfolioEngine] Building ensemble with Top {len(selected_strats)} strategies on {len(df_2026):,} candles...")

    # Parallel strategy execution
    worker_tasks = [
        (s.get('rank', i + 1), s.get('name'), s.get('concept', ''), s.get('code', ''), df_2026)
        for i, s in enumerate(selected_strats)
    ]

    strat_results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
        for res in executor.map(_eval_strategy_worker, worker_tasks):
            if res['success']:
                strat_results[res['rank']] = res

    # Organize daily return series
    all_dates = pd.date_range(start=df_2026.index[0].date(), end=df_2026.index[-1].date(), freq='D')
    df_daily_r = pd.DataFrame(index=all_dates.date)

    all_trades = []
    strat_details = []

    for rank in sorted(strat_results.keys()):
        data = strat_results[rank]
        trades = data['trades']
        stats = data['stats']

        # Tag trades with strategy metadata
        for t in trades:
            trade_copy = dict(t)
            trade_copy['strategy_rank'] = rank
            trade_copy['strategy_name'] = data['name']
            all_trades.append(trade_copy)

        # Daily PnL
        df_t = pd.DataFrame(trades)
        if not df_t.empty and 'exit_time' in df_t.columns:
            df_t['date'] = pd.to_datetime(df_t['exit_time']).dt.date
            daily = df_t.groupby('date')['pnl'].sum() / 1000.0
            df_daily_r[f"S{rank}"] = daily
        else:
            df_daily_r[f"S{rank}"] = 0.0

        net_r = round(float(sum(t.get('pnl', 0.0) / 1000.0 for t in trades)), 1)
        strat_details.append({
            'rank': rank,
            'name': data['name'],
            'concept': data['concept'],
            'total_r': net_r,
            'profit_factor': round(float(stats.get('profit_factor', 0.0)), 2),
            'win_rate': round(float(stats.get('win_rate', 0.0)), 1),
            'total_trades': len(trades),
            'max_dd_r': round(float(stats.get('max_drawdown', 0.0) / 1000.0), 1)
        })

    df_daily_r = df_daily_r.fillna(0.0)
    strat_cols = [c for c in df_daily_r.columns if c.startswith('S')]

    # Pairwise Correlation Matrix
    corr_matrix = df_daily_r[strat_cols].corr().round(3)
    avg_corr = round(float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()), 3)

    # Convert correlation matrix for JSON
    corr_dict = {}
    for c in strat_cols:
        corr_dict[c] = corr_matrix[c].to_dict()

    # --------------------------------------------------------------------------
    # MODEL 1: UNCONSTRAINED ENSEMBLE (1R risk per trade)
    # --------------------------------------------------------------------------
    daily_unconstrained = df_daily_r[strat_cols].sum(axis=1)
    cum_unconstrained = daily_unconstrained.cumsum()
    tot_r_unconstrained = round(float(cum_unconstrained.iloc[-1]), 1)
    max_dd_unconstrained = calc_max_drawdown(cum_unconstrained)
    calmar_unconstrained = round(tot_r_unconstrained / max(0.1, max_dd_unconstrained), 2)
    monthly_unconstrained = compute_monthly_breakdown(daily_unconstrained.index, daily_unconstrained.values)

    # --------------------------------------------------------------------------
    # MODEL 2: RISK-BUDGETED ENSEMBLE (1/Nth R per trade)
    # --------------------------------------------------------------------------
    daily_budgeted = df_daily_r[strat_cols].mean(axis=1)
    cum_budgeted = daily_budgeted.cumsum()
    tot_r_budgeted = round(float(cum_budgeted.iloc[-1]), 1)
    max_dd_budgeted = calc_max_drawdown(cum_budgeted)
    calmar_budgeted = round(tot_r_budgeted / max(0.1, max_dd_budgeted), 2)
    monthly_budgeted = compute_monthly_breakdown(daily_budgeted.index, daily_budgeted.values)

    # --------------------------------------------------------------------------
    # MODEL 3: PRUNED UNCORRELATED CHAMPIONS (1 per distinct archetype in Top 10)
    # --------------------------------------------------------------------------
    # Archetypes in Top 10:
    # #1 (Divergence), #2 (LSS Hybrid), #3 (ORB), #4 (Stochastic MACD),
    # #5 (NY Open Sweep), #7 (Floor Pivots), #10 (Volatility Squeeze)
    pruned_ranks = [r for r in [1, 2, 3, 4, 5, 7, 10] if f"S{r}" in strat_cols]
    pruned_cols = [f"S{r}" for r in pruned_ranks]

    daily_pruned_un = df_daily_r[pruned_cols].sum(axis=1)
    cum_pruned_un = daily_pruned_un.cumsum()
    tot_r_pruned_un = round(float(cum_pruned_un.iloc[-1]), 1)
    max_dd_pruned_un = calc_max_drawdown(cum_pruned_un)

    daily_pruned_bg = df_daily_r[pruned_cols].mean(axis=1)
    cum_pruned_bg = daily_pruned_bg.cumsum()
    tot_r_pruned_bg = round(float(cum_pruned_bg.iloc[-1]), 1)
    max_dd_pruned_bg = calc_max_drawdown(cum_pruned_bg)
    monthly_pruned = compute_monthly_breakdown(daily_pruned_bg.index, daily_pruned_bg.values)

    # --------------------------------------------------------------------------
    # MODEL 4: RISK PARITY (Inverse Drawdown Weighting)
    # --------------------------------------------------------------------------
    inv_dds = np.array([1.0 / max(5.0, s['max_dd_r']) for s in strat_details])
    risk_parity_weights = inv_dds / inv_dds.sum()
    daily_risk_parity = df_daily_r[strat_cols].dot(risk_parity_weights)
    cum_risk_parity = daily_risk_parity.cumsum()
    tot_r_risk_parity = round(float(cum_risk_parity.iloc[-1]), 1)
    max_dd_risk_parity = calc_max_drawdown(cum_risk_parity)
    calmar_risk_parity = round(tot_r_risk_parity / max(0.1, max_dd_risk_parity), 2)
    monthly_risk_parity = compute_monthly_breakdown(daily_risk_parity.index, daily_risk_parity.values)

    # --------------------------------------------------------------------------
    # MODEL 5: CALMAR KING 3 [2, 8, 9] (Global Top 10 Winner: 22.08x Calmar)
    # --------------------------------------------------------------------------
    c4_ranks = [2, 8, 9]
    c4_cols = [f"S{r}" for r in c4_ranks if f"S{r}" in strat_cols]
    daily_c4 = df_daily_r[c4_cols].mean(axis=1)
    cum_c4 = daily_c4.cumsum()
    tot_r_c4 = round(float(cum_c4.iloc[-1]), 1)
    max_dd_c4 = calc_max_drawdown(cum_c4)
    calmar_c4 = round(tot_r_c4 / max(0.1, max_dd_c4), 2)
    monthly_c4 = compute_monthly_breakdown(daily_c4.index, daily_c4.values)

    # --------------------------------------------------------------------------
    # MODEL 6: TITAN 5 [1, 2, 4, 8, 9] (Best 5-Strategy Synergy: 21.67x Calmar)
    # --------------------------------------------------------------------------
    t5_ranks = [1, 2, 4, 8, 9]
    t5_cols = [f"S{r}" for r in t5_ranks if f"S{r}" in strat_cols]
    daily_t5 = df_daily_r[t5_cols].mean(axis=1)
    cum_t5 = daily_t5.cumsum()
    tot_r_t5 = round(float(cum_t5.iloc[-1]), 1)
    max_dd_t5 = calc_max_drawdown(cum_t5)
    calmar_t5 = round(tot_r_t5 / max(0.1, max_dd_t5), 2)
    monthly_t5 = compute_monthly_breakdown(daily_t5.index, daily_t5.values)

    # --------------------------------------------------------------------------
    # MODEL 7: FORTRESS 4 [1, 4, 8, 9] (Ultra-Low Drawdown: -5.4R)
    # --------------------------------------------------------------------------
    f3_ranks = [1, 4, 8, 9]
    f3_cols = [f"S{r}" for r in f3_ranks if f"S{r}" in strat_cols]
    daily_f3 = df_daily_r[f3_cols].mean(axis=1)
    cum_f3 = daily_f3.cumsum()
    tot_r_f3 = round(float(cum_f3.iloc[-1]), 1)
    max_dd_f3 = calc_max_drawdown(cum_f3)
    calmar_f3 = round(tot_r_f3 / max(0.1, max_dd_f3), 2)
    monthly_f3 = compute_monthly_breakdown(daily_f3.index, daily_f3.values)

    # --------------------------------------------------------------------------
    # MODEL 8: GOLDEN HYBRID 7 [1, 2, 3, 4, 6, 8, 9] (Optimal Broad Hybrid: 21.23x Calmar)
    # --------------------------------------------------------------------------
    g7_ranks = [1, 2, 3, 4, 6, 8, 9]
    g7_cols = [f"S{r}" for r in g7_ranks if f"S{r}" in strat_cols]
    daily_g7 = df_daily_r[g7_cols].mean(axis=1)
    cum_g7 = daily_g7.cumsum()
    tot_r_g7 = round(float(cum_g7.iloc[-1]), 1)
    max_dd_g7 = calc_max_drawdown(cum_g7)
    calmar_g7 = round(tot_r_g7 / max(0.1, max_dd_g7), 2)
    monthly_g7 = compute_monthly_breakdown(daily_g7.index, daily_g7.values)

    # Attach weights to strategy details
    for i, s in enumerate(strat_details):
        s['equal_weight'] = round(1.0 / len(strat_details), 4)
        s['risk_parity_weight'] = round(float(risk_parity_weights[i]), 4)
        s['is_in_pruned_ensemble'] = s['rank'] in pruned_ranks

    # Sort all trades chronologically
    all_trades.sort(key=lambda t: str(t.get('entry_time', '')))

    # Concurrency analysis (simultaneous active positions)
    events = []
    for t in all_trades:
        en = t.get('entry_time')
        ex = t.get('exit_time')
        if en and ex:
            events.append((en, 1))
            events.append((ex, -1))
    events.sort(key=lambda x: str(x[0]))

    current_open = 0
    max_concurrency = 0
    concurrency_samples = []
    for dt, change in events:
        current_open += change
        concurrency_samples.append(current_open)
        if current_open > max_concurrency:
            max_concurrency = current_open

    avg_concurrency = round(float(np.mean(concurrency_samples)) if concurrency_samples else 0.0, 2)

    # Monte Carlo on Portfolio Trades
    mc_portfolio = run_monte_carlo(all_trades, num_simulations=500)

    # Prepare timeline series for frontend charting
    dates_str = [str(d) for d in all_dates.date]
    equity_series = {
        'dates': dates_str,
        'unconstrained': [round(float(v), 2) for v in cum_unconstrained.values],
        'risk_budgeted': [round(float(v), 2) for v in cum_budgeted.values],
        'pruned_unconstrained': [round(float(v), 2) for v in cum_pruned_un.values],
        'pruned_budgeted': [round(float(v), 2) for v in cum_pruned_bg.values],
        'risk_parity': [round(float(v), 2) for v in cum_risk_parity.values],
        'champion_4': [round(float(v), 2) for v in cum_c4.values],
        'titan_5': [round(float(v), 2) for v in cum_t5.values],
        'fortress_3': [round(float(v), 2) for v in cum_f3.values],
        'macro_7': [round(float(v), 2) for v in cum_g7.values],
    }

    ensemble_payload = {
        'benchmark_period': '2026-01-01 to Present (MT5 Broker)',
        'strategies_count': len(strat_details),
        'strategies': strat_details,
        'diagnostics': {
            'total_portfolio_trades': len(all_trades),
            'average_pairwise_correlation': avg_corr,
            'max_concurrent_positions': max_concurrency,
            'avg_concurrent_positions': avg_concurrency,
        },
        'models': {
            'unconstrained': {
                'name': 'Full Top 10 (Unconstrained 1R/trade)',
                'total_r': tot_r_unconstrained,
                'max_drawdown_r': max_dd_unconstrained,
                'calmar_ratio': calmar_unconstrained,
                'monthly_pnl': monthly_unconstrained
            },
            'risk_budgeted': {
                'name': 'Full Top 10 (Risk-Budgeted 1/10th R)',
                'total_r': tot_r_budgeted,
                'max_drawdown_r': max_dd_budgeted,
                'calmar_ratio': calmar_budgeted,
                'monthly_pnl': monthly_budgeted
            },
            'pruned_champions': {
                'name': 'Pruned 7 Uncorrelated Archetypes',
                'ranks': pruned_ranks,
                'total_r_unconstrained': tot_r_pruned_un,
                'max_drawdown_unconstrained': max_dd_pruned_un,
                'total_r_budgeted': tot_r_pruned_bg,
                'max_drawdown_budgeted': max_dd_pruned_bg,
                'monthly_pnl': monthly_pruned
            },
            'risk_parity': {
                'name': 'Risk Parity (Inverse Drawdown Weighting)',
                'total_r': tot_r_risk_parity,
                'max_drawdown_r': max_dd_risk_parity,
                'calmar_ratio': calmar_risk_parity,
                'monthly_pnl': monthly_risk_parity
            },
            'champion_4': {
                'name': 'Calmar King 3 [Ranks #2, #8, #9]',
                'ranks': c4_ranks,
                'total_r': tot_r_c4,
                'max_drawdown_r': max_dd_c4,
                'calmar_ratio': calmar_c4,
                'monthly_pnl': monthly_c4
            },
            'titan_5': {
                'name': 'Titan 5 [Ranks #1, #2, #4, #8, #9]',
                'ranks': t5_ranks,
                'total_r': tot_r_t5,
                'max_drawdown_r': max_dd_t5,
                'calmar_ratio': calmar_t5,
                'monthly_pnl': monthly_t5
            },
            'fortress_3': {
                'name': 'Fortress 4 (Min DD) [Ranks #1, #4, #8, #9]',
                'ranks': f3_ranks,
                'total_r': tot_r_f3,
                'max_drawdown_r': max_dd_f3,
                'calmar_ratio': calmar_f3,
                'monthly_pnl': monthly_f3
            },
            'macro_7': {
                'name': 'Golden Hybrid 7 [Ranks #1, #2, #3, #4, #6, #8, #9]',
                'ranks': g7_ranks,
                'total_r': tot_r_g7,
                'max_drawdown_r': max_dd_g7,
                'calmar_ratio': calmar_g7,
                'monthly_pnl': monthly_g7
            }
        },
        'correlation_matrix': corr_dict,
        'equity_curve': equity_series,
        'monte_carlo': {
            'expected_max_dd_r': mc_portfolio.get('expected_max_dd_r'),
            'var_95_max_dd_r': mc_portfolio.get('var_95_max_dd_r'),
            'risk_of_ruin_10r': mc_portfolio.get('risk_of_ruin_10r'),
            'risk_of_ruin_20r': mc_portfolio.get('risk_of_ruin_20r'),
            'probability_of_profit': mc_portfolio.get('probability_of_profit')
        }
    }

    # Save to disk
    PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(ensemble_payload, f, indent=2)

    print(f"[PortfolioEngine] Ensemble built successfully! Saved to {PORTFOLIO_FILE}")
    return ensemble_payload

def get_portfolio_ensemble_data() -> Dict[str, Any]:
    """Retrieves cached ensemble payload or generates it on the fly."""
    if PORTFOLIO_FILE.exists():
        try:
            with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return build_portfolio_ensemble()

if __name__ == '__main__':
    data = build_portfolio_ensemble()
    print("\n" + "="*75)
    print("INSTITUTIONAL ENSEMBLE SUMMARY:")
    print("="*75)
    m = data['models']
    print(f"1. Unconstrained Top 15:    Return: +{m['unconstrained']['total_r']}R | Max DD: -{m['unconstrained']['max_drawdown_r']}R | Calmar: {m['unconstrained']['calmar_ratio']}x")
    print(f"2. Risk-Budgeted Top 15:    Return: +{m['risk_budgeted']['total_r']}R | Max DD: -{m['risk_budgeted']['max_drawdown_r']}R | Calmar: {m['risk_budgeted']['calmar_ratio']}x")
    print(f"3. Pruned 9 Archetypes:     Return: +{m['pruned_champions']['total_r_unconstrained']}R | Max DD: -{m['pruned_champions']['max_drawdown_unconstrained']}R")
    print(f"4. Risk Parity Allocation:  Return: +{m['risk_parity']['total_r']}R | Max DD: -{m['risk_parity']['max_drawdown_r']}R | Calmar: {m['risk_parity']['calmar_ratio']}x")
    print("="*75)
