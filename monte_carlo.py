"""
Monte Carlo Sequence-Risk Bootstrap Resampling Engine for Algorithmic Trading Strategies.

Performs bootstrap trade-order resampling simulations (1,000 runs) on historical
trade sequences to evaluate sequence-of-returns risk:
  - Percentile equity corridors (5th, 25th, Median 50th, 75th, 95th)
  - 95% Confidence Max Drawdown (Value at Risk) in R and USD
  - Expected (Mean) Drawdown in R and USD
  - Risk of Ruin probabilities (Chance of reaching 10R DD, 20R DD, 20% Account DD)
  - Probability of Profit

Note: This evaluates sequence risk via empirical trade shuffling; it does NOT
model synthetic price paths, slippage regimes, or structural volatility shocks.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


def run_monte_carlo(trades: List[Dict[str, Any]],
                     initial_capital: float = 100000.0,
                     lot_size: float = 100.0,
                     num_simulations: int = 1000,
                     seed: int = 42) -> Dict[str, Any]:
    """
    Runs Monte Carlo bootstrap simulation on a list of executed trades.

    Returns:
        metrics: dict of risk of ruin, expected drawdown, 95% VaR
        corridors: percentiles along normalized trade progression for fan charts
    """
    if not trades or len(trades) < 2:
        return {
            'simulations_count': 0,
            'probability_of_profit': 0.0,
            'expected_max_dd_usd': 0.0,
            'expected_max_dd_r': 0.0,
            'var_95_max_dd_usd': 0.0,
            'var_95_max_dd_r': 0.0,
            'risk_of_ruin_10r': 0.0,
            'risk_of_ruin_20r': 0.0,
            'risk_of_ruin_20pct': 0.0,
            'median_final_equity': initial_capital,
            'median_final_r': 0.0,
            'corridors': {
                'indices': [],
                'p5': [],
                'p25': [],
                'p50': [],
                'p75': [],
                'p95': [],
            }
        }

    rng = np.random.default_rng(seed)
    n_trades = len(trades)

    pnls = np.array([t.get('pnl', 0.0) for t in trades], dtype=np.float64)

    # Compute R-multiples for each trade
    r_multiples = []
    for t in trades:
        if 'r_return' in t and t['r_return'] is not None:
            r_multiples.append(float(t['r_return']))
        else:
            pnl = t.get('pnl', 0.0)
            entry = t.get('entry_price', 0.0)
            sl = t.get('sl', None)
            if sl is not None and entry > 0 and abs(entry - sl) > 0:
                risk_usd = abs(entry - sl) * lot_size
                r_multiples.append(pnl / risk_usd if risk_usd > 0 else (1.0 if pnl > 0 else -1.0))
            else:
                # Default baseline risk estimate ($10/oz * 100oz = $1,000 per R)
                r_multiples.append(pnl / 1000.0)

    r_multiples = np.array(r_multiples, dtype=np.float64)

    # Matrix of random resampled indices: (num_simulations, n_trades)
    resample_indices = rng.integers(0, n_trades, size=(num_simulations, n_trades))

    sim_pnls = pnls[resample_indices]
    sim_rs = r_multiples[resample_indices]

    # Cumulative equity paths: (num_simulations, n_trades + 1)
    cum_pnls = np.cumsum(sim_pnls, axis=1)
    equity_paths = initial_capital + np.hstack([np.zeros((num_simulations, 1)), cum_pnls])

    cum_rs = np.cumsum(sim_rs, axis=1)
    r_paths = np.hstack([np.zeros((num_simulations, 1)), cum_rs])

    # Compute Maximum Drawdown for each simulation in USD and R
    # Peak equity running max along rows
    peaks_usd = np.maximum.accumulate(equity_paths, axis=1)
    dds_usd = peaks_usd - equity_paths
    max_dds_usd = np.max(dds_usd, axis=1)
    max_dd_pcts = (max_dds_usd / np.max(peaks_usd, axis=1)) * 100.0

    peaks_r = np.maximum.accumulate(r_paths, axis=1)
    dds_r = peaks_r - r_paths
    max_dds_r = np.max(dds_r, axis=1)

    final_equities = equity_paths[:, -1]
    final_rs = r_paths[:, -1]

    # Risk of Ruin metrics
    risk_ruin_10r = float(np.mean(max_dds_r >= 10.0) * 100.0)
    risk_ruin_20r = float(np.mean(max_dds_r >= 20.0) * 100.0)
    risk_ruin_20pct = float(np.mean(max_dd_pcts >= 20.0) * 100.0)
    prob_profit = float(np.mean(final_equities > initial_capital) * 100.0)

    # Downsample points for fan chart (max 50 points along x-axis for fast UI rendering)
    num_steps = min(50, n_trades + 1)
    step_indices = np.linspace(0, n_trades, num_steps, dtype=int)
    sampled_r_paths = r_paths[:, step_indices]

    p5 = np.percentile(sampled_r_paths, 5, axis=0)
    p25 = np.percentile(sampled_r_paths, 25, axis=0)
    p50 = np.percentile(sampled_r_paths, 50, axis=0)
    p75 = np.percentile(sampled_r_paths, 75, axis=0)
    p95 = np.percentile(sampled_r_paths, 95, axis=0)

    return {
        'simulations_count': num_simulations,
        'probability_of_profit': round(prob_profit, 1),
        'expected_max_dd_usd': round(float(np.mean(max_dds_usd)), 2),
        'expected_max_dd_r': round(float(np.mean(max_dds_r)), 2),
        'var_95_max_dd_usd': round(float(np.percentile(max_dds_usd, 95)), 2),
        'var_95_max_dd_r': round(float(np.percentile(max_dds_r, 95)), 2),
        'median_max_dd_r': round(float(np.median(max_dds_r)), 2),
        'risk_of_ruin_10r': round(risk_ruin_10r, 1),
        'risk_of_ruin_20r': round(risk_ruin_20r, 1),
        'risk_of_ruin_20pct': round(risk_ruin_20pct, 1),
        'median_final_equity': round(float(np.median(final_equities)), 2),
        'median_final_r': round(float(np.median(final_rs)), 2),
        'corridors': {
            'indices': step_indices.tolist(),
            'p5': [round(float(v), 2) for v in p5],
            'p25': [round(float(v), 2) for v in p25],
            'p50': [round(float(v), 2) for v in p50],
            'p75': [round(float(v), 2) for v in p75],
            'p95': [round(float(v), 2) for v in p95],
        }
    }
