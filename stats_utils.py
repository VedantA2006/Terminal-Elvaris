"""
Shared Statistical Metrics Utilities.

Single source of truth for Sharpe Ratio, Sortino Ratio, and other
risk-adjusted return metrics used by both backtest.py and strategy_executor.py.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple


def compute_sharpe_sortino(trades: List[Dict[str, Any]],
                           equity_curve: List[Dict[str, Any]] = None,
                           initial_capital: float = 100000.0) -> Tuple[float, float]:
    """
    Compute annualized Sharpe and Sortino ratios.

    Method 1 (preferred): Daily returns from equity curve, annualized by sqrt(252).
    Method 2 (fallback): Trade-level PnL returns if daily curve is insufficient.

    Returns:
        (sharpe_ratio, sortino_ratio)
    """
    sharpe = 0.0
    sortino = 0.0

    # Method 1: Daily periodic return based (preferred)
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

    # Method 2: Trade-based fallback if daily curve insufficient
    if sharpe == 0.0 and trades:
        pnls = [t.get('pnl', 0.0) for t in trades]
        if len(pnls) > 1 and np.std(pnls) > 0:
            mean_r = np.mean(pnls)
            std_r = np.std(pnls)
            sharpe = round((mean_r / std_r) * np.sqrt(min(len(pnls), 252)), 2)
            downside = [p for p in pnls if p < 0]
            if downside and np.std(downside) > 0:
                sortino = round((mean_r / np.std(downside)) * np.sqrt(min(len(pnls), 252)), 2)
            else:
                sortino = sharpe

    return sharpe, sortino
