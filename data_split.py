"""
Chronological Train / Validation / Test Data Splitter.

Splits a time-series DataFrame into three non-overlapping chronological segments:
  - Train (default 70%): Used for strategy generation, parameter optimization, grid search
  - Validation (default 15%): Used as mandatory confirmation gate before leaderboard admission
  - Test/Holdout (default 15%): Used ONLY for final champion reporting — never seen during fitting

This prevents the multiple-comparisons / data-dredging problem where strategies
appear profitable purely by overfitting to a single static dataset.
"""

import pandas as pd
from typing import Tuple


def split_data(df: pd.DataFrame,
               train_pct: float = 0.70,
               val_pct: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame chronologically into train, validation, and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLCV DataFrame with DatetimeIndex, sorted chronologically.
    train_pct : float
        Fraction of data for training (default 0.70).
    val_pct : float
        Fraction of data for validation (default 0.15).
        Test fraction is implicitly 1 - train_pct - val_pct.

    Returns
    -------
    (train_df, val_df, test_df) : Tuple of DataFrames
    """
    assert 0 < train_pct < 1, f"train_pct must be in (0, 1), got {train_pct}"
    assert 0 < val_pct < 1, f"val_pct must be in (0, 1), got {val_pct}"
    assert train_pct + val_pct < 1.0, f"train_pct + val_pct must be < 1.0, got {train_pct + val_pct}"

    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Log split boundaries for auditability
    print(f"  [Data Split] Train: {len(train_df):,} bars "
          f"({train_df.index[0].date() if hasattr(train_df.index[0], 'date') else '?'} -> "
          f"{train_df.index[-1].date() if hasattr(train_df.index[-1], 'date') else '?'})")
    print(f"  [Data Split] Validation: {len(val_df):,} bars "
          f"({val_df.index[0].date() if hasattr(val_df.index[0], 'date') else '?'} -> "
          f"{val_df.index[-1].date() if hasattr(val_df.index[-1], 'date') else '?'})")
    print(f"  [Data Split] Test/Holdout: {len(test_df):,} bars "
          f"({test_df.index[0].date() if hasattr(test_df.index[0], 'date') else '?'} -> "
          f"{test_df.index[-1].date() if hasattr(test_df.index[-1], 'date') else '?'})")

    return train_df, val_df, test_df
