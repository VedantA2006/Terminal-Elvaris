import json
import pandas as pd
import numpy as np
import math
import strategy_executor
import traceback
import warnings

_STRATEGIES = None

def load_strategies():
    global _STRATEGIES
    if _STRATEGIES is not None:
        return _STRATEGIES
    
    with open('data/leaderboard.json', 'r', encoding='utf-16') as f:
        leaderboard = json.load(f)
    
    # Exclude itself to avoid infinite recursion, and ensure only profitable strats are used
    _STRATEGIES = [s for s in leaderboard if s.get('total_pnl', 0) > 0 and s.get('name') != 'Mega Ensemble Strategy (Top 100 Consensus)']
    return _STRATEGIES

def _memoize_indicator(name, func, *args, **kwargs):
    return func(*args, **kwargs)

def calculate_signals(df):
    strats = load_strategies()
    
    # Try to extract the dt column correctly for resampling
    dt_col = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('dt', df.get('datetime', df.index)))
    df_temp = df.copy()
    df_temp.index = dt_col
    
    df_1h = df_temp.resample('1h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    df_4h = df_temp.resample('4h').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    
    bull_signals = []
    bear_signals = []
    sl_longs = []
    tp_longs = []
    sl_shorts = []
    tp_shorts = []
    
    # Prepare global environment for AST execution
    local_env = {
        'df': df.copy(), 'df_1h': df_1h.copy(), 'df_4h': df_4h.copy(),
        'pd': pd, 'np': np, 'math': math,
        '_memoize_indicator': _memoize_indicator,
        'SAFE_BUILTINS': strategy_executor.SAFE_BUILTINS,
    }
    
    # Inject all standard indicators from strategy_executor
    for k in dir(strategy_executor):
        if not k.startswith('_') and callable(getattr(strategy_executor, k)):
            local_env[k] = getattr(strategy_executor, k)
            
    for i, s in enumerate(strats):
        # We need a fresh env because execution might mutate globals
        env = dict(local_env)
        try:
            exec(s['code'], env, env)
            df_sig = None
            df_copy = df.copy()
            for func_name in ['calculate_signals', 'generate_signals', 'compute_signals', 'get_signals', 'strategy']:
                if func_name in env and callable(env[func_name]):
                    df_sig = env[func_name](df_copy)
                    break
            
            if df_sig is None:
                if 'bull_signal' in env.get('df', pd.DataFrame()).columns:
                    df_sig = env['df']
                else:
                    continue
                    
            if 'bull_signal' in df_sig.columns and 'sl_long' in df_sig.columns and 'tp1_long' in df_sig.columns:
                bull_signals.append(df_sig['bull_signal'].fillna(False).astype(bool).values)
                sl_longs.append(df_sig['sl_long'].values)
                tp_longs.append(df_sig['tp1_long'].values)
                
            if 'bear_signal' in df_sig.columns and 'sl_short' in df_sig.columns and 'tp1_short' in df_sig.columns:
                bear_signals.append(df_sig['bear_signal'].fillna(False).astype(bool).values)
                sl_shorts.append(df_sig['sl_short'].values)
                tp_shorts.append(df_sig['tp1_short'].values)
                
        except Exception as e:
            pass
            
    # Convert to numpy arrays for fast vectorized summation
    arr_bull = np.array(bull_signals, dtype=bool)
    arr_bear = np.array(bear_signals, dtype=bool)
    arr_sl_long = np.array(sl_longs, dtype=float)
    arr_tp_long = np.array(tp_longs, dtype=float)
    arr_sl_short = np.array(sl_shorts, dtype=float)
    arr_tp_short = np.array(tp_shorts, dtype=float)
    
    total_bull = np.sum(arr_bull, axis=0) if len(arr_bull) > 0 else np.zeros(len(df))
    total_bear = np.sum(arr_bear, axis=0) if len(arr_bear) > 0 else np.zeros(len(df))
    
    warnings.filterwarnings('ignore')
    
    # Calculate average TP and SL only among strategies that signaled a trade
    if len(arr_bull) > 0:
        mask_bull = np.where(arr_bull, 1.0, np.nan)
        avg_sl_long = np.nanmean(arr_sl_long * mask_bull, axis=0)
        avg_tp_long = np.nanmean(arr_tp_long * mask_bull, axis=0)
    else:
        avg_sl_long = np.zeros(len(df))
        avg_tp_long = np.zeros(len(df))
        
    if len(arr_bear) > 0:
        mask_bear = np.where(arr_bear, 1.0, np.nan)
        avg_sl_short = np.nanmean(arr_sl_short * mask_bear, axis=0)
        avg_tp_short = np.nanmean(arr_tp_short * mask_bear, axis=0)
    else:
        avg_sl_short = np.zeros(len(df))
        avg_tp_short = np.zeros(len(df))
    
    THRESHOLD = 100
    df_out = df.copy()
    df_out['bull_signal'] = total_bull >= THRESHOLD
    df_out['bear_signal'] = total_bear >= THRESHOLD
    
    df_out['sl_long'] = avg_sl_long
    df_out['tp1_long'] = avg_tp_long
    df_out['sl_short'] = avg_sl_short
    df_out['tp1_short'] = avg_tp_short
    
    return df_out
