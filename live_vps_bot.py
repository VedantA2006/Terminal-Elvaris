import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

# ==============================================================================
# INDICATOR FUNCTIONS (Copied for standalone execution)
# ==============================================================================

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def smma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return smma(tr, period)

def bollinger_bands(series: pd.Series, period: int = 20, mult: float = 2.0):
    basis = series.rolling(period).mean()
    dev = series.rolling(period).std() * mult
    return basis + dev, basis - dev, basis

def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    vol = df['volume'] if 'volume' in df.columns and (df['volume'] > 0).any() else pd.Series(1.0, index=df.index)
    date_col = df.index.date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('dt', df.get('datetime', df.index))).dt.date
    pv_cum = (tp * vol).groupby(date_col).cumsum()
    v_cum = vol.groupby(date_col).cumsum()
    return pv_cum / v_cum.replace(0, np.nan)

def keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14, mult: float = 2.0):
    mid = ema(df['close'], ema_period)
    atr_v = atr(df, atr_period)
    return mid + mult * atr_v, mid - mult * atr_v, mid

def find_swings(df: pd.DataFrame, swing_len: int = 7):
    n = len(df)
    h_vals, l_vals = df['high'].values, df['low'].values
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(swing_len * 2, n):
        mid = i - swing_len
        if h_vals[mid] == max(h_vals[i - 2 * swing_len : i + 1]):
            sh[i] = h_vals[mid]
        if l_vals[mid] == min(l_vals[i - 2 * swing_len : i + 1]):
            sl[i] = l_vals[mid]
    s_h = pd.Series(sh, index=df.index).ffill()
    s_l = pd.Series(sl, index=df.index).ffill()
    return s_h, s_l

def session_mask(df: pd.DataFrame, session: str = 'london_ny') -> pd.Series:
    times = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('dt', df.get('datetime', df.index)))
    mins = times.hour * 60 + times.minute
    if session == 'london':
        m = (6 * 60 <= mins) & (mins < 11 * 60)
    elif session == 'ny':
        m = (12 * 60 + 20 <= mins) & (mins < 17 * 60 + 30)
    elif session == 'asia':
        m = (0 <= mins) & (mins < 6 * 60)
    elif session == 'london_fix':
        m = (14 * 60 + 30 <= mins) & (mins < 15 * 60 + 30)
    else:
        m = (6 * 60 <= mins) & (mins < 18 * 60)
    return pd.Series(m, index=df.index)

def daily_levels(df: pd.DataFrame) -> pd.DataFrame:
    date_col = df.index.date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('dt', df.get('datetime', df.index))).dt.date
    daily = df.groupby(date_col).agg({'high': 'max', 'low': 'min', 'close': 'last'})
    daily_shifted = daily.shift(1)
    pp = (daily_shifted['high'] + daily_shifted['low'] + daily_shifted['close']) / 3.0
    r1 = (2.0 * pp) - daily_shifted['low']
    s1 = (2.0 * pp) - daily_shifted['high']
    r2 = pp + (daily_shifted['high'] - daily_shifted['low'])
    s2 = pp - (daily_shifted['high'] - daily_shifted['low'])
    levels_df = pd.DataFrame({
        'pdh': daily_shifted['high'],
        'pdl': daily_shifted['low'],
        'pdc': daily_shifted['close'],
        'pp': pp, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2
    })
    date_series = pd.Series(date_col, index=df.index)
    mapped = levels_df.reindex(date_series.values)
    mapped.index = df.index
    return mapped

def volume_profile_levels(df: pd.DataFrame) -> pd.DataFrame:
    date_col = df.index.date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('dt', df.get('datetime', df.index))).dt.date
    daily_groups = df.groupby(date_col)

    profiles = {}
    for d, g in daily_groups:
        vol = g['volume'] if 'volume' in g.columns and (g['volume'] > 0).any() else pd.Series(1.0, index=g.index)
        p_min, p_max = g['low'].min(), g['high'].max()
        if p_max - p_min < 0.5:
            p_max = p_min + 1.0
        bins = np.linspace(p_min, p_max, 25)
        typ = (g['high'] + g['low'] + g['close']) / 3.0
        binned = pd.cut(typ, bins=bins, labels=bins[:-1])
        vol_per_bin = vol.groupby(binned, observed=False).sum()

        if len(vol_per_bin) > 0 and vol_per_bin.max() > 0:
            poc_level = float(vol_per_bin.idxmax())
            total_vol = vol_per_bin.sum()
            sorted_bins = vol_per_bin.sort_values(ascending=False)
            cum_vol = sorted_bins.cumsum()
            val_bins = sorted_bins[cum_vol <= total_vol * 0.70].index
            if len(val_bins) > 0:
                vah_level = float(max(val_bins))
                val_level = float(min(val_bins))
            else:
                vah_level = poc_level + 2.0
                val_level = poc_level - 2.0
        else:
            poc_level = (p_min + p_max) / 2.0
            vah_level = poc_level + 2.0
            val_level = poc_level - 2.0

        profiles[d] = {'poc': poc_level, 'vah': vah_level, 'val': val_level}

    prof_df = pd.DataFrame(profiles).T
    prof_df_shifted = prof_df.shift(1)
    df_date = pd.Series(date_col, index=df.index)
    res = df_date.map(prof_df_shifted.to_dict(orient='index'))
    return pd.DataFrame(res.tolist(), index=df.index)

# ==============================================================================
# STRATEGY LOGIC
# ==============================================================================

def calculate_signals(df):
    times = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('dt', df.get('datetime', df.index)))
    mins = times.hour * 60 + times.minute
    sess = (6 * 60 <= mins) & (mins < 18 * 60)

    # Swing Highs and Lows
    sw_highs, sw_lows = find_swings(df, swing_len=14)

    # Daily Levels
    levels = daily_levels(df)

    # VWAP
    vwap_line = vwap(df)

    # Volatility Squeeze Breakout
    bb_upper, bb_lower, bb_basis = bollinger_bands(df['close'], period=20, mult=2.0)
    keltner_upper, keltner_lower, keltner_mid = keltner_channels(df, ema_period=20, atr_period=14, mult=2.0)

    # Volatility Squeeze Condition
    vol_squeeze = (bb_lower > keltner_lower) & (bb_upper < keltner_upper)

    # SSL and BSL Sweeps
    ssl_sweep = (df['low'] < sw_lows) & (df['close'] > sw_lows)
    bsl_sweep = (df['high'] > sw_highs) & (df['close'] < sw_highs)

    # Armed Conditions
    armed_long = ssl_sweep.rolling(10, min_periods=1).max() == 1
    armed_short = bsl_sweep.rolling(10, min_periods=1).max() == 1

    # Volume Profile Levels
    vol_profile = volume_profile_levels(df)

    # Raw Signals
    raw_bull = (armed_long & (df['close'] > vwap_line) & (df['close'] > levels['s1']) & vol_squeeze & sess)
    df['bull_signal'] = raw_bull & (~raw_bull.shift(1).fillna(False))

    raw_bear = (armed_short & (df['close'] < vwap_line) & (df['close'] < levels['r1']) & vol_squeeze & sess)
    df['bear_signal'] = raw_bear & (~raw_bear.shift(1).fillna(False))

    # ATR-based Stop Loss
    atr_val = atr(df, 14)
    df['sl_long'] = np.minimum(df['low'], sw_lows) - (1.6 * atr_val)
    df['sl_short'] = np.maximum(df['high'], sw_highs) + (1.6 * atr_val)

    # Risk Calculation
    risk_long = np.maximum(df['close'] - df['sl_long'], 4.00)
    risk_short = np.maximum(df['sl_short'] - df['close'], 4.00)

    # Multi-Target Take Profit
    df['tp1_long'] = df['close'] + (risk_long * 6.0)
    df['tp1_short'] = df['close'] - (risk_short * 6.0)

    # Optional multi-target runner
    df['tp2_long'] = df['close'] + (risk_long * 13.2)
    df['tp2_short'] = df['close'] - (risk_short * 13.2)
    df['use_breakeven'] = True
    df['use_trailing'] = True

    # Risk Officer Injected Session Gating (London/NY Killzones)
    _sess_mask = session_mask(df, 'london_ny')
    if 'bull_signal' in df.columns:
        df['bull_signal'] = df['bull_signal'] & _sess_mask
    if 'bear_signal' in df.columns:
        df['bear_signal'] = df['bear_signal'] & _sess_mask

    return df

# ==============================================================================
# MT5 LIVE TRADING LOGIC
# ==============================================================================

SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
MAGIC_NUMBER = 117117
LOT_SIZE = 0.1  # Change as per risk management

def get_rates(bars=1000):
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, bars)
    if rates is None:
        print(f"[{datetime.now()}] Error copying rates: {mt5.last_error()}")
        return pd.DataFrame()
    
    df = pd.DataFrame(rates)
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    if 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']
    return df

def place_order(order_type, price, sl, tp):
    point = mt5.symbol_info(SYMBOL).point
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": order_type,
        "price": price,
        "sl": float(sl),
        "tp": float(tp),
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "Bot_Signal",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[{datetime.now()}] Order failed, retcode={result.retcode}")
        return False
    else:
        print(f"[{datetime.now()}] Order executed: {result}")
        return True

def main():
    print("Initializing MT5...")
    if not mt5.initialize():
        print(f"initialize() failed, error code = {mt5.last_error()}")
        return
        
    print(f"MT5 Initialized. Connected to {mt5.account_info().server} - Account {mt5.account_info().login}")
    
    # Enable symbol
    if not mt5.symbol_select(SYMBOL, True):
        print(f"Symbol {SYMBOL} not found.")
        mt5.shutdown()
        return
        
    print(f"Starting Live Trading loop on {SYMBOL}...")
    last_processed_time = None
    
    while True:
        try:
            # Poll every 10 seconds
            time.sleep(10)
            
            df = get_rates(bars=500)
            if df.empty:
                continue
                
            # Current unclosed bar is the last one [-1], last fully closed bar is [-2]
            current_time = df.index[-2]
            
            # Wait for a new closed bar to evaluate
            if last_processed_time is not None and current_time <= last_processed_time:
                continue
                
            print(f"[{datetime.now()}] Evaluating new 5M bar closed at {current_time}")
            
            # Evaluate signals
            sig_df = calculate_signals(df)
            
            # Check the last fully closed candle for signals
            closed_bar = sig_df.iloc[-2]
            
            bull_sig = closed_bar.get('bull_signal', False)
            bear_sig = closed_bar.get('bear_signal', False)
            
            if bull_sig:
                sl = closed_bar['sl_long']
                tp = closed_bar['tp1_long'] # Use TP1, TP2 can be implemented with split orders
                price = mt5.symbol_info_tick(SYMBOL).ask
                print(f"[{datetime.now()}] 🟢 BULL SIGNAL DETECTED. SL: {sl}, TP: {tp}")
                place_order(mt5.ORDER_TYPE_BUY, price, sl, tp)
                
            elif bear_sig:
                sl = closed_bar['sl_short']
                tp = closed_bar['tp1_short']
                price = mt5.symbol_info_tick(SYMBOL).bid
                print(f"[{datetime.now()}] 🔴 BEAR SIGNAL DETECTED. SL: {sl}, TP: {tp}")
                place_order(mt5.ORDER_TYPE_SELL, price, sl, tp)
            
            last_processed_time = current_time
            
        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            print(f"[{datetime.now()}] Loop error: {e}")
            time.sleep(10)
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
