def calculate_signals(df):
    # Continuous Institutional Gold Session: London Open through US Floor Settlement (06:00 - 18:00 UTC)
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
    df['tp1_long'] = df['close'] + (risk_long * 7.5)
    df['tp1_short'] = df['close'] - (risk_short * 7.5)

    # Optional multi-target runner
    df['tp2_long'] = df['close'] + (risk_long * 16.5)
    df['tp2_short'] = df['close'] - (risk_short * 16.5)
    df['use_breakeven'] = True
    df['use_trailing'] = True


    # Risk Officer Injected Session Gating (London/NY Killzones)
    _sess_mask = session_mask(df, 'london_ny')
    if 'bull_signal' in df.columns:
        df['bull_signal'] = df['bull_signal'] & _sess_mask
    if 'bear_signal' in df.columns:
        df['bear_signal'] = df['bear_signal'] & _sess_mask

    return df