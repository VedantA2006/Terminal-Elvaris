"""
Multi-Instrument Data Downloader.

Primary source: Local MetaTrader 5 Terminal
Supported Instruments: XAUUSD, US100, SPX500, EURUSD, GBPUSD
Supported Timeframes: 1m, 5m, 15m, 1h, 4h, 1d
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package not installed. Please install it.")
    sys.exit(1)

DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

INSTRUMENTS = ['XAUUSD']

TIMEFRAME_MAP = {
    '1m': (mt5.TIMEFRAME_M1, 80000),
    '5m': (mt5.TIMEFRAME_M5, 400000),
    '15m': (mt5.TIMEFRAME_M15, 65000),
    '1h': (mt5.TIMEFRAME_H1, 65000),
    '4h': (mt5.TIMEFRAME_H4, 20000),
    '1d': (mt5.TIMEFRAME_D1, 10000),
}

def get_filename(symbol: str, timeframe: str) -> Path:
    return DATA_DIR / f'{symbol}_{timeframe}.csv'

def download_from_mt5(symbol: str, timeframe_str: str = '5m', max_bars: int = 400000) -> pd.DataFrame:
    if not mt5.initialize():
        print(f"initialize() failed, error code = {mt5.last_error()}")
        return None

    if timeframe_str not in TIMEFRAME_MAP:
        print(f"Unknown timeframe: {timeframe_str}")
        return None

    mt5_tf, default_max = TIMEFRAME_MAP[timeframe_str]
    max_bars = min(max_bars, default_max)

    mt5.symbol_select(symbol, True)
    print(f"Downloading {max_bars} bars for {symbol} on {timeframe_str}...")
    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, max_bars)
    
    if rates is None or len(rates) == 0:
        print(f"Failed to get rates for {symbol}, error code = {mt5.last_error()}")
        return None

    df = pd.DataFrame(rates)
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    
    # Rename columns to match existing system
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    cols = ['open', 'high', 'low', 'close', 'volume']
    if 'spread' in df.columns:
        cols.append('spread')
    
    df = df[cols]
    
    save_file = get_filename(symbol, timeframe_str)
    df.to_csv(save_file)
    print(f"   [OK] Saved {len(df)} bars to {save_file.name} ({df.index[0]} to {df.index[-1]})")
    return df

def add_news_and_weekend_flags(df: pd.DataFrame) -> pd.DataFrame:
    df['is_weekend_close'] = 0
    weekday = df.index.dayofweek
    hour = df.index.hour
    minute = df.index.minute
    
    fri_mask = (weekday == 4) & ((hour > 21) | ((hour == 21) & (minute >= 30)))
    sat_mask = (weekday == 5)
    sun_mask = (weekday == 6) & (hour < 21)
    df.loc[fri_mask | sat_mask | sun_mask, 'is_weekend_close'] = 1
    
    df['is_news_time'] = 0
    try:
        news_file = Path('XAUUSD_News_FOMC_NFP_CPI_PPI_PMI_2Y.csv')
        if news_file.exists():
            try:
                news_df = pd.read_csv(news_file, encoding='utf-8')
            except UnicodeDecodeError:
                news_df = pd.read_csv(news_file, encoding='utf-16')
            if 'Importance' in news_df.columns:
                news_df = news_df[news_df['Importance'].isin(['CALENDAR_IMPORTANCE_HIGH', 'CALENDAR_IMPORTANCE_MODERATE'])].copy()
            news_df['datetime'] = pd.to_datetime(news_df['Date'] + ' ' + news_df['Time'], format='%Y.%m.%d %H:%M')
            news_df['datetime'] = news_df['datetime'].dt.tz_localize('UTC')
            
            for event_time in news_df['datetime']:
                start_time = event_time - pd.Timedelta(minutes=15)
                end_time = event_time + pd.Timedelta(minutes=15)
                mask = (df.index >= start_time) & (df.index <= end_time)
                df.loc[mask, 'is_news_time'] = 1
    except Exception as e:
        print(f"Failed to load news data: {e}")
    return df

def load_or_download(symbol: str = 'XAUUSD', timeframe: str = '5m') -> pd.DataFrame:
    save_file = get_filename(symbol, timeframe)
    if save_file.exists():
        df = pd.read_csv(save_file, index_col='datetime', parse_dates=True)
        if len(df) > 1000:
            print(f"Using cached MT5 data for {symbol} {timeframe}: {len(df):,} bars")
            return add_news_and_weekend_flags(df)
            
    df = download_from_mt5(symbol, timeframe)
    if df is not None and len(df) > 1000:
        return add_news_and_weekend_flags(df)
        
    raise RuntimeError(f"Failed to load or generate data for {symbol} {timeframe}")

def download_all():
    mt5.initialize()
    for symbol in INSTRUMENTS:
        download_from_mt5(symbol, '5m')
    mt5.shutdown()

if __name__ == '__main__':
    download_all()

def load_timeframe_data(timeframe: str, symbol: str = 'XAUUSD'):
    return load_or_download(symbol, timeframe)
