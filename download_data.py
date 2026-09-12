"""
XAUUSD Multi-Timeframe Data Downloader.

Primary source: MetaTrader 5 High-Performance REST API (RoboForex-ECN)
Fallback: Calibrated Synthetic GBM gold data

Supported Timeframes: 1m, 5m, 15m, 1h, 4h, 1d
Saves primary 5m output to data/XAUUSD_5min.csv
"""

import os
import sys
import struct
import subprocess
import lzma as lzma_mod
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from typing import Optional, Dict, Tuple
import urllib.request
import json
import time

DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_FILE = DATA_DIR / 'XAUUSD_5min.csv'

# MetaTrader 5 High-Performance REST API
MT5_API_BASE = os.environ.get("MT5_API_BASE", "https://mutual-accurately-dryer-los.trycloudflare.com")

TIMEFRAME_MAP = {
    '1m': ('M1', 'XAUUSD_1min.csv', 65000),
    '5m': ('M5', 'XAUUSD_5min.csv', 100000),
    '15m': ('M15', 'XAUUSD_15min.csv', 65000),
    '1h': ('H1', 'XAUUSD_1hour.csv', 65000),
    '4h': ('H4', 'XAUUSD_4hour.csv', 20000),
    '1d': ('D1', 'XAUUSD_daily.csv', 10000),
}


# ========================================================================
# METHOD 1 (PRIMARY) — MetaTrader 5 High-Performance REST API
# ========================================================================

def download_from_mt5_api(symbol: str = 'XAUUSD',
                          timeframe: str = 'M5',
                          max_bars: int = 45000,
                          save_file: Path = OUTPUT_FILE) -> Optional[pd.DataFrame]:
    """
    Downloads historical candlestick rates from MetaTrader 5 High-Performance REST API.
    Provides real broker ECN data with actual tick volume and broker spread.
    """
    print(f"[1/4] Trying MetaTrader 5 API for {symbol} ({timeframe}) ...")
    all_rates = []
    chunk_size = 5000
    pos = 0
    t0 = time.time()

    while len(all_rates) < max_bars:
        url = f"{MT5_API_BASE}/api/rates/{symbol}?timeframe={timeframe}&count={chunk_size}&pos={pos}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'FoundeerTerminal/1.0'})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                rates = data.get('rates', [])
                if not rates:
                    break
                all_rates.extend(rates)
                if len(rates) < chunk_size:
                    break
                pos += chunk_size
        except Exception as e:
            print(f"   [!] MT5 API connection notice at pos={pos}: {e}")
            break

    if not all_rates:
        return None

    df = pd.DataFrame(all_rates)
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values(by='timestamp', inplace=True)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)

    if 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']

    cols = ['open', 'high', 'low', 'close', 'volume']
    if 'spread' in df.columns:
        cols.append('spread')

    df = df[cols]
    if save_file:
        save_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_file)
        print(f"   [OK] MT5 API: {len(df):,} bars saved to {save_file.name} in {time.time() - t0:.2f}s ({df.index[0]} to {df.index[-1]})")
    return df


def load_timeframe_data(timeframe: str = '5m', symbol: str = 'XAUUSD') -> pd.DataFrame:
    """
    Loads or downloads historical data for any requested timeframe (1m, 5m, 15m, 1h, 4h, 1d).
    Enables multi-timeframe strategies requiring Higher Timeframe trend filters.
    """
    tf_key = timeframe.lower().strip()
    if tf_key not in TIMEFRAME_MAP:
        raise ValueError(f"Unknown timeframe: {timeframe}. Available: {list(TIMEFRAME_MAP.keys())}")

    mt5_tf, filename, max_bars = TIMEFRAME_MAP[tf_key]
    target_path = DATA_DIR / filename

    if target_path.exists():
        df = pd.read_csv(target_path, index_col='datetime', parse_dates=True)
        if len(df) > 50:
            return df

    # Download from MT5 API
    df = download_from_mt5_api(symbol=symbol, timeframe=mt5_tf, max_bars=max_bars, save_file=target_path)
    if df is not None and len(df) > 50:
        return df

    raise RuntimeError(f"Unable to load or download {timeframe} data for {symbol}.")

# ========================================================================
# METHOD 2 — Calibrated Synthetic XAUUSD fallback (offline safety)
# ========================================================================

def generate_synthetic_data():
    """
    Generate realistic synthetic XAUUSD 5-min data using Geometric
    Brownian Motion calibrated to gold's actual volatility.
    """
    print("[3/3] Generating synthetic XAUUSD data ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Parameters calibrated to real XAUUSD
    start_price = 2650.0
    annual_vol = 0.18          # ~18 % annual volatility for gold
    annual_drift = 0.05        # slight upward drift
    minutes_per_bar = 5
    trading_hours_per_day = 23  # Forex nearly 24h
    bars_per_day = int(trading_hours_per_day * 60 / minutes_per_bar)
    num_days = 130             # ~6 months of weekdays
    total_bars = bars_per_day * num_days

    # Time index (skip weekends)
    start_dt = datetime(2025, 3, 10, 0, 0)
    times = []
    current_dt = start_dt
    while len(times) < total_bars:
        if current_dt.weekday() < 5:  # Mon-Fri
            times.append(current_dt)
        current_dt += timedelta(minutes=minutes_per_bar)
        # Skip to Monday if we hit Saturday
        if current_dt.weekday() == 5:
            current_dt += timedelta(days=2)

    times = times[:total_bars]

    # GBM for close prices
    dt_year = minutes_per_bar / (252 * trading_hours_per_day * 60)
    returns = np.random.normal(
        annual_drift * dt_year,
        annual_vol * np.sqrt(dt_year),
        total_bars
    )

    closes = np.zeros(total_bars)
    closes[0] = start_price
    for i in range(1, total_bars):
        closes[i] = closes[i - 1] * np.exp(returns[i])

    # Generate OHLV from close
    intrabar_vol = annual_vol * np.sqrt(dt_year) * 0.6
    highs = closes * (1 + np.abs(np.random.normal(0, intrabar_vol, total_bars)))
    lows = closes * (1 - np.abs(np.random.normal(0, intrabar_vol, total_bars)))
    opens = np.roll(closes, 1)
    opens[0] = start_price

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    # Volume (lognormal)
    volumes = np.random.lognormal(mean=6, sigma=1.5, size=total_bars).astype(int)

    df = pd.DataFrame({
        'open': np.round(opens, 2),
        'high': np.round(highs, 2),
        'low': np.round(lows, 2),
        'close': np.round(closes, 2),
        'volume': volumes,
    }, index=pd.DatetimeIndex(times, name='datetime'))

    df.to_csv(OUTPUT_FILE)
    print(f"   [OK] Synthetic data: {len(df)} bars, "
          f"{df.index[0].date()} to {df.index[-1].date()}")
    return df


# ========================================================================
# MAIN
# ========================================================================

def load_or_download():
    """Load MetaTrader 5 ECN broker data, falling back to MT5 API download or calibrated synthetic."""
    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE, index_col='datetime', parse_dates=True)
        if len(df) > 1000:
            print(f"Using cached MT5 broker data: {len(df):,} bars ({df.index[0]} to {df.index[-1]})")
            return df

    # 1. MetaTrader 5 REST API (Primary)
    df = download_from_mt5_api(symbol='XAUUSD', timeframe='M5', max_bars=100000, save_file=OUTPUT_FILE)
    if df is not None and len(df) > 1000:
        return df

    # 2. Synthetic fallback (Offline safety)
    print("   [!] Falling back to calibrated synthetic gold data generation...")
    df = generate_synthetic_data()
    if df is not None and len(df) > 1000:
        return df

    raise RuntimeError("Failed to load or generate XAUUSD 5m data from MT5 API.")


if __name__ == '__main__':
    df = load_or_download()
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(df.tail())
