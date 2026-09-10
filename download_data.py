"""
XAUUSD 5-minute data downloader.

Attempts three methods in order:
  1. dukascopy-node (npm) — official Dukascopy historical data
  2. yfinance (GC=F gold futures) — limited to ~60 days of 5-min
  3. Synthetic generation — realistic GBM-based XAUUSD data

Saves output to  data/XAUUSD_5min.csv
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

DATA_DIR = Path(__file__).parent / 'data'
OUTPUT_FILE = DATA_DIR / 'XAUUSD_5min.csv'

# Dukascopy data feed constants
DUKASCOPY_BASE = "https://datafeed.dukascopy.com/datafeed"
XAUUSD_POINT_VALUE = 1000  # prices stored as int / 1000


# ========================================================================
# METHOD 1 — dukascopy-node (npx)
# ========================================================================

def download_dukascopy_node():
    """Use the npm `dukascopy-node` CLI to download data."""
    print("[1/3] Trying dukascopy-node ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=180)

    cmd = f"npx -y dukascopy-node -i xauusd -from {start_date.strftime('%Y-%m-%d')} -to {end_date.strftime('%Y-%m-%d')} -t m5 -v -f csv -dir {str(DATA_DIR)}"
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, shell=True
        )
        if result.returncode == 0:
            # Find the generated CSV
            csvs = sorted(DATA_DIR.glob("*.csv"))
            for csv_path in csvs:
                if csv_path.name != OUTPUT_FILE.name:
                    df = pd.read_csv(csv_path)
                    # Normalise columns
                    col_map = {}
                    for c in df.columns:
                        cl = c.strip().lower()
                        if 'time' in cl or 'date' in cl:
                            col_map[c] = 'datetime'
                        elif cl == 'open':
                            col_map[c] = 'open'
                        elif cl == 'high':
                            col_map[c] = 'high'
                        elif cl == 'low':
                            col_map[c] = 'low'
                        elif cl == 'close':
                            col_map[c] = 'close'
                        elif 'vol' in cl:
                            col_map[c] = 'volume'
                    df.rename(columns=col_map, inplace=True)
                    if 'datetime' in df.columns:
                        # Dukascopy-node outputs timestamp in milliseconds
                        if pd.api.types.is_numeric_dtype(df['datetime']):
                            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
                        else:
                            df['datetime'] = pd.to_datetime(df['datetime'])
                        df.set_index('datetime', inplace=True)
                        df = df[['open', 'high', 'low', 'close', 'volume']]
                        df.to_csv(OUTPUT_FILE)
                        print(f"   [OK] dukascopy-node: {len(df)} bars saved")
                        return df
        else:
            print(f"   [X] dukascopy-node failed: {result.stderr[:200]}")
    except FileNotFoundError:
        print("   [X] npx/node not found on PATH")
    except subprocess.TimeoutExpired:
        print("   [X] dukascopy-node timed out")
    except Exception as e:
        print(f"   [X] dukascopy-node error: {e}")
    return None


# ========================================================================
# METHOD 2 — Dukascopy direct download (Python)
# ========================================================================

def download_dukascopy_direct():
    """Download directly from Dukascopy data feed using requests."""
    print("[2/3] Trying direct Dukascopy download ...")
    try:
        import requests
    except ImportError:
        print("   ✗ requests not installed")
        return None

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=180)

    all_records = []
    current = start_date
    downloaded_days = 0

    while current <= end_date:
        # Dukascopy months are 0-indexed
        month_idx = current.month - 1
        url = (f"{DUKASCOPY_BASE}/XAUUSD/{current.year}/"
               f"{month_idx:02d}/{current.day:02d}/BID_candles_min_5.bi5")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 0:
                records = _parse_bi5(resp.content, current)
                if records:
                    all_records.extend(records)
                    downloaded_days += 1
                    if downloaded_days % 10 == 0:
                        print(f"   ... downloaded {downloaded_days} days")
            else:
                print(f"   [Debug] Failed to fetch {current.date()}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"   [Debug] Exception on {current.date()}: {e}")
        current += timedelta(days=1)

    if all_records:
        df = pd.DataFrame(all_records, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df.to_csv(OUTPUT_FILE)
        print(f"   [OK] Direct download: {len(df)} bars ({downloaded_days} days)")
        return df

    print("   [X] Direct download yielded no data")
    return None


def _parse_bi5(data, date):
    """Parse Dukascopy .bi5 LZMA-compressed binary candle data."""
    try:
        decompressed = lzma_mod.decompress(data)
    except Exception:
        return []

    records = []
    # Each record: 4 uint32 (time, O, H, L, C) + 1 float32 (vol) = 24 bytes
    rec_size = 24
    for i in range(0, len(decompressed) - rec_size + 1, rec_size):
        chunk = decompressed[i:i + rec_size]
        try:
            time_ms, o, h, l, c, v = struct.unpack('>IIIIIf', chunk)
        except struct.error:
            continue
        dt = datetime(date.year, date.month, date.day) + timedelta(milliseconds=time_ms)
        records.append([
            dt,
            o / XAUUSD_POINT_VALUE,
            h / XAUUSD_POINT_VALUE,
            l / XAUUSD_POINT_VALUE,
            c / XAUUSD_POINT_VALUE,
            round(v, 2),
        ])
    return records


# ========================================================================
# METHOD 3 — Synthetic XAUUSD data (always works)
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
    """Load real Dukascopy 6-month data up to today."""
    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE, index_col='datetime', parse_dates=True)
        if len(df) > 1000:
            print(f"Using cached real Dukascopy data: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
            return df

    df = download_dukascopy_node()
    if df is not None and len(df) > 1000:
        return df

    raise RuntimeError("Failed to load real Dukascopy data from today to last 6 months.")


if __name__ == '__main__':
    df = load_or_download()
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(df.tail())
