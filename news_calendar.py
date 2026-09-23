import pandas as pd
import os
from datetime import timedelta

_NEWS_CACHE = None

def get_news_events():
    global _NEWS_CACHE
    if _NEWS_CACHE is not None:
        return _NEWS_CACHE

    csv_path = 'XAUUSD_News_FOMC_NFP_CPI_PPI_PMI_2Y.csv'
    if not os.path.exists(csv_path):
        print(f"Warning: News file not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path, encoding='utf-16')
    
    if 'Importance' in df.columns:
        high_impact = df[(df['Importance'] == 'CALENDAR_IMPORTANCE_HIGH') | (df['Importance'] == 'CALENDAR_IMPORTANCE_MODERATE')].copy()
    else:
        high_impact = df.copy()
    
    if high_impact.empty:
        _NEWS_CACHE = pd.DataFrame()
        return _NEWS_CACHE

    # Parse Date and Time into datetime
    # Format: Date: 2024.09.18, Time: 19:00
    high_impact['datetime'] = pd.to_datetime(high_impact['Date'] + ' ' + high_impact['Time'], format='%Y.%m.%d %H:%M')
    high_impact['datetime'] = high_impact['datetime'].dt.tz_localize('UTC')
    
    _NEWS_CACHE = high_impact
    return _NEWS_CACHE


def get_mapped_instrument(symbol: str) -> str:
    mapping = {
        'XAUUSD': 'XAUUSD',
        'US100': 'NASDAQ100',
        'SPX500': 'SP500',
        'EURUSD': 'EURUSD',
        'GBPUSD': 'GBPUSD'
    }
    return mapping.get(symbol, symbol)

def is_news_embargo(current_time: pd.Timestamp, symbol: str = 'XAUUSD', buffer_minutes: int = 15) -> bool:
    if not isinstance(current_time, pd.Timestamp):
        current_time = pd.to_datetime(current_time)
        
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize('UTC')
            
    events_df = get_news_events()
    
    if events_df.empty:
        return False
        
    # The current XAUUSD CSV only contains XAUUSD events, so no need to filter by instrument if column doesn't exist
    if 'Instrument' in events_df.columns:
        mapped_inst = get_mapped_instrument(symbol)
        symbol_events = events_df[events_df['Instrument'] == mapped_inst]
    else:
        symbol_events = events_df
    
    if symbol_events.empty:
        return False
        
    spikes = symbol_events['datetime']
    
    diffs = abs(spikes - current_time)
    min_diff = diffs.min()
    
    return min_diff <= timedelta(minutes=buffer_minutes)

def is_weekend_embargo(current_time: pd.Timestamp) -> bool:
    if not isinstance(current_time, pd.Timestamp):
        current_time = pd.to_datetime(current_time)
        
    weekday = current_time.weekday()
    hour = current_time.hour
    minute = current_time.minute
    
    # Friday close at 4:30 PM ET = 21:30 UTC. (or 20:30 DST). We use 21:30 UTC.
    if weekday == 4:
        if hour > 21 or (hour == 21 and minute >= 30):
            return True
            
    if weekday == 5:
        return True
        
    if weekday == 6:
        # Sunday open is at 17:00 ET = 21:00 UTC (or 22:00 UTC).
        if hour < 21:
            return True
            
    return False
