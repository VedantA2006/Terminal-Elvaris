import pandas as pd
import os
from datetime import timedelta

_NEWS_CACHE = None

def get_news_events():
    global _NEWS_CACHE
    if _NEWS_CACHE is not None:
        return _NEWS_CACHE

    csv_path = 'EquityEdge_5_Instrument_HighImpact_News.csv'
    if not os.path.exists(csv_path):
        print(f"Warning: News file not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    
    # Filter for high impact news
    high_impact = df[df['Importance'] == 'CALENDAR_IMPORTANCE_HIGH'].copy()
    
    if high_impact.empty:
        _NEWS_CACHE = pd.DataFrame()
        return _NEWS_CACHE

    # Parse News_Time_MT5_Server to datetime
    high_impact['datetime'] = pd.to_datetime(high_impact['News_Time_MT5_Server'])
    high_impact['datetime'] = pd.to_datetime(high_impact['datetime'], utc=True)
    
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
        
    mapped_inst = get_mapped_instrument(symbol)
    symbol_events = events_df[events_df['Instrument'] == mapped_inst]
    
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
    
    if weekday == 4:
        if hour > 21 or (hour == 21 and minute >= 45):
            return True
            
    if weekday == 5:
        return True
        
    if weekday == 6:
        if hour < 22:
            return True
            
    return False
