import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from strategy_executor import execute_strategy
from autonomous_research_loop import ResearchLoopManager
import os

def extract_features(df: pd.DataFrame, trade_indices: list):
    """Extracts ML features for each trade from the original dataframe."""
    features = []
    
    # Pre-calculate some indicators for the whole df to save time
    df['rsi_14'] = df['close'].diff().clip(lower=0).rolling(14).mean() / (df['close'].diff().abs().rolling(14).mean() + 1e-9)
    df['atr_14'] = (df['high'] - df['low']).rolling(14).mean()
    df['vol_sma'] = df['volume'].rolling(20).mean()
    
    for idx in trade_indices:
        try:
            # We look at the candle exactly where the signal fired
            row = df.iloc[idx]
            dt = df.index[idx]
            
            feat = {
                'hour': dt.hour,
                'minute': dt.minute,
                'day_of_week': dt.dayofweek,
                'rsi': row.get('rsi_14', 50),
                'atr_ratio': row.get('atr_14', 1.0) / (row['close'] * 0.001),
                'vol_ratio': row['volume'] / (row.get('vol_sma', 1.0) + 1e-9),
                'is_london': 1 if 6 <= dt.hour < 11 else 0,
                'is_ny': 1 if 12 <= dt.hour < 17 else 0
            }
            features.append(feat)
        except Exception:
            # Fallback for missing data
            features.append({
                'hour': 12, 'minute': 0, 'day_of_week': 2, 'rsi': 50, 
                'atr_ratio': 1.0, 'vol_ratio': 1.0, 'is_london': 0, 'is_ny': 1
            })
            
    return pd.DataFrame(features)

def train_meta_labeler():
    print("Loading data...")
    train_df, _, _ = ResearchLoopManager()._ensure_data()
    
    with open('data/leaderboard.json', 'r', encoding='utf-8') as f:
        leaderboard = json.load(f)
        
    # Get #1 strategy
    champ = leaderboard[0]
    print(f"Training Meta-Labeler on Champion: {champ['name']}")
    
    res = execute_strategy(champ['code'], train_df)
    trades = res.get('trades', [])
    
    if not trades:
        print("No trades found to train on.")
        return
        
    print(f"Extracting features for {len(trades)} historical trades...")
    
    # 1 = Win (PnL > 0), 0 = Loss
    y = np.array([1 if t['pnl'] > 0 else 0 for t in trades])
    entry_indices = [t.get('entry_bar_idx', 0) for t in trades]
    
    X = extract_features(train_df, entry_indices)
    X.fillna(0, inplace=True)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training RandomForest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    
    print(f"\n[ML Meta-Labeler Results]")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision (Win Prediction): {prec:.2f}")
    
    # Analyze Impact
    original_wins = sum(y_test)
    original_total = len(y_test)
    original_wr = original_wins / original_total
    
    # Only take trades the ML model predicts as wins
    ml_approved_trades = [i for i in range(len(y_test)) if y_pred[i] == 1]
    ml_actual_wins = sum([y_test[i] for i in ml_approved_trades])
    ml_wr = ml_actual_wins / max(1, len(ml_approved_trades))
    
    print(f"\n[Impact on Win Rate]")
    print(f"Original Win Rate: {original_wr*100:.1f}%")
    print(f"ML Filtered Win Rate: {ml_wr*100:.1f}%")
    
    # Feature Importances
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print("\n[Top Features deciding Wins vs Losses]")
    print(feat_imp.head(3))
    
    print("\nMeta-labeling model successfully trained and ready for injection into executor!")
    return clf

if __name__ == "__main__":
    train_meta_labeler()
