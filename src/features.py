import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'], window=14)
    df = df.dropna()
    return df

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df

def create_sliding_windows(df: pd.DataFrame, window_size: int, feature_cols: list) -> np.ndarray:
    X = []
    for i in range(len(df) - window_size + 1):
        X.append(df[feature_cols].iloc[i:i+window_size].values.flatten())
    return np.array(X)
