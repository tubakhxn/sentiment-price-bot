from src.data import fetch_data
from src.features import add_features, normalize_features, create_sliding_windows
from src.model import MarketRegimeModel
from src.visualize import plot_regimes, plot_cluster_distribution
import pandas as pd
import numpy as np

# Parameters
ticker = 'SPY'
start = '2015-01-01'
end = '2024-01-01'
window_size = 10
method = 'kmeans'  # or 'rf' for RandomForest

# 1. Fetch data
df = fetch_data(ticker, start, end)

# 2. Feature engineering
df_feat = add_features(df)
feature_cols = ['Return', 'Volatility', 'MA20', 'MA50', 'RSI']
df_feat = normalize_features(df_feat, feature_cols)
X = create_sliding_windows(df_feat, window_size, feature_cols)

# 3. Model (unsupervised KMeans)
model = MarketRegimeModel(method=method, n_clusters=3)
regimes = model.fit(X)

# Align regimes with original df
df_feat = df_feat.iloc[window_size-1:].copy()
df_feat['Regime'] = regimes

# 4. Visualization
plot_regimes(df_feat, regimes)
plot_cluster_distribution(regimes)

# 5. Evaluation
print('Regime value counts:')
print(df_feat['Regime'].value_counts())

# 6. Optional: Simple strategy example
# Only trade in trending regime (e.g., regime 0)
# This is a placeholder for further strategy logic
