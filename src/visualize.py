import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_regimes(df: pd.DataFrame, regimes: np.ndarray, regime_labels=None):
    plt.figure(figsize=(16, 6))
    colors = ['#a6cee3', '#b2df8a', '#fb9a99']
    if regime_labels is None:
        regime_labels = ['Regime 0', 'Regime 1', 'Regime 2']
    for i, label in enumerate(np.unique(regimes)):
        mask = regimes == label
        plt.fill_between(df.index[mask], df['Close'][mask].min(), df['Close'][mask].max(), color=colors[i % len(colors)], alpha=0.2, label=regime_labels[i])
    plt.plot(df.index, df['Close'], color='black', lw=1.5, label='Close Price')
    plt.legend()
    plt.title('Market Regimes')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

def plot_cluster_distribution(regimes: np.ndarray, regime_labels=None):
    unique, counts = np.unique(regimes, return_counts=True)
    if regime_labels is None:
        regime_labels = [f'Regime {i}' for i in unique]
    plt.figure(figsize=(6, 4))
    plt.bar(regime_labels, counts, color=['#a6cee3', '#b2df8a', '#fb9a99'])
    plt.title('Regime Distribution')
    plt.ylabel('Count')
    plt.show()
