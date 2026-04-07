import yfinance as yf
import pandas as pd

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical price data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end)
    return df
