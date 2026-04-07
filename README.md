## dev/creator = tubakhxn

# ML-Based Market Regime Detection System

This project is a machine learning system for classifying financial market regimes (trending, sideways, volatile) using unsupervised learning and technical indicators. It helps users understand market behavior with ML, not just traditional indicators.

## What is this project about?
- Fetches historical price data (Yahoo Finance)
- Engineers features: returns, volatility, moving averages, RSI
- Normalizes and creates sliding windows for time-series context
- Uses KMeans clustering (or RandomForest if labeled) to classify regimes
- Visualizes price with regime backgrounds and regime distribution
- Evaluates and analyzes regime behavior
- (Optional) Can be extended to include simple trading strategies

## How to fork and run
1. Fork this repository on GitHub (click the "Fork" button at the top right)
2. Clone your fork:
   ```
   git clone https://github.com/YOUR_USERNAME/ML-Based-Market-Regime-Detection-System.git
   ```
3. Install requirements:
   ```
   pip install -r requirements.txt
   ```
4. Run the main script:
   ```
   python main.py
   ```

## Relevant Links & Resources
- [yfinance documentation](https://github.com/ranaroussi/yfinance)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [KMeans clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [RSI explanation](https://www.investopedia.com/terms/r/rsi.asp)
- [Moving averages](https://www.investopedia.com/terms/m/movingaverage.asp)
- [Rolling volatility](https://www.investopedia.com/terms/v/volatility.asp)
- [Time series sliding window](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)

## Project Structure
- `src/data.py` - Data fetching
- `src/features.py` - Feature engineering
- `src/model.py` - ML model
- `src/visualize.py` - Visualization
- `main.py` - Pipeline entry point

---

For questions, open an issue or contact tubakhxn.
