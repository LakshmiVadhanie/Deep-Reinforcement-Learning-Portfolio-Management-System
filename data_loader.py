import numpy as np
import pandas as pd
import yfinance as yf

class DataLoader:
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        print("Fetching market data")

        # Fetch stock data
        stock_data = {}
        for symbol in self.symbols:
            data = yf.download(symbol, start=self.start_date, end=self.end_date, progress=False)
            if not data.empty:
                stock_data[symbol] = data
                print(f"  {symbol}: {data.shape[0]} trading days")

        # Fetch VIX and SPY
        vix_data = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
        spy_data = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)

        return stock_data, vix_data, spy_data

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100

    def preprocess_data(self, stock_data, vix_data, spy_data):
        print("Engineering features")

        # Extract price data
        price_data = pd.DataFrame()
        for symbol, data in stock_data.items():
            if 'Adj Close' in data.columns:
                price_data[symbol] = data['Adj Close']
            else:
                price_data[symbol] = data['Close']

        # Calculate returns
        returns_data = price_data.pct_change()

        # Create feature matrix
        features = pd.DataFrame(index=price_data.index)

        # Technical indicators for each stock
        for symbol in price_data.columns:
            features[f'{symbol}_return'] = returns_data[symbol]
            features[f'{symbol}_ma_5'] = price_data[symbol].rolling(5).mean() / price_data[symbol]
            features[f'{symbol}_ma_20'] = price_data[symbol].rolling(20).mean() / price_data[symbol]
            features[f'{symbol}_volatility'] = returns_data[symbol].rolling(20).std()
            features[f'{symbol}_rsi'] = self.calculate_rsi(price_data[symbol])

        # Add VIX volatility index
        if not vix_data.empty:
            vix_close = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
            vix_aligned = vix_close.reindex(price_data.index, method='ffill')
            features['vix'] = vix_aligned
            features['vix_ma'] = vix_aligned.rolling(10).mean()
        else:
            market_vol = returns_data.std(axis=1).rolling(20).mean() * 100
            features['vix'] = market_vol
            features['vix_ma'] = market_vol.rolling(10).mean()

        # Add market benchmark (SPY)
        if not spy_data.empty:
            spy_close = spy_data['Adj Close'] if 'Adj Close' in spy_data.columns else spy_data['Close']
            spy_aligned = spy_close.reindex(price_data.index, method='ffill')
            features['market_return'] = spy_aligned.pct_change()
            features['market_ma'] = spy_aligned.rolling(20).mean() / spy_aligned

        # Fill missing values
        features = features.fillna(method='ffill').fillna(0)

        return price_data, features
