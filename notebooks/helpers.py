from common.paths import STOCKS_PRICES
import os
import yfinance as yf


def save_stock_prices(symbol, security, dataset_dir=STOCKS_PRICES):
    try:
        stock_prices = yf.Ticker(symbol).history(period="max")
        if stock_prices.empty:
            print(f"No data found for {symbol}")
            return None
        # New path file
        file_path = os.path.join(dataset_dir, f"{symbol.lower()} ({security.lower()}).csv")
        stock_prices.to_csv(file_path)
        print(f"Saved {symbol} history to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None