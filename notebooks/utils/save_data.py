import yfinance as yf
import json
import os

from datetime import datetime
from .paths import ROOT, DATA_DIR, RAW_DIR, PROCESSED_DIR

ASSET_CLASS = "equity"
INTERVAL = "1d"
START_DATE = "2025-01-01"

def save_price_data( tickers,
                     asset_class=ASSET_CLASS,  # equity, futures, options
                     interval=INTERVAL,         # Timeframe
                     start_date=START_DATE, 
                     end_date=datetime.now().strftime('%Y-%m-%d')):
    """
    Avaliable both Ticker and Tickers
    Save Asset type (asset_class) and Timeframe (interval)
    
    Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    # 1. Convert Tickers to list (If it were a single str)
    if isinstance(tickers, str):
        tickers = [tickers]
        
    print(f"--- Processing these {len(tickers)} Tickers | Type: {asset_class} | Timeframe: {interval} ---")

    # 2. Loop each Ticker for save each file.
    for ticker in tickers:
        
        # 3. Define Path for saving
        #    such as data/raw/equity/1d/AAPL.csv
        #    or data/raw/futures/1h/ES_F.csv
        save_dir = RAW_DIR / asset_class / interval
        
        # If there has special character.
        safe_ticker_name = ticker.replace('=F', '_F').replace('=X', '_X')
        save_path = save_dir / f"{safe_ticker_name}.csv"
        
        # 4. New a folder if not exists
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading: {ticker} ({interval})...")
        
        try:
            # 5. Load the data
            data = yf.download(ticker, 
                               start=start_date, 
                               end=end_date, 
                               interval=interval)
            
            if data.empty:
                print(f"Not found the {ticker} (Timeframe: {interval})")
                continue # Skip next Ticker
                
            # 6. Save a File
            data.to_csv(save_path)
            print(f"✅ Saved {ticker} At: {save_path}")
        
        except Exception as e:
            print(f"❌ Error to download {ticker}: {e}")
            
    print(f"--- Done ---\n")


def save_stock_fundamental(tickers):
    """
    Download Stock Fundemental (Info, Financials, Balance Sheet) with a single Ticker or Tickers.
    """
    
    # 1. Convert Tickers to list (If it were a single str)
    if isinstance(tickers, str):
        tickers = [tickers]
        
    print(f"--- Processing Fundemental {len(tickers)} Tickers ---")
    
    # 2. Define Path for saving
    save_dir = RAW_DIR / "equity" / "fundamental"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3. Loop each Ticker for save each file.
    for ticker in tickers:
        print(f"Dowloading the Fundemental: {ticker}...")
        try:
            t = yf.Ticker(ticker)
            
            # Saveing Info (JSON)
            info_data = t.info
            info_path = save_dir / f"{ticker}_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info_data, f, indent=4, ensure_ascii=False)

            # Saveing Financials (CSV)
            fin_data = t.financials
            fin_path = save_dir / f"{ticker}_financials.csv"
            fin_data.to_csv(fin_path)

            # Saveing Balance Sheet (CSV)
            bs_data = t.balance_sheet
            bs_path = save_dir / f"{ticker}_balance_sheet.csv"
            bs_data.to_csv(bs_path)
            
            print(f"✅ Saved Fundemental {ticker} .")

        except Exception as e:
            print(f"❌ Error to download fundemental {ticker}: {e}")
            
    print(f"--- Done ---\n")


# Multi Stocks
def save_price_data_grouped(tickers, 
                            group_name,
                            asset_class=ASSET_CLASS, 
                            interval=INTERVAL, 
                            start_date=START_DATE, 
                            end_date=datetime.now().strftime('%Y-%m-%d')):
    """
    Downloads price data (OHLCV) for multiple tickers (Batch)
    and saves them as a single, standardized file in the PROCESSED directory.
    
    (EN) Saves to: PROCESSED_DIR / {group_name}_{asset_class}_{interval}.parquet
    """
    
    # 1. Validate Tickers (must be a list)
    if not isinstance(tickers, list) or len(tickers) == 0:
        print(f"❌ Error: 'tickers' must be a list with at least one item.")
        return

    print(f"--- 🚀 Processing: {group_name} ({len(tickers)} Tickers) | Timeframe: {interval} ---")

    # 2. (NEW) Define the standard filename
    standard_filename = f"{group_name}_{asset_class}_{interval}.csv"
    
    # 3. (NEW) Define the save path in PROCESSED_DIR
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PROCESSED_DIR / standard_filename
    
    print(f"Downloading {tickers} ({interval})...")
    
    try:
        # 4. Download data
        data = yf.download(tickers, 
                           start=start_date, 
                           end=end_date, 
                           interval=interval)
        
        if data.empty:
            print(f"Not found {group_name} (Timeframe: {interval})")
            return 
        
        # 5. (Optional) Rename 'Adj Close' to 'Price'
        if 'Adj Close' in data.columns.levels[0]:
            print("Renaming 'Adj Close' to 'Price'...")
            data.columns = data.columns.set_levels(
                ['Price' if x == 'Adj Close' else x for x in data.columns.levels[0]],
                level=0
            )

        # 6. Save the file to the new standard path
        data.to_csv(save_path)
            
        # (Update Log for easy to read)
        print(f"✅ Saved '{group_name}' at: {save_path}") 
        print(f"--- Processed {group_name} ---\n")
        
        return data # Return the DataFrame for immediate use
    
    except Exception as e:
        print(f"❌ Failing to download {group_name}: {e}")
        return None
