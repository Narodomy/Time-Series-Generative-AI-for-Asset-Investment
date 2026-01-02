import os
import pandas as pd

from utils.paths import EQUITY_DIR, PRICE_DIR

def read_equity(symbol, freq="1d"):
    path = os.path.join(PRICE_DIR, freq, f"{symbol.upper()}.csv")
    cols = ["Date", "Close", "High", "Low", "Open", "Volume"]
    # Read file as a csv and set up the features.
    df = pd.read_csv(path, skiprows=2)
    df.columns = cols
    # Covert from Date feature as index of dataframe.
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    # Return the dataframe.
    return df
    
def read_fundamental(symbol):
    path = os.path.join(fundamental, f"{symbol.upper()}.csv")
    return path