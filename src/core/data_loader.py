import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def load_prices():
    return pd.read_csv(DATA_DIR / "prices.csv", index_col=0, parse_dates=True)

def load_returns():
    return pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)

def load_portfolio_returns():
    return pd.read_csv(DATA_DIR / "portfolio_returns.csv", index_col=0, parse_dates=True)["portfolio_return"]