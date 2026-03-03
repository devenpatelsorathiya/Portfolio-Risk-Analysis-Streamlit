import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.core.db import connect, init_db

def upsert_instrument(cur, ticker):
    cur.execute("INSERT OR IGNORE INTO instruments(ticker) VALUES (?)", (ticker,))
    cur.execute("SELECT instrument_id FROM instruments WHERE ticker=?", (ticker,))
    return cur.fetchone()[0]

def main():
    init_db()
    con = connect()
    cur = con.cursor()

    prices_file = ROOT_DIR / "data" / "prices.csv"
    if not prices_file.exists():
        raise FileNotFoundError(f"prices.csv not found at {prices_file}")

    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True).sort_index()
    prices = prices.dropna(how="all").ffill().dropna()

    returns = prices.pct_change().dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    for ticker in prices.columns:
        iid = upsert_instrument(cur, ticker)

        p = prices[[ticker]].dropna().copy()
        p["date"] = p.index.strftime("%Y-%m-%d")
        p.rename(columns={ticker: "adj_close"}, inplace=True)

        cur.executemany(
            "INSERT OR REPLACE INTO prices_daily(instrument_id, date, adj_close, close, volume) VALUES (?, ?, ?, NULL, NULL)",
            [(iid, d, float(v)) for d, v in zip(p["date"], p["adj_close"])]
        )

        r = returns[[ticker]].dropna().copy()
        lr = log_returns[[ticker]].dropna().copy()
        r["date"] = r.index.strftime("%Y-%m-%d")

        cur.executemany(
            "INSERT OR REPLACE INTO returns_daily(instrument_id, date, ret_simple, ret_log) VALUES (?, ?, ?, ?)",
            [(iid, d, float(rv), float(lrv)) for d, rv, lrv in zip(r["date"], r[ticker], lr[ticker])]
        )

    con.commit()
    con.close()
    print("Loaded prices and returns into SQLite database.")

if __name__ == "__main__":
    main()