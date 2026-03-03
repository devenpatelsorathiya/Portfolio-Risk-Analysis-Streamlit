from pathlib import Path
import sqlite3

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DB_FILE = DATA_DIR / "alphapulse.db"

def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_FILE)

def init_db():
    con = connect()
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS instruments (
        instrument_id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT UNIQUE NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS prices_daily (
        instrument_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        adj_close REAL NOT NULL,
        close REAL,
        volume REAL,
        PRIMARY KEY (instrument_id, date),
        FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS returns_daily (
        instrument_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        ret_simple REAL NOT NULL,
        ret_log REAL NOT NULL,
        PRIMARY KEY (instrument_id, date),
        FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS portfolio_metrics_daily (
        date TEXT PRIMARY KEY,
        vol_30d REAL,
        var_hist_95 REAL,
        var_param_95 REAL,
        var_mc_95 REAL,
        var_mc_corr_95 REAL,
        max_drawdown REAL,
        sharpe REAL
    )
    """)

    con.commit()
    con.close()