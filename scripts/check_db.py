import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.core.db import connect

con = connect()
cur = con.cursor()

cur.execute("SELECT COUNT(*) FROM instruments")
print("instruments:", cur.fetchone()[0])

cur.execute("SELECT COUNT(*) FROM prices_daily")
print("prices_daily:", cur.fetchone()[0])

cur.execute("SELECT COUNT(*) FROM returns_daily")
print("returns_daily:", cur.fetchone()[0])

con.close()