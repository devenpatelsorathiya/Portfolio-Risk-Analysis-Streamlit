import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import norm, chi2

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_FILE = DATA_DIR / "alphapulse.db"

st.set_page_config(page_title="AlphaPulse", layout="wide")


@st.cache_data(show_spinner=False)
def load_returns_from_db(db_file: Path) -> pd.DataFrame:
    if not db_file.exists():
        raise FileNotFoundError(f"Missing database file: {db_file}")

    con = sqlite3.connect(str(db_file))
    q = """
    SELECT i.ticker, r.date, r.ret_simple
    FROM returns_daily r
    JOIN instruments i ON i.instrument_id = r.instrument_id
    """
    df = pd.read_sql(q, con)
    con.close()

    df["date"] = pd.to_datetime(df["date"])
    returns = df.pivot(index="date", columns="ticker", values="ret_simple").sort_index()
    returns = returns.dropna(how="all").dropna()
    return returns


def portfolio_series(returns: pd.DataFrame, w: np.ndarray) -> pd.Series:
    rp = returns @ w
    rp.name = "portfolio_return"
    return rp


def historical_var(rp: pd.Series, alpha: float) -> float:
    x = rp.dropna().values
    if x.size == 0:
        return float("nan")
    return -float(np.quantile(x, alpha))


def parametric_var(rp: pd.Series, alpha: float) -> float:
    rp = rp.dropna()
    if rp.empty:
        return float("nan")
    mu = float(rp.mean())
    sigma = float(rp.std(ddof=1))
    z = float(norm.ppf(alpha))
    return -(mu + z * sigma)


def historical_es(rp: pd.Series, alpha: float) -> float:
    x = rp.dropna().values
    if x.size == 0:
        return float("nan")
    q = np.quantile(x, alpha)
    tail = x[x <= q]
    if tail.size == 0:
        return float("nan")
    return -float(tail.mean())


def mc_var_normal(rp: pd.Series, alpha: float, sims: int, seed: int | None = None):
    rp = rp.dropna()
    if rp.empty:
        return float("nan"), np.array([])
    mu = float(rp.mean())
    sigma = float(rp.std(ddof=1))
    rng = np.random.default_rng(seed)
    sim = rng.normal(loc=mu, scale=sigma, size=int(sims))
    return -float(np.quantile(sim, alpha)), sim


def mc_var_correlated(returns: pd.DataFrame, w: np.ndarray, alpha: float, sims: int, seed: int | None = None):
    if returns.dropna().empty:
        return float("nan"), np.array([])
    mean = returns.mean().values
    cov = returns.cov().values
    rng = np.random.default_rng(seed)
    sim = rng.multivariate_normal(mean=mean, cov=cov, size=int(sims))
    port = sim @ w
    return -float(np.quantile(port, alpha)), port


def var_backtest_exceptions(rp: pd.Series, var_value: float):
    losses = -rp.dropna()
    exc = losses > var_value
    n = int(exc.shape[0])
    x = int(exc.sum())
    return n, x, exc


def kupiec_pof_test(n: int, x: int, alpha: float):
    if n <= 0:
        return float("nan"), float("nan")
    p = float(alpha)
    phat = x / n

    if phat == 0 or phat == 1:
        return float("inf"), 0.0

    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - phat) + x * np.log(phat)
    lr = float(-2 * (ll0 - ll1))
    p_value = float(1 - chi2.cdf(lr, df=1))
    return lr, p_value


def max_drawdown_from_returns(rp: pd.Series) -> float:
    rp = rp.dropna()
    if rp.empty:
        return float("nan")
    equity = (1.0 + rp).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def annualized_sharpe(rp: pd.Series, ann: int = 252) -> float:
    rp = rp.dropna()
    if rp.empty:
        return float("nan")
    mu = float(rp.mean()) * ann
    vol = float(rp.std(ddof=1)) * np.sqrt(ann)
    if vol == 0:
        return float("nan")
    return mu / vol


def ensure_portfolio_metrics_schema(cur: sqlite3.Cursor):
    cur.execute(
        """
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
        """
    )

    cur.execute("PRAGMA table_info(portfolio_metrics_daily)")
    existing_cols = {row[1] for row in cur.fetchall()}

    required_cols = {
        "vol_30d": "REAL",
        "var_hist_95": "REAL",
        "var_param_95": "REAL",
        "var_mc_95": "REAL",
        "var_mc_corr_95": "REAL",
        "es_hist_95": "REAL",
        "max_drawdown": "REAL",
        "sharpe": "REAL",
    }

    for col, coltype in required_cols.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE portfolio_metrics_daily ADD COLUMN {col} {coltype}")


def save_metrics_to_db(
    db_file: Path,
    asof_date: pd.Timestamp,
    vol_30d: float,
    var_hist_95: float,
    var_param_95: float,
    var_mc_95: float,
    var_mc_corr_95: float,
    es_hist_95: float,
    max_drawdown: float,
    sharpe: float,
):
    con = sqlite3.connect(str(db_file))
    cur = con.cursor()

    ensure_portfolio_metrics_schema(cur)

    cur.execute(
        """
        INSERT OR REPLACE INTO portfolio_metrics_daily
        (date, vol_30d, var_hist_95, var_param_95, var_mc_95, var_mc_corr_95, es_hist_95, max_drawdown, sharpe)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            asof_date.strftime("%Y-%m-%d"),
            float(vol_30d) if np.isfinite(vol_30d) else None,
            float(var_hist_95) if np.isfinite(var_hist_95) else None,
            float(var_param_95) if np.isfinite(var_param_95) else None,
            float(var_mc_95) if np.isfinite(var_mc_95) else None,
            float(var_mc_corr_95) if np.isfinite(var_mc_corr_95) else None,
            float(es_hist_95) if np.isfinite(es_hist_95) else None,
            float(max_drawdown) if np.isfinite(max_drawdown) else None,
            float(sharpe) if np.isfinite(sharpe) else None,
        ),
    )

    con.commit()
    con.close()


@st.cache_data(show_spinner=False)
def load_metrics_history(db_file: Path) -> pd.DataFrame:
    if not db_file.exists():
        return pd.DataFrame()

    con = sqlite3.connect(str(db_file))
    cur = con.cursor()

    cur.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='portfolio_metrics_daily'
        """
    )
    exists = cur.fetchone() is not None
    if not exists:
        con.close()
        return pd.DataFrame()

    cur.execute("PRAGMA table_info(portfolio_metrics_daily)")
    cols = [row[1] for row in cur.fetchall()]
    con.close()

    wanted = [
        "date",
        "vol_30d",
        "var_hist_95",
        "var_param_95",
        "var_mc_95",
        "var_mc_corr_95",
        "es_hist_95",
        "max_drawdown",
        "sharpe",
    ]
    select_cols = [c for c in wanted if c in cols]
    if not select_cols:
        return pd.DataFrame()

    con = sqlite3.connect(str(db_file))
    df = pd.read_sql(
        f"SELECT {', '.join(select_cols)} FROM portfolio_metrics_daily ORDER BY date ASC",
        con,
    )
    con.close()

    if "date" in df.columns and not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def apply_stress_scenario(returns_sel: pd.DataFrame, shock: float, vol_mult: float, corr_to: float | None) -> pd.DataFrame:
    r = returns_sel.copy()

    if corr_to is not None:
        cov = r.cov().values
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-12
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1.0, 1.0)

        target = float(corr_to)
        corr_new = np.full_like(corr, target, dtype=float)
        np.fill_diagonal(corr_new, 1.0)

        cov_new = np.outer(std, std) * corr_new
        mean = r.mean().values

        rng = np.random.default_rng(123)
        sim = rng.multivariate_normal(mean=mean, cov=cov_new, size=r.shape[0])
        r = pd.DataFrame(sim, index=r.index, columns=r.columns)

    if vol_mult != 1.0:
        r = r * float(vol_mult)

    if shock != 0.0:
        r = r + float(shock)

    return r.dropna()


try:
    returns = load_returns_from_db(DB_FILE)
except Exception as e:
    st.error(str(e))
    st.stop()

st.title("AlphaPulse Risk Platform")

with st.sidebar:
    st.header("Portfolio")

    assets = returns.columns.tolist()
    selected_assets = st.multiselect("Assets", assets, default=assets)

    if len(selected_assets) < 2:
        st.warning("Select at least 2 assets")
        st.stop()

    returns_sel_raw = returns[selected_assets].dropna()

    st.subheader("Weights")
    w_vals = []
    default_w = 1.0 / len(selected_assets)
    for a in selected_assets:
        w_vals.append(
            st.number_input(
                a,
                min_value=0.0,
                max_value=1.0,
                value=float(default_w),
                step=0.01,
            )
        )
    w = np.array(w_vals, dtype=float)

    if float(w.sum()) <= 0.0:
        st.warning("Sum of weights must be > 0")
        st.stop()

    w = w / w.sum()

    st.subheader("Risk Settings")
    confidence = st.slider("Confidence Level", min_value=90, max_value=99, value=95, step=1)
    alpha = 1 - (confidence / 100)

    sims = st.selectbox("Monte Carlo Simulations", [5000, 10000, 25000, 50000], index=1)
    portfolio_value = st.number_input("Portfolio Value (INR)", min_value=10000, value=1000000, step=10000)

    st.subheader("Reproducibility")
    use_seed = st.checkbox("Use fixed random seed", value=True)
    seed = 42 if use_seed else None

    st.subheader("Stress Testing")
    scenario = st.selectbox(
        "Scenario",
        ["Normal", "2008 Crash", "Covid Shock", "Rates Shock", "Custom"],
        index=0,
    )

    enable_stress = scenario != "Normal"

    if scenario == "2008 Crash":
        shock_pct = -4.0
        vol_mult = 3.0
        corr_mode = "Set all pairwise corr to 0.90"
    elif scenario == "Covid Shock":
        shock_pct = -3.0
        vol_mult = 2.5
        corr_mode = "Set all pairwise corr to 0.90"
    elif scenario == "Rates Shock":
        shock_pct = -1.0
        vol_mult = 2.0
        corr_mode = "Set all pairwise corr to 0.70"
    else:
        shock_pct = st.slider("Additive daily shock (%)", min_value=-10.0, max_value=10.0, value=-3.0, step=0.5)
        vol_mult = st.slider("Volatility multiplier", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        corr_mode = st.selectbox(
            "Correlation override",
            ["None", "Set all pairwise corr to 0.70", "Set all pairwise corr to 0.90"],
            index=0,
        )

    st.subheader("Database Logging")
    auto_save = st.checkbox("Auto-save latest metrics to DB", value=True)

corr_to = None
if corr_mode == "Set all pairwise corr to 0.70":
    corr_to = 0.70
elif corr_mode == "Set all pairwise corr to 0.90":
    corr_to = 0.90

returns_sel = returns_sel_raw
if enable_stress:
    returns_sel = apply_stress_scenario(
        returns_sel_raw,
        shock=shock_pct / 100.0,
        vol_mult=float(vol_mult),
        corr_to=corr_to,
    )

rp = portfolio_series(returns_sel, w)
ANN = 252

vol_daily = float(rp.std(ddof=1))
vol_annual = vol_daily * np.sqrt(ANN)

hvar = historical_var(rp, alpha)
pvar = parametric_var(rp, alpha)
hes = historical_es(rp, alpha)

mc_var_norm, sim_norm = mc_var_normal(rp, alpha, int(sims), seed=seed)
mc_var_corr, sim_corr = mc_var_correlated(returns_sel, w, alpha, int(sims), seed=seed)

rp_clean = rp.dropna()
asof_date = pd.to_datetime(rp_clean.index.max()) if not rp_clean.empty else pd.NaT

vol_30d = float("nan")
if rp_clean.shape[0] >= 30:
    vol_30d = float(rp_clean.rolling(30).std(ddof=1).iloc[-1]) * np.sqrt(ANN)

max_dd = max_drawdown_from_returns(rp)
sharpe = annualized_sharpe(rp, ANN)

if auto_save and pd.notna(asof_date):
    save_metrics_to_db(
        DB_FILE,
        asof_date=asof_date,
        vol_30d=vol_30d,
        var_hist_95=hvar,
        var_param_95=pvar,
        var_mc_95=mc_var_norm,
        var_mc_corr_95=mc_var_corr,
        es_hist_95=hes,
        max_drawdown=max_dd,
        sharpe=sharpe,
    )
    load_metrics_history.clear()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Annual Volatility", f"{vol_annual:.2%}")
col2.metric(f"Historical VaR ({confidence}%)", f"{hvar:.2%}", f"₹{(hvar * portfolio_value):,.0f}")
col3.metric(f"Parametric VaR ({confidence}%)", f"{pvar:.2%}", f"₹{(pvar * portfolio_value):,.0f}")
col4.metric(f"MC VaR Correlated ({confidence}%)", f"{mc_var_corr:.2%}", f"₹{(mc_var_corr * portfolio_value):,.0f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric(f"Historical ES ({confidence}%)", f"{hes:.2%}", f"₹{(hes * portfolio_value):,.0f}")
col6.metric(f"MC VaR Normal ({confidence}%)", f"{mc_var_norm:.2%}", f"₹{(mc_var_norm * portfolio_value):,.0f}")
col7.metric("Max Drawdown", f"{max_dd:.2%}")
col8.metric("Sharpe (ann.)", f"{sharpe:.2f}" if np.isfinite(sharpe) else "nan")

st.caption(f"Last saved date in DB: {asof_date.strftime('%Y-%m-%d') if pd.notna(asof_date) else 'NA'}")

st.subheader("Scenario Report")

base_rp = portfolio_series(returns_sel_raw, w)
base_hvar = historical_var(base_rp, alpha)
base_hes = historical_es(base_rp, alpha)

stressed_rp = rp
stressed_hvar = hvar
stressed_hes = hes

s1, s2, s3, s4 = st.columns(4)
s1.metric("Base Hist VaR", f"{base_hvar:.2%}", f"₹{(base_hvar * portfolio_value):,.0f}")
s2.metric("Scenario Hist VaR", f"{stressed_hvar:.2%}", f"₹{(stressed_hvar * portfolio_value):,.0f}")
s3.metric("Base Hist ES", f"{base_hes:.2%}", f"₹{(base_hes * portfolio_value):,.0f}")
s4.metric("Scenario Hist ES", f"{stressed_hes:.2%}", f"₹{(stressed_hes * portfolio_value):,.0f}")

if enable_stress:
    dvar = (stressed_hvar - base_hvar) * portfolio_value
    des = (stressed_hes - base_hes) * portfolio_value
    st.write(f"Delta VaR (₹): {dvar:,.0f}")
    st.write(f"Delta ES (₹): {des:,.0f}")

fig = plt.figure(figsize=(10, 4))
plt.hist(base_rp.dropna().values, bins=60, alpha=0.6, label="Base")
plt.hist(stressed_rp.dropna().values, bins=60, alpha=0.6, label="Scenario")
plt.title("Return Distribution: Base vs Scenario")
plt.legend()
plt.tight_layout()
st.pyplot(fig)

left, right = st.columns(2)

with left:
    st.subheader("Portfolio Return Series")
    fig = plt.figure(figsize=(10, 4))
    plt.plot(rp.index, rp.values)
    plt.title("Daily Portfolio Returns")
    plt.tight_layout()
    st.pyplot(fig)

with right:
    st.subheader("Monte Carlo Distribution (Correlated)")
    fig = plt.figure(figsize=(10, 4))
    if sim_corr.size > 0:
        plt.hist(sim_corr, bins=60)
    plt.title("Simulated Portfolio Returns")
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("Correlation Heatmap")
corr = returns_sel.corr()
fig = plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5)
plt.tight_layout()
st.pyplot(fig)

st.subheader("VaR Backtesting (Historical VaR)")
n, x, exc = var_backtest_exceptions(rp, hvar)
lr_stat, lr_p = kupiec_pof_test(n, x, alpha)

b1, b2, b3, b4 = st.columns(4)
b1.metric("Observations", f"{n}")
b2.metric("Exceptions", f"{x}")
b3.metric("Exception Rate", f"{(x / n if n else 0):.2%}")
b4.metric("Kupiec POF p-value", f"{lr_p:.4f}")

fig = plt.figure(figsize=(10, 3))
plt.plot(exc.index, exc.astype(int).values)
plt.title("VaR Exceptions Over Time (1 = breach)")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Weights")
weights_df = pd.DataFrame({"Asset": selected_assets, "Weight": w})
st.dataframe(weights_df, use_container_width=True)

st.subheader("Stored Risk Metrics History (Database)")
hist = load_metrics_history(DB_FILE)

if hist.empty:
    st.write("No history saved yet.")
else:
    st.dataframe(hist.tail(60), use_container_width=True)

    def _clean_series(df: pd.DataFrame, date_col: str, y_col: str) -> pd.DataFrame:
        out = df.copy()
        if date_col in out.columns:
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        if y_col in out.columns:
            out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
        out = out.dropna(subset=[date_col, y_col]).sort_values(date_col)
        return out

    vdf = _clean_series(hist, "date", "vol_30d")
    if vdf.shape[0] == 0:
        st.info("No valid vol_30d values saved yet (all NULL/NaN).")
    else:
        fig = plt.figure(figsize=(10, 4))
        if vdf.shape[0] == 1:
            plt.scatter(vdf["date"], vdf["vol_30d"])
        else:
            plt.plot(vdf["date"], vdf["vol_30d"])
        plt.title("Volatility (30d) History")
        plt.tight_layout()
        st.pyplot(fig)

    vardf = hist.copy()
    for c in ["var_hist_95", "var_mc_corr_95"]:
        if c in vardf.columns:
            vardf[c] = pd.to_numeric(vardf[c], errors="coerce")
    vardf["date"] = pd.to_datetime(vardf["date"], errors="coerce")
    vardf = vardf.dropna(subset=["date"]).sort_values("date")

    if ("var_hist_95" in vardf.columns) and ("var_mc_corr_95" in vardf.columns):
        v1 = vardf.dropna(subset=["var_hist_95"])
        v2 = vardf.dropna(subset=["var_mc_corr_95"])

        if v1.shape[0] == 0 and v2.shape[0] == 0:
            st.info("No valid VaR history values saved yet (all NULL/NaN).")
        else:
            fig = plt.figure(figsize=(10, 4))
            if v1.shape[0] == 1:
                plt.scatter(v1["date"], v1["var_hist_95"], label="Hist VaR")
            elif v1.shape[0] > 1:
                plt.plot(v1["date"], v1["var_hist_95"], label="Hist VaR")

            if v2.shape[0] == 1:
                plt.scatter(v2["date"], v2["var_mc_corr_95"], label="MC Corr VaR")
            elif v2.shape[0] > 1:
                plt.plot(v2["date"], v2["var_mc_corr_95"], label="MC Corr VaR")

            plt.title("VaR History")
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)