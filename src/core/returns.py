import numpy as np

def compute_returns(prices):
    returns = prices.pct_change().dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return returns, log_returns