import numpy as np

def portfolio_returns(returns, weights):
    return returns @ weights

def portfolio_volatility(returns, weights):
    cov = returns.cov()
    var = weights.T @ cov.values @ weights
    return np.sqrt(var)