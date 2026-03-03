import numpy as np
from scipy.stats import norm

def historical_var(portfolio_returns, alpha=0.05):
    return -np.quantile(portfolio_returns, alpha)

def parametric_var(portfolio_returns, alpha=0.05):
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    z = norm.ppf(alpha)
    return -(mu + z*sigma)