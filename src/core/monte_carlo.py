import numpy as np

def monte_carlo_var(portfolio_returns, sims=10000, alpha=0.05):
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    sim = np.random.normal(mu, sigma, sims)
    return -np.quantile(sim, alpha)

def advanced_mc_var(returns, weights, sims=10000, alpha=0.05):
    mean = returns.mean().values
    cov = returns.cov().values
    sim = np.random.multivariate_normal(mean, cov, sims)
    port = sim @ weights
    return -np.quantile(port, alpha)