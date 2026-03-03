import numpy as np
from src.data_loader import load_returns, load_portfolio_returns
from src.risk_metrics import historical_var, parametric_var
from src.monte_carlo import monte_carlo_var, advanced_mc_var

returns = load_returns()
portfolio_returns = load_portfolio_returns()

weights = np.array([0.25,0.25,0.25,0.25])

print("Historical VaR:", historical_var(portfolio_returns))
print("Parametric VaR:", parametric_var(portfolio_returns))
print("Monte Carlo VaR:", monte_carlo_var(portfolio_returns))
print("Advanced Monte Carlo VaR:", advanced_mc_var(returns, weights))