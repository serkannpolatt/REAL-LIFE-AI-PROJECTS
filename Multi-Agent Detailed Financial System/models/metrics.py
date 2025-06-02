"""
Portfolio metrics calculation module for FinAgents.
This module provides functions for calculating various portfolio performance metrics.
"""

import pandas as pd
import numpy as np


def calculate_portfolio_metrics(stock_data, weights):
    """
    Calculate portfolio performance metrics.

    Args:
        stock_data (DataFrame): Dataframe with stock prices
        weights (dict): Dictionary with ticker symbols as keys and portfolio weights as values

    Returns:
        dict: Dictionary containing various portfolio metrics
    """
    returns = stock_data.pct_change().dropna()
    portfolio_returns = pd.Series(0.0, index=returns.index)
    for stock, weight in weights.items():
        if stock in returns.columns:
            portfolio_returns += returns[stock] * weight

    annual_return = portfolio_returns.mean() * 252
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_volatility

    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_drawdown = drawdown.min()

    stock_metrics = {}
    for stock in weights.keys():
        if stock in returns.columns:
            stock_returns = returns[stock]
            stock_metrics[stock] = {
                "annual_return": stock_returns.mean() * 252,
                "annual_volatility": stock_returns.std() * np.sqrt(252),
                "sharpe_ratio": (stock_returns.mean() * 252)
                / (stock_returns.std() * np.sqrt(252)),
                "beta": stock_returns.cov(portfolio_returns) / portfolio_returns.var(),
            }

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "returns": returns,
        "portfolio_returns": portfolio_returns,
        "stock_metrics": stock_metrics,
    }
