"""
Visualization module for FinAgents.
This module provides functions for creating charts and visualizations.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend


def generate_charts(stock_data, portfolio_metrics, chart_dir="charts"):
    """
    Generate charts for portfolio analysis.

    Args:
        stock_data (DataFrame): DataFrame with stock prices
        portfolio_metrics (dict): Dictionary with portfolio metrics
        chart_dir (str): Directory to save charts

    Returns:
        dict: Dictionary with paths to generated charts
    """
    os.makedirs(chart_dir, exist_ok=True)
    chart_paths = {}

    # Get stocks list
    stocks = portfolio_metrics["stock_metrics"].keys()

    # Normalized Stock Performance
    plt.figure(figsize=(10, 6))
    normalized_prices = stock_data / stock_data.iloc[0]
    normalized_prices.plot(title="Normalized Stock Performance")
    plt.ylabel("Normalized Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    performance_path = os.path.join(chart_dir, "normalized_performance.png")
    plt.savefig(performance_path, dpi=300)
    plt.close()
    chart_paths["performance"] = performance_path

    # Portfolio Returns Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(portfolio_metrics["portfolio_returns"], kde=True, bins=50)
    plt.title("Portfolio Daily Returns Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    returns_path = os.path.join(chart_dir, "returns_distribution.png")
    plt.savefig(returns_path, dpi=300)
    plt.close()
    chart_paths["returns"] = returns_path

    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    correlation = portfolio_metrics["returns"].corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("Stock Correlation Matrix")
    plt.tight_layout()
    correlation_path = os.path.join(chart_dir, "correlation_matrix.png")
    plt.savefig(correlation_path, dpi=300)
    plt.close()
    chart_paths["correlation"] = correlation_path

    # Cumulative Returns
    plt.figure(figsize=(10, 6))
    cumulative_returns = (1 + portfolio_metrics["portfolio_returns"]).cumprod()
    cumulative_returns.plot(title="Portfolio Cumulative Returns")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    cumulative_path = os.path.join(chart_dir, "cumulative_returns.png")
    plt.savefig(cumulative_path, dpi=300)
    plt.close()
    chart_paths["cumulative"] = cumulative_path

    # Risk-Return Scatter Plot
    plt.figure(figsize=(10, 6))
    stock_metrics = portfolio_metrics["stock_metrics"]
    returns_list = [stock_metrics[stock]["annual_return"] * 100 for stock in stocks]
    volatilities = [stock_metrics[stock]["annual_volatility"] * 100 for stock in stocks]
    plt.scatter(volatilities, returns_list, s=100)
    for i, stock in enumerate(stocks):
        plt.annotate(
            stock,
            (volatilities[i], returns_list[i]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
        )
    plt.title("Risk-Return Profile")
    plt.xlabel("Annual Volatility (%)")
    plt.ylabel("Annual Return (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    risk_return_path = os.path.join(chart_dir, "risk_return.png")
    plt.savefig(risk_return_path, dpi=300)
    plt.close()
    chart_paths["risk_return"] = risk_return_path

    print(f"Charts saved to {chart_dir} directory")
    return chart_paths


def generate_allocation_chart(weights, chart_dir="charts"):
    """
    Generate a pie chart for portfolio allocation.

    Args:
        weights (dict): Dictionary with ticker symbols and weights
        chart_dir (str): Directory to save the chart

    Returns:
        str: Path to the generated chart
    """
    os.makedirs(chart_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.pie(
        list(weights.values()),
        labels=list(weights.keys()),
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.axis("equal")
    plt.title("Portfolio Allocation")
    plt.tight_layout()
    allocation_path = os.path.join(chart_dir, "current_allocation.png")
    plt.savefig(allocation_path, dpi=300)
    plt.close()

    return allocation_path
