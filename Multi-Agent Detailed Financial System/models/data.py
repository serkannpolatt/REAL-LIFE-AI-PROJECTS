"""
Data handling module for FinAgents.
This module provides functions for fetching and processing financial data.
"""

import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()


def get_stock_data(tickers, period="1y"):
    """
    Fetches stock data for given tickers.

    Args:
        tickers (list): List of ticker symbols
        period (str): Time period for data, e.g., "1y" for 1 year

    Returns:
        DataFrame: DataFrame containing adjusted close prices
    """
    data = yf.download(tickers, period=period, interval="1d")
    if "Adj Close" in data:
        return data["Adj Close"]
    else:
        print("⚠️ Warning: 'Adj Close' not found. Returning closing prices.")
        return data["Close"]
