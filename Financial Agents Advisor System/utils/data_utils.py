"""
Data Utilities
=============

Helper functions for data processing and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime, timedelta


class DataProcessor:
    """Handles data processing and validation for financial data."""

    @staticmethod
    def validate_stock_symbol(symbol: str) -> bool:
        """
        Validate stock symbol format.

        Args:
            symbol: Stock symbol to validate

        Returns:
            True if valid format, False otherwise
        """
        if not symbol or not isinstance(symbol, str):
            return False

        # Basic validation: 1-5 characters, alphanumeric
        symbol = symbol.strip().upper()
        return len(symbol) >= 1 and len(symbol) <= 5 and symbol.isalnum()

    @staticmethod
    def clean_financial_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare financial data for analysis.

        Args:
            data: Raw financial data DataFrame

        Returns:
            Cleaned DataFrame
        """
        if data.empty:
            return data

        # Remove rows with all NaN values
        data = data.dropna(how="all")

        # Forward fill missing values for price data
        price_columns = ["Open", "High", "Low", "Close", "Adj Close"]
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].fillna(method="ffill")

        # Remove outliers (values more than 3 standard deviations from mean)
        for col in price_columns:
            if col in data.columns:
                mean = data[col].mean()
                std = data[col].std()
                data = data[abs(data[col] - mean) <= 3 * std]

        return data

    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
        """
        Calculate returns from price series.

        Args:
            prices: Price series
            method: 'simple' or 'log' returns

        Returns:
            Returns series
        """
        if method == "simple":
            return prices.pct_change()
        elif method == "log":
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError("Method must be 'simple' or 'log'")

    @staticmethod
    def get_date_range(period: str) -> tuple:
        """
        Get start and end dates for common periods.

        Args:
            period: Period string ('1d', '1w', '1m', '3m', '6m', '1y', '2y', '5y')

        Returns:
            Tuple of (start_date, end_date)
        """
        end_date = datetime.now()

        period_map = {
            "1d": timedelta(days=1),
            "1w": timedelta(weeks=1),
            "1m": timedelta(days=30),
            "3m": timedelta(days=90),
            "6m": timedelta(days=180),
            "1y": timedelta(days=365),
            "2y": timedelta(days=730),
            "5y": timedelta(days=1825),
        }

        if period not in period_map:
            raise ValueError(f"Unsupported period: {period}")

        start_date = end_date - period_map[period]
        return start_date, end_date


class MarketDataValidator:
    """Validates market data quality and completeness."""

    @staticmethod
    def validate_price_data(data: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate price data DataFrame.

        Args:
            data: Price data DataFrame

        Returns:
            Dictionary with validation results
        """
        results = {
            "has_data": not data.empty,
            "has_required_columns": False,
            "no_negative_prices": True,
            "logical_ohlc": True,
            "no_excessive_gaps": True,
        }

        if not results["has_data"]:
            return results

        # Check required columns
        required_cols = ["Open", "High", "Low", "Close"]
        results["has_required_columns"] = all(
            col in data.columns for col in required_cols
        )

        if not results["has_required_columns"]:
            return results

        # Check for negative prices
        for col in required_cols:
            if (data[col] < 0).any():
                results["no_negative_prices"] = False
                break

        # Check OHLC logic (High >= Open, Low, Close and Low <= Open, High, Close)
        ohlc_valid = (
            data["High"] >= data[["Open", "Low", "Close"]].max(axis=1)
        ).all() and (data["Low"] <= data[["Open", "High", "Close"]].min(axis=1)).all()
        results["logical_ohlc"] = ohlc_valid

        # Check for excessive price gaps (more than 50% change)
        if "Close" in data.columns:
            returns = data["Close"].pct_change().abs()
            results["no_excessive_gaps"] = (returns < 0.5).all()

        return results

    @staticmethod
    def get_data_quality_score(data: pd.DataFrame) -> float:
        """
        Calculate data quality score (0-1).

        Args:
            data: Price data DataFrame

        Returns:
            Quality score between 0 and 1
        """
        validation = MarketDataValidator.validate_price_data(data)
        score = sum(validation.values()) / len(validation)
        return score


def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format currency amount for display.

    Args:
        amount: Amount to format
        currency: Currency code

    Returns:
        Formatted currency string
    """
    if currency == "USD":
        return f"${amount:,.2f}"
    elif currency == "EUR":
        return f"€{amount:,.2f}"
    elif currency == "GBP":
        return f"£{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format percentage for display.

    Args:
        value: Percentage value (0.1 = 10%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"
