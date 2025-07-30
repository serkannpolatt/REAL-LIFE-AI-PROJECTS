"""
Finance Advisor AI Agent - Configuration Module
==============================================

This module manages the general configuration settings of the application.
Environment variables, default values, and system settings are defined here.
"""

import os
from typing import Dict
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration settings."""

    # API Keys
    openai_api_key: str
    groq_api_key: str

    # System Settings
    debug_mode: bool = False
    log_level: str = "INFO"
    max_response_length: int = 4000

    # Model Settings
    groq_model: str = "llama3-groq-70b-8192-tool-use-preview"
    temperature: float = 0.1

    # Timeout Settings
    api_timeout: int = 30
    max_retries: int = 3


def load_config() -> AppConfig:
    """
    Load configuration from environment variables.

    Returns:
        AppConfig: Loaded configuration

    Raises:
        ValueError: If required environment variables are missing
    """
    # Check required API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable must be defined")
    if not groq_key:
        raise ValueError("GROQ_API_KEY environment variable must be defined")

    return AppConfig(
        openai_api_key=openai_key,
        groq_api_key=groq_key,
        debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        max_response_length=int(os.getenv("MAX_RESPONSE_LENGTH", "4000")),
    )


def get_financial_symbols() -> Dict[str, str]:
    """
    Return list of popular financial symbols.

    Returns:
        Dict[str, str]: Symbol code and company name mapping
    """
    return {
        # Technology
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc.",
        # Financial
        "JPM": "JPMorgan Chase & Co.",
        "BAC": "Bank of America Corp.",
        "WFC": "Wells Fargo & Company",
        "GS": "Goldman Sachs Group Inc.",
        # Healthcare
        "JNJ": "Johnson & Johnson",
        "PFE": "Pfizer Inc.",
        "UNH": "UnitedHealth Group Inc.",
        # Consumer
        "KO": "Coca-Cola Company",
        "PEP": "PepsiCo Inc.",
        "WMT": "Walmart Inc.",
        # Crypto ETFs
        "BITO": "Bitcoin ETF",
        # Turkish Stocks (BIST)
        "THYAO.IS": "Turkish Airlines",
        "BIMAS.IS": "BIM",
        "AKBNK.IS": "Akbank",
        "GARAN.IS": "Garanti BBVA",
        "ISCTR.IS": "Is Bank",
    }


def get_market_indices() -> Dict[str, str]:
    """
    Return list of important market indices.

    Returns:
        Dict[str, str]: Index code and name mapping
    """
    return {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones Industrial Average",
        "^IXIC": "NASDAQ Composite",
        "^RUT": "Russell 2000",
        "^VIX": "CBOE Volatility Index",
        "^TNX": "10-Year Treasury Yield",
        "XU100.IS": "BIST 100",
        "^FTSE": "FTSE 100",
        "^GDAXI": "DAX",
        "^N225": "Nikkei 225",
    }
