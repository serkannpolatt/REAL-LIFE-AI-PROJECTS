"""
Config module for FinAgents.
This module contains configuration settings for the application.
"""

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Default portfolio details
DEFAULT_PORTFOLIO = {
    "AAPL": {"weight": 0.25},
    "MSFT": {"weight": 0.25},
    "GOOGL": {"weight": 0.20},
    "AMZN": {"weight": 0.15},
    "TSLA": {"weight": 0.15},
}

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.2

# Data fetching configuration
DEFAULT_DATA_PERIOD = "1y"
DEFAULT_DATA_INTERVAL = "1d"

# File paths
CHART_DIR = "charts"
