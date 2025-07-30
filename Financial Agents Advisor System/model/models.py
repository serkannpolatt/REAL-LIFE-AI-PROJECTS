"""
Agent Configuration Models - AI Agent Configuration Module
=========================================================

This module manages the configurations of financial analysis AI agents.
It provides integration of Groq LLM, DuckDuckGo, and YFinance tools.

Classes:
    AgentConfigManager: Main class that manages agent configurations

Functions:
    get_web_search_agent_config: Web search agent configuration (backward compatibility)
    get_finance_agent_config: Financial analysis agent configuration (backward compatibility)
"""

import logging
from typing import Dict, Any, List
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

logger = logging.getLogger(__name__)


class AgentConfigManager:
    """
    Main class that manages AI agent configurations.

    This class centrally manages the required configurations for different
    AI agents and provides consistent settings.
    """

    def __init__(self):
        """Initialize ConfigManager and set basic settings."""
        self.groq_model_id = "llama3-groq-70b-8192-tool-use-preview"
        self.base_instructions = [
            "Always cite from reliable sources",
            "Present analysis results clearly and understandably",
        ]
        logger.info("AgentConfigManager initialized")

    def _get_base_agent_config(self) -> Dict[str, Any]:
        """
        Return common base configuration for all agents.

        Returns:
            Dict[str, Any]: Base agent configuration
        """
        return {
            "model": Groq(id=self.groq_model_id),
            "show_tool_calls": True,
            "markdown": True,
            "debug_mode": False,
        }

    def get_web_search_agent_config(self) -> Dict[str, Any]:
        """
        Return configuration for web search capable agent.

        Returns:
            Dict[str, Any]: Web search agent configuration
        """
        config = self._get_base_agent_config()
        config.update(
            {
                "name": "Web Research Agent",
                "role": "Conducting current information research on the web",
                "description": "Searches for current news and information using DuckDuckGo",
                "tools": [DuckDuckGo()],
                "instructions": self.base_instructions
                + [
                    "Collect the most current and reliable information from the web",
                    "Always specify source links",
                    "Verify from multiple sources",
                    "Check date information and specify recency",
                ],
            }
        )

        logger.info("Web search agent configuration created")
        return config

    def get_finance_agent_config(self) -> Dict[str, Any]:
        """
        Return configuration for financial analysis capable agent.

        Returns:
            Dict[str, Any]: Financial analysis agent configuration
        """
        config = self._get_base_agent_config()

        # Configure YFinance tools
        yfinance_tools = YFinanceTools(
            stock_price=True,  # Stock prices
            analyst_recommendations=True,  # Analyst recommendations
            stock_fundamentals=True,  # Fundamental analysis data
            company_news=True,  # Company news
            technical_indicators=True,  # Technical indicators
            company_info=True,  # Company information
        )

        config.update(
            {
                "name": "Financial Analysis Agent",
                "role": "Financial data analysis and investment recommendations",
                "description": "Analyzes stocks and financial data using YFinance",
                "tools": [yfinance_tools],
                "instructions": self.base_instructions
                + [
                    "Organize financial data in tables",
                    "Summarize and evaluate analyst recommendations",
                    "Always specify risk factors",
                    "Conduct both technical and fundamental analysis",
                    "Compare past performance with current situation",
                    "Compare with sector averages",
                    "Be careful when giving investment advice and emphasize risks",
                ],
            }
        )

        logger.info("Financial analysis agent configuration created")
        return config

    def get_specialized_agent_config(
        self, agent_type: str, custom_instructions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Create specialized agent configuration.

        Args:
            agent_type (str): Agent type ("technical", "fundamental", "news")
            custom_instructions (List[str], optional): Custom instructions

        Returns:
            Dict[str, Any]: Specialized agent configuration
        """
        config = self._get_base_agent_config()
        instructions = self.base_instructions.copy()

        if custom_instructions:
            instructions.extend(custom_instructions)

        if agent_type == "technical":
            config.update(
                {
                    "name": "Technical Analysis Expert",
                    "role": "Technical analysis and chart interpretation",
                    "tools": [
                        YFinanceTools(technical_indicators=True, stock_price=True)
                    ],
                    "instructions": instructions
                    + [
                        "Analyze technical indicators in detail",
                        "Perform trend analysis",
                        "Identify support and resistance levels",
                    ],
                }
            )
        elif agent_type == "fundamental":
            config.update(
                {
                    "name": "Fundamental Analysis Expert",
                    "role": "Company financials and fundamental analysis",
                    "tools": [
                        YFinanceTools(stock_fundamentals=True, company_info=True)
                    ],
                    "instructions": instructions
                    + [
                        "Calculate and evaluate financial ratios",
                        "Compare company performance with sector",
                        "Analyze growth potential",
                    ],
                }
            )
        elif agent_type == "news":
            config.update(
                {
                    "name": "News Analysis Expert",
                    "role": "Analyzing financial news",
                    "tools": [DuckDuckGo(), YFinanceTools(company_news=True)],
                    "instructions": instructions
                    + [
                        "Evaluate the impact of news on stock prices",
                        "Analyze market sentiment",
                        "Extract chronology of important events",
                    ],
                }
            )

        logger.info(f"Specialized '{agent_type}' agent configuration created")
        return config


# Wrapper functions for backward compatibility
def get_web_search_agent_config() -> Dict[str, Any]:
    """
    Return web search agent configuration (backward compatibility).

    Returns:
        Dict[str, Any]: Web search agent configuration
    """
    manager = AgentConfigManager()
    return manager.get_web_search_agent_config()


def get_finance_agent_config() -> Dict[str, Any]:
    """
    Return financial analysis agent configuration (backward compatibility).

    Returns:
        Dict[str, Any]: Financial analysis agent configuration
    """
    manager = AgentConfigManager()
    return manager.get_finance_agent_config()
