"""
Financial Agents Module - Advanced AI Agent Collection
=====================================================

This module provides a comprehensive collection of specialized financial
analysis agents, each designed for specific aspects of financial analysis.

Available Agents:
    - BaseFinancialAgent: Abstract base class for all financial agents
    - TechnicalAnalysisAgent: Technical analysis and chart patterns
    - SentimentAnalysisAgent: Market sentiment and news analysis
    - RiskAssessmentAgent: Comprehensive risk analysis
    - CryptoAnalysisAgent: Cryptocurrency market analysis
    - PortfolioOptimizationAgent: Portfolio optimization and asset allocation

Agent Registry:
    - AgentRegistry: Centralized agent management system
    - agent_registry: Global registry instance

Usage Example:
    ```python
    from agents import agent_registry, TechnicalAnalysisAgent, RiskAssessmentAgent

    # Initialize agents
    tech_agent = TechnicalAnalysisAgent()
    risk_agent = RiskAssessmentAgent()

    # Register agents
    agent_registry.register_agent(tech_agent)
    agent_registry.register_agent(risk_agent)

    # Use agents
    tech_agent.initialize()
    analysis = tech_agent.analyze('AAPL', AnalysisType.PRICE_MOVEMENT)
    ```
"""

from .base_agent import (
    BaseFinancialAgent,
    AgentCapability,
    AnalysisType,
    AgentResponse,
    AgentRegistry,
    agent_registry,
)

from .technical_agent import TechnicalAnalysisAgent
from .sentiment_agent import SentimentAnalysisAgent
from .risk_agent import RiskAssessmentAgent
from .crypto_agent import CryptoAnalysisAgent
from .portfolio_agent import PortfolioOptimizationAgent


# Agent registry setup
def initialize_all_agents():
    """
    Initialize and register all available financial agents.

    Returns:
        bool: True if all agents initialized successfully
    """
    agents = [
        TechnicalAnalysisAgent(),
        SentimentAnalysisAgent(),
        RiskAssessmentAgent(),
        CryptoAnalysisAgent(),
        PortfolioOptimizationAgent(),
    ]

    success_count = 0
    for agent in agents:
        try:
            if agent.initialize():
                agent_registry.register_agent(agent)
                success_count += 1
        except Exception as e:
            print(f"Failed to initialize {agent.name}: {e}")

    return success_count == len(agents)


# Export all agent classes and utilities
__all__ = [
    # Base classes
    "BaseFinancialAgent",
    "AgentCapability",
    "AnalysisType",
    "AgentResponse",
    "AgentRegistry",
    # Specialized agents
    "TechnicalAnalysisAgent",
    "SentimentAnalysisAgent",
    "RiskAssessmentAgent",
    "CryptoAnalysisAgent",
    "PortfolioOptimizationAgent",
    # Registry
    "agent_registry",
    "initialize_all_agents",
]

# Version information
__version__ = "1.0.0"
__author__ = "Finance AI Team"
__description__ = "Advanced Financial Analysis Agent Collection"
