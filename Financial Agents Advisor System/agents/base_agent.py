"""
Base Agent Module - Abstract Base Classes for Financial Agents
============================================================

This module provides abstract base classes and interfaces for all financial agents.
It ensures consistency and standardization across different agent implementations.

Classes:
    BaseFinancialAgent: Abstract base class for all financial agents
    AgentCapability: Enumeration of agent capabilities
    AnalysisType: Types of financial analysis
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Enumeration of different agent capabilities."""

    STOCK_ANALYSIS = "stock_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    NEWS_ANALYSIS = "news_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    MARKET_RESEARCH = "market_research"
    CRYPTO_ANALYSIS = "crypto_analysis"
    FOREX_ANALYSIS = "forex_analysis"


class AnalysisType(Enum):
    """Types of financial analysis that can be performed."""

    PRICE_MOVEMENT = "price_movement"
    VOLUME_ANALYSIS = "volume_analysis"
    EARNINGS_ANALYSIS = "earnings_analysis"
    VALUATION = "valuation"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    SECTOR_ANALYSIS = "sector_analysis"
    MACRO_ECONOMIC = "macro_economic"
    REGULATORY_IMPACT = "regulatory_impact"


@dataclass
class AgentResponse:
    """Standardized response format for all agents."""

    success: bool
    data: Dict[str, Any]
    message: str
    confidence_score: float
    sources: List[str]
    analysis_type: AnalysisType
    timestamp: str


class BaseFinancialAgent(ABC):
    """
    Abstract base class for all financial agents.

    This class defines the common interface and behavior that all
    financial agents must implement.
    """

    def __init__(self, name: str, capabilities: List[AgentCapability]):
        """
        Initialize the base financial agent.

        Args:
            name (str): Name of the agent
            capabilities (List[AgentCapability]): List of agent capabilities
        """
        self.name = name
        self.capabilities = capabilities
        self.is_initialized = False
        logger.info(f"Base agent '{name}' created with capabilities: {capabilities}")

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the agent with required resources.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def analyze(
        self, symbol: str, analysis_type: AnalysisType, **kwargs
    ) -> AgentResponse:
        """
        Perform analysis on a given financial symbol.

        Args:
            symbol (str): Financial symbol to analyze (e.g., 'AAPL', 'BTC-USD')
            analysis_type (AnalysisType): Type of analysis to perform
            **kwargs: Additional parameters for specific analysis types

        Returns:
            AgentResponse: Standardized response with analysis results
        """
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """
        Get list of financial symbols supported by this agent.

        Returns:
            List[str]: List of supported symbols
        """
        pass

    def can_handle(self, capability: AgentCapability) -> bool:
        """
        Check if agent can handle a specific capability.

        Args:
            capability (AgentCapability): Capability to check

        Returns:
            bool: True if agent supports the capability
        """
        return capability in self.capabilities

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the agent.

        Returns:
            Dict[str, Any]: Agent information including capabilities and status
        """
        return {
            "name": self.name,
            "capabilities": [cap.value for cap in self.capabilities],
            "is_initialized": self.is_initialized,
            "supported_symbols_count": len(self.get_supported_symbols())
            if self.is_initialized
            else 0,
            "analysis_types": [atype.value for atype in AnalysisType],
        }

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol is supported by this agent.

        Args:
            symbol (str): Symbol to validate

        Returns:
            bool: True if symbol is supported
        """
        if not self.is_initialized:
            logger.warning(f"Agent '{self.name}' not initialized for symbol validation")
            return False

        supported_symbols = self.get_supported_symbols()
        return symbol.upper() in [s.upper() for s in supported_symbols]


class AgentRegistry:
    """
    Registry for managing multiple financial agents.

    This class provides a centralized way to register, discover,
    and manage different financial agents.
    """

    def __init__(self):
        """Initialize the agent registry."""
        self._agents: Dict[str, BaseFinancialAgent] = {}
        self._capability_map: Dict[AgentCapability, List[str]] = {}
        logger.info("Agent registry initialized")

    def register_agent(self, agent: BaseFinancialAgent) -> bool:
        """
        Register a new agent in the registry.

        Args:
            agent (BaseFinancialAgent): Agent to register

        Returns:
            bool: True if registration successful
        """
        try:
            if agent.name in self._agents:
                logger.warning(f"Agent '{agent.name}' already registered, updating...")

            self._agents[agent.name] = agent

            # Update capability map
            for capability in agent.capabilities:
                if capability not in self._capability_map:
                    self._capability_map[capability] = []
                if agent.name not in self._capability_map[capability]:
                    self._capability_map[capability].append(agent.name)

            logger.info(f"Agent '{agent.name}' registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent '{agent.name}': {e}")
            return False

    def get_agent(self, name: str) -> Optional[BaseFinancialAgent]:
        """
        Get an agent by name.

        Args:
            name (str): Name of the agent

        Returns:
            Optional[BaseFinancialAgent]: Agent if found, None otherwise
        """
        return self._agents.get(name)

    def get_agents_by_capability(
        self, capability: AgentCapability
    ) -> List[BaseFinancialAgent]:
        """
        Get all agents that support a specific capability.

        Args:
            capability (AgentCapability): Capability to search for

        Returns:
            List[BaseFinancialAgent]: List of agents supporting the capability
        """
        agent_names = self._capability_map.get(capability, [])
        return [self._agents[name] for name in agent_names if name in self._agents]

    def get_all_agents(self) -> List[BaseFinancialAgent]:
        """
        Get all registered agents.

        Returns:
            List[BaseFinancialAgent]: List of all registered agents
        """
        return list(self._agents.values())

    def remove_agent(self, name: str) -> bool:
        """
        Remove an agent from the registry.

        Args:
            name (str): Name of the agent to remove

        Returns:
            bool: True if removal successful
        """
        if name not in self._agents:
            logger.warning(f"Agent '{name}' not found in registry")
            return False

        # Remove from capability map
        agent = self._agents[name]
        for capability in agent.capabilities:
            if capability in self._capability_map:
                if name in self._capability_map[capability]:
                    self._capability_map[capability].remove(name)
                if not self._capability_map[capability]:
                    del self._capability_map[capability]

        # Remove from agents
        del self._agents[name]
        logger.info(f"Agent '{name}' removed from registry")
        return True

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the agent registry.

        Returns:
            Dict[str, Any]: Registry statistics
        """
        return {
            "total_agents": len(self._agents),
            "agent_names": list(self._agents.keys()),
            "capabilities_covered": list(self._capability_map.keys()),
            "capability_distribution": {
                cap.value: len(agents) for cap, agents in self._capability_map.items()
            },
        }


# Global agent registry instance
agent_registry = AgentRegistry()
