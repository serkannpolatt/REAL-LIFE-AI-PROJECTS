"""
Finance Agent Controller - AI Agent Management Module
====================================================

This module is responsible for initializing and coordinating AI agents
used for financial analysis. It brings together web search and financial
data analysis agents to create a powerful financial advisor AI system.

Classes:
    FinanceAgentController: Main control class

Functions:
    initialize_agents: Wrapper function for backward compatibility
"""

import logging
from typing import Dict, Any
from phi.agent import Agent
from model.models import AgentConfigManager

logger = logging.getLogger(__name__)


class FinanceAgentController:
    """
    Main control class responsible for managing financial AI agents.

    This class initializes, configures, and coordinates web search and financial
    data analysis agents to create a multi-agent system.
    """

    def __init__(self):
        """Initialize the controller."""
        self.config_manager = AgentConfigManager()
        self.web_search_agent = None
        self.finance_agent = None
        self.multi_agent = None
        logger.info("FinanceAgentController initialized")

    def create_web_search_agent(self) -> Agent:
        """
        Create an AI agent capable of web search.

        Returns:
            Agent: Agent capable of web search with DuckDuckGo
        """
        try:
            config = self.config_manager.get_web_search_agent_config()
            agent = Agent(**config)
            logger.info("Web search agent created successfully")
            return agent
        except Exception as e:
            logger.error(f"Error while creating web search agent: {e}")
            raise

    def create_finance_agent(self) -> Agent:
        """
        Create an AI agent capable of financial data analysis.

        Returns:
            Agent: Agent capable of financial analysis with YFinance
        """
        try:
            config = self.config_manager.get_finance_agent_config()
            agent = Agent(**config)
            logger.info("Financial analysis agent created successfully")
            return agent
        except Exception as e:
            logger.error(f"Error while creating financial analysis agent: {e}")
            raise

    def create_multi_agent_system(
        self, web_agent: Agent, finance_agent: Agent
    ) -> Agent:
        """
        Create the main system that coordinates multiple agents.

        Args:
            web_agent (Agent): Web search agent
            finance_agent (Agent): Financial analysis agent

        Returns:
            Agent: Coordinating multi-agent system
        """
        try:
            multi_agent_config = {
                "name": "Finance Advisor Multi-Agent System",
                "description": "Financial analysis and web search coordinator",
                "team": [web_agent, finance_agent],
                "instructions": [
                    "Always cite from reliable sources",
                    "Display data in table format",
                    "Present analysis results clearly and understandably",
                    "Consider current market conditions",
                    "Specify risk factors",
                    "Respond in Turkish (unless requested otherwise)",
                ],
                "show_tool_calls": True,
                "markdown": True,
                "debug_mode": False,
            }

            agent = Agent(**multi_agent_config)
            logger.info("Multi-agent system created successfully")
            return agent
        except Exception as e:
            logger.error(f"Error while creating multi-agent system: {e}")
            raise

    def initialize_agents(self) -> Agent:
        """
        Initialize all agents and return the coordinating system.

        Returns:
            Agent: Ready-to-use multi-agent financial advisor system

        Raises:
            RuntimeError: If agents cannot be initialized
        """
        try:
            logger.info("Initializing AI agents...")

            # Create agents
            self.web_search_agent = self.create_web_search_agent()
            self.finance_agent = self.create_finance_agent()

            # Create multi-agent system
            self.multi_agent = self.create_multi_agent_system(
                self.web_search_agent, self.finance_agent
            )

            logger.info("All agents initialized successfully")
            return self.multi_agent

        except Exception as e:
            logger.error(f"Critical error while initializing agents: {e}")
            raise RuntimeError(f"AI agents could not be initialized: {e}")

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Return the status of current agents.

        Returns:
            Dict[str, Any]: Agent status information
        """
        return {
            "web_search_agent": self.web_search_agent is not None,
            "finance_agent": self.finance_agent is not None,
            "multi_agent": self.multi_agent is not None,
            "system_ready": all(
                [
                    self.web_search_agent is not None,
                    self.finance_agent is not None,
                    self.multi_agent is not None,
                ]
            ),
        }


# Wrapper function for backward compatibility
def initialize_agents() -> Agent:
    """
    Wrapper function for backward compatibility.

    Returns:
        Agent: Initialized multi-agent system
    """
    controller = FinanceAgentController()
    return controller.initialize_agents()
