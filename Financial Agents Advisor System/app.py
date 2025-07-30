"""
Finance Advisor AI Agent - Main Application
==========================================

This module serves as the main entry point for the intelligent AI agent
developed for financial analysis and market research.

Features:
- Stock analysis and price tracking
- Analyst recommendations summarization
- Latest company news collection
- Web-based research support

Usage:
    python app.py

Requirements:
    - OPENAI_API_KEY and GROQ_API_KEY must be defined in .env file
    - Packages in requirements.txt must be installed
"""

import os
import sys
import logging
from dotenv import load_dotenv
import openai

from controller.agent_controller import FinanceAgentController

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("finance_agent.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class FinanceAdvisorApp:
    """
    Finance Advisor AI Agent main application class.

    This class is responsible for initializing and managing the financial AI agent.
    It processes user requests and generates appropriate responses.
    """

    def __init__(self):
        """Initialize the application and configure necessary settings."""
        self._load_environment()
        self._setup_openai()
        self.controller = FinanceAgentController()
        self.agent = None
        logger.info("Finance Advisor AI Agent initialized")

    def _load_environment(self) -> None:
        """Load environment variables."""
        load_dotenv()

        # Check required API keys
        required_keys = ["OPENAI_API_KEY", "GROQ_API_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]

        if missing_keys:
            logger.error(f"Missing API keys: {missing_keys}")
            raise ValueError(f"Please define these keys in .env file: {missing_keys}")

        logger.info("Environment variables loaded successfully")

    def _setup_openai(self) -> None:
        """Set up OpenAI API key."""
        openai.api_key = os.getenv("OPENAI_API_KEY")
        logger.info("OpenAI API key configured")

    def initialize_agent(self) -> None:
        """Initialize the AI agent."""
        try:
            self.agent = self.controller.initialize_agents()
            logger.info("AI agent initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AI agent: {e}")
            raise

    def process_query(self, query: str, stream: bool = True) -> None:
        """
        Process user query and generate response.

        Args:
            query (str): User's financial question
            stream (bool): Whether to display response in streaming mode
        """
        if not self.agent:
            logger.error("Agent not yet initialized")
            raise RuntimeError("Please call initialize_agent() method first")

        try:
            logger.info(f"Processing query: {query[:50]}...")
            self.agent.print_response(query, stream=stream)
            logger.info("Query processed successfully")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def run_interactive_mode(self) -> None:
        """
        Start interactive mode - user can ask continuous questions.
        """
        print("\n" + "=" * 60)
        print("🤖 Finance Advisor AI Agent - Interactive Mode")
        print("=" * 60)
        print("You can ask financial questions. Type 'quit' to exit.")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not user_input:
                    print("⚠️  Please enter a question.")
                    continue

                print("\n🤔 Analyzing...")
                self.process_query(user_input)
                print("\n" + "-" * 60)

            except KeyboardInterrupt:
                print("\n\n👋 Program terminated.")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"❌ Error occurred: {e}")


def main():
    """Main function - starts the application."""
    try:
        # Start the application
        app = FinanceAdvisorApp()
        app.initialize_agent()

        # Sample query (optional)
        sample_query = (
            "Summarize analyst recommendations and share the latest news for NVDA"
        )
        print(f"\n🚀 Running sample query: {sample_query}")
        print("=" * 80)
        app.process_query(sample_query, stream=True)

        # Start interactive mode
        app.run_interactive_mode()

    except Exception as e:
        logger.error(f"Critical error in application: {e}")
        print(f"❌ Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
