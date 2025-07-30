"""
Financial AI Agent Examples
==========================

This file contains comprehensive examples demonstrating the use of all
available financial analysis agents including technical analysis, sentiment analysis,
risk assessment, cryptocurrency analysis, and portfolio optimization.
"""

from dotenv import load_dotenv
from datetime import datetime

from model.models import AgentConfigManager
from controller.agent_controller import FinanceAgentController
from agents import (
    agent_registry,
    initialize_all_agents,
    TechnicalAnalysisAgent,
    SentimentAnalysisAgent,
    RiskAssessmentAgent,
    CryptoAnalysisAgent,
    PortfolioOptimizationAgent,
    AnalysisType,
)

# Load environment variables
load_dotenv()


def example_basic_analysis():
    """
    Example of basic financial analysis using the main agent.
    """
    print("=== Basic Financial Analysis Example ===")

    # Initialize components
    config_manager = AgentConfigManager()
    controller = FinanceAgentController(config_manager)

    # Initialize main agent
    agent = controller.create_main_agent()

    # Perform analysis
    query = "Analyze AAPL stock and provide investment recommendations"
    print(f"Query: {query}")

    response = agent.run(query)
    print(f"Response: {response}")
    print()


def example_technical_analysis():
    """
    Example of advanced technical analysis using TechnicalAnalysisAgent.
    """
    print("=== Technical Analysis Example ===")

    try:
        # Initialize technical analysis agent
        tech_agent = TechnicalAnalysisAgent()
        if not tech_agent.initialize():
            print("Failed to initialize technical analysis agent")
            return

        # Analyze AAPL stock
        symbol = "AAPL"
        print(f"Performing technical analysis for {symbol}...")

        # Different types of technical analysis
        analyses = [
            (AnalysisType.PRICE_MOVEMENT, "Price Movement Analysis"),
            (AnalysisType.TREND_ANALYSIS, "Trend Analysis"),
            (AnalysisType.VOLATILITY, "Volatility Analysis"),
        ]

        for analysis_type, description in analyses:
            print(f"\n--- {description} ---")
            result = tech_agent.analyze(symbol, analysis_type)
            if result.success:
                print(f"Analysis: {result.data}")
                print(f"Confidence: {result.confidence}")
            else:
                print(f"Error: {result.error}")

    except Exception as e:
        print(f"Technical analysis error: {e}")
    print()


def example_sentiment_analysis():
    """
    Example of market sentiment analysis using SentimentAnalysisAgent.
    """
    print("=== Sentiment Analysis Example ===")

    try:
        # Initialize sentiment analysis agent
        sentiment_agent = SentimentAnalysisAgent()
        if not sentiment_agent.initialize():
            print("Failed to initialize sentiment analysis agent")
            return

        # Analyze market sentiment for different stocks
        symbols = ["AAPL", "TSLA", "GOOGL"]

        for symbol in symbols:
            print(f"\n--- Sentiment Analysis for {symbol} ---")
            result = sentiment_agent.analyze(symbol, AnalysisType.SENTIMENT)

            if result.success:
                print(f"Sentiment Score: {result.data.get('sentiment_score', 'N/A')}")
                print(f"Market Mood: {result.data.get('market_mood', 'N/A')}")
                print(f"News Summary: {result.data.get('news_summary', 'N/A')}")
            else:
                print(f"Error: {result.error}")

    except Exception as e:
        print(f"Sentiment analysis error: {e}")
    print()


def example_risk_assessment():
    """
    Example of comprehensive risk assessment using RiskAssessmentAgent.
    """
    print("=== Risk Assessment Example ===")

    try:
        # Initialize risk assessment agent
        risk_agent = RiskAssessmentAgent()
        if not risk_agent.initialize():
            print("Failed to initialize risk assessment agent")
            return

        # Portfolio for risk analysis
        portfolio = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

        print(f"Analyzing risk for portfolio: {', '.join(portfolio)}")

        # Perform different types of risk analysis
        risk_analyses = [
            (AnalysisType.RISK_METRICS, "Risk Metrics"),
            (AnalysisType.VOLATILITY, "Volatility Analysis"),
        ]

        for analysis_type, description in risk_analyses:
            print(f"\n--- {description} ---")

            # For portfolio analysis, we'll analyze each symbol
            for symbol in portfolio[:2]:  # Limit to 2 for demo
                result = risk_agent.analyze(symbol, analysis_type)
                if result.success:
                    print(f"{symbol}: {result.data}")
                else:
                    print(f"{symbol} error: {result.error}")

    except Exception as e:
        print(f"Risk assessment error: {e}")
    print()


def example_crypto_analysis():
    """
    Example of cryptocurrency analysis using CryptoAnalysisAgent.
    """
    print("=== Cryptocurrency Analysis Example ===")

    try:
        # Initialize crypto analysis agent
        crypto_agent = CryptoAnalysisAgent()
        if not crypto_agent.initialize():
            print("Failed to initialize crypto analysis agent")
            return

        # Analyze popular cryptocurrencies
        crypto_symbols = ["BTC-USD", "ETH-USD"]

        for symbol in crypto_symbols:
            print(f"\n--- Crypto Analysis for {symbol} ---")

            # Different types of crypto analysis
            analyses = [
                (AnalysisType.PRICE_MOVEMENT, "Price Movement"),
                (AnalysisType.VOLATILITY, "Volatility"),
                (AnalysisType.TREND_ANALYSIS, "Trend Analysis"),
            ]

            for analysis_type, description in analyses:
                result = crypto_agent.analyze(symbol, analysis_type)
                if result.success:
                    print(f"{description}: {result.data}")
                else:
                    print(f"{description} error: {result.error}")
                    break  # Skip further analysis if one fails

    except Exception as e:
        print(f"Crypto analysis error: {e}")
    print()


def example_portfolio_optimization():
    """
    Example of portfolio optimization using PortfolioOptimizationAgent.
    """
    print("=== Portfolio Optimization Example ===")

    try:
        # Initialize portfolio optimization agent
        portfolio_agent = PortfolioOptimizationAgent()
        if not portfolio_agent.initialize():
            print("Failed to initialize portfolio optimization agent")
            return

        # Define a sample portfolio
        portfolio_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        print(f"Optimizing portfolio: {', '.join(portfolio_symbols)}")

        # For demonstration, we'll do a general portfolio analysis
        result = portfolio_agent.analyze(
            f"Optimize portfolio with {', '.join(portfolio_symbols)}",
            AnalysisType.PORTFOLIO_OPTIMIZATION,
        )

        if result.success:
            print("Portfolio Optimization Results:")
            print(f"Analysis: {result.data}")
            print(f"Confidence: {result.confidence}")
        else:
            print(f"Optimization error: {result.error}")

    except Exception as e:
        print(f"Portfolio optimization error: {e}")
    print()


def example_multi_agent_analysis():
    """
    Example of coordinated multi-agent analysis.
    """
    print("=== Multi-Agent Coordinated Analysis Example ===")

    # Initialize all agents
    if not initialize_all_agents():
        print("Failed to initialize all agents")
        return

    symbol = "AAPL"
    print(f"Performing comprehensive analysis for {symbol} using multiple agents...")

    # Get all registered agents
    registered_agents = agent_registry.get_all_agents()

    if not registered_agents:
        print("No agents registered")
        return

    print(f"Using {len(registered_agents)} specialized agents:")
    for agent in registered_agents:
        print(f"  - {agent.name}")

    # Perform analysis with each agent
    results = {}
    for agent in registered_agents:
        try:
            # Use appropriate analysis type for each agent
            analysis_type = AnalysisType.PRICE_MOVEMENT
            if "sentiment" in agent.name.lower():
                analysis_type = AnalysisType.SENTIMENT
            elif "risk" in agent.name.lower():
                analysis_type = AnalysisType.RISK_METRICS
            elif "portfolio" in agent.name.lower():
                analysis_type = AnalysisType.PORTFOLIO_OPTIMIZATION

            result = agent.analyze(symbol, analysis_type)
            results[agent.name] = result

        except Exception as e:
            print(f"Error with {agent.name}: {e}")

    # Display consolidated results
    print(f"\n--- Consolidated Analysis Results for {symbol} ---")
    for agent_name, result in results.items():
        print(f"\n{agent_name}:")
        if result.success:
            print(f"  Status: Success (Confidence: {result.confidence})")
            print(f"  Analysis: {str(result.data)[:200]}...")  # Truncate for display
        else:
            print(f"  Status: Failed - {result.error}")

    print()


def example_custom_portfolio():
    """
    Example of custom portfolio analysis.
    """
    print("=== Custom Portfolio Analysis Example ===")

    # Initialize components
    config_manager = AgentConfigManager()
    controller = FinanceAgentController(config_manager)

    # Create portfolio agent
    portfolio_agent = controller.create_portfolio_agent()

    # Analyze portfolio
    portfolio_query = """
    Analyze this portfolio:
    - 40% AAPL
    - 30% GOOGL
    - 20% MSFT
    - 10% TSLA
    
    Provide risk assessment and optimization suggestions.
    """

    print(f"Query: {portfolio_query}")
    response = portfolio_agent.run(portfolio_query)
    print(f"Response: {response}")
    print()


def example_agent_registry_management():
    """
    Example of agent registry management and monitoring.
    """
    print("=== Agent Registry Management Example ===")

    # Initialize agents
    if initialize_all_agents():
        print("All agents initialized successfully!")
    else:
        print("Some agents failed to initialize")

    # Display registry status
    print("\n--- Agent Registry Status ---")
    all_agents = agent_registry.get_all_agents()

    print(f"Total registered agents: {len(all_agents)}")
    for agent in all_agents:
        status = "✓ Active" if agent.is_initialized else "✗ Inactive"
        capabilities = ", ".join([cap.value for cap in agent.capabilities])
        print(f"  {agent.name}: {status}")
        print(f"    Capabilities: {capabilities}")
        print(f"    Version: {agent.version}")

    # Test agent retrieval by capability
    print("\n--- Agents by Capability ---")
    from agents.base_agent import AgentCapability

    for capability in AgentCapability:
        capable_agents = agent_registry.get_agents_by_capability(capability)
        if capable_agents:
            agent_names = [agent.name for agent in capable_agents]
            print(f"  {capability.value}: {', '.join(agent_names)}")

    print()


def interactive_demo():
    """
    Interactive demo mode with specialized agent selection.
    """
    print("=== Interactive Demo Mode ===")
    print("You can ask financial questions and choose which agent to use.")
    print("Type 'quit' to exit.")
    print()

    # Initialize agents
    if not initialize_all_agents():
        print("Failed to initialize all agents")
        return

    config_manager = AgentConfigManager()
    controller = FinanceAgentController(config_manager)
    main_agent = controller.create_main_agent()

    while True:
        try:
            print("\nAvailable analysis options:")
            print("1. General Financial Analysis (Main Agent)")
            print("2. Technical Analysis")
            print("3. Sentiment Analysis")
            print("4. Risk Assessment")
            print("5. Cryptocurrency Analysis")
            print("6. Portfolio Optimization")
            print("7. Multi-Agent Analysis")
            print("0. Quit")

            choice = input("\nSelect analysis type (0-7): ").strip()

            if choice == "0" or choice.lower() in ["quit", "exit", "q"]:
                print("Demo completed!")
                break

            if choice not in ["1", "2", "3", "4", "5", "6", "7"]:
                print("Invalid choice. Please try again.")
                continue

            query = input("\nEnter your financial question: ").strip()
            if not query:
                print("Please enter a question.")
                continue

            print("\nAnalyzing...")
            print("-" * 50)

            if choice == "1":
                response = main_agent.run(query)
                print(f"Response: {response}")
            elif choice == "2":
                tech_agent = TechnicalAnalysisAgent()
                if tech_agent.initialize():
                    result = tech_agent.analyze(query, AnalysisType.PRICE_MOVEMENT)
                    print(
                        f"Technical Analysis: {result.data if result.success else result.error}"
                    )
                else:
                    print("Failed to initialize technical analysis agent")
            elif choice == "3":
                sentiment_agent = SentimentAnalysisAgent()
                if sentiment_agent.initialize():
                    result = sentiment_agent.analyze(query, AnalysisType.SENTIMENT)
                    print(
                        f"Sentiment Analysis: {result.data if result.success else result.error}"
                    )
                else:
                    print("Failed to initialize sentiment analysis agent")
            elif choice == "4":
                risk_agent = RiskAssessmentAgent()
                if risk_agent.initialize():
                    result = risk_agent.analyze(query, AnalysisType.RISK_METRICS)
                    print(
                        f"Risk Assessment: {result.data if result.success else result.error}"
                    )
                else:
                    print("Failed to initialize risk assessment agent")
            elif choice == "5":
                crypto_agent = CryptoAnalysisAgent()
                if crypto_agent.initialize():
                    result = crypto_agent.analyze(query, AnalysisType.PRICE_MOVEMENT)
                    print(
                        f"Crypto Analysis: {result.data if result.success else result.error}"
                    )
                else:
                    print("Failed to initialize crypto analysis agent")
            elif choice == "6":
                portfolio_agent = PortfolioOptimizationAgent()
                if portfolio_agent.initialize():
                    result = portfolio_agent.analyze(
                        query, AnalysisType.PORTFOLIO_OPTIMIZATION
                    )
                    print(
                        f"Portfolio Optimization: {result.data if result.success else result.error}"
                    )
                else:
                    print("Failed to initialize portfolio optimization agent")
            elif choice == "7":
                print("Running multi-agent analysis...")
                # Simulate multi-agent analysis
                all_agents = agent_registry.get_all_agents()
                for agent in all_agents[:3]:  # Limit to first 3 for demo
                    try:
                        result = agent.analyze(query, AnalysisType.PRICE_MOVEMENT)
                        print(
                            f"{agent.name}: {str(result.data)[:100]}..."
                            if result.success
                            else f"{agent.name}: {result.error}"
                        )
                    except Exception as e:
                        print(f"{agent.name}: Error - {e}")

            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\nDemo terminated.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("Financial AI Agent Examples")
    print("=" * 60)
    print(f"Starting examples at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("Select an example to run:")
    print("1. Basic Financial Analysis")
    print("2. Technical Analysis")
    print("3. Sentiment Analysis")
    print("4. Risk Assessment")
    print("5. Cryptocurrency Analysis")
    print("6. Portfolio Optimization")
    print("7. Multi-Agent Analysis")
    print("8. Agent Registry Management")
    print("9. Interactive Demo Mode")
    print("0. Run All Examples")

    try:
        choice = input("\nEnter your choice (0-9): ").strip()

        if choice == "1":
            example_basic_analysis()
        elif choice == "2":
            example_technical_analysis()
        elif choice == "3":
            example_sentiment_analysis()
        elif choice == "4":
            example_risk_assessment()
        elif choice == "5":
            example_crypto_analysis()
        elif choice == "6":
            example_portfolio_optimization()
        elif choice == "7":
            example_multi_agent_analysis()
        elif choice == "8":
            example_agent_registry_management()
        elif choice == "9":
            interactive_demo()
        elif choice == "0":
            # Run all examples
            example_basic_analysis()
            example_custom_portfolio()
            example_agent_registry_management()
            example_technical_analysis()
            example_sentiment_analysis()
            example_risk_assessment()
            example_crypto_analysis()
            example_portfolio_optimization()
            example_multi_agent_analysis()
        else:
            print("Invalid choice!")

        print("=" * 60)
        print("Examples completed successfully!")
        print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyboardInterrupt:
        print("\n\nProgram terminated.")
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have set up your environment variables properly.")
        print("Required environment variables:")
        print("  - GROQ_API_KEY: Your Groq API key")
        print("  - Additional API keys may be required for specific agents")
