"""
Main module for FinAgents.
This module serves as the entry point for the portfolio analysis system.
"""

import os
import sys
from datetime import datetime
from crewai import Crew
from langchain_openai import ChatOpenAI

# Add project root to system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import application modules
from models.data import get_stock_data
from models.metrics import calculate_portfolio_metrics
from models.agents import create_agents, create_tasks
from visualization.charts import generate_charts, generate_allocation_chart
from reporting.pdf_report import create_pdf_report
from utils.helpers import (
    parse_report_sections,
    generate_workflow_diagram,
    save_text_report,
)
from utils.config import (
    DEFAULT_PORTFOLIO,
    OPENAI_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    CHART_DIR,
)


def setup_environment():
    """Set up the environment for the application"""
    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Initialize LLM
    llm = ChatOpenAI(model_name=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)
    return llm


def prepare_data(portfolio_details):
    """
    Prepare data for portfolio analysis

    Args:
        portfolio_details (dict): Portfolio details with weights

    Returns:
        tuple: Tuple containing stock data, portfolio metrics, and other data
    """
    # Extract ticker symbols
    stocks = list(portfolio_details.keys())

    # Get current weights
    current_weights = {
        stock: info["weight"] for stock, info in portfolio_details.items()
    }

    # Get stock data
    stock_data = get_stock_data(stocks)

    # Calculate portfolio metrics
    pm_metrics = calculate_portfolio_metrics(stock_data, current_weights)

    # Get recent data for analysis
    recent_data = stock_data.tail(30)
    recent_data_str = recent_data.to_csv(index=True)

    # Create metrics summary
    metrics_summary = f"""
    Portfolio Metrics:
    - Annual Return: {pm_metrics["annual_return"]:.4f} ({pm_metrics["annual_return"] * 100:.2f}%)
    - Annual Volatility: {pm_metrics["annual_volatility"]:.4f} ({pm_metrics["annual_volatility"] * 100:.2f}%)
    - Sharpe Ratio: {pm_metrics["sharpe_ratio"]:.4f}
    - Maximum Drawdown: {pm_metrics["max_drawdown"]:.4f} ({pm_metrics["max_drawdown"] * 100:.2f}%)
    """

    # Create stock metrics string
    stock_metrics_str = "Individual Stock Metrics:\n"
    for stock, metrics in pm_metrics["stock_metrics"].items():
        stock_metrics_str += f"\n{stock}:\n"
        stock_metrics_str += f"- Annual Return: {metrics['annual_return'] * 100:.2f}%\n"
        stock_metrics_str += (
            f"- Annual Volatility: {metrics['annual_volatility'] * 100:.2f}%\n"
        )
        stock_metrics_str += f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        stock_metrics_str += f"- Beta: {metrics['beta']:.2f}\n"

    return (
        stock_data,
        pm_metrics,
        recent_data_str,
        metrics_summary,
        stock_metrics_str,
        current_weights,
    )


def main(portfolio_details=None):
    """
    Main function to run the portfolio analysis

    Args:
        portfolio_details (dict, optional): Portfolio details with weights,
                                           defaults to DEFAULT_PORTFOLIO if None

    Returns:
        str: Path to the generated PDF report
    """
    # Use default portfolio if none is provided
    if portfolio_details is None:
        portfolio_details = DEFAULT_PORTFOLIO

    try:
        print(f"\n{'=' * 50}\nStarting Portfolio Analysis Crew\n{'=' * 50}\n")

        # Set up environment and LLM
        llm = setup_environment()

        # Prepare data for analysis
        (
            stock_data,
            pm_metrics,
            recent_data_str,
            metrics_summary,
            stock_metrics_str,
            current_weights,
        ) = prepare_data(portfolio_details)

        # Create agents
        agents = create_agents(llm)

        # Create tasks for agents
        tasks = create_tasks(
            agents,
            portfolio_details,
            metrics_summary,
            stock_metrics_str,
            current_weights,
            recent_data_str,
        )

        # Create crew with agents and tasks
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            verbose=True,
        )

        # Run the crew
        result = crew.kickoff()

        print("\n\nFINAL REPORT:")
        print(result)

        # Parse the report into sections
        report_sections = parse_report_sections(str(result))

        # Debug: Print the extracted sections
        print("\nExtracted report sections:")
        for section, content in report_sections.items():
            print(f"Section: {section}")
            print(f"Content length: {len(content)} characters")
            if len(content) < 10:
                print(f"Warning: Section '{section}' has very little content!")

        # Generate charts
        charts = generate_charts(stock_data, pm_metrics, CHART_DIR)

        # Add allocation chart
        allocation_chart = generate_allocation_chart(current_weights, CHART_DIR)
        charts["allocation"] = allocation_chart

        # Generate current date for filename
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Create PDF report
        pdf_filename = f"Portfolio_Investment_Report_{current_date}.pdf"
        create_pdf_report(report_sections, charts, pm_metrics, pdf_filename)

        # Save text report
        save_text_report(str(result), f"Portfolio_Investment_Report_{current_date}.txt")

        # Generate workflow diagram
        generate_workflow_diagram()

        return pdf_filename

    except Exception as e:
        print(f"Error executing the crew: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
