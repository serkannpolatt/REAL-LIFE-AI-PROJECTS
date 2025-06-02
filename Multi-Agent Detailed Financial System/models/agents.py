"""
Agent definitions for FinAgents.
This module defines the agents and tasks used in the portfolio analysis system.
"""

from crewai import Agent, Task
from langchain_openai import ChatOpenAI
import os


def create_agents(llm):
    """
    Create the portfolio analysis agents

    Args:
        llm: Language model instance

    Returns:
        dict: Dictionary containing agent instances
    """
    # Risk analyst agent
    risk_analyst = Agent(
        role="Risk Analyst",
        goal="Evaluate portfolio volatility and risks, suggesting specific changes to improve risk-adjusted returns, including diversification across different asset classes.",
        backstory="You are a financial analyst with expertise in risk assessment, quantitative strategies, and multi-asset portfolio construction. You have deep knowledge of various asset classes including stocks, bonds, REITs, commodities, and alternative investments.",
        verbose=True,
        llm=llm,
    )

    # Market analyst agent
    market_analyst = Agent(
        role="Market Analyst",
        goal="Provide a deep analysis of current market conditions and a 12-month outlook for each sector in the portfolio, identifying new sectors and specific companies that could enhance diversification.",
        backstory="You are a seasoned market analyst with expertise in macroeconomic trends, sector analysis, and stock selection. You have a proven track record of identifying emerging sectors and high-potential companies across global markets.",
        verbose=True,
        llm=llm,
    )

    # Allocation optimizer agent
    allocation_optimizer = Agent(
        role="Allocation Optimizer",
        goal="Propose specific changes to portfolio allocation to maximize risk-adjusted returns, including new asset classes and specific investment vehicles.",
        backstory="You are an expert in portfolio optimization and quantitative finance using modern portfolio theory. You specialize in multi-asset allocation strategies and have extensive knowledge of ETFs, mutual funds, and individual securities across global markets.",
        verbose=True,
        llm=llm,
    )

    # Portfolio manager agent
    portfolio_manager = Agent(
        role="Portfolio Manager",
        goal="Make final allocation decisions based on risk analysis and optimization suggestions, providing specific implementation steps and investment vehicles.",
        backstory="You are an experienced portfolio manager responsible for strategic investment decisions. You have deep expertise in asset allocation, security selection, and portfolio implementation across various market conditions. You provide actionable advice with specific investment recommendations.",
        verbose=True,
        llm=llm,
    )

    # Report generator agent
    report_generator = Agent(
        role="Investment Report Writer",
        goal="Create a comprehensive investment report for the client that includes all required sections with specific, actionable recommendations.",
        backstory="You are a skilled financial writer who translates complex analysis into clear, detailed reports for high-net-worth clients. Your reports include specific investment recommendations, including ticker symbols, allocation percentages, and implementation steps. Your report must include the following sections with standardized headers:\n\n"
        "EXECUTIVE SUMMARY\nMARKET OVERVIEW\nPORTFOLIO PERFORMANCE ANALYSIS\nRISK ASSESSMENT\nALLOCATION RECOMMENDATIONS\nIMPLEMENTATION STRATEGY\nFUTURE OUTLOOK\nCONCLUSION",
        verbose=True,
        llm=llm,
    )

    return {
        "risk_analyst": risk_analyst,
        "market_analyst": market_analyst,
        "allocation_optimizer": allocation_optimizer,
        "portfolio_manager": portfolio_manager,
        "report_generator": report_generator,
    }


def create_tasks(
    agents,
    portfolio_details,
    metrics_summary,
    stock_metrics_str,
    current_weights,
    recent_data_str,
):
    """
    Create tasks for the portfolio analysis agents

    Args:
        agents (dict): Dictionary containing agent instances
        portfolio_details (dict): Portfolio details
        metrics_summary (str): Summary of portfolio metrics
        stock_metrics_str (str): String representation of stock metrics
        current_weights (dict): Current portfolio weights
        recent_data_str (str): Recent stock data in string format

    Returns:
        list: List of Task instances
    """
    # Risk analysis task
    risk_analysis_task = Task(
        description=f"""Analyze the portfolio risks based on recent data and calculated metrics.

    Recent stock data:
    {recent_data_str}

    {metrics_summary}

    {stock_metrics_str}

    Current portfolio weights:
    {current_weights}

    Identify volatility patterns, correlation risks, and potential market risks.
    Suggest risk mitigation strategies with detailed reasoning.

    Go beyond basic rebalancing and provide specific recommendations for:
    1. Asset class diversification (bonds, REITs, commodities, etc.) with specific ETFs or securities
    2. Geographic diversification with specific international market exposure recommendations
    3. Factor-based diversification strategies (value, growth, quality, etc.)
    4. Hedging strategies with specific implementation methods

    For each recommendation, provide specific investment vehicles (with ticker symbols where applicable) and suggested allocation percentages.
    """,
        agent=agents["risk_analyst"],
        expected_output="Detailed risk analysis report with specific diversification and risk mitigation recommendations.",
    )

    # Market analysis task
    tickers_str = ""
    for ticker, info in portfolio_details.items():
        tickers_str += f"- {ticker}: {info['weight'] * 100}%\n"

    market_analysis_task = Task(
        description=f"""Provide a deep analysis of current market conditions and a 12-month outlook for each sector represented in the portfolio.

    The portfolio contains the following stocks:
    {tickers_str}

    First, classify each stock into appropriate sectors and industries based on their business activities.

    Then, for each identified sector, please:
    1. Analyze current conditions and trends.
    2. Identify growth opportunities and risks.
    3. Provide a 12-month outlook with key factors to watch.
    4. Suggest how these trends might impact each stock within the sector.

    Additionally, identify 3-5 new sectors not currently represented in the portfolio that offer diversification benefits and growth potential.
    For each new sector, recommend 2-3 specific companies (with ticker symbols) that are leaders or emerging players.

    Your analysis should be data-driven, forward-looking, and include specific investment recommendations.
    """,
        agent=agents["market_analyst"],
        expected_output="Comprehensive market analysis with specific sector and company recommendations for diversification.",
    )

    # Allocation task
    allocation_task = Task(
        description=f"""Based on the risk analysis and market outlook, suggest adjustments to the portfolio allocation.

    {metrics_summary}

    {stock_metrics_str}

    Current portfolio weights:
    {current_weights}

    Provide specific weight recommendations for each existing stock, ensuring balance between risk and return.
    Additionally, recommend:

    1. New asset classes to add to the portfolio (bonds, REITs, commodities, etc.) with specific ETFs or securities and allocation percentages
    2. Specific new stocks from underrepresented sectors with suggested allocation percentages
    3. Geographic diversification with international market exposure recommendations
    4. Factor-based allocation strategies (value, growth, quality, etc.)

    For each recommendation, provide specific investment vehicles (with ticker symbols) and suggested allocation percentages.
    Justify your recommendations using quantitative and qualitative reasoning.

    IMPORTANT: All allocation percentages MUST sum to EXACTLY 100%. Double-check your math to ensure the total allocation is precisely 100%.
    """,
        agent=agents["allocation_optimizer"],
        expected_output="Comprehensive portfolio allocation proposal with specific investment recommendations across asset classes and sectors.",
    )

    # Manager task
    manager_task = Task(
        description=f"""Review the suggestions from the risk analyst, market analyst, and allocation optimizer.
    Decide on the final allocation while considering transaction costs and tax implications.
    Provide a detailed implementation strategy with specific steps and investment vehicles.

    Your final recommendations should include:
    1. Specific allocation percentages for all recommended investments (existing and new)
    2. Specific ETFs, mutual funds, or individual securities for each asset class and sector (with ticker symbols)
    3. Implementation priority and timeline
    4. Tax-efficient implementation strategies
    5. Ongoing monitoring recommendations

    IMPORTANT: All allocation percentages MUST sum to EXACTLY 100%. Double-check your math to ensure the total allocation is precisely 100%.

    Be specific and actionable in your recommendations, providing a clear roadmap for portfolio implementation.
    """,
        agent=agents["portfolio_manager"],
        expected_output="Final portfolio allocation decision with detailed implementation strategy and specific investment recommendations.",
    )

    # Report task
    report_task = Task(
        description=f"""Create a comprehensive investment report for the client based on all previous analyses.
    The report must include the following sections with the exact headers:
        
    EXECUTIVE SUMMARY
    Provide a summary of key findings and recommendations, including the most important specific investment actions to take.

    MARKET OVERVIEW
    Detail current market conditions and sector insights, including specific sectors to increase or decrease exposure to.

    PORTFOLIO PERFORMANCE ANALYSIS
    Include detailed analysis of portfolio metrics and historical trends, with specific performance drivers and detractors.

    RISK ASSESSMENT
    Discuss identified risks and proposed mitigation strategies, including specific diversification recommendations across asset classes.

    ALLOCATION RECOMMENDATIONS
    Present specific portfolio rebalancing suggestions with exact allocation percentages and ticker symbols for all recommended investments.
    Include a clear table showing current vs. recommended allocations for all assets.
    Ensure recommendations include diversification across:
    - Asset classes (stocks, bonds, alternatives, etc.)
    - Sectors (both existing and new sectors)
    - Geographic regions
    - Investment styles/factors

    IMPORTANT: All allocation percentages MUST sum to EXACTLY 100%. Double-check your math to ensure the total allocation is precisely 100%.

    IMPLEMENTATION STRATEGY
    Outline a step-by-step plan for executing the recommendations, including:
    - Specific securities to buy and sell with ticker symbols
    - Implementation timeline and priority
    - Tax considerations
    - Cost-efficient implementation methods

    FUTURE OUTLOOK
    Provide projections and a 12-month outlook, including specific market catalysts and risks to monitor.

    CONCLUSION
    Summarize the final recommendations and key takeaways, emphasizing the most important actions to take.

    Ensure that every section is complete, specific, and actionable with concrete investment recommendations.
    """,
        agent=agents["report_generator"],
        expected_output="A professional investment report in plain text with all required sections complete and specific actionable recommendations.",
    )

    return [
        risk_analysis_task,
        market_analysis_task,
        allocation_task,
        manager_task,
        report_task,
    ]
