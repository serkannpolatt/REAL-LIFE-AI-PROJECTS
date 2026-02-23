"""
AI Stock Analysis Assistant - Backend Server

This module provides a FastAPI-based backend server that integrates with
LangChain to create an intelligent stock analysis chatbot. The chatbot
can fetch real-time stock data, historical prices, balance sheets,
and news using the yfinance library.

Author: AI Stock Analysis Team
Version: 1.0.0
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

# Environment and Configuration
from dotenv import load_dotenv  # Load environment variables from .env file

# Data Validation
from pydantic import BaseModel  # For request/response data validation

# Web Framework
import uvicorn  # ASGI server for running FastAPI
from fastapi import FastAPI  # Main web framework
from fastapi.middleware.cors import CORSMiddleware  # Cross-Origin Resource Sharing
from fastapi.responses import StreamingResponse  # For streaming chat responses

# LangChain - AI Agent Framework
from langchain.agents import create_agent  # Create AI agents with tools
from langchain.tools import tool  # Decorator for creating agent tools
from langchain.messages import SystemMessage, HumanMessage  # Message types for chat
from langchain_openai import ChatOpenAI  # OpenAI-compatible chat model
from langgraph.checkpoint.memory import InMemorySaver  # In-memory state persistence

# Financial Data
import yfinance as yf  # Yahoo Finance API for stock data

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Load environment variables from .env file (e.g., API keys)
load_dotenv()

# Initialize FastAPI application
app = FastAPI(
    title="AI Stock Analysis Assistant API",
    description="An intelligent stock analysis chatbot powered by LangChain",
    version="1.0.0",
)

# Configure the AI model
# Using TheSys API endpoint with GPT-5 model for intelligent responses
model = ChatOpenAI(
    model="c1/openai/gpt-5/v-20250930", base_url="https://api.thesys.dev/v1/embed/"
)

# Initialize memory checkpointer for conversation persistence
# This allows the agent to remember context within a conversation thread
checkpointer = InMemorySaver()


# ==============================================================================
# STOCK DATA TOOLS
# ==============================================================================


@tool(
    "get_stock_price",
    description="A function that returns the current stock price based on a ticker symbol.",
)
def get_stock_price(ticker: str) -> float:
    """
    Fetch the most recent closing price for a given stock.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')

    Returns:
        float: The most recent closing price of the stock

    Example:
        >>> get_stock_price('AAPL')
        175.50
    """
    print(f"[Tool] get_stock_price called with ticker: {ticker}")
    stock = yf.Ticker(ticker)
    return stock.history()["Close"].iloc[-1]


@tool(
    "get_historical_stock_price",
    description="A function that returns the historical stock prices over time based on a ticker symbol and a start and end date.",
)
def get_historical_stock_price(ticker: str, start_date: str, end_date: str) -> dict:
    """
    Fetch historical stock price data for a given date range.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        dict: Dictionary containing historical OHLCV data with dates as keys

    Example:
        >>> get_historical_stock_price('AAPL', '2024-01-01', '2024-12-31')
        {'Open': {...}, 'High': {...}, 'Low': {...}, 'Close': {...}, 'Volume': {...}}
    """
    print(
        f"[Tool] get_historical_stock_price called with ticker: {ticker}, start: {start_date}, end: {end_date}"
    )
    stock = yf.Ticker(ticker)
    return stock.history(start=start_date, end=end_date).to_dict()


@tool(
    "get_balance_sheet",
    description="A function that returns the balance sheet based on a ticker symbol.",
)
def get_balance_sheet(ticker: str):
    """
    Fetch the balance sheet data for a given stock.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')

    Returns:
        pandas.DataFrame: Balance sheet data including assets, liabilities,
                         and shareholders' equity

    Example:
        >>> get_balance_sheet('AAPL')
        DataFrame with balance sheet data
    """
    print(f"[Tool] get_balance_sheet called with ticker: {ticker}")
    stock = yf.Ticker(ticker)
    return stock.balance_sheet


@tool(
    "get_stock_news",
    description="A function that returns news based on a ticker symbol.",
)
def get_stock_news(ticker: str) -> list:
    """
    Fetch recent news articles related to a given stock.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')

    Returns:
        list: List of news articles with titles, links, and publication dates

    Example:
        >>> get_stock_news('AAPL')
        [{'title': 'Apple announces new product...', 'link': '...', ...}]
    """
    print(f"[Tool] get_stock_news called with ticker: {ticker}")
    stock = yf.Ticker(ticker)
    return stock.news


# ==============================================================================
# AI AGENT CONFIGURATION
# ==============================================================================

# Create the AI agent with all available tools
# The agent can intelligently decide which tool to use based on user queries
agent = create_agent(
    model=model,
    checkpointer=checkpointer,
    tools=[
        get_stock_price,
        get_historical_stock_price,
        get_balance_sheet,
        get_stock_news,
    ],
)


# ==============================================================================
# REQUEST/RESPONSE MODELS
# ==============================================================================


class PromptObject(BaseModel):
    """
    Model representing a chat prompt from the user.

    Attributes:
        content (str): The actual message content from the user
        id (str): Unique identifier for the message
        role (str): Role of the sender (e.g., 'user', 'assistant')
    """

    content: str
    id: str
    role: str


class RequestObject(BaseModel):
    """
    Model representing a complete chat request.

    Attributes:
        prompt (PromptObject): The user's message
        threadId (str): Unique identifier for the conversation thread
        responseId (str): Unique identifier for the expected response
    """

    prompt: PromptObject
    threadId: str
    responseId: str


# ==============================================================================
# API ENDPOINTS
# ==============================================================================


@app.post("/api/chat")
async def chat(request: RequestObject):
    """
    Handle chat requests and stream AI responses.

    This endpoint receives user messages, processes them through the AI agent,
    and streams back the response in real-time using Server-Sent Events (SSE).

    Args:
        request (RequestObject): The incoming chat request containing
                                 the user's message and thread information

    Returns:
        StreamingResponse: A streaming response that sends tokens as they
                          are generated by the AI model

    Example:
        POST /api/chat
        {
            "prompt": {"content": "What is Apple's stock price?", "id": "1", "role": "user"},
            "threadId": "thread-123",
            "responseId": "response-456"
        }
    """
    # Configure the agent with the conversation thread ID for context persistence
    config = {"configurable": {"thread_id": request.threadId}}

    def generate():
        """
        Generator function that yields tokens from the AI agent stream.

        The agent processes the user's message along with a system prompt
        that defines its role as a stock analysis assistant.
        """
        # System prompt defining the AI assistant's capabilities and behavior
        system_prompt = (
            "You are a stock analysis assistant. You have the ability to get "
            "real-time stock prices, historical stock prices (given a date range), "
            "news and balance sheet data for a given ticker symbol."
        )

        # Stream tokens from the agent
        for token, _ in agent.stream(
            {
                "messages": [
                    SystemMessage(system_prompt),
                    HumanMessage(request.prompt.content),
                ]
            },
            stream_mode="messages",
            config=config,
        ):
            yield token.content

    # Return a streaming response for real-time token delivery
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",  # Prevent caching
            "Connection": "keep-alive",  # Keep connection open for streaming
        },
    )


# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Run the FastAPI application with Uvicorn ASGI server
    # Host: 0.0.0.0 allows connections from any IP address
    # Port: 8888 is the default port for the API server
    uvicorn.run(app, host="0.0.0.0", port=8888)
