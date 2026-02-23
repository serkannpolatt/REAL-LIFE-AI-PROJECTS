# 🔧 AI Stock Analysis Assistant - Backend

The backend server for the AI Stock Analysis Assistant, built with FastAPI and LangChain.

## 📋 Overview

This backend provides:
- RESTful API endpoints for chat functionality
- AI agent integration with LangChain
- Real-time stock data through Yahoo Finance
- Streaming response support via Server-Sent Events (SSE)

## 🏗 Architecture

```
main.py
├── Configuration
│   ├── FastAPI App Setup
│   ├── AI Model Configuration
│   └── Memory Checkpointer
├── Stock Data Tools
│   ├── get_stock_price()
│   ├── get_historical_stock_price()
│   ├── get_balance_sheet()
│   └── get_stock_news()
├── Request/Response Models
│   ├── PromptObject
│   └── RequestObject
└── API Endpoints
    └── POST /api/chat
```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | ≥0.123.7 | Web framework for building APIs |
| langchain[openai] | ≥1.1.0 | AI agent framework |
| pydantic | ≥2.12.5 | Data validation |
| python-dotenv | ≥1.2.1 | Environment variable management |
| uvicorn | ≥0.38.0 | ASGI server |
| yfinance | ≥0.2.66 | Yahoo Finance data |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install fastapi langchain[openai] pydantic python-dotenv uvicorn yfinance
```

### 2. Configure Environment

Create a `.env` file:

```env
THESYS_API_KEY=your_api_key_here
```

### 3. Run the Server

```bash
# Development mode with auto-reload
uv run uvicorn main:app --reload --port 8888

# Production mode
uv run python main.py
```

## 🔌 API Endpoints

### POST `/api/chat`

Process chat messages and return AI-generated responses.

**Request:**
```json
{
  "prompt": {
    "content": "What is AAPL stock price?",
    "id": "msg-123",
    "role": "user"
  },
  "threadId": "thread-456",
  "responseId": "resp-789"
}
```

**Response:** Server-Sent Events stream with AI tokens

## 🛠 Available Tools

### 1. `get_stock_price`
Retrieves the current closing price for a stock.

```python
get_stock_price(ticker="AAPL")
# Returns: 175.50
```

### 2. `get_historical_stock_price`
Retrieves historical price data for a date range.

```python
get_historical_stock_price(
    ticker="AAPL",
    start_date="2024-01-01",
    end_date="2024-06-30"
)
# Returns: {'Open': {...}, 'High': {...}, 'Low': {...}, 'Close': {...}, 'Volume': {...}}
```

### 3. `get_balance_sheet`
Retrieves company financial statements.

```python
get_balance_sheet(ticker="AAPL")
# Returns: DataFrame with balance sheet data
```

### 4. `get_stock_news`
Retrieves recent news articles for a stock.

```python
get_stock_news(ticker="AAPL")
# Returns: [{'title': '...', 'link': '...', ...}]
```

## 📁 File Structure

```
backend/
├── main.py           # Main application code
├── pyproject.toml    # Python project configuration
├── README.md         # This file
└── .env              # Environment variables (create this)
```

## 🔒 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `THESYS_API_KEY` | Yes | API key for TheSys AI service |
| `OPENAI_API_KEY` | No | OpenAI API key (alternative) |

## 🐛 Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Issues

1. **Connection Refused**: Ensure the server is running on port 8888
2. **API Key Error**: Check that your `.env` file is properly configured
3. **yfinance Errors**: Some tickers may not be available or may have limited data


