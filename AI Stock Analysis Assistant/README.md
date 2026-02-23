# 🤖 AI Stock Analysis Assistant

A powerful AI-powered stock analysis chatbot that provides real-time stock information, historical data, financial statements, and news. Built with FastAPI, LangChain, and React.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12+-green.svg)
![React](https://img.shields.io/badge/react-19.2+-blue.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.123+-teal.svg)

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### Stock Data Capabilities
- **Real-time Stock Prices**: Get current stock prices for any ticker symbol
- **Historical Data**: Retrieve historical stock prices with custom date ranges
- **Balance Sheets**: Access company financial statements and balance sheet data
- **Stock News**: Get latest news articles related to specific stocks

### Technical Features
- **AI-Powered Responses**: Intelligent responses using GPT-5 via LangChain
- **Streaming Responses**: Real-time token streaming for instant feedback
- **Conversation Memory**: Maintains context within conversation threads
- **Modern UI**: Beautiful dark-themed chat interface
- **Type Safety**: Full TypeScript support in the frontend

## 🏗 Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  React Frontend │────▶│  FastAPI Server │────▶│  Yahoo Finance  │
│  (Port 3000)    │     │  (Port 8888)    │     │     API         │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │                 │
                        │  LangChain +    │
                        │  GPT-5 Model    │
                        │                 │
                        └─────────────────┘
```

### Data Flow
1. User enters a question in the chat interface
2. Frontend sends the message to the FastAPI backend via `/api/chat`
3. Backend processes the message through the LangChain agent
4. Agent determines which tools to use (stock price, history, news, etc.)
5. Tools fetch data from Yahoo Finance API
6. AI generates a response based on the retrieved data
7. Response is streamed back to the frontend in real-time

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12+** - [Download Python](https://www.python.org/downloads/)
- **Node.js 18+** - [Download Node.js](https://nodejs.org/)
- **Git** - [Download Git](https://git-scm.com/downloads/)
- **uv** (Python package manager) - [Install uv](https://github.com/astral-sh/uv)

## 🚀 Installation



## ⚙ Configuration

### Environment Variables

Create a `.env` file in the `backend` directory:

```env
# TheSys API Key (required for AI functionality)
THESYS_API_KEY=your_api_key_here

# Optional: OpenAI API Key (if using OpenAI directly)
OPENAI_API_KEY=your_openai_api_key
```

### Getting API Keys

1. **TheSys API Key**: Sign up at [TheSys](https://thesys.dev) to get your API key
2. The API key is required for the AI agent to function

## 🎮 Usage

### Starting the Backend Server

```bash
cd backend

# Using uv
uv run python main.py

# Or directly with Python
python main.py
```

The backend server will start at `http://localhost:8888`

### Starting the Frontend Development Server

```bash
cd frontend

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Example Queries

Once both servers are running, you can ask questions like:

- "What is Apple's current stock price?"
- "Show me Tesla's stock history from January to June 2024"
- "Get me the latest news about Microsoft"
- "What does Amazon's balance sheet look like?"
- "Compare Google and Meta stock prices"

## 📡 API Reference

### POST `/api/chat`

Send a chat message and receive a streaming response.

#### Request Body

```json
{
  "prompt": {
    "content": "What is Apple's stock price?",
    "id": "unique-message-id",
    "role": "user"
  },
  "threadId": "conversation-thread-id",
  "responseId": "expected-response-id"
}
```

#### Response

Server-Sent Events (SSE) stream containing AI-generated tokens.

#### Headers

| Header | Value |
|--------|-------|
| Content-Type | text/event-stream |
| Cache-Control | no-cache, no-transform |
| Connection | keep-alive |

### Available Tools

The AI agent has access to the following tools:

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_stock_price` | Get current stock price | `ticker: string` |
| `get_historical_stock_price` | Get historical prices | `ticker: string`, `start_date: string`, `end_date: string` |
| `get_balance_sheet` | Get company balance sheet | `ticker: string` |
| `get_stock_news` | Get stock-related news | `ticker: string` |

## 📁 Project Structure

```
ai-stock-analysis-assistant/
├── README.md                 # This file
├── backend/                  # Python FastAPI backend
│   ├── main.py              # Main application entry point
│   ├── pyproject.toml       # Python dependencies and config
│   ├── README.md            # Backend-specific documentation
│   └── .env                 # Environment variables (create this)
├── frontend/                 # React TypeScript frontend
│   ├── src/
│   │   ├── App.tsx          # Main React component
│   │   ├── App.css          # Application styles
│   │   ├── main.tsx         # React entry point
│   │   └── assets/          # Static assets
│   ├── public/              # Public static files
│   ├── index.html           # HTML entry point
│   ├── package.json         # Node.js dependencies
│   ├── vite.config.ts       # Vite configuration
│   ├── tsconfig.json        # TypeScript configuration
│   └── README.md            # Frontend-specific documentation
```

## 🛠 Technologies Used

### Backend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance web framework |
| **LangChain** | AI agent framework and tools |
| **LangGraph** | Agent memory and checkpointing |
| **yfinance** | Yahoo Finance API wrapper |
| **Pydantic** | Data validation |
| **Uvicorn** | ASGI server |
| **python-dotenv** | Environment configuration |

### Frontend
| Technology | Purpose |
|------------|---------|
| **React 19** | UI framework |
| **TypeScript** | Type-safe JavaScript |
| **Vite** | Build tool and dev server |
| **TheSys GenUI SDK** | Pre-built chat components |
| **Crayon AI React UI** | UI component styles |

### AI/ML
| Technology | Purpose |
|------------|---------|
| **GPT-5** | Large language model |
| **TheSys API** | AI model hosting |
| **LangChain Tools** | Function calling capabilities |

## 🔧 Development

### Running in Development Mode

```bash
# Terminal 1: Start backend with auto-reload
cd backend
uv run uvicorn main:app --reload --port 8888

# Terminal 2: Start frontend with HMR
cd frontend
npm run dev
```

### Building for Production

```bash
# Build frontend
cd frontend
npm run build

# The built files will be in frontend/dist/
```

### Linting

```bash
# Frontend linting
cd frontend
npm run lint
```
