# 🤖 Advanced Financial AI Agent System

> **Comprehensive financial analysis platform powered by specialized AI agents**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA3-green.svg)](https://groq.com)
[![PhiData](https://img.shields.io/badge/Framework-PhiData-orange.svg)](https://github.com/phidatahq/phidata)

This project is a **comprehensive financial analysis AI system** built with modular architecture using **Groq LLaMA3**, **YFinance**, and **DuckDuckGo** technologies. It features a multi-agent system with specialized capabilities for stock analysis, market research, risk assessment, and portfolio optimization.

---

## 🎯 Key Features

### 🧠 **Specialized AI Agents**
- **Technical Analysis Agent**: Advanced technical indicators, chart patterns, and trading signals
- **Sentiment Analysis Agent**: Market sentiment analysis from news and web sources
- **Risk Assessment Agent**: Comprehensive risk metrics, VaR calculations, and stress testing
- **Cryptocurrency Agent**: Crypto market analysis with DeFi and on-chain metrics
- **Portfolio Optimization Agent**: Modern Portfolio Theory implementation and asset allocation

### 🌐 **Real-time Data Integration**
- **Live Market Data**: Yahoo Finance API integration for real-time stock prices
- **News Analysis**: Current company and market news aggregation and sentiment analysis
- **Web Research**: DuckDuckGo integration for comprehensive market research
- **Multi-source Validation**: Cross-reference data from multiple sources

### ⚡ **Performance & Scalability**
- **Fast Processing**: Optimized AI processing with Groq LLaMA3
- **Streaming Responses**: Real-time response display
- **Agent Registry**: Centralized management of specialized agents
- **Comprehensive Logging**: System monitoring and error tracking

---

## 🏗️ System Architecture

```
Finance-Advisor-AI-Agent/
├── 🎯 app.py                           # Main application and interactive interface
├── 📋 requirements.txt                 # Project dependencies
├── 📖 README.md                       # Project documentation
├── � .env.example                    # Environment variables template
├── ⚙️ config.py                       # System configuration
├── � examples.py                     # Comprehensive usage examples
├── 🧪 test_integration.py            # System integration tests
│
├── 🎮 controller/                     # Agent Management Layer
│   └── agent_controller.py            # Multi-agent coordination and management
│
├── 🧠 model/                          # Configuration Layer
│   └── models.py                      # Agent configurations and tool integrations
│
└── 🤖 agents/                         # Specialized Agent Framework
    ├── __init__.py                    # Agent registry and initialization
    ├── base_agent.py                  # Abstract base classes and interfaces
    ├── technical_agent.py             # Technical analysis and chart patterns
    ├── sentiment_agent.py             # Market sentiment and news analysis
    ├── risk_agent.py                  # Risk assessment and portfolio metrics
    ├── crypto_agent.py                # Cryptocurrency market analysis
    └── portfolio_agent.py             # Portfolio optimization and asset allocation
│
├── 🔧 utils/                          # Utility Functions
│   ├── __init__.py                    # Utilities package initialization
│   ├── api_utils.py                   # API key management and validation
│   ├── data_utils.py                  # Data processing and validation
│   └── logger_utils.py                # Centralized logging system
│
├── 🧪 tests/                          # Test Suite
│   ├── __init__.py                    # Test package initialization
│   ├── test_integration.py            # System integration tests
│   └── test_agents.py                 # Individual agent unit tests
│
├── 📚 docs/                           # Documentation
│   ├── agent_architecture.md          # Agent system architecture
│   └── api_documentation.md           # Comprehensive API documentation
│
└── 📜 scripts/                        # Utility Scripts
    └── setup.py                       # System setup and validation script
```

### 🔄 **Agent-Based Architecture**

- **Modular Design**: Each agent specializes in specific financial analysis domains
- **Extensible Framework**: Easy integration of new agents and capabilities
- **Centralized Registry**: Unified management and coordination of all agents
- **Standardized Interfaces**: Consistent APIs across all specialized agents

---

## 🚀 Quick Start

### 📋 **Prerequisites**

- Python 3.8+ 
- Groq API key (free tier available)
- Internet connection for real-time data

### 🔧 **Installation Steps**

#### Option 1: Automated Setup (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/username/finance-advisor-ai-agent.git
cd finance-advisor-ai-agent
```

2. **Run automated setup script**
```bash
python scripts/setup.py
```

The setup script will:
- ✅ Check Python version compatibility
- ✅ Verify required dependencies
- ✅ Create necessary directories
- ✅ Set up environment file from template
- ✅ Validate API key configuration
- ✅ Run basic system test

#### Option 2: Manual Setup

1. **Clone the repository**
```bash
git clone https://github.com/username/finance-advisor-ai-agent.git
cd finance-advisor-ai-agent
```

2. **Create virtual environment (recommended)**
```bash
python -m venv finance_ai_env
finance_ai_env\Scripts\activate  # Windows
source finance_ai_env/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file and add the following:
GROQ_API_KEY=your_groq_api_key_here
```

5. **Run the application**
```bash
python app.py
```

---

## 💡 Usage Examples

### 🎯 **Basic Stock Analysis**
```python
from agents import TechnicalAnalysisAgent, AnalysisType

# Initialize technical analysis agent
tech_agent = TechnicalAnalysisAgent()
tech_agent.initialize()

# Analyze Apple stock
result = tech_agent.analyze('AAPL', AnalysisType.PRICE_MOVEMENT)
print(f"Analysis: {result.data}")
```

### 📊 **Multi-Agent Analysis**
```python
from agents import initialize_all_agents, agent_registry

# Initialize all specialized agents
initialize_all_agents()

# Get all available agents
agents = agent_registry.get_all_agents()
print(f"Available agents: {[agent.name for agent in agents]}")

# Perform coordinated analysis
for agent in agents:
    result = agent.analyze('TSLA', AnalysisType.PRICE_MOVEMENT)
    print(f"{agent.name}: {result.data if result.success else result.error}")
```

### 📈 **Portfolio Optimization**
```python
from agents import PortfolioOptimizationAgent, AnalysisType

# Initialize portfolio optimization agent
portfolio_agent = PortfolioOptimizationAgent()
portfolio_agent.initialize()

# Optimize portfolio allocation
portfolio_query = "Optimize portfolio: 40% AAPL, 30% GOOGL, 20% MSFT, 10% TSLA"
result = portfolio_agent.analyze(portfolio_query, AnalysisType.PORTFOLIO_OPTIMIZATION)
print(f"Optimization: {result.data}")
```

### 🔍 **Interactive Demo**
```bash
# Run comprehensive examples
python examples.py

# Available options:
# 1. Basic Financial Analysis
# 2. Technical Analysis
# 3. Sentiment Analysis  
# 4. Risk Assessment
# 5. Cryptocurrency Analysis
# 6. Portfolio Optimization
# 7. Multi-Agent Analysis
# 8. Agent Registry Management
# 9. Interactive Demo Mode
```

---

## 🛠️ Advanced Configuration

### 🎛️ **Agent Customization**

The system supports creating specialized agent configurations:

```python
from model.models import AgentConfigManager
from controller.agent_controller import FinanceAgentController

# Initialize configuration manager
config_manager = AgentConfigManager()
controller = FinanceAgentController(config_manager)

# Create specialized agents
main_agent = controller.create_main_agent()
portfolio_agent = controller.create_portfolio_agent()
```

### 📊 **Agent Registry Management**

```python
from agents import agent_registry, AgentCapability

# Register agents
agent_registry.register_agent(tech_agent)

# Find agents by capability
risk_agents = agent_registry.get_agents_by_capability(AgentCapability.RISK_ANALYSIS)

# Get agent information
agent_info = agent_registry.get_agent_info()
print(agent_info)
```

### 🔧 **Custom Analysis Types**

```python
from agents import AnalysisType

# Available analysis types:
# - PRICE_MOVEMENT: Stock price movement analysis
# - TREND_ANALYSIS: Technical trend identification  
# - VOLATILITY: Volatility calculations and analysis
# - SENTIMENT: Market sentiment from news and social media
# - RISK_METRICS: Comprehensive risk assessment
# - PORTFOLIO_OPTIMIZATION: Modern Portfolio Theory optimization
```

---

## 🧪 Testing

### 🔬 **Run Integration Tests**
```bash
# Run system integration tests
python tests/test_integration.py

# Run individual agent tests
python tests/test_agents.py

# Expected output:
# ✓ Agent creation successful
# ✓ Agent registry functionality working
# ✓ Configuration manager working
# ✓ System integration working with X agents
```

### 🛠️ **Development Testing**
```python
# Test individual agents
from agents import TechnicalAnalysisAgent

agent = TechnicalAnalysisAgent()
assert agent is not None
assert not agent.is_initialized  # Before initialization

# Test agent registry
from agents import agent_registry, initialize_all_agents

initialize_all_agents()
agents = agent_registry.get_all_agents()
assert len(agents) > 0
```

### 📊 **System Validation**
```bash
# Run setup script to validate system
python scripts/setup.py

# This will check:
# - Python version compatibility
# - Required dependencies
# - API key configuration
# - Basic system functionality
```

### �️ **Development Testing**
```python
# Test individual agents
from agents import TechnicalAnalysisAgent

agent = TechnicalAnalysisAgent()
assert agent is not None
assert not agent.is_initialized  # Before initialization
```

---

## 📊 Agent Capabilities

### 🔍 **Technical Analysis Agent**
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Chart Patterns**: Support/Resistance, Trend Lines, Pattern Recognition
- **Volume Analysis**: Volume indicators and price-volume relationships
- **Signal Generation**: Buy/Sell signals based on technical criteria

### 📰 **Sentiment Analysis Agent**
- **News Processing**: Financial news aggregation and analysis
- **Sentiment Scoring**: Advanced sentiment classification algorithms
- **Market Mood**: Overall market sentiment indicators
- **Source Filtering**: Credible financial news source prioritization

### ⚠️ **Risk Assessment Agent**
- **Value at Risk (VaR)**: Monte Carlo and historical simulation methods
- **Portfolio Metrics**: Beta, Sharpe ratio, maximum drawdown
- **Stress Testing**: Scenario analysis and sensitivity testing
- **Risk Classification**: Risk level categorization and recommendations

### 🪙 **Cryptocurrency Analysis Agent**
- **Crypto Indicators**: Crypto-specific technical analysis
- **DeFi Metrics**: Decentralized finance protocol analysis
- **On-chain Analysis**: Blockchain data interpretation (simulated)
- **Market Correlation**: Crypto-traditional market relationships

### 📊 **Portfolio Optimization Agent**
- **Modern Portfolio Theory**: Efficient frontier calculation
- **Asset Allocation**: Strategic and tactical allocation strategies
- **Optimization Algorithms**: Mathematical portfolio optimization
- **Backtesting**: Historical performance simulation

---

## 🔧 API Integrations

### 📈 **YFinance (Yahoo Finance)**
- Real-time stock prices and historical data
- Financial statements and fundamental metrics
- Analyst recommendations and price targets
- Company news and announcements

### 🔍 **DuckDuckGo Search**
- Current market news aggregation
- Company announcements and press releases
- Economic indicators and analysis
- Sector developments and trends

### 🤖 **Groq LLaMA3**
- Ultra-fast language model processing
- Advanced financial analysis capabilities
- Context-aware response generation
- Multi-language support

---

## 🚨 Important Notes

### ⚠️ **Investment Disclaimer**
```
🔴 IMPORTANT: This system is for informational purposes only.
   Before making investment decisions:
   
   ✅ Consult with professional financial advisors
   ✅ Assess your own risk tolerance
   ✅ Conduct research from multiple sources
   ✅ Only invest amounts you can afford to lose
```

### 🔒 **Security Best Practices**
- Never share API keys in public repositories
- Add `.env` files to `.gitignore`
- Regularly rotate API keys
- Use environment variables for sensitive data

---
