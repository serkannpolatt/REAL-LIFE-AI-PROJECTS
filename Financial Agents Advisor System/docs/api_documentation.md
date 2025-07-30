# API Documentation

## Overview

This document provides comprehensive API documentation for the Financial AI Agent system.

## Agent Controller API

### FinanceAgentController

Main controller for managing financial agents.

```python
from controller.agent_controller import FinanceAgentController
from model.models import AgentConfigManager

# Initialize
config_manager = AgentConfigManager()
controller = FinanceAgentController(config_manager)
```

#### Methods

##### `create_main_agent()`
Creates the main financial analysis agent.

**Returns:** `Agent` - Configured main agent

**Example:**
```python
main_agent = controller.create_main_agent()
response = main_agent.run("Analyze AAPL stock")
```

##### `create_portfolio_agent()`
Creates a specialized portfolio analysis agent.

**Returns:** `Agent` - Configured portfolio agent

**Example:**
```python
portfolio_agent = controller.create_portfolio_agent()
response = portfolio_agent.run("Optimize my portfolio")
```

## Agent Registry API

### AgentRegistry

Centralized registry for managing specialized agents.

```python
from agents import agent_registry, initialize_all_agents

# Initialize all agents
initialize_all_agents()
```

#### Methods

##### `register_agent(agent: BaseFinancialAgent)`
Register a new agent.

**Parameters:**
- `agent`: Agent instance to register

**Example:**
```python
from agents import TechnicalAnalysisAgent

tech_agent = TechnicalAnalysisAgent()
agent_registry.register_agent(tech_agent)
```

##### `get_all_agents() -> List[BaseFinancialAgent]`
Get all registered agents.

**Returns:** List of all registered agents

##### `get_agents_by_capability(capability: AgentCapability) -> List[BaseFinancialAgent]`
Get agents with specific capability.

**Parameters:**
- `capability`: Required capability

**Returns:** List of agents with the capability

**Example:**
```python
from agents import AgentCapability

risk_agents = agent_registry.get_agents_by_capability(
    AgentCapability.RISK_ANALYSIS
)
```

##### `get_agent_info() -> Dict[str, Dict]`
Get information about all registered agents.

**Returns:** Dictionary with agent information

## Individual Agent APIs

### TechnicalAnalysisAgent

Technical analysis and chart pattern recognition.

```python
from agents import TechnicalAnalysisAgent, AnalysisType

agent = TechnicalAnalysisAgent()
agent.initialize()
```

#### Analysis Types Supported
- `PRICE_MOVEMENT`
- `TREND_ANALYSIS`
- `VOLATILITY`

#### Example Usage
```python
result = agent.analyze('AAPL', AnalysisType.PRICE_MOVEMENT)
if result.success:
    print(f"Technical Analysis: {result.data}")
```

### SentimentAnalysisAgent

Market sentiment analysis from news and web sources.

```python
from agents import SentimentAnalysisAgent, AnalysisType

agent = SentimentAnalysisAgent()
agent.initialize()
```

#### Analysis Types Supported
- `SENTIMENT`

#### Example Usage
```python
result = agent.analyze('TSLA', AnalysisType.SENTIMENT)
if result.success:
    sentiment_score = result.data.get('sentiment_score')
    market_mood = result.data.get('market_mood')
```

### RiskAssessmentAgent

Comprehensive risk analysis and portfolio metrics.

```python
from agents import RiskAssessmentAgent, AnalysisType

agent = RiskAssessmentAgent()
agent.initialize()
```

#### Analysis Types Supported
- `RISK_METRICS`
- `VOLATILITY`

#### Example Usage
```python
result = agent.analyze('AAPL', AnalysisType.RISK_METRICS)
if result.success:
    var = result.data.get('value_at_risk')
    beta = result.data.get('beta')
```

### CryptoAnalysisAgent

Cryptocurrency market analysis with DeFi metrics.

```python
from agents import CryptoAnalysisAgent, AnalysisType

agent = CryptoAnalysisAgent()
agent.initialize()
```

#### Analysis Types Supported
- `PRICE_MOVEMENT`
- `VOLATILITY`
- `TREND_ANALYSIS`

#### Example Usage
```python
result = agent.analyze('BTC-USD', AnalysisType.PRICE_MOVEMENT)
if result.success:
    print(f"Crypto Analysis: {result.data}")
```

### PortfolioOptimizationAgent

Modern Portfolio Theory implementation and asset allocation.

```python
from agents import PortfolioOptimizationAgent, AnalysisType

agent = PortfolioOptimizationAgent()
agent.initialize()
```

#### Analysis Types Supported
- `PORTFOLIO_OPTIMIZATION`

#### Example Usage
```python
portfolio_query = "Optimize portfolio: 40% AAPL, 30% GOOGL, 20% MSFT, 10% TSLA"
result = agent.analyze(portfolio_query, AnalysisType.PORTFOLIO_OPTIMIZATION)
```

## Response Format

All agents return `AgentResponse` objects with the following structure:

```python
@dataclass
class AgentResponse:
    success: bool                    # Operation success status
    data: Optional[Dict] = None      # Analysis results
    error: Optional[str] = None      # Error message if failed
    confidence: float = 0.0          # Confidence score (0-1)
    analysis_type: Optional[AnalysisType] = None  # Type of analysis performed
    timestamp: datetime = field(default_factory=datetime.now)  # Timestamp
```

### Example Response Handling

```python
result = agent.analyze('AAPL', AnalysisType.PRICE_MOVEMENT)

if result.success:
    print(f"Analysis successful with {result.confidence:.2%} confidence")
    print(f"Data: {result.data}")
else:
    print(f"Analysis failed: {result.error}")
```

## Configuration API

### AgentConfigManager

Manages configuration for all agents.

```python
from model.models import AgentConfigManager

config_manager = AgentConfigManager()
```

#### Methods

##### `get_main_agent_config() -> Dict`
Get configuration for the main agent.

##### `get_portfolio_agent_config() -> Dict`
Get configuration for the portfolio agent.

##### `get_groq_model() -> Any`
Get configured Groq model instance.

## Utility APIs

### API Key Management

```python
from utils.api_utils import APIKeyManager, setup_environment

# Check system setup
if setup_environment():
    print("System ready!")

# Manage API keys
manager = APIKeyManager()
status = manager.validate_api_keys()
print(manager.get_status_report())
```

### Data Processing

```python
from utils.data_utils import DataProcessor, MarketDataValidator

# Validate stock symbol
is_valid = DataProcessor.validate_stock_symbol('AAPL')

# Process financial data
cleaned_data = DataProcessor.clean_financial_data(raw_data)

# Validate data quality
quality_score = MarketDataValidator.get_data_quality_score(data)
```

### Logging

```python
from utils.logger_utils import get_logger, setup_logging

# Set up logging
logger = setup_logging('INFO')

# Use logger
logger.info("System started")
```

## Error Handling

### Common Error Types

1. **API Key Missing**: Required API keys not configured
2. **Data Unavailable**: Market data not available for symbol
3. **Agent Not Initialized**: Agent not properly initialized
4. **Invalid Symbol**: Stock symbol format invalid
5. **Network Error**: Connection issues with data providers

### Error Response Example

```python
{
    "success": false,
    "error": "Invalid stock symbol: INVALID123",
    "confidence": 0.0,
    "analysis_type": "price_movement",
    "timestamp": "2024-07-30T10:30:00"
}
```

## Rate Limiting

The system implements automatic rate limiting for external APIs:

- **YFinance**: Respects Yahoo Finance rate limits
- **Groq API**: Handles API rate limiting gracefully
- **DuckDuckGo**: Implements request throttling

## Best Practices

1. **Always check `result.success`** before accessing data
2. **Handle errors gracefully** with appropriate user feedback
3. **Use appropriate analysis types** for each agent
4. **Initialize agents once** and reuse them
5. **Validate input symbols** before analysis
6. **Implement proper logging** for debugging
7. **Use the agent registry** for multi-agent coordination
