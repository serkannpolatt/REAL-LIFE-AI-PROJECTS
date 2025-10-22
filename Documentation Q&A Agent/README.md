# 📚 Documentation Q&A Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Demo](./assets/demo.gif)

An intelligent Streamlit-based AI agent that transforms static documentation into interactive, conversational knowledge bases. Built with the Model Context Protocol (MCP) and powered by Nebius AI for natural language understanding and contextual responses.

## ✨ Key Features

### 🤖 **Intelligent Conversational Interface**
- **Natural Language Queries** - Ask questions in plain English about any documentation
- **Contextual Understanding** - AI maintains conversation context for follow-up questions
- **Source Citations** - Responses include references to specific documentation sections
- **Real-time Processing** - Instant responses with streaming output

### 🔗 **Advanced MCP Integration**
- **Model Context Protocol** - Seamless integration with documentation servers
- **Multiple Transport Methods** - HTTP streaming and WebSocket support
- **Tool Orchestration** - Dynamic tool selection for optimal information retrieval
- **Extensible Architecture** - Easy integration with custom documentation sources

### 🎯 **User Experience Excellence**
- **Interactive Chat Interface** - Clean, intuitive Streamlit-based UI
- **Example-Driven Onboarding** - Pre-built questions for quick exploration
- **Session Management** - Persistent conversation history during sessions
- **Responsive Design** - Optimized for desktop and mobile viewing

### � **Security & Configuration**
- **Secure API Management** - Safe handling of API credentials with validation
- **Environment Configuration** - Flexible settings via environment variables
- **Error Handling** - Comprehensive error recovery and user feedback
- **Connection Testing** - Built-in tools to verify server connectivity

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   Agno Agent     │    │ Documentation   │
│   (Frontend)    │◄──►│   (Orchestrator) │◄──►│ MCP Server      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Session State   │    │   MCP Tools      │    │   Nebius AI     │
│ (Chat History)  │    │  (Doc Access)    │    │ (LLM Processing)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10 or higher**
- **Nebius AI API Key** - Sign up at [Nebius Studio](https://studio.nebius.ai/)
- **Documentation MCP Server** - Access to a documentation server or use default
- **Internet Connection** - For API calls and documentation access


## 📖 Usage Guide

### Getting Started

1. **Enter API Key**
   - Navigate to the sidebar
   - Enter your Nebius API key
   - Look for validation confirmation

2. **Configure Documentation Source**
   - Set the documentation MCP server URL
   - Use the default Nebius docs or enter your own
   - Test connection with the built-in validator

3. **Start Asking Questions**
   - Use the example questions for quick exploration
   - Type custom questions in the chat input
   - Follow up with additional questions for clarification

### Example Interactions

**Basic Documentation Queries**
```
Q: How to create an Agent with Google SDK & Nebius?
A: [Detailed response with code examples and step-by-step instructions]

Q: What are the authentication requirements?
A: [Comprehensive auth guide with API key setup and security best practices]
```

**Advanced Feature Exploration**
```
Q: How to implement streaming responses?
A: [Technical implementation details with code snippets]

Q: Can you provide more specific examples for fine-tuning?
A: [Detailed examples building on previous context]
```

**Integration Questions**
```
Q: How does this integrate with other AI tools?
A: [Integration patterns and compatibility information]
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEBIUS_API_KEY` | Nebius AI API key for model access | - | ✅ |
| `DOCUMENTATION_URL` | Default MCP server URL | `https://docs.studio.nebius.com/mcp` | ❌ |
| `LOG_LEVEL` | Application logging level | `INFO` | ❌ |
| `STREAMLIT_SERVER_PORT` | Custom port for Streamlit | `8501` | ❌ |

### Model Configuration

The application uses the following AI models by default:
- **Primary Model**: `deepseek-ai/DeepSeek-V3-0324`
- **Temperature**: `0.1` (optimized for factual responses)
- **Max Tokens**: `4096` (for detailed responses)

### MCP Transport Options

- **Streamable HTTP** - Default, works with most documentation servers
- **WebSocket** - For real-time documentation servers
- **Custom Transport** - Extensible for proprietary protocols

## 🔧 Advanced Usage

### Custom Documentation Sources

To integrate with your own documentation:

1. **Set up MCP Server**
   ```bash
   # Example with custom documentation
   export DOCUMENTATION_URL="https://your-docs.com/mcp"
   ```

2. **Configure Access**
   ```python
   # Custom MCP configuration
   mcp_tools = MCPTools(
       url="https://your-docs.com/mcp",
       transport="streamable-http",
       headers={"Authorization": "Bearer your-token"}
   )
   ```

### Extending Functionality

**Add Custom Instructions**
```python
agent = Agent(
    model=Nebius(id="deepseek-ai/DeepSeek-V3-0324", api_key=api_key),
    tools=[mcp_tools],
    instructions="""
    Custom instructions for specialized documentation handling:
    - Focus on technical implementation details
    - Provide code examples in Python
    - Include performance considerations
    """
)
```

**Custom Example Questions**
```python
CUSTOM_EXAMPLES = [
    "How to optimize performance for large datasets?",
    "What are the security best practices?",
    "How to implement custom authentication?"
]
```

## 🔍 Troubleshooting

### Common Issues

**Authentication Errors**
```bash
❌ Invalid API key format
```
**Solution**: Verify your Nebius API key is correct and active

**Connection Issues**
```bash
🌐 Connection Error: Unable to reach documentation server
```
**Solutions**:
- Check internet connectivity
- Verify documentation URL is accessible
- Test with default Nebius documentation URL

**Performance Issues**
```bash
⏳ Slow response times
```
**Solutions**:
- Check API rate limits
- Simplify complex queries
- Use more specific questions

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
streamlit run main.py
```

Check application logs:
```bash
# View real-time logs
tail -f ~/.streamlit/logs/streamlit.log
```

### Support Resources

- **API Documentation**: [Nebius Studio Docs](https://docs.studio.nebius.ai/)
- **MCP Specification**: [Model Context Protocol](https://modelcontextprotocol.io/)
- **Streamlit Guide**: [Streamlit Documentation](https://docs.streamlit.io/)

## 🧪 Development

### Project Structure
```
docs_qna_agent/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
├── .env.example           # Environment variables template
├── README.md              # This documentation
├── assets/                # Static assets
│   ├── Nebius.png         # Branding image
│   └── demo.gif           # Demo animation
└── tests/                 # Test suite (future)
```

### Code Quality

**Formatting and Linting**
```bash
# Format code
black main.py
isort main.py

# Type checking (if added)
mypy main.py

# Linting
flake8 main.py
```

**Testing**
```bash
# Run tests (when implemented)
pytest tests/

# Coverage report
pytest --cov=main tests/
```

### Contributing

1. **Fork the Repository**
   ```bash
   git fork https://github.com/original/docs-qna-agent.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow existing code style
   - Add appropriate error handling
   - Update documentation

4. **Submit Pull Request**
   - Include clear description
   - Add tests if applicable
   - Update README if needed

## 📊 Performance Metrics

### Response Times
- **Average Query Processing**: 2-5 seconds
- **MCP Connection Setup**: <1 second
- **Streaming Response Start**: <500ms

### Supported Scale
- **Concurrent Users**: 10-50 (depends on Streamlit deployment)
- **Documentation Size**: No inherent limits (depends on MCP server)
- **Chat History**: Session-based (cleared on refresh)

## 🚀 Deployment

### Local Development
```bash
streamlit run main.py --server.port 8501
```

### Production Deployment

**Docker Deployment**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0"]
```

**Streamlit Cloud**
- Push to GitHub repository
- Connect via [Streamlit Cloud](https://streamlit.io/cloud)
- Configure environment variables in dashboard


## 🙏 Acknowledgments

- **Nebius AI** - For providing powerful language models and embedding services
- **Streamlit Team** - For the excellent web application framework
- **Model Context Protocol** - For standardizing AI-documentation integration
- **Agno Framework** - For AI agent orchestration and tool management

## 🤝 Community

- **Issues**: [GitHub Issues](https://github.com/your-repo/docs-qna-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/docs-qna-agent/discussions)
- **Documentation**: [Project Wiki](https://github.com/your-repo/docs-qna-agent/wiki)

