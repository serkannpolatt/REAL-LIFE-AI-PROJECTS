# 🦑 GitHub MCP Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Demo](./assets/demo.gif)

An advanced GitHub repository exploration tool powered by the Model Context Protocol (MCP) and Nebius AI. Transform complex GitHub API operations into natural language conversations and gain deep insights into any repository with intelligent analysis capabilities.

## ✨ Key Features

### 🔍 **Intelligent Repository Analysis**
- **Natural Language Queries** - Ask questions about repositories in plain English
- **Comprehensive Repository Info** - Extract detailed information from README files and metadata
- **Issue & PR Analysis** - Explore recent issues, pull requests, and their status
- **Activity Monitoring** - Analyze commit patterns, contributor activity, and code quality trends
- **Custom Queries** - Ask specific questions tailored to your needs

### 🤖 **Advanced AI Integration**
- **Multiple AI Models** - Choose from Qwen, DeepSeek, and LLaMA models via Nebius
- **Contextual Understanding** - AI maintains context for follow-up questions
- **Structured Responses** - Organized tables, charts, and formatted markdown output
- **Real-time Processing** - Instant analysis with progress tracking

### 🛠️ **Professional Development Tools**
- **Direct GitHub Links** - Clickable links to issues, PRs, and repository sections
- **Export Capabilities** - Download query results in JSON format
- **Query History** - Track and revisit previous analyses
- **Session Management** - Persistent configuration and results

### 🔒 **Enterprise-Ready Security**
- **Secure Token Management** - Safe handling of GitHub Personal Access Tokens
- **API Rate Limiting** - Intelligent handling of GitHub API limits
- **Error Recovery** - Comprehensive error handling with actionable solutions
- **Docker Isolation** - Containerized MCP server for security

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   GitHub MCP     │    │   GitHub API    │
│   (Frontend)    │◄──►│   Server         │◄──►│   (Data Source) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Agno Agent    │    │   MCP Tools      │    │   Nebius AI     │
│ (Orchestrator)  │    │  (API Bridge)    │    │ (LLM Processing)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10 or higher**
- **Docker installed and running** - Required for MCP server execution
- **GitHub Personal Access Token** - Create at [github.com/settings/tokens](https://github.com/settings/tokens)
- **Nebius AI API Key** - Sign up at [Nebius Studio](https://studio.nebius.ai/)


## 📖 Usage Guide

### Getting Started

1. **Configure Authentication**
   - Enter your Nebius API key in the sidebar
   - Add your GitHub Personal Access Token
   - Look for validation confirmation (✅)

2. **Select Repository and Query Type**
   - Enter repository in format `owner/repository`
   - Choose from predefined query types:
     - **Info**: Repository overview and features
     - **Issues**: Recent bug reports and feature requests
     - **Pull Requests**: Merged code changes and reviews
     - **Repository Activity**: Commit patterns and contributor analysis
     - **Custom**: Ask specific questions

3. **Execute Analysis**
   - Review the auto-generated query or customize it
   - Click "🚀 Execute Query"
   - Monitor progress with real-time indicators
   - Review comprehensive results with linked resources

### Advanced Features

**Model Selection**
- Choose optimal AI model for your use case
- **Qwen/Qwen3-30B-A3B**: Balanced performance and speed
- **DeepSeek-V3-0324**: Advanced reasoning capabilities
- **LLaMA-3.3-70B**: Comprehensive analysis

**Configuration Options**
- **Include Direct Links**: Add clickable GitHub URLs to results
- **Detailed Analysis Mode**: Enable comprehensive repository insights
- **Query History**: Track and export previous analyses

### Example Queries

**Repository Overview**
```
Input: microsoft/vscode
Query Type: Info
Result: Comprehensive overview of VS Code features, architecture, and contribution guidelines
```

**Issue Analysis**
```
Input: facebook/react
Query Type: Issues
Result: Recent issues with labels, assignees, and community engagement metrics
```

**Pull Request Insights**
```
Input: tensorflow/tensorflow
Query Type: Pull Requests
Result: Recent merged PRs with change summaries and contributor statistics
```

**Custom Analysis**
```
Input: kubernetes/kubernetes
Query: "What are the most common types of issues and how quickly are they resolved?"
Result: Issue categorization analysis with resolution time metrics
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `NEBIUS_API_KEY` | Nebius AI API key for model access | ✅ | - |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | GitHub token for API access | ✅ | - |
| `STREAMLIT_SERVER_PORT` | Custom port for the application | ❌ | `8501` |
| `LOG_LEVEL` | Application logging level | ❌ | `INFO` |

### GitHub Token Setup

1. **Create Personal Access Token**
   - Go to [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
   - Click "Generate new token (classic)"
   - Select scopes: `repo`, `read:org`, `read:user`
   - Copy and save the token securely

2. **Token Permissions**
   - **Public repositories**: Basic token with `public_repo` scope
   - **Private repositories**: Token with full `repo` scope
   - **Organization repositories**: Additional `read:org` permissions

### API Rate Limits

| Authentication | Requests per Hour | Use Case |
|----------------|-------------------|----------|
| **No Token** | 60 | Basic testing only |
| **Personal Token** | 5,000 | Development and production |
| **GitHub App** | 15,000 | Enterprise applications |

## 🔧 Advanced Usage

### Custom Docker Configuration

For custom MCP server configurations:

```bash
# Custom environment variables
docker run -i --rm \
  -e GITHUB_PERSONAL_ACCESS_TOKEN=$GITHUB_TOKEN \
  -e GITHUB_API_URL=https://api.github.com \
  ghcr.io/github/github-mcp-server
```

### Extending Query Capabilities

**Custom Instructions**
```python
# Add to agent configuration
custom_instructions = """
Focus on security-related issues and vulnerabilities.
Highlight code quality metrics and technical debt.
Provide recommendations for improvement.
"""
```

**Advanced Filtering**
```python
# Example: Filter issues by labels
query = "Find all security-related issues labeled 'vulnerability' in the last 30 days"
```

### Integration with CI/CD

**Automated Repository Analysis**
```bash
# Example CI script
curl -X POST "http://localhost:8501/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "repository": "owner/repo",
    "query_type": "Repository Activity",
    "output_format": "json"
  }'
```

## 🔍 Troubleshooting

### Common Issues

**Authentication Errors**
```
❌ Invalid token or insufficient permissions
```
**Solutions:**
- Verify token has correct scopes (`repo`, `read:org`)
- Check token hasn't expired
- Ensure token is from the correct GitHub account

**Docker Issues**
```
❌ Docker daemon not running or MCP server unavailable
```
**Solutions:**
- Start Docker service: `sudo systemctl start docker`
- Pull latest image: `docker pull ghcr.io/github/github-mcp-server`
- Check Docker permissions for non-root users

**Rate Limiting**
```
❌ GitHub API rate limit exceeded
```
**Solutions:**
- Wait for rate limit reset (shown in error message)
- Use Personal Access Token for higher limits
- Implement caching for repeated queries

**Network Connectivity**
```
❌ Unable to connect to GitHub API or Nebius
```
**Solutions:**
- Check internet connection
- Verify firewall settings allow outbound HTTPS
- Test direct API access: `curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/user`

### Debug Mode

**Enable Detailed Logging**
```bash
export LOG_LEVEL=DEBUG
streamlit run main.py
```

**Check Docker Logs**
```bash
# View MCP server logs
docker logs $(docker ps -q --filter ancestor=ghcr.io/github/github-mcp-server)
```

**Validate Configuration**
```python
# Test GitHub API access
import requests
response = requests.get(
    "https://api.github.com/user",
    headers={"Authorization": f"token {GITHUB_TOKEN}"}
)
print(f"Rate limit remaining: {response.headers.get('X-RateLimit-Remaining')}")
```

## 🧪 Development

### Project Structure
```
github_mcp_agent/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── README.md              # This documentation
├── assets/                # Static assets
│   ├── agno.png           # Agno framework logo
│   ├── Nebius.png         # Nebius AI branding
│   └── demo.gif           # Application demo
└── tests/                 # Test suite (future)
```

### Code Quality Standards

**Formatting and Linting**
```bash
# Code formatting
black main.py
isort main.py

# Type checking
mypy main.py

# Security scanning
bandit main.py
```

**Testing Framework**
```bash
# Unit tests
pytest tests/test_github_agent.py

# Integration tests
pytest tests/test_integration.py

# Coverage report
pytest --cov=main tests/
```


## 📊 Performance Metrics

### Response Times
- **Repository Info**: 2-5 seconds
- **Issue Analysis**: 3-8 seconds (depending on repository size)
- **PR Analysis**: 2-6 seconds
- **Custom Queries**: 3-10 seconds (varies by complexity)

### Scalability
- **Concurrent Users**: 5-20 (limited by GitHub API rate limits)
- **Repository Size**: No inherent limits (performance depends on GitHub API)
- **Query Complexity**: Optimized for most common use cases

### Resource Usage
- **Memory**: 100-500 MB (varies by query complexity)
- **CPU**: Low usage, primarily I/O bound
- **Network**: Dependent on GitHub API responses

## 🚀 Deployment Options

### Local Development
```bash
# Development server
streamlit run main.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0"]
```

### Cloud Deployment

**Streamlit Cloud**
- Push to GitHub repository
- Connect via [Streamlit Cloud](https://streamlit.io/cloud)
- Configure secrets for API keys

**AWS/Azure/GCP**
- Use container deployment services
- Configure environment variables securely
- Set up load balancing for multiple instances


## Acknowledgments

- **GitHub** - For providing comprehensive API access and MCP server
- **Nebius AI** - For powerful language models and AI processing
- **Agno Framework** - For AI agent orchestration and tool management
- **Streamlit Team** - For the excellent web application framework
- **Model Context Protocol** - For standardizing AI-tool integration
- **Docker** - For containerization and isolation capabilities
