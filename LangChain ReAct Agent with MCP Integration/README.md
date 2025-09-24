# 🦜🔗 LangChain ReAct Agent with MCP Integration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-FF6B6B?logo=langchain&logoColor=white)](https://langchain.com/langgraph)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io/)
[![Couchbase](https://img.shields.io/badge/Couchbase-EA2328?logo=couchbase&logoColor=white)](https://www.couchbase.com/)

A comprehensive tutorial demonstrating how to build a ReAct (Reasoning and Acting) agent using LangChain and LangGraph that can interact with a Couchbase database through the Model Context Protocol (MCP). This tutorial showcases the power of combining AI reasoning with real-time database operations.

## 📚 What You'll Learn

### 🧠 **ReAct Agent Architecture**
- **Reasoning and Acting** - Build agents that can think and act iteratively
- **LangGraph Integration** - Use LangGraph's prebuilt ReAct agent framework
- **Tool Orchestration** - Seamlessly combine multiple tools and data sources
- **Context Management** - Maintain conversation context across complex workflows

### 🔗 **Model Context Protocol (MCP) Mastery**
- **Universal AI-Data Integration** - Standardized communication between AI and systems
- **Secure Data Access** - Controlled exposure of data through MCP servers
- **Tool Use and Actionability** - Enable LLMs to trigger actions in external systems
- **Interoperability** - Connect different AI tools and data sources cohesively

### 🗄️ **Couchbase Database Operations**
- **NoSQL Document Operations** - Query, insert, update, and delete operations
- **N1QL Query Language** - Advanced SQL-like queries for JSON documents
- **Real-time Data Access** - Live database interactions through MCP
- **Enterprise-Grade Database** - Production-ready database integration patterns

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Jupyter       │    │   LangChain      │    │   Couchbase     │
│   Notebook      │◄──►│   ReAct Agent    │◄──►│   MCP Server    │
│   (Interface)   │    │   (Orchestrator) │    │   (Data Bridge) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Interactive   │    │   LangGraph      │    │   Couchbase     │
│   Tutorial      │    │   Tools & State  │    │   Capella       │
│   (Learning)    │    │   (Execution)    │    │   (Database)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## ✨ Key Features

### 🤖 **Advanced AI Agent Capabilities**
- **Multi-Step Reasoning** - Agents that can plan, execute, and reflect
- **Dynamic Tool Selection** - Intelligent choice of appropriate tools
- **Error Recovery** - Robust handling of failures and retries
- **Conversational Memory** - Persistent context across interactions

### 🛠️ **Comprehensive MCP Integration**
- **Client-Server Architecture** - Standardized communication protocols
- **Secure Authentication** - Safe database access with proper credentials
- **Real-time Operations** - Live database queries and modifications
- **Extensible Framework** - Easy integration with other systems

### 📊 **Production-Ready Database Integration**
- **Couchbase Capella** - Cloud-native NoSQL database platform
- **Travel Sample Data** - Real-world dataset for learning and testing
- **Advanced Querying** - Complex N1QL queries with joins and aggregations
- **Scalable Architecture** - Enterprise-grade database operations

## 🚀 Quick Start Guide

### Prerequisites

Before starting this tutorial, ensure you have:

- **Python 3.10 or higher**
- **Jupyter Notebook or JupyterLab**
- **Git** for cloning repositories
- **Nebius AI Studio Account** - Get API key from [Nebius Studio](https://studio.nebius.com/)
- **Couchbase Capella Account** - Free tier available at [Couchbase Cloud](https://cloud.couchbase.com/)

### Installation Steps

1. **Clone the Required Repositories**
   ```bash
   # Clone this tutorial
   git clone https://github.com/Arindam200/awesome-ai-apps.git
   cd mcp_ai_agents/langchain_langgraph_mcp_agent

   # Clone the Couchbase MCP Server
   git clone https://github.com/Couchbase-Ecosystem/mcp-server-couchbase.git
   ```

2. **Set Up Python Environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows

   # Install dependencies
   pip install langchain langgraph
   pip install python-dotenv jupyter
   pip install mcp
   ```

3. **Configure Environment Variables**
   
   Create a `.env` file in the project directory:
   ```env
   # Nebius AI Configuration
   NEBIUS_API_KEY=your_nebius_api_key_here
   NEBIUS_BASE_URL=https://api.studio.nebius.ai/v1/

   # Couchbase Configuration  
   COUCHBASE_CONNECTION_STRING=couchbases://your-cluster.cloud.couchbase.com
   COUCHBASE_USERNAME=your_username
   COUCHBASE_PASSWORD=your_password
   COUCHBASE_BUCKET_NAME=travel-sample

   # MCP Server Configuration
   MCP_SERVER_PATH=./mcp-server-couchbase
   ```

4. **Set Up Couchbase Capella**
   
   Follow these steps to configure your database:
   
   a. **Create Free Tier Account**
   - Visit [Couchbase Capella](https://cloud.couchbase.com/sign-in)
   - Sign up for a free tier account
   - Deploy a forever-free operational cluster

   b. **Configure Database Access**
   - Create [database credentials](https://docs.couchbase.com/cloud/clusters/manage-database-users.html) with Read/Write access
   - [Allow IP access](https://docs.couchbase.com/cloud/clusters/allow-ip-address.html) to your cluster
   - Ensure the `travel-sample` bucket is available

   c. **Test Connection**
   ```bash
   # Test your connection string
   curl -u username:password \
     "https://your-cluster.cloud.couchbase.com/pools/default"
   ```

5. **Launch the Tutorial**
   ```bash
   # Start Jupyter Notebook
   jupyter notebook LangChain_MCP_Adapter_Tutorial.ipynb

   # Or use JupyterLab
   jupyter lab LangChain_MCP_Adapter_Tutorial.ipynb
   ```

## 📖 Tutorial Structure

### Part 1: Foundation Concepts

**Understanding MCP (Model Context Protocol)**
- What is MCP and why it matters
- Client-server architecture patterns
- Security and authentication best practices
- Integration with AI systems

**ReAct Agent Framework**
- Reasoning and Acting paradigm
- LangGraph agent architecture
- Tool selection and execution
- State management and memory

### Part 2: Setup and Configuration

**Environment Preparation**
- Installing dependencies and tools
- Configuring API keys and credentials
- Setting up Couchbase Capella
- Testing connections and authentication

**MCP Server Deployment**
- Cloning and configuring mcp-server-couchbase
- Understanding server parameters
- Establishing client-server communication
- Debugging connection issues

### Part 3: Building the ReAct Agent

**Agent Construction**
- Creating LangChain ReAct agent
- Integrating MCP tools
- Configuring LLM models (Nebius AI)
- Setting up agent instructions

**Tool Integration**
- Connecting to Couchbase via MCP
- Defining available operations
- Handling tool responses
- Error management and recovery

### Part 4: Advanced Operations

**Database Querying**
- Simple document retrieval
- Complex N1QL queries
- Aggregations and joins
- Real-time data analysis

**Agent Interactions**
- Natural language to database operations
- Multi-step reasoning workflows
- Context-aware responses
- Performance optimization

### Part 5: Real-World Examples

**Travel Booking Scenarios**
- Finding flights and hotels
- Analyzing travel patterns
- Booking management operations
- Customer service automation

**Data Analysis Tasks**
- Exploratory data analysis
- Trend identification
- Report generation
- Business intelligence queries

## 🎯 Learning Objectives

By completing this tutorial, you will:

### Technical Skills
- ✅ Build production-ready ReAct agents with LangChain/LangGraph
- ✅ Implement MCP integration for database connectivity
- ✅ Design secure AI-database interaction patterns
- ✅ Create conversational interfaces for data operations

### Conceptual Understanding
- ✅ Master the ReAct (Reasoning and Acting) paradigm
- ✅ Understand MCP architecture and benefits
- ✅ Learn enterprise AI integration patterns
- ✅ Grasp database-AI interaction principles

### Practical Applications
- ✅ Develop customer service chatbots with database access
- ✅ Create data analysis assistants
- ✅ Build intelligent travel booking systems
- ✅ Design enterprise knowledge management tools

## 🔧 Configuration Details

### Nebius AI Models

The tutorial supports multiple AI models via Nebius:

| Model | Strengths | Best For |
|-------|-----------|----------|
| **DeepSeek-V3** | Advanced reasoning, code understanding | Complex multi-step operations |
| **Qwen-2.5** | Efficient processing, good accuracy | General database interactions |
| **LLaMA-3.3** | Comprehensive analysis, detailed responses | Data analysis and reporting |

### Couchbase Configuration

**Database Setup:**
```json
{
  "cluster": "couchbases://your-cluster.cloud.couchbase.com",
  "bucket": "travel-sample",
  "scope": "_default",
  "collection": "_default",
  "credentials": {
    "username": "your_username",
    "password": "your_password"
  }
}
```

**Required Permissions:**
- Data Reader (for query operations)
- Data Writer (for insert/update operations)
- Query Select (for N1QL queries)
- Query Manage Index (for index operations)

### MCP Server Configuration

**Server Parameters:**
```python
server_params = StdioServerParameters(
    command="node",
    args=["path/to/mcp-server-couchbase/build/index.js"],
    env={
        "COUCHBASE_CONNECTION_STRING": "couchbases://...",
        "COUCHBASE_USERNAME": "username",
        "COUCHBASE_PASSWORD": "password",
        "COUCHBASE_BUCKET_NAME": "travel-sample"
    }
)
```

## 🔍 Troubleshooting Guide

### Common Issues

**Connection Problems**
```
❌ Unable to connect to Couchbase cluster
```
**Solutions:**
- Verify connection string format
- Check IP allowlist in Capella dashboard
- Validate username/password credentials
- Ensure cluster is running and accessible

**MCP Server Issues**
```
❌ MCP server failed to start
```
**Solutions:**
- Check Node.js installation (`node --version`)
- Verify mcp-server-couchbase build (`npm run build`)
- Review environment variables
- Check server logs for detailed errors

**Authentication Errors**
```
❌ Invalid credentials or insufficient permissions
```
**Solutions:**
- Verify database user permissions
- Check bucket access rights
- Ensure credentials match Capella configuration
- Test connection with Couchbase SDK directly

**Agent Execution Problems**
```
❌ Agent failed to execute tools
```
**Solutions:**
- Review agent instructions and tool definitions
- Check LLM model permissions and quotas
- Validate MCP tool responses
- Enable debug logging for detailed information

### Debug Mode

**Enable Detailed Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# MCP client debugging
mcp_client.set_debug(True)

# LangChain debugging
import langchain
langchain.debug = True
```

**Test Individual Components:**
```python
# Test Couchbase connection
from couchbase.cluster import Cluster
cluster = Cluster(connection_string, ClusterOptions(...))

# Test MCP server
from mcp.client.stdio import stdio_client
async with stdio_client(server_params) as (read, write):
    # Test communication
    pass

# Test LLM access
from langchain_community.llms import Nebius
llm = Nebius(api_key="your_key")
response = llm.invoke("Hello, world!")
```

## 🧪 Advanced Topics

### Custom Tool Development

**Creating Custom MCP Tools:**
```python
class CustomDatabaseTool:
    def __init__(self, connection):
        self.connection = connection
    
    async def execute_custom_query(self, query: str) -> str:
        # Custom database operation
        result = await self.connection.query(query)
        return self.format_response(result)
```

**Agent Customization:**
```python
custom_agent = create_react_agent(
    llm=llm,
    tools=tools,
    state_modifier="You are a specialized travel assistant with access to booking data."
)
```

### Performance Optimization

**Query Optimization:**
- Use appropriate indexes for N1QL queries
- Implement query result caching
- Optimize data transfer with projection
- Use prepared statements for repeated queries

**Agent Optimization:**
- Implement tool result caching
- Use streaming for long-running operations
- Optimize context window usage
- Implement parallel tool execution

### Security Best Practices

**Credential Management:**
- Use environment variables for sensitive data
- Implement credential rotation
- Use least-privilege access principles
- Monitor and log access patterns

**Data Protection:**
- Validate all user inputs
- Implement rate limiting
- Use secure communication channels
- Audit database operations

## 📊 Example Use Cases

### Customer Service Chatbot
```python
# Natural language query
"Find all bookings for customer John Doe from the last month"

# Agent reasoning and execution
# 1. Parse customer name and date range
# 2. Construct N1QL query
# 3. Execute database search
# 4. Format results for user
```

### Data Analysis Assistant
```python
# Business intelligence query
"What are the most popular travel destinations this quarter?"

# Agent workflow
# 1. Analyze travel-sample data
# 2. Aggregate booking statistics
# 3. Identify trends and patterns
# 4. Generate comprehensive report
```

### Intelligent Travel Planner
```python
# Complex multi-step planning
"Plan a 3-day trip to Paris with hotel and flight recommendations"

# Agent process
# 1. Search available flights
# 2. Find hotels near attractions
# 3. Check availability and pricing
# 4. Create integrated itinerary
```

## 📄 License

This tutorial is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain Team** - For the excellent framework and agent tools
- **LangGraph Developers** - For the ReAct agent implementation
- **Couchbase** - For the robust NoSQL database platform and MCP server
- **Model Context Protocol** - For standardizing AI-system integration
- **Nebius AI** - For providing powerful language models
- **Open Source Community** - For continuous improvements and contributions

## 🤝 Community & Support

### Getting Help
- **Tutorial Issues**: [Report problems or ask questions](https://github.com/Arindam200/awesome-ai-apps/issues)
- **LangChain Documentation**: [Official LangChain Docs](https://python.langchain.com/)
- **Couchbase Support**: [Couchbase Developer Portal](https://developer.couchbase.com/)
- **MCP Resources**: [Model Context Protocol Documentation](https://modelcontextprotocol.io/)

### Contributing
- **Code Examples**: Submit additional use cases and examples
- **Documentation**: Help improve tutorial clarity and completeness
- **Bug Fixes**: Report and fix issues in the tutorial code
- **Feature Requests**: Suggest new topics and advanced scenarios

