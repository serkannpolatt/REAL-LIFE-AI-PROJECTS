# 📚 Doc-MCP: Advanced Documentation RAG System

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MongoDB Atlas](https://img.shields.io/badge/Database-MongoDB%20Atlas-green.svg)](https://www.mongodb.com/atlas)
[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-purple.svg)](https://modelcontextprotocol.io/)

Transform GitHub documentation repositories into intelligent, queryable knowledge bases using advanced RAG (Retrieval-Augmented Generation) technology and Model Context Protocol (MCP) integration.

## ✨ Key Features

### 🔍 **Intelligent Search & Analysis**
- **Semantic Vector Search** - Natural language queries with MongoDB Atlas Vector Search
- **AI-Powered Q&A** - Context-aware responses with precise source citations
- **Hybrid Search** - Combines vector similarity and full-text search capabilities
- **Multi-format Support** - Processes Markdown, reStructuredText, and plain text files

### 🤖 **MCP Integration**
- **Native MCP Server** - Seamless integration with AI agents and assistants
- **RESTful API** - HTTP endpoints for programmatic access
- **Real-time Streaming** - Server-Sent Events (SSE) for live responses
- **Tool Integration** - Works with Claude Desktop, Codeium, and other MCP clients

### 📊 **Repository Management**
- **Batch Processing** - Ingest entire repositories with progress tracking
- **Incremental Updates** - Smart sync that processes only changed files
- **Repository Analytics** - Detailed statistics and file metrics
- **CRUD Operations** - Complete repository lifecycle management

### 🚀 **Production Features**
- **Scalable Architecture** - Modular design with separation of concerns
- **Error Handling** - Comprehensive error recovery and logging
- **Configuration Management** - Environment-based settings with validation
- **Performance Optimization** - Efficient chunking and concurrent processing

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio UI     │    │   MCP Server     │    │  MongoDB Atlas  │
│   (Web Interface)│◄──►│  (API Layer)     │◄──►│ (Vector Store)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  GitHub Loader  │    │   RAG Engine     │    │  Nebius AI      │
│  (File Ingestion)│    │  (Query Processing)│   │ (LLM & Embeddings)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.13+** with pip or uv package manager
- **MongoDB Atlas** cluster with Vector Search enabled
- **Nebius AI API** key for embeddings and language models
- **GitHub Personal Access Token** (optional, for private repos and higher rate limits)

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/doc-mcp.git
cd doc-mcp
```

2. **Set Up Python Environment**
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Using pip
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

3. **Install Dependencies**
```bash
# Using uv
uv sync

# Using pip
pip install -r requirements.txt
```

4. **Configure Environment**
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
# Required API Keys
NEBIUS_API_KEY=your_nebius_api_key_here
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/

# Optional Configuration
GITHUB_API_KEY=your_github_token_here
NEBIUS_LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-fast
NEBIUS_EMBEDDING_MODEL=BAAI/bge-en-icl

# Performance Tuning
CHUNK_SIZE=3072
SIMILARITY_TOP_K=5
GITHUB_CONCURRENT_REQUESTS=10
```

5. **Launch Application**
```bash
python app.py
```

The application will be available at:
- **Web Interface**: `http://localhost:7860`
- **MCP Server**: `http://127.0.0.1:7860/gradio_api/mcp/sse`

## 📖 Usage Guide

### 1. Documentation Ingestion

**Step 1: Repository Selection**
- Navigate to "📥 Documentation Ingestion" tab
- Enter GitHub repository URL (format: `owner/repository`)
- The system will automatically discover documentation files

**Step 2: File Processing**
- Review and select Markdown files to process
- Configure processing parameters (chunk size, filters)
- Execute the two-phase pipeline:
  1. **Load Files** - Download and parse documentation
  2. **Generate Embeddings** - Create vector representations

**Step 3: Verification**
- Monitor progress with real-time status updates
- Review ingestion statistics and any processing errors
- Verify successful embedding generation

### 2. AI-Powered Querying

**Natural Language Search**
- Go to "🤖 AI Documentation Assistant" tab
- Select your ingested repository from the dropdown
- Ask questions in natural language
- Receive AI-generated responses with source citations

**Advanced Query Examples**
```
- "How do I configure authentication in this project?"
- "What are the deployment requirements and steps?"
- "Explain the API endpoints and their parameters"
- "Show me examples of error handling patterns"
```

### 3. Repository Management

**Analytics Dashboard**
- Use "🛠️ Repository Management" tab
- View comprehensive statistics:
  - Total files and chunks processed
  - Repository metadata and last update
  - Processing performance metrics

**Maintenance Operations**
- **Update Repository** - Sync changes from GitHub
- **Delete Repository** - Remove all associated data
- **Reprocess Files** - Regenerate embeddings with new settings

### 4. MCP Integration

**Connect with AI Agents**
```python
# Example: Claude Desktop configuration
{
  "servers": {
    "doc-mcp": {
      "command": "curl",
      "args": ["http://127.0.0.1:7860/gradio_api/mcp/sse"]
    }
  }
}
```

**API Usage**
```bash
# Query repository via HTTP API
curl -X POST "http://localhost:7860/gradio_api/query" \
  -H "Content-Type: application/json" \
  -d '{"repository": "owner/repo", "query": "How to deploy?"}'
```

## ⚙️ Configuration Reference

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEBIUS_API_KEY` | Nebius AI API key for LLM and embeddings | - | ✅ |
| `MONGODB_URI` | MongoDB Atlas connection string | - | ✅ |
| `GITHUB_API_KEY` | GitHub Personal Access Token | - | ❌ |
| `NEBIUS_LLM_MODEL` | Language model for responses | `meta-llama/Llama-3.3-70B-Instruct-fast` | ❌ |
| `NEBIUS_EMBEDDING_MODEL` | Embedding model for vectors | `BAAI/bge-en-icl` | ❌ |
| `CHUNK_SIZE` | Text chunk size for processing | `3072` | ❌ |
| `SIMILARITY_TOP_K` | Number of similar chunks to retrieve | `5` | ❌ |
| `GITHUB_CONCURRENT_REQUESTS` | Parallel GitHub API requests | `10` | ❌ |

### Performance Optimization

**For Large Repositories**
```env
CHUNK_SIZE=2048              # Smaller chunks for better granularity
GITHUB_CONCURRENT_REQUESTS=5 # Reduce API load
SIMILARITY_TOP_K=10         # More context for complex queries
```

**For Better Accuracy**
```env
CHUNK_SIZE=4096             # Larger chunks for more context
SIMILARITY_TOP_K=3          # Focused retrieval
NEBIUS_LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct-fast
```

## 🗄️ Database Setup

### MongoDB Atlas Configuration

1. **Create Cluster**
   - Sign up at [MongoDB Atlas](https://www.mongodb.com/atlas)
   - Create a new cluster with **Vector Search** enabled
   - Note your connection string

2. **Database Structure**
   - `doc_rag` - Document chunks with embeddings
   - `ingested_repos` - Repository metadata and statistics

3. **Vector Search Index**
   The application automatically creates the required vector search index:
   ```json
   {
     "fields": [
       {
         "numDimensions": 4096,
         "path": "embedding",
         "similarity": "cosine",
         "type": "vector"
       }
     ]
   }
   ```

## � Troubleshooting

### Common Issues

**Authentication Errors**
```bash
# Verify MongoDB connection
python -c "from pymongo import MongoClient; print(MongoClient('your_uri').admin.command('ping'))"

# Test Nebius API
curl -H "Authorization: Bearer your_api_key" https://api.studio.nebius.ai/v1/models
```

**Rate Limiting**
- **GitHub API**: Add personal access token for 5,000 requests/hour (vs 60 anonymous)
- **Nebius API**: Monitor usage in Nebius Studio dashboard
- **MongoDB**: Check Atlas cluster metrics for connection limits

**Memory Issues**
- Reduce `CHUNK_SIZE` to `2048` or lower
- Decrease `GITHUB_CONCURRENT_REQUESTS` to `5`
- Process repositories in smaller batches

**Vector Search Issues**
- Ensure Atlas cluster has Vector Search enabled
- Verify index creation in Atlas dashboard
- Check embedding dimensions match model output

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python app.py
```

Check application logs:
```bash
tail -f doc_mcp.log
```

## 🧪 Development

### Project Structure
```
doc_mcp/
├── app.py                    # Application entry point
├── src/
│   ├── core/                # Core configuration and types
│   │   ├── config.py        # Settings management
│   │   ├── exceptions.py    # Custom exceptions
│   │   └── types.py         # Type definitions
│   ├── database/            # Database layer
│   │   ├── mongodb.py       # MongoDB connection
│   │   ├── repository.py    # Data access layer
│   │   └── vector_store.py  # Vector operations
│   ├── github/              # GitHub integration
│   │   ├── client.py        # GitHub API client
│   │   ├── file_loader.py   # File downloading
│   │   └── parser.py        # Content parsing
│   ├── rag/                 # RAG pipeline
│   │   ├── ingestion.py     # Document processing
│   │   ├── models.py        # AI model integration
│   │   └── query.py         # Query processing
│   └── ui/                  # User interface
│       ├── main.py          # Gradio application
│       └── tabs/            # UI components
├── tests/                   # Test suite
├── docs/                    # Documentation
├── requirements.txt         # Dependencies
└── pyproject.toml          # Project configuration
```

### Running Tests
```bash
# Install test dependencies
uv sync --dev

# Run test suite
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork and Clone**
   ```bash
   git fork https://github.com/original/doc-mcp.git
   git clone https://github.com/yourusername/doc-mcp.git
   cd doc-mcp
   ```

2. **Development Setup**
   ```bash
   uv sync --dev
   pre-commit install
   ```

3. **Make Changes**
   - Follow existing code style and patterns
   - Add tests for new functionality
   - Update documentation as needed

4. **Submit Pull Request**
   - Ensure all tests pass
   - Include clear description of changes
   - Reference related issues

### Development Guidelines

- **Code Style**: Black formatting, isort imports, type hints
- **Testing**: pytest with >90% coverage requirement
- **Documentation**: Docstrings for all public functions
- **Commits**: Conventional commit messages


