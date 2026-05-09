# 🧠 Memory-Enabled AI Agent

An intelligent AI agent with persistent memory capabilities powered by Agno framework and Nebius AI. This agent remembers user interactions and leverages them in future conversations to provide personalized experiences.

## 🌟 Features

### 🧠 Persistent Memory System
💾 **Long-term Memory**: SQLite-based storage for persistent user information  
🔄 **Session Continuity**: Maintains context across application restarts  
👤 **User Profiling**: Automatic extraction and storage of user preferences  
📚 **Conversation History**: Remembers past interactions and references  

### ⚡ Real-time Interactions
🎯 **Streaming Responses**: Real-time response generation with progress indicators  
💬 **Natural Conversations**: Context-aware dialogue management  
🧮 **Smart Context**: Uses last 3 conversation turns for optimal context  
🎨 **Rich UI**: Beautiful console interface with colors and formatting  

### 🔧 Advanced Configuration
🛠️ **Flexible Setup**: Configurable user IDs and database locations  
🔐 **Secure API**: Environment variable-based API key management  
📊 **Memory Inspection**: View and analyze stored memories  
🧹 **Memory Management**: Clear memories when needed

## 📋 System Requirements

### Basic Requirements
- **Python**: 3.11 or higher
- **RAM**: 2GB minimum (4GB recommended)
- **Storage**: 1GB free space
- **Internet**: Stable connection for API access

### API Services
- **Nebius AI**: For intelligent conversation and memory processing
- **Agno Framework**: For agent orchestration and memory management



### Configure Environment Variables
Create a `.env` file:

```bash
# Nebius AI Configuration
NEBIUS_API_KEY=your_nebius_api_key_here
```

### 4. Run the Application
```bash
# Using UV (recommended)
uv run python main.py

# Manual execution
python main.py
```

## 💡 Usage Guide

### 🎬 Quick Start Demo

The application includes an interactive demo that showcases the memory capabilities:

1. **Launch Application**: Run the main script
2. **Follow Demo**: The agent will guide you through 3 demonstration steps
3. **Observe Memory**: Watch how the agent remembers and references your information
4. **View Memories**: See stored memories displayed after each interaction

### 📝 Demo Conversation Flow

**Step 1 - Introduction**
```
User: "Hello! I'm John, I work as a software developer."
Agent: [Responds and stores profession and name information]
```

**Step 2 - Additional Details**  
```
User: "I live in New York, I'm interested in Python and AI topics."
Agent: [Stores location and interests, builds user profile]
```

**Step 3 - Memory Recall**
```
User: "Do you know me? What do you remember about me?"
Agent: [Recalls and summarizes all stored information about the user]
```

## 🏗️ Technical Architecture

### 📁 Project Structure
```
agno_memory_agent/
├── main.py                # Main application with MemoryAgent class
├── pyproject.toml         # Project configuration and dependencies
├── README.md             # This documentation
├── .env                  # Environment variables (create this)
└── tmp/                  # Database storage (auto-created)
    └── agent.db          # SQLite database for memories and sessions
```

### 🔧 Core Components

**MemoryAgent Class**
```python
class MemoryAgent:
    """Memory-enabled AI Agent class"""
    
    # Core methods
    __init__()              # Initialize agent and memory systems
    _init_memory_system()   # Setup SQLite memory database
    _init_agent()           # Configure Agno agent with Nebius AI
    chat()                  # Handle user conversations
    clear_memory()          # Memory management utilities
    get_user_memories()     # Retrieve stored user information
    run_demo()              # Interactive demonstration
```

**Memory System Architecture**
- **SQLite Database**: Local storage for persistent memories
- **Memory Tables**: Separate tables for user memories and sessions
- **Agno Framework**: Handles memory operations and context management
- **Nebius AI**: Processes conversations and extracts memorable information

## ⚙️ Configuration Options

### 🔧 Customizable Settings

**Database Configuration**
```python
# Default values
DEFAULT_USER_ID = "user"
DEFAULT_DB_FILE = "tmp/agent.db"
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V3-0324"

# Custom initialization
agent = MemoryAgent(
    user_id="custom_user",
    db_file="custom/path/agent.db"
)
```

**Nebius AI Model Settings**
- **Model**: DeepSeek-V3 (configurable)
- **Context Window**: Optimized for conversation history
- **Memory Integration**: Automatic user information extraction
- **Streaming**: Real-time response generation

### 🧠 Memory System Features

**Automatic Memory Management**
- **User Information**: Names, professions, locations automatically stored
- **Preferences**: Interests and topics of conversation remembered
- **Context Awareness**: References previous conversations appropriately
- **Session Persistence**: Memories survive application restarts

**Memory Operations**
```python
# View current memories
memories = agent.get_user_memories()

# Clear all memories
agent.clear_memory()

# Chat with memory integration
agent.chat("Remember our previous conversation?")
```

## 🎯 Use Cases and Applications

### 👤 Personal Assistant
- **Daily Interactions**: Remember preferences and past conversations
- **Project Tracking**: Keep track of ongoing work and deadlines
- **Learning Assistant**: Adapt to your learning style and progress
- **Productivity Support**: Remember your goals and provide relevant suggestions

### 💼 Professional Applications
- **Customer Service**: Maintain customer interaction history
- **Team Collaboration**: Remember team member preferences and work styles
- **Meeting Assistant**: Recall previous meeting discussions and action items
- **Knowledge Management**: Build institutional knowledge from conversations

### 🎓 Educational Scenarios
- **Tutoring Systems**: Adapt to student learning pace and style
- **Research Assistance**: Remember research topics and progress
- **Study Companion**: Track learning goals and achievements
- **Skill Development**: Monitor progress and provide personalized guidance

## 🐛 Troubleshooting

### ⚠️ Common Issues and Solutions

**API Key Errors**
```bash
# Error: NEBIUS_API_KEY environment variable not found
# Solution: Check .env file configuration
echo $NEBIUS_API_KEY  # Linux/Mac
echo %NEBIUS_API_KEY%  # Windows
```

**Database Connection Issues**
```bash
# Error: Cannot create database file
# Solution: Ensure tmp directory permissions
mkdir tmp
chmod 755 tmp  # Linux/Mac
```

**Memory Not Persisting**
```bash
# Error: Agent doesn't remember previous conversations
# Solution: Check database file location and permissions
ls -la tmp/agent.db  # Verify file exists and is writable
```

**Import Errors**
```bash
# Error: Module 'agno' not found
# Solution: Ensure dependencies are installed
uv sync  # or pip install agno
```

### 🔧 Debug Mode

**Enable Detailed Logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
agent = MemoryAgent()
agent.run_demo()
```

## 📈 Performance and Optimization

### ⚡ Performance Characteristics
- **Memory Queries**: < 100ms for user memory retrieval
- **Conversation Response**: 2-5 seconds depending on complexity
- **Database Operations**: Optimized SQLite queries
- **Streaming Output**: Real-time response generation

### 🔧 Optimization Tips
- **Database Location**: Use SSD storage for better performance
- **Memory Cleanup**: Periodically clear old memories if needed
- **Model Selection**: Choose appropriate Nebius AI model for your use case
- **Context Management**: Configure history length based on needs

## 🔮 Future Enhancements

### 📅 Planned Features
- [ ] **Multi-user Support**: Handle multiple users in the same database
- [ ] **Memory Export**: Export/import memory databases
- [ ] **Web Interface**: Browser-based interaction
- [ ] **Voice Integration**: Speech-to-text and text-to-speech
- [ ] **Memory Analytics**: Insights into conversation patterns

### 🧠 Advanced Memory Features
- [ ] **Semantic Search**: Find memories by meaning, not just keywords
- [ ] **Memory Prioritization**: Automatically prioritize important memories
- [ ] **Temporal Memory**: Remember when things happened
- [ ] **Contextual Associations**: Link related memories automatically


### 📏 Code Standards

**🐍 Python Code Style**
- **PEP 8**: Follow Python official style guide
- **Type Hints**: Use type hints in function signatures
- **Docstrings**: Provide descriptive documentation for each function
- **Error Handling**: Comprehensive error management and meaningful messages


### 📊 API Provider Support

**🧠 Nebius AI**: [nebius.ai](https://nebius.ai) - AI model platform  
**🤖 Agno Framework**: [agno.ai](https://agno.ai) - Agent orchestration framework

---

**🚀 Memory-Enabled AI Agent** - Start building intelligent conversations with persistent memory! 

Feel free to use GitHub Issues for questions or suggestions. Together we're building stronger AI agents! 🌟Agent

An intelligent AI agent with persistent memory capabilities powered by Agno framework and Nebius AI. This agent remembers user interactions and leverages them in future conversations to provide personalized experiences.

## 🌟 Features

### 🧠 Persistent Memory System
💾 **Long-term Memory**: SQLite-based storage for persistent user information  
🔄 **Session Continuity**: Maintains context across application restarts  
👤 **User Profiling**: Automatic extraction and storage of user preferences  
📚 **Conversation History**: Remembers past interactions and references  

### ⚡ Real-time Interactions
🎯 **Streaming Responses**: Real-time response generation with progress indicators  
💬 **Natural Conversations**: Context-aware dialogue management  
🧮 **Smart Context**: Uses last 3 conversation turns for optimal context  
🎨 **Rich UI**: Beautiful console interface with colors and formatting  

### � Advanced Configuration
🛠️ **Flexible Setup**: Configurable user IDs and database locations  
� **Secure API**: Environment variable-based API key management  
📊 **Memory Inspection**: View and analyze stored memories  
🧹 **Memory Management**: Clear memories when needed
