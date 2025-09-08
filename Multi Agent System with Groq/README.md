# Multi-Agent AI System (Groq)

## Overview
This project demonstrates a simple and hierarchical multi-agent architecture using Python and LangChain, powered by Groq LLM. The system coordinates multiple specialized agents (Researcher, Analyst, Writer) under the supervision of a Supervisor agent to solve complex tasks collaboratively.

## Features
- **Modular Agents:** Researcher, Analyst, and Writer agents, each with a distinct role.
- **Supervisor Agent:** Orchestrates workflow, assigns tasks, and manages agent collaboration.
- **Tool Integration:** Includes web search and summary tools for enhanced capabilities.
- **Hierarchical Team Structure:** Example of organizing agents in teams with leaders (e.g., CEO, Team Leaders).
- **LangChain & Groq:** Utilizes LangChain's agent and workflow features with Groq LLM for reasoning and generation.

## How It Works
1. **Supervisor Agent** receives a task and decides which agent should act next.
2. **Researcher Agent** gathers information and data relevant to the task.
3. **Analyst Agent** analyzes the research data and provides insights.
4. **Writer Agent** compiles a final report based on research and analysis.
5. The process continues until the Supervisor determines the task is complete.

## Example Use Cases
- Business research and reporting
- Technical analysis and documentation
- Automated multi-step reasoning tasks

## Setup
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your `GROQ_API_KEY` in a `.env` file:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
4. Run the notebook `main.ipynb` in Jupyter or VS Code.

## Main Files
- `main.ipynb`: Contains all code for agent definitions, workflow setup, and example executions.

## Dependencies
- Python 3.8+
- langchain
- langchain_groq
- langgraph
- python-dotenv


## Author
Serkan Polat
