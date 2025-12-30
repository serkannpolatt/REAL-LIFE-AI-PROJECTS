#!/usr/bin/env python3
"""
arXiv Research Agent - With OpenAI Agents & Memori

This module contains AI agents specifically designed for conducting
academic research and storing it in memory.

Features:
- arXiv paper search and analysis
- Persistent memory system
- Academic reporting
- Tavily integration

Requirements:
- pip install agents memori tavily openai python-dotenv
- Nebius API key
- Tavily API key

Usage:
    from researcher import Researcher
    researcher = Researcher()
    researcher.define_agents()
"""

import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Optional

from agents import Agent, Runner, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel
from pydantic import BaseModel
from dotenv import load_dotenv
from tavily import TavilyClient

from memori import Memori, create_memory_tool

# Load environment variables
load_dotenv()

# Check API keys
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not NEBIUS_API_KEY:
    raise ValueError("NEBIUS_API_KEY not defined in environment variables")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not defined in environment variables")

print("🤖 Initializing Nebius AI client...")

# Model configuration
MODEL_NAME = os.getenv("EXAMPLE_MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
BASE_URL = os.getenv("EXAMPLE_BASE_URL", "https://api.studio.nebius.ai/v1/")

model = OpenAIChatCompletionsModel(
    model=MODEL_NAME,
    openai_client=AsyncOpenAI(base_url=BASE_URL, api_key=NEBIUS_API_KEY),
)

# Create tmp directory
cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
tmp.mkdir(exist_ok=True, parents=True)

today = datetime.now().strftime("%Y-%m-%d")


class ArxivSearchResult(BaseModel):
    """arXiv search result model"""

    query: str
    results: str
    found_papers: bool
    paper_count: int = 0


class MemorySearchResult(BaseModel):
    """Memory search result model"""

    query: str
    results: str
    found_memories: bool


# Global researcher instance
_researcher_instance: Optional["Researcher"] = None


@function_tool
def search_memory(query: str) -> MemorySearchResult:
    """
    Search the agent's memory for past conversations and research information.

    Args:
        query: Term to search in memory (e.g., "AI research", "quantum computing findings")

    Returns:
        MemorySearchResult: Search results from the agent's memory
    """
    global _researcher_instance
    if _researcher_instance is None:
        return MemorySearchResult(
            query=query, results="Memory system not initialized", found_memories=False
        )

    try:
        if not query.strip():
            return MemorySearchResult(
                query=query, results="Please enter a search query", found_memories=False
            )

        result = _researcher_instance.memory_tool.execute(query=query.strip())
        found_memories = bool(
            result
            and "No relevant memory found" not in result
            and "No relevant memories found" not in result
            and "Error" not in result
        )

        return MemorySearchResult(
            query=query,
            results=result if result else "No relevant memories found",
            found_memories=found_memories,
        )

    except Exception as e:
        return MemorySearchResult(
            query=query, results=f"Memory search error: {str(e)}", found_memories=False
        )


@function_tool
def search_arxiv(query: str) -> ArxivSearchResult:
    """
    Search for arXiv research papers related to the given topic.

    Args:
        query: Research topic to search (e.g., "quantum computing", "machine learning", "neuroscience")

    Returns:
        ArxivSearchResult: Search results from arXiv with paper details
    """
    global _researcher_instance
    if _researcher_instance is None:
        return ArxivSearchResult(
            query=query, results="Research system not initialized", found_papers=False
        )

    try:
        if not query.strip():
            return ArxivSearchResult(
                query=query,
                results="Please enter a research topic to search",
                found_papers=False,
            )

        # Search arXiv papers with Tavily
        search_query = (
            f"arXiv research papers {query} latest developments academic research"
        )

        # Perform advanced search with Tavily
        search_result = _researcher_instance.tavily_client.search(
            query=search_query,
            search_depth="advanced",
            include_domains=["arxiv.org", "scholar.google.com", "researchgate.net"],
            max_results=10,
        )

        if not search_result.get("results"):
            return ArxivSearchResult(
                query=query,
                results=f"No arXiv papers found for {query}",
                found_papers=False,
            )

        # Process and format results
        papers = []
        for result in search_result["results"][:5]:  # Limit to first 5 results
            title = result.get("title", "Title not available")
            url = result.get("url", "")
            content = result.get("content", "")

            # Extract basic information
            paper_info = {
                "title": title,
                "url": url,
                "summary": content[:300] + "..." if len(content) > 300 else content,
            }
            papers.append(paper_info)

        # Format results as structured output
        result_text = f"## arXiv Research Papers for {query}\n\n"
        result_text += f"{len(papers)} relevant research papers found:\n\n"

        for i, paper in enumerate(papers, 1):
            result_text += f"### {i}. {paper['title']}\n"
            result_text += f"**URL:** {paper['url']}\n"
            result_text += f"**Summary:** {paper['summary']}\n\n"

        result_text += "---\n"
        result_text += (
            f"*Search performed with Tavily for academic research papers on {query}*"
        )

        return ArxivSearchResult(
            query=query, results=result_text, found_papers=True, paper_count=len(papers)
        )

    except Exception as e:
        return ArxivSearchResult(
            query=query, results=f"arXiv search error: {str(e)}", found_papers=False
        )


class Researcher:
    """
    Researcher class that manages Memori initialization and agent creation
    """

    def __init__(self):
        """Initialize the researcher"""
        global _researcher_instance
        _researcher_instance = self

        print("🧠 Initializing Memori memory system...")

        # Initialize Memori memory system
        self.memori = Memori(
            database_connect="sqlite:///research_memori.db",
            conscious_ingest=True,  # Working memory
            auto_ingest=True,  # Dynamic search
            verbose=True,
        )
        self.memori.enable()
        self.memory_tool = create_memory_tool(self.memori)

        # Initialize Tavily client
        print("📚 Initializing Tavily research client...")
        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

        self.research_agent = None
        self.memory_agent = None

        print("✅ Researcher successfully initialized!")

    async def run_agent_with_memory(self, agent, user_input: str):
        """Run agent and save conversation to memory"""
        try:
            print(f"🔍 Running agent: {user_input[:50]}...")

            # Run agent with user input
            result = await Runner.run(agent, input=user_input)

            # Get response content
            response_content = (
                result.final_output if hasattr(result, "final_output") else str(result)
            )

            # Save conversation to memory
            print("💾 Saving conversation to memory...")
            self.memori.record_conversation(
                user_input=user_input, ai_output=response_content
            )

            print("✅ Agent run and saved to memory!")
            return result

        except Exception as e:
            print(f"❌ Agent execution error: {str(e)}")
            raise

    def define_agents(self):
        """Define and create research and memory agents"""
        print("🤖 Defining agents...")

        # Create research agent
        self.research_agent = self._create_research_agent()

        # Create memory agent
        self.memory_agent = self._create_memory_agent()

        print("✅ Agents successfully defined!")

        return self.research_agent, self.memory_agent

    def get_research_agent(self):
        """Get research agent, create if necessary"""
        if self.research_agent is None:
            self.define_agents()
        return self.research_agent

    def get_memory_agent(self):
        """Get memory agent, create if necessary"""
        if self.memory_agent is None:
            self.define_agents()
        return self.memory_agent

    def _create_research_agent(self):
        """Create research agent with Memori memory capabilities and arXiv search"""
        agent = Agent(
            name="arXiv Research Agent",
            model=model,
            instructions=dedent(
                """\
                You are Professor X-1000, an elite AI research scientist with MEMORY CAPABILITIES!

                🧠 Your advanced capabilities:
                - Advanced research with Tavily-powered arXiv paper search
                - Persistent memory of all research sessions
                - Reference and build upon previous research
                - Generate comprehensive, fact-based research reports

                Your writing style:
                - Clear and authoritative
                - Engaging but professional  
                - Fact-focused with proper citations
                - Accessible to educated experts
                - Building on previous research when relevant

                RESEARCH WORKFLOW:
                1. FIRST: Use search_memory to find any previous research on this topic
                2. Use search_arxiv to find research papers on the topic
                3. Analyze and cross-reference sources for accuracy and relevance
                4. If you find relevant previous research, note how this builds on it
                5. Follow academic standards but maintain readability
                6. Include only verifiable facts with proper citations
                7. Create an engaging narrative that guides readers through complex topics
                8. End with actionable insights and future implications

                Always note when you're building on previous research sessions!
                Focus on academic research papers and scientific sources.
                
                When presenting research findings, clearly structure them as:
                - Key research questions addressed
                - Methodology and approach
                - Main findings and results
                - Implications for the field
                - Future research directions
            """
            ),
            tools=[search_memory, search_arxiv],
        )
        return agent

    def _create_memory_agent(self):
        """Create an agent specialized in retrieving research memories"""
        agent = Agent(
            name="Research Memory Assistant",
            instructions=dedent(
                """\
                You are the Research Memory Assistant, specialized in helping users recall their research history!

                🧠 Your capabilities:
                - Search through all past research sessions
                - Summarize previous research topics and findings
                - Help users find specific research they've done before
                - Connect related research across different sessions

                Your style:
                - Friendly and helpful
                - Organized and clear in presenting research history
                - Good at summarizing complex research into digestible insights

                When users ask about their research history:
                1. Use search_memory to find relevant past research
                2. Organize results chronologically or by topic
                3. Provide clear summaries of each research session
                4. Highlight key findings and connections between research
                5. If they ask for specific research, provide detailed information

                Always search memory first, then organize and provide helpful summaries!
            """
            ),
            tools=[search_memory],
        )
        return agent
