#!/usr/bin/env python3
"""
Memory-Enabled AI Agent Demo - Using Agno & Nebius AI

This application demonstrates an intelligent AI agent that remembers user
interactions and uses them in future conversations.

Features:
- Persistent memory system
- User profile tracking
- Conversation history
- Real-time streaming responses

Requirements:
- Python 3.11+
- Nebius API key
- agno library

Usage:
    python main.py
"""

import os
import sys
from pathlib import Path

from agno.agent import Agent
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
from agno.models.nebius import Nebius
from agno.storage.sqlite import SqliteStorage
from rich.pretty import pprint
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Use Rich for console output
console = Console()

# Configuration constants
DEFAULT_USER_ID = "user"
DEFAULT_DB_FILE = "tmp/agent.db"
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V3-0324"
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")


class MemoryAgent:
    """Memory-enabled AI Agent class"""

    def __init__(self, user_id: str = DEFAULT_USER_ID, db_file: str = DEFAULT_DB_FILE):
        """
        Initialize the agent

        Args:
            user_id: User identifier
            db_file: Database file path
        """
        self.user_id = user_id
        self.db_file = db_file

        # Check API key
        if not NEBIUS_API_KEY:
            raise ValueError(
                "NEBIUS_API_KEY environment variable not found. "
                "Please define it in .env file."
            )

        # Create tmp directory
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)

        console.print(
            "[bold blue]🤖 Initializing Memory-Enabled AI Agent...[/bold blue]"
        )

        # Initialize memory system
        self._init_memory_system()

        # Initialize agent
        self._init_agent()

        console.print("[bold green]✅ Agent successfully initialized![/bold green]")

    def _init_memory_system(self):
        """Initialize memory system"""
        self.memory = Memory(
            model=Nebius(id=DEFAULT_MODEL_ID, api_key=NEBIUS_API_KEY),
            db=SqliteMemoryDb(table_name="user_memories", db_file=self.db_file),
        )

        # Storage system
        self.storage = SqliteStorage(table_name="agent_sessions", db_file=self.db_file)

    def _init_agent(self):
        """Initialize agent"""
        self.agent = Agent(
            model=Nebius(id=DEFAULT_MODEL_ID, api_key=NEBIUS_API_KEY),
            memory=self.memory,
            enable_agentic_memory=True,
            enable_user_memories=True,
            storage=self.storage,
            add_history_to_messages=True,
            num_history_runs=3,
            markdown=True,
        )

    def clear_memory(self):
        """Clear memory"""
        self.memory.clear()
        console.print("[yellow]🧹 Memory cleared[/yellow]")

    def get_user_memories(self) -> dict:
        """Get user memories"""
        return self.memory.get_user_memories(user_id=self.user_id)

    def chat(self, message: str, show_memories: bool = True):
        """
        Chat with the agent

        Args:
            message: User message
            show_memories: Whether to show/hide memories
        """
        console.print(f"\n[bold cyan]👤 User:[/bold cyan] {message}")
        console.print("[dim]Agent is thinking...[/dim]")

        # Agent response
        self.agent.print_response(
            message,
            user_id=self.user_id,
            stream=True,
            stream_intermediate_steps=True,
        )

        if show_memories:
            self._display_memories()

    def _display_memories(self):
        """Display memories"""
        memories = self.get_user_memories()

        if memories:
            console.print("\n[bold magenta]🧠 Memories:[/bold magenta]")
            pprint(memories)
        else:
            console.print("\n[dim]No memories found yet[/dim]")

    def run_demo(self):
        """Run demo conversations"""
        console.print(
            Panel.fit(
                "[bold]Memory-Enabled AI Agent Demo[/bold]\n\n"
                "This demo shows how the agent remembers you.",
                title="🤖 Demo Start",
            )
        )

        # Demo conversations
        demo_messages = [
            "Hello! I'm Serkan, I work as a software developer.",
            "I live in London, I'm interested in Python and AI topics.",
            "Do you know me? What do you remember about me?",
        ]

        for i, message in enumerate(demo_messages, 1):
            console.print(f"\n[bold yellow]📝 Demo Step {i}:[/bold yellow]")
            self.chat(message)

            if i < len(demo_messages):
                input("\n[dim]Press Enter to continue...[/dim]")

        console.print(
            Panel.fit(
                "[bold green]Demo completed![/bold green]\n\n"
                "The agent now knows you and will use this information\n"
                "in future conversations.",
                title="✅ Demo End",
            )
        )


def main():
    """Main function"""
    try:
        # Initialize agent
        agent = MemoryAgent()

        # Run demo
        agent.run_demo()

    except KeyboardInterrupt:
        console.print("\n[yellow]Demo stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
