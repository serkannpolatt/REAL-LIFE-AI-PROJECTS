"""
Agent Manager - Handles dynamic loading and configuration of agents
"""
import sys
import os

# Add project root to sys.path for imports
project_root = r"C:\Users\Serkan POLAT\Desktop\agentic-ai-system-main"
if project_root not in sys.path:
    sys.path.append(project_root)



import os
import json
import importlib.util
from typing import List, Dict, Any
from langchain.agents import Tool


class AgentManager:
    """Manages dynamic loading and configuration of domain-specific agents"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "agents_config.json"
        )
        self.tools = []
        self.agent_config = []

    def load_config(self) -> List[Dict[str, Any]]:
        """Load agent configuration from JSON file"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.agent_config = json.load(f)
            return self.agent_config
        except FileNotFoundError:
            raise FileNotFoundError(f"Agent config file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

    def load_agents(self) -> List[Tool]:
        """Dynamically load all agents as LangChain tools"""
        if not self.agent_config:
            self.load_config()

        self.tools = []

        for agent in self.agent_config:
            try:
                tool = self._create_tool_from_config(agent)
                self.tools.append(tool)
                print(f"✅ Loaded agent: {agent['name']}")
            except Exception as e:
                print(f"❌ Failed to load agent {agent['name']}: {e}")

        return self.tools

    def _create_tool_from_config(self, agent_config: Dict[str, Any]) -> Tool:
        """Create a LangChain Tool from agent configuration"""
        module_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "agents",
            f"{agent_config['file']}.py",
        )

        # Load module dynamically
        spec = importlib.util.spec_from_file_location(agent_config["file"], module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the function from module
        func = getattr(module, agent_config["function"])

        return Tool(
            name=agent_config["name"],
            func=func,
            description=agent_config["description"],
        )

    def get_tool_names(self) -> List[str]:
        """Get list of all loaded tool names"""
        return [tool.name for tool in self.tools]

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get mapping of tool names to descriptions"""
        return {tool.name: tool.description for tool in self.tools}
