"""
System Orchestrator - Main system coordination and workflow management
"""
import sys
import os

# Add project root to sys.path for imports
project_root = r"C:\Users\Serkan POLAT\Desktop\agentic-ai-system-main"
if project_root not in sys.path:
    sys.path.append(project_root)



from typing import Dict, Any
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent_manager import AgentManager
from core.intent_classifier import IntentClassifier
from utils.llm_instance import llm
from utils.db_utils import init_db
from utils.ticket_parser import extract_ticket_info_and_intent


class SystemOrchestrator:
    """Main system orchestrator that coordinates all components"""

    def __init__(self):
        self.agent_manager = AgentManager()
        self.intent_classifier = IntentClassifier()
        self.agent = None
        self._initialize_system()

    def _initialize_system(self):
        """Initialize the system components"""
        # Initialize database
        init_db()
        print("✅ Database initialized")

        # Load agents
        tools = self.agent_manager.load_agents()
        print(f"✅ Loaded {len(tools)} agents")

        # Initialize LangChain agent
        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )
        print("✅ LangChain agent initialized")

    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input and return appropriate response

        Args:
            user_input: The user's message

        Returns:
            Dictionary containing response and metadata
        """
        try:
            # Classify intent
            intent = self.intent_classifier.classify(user_input)

            response_data = {
                "intent": intent,
                "input": user_input,
                "response": "",
                "error": None,
            }

            if intent == "ticket":
                response_data["response"] = self._handle_ticket_creation(user_input)
            elif intent in ["status", "close"]:
                response_data["response"] = self._handle_ticket_management(user_input)
            elif intent == "info":
                response_data["response"] = self._handle_info_request(user_input)
            else:
                response_data["response"] = (
                    "❓ I couldn't understand your message. Please rephrase your question."
                )

            return response_data

        except Exception as e:
            return {
                "intent": "error",
                "input": user_input,
                "response": f"❌ An error occurred: {str(e)}",
                "error": str(e),
            }

    def _handle_ticket_creation(self, user_input: str) -> str:
        """Handle ticket creation requests"""
        try:
            return self.agent.run(user_input)
        except Exception as e:
            return f"❌ Error while creating ticket: {str(e)}"

    def _handle_ticket_management(self, user_input: str) -> str:
        """Handle ticket status checking and closing"""
        try:
            structured_input = extract_ticket_info_and_intent(user_input)

            if structured_input["intent"] == "close":
                structured_input["status"] = "Closed"

            return self.agent.run(f"ticket status checker input: {structured_input}")
        except Exception as e:
            return f"❌ Error during ticket operation: {str(e)}"

    def _handle_info_request(self, user_input: str) -> str:
        """Handle general information requests"""
        try:
            ai_response = llm.invoke(user_input)
            return ai_response.content
        except Exception as e:
            return f"❌ Error during information query: {str(e)}"

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "agents_loaded": len(self.agent_manager.tools),
            "agent_names": self.agent_manager.get_tool_names(),
            "llm_available": self.agent is not None,
            "database_initialized": True,
        }
