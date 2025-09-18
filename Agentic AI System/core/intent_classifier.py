"""
Intent Classifier - Handles classification of user intents
"""
import sys
import os

# Add project root to sys.path for imports
project_root = r"C:\Users\Serkan POLAT\Desktop\agentic-ai-system-main"
if project_root not in sys.path:
    sys.path.append(project_root)



from typing import Literal
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_instance import llm


class IntentClassifier:
    """Classifies user intents for proper routing"""

    def __init__(self):
        self.valid_intents = ("ticket", "info", "status", "close")

    def classify(self, user_input: str) -> Literal["ticket", "info", "status", "close"]:
        """
        Classify user intent into predefined categories

        Args:
            user_input: The user's message

        Returns:
            One of: "ticket", "info", "status", "close"
        """
        intent_prompt = self._build_intent_prompt(user_input)

        try:
            result = llm.invoke(intent_prompt).content.strip().lower()
            return result if result in self.valid_intents else "info"
        except Exception as e:
            print(f"[Intent Classification Error] {e}")
            return "info"

    def _build_intent_prompt(self, user_input: str) -> str:
        """Build the prompt for intent classification"""
        return f"""
        Classify the user's message into one of these categories:
        
        - "ticket": User is reporting a problem, requesting support, or wants to create a new ticket
        - "info": General information question, policy inquiry, or informational message  
        - "status": Wants to check the status of an existing ticket
        - "close": Wants to close a ticket
        
        User message: "{user_input}"
        
        Return only one word: ticket, info, status, or close
        """
