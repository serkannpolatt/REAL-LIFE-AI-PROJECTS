"""
IT Agent - Specialized agent for technology issues
===============================================

This agent helps with:
- Software issues
- Hardware failures
- Login problems
- Network connectivity issues
- System performance problems
"""
import sys
import os

# Add project root to sys.path for imports
project_root = r"C:\Users\Serkan POLAT\Desktop\agentic-ai-system-main"
if project_root not in sys.path:
    sys.path.append(project_root)



import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.db_utils import insert_it_ticket

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ITAgent:
    """Agent class for managing IT issues"""

    def __init__(self):
        self.department = "IT"
        self.supported_issues = [
            "software",
            "hardware",
            "login",
            "password",
            "network",
            "internet",
            "computer",
            "laptop",
            "server",
            "email",
            "system",
            "application",
        ]

    def handle_issue(self, prompt: str) -> str:
        """
        Handle IT issue and save to database

        Args:
            prompt: User's text describing the problem

        Returns:
            Operation result message
        """
        try:
            # Determine issue type
            issue_type = self._categorize_issue(prompt)

            # Create ticket
            result = insert_it_ticket(prompt)

            # Log the action
            logger.info(f"IT ticket created: {issue_type} - {prompt[:50]}...")

            # Return appropriate message to user
            return self._format_response(result, issue_type)

        except Exception as e:
            logger.error(f"IT agent error: {e}")
            return f"❌ Error occurred while creating IT ticket: {str(e)}"

    def _categorize_issue(self, prompt: str) -> str:
        """Categorize the issue type"""
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["password", "login", "signin"]):
            return "Login/Password Issue"
        elif any(
            word in prompt_lower for word in ["software", "application", "program"]
        ):
            return "Software Issue"
        elif any(
            word in prompt_lower
            for word in ["hardware", "computer", "laptop", "mouse", "keyboard"]
        ):
            return "Hardware Issue"
        elif any(word in prompt_lower for word in ["network", "internet", "wifi"]):
            return "Network Issue"
        elif any(word in prompt_lower for word in ["email", "mail"]):
            return "Email Issue"
        else:
            return "General IT Issue"

    def _format_response(self, db_result: str, issue_type: str) -> str:
        """Format appropriate response for user"""
        return f"""
🔧 **IT Support**

**Issue Type:** {issue_type}
**Status:** {db_result}

**Next Steps:**
• Our IT team will review your issue as soon as possible
• For urgent matters, you can call extension 101
• You can check your ticket status by saying "Check ticket status"

💡 **Tip:** Providing more details about the issue helps speed up the resolution process.
        """.strip()


# Global instance and legacy function
_it_agent_instance = ITAgent()


def handle_it_issue(prompt: str) -> str:
    """Legacy function - for backwards compatibility"""
    return _it_agent_instance.handle_issue(prompt)
