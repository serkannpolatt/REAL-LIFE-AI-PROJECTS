"""
Admin Agent - Specialized agent for Administrative matters
=========================================================

This agent helps with:
- Access requests and permissions
- Office supplies and equipment
- Meeting room reservations
- General administrative support
- Facility management requests
"""

import logging
import sys
import os

# Add project root to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.db_utils import insert_admin_ticket

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdminAgent:
    """Agent class for managing Administrative issues"""

    def __init__(self):
        self.department = "Admin"
        self.supported_issues = [
            "access",
            "permission",
            "card",
            "badge",
            "key",
            "supplies",
            "stationery",
            "equipment",
            "furniture",
            "meeting",
            "room",
            "reservation",
            "booking",
            "parking",
            "security",
            "visitor",
            "mail",
            "logistics",
            "transport",
            "delivery",
        ]

    def handle_issue(self, prompt: str) -> str:
        """
        Handle Administrative issue and save to database

        Args:
            prompt: User's text describing the administrative matter

        Returns:
            Operation result message
        """
        try:
            # Determine issue type
            issue_type = self._categorize_issue(prompt)

            # Create ticket
            result = insert_admin_ticket(prompt)

            # Log the action
            logger.info(f"Admin ticket created: {issue_type} - {prompt[:50]}...")

            # Return appropriate message to user
            return self._format_response(result, issue_type)

        except Exception as e:
            logger.error(f"Admin agent error: {e}")
            return f"❌ Error occurred while creating Admin ticket: {str(e)}"

    def _categorize_issue(self, prompt: str) -> str:
        """Categorize the Administrative issue type"""
        prompt_lower = prompt.lower()

        if any(
            word in prompt_lower
            for word in ["access", "permission", "card", "badge", "key"]
        ):
            return "Access Request"
        elif any(
            word in prompt_lower
            for word in ["supplies", "stationery", "equipment", "furniture"]
        ):
            return "Office Supplies/Equipment"
        elif any(
            word in prompt_lower
            for word in ["meeting", "room", "reservation", "booking"]
        ):
            return "Meeting Room/Booking"
        elif any(word in prompt_lower for word in ["parking", "security", "visitor"]):
            return "Security/Parking"
        elif any(
            word in prompt_lower
            for word in ["logistics", "transport", "delivery", "mail"]
        ):
            return "Logistics/Delivery"
        else:
            return "General Administrative"

    def _format_response(self, db_result: str, issue_type: str) -> str:
        """Format appropriate response for user"""
        return f"""
🏢 **Administrative Support**

**Issue Type:** {issue_type}
**Status:** {db_result}

**Next Steps:**
• Admin team will handle your request within 1-2 business days
• For urgent access issues, contact security at extension 103
• For supply requests, approval may be required
• You can check your ticket status by saying "Check ticket status"

💡 **Note:** Some requests may require manager approval before processing.
        """.strip()


# Global instance and legacy function
_admin_agent_instance = AdminAgent()


def handle_admin_issue(prompt: str) -> str:
    """Legacy function - for backwards compatibility"""
    return _admin_agent_instance.handle_issue(prompt)
