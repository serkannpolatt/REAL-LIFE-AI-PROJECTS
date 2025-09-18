"""
Infrastructure Agent - Specialized agent for Infrastructure issues
=================================================================

This agent helps with:
- WiFi and network connectivity problems
- Power outages and electrical issues
- Air conditioning and heating
- Physical facility problems
- Equipment maintenance requests
"""

import logging
import sys
import os

# Add project root to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.db_utils import insert_infra_ticket

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InfraAgent:
    """Agent class for managing Infrastructure issues"""

    def __init__(self):
        self.department = "Infrastructure"
        self.supported_issues = [
            "wifi",
            "network",
            "internet",
            "connectivity",
            "power",
            "electricity",
            "outage",
            "electrical",
            "aircon",
            "heating",
            "cooling",
            "temperature",
            "elevator",
            "lift",
            "stairs",
            "lighting",
            "plumbing",
            "water",
            "restroom",
            "facility",
        ]

    def handle_issue(self, prompt: str) -> str:
        """
        Handle Infrastructure issue and save to database

        Args:
            prompt: User's text describing the infrastructure problem

        Returns:
            Operation result message
        """
        try:
            # Determine issue type
            issue_type = self._categorize_issue(prompt)

            # Create ticket
            result = insert_infra_ticket(prompt)

            # Log the action
            logger.info(
                f"Infrastructure ticket created: {issue_type} - {prompt[:50]}..."
            )

            # Return appropriate message to user
            return self._format_response(result, issue_type)

        except Exception as e:
            logger.error(f"Infrastructure agent error: {e}")
            return f"❌ Error occurred while creating Infrastructure ticket: {str(e)}"

    def _categorize_issue(self, prompt: str) -> str:
        """Categorize the Infrastructure issue type"""
        prompt_lower = prompt.lower()

        if any(
            word in prompt_lower
            for word in ["wifi", "network", "internet", "connectivity"]
        ):
            return "Network/WiFi Issue"
        elif any(
            word in prompt_lower
            for word in ["power", "electricity", "outage", "electrical"]
        ):
            return "Power/Electrical Issue"
        elif any(
            word in prompt_lower
            for word in ["aircon", "heating", "cooling", "temperature"]
        ):
            return "HVAC Issue"
        elif any(word in prompt_lower for word in ["elevator", "lift", "stairs"]):
            return "Building Access Issue"
        elif any(word in prompt_lower for word in ["plumbing", "water", "restroom"]):
            return "Plumbing/Water Issue"
        elif any(word in prompt_lower for word in ["lighting", "light", "bulb"]):
            return "Lighting Issue"
        else:
            return "General Infrastructure Issue"

    def _format_response(self, db_result: str, issue_type: str) -> str:
        """Format appropriate response for user"""
        return f"""
⚡ **Infrastructure Support**

**Issue Type:** {issue_type}
**Status:** {db_result}

**Next Steps:**
• Facilities team will investigate and resolve the issue promptly
• For emergency issues (power/safety), call security immediately
• Please specify location details for faster response
• You can check your ticket status by saying "Check ticket status"

💡 **Emergency Contact:** For urgent facility issues, call extension 104.
        """.strip()


# Global instance and legacy function
_infra_agent_instance = InfraAgent()


def handle_infra_issue(prompt: str) -> str:
    """Legacy function - for backwards compatibility"""
    return _infra_agent_instance.handle_issue(prompt)
