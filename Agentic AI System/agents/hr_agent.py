"""
HR Agent - Specialized agent for Human Resources matters
=======================================================

This agent helps with:
- Leave applications and time-off requests
- Employee complaints and grievances
- Company policy inquiries
- Resignation processes and procedures
- Benefits and compensation questions
"""

import logging
import sys
import os

# Add project root to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.db_utils import insert_hr_ticket

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HRAgent:
    """Agent class for managing HR issues"""

    def __init__(self):
        self.department = "HR"
        self.supported_issues = [
            "leave",
            "vacation",
            "sick",
            "time off",
            "absence",
            "complaint",
            "grievance",
            "policy",
            "resignation",
            "quit",
            "benefits",
            "insurance",
            "compensation",
            "performance",
            "harassment",
            "discrimination",
            "workplace",
        ]

    def handle_issue(self, prompt: str) -> str:
        """
        Handle HR issue and save to database

        Args:
            prompt: User's text describing the HR matter

        Returns:
            Operation result message
        """
        try:
            # Determine issue type
            issue_type = self._categorize_issue(prompt)

            # Create ticket
            result = insert_hr_ticket(prompt)

            # Log the action
            logger.info(f"HR ticket created: {issue_type} - {prompt[:50]}...")

            # Return appropriate message to user
            return self._format_response(result, issue_type)

        except Exception as e:
            logger.error(f"HR agent error: {e}")
            return f"❌ Error occurred while creating HR ticket: {str(e)}"

    def _categorize_issue(self, prompt: str) -> str:
        """Categorize the HR issue type"""
        prompt_lower = prompt.lower()

        if any(
            word in prompt_lower
            for word in ["leave", "vacation", "sick", "time off", "absence"]
        ):
            return "Leave Request"
        elif any(
            word in prompt_lower
            for word in ["complaint", "grievance", "harassment", "discrimination"]
        ):
            return "Employee Complaint"
        elif any(
            word in prompt_lower
            for word in ["policy", "rule", "procedure", "guideline"]
        ):
            return "Policy Inquiry"
        elif any(
            word in prompt_lower
            for word in ["resignation", "quit", "leave company", "terminate"]
        ):
            return "Resignation Process"
        elif any(
            word in prompt_lower
            for word in ["benefits", "insurance", "compensation", "salary"]
        ):
            return "Benefits & Compensation"
        elif any(
            word in prompt_lower for word in ["performance", "review", "evaluation"]
        ):
            return "Performance Management"
        else:
            return "General HR Issue"

    def _format_response(self, db_result: str, issue_type: str) -> str:
        """Format appropriate response for user"""
        return f"""
👥 **HR Support**

**Issue Type:** {issue_type}
**Status:** {db_result}

**Next Steps:**
• HR team will review your request within 1-2 business days
• For urgent matters, contact HR directly at extension 101
• Confidentiality is maintained for all HR matters
• You can check your ticket status by saying "Check ticket status"

💡 **Tip:** Provide specific details and dates for faster processing.
        """.strip()


# Global instance and legacy function
_hr_agent_instance = HRAgent()


def handle_hr_issue(prompt: str) -> str:
    """Legacy function - for backwards compatibility"""
    return _hr_agent_instance.handle_issue(prompt)
