"""
Finance Agent - Specialized agent for Financial matters
======================================================

This agent helps with:
- Salary and payroll issues
- Expense reimbursements
- Payment delays and inquiries
- Invoice and billing questions
- Budget-related requests
"""

import logging
import sys
import os

# Add project root to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.db_utils import insert_finance_ticket

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinanceAgent:
    """Agent class for managing Finance issues"""

    def __init__(self):
        self.department = "Finance"
        self.supported_issues = [
            "salary",
            "payroll",
            "payment",
            "pay",
            "wage",
            "expense",
            "reimbursement",
            "receipt",
            "claim",
            "invoice",
            "billing",
            "budget",
            "cost",
            "overtime",
            "bonus",
            "allowance",
            "deduction",
        ]

    def handle_issue(self, prompt: str) -> str:
        """
        Handle Finance issue and save to database

        Args:
            prompt: User's text describing the financial matter

        Returns:
            Operation result message
        """
        try:
            # Determine issue type
            issue_type = self._categorize_issue(prompt)

            # Create ticket
            result = insert_finance_ticket(prompt)

            # Log the action
            logger.info(f"Finance ticket created: {issue_type} - {prompt[:50]}...")

            # Return appropriate message to user
            return self._format_response(result, issue_type)

        except Exception as e:
            logger.error(f"Finance agent error: {e}")
            return f"❌ Error occurred while creating Finance ticket: {str(e)}"

    def _categorize_issue(self, prompt: str) -> str:
        """Categorize the Finance issue type"""
        prompt_lower = prompt.lower()

        if any(
            word in prompt_lower
            for word in ["salary", "payroll", "payment", "pay", "wage"]
        ):
            return "Salary/Payroll Issue"
        elif any(
            word in prompt_lower
            for word in ["expense", "reimbursement", "receipt", "claim"]
        ):
            return "Expense Reimbursement"
        elif any(word in prompt_lower for word in ["invoice", "billing", "bill"]):
            return "Invoice/Billing Issue"
        elif any(word in prompt_lower for word in ["overtime", "bonus", "allowance"]):
            return "Additional Compensation"
        elif any(word in prompt_lower for word in ["budget", "cost", "approve"]):
            return "Budget Request"
        else:
            return "General Finance Issue"

    def _format_response(self, db_result: str, issue_type: str) -> str:
        """Format appropriate response for user"""
        return f"""
💰 **Finance Support**

**Issue Type:** {issue_type}
**Status:** {db_result}

**Next Steps:**
• Finance team will process your request within 3-5 business days
• For urgent payment issues, contact extension 102
• Keep receipts and documentation ready if requested
• You can check your ticket status by saying "Check ticket status"

💡 **Tip:** Include relevant details like dates, amounts, and reference numbers.
        """.strip()


# Global instance and legacy function
_finance_agent_instance = FinanceAgent()


def handle_finance_issue(prompt: str) -> str:
    """Legacy function - for backwards compatibility"""
    return _finance_agent_instance.handle_issue(prompt)
