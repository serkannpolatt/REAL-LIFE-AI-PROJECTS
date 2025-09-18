"""
Ticket status enhancement utilities for generating user-friendly status summaries.
"""

import sys
import os
from typing import Dict

# Add project root to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_instance import llm


class StatusEnhancer:
    """Class for enhancing ticket status information with user-friendly summaries."""

    def __init__(self):
        """Initialize the status enhancer."""
        self.status_templates = {
            "open": "Your {department} ticket #{ticket_id} is currently being reviewed by our team.",
            "in_progress": "Your {department} ticket #{ticket_id} is actively being worked on.",
            "resolved": "Your {department} ticket #{ticket_id} has been successfully resolved.",
            "closed": "Your {department} ticket #{ticket_id} has been closed.",
            "pending": "Your {department} ticket #{ticket_id} is waiting for additional information.",
        }

    def _get_fallback_summary(
        self, department: str, ticket_id: int, issue: str, status: str
    ) -> str:
        """Generate a fallback summary when LLM is unavailable."""
        template = self.status_templates.get(
            status.lower(), "Your {department} ticket #{ticket_id} status: {status}"
        )

        base_summary = template.format(
            department=department.upper(), ticket_id=ticket_id, status=status
        )

        if issue and len(issue) > 0:
            return f"{base_summary} Issue: {issue[:100]}{'...' if len(issue) > 100 else ''}"

        return base_summary

    def enhance_ticket_status(
        self, department: str, ticket_id: int, issue: str, status: str
    ) -> str:
        """
        Generate a user-friendly summary of ticket status.

        Args:
            department: Department handling the ticket
            ticket_id: Unique ticket identifier
            issue: Description of the issue
            status: Current status of the ticket

        Returns:
            User-friendly status summary
        """
        try:
            # Try LLM-based enhancement first
            prompt = f"""
            Write a friendly and professional summary of this support ticket:
            
            Department: {department.upper()}
            Ticket ID: #{ticket_id}
            Issue: {issue}
            Current Status: {status}
            
            Requirements:
            - Keep it concise (1-2 sentences)
            - Use friendly, professional tone
            - Avoid technical jargon
            - Include next steps if applicable
            - Start with the ticket reference
            """

            response = llm.invoke(prompt)
            enhanced_summary = response.content.strip()

            # Validate the response
            if enhanced_summary and len(enhanced_summary) > 10:
                return enhanced_summary
            else:
                return self._get_fallback_summary(department, ticket_id, issue, status)

        except Exception as e:
            print(f"[Status Enhancement Error] {e}")
            return self._get_fallback_summary(department, ticket_id, issue, status)

    def enhance_multiple_tickets(self, tickets: list) -> Dict[int, str]:
        """
        Enhance status for multiple tickets.

        Args:
            tickets: List of ticket dictionaries

        Returns:
            Dictionary mapping ticket_id to enhanced status
        """
        enhanced_statuses = {}

        for ticket in tickets:
            try:
                ticket_id = ticket.get("id")
                if ticket_id:
                    enhanced_statuses[ticket_id] = self.enhance_ticket_status(
                        ticket.get("department", "General"),
                        ticket_id,
                        ticket.get("issue", "No description available"),
                        ticket.get("status", "Unknown"),
                    )
            except Exception as e:
                print(
                    f"[Multiple Enhancement Error] Ticket {ticket.get('id', 'Unknown')}: {e}"
                )
                enhanced_statuses[ticket.get("id", 0)] = (
                    "Status information unavailable"
                )

        return enhanced_statuses

    def get_status_priority(self, status: str) -> int:
        """
        Get priority level for status sorting.

        Args:
            status: Ticket status

        Returns:
            Priority level (lower number = higher priority)
        """
        priority_map = {
            "open": 1,
            "in_progress": 2,
            "pending": 3,
            "resolved": 4,
            "closed": 5,
        }
        return priority_map.get(status.lower(), 6)


# Global enhancer instance for backward compatibility
_enhancer = StatusEnhancer()


def enhance_ticket_status(
    department: str, ticket_id: int, issue: str, status: str
) -> str:
    """
    Legacy function for backward compatibility.
    Summarize ticket status into a user-friendly sentence.

    Args:
        department: Department handling the ticket
        ticket_id: Unique ticket identifier
        issue: Description of the issue
        status: Current status of the ticket

    Returns:
        User-friendly status summary
    """
    return _enhancer.enhance_ticket_status(department, ticket_id, issue, status)
