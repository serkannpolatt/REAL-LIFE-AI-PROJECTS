"""
Ticket parsing utilities for extracting ticket information from user prompts.
"""

import re
import sys
import os
from typing import Dict, Optional, Any

# Add project root to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.llm_instance import llm


class TicketParser:
    """Parser class for extracting ticket information from user messages."""

    def __init__(self):
        """Initialize the ticket parser."""
        self.valid_departments = ["it", "hr", "finance", "admin", "infra"]
        self.valid_intents = ["check", "close", "status", "update"]

    def _extract_ticket_id(self, text: str) -> Optional[int]:
        """Extract ticket ID from text using regex patterns."""
        patterns = [r"ticket\s*#?(\d+)", r"id\s*#?(\d+)", r"#(\d+)", r"\b(\d+)\b"]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return None

    def _extract_department(self, text: str) -> Optional[str]:
        """Extract department from text."""
        text_lower = text.lower()
        for dept in self.valid_departments:
            if dept in text_lower:
                return dept
        return None

    def _extract_intent(self, text: str) -> str:
        """Extract intent from text."""
        text_lower = text.lower()

        # Check for specific intent keywords
        if any(
            word in text_lower for word in ["close", "resolved", "done", "finished"]
        ):
            return "close"
        elif any(
            word in text_lower for word in ["check", "status", "see", "view", "show"]
        ):
            return "check"
        elif any(word in text_lower for word in ["update", "modify", "change"]):
            return "update"

        return "check"  # Default intent

    def parse_ticket_request(self, prompt: str) -> Dict[str, Any]:
        """
        Parse user prompt to extract ticket information.

        Args:
            prompt: User's message containing ticket request

        Returns:
            Dictionary containing ticket_id, department, and intent
        """
        try:
            # First try regex-based extraction
            ticket_id = self._extract_ticket_id(prompt)
            department = self._extract_department(prompt)
            intent = self._extract_intent(prompt)

            # If we have enough information, return it
            if ticket_id and department:
                return {
                    "ticket_id": ticket_id,
                    "department": department,
                    "intent": intent,
                    "method": "regex",
                }

            # Fall back to LLM-based extraction
            return self._llm_extract(prompt)

        except Exception as e:
            print(f"[Ticket Parser Error] {e}")
            return {
                "ticket_id": None,
                "department": None,
                "intent": "missing",
                "error": str(e),
            }

    def _llm_extract(self, prompt: str) -> Dict[str, Any]:
        """Use LLM to extract ticket information when regex fails."""
        extract_prompt = f"""
        From the following user message, extract:
        - ticket_id (as a number)
        - department (one of: it, hr, finance, admin, infra)
        - intent (check, close, status, or update)

        Return format (lowercase, no punctuation):
        ticket_id=3, department=finance, intent=close

        If anything is missing or unclear, return: missing

        User message: "{prompt}"
        """

        try:
            response = llm.invoke(extract_prompt).content.strip().lower()

            if (
                "ticket_id" in response
                and "department" in response
                and "intent" in response
            ):
                parts = {}
                for part in response.split(","):
                    if "=" in part:
                        key, value = part.strip().split("=", 1)
                        parts[key.strip()] = value.strip()

                ticket_id = None
                if parts.get("ticket_id", "").isdigit():
                    ticket_id = int(parts["ticket_id"])

                return {
                    "ticket_id": ticket_id,
                    "department": parts.get("department"),
                    "intent": parts.get("intent", "check"),
                    "method": "llm",
                }
            else:
                return {
                    "ticket_id": None,
                    "department": None,
                    "intent": "missing",
                    "method": "llm_failed",
                }

        except Exception as e:
            print(f"[LLM Extraction Error] {e}")
            return {
                "ticket_id": None,
                "department": None,
                "intent": "missing",
                "error": str(e),
            }


# Global parser instance for backward compatibility
_parser = TicketParser()


def extract_ticket_info_and_intent(prompt: str) -> dict:
    """
    Legacy function for backward compatibility.
    Extract ticket_id, department, and intent from user prompt.

    Args:
        prompt: User's message

    Returns:
        Dictionary with ticket information
    """
    return _parser.parse_ticket_request(prompt)
