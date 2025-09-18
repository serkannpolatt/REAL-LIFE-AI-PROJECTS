"""
Ticket Status Agent - Handles ticket status checking and closing operations
==========================================================================

This agent manages:
- Checking ticket status by ID and department
- Closing tickets when requested
- Providing enhanced status information
"""

import re
import sqlite3
import sys
import os

# Add project root to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.enhance_status import enhance_ticket_status
from utils.ticket_parser import extract_ticket_info_and_intent
from utils.db_utils import get_db_path, get_ticket_by_id, update_ticket_status


def check_ticket_status(query: str) -> str:
    """
    Check the status of a ticket based on user query

    Args:
        query: User query containing ticket information

    Returns:
        str: Ticket status information
    """
    # Parse structured input if available
    if "ticket_id=" in query and "department=" in query:
        try:
            parts = dict(item.strip().split("=") for item in query.split(","))
            ticket_id = int(parts["ticket_id"])
        except Exception:
            return "❌ Failed to parse structured ticket info."
    else:
        # Extract ticket ID from natural language
        ticket_id_match = re.search(r"ticket\s*#?(\d+)", query, re.IGNORECASE)
        if not ticket_id_match:
            return "❓ Please provide a ticket number like 'ticket #3' or 'check ticket 5'."

        ticket_id = int(ticket_id_match.group(1))

    # Department mapping
    department_map = {
        "it": ("it_tickets", "issue"),
        "infra": ("infra_tickets", "issue"),
        "hr": ("hr_tickets", "issue"),
        "finance": ("finance_tickets", "query"),
        "admin": ("admin_tickets", "issue"),
    }

    # Detect department from query
    query_lower = query.lower()
    selected_table = None

    for keyword, (table, column) in department_map.items():
        if keyword in query_lower:
            selected_table = (table, column, keyword)
            break

    db_path = get_db_path()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        if selected_table:
            table, column, dept_name = selected_table
            cursor.execute(
                f"SELECT id, {column}, status, created_at FROM {table} WHERE id = ?",
                (ticket_id,),
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return enhance_ticket_status(dept_name.upper(), row[0], row[1], row[2])
            else:
                return f"❌ No {dept_name.upper()} ticket found with ID #{ticket_id}."
        else:
            # Search all departments
            for dept_key, (table, column) in department_map.items():
                cursor.execute(
                    f"SELECT id, {column}, status, created_at FROM {table} WHERE id = ?",
                    (ticket_id,),
                )
                row = cursor.fetchone()
                if row:
                    conn.close()
                    return enhance_ticket_status(
                        dept_key.upper(), row[0], row[1], row[2]
                    )

            conn.close()
            return f"❌ Ticket ID #{ticket_id} not found in any department."

    except Exception as e:
        return f"❌ Error checking ticket status: {str(e)}"


def close_ticket_status(query: str) -> str:
    """
    Close a ticket based on user query

    Args:
        query: User query containing ticket close request

    Returns:
        str: Operation result message
    """
    try:
        info = extract_ticket_info_and_intent(query)

        if not isinstance(info, dict) or info.get("intent") == "missing":
            return "❌ Could not extract ticket info. Please specify both ticket ID and department (e.g., 'close IT ticket 5')."

        ticket_id = info.get("ticket_id")
        department = info.get("department")

        if not ticket_id or not department:
            return "❌ Please provide both ticket ID and department name."

        # Use the improved database functions
        ticket_data = get_ticket_by_id(department, ticket_id)

        if not ticket_data:
            return (
                f"❌ Ticket #{ticket_id} not found in {department.title()} department."
            )

        current_status = ticket_data[2]  # Status is the 3rd column

        if current_status.lower() == "closed":
            return f"ℹ️ Ticket #{ticket_id} is already closed."

        # Update ticket status
        success = update_ticket_status(department, ticket_id, "Closed")

        if success:
            return f"✅ Ticket #{ticket_id} in {department.title()} department has been marked as Closed."
        else:
            return f"❌ Failed to close ticket #{ticket_id}."

    except Exception as e:
        return f"❌ Error closing ticket: {str(e)}"


def preprocess_user_query(query: str) -> dict:
    """
    Extract intent, ticket ID, and department from user query

    Args:
        query: User query string

    Returns:
        dict: Extracted information
    """
    intent = (
        "check"
        if "check" in query.lower()
        else "close"
        if "close" in query.lower()
        else None
    )

    ticket_id_match = re.search(r"ticket[^\d]*(\d+)", query, re.IGNORECASE)
    ticket_id = int(ticket_id_match.group(1)) if ticket_id_match else None

    department = None
    for dept in ["infra", "it", "admin", "hr", "finance"]:
        if dept in query.lower():
            department = dept
            break

    return {"intent": intent, "ticket_id": ticket_id, "department": department}


def handle_ticket_status(query: str) -> str:
    """
    Main handler for ticket status operations

    Args:
        query: User query for ticket status operation

    Returns:
        str: Operation result message
    """
    try:
        # Try to extract structured information
        info = extract_ticket_info_and_intent(query)

        # If extraction fails, use fallback parsing
        if not isinstance(info, dict) or info.get("intent") == "missing":
            info = preprocess_user_query(query)

            if not info.get("ticket_id"):
                return "❓ Please provide a ticket ID (e.g., 'check ticket 5' or 'close IT ticket 3')."

        intent = info.get("intent")

        if not intent:
            return (
                "❓ Please specify what you want to do: 'check' or 'close' the ticket."
            )

        if intent == "check":
            return check_ticket_status(query)
        elif intent == "close":
            return close_ticket_status(query)
        else:
            return "❌ Unknown operation. Please specify if you want to 'check' or 'close' a ticket."

    except Exception as e:
        return f"❌ Error handling ticket status request: {str(e)}"
