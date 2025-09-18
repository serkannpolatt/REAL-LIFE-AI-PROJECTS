"""
Database Utilities - SQLite database operations for ticket management
====================================================================

Handles all database operations including table creation, ticket insertion,
and data retrieval for the ticket management system.
"""

import sqlite3
import os
from typing import Dict, List, Tuple


def get_db_path() -> str:
    """
    Get the database file path relative to project root

    Returns:
        str: Path to the database file
    """
    # Get project root directory (parent of utils directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_dir = os.path.join(project_root, "db")

    # Create db directory if it doesn't exist
    os.makedirs(db_dir, exist_ok=True)

    return os.path.join(db_dir, "system.db")


def init_db():
    """Initialize the database with required tables"""
    db_path = get_db_path()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Infrastructure tickets table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS infra_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue TEXT NOT NULL,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # HR tickets table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS hr_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue TEXT NOT NULL,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # IT tickets table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS it_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue TEXT NOT NULL,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Finance tickets table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS finance_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Admin tickets table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue TEXT NOT NULL,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()
        conn.close()
        print(f"✅ Database initialized at: {db_path}")

    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        raise


def _insert_ticket(table: str, column: str, value: str) -> str:
    """
    Insert a ticket into the specified table

    Args:
        table: Table name to insert into
        column: Column name for the ticket content
        value: Ticket content/issue description

    Returns:
        str: Success message with ticket ID
    """
    db_path = get_db_path()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(f"INSERT INTO {table} ({column}) VALUES (?)", (value,))

        conn.commit()
        ticket_id = cursor.lastrowid
        conn.close()

        department = table.replace("_tickets", "").title()
        return f"✅ {department} ticket #{ticket_id} created successfully"

    except Exception as e:
        print(f"❌ Error inserting ticket: {e}")
        return f"❌ Failed to create ticket: {str(e)}"


# Department-specific ticket insertion functions
def insert_infra_ticket(issue: str) -> str:
    """Insert an infrastructure ticket"""
    return _insert_ticket("infra_tickets", "issue", issue)


def insert_it_ticket(issue: str) -> str:
    """Insert an IT ticket"""
    return _insert_ticket("it_tickets", "issue", issue)


def insert_finance_ticket(query: str) -> str:
    """Insert a finance ticket"""
    return _insert_ticket("finance_tickets", "query", query)


def insert_hr_ticket(issue: str) -> str:
    """Insert an HR ticket"""
    return _insert_ticket("hr_tickets", "issue", issue)


def insert_admin_ticket(issue: str) -> str:
    """Insert an admin ticket"""
    return _insert_ticket("admin_tickets", "issue", issue)


def fetch_all_tickets() -> Dict[str, List[Tuple]]:
    """
    Fetch all tickets from all departments

    Returns:
        Dict: Dictionary with department names as keys and ticket lists as values
    """
    db_path = get_db_path()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        tables = {
            "Infrastructure": ("infra_tickets", "issue"),
            "IT": ("it_tickets", "issue"),
            "HR": ("hr_tickets", "issue"),
            "Finance": ("finance_tickets", "query"),
            "Admin": ("admin_tickets", "issue"),
        }

        all_data = {}

        for department, (table, column) in tables.items():
            cursor.execute(
                f"SELECT id, {column}, status, created_at FROM {table} ORDER BY id DESC"
            )
            all_data[department] = cursor.fetchall()

        conn.close()
        return all_data

    except Exception as e:
        print(f"❌ Error fetching tickets: {e}")
        return {}


def get_ticket_by_id(department: str, ticket_id: int) -> Tuple:
    """
    Get a specific ticket by department and ID

    Args:
        department: Department name (it, hr, finance, admin, infra)
        ticket_id: Ticket ID number

    Returns:
        Tuple: Ticket data or None if not found
    """
    db_path = get_db_path()

    department_map = {
        "it": ("it_tickets", "issue"),
        "hr": ("hr_tickets", "issue"),
        "finance": ("finance_tickets", "query"),
        "admin": ("admin_tickets", "issue"),
        "infra": ("infra_tickets", "issue"),
    }

    if department.lower() not in department_map:
        return None

    table, column = department_map[department.lower()]

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT id, {column}, status, created_at FROM {table} WHERE id = ?",
            (ticket_id,),
        )

        result = cursor.fetchone()
        conn.close()

        return result

    except Exception as e:
        print(f"❌ Error fetching ticket: {e}")
        return None


def update_ticket_status(department: str, ticket_id: int, new_status: str) -> bool:
    """
    Update the status of a specific ticket

    Args:
        department: Department name
        ticket_id: Ticket ID
        new_status: New status value

    Returns:
        bool: True if update successful, False otherwise
    """
    db_path = get_db_path()

    department_map = {
        "it": "it_tickets",
        "hr": "hr_tickets",
        "finance": "finance_tickets",
        "admin": "admin_tickets",
        "infra": "infra_tickets",
    }

    if department.lower() not in department_map:
        return False

    table = department_map[department.lower()]

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"UPDATE {table} SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (new_status, ticket_id),
        )

        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()

        return rows_affected > 0

    except Exception as e:
        print(f"❌ Error updating ticket status: {e}")
        return False
