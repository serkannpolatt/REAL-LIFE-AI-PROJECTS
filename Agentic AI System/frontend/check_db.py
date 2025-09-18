"""
Database Viewer - Streamlit interface for viewing ticket database
==============================================================

Provides an interactive interface to view all tickets stored in the SQLite database.
"""

import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.db_utils import fetch_all_tickets


def init_page_config():
    """Initialize page configuration"""
    st.set_page_config(
        page_title="Ticket Database Viewer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_custom_css():
    """Apply custom CSS styles"""
    st.markdown(
        """
        <style>
            .main-header {
                text-align: center;
                color: #2c3e50;
                font-size: 2.5rem;
                margin-bottom: 2rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 1rem;
            }
            
            .status-open {
                background-color: #e74c3c;
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8rem;
            }
            
            .status-closed {
                background-color: #27ae60;
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8rem;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )


def format_dataframe(tickets, department):
    """Format ticket data for better display"""
    if not tickets:
        return pd.DataFrame()

    # Convert to DataFrame
    column_names = ["ID", "Issue/Query", "Status", "Created At"]
    df = pd.DataFrame(tickets, columns=column_names)

    # Format datetime
    df["Created At"] = pd.to_datetime(df["Created At"]).dt.strftime("%Y-%m-%d %H:%M")

    # Add status styling
    def style_status(status):
        if status.lower() == "open":
            return f'<span class="status-open">{status}</span>'
        else:
            return f'<span class="status-closed">{status}</span>'

    return df


def display_statistics(all_tickets):
    """Display ticket statistics"""
    total_tickets = sum(len(tickets) for tickets in all_tickets.values())

    if total_tickets == 0:
        st.info("📭 No tickets found in the database.")
        return

    # Calculate statistics
    stats = {}
    total_open = 0
    total_closed = 0

    for dept, tickets in all_tickets.items():
        dept_open = sum(1 for ticket in tickets if ticket[2].lower() == "open")
        dept_closed = len(tickets) - dept_open
        total_open += dept_open
        total_closed += dept_closed

        stats[dept.strip()] = {
            "total": len(tickets),
            "open": dept_open,
            "closed": dept_closed,
        }

    # Display overview metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 Total Tickets", total_tickets)

    with col2:
        st.metric("🔓 Open Tickets", total_open, delta=None)

    with col3:
        st.metric("✅ Closed Tickets", total_closed, delta=None)

    # Department breakdown
    st.markdown("### 📈 Department Breakdown")

    cols = st.columns(len(stats))
    for i, (dept, data) in enumerate(stats.items()):
        with cols[i]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>{dept}</h4>
                    <p><strong>{data["total"]}</strong> total</p>
                    <p>🔓 {data["open"]} open | ✅ {data["closed"]} closed</p>
                </div>
            """,
                unsafe_allow_html=True,
            )


def display_department_tickets(dept_name, tickets):
    """Display tickets for a specific department"""
    st.markdown(f"### 🎫 {dept_name} Tickets")

    if not tickets:
        st.info(f"No tickets found for {dept_name} department.")
        return

    # Format data
    df = format_dataframe(tickets, dept_name)

    if not df.empty:
        # Display as interactive table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn("Status", help="Ticket status"),
                "Created At": st.column_config.DatetimeColumn(
                    "Created At", help="When the ticket was created"
                ),
            },
        )

        # Show latest ticket info
        latest_ticket = tickets[0]  # Assuming sorted by latest
        with st.expander(f"Latest {dept_name} Ticket Details"):
            st.write(f"**ID:** #{latest_ticket[0]}")
            st.write(f"**Issue:** {latest_ticket[1]}")
            st.write(f"**Status:** {latest_ticket[2]}")
            st.write(f"**Created:** {latest_ticket[3]}")


def main():
    """Main application function"""
    init_page_config()
    apply_custom_css()

    # Header
    st.markdown(
        '<h1 class="main-header">📊 Ticket Database Viewer</h1>', unsafe_allow_html=True
    )
    st.markdown("---")

    # Sidebar controls
    with st.sidebar:
        st.markdown("## 🎛️ Controls")

        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

        # Navigation
        st.markdown("### 📋 Navigation")
        view_mode = st.radio(
            "Select View Mode:",
            ["📈 Overview", "🏢 By Department", "📊 Statistics Only"],
        )

        # Department filter for department view
        if view_mode == "🏢 By Department":
            st.markdown("### 🎯 Department Filter")

    try:
        # Fetch all tickets
        with st.spinner("📡 Loading ticket data..."):
            all_tickets = fetch_all_tickets()

        if view_mode == "📈 Overview":
            # Display statistics
            display_statistics(all_tickets)
            st.markdown("---")

            # Display all departments
            for dept_name, tickets in all_tickets.items():
                if tickets:  # Only show departments with tickets
                    display_department_tickets(dept_name, tickets)
                    st.markdown("---")

        elif view_mode == "🏢 By Department":
            # Department selection
            with st.sidebar:
                dept_options = [
                    dept for dept, tickets in all_tickets.items() if tickets
                ]
                if dept_options:
                    selected_dept = st.selectbox(
                        "Choose Department:", dept_options, key="dept_selector"
                    )

                    # Display selected department
                    display_department_tickets(
                        selected_dept, all_tickets[selected_dept]
                    )
                else:
                    st.warning("No departments with tickets found.")

        elif view_mode == "📊 Statistics Only":
            # Only display statistics
            display_statistics(all_tickets)

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("💡 Make sure the database exists and is accessible.")


if __name__ == "__main__":
    main()
