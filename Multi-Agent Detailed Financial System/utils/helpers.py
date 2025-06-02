"""
Utility functions for FinAgents project.
This module contains helper functions used across the application.
"""

import os
from datetime import datetime
from graphviz import Digraph


def parse_report_sections(report_text):
    """
    Parse the report text into sections.

    Args:
        report_text (str): The full text of the report

    Returns:
        dict: Dictionary with report sections
    """
    report_sections = {}
    current_section = None
    content = []

    # Convert result to string and split by lines
    report_lines = report_text.split("\n")

    # Define section headers to look for
    section_headers = [
        "EXECUTIVE SUMMARY",
        "MARKET OVERVIEW",
        "PORTFOLIO PERFORMANCE ANALYSIS",
        "RISK ASSESSMENT",
        "ALLOCATION RECOMMENDATIONS",
        "IMPLEMENTATION STRATEGY",
        "FUTURE OUTLOOK",
        "CONCLUSION",
    ]

    # Process each line to extract sections
    for line in report_lines:
        line_clean = line.strip()
        # Check if this line is a section header
        is_header = False
        for header in section_headers:
            if header in line_clean.upper():
                is_header = True
                if current_section:
                    report_sections[current_section.lower().replace(" ", "_")] = (
                        "\n".join(content)
                    )
                current_section = header
                content = []
                break

        # If not a header and we have a current section, add to content
        if not is_header and current_section:
            content.append(line)

    # Add the last section if there is one
    if current_section and content:
        report_sections[current_section.lower().replace(" ", "_")] = "\n".join(content)

    # If any sections are missing, add placeholders
    for header in section_headers:
        key = header.lower().replace(" ", "_")
        if key not in report_sections:
            report_sections[key] = f"No {header} content provided."

    return report_sections


def generate_workflow_diagram(output_path="portfolio_workflow"):
    """
    Generate a diagram of the portfolio analysis workflow.

    Args:
        output_path (str): Path to save the diagram

    Returns:
        str: Path to the generated diagram
    """
    try:
        # Create a new Digraph with improved styling
        dot = Digraph(comment="Portfolio Analysis Workflow", format="png", engine="dot")

        # Set graph attributes for better appearance
        dot.attr(
            rankdir="TB",  # Top to bottom layout
            size="8,5",  # Size in inches
            dpi="300",  # Higher resolution
            bgcolor="#f7f7f7",  # Light background
            fontname="Arial",
            fontsize="14",
            margin="0.5,0.5",
        )

        # Set default node attributes
        dot.attr(
            "node",
            shape="ellipse",
            style="filled,rounded",
            color="#333333",
            fontname="Arial",
            fontsize="14",
            height="1.2",
            width="2.5",
            penwidth="2",
        )

        # Set default edge attributes
        dot.attr(
            "edge",
            fontname="Arial",
            fontsize="12",
            fontcolor="#505050",
            color="#666666",
            penwidth="1.5",
            arrowsize="0.8",
        )

        # Create nodes with custom colors for each agent - using proper HTML labels
        # Note: We need to set HTML=True to enable HTML-like labels
        dot.node(
            "A",
            label='<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><FONT POINT-SIZE="16"><B>📊 Risk Analyst</B></FONT></TD></TR></TABLE>>',
            fillcolor="#E6F3FF",
            fontcolor="#0066CC",
            style="filled,rounded",
            shape="ellipse",
            _attributes={"fontname": "Arial"},
        )

        dot.node(
            "B",
            label='<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><FONT POINT-SIZE="16"><B>🌎 Market Analyst</B></FONT></TD></TR></TABLE>>',
            fillcolor="#E6FFE6",
            fontcolor="#006600",
            style="filled,rounded",
            shape="ellipse",
            _attributes={"fontname": "Arial"},
        )

        dot.node(
            "C",
            label='<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><FONT POINT-SIZE="16"><B>📈 Allocation Optimizer</B></FONT></TD></TR></TABLE>>',
            fillcolor="#FFF0E6",
            fontcolor="#CC6600",
            style="filled,rounded",
            shape="ellipse",
            _attributes={"fontname": "Arial"},
        )

        dot.node(
            "D",
            label='<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><FONT POINT-SIZE="16"><B>📝 Portfolio Manager</B></FONT></TD></TR></TABLE>>',
            fillcolor="#F3E6FF",
            fontcolor="#660099",
            style="filled,rounded",
            shape="ellipse",
            _attributes={"fontname": "Arial"},
        )

        dot.node(
            "E",
            label='<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><FONT POINT-SIZE="16"><B>📄 Report Generator</B></FONT></TD></TR></TABLE>>',
            fillcolor="#FFE6E6",
            fontcolor="#CC0000",
            style="filled,rounded",
            shape="ellipse",
            _attributes={"fontname": "Arial"},
        )

        # Create edges with meaningful connections
        dot.edge("A", "C", label="Risk analysis")
        dot.edge("B", "C", label="Market outlook")
        dot.edge("C", "D", label="Allocation proposal")
        dot.edge("D", "E", label="Final allocation decision")
        dot.edge("A", "E", label="Risk assessment")
        dot.edge("B", "E", label="Market insights")

        # Render the graph
        dot.render(output_path, format="png", cleanup=True)
        print(f"Enhanced diagram generated: {output_path}.png")
        return f"{output_path}.png"
    except Exception as e:
        print("Error generating workflow diagram:", e)
        print("Make sure Graphviz is properly installed.")
        return None


def save_text_report(report_text, filename=None):
    """
    Save the report as a text file.

    Args:
        report_text (str): The report text to save
        filename (str, optional): Filename to save the report as

    Returns:
        str: Path to the saved report
    """
    if not filename:
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"Portfolio_Investment_Report_{current_date}.txt"

    try:
        with open(filename, "w") as f:
            f.write(report_text)
        print(f"Text report saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving text report: {e}")
        return None
