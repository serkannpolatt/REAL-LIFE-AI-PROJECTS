"""
Reporting module for FinAgents.
This module provides functionality for generating PDF reports.
"""

from fpdf import FPDF
from datetime import datetime
import os
import sys

# Add the project root directory to Python path to import other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.validation import validate_and_normalize_allocations


class PortfolioPDF(FPDF):
    """PDF class for portfolio reports"""

    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297

    def header(self):
        """Define the header for each page"""
        self.set_font("Arial", "B", 15)
        self.cell(self.WIDTH - 20, 10, "Investment Portfolio Analysis", 0, 1, "R")
        self.set_font("Arial", "I", 10)
        self.cell(
            self.WIDTH - 20,
            10,
            f"Generated on {datetime.now().strftime('%Y-%m-%d')}",
            0,
            1,
            "R",
        )
        self.line(10, 30, self.WIDTH - 10, 30)
        self.ln(20)

    def footer(self):
        """Define the footer for each page"""
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def chapter_title(self, title):
        """Add a chapter title to the PDF"""
        self.set_font("Arial", "B", 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, "L", 1)
        self.ln(5)

    def chapter_body(self, body):
        """Add chapter content to the PDF"""
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 6, body)
        self.ln(5)

    def add_image(self, image_path, w=190):
        """Add an image to the PDF"""
        self.image(image_path, x=10, w=w)
        self.ln(5)

    def add_table(self, headers, data, col_widths=None):
        """Add a table to the PDF"""
        # Calculate column widths if not provided
        if col_widths is None:
            # Ensure table fits within page margins (190mm width)
            available_width = 190
            col_widths = [available_width / len(headers)] * len(headers)

            # Adjust column widths based on content
            if len(headers) > 3:
                # For tables with many columns, make first column wider for readability
                col_widths[0] = available_width * 0.25  # 25% for first column
                remaining_width = available_width * 0.75  # 75% for remaining columns
                for i in range(1, len(headers)):
                    col_widths[i] = remaining_width / (len(headers) - 1)

        line_height = 7
        self.set_font("Arial", "B", 10)

        # Draw header row
        for i, header in enumerate(headers):
            self.cell(col_widths[i], line_height, header, 1, 0, "C")
        self.ln(line_height)

        # Draw data rows
        self.set_font("Arial", "", 10)
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], line_height, str(item), 1, 0, "C")
            self.ln(line_height)
        self.ln(5)


def create_pdf_report(
    report_content, chart_paths, portfolio_metrics, filename="Portfolio_Report.pdf"
):
    """
    Create a PDF report with portfolio analysis.

    Args:
        report_content (dict): Dictionary with report sections
        chart_paths (dict): Dictionary with paths to charts
        portfolio_metrics (dict): Dictionary with portfolio metrics
        filename (str): Output filename for the PDF

    Returns:
        str: Path to the generated PDF file
    """
    # Validate and normalize allocation recommendations
    if "allocation_recommendations" in report_content:
        report_content["allocation_recommendations"] = (
            validate_and_normalize_allocations(
                report_content["allocation_recommendations"]
            )
        )

    pdf = PortfolioPDF()

    pdf.add_page()
    pdf.chapter_title("Executive Summary")
    pdf.chapter_body(
        report_content.get("executive_summary", "No Executive Summary provided.")
    )

    pdf.add_page()
    pdf.chapter_title("Market Overview")
    pdf.chapter_body(
        report_content.get("market_overview", "No Market Overview provided.")
    )

    pdf.add_page()
    pdf.chapter_title("Portfolio Performance Analysis")
    portfolio_performance = report_content.get(
        "portfolio_performance_analysis",
        report_content.get(
            "portfolio_performance", "No Portfolio Performance Analysis provided."
        ),
    )
    pdf.chapter_body(portfolio_performance)
    if "performance" in chart_paths:
        pdf.add_image(chart_paths["performance"])
    if "cumulative" in chart_paths:
        pdf.add_image(chart_paths["cumulative"])

    pdf.chapter_title("Portfolio Metrics")
    headers = ["Metric", "Value"]
    data = [
        ["Annual Return", f"{portfolio_metrics['annual_return'] * 100:.2f}%"],
        ["Annual Volatility", f"{portfolio_metrics['annual_volatility'] * 100:.2f}%"],
        ["Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.2f}"],
        ["Maximum Drawdown", f"{portfolio_metrics['max_drawdown'] * 100:.2f}%"],
    ]
    # Use custom column widths for this table
    col_widths = [95, 95]  # Equal width for both columns
    pdf.add_table(headers, data, col_widths)

    pdf.add_page()
    pdf.chapter_title("Risk Assessment")
    pdf.chapter_body(
        report_content.get("risk_assessment", "No Risk Assessment provided.")
    )
    if "correlation" in chart_paths:
        pdf.add_image(chart_paths["correlation"])
    if "returns" in chart_paths:
        pdf.add_image(chart_paths["returns"])
    if "risk_return" in chart_paths:
        pdf.add_image(chart_paths["risk_return"])

    pdf.add_page()
    pdf.chapter_title("Individual Stock Analysis")
    headers = ["Stock", "Weight", "Return", "Volatility", "Sharpe", "Beta"]
    data = []

    # Get the list of stocks from the portfolio_metrics
    stocks = portfolio_metrics["stock_metrics"].keys()

    for stock in stocks:
        metrics = portfolio_metrics["stock_metrics"].get(stock, {})
        # Get current weights from portfolio_metrics
        current_weights = {
            s: metrics.get("weight", 0)
            for s, metrics in portfolio_metrics["stock_metrics"].items()
        }
        data.append(
            [
                stock,
                f"{current_weights.get(stock, 0) * 100:.1f}%",
                f"{metrics.get('annual_return', 0) * 100:.2f}%",
                f"{metrics.get('annual_volatility', 0) * 100:.2f}%",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                f"{metrics.get('beta', 0):.2f}",
            ]
        )
    # Use custom column widths for this table to ensure all columns fit
    col_widths = [40, 30, 30, 30, 30, 30]  # Total: 190mm (fits within page margins)
    pdf.add_table(headers, data, col_widths)

    pdf.add_page()
    pdf.chapter_title("Allocation Recommendations")
    # Use a smaller font size for this section
    pdf.set_font("Arial", "", 9)  # Decreased from 11 to 9
    allocation_text = report_content.get(
        "allocation_recommendations", "No Allocation Recommendations provided."
    )

    # Check if there's a table in the allocation recommendations
    table_start = allocation_text.find("|")
    if table_start != -1:
        # Split the text before and after the table
        before_table = allocation_text[:table_start].strip()

        # Find the end of the table (empty line after table)
        table_lines = []
        after_table = ""
        in_table = False
        for line in allocation_text[table_start:].split("\n"):
            if line.strip().startswith("|"):
                table_lines.append(line.strip())
                in_table = True
            elif in_table and not line.strip():
                # Empty line after table
                in_table = False
                after_table = line
            elif not in_table:
                after_table += line + "\n"

        # Add the text before the table
        pdf.multi_cell(0, 5, before_table)  # Decreased line height from 6 to 5

        # Parse and add the table
        if len(table_lines) >= 2:  # At least header and separator
            # Parse header
            header_cells = [cell.strip() for cell in table_lines[0].split("|")[1:-1]]

            # Skip separator line
            data_rows = []
            for i in range(2, len(table_lines)):
                cells = [cell.strip() for cell in table_lines[i].split("|")[1:-1]]
                if cells:
                    data_rows.append(cells)

            # Add the table with dynamic column widths
            num_columns = len(header_cells)
            if num_columns > 0:
                # Calculate column widths based on number of columns
                available_width = 190  # Total available width in mm
                col_widths = [available_width / num_columns] * num_columns

                # Adjust first column to be wider if there are multiple columns
                if num_columns > 2:
                    col_widths[0] = available_width * 0.4  # 40% for first column
                    remaining_width = available_width * 0.6  # 60% for remaining columns
                    for i in range(1, num_columns):
                        col_widths[i] = remaining_width / (num_columns - 1)

                # Use smaller font for table
                pdf.set_font("Arial", "B", 8)  # Header font
                line_height = 6  # Decreased from 7

                # Draw header row
                for i, header in enumerate(header_cells):
                    pdf.cell(col_widths[i], line_height, header, 1, 0, "C")
                pdf.ln(line_height)

                # Draw data rows
                pdf.set_font("Arial", "", 8)  # Data font
                for row in data_rows:
                    for i, item in enumerate(row):
                        if i < len(col_widths):  # Ensure we don't exceed column widths
                            pdf.cell(col_widths[i], line_height, str(item), 1, 0, "C")
                    pdf.ln(line_height)
                pdf.ln(3)  # Decreased from 5

        # Add the text after the table
        pdf.set_font("Arial", "", 9)  # Reset to section font size
        pdf.multi_cell(0, 5, after_table)  # Decreased line height
    else:
        # No table found, just add the text
        pdf.multi_cell(0, 5, allocation_text)  # Decreased line height

    if "allocation" in chart_paths:
        pdf.add_image(chart_paths["allocation"])

    # Reset font size for next sections
    pdf.set_font("Arial", "", 11)

    pdf.add_page()
    pdf.chapter_title("Implementation Strategy")
    pdf.chapter_body(
        report_content.get(
            "implementation_strategy", "No Implementation Strategy provided."
        )
    )

    pdf.add_page()
    pdf.chapter_title("Future Outlook")
    pdf.chapter_body(
        report_content.get("future_outlook", "No Future Outlook provided.")
    )

    pdf.add_page()
    pdf.chapter_title("Conclusion")
    pdf.chapter_body(report_content.get("conclusion", "No Conclusion provided."))

    pdf.output(filename)
    print(f"PDF report saved as {filename}")
    return filename
