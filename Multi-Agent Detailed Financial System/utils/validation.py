"""
Validation module for FinAgents.
This module provides functions for validating portfolio allocations.
"""

import re


def validate_and_normalize_allocations(allocation_text):
    """
    Validates and normalizes portfolio allocations to ensure they sum to exactly 100%.

    Args:
        allocation_text (str): The allocation recommendations text containing percentages

    Returns:
        str: Updated allocation text with normalized percentages
    """
    print("\n=== VALIDATING ALLOCATIONS ===")
    print(f"Original text length: {len(allocation_text)} characters")

    # Check if there's a table in the allocation recommendations
    table_start = allocation_text.find("|")
    if table_start != -1:
        print("Found a table in the allocation recommendations")

        # Extract the table
        table_lines = []
        in_table = False
        for line in allocation_text[table_start:].split("\n"):
            if line.strip().startswith("|"):
                table_lines.append(line.strip())
                in_table = True
            elif in_table and not line.strip():
                # Empty line after table
                break

        if len(table_lines) >= 3:  # At least header, separator, and one data row
            print(f"Found {len(table_lines)} table lines")

            # Parse the table to extract allocations
            allocations = {}
            for i in range(2, len(table_lines)):  # Skip header and separator
                cells = [cell.strip() for cell in table_lines[i].split("|")[1:-1]]
                if len(cells) >= 3:
                    asset = cells[0].strip()
                    if asset and "%" in cells[2]:
                        percentage_str = cells[2].replace("%", "").strip()
                        try:
                            percentage = float(percentage_str)
                            allocations[asset] = percentage
                            print(f"  Found in table: {asset} = {percentage}%")
                        except ValueError:
                            print(f"  Error parsing percentage: {percentage_str}")

            # Calculate total allocation
            total_allocation = sum(allocations.values())
            print(f"\nOriginal total allocation from table: {total_allocation}%")
            print("Original allocations from table:", allocations)

            if (
                abs(total_allocation - 100) > 0.01
            ):  # Allow for small floating point differences
                print(
                    f"\nNeed to normalize allocations to 100% (current: {total_allocation}%)"
                )
                # Normalize allocations to sum to 100%
                scaling_factor = 100 / total_allocation
                print(f"Scaling factor: {scaling_factor}")

                # Update allocations
                normalized_allocations = {}
                for asset in allocations:
                    # Scale and round to nearest integer or one decimal place
                    normalized_allocations[asset] = (
                        round(allocations[asset] * scaling_factor * 10) / 10
                    )

                # Ensure the sum is exactly 100% after rounding
                total_after_scaling = sum(normalized_allocations.values())
                if abs(total_after_scaling - 100) > 0.01:
                    # Adjust the largest allocation to make the sum exactly 100%
                    largest_asset = max(
                        normalized_allocations.items(), key=lambda x: x[1]
                    )[0]
                    adjustment = round((100 - total_after_scaling) * 10) / 10
                    print(
                        f"Adjusting largest asset ({largest_asset}) by {adjustment}% to make total exactly 100%"
                    )
                    normalized_allocations[largest_asset] += adjustment

                print(
                    f"Normalized total allocation: {sum(normalized_allocations.values())}%"
                )
                print("Normalized allocations:", normalized_allocations)

                # Update the table with normalized percentages
                print("\nUpdating table with normalized percentages...")
                updated_table_lines = table_lines[:2]  # Keep header and separator

                for i in range(2, len(table_lines)):
                    line = table_lines[i]
                    cells = [cell.strip() for cell in line.split("|")[1:-1]]
                    if len(cells) >= 3:
                        asset = cells[0].strip()
                        if asset in normalized_allocations:
                            percentage = normalized_allocations[asset]
                            percentage_str = (
                                f"{int(percentage)}%"
                                if percentage.is_integer()
                                else f"{percentage:.1f}%"
                            )
                            cells[2] = percentage_str
                            updated_line = "| " + " | ".join(cells) + " |"
                            print(f"  Updated: {updated_line}")
                            updated_table_lines.append(updated_line)
                        else:
                            updated_table_lines.append(line)
                    else:
                        updated_table_lines.append(line)

                # Replace the table in the allocation text
                updated_table = "\n".join(updated_table_lines)
                allocation_text = (
                    allocation_text[:table_start]
                    + updated_table
                    + allocation_text[table_start + len("\n".join(table_lines)) :]
                )

                print("Table updated successfully")
                return allocation_text

    # If no table found or table processing failed, use the regex approach
    # Extract all percentage allocations using regex
    # Main pattern for allocations like "AAPL: 15%" or "AGG (Bonds): 15%"
    percentage_pattern = r"(\w+(?:\s+\([^)]+\))?)\s*:\s*(\d+(?:\.\d+)?)%"

    # Pattern for nested allocations like "JNJ 3%" within category sections
    nested_percentage_pattern = r"(\w+)\s+(\d+(?:\.\d+)?)%"

    # Pattern for markdown table rows like "| AAPL | 20% | 15% |"
    table_row_pattern = (
        r"\|\s*([^|]+)\s*\|\s*\d+(?:\.\d+)?%\s*\|\s*(\d+(?:\.\d+)?)%\s*\|"
    )

    allocations = {}

    # List of phrases to exclude from being considered as assets
    excluded_phrases = ["exactly", "precisely", "total", "sum", "allocation"]

    # Find all main allocations (e.g., "AAPL: 15%")
    print("\nSearching for main allocations (e.g., 'AAPL: 15%'):")
    for match in re.finditer(percentage_pattern, allocation_text):
        asset, percentage = match.groups()
        asset = asset.strip()

        # Skip excluded phrases
        if any(phrase in asset.lower() for phrase in excluded_phrases):
            print(f"  Skipping excluded phrase: {asset}")
            continue

        print(f"  Found: {asset} = {percentage}%")
        allocations[asset] = float(percentage)

    # Find all nested allocations (e.g., "JNJ 3%")
    print("\nSearching for nested allocations (e.g., 'JNJ 3%'):")
    for match in re.finditer(nested_percentage_pattern, allocation_text):
        asset, percentage = match.groups()
        asset = asset.strip()

        # Skip excluded phrases
        if any(phrase in asset.lower() for phrase in excluded_phrases):
            print(f"  Skipping excluded phrase: {asset}")
            continue

        if asset not in allocations:  # Avoid duplicates
            print(f"  Found: {asset} = {percentage}%")
            allocations[asset] = float(percentage)

    # Find all table row allocations
    print("\nSearching for table row allocations (e.g., '| AAPL | ... | 15% |'):")
    for match in re.finditer(table_row_pattern, allocation_text):
        asset, percentage = match.groups()
        asset = asset.strip()

        # Skip excluded phrases
        if any(phrase in asset.lower() for phrase in excluded_phrases):
            print(f"  Skipping excluded phrase: {asset}")
            continue

        if asset not in allocations:  # Avoid duplicates
            print(f"  Found: {asset} = {percentage}%")
            allocations[asset] = float(percentage)

    # Calculate total allocation
    total_allocation = sum(allocations.values())
    print(f"\nOriginal total allocation: {total_allocation}%")
    print("Original allocations:", allocations)

    if abs(total_allocation - 100) > 0.01:  # Allow for small floating point differences
        print(f"\nNeed to normalize allocations to 100% (current: {total_allocation}%)")
        # Normalize allocations to sum to 100%
        scaling_factor = 100 / total_allocation
        print(f"Scaling factor: {scaling_factor}")

        # Update allocations
        normalized_allocations = {}
        for asset in allocations:
            # Scale and round to nearest integer or one decimal place
            normalized_allocations[asset] = (
                round(allocations[asset] * scaling_factor * 10) / 10
            )

        # Ensure the sum is exactly 100% after rounding
        total_after_scaling = sum(normalized_allocations.values())
        if abs(total_after_scaling - 100) > 0.01:
            # Adjust the largest allocation to make the sum exactly 100%
            largest_asset = max(normalized_allocations.items(), key=lambda x: x[1])[0]
            adjustment = round((100 - total_after_scaling) * 10) / 10
            print(
                f"Adjusting largest asset ({largest_asset}) by {adjustment}% to make total exactly 100%"
            )
            normalized_allocations[largest_asset] += adjustment

        print(f"Normalized total allocation: {sum(normalized_allocations.values())}%")
        print("Normalized allocations:", normalized_allocations)

        # Update the allocation text with normalized percentages
        print("\nUpdating allocation text with normalized percentages...")

        # Replace main allocations
        for asset, percentage in normalized_allocations.items():
            # Format percentage with one decimal place if it's not a whole number
            percentage_str = (
                f"{int(percentage)}%"
                if percentage.is_integer()
                else f"{percentage:.1f}%"
            )

            # Replace in main pattern
            pattern = rf"{re.escape(asset)}\s*:\s*\d+(?:\.\d+)?%"
            if re.search(pattern, allocation_text):
                print(f"  Replacing main allocation: {asset}: {percentage_str}")
                allocation_text = re.sub(
                    pattern, f"{asset}: {percentage_str}", allocation_text
                )

            # Replace in nested pattern
            pattern = rf"{re.escape(asset)}\s+\d+(?:\.\d+)?%"
            if re.search(pattern, allocation_text):
                print(f"  Replacing nested allocation: {asset} {percentage_str}")
                allocation_text = re.sub(
                    pattern, f"{asset} {percentage_str}", allocation_text
                )

            # Replace in table rows
            pattern = rf"\|\s*{re.escape(asset)}\s*\|\s*\d+(?:\.\d+)?%\s*\|\s*\d+(?:\.\d+)?%\s*\|"
            if re.search(pattern, allocation_text):
                print(f"  Replacing table row: | {asset} | x% | {percentage_str} |")

                # Preserve the middle column content
                def replace_table_row(match):
                    full_match = match.group(0)
                    parts = full_match.split("|")
                    if len(parts) >= 5:
                        parts[0] = "|"
                        parts[1] = f" {asset} "
                        # Keep parts[2] (current allocation) unchanged
                        parts[3] = f" {percentage_str} "
                        parts[4] = "|"
                        return "|".join(parts[:5])
                    return full_match

                allocation_text = re.sub(pattern, replace_table_row, allocation_text)

    print("\n=== VALIDATION COMPLETE ===")
    return allocation_text
