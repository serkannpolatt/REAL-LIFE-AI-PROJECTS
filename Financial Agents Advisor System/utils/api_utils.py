"""
API Utilities
============

Helper functions for API key management and validation.
"""

import os
from typing import Optional, Dict
from dotenv import load_dotenv


class APIKeyManager:
    """Manages API keys and validates their presence."""

    def __init__(self):
        load_dotenv()
        self.required_keys = ["GROQ_API_KEY"]
        self.optional_keys = ["OPENAI_API_KEY", "ALPHA_VANTAGE_API_KEY"]

    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate the presence of required and optional API keys.

        Returns:
            Dict mapping API key names to their validation status
        """
        status = {}

        # Check required keys
        for key in self.required_keys:
            status[key] = bool(os.getenv(key))

        # Check optional keys
        for key in self.optional_keys:
            status[key] = bool(os.getenv(key))

        return status

    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get API key value safely.

        Args:
            key_name: Name of the API key

        Returns:
            API key value or None if not found
        """
        return os.getenv(key_name)

    def check_system_ready(self) -> bool:
        """
        Check if system has minimum required API keys.

        Returns:
            True if system is ready, False otherwise
        """
        status = self.validate_api_keys()
        return all(status[key] for key in self.required_keys)

    def get_status_report(self) -> str:
        """
        Get formatted status report of API keys.

        Returns:
            Formatted string with API key status
        """
        status = self.validate_api_keys()
        report = ["API Key Status Report:", "=" * 25]

        report.append("\nRequired Keys:")
        for key in self.required_keys:
            symbol = "✓" if status[key] else "✗"
            report.append(f"  {symbol} {key}")

        report.append("\nOptional Keys:")
        for key in self.optional_keys:
            symbol = "✓" if status[key] else "✗"
            report.append(f"  {symbol} {key}")

        return "\n".join(report)


def setup_environment() -> bool:
    """
    Set up environment and validate API keys.

    Returns:
        True if setup successful, False otherwise
    """
    manager = APIKeyManager()

    if not manager.check_system_ready():
        print("❌ System setup incomplete!")
        print(manager.get_status_report())
        print("\nPlease check your .env file and ensure required API keys are set.")
        return False

    print("✅ System ready!")
    print(manager.get_status_report())
    return True
