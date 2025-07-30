"""
Logging Utilities
================

Centralized logging configuration for the Financial AI Agent system.
"""

import logging
import sys
from datetime import datetime
from typing import Optional


class FinancialAgentLogger:
    """Centralized logger for the Financial AI Agent system."""

    def __init__(self, name: str = "FinancialAgent", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(log_file)

    def _setup_handlers(self, log_file: Optional[str] = None):
        """Set up logging handlers for console and file output."""

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file is None:
            log_file = f"finance_agent_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)

    def log_agent_action(self, agent_name: str, action: str, details: str = ""):
        """Log agent-specific actions."""
        message = f"[{agent_name}] {action}"
        if details:
            message += f" - {details}"
        self.info(message)

    def log_api_call(self, api_name: str, endpoint: str, status: str):
        """Log API call information."""
        self.info(f"API Call - {api_name}:{endpoint} - Status: {status}")

    def log_analysis_start(self, symbol: str, analysis_type: str):
        """Log start of analysis."""
        self.info(f"Starting {analysis_type} analysis for {symbol}")

    def log_analysis_complete(self, symbol: str, analysis_type: str, duration: float):
        """Log completion of analysis."""
        self.info(f"Completed {analysis_type} analysis for {symbol} in {duration:.2f}s")

    def log_error_with_context(self, error: Exception, context: str):
        """Log error with additional context."""
        self.error(f"Error in {context}: {type(error).__name__}: {str(error)}")


# Global logger instance
_global_logger = None


def get_logger(name: str = "FinancialAgent") -> FinancialAgentLogger:
    """
    Get or create global logger instance.

    Args:
        name: Logger name

    Returns:
        FinancialAgentLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = FinancialAgentLogger(name)
    return _global_logger


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration for the entire application.

    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Get global logger
    logger = get_logger()
    logger.info(f"Logging initialized - Level: {log_level}")

    return logger


class LoggingMixin:
    """Mixin class to add logging capabilities to any class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(self.__class__.__name__)

    def log_info(self, message: str):
        """Log info message with class context."""
        self._logger.info(f"[{self.__class__.__name__}] {message}")

    def log_debug(self, message: str):
        """Log debug message with class context."""
        self._logger.debug(f"[{self.__class__.__name__}] {message}")

    def log_warning(self, message: str):
        """Log warning message with class context."""
        self._logger.warning(f"[{self.__class__.__name__}] {message}")

    def log_error(self, message: str):
        """Log error message with class context."""
        self._logger.error(f"[{self.__class__.__name__}] {message}")

    def log_method_entry(self, method_name: str, **kwargs):
        """Log method entry with parameters."""
        params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.log_debug(f"Entering {method_name}({params})")

    def log_method_exit(self, method_name: str, result=None):
        """Log method exit with result."""
        if result is not None:
            self.log_debug(f"Exiting {method_name} - Result: {type(result).__name__}")
        else:
            self.log_debug(f"Exiting {method_name}")


def log_performance(func):
    """Decorator to log function performance."""

    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger = get_logger()

        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"Function {func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Function {func.__name__} failed after {duration:.3f}s: {e}")
            raise

    return wrapper
