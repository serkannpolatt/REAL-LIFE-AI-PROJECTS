#!/usr/bin/env python3
"""
Doc-MCP Application Launcher

A production-ready documentation RAG system that transforms GitHub repositories
into intelligent, queryable knowledge bases using Vector Search and MCP integration.

Features:
- Semantic search across documentation
- AI-powered Q&A with source citations
- MCP server integration for AI agents
- Repository management and incremental updates
- MongoDB Atlas Vector Search backend
"""

import logging
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ui.main import main


def setup_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("doc_mcp.log", mode="a"),
        ],
    )


if __name__ == "__main__":
    try:
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting Doc-MCP Application...")
        main()
    except KeyboardInterrupt:
        logger.info("Application shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)
