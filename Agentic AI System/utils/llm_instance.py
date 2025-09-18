"""
LLM Instance - Groq Language Model Configuration
===============================================

Configures and initializes the Groq LLM instance for the application.
Handles API key management and model settings.
"""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_groq_api_key() -> str:
    """
    Get Groq API key from environment variables

    Returns:
        str: API key

    Raises:
        ValueError: If API key is not found
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "❌ GROQ_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )

    return api_key


def initialize_llm():
    """
    Initialize and configure the Groq LLM instance

    Returns:
        ChatGroq: Configured LLM instance
    """
    try:
        api_key = get_groq_api_key()

        llm = ChatGroq(
            model="llama-3.1-70b-versatile",  # Updated to a valid Groq model
            temperature=0,
            groq_api_key=api_key,
            max_tokens=1024,
            timeout=30,
        )

        return llm

    except Exception as e:
        print(f"❌ Error initializing LLM: {e}")
        print("💡 Make sure you have set GROQ_API_KEY in your environment variables")
        raise


# Initialize the global LLM instance
try:
    llm = initialize_llm()
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize LLM: {e}")
    # Create a dummy LLM for development/testing
    llm = None
