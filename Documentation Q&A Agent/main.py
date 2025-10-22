#!/usr/bin/env python3
"""
Documentation Q&A Agent

An intelligent Streamlit application that enables conversational interaction with any documentation
using Model Context Protocol (MCP) integration and Nebius AI. Transforms static documentation
into an interactive knowledge base with real-time Q&A capabilities.

Features:
- Natural language queries against documentation
- MCP-powered tool integration for seamless data access
- Real-time streaming responses with contextual understanding
- Secure API key management and session handling
- Example-driven interface for quick onboarding
"""

import asyncio
import logging
import os
import sys

import streamlit as st
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.nebius import Nebius
from agno.tools.mcp import MCPTools

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Application constants
DEFAULT_SERVER_URL = "https://docs.studio.nebius.com/mcp"
APP_TITLE = "Talk to Your Docs"
APP_ICON = "📚"

# Example questions for user guidance
EXAMPLE_QUESTIONS = [
    "How to create an Agent with Google ADK & Nebius?",
    "How to fine-tune your custom model?",
    "How to get structured output from our text models?",
    "How to use the Nebius API for embeddings?",
    "What are the authentication requirements?",
    "How to implement streaming responses?",
]


class DocumentationQnAAgent:
    """Main application class for Documentation Q&A functionality."""

    def __init__(self):
        """Initialize the application with default settings."""
        self.setup_page_config()
        self.initialize_session_state()

    def setup_page_config(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon=APP_ICON,
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def initialize_session_state(self) -> None:
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "api_key_validated" not in st.session_state:
            st.session_state.api_key_validated = False
        if "last_doc_url" not in st.session_state:
            st.session_state.last_doc_url = DEFAULT_SERVER_URL

    async def run_mcp_agent(self, url: str, query: str, api_key: str) -> str:
        """
        Execute MCP agent query with enhanced error handling.

        Args:
            url: Documentation MCP server URL
            query: User query to process
            api_key: Nebius API key for authentication

        Returns:
            AI-generated response content

        Raises:
            Exception: For connection, authentication, or processing errors
        """
        mcp_tools = None
        try:
            # Validate inputs
            if not api_key.strip():
                raise ValueError("API key is required")
            if not url.strip():
                raise ValueError("Documentation URL is required")
            if not query.strip():
                raise ValueError("Query cannot be empty")

            # Initialize MCP connection
            logger.info(f"Connecting to MCP server: {url}")
            mcp_tools = MCPTools(url=url, transport="streamable-http")
            await mcp_tools.connect()

            # Create agent with optimized settings
            agent = Agent(
                model=Nebius(
                    id="deepseek-ai/DeepSeek-V3-0324",
                    api_key=api_key,
                    temperature=0.1,  # Low temperature for factual responses
                    max_tokens=4096,  # Increased for detailed responses
                ),
                tools=[mcp_tools],
                instructions="""
                You are a helpful documentation assistant. Provide accurate, detailed responses
                based on the documentation content. Always:
                - Cite sources when possible
                - Provide code examples when relevant
                - Break down complex concepts into clear steps
                - Suggest related topics that might be helpful
                - If information is not available, clearly state this
                """,
                show_tool_calls=False,  # Cleaner user experience
                markdown=True,
            )

            # Execute query
            logger.info(f"Processing query: {query[:50]}...")
            response = await agent.arun(query)
            return response.content

        except Exception as e:
            logger.error(f"MCP agent error: {str(e)}")
            raise
        finally:
            # Ensure cleanup
            if mcp_tools:
                try:
                    await mcp_tools.close()
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup error: {cleanup_error}")

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate API key format and set validation status.

        Args:
            api_key: The API key to validate

        Returns:
            True if key appears valid, False otherwise
        """
        if not api_key or len(api_key) < 10:  # Basic validation
            st.session_state.api_key_validated = False
            return False

        st.session_state.api_key_validated = True
        return True

    def render_sidebar(self) -> tuple[str, str]:
        """
        Render sidebar with configuration options.

        Returns:
            Tuple of (api_key, doc_url)
        """
        with st.sidebar:
            # Logo and branding
            try:
                st.image("./assets/Nebius.png", width=150)
            except FileNotFoundError:
                st.markdown("### 🤖 Nebius AI")

            st.markdown("### ⚙️ Configuration")

            # API Key input with validation
            api_key = st.text_input(
                "Nebius API Key",
                value=os.getenv("NEBIUS_API_KEY", ""),
                type="password",
                help="Enter your Nebius API key for AI model access",
                placeholder="Enter your API key...",
            )

            # Visual validation feedback
            if api_key:
                if self.validate_api_key(api_key):
                    st.success("✅ API key format valid")
                else:
                    st.error("❌ Invalid API key format")

            st.divider()

            # Documentation URL configuration
            doc_url = st.text_input(
                "Documentation MCP Server URL",
                value=st.session_state.last_doc_url,
                help="URL of the documentation MCP server to query",
                placeholder="https://docs.example.com/mcp",
            )

            # Update session state
            if doc_url != st.session_state.last_doc_url:
                st.session_state.last_doc_url = doc_url

            # Server connection test
            if st.button("🔍 Test Connection", help="Test connection to MCP server"):
                with st.spinner("Testing connection..."):
                    # Simple URL validation
                    if doc_url.startswith(("http://", "https://")):
                        st.success("✅ URL format valid")
                    else:
                        st.error("❌ Invalid URL format")

            st.divider()

            # Example questions
            st.markdown("### 💡 Example Questions")
            st.markdown("*Click any question to try it:*")

            for i, question in enumerate(EXAMPLE_QUESTIONS):
                if st.button(
                    question,
                    key=f"example_{i}",
                    help=f"Ask: {question}",
                    use_container_width=True,
                ):
                    st.session_state.messages.append(
                        {"role": "user", "content": question}
                    )
                    st.rerun()

            st.divider()

            # Usage statistics
            st.markdown("### 📊 Session Stats")
            message_count = len(st.session_state.messages)
            st.metric("Messages", message_count)
            if message_count > 0:
                user_messages = len(
                    [m for m in st.session_state.messages if m["role"] == "user"]
                )
                st.metric("Questions Asked", user_messages)

        return api_key, doc_url

    def render_header(self) -> None:
        """Render application header with title and controls."""
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 10px;">
                    <h1 style="margin: 0;">{APP_ICON} {APP_TITLE}</h1>
                </div>
                <p style="color: #666; margin-top: 5px;">
                    Intelligent Q&A powered by MCP and Nebius AI
                </p>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            if st.button("🗑️ Clear Chat", help="Clear conversation history"):
                st.session_state.messages = []
                st.success("Chat cleared!")
                st.rerun()

    def render_chat_interface(self, api_key: str, doc_url: str) -> None:
        """
        Render the main chat interface.

        Args:
            api_key: Validated API key
            doc_url: Documentation server URL
        """
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input(
            "Ask a question about the documentation...",
            disabled=not st.session_state.api_key_validated,
        ):
            self.handle_user_input(prompt, api_key, doc_url)

        # Handle example questions
        self.handle_pending_user_message(api_key, doc_url)

    def handle_user_input(self, prompt: str, api_key: str, doc_url: str) -> None:
        """
        Process user input and generate response.

        Args:
            prompt: User's question/input
            api_key: Nebius API key
            doc_url: Documentation server URL
        """
        if not self.validate_api_key(api_key):
            st.error("❌ Please enter a valid Nebius API key in the sidebar.")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            self.generate_assistant_response(prompt, api_key, doc_url)

    def handle_pending_user_message(self, api_key: str, doc_url: str) -> None:
        """Handle messages added via example buttons."""
        if st.session_state.messages:
            last_message = st.session_state.messages[-1]
            if last_message["role"] == "user" and (
                len(st.session_state.messages) == 1
                or st.session_state.messages[-2]["role"] != "assistant"
            ):
                with st.chat_message("user"):
                    st.markdown(last_message["content"])

                with st.chat_message("assistant"):
                    self.generate_assistant_response(
                        last_message["content"], api_key, doc_url
                    )

    def generate_assistant_response(
        self, prompt: str, api_key: str, doc_url: str
    ) -> None:
        """
        Generate and display assistant response with comprehensive error handling.

        Args:
            prompt: User's question
            api_key: Nebius API key
            doc_url: Documentation server URL
        """
        try:
            with st.spinner("🤔 Analyzing documentation..."):
                # Show processing status
                status_placeholder = st.empty()
                status_placeholder.info("🔌 Connecting to documentation server...")

                # Execute agent query
                response_text = asyncio.run(
                    self.run_mcp_agent(doc_url, prompt, api_key)
                )

                status_placeholder.empty()

                if response_text and response_text.strip():
                    st.markdown(response_text)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )

                    # Add helpful follow-up suggestions
                    with st.expander("💡 Related Questions", expanded=False):
                        st.markdown("""
                        - Can you provide more specific examples?
                        - Are there any prerequisites I should know about?
                        - How does this integrate with other features?
                        - What are the common troubleshooting steps?
                        """)
                else:
                    error_msg = (
                        "No response received. Please try rephrasing your question."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

        except ValueError as ve:
            error_msg = f"❌ Input Error: {str(ve)}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

        except ConnectionError:
            error_msg = "🌐 Connection Error: Unable to reach documentation server. Please check the URL and try again."
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

        except Exception as e:
            error_msg = f"⚠️ Unexpected error: {str(e)}"
            st.error(error_msg)
            logger.error(f"Assistant response error: {e}")
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )

            # Provide troubleshooting guidance
            with st.expander("🔧 Troubleshooting Tips", expanded=True):
                st.markdown("""
                **Common solutions:**
                - Verify your API key is correct and active
                - Check that the documentation URL is accessible
                - Try a simpler question to test connectivity
                - Refresh the page and try again
                - Contact support if the issue persists
                """)

    def run(self) -> None:
        """Main application entry point."""
        try:
            logger.info("Starting Documentation Q&A Agent")

            # Render UI components
            self.render_header()
            api_key, doc_url = self.render_sidebar()

            # Main chat interface
            if not api_key:
                st.info(
                    "👈 Please enter your Nebius API key in the sidebar to get started."
                )
                st.markdown("""
                ### 🚀 Getting Started
                
                1. **Get API Key**: Sign up at [Nebius Studio](https://studio.nebius.ai/)
                2. **Enter Credentials**: Add your API key in the sidebar
                3. **Select Documentation**: Choose or enter documentation URL
                4. **Start Asking**: Use example questions or type your own
                
                ### ✨ Features
                - **Natural Language Queries** - Ask questions in plain English
                - **Real-time Responses** - Get instant, contextual answers
                - **Source Citations** - Responses include documentation references
                - **Example Questions** - Quick start with pre-built queries
                """)
            else:
                self.render_chat_interface(api_key, doc_url)

        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error(f"Application error: {str(e)}")


def main():
    """Application entry point with error handling."""
    try:
        app = DocumentationQnAAgent()
        app.run()
    except KeyboardInterrupt:
        logger.info("Application shutdown by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        st.error(f"Failed to start application: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
