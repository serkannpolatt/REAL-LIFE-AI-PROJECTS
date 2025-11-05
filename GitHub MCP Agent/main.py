#!/usr/bin/env python3
"""
GitHub MCP Agent

An advanced GitHub repository exploration tool that leverages the Model Context Protocol (MCP)
and Nebius AI to provide intelligent, natural language interactions with GitHub repositories.
This application transforms complex GitHub API operations into conversational queries.

Features:
- Natural language GitHub repository queries
- Comprehensive repository analysis (issues, PRs, activity, code quality)
- Real-time data retrieval with proper error handling
- Interactive Streamlit interface with organized results
- Secure credential management and API rate limiting
"""

import asyncio
import base64
import logging
import os
import sys
from datetime import datetime
from textwrap import dedent
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.nebius import Nebius
from agno.tools.mcp import MCPTools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Application constants
APP_TITLE = "GitHub MCP Agent"
APP_ICON = "🦑"
DEFAULT_REPO = "Arindam200/awesome-ai-apps"

# Query type configurations
QUERY_TYPES = {
    "Info": {
        "description": "Get comprehensive repository information from README.md",
        "template": "Tell me all about {repo} - provide overview, features, and key information",
    },
    "Issues": {
        "description": "Explore recent issues and bug reports",
        "template": "Find and analyze recent issues in {repo}, including open and recently closed issues",
    },
    "Pull Requests": {
        "description": "View recent merged pull requests",
        "template": "Show me recent merged PRs in {repo} with details about changes and contributors",
    },
    "Repository Activity": {
        "description": "Analyze code quality trends and repository metrics",
        "template": "Analyze code quality trends, commit activity, and repository health metrics for {repo}",
    },
    "Custom": {
        "description": "Ask any specific questions about the repository",
        "template": "",
    },
}


class GitHubMCPAgent:
    """Main application class for GitHub MCP Agent functionality."""

    def __init__(self):
        """Initialize the GitHub MCP Agent application."""
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
        if "credentials_validated" not in st.session_state:
            st.session_state.credentials_validated = False
        if "last_query_result" not in st.session_state:
            st.session_state.last_query_result = None
        if "query_history" not in st.session_state:
            st.session_state.query_history = []

    def load_logo(self) -> str:
        """Load and encode logo image."""
        try:
            with open("./assets/agno.png", "rb") as file:
                return base64.b64encode(file.read()).decode()
        except FileNotFoundError:
            logger.warning("Logo file not found: ./assets/agno.png")
            return ""

    def render_header(self) -> None:
        """Render application header with logo and title."""
        agno_base64 = self.load_logo()

        if agno_base64:
            title_html = f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                <h1 style="margin: 0; padding: 0;">
                {APP_TITLE} with 
                <img src="data:image/png;base64,{agno_base64}" 
                     style="height: 70px; margin: 0; padding: 0;"/> 
                </h1>
            </div>
            """
        else:
            title_html = f"<h1>{APP_ICON} {APP_TITLE}</h1>"

        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(
            "**Explore GitHub repositories with natural language using the Model Context Protocol**"
        )

    def render_sidebar(self) -> tuple[str, str]:
        """
        Render sidebar with authentication and configuration.

        Returns:
            Tuple of (nebius_api_key, github_token)
        """
        with st.sidebar:
            # Nebius branding
            try:
                st.image("./assets/Nebius.png", width=150)
            except FileNotFoundError:
                st.markdown("### 🤖 Nebius AI")

            st.markdown("### 🔑 Authentication")

            # API key inputs with validation
            api_key = st.text_input(
                "Nebius API Key",
                type="password",
                help="Get your API key from Nebius Studio",
                placeholder="Enter your Nebius API key",
            )

            github_token = st.text_input(
                "GitHub Personal Access Token",
                type="password",
                help="Create a token with repo scope at github.com/settings/tokens",
                placeholder="Enter your GitHub token",
            )

            # Validation feedback
            if api_key and github_token:
                if self.validate_credentials(api_key, github_token):
                    st.success("✅ Credentials validated")
                else:
                    st.warning("⚠️ Please verify your credentials")
            elif api_key or github_token:
                st.info("ℹ️ Both credentials required")

            # Set environment variables
            if api_key:
                os.environ["NEBIUS_API_KEY"] = api_key
            if github_token:
                os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = github_token

            st.divider()

            # Configuration options
            st.markdown("### ⚙️ Configuration")

            model_choice = st.selectbox(
                "AI Model",
                [
                    "Qwen/Qwen3-30B-A3B",
                    "deepseek-ai/DeepSeek-V3-0324",
                    "meta-llama/Llama-3.3-70B-Instruct",
                ],
                help="Select the AI model for processing queries",
            )

            include_links = st.checkbox(
                "Include Direct Links",
                value=True,
                help="Add GitHub links to issues and PRs in responses",
            )

            detailed_analysis = st.checkbox(
                "Detailed Analysis Mode",
                value=False,
                help="Provide more comprehensive analysis (slower)",
            )

            st.divider()

            # Usage statistics
            st.markdown("### 📊 Session Stats")
            query_count = len(st.session_state.query_history)
            st.metric("Queries Executed", query_count)

            if query_count > 0:
                last_query_time = st.session_state.query_history[-1].get(
                    "timestamp", "Unknown"
                )
                st.metric("Last Query", last_query_time)

        return api_key, github_token, model_choice, include_links, detailed_analysis

    def validate_credentials(self, api_key: str, github_token: str) -> bool:
        """
        Validate API credentials format.

        Args:
            api_key: Nebius API key
            github_token: GitHub personal access token

        Returns:
            True if credentials appear valid
        """
        # Basic validation - in production, consider API calls for verification
        api_key_valid = len(api_key) > 20 if api_key else False
        github_token_valid = len(github_token) > 30 if github_token else False

        is_valid = api_key_valid and github_token_valid
        st.session_state.credentials_validated = is_valid

        return is_valid

    def render_query_interface(self) -> tuple[str, str, str]:
        """
        Render the main query interface.

        Returns:
            Tuple of (repository, query_type, query_text)
        """
        st.markdown("### 🔍 Repository Query Interface")

        # Repository and query type selection
        col1, col2 = st.columns([3, 1])

        with col1:
            repository = st.text_input(
                "📁 Repository",
                value=DEFAULT_REPO,
                help="Format: owner/repository (e.g., microsoft/vscode)",
                placeholder="owner/repository",
            )

        with col2:
            query_type = st.selectbox(
                "Query Type",
                list(QUERY_TYPES.keys()),
                help="Select the type of information you want to retrieve",
            )

        # Query input with smart templates
        query_config = QUERY_TYPES[query_type]
        template = (
            query_config["template"].format(repo=repository) if repository else ""
        )

        st.markdown(f"**{query_config['description']}**")

        query_text = st.text_area(
            "🗣️ Your Query",
            value=template,
            height=120,
            help="Describe what you want to know about this repository",
            placeholder="What would you like to know about this repository?",
        )

        return repository, query_type, query_text

    async def run_github_agent(
        self,
        query: str,
        model_choice: str,
        include_links: bool = True,
        detailed_analysis: bool = False,
    ) -> str:
        """
        Execute GitHub agent query with comprehensive error handling.

        Args:
            query: The user query to process
            model_choice: Selected AI model
            include_links: Whether to include GitHub links
            detailed_analysis: Whether to perform detailed analysis

        Returns:
            Formatted response from the GitHub agent

        Raises:
            Various exceptions for different error conditions
        """
        # Validate prerequisites
        if not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"):
            raise ValueError("GitHub token not provided")
        if not os.getenv("NEBIUS_API_KEY"):
            raise ValueError("Nebius API key not provided")

        try:
            # Configure Docker server parameters
            server_params = StdioServerParameters(
                command="docker",
                args=[
                    "run",
                    "-i",
                    "--rm",
                    "-e",
                    "GITHUB_PERSONAL_ACCESS_TOKEN",
                    "ghcr.io/github/github-mcp-server",
                ],
                env={
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv(
                        "GITHUB_PERSONAL_ACCESS_TOKEN"
                    )
                },
            )

            # Create MCP client connection
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize MCP tools
                    mcp_tools = MCPTools(session=session)
                    await mcp_tools.initialize()

                    # Configure agent instructions
                    instructions = self.build_agent_instructions(
                        include_links, detailed_analysis
                    )

                    # Create and configure agent
                    agent = Agent(
                        tools=[mcp_tools],
                        instructions=instructions,
                        markdown=True,
                        show_tool_calls=True,
                        model=Nebius(
                            id=model_choice,
                            api_key=os.getenv("NEBIUS_API_KEY"),
                            temperature=0.2,  # Balanced creativity for factual responses
                            max_tokens=4096,
                        ),
                    )

                    # Execute query
                    logger.info(f"Executing query: {query[:50]}...")
                    response = await agent.arun(query)
                    return response.content

        except Exception as e:
            logger.error(f"GitHub agent execution error: {e}")
            raise

    def build_agent_instructions(
        self, include_links: bool, detailed_analysis: bool
    ) -> str:
        """
        Build dynamic agent instructions based on configuration.

        Args:
            include_links: Whether to include GitHub links
            detailed_analysis: Whether to perform detailed analysis

        Returns:
            Formatted instruction string
        """
        base_instructions = dedent("""\
            You are an expert GitHub assistant specializing in repository analysis and exploration.
            
            **Core Responsibilities:**
            - Provide organized, factual insights about GitHub repositories
            - Use GitHub API data to deliver accurate, up-to-date information
            - Present information in clear, structured formats
            
            **Response Guidelines:**
            - Use markdown formatting for better readability
            - Present numerical data in tables when appropriate
            - Focus on facts and verified data from the GitHub API
            - For repository info: Extract details from README.md and repository metadata
            - For issues: Provide issue details, status, labels, and activity
            - For pull requests: Include PR details, changes, and merge information
            - For activity analysis: Show commit patterns, contributor activity, and trends
        """)

        if include_links:
            base_instructions += dedent("""\
                
                **Link Requirements:**
                - Include direct GitHub links for all issues and pull requests
                - Format links as: [#123 Issue Title](https://github.com/owner/repo/issues/123)
                - Add "🔗 View on GitHub" links for repositories and major sections
            """)

        if detailed_analysis:
            base_instructions += dedent("""\
                
                **Detailed Analysis Mode:**
                - Provide comprehensive analysis with deeper insights
                - Include code quality metrics when available
                - Analyze contributor patterns and collaboration trends
                - Examine issue and PR lifecycle patterns
                - Consider repository health indicators
            """)

        base_instructions += dedent("""\
            
            **Technical Requirements:**
            - Convert string numbers to float64 for MCP tool parameters
            - Use valid ISO 8601 date formats for date parameters
            - For pagination, use float64 values for page numbers
            - Never pass nil/null values for required parameters
            - Use default date of 30 days ago for time-based queries when not specified
        """)

        return base_instructions

    def handle_query_execution(
        self,
        repository: str,
        query_type: str,
        query_text: str,
        api_key: str,
        github_token: str,
        model_choice: str,
        include_links: bool,
        detailed_analysis: bool,
    ) -> None:
        """
        Handle the execution of a GitHub query with comprehensive error handling.

        Args:
            repository: Target repository
            query_type: Type of query being executed
            query_text: The actual query text
            api_key: Nebius API key
            github_token: GitHub token
            model_choice: Selected AI model
            include_links: Whether to include links
            detailed_analysis: Whether to perform detailed analysis
        """
        # Validation
        validation_error = self.validate_query_inputs(
            repository, query_text, api_key, github_token
        )
        if validation_error:
            st.error(validation_error)
            return

        # Prepare query
        if repository and repository not in query_text:
            full_query = f"{query_text} in {repository}"
        else:
            full_query = query_text

        # Execute with progress tracking
        with st.spinner("🔍 Analyzing GitHub repository..."):
            try:
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Initializing GitHub MCP connection...")
                progress_bar.progress(20)

                status_text.text("Authenticating with GitHub API...")
                progress_bar.progress(40)

                status_text.text("Processing repository query...")
                progress_bar.progress(60)

                # Execute query
                result = asyncio.run(
                    self.run_github_agent(
                        full_query, model_choice, include_links, detailed_analysis
                    )
                )

                progress_bar.progress(80)
                status_text.text("Formatting results...")

                # Store results
                query_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "repository": repository,
                    "query_type": query_type,
                    "query": query_text,
                    "result_length": len(result),
                    "model": model_choice,
                }
                st.session_state.query_history.append(query_record)
                st.session_state.last_query_result = result

                progress_bar.progress(100)
                status_text.text("Query completed!")

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Display results
                self.display_results(result, repository, query_type)

            except Exception as e:
                st.error(f"❌ **Query Execution Error**: {str(e)}")
                self.display_troubleshooting_guide(e)

    def validate_query_inputs(
        self, repository: str, query_text: str, api_key: str, github_token: str
    ) -> Optional[str]:
        """
        Validate all query inputs and return error message if invalid.

        Returns:
            Error message if validation fails, None if valid
        """
        if not github_token:
            return "❌ Please enter your GitHub Personal Access Token in the sidebar"
        if not api_key:
            return "❌ Please enter your Nebius API key in the sidebar"
        if not repository:
            return "❌ Please enter a repository name"
        if not query_text.strip():
            return "❌ Please enter a query"
        if "/" not in repository:
            return "❌ Repository must be in format 'owner/repo'"

        return None

    def display_results(self, result: str, repository: str, query_type: str) -> None:
        """
        Display query results with enhanced formatting.

        Args:
            result: The query result to display
            repository: Repository that was queried
            query_type: Type of query that was executed
        """
        st.markdown("---")

        # Results header
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### 📊 Results for {repository}")
        with col2:
            st.markdown(f"**Query Type:** {query_type}")
        with col3:
            st.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")

        # Main results
        st.markdown(result)

        # Additional actions
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📋 Copy Results", help="Copy results to clipboard"):
                st.code(result, language="markdown")

        with col2:
            if st.button("🔗 Open Repository", help="Open repository in new tab"):
                st.markdown(
                    f"[🔗 View {repository} on GitHub](https://github.com/{repository})"
                )

        with col3:
            if st.button("📊 Export Data", help="Export query data"):
                self.export_query_data(repository, query_type, result)

    def display_troubleshooting_guide(self, error: Exception) -> None:
        """
        Display contextual troubleshooting information based on error type.

        Args:
            error: The exception that occurred
        """
        st.markdown("### 🔧 Troubleshooting Guide")

        error_msg = str(error).lower()

        if "token" in error_msg or "authentication" in error_msg:
            st.error("**Authentication Issue**")
            st.markdown("""
            **Solutions:**
            - Verify your GitHub token has the correct permissions
            - Ensure the token hasn't expired
            - Check that the token has 'repo' scope for private repositories
            """)

        elif "network" in error_msg or "connection" in error_msg:
            st.error("**Network/Connection Issue**")
            st.markdown("""
            **Solutions:**
            - Check your internet connection
            - Verify Docker is running and accessible
            - Try again in a few moments
            """)

        elif "rate limit" in error_msg:
            st.error("**API Rate Limit Exceeded**")
            st.markdown("""
            **Solutions:**
            - Wait for the rate limit to reset (usually 1 hour)
            - Use a GitHub token with higher rate limits
            - Reduce the frequency of queries
            """)

        else:
            st.error("**General Error**")
            st.markdown("""
            **General Solutions:**
            - Simplify your query and try again
            - Check repository name spelling
            - Verify the repository is accessible
            - Contact support if the issue persists
            """)

    def export_query_data(self, repository: str, query_type: str, result: str) -> None:
        """
        Export query data for external use.

        Args:
            repository: Repository name
            query_type: Type of query
            result: Query result
        """
        export_data = {
            "repository": repository,
            "query_type": query_type,
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "word_count": len(result.split()),
            "character_count": len(result),
        }

        st.download_button(
            label="💾 Download Results (JSON)",
            data=str(export_data),
            file_name=f"github_query_{repository.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    def run(self) -> None:
        """Main application entry point."""
        try:
            logger.info("Starting GitHub MCP Agent")

            # Render UI components
            self.render_header()
            api_key, github_token, model_choice, include_links, detailed_analysis = (
                self.render_sidebar()
            )
            repository, query_type, query_text = self.render_query_interface()

            # Query execution controls
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                execute_query = st.button(
                    "🚀 Execute Query",
                    type="primary",
                    use_container_width=True,
                    disabled=not (
                        api_key and github_token and repository and query_text
                    ),
                )

            with col2:
                if st.button("🔄 Clear Results", use_container_width=True):
                    if "last_query_result" in st.session_state:
                        del st.session_state["last_query_result"]
                    st.rerun()

            with col3:
                if st.button("📋 View History", use_container_width=True):
                    self.display_query_history()

            # Execute query if requested
            if execute_query:
                self.handle_query_execution(
                    repository,
                    query_type,
                    query_text,
                    api_key,
                    github_token,
                    model_choice,
                    include_links,
                    detailed_analysis,
                )

            # Display last result if available
            if st.session_state.get("last_query_result"):
                self.display_results(
                    st.session_state["last_query_result"], repository, query_type
                )

        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error(f"Application error: {str(e)}")

    def display_query_history(self) -> None:
        """Display query history in an expandable section."""
        if not st.session_state.query_history:
            st.info("No query history available")
            return

        with st.expander("📋 Query History", expanded=True):
            for i, query in enumerate(
                reversed(st.session_state.query_history[-10:])
            ):  # Last 10
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{query['repository']}** - {query['query_type']}")
                with col2:
                    st.markdown(f"{query['timestamp']}")
                with col3:
                    st.markdown(f"{query['result_length']} chars")


def main():
    """Application entry point with error handling."""
    try:
        app = GitHubMCPAgent()
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
