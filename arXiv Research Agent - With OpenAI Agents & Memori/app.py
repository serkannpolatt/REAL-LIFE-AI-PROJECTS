#!/usr/bin/env python3
"""
arXiv Research Agent Streamlit Application

Advanced academic research assistant with OpenAI Agents and Memori integration.
This application provides persistent, organized research support that remembers
research sessions, builds on previous findings, and focuses on arXiv papers.

Features:
- arXiv paper search and analysis
- Persistent research memory
- Tavily integration
- Streamlit web interface

This file contains the Streamlit interface while agent logic is in researcher.py.
"""

import os
import asyncio
import base64
import streamlit as st
from researcher import Researcher

# Page configuration
st.set_page_config(page_title="arXiv Research Agent", page_icon="🔬", layout="wide")


def display_header():
    """Display application header and logos"""
    # Load SVG and PNG logos
    with open("./assets/gibson.svg", "r", encoding="utf-8") as gibson_file:
        gibson_svg = (
            gibson_file.read()
            .replace("\n", "")
            .replace("\r", "")
            .replace("  ", "")
            .replace('"', "'")
        )

    with open("./assets/tavily.png", "rb") as tavily_file:
        tavily_base64 = base64.b64encode(tavily_file.read()).decode()

    # Create inline SVG and PNG elements
    gibson_svg_inline = f'<span style="height:80px; width:200px; display:inline-block; vertical-align:middle; margin-left:8px;margin-top:20px;margin-right:8px;">{gibson_svg}</span>'

    # Create header HTML
    title_html = f"""
    <div style="display: flex; width: 100%; ">
        <h1 style="margin: 0; padding: 0; font-size: 2.5rem; font-weight: bold;">
            <span style="font-size:2.5rem;">🕵🏻‍♂️</span> arXiv Research Agent
            {gibson_svg_inline}
            <span style="">Memori</span> & 
            <img src="data:image/png;base64,{tavily_base64}" style="height: 60px; vertical-align: middle; bottom: 5px;"/>
        </h1>
    </div>
    """

    st.markdown(title_html, unsafe_allow_html=True)


def setup_sidebar():
    """Configure sidebar"""
    with st.sidebar:
        # Nebius logo
        st.image("./assets/nebius.png", width=150)

        # API keys
        nebius_key = st.text_input(
            "Enter Your Nebius API Key",
            value=os.getenv("NEBIUS_API_KEY", ""),
            type="password",
        )

        tavily_api_key = st.text_input(
            "Enter Your Tavily API Key",
            value=os.getenv("TAVILY_API_KEY", ""),
            type="password",
        )

        st.divider()

        # Mode selection
        tab_choice = st.radio(
            "Select Mode:", ["🔬 Research Chat", "🧠 Memory Chat"], key="tab_choice"
        )

        st.divider()

        # Research examples
        if tab_choice == "🔬 Research Chat":
            st.markdown("### 🔬 Example Research Topics:")

            research_examples = [
                (
                    "🧠 Brain-Computer Interfaces",
                    "Research recent developments in brain-computer interfaces",
                ),
                (
                    "🔋 Solid State Batteries",
                    "Analyze the current state of solid-state batteries",
                ),
                (
                    "🧬 CRISPR Gene Editing",
                    "Research recent breakthroughs in CRISPR gene editing",
                ),
                (
                    "🚗 Autonomous Vehicles",
                    "Examine the development of autonomous vehicles",
                ),
            ]

            for title, prompt in research_examples:
                if st.button(title):
                    st.session_state.research_messages.append(
                        {"role": "user", "content": prompt}
                    )

        elif tab_choice == "🧠 Memory Chat":
            st.markdown("### 🧠 Example Memory Queries:")

            memory_examples = [
                (
                    "📊 Summarize my research history",
                    "Can you summarize my research history and main findings?",
                ),
                (
                    "🧬 Find my biotech research",
                    "Find all my research on biotechnology and gene editing",
                ),
                (
                    "📋 What were my recent research topics?",
                    "What were my recent research topics?",
                ),
                (
                    "🤖 Show my AI research",
                    "Show all my previous research on artificial intelligence",
                ),
            ]

            for title, prompt in memory_examples:
                if st.button(title):
                    st.session_state.memory_messages.append(
                        {"role": "user", "content": prompt}
                    )

        # Research history
        st.header("Research History")

        if st.button("📊 View All Research"):
            st.session_state.show_all_research = True

        if st.button("🗑️ Clear All Memory", type="secondary"):
            if st.session_state.get("confirm_clear_research"):
                st.success("Research memory cleared!")
                st.session_state.confirm_clear_research = False
                st.rerun()
            else:
                st.session_state.confirm_clear_research = True
                st.warning("Click again to confirm")

        return tab_choice


async def run_research_agent(researcher, agent, user_input):
    """Run research agent asynchronously"""
    try:
        response = await researcher.run_agent_with_memory(agent, user_input)
        return response
    except Exception as e:
        raise e


async def run_memory_agent(researcher, agent, user_input):
    """Run memory agent asynchronously"""
    try:
        response = await researcher.run_agent_with_memory(agent, user_input)
        return response
    except Exception as e:
        raise e


def display_about_section():
    """Display about section"""
    st.markdown("## About This Demo")
    st.markdown("""
    This demo showcases:
    - **Research Agent**: Uses Tavily for arXiv research paper search
    - **Memori Integration**: Remembers all research sessions
    - **Memory Chat**: Query your research history

    The research agent can:
    - 🔍 Conduct comprehensive research using arXiv papers
    - 🧠 Remember all previous research 
    - 📚 Build upon past research
    - 💾 Store findings for future reference
    """)


def handle_research_chat():
    """Handle research chat"""
    # Display research chat messages
    for message in st.session_state.research_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Research chat input
    if research_prompt := st.chat_input("What would you like me to research?"):
        # Add user message to chat history
        st.session_state.research_messages.append(
            {"role": "user", "content": research_prompt}
        )

        with st.chat_message("user"):
            st.markdown(research_prompt)

        # Generate research response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Researching and searching memory..."):
                try:
                    # Get response from research agent with automatic memory saving
                    response = asyncio.run(
                        run_research_agent(
                            st.session_state.researcher,
                            st.session_state.research_agent,
                            research_prompt,
                        )
                    )

                    # Extract response content
                    if hasattr(response, "final_output"):
                        response_content = response.final_output
                    elif hasattr(response, "content"):
                        response_content = response.content
                    else:
                        response_content = str(response)

                    # Display response
                    st.markdown(response_content)

                    # Confirm individual conversations were saved
                    st.success("✅ All agent conversations saved to memory!", icon="🧠")

                    # Add assistant response to chat history
                    st.session_state.research_messages.append(
                        {"role": "assistant", "content": response_content}
                    )

                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.research_messages.append(
                        {"role": "assistant", "content": error_message}
                    )


def handle_memory_chat():
    """Handle memory chat"""
    for message in st.session_state.memory_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Memory chat input
    if memory_prompt := st.chat_input(
        "What would you like to know about your research history?"
    ):
        # Add user message to chat history
        st.session_state.memory_messages.append(
            {"role": "user", "content": memory_prompt}
        )

        with st.chat_message("user"):
            st.markdown(memory_prompt)

        # Generate memory response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Searching your research history..."):
                try:
                    # Get response from memory agent with automatic memory saving
                    response = asyncio.run(
                        run_memory_agent(
                            st.session_state.researcher,
                            st.session_state.memory_agent,
                            memory_prompt,
                        )
                    )

                    # Extract response content
                    if hasattr(response, "final_output"):
                        response_content = response.final_output
                    elif hasattr(response, "content"):
                        response_content = response.content
                    else:
                        response_content = str(response)

                    # Display response
                    st.markdown(response_content)

                    # Confirm conversations are saved
                    st.success("✅ Memory agent conversations saved!", icon="🧠")

                    # Add assistant response to chat history
                    st.session_state.memory_messages.append(
                        {"role": "assistant", "content": response_content}
                    )

                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.memory_messages.append(
                        {"role": "assistant", "content": error_message}
                    )


def initialize_session_state():
    """Initialize session state"""
    # Initialize researcher
    if "researcher" not in st.session_state:
        with st.spinner("Initializing Researcher with Memori..."):
            st.session_state.researcher = Researcher()
            st.session_state.researcher.define_agents()

    # Get agents
    if "research_agent" not in st.session_state:
        st.session_state.research_agent = (
            st.session_state.researcher.get_research_agent()
        )

    if "memory_agent" not in st.session_state:
        st.session_state.memory_agent = st.session_state.researcher.get_memory_agent()

    # Initialize chat histories
    if "research_messages" not in st.session_state:
        st.session_state.research_messages = []

    if "memory_messages" not in st.session_state:
        st.session_state.memory_messages = []


def main():
    """Main function"""
    # Check required environment variables
    missing = []
    if not os.getenv("NEBIUS_API_KEY"):
        missing.append("NEBIUS_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if missing:
        st.error(f"Please set the required environment variables: {', '.join(missing)}")
        st.stop()

    # Display header
    display_header()

    # Initialize session state
    initialize_session_state()

    # Setup sidebar
    tab_choice = setup_sidebar()

    # Main content
    if not st.session_state.research_messages and tab_choice == "🔬 Research Chat":
        display_about_section()

    # Handle chat based on selected mode
    if tab_choice == "🔬 Research Chat":
        handle_research_chat()
    elif tab_choice == "🧠 Memory Chat":
        handle_memory_chat()


if __name__ == "__main__":
    main()
