"""
Agentic AI System - Streamlit Web Interface
==========================================

User-friendly chat interface for intelligent ticket management system.
"""

import streamlit as st
import sys
import os

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ticket_parser import extract_ticket_info_and_intent
from utils.llm_instance import llm
from main import agent, classify_intent


def init_page_config():
    """Initialize page configuration"""
    st.set_page_config(
        page_title="Agentic AI Assistant",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )


def apply_custom_css():
    """Apply custom CSS styles"""
    st.markdown(
        """
        <style>
            /* Ana tema */
            .main {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            }
            
            /* Chat mesajları */
            .stChatMessage {
                border-radius: 15px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            /* Kullanıcı mesajları */
            .user-message {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            /* Assistant mesajları */
            .assistant-message {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
            }
            
            /* Başlık */
            .main-title {
                text-align: center;
                color: #2c3e50;
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            
            /* Alt başlık */
            .subtitle {
                text-align: center;
                color: #7f8c8d;
                font-size: 1.2rem;
                margin-bottom: 2rem;
            }
            
            /* Sidebar */
            .css-1d391kg {
                background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
                color: white;
            }
            
            /* Intent badge */
            .intent-badge {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: bold;
                margin-bottom: 10px;
            }
            
            .intent-ticket {
                background-color: #e74c3c;
                color: white;
            }
            
            .intent-info {
                background-color: #3498db;
                color: white;
            }
            
            .intent-status {
                background-color: #f39c12;
                color: white;
            }
            
            .intent-close {
                background-color: #27ae60;
                color: white;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )


def display_header():
    """Display main header"""
    st.markdown(
        '<h1 class="main-title">🤖 Agentic AI Assistant</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">Intelligent Ticket Management System</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")


def setup_sidebar():
    """Setup sidebar"""
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")

        # System information
        st.markdown("### 📊 System Status")
        st.success("✅ System Active")
        st.info("🤖 Active Agents: IT, HR, Finance, Admin, Infra")

        # Chat controls
        st.markdown("### 💬 Chat Controls")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        # Help section
        st.markdown("### ❓ How to Use")
        st.markdown("""
        **Example Questions:**
        - 💻 "My computer won't start"
        - 📝 "I want to apply for leave"  
        - 💰 "I can't access my payslip"
        - 🔌 "Office power is out"
        - 📋 "Check ticket status"
        """)


def get_intent_badge(intent: str) -> str:
    """Generate badge HTML for intent"""
    intent_labels = {
        "ticket": "🎫 New Ticket",
        "info": "ℹ️ Info",
        "status": "📋 Status Check",
        "close": "✅ Close Ticket",
    }

    label = intent_labels.get(intent, "❓ Unknown")
    return f'<span class="intent-badge intent-{intent}">{label}</span>'


def process_user_message(user_input: str):
    """Process user message"""
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Intent classification
                intent = classify_intent(user_input)

                # Show intent badge
                intent_badge = get_intent_badge(intent)
                st.markdown(intent_badge, unsafe_allow_html=True)

                # Process request
                if intent == "ticket":
                    response = handle_ticket_request(user_input)
                elif intent in ["status", "close"]:
                    response = handle_ticket_management(user_input)
                elif intent == "info":
                    response = handle_info_request(user_input)
                else:
                    response = "❓ I couldn't understand your message. Please rephrase your question."

                st.markdown(response)
                return response

            except Exception as e:
                error_msg = f"❌ An error occurred: {str(e)}"
                st.error(error_msg)
                return error_msg


def handle_ticket_request(user_input: str) -> str:
    """Handle ticket creation request"""
    try:
        return agent.run(user_input)
    except Exception as e:
        return f"❌ Error while creating ticket: {str(e)}"


def handle_ticket_management(user_input: str) -> str:
    """Handle ticket management (status/closing)"""
    try:
        structured_input = extract_ticket_info_and_intent(user_input)

        if structured_input["intent"] == "close":
            structured_input["status"] = "Closed"

        return agent.run(f"ticket status checker input: {structured_input}")
    except Exception as e:
        return f"❌ Error during ticket operation: {str(e)}"


def handle_info_request(user_input: str) -> str:
    """Handle information request"""
    try:
        ai_response = llm.invoke(user_input)
        return ai_response.content
    except Exception as e:
        return f"❌ Error during information query: {str(e)}"


def main():
    """Main application function"""
    # Page settings
    init_page_config()
    apply_custom_css()

    # Page layout
    display_header()
    setup_sidebar()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Welcome message
        welcome_msg = """
        👋 **Hello! Welcome to Agentic AI Assistant!**
        
        I'm here to help solve problems in your company. I can:
        
        🎫 **Create Tickets** - IT, HR, Finance, Admin or Infrastructure issues
        📋 **Check Ticket Status** - Monitor your existing tickets  
        ✅ **Close Tickets** - Close resolved tickets
        ℹ️ **Provide Information** - Answer general questions
        
        How can I help you?
        """
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_input := st.chat_input("Type your message..."):
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Process message
        response = process_user_message(user_input)

        # Save response
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
