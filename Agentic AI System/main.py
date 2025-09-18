"""
Agentic AI System - Main Entry Point
==================================

Main entry point for intelligent ticket management system.
Uses LangChain, Groq Cloud and various domain-specific agents.

Features:
- Automatic intent classification
- Domain-specific agent routing
- Ticket creation, status checking and closing
- SQLite database integration
"""

import sys
import os

# Add project root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import SystemOrchestrator


def main():
    """Main system entry point"""
    print("🚀 Starting Agentic AI System...")

    try:
        # Initialize system orchestrator
        orchestrator = SystemOrchestrator()

        # Check system status
        status = orchestrator.get_system_status()
        print(f"✅ System ready - {status['agents_loaded']} agents loaded")
        print(f"📋 Loaded agents: {', '.join(status['agent_names'])}")

        return orchestrator

    except Exception as e:
        print(f"❌ System startup error: {e}")
        raise


# Global orchestrator instance for backwards compatibility
print("🔧 Loading system components...")
orchestrator = main()


# Legacy functions for backwards compatibility with frontend
def classify_intent(prompt: str) -> str:
    """Intent classification function for legacy frontend compatibility"""
    result = orchestrator.process_user_input(prompt)
    return result["intent"]


# Legacy agent variable for backwards compatibility
agent = orchestrator.agent


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("    🤖 AGENTIC AI SYSTEM")
    print("=" * 50)
    print("💡 This system is designed for ticket management.")
    print("🌐 For web interface: streamlit run frontend/app.py")
    print("📊 For database view: streamlit run frontend/check_db.py")
    print("=" * 50 + "\n")
