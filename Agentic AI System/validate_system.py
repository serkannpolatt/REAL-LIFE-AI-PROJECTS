"""
System Validation Script
========================

Validates that all components of the Agentic AI System work properly.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def test_imports():
    """Test that all critical imports work"""
    print("🔍 Testing module imports...")

    try:
        # Test utility imports
        from utils.db_utils import init_db, get_db_path

        print("✅ Database utilities import successful")

        from utils.llm_instance import llm

        print("✅ LLM instance import successful")

        from utils.ticket_parser import extract_ticket_info_and_intent

        print("✅ Ticket parser import successful")

        from utils.enhance_status import enhance_ticket_status

        print("✅ Status enhancer import successful")

        # Test core imports
        from core.agent_manager import AgentManager

        print("✅ Agent manager import successful")

        from core.intent_classifier import IntentClassifier

        print("✅ Intent classifier import successful")

        from core.orchestrator import SystemOrchestrator

        print("✅ System orchestrator import successful")

        # Test agent imports
        from agents.it_agent import handle_it_issue
        from agents.hr_agent import handle_hr_issue
        from agents.finance_agent import handle_finance_issue
        from agents.admin_agent import handle_admin_issue
        from agents.infra_agent import handle_infra_issue
        from agents.ticket_status_agent import handle_ticket_status

        print("✅ All agent imports successful")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def test_database():
    """Test database functionality"""
    print("\n📁 Testing database functionality...")

    try:
        from utils.db_utils import init_db, get_db_path

        # Get database path
        db_path = get_db_path()
        print(f"📍 Database path: {db_path}")

        # Initialize database
        init_db()
        print("✅ Database initialization successful")

        # Check if database file exists
        if os.path.exists(db_path):
            print("✅ Database file created successfully")
            return True
        else:
            print("❌ Database file not found")
            return False

    except Exception as e:
        print(f"❌ Database error: {e}")
        return False


def test_orchestrator():
    """Test system orchestrator"""
    print("\n🎯 Testing system orchestrator...")

    try:
        from core.orchestrator import SystemOrchestrator

        # Try to create orchestrator (this will test all integrations)
        orchestrator = SystemOrchestrator()
        print("✅ System orchestrator created successfully")

        # Get system status
        status = orchestrator.get_system_status()
        print(f"✅ System status: {status['agents_loaded']} agents loaded")
        print(f"📋 Available agents: {', '.join(status['agent_names'])}")

        return True

    except Exception as e:
        print(f"❌ Orchestrator error: {e}")
        return False


def test_config():
    """Test configuration files"""
    print("\n⚙️  Testing configuration...")

    try:
        import json

        # Test agents config
        with open("agents_config.json", "r") as f:
            config = json.load(f)

        print(f"✅ Agent configuration loaded: {len(config)} agents configured")

        # Test .env.example exists
        if os.path.exists(".env.example"):
            print("✅ Environment template (.env.example) exists")
        else:
            print("⚠️  Environment template (.env.example) not found")

        return True

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def main():
    """Main validation function"""
    print("🚀 Starting Agentic AI System Validation")
    print("=" * 50)

    tests = [
        ("Import Tests", test_imports),
        ("Database Tests", test_database),
        ("Configuration Tests", test_config),
        ("Orchestrator Tests", test_orchestrator),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\n💡 Quick start commands:")
        print("   Web interface: streamlit run frontend/app.py")
        print("   Database viewer: streamlit run frontend/check_db.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
