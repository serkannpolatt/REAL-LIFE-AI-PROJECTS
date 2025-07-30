#!/usr/bin/env python3
"""
Setup Script for Financial AI Agent System
==========================================

This script helps set up the Financial AI Agent system with proper configuration.
"""

import os
import sys
from pathlib import Path


def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")

    if env_file.exists():
        print("✅ .env file already exists")
        return True

    if not env_example.exists():
        print("❌ .env.example template not found")
        return False

    # Copy template to .env
    with open(env_example, "r") as template:
        content = template.read()

    with open(env_file, "w") as env:
        env.write(content)

    print("✅ Created .env file from template")
    print("📝 Please edit .env file and add your API keys")
    return True


def check_dependencies():
    """Check if required Python packages are installed."""
    required_packages = [
        "phidata",
        "groq",
        "yfinance",
        "duckduckgo-search",
        "pandas",
        "numpy",
        "scipy",
        "python-dotenv",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False

    return True


def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info

    if version >= (3, 8):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(
            f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+"
        )
        return False


def create_directories():
    """Create necessary directories."""
    directories = ["logs", "data", "cache"]

    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(exist_ok=True)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")


def validate_api_keys():
    """Validate API keys in environment."""
    from dotenv import load_dotenv

    load_dotenv()

    required_keys = ["GROQ_API_KEY"]
    optional_keys = ["OPENAI_API_KEY"]

    print("\n🔑 API Key Validation:")

    all_valid = True
    for key in required_keys:
        if os.getenv(key):
            print(f"✅ {key} - Set")
        else:
            print(f"❌ {key} - Missing (Required)")
            all_valid = False

    for key in optional_keys:
        if os.getenv(key):
            print(f"✅ {key} - Set (Optional)")
        else:
            print(f"⚠️ {key} - Not set (Optional)")

    return all_valid


def run_basic_test():
    """Run basic system test."""
    try:
        # Test import
        sys.path.append(".")
        from utils.api_utils import setup_environment

        print("\n🧪 Running basic system test...")

        if setup_environment():
            print("✅ Basic system test passed")
            return True
        else:
            print("❌ Basic system test failed")
            return False

    except Exception as e:
        print(f"❌ Basic system test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🤖 Financial AI Agent System Setup")
    print("=" * 50)

    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        print("❌ Setup failed: Incompatible Python version")
        return False

    # Check dependencies
    print("\n2. Checking dependencies...")
    if not check_dependencies():
        print("❌ Setup failed: Missing dependencies")
        return False

    # Create directories
    print("\n3. Creating directories...")
    create_directories()

    # Create .env file
    print("\n4. Setting up environment file...")
    create_env_file()

    # Validate API keys
    print("\n5. Validating API keys...")
    api_keys_valid = validate_api_keys()

    # Run basic test
    if api_keys_valid:
        print("\n6. Running basic test...")
        run_basic_test()
    else:
        print("\n⚠️ Skipping basic test - API keys not configured")

    print("\n" + "=" * 50)

    if api_keys_valid:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("  - Run: python app.py")
        print("  - Or: python examples.py")
    else:
        print("⚠️ Setup partially completed")
        print("\nNext steps:")
        print("  1. Edit .env file and add your API keys")
        print("  2. Run this setup script again")
        print("  3. Run: python app.py")

    return True


if __name__ == "__main__":
    main()
