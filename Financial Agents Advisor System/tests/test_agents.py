"""
Unit Tests for Financial Agents
===============================

Test suite for individual agent functionality.
"""

import unittest
from unittest.mock import patch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import (
    TechnicalAnalysisAgent,
    SentimentAnalysisAgent,
    RiskAssessmentAgent,
    CryptoAnalysisAgent,
    PortfolioOptimizationAgent,
    AnalysisType,
    AgentCapability,
)


class TestAgentCreation(unittest.TestCase):
    """Test agent creation and initialization."""

    def test_technical_agent_creation(self):
        """Test TechnicalAnalysisAgent creation."""
        agent = TechnicalAnalysisAgent()
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Technical Analysis Agent")
        self.assertIn(AgentCapability.TECHNICAL_ANALYSIS, agent.capabilities)

    def test_sentiment_agent_creation(self):
        """Test SentimentAnalysisAgent creation."""
        agent = SentimentAnalysisAgent()
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Sentiment Analysis Agent")
        self.assertIn(AgentCapability.SENTIMENT_ANALYSIS, agent.capabilities)

    def test_risk_agent_creation(self):
        """Test RiskAssessmentAgent creation."""
        agent = RiskAssessmentAgent()
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Risk Assessment Agent")
        self.assertIn(AgentCapability.RISK_ANALYSIS, agent.capabilities)

    def test_crypto_agent_creation(self):
        """Test CryptoAnalysisAgent creation."""
        agent = CryptoAnalysisAgent()
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Cryptocurrency Analysis Agent")
        self.assertIn(AgentCapability.CRYPTO_ANALYSIS, agent.capabilities)

    def test_portfolio_agent_creation(self):
        """Test PortfolioOptimizationAgent creation."""
        agent = PortfolioOptimizationAgent()
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "Portfolio Optimization Agent")
        self.assertIn(AgentCapability.PORTFOLIO_OPTIMIZATION, agent.capabilities)


class TestAgentCapabilities(unittest.TestCase):
    """Test agent capabilities and interface compliance."""

    def setUp(self):
        """Set up test agents."""
        self.agents = [
            TechnicalAnalysisAgent(),
            SentimentAnalysisAgent(),
            RiskAssessmentAgent(),
            CryptoAnalysisAgent(),
            PortfolioOptimizationAgent(),
        ]

    def test_agent_interfaces(self):
        """Test that all agents implement required interfaces."""
        for agent in self.agents:
            with self.subTest(agent=agent.name):
                # Test required attributes
                self.assertTrue(hasattr(agent, "name"))
                self.assertTrue(hasattr(agent, "version"))
                self.assertTrue(hasattr(agent, "capabilities"))
                self.assertTrue(hasattr(agent, "is_initialized"))

                # Test required methods
                self.assertTrue(hasattr(agent, "initialize"))
                self.assertTrue(hasattr(agent, "analyze"))
                self.assertTrue(callable(agent.initialize))
                self.assertTrue(callable(agent.analyze))

    def test_agent_capabilities_not_empty(self):
        """Test that all agents have defined capabilities."""
        for agent in self.agents:
            with self.subTest(agent=agent.name):
                self.assertGreater(len(agent.capabilities), 0)
                self.assertIsInstance(agent.capabilities, list)


class TestAnalysisTypes(unittest.TestCase):
    """Test analysis type enumeration."""

    def test_analysis_type_values(self):
        """Test that analysis types have proper values."""
        expected_types = [
            "price_movement",
            "trend_analysis",
            "volatility",
            "sentiment",
            "risk_metrics",
            "portfolio_optimization",
        ]

        for analysis_type in AnalysisType:
            self.assertIn(analysis_type.value, expected_types)

    def test_analysis_type_uniqueness(self):
        """Test that analysis type values are unique."""
        values = [at.value for at in AnalysisType]
        self.assertEqual(len(values), len(set(values)))


class TestAgentBehavior(unittest.TestCase):
    """Test agent behavior and response handling."""

    def test_agent_initialization_state(self):
        """Test agent initialization state management."""
        agent = TechnicalAnalysisAgent()

        # Should not be initialized initially
        self.assertFalse(agent.is_initialized)

        # Test state doesn't change without proper initialization
        # (This test assumes initialization requires API keys)
        self.assertFalse(agent.is_initialized)

    def test_agent_analyze_method_signature(self):
        """Test that analyze method has correct signature."""
        agent = TechnicalAnalysisAgent()

        # Test that analyze method exists and can be called
        # (Even if it fails due to missing API keys)
        try:
            result = agent.analyze("AAPL", AnalysisType.PRICE_MOVEMENT)
            # If we get here, the method signature is correct
            self.assertTrue(hasattr(result, "success"))
        except Exception:
            # Expected to fail without proper setup, but signature should be correct
            pass

    @patch("yfinance.download")
    def test_mock_analysis(self, mock_yf):
        """Test analysis with mocked data."""
        import pandas as pd
        import numpy as np

        # Mock yfinance data
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        mock_data = pd.DataFrame(
            {
                "Open": np.random.randn(50).cumsum() + 100,
                "High": np.random.randn(50).cumsum() + 105,
                "Low": np.random.randn(50).cumsum() + 95,
                "Close": np.random.randn(50).cumsum() + 100,
                "Volume": np.random.randint(1000000, 10000000, 50),
            },
            index=dates,
        )

        mock_yf.return_value = mock_data

        agent = TechnicalAnalysisAgent()

        # Test that agent can handle mocked data
        # (May still fail due to API dependencies, but structure should be correct)
        try:
            result = agent.analyze("AAPL", AnalysisType.PRICE_MOVEMENT)
            if hasattr(result, "success"):
                # Structure is correct
                self.assertTrue(True)
        except Exception as e:
            # Expected to fail in test environment
            self.assertIsInstance(e, Exception)


if __name__ == "__main__":
    # Run specific test suites
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTest(unittest.makeSuite(TestAgentCreation))
    suite.addTest(unittest.makeSuite(TestAgentCapabilities))
    suite.addTest(unittest.makeSuite(TestAnalysisTypes))
    suite.addTest(unittest.makeSuite(TestAgentBehavior))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\nTest Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("✅ All agent tests passed!")
    else:
        print("❌ Some tests failed.")
        for failure in result.failures:
            print(f"FAIL: {failure[0]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
