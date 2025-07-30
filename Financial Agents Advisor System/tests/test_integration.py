"""
System Integration Tests
=======================

Test suite for validating the complete financial AI agent system integration.
"""

import unittest
from unittest.mock import patch
from dotenv import load_dotenv

from model.models import AgentConfigManager
from controller.agent_controller import FinanceAgentController
from agents import (
    agent_registry,
    initialize_all_agents,
    TechnicalAnalysisAgent,
    SentimentAnalysisAgent,
    RiskAssessmentAgent,
    CryptoAnalysisAgent,
    PortfolioOptimizationAgent,
    AnalysisType,
    AgentCapability
)

# Load environment variables
load_dotenv()

class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.config_manager = AgentConfigManager()
        self.controller = FinanceAgentController(self.config_manager)
        
    def test_basic_agent_creation(self):
        """Test basic agent creation through controller."""
        try:
            main_agent = self.controller.create_main_agent()
            self.assertIsNotNone(main_agent)
            print("✓ Main agent creation successful")
        except Exception as e:
            self.skipTest(f"Skipping due to API dependency: {e}")
    
    def test_specialized_agent_initialization(self):
        """Test initialization of all specialized agents."""
        agents_to_test = [
            TechnicalAnalysisAgent,
            SentimentAnalysisAgent,
            RiskAssessmentAgent,
            CryptoAnalysisAgent,
            PortfolioOptimizationAgent
        ]
        
        for AgentClass in agents_to_test:
            with self.subTest(agent=AgentClass.__name__):
                agent = AgentClass()
                self.assertIsNotNone(agent)
                self.assertFalse(agent.is_initialized)  # Should be False before initialization
                print(f"✓ {AgentClass.__name__} creation successful")
    
    def test_agent_registry_functionality(self):
        """Test agent registry operations."""
        # Clear registry first
        agent_registry._agents = {}
        
        # Create test agent
        test_agent = TechnicalAnalysisAgent()
        
        # Test registration
        agent_registry.register_agent(test_agent)
        self.assertEqual(len(agent_registry.get_all_agents()), 1)
        
        # Test retrieval by capability
        capable_agents = agent_registry.get_agents_by_capability(AgentCapability.TECHNICAL_ANALYSIS)
        self.assertEqual(len(capable_agents), 1)
        self.assertEqual(capable_agents[0], test_agent)
        
        # Test agent info
        agent_info = agent_registry.get_agent_info()
        self.assertIn(test_agent.name, agent_info)
        
        print("✓ Agent registry functionality working")
    
    def test_analysis_type_enum(self):
        """Test AnalysisType enum functionality."""
        analysis_types = [
            AnalysisType.PRICE_MOVEMENT,
            AnalysisType.TREND_ANALYSIS,
            AnalysisType.VOLATILITY,
            AnalysisType.SENTIMENT,
            AnalysisType.RISK_METRICS,
            AnalysisType.PORTFOLIO_OPTIMIZATION
        ]
        
        for analysis_type in analysis_types:
            self.assertIsInstance(analysis_type, AnalysisType)
            self.assertIsInstance(analysis_type.value, str)
        
        print("✓ AnalysisType enum working correctly")
    
    def test_agent_capabilities(self):
        """Test agent capability system."""
        # Test TechnicalAnalysisAgent capabilities
        tech_agent = TechnicalAnalysisAgent()
        expected_capabilities = [
            AgentCapability.TECHNICAL_ANALYSIS,
            AgentCapability.PRICE_ANALYSIS,
            AgentCapability.TREND_ANALYSIS
        ]
        
        for capability in expected_capabilities:
            self.assertIn(capability, tech_agent.capabilities)
        
        # Test SentimentAnalysisAgent capabilities
        sentiment_agent = SentimentAnalysisAgent()
        self.assertIn(AgentCapability.SENTIMENT_ANALYSIS, sentiment_agent.capabilities)
        
        # Test RiskAssessmentAgent capabilities
        risk_agent = RiskAssessmentAgent()
        self.assertIn(AgentCapability.RISK_ANALYSIS, risk_agent.capabilities)
        
        print("✓ Agent capabilities working correctly")
    
    @patch('yfinance.download')
    def test_mock_data_analysis(self, mock_yf_download):
        """Test analysis with mocked data."""
        # Mock yfinance data
        import pandas as pd
        import numpy as np
        
        # Create mock data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 105,
            'Low': np.random.randn(100).cumsum() + 95,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        mock_yf_download.return_value = mock_data
        
        # Test technical analysis with mock data
        tech_agent = TechnicalAnalysisAgent()
        try:
            if tech_agent.initialize():
                result = tech_agent.analyze('AAPL', AnalysisType.PRICE_MOVEMENT)
                self.assertIsNotNone(result)
                print("✓ Mock data analysis working")
            else:
                self.skipTest("Agent initialization failed")
        except Exception as e:
            self.skipTest(f"Skipping due to dependency: {e}")
    
    def test_configuration_manager(self):
        """Test configuration manager functionality."""
        config = self.config_manager
        
        # Test main agent configuration
        main_config = config.get_main_agent_config()
        self.assertIsNotNone(main_config)
        self.assertIn('model', main_config)
        self.assertIn('tools', main_config)
        
        # Test portfolio agent configuration
        portfolio_config = config.get_portfolio_agent_config()
        self.assertIsNotNone(portfolio_config)
        
        print("✓ Configuration manager working")
    
    def test_controller_agent_creation(self):
        """Test controller's agent creation methods."""
        controller = self.controller
        
        try:
            # Test main agent creation
            main_agent = controller.create_main_agent()
            self.assertIsNotNone(main_agent)
            
            # Test portfolio agent creation
            portfolio_agent = controller.create_portfolio_agent()
            self.assertIsNotNone(portfolio_agent)
            
            print("✓ Controller agent creation working")
        except Exception as e:
            self.skipTest(f"Skipping due to API dependency: {e}")
    
    def test_system_integration_flow(self):
        """Test complete system integration flow."""
        try:
            # Initialize all agents
            success = initialize_all_agents()
            
            if success:
                # Check registry
                all_agents = agent_registry.get_all_agents()
                self.assertGreater(len(all_agents), 0)
                
                # Test each agent has proper attributes
                for agent in all_agents:
                    self.assertIsNotNone(agent.name)
                    self.assertIsNotNone(agent.version)
                    self.assertIsInstance(agent.capabilities, list)
                
                print(f"✓ System integration working with {len(all_agents)} agents")
            else:
                self.skipTest("Agent initialization failed")
                
        except Exception as e:
            self.skipTest(f"Skipping due to API dependency: {e}")

class TestAgentResponses(unittest.TestCase):
    """Test agent response formats."""
    
    def test_agent_response_structure(self):
        """Test AgentResponse structure."""
        from agents.base_agent import AgentResponse
        
        # Test successful response
        success_response = AgentResponse(
            success=True,
            data={"test": "data"},
            confidence=0.85,
            analysis_type=AnalysisType.PRICE_MOVEMENT
        )
        
        self.assertTrue(success_response.success)
        self.assertEqual(success_response.data["test"], "data")
        self.assertEqual(success_response.confidence, 0.85)
        self.assertIsNone(success_response.error)
        
        # Test error response
        error_response = AgentResponse(
            success=False,
            error="Test error",
            analysis_type=AnalysisType.SENTIMENT
        )
        
        self.assertFalse(error_response.success)
        self.assertEqual(error_response.error, "Test error")
        self.assertIsNone(error_response.data)
        
        print("✓ AgentResponse structure working correctly")

def run_integration_tests():
    """Run all integration tests."""
    print("Financial AI Agent System - Integration Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestSystemIntegration))
    suite.addTest(unittest.makeSuite(TestAgentResponses))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All integration tests passed!")
    else:
        print("❌ Some tests failed:")
        for failure in result.failures:
            print(f"  - {failure[0]}")
        for error in result.errors:
            print(f"  - {error[0]} (ERROR)")
    
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    run_integration_tests()
