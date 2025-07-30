"""
Portfolio Optimization Agent - Advanced Portfolio Management
==========================================================

This module implements a specialized agent for portfolio optimization,
asset allocation, and portfolio risk management using modern portfolio theory.

Classes:
    PortfolioOptimizationAgent: Main portfolio optimization agent
    ModernPortfolioTheory: MPT calculations and optimization
    AssetAllocationStrategies: Various asset allocation strategies
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from phi.tools.yfinance import YFinanceTools

from .base_agent import BaseFinancialAgent, AgentCapability, AnalysisType, AgentResponse

logger = logging.getLogger(__name__)


class ModernPortfolioTheory:
    """
    Modern Portfolio Theory calculations and optimization algorithms.
    """

    @staticmethod
    def calculate_portfolio_metrics(
        weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.

        Args:
            weights (np.ndarray): Portfolio weights
            returns (pd.DataFrame): Historical returns matrix
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation

        Returns:
            Dict[str, float]: Portfolio metrics
        """
        if returns.empty:
            return {"return": 0.0, "volatility": 0.0, "sharpe_ratio": 0.0}

        # Portfolio returns
        portfolio_returns = returns.dot(weights)

        # Annualized return and volatility
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe_ratio = (
            (annual_return - risk_free_rate) / annual_volatility
            if annual_volatility > 0
            else 0
        )

        return {
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe_ratio,
        }

    @staticmethod
    def efficient_frontier(
        returns: pd.DataFrame, num_portfolios: int = 10000, risk_free_rate: float = 0.02
    ) -> Dict[str, Any]:
        """
        Calculate efficient frontier using Monte Carlo simulation.

        Args:
            returns (pd.DataFrame): Historical returns matrix
            num_portfolios (int): Number of random portfolios to generate
            risk_free_rate (float): Risk-free rate

        Returns:
            Dict[str, Any]: Efficient frontier data
        """
        if returns.empty or returns.shape[1] < 2:
            return {"efficient_portfolios": [], "optimal_weights": []}

        num_assets = returns.shape[1]
        results = np.zeros((3, num_portfolios))
        weights_array = np.zeros((num_portfolios, num_assets))

        for i in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)  # Normalize to sum to 1
            weights_array[i, :] = weights

            # Calculate portfolio metrics
            metrics = ModernPortfolioTheory.calculate_portfolio_metrics(
                weights, returns, risk_free_rate
            )
            results[0, i] = metrics["annual_return"]
            results[1, i] = metrics["annual_volatility"]
            results[2, i] = metrics["sharpe_ratio"]

        # Find optimal portfolios
        max_sharpe_idx = np.argmax(results[2])
        min_vol_idx = np.argmin(results[1])

        return {
            "returns": results[0],
            "volatilities": results[1],
            "sharpe_ratios": results[2],
            "weights": weights_array,
            "max_sharpe_portfolio": {
                "weights": weights_array[max_sharpe_idx],
                "return": results[0, max_sharpe_idx],
                "volatility": results[1, max_sharpe_idx],
                "sharpe_ratio": results[2, max_sharpe_idx],
            },
            "min_volatility_portfolio": {
                "weights": weights_array[min_vol_idx],
                "return": results[0, min_vol_idx],
                "volatility": results[1, min_vol_idx],
                "sharpe_ratio": results[2, min_vol_idx],
            },
        }

    @staticmethod
    def optimize_portfolio(
        returns: pd.DataFrame, target_return: float = None, minimize_risk: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize portfolio using mathematical optimization.

        Args:
            returns (pd.DataFrame): Historical returns matrix
            target_return (float): Target annual return (if None, maximize Sharpe ratio)
            minimize_risk (bool): Whether to minimize risk or maximize return

        Returns:
            Dict[str, Any]: Optimized portfolio
        """
        if returns.empty:
            return {"success": False, "message": "No data available"}

        num_assets = returns.shape[1]

        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        ]  # Weights sum to 1

        # Add target return constraint if specified
        if target_return is not None:
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x: np.dot(x, expected_returns) - target_return,
                }
            )

        # Bounds (weights between 0 and 1)
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Initial guess (equal weights)
        initial_weights = np.array([1 / num_assets] * num_assets)

        # Objective function
        if minimize_risk or target_return is not None:
            # Minimize portfolio variance
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
        else:
            # Maximize Sharpe ratio (minimize negative Sharpe ratio)
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                return -sharpe

        # Optimize
        try:
            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                optimal_weights = result.x
                metrics = ModernPortfolioTheory.calculate_portfolio_metrics(
                    optimal_weights, returns
                )

                return {
                    "success": True,
                    "optimal_weights": optimal_weights,
                    "metrics": metrics,
                    "asset_allocation": dict(zip(returns.columns, optimal_weights)),
                }
            else:
                return {"success": False, "message": "Optimization failed"}

        except Exception as e:
            return {"success": False, "message": f"Optimization error: {str(e)}"}


class AssetAllocationStrategies:
    """
    Various asset allocation strategies and portfolio construction methods.
    """

    @staticmethod
    def equal_weight_portfolio(assets: List[str]) -> Dict[str, float]:
        """Create equal-weighted portfolio."""
        weight = 1.0 / len(assets)
        return {asset: weight for asset in assets}

    @staticmethod
    def market_cap_weighted_portfolio(
        market_caps: Dict[str, float],
    ) -> Dict[str, float]:
        """Create market cap weighted portfolio."""
        total_market_cap = sum(market_caps.values())
        return {asset: cap / total_market_cap for asset, cap in market_caps.items()}

    @staticmethod
    def risk_parity_portfolio(returns: pd.DataFrame) -> Dict[str, float]:
        """
        Create risk parity portfolio where each asset contributes equally to portfolio risk.

        Args:
            returns (pd.DataFrame): Historical returns matrix

        Returns:
            Dict[str, float]: Risk parity weights
        """
        if returns.empty:
            return {}

        # Calculate volatilities
        volatilities = returns.std() * np.sqrt(252)

        # Risk parity weights (inverse volatility)
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()

        return dict(zip(returns.columns, weights))

    @staticmethod
    def target_date_allocation(
        years_to_retirement: int, risk_tolerance: str = "moderate"
    ) -> Dict[str, float]:
        """
        Create target-date fund allocation based on years to retirement.

        Args:
            years_to_retirement (int): Years until retirement
            risk_tolerance (str): Risk tolerance level

        Returns:
            Dict[str, float]: Asset allocation
        """
        # Basic glide path: Stock % = 100 - age
        # Assuming current age = 65 - years_to_retirement
        current_age = max(25, 65 - years_to_retirement)
        base_stock_allocation = max(0.2, min(0.9, (100 - current_age) / 100))

        # Adjust for risk tolerance
        risk_adjustments = {"conservative": -0.1, "moderate": 0.0, "aggressive": 0.1}

        adjustment = risk_adjustments.get(risk_tolerance, 0.0)
        stock_allocation = max(0.1, min(0.95, base_stock_allocation + adjustment))
        bond_allocation = 1.0 - stock_allocation

        return {"stocks": stock_allocation, "bonds": bond_allocation}

    @staticmethod
    def strategic_asset_allocation(
        risk_profile: str, investment_horizon: str
    ) -> Dict[str, float]:
        """
        Create strategic asset allocation based on risk profile and horizon.

        Args:
            risk_profile (str): Conservative, moderate, or aggressive
            investment_horizon (str): Short, medium, or long term

        Returns:
            Dict[str, float]: Strategic allocation
        """
        allocations = {
            "conservative": {
                "short": {"stocks": 0.3, "bonds": 0.6, "cash": 0.1},
                "medium": {"stocks": 0.4, "bonds": 0.5, "cash": 0.1},
                "long": {"stocks": 0.5, "bonds": 0.4, "cash": 0.1},
            },
            "moderate": {
                "short": {"stocks": 0.5, "bonds": 0.4, "cash": 0.1},
                "medium": {"stocks": 0.6, "bonds": 0.3, "cash": 0.1},
                "long": {"stocks": 0.7, "bonds": 0.2, "cash": 0.1},
            },
            "aggressive": {
                "short": {"stocks": 0.7, "bonds": 0.2, "cash": 0.1},
                "medium": {"stocks": 0.8, "bonds": 0.15, "cash": 0.05},
                "long": {"stocks": 0.9, "bonds": 0.05, "cash": 0.05},
            },
        }

        return allocations.get(risk_profile, {}).get(investment_horizon, {})


class PortfolioOptimizationAgent(BaseFinancialAgent):
    """
    Specialized agent for portfolio optimization and asset allocation.

    This agent provides portfolio optimization services including asset allocation,
    risk management, and portfolio construction using modern portfolio theory.
    """

    def __init__(self):
        """Initialize the Portfolio Optimization Agent."""
        capabilities = [
            AgentCapability.PORTFOLIO_OPTIMIZATION,
            AgentCapability.RISK_ASSESSMENT,
            AgentCapability.STOCK_ANALYSIS,
        ]
        super().__init__("PortfolioOptimizationAgent", capabilities)
        self.yfinance_tools = None
        self.mpt = ModernPortfolioTheory()
        self.allocation_strategies = AssetAllocationStrategies()

    def initialize(self) -> bool:
        """Initialize the agent with required resources."""
        try:
            self.yfinance_tools = YFinanceTools(
                stock_price=True,
                technical_indicators=False,
                stock_fundamentals=True,
                company_news=False,
            )
            self.is_initialized = True
            logger.info("Portfolio Optimization Agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Portfolio Optimization Agent: {e}")
            return False

    def analyze(
        self, symbols: List[str], analysis_type: AnalysisType, **kwargs
    ) -> AgentResponse:
        """
        Perform portfolio optimization analysis.

        Args:
            symbols (List[str]): List of symbols to include in portfolio
            analysis_type (AnalysisType): Type of analysis to perform
            **kwargs: Additional parameters (target_return, risk_tolerance, etc.)

        Returns:
            AgentResponse: Portfolio optimization results
        """
        if not self.is_initialized:
            return AgentResponse(
                success=False,
                data={},
                message="Agent not initialized",
                confidence_score=0.0,
                sources=[],
                analysis_type=analysis_type,
                timestamp=datetime.now().isoformat(),
            )

        try:
            # Extract parameters
            optimization_type = kwargs.get("optimization_type", "max_sharpe")
            target_return = kwargs.get("target_return", None)
            risk_tolerance = kwargs.get("risk_tolerance", "moderate")
            investment_horizon = kwargs.get("investment_horizon", "long")
            period = kwargs.get("period", "2y")

            # Get historical data for all symbols
            returns_data = self._get_portfolio_returns(symbols, period)

            if returns_data is None or returns_data.empty:
                return AgentResponse(
                    success=False,
                    data={},
                    message="Insufficient data for portfolio optimization",
                    confidence_score=0.0,
                    sources=[],
                    analysis_type=analysis_type,
                    timestamp=datetime.now().isoformat(),
                )

            # Perform portfolio optimization
            optimization_result = self._optimize_portfolio(
                returns_data,
                symbols,
                optimization_type,
                target_return,
                risk_tolerance,
                investment_horizon,
            )

            return AgentResponse(
                success=True,
                data=optimization_result,
                message=f"Portfolio optimization completed for {len(symbols)} assets",
                confidence_score=optimization_result.get("confidence_score", 0.8),
                sources=[
                    "Yahoo Finance",
                    "Modern Portfolio Theory",
                    "Optimization Algorithms",
                ],
                analysis_type=analysis_type,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return AgentResponse(
                success=False,
                data={},
                message=f"Portfolio optimization failed: {str(e)}",
                confidence_score=0.0,
                sources=[],
                analysis_type=analysis_type,
                timestamp=datetime.now().isoformat(),
            )

    def _get_portfolio_returns(
        self, symbols: List[str], period: str
    ) -> Optional[pd.DataFrame]:
        """Get historical returns for portfolio assets."""
        try:
            import yfinance as yf

            # Download data for all symbols
            data = yf.download(symbols, period=period, progress=False)

            if data.empty:
                return None

            # Extract closing prices
            if len(symbols) == 1:
                prices = data["Close"].to_frame(symbols[0])
            else:
                prices = data["Close"]

            # Calculate returns
            returns = prices.pct_change().dropna()

            return returns

        except Exception as e:
            logger.error(f"Failed to get portfolio data: {e}")
            return None

    def _optimize_portfolio(
        self,
        returns_data: pd.DataFrame,
        symbols: List[str],
        optimization_type: str,
        target_return: Optional[float],
        risk_tolerance: str,
        investment_horizon: str,
    ) -> Dict[str, Any]:
        """Perform comprehensive portfolio optimization."""

        # Basic portfolio statistics
        portfolio_stats = self._calculate_portfolio_statistics(returns_data)

        # Generate different allocation strategies
        allocation_strategies = self._generate_allocation_strategies(
            returns_data, symbols, risk_tolerance, investment_horizon
        )

        # Perform optimization based on type
        if optimization_type == "max_sharpe":
            optimization_result = self.mpt.optimize_portfolio(
                returns_data, minimize_risk=False
            )
        elif optimization_type == "min_volatility":
            optimization_result = self.mpt.optimize_portfolio(
                returns_data, minimize_risk=True
            )
        elif optimization_type == "target_return" and target_return is not None:
            optimization_result = self.mpt.optimize_portfolio(
                returns_data, target_return=target_return, minimize_risk=True
            )
        else:
            # Default to max Sharpe ratio
            optimization_result = self.mpt.optimize_portfolio(
                returns_data, minimize_risk=False
            )

        # Calculate efficient frontier
        efficient_frontier = self.mpt.efficient_frontier(returns_data)

        # Generate recommendations
        recommendations = self._generate_portfolio_recommendations(
            optimization_result, allocation_strategies, portfolio_stats, risk_tolerance
        )

        # Backtesting (simplified)
        backtest_results = self._simple_backtest(returns_data, optimization_result)

        return {
            "symbols": symbols,
            "optimization_type": optimization_type,
            "portfolio_statistics": portfolio_stats,
            "optimal_portfolio": optimization_result,
            "allocation_strategies": allocation_strategies,
            "efficient_frontier": efficient_frontier,
            "backtest_results": backtest_results,
            "recommendations": recommendations,
            "confidence_score": self._calculate_optimization_confidence(
                optimization_result, returns_data
            ),
        }

    def _calculate_portfolio_statistics(
        self, returns_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate basic portfolio statistics."""
        if returns_data.empty:
            return {}

        # Individual asset statistics
        annual_returns = returns_data.mean() * 252
        annual_volatilities = returns_data.std() * np.sqrt(252)
        sharpe_ratios = annual_returns / annual_volatilities

        # Correlation matrix
        correlation_matrix = returns_data.corr()

        # Diversification metrics
        avg_correlation = correlation_matrix.values[
            np.triu_indices_from(correlation_matrix.values, k=1)
        ].mean()

        return {
            "asset_statistics": {
                "annual_returns": dict(zip(returns_data.columns, annual_returns)),
                "annual_volatilities": dict(
                    zip(returns_data.columns, annual_volatilities)
                ),
                "sharpe_ratios": dict(zip(returns_data.columns, sharpe_ratios)),
            },
            "correlation_matrix": correlation_matrix.to_dict(),
            "average_correlation": avg_correlation,
            "diversification_potential": 1 - avg_correlation,
            "data_period": f"{returns_data.index[0].date()} to {returns_data.index[-1].date()}",
            "number_of_observations": len(returns_data),
        }

    def _generate_allocation_strategies(
        self,
        returns_data: pd.DataFrame,
        symbols: List[str],
        risk_tolerance: str,
        investment_horizon: str,
    ) -> Dict[str, Any]:
        """Generate various allocation strategies for comparison."""
        strategies = {}

        # Equal weight allocation
        strategies["equal_weight"] = self.allocation_strategies.equal_weight_portfolio(
            symbols
        )

        # Risk parity allocation
        strategies["risk_parity"] = self.allocation_strategies.risk_parity_portfolio(
            returns_data
        )

        # Strategic asset allocation (simplified for stock/bond mix)
        if len(symbols) >= 2:
            strategic_allocation = (
                self.allocation_strategies.strategic_asset_allocation(
                    risk_tolerance, investment_horizon
                )
            )
            strategies["strategic"] = strategic_allocation

        # Calculate performance for each strategy
        for strategy_name, weights in strategies.items():
            if isinstance(weights, dict) and weights:
                if strategy_name in ["equal_weight", "risk_parity"]:
                    # These have weights for specific symbols
                    weight_array = np.array(
                        [weights.get(symbol, 0) for symbol in symbols]
                    )
                    if np.sum(weight_array) > 0:
                        weight_array = weight_array / np.sum(weight_array)  # Normalize
                        metrics = self.mpt.calculate_portfolio_metrics(
                            weight_array, returns_data
                        )
                        strategies[strategy_name] = {
                            "weights": dict(zip(symbols, weight_array)),
                            "metrics": metrics,
                        }

        return strategies

    def _simple_backtest(
        self, returns_data: pd.DataFrame, optimization_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform simple backtest of optimized portfolio."""
        if not optimization_result.get("success") or returns_data.empty:
            return {"backtest_available": False}

        optimal_weights = optimization_result["optimal_weights"]

        # Calculate portfolio returns
        portfolio_returns = returns_data.dot(optimal_weights)

        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        max_drawdown = (
            cumulative_returns / cumulative_returns.expanding().max() - 1
        ).min()

        # Sharpe ratio
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        return {
            "backtest_available": True,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": abs(max_drawdown),
            "best_month": portfolio_returns.max(),
            "worst_month": portfolio_returns.min(),
            "positive_months": (portfolio_returns > 0).sum() / len(portfolio_returns),
        }

    def _generate_portfolio_recommendations(
        self,
        optimization_result: Dict[str, Any],
        allocation_strategies: Dict[str, Any],
        portfolio_stats: Dict[str, Any],
        risk_tolerance: str,
    ) -> List[str]:
        """Generate portfolio recommendations based on optimization results."""
        recommendations = []

        if optimization_result.get("success"):
            metrics = optimization_result.get("metrics", {})
            sharpe_ratio = metrics.get("sharpe_ratio", 0)
            volatility = metrics.get("annual_volatility", 0)

            # Performance-based recommendations
            if sharpe_ratio > 1.5:
                recommendations.append(
                    "Excellent risk-adjusted returns expected from optimized portfolio"
                )
            elif sharpe_ratio > 1.0:
                recommendations.append(
                    "Good risk-adjusted returns expected from optimized portfolio"
                )
            elif sharpe_ratio > 0.5:
                recommendations.append("Moderate risk-adjusted returns expected")
            else:
                recommendations.append(
                    "Consider alternative assets or strategies for better risk-adjusted returns"
                )

            # Volatility-based recommendations
            if volatility > 0.25:
                recommendations.append(
                    "High portfolio volatility - consider position sizing and risk management"
                )
            elif volatility > 0.15:
                recommendations.append(
                    "Moderate portfolio volatility - suitable for balanced investors"
                )
            else:
                recommendations.append(
                    "Low portfolio volatility - suitable for conservative investors"
                )

            # Diversification recommendations
            avg_correlation = portfolio_stats.get("average_correlation", 0)
            if avg_correlation > 0.7:
                recommendations.append(
                    "High correlation between assets - consider adding uncorrelated assets"
                )
            elif avg_correlation > 0.5:
                recommendations.append(
                    "Moderate correlation - reasonable diversification present"
                )
            else:
                recommendations.append(
                    "Good diversification with low correlation between assets"
                )

        # Risk tolerance alignment
        if risk_tolerance == "conservative":
            recommendations.append(
                "Consider increasing bond allocation for more conservative approach"
            )
        elif risk_tolerance == "aggressive":
            recommendations.append("Portfolio aligns with aggressive risk tolerance")

        return recommendations[:5]  # Limit to top 5 recommendations

    def _calculate_optimization_confidence(
        self, optimization_result: Dict[str, Any], returns_data: pd.DataFrame
    ) -> float:
        """Calculate confidence score for optimization results."""
        confidence_factors = []

        # Optimization success
        if optimization_result.get("success"):
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.2)

        # Data quality
        if len(returns_data) > 500:  # ~2 years of data
            confidence_factors.append(0.9)
        elif len(returns_data) > 250:  # ~1 year of data
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)

        # Number of assets (diversification potential)
        num_assets = returns_data.shape[1]
        if num_assets >= 10:
            confidence_factors.append(0.9)
        elif num_assets >= 5:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)

        # Sharpe ratio quality
        if optimization_result.get("success"):
            sharpe_ratio = optimization_result.get("metrics", {}).get("sharpe_ratio", 0)
            if sharpe_ratio > 1.0:
                confidence_factors.append(0.9)
            elif sharpe_ratio > 0.5:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)

        return np.mean(confidence_factors) if confidence_factors else 0.5

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols for portfolio optimization."""
        return [
            # Large Cap US Stocks
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            # Financial Sector
            "JPM",
            "BAC",
            "WFC",
            "GS",
            # Healthcare
            "JNJ",
            "PFE",
            "UNH",
            # ETFs
            "SPY",
            "QQQ",
            "IWM",
            "VTI",
            "BND",
            "TLT",
            "GLD",
            "VNQ",
            # International
            "VEA",
            "VWO",
            "EFA",
            # Commodities
            "USO",
            "DBA",
            # Crypto
            "BTC-USD",
            "ETH-USD",
        ]
