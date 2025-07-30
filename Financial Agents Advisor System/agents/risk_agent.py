"""
Risk Assessment Agent - Comprehensive Financial Risk Analysis
============================================================

This module implements a specialized agent for assessing various types of
financial risks including market risk, credit risk, operational risk, and more.

Classes:
    RiskAssessmentAgent: Main risk assessment agent
    RiskCalculator: Risk metrics and calculations
    PortfolioRiskAnalyzer: Portfolio-level risk analysis
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from phi.tools.yfinance import YFinanceTools

from .base_agent import BaseFinancialAgent, AgentCapability, AnalysisType, AgentResponse

logger = logging.getLogger(__name__)


class RiskCalculator:
    """
    Collection of risk calculation methods for financial analysis.
    """

    @staticmethod
    def value_at_risk(
        returns: pd.Series, confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) using historical simulation method.

        Args:
            returns (pd.Series): Historical returns
            confidence_level (float): Confidence level (default 0.95)

        Returns:
            Dict[str, float]: VaR metrics
        """
        if returns.empty:
            return {"var_95": 0.0, "var_99": 0.0, "expected_shortfall": 0.0}

        sorted_returns = returns.sort_values()

        # Calculate VaR at different confidence levels
        var_95 = np.percentile(sorted_returns, 5)  # 95% VaR
        var_99 = np.percentile(sorted_returns, 1)  # 99% VaR

        # Calculate Expected Shortfall (Conditional VaR)
        var_threshold = np.percentile(sorted_returns, (1 - confidence_level) * 100)
        expected_shortfall = sorted_returns[sorted_returns <= var_threshold].mean()

        return {
            "var_95": abs(var_95),
            "var_99": abs(var_99),
            "expected_shortfall": abs(expected_shortfall),
            "confidence_level": confidence_level,
        }

    @staticmethod
    def volatility_metrics(returns: pd.Series) -> Dict[str, float]:
        """
        Calculate various volatility metrics.

        Args:
            returns (pd.Series): Historical returns

        Returns:
            Dict[str, float]: Volatility metrics
        """
        if returns.empty:
            return {"daily_vol": 0.0, "annualized_vol": 0.0, "downside_vol": 0.0}

        # Daily volatility
        daily_vol = returns.std()

        # Annualized volatility (assuming 252 trading days)
        annualized_vol = daily_vol * np.sqrt(252)

        # Downside volatility (volatility of negative returns only)
        negative_returns = returns[returns < 0]
        downside_vol = (
            negative_returns.std() * np.sqrt(252) if not negative_returns.empty else 0.0
        )

        return {
            "daily_volatility": daily_vol,
            "annualized_volatility": annualized_vol,
            "downside_volatility": downside_vol,
        }

    @staticmethod
    def beta_calculation(
        stock_returns: pd.Series, market_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate beta and related risk metrics.

        Args:
            stock_returns (pd.Series): Stock returns
            market_returns (pd.Series): Market returns (benchmark)

        Returns:
            Dict[str, float]: Beta and related metrics
        """
        if stock_returns.empty or market_returns.empty:
            return {"beta": 1.0, "alpha": 0.0, "correlation": 0.0, "r_squared": 0.0}

        # Align the series
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        if aligned_data.shape[0] < 2:
            return {"beta": 1.0, "alpha": 0.0, "correlation": 0.0, "r_squared": 0.0}

        stock_aligned = aligned_data.iloc[:, 0]
        market_aligned = aligned_data.iloc[:, 1]

        # Calculate beta
        covariance = np.cov(stock_aligned, market_aligned)[0, 1]
        market_variance = np.var(market_aligned)
        beta = covariance / market_variance if market_variance != 0 else 1.0

        # Calculate alpha
        stock_mean = stock_aligned.mean()
        market_mean = market_aligned.mean()
        alpha = stock_mean - (beta * market_mean)

        # Calculate correlation and R-squared
        correlation = np.corrcoef(stock_aligned, market_aligned)[0, 1]
        r_squared = correlation**2

        return {
            "beta": beta,
            "alpha": alpha,
            "correlation": correlation,
            "r_squared": r_squared,
        }

    @staticmethod
    def maximum_drawdown(prices: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics.

        Args:
            prices (pd.Series): Price series

        Returns:
            Dict[str, float]: Drawdown metrics
        """
        if prices.empty:
            return {"max_drawdown": 0.0, "current_drawdown": 0.0, "recovery_time": 0}

        # Calculate running maximum
        running_max = prices.expanding().max()

        # Calculate drawdown
        drawdown = (prices - running_max) / running_max

        # Maximum drawdown
        max_drawdown = drawdown.min()

        # Current drawdown
        current_drawdown = drawdown.iloc[-1]

        # Calculate recovery time (days to recover from max drawdown)
        max_dd_idx = drawdown.idxmin()
        recovery_time = 0

        if max_dd_idx in prices.index:
            post_max_dd = prices[prices.index > max_dd_idx]
            peak_before_dd = running_max.loc[max_dd_idx]

            recovery_idx = post_max_dd[post_max_dd >= peak_before_dd].index
            if not recovery_idx.empty:
                recovery_time = (recovery_idx[0] - max_dd_idx).days

        return {
            "max_drawdown": abs(max_drawdown),
            "current_drawdown": abs(current_drawdown),
            "recovery_time_days": recovery_time,
        }


class PortfolioRiskAnalyzer:
    """
    Portfolio-level risk analysis tools.
    """

    def __init__(self):
        """Initialize portfolio risk analyzer."""
        self.risk_calculator = RiskCalculator()

    def calculate_portfolio_risk(
        self, weights: List[float], returns_matrix: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate portfolio risk metrics.

        Args:
            weights (List[float]): Portfolio weights
            returns_matrix (pd.DataFrame): Historical returns matrix

        Returns:
            Dict[str, Any]: Portfolio risk metrics
        """
        if returns_matrix.empty or len(weights) != returns_matrix.shape[1]:
            return {"portfolio_volatility": 0.0, "diversification_ratio": 0.0}

        weights = np.array(weights)

        # Calculate portfolio returns
        portfolio_returns = (returns_matrix * weights).sum(axis=1)

        # Portfolio volatility
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)

        # Individual asset volatilities
        individual_vols = returns_matrix.std() * np.sqrt(252)
        weighted_avg_vol = (weights * individual_vols).sum()

        # Diversification ratio
        diversification_ratio = (
            weighted_avg_vol / portfolio_vol if portfolio_vol != 0 else 1.0
        )

        # Portfolio VaR
        portfolio_var = self.risk_calculator.value_at_risk(portfolio_returns)

        return {
            "portfolio_volatility": portfolio_vol,
            "weighted_average_volatility": weighted_avg_vol,
            "diversification_ratio": diversification_ratio,
            "portfolio_var": portfolio_var,
            "portfolio_returns_std": portfolio_returns.std(),
        }

    def stress_test_scenarios(self, prices: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Perform stress testing under various market scenarios.

        Args:
            prices (pd.Series): Historical price series

        Returns:
            Dict[str, Dict[str, float]]: Stress test results
        """
        if prices.empty:
            return {}

        current_price = prices.iloc[-1]
        scenarios = {}

        # Market crash scenario (-20%)
        crash_price = current_price * 0.8
        scenarios["market_crash"] = {
            "price_change": -0.2,
            "new_price": crash_price,
            "loss_amount": current_price - crash_price,
        }

        # Severe market crash scenario (-40%)
        severe_crash_price = current_price * 0.6
        scenarios["severe_crash"] = {
            "price_change": -0.4,
            "new_price": severe_crash_price,
            "loss_amount": current_price - severe_crash_price,
        }

        # Volatility spike scenario (based on historical volatility)
        returns = prices.pct_change().dropna()
        if not returns.empty:
            hist_vol = returns.std()
            vol_spike_change = -3 * hist_vol  # 3 standard deviations down
            vol_spike_price = current_price * (1 + vol_spike_change)

            scenarios["volatility_spike"] = {
                "price_change": vol_spike_change,
                "new_price": vol_spike_price,
                "loss_amount": current_price - vol_spike_price,
            }

        return scenarios


class RiskAssessmentAgent(BaseFinancialAgent):
    """
    Specialized agent for comprehensive financial risk assessment.

    This agent evaluates various types of financial risks including market risk,
    volatility risk, liquidity risk, and systematic risk.
    """

    def __init__(self):
        """Initialize the Risk Assessment Agent."""
        capabilities = [
            AgentCapability.RISK_ASSESSMENT,
            AgentCapability.STOCK_ANALYSIS,
            AgentCapability.PORTFOLIO_OPTIMIZATION,
        ]
        super().__init__("RiskAssessmentAgent", capabilities)
        self.yfinance_tools = None
        self.risk_calculator = RiskCalculator()
        self.portfolio_analyzer = PortfolioRiskAnalyzer()

    def initialize(self) -> bool:
        """Initialize the agent with required resources."""
        try:
            self.yfinance_tools = YFinanceTools(
                stock_price=True,
                technical_indicators=True,
                stock_fundamentals=True,
                company_news=False,
            )
            self.is_initialized = True
            logger.info("Risk Assessment Agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Risk Assessment Agent: {e}")
            return False

    def analyze(
        self, symbol: str, analysis_type: AnalysisType, **kwargs
    ) -> AgentResponse:
        """
        Perform risk assessment for a given symbol.

        Args:
            symbol (str): Financial symbol to analyze
            analysis_type (AnalysisType): Type of analysis to perform
            **kwargs: Additional parameters (period, benchmark, etc.)

        Returns:
            AgentResponse: Risk assessment results
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
            period = kwargs.get(
                "period", "2y"
            )  # 2 years of data for better risk assessment
            benchmark = kwargs.get("benchmark", "^GSPC")  # S&P 500 as default benchmark

            # Get historical data
            stock_data = self._get_historical_data(symbol, period)
            benchmark_data = self._get_historical_data(benchmark, period)

            if stock_data is None or stock_data.empty:
                return AgentResponse(
                    success=False,
                    data={},
                    message=f"No data available for symbol {symbol}",
                    confidence_score=0.0,
                    sources=[],
                    analysis_type=analysis_type,
                    timestamp=datetime.now().isoformat(),
                )

            # Perform risk analysis
            analysis_result = self._comprehensive_risk_analysis(
                stock_data, benchmark_data, symbol, benchmark
            )

            return AgentResponse(
                success=True,
                data=analysis_result,
                message=f"Risk assessment completed for {symbol}",
                confidence_score=analysis_result.get("overall_risk_score", 0.8),
                sources=["Yahoo Finance", "Historical Price Data", "Market Benchmarks"],
                analysis_type=analysis_type,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Error in risk assessment for {symbol}: {e}")
            return AgentResponse(
                success=False,
                data={},
                message=f"Risk assessment failed: {str(e)}",
                confidence_score=0.0,
                sources=[],
                analysis_type=analysis_type,
                timestamp=datetime.now().isoformat(),
            )

    def _get_historical_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Get historical price data for a symbol."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None

    def _comprehensive_risk_analysis(
        self,
        stock_data: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        symbol: str,
        benchmark: str,
    ) -> Dict[str, Any]:
        """Perform comprehensive risk analysis."""

        # Calculate returns
        stock_returns = stock_data["Close"].pct_change().dropna()
        stock_prices = stock_data["Close"]

        benchmark_returns = pd.Series()
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_returns = benchmark_data["Close"].pct_change().dropna()

        # Market risk metrics
        market_risk = self._calculate_market_risk(stock_returns, benchmark_returns)

        # Volatility analysis
        volatility_analysis = self._analyze_volatility(stock_returns, stock_prices)

        # Liquidity risk (simplified)
        liquidity_risk = self._assess_liquidity_risk(stock_data)

        # Drawdown analysis
        drawdown_analysis = self.risk_calculator.maximum_drawdown(stock_prices)

        # Value at Risk
        var_analysis = self.risk_calculator.value_at_risk(stock_returns)

        # Stress testing
        stress_test = self.portfolio_analyzer.stress_test_scenarios(stock_prices)

        # Overall risk score calculation
        overall_risk_score = self._calculate_overall_risk_score(
            market_risk, volatility_analysis, drawdown_analysis, var_analysis
        )

        # Risk classification
        risk_classification = self._classify_risk_level(overall_risk_score)

        return {
            "symbol": symbol,
            "analysis_period": stock_data.index[0].strftime("%Y-%m-%d")
            + " to "
            + stock_data.index[-1].strftime("%Y-%m-%d"),
            "market_risk": market_risk,
            "volatility_analysis": volatility_analysis,
            "liquidity_risk": liquidity_risk,
            "drawdown_analysis": drawdown_analysis,
            "value_at_risk": var_analysis,
            "stress_testing": stress_test,
            "overall_risk_score": overall_risk_score,
            "risk_classification": risk_classification,
            "recommendations": self._generate_risk_recommendations(
                risk_classification, overall_risk_score
            ),
        }

    def _calculate_market_risk(
        self, stock_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Dict[str, Any]:
        """Calculate market risk metrics."""
        if benchmark_returns.empty:
            return {
                "beta": 1.0,
                "alpha": 0.0,
                "correlation": 0.0,
                "systematic_risk": "unknown",
            }

        beta_metrics = self.risk_calculator.beta_calculation(
            stock_returns, benchmark_returns
        )

        # Classify systematic risk based on beta
        beta = beta_metrics["beta"]
        if beta > 1.2:
            systematic_risk = "high"
        elif beta > 0.8:
            systematic_risk = "moderate"
        else:
            systematic_risk = "low"

        beta_metrics["systematic_risk"] = systematic_risk
        return beta_metrics

    def _analyze_volatility(
        self, returns: pd.Series, prices: pd.Series
    ) -> Dict[str, Any]:
        """Analyze volatility patterns."""
        volatility_metrics = self.risk_calculator.volatility_metrics(returns)

        # Volatility classification
        annualized_vol = volatility_metrics["annualized_volatility"]
        if annualized_vol > 0.4:  # 40%+
            vol_classification = "very_high"
        elif annualized_vol > 0.25:  # 25-40%
            vol_classification = "high"
        elif annualized_vol > 0.15:  # 15-25%
            vol_classification = "moderate"
        else:  # <15%
            vol_classification = "low"

        # Volatility trend (comparing recent vs historical)
        recent_returns = returns.tail(63)  # Last quarter
        historical_returns = returns.head(-63)  # Earlier data

        recent_vol = (
            recent_returns.std() * np.sqrt(252) if not recent_returns.empty else 0
        )
        historical_vol = (
            historical_returns.std() * np.sqrt(252)
            if not historical_returns.empty
            else 0
        )

        vol_trend = (
            "increasing"
            if recent_vol > historical_vol * 1.1
            else "decreasing"
            if recent_vol < historical_vol * 0.9
            else "stable"
        )

        volatility_metrics.update(
            {
                "volatility_classification": vol_classification,
                "recent_volatility": recent_vol,
                "historical_volatility": historical_vol,
                "volatility_trend": vol_trend,
            }
        )

        return volatility_metrics

    def _assess_liquidity_risk(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess liquidity risk based on volume data."""
        volume = stock_data["Volume"]

        if volume.empty:
            return {"liquidity_risk": "unknown", "average_volume": 0}

        avg_volume = volume.mean()
        recent_volume = volume.tail(20).mean()  # Last 20 days

        # Volume trend
        volume_trend = (
            "increasing"
            if recent_volume > avg_volume * 1.1
            else "decreasing"
            if recent_volume < avg_volume * 0.9
            else "stable"
        )

        # Liquidity classification (simplified)
        if avg_volume > 1_000_000:
            liquidity_risk = "low"
        elif avg_volume > 100_000:
            liquidity_risk = "moderate"
        else:
            liquidity_risk = "high"

        return {
            "liquidity_risk": liquidity_risk,
            "average_volume": avg_volume,
            "recent_volume": recent_volume,
            "volume_trend": volume_trend,
        }

    def _calculate_overall_risk_score(
        self,
        market_risk: Dict,
        volatility_analysis: Dict,
        drawdown_analysis: Dict,
        var_analysis: Dict,
    ) -> float:
        """Calculate overall risk score (0-1, where 1 is highest risk)."""

        # Beta component (0-0.3)
        beta = market_risk.get("beta", 1.0)
        beta_score = min(abs(beta - 1) * 0.3, 0.3)

        # Volatility component (0-0.3)
        annualized_vol = volatility_analysis.get("annualized_volatility", 0.2)
        vol_score = min(annualized_vol, 0.3)

        # Drawdown component (0-0.2)
        max_drawdown = drawdown_analysis.get("max_drawdown", 0.1)
        drawdown_score = min(max_drawdown, 0.2)

        # VaR component (0-0.2)
        var_95 = var_analysis.get("var_95", 0.02)
        var_score = min(var_95 * 10, 0.2)  # Scale VaR

        overall_score = beta_score + vol_score + drawdown_score + var_score
        return min(overall_score, 1.0)

    def _classify_risk_level(self, risk_score: float) -> Dict[str, str]:
        """Classify overall risk level."""
        if risk_score > 0.7:
            level = "very_high"
            description = "Very High Risk - Suitable only for aggressive investors"
        elif risk_score > 0.5:
            level = "high"
            description = (
                "High Risk - Suitable for aggressive investors with high risk tolerance"
            )
        elif risk_score > 0.3:
            level = "moderate"
            description = "Moderate Risk - Suitable for balanced investment strategies"
        elif risk_score > 0.15:
            level = "low"
            description = "Low Risk - Suitable for conservative investors"
        else:
            level = "very_low"
            description = "Very Low Risk - Suitable for highly conservative investors"

        return {"level": level, "description": description, "score": risk_score}

    def _generate_risk_recommendations(
        self, risk_classification: Dict, risk_score: float
    ) -> List[str]:
        """Generate risk-based recommendations."""
        recommendations = []

        level = risk_classification["level"]

        if level in ["very_high", "high"]:
            recommendations.extend(
                [
                    "Consider position sizing carefully due to high volatility",
                    "Implement stop-loss orders to limit downside risk",
                    "Diversify portfolio to reduce concentration risk",
                    "Monitor market conditions closely",
                ]
            )
        elif level == "moderate":
            recommendations.extend(
                [
                    "Suitable for balanced portfolio allocation",
                    "Consider dollar-cost averaging for entry",
                    "Regular portfolio rebalancing recommended",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Good for conservative portfolio base",
                    "Consider as defensive holding",
                    "Suitable for income-focused strategies",
                ]
            )

        # Add specific recommendations based on risk score
        if risk_score > 0.6:
            recommendations.append(
                "Consider reducing position size due to elevated risk metrics"
            )

        return recommendations

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols for risk assessment."""
        return [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "JPM",
            "BAC",
            "WFC",
            "GS",
            "JNJ",
            "PFE",
            "UNH",
            "SPY",
            "QQQ",
            "IWM",
            "VTI",
            "BTC-USD",
            "ETH-USD",
        ]
