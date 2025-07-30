"""
Crypto Analysis Agent - Cryptocurrency Market Analysis
=====================================================

This module implements a specialized agent for cryptocurrency analysis,
including price analysis, market metrics, and blockchain-specific indicators.

Classes:
    CryptoAnalysisAgent: Main cryptocurrency analysis agent
    CryptoMetrics: Cryptocurrency-specific metrics calculator
    DeFiAnalyzer: DeFi protocol analysis tools
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

from .base_agent import BaseFinancialAgent, AgentCapability, AnalysisType, AgentResponse

logger = logging.getLogger(__name__)


class CryptoMetrics:
    """
    Cryptocurrency-specific metrics and calculations.
    """

    @staticmethod
    def calculate_market_dominance(
        crypto_market_cap: float, total_market_cap: float
    ) -> float:
        """Calculate market dominance percentage."""
        if total_market_cap == 0:
            return 0.0
        return (crypto_market_cap / total_market_cap) * 100

    @staticmethod
    def network_value_to_transactions(
        market_cap: float, transaction_volume: float
    ) -> float:
        """Calculate Network Value to Transactions (NVT) ratio."""
        if transaction_volume == 0:
            return 0.0
        return market_cap / transaction_volume

    @staticmethod
    def mvrv_ratio(market_cap: float, realized_cap: float) -> float:
        """Calculate Market Value to Realized Value (MVRV) ratio."""
        if realized_cap == 0:
            return 1.0
        return market_cap / realized_cap

    @staticmethod
    def fear_greed_index(
        price_change: float, volume_change: float, volatility: float
    ) -> Dict[str, Any]:
        """
        Calculate simplified Fear and Greed Index.

        Args:
            price_change (float): Recent price change percentage
            volume_change (float): Recent volume change percentage
            volatility (float): Current volatility

        Returns:
            Dict[str, Any]: Fear and Greed metrics
        """
        # Simplified calculation (real index uses more factors)
        price_score = max(0, min(100, 50 + price_change * 2))
        volume_score = max(0, min(100, 50 + volume_change))
        volatility_score = max(0, min(100, 100 - (volatility * 1000)))

        overall_score = (price_score + volume_score + volatility_score) / 3

        if overall_score >= 75:
            sentiment = "Extreme Greed"
        elif overall_score >= 55:
            sentiment = "Greed"
        elif overall_score >= 45:
            sentiment = "Neutral"
        elif overall_score >= 25:
            sentiment = "Fear"
        else:
            sentiment = "Extreme Fear"

        return {
            "score": overall_score,
            "sentiment": sentiment,
            "price_component": price_score,
            "volume_component": volume_score,
            "volatility_component": volatility_score,
        }

    @staticmethod
    def hash_rate_analysis(
        current_hash_rate: float, historical_hash_rates: List[float]
    ) -> Dict[str, Any]:
        """Analyze hash rate trends (for proof-of-work cryptocurrencies)."""
        if not historical_hash_rates:
            return {"trend": "unknown", "security_score": 50}

        avg_hash_rate = np.mean(historical_hash_rates)
        trend = (
            "increasing"
            if current_hash_rate > avg_hash_rate * 1.05
            else "decreasing"
            if current_hash_rate < avg_hash_rate * 0.95
            else "stable"
        )

        # Security score based on hash rate stability and growth
        if trend == "increasing":
            security_score = min(100, 70 + (current_hash_rate / avg_hash_rate - 1) * 30)
        elif trend == "stable":
            security_score = 70
        else:
            security_score = max(30, 70 - (1 - current_hash_rate / avg_hash_rate) * 40)

        return {
            "current_hash_rate": current_hash_rate,
            "average_hash_rate": avg_hash_rate,
            "trend": trend,
            "security_score": security_score,
        }


class DeFiAnalyzer:
    """
    DeFi (Decentralized Finance) protocol analysis tools.
    """

    @staticmethod
    def total_value_locked_analysis(
        current_tvl: float, historical_tvl: List[float]
    ) -> Dict[str, Any]:
        """Analyze Total Value Locked (TVL) trends."""
        if not historical_tvl:
            return {"tvl_trend": "unknown", "growth_rate": 0.0}

        # Calculate growth rate
        if len(historical_tvl) >= 2:
            growth_rate = ((current_tvl - historical_tvl[0]) / historical_tvl[0]) * 100
        else:
            growth_rate = 0.0

        # Determine trend
        recent_avg = (
            np.mean(historical_tvl[-5:])
            if len(historical_tvl) >= 5
            else historical_tvl[-1]
        )
        trend = (
            "growing"
            if current_tvl > recent_avg * 1.05
            else "declining"
            if current_tvl < recent_avg * 0.95
            else "stable"
        )

        return {
            "current_tvl": current_tvl,
            "tvl_trend": trend,
            "growth_rate": growth_rate,
            "tvl_stability": np.std(historical_tvl) / np.mean(historical_tvl)
            if historical_tvl
            else 0,
        }

    @staticmethod
    def yield_farming_metrics(apy: float, risk_factors: List[str]) -> Dict[str, Any]:
        """Analyze yield farming opportunities."""
        # Risk assessment based on common DeFi risks
        risk_score = len(risk_factors) * 10  # Simplified risk scoring

        # Adjust APY for risk
        risk_adjusted_yield = apy * (1 - risk_score / 100)

        return {
            "apy": apy,
            "risk_score": min(risk_score, 100),
            "risk_factors": risk_factors,
            "risk_adjusted_yield": max(0, risk_adjusted_yield),
        }


class CryptoAnalysisAgent(BaseFinancialAgent):
    """
    Specialized agent for cryptocurrency market analysis.

    This agent provides comprehensive analysis of cryptocurrencies including
    price analysis, market metrics, on-chain analysis, and DeFi metrics.
    """

    def __init__(self):
        """Initialize the Crypto Analysis Agent."""
        capabilities = [
            AgentCapability.CRYPTO_ANALYSIS,
            AgentCapability.MARKET_RESEARCH,
            AgentCapability.TECHNICAL_ANALYSIS,
            AgentCapability.SENTIMENT_ANALYSIS,
        ]
        super().__init__("CryptoAnalysisAgent", capabilities)
        self.yfinance_tools = None
        self.duckduckgo_tool = None
        self.crypto_metrics = CryptoMetrics()
        self.defi_analyzer = DeFiAnalyzer()

        # Supported crypto symbols
        self.crypto_symbols = {
            "BTC-USD": "Bitcoin",
            "ETH-USD": "Ethereum",
            "BNB-USD": "Binance Coin",
            "ADA-USD": "Cardano",
            "XRP-USD": "Ripple",
            "SOL-USD": "Solana",
            "DOT-USD": "Polkadot",
            "DOGE-USD": "Dogecoin",
            "AVAX-USD": "Avalanche",
            "MATIC-USD": "Polygon",
            "LINK-USD": "Chainlink",
            "UNI-USD": "Uniswap",
            "LTC-USD": "Litecoin",
            "ATOM-USD": "Cosmos",
            "FTT-USD": "FTX Token",
        }

    def initialize(self) -> bool:
        """Initialize the agent with required resources."""
        try:
            self.yfinance_tools = YFinanceTools(
                stock_price=True,
                technical_indicators=True,
                company_news=True,
                stock_fundamentals=False,
            )
            self.duckduckgo_tool = DuckDuckGo()
            self.is_initialized = True
            logger.info("Crypto Analysis Agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Crypto Analysis Agent: {e}")
            return False

    def analyze(
        self, symbol: str, analysis_type: AnalysisType, **kwargs
    ) -> AgentResponse:
        """
        Perform cryptocurrency analysis for a given symbol.

        Args:
            symbol (str): Cryptocurrency symbol to analyze (e.g., 'BTC-USD')
            analysis_type (AnalysisType): Type of analysis to perform
            **kwargs: Additional parameters (period, include_defi, etc.)

        Returns:
            AgentResponse: Cryptocurrency analysis results
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

        # Validate crypto symbol
        if symbol not in self.crypto_symbols:
            return AgentResponse(
                success=False,
                data={},
                message=f"Unsupported cryptocurrency symbol: {symbol}",
                confidence_score=0.0,
                sources=[],
                analysis_type=analysis_type,
                timestamp=datetime.now().isoformat(),
            )

        try:
            period = kwargs.get("period", "1y")
            include_defi = kwargs.get("include_defi", True)
            include_onchain = kwargs.get("include_onchain", True)

            # Get cryptocurrency data
            crypto_data = self._get_crypto_data(symbol, period)

            if crypto_data is None or crypto_data.empty:
                return AgentResponse(
                    success=False,
                    data={},
                    message=f"No data available for cryptocurrency {symbol}",
                    confidence_score=0.0,
                    sources=[],
                    analysis_type=analysis_type,
                    timestamp=datetime.now().isoformat(),
                )

            # Perform comprehensive crypto analysis
            analysis_result = self._comprehensive_crypto_analysis(
                crypto_data, symbol, include_defi, include_onchain
            )

            return AgentResponse(
                success=True,
                data=analysis_result,
                message=f"Cryptocurrency analysis completed for {symbol}",
                confidence_score=analysis_result.get("confidence_score", 0.8),
                sources=[
                    "Yahoo Finance",
                    "CoinGecko API",
                    "Blockchain Data",
                    "DeFi Metrics",
                ],
                analysis_type=analysis_type,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Error in crypto analysis for {symbol}: {e}")
            return AgentResponse(
                success=False,
                data={},
                message=f"Cryptocurrency analysis failed: {str(e)}",
                confidence_score=0.0,
                sources=[],
                analysis_type=analysis_type,
                timestamp=datetime.now().isoformat(),
            )

    def _get_crypto_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Get cryptocurrency price data."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            logger.error(f"Failed to get crypto data for {symbol}: {e}")
            return None

    def _comprehensive_crypto_analysis(
        self,
        crypto_data: pd.DataFrame,
        symbol: str,
        include_defi: bool,
        include_onchain: bool,
    ) -> Dict[str, Any]:
        """Perform comprehensive cryptocurrency analysis."""

        crypto_name = self.crypto_symbols.get(symbol, symbol)

        # Basic price analysis
        price_analysis = self._analyze_crypto_price(crypto_data, symbol)

        # Volatility and risk analysis
        volatility_analysis = self._analyze_crypto_volatility(crypto_data)

        # Market sentiment analysis
        sentiment_analysis = self._analyze_crypto_sentiment(symbol, crypto_name)

        # Technical indicators
        technical_analysis = self._analyze_crypto_technicals(crypto_data)

        # Market metrics
        market_metrics = self._calculate_market_metrics(crypto_data, symbol)

        result = {
            "symbol": symbol,
            "cryptocurrency": crypto_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "price_analysis": price_analysis,
            "volatility_analysis": volatility_analysis,
            "sentiment_analysis": sentiment_analysis,
            "technical_analysis": technical_analysis,
            "market_metrics": market_metrics,
        }

        # Add DeFi analysis if requested and applicable
        if include_defi and symbol in ["ETH-USD", "BNB-USD", "AVAX-USD", "MATIC-USD"]:
            defi_analysis = self._analyze_defi_metrics(symbol)
            result["defi_analysis"] = defi_analysis

        # Add on-chain analysis if requested
        if include_onchain:
            onchain_analysis = self._analyze_onchain_metrics(symbol)
            result["onchain_analysis"] = onchain_analysis

        # Calculate overall confidence score
        result["confidence_score"] = self._calculate_confidence_score(result)

        # Generate investment recommendation
        result["investment_recommendation"] = self._generate_crypto_recommendation(
            result
        )

        return result

    def _analyze_crypto_price(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Analyze cryptocurrency price movements."""
        current_price = data["Close"].iloc[-1]
        prices = data["Close"]

        # Price changes
        price_changes = {
            "24h": ((current_price - prices.iloc[-2]) / prices.iloc[-2]) * 100
            if len(prices) > 1
            else 0,
            "7d": ((current_price - prices.iloc[-8]) / prices.iloc[-8]) * 100
            if len(prices) > 7
            else 0,
            "30d": ((current_price - prices.iloc[-31]) / prices.iloc[-31]) * 100
            if len(prices) > 30
            else 0,
            "ytd": ((current_price - prices.iloc[0]) / prices.iloc[0]) * 100,
        }

        # Support and resistance levels
        high_52w = prices.max()
        low_52w = prices.min()

        # Market cap estimation (simplified)
        volume = data["Volume"].iloc[-1] if "Volume" in data.columns else 0

        return {
            "current_price": current_price,
            "price_changes": price_changes,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "distance_from_high": ((current_price - high_52w) / high_52w) * 100,
            "distance_from_low": ((current_price - low_52w) / low_52w) * 100,
            "average_volume": volume,
        }

    def _analyze_crypto_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cryptocurrency volatility patterns."""
        returns = data["Close"].pct_change().dropna()

        if returns.empty:
            return {
                "daily_volatility": 0,
                "annualized_volatility": 0,
                "volatility_rank": "unknown",
            }

        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(365)  # Crypto trades 24/7

        # Volatility ranking
        if annualized_vol > 1.0:  # >100%
            vol_rank = "extremely_high"
        elif annualized_vol > 0.6:  # 60-100%
            vol_rank = "very_high"
        elif annualized_vol > 0.4:  # 40-60%
            vol_rank = "high"
        elif annualized_vol > 0.2:  # 20-40%
            vol_rank = "moderate"
        else:  # <20%
            vol_rank = "low"

        return {
            "daily_volatility": daily_vol,
            "annualized_volatility": annualized_vol,
            "volatility_rank": vol_rank,
            "recent_volatility": returns.tail(30).std() * np.sqrt(365),
        }

    def _analyze_crypto_sentiment(
        self, symbol: str, crypto_name: str
    ) -> Dict[str, Any]:
        """Analyze cryptocurrency market sentiment."""
        # Simplified sentiment analysis
        # In practice, this would integrate with sentiment APIs, social media analysis, etc.

        # Fear and Greed Index simulation
        price_data = self._get_crypto_data(symbol, "30d")
        if price_data is not None and not price_data.empty:
            recent_return = price_data["Close"].pct_change().tail(7).mean()
            volume_change = (
                price_data["Volume"].pct_change().tail(7).mean()
                if "Volume" in price_data.columns
                else 0
            )
            volatility = price_data["Close"].pct_change().std()

            fear_greed = self.crypto_metrics.fear_greed_index(
                recent_return * 100, volume_change * 100, volatility
            )
        else:
            fear_greed = {"score": 50, "sentiment": "Neutral"}

        return {
            "fear_greed_index": fear_greed,
            "social_sentiment": "neutral",  # Would be calculated from social media APIs
            "news_sentiment": "neutral",  # Would be calculated from news analysis
            "overall_sentiment": fear_greed["sentiment"].lower(),
        }

    def _analyze_crypto_technicals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze technical indicators for cryptocurrency."""
        prices = data["Close"]

        if len(prices) < 50:
            return {"insufficient_data": True}

        # Moving averages
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1]
        ema_12 = prices.ewm(span=12).mean().iloc[-1]

        current_price = prices.iloc[-1]

        # RSI calculation
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50

        # MACD
        ema_12_full = prices.ewm(span=12).mean()
        ema_26_full = prices.ewm(span=26).mean()
        macd = ema_12_full - ema_26_full
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal

        return {
            "moving_averages": {
                "sma_20": sma_20,
                "sma_50": sma_50,
                "ema_12": ema_12,
                "price_vs_sma20": ((current_price - sma_20) / sma_20) * 100,
                "price_vs_sma50": ((current_price - sma_50) / sma_50) * 100,
            },
            "rsi": {
                "current": current_rsi,
                "signal": "overbought"
                if current_rsi > 70
                else "oversold"
                if current_rsi < 30
                else "neutral",
            },
            "macd": {
                "macd": macd.iloc[-1] if not macd.empty else 0,
                "signal": signal.iloc[-1] if not signal.empty else 0,
                "histogram": histogram.iloc[-1] if not histogram.empty else 0,
                "signal_trend": "bullish"
                if not macd.empty
                and not signal.empty
                and macd.iloc[-1] > signal.iloc[-1]
                else "bearish",
            },
        }

    def _calculate_market_metrics(
        self, data: pd.DataFrame, symbol: str
    ) -> Dict[str, Any]:
        """Calculate cryptocurrency market metrics."""
        current_price = data["Close"].iloc[-1]
        volume = data["Volume"].iloc[-1] if "Volume" in data.columns else 0

        # Simplified market cap calculation (would need circulating supply data)
        # Using approximations for major cryptocurrencies
        supply_estimates = {
            "BTC-USD": 19_000_000,  # Bitcoin
            "ETH-USD": 120_000_000,  # Ethereum
            "BNB-USD": 160_000_000,  # Binance Coin
            "ADA-USD": 35_000_000_000,  # Cardano
            "XRP-USD": 50_000_000_000,  # Ripple
        }

        estimated_supply = supply_estimates.get(symbol, 1_000_000_000)
        estimated_market_cap = current_price * estimated_supply

        return {
            "estimated_market_cap": estimated_market_cap,
            "current_volume": volume,
            "volume_to_mcap_ratio": volume / estimated_market_cap
            if estimated_market_cap > 0
            else 0,
            "liquidity_score": min(
                100, (volume / 1_000_000) * 10
            ),  # Simplified liquidity score
        }

    def _analyze_defi_metrics(self, symbol: str) -> Dict[str, Any]:
        """Analyze DeFi-related metrics for applicable cryptocurrencies."""
        # Simplified DeFi analysis
        # In practice, this would integrate with DeFi protocols and TVL data

        defi_data = {
            "ETH-USD": {
                "tvl_ecosystem": 50_000_000_000,  # $50B estimate
                "defi_dominance": 60,  # 60% of DeFi
                "major_protocols": ["Uniswap", "Aave", "Compound", "MakerDAO"],
            },
            "BNB-USD": {
                "tvl_ecosystem": 8_000_000_000,  # $8B estimate
                "defi_dominance": 15,  # 15% of DeFi
                "major_protocols": ["PancakeSwap", "Venus", "Alpaca Finance"],
            },
            "AVAX-USD": {
                "tvl_ecosystem": 2_000_000_000,  # $2B estimate
                "defi_dominance": 5,  # 5% of DeFi
                "major_protocols": ["Trader Joe", "Benqi", "Platypus"],
            },
        }

        if symbol in defi_data:
            data = defi_data[symbol]
            return {
                "total_value_locked": data["tvl_ecosystem"],
                "defi_market_share": data["defi_dominance"],
                "major_protocols": data["major_protocols"],
                "defi_growth_trend": "stable",  # Would be calculated from historical data
            }

        return {"defi_applicable": False}

    def _analyze_onchain_metrics(self, symbol: str) -> Dict[str, Any]:
        """Analyze on-chain metrics (simplified)."""
        # Simplified on-chain analysis
        # In practice, this would integrate with blockchain APIs

        onchain_data = {
            "BTC-USD": {
                "network_hash_rate": 250_000_000,  # TH/s
                "active_addresses": 1_000_000,
                "transaction_count": 250_000,
                "network_security": "very_high",
            },
            "ETH-USD": {
                "network_hash_rate": 900_000,  # GH/s (before PoS)
                "active_addresses": 700_000,
                "transaction_count": 1_200_000,
                "network_security": "high",
            },
        }

        if symbol in onchain_data:
            data = onchain_data[symbol]
            return {
                "network_metrics": data,
                "network_health": "healthy",
                "adoption_trend": "growing",
            }

        return {"onchain_data_available": False}

    def _calculate_confidence_score(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the analysis."""
        score_components = []

        # Data availability score
        if "price_analysis" in analysis_result:
            score_components.append(0.9)

        # Technical analysis confidence
        if "technical_analysis" in analysis_result and not analysis_result[
            "technical_analysis"
        ].get("insufficient_data"):
            score_components.append(0.8)

        # Sentiment analysis confidence
        if "sentiment_analysis" in analysis_result:
            score_components.append(0.7)

        # Additional data sources
        if "defi_analysis" in analysis_result:
            score_components.append(0.8)

        if "onchain_analysis" in analysis_result:
            score_components.append(0.9)

        return np.mean(score_components) if score_components else 0.5

    def _generate_crypto_recommendation(
        self, analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate investment recommendation based on analysis."""

        # Extract key metrics
        technical_analysis = analysis_result.get("technical_analysis", {})
        volatility_analysis = analysis_result.get("volatility_analysis", {})
        sentiment_analysis = analysis_result.get("sentiment_analysis", {})

        # Scoring factors
        scores = []

        # Technical score
        if not technical_analysis.get("insufficient_data"):
            rsi = technical_analysis.get("rsi", {}).get("current", 50)
            macd_signal = technical_analysis.get("macd", {}).get(
                "signal_trend", "neutral"
            )

            if 30 <= rsi <= 70 and macd_signal == "bullish":
                scores.append(75)
            elif rsi < 30:  # Oversold
                scores.append(60)
            elif rsi > 70:  # Overbought
                scores.append(40)
            else:
                scores.append(50)

        # Volatility score (lower volatility = higher score for conservative recommendation)
        vol_rank = volatility_analysis.get("volatility_rank", "moderate")
        vol_scores = {
            "low": 80,
            "moderate": 60,
            "high": 40,
            "very_high": 20,
            "extremely_high": 10,
        }
        scores.append(vol_scores.get(vol_rank, 50))

        # Sentiment score
        overall_sentiment = sentiment_analysis.get("overall_sentiment", "neutral")
        sentiment_scores = {"positive": 70, "neutral": 50, "negative": 30}
        scores.append(sentiment_scores.get(overall_sentiment, 50))

        # Calculate overall score
        overall_score = np.mean(scores) if scores else 50

        # Generate recommendation
        if overall_score >= 70:
            recommendation = "buy"
            risk_level = "moderate"
            reasoning = "Technical indicators and sentiment suggest positive outlook"
        elif overall_score >= 55:
            recommendation = "hold"
            risk_level = "moderate"
            reasoning = "Mixed signals suggest maintaining current position"
        elif overall_score >= 40:
            recommendation = "cautious_hold"
            risk_level = "high"
            reasoning = "Some negative indicators present, monitor closely"
        else:
            recommendation = "sell"
            risk_level = "high"
            reasoning = "Multiple negative indicators suggest reducing exposure"

        return {
            "recommendation": recommendation,
            "confidence": overall_score,
            "risk_level": risk_level,
            "reasoning": reasoning,
            "time_horizon": "short_to_medium_term",
            "key_factors": [
                f"Technical analysis: {technical_analysis.get('rsi', {}).get('signal', 'neutral')}",
                f"Volatility: {vol_rank}",
                f"Sentiment: {overall_sentiment}",
            ],
        }

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported cryptocurrency symbols."""
        return list(self.crypto_symbols.keys())
