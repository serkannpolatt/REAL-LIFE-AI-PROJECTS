"""
Technical Analysis Agent - Advanced Technical Analysis and Chart Patterns
========================================================================

This module implements a specialized agent for technical analysis of financial instruments.
It provides technical indicators, chart pattern recognition, and trading signals.

Classes:
    TechnicalAnalysisAgent: Main technical analysis agent
    TechnicalIndicators: Collection of technical indicators
    ChartPatterns: Chart pattern recognition
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from phi.tools.yfinance import YFinanceTools

from .base_agent import BaseFinancialAgent, AgentCapability, AnalysisType, AgentResponse

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Collection of technical indicators for financial analysis.
    """

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(
        data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line

        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    @staticmethod
    def bollinger_bands(
        data: pd.Series, period: int = 20, std_dev: int = 2
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()

        return {
            "upper": sma + (std * std_dev),
            "middle": sma,
            "lower": sma - (std * std_dev),
        }

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return {"k_percent": k_percent, "d_percent": d_percent}

    @staticmethod
    def support_resistance_levels(
        data: pd.Series, window: int = 5
    ) -> Dict[str, List[float]]:
        """Identify support and resistance levels."""
        highs = data.rolling(window=window, center=True).max()
        lows = data.rolling(window=window, center=True).min()

        resistance_levels = []
        support_levels = []

        for i in range(len(data)):
            if data.iloc[i] == highs.iloc[i] and not pd.isna(highs.iloc[i]):
                resistance_levels.append(data.iloc[i])
            if data.iloc[i] == lows.iloc[i] and not pd.isna(lows.iloc[i]):
                support_levels.append(data.iloc[i])

        return {
            "resistance": sorted(list(set(resistance_levels)), reverse=True)[:5],
            "support": sorted(list(set(support_levels)))[:5],
        }


class ChartPatterns:
    """
    Chart pattern recognition algorithms.
    """

    @staticmethod
    def detect_head_shoulders(data: pd.Series, window: int = 5) -> Dict[str, Any]:
        """Detect Head and Shoulders pattern."""
        peaks = []
        troughs = []

        for i in range(window, len(data) - window):
            # Peak detection
            if all(
                data.iloc[i] > data.iloc[i - j] for j in range(1, window + 1)
            ) and all(data.iloc[i] > data.iloc[i + j] for j in range(1, window + 1)):
                peaks.append((i, data.iloc[i]))

            # Trough detection
            if all(
                data.iloc[i] < data.iloc[i - j] for j in range(1, window + 1)
            ) and all(data.iloc[i] < data.iloc[i + j] for j in range(1, window + 1)):
                troughs.append((i, data.iloc[i]))

        # Head and shoulders pattern requires 3 peaks
        if len(peaks) >= 3:
            for i in range(len(peaks) - 2):
                left_shoulder = peaks[i][1]
                head = peaks[i + 1][1]
                right_shoulder = peaks[i + 2][1]

                # Check if middle peak (head) is higher than shoulders
                if head > left_shoulder and head > right_shoulder:
                    # Check if shoulders are approximately equal
                    if abs(left_shoulder - right_shoulder) / left_shoulder < 0.05:
                        return {
                            "pattern_found": True,
                            "type": "head_and_shoulders",
                            "left_shoulder": left_shoulder,
                            "head": head,
                            "right_shoulder": right_shoulder,
                            "confidence": 0.8,
                        }

        return {"pattern_found": False}

    @staticmethod
    def detect_double_top_bottom(data: pd.Series, window: int = 5) -> Dict[str, Any]:
        """Detect Double Top/Bottom patterns."""
        peaks = []
        troughs = []

        for i in range(window, len(data) - window):
            # Peak detection
            if all(
                data.iloc[i] > data.iloc[i - j] for j in range(1, window + 1)
            ) and all(data.iloc[i] > data.iloc[i + j] for j in range(1, window + 1)):
                peaks.append((i, data.iloc[i]))

            # Trough detection
            if all(
                data.iloc[i] < data.iloc[i - j] for j in range(1, window + 1)
            ) and all(data.iloc[i] < data.iloc[i + j] for j in range(1, window + 1)):
                troughs.append((i, data.iloc[i]))

        # Double top pattern
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                peak1 = peaks[i][1]
                peak2 = peaks[i + 1][1]

                if abs(peak1 - peak2) / peak1 < 0.03:  # Peaks are approximately equal
                    return {
                        "pattern_found": True,
                        "type": "double_top",
                        "peak1": peak1,
                        "peak2": peak2,
                        "confidence": 0.75,
                    }

        # Double bottom pattern
        if len(troughs) >= 2:
            for i in range(len(troughs) - 1):
                trough1 = troughs[i][1]
                trough2 = troughs[i + 1][1]

                if (
                    abs(trough1 - trough2) / trough1 < 0.03
                ):  # Troughs are approximately equal
                    return {
                        "pattern_found": True,
                        "type": "double_bottom",
                        "trough1": trough1,
                        "trough2": trough2,
                        "confidence": 0.75,
                    }

        return {"pattern_found": False}


class TechnicalAnalysisAgent(BaseFinancialAgent):
    """
    Specialized agent for technical analysis of financial instruments.

    This agent provides comprehensive technical analysis including indicators,
    chart patterns, and trading signals.
    """

    def __init__(self):
        """Initialize the Technical Analysis Agent."""
        capabilities = [
            AgentCapability.TECHNICAL_ANALYSIS,
            AgentCapability.STOCK_ANALYSIS,
        ]
        super().__init__("TechnicalAnalysisAgent", capabilities)
        self.yfinance_tools = None
        self.indicators = TechnicalIndicators()
        self.patterns = ChartPatterns()

    def initialize(self) -> bool:
        """Initialize the agent with required resources."""
        try:
            self.yfinance_tools = YFinanceTools(
                stock_price=True,
                technical_indicators=True,
                stock_fundamentals=False,
                company_news=False,
            )
            self.is_initialized = True
            logger.info("Technical Analysis Agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Technical Analysis Agent: {e}")
            return False

    def analyze(
        self, symbol: str, analysis_type: AnalysisType, **kwargs
    ) -> AgentResponse:
        """
        Perform technical analysis on a given symbol.

        Args:
            symbol (str): Financial symbol to analyze
            analysis_type (AnalysisType): Type of analysis to perform
            **kwargs: Additional parameters (period, timeframe, etc.)

        Returns:
            AgentResponse: Analysis results
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
            period = kwargs.get("period", "1y")
            interval = kwargs.get("interval", "1d")

            # Get historical data
            historical_data = self._get_historical_data(symbol, period, interval)

            if historical_data is None or historical_data.empty:
                return AgentResponse(
                    success=False,
                    data={},
                    message=f"No data available for symbol {symbol}",
                    confidence_score=0.0,
                    sources=[],
                    analysis_type=analysis_type,
                    timestamp=datetime.now().isoformat(),
                )

            # Perform specific analysis based on type
            if analysis_type == AnalysisType.PRICE_MOVEMENT:
                analysis_result = self._analyze_price_movement(historical_data, symbol)
            elif analysis_type == AnalysisType.VOLUME_ANALYSIS:
                analysis_result = self._analyze_volume(historical_data, symbol)
            else:
                analysis_result = self._comprehensive_technical_analysis(
                    historical_data, symbol
                )

            return AgentResponse(
                success=True,
                data=analysis_result,
                message=f"Technical analysis completed for {symbol}",
                confidence_score=analysis_result.get("confidence_score", 0.8),
                sources=["Yahoo Finance", "Technical Indicators"],
                analysis_type=analysis_type,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Error in technical analysis for {symbol}: {e}")
            return AgentResponse(
                success=False,
                data={},
                message=f"Analysis failed: {str(e)}",
                confidence_score=0.0,
                sources=[],
                analysis_type=analysis_type,
                timestamp=datetime.now().isoformat(),
            )

    def _get_historical_data(
        self, symbol: str, period: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Get historical price data for a symbol."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None

    def _analyze_price_movement(
        self, data: pd.DataFrame, symbol: str
    ) -> Dict[str, Any]:
        """Analyze price movement patterns."""
        close_prices = data["Close"]

        # Calculate moving averages
        sma_20 = self.indicators.sma(close_prices, 20)
        sma_50 = self.indicators.sma(close_prices, 50)
        ema_12 = self.indicators.ema(close_prices, 12)

        # Calculate trend
        current_price = close_prices.iloc[-1]
        trend = (
            "bullish"
            if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]
            else "bearish"
        )

        # Support and resistance levels
        support_resistance = self.indicators.support_resistance_levels(close_prices)

        return {
            "symbol": symbol,
            "current_price": current_price,
            "trend": trend,
            "sma_20": sma_20.iloc[-1],
            "sma_50": sma_50.iloc[-1],
            "ema_12": ema_12.iloc[-1],
            "support_levels": support_resistance["support"],
            "resistance_levels": support_resistance["resistance"],
            "confidence_score": 0.85,
        }

    def _analyze_volume(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Analyze volume patterns."""
        volume = data["Volume"]
        close_prices = data["Close"]

        # Volume moving average
        volume_ma = self.indicators.sma(volume, 20)

        # Volume trend
        current_volume = volume.iloc[-1]
        avg_volume = volume_ma.iloc[-1]
        volume_ratio = current_volume / avg_volume

        # Price-volume relationship
        price_change = (
            close_prices.iloc[-1] - close_prices.iloc[-2]
        ) / close_prices.iloc[-2]
        volume_confirmation = (price_change > 0 and volume_ratio > 1.2) or (
            price_change < 0 and volume_ratio > 1.2
        )

        return {
            "symbol": symbol,
            "current_volume": current_volume,
            "average_volume": avg_volume,
            "volume_ratio": volume_ratio,
            "volume_trend": "high"
            if volume_ratio > 1.5
            else "normal"
            if volume_ratio > 0.8
            else "low",
            "volume_confirmation": volume_confirmation,
            "confidence_score": 0.75,
        }

    def _comprehensive_technical_analysis(
        self, data: pd.DataFrame, symbol: str
    ) -> Dict[str, Any]:
        """Perform comprehensive technical analysis."""
        close_prices = data["Close"]
        high_prices = data["High"]
        low_prices = data["Low"]

        # Technical indicators
        rsi = self.indicators.rsi(close_prices)
        macd_data = self.indicators.macd(close_prices)
        bollinger = self.indicators.bollinger_bands(close_prices)
        stochastic = self.indicators.stochastic(high_prices, low_prices, close_prices)

        # Chart patterns
        head_shoulders = self.patterns.detect_head_shoulders(close_prices)
        double_pattern = self.patterns.detect_double_top_bottom(close_prices)

        # Trading signals
        signals = self._generate_trading_signals(
            rsi, macd_data, bollinger, close_prices
        )

        return {
            "symbol": symbol,
            "technical_indicators": {
                "rsi": rsi.iloc[-1] if not rsi.empty else None,
                "macd": {
                    "macd": macd_data["macd"].iloc[-1]
                    if not macd_data["macd"].empty
                    else None,
                    "signal": macd_data["signal"].iloc[-1]
                    if not macd_data["signal"].empty
                    else None,
                    "histogram": macd_data["histogram"].iloc[-1]
                    if not macd_data["histogram"].empty
                    else None,
                },
                "bollinger_bands": {
                    "upper": bollinger["upper"].iloc[-1]
                    if not bollinger["upper"].empty
                    else None,
                    "middle": bollinger["middle"].iloc[-1]
                    if not bollinger["middle"].empty
                    else None,
                    "lower": bollinger["lower"].iloc[-1]
                    if not bollinger["lower"].empty
                    else None,
                },
                "stochastic": {
                    "k_percent": stochastic["k_percent"].iloc[-1]
                    if not stochastic["k_percent"].empty
                    else None,
                    "d_percent": stochastic["d_percent"].iloc[-1]
                    if not stochastic["d_percent"].empty
                    else None,
                },
            },
            "chart_patterns": {
                "head_shoulders": head_shoulders,
                "double_pattern": double_pattern,
            },
            "trading_signals": signals,
            "confidence_score": 0.9,
        }

    def _generate_trading_signals(
        self,
        rsi: pd.Series,
        macd_data: Dict[str, pd.Series],
        bollinger: Dict[str, pd.Series],
        prices: pd.Series,
    ) -> Dict[str, str]:
        """Generate trading signals based on technical indicators."""
        signals = {}

        # RSI signals
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        if current_rsi > 70:
            signals["rsi"] = "overbought"
        elif current_rsi < 30:
            signals["rsi"] = "oversold"
        else:
            signals["rsi"] = "neutral"

        # MACD signals
        if not macd_data["macd"].empty and not macd_data["signal"].empty:
            macd_line = macd_data["macd"].iloc[-1]
            signal_line = macd_data["signal"].iloc[-1]
            if macd_line > signal_line:
                signals["macd"] = "bullish"
            else:
                signals["macd"] = "bearish"

        # Bollinger Bands signals
        if not bollinger["upper"].empty and not bollinger["lower"].empty:
            current_price = prices.iloc[-1]
            upper_band = bollinger["upper"].iloc[-1]
            lower_band = bollinger["lower"].iloc[-1]

            if current_price > upper_band:
                signals["bollinger"] = "overbought"
            elif current_price < lower_band:
                signals["bollinger"] = "oversold"
            else:
                signals["bollinger"] = "neutral"

        return signals

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols for technical analysis."""
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
            "BTC-USD",
            "ETH-USD",
            "XU100.IS",
            "THYAO.IS",
        ]
