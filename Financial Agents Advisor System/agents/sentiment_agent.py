"""
Sentiment Analysis Agent - Market Sentiment and News Analysis
===========================================================

This module implements a specialized agent for analyzing market sentiment
from news, social media, and other textual sources.

Classes:
    SentimentAnalysisAgent: Main sentiment analysis agent
    NewsProcessor: News text processing and analysis
    SentimentScorer: Sentiment scoring algorithms
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
import re
from collections import Counter
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

from .base_agent import BaseFinancialAgent, AgentCapability, AnalysisType, AgentResponse

logger = logging.getLogger(__name__)


class SentimentScorer:
    """
    Sentiment scoring algorithms for financial text analysis.
    """

    def __init__(self):
        """Initialize sentiment scorer with financial lexicons."""
        # Positive financial terms
        self.positive_terms = {
            "bull",
            "bullish",
            "up",
            "rise",
            "gain",
            "profit",
            "growth",
            "increase",
            "strong",
            "rally",
            "surge",
            "boom",
            "outperform",
            "beat",
            "exceed",
            "positive",
            "upgrade",
            "buy",
            "momentum",
            "recovery",
            "expansion",
            "optimistic",
            "confidence",
            "breakout",
        }

        # Negative financial terms
        self.negative_terms = {
            "bear",
            "bearish",
            "down",
            "fall",
            "decline",
            "loss",
            "drop",
            "crash",
            "weak",
            "plunge",
            "slump",
            "recession",
            "underperform",
            "miss",
            "negative",
            "downgrade",
            "sell",
            "fear",
            "uncertainty",
            "volatility",
            "risk",
            "concern",
            "pessimistic",
        }

        # Neutral/uncertainty terms
        self.neutral_terms = {
            "stable",
            "flat",
            "sideways",
            "neutral",
            "hold",
            "wait",
            "cautious",
            "mixed",
            "uncertain",
            "unclear",
            "pending",
            "monitor",
            "watch",
            "evaluate",
        }

        # Intensity modifiers
        self.intensifiers = {
            "very": 1.5,
            "extremely": 2.0,
            "highly": 1.5,
            "significantly": 1.8,
            "dramatically": 2.0,
            "substantially": 1.7,
            "moderately": 1.2,
            "slightly": 0.8,
        }

    def score_text(self, text: str) -> Dict[str, Any]:
        """
        Score sentiment of financial text.

        Args:
            text (str): Text to analyze

        Returns:
            Dict[str, Any]: Sentiment analysis results
        """
        if not text:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}

        # Clean and tokenize text
        clean_text = self._clean_text(text)
        tokens = clean_text.lower().split()

        # Calculate sentiment scores
        positive_score = 0
        negative_score = 0
        total_words = len(tokens)

        for i, token in enumerate(tokens):
            # Check for intensifiers
            intensity = 1.0
            if i > 0 and tokens[i - 1] in self.intensifiers:
                intensity = self.intensifiers[tokens[i - 1]]

            # Score terms
            if token in self.positive_terms:
                positive_score += intensity
            elif token in self.negative_terms:
                negative_score += intensity

        # Calculate final sentiment
        net_score = positive_score - negative_score
        normalized_score = net_score / max(total_words, 1)  # Normalize by text length

        # Determine sentiment label
        if normalized_score > 0.1:
            sentiment = "positive"
        elif normalized_score < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Calculate confidence based on number of sentiment words found
        sentiment_words = positive_score + negative_score
        confidence = min(sentiment_words / max(total_words * 0.1, 1), 1.0)

        return {
            "sentiment": sentiment,
            "score": normalized_score,
            "confidence": confidence,
            "positive_words": positive_score,
            "negative_words": negative_score,
            "word_count": total_words,
        }

    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis."""
        # Remove URLs, special characters, extra whitespace
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"[^\w\s]", " ", text)  # Remove special characters
        text = re.sub(r"\s+", " ", text)  # Remove extra whitespace
        return text.strip()


class NewsProcessor:
    """
    News processing and filtering for financial analysis.
    """

    def __init__(self):
        """Initialize news processor."""
        self.financial_keywords = {
            "earnings",
            "revenue",
            "profit",
            "stock",
            "share",
            "market",
            "trading",
            "investor",
            "investment",
            "financial",
            "economic",
            "economy",
            "price",
            "analyst",
            "forecast",
            "guidance",
            "quarter",
            "annual",
            "report",
        }

    def filter_financial_news(
        self, news_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter news items to keep only financial-related content.

        Args:
            news_items (List[Dict]): List of news items

        Returns:
            List[Dict]: Filtered financial news items
        """
        financial_news = []

        for item in news_items:
            title = item.get("title", "").lower()
            content = item.get("content", "").lower()

            # Check if news contains financial keywords
            text_to_check = f"{title} {content}"
            keyword_count = sum(
                1 for keyword in self.financial_keywords if keyword in text_to_check
            )

            if keyword_count >= 2:  # At least 2 financial keywords
                item["financial_relevance"] = keyword_count / len(
                    self.financial_keywords
                )
                financial_news.append(item)

        # Sort by relevance
        financial_news.sort(key=lambda x: x.get("financial_relevance", 0), reverse=True)
        return financial_news

    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key financial phrases from text."""
        # Simple key phrase extraction (can be enhanced with NLP libraries)
        phrases = []

        # Common financial phrase patterns
        patterns = [
            r"\b\w+\s+earnings\b",
            r"\b\w+\s+revenue\b",
            r"\b\w+\s+profit\b",
            r"\b\w+\s+growth\b",
            r"\b\w+\s+decline\b",
            r"\b\w+\s+increase\b",
            r"\b\w+\s+decrease\b",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            phrases.extend(matches)

        return list(set(phrases))  # Remove duplicates


class SentimentAnalysisAgent(BaseFinancialAgent):
    """
    Specialized agent for market sentiment analysis from news and text sources.

    This agent analyzes sentiment from financial news, social media mentions,
    and other textual sources to gauge market sentiment.
    """

    def __init__(self):
        """Initialize the Sentiment Analysis Agent."""
        capabilities = [
            AgentCapability.SENTIMENT_ANALYSIS,
            AgentCapability.NEWS_ANALYSIS,
            AgentCapability.MARKET_RESEARCH,
        ]
        super().__init__("SentimentAnalysisAgent", capabilities)
        self.duckduckgo_tool = None
        self.yfinance_tools = None
        self.sentiment_scorer = SentimentScorer()
        self.news_processor = NewsProcessor()

    def initialize(self) -> bool:
        """Initialize the agent with required resources."""
        try:
            self.duckduckgo_tool = DuckDuckGo()
            self.yfinance_tools = YFinanceTools(
                stock_price=False,
                company_news=True,
                stock_fundamentals=False,
                analyst_recommendations=False,
            )
            self.is_initialized = True
            logger.info("Sentiment Analysis Agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Sentiment Analysis Agent: {e}")
            return False

    def analyze(
        self, symbol: str, analysis_type: AnalysisType, **kwargs
    ) -> AgentResponse:
        """
        Perform sentiment analysis for a given symbol.

        Args:
            symbol (str): Financial symbol to analyze
            analysis_type (AnalysisType): Type of analysis to perform
            **kwargs: Additional parameters (timeframe, source, etc.)

        Returns:
            AgentResponse: Sentiment analysis results
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
            timeframe = kwargs.get("timeframe", "7d")  # Last 7 days by default
            sources = kwargs.get("sources", ["news", "web"])  # Sources to analyze

            analysis_result = {}

            if "news" in sources:
                news_sentiment = self._analyze_news_sentiment(symbol, timeframe)
                analysis_result["news_sentiment"] = news_sentiment

            if "web" in sources:
                web_sentiment = self._analyze_web_sentiment(symbol, timeframe)
                analysis_result["web_sentiment"] = web_sentiment

            # Aggregate sentiment scores
            overall_sentiment = self._aggregate_sentiment(analysis_result)
            analysis_result["overall_sentiment"] = overall_sentiment

            # Generate sentiment summary
            summary = self._generate_sentiment_summary(symbol, analysis_result)
            analysis_result["summary"] = summary

            return AgentResponse(
                success=True,
                data=analysis_result,
                message=f"Sentiment analysis completed for {symbol}",
                confidence_score=overall_sentiment.get("confidence", 0.7),
                sources=["Financial News", "Web Search", "Company Reports"],
                analysis_type=analysis_type,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Error in sentiment analysis for {symbol}: {e}")
            return AgentResponse(
                success=False,
                data={},
                message=f"Sentiment analysis failed: {str(e)}",
                confidence_score=0.0,
                sources=[],
                analysis_type=analysis_type,
                timestamp=datetime.now().isoformat(),
            )

    def _analyze_news_sentiment(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze sentiment from financial news."""
        try:
            # Get company news (this would typically use the YFinance news API)
            # For now, we'll simulate news data
            news_items = self._get_company_news(symbol, timeframe)

            if not news_items:
                return {
                    "sentiment": "neutral",
                    "score": 0.0,
                    "confidence": 0.0,
                    "article_count": 0,
                }

            # Filter financial news
            financial_news = self.news_processor.filter_financial_news(news_items)

            # Analyze sentiment for each article
            sentiment_scores = []
            for article in financial_news[:10]:  # Analyze top 10 articles
                title = article.get("title", "")
                content = article.get("content", "")
                text = f"{title} {content}"

                score_result = self.sentiment_scorer.score_text(text)
                sentiment_scores.append(score_result)

            # Aggregate news sentiment
            if sentiment_scores:
                avg_score = sum(s["score"] for s in sentiment_scores) / len(
                    sentiment_scores
                )
                avg_confidence = sum(s["confidence"] for s in sentiment_scores) / len(
                    sentiment_scores
                )

                # Determine overall sentiment
                if avg_score > 0.1:
                    overall_sentiment = "positive"
                elif avg_score < -0.1:
                    overall_sentiment = "negative"
                else:
                    overall_sentiment = "neutral"

                return {
                    "sentiment": overall_sentiment,
                    "score": avg_score,
                    "confidence": avg_confidence,
                    "article_count": len(financial_news),
                    "analyzed_articles": len(sentiment_scores),
                }

            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "article_count": 0,
            }

        except Exception as e:
            logger.error(f"Error analyzing news sentiment for {symbol}: {e}")
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "article_count": 0,
            }

    def _analyze_web_sentiment(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Analyze sentiment from web search results."""
        try:
            # Search for recent mentions of the symbol
            search_query = f"{symbol} stock market news {timeframe}"

            # This would use DuckDuckGo to search for recent mentions
            # For now, we'll simulate web search results
            web_results = self._search_web_mentions(search_query)

            if not web_results:
                return {
                    "sentiment": "neutral",
                    "score": 0.0,
                    "confidence": 0.0,
                    "result_count": 0,
                }

            # Analyze sentiment for each result
            sentiment_scores = []
            for result in web_results[:15]:  # Analyze top 15 results
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                text = f"{title} {snippet}"

                score_result = self.sentiment_scorer.score_text(text)
                sentiment_scores.append(score_result)

            # Aggregate web sentiment
            if sentiment_scores:
                avg_score = sum(s["score"] for s in sentiment_scores) / len(
                    sentiment_scores
                )
                avg_confidence = sum(s["confidence"] for s in sentiment_scores) / len(
                    sentiment_scores
                )

                # Determine overall sentiment
                if avg_score > 0.1:
                    overall_sentiment = "positive"
                elif avg_score < -0.1:
                    overall_sentiment = "negative"
                else:
                    overall_sentiment = "neutral"

                return {
                    "sentiment": overall_sentiment,
                    "score": avg_score,
                    "confidence": avg_confidence,
                    "result_count": len(web_results),
                    "analyzed_results": len(sentiment_scores),
                }

            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "result_count": 0,
            }

        except Exception as e:
            logger.error(f"Error analyzing web sentiment for {symbol}: {e}")
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "result_count": 0,
            }

    def _get_company_news(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Get company news from various sources."""
        # This would integrate with actual news APIs
        # For demonstration, returning simulated data
        return [
            {
                "title": f"{symbol} reports strong quarterly earnings",
                "content": "The company exceeded analyst expectations with strong revenue growth...",
                "source": "Financial News",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "title": f"{symbol} faces market volatility amid economic concerns",
                "content": "Shares declined following broader market uncertainty...",
                "source": "Market Watch",
                "timestamp": datetime.now().isoformat(),
            },
        ]

    def _search_web_mentions(self, query: str) -> List[Dict[str, Any]]:
        """Search web for mentions of the symbol."""
        # This would use DuckDuckGo API for actual web search
        # For demonstration, returning simulated data
        return [
            {
                "title": "Stock Analysis: Strong Buy Recommendation",
                "snippet": "Analysts are bullish on the stock with positive outlook...",
                "url": "https://example.com/analysis1",
            },
            {
                "title": "Market Concerns Rise Over Recent Developments",
                "snippet": "Investors show concern over regulatory changes affecting the sector...",
                "url": "https://example.com/analysis2",
            },
        ]

    def _aggregate_sentiment(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate sentiment from multiple sources."""
        sentiments = []
        confidences = []
        scores = []

        for source, result in analysis_results.items():
            if isinstance(result, dict) and "sentiment" in result:
                sentiments.append(result["sentiment"])
                confidences.append(result.get("confidence", 0.0))
                scores.append(result.get("score", 0.0))

        if not sentiments:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}

        # Calculate weighted average (can be enhanced with source-specific weights)
        avg_score = sum(scores) / len(scores)
        avg_confidence = sum(confidences) / len(confidences)

        # Count sentiment occurrences
        sentiment_counts = Counter(sentiments)
        overall_sentiment = sentiment_counts.most_common(1)[0][0]

        return {
            "sentiment": overall_sentiment,
            "score": avg_score,
            "confidence": avg_confidence,
            "source_breakdown": sentiment_counts,
        }

    def _generate_sentiment_summary(
        self, symbol: str, analysis_results: Dict[str, Any]
    ) -> str:
        """Generate a human-readable sentiment summary."""
        overall = analysis_results.get("overall_sentiment", {})
        sentiment = overall.get("sentiment", "neutral")
        confidence = overall.get("confidence", 0.0)

        if sentiment == "positive":
            sentiment_desc = "positive market sentiment"
        elif sentiment == "negative":
            sentiment_desc = "negative market sentiment"
        else:
            sentiment_desc = "neutral market sentiment"

        confidence_desc = (
            "high" if confidence > 0.7 else "moderate" if confidence > 0.4 else "low"
        )

        return f"{symbol} shows {sentiment_desc} with {confidence_desc} confidence based on recent news and web analysis."

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols for sentiment analysis."""
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
            "SPY",
            "QQQ",
        ]
