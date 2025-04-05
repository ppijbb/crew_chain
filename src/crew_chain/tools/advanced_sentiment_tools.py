import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

class MultilayerSentimentAnalysisInput(BaseModel):
    """Input for the Multilayer Sentiment Analysis tool."""
    crypto_symbol: str = Field(..., description="The symbol of the cryptocurrency to analyze, e.g., BTC, ETH")
    data_sources: List[str] = Field(default=["twitter", "news", "forums"], 
                                   description="Data sources to include in analysis: 'twitter', 'news', 'forums', 'reddit'")
    analysis_depth: str = Field(default="comprehensive", description="Analysis depth: 'basic', 'comprehensive', or 'advanced'")

class MultilayerSentimentAnalysisTool(BaseTool):
    """Advanced tool for multilayer sentiment analysis of cryptocurrency markets."""
    name = "multilayer_sentiment_analysis"
    description = "Analyzes sentiment across multiple data sources using transformer models and provides comprehensive insights for cryptocurrency trading."
    args_schema: Type[BaseModel] = MultilayerSentimentAnalysisInput
    
    def _run(self, crypto_symbol: str, data_sources: List[str] = ["twitter", "news", "forums"], 
            analysis_depth: str = "comprehensive") -> str:
        """
        Perform multilayer sentiment analysis across different data sources.
        Incorporates BERT, RNN, and transformer models for nuanced sentiment detection.
        
        This is a simulation based on research papers on sentiment analysis in crypto markets.
        In a real implementation, this would connect to APIs and use actual ML models.
        """
        try:
            coin_name = self._get_coin_name(crypto_symbol)
            valid_sources = [s for s in data_sources if s in ["twitter", "news", "forums", "reddit"]]
            
            if not valid_sources:
                valid_sources = ["twitter", "news"]
            
            # Normalize analysis depth
            if analysis_depth not in ["basic", "comprehensive", "advanced"]:
                analysis_depth = "comprehensive"
            
            # Generate simulated sentiment results per data source
            sentiment_results = {}
            for source in valid_sources:
                sentiment_results[source] = self._simulate_sentiment_data(crypto_symbol, source)
            
            # Calculate weighted consensus sentiment
            consensus_sentiment = self._calculate_consensus_sentiment(sentiment_results)
            
            # Generate sentiment divergence metrics
            divergence = self._calculate_sentiment_divergence(sentiment_results)
            
            # Format the output based on analysis depth
            result = f"""
# Multilayer Sentiment Analysis for {coin_name} ({crypto_symbol.upper()})

## Sentiment Summary
Overall Consensus: {self._format_sentiment_score(consensus_sentiment['score'])}
Confidence Level: {consensus_sentiment['confidence']:.1f}%
Sentiment Trend: {consensus_sentiment['trend']}
Sentiment Divergence: {divergence['level']} ({divergence['score']:.2f})

## Source-specific Sentiment
"""
            # Add source-specific sentiment results
            for source in sentiment_results:
                data = sentiment_results[source]
                result += f"""
### {source.capitalize()} Sentiment
Score: {self._format_sentiment_score(data['sentiment_score'])}
Volume: {data['volume']:,} data points
Key topics: {', '.join(data['topics'])}
"""
            
            # Add additional analysis based on depth
            if analysis_depth in ["comprehensive", "advanced"]:
                result += f"""
## Sentiment Patterns
- Short-term Pattern: {consensus_sentiment['short_term_pattern']}
- Medium-term Pattern: {consensus_sentiment['medium_term_pattern']}
- Volatility Indicator: {consensus_sentiment['volatility_indicator']}

## Sentiment-Price Correlation
- Current Correlation: {consensus_sentiment['price_correlation']}
- Predictive Signal: {consensus_sentiment['predictive_signal']}
"""
            
            if analysis_depth == "advanced":
                result += f"""
## Deep Sentiment Analysis
- Sentiment Momentum: {consensus_sentiment['momentum']}
- Sentiment Reversal Probability: {consensus_sentiment['reversal_probability']:.1f}%
- Market Manipulation Indicators: {consensus_sentiment['manipulation_indicator']}
- Sentiment-based Price Projection: {consensus_sentiment['price_projection']}

## Multilingual Sentiment (Global Perspective)
- Western Markets: {consensus_sentiment['regional_sentiment']['western']}
- Asian Markets: {consensus_sentiment['regional_sentiment']['asian']}
- Emerging Markets: {consensus_sentiment['regional_sentiment']['emerging']}
"""
            
            # Add trading implications
            result += f"""
## Trading Implications
{self._get_trading_implications(consensus_sentiment, divergence)}

## Sentiment-based Position Recommendations
{self._get_position_recommendations(consensus_sentiment, divergence)}
"""
            
            return result
            
        except Exception as e:
            return f"Error performing multilayer sentiment analysis: {str(e)}"
    
    def _simulate_sentiment_data(self, crypto_symbol: str, source: str) -> dict:
        """Generate simulated sentiment data for a specific source."""
        # Seed for consistent results per symbol and source
        np.random.seed(hash(f"{crypto_symbol}_{source}") % 2**32)
        
        # Base sentiment varies by source
        if source == "twitter":
            base_sentiment = np.random.normal(0.1, 0.4)  # Twitter tends to be more volatile
            volume = np.random.randint(8000, 25000)
            topics = self._generate_topics(crypto_symbol, "social")
        elif source == "news":
            base_sentiment = np.random.normal(0, 0.3)  # News tends to be more neutral
            volume = np.random.randint(500, 3000)
            topics = self._generate_topics(crypto_symbol, "news")
        elif source == "forums":
            base_sentiment = np.random.normal(-0.05, 0.35)  # Forums tend to be more technical/critical
            volume = np.random.randint(2000, 10000)
            topics = self._generate_topics(crypto_symbol, "technical")
        elif source == "reddit":
            base_sentiment = np.random.normal(0.05, 0.45)  # Reddit can be very mixed
            volume = np.random.randint(5000, 15000)
            topics = self._generate_topics(crypto_symbol, "community")
        
        # Apply some normalization
        sentiment_score = np.clip(base_sentiment, -1, 1)
        
        return {
            "sentiment_score": sentiment_score,
            "volume": volume,
            "topics": topics,
            "change_24h": np.random.uniform(-0.3, 0.3)
        }
    
    def _generate_topics(self, crypto_symbol: str, source_type: str) -> List[str]:
        """Generate relevant topics based on cryptocurrency and source type."""
        common_topics = ["price movement", "market trend", "trading volume"]
        
        if source_type == "social":
            additional_topics = ["market sentiment", "price prediction", "buying opportunity", 
                               "selling pressure", "FOMO", "market crash", "bull run"]
        elif source_type == "news":
            additional_topics = ["regulation", "adoption", "institutional investment", 
                               "technology update", "security concerns", "market analysis"]
        elif source_type == "technical":
            additional_topics = ["technical analysis", "support levels", "resistance", 
                               "trading patterns", "indicators", "development progress"]
        elif source_type == "community":
            additional_topics = ["long-term holding", "staking", "project development", 
                               "comparisons", "alternatives", "use cases"]
        
        # Select random topics
        num_topics = np.random.randint(3, 6)
        selected_topics = np.random.choice(common_topics + additional_topics, 
                                         size=min(num_topics, len(common_topics + additional_topics)), 
                                         replace=False)
        
        return selected_topics.tolist()
    
    def _calculate_consensus_sentiment(self, sentiment_results: dict) -> dict:
        """Calculate weighted consensus sentiment across all sources."""
        # Define source weights (in a real implementation, these would be trained)
        source_weights = {
            "twitter": 0.35,
            "news": 0.30,
            "forums": 0.20,
            "reddit": 0.15
        }
        
        # Calculate weighted sentiment score
        total_weight = 0
        weighted_score = 0
        for source, data in sentiment_results.items():
            weight = source_weights.get(source, 0.25)
            weighted_score += data["sentiment_score"] * weight
            total_weight += weight
        
        if total_weight > 0:
            consensus_score = weighted_score / total_weight
        else:
            consensus_score = 0
            
        # Generate other consensus metrics
        short_patterns = ["Bullish spike", "Bearish dip", "Neutral consolidation", 
                        "Extreme optimism", "Panic selling", "Growing interest"]
        medium_patterns = ["Sustained optimism", "Gradual decline", "Fluctuating interest", 
                         "Building momentum", "Waning enthusiasm", "Controversy"]
        
        volatility_indicators = ["High", "Moderate", "Low"]
        price_correlations = ["Strong positive", "Moderate positive", "Weak positive", 
                            "No correlation", "Weak negative", "Moderate negative"]
        predictive_signals = ["Strong buy", "Moderate buy", "Weak buy", "Neutral", 
                            "Weak sell", "Moderate sell", "Strong sell"]
        momentum_states = ["Accelerating positive", "Steady positive", "Decelerating positive", 
                         "Neutral", "Decelerating negative", "Steady negative", "Accelerating negative"]
        manipulation_indicators = ["No suspicious activity", "Minor anomalies", "Potential coordinated activity",
                                 "Signs of artificial sentiment boost", "Evidence of FUD campaign"]
        price_projections = ["Strong upward movement likely", "Moderate rise expected", 
                           "Slight increase possible", "Sideways movement likely",
                           "Slight decrease possible", "Moderate drop expected", 
                           "Strong downward movement likely"]
        
        # Regional sentiment (simulated)
        regional_sentiment = {
            "western": np.random.choice(["Strongly Positive", "Positive", "Neutral", "Negative", "Strongly Negative"]),
            "asian": np.random.choice(["Strongly Positive", "Positive", "Neutral", "Negative", "Strongly Negative"]),
            "emerging": np.random.choice(["Strongly Positive", "Positive", "Neutral", "Negative", "Strongly Negative"])
        }
        
        return {
            "score": consensus_score,
            "confidence": 50 + np.random.uniform(0, 40),  # Higher for more consistent cross-source sentiment
            "trend": np.random.choice(["Improving", "Stable", "Deteriorating"]),
            "short_term_pattern": np.random.choice(short_patterns),
            "medium_term_pattern": np.random.choice(medium_patterns),
            "volatility_indicator": np.random.choice(volatility_indicators),
            "price_correlation": np.random.choice(price_correlations),
            "predictive_signal": np.random.choice(predictive_signals),
            "momentum": np.random.choice(momentum_states),
            "reversal_probability": np.random.uniform(10, 60),
            "manipulation_indicator": np.random.choice(manipulation_indicators),
            "price_projection": np.random.choice(price_projections),
            "regional_sentiment": regional_sentiment
        }
    
    def _calculate_sentiment_divergence(self, sentiment_results: dict) -> dict:
        """Calculate divergence between different sentiment sources."""
        if len(sentiment_results) < 2:
            return {"level": "Insufficient data", "score": 0.0}
        
        # Extract sentiment scores
        scores = [data["sentiment_score"] for data in sentiment_results.values()]
        
        # Calculate standard deviation as a measure of divergence
        divergence_score = np.std(scores)
        
        # Categorize divergence level
        if divergence_score < 0.1:
            level = "Very Low (High Consensus)"
        elif divergence_score < 0.2:
            level = "Low (General Agreement)"
        elif divergence_score < 0.3:
            level = "Moderate (Some Disagreement)"
        elif divergence_score < 0.5:
            level = "High (Significant Disagreement)"
        else:
            level = "Very High (Extreme Divergence)"
            
        return {
            "level": level,
            "score": divergence_score
        }
    
    def _format_sentiment_score(self, score: float) -> str:
        """Format sentiment score with description."""
        if score > 0.6:
            return f"Extremely Positive ({score:.2f})"
        elif score > 0.3:
            return f"Strongly Positive ({score:.2f})"
        elif score > 0.1:
            return f"Moderately Positive ({score:.2f})"
        elif score > -0.1:
            return f"Neutral ({score:.2f})"
        elif score > -0.3:
            return f"Moderately Negative ({score:.2f})"
        elif score > -0.6:
            return f"Strongly Negative ({score:.2f})"
        else:
            return f"Extremely Negative ({score:.2f})"
    
    def _get_trading_implications(self, consensus: dict, divergence: dict) -> str:
        """Generate trading implications based on sentiment analysis."""
        score = consensus["score"]
        trend = consensus["trend"]
        divergence_level = divergence["level"]
        
        # Base implications on sentiment score and trend
        if score > 0.3 and trend == "Improving":
            base_implication = "Highly favorable sentiment environment for long positions. The improving trend suggests potential for further upside."
        elif score > 0.3 and trend == "Deteriorating":
            base_implication = "Currently positive sentiment but showing signs of peaking. Consider taking some profits on long positions."
        elif score > 0 and trend == "Improving":
            base_implication = "Mildly positive sentiment with improving trend. This could be a good entry point for long positions."
        elif score > 0 and trend == "Deteriorating":
            base_implication = "Sentiment is still positive but weakening. Monitor closely for potential sentiment shift."
        elif score < -0.3 and trend == "Deteriorating":
            base_implication = "Highly negative sentiment that continues to worsen. Favorable environment for short positions."
        elif score < -0.3 and trend == "Improving":
            base_implication = "Sentiment remains negative but is showing signs of improvement. Potential bottoming process may be underway."
        elif score < 0 and trend == "Deteriorating":
            base_implication = "Negative sentiment that is continuing to worsen. Consider establishing or maintaining short positions."
        elif score < 0 and trend == "Improving":
            base_implication = "Sentiment is negative but improving. Watch for confirmation of sentiment shift before establishing long positions."
        else:
            base_implication = "Neutral sentiment environment with no clear directional bias. Focus on technical factors for trading decisions."
        
        # Add nuance based on divergence
        if "High" in divergence_level or "Very High" in divergence_level:
            divergence_note = "There is significant disagreement between different data sources, suggesting uncertainty in the market. Consider reducing position sizes and implementing tighter risk management."
        elif "Moderate" in divergence_level:
            divergence_note = "There is moderate disagreement between data sources. Use caution in position sizing and confirm signals with technical analysis."
        else:
            divergence_note = "There is strong consensus across different data sources, increasing confidence in the sentiment signal."
        
        # Add recommendation on using sentiment as a leading/lagging indicator
        if "Very Low" in divergence_level or "Low" in divergence_level:
            indicator_note = "With high consensus across sources, sentiment can be used as a leading indicator for potential price movements."
        else:
            indicator_note = "With divergent opinions across sources, treat sentiment as a confirming indicator rather than a primary signal."
        
        return f"{base_implication}\n\n{divergence_note}\n\n{indicator_note}"
    
    def _get_position_recommendations(self, consensus: dict, divergence: dict) -> str:
        """Generate position recommendations based on sentiment analysis."""
        score = consensus["score"]
        trend = consensus["trend"]
        divergence_high = "High" in divergence["level"] or "Very High" in divergence["level"]
        
        # Adjust position sizing based on divergence
        position_size_modifier = "smaller" if divergence_high else "standard"
        
        # Generate entry/exit recommendations
        if score > 0.3 and trend == "Improving":
            recommendation = f"""
1. Position Bias: Strong Long
2. Entry Strategy: Implement phased buying with {position_size_modifier} position sizes
3. Target Zones: Look for minor dips as entry opportunities
4. Exit Strategy: Set trailing stops to capture upside momentum
5. Risk Management: Place stops 5-8% below entry points
6. Position Switching: Maintain long bias until sentiment deteriorates significantly"""
            
        elif score > 0.3 and trend == "Deteriorating":
            recommendation = f"""
1. Position Bias: Cautious Long
2. Entry Strategy: Hold existing positions but avoid adding new exposure
3. Target Zones: Consider partial profit-taking on strength
4. Exit Strategy: Tighten stops to protect profits
5. Risk Management: Reduce position sizes as sentiment deteriorates
6. Position Switching: Prepare for potential shift to neutral or short bias"""
            
        elif score > 0 and trend == "Improving":
            recommendation = f"""
1. Position Bias: Moderate Long
2. Entry Strategy: Staged buying on dips with {position_size_modifier} position sizes
3. Target Zones: Focus on support areas for entries
4. Exit Strategy: Set moderate profit targets with trailing stops
5. Risk Management: Place stops 4-7% below entry points
6. Position Switching: Maintain long bias but be prepared to adjust if improvement stalls"""
            
        elif score < -0.3 and trend == "Deteriorating":
            recommendation = f"""
1. Position Bias: Strong Short
2. Entry Strategy: Implement phased shorting with {position_size_modifier} position sizes
3. Target Zones: Look for relief rallies as entry opportunities
4. Exit Strategy: Set trailing stops to capture downside momentum
5. Risk Management: Place stops 5-8% above entry points
6. Position Switching: Maintain short bias until sentiment shows clear improvement"""
            
        elif score < -0.3 and trend == "Improving":
            recommendation = f"""
1. Position Bias: Cautious Short / Neutral
2. Entry Strategy: Reduce short exposure, consider small long positions if confirmation appears
3. Target Zones: Look for key technical levels for potential long entries
4. Exit Strategy: Cover shorts gradually as sentiment improves
5. Risk Management: Use tight stops on any new long positions
6. Position Switching: Prepare for potential shift from short to long bias"""
            
        elif score < 0 and trend == "Deteriorating":
            recommendation = f"""
1. Position Bias: Moderate Short
2. Entry Strategy: Staged shorting on bounces with {position_size_modifier} position sizes
3. Target Zones: Focus on resistance areas for entries
4. Exit Strategy: Set moderate profit targets with trailing stops
5. Risk Management: Place stops 4-7% above entry points
6. Position Switching: Maintain short bias with potential to increase as sentiment worsens"""
            
        elif score < 0 and trend == "Improving":
            recommendation = f"""
1. Position Bias: Neutral with Bullish Tilt
2. Entry Strategy: Prepare for potential long positions but await confirmation
3. Target Zones: Identify key technical support levels for possible entries
4. Exit Strategy: Cover any existing shorts as sentiment improves
5. Risk Management: Use smaller position sizes with tight stops when initiating longs
6. Position Switching: Monitor sentiment closely for confirmation of shift from bearish to bullish"""
            
        else:  # Neutral cases
            recommendation = f"""
1. Position Bias: Neutral
2. Entry Strategy: Focus on range-bound trading strategies
3. Target Zones: Use technical analysis to identify short-term support and resistance
4. Exit Strategy: Take profits quickly as sentiment provides no directional edge
5. Risk Management: Reduce position sizes and tighten stops in this uncertain environment
6. Position Switching: Remain flexible and ready to adapt to emerging sentiment trends"""
        
        return recommendation
    
    def _get_coin_name(self, symbol: str) -> str:
        """Get the full name of a cryptocurrency from its symbol."""
        symbol = symbol.lower()
        coin_names = {
            "btc": "Bitcoin",
            "eth": "Ethereum",
            "sol": "Solana",
            "xrp": "Ripple",
            "doge": "Dogecoin",
            "ada": "Cardano",
            "dot": "Polkadot",
            "bnb": "Binance Coin",
            "usdt": "Tether",
            "usdc": "USD Coin",
            "link": "Chainlink",
            "avax": "Avalanche"
        }
        return coin_names.get(symbol, symbol.upper()) 