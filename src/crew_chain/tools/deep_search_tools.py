import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import json

class TwitterSentimentAnalysisInput(BaseModel):
    """Input for the Twitter sentiment analysis tool."""
    crypto_symbol: str = Field(..., description="The symbol of the cryptocurrency to analyze sentiment for, e.g., BTC, ETH")
    days: int = Field(default=7, description="Number of days of historical sentiment data to analyze (max 30)")

class TwitterSentimentAnalysisTool(BaseTool):
    """Tool for analyzing Twitter sentiment about cryptocurrencies."""
    name = "twitter_sentiment_analysis"
    description = "Analyzes sentiment from Twitter/social media about a specific cryptocurrency to gauge market sentiment."
    args_schema: Type[BaseModel] = TwitterSentimentAnalysisInput
    
    def _run(self, crypto_symbol: str, days: int = 7) -> str:
        """
        Analyze Twitter sentiment for a specific cryptocurrency.
        
        This is a simplified implementation that would need to be connected to
        a real Twitter API or sentiment analysis service in production.
        """
        try:
            # Simulate sentiment analysis results
            # In a real implementation, this would connect to Twitter API and use NLP
            coin_name = self._get_coin_name(crypto_symbol)
            
            # Generate simulated sentiment data
            np.random.seed(hash(crypto_symbol + str(days)) % 2**32)
            
            dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
            dates.reverse()
            
            # Generate more realistic sentiment patterns
            sentiment_base = np.random.normal(0, 0.3, days)
            # Add some trend
            trend = np.linspace(-0.2, 0.2, days)
            # Add some cyclical patterns
            cycles = 0.15 * np.sin(np.linspace(0, 3*np.pi, days))
            
            sentiment_scores = sentiment_base + trend + cycles
            # Normalize to -1 to 1 range
            sentiment_scores = np.clip(sentiment_scores, -1, 1)
            
            # Convert to percentage positive sentiment (0-100%)
            positive_sentiment = [(score + 1) * 50 for score in sentiment_scores]
            
            # Calculate tweet volumes (simulated)
            base_volume = 10000 if crypto_symbol.lower() == "btc" else 5000
            volumes = [int(base_volume * (0.8 + 0.4 * np.random.random())) for _ in range(days)]
            
            # Create dictionary for visualization
            sentiment_data = []
            for i in range(days):
                sentiment_data.append({
                    "date": dates[i],
                    "sentiment_score": round(sentiment_scores[i], 2),
                    "positive_percentage": round(positive_sentiment[i], 1),
                    "volume": volumes[i]
                })
            
            # Calculate overall sentiment metrics
            avg_sentiment = sum(s["sentiment_score"] for s in sentiment_data) / len(sentiment_data)
            avg_positive = sum(s["positive_percentage"] for s in sentiment_data) / len(sentiment_data)
            sentiment_trend = sentiment_data[-1]["sentiment_score"] - sentiment_data[0]["sentiment_score"]
            
            sentiment_interpretation = self._interpret_sentiment(avg_sentiment, sentiment_trend)
            
            # Format the output
            result = f"""
Twitter Sentiment Analysis for {coin_name} ({crypto_symbol.upper()}) over the past {days} days:

Overall Sentiment: {self._format_sentiment_score(avg_sentiment)} ({avg_positive:.1f}% positive)
Sentiment Trend: {self._format_trend(sentiment_trend)}
Tweet Volume: {sum(volumes):,} tweets analyzed

Daily Sentiment Breakdown:
"""
            
            for data in sentiment_data:
                result += f"- {data['date']}: {self._format_sentiment_score(data['sentiment_score'])} ({data['positive_percentage']}% positive, {data['volume']:,} tweets)\n"
            
            result += f"""
Sentiment Interpretation:
{sentiment_interpretation}

Trading Implication:
{self._get_trading_recommendation(avg_sentiment, sentiment_trend)}
"""
            
            return result
            
        except Exception as e:
            return f"Error analyzing Twitter sentiment: {str(e)}"
    
    def _format_sentiment_score(self, score: float) -> str:
        """Format sentiment score with description."""
        if score > 0.5:
            return f"Very Positive ({score:.2f})"
        elif score > 0.1:
            return f"Positive ({score:.2f})"
        elif score > -0.1:
            return f"Neutral ({score:.2f})"
        elif score > -0.5:
            return f"Negative ({score:.2f})"
        else:
            return f"Very Negative ({score:.2f})"
    
    def _format_trend(self, trend: float) -> str:
        """Format sentiment trend with description."""
        if trend > 0.3:
            return f"Strongly Improving ({trend:.2f})"
        elif trend > 0.1:
            return f"Improving ({trend:.2f})"
        elif trend > -0.1:
            return f"Stable ({trend:.2f})"
        elif trend > -0.3:
            return f"Declining ({trend:.2f})"
        else:
            return f"Strongly Declining ({trend:.2f})"
    
    def _interpret_sentiment(self, avg_sentiment: float, trend: float) -> str:
        """Provide interpretation of sentiment analysis."""
        if avg_sentiment > 0.3 and trend > 0.1:
            return "The market is highly optimistic and sentiment is improving, suggesting strong bullish momentum."
        elif avg_sentiment > 0.3 and trend < -0.1:
            return "The market is currently optimistic but sentiment is starting to decline, which may indicate an upcoming correction."
        elif avg_sentiment > 0 and trend > 0.1:
            return "The market is moderately positive and sentiment is improving, suggesting a potential upward trend."
        elif avg_sentiment > 0 and trend < -0.1:
            return "The market is slightly positive but sentiment is decreasing, suggesting caution."
        elif avg_sentiment < -0.3 and trend < -0.1:
            return "The market is very pessimistic and sentiment continues to worsen, indicating strong bearish momentum."
        elif avg_sentiment < -0.3 and trend > 0.1:
            return "The market is pessimistic but sentiment is improving, which could signal a potential market bottom."
        elif avg_sentiment < 0 and trend < -0.1:
            return "The market is negatively biased and sentiment is deteriorating, suggesting continued downward pressure."
        elif avg_sentiment < 0 and trend > 0.1:
            return "The market is slightly negative but sentiment is improving, which could indicate a potential trend reversal."
        else:
            return "Market sentiment is neutral with no clear directional bias."
    
    def _get_trading_recommendation(self, avg_sentiment: float, trend: float) -> str:
        """Generate trading recommendations based on sentiment analysis."""
        if avg_sentiment > 0.3 and trend > 0.1:
            return "Consider phased buying with a long bias. Set tight stop losses as extremely positive sentiment can indicate a local top."
        elif avg_sentiment > 0.3 and trend < -0.1:
            return "Be cautious with new long positions. Consider reducing exposure and tightening stop losses as sentiment may be peaking."
        elif avg_sentiment > 0 and trend > 0.1:
            return "Favorable conditions for long positions with improving sentiment. Consider phased buying on dips."
        elif avg_sentiment > 0 and trend < -0.1:
            return "Mixed signals - maintain existing long positions but avoid adding new exposure until sentiment stabilizes."
        elif avg_sentiment < -0.3 and trend < -0.1:
            return "Consider short positions or staying out of the market. Extremely negative sentiment could accelerate selling pressure."
        elif avg_sentiment < -0.3 and trend > 0.1:
            return "Potential bottoming process - consider small phased buying with strict risk management as sentiment improves."
        elif avg_sentiment < 0 and trend < -0.1:
            return "Maintain a defensive stance. Short-term traders may look for short opportunities while sentiment deteriorates."
        elif avg_sentiment < 0 and trend > 0.1:
            return "Early signs of improvement - watch for confirmation of sentiment shift before establishing new long positions."
        else:
            return "No clear sentiment edge - focus on technical factors and manage position sizes conservatively."
    
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


class DeepMarketAnalysisInput(BaseModel):
    """Input for the deep market analysis tool."""
    crypto_symbol: str = Field(..., description="The symbol of the cryptocurrency to analyze, e.g., BTC, ETH")
    analysis_depth: str = Field(default="medium", description="Depth of analysis: 'basic', 'medium', or 'deep'")

class DeepMarketAnalysisTool(BaseTool):
    """Tool that performs deep market analysis using ML-DQN approach."""
    name = "deep_market_analysis"
    description = "Performs comprehensive market analysis using ML-DQN approach, incorporating price data, sentiment, and technical indicators."
    args_schema: Type[BaseModel] = DeepMarketAnalysisInput
    
    def _run(self, crypto_symbol: str, analysis_depth: str = "medium") -> str:
        """
        Perform deep market analysis based on the M-DQN approach from the research paper.
        
        In a real implementation, this would use actual DQN models and real-time data.
        This is a simplified simulation to demonstrate the concept.
        """
        try:
            coin_name = self._get_coin_name(crypto_symbol)
            
            # Fetch historical price data (simulated)
            price_data = self._get_simulated_price_data(crypto_symbol)
            
            # Generate technical indicators
            indicators = self._calculate_technical_indicators(price_data)
            
            # Simulate M-DQN analysis results
            dqn_prediction = self._simulate_mdqn_prediction(price_data, indicators, crypto_symbol)
            
            # Determine analysis detail level
            if analysis_depth.lower() not in ["basic", "medium", "deep"]:
                analysis_depth = "medium"
            
            # Format the output based on analysis depth
            result = f"""
Deep Market Analysis for {coin_name} ({crypto_symbol.upper()}) using M-DQN Approach:

M-DQN Trade Signal: {dqn_prediction["trade_signal"]}
Confidence Level: {dqn_prediction["confidence_level"]:.1f}%
Price Prediction (24h): {dqn_prediction["price_prediction_24h"]:+.2f}%
Price Prediction (7d): {dqn_prediction["price_prediction_7d"]:+.2f}%
Risk Assessment: {dqn_prediction["risk_level"]}

"""
            # Add more details based on analysis depth
            if analysis_depth.lower() in ["medium", "deep"]:
                result += f"""
Technical Analysis Summary:
- Trend Direction: {indicators["trend_direction"]}
- Support Level: ${indicators["support_level"]:.2f}
- Resistance Level: ${indicators["resistance_level"]:.2f}
- RSI (14): {indicators["rsi"]:.1f} ({self._interpret_rsi(indicators["rsi"])})
- MACD Signal: {indicators["macd_signal"]}
- Volume Trend: {indicators["volume_trend"]}

"""
            
            if analysis_depth.lower() == "deep":
                result += f"""
Detailed M-DQN Analysis:
- Price Analysis Contribution: {dqn_prediction["price_analysis_weight"]:.1f}%
- Sentiment Analysis Contribution: {dqn_prediction["sentiment_analysis_weight"]:.1f}%
- Technical Indicator Contribution: {dqn_prediction["technical_analysis_weight"]:.1f}%
- Market Cycle Position: {dqn_prediction["market_cycle_position"]}
- Volatility Forecast: {dqn_prediction["volatility_forecast"]}

Pattern Recognition:
- Detected Patterns: {", ".join(dqn_prediction["detected_patterns"])}
- Pattern Reliability: {dqn_prediction["pattern_reliability"]}

"""
            
            # Add trading strategy recommendation
            result += f"""
Trading Strategy Recommendation:
{self._get_trading_strategy(dqn_prediction, indicators)}

Position Management:
{self._get_position_management(dqn_prediction, indicators)}
"""
            
            return result
            
        except Exception as e:
            return f"Error performing deep market analysis: {str(e)}"
    
    def _get_simulated_price_data(self, crypto_symbol: str) -> dict:
        """Generate simulated price data for the cryptocurrency."""
        # Seed the random number generator to get consistent results for the same symbol
        np.random.seed(hash(crypto_symbol) % 2**32)
        
        days = 30
        current_price = 100 if crypto_symbol.lower() != "btc" else 30000
        
        # Generate price data with some realistic patterns
        noise = np.random.normal(0, 0.02, days)
        trend = np.linspace(-0.1, 0.1, days) if np.random.random() > 0.5 else np.linspace(0.1, -0.1, days)
        cycles = 0.08 * np.sin(np.linspace(0, 4*np.pi, days))
        
        daily_returns = noise + trend + cycles
        
        # Calculate price series
        prices = [current_price]
        for ret in daily_returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate volume data
        base_volume = 10000 if crypto_symbol.lower() == "btc" else 5000
        volumes = [int(base_volume * (0.7 + 0.6 * np.random.random())) for _ in range(days+1)]
        
        return {
            "prices": prices,
            "volumes": volumes,
            "returns": daily_returns
        }
    
    def _calculate_technical_indicators(self, price_data: dict) -> dict:
        """Calculate simulated technical indicators based on price data."""
        prices = price_data["prices"]
        
        # Calculate simulated indicators
        last_price = prices[-1]
        
        # Simple trend direction
        short_ma = sum(prices[-5:]) / 5
        long_ma = sum(prices[-20:]) / 20
        trend_direction = "Uptrend" if short_ma > long_ma else "Downtrend"
        
        # Support and resistance levels (simplified)
        recent_prices = prices[-10:]
        support_level = min(recent_prices) * 0.98
        resistance_level = max(recent_prices) * 1.02
        
        # Simulated RSI (simplified)
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, ret) for ret in returns[-14:]]
        losses = [max(0, -ret) for ret in returns[-14:]]
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Simulated MACD signal
        ema12 = sum(prices[-12:]) / 12
        ema26 = sum(prices[-26:]) / 26
        macd = ema12 - ema26
        macd_signal = "Bullish" if macd > 0 else "Bearish"
        
        # Volume trend
        recent_volumes = price_data["volumes"][-5:]
        earlier_volumes = price_data["volumes"][-10:-5]
        volume_trend = "Increasing" if sum(recent_volumes) > sum(earlier_volumes) else "Decreasing"
        
        return {
            "trend_direction": trend_direction,
            "support_level": support_level,
            "resistance_level": resistance_level,
            "rsi": rsi,
            "macd_signal": macd_signal,
            "volume_trend": volume_trend
        }
    
    def _simulate_mdqn_prediction(self, price_data: dict, indicators: dict, crypto_symbol: str) -> dict:
        """
        Simulate M-DQN model predictions based on the research paper approach.
        
        This function simulates the multi-level DQN model that combines price data analysis
        and sentiment analysis to generate trading signals.
        """
        # Seed for consistent results per symbol
        np.random.seed(hash(crypto_symbol + "mdqn") % 2**32)
        
        # Determine base signal based on technical indicators
        signal_score = 0
        
        # Factor in trend
        if indicators["trend_direction"] == "Uptrend":
            signal_score += np.random.uniform(0.2, 0.4)
        else:
            signal_score -= np.random.uniform(0.2, 0.4)
        
        # Factor in RSI
        if indicators["rsi"] > 70:
            signal_score -= np.random.uniform(0.1, 0.3)  # Overbought
        elif indicators["rsi"] < 30:
            signal_score += np.random.uniform(0.1, 0.3)  # Oversold
        
        # Factor in MACD
        if indicators["macd_signal"] == "Bullish":
            signal_score += np.random.uniform(0.1, 0.25)
        else:
            signal_score -= np.random.uniform(0.1, 0.25)
        
        # Simulate sentiment influence (in real implementation, this would come from actual sentiment analysis)
        sentiment_factor = np.random.normal(0, 0.2)
        signal_score += sentiment_factor
        
        # Determine trading signal based on composite score
        if signal_score > 0.3:
            trade_signal = "Strong Buy"
            confidence_level = 70 + np.random.uniform(0, 20)
        elif signal_score > 0.1:
            trade_signal = "Buy"
            confidence_level = 60 + np.random.uniform(0, 15)
        elif signal_score > -0.1:
            trade_signal = "Hold"
            confidence_level = 50 + np.random.uniform(0, 10)
        elif signal_score > -0.3:
            trade_signal = "Sell"
            confidence_level = 60 + np.random.uniform(0, 15)
        else:
            trade_signal = "Strong Sell"
            confidence_level = 70 + np.random.uniform(0, 20)
        
        # Generate price predictions
        price_prediction_24h = signal_score * 5 + np.random.normal(0, 1)
        price_prediction_7d = signal_score * 15 + np.random.normal(0, 3)
        
        # Risk assessment
        volatility = np.std([r for r in price_data["returns"]])
        if volatility > 0.03:
            risk_level = "High"
        elif volatility > 0.015:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # For deep analysis, generate additional insights
        detected_patterns = []
        if price_data["prices"][-1] > price_data["prices"][-2] > price_data["prices"][-3]:
            detected_patterns.append("Rising Three Methods")
        if indicators["rsi"] < 30 and indicators["trend_direction"] == "Downtrend":
            detected_patterns.append("Oversold Reversal Potential")
        if indicators["rsi"] > 70 and indicators["trend_direction"] == "Uptrend":
            detected_patterns.append("Overbought Condition")
        if not detected_patterns:
            detected_patterns.append("No Clear Pattern")
        
        # Market cycle position (simplified simulation)
        cycles = ["Accumulation", "Uptrend", "Distribution", "Downtrend"]
        market_cycle_position = np.random.choice(cycles)
        
        # Volatility forecast
        vol_forecasts = ["Increasing", "Stable", "Decreasing"]
        volatility_forecast = np.random.choice(vol_forecasts)
        
        # Analysis weights (simulating the different DQN components from the paper)
        price_weight = np.random.uniform(30, 50)
        sentiment_weight = np.random.uniform(20, 40)
        technical_weight = 100 - price_weight - sentiment_weight
        
        return {
            "trade_signal": trade_signal,
            "confidence_level": confidence_level,
            "price_prediction_24h": price_prediction_24h,
            "price_prediction_7d": price_prediction_7d,
            "risk_level": risk_level,
            "price_analysis_weight": price_weight,
            "sentiment_analysis_weight": sentiment_weight,
            "technical_analysis_weight": technical_weight,
            "market_cycle_position": market_cycle_position,
            "volatility_forecast": volatility_forecast,
            "detected_patterns": detected_patterns,
            "pattern_reliability": "Medium" if len(detected_patterns) > 1 else "Low"
        }
    
    def _interpret_rsi(self, rsi: float) -> str:
        """Interpret RSI value."""
        if rsi > 70:
            return "Overbought"
        elif rsi < 30:
            return "Oversold"
        else:
            return "Neutral"
    
    def _get_trading_strategy(self, dqn_prediction: dict, indicators: dict) -> str:
        """Generate trading strategy recommendation based on M-DQN analysis."""
        signal = dqn_prediction["trade_signal"]
        
        if signal in ["Strong Buy", "Buy"]:
            if dqn_prediction["risk_level"] == "High":
                return """
1. Implement a phased buying strategy with 4-5 entry points
2. Allocate smaller position sizes due to high volatility
3. Focus on accumulating near the support level of ${:.2f}
4. Set stop losses at 5-7% below entry points
5. Consider using a mix of market and limit orders for entries""".format(indicators["support_level"])
            else:
                return """
1. Implement a phased buying strategy with 3-4 entry points
2. Allocate standard position sizes due to moderate/low volatility
3. Focus on accumulating between current price and the support level
4. Set stop losses at 3-5% below entry points
5. Use primarily limit orders for better entry prices"""
        
        elif signal == "Hold":
            return """
1. Maintain current positions without adding new exposure
2. Tighten stop losses on existing positions to protect profits
3. Consider taking partial profits if price approaches resistance at ${:.2f}
4. Monitor for changes in sentiment or technical indicators
5. Prepare conditional orders for both potential breakouts and breakdowns""".format(indicators["resistance_level"])
        
        elif signal in ["Sell", "Strong Sell"]:
            if dqn_prediction["risk_level"] == "High":
                return """
1. Implement a phased selling strategy with 3-4 exit points
2. Close positions more aggressively due to high downside risk
3. Consider establishing short positions after confirmation of trend
4. Set take-profit targets at 5-7% intervals
5. Use mainly market orders to ensure execution in volatile conditions"""
            else:
                return """
1. Implement a phased selling strategy with 4-5 exit points
2. Gradually reduce exposure as price approaches support levels
3. Consider partial conversion to stablecoins to preserve capital
4. Set take-profit targets at 3-5% intervals
5. Use limit orders for optimal exit prices"""
    
    def _get_position_management(self, dqn_prediction: dict, indicators: dict) -> str:
        """Generate position management recommendations."""
        signal = dqn_prediction["trade_signal"]
        risk = dqn_prediction["risk_level"]
        
        if signal in ["Strong Buy", "Buy"]:
            if risk == "High":
                position_size = "15-20% of available capital, divided across 4-5 entries"
                stop_loss = f"5-7% below each entry point, or ${indicators['support_level'] * 0.93:.2f} absolute floor"
            elif risk == "Medium":
                position_size = "20-30% of available capital, divided across 3-4 entries"
                stop_loss = f"4-5% below each entry point, or ${indicators['support_level'] * 0.95:.2f} absolute floor"
            else:
                position_size = "30-40% of available capital, divided across 3 entries"
                stop_loss = f"3-4% below each entry point, or ${indicators['support_level'] * 0.97:.2f} absolute floor"
                
            return f"""
Position Sizing: {position_size}
Stop Loss Levels: {stop_loss}
Take Profit Targets:
1. First target: ${indicators['resistance_level'] * 0.95:.2f} (close 25% of position)
2. Second target: ${indicators['resistance_level']:.2f} (close 25% of position)
3. Final target: ${indicators['resistance_level'] * 1.05:.2f} (close remaining position)
4. Trailing stop: Consider 10% trailing stop after breaking through resistance

Position Switching: Maintain long bias, no short position recommended at this time."""
        
        elif signal == "Hold":
            return f"""
Position Sizing: Maintain current position sizes without adding
Stop Loss Levels: Tighten stops to 5% below current price
Take Profit Targets: Set partial profit targets at ${indicators['resistance_level']:.2f}
Position Switching: Prepare for potential direction shift, but no immediate action needed

Hedging: Consider small hedge positions (5-10% of portfolio) if volatility increases."""
        
        elif signal in ["Sell", "Strong Sell"]:
            if risk == "High":
                position_size = "15-20% of available capital for short positions, divided across 3-4 entries"
                stop_loss = f"5-7% above each entry point, or ${indicators['resistance_level'] * 1.07:.2f} absolute ceiling"
            elif risk == "Medium":
                position_size = "20-25% of available capital for short positions, divided across 3 entries"
                stop_loss = f"4-6% above each entry point, or ${indicators['resistance_level'] * 1.05:.2f} absolute ceiling"
            else:
                position_size = "25-30% of available capital for short positions, divided across 2-3 entries"
                stop_loss = f"3-5% above each entry point, or ${indicators['resistance_level'] * 1.03:.2f} absolute ceiling"
                
            return f"""
Position Sizing: {position_size}
Stop Loss Levels: {stop_loss}
Take Profit Targets:
1. First target: ${indicators['support_level'] * 1.05:.2f} (close 25% of position)
2. Second target: ${indicators['support_level']:.2f} (close 25% of position)
3. Final target: ${indicators['support_level'] * 0.95:.2f} (close remaining position)
4. Trailing stop: Consider 10% trailing stop after breaking through support

Position Switching: Shift from long to short positions with confirmation of downtrend."""
    
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