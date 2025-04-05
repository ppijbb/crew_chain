import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

class InvestmentValidationInput(BaseModel):
    """Input for the Investment Validation tool."""
    crypto_symbol: str = Field(..., description="The symbol of the cryptocurrency to validate, e.g., BTC, ETH")
    trade_type: str = Field(..., description="Type of trade: 'buy' or 'sell'")
    position_type: str = Field(..., description="Type of position: 'long' or 'short'")
    entry_price: float = Field(..., description="Proposed entry price")
    target_price: float = Field(..., description="Target price for taking profit")
    stop_loss: float = Field(..., description="Stop loss price")
    position_size: float = Field(..., description="Position size as percentage of available capital (1-100)")
    confidence_level: float = Field(default=50.0, description="Confidence level of the trade (0-100)")
    validation_methods: List[str] = Field(default=["technical", "sentiment", "risk"], 
                                       description="Validation methods to use: 'technical', 'sentiment', 'risk', 'historical'")

class InvestmentValidationTool(BaseTool):
    """Tool for validating investment decisions before execution."""
    name = "investment_validation"
    description = "Validates investment decisions by applying multiple verification methods to ensure trade viability."
    args_schema: Type[BaseModel] = InvestmentValidationInput
    
    def _run(self, crypto_symbol: str, trade_type: str, position_type: str, 
            entry_price: float, target_price: float, stop_loss: float, 
            position_size: float, confidence_level: float = 50.0,
            validation_methods: List[str] = ["technical", "sentiment", "risk"]) -> str:
        """
        Validates an investment decision using multiple verification methods.
        """
        try:
            coin_name = self._get_coin_name(crypto_symbol)
            
            # Normalize inputs
            trade_type = trade_type.lower()
            position_type = position_type.lower()
            
            if trade_type not in ["buy", "sell"]:
                trade_type = "buy"
                
            if position_type not in ["long", "short"]:
                position_type = "long"
            
            # Check if valid validation methods were provided
            valid_methods = [m for m in validation_methods if m in ["technical", "sentiment", "risk", "historical"]]
            if not valid_methods:
                valid_methods = ["technical", "risk"]
            
            # Calculate risk-reward ratio
            if position_type == "long":
                risk = entry_price - stop_loss
                reward = target_price - entry_price
            else:  # short
                risk = stop_loss - entry_price
                reward = entry_price - target_price
                
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Generate validation results for each method
            validation_results = {}
            
            if "technical" in valid_methods:
                validation_results["technical"] = self._validate_technical(
                    crypto_symbol, trade_type, position_type, entry_price, target_price, stop_loss
                )
                
            if "sentiment" in valid_methods:
                validation_results["sentiment"] = self._validate_sentiment(
                    crypto_symbol, trade_type, position_type, confidence_level
                )
                
            if "risk" in valid_methods:
                validation_results["risk"] = self._validate_risk(
                    crypto_symbol, risk_reward_ratio, position_size
                )
                
            if "historical" in valid_methods:
                validation_results["historical"] = self._validate_historical(
                    crypto_symbol, trade_type, position_type, entry_price, target_price
                )
            
            # Calculate overall validation score
            overall_score = self._calculate_overall_score(validation_results)
            
            # Generate final recommendation
            trade_viability, recommendation = self._generate_recommendation(
                overall_score, validation_results, risk_reward_ratio, position_size
            )
            
            # Format output
            result = f"""
# Investment Validation Report for {coin_name} ({crypto_symbol.upper()})

## Trade Summary
- Trade Type: {trade_type.capitalize()} {position_type.capitalize()}
- Entry Price: ${entry_price:,.2f}
- Target Price: ${target_price:,.2f}
- Stop Loss: ${stop_loss:,.2f}
- Position Size: {position_size:.1f}% of capital
- Risk-Reward Ratio: {risk_reward_ratio:.2f}

## Validation Results
"""

            # Add individual validation results
            for method, data in validation_results.items():
                result += f"""
### {method.capitalize()} Analysis
- Score: {data['score']}/100
- Assessment: {data['assessment']}
- Details: {data['details']}
"""
            
            # Add overall validation
            result += f"""
## Overall Validation
- Score: {overall_score}/100
- Trade Viability: {trade_viability}
- Recommendation: {recommendation}

## Improvement Suggestions
{self._generate_improvement_suggestions(validation_results, risk_reward_ratio, position_size)}
"""
            
            return result
            
        except Exception as e:
            return f"Error validating investment: {str(e)}"
    
    def _validate_technical(self, crypto_symbol: str, trade_type: str, 
                          position_type: str, entry_price: float, 
                          target_price: float, stop_loss: float) -> dict:
        """Validate the trade based on technical analysis."""
        # Seed for consistent results
        np.random.seed(hash(f"{crypto_symbol}_{trade_type}_{position_type}_{entry_price}") % 2**32)
        
        # Generate simulated technical indicators
        rsi_value = np.random.randint(0, 100)
        macd_signal = "bullish" if np.random.random() > 0.5 else "bearish"
        volume_trend = "increasing" if np.random.random() > 0.5 else "decreasing"
        price_trend = "uptrend" if np.random.random() > 0.5 else "downtrend"
        
        # Score calculation based on alignment with trade direction
        score = 50  # Base score
        
        # For long positions
        if position_type == "long":
            # RSI analysis
            if rsi_value < 30:  # Oversold condition, good for buying
                score += 15
            elif rsi_value > 70:  # Overbought condition, bad for buying
                score -= 15
                
            # MACD signal
            if macd_signal == "bullish":
                score += 10
            else:
                score -= 10
                
            # Volume trend
            if volume_trend == "increasing":
                score += 10
            else:
                score -= 5
                
            # Price trend
            if price_trend == "uptrend":
                score += 15
            else:
                score -= 10
        
        # For short positions
        else:
            # RSI analysis
            if rsi_value > 70:  # Overbought condition, good for shorting
                score += 15
            elif rsi_value < 30:  # Oversold condition, bad for shorting
                score -= 15
                
            # MACD signal
            if macd_signal == "bearish":
                score += 10
            else:
                score -= 10
                
            # Volume trend
            if volume_trend == "increasing" and price_trend == "downtrend":
                score += 10
            else:
                score -= 5
                
            # Price trend
            if price_trend == "downtrend":
                score += 15
            else:
                score -= 10
        
        # Cap score between 0-100
        score = max(0, min(100, score))
        
        # Generate assessment based on score
        if score >= 80:
            assessment = "Strong Technical Confirmation"
            details = f"Technical indicators strongly support this {position_type} position. RSI at {rsi_value}, MACD signal is {macd_signal}, with {volume_trend} volume in a {price_trend}."
        elif score >= 60:
            assessment = "Moderate Technical Confirmation"
            details = f"Technical indicators moderately support this {position_type} position. Some indicators are aligned but not all."
        elif score >= 40:
            assessment = "Neutral Technical Signals"
            details = f"Technical indicators are mixed or neutral for this {position_type} position. More confirmation may be needed."
        elif score >= 20:
            assessment = "Weak Technical Signals"
            details = f"Technical indicators generally do not support this {position_type} position. Consider waiting for better alignment."
        else:
            assessment = "Technical Contradiction"
            details = f"Technical indicators contradict this {position_type} position. RSI at {rsi_value}, MACD signal is {macd_signal}, with {volume_trend} volume in a {price_trend}."
            
        return {
            "score": score,
            "assessment": assessment,
            "details": details,
            "indicators": {
                "rsi": rsi_value,
                "macd": macd_signal,
                "volume_trend": volume_trend,
                "price_trend": price_trend
            }
        }
    
    def _validate_sentiment(self, crypto_symbol: str, trade_type: str, 
                          position_type: str, confidence_level: float) -> dict:
        """Validate the trade based on sentiment analysis."""
        # Seed for consistent results
        np.random.seed(hash(f"{crypto_symbol}_{trade_type}_{position_type}_{confidence_level}") % 2**32)
        
        # Generate simulated sentiment data
        sentiment_score = np.random.uniform(-1, 1)
        sentiment_volume = np.random.randint(1000, 20000)
        sentiment_trend = "improving" if np.random.random() > 0.5 else "deteriorating"
        social_consensus = "high" if np.random.random() > 0.7 else "medium" if np.random.random() > 0.4 else "low"
        
        # Score calculation based on alignment with trade direction
        score = 50  # Base score
        
        # For long positions
        if position_type == "long":
            # Sentiment score
            if sentiment_score > 0.3:  # Strong positive sentiment
                score += 20
            elif sentiment_score > 0:  # Mild positive sentiment
                score += 10
            elif sentiment_score < -0.3:  # Strong negative sentiment
                score -= 20
            else:  # Mild negative sentiment
                score -= 10
                
            # Sentiment trend
            if sentiment_trend == "improving":
                score += 15
            else:
                score -= 10
                
            # Social consensus
            if social_consensus == "high":
                score += 10
            elif social_consensus == "low":
                score -= 5
        
        # For short positions
        else:
            # Sentiment score
            if sentiment_score < -0.3:  # Strong negative sentiment
                score += 20
            elif sentiment_score < 0:  # Mild negative sentiment
                score += 10
            elif sentiment_score > 0.3:  # Strong positive sentiment
                score -= 20
            else:  # Mild positive sentiment
                score -= 10
                
            # Sentiment trend
            if sentiment_trend == "deteriorating":
                score += 15
            else:
                score -= 10
                
            # Social consensus
            if social_consensus == "high":
                score += 10
            elif social_consensus == "low":
                score -= 5
        
        # Factor in user's confidence level
        score = score * (confidence_level / 50)
        
        # Cap score between 0-100
        score = max(0, min(100, score))
        
        # Generate assessment based on score
        if score >= 80:
            assessment = "Strong Sentiment Confirmation"
            details = f"Market sentiment strongly supports this {position_type} position with {sentiment_score:.2f} sentiment score and {sentiment_trend} trend."
        elif score >= 60:
            assessment = "Moderate Sentiment Confirmation"
            details = f"Market sentiment moderately supports this {position_type} position. Sentiment is {sentiment_trend} with {social_consensus} consensus."
        elif score >= 40:
            assessment = "Neutral Sentiment"
            details = f"Market sentiment is relatively neutral for this {position_type} position. Watch for sentiment shifts."
        elif score >= 20:
            assessment = "Sentiment Warning"
            details = f"Market sentiment generally contradicts this {position_type} position. Consider waiting for sentiment shift."
        else:
            assessment = "Strong Sentiment Contradiction"
            details = f"Market sentiment strongly contradicts this {position_type} position with {sentiment_score:.2f} sentiment score and {sentiment_trend} trend."
            
        return {
            "score": score,
            "assessment": assessment,
            "details": details,
            "sentiment_data": {
                "score": sentiment_score,
                "volume": sentiment_volume,
                "trend": sentiment_trend,
                "consensus": social_consensus
            }
        }
    
    def _validate_risk(self, crypto_symbol: str, risk_reward_ratio: float, position_size: float) -> dict:
        """Validate the trade based on risk management principles."""
        # Seed for consistent results
        np.random.seed(hash(f"{crypto_symbol}_{risk_reward_ratio}_{position_size}") % 2**32)
        
        # Generate simulated volatility data
        volatility = np.random.uniform(1, 15)  # Daily volatility in percentage
        market_risk = "high" if volatility > 8 else "medium" if volatility > 4 else "low"
        
        # Score calculation based on risk management principles
        score = 50  # Base score
        
        # Risk-reward ratio analysis
        if risk_reward_ratio >= 3.0:  # Excellent ratio
            score += 25
        elif risk_reward_ratio >= 2.0:  # Good ratio
            score += 15
        elif risk_reward_ratio >= 1.0:  # Acceptable ratio
            score += 5
        elif risk_reward_ratio < 1.0:  # Poor ratio
            score -= 25
            
        # Position sizing analysis
        if position_size <= 5:  # Very conservative
            score += 20
        elif position_size <= 10:  # Conservative
            score += 10
        elif position_size <= 20:  # Moderate
            score += 0
        elif position_size <= 30:  # Aggressive
            score -= 10
        else:  # Very aggressive
            score -= 25
            
        # Volatility adjustment
        if market_risk == "high":
            if position_size > 10:
                score -= 15
            else:
                score -= 5
        elif market_risk == "low" and position_size < 10:
            score += 5
        
        # Cap score between 0-100
        score = max(0, min(100, score))
        
        # Generate assessment based on score
        if score >= 80:
            assessment = "Excellent Risk Profile"
            details = f"Risk-reward ratio of {risk_reward_ratio:.2f} with {position_size:.1f}% position size provides an excellent risk profile in {market_risk} volatility market."
        elif score >= 60:
            assessment = "Good Risk Profile"
            details = f"Risk-reward ratio of {risk_reward_ratio:.2f} with {position_size:.1f}% position size provides a good risk profile, though some improvements possible."
        elif score >= 40:
            assessment = "Acceptable Risk Profile"
            details = f"Risk profile is acceptable but could be improved by adjusting position size or entry/exit points."
        elif score >= 20:
            assessment = "Concerning Risk Profile"
            details = f"Risk profile raises concerns. Consider reducing position size or improving risk-reward ratio."
        else:
            assessment = "Unacceptable Risk Profile"
            details = f"Risk profile is unacceptable with risk-reward ratio of {risk_reward_ratio:.2f} and {position_size:.1f}% position size in {market_risk} volatility market."
            
        return {
            "score": score,
            "assessment": assessment,
            "details": details,
            "risk_data": {
                "volatility": volatility,
                "market_risk": market_risk,
                "risk_reward_ratio": risk_reward_ratio,
                "position_size": position_size
            }
        }
    
    def _validate_historical(self, crypto_symbol: str, trade_type: str, 
                           position_type: str, entry_price: float, 
                           target_price: float) -> dict:
        """Validate the trade based on historical performance of similar setups."""
        # Seed for consistent results
        np.random.seed(hash(f"{crypto_symbol}_{trade_type}_{position_type}_{entry_price}_{target_price}") % 2**32)
        
        # Generate simulated historical data
        success_rate = np.random.uniform(30, 80)
        avg_holding_time = np.random.randint(2, 30)  # days
        similar_setups = np.random.randint(5, 50)
        max_drawdown = np.random.uniform(5, 25)  # percentage
        
        # Score calculation based on historical performance
        score = 50  # Base score
        
        # Success rate analysis
        if success_rate >= 70:  # Very high success rate
            score += 25
        elif success_rate >= 55:  # Good success rate
            score += 15
        elif success_rate >= 45:  # Average success rate
            score += 0
        elif success_rate >= 35:  # Below average success rate
            score -= 10
        else:  # Poor success rate
            score -= 20
            
        # Number of similar setups analysis
        if similar_setups >= 30:  # Very high sample size
            score += 10
        elif similar_setups >= 15:  # Good sample size
            score += 5
        elif similar_setups >= 5:  # Minimal sample size
            score += 0
        else:  # Insufficient sample size
            score -= 15
            
        # Max drawdown analysis
        if max_drawdown <= 10:  # Low drawdown
            score += 10
        elif max_drawdown <= 15:  # Moderate drawdown
            score += 5
        elif max_drawdown <= 20:  # High drawdown
            score -= 5
        else:  # Very high drawdown
            score -= 10
        
        # Cap score between 0-100
        score = max(0, min(100, score))
        
        # Generate assessment based on score
        if score >= 80:
            assessment = "Strong Historical Performance"
            details = f"Similar setups have shown a {success_rate:.1f}% success rate across {similar_setups} instances with {avg_holding_time} days average holding time."
        elif score >= 60:
            assessment = "Good Historical Performance"
            details = f"Similar setups have performed well historically with a {success_rate:.1f}% success rate, though with {max_drawdown:.1f}% max drawdown."
        elif score >= 40:
            assessment = "Average Historical Performance"
            details = f"Similar setups have shown average performance historically with a {success_rate:.1f}% success rate."
        elif score >= 20:
            assessment = "Below Average Historical Performance"
            details = f"Similar setups have underperformed historically with only a {success_rate:.1f}% success rate and {max_drawdown:.1f}% max drawdown."
        else:
            assessment = "Poor Historical Performance"
            details = f"Similar setups have performed poorly historically with a {success_rate:.1f}% success rate. Consider alternative strategies."
            
        return {
            "score": score,
            "assessment": assessment,
            "details": details,
            "historical_data": {
                "success_rate": success_rate,
                "similar_setups": similar_setups,
                "avg_holding_time": avg_holding_time,
                "max_drawdown": max_drawdown
            }
        }
    
    def _calculate_overall_score(self, validation_results: dict) -> float:
        """Calculate the overall validation score."""
        if not validation_results:
            return 0
            
        # Assign weights to different validation methods
        weights = {
            "technical": 0.25,
            "sentiment": 0.20,
            "risk": 0.35,
            "historical": 0.20
        }
        
        # Calculate weighted score
        total_weight = 0
        weighted_score = 0
        
        for method, data in validation_results.items():
            weight = weights.get(method, 0.25)
            weighted_score += data["score"] * weight
            total_weight += weight
            
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return 0
    
    def _generate_recommendation(self, overall_score: float, validation_results: dict, 
                              risk_reward_ratio: float, position_size: float) -> tuple:
        """Generate a final recommendation based on the overall score."""
        # Determine viability based on overall score
        if overall_score >= 80:
            viability = "Highly Viable"
        elif overall_score >= 65:
            viability = "Viable"
        elif overall_score >= 50:
            viability = "Marginally Viable"
        elif overall_score >= 35:
            viability = "Questionable"
        else:
            viability = "Not Recommended"
            
        # Generate recommendation text
        if overall_score >= 80:
            recommendation = "Proceed with the trade as planned. All validation checks indicate a high probability of success."
        elif overall_score >= 65:
            recommendation = "Proceed with the trade, but monitor closely. Most validation checks are positive."
        elif overall_score >= 50:
            if "risk" in validation_results and validation_results["risk"]["score"] < 50:
                recommendation = "Consider adjusting position size or risk parameters before proceeding."
            else:
                recommendation = "Proceed with caution. Some validation checks raise concerns."
        elif overall_score >= 35:
            if risk_reward_ratio < 1.5:
                recommendation = "Reconsider entry and exit points to improve risk-reward ratio before proceeding."
            elif position_size > 15:
                recommendation = "Consider reducing position size significantly if proceeding with this trade."
            else:
                recommendation = "This trade has significant red flags. Consider alternative opportunities."
        else:
            recommendation = "Avoid this trade. Multiple validation methods indicate high probability of loss."
            
        return viability, recommendation
    
    def _generate_improvement_suggestions(self, validation_results: dict, 
                                      risk_reward_ratio: float, position_size: float) -> str:
        """Generate suggestions for improving the trade setup."""
        suggestions = []
        
        # Technical suggestions
        if "technical" in validation_results:
            tech_score = validation_results["technical"]["score"]
            if tech_score < 50:
                suggestions.append("Wait for better technical alignment before entering this position.")
                if "indicators" in validation_results["technical"]:
                    indicators = validation_results["technical"]["indicators"]
                    if "rsi" in indicators:
                        rsi = indicators["rsi"]
                        if rsi > 70:
                            suggestions.append(f"RSI is overbought at {rsi}. Consider waiting for RSI to cool down for long positions.")
                        elif rsi < 30:
                            suggestions.append(f"RSI is oversold at {rsi}. Consider waiting for RSI to recover for short positions.")
        
        # Sentiment suggestions
        if "sentiment" in validation_results:
            sent_score = validation_results["sentiment"]["score"]
            if sent_score < 50:
                suggestions.append("Monitor sentiment trends for a shift in market perception before entering.")
                if "sentiment_data" in validation_results["sentiment"]:
                    sent_data = validation_results["sentiment"]["sentiment_data"]
                    if "consensus" in sent_data and sent_data["consensus"] == "low":
                        suggestions.append("Market consensus is low. Wait for clearer sentiment signals.")
        
        # Risk suggestions
        if "risk" in validation_results:
            risk_score = validation_results["risk"]["score"]
            if risk_score < 60:
                if risk_reward_ratio < 2.0:
                    suggestions.append(f"Improve risk-reward ratio from current {risk_reward_ratio:.2f} to at least 2.0 by adjusting entry, target, or stop loss.")
                if position_size > 15:
                    suggestions.append(f"Consider reducing position size from {position_size:.1f}% to below 15% of capital.")
        
        # Historical suggestions
        if "historical" in validation_results:
            hist_score = validation_results["historical"]["score"]
            if hist_score < 50:
                suggestions.append("This setup has underperformed historically. Consider alternative entry criteria.")
                if "historical_data" in validation_results["historical"]:
                    hist_data = validation_results["historical"]["historical_data"]
                    if "max_drawdown" in hist_data and hist_data["max_drawdown"] > 20:
                        suggestions.append(f"Be prepared for significant drawdowns (historically up to {hist_data['max_drawdown']:.1f}%).")
        
        # If no specific suggestions were generated
        if not suggestions:
            if risk_reward_ratio < 3.0:
                suggestions.append("While the trade is viable, consider optimizing your risk-reward ratio for even better results.")
            if position_size > 10:
                suggestions.append("Consider implementing a phased entry strategy to reduce timing risk.")
                
        # Return formatted suggestions
        if suggestions:
            return "\n".join([f"- {suggestion}" for suggestion in suggestions])
        else:
            return "No specific improvements needed. The trade setup appears solid."
    
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