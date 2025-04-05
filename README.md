# Crew Chain - Cryptocurrency Trading System

An AI-powered cryptocurrency trading automation system built using CrewAI.

## Overview

This project leverages CrewAI to create an automated cryptocurrency trading system that follows these key principles:

1. **Buy Low, Sell High** - The system identifies optimal entry and exit points based on detailed market analysis
2. **Phased Buying & Selling** - Implements systematic scaling in and out of positions
3. **Rapid Position Switching** - Capable of quickly transitioning between long and short positions based on market conditions

## System Architecture

The system uses a crew of specialized AI agents working together:

- **Market Analyst** - Analyzes overall cryptocurrency market trends, sentiment, and events
- **Crypto Researcher** - Conducts in-depth research on specific cryptocurrencies
- **Trading Strategist** - Develops comprehensive trading strategies with clear entry/exit points
- **Trade Executor** - Executes trades according to defined strategies with precision

## Advanced AI Capabilities

Our system implements cutting-edge AI technologies for cryptocurrency market analysis and prediction:

### Neural Network Price Prediction

Our system utilizes state-of-the-art deep learning models for price prediction, incorporating:

- **LSTM with Attention Mechanism** - Captures temporal dependencies and focuses on the most relevant time periods
- **Temporal Convolutional Networks** - Identifies complex patterns across different timeframes
- **Transformer-based Models** - Processes market data with sophisticated attention mechanisms
- **Bidirectional GRU with Self-Attention** - Analyzes both past and future context for more accurate predictions

The neural prediction system integrates both price data and sentiment analysis, using separate pathways that are combined through a sophisticated fusion mechanism, significantly improving prediction accuracy compared to traditional models.

### Multilayer Sentiment Analysis

Our advanced sentiment analysis system analyzes data from multiple sources:

- **Social Media Analysis** - Processes Twitter, Reddit, and forum discussions using NLP techniques
- **News Sentiment Processing** - Analyzes news articles and press releases to gauge market sentiment
- **Cross-Source Sentiment Fusion** - Combines sentiment from multiple sources with appropriate weighting
- **Sentiment Divergence Detection** - Identifies discrepancies between different sources that might signal market shifts

Research has shown that incorporating sentiment analysis can approximately double the accuracy of cryptocurrency price prediction models.

### Multi-level Deep Q-Network (M-DQN) Approach

The system's deep search capabilities are powered by a Multi-level Deep Q-Network (M-DQN) approach inspired by recent research in cryptocurrency trading. The model:

1. Processes historical price data and market indicators in one pathway
2. Analyzes sentiment data from social media in a parallel pathway
3. Combines both analyses to produce final trading recommendations

This multi-level structure increases learning effectiveness by processing different data types through specialized pathways before integration, resulting in more accurate trading signals.

## Bidirectional Trading Strategy

The system employs a sophisticated bidirectional trading approach:

- **Long and Short Positions**: Provides entry and exit points for both long and short positions
- **Dynamic Position Switching**: Quickly transitions between long and short positions based on sentiment shifts and technical indicators
- **Adaptive Position Sizing**: Adjusts position sizes based on market volatility and signal conviction
- **Integrated Risk Management**: Implements specific stop-loss levels for both long and short positions

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd crew_chain
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the cryptocurrency trading system:

```
python -m src.crew_chain.crypto_trading_main run --crypto BTC
```

You can specify which cryptocurrency to analyze and trade by changing the `--crypto` parameter (default is BTC).

## How It Works

1. The **Market Analyst** examines overall market conditions, sentiment, and trends using deep search tools
2. The **Crypto Researcher** analyzes the specific cryptocurrency's fundamentals, technicals, and price patterns
3. The **Trading Strategist** develops a bidirectional trading strategy with entry/exit points for both long and short positions
4. The **Trade Executor** implements the strategy with proper discipline and records all trading activity

## Custom Tools

The system includes sophisticated tools for cryptocurrency analysis:

- **CryptoPriceCheckTool**: Gets real-time cryptocurrency price and market data
- **CryptoHistoricalDataTool**: Retrieves historical price data for technical analysis
- **TwitterSentimentAnalysisTool**: Analyzes Twitter sentiment about cryptocurrencies
- **DeepMarketAnalysisTool**: Performs market analysis using the M-DQN approach
- **MultilayerSentimentAnalysisTool**: Conducts advanced sentiment analysis across multiple data sources
- **DeepLearningPredictionTool**: Generates price predictions using neural network models

## Configuration

You can modify the agent configurations in `src/crew_chain/config/crypto_agents.yaml` and task configurations in `src/crew_chain/config/crypto_tasks.yaml`.

## Output

The system generates a comprehensive trading report that includes:
- Market analysis with sentiment insights
- Cryptocurrency research with bidirectional perspectives
- Trading strategy with entry/exit points for both long and short positions
- Trade execution details and performance tracking

## References

The AI models in this system are based on research findings from:

- Yasir, M., et al. (2023), "Deep-learning-assisted business intelligence model for cryptocurrency forecasting using social media sentiment", Journal of Enterprise Information Management
- Research on sentiment analysis in cryptocurrency markets demonstrates that incorporating social media sentiment data can significantly improve the performance of prediction models, approximately doubling the overall accuracy.
