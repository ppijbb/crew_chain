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

1. The **Market Analyst** examines overall market conditions, sentiment, and trends
2. The **Crypto Researcher** analyzes the specific cryptocurrency's fundamentals, technicals, and price patterns
3. The **Trading Strategist** develops a trading strategy with entry/exit points, position sizing, and risk management
4. The **Trade Executor** implements the strategy with proper discipline and records all trading activity

## Custom Tools

The system includes custom tools for cryptocurrency analysis:

- **CryptoPriceCheckTool**: Gets real-time cryptocurrency price and market data
- **CryptoHistoricalDataTool**: Retrieves historical price data for technical analysis and support/resistance identification

## Configuration

You can modify the agent configurations in `src/crew_chain/config/crypto_agents.yaml` and task configurations in `src/crew_chain/config/crypto_tasks.yaml`.

## Output

The system generates a comprehensive trading report that includes:
- Market analysis
- Cryptocurrency research
- Trading strategy with entry/exit points
- Trade execution details and performance tracking
