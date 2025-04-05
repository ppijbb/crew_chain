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

## Advanced Features

### Investment Validation System

The cryptocurrency trading system includes a robust investment validation system that verifies trading decisions before execution. This adds an essential layer of risk management and helps ensure that all trading actions align with the overall strategy. 

Key features of the investment validation system:

- **Multi-method Validation**: Trading decisions are validated through multiple methods including technical analysis alignment, sentiment correlation, risk assessment, and historical performance comparison.
- **Risk-Reward Analysis**: Every trade is analyzed for its risk-reward ratio to ensure favorable trading conditions.
- **Position Sizing Verification**: The system checks that position sizes are appropriate given market volatility and other risk factors.
- **Comprehensive Reporting**: Detailed validation reports provide actionable insights on trade viability with specific improvement suggestions.

### Recurring Operations

The system supports automated scheduling of critical trading operations, ensuring consistent execution without manual intervention:

- **Flexible Scheduling**: Operations can be scheduled at intervals (minutes/hours), daily at specific times, or weekly on designated days.
- **Key Recurring Operations**:
  - **Market Scans**: Regular technical and fundamental analysis of target cryptocurrencies.
  - **Portfolio Rebalancing**: Automatic portfolio adjustments to maintain target allocations.
  - **Trade Verification**: Regular checks of recent trading activity to identify patterns and verify performance.
  - **Sentiment Analysis**: Scheduled sentiment monitoring across social media, news, and other sources.
  
- **Execution Tracking**: All recurring operations are tracked with detailed execution history, success rates, and error reporting.
- **Dynamic Management**: Operations can be paused, resumed, or modified as market conditions change.

This automated scheduling system ensures that critical analysis and trading functions occur consistently even in fast-moving markets, maintaining strategic discipline while adapting to changing conditions.

### MCP Kafka Integration

The system now incorporates Message Consumption Protocol (MCP) via Kafka for real-time data streaming and processing:

- **Real-Time Data Flow**: Seamless streaming of market data, trading signals, and system events through Kafka topics.
- **Bidirectional Communication**: Both publishes trading decisions and consumes market data through the same infrastructure.
- **Multi-Database Storage**: Automatically stores trading data in both MongoDB (for flexible schema) and PostgreSQL (for structured queries).
- **Event-Driven Architecture**: Enables reactive trading based on real-time market events through registered handlers.

Key components of the MCP Kafka integration:

- **Kafka Message Broker**: High-throughput, distributed messaging system for real-time data processing.
- **MongoDB Integration**: For storing unstructured and semi-structured trading data with flexible schemas.
- **PostgreSQL Integration**: For structured financial data storage and complex analytical queries.
- **Custom MCP API**: RESTful API service for interacting with the Kafka messaging system.

This integration enables the system to handle high volumes of real-time market data while maintaining data consistency across multiple storage systems, making it suitable for mission-critical financial operations.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd crew_chain
```

2. Install dependencies:

### Using uv (recommended)
[uv](https://github.com/astral-sh/uv) is a faster, more reliable Python package installer and dependency resolver, up to 10-100x faster than pip.

```bash
# Install uv if you don't have it
pip install uv

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Generate a lock file for reproducible builds
uv lock
```

### Using the installation script
For the easiest installation, use our provided script:

```bash
# Make the script executable
chmod +x install_with_uv.sh

# Run the installation script
./install_with_uv.sh
```

This script will:
1. Install uv if not already installed
2. Create a virtual environment
3. Install all dependencies using uv (from requirements.txt or pyproject.toml)
4. Generate a lock file (uv.lock) for reproducible builds

#### Reproducible builds with uv.lock

Once you have a lock file, you can ensure that all team members and deployment environments use the exact same dependency versions:

```bash
# Install exact dependency versions from the lock file
uv sync
```

This approach prevents "it works on my machine" problems by guaranteeing consistent environments across development and production.

### Using pip (traditional method)
If you prefer using pip:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

### Using the run script (recommended)

The easiest way to run the application is with the provided run script:

On Linux/Mac:
```bash
# Make the script executable
chmod +x run.sh

# Run the application
./run.sh
```

On Windows:
```
# Run the application
run.bat
```

These scripts will:
1. Check if a virtual environment exists, and create one if needed
2. Run the application using uv, which ensures all dependencies are available
3. Pass any command-line arguments to the application

### Manual execution

If you prefer to run the application manually:

```bash
# With uv (recommended - handles dependencies automatically)
uv run src/crew_chain/crypto_trading_main.py

# Traditional method (requires activated virtual environment)
python src/crew_chain/crypto_trading_main.py
```

## Model Context Protocol (MCP) Integration

Crew Chain now includes support for the Model Context Protocol (MCP), enabling integration with various AI services and tools:

- **AI Agent Communication**: Allows agents to communicate with external AI services using the standardized MCP protocol.
- **MCP Tool Integration**: Supports using external MCP servers as tools within the trading system.
- **Extensible Architecture**: Easily add new MCP-based capabilities without changing the core system.

The MCP integration enables the system to leverage specialized AI tools from the growing MCP ecosystem for tasks like:
- Advanced data analysis
- Market sentiment evaluation
- Technical pattern recognition
- Trade decision validation

To use MCP capabilities, follow the standard MCP configuration patterns in your configuration files.

## Docker Setup

The system can be deployed using Docker, which simplifies setup and ensures consistent environments:

```bash
# Start all services with Docker Compose
docker-compose up -d
```

This will start the following services:
- Zookeeper and Kafka for message brokering
- MongoDB and PostgreSQL for data storage
- Kafka UI, MongoDB Express, and pgAdmin for administration
- MCP API service for interacting with the system