from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_community.tools import DuckDuckGoSearchRun, SearxSearchRun
from langchain.agents import Tool
from src.crew_chain.tools.crypto_tools import CryptoPriceCheckTool, CryptoHistoricalDataTool
from src.crew_chain.tools.deep_search_tools import TwitterSentimentAnalysisTool, DeepMarketAnalysisTool
from src.crew_chain.tools.advanced_sentiment_tools import MultilayerSentimentAnalysisTool
from src.crew_chain.tools.neural_prediction_tools import DeepLearningPredictionTool
from src.crew_chain.tools.investment_validation_tools import InvestmentValidationTool
from src.crew_chain.tools.recurring_operations import RecurringOperationTool
from src.crew_chain.tools.mcp_kafka_connector import MCPKafkaProducerTool, MCPKafkaConsumerTool, KafkaConfig
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CryptoTradingCrew")

@CrewBase
class CryptoTradingCrew():
    """Crypto Trading Crew for automated cryptocurrency trading"""

    agents_config = 'config/crypto_agents.yaml'
    tasks_config = 'config/crypto_tasks.yaml'
    
    def __init__(self, config=None):
        """Initialize the crypto trading crew with configuration."""
        self.config = config or {}
        self.logger = logger
        
        # Initialize MCP Kafka connectors if enabled
        if self.config.get("use_mcp", True):
            self._initialize_mcp_kafka()
        
        self.market_data_handlers = {}
        self.trade_signal_handlers = {}
    
    def _initialize_mcp_kafka(self):
        """Initialize MCP Kafka connectors."""
        try:
            # Get Kafka configuration from config
            kafka_config = KafkaConfig(
                bootstrap_servers=self.config.get("kafka_bootstrap_servers", "localhost:9092"),
                consumer_group_id=self.config.get("kafka_consumer_group", "crew_chain_consumer"),
                auto_offset_reset=self.config.get("kafka_auto_offset_reset", "latest"),
                security_protocol=self.config.get("kafka_security_protocol"),
                sasl_mechanism=self.config.get("kafka_sasl_mechanism"),
                sasl_username=self.config.get("kafka_sasl_username"),
                sasl_password=self.config.get("kafka_sasl_password")
            )
            
            # Initialize producer tool
            self.mcp_producer = MCPKafkaProducerTool(config=kafka_config)
            
            # Initialize consumer tool and register handlers
            self.mcp_consumer = MCPKafkaConsumerTool(config=kafka_config)
            
            # Register default handlers
            self.mcp_consumer.register_handler("market_data_handler", self._market_data_handler)
            self.mcp_consumer.register_handler("trade_signal_handler", self._trade_signal_handler)
            
            # Start consumers if auto_consume is enabled
            if self.config.get("auto_consume_market_data", True):
                self.mcp_consumer._run(
                    topic="market_data",
                    handler_name="market_data_handler"
                )
                
            if self.config.get("auto_consume_trade_signals", True):
                self.mcp_consumer._run(
                    topic="trading_signals",
                    handler_name="trade_signal_handler"
                )
                
            self.logger.info("MCP Kafka connectors initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing MCP Kafka connectors: {str(e)}")
            raise

    def _market_data_handler(self, message: Dict[str, Any]):
        """
        Handle market data messages from Kafka.
        
        Args:
            message: Market data message
        """
        try:
            # Extract data from the message
            message_type = message.get("message_type", "")
            payload = message.get("payload", {})
            
            if message_type == "market_data":
                symbol = payload.get("symbol", "")
                price = payload.get("price", 0.0)
                
                self.logger.info(f"Received market data for {symbol}: ${price}")
                
                # Execute any registered callbacks for this symbol
                handlers = self.market_data_handlers.get(symbol.upper(), [])
                for handler in handlers:
                    try:
                        handler(payload)
                    except Exception as e:
                        self.logger.error(f"Error in market data handler for {symbol}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Error handling market data message: {str(e)}")

    def _trade_signal_handler(self, message: Dict[str, Any]):
        """
        Handle trade signal messages from Kafka.
        
        Args:
            message: Trade signal message
        """
        try:
            # Extract data from the message
            message_type = message.get("message_type", "")
            payload = message.get("payload", {})
            
            if message_type == "trade_signal":
                symbol = payload.get("symbol", "")
                action = payload.get("action", "")
                position_type = payload.get("position_type", "")
                
                self.logger.info(f"Received trade signal for {symbol}: {action} {position_type}")
                
                # Execute any registered callbacks for this symbol
                handlers = self.trade_signal_handlers.get(symbol.upper(), [])
                for handler in handlers:
                    try:
                        handler(payload)
                    except Exception as e:
                        self.logger.error(f"Error in trade signal handler for {symbol}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Error handling trade signal message: {str(e)}")

    def register_market_data_handler(self, symbol: str, handler_func):
        """
        Register a handler for market data for a specific symbol.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., BTC)
            handler_func: Function to call when market data is received
        """
        symbol = symbol.upper()
        if symbol not in self.market_data_handlers:
            self.market_data_handlers[symbol] = []
            
        self.market_data_handlers[symbol].append(handler_func)
        self.logger.info(f"Registered market data handler for {symbol}")

    def register_trade_signal_handler(self, symbol: str, handler_func):
        """
        Register a handler for trade signals for a specific symbol.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., BTC)
            handler_func: Function to call when a trade signal is received
        """
        symbol = symbol.upper()
        if symbol not in self.trade_signal_handlers:
            self.trade_signal_handlers[symbol] = []
            
        self.trade_signal_handlers[symbol].append(handler_func)
        self.logger.info(f"Registered trade signal handler for {symbol}")

    def send_market_data(self, symbol: str, price: float, volume: Optional[float] = None, 
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Send market data to the MCP Kafka system.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., BTC)
            price: Current price
            volume: Trading volume (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Result message
        """
        if not hasattr(self, 'mcp_producer'):
            return "MCP Kafka producer not initialized"
            
        payload = {
            'symbol': symbol.upper(),
            'price': float(price),
            'volume': float(volume or 0),
            'timestamp': datetime.now().isoformat()
        }
        
        return self.mcp_producer._run(
            topic="market_data",
            message_type="market_data",
            payload=payload,
            metadata=metadata or {}
        )

    def send_trade_signal(self, symbol: str, action: str, position_type: str = "long",
                       entry_price: Optional[float] = None, target_price: Optional[float] = None,
                       stop_loss: Optional[float] = None, confidence: Optional[float] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a trade signal to the MCP Kafka system.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., BTC)
            action: Trade action ('buy' or 'sell')
            position_type: Position type ('long' or 'short')
            entry_price: Entry price (optional)
            target_price: Target price (optional)
            stop_loss: Stop loss price (optional)
            confidence: Confidence level (0-100, optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Result message
        """
        if not hasattr(self, 'mcp_producer'):
            return "MCP Kafka producer not initialized"
            
        payload = {
            'symbol': symbol.upper(),
            'action': action.lower(),
            'position_type': position_type.lower(),
            'entry_price': entry_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'confidence': confidence or 50.0,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.mcp_producer._run(
            topic="trading_signals",
            message_type="trade_signal",
            payload=payload,
            metadata=metadata or {}
        )

    @agent
    def market_analyst(self) -> Agent:
        """Agent responsible for analyzing market trends and news"""
        search_tool = DuckDuckGoSearchRun()
        tools = [
            Tool(
                name="Web Search",
                func=search_tool.run,
                description="Useful for searching information about cryptocurrency market trends and news"
            ),
            CryptoPriceCheckTool(),
            TwitterSentimentAnalysisTool(),
            DeepMarketAnalysisTool(),
            MultilayerSentimentAnalysisTool()
        ]
        
        # Add MCP tools if enabled
        if hasattr(self, 'mcp_producer'):
            tools.append(self.mcp_producer)
        
        return Agent(
            config=self.agents_config['market_analyst'],
            tools=tools,
            verbose=True
        )

    @agent
    def crypto_researcher(self) -> Agent:
        """Agent responsible for deep research on specific cryptocurrencies"""
        search_tool = DuckDuckGoSearchRun()
        tools = [
            Tool(
                name="Web Search",
                func=search_tool.run,
                description="Useful for researching specific cryptocurrencies and their technology"
            ),
            CryptoPriceCheckTool(),
            CryptoHistoricalDataTool(),
            TwitterSentimentAnalysisTool(),
            MultilayerSentimentAnalysisTool(),
            DeepLearningPredictionTool()
        ]
        
        # Add MCP tools if enabled
        if hasattr(self, 'mcp_producer'):
            tools.append(self.mcp_producer)
            
        return Agent(
            config=self.agents_config['crypto_researcher'],
            tools=tools,
            verbose=True
        )

    @agent
    def trading_strategist(self) -> Agent:
        """Agent responsible for creating trading strategies"""
        tools = [
            CryptoPriceCheckTool(),
            CryptoHistoricalDataTool(),
            DeepMarketAnalysisTool(),
            TwitterSentimentAnalysisTool(),
            MultilayerSentimentAnalysisTool(),
            DeepLearningPredictionTool()
        ]
        
        # Add MCP tools if enabled
        if hasattr(self, 'mcp_producer'):
            tools.append(self.mcp_producer)
            
        # Add investment validation if enabled
        if self.config.get("use_investment_validation", True):
            tools.append(InvestmentValidationTool())
            
        return Agent(
            config=self.agents_config['trading_strategist'],
            tools=tools,
            verbose=True
        )

    @agent
    def trade_executor(self) -> Agent:
        """Agent responsible for executing trades based on strategies"""
        tools = [
            CryptoPriceCheckTool(),
            DeepMarketAnalysisTool()
        ]
        
        # Add MCP tools if enabled
        if hasattr(self, 'mcp_producer'):
            tools.append(self.mcp_producer)
            
        # Add investment validation if enabled
        if self.config.get("use_investment_validation", True):
            tools.append(InvestmentValidationTool())
            
        return Agent(
            config=self.agents_config['trade_executor'],
            tools=tools,
            verbose=True
        )

    @task
    def market_analysis_task(self) -> Task:
        """Task to analyze current market conditions"""
        return Task(
            config=self.tasks_config['market_analysis_task'],
        )

    @task
    def crypto_research_task(self) -> Task:
        """Task to research specific cryptocurrencies"""
        return Task(
            config=self.tasks_config['crypto_research_task'],
        )

    @task
    def strategy_development_task(self) -> Task:
        """Task to develop trading strategies"""
        return Task(
            config=self.tasks_config['strategy_development_task'],
        )

    @task
    def trade_execution_task(self) -> Task:
        """Task to execute trades based on strategies"""
        return Task(
            config=self.tasks_config['trade_execution_task'],
            output_file='trading_report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Crypto Trading crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    def get_tools(self):
        """Initialize and return all tools needed for the crew."""
        tools = []
        
        # Add existing tools
        tools.append(CryptoPriceCheckTool())
        tools.append(CryptoHistoricalDataTool())
        tools.append(TwitterSentimentAnalysisTool())
        tools.append(DeepMarketAnalysisTool())
        tools.append(MultilayerSentimentAnalysisTool())
        tools.append(DeepLearningPredictionTool())
        
        # Add investment validation tool if enabled
        if self.config.get("use_investment_validation", True):
            tools.append(InvestmentValidationTool())
            
        # Add recurring operation tool if enabled
        if self.config.get("use_recurring_operations", True):
            tools.append(RecurringOperationTool())
            
        # Add MCP Kafka tools if enabled
        if self.config.get("use_mcp", True):
            # Initialize Kafka config
            kafka_config = KafkaConfig(
                bootstrap_servers=self.config.get("kafka_bootstrap_servers", "localhost:9092"),
                consumer_group_id=self.config.get("kafka_consumer_group", "crew_chain_consumer"),
                auto_offset_reset=self.config.get("kafka_auto_offset_reset", "latest"),
                security_protocol=self.config.get("kafka_security_protocol"),
                sasl_mechanism=self.config.get("kafka_sasl_mechanism"),
                sasl_username=self.config.get("kafka_sasl_username"),
                sasl_password=self.config.get("kafka_sasl_password")
            )
            
            # Add producer and consumer tools
            tools.append(MCPKafkaProducerTool(config=kafka_config))
            tools.append(MCPKafkaConsumerTool(config=kafka_config))
            
        return tools

    def run(self):
        """Execute the crypto trading crew workflow."""
        try:
            self.logger.info("Starting Crypto Trading Crew")
            
            # Setup recurring operations if enabled
            if self.config.get("use_recurring_operations", True):
                self._setup_recurring_operations()
            
            # Create agents with the tools
            self.logger.info("Creating agents...")
            agents = self._create_agents()
            
            # Create the crew with the agents
            self.logger.info("Creating crew...")
            crew = self._create_crew(agents)
            
            # Start the crew tasks
            self.logger.info("Executing crew tasks...")
            result = crew.kickoff()
            
            # Validate investment decisions if enabled
            if self.config.get("use_investment_validation", True):
                self._validate_investments(result)
                
            # Send trading decisions to MCP Kafka if enabled
            if hasattr(self, 'mcp_producer') and self.config.get("publish_trading_decisions", True):
                self._publish_trading_decisions(result)
            
            self.logger.info("Crypto Trading Crew completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Crypto Trading Crew: {str(e)}")
            raise
    
    def _publish_trading_decisions(self, result):
        """
        Publish trading decisions to MCP Kafka.
        
        Args:
            result: Result from the crew
        """
        try:
            if not isinstance(result, dict):
                self.logger.warning("Cannot publish trading decisions: result is not a dictionary")
                return
                
            # Extract trading decision
            trading_decision = result.get("trading_decision")
            if not trading_decision:
                self.logger.warning("No trading decision found in result")
                return
                
            # Get target symbol
            symbol = trading_decision.get("crypto_symbol", self.config.get("target_crypto", "BTC"))
            
            # Send trade signal
            self.send_trade_signal(
                symbol=symbol,
                action=trading_decision.get("action", "hold"),
                position_type=trading_decision.get("position", "long"),
                entry_price=trading_decision.get("entry_price"),
                target_price=trading_decision.get("target_price"),
                stop_loss=trading_decision.get("stop_loss"),
                confidence=trading_decision.get("confidence", 50.0),
                metadata={
                    "source": "crew_chain",
                    "strategy": trading_decision.get("strategy_name", "default"),
                    "validation_score": trading_decision.get("validation_score"),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"Published trading decision for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error publishing trading decisions: {str(e)}")
    
    def _setup_recurring_operations(self):
        """Set up recurring operations based on configuration."""
        self.logger.info("Setting up recurring operations...")
        
        recurring_tool = RecurringOperationTool()
        
        # Configure market scan
        if self.config.get("recurring_market_scan", True):
            interval = self.config.get("market_scan_interval", 60)  # Default: every hour
            self.logger.info(f"Setting up recurring market scan every {interval} minutes")
            
            recurring_tool._run(
                operation_type='market_scan',
                schedule_type='interval',
                interval=interval,
                parameters={
                    'crypto_symbol': self.config.get("target_crypto", "BTC"),
                    'scan_type': 'comprehensive'
                }
            )
        
        # Configure daily portfolio rebalance
        if self.config.get("recurring_portfolio_rebalance", True):
            time_of_day = self.config.get("portfolio_rebalance_time", "16:00")
            self.logger.info(f"Setting up daily portfolio rebalance at {time_of_day}")
            
            recurring_tool._run(
                operation_type='portfolio_rebalance',
                schedule_type='daily',
                time_of_day=time_of_day,
                parameters={
                    'tolerance': self.config.get("rebalance_tolerance", 5.0)
                }
            )
        
        # Configure weekly trade verification
        if self.config.get("recurring_trade_verification", True):
            day = self.config.get("trade_verification_day", "monday")
            time = self.config.get("trade_verification_time", "09:00")
            self.logger.info(f"Setting up weekly trade verification on {day} at {time}")
            
            recurring_tool._run(
                operation_type='trade_verification',
                schedule_type='weekly',
                day_of_week=day,
                time_of_day=time,
                parameters={
                    'lookback_days': self.config.get("verification_lookback_days", 7),
                    'min_profit': self.config.get("verification_min_profit", 1.5)
                }
            )
        
        # Configure sentiment analysis
        if self.config.get("recurring_sentiment_analysis", True):
            frequency = self.config.get("sentiment_analysis_frequency", "daily")
            time = self.config.get("sentiment_analysis_time", "09:30")
            self.logger.info(f"Setting up {frequency} sentiment analysis at {time}")
            
            recurring_tool._run(
                operation_type='sentiment_analysis',
                schedule_type=frequency,
                time_of_day=time,
                parameters={
                    'crypto_symbol': self.config.get("target_crypto", "BTC"),
                    'sources': self.config.get("sentiment_sources", ["twitter", "reddit", "news"])
                }
            )
    
    def _validate_investments(self, crew_result):
        """Validate investment decisions from crew results."""
        self.logger.info("Validating investment decisions...")
        
        # Extract trading recommendations from crew results
        if not crew_result or not isinstance(crew_result, dict):
            self.logger.warning("No valid crew results to validate")
            return
        
        # Extract trading decision
        trading_decision = crew_result.get("trading_decision", {})
        if not trading_decision:
            self.logger.warning("No trading decision found in crew results")
            return
        
        # Create validation tool
        validation_tool = InvestmentValidationTool()
        
        # Prepare validation parameters
        try:
            validation_input = {
                'crypto_symbol': trading_decision.get("crypto_symbol", self.config.get("target_crypto", "BTC")),
                'trade_type': trading_decision.get("action", "buy"),
                'position_type': trading_decision.get("position", "long"),
                'entry_price': float(trading_decision.get("entry_price", 0)),
                'target_price': float(trading_decision.get("target_price", 0)),
                'stop_loss': float(trading_decision.get("stop_loss", 0)),
                'position_size': float(trading_decision.get("position_size", 5.0)),
                'confidence_level': float(trading_decision.get("confidence", 50.0)),
                'validation_methods': self.config.get("validation_methods", ["technical", "sentiment", "risk", "historical"])
            }
            
            # Run validation
            validation_result = validation_tool._run(**validation_input)
            
            # Log validation result
            self.logger.info(f"Investment validation completed")
            
            # Add validation results to crew results
            crew_result["investment_validation"] = validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating investment: {str(e)}")
            
    def cleanup(self):
        """Clean up resources when the crew is no longer needed."""
        if hasattr(self, 'mcp_producer'):
            self.mcp_producer.cleanup()
            
        if hasattr(self, 'mcp_consumer'):
            self.mcp_consumer.cleanup() 