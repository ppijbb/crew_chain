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
import os

@CrewBase
class CryptoTradingCrew():
    """Crypto Trading Crew for automated cryptocurrency trading"""

    agents_config = 'config/crypto_agents.yaml'
    tasks_config = 'config/crypto_tasks.yaml'

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
            DeepMarketAnalysisTool(),
            MultilayerSentimentAnalysisTool(),
            DeepLearningPredictionTool()
        ]
        
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
            
            self.logger.info("Crypto Trading Crew completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Crypto Trading Crew: {str(e)}")
            raise
    
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
            self.logger.info(f"Investment validation completed: {validation_result}")
            
            # Add validation results to crew results
            crew_result["investment_validation"] = validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating investment: {str(e)}") 