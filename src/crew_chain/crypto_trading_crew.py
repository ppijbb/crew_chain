from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_community.tools import DuckDuckGoSearchRun, SearxSearchRun
from langchain.agents import Tool
from src.crew_chain.tools.crypto_tools import CryptoPriceCheckTool, CryptoHistoricalDataTool
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
            CryptoPriceCheckTool()
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
            CryptoHistoricalDataTool()
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
            CryptoHistoricalDataTool()
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
            CryptoPriceCheckTool()
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