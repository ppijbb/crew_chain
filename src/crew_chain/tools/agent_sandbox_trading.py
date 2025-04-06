"""
AI Agent Sandbox Trading Integration

This module integrates the AI trading agents with the sandbox trading environment
to test and evaluate trading strategies with virtual funds.
"""
import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from src.crew_chain.tools.sandbox_trading import SandboxTradingTool
from src.crew_chain.crypto_trading_crew import CryptoTradingCrew

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentSandboxTrading:
    """Class to integrate AI agents with sandbox trading for evaluation."""
    
    def __init__(self, config_path: Optional[str] = None, account_id: Optional[str] = None):
        """
        Initialize the agent sandbox trading integration.
        
        Args:
            config_path: Path to agent configuration file
            account_id: Sandbox account ID to use (creates new if None)
        """
        self.sandbox_tool = SandboxTradingTool()
        self.crypto_crew = CryptoTradingCrew(config_path=config_path)
        self.account_id = account_id
        
        # Create a new account if none provided
        if not self.account_id:
            account = self.sandbox_tool.create_sandbox_account()
            self.account_id = account["account_id"]
            logger.info(f"Created new sandbox account: {self.account_id}")
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get the current account summary."""
        return self.sandbox_tool._run("account_summary", account_id=self.account_id)
    
    def get_trade_history(self, limit: int = 20) -> Dict[str, Any]:
        """Get the trading history for the account."""
        return self.sandbox_tool._run("trade_history", account_id=self.account_id, limit=limit)
    
    def get_price(self, symbol: str) -> Dict[str, Any]:
        """Get the current price for a symbol."""
        return self.sandbox_tool._run("get_price", symbol=symbol)
    
    def execute_trade(self, symbol: str, action: str, amount: float, position_type: str = "long") -> Dict[str, Any]:
        """
        Execute a trade on the sandbox account.
        
        Args:
            symbol: Cryptocurrency symbol
            action: Trade action ('buy' or 'sell')
            amount: Amount in USD
            position_type: Type of position ('long' or 'short')
            
        Returns:
            Trade result
        """
        return self.sandbox_tool._run(
            "execute_trade",
            account_id=self.account_id,
            symbol=symbol,
            action=action,
            amount=amount,
            position_type=position_type
        )
    
    def run_agent_evaluation(self, target_crypto: str = "bitcoin", evaluation_days: int = 7, 
                           trade_interval: int = 1800, verbose: bool = True) -> Dict[str, Any]:
        """
        Run an evaluation of the AI trading agent using sandbox trading.
        
        Args:
            target_crypto: Target cryptocurrency to trade
            evaluation_days: Number of days to simulate
            trade_interval: Seconds between trades (simulated time)
            verbose: Whether to print detailed logs
        
        Returns:
            Evaluation results
        """
        if verbose:
            print(f"\n=== Starting AI Agent Evaluation for {target_crypto} ===")
            print(f"Simulating {evaluation_days} days of trading")
        
        # Get initial account status
        initial_summary = self.get_account_summary()
        initial_value = initial_summary["portfolio_value_usd"]
        
        if verbose:
            print(f"\nInitial Portfolio Value: ${initial_value:.2f}")
        
        # Simulate days of trading
        total_trades = 0
        successful_trades = 0
        total_intervals = (evaluation_days * 24 * 60 * 60) // trade_interval
        
        for interval in range(total_intervals):
            # Get current time (simulated)
            current_time = datetime.now()
            day_number = interval // (24 * 60 * 60 // trade_interval) + 1
            
            if verbose and interval % 5 == 0:
                print(f"\nDay {day_number}, Interval {interval + 1}/{total_intervals}")
                print(f"Simulated time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get current market data
            price_data = self.get_price(target_crypto)
            current_price = price_data["price_usd"]
            
            if verbose and interval % 5 == 0:
                print(f"{target_crypto} price: ${current_price:.2f}")
            
            # Prepare input for the AI agent
            agent_input = {
                "target_crypto": target_crypto,
                "current_price": current_price,
                "timestamp": current_time.isoformat(),
                "account_summary": self.get_account_summary()
            }
            
            # Run the AI agent to get trading decisions
            try:
                agent_result = self.crypto_crew.run()
                
                if "trading_decision" in agent_result:
                    trading_decision = agent_result["trading_decision"]
                    
                    # Extract decision details
                    action = trading_decision.get("action", "hold").lower()
                    if action != "hold":
                        position_type = trading_decision.get("position", "long")
                        
                        # Determine trade amount
                        account_summary = self.get_account_summary()
                        available_balance = account_summary["balance_usd"]
                        portfolio_value = account_summary["portfolio_value_usd"]
                        
                        # Use position size from decision or default to 10% of portfolio
                        position_size_pct = trading_decision.get("position_size_pct", 10.0)
                        amount = (portfolio_value * position_size_pct / 100.0)
                        
                        # Cap at available balance for buys
                        if action == "buy" and amount > available_balance:
                            amount = available_balance * 0.95  # Leave some buffer
                        
                        if amount > 10.0:  # Minimum trade size $10
                            if verbose:
                                print(f"\nExecuting trade: {action.upper()} {target_crypto}")
                                print(f"Amount: ${amount:.2f}, Position: {position_type}")
                            
                            # Execute the trade
                            trade_result = self.execute_trade(
                                symbol=target_crypto,
                                action=action,
                                amount=amount,
                                position_type=position_type
                            )
                            
                            total_trades += 1
                            if trade_result.get("status") == "success":
                                successful_trades += 1
                                
                            if verbose:
                                print(f"Trade result: {trade_result.get('status', 'unknown').upper()}")
                        elif verbose:
                            print(f"\nSkipping small trade (${amount:.2f} < $10 minimum)")
                    elif verbose and interval % 5 == 0:
                        print(f"AI decision: HOLD {target_crypto}")
                
            except Exception as e:
                logger.error(f"Error running AI agent: {e}")
                if verbose:
                    print(f"Error running AI agent: {e}")
            
            # Simulate time passing
            time.sleep(0.5)  # Small delay for simulation
        
        # Get final account status
        final_summary = self.get_account_summary()
        final_value = final_summary["portfolio_value_usd"]
        
        # Calculate performance metrics
        profit_loss = final_value - initial_value
        profit_loss_pct = (profit_loss / initial_value) * 100 if initial_value > 0 else 0
        trade_success_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
        
        results = {
            "target_crypto": target_crypto,
            "evaluation_days": evaluation_days,
            "initial_value": initial_value,
            "final_value": final_value,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss_pct,
            "total_trades": total_trades,
            "successful_trades": successful_trades,
            "trade_success_rate": trade_success_rate,
            "account_id": self.account_id,
            "final_positions": final_summary.get("positions", []),
            "completed_at": datetime.now().isoformat()
        }
        
        if verbose:
            print("\n=== Evaluation Results ===")
            print(f"Target Cryptocurrency: {target_crypto}")
            print(f"Evaluation Period: {evaluation_days} days")
            print(f"Initial Portfolio Value: ${initial_value:.2f}")
            print(f"Final Portfolio Value: ${final_value:.2f}")
            print(f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
            print(f"Total Trades: {total_trades}")
            print(f"Successful Trades: {successful_trades}")
            print(f"Trade Success Rate: {trade_success_rate:.2f}%")
        
        return results

    def compare_strategies(self, strategies: List[Dict[str, Any]], target_crypto: str = "bitcoin", 
                          evaluation_days: int = 5) -> Dict[str, Any]:
        """
        Compare multiple trading strategies using the same initial conditions.
        
        Args:
            strategies: List of strategy configurations to test
            target_crypto: Target cryptocurrency to trade
            evaluation_days: Number of days to simulate
            
        Returns:
            Comparison results
        """
        print(f"\n=== Comparing {len(strategies)} Trading Strategies for {target_crypto} ===")
        
        # Save original account ID
        original_account_id = self.account_id
        
        # Results for each strategy
        strategy_results = []
        
        for i, strategy in enumerate(strategies):
            strategy_name = strategy.get("name", f"Strategy {i+1}")
            print(f"\n--- Testing {strategy_name} ---")
            
            # Create a new account for this strategy test
            account = self.sandbox_tool.create_sandbox_account()
            self.account_id = account["account_id"]
            
            # Configure the crew with this strategy
            if "config_path" in strategy:
                self.crypto_crew = CryptoTradingCrew(config_path=strategy["config_path"])
            
            # Run evaluation
            results = self.run_agent_evaluation(
                target_crypto=target_crypto,
                evaluation_days=evaluation_days,
                verbose=False
            )
            
            # Add strategy info to results
            results["strategy_name"] = strategy_name
            results["strategy_config"] = strategy
            strategy_results.append(results)
            
            print(f"Strategy: {strategy_name}")
            print(f"Profit/Loss: ${results['profit_loss']:.2f} ({results['profit_loss_pct']:.2f}%)")
            print(f"Trade Success Rate: {results['trade_success_rate']:.2f}%")
        
        # Restore original account
        self.account_id = original_account_id
        
        # Rank strategies by profit/loss
        ranked_strategies = sorted(
            strategy_results, 
            key=lambda x: x["profit_loss_pct"], 
            reverse=True
        )
        
        print("\n=== Strategy Comparison Results ===")
        print("Ranked by profit percentage:")
        
        for i, result in enumerate(ranked_strategies):
            print(f"{i+1}. {result['strategy_name']}: {result['profit_loss_pct']:.2f}% " +
                  f"(${result['profit_loss']:.2f}), Success Rate: {result['trade_success_rate']:.2f}%")
        
        return {
            "target_crypto": target_crypto,
            "evaluation_days": evaluation_days,
            "strategy_results": strategy_results,
            "ranked_strategies": ranked_strategies
        }

    def backtest_strategy(self, historical_data_path: str, target_crypto: str = "bitcoin") -> Dict[str, Any]:
        """
        Backtest a trading strategy using historical price data.
        
        Args:
            historical_data_path: Path to CSV file with historical data
            target_crypto: Target cryptocurrency to trade
            
        Returns:
            Backtesting results
        """
        try:
            import pandas as pd
            
            # Load historical data
            historical_data = pd.read_csv(historical_data_path)
            print(f"\n=== Backtesting Strategy for {target_crypto} ===")
            print(f"Loaded {len(historical_data)} historical data points")
            
            # Create a new account for backtesting
            account = self.sandbox_tool.create_sandbox_account()
            self.account_id = account["account_id"]
            initial_balance = account["balance_usd"]
            
            # Track performance metrics
            total_trades = 0
            successful_trades = 0
            
            # Iterate through historical data points
            for i, row in historical_data.iterrows():
                # Extract price and timestamp from row
                timestamp = row.get("timestamp", datetime.now().isoformat())
                price = row.get("price", 0.0)
                
                if i % 100 == 0:
                    print(f"Processing data point {i+1}/{len(historical_data)}")
                
                # Prepare input for the AI agent
                agent_input = {
                    "target_crypto": target_crypto,
                    "current_price": price,
                    "timestamp": timestamp,
                    "account_summary": self.get_account_summary()
                }
                
                # Run the AI agent to get trading decisions
                try:
                    agent_result = self.crypto_crew.run()
                    
                    if "trading_decision" in agent_result:
                        trading_decision = agent_result["trading_decision"]
                        
                        # Extract decision details
                        action = trading_decision.get("action", "hold").lower()
                        if action != "hold":
                            position_type = trading_decision.get("position", "long")
                            
                            # Determine trade amount
                            account_summary = self.get_account_summary()
                            available_balance = account_summary["balance_usd"]
                            
                            # Use position size from decision or default to 10% of portfolio
                            position_size_pct = trading_decision.get("position_size_pct", 10.0)
                            portfolio_value = account_summary["portfolio_value_usd"]
                            amount = (portfolio_value * position_size_pct / 100.0)
                            
                            # Cap at available balance for buys
                            if action == "buy" and amount > available_balance:
                                amount = available_balance * 0.95  # Leave some buffer
                            
                            if amount > 10.0:  # Minimum trade size $10
                                # Override the price in the sandbox to use historical price
                                self.sandbox_tool.exchange.price_cache[target_crypto] = (price, datetime.now())
                                
                                # Execute the trade
                                trade_result = self.execute_trade(
                                    symbol=target_crypto,
                                    action=action,
                                    amount=amount,
                                    position_type=position_type
                                )
                                
                                total_trades += 1
                                if trade_result.get("status") == "success":
                                    successful_trades += 1
                
                except Exception as e:
                    logger.error(f"Error running AI agent: {e}")
            
            # Get final account status
            final_summary = self.get_account_summary()
            final_value = final_summary["portfolio_value_usd"]
            
            # Calculate performance metrics
            profit_loss = final_value - initial_balance
            profit_loss_pct = (profit_loss / initial_balance) * 100 if initial_balance > 0 else 0
            trade_success_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
            
            results = {
                "target_crypto": target_crypto,
                "data_points": len(historical_data),
                "initial_value": initial_balance,
                "final_value": final_value,
                "profit_loss": profit_loss,
                "profit_loss_pct": profit_loss_pct,
                "total_trades": total_trades,
                "successful_trades": successful_trades,
                "trade_success_rate": trade_success_rate,
                "account_id": self.account_id,
                "final_positions": final_summary.get("positions", []),
                "completed_at": datetime.now().isoformat()
            }
            
            print("\n=== Backtesting Results ===")
            print(f"Target Cryptocurrency: {target_crypto}")
            print(f"Data Points Analyzed: {len(historical_data)}")
            print(f"Initial Portfolio Value: ${initial_balance:.2f}")
            print(f"Final Portfolio Value: ${final_value:.2f}")
            print(f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
            print(f"Total Trades: {total_trades}")
            print(f"Successful Trades: {successful_trades}")
            print(f"Trade Success Rate: {trade_success_rate:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return {"error": str(e)} 