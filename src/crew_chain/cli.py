#!/usr/bin/env python
"""
CLI interface for the Crew Chain cryptocurrency trading system.
"""
import argparse
import logging
from typing import List, Optional

from crew_chain.crypto_trading_crew import CryptoTradingCrew

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the crew-chain CLI."""
    parser = argparse.ArgumentParser(description="Crew Chain cryptocurrency trading system")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the trading system")
    start_parser.add_argument(
        "--config", 
        default="config/crypto_agents.yaml", 
        help="Path to the agent configuration file"
    )
    start_parser.add_argument(
        "--mode",
        choices=["analyze", "backtest", "trade", "simulate"],
        default="analyze",
        help="Trading mode to run in"
    )
    
    # Status command
    subparsers.add_parser("status", help="Check the status of the trading system")
    
    # Run backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest on historical data")
    backtest_parser.add_argument(
        "--period", 
        default="1m", 
        help="Period to backtest (e.g. 1d, 1w, 1m, 1y)"
    )
    backtest_parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC", "ETH"],
        help="Symbols to include in backtest"
    )
    
    parsed_args = parser.parse_args(args)
    
    if not parsed_args.command:
        parser.print_help()
        return 1
    
    try:
        if parsed_args.command == "start":
            logger.info(f"Starting trading system in {parsed_args.mode} mode")
            crew = CryptoTradingCrew(config_path=parsed_args.config)
            crew.run()
            return 0
        
        elif parsed_args.command == "status":
            logger.info("Checking trading system status")
            # TODO: Implement status check
            print("Trading system is currently not running")
            return 0
        
        elif parsed_args.command == "backtest":
            logger.info(f"Running backtest for {parsed_args.symbols} over {parsed_args.period}")
            crew = CryptoTradingCrew(config_path="config/crypto_agents.yaml")
            # TODO: Implement backtest mode
            print(f"Backtest complete for {parsed_args.symbols} over {parsed_args.period}")
            return 0
    
    except Exception as e:
        logger.error(f"Error running command {parsed_args.command}: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 