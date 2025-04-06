#!/usr/bin/env python
"""
Command-line interface for evaluating AI trading agents with sandbox accounts.
This allows testing and evaluating agents with virtual funds before using real accounts.
"""
import os
import sys
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.crew_chain.tools.agent_sandbox_trading import AgentSandboxTrading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def run_agent_evaluation(args):
    """Run an evaluation of the AI trading agent using sandbox trading."""
    # Create the agent sandbox trading integration
    sandbox_trading = AgentSandboxTrading(
        config_path=args.config,
        account_id=args.account_id
    )
    
    # Run the evaluation
    results = sandbox_trading.run_agent_evaluation(
        target_crypto=args.symbol,
        evaluation_days=args.days,
        trade_interval=args.interval,
        verbose=not args.quiet
    )
    
    # Save results if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
    
    return 0

def compare_strategies(args):
    """Compare multiple trading strategies using the same initial conditions."""
    # Load strategy configurations
    strategies = []
    try:
        with open(args.strategies_file, 'r') as f:
            strategies = json.load(f)
    except Exception as e:
        logger.error(f"Error loading strategies file: {e}")
        return 1
    
    # Create the agent sandbox trading integration
    sandbox_trading = AgentSandboxTrading(
        config_path=args.config,
        account_id=args.account_id
    )
    
    # Run the comparison
    results = sandbox_trading.compare_strategies(
        strategies=strategies,
        target_crypto=args.symbol,
        evaluation_days=args.days
    )
    
    # Save results if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            print(f"Comparison results saved to {args.output}")
    
    return 0

def backtest_strategy(args):
    """Backtest a trading strategy using historical price data."""
    # Create the agent sandbox trading integration
    sandbox_trading = AgentSandboxTrading(
        config_path=args.config,
        account_id=args.account_id
    )
    
    # Run the backtest
    results = sandbox_trading.backtest_strategy(
        historical_data_path=args.data_file,
        target_crypto=args.symbol
    )
    
    # Save results if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            print(f"Backtest results saved to {args.output}")
    
    return 0

def create_sandbox_account(args):
    """Create a new sandbox trading account."""
    from src.crew_chain.tools.sandbox_trading import SandboxTradingTool
    
    # Create the sandbox trading tool
    sandbox_tool = SandboxTradingTool()
    
    # Create a new account
    result = sandbox_tool._run("create_account", initial_balance_usd=args.balance)
    
    print(f"\n=== New Sandbox Account Created ===")
    print(f"Account ID: {result['account_id']}")
    print(f"Balance: ${result['balance_usd']:.2f}")
    print(f"Created at: {result['created_at']}")
    print(f"\nTo use this account in future commands, use:")
    print(f"  --account-id {result['account_id']}")
    
    # Save account ID to file if requested
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, 'w') as f:
            f.write(result['account_id'])
            print(f"Account ID saved to {args.save}")
    
    return 0

def get_account_summary(args):
    """Get a summary of account status including balance and positions."""
    from src.crew_chain.tools.sandbox_trading import SandboxTradingTool
    
    # Create the sandbox trading tool
    sandbox_tool = SandboxTradingTool()
    
    # Get account summary
    summary = sandbox_tool._run("account_summary", account_id=args.account_id)
    
    if "error" in summary:
        print(f"Error: {summary['error']}")
        return 1
    
    print("\n=== Account Summary ===")
    print(f"Account: {summary['account_id']}")
    print(f"Cash Balance: ${summary['balance_usd']:.2f}")
    print(f"Portfolio Value: ${summary['portfolio_value_usd']:.2f}")
    print("\nPositions:")
    
    if 'positions' in summary and summary['positions']:
        for position in summary['positions']:
            print(f"  {position['symbol']} ({position['position_type']}):")
            print(f"    Amount: {position['crypto_amount']:.6f}")
            print(f"    Current Price: ${position['current_price']:.2f}")
            print(f"    Value: ${position['current_value_usd']:.2f}")
            print(f"    P/L: ${position['pnl_usd']:.2f} ({position['pnl_percentage']:.2f}%)")
    else:
        print("  No open positions")
    
    # Save results if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
            print(f"Account summary saved to {args.output}")
    
    return 0

def execute_trade(args):
    """Execute a trade on the sandbox account."""
    from src.crew_chain.tools.sandbox_trading import SandboxTradingTool
    
    # Create the sandbox trading tool
    sandbox_tool = SandboxTradingTool()
    
    # Execute the trade
    result = sandbox_tool._run(
        "execute_trade",
        account_id=args.account_id,
        symbol=args.symbol,
        action=args.action,
        amount=args.amount,
        position_type=args.position_type
    )
    
    print("\n=== Trade Execution Result ===")
    if result.get("status") == "success":
        print(f"Success! {args.action.upper()} {result['symbol']}")
        print(f"Amount: ${result['amount_usd']:.2f} ({result['crypto_amount']:.6f} {result['symbol']})")
        print(f"Price: ${result['price']:.2f}")
        print(f"Position Type: {result['position_type']}")
        print(f"Timestamp: {result['timestamp']}")
    else:
        print(f"Trade failed: {result.get('error', 'Unknown error')}")
    
    # Save results if requested
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
            print(f"Trade result saved to {args.output}")
    
    return 0 if result.get("status") == "success" else 1

def main(args=None):
    """Main entry point for the evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate AI trading agents with sandbox accounts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Evaluation command")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate", 
        help="Run an evaluation of the AI trading agent"
    )
    evaluate_parser.add_argument(
        "--config", 
        type=str, 
        default="config/crypto_agents.yaml",
        help="Path to agent configuration file"
    )
    evaluate_parser.add_argument(
        "--account-id", 
        type=str, 
        help="Sandbox account ID to use"
    )
    evaluate_parser.add_argument(
        "--symbol", 
        type=str, 
        default="bitcoin",
        help="Cryptocurrency symbol to trade"
    )
    evaluate_parser.add_argument(
        "--days", 
        type=int, 
        default=7,
        help="Number of days to simulate"
    )
    evaluate_parser.add_argument(
        "--interval", 
        type=int, 
        default=1800,
        help="Seconds between trades (simulated time)"
    )
    evaluate_parser.add_argument(
        "--output", 
        type=str, 
        help="Path to save results to"
    )
    evaluate_parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress detailed output"
    )
    evaluate_parser.set_defaults(func=run_agent_evaluation)
    
    # Compare command
    compare_parser = subparsers.add_parser(
        "compare", 
        help="Compare multiple trading strategies"
    )
    compare_parser.add_argument(
        "--strategies-file", 
        type=str, 
        required=True,
        help="Path to JSON file with strategy configurations"
    )
    compare_parser.add_argument(
        "--config", 
        type=str, 
        default="config/crypto_agents.yaml",
        help="Path to default agent configuration file"
    )
    compare_parser.add_argument(
        "--account-id", 
        type=str, 
        help="Sandbox account ID to use"
    )
    compare_parser.add_argument(
        "--symbol", 
        type=str, 
        default="bitcoin",
        help="Cryptocurrency symbol to trade"
    )
    compare_parser.add_argument(
        "--days", 
        type=int, 
        default=5,
        help="Number of days to simulate for each strategy"
    )
    compare_parser.add_argument(
        "--output", 
        type=str, 
        help="Path to save results to"
    )
    compare_parser.set_defaults(func=compare_strategies)
    
    # Backtest command
    backtest_parser = subparsers.add_parser(
        "backtest", 
        help="Backtest a trading strategy using historical data"
    )
    backtest_parser.add_argument(
        "--data-file", 
        type=str, 
        required=True,
        help="Path to CSV file with historical price data"
    )
    backtest_parser.add_argument(
        "--config", 
        type=str, 
        default="config/crypto_agents.yaml",
        help="Path to agent configuration file"
    )
    backtest_parser.add_argument(
        "--account-id", 
        type=str, 
        help="Sandbox account ID to use"
    )
    backtest_parser.add_argument(
        "--symbol", 
        type=str, 
        default="bitcoin",
        help="Cryptocurrency symbol to trade"
    )
    backtest_parser.add_argument(
        "--output", 
        type=str, 
        help="Path to save results to"
    )
    backtest_parser.set_defaults(func=backtest_strategy)
    
    # Account commands
    account_parser = subparsers.add_parser(
        "account", 
        help="Manage sandbox trading accounts"
    )
    account_subparsers = account_parser.add_subparsers(dest="account_command", help="Account management command")
    
    # Create account command
    create_account_parser = account_subparsers.add_parser(
        "create", 
        help="Create a new sandbox trading account"
    )
    create_account_parser.add_argument(
        "--balance", 
        type=float, 
        default=10000.0,
        help="Initial balance in USD"
    )
    create_account_parser.add_argument(
        "--save", 
        type=str, 
        help="Path to save account ID to"
    )
    create_account_parser.set_defaults(func=create_sandbox_account)
    
    # Summary command
    summary_parser = account_subparsers.add_parser(
        "summary", 
        help="Get a summary of account status"
    )
    summary_parser.add_argument(
        "--account-id", 
        type=str, 
        required=True,
        help="Sandbox account ID to use"
    )
    summary_parser.add_argument(
        "--output", 
        type=str, 
        help="Path to save results to"
    )
    summary_parser.set_defaults(func=get_account_summary)
    
    # Trade command
    trade_parser = account_subparsers.add_parser(
        "trade", 
        help="Execute a trade on the sandbox account"
    )
    trade_parser.add_argument(
        "--account-id", 
        type=str, 
        required=True,
        help="Sandbox account ID to use"
    )
    trade_parser.add_argument(
        "--symbol", 
        type=str, 
        required=True,
        help="Cryptocurrency symbol to trade"
    )
    trade_parser.add_argument(
        "--action", 
        type=str, 
        required=True,
        choices=["buy", "sell"],
        help="Trade action"
    )
    trade_parser.add_argument(
        "--amount", 
        type=float, 
        required=True,
        help="Amount in USD"
    )
    trade_parser.add_argument(
        "--position-type", 
        type=str, 
        default="long",
        choices=["long", "short"],
        help="Type of position"
    )
    trade_parser.add_argument(
        "--output", 
        type=str, 
        help="Path to save results to"
    )
    trade_parser.set_defaults(func=execute_trade)
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Execute the appropriate function
    if hasattr(parsed_args, "func"):
        return parsed_args.func(parsed_args)
    else:
        if parsed_args.command == "account" and not parsed_args.account_command:
            account_parser.print_help()
        else:
            parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 