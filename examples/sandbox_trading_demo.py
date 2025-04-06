#!/usr/bin/env python
"""
Sandbox Trading Demo for testing cryptocurrency trading strategies
with virtual funds.
"""
import os
import sys
import time
import json
from datetime import datetime
import logging
from pprint import pprint

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.crew_chain.tools.sandbox_trading import SandboxTradingTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def create_sandbox_account(sandbox_tool, initial_balance=10000.0):
    """Create a new sandbox trading account."""
    result = sandbox_tool._run("create_account", initial_balance_usd=initial_balance)
    print("\n=== New Sandbox Account Created ===")
    pprint(result)
    return result["account_id"]

def get_price_data(sandbox_tool, symbols=["bitcoin", "ethereum"]):
    """Get price data for cryptocurrencies."""
    print("\n=== Current Cryptocurrency Prices ===")
    for symbol in symbols:
        price_data = sandbox_tool._run("get_price", symbol=symbol)
        pprint(price_data)
        time.sleep(1)  # Small delay to avoid rate limits

def execute_demo_trades(sandbox_tool, account_id):
    """Execute some demo trades for testing."""
    print("\n=== Executing Demo Trades ===")
    
    # Execute a BTC buy trade
    btc_buy = sandbox_tool._run(
        "execute_trade",
        account_id=account_id,
        symbol="bitcoin",
        action="buy",
        amount=2000.0,
        position_type="long"
    )
    print("\n--- BTC Buy Trade ---")
    pprint(btc_buy)
    
    # Execute an ETH buy trade
    eth_buy = sandbox_tool._run(
        "execute_trade",
        account_id=account_id,
        symbol="ethereum",
        action="buy",
        amount=3000.0,
        position_type="long"
    )
    print("\n--- ETH Buy Trade ---")
    pprint(eth_buy)
    
    # Let's wait a bit to simulate time passing and prices changing
    print("\nWaiting 5 seconds to simulate price changes...")
    time.sleep(5)
    
    # Execute a partial ETH sell trade
    eth_sell = sandbox_tool._run(
        "execute_trade",
        account_id=account_id,
        symbol="ethereum",
        action="sell",
        amount=1500.0,
        position_type="long"
    )
    print("\n--- ETH Partial Sell Trade ---")
    pprint(eth_sell)

def get_account_summary(sandbox_tool, account_id):
    """Get the account summary with current positions and values."""
    summary = sandbox_tool._run("account_summary", account_id=account_id)
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
    
    return summary

def get_trade_history(sandbox_tool, account_id):
    """Get the trading history for the account."""
    history = sandbox_tool._run("trade_history", account_id=account_id, limit=10)
    print("\n=== Trade History ===")
    if 'trades' in history and history['trades']:
        for i, trade in enumerate(history['trades']):
            print(f"\nTrade #{i+1}:")
            print(f"  Symbol: {trade['symbol']}")
            print(f"  Action: {trade['action'].upper()}")
            print(f"  Amount: ${trade['amount_usd']:.2f} ({trade['crypto_amount']:.6f} {trade['symbol']})")
            print(f"  Price: ${trade['price']:.2f}")
            print(f"  Time: {trade['timestamp']}")
            print(f"  Status: {trade['status'].upper()}")
    else:
        print("No trade history found")

def run_evaluation_test(sandbox_tool, account_id, symbols=["bitcoin", "ethereum"]):
    """Run a simple evaluation test with multiple trades."""
    print("\n=== Running Evaluation Test ===")
    print("This test will simulate multiple trades over time to evaluate performance.")
    
    # Initial account status
    print("\nInitial account status:")
    initial_summary = get_account_summary(sandbox_tool, account_id)
    initial_value = initial_summary['balance_usd']
    
    # Simulate trades over time
    for i in range(5):
        print(f"\n--- Day {i+1} Trades ---")
        
        # Simulated trading strategy (this would be replaced by AI strategy)
        if i % 2 == 0:  # Even days - buy
            for symbol in symbols:
                amount = 500.0  # Fixed amount per trade
                trade = sandbox_tool._run(
                    "execute_trade",
                    account_id=account_id,
                    symbol=symbol,
                    action="buy",
                    amount=amount,
                    position_type="long"
                )
                print(f"Bought {symbol}: ${amount:.2f}")
        else:  # Odd days - sell partial positions
            summary = sandbox_tool._run("account_summary", account_id=account_id)
            if 'positions' in summary and summary['positions']:
                for position in summary['positions']:
                    symbol = position['symbol']
                    sell_amount = position['current_value_usd'] * 0.2  # Sell 20% of position
                    trade = sandbox_tool._run(
                        "execute_trade",
                        account_id=account_id,
                        symbol=symbol,
                        action="sell",
                        amount=sell_amount,
                        position_type="long"
                    )
                    print(f"Sold {symbol}: ${sell_amount:.2f}")
        
        # Simulate time passing and price changes
        print(f"Simulating market movement for day {i+1}...")
        time.sleep(2)
    
    # Final account status
    print("\nFinal account status after evaluation test:")
    final_summary = get_account_summary(sandbox_tool, account_id)
    final_value = final_summary['portfolio_value_usd']
    
    # Calculate overall performance
    profit_loss = final_value - initial_value
    profit_loss_percent = (profit_loss / initial_value) * 100
    
    print("\n=== Performance Evaluation ===")
    print(f"Initial Portfolio Value: ${initial_value:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_percent:.2f}%)")
    
    return {
        "initial_value": initial_value,
        "final_value": final_value,
        "profit_loss": profit_loss,
        "profit_loss_percent": profit_loss_percent
    }

def main():
    """Run the sandbox trading demo."""
    print("=== Crew Chain Sandbox Trading Demo ===")
    print("This demo will create a virtual trading account and execute test trades.")
    
    # Initialize the sandbox trading tool
    sandbox_tool = SandboxTradingTool()
    
    # Create a new sandbox account or use existing one
    account_id = None
    if os.path.exists("data/sandbox/accounts.json"):
        try:
            with open("data/sandbox/accounts.json", "r") as f:
                accounts_data = json.load(f)
                if accounts_data:
                    first_account_id = next(iter(accounts_data.keys()))
                    print(f"\nFound existing account: {first_account_id}")
                    use_existing = input("Use this account? (y/n): ").lower() == 'y'
                    if use_existing:
                        account_id = first_account_id
        except Exception as e:
            logger.error(f"Error loading existing accounts: {e}")
    
    if not account_id:
        initial_balance = float(input("\nEnter initial balance in USD (default: 10000): ") or 10000)
        account_id = create_sandbox_account(sandbox_tool, initial_balance)
    
    # Get current cryptocurrency prices
    get_price_data(sandbox_tool)
    
    # Execute demo trades
    execute_demo_trades(sandbox_tool, account_id)
    
    # Get account summary
    get_account_summary(sandbox_tool, account_id)
    
    # Get trade history
    get_trade_history(sandbox_tool, account_id)
    
    # Run evaluation test
    run_test = input("\nRun evaluation test with multiple trades? (y/n): ").lower() == 'y'
    if run_test:
        run_evaluation_test(sandbox_tool, account_id)
    
    print("\n=== Demo Completed ===")

if __name__ == "__main__":
    main() 