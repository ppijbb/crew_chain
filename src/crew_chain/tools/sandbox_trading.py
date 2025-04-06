"""
Sandbox Trading Environment for Crew Chain

This module provides a simulated trading environment with paper trading
capabilities for testing the trading strategies before using real funds.
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import uuid

import pandas as pd
import numpy as np
import requests
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SandboxAccount(BaseModel):
    """Represents a virtual trading account for sandbox testing."""
    account_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    balance_usd: float = 10000.0  # Default starting balance in USD
    created_at: datetime = Field(default_factory=datetime.now)
    positions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    trade_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "account_id": self.account_id,
            "balance_usd": self.balance_usd,
            "created_at": self.created_at.isoformat(),
            "positions": self.positions,
            "trade_history": self.trade_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SandboxAccount':
        """Create from dictionary after deserialization."""
        if isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)
    
    def add_trade(self, trade_data: Dict[str, Any]) -> None:
        """Add a trade to the history."""
        trade_data["timestamp"] = datetime.now().isoformat()
        trade_data["trade_id"] = str(uuid.uuid4())
        self.trade_history.append(trade_data)

class SandboxExchange:
    """Simulated cryptocurrency exchange for sandbox trading."""
    
    def __init__(self, data_source: str = "binance"):
        """
        Initialize the sandbox exchange.
        
        Args:
            data_source: Source for price data ('binance', 'coinbase', etc.)
        """
        self.data_source = data_source
        self.accounts: Dict[str, SandboxAccount] = {}
        self.price_cache: Dict[str, Tuple[float, datetime]] = {}
        self.price_cache_timeout = 60  # seconds
        self._load_accounts()
    
    def _load_accounts(self) -> None:
        """Load existing sandbox accounts from storage."""
        os.makedirs("data/sandbox", exist_ok=True)
        try:
            if os.path.exists("data/sandbox/accounts.json"):
                with open("data/sandbox/accounts.json", "r") as f:
                    accounts_data = json.load(f)
                    for account_id, account_data in accounts_data.items():
                        self.accounts[account_id] = SandboxAccount.from_dict(account_data)
                logger.info(f"Loaded {len(self.accounts)} sandbox accounts")
        except Exception as e:
            logger.error(f"Error loading sandbox accounts: {e}")
    
    def _save_accounts(self) -> None:
        """Save sandbox accounts to storage."""
        try:
            accounts_data = {
                account_id: account.to_dict() 
                for account_id, account in self.accounts.items()
            }
            with open("data/sandbox/accounts.json", "w") as f:
                json.dump(accounts_data, f, indent=2)
            logger.info(f"Saved {len(self.accounts)} sandbox accounts")
        except Exception as e:
            logger.error(f"Error saving sandbox accounts: {e}")
    
    def create_account(self, initial_balance_usd: float = 10000.0) -> SandboxAccount:
        """
        Create a new sandbox trading account.
        
        Args:
            initial_balance_usd: Initial USD balance for the account
            
        Returns:
            The newly created account
        """
        account = SandboxAccount(balance_usd=initial_balance_usd)
        self.accounts[account.account_id] = account
        self._save_accounts()
        return account
    
    def get_account(self, account_id: str) -> Optional[SandboxAccount]:
        """
        Get a sandbox account by ID.
        
        Args:
            account_id: ID of the account to retrieve
            
        Returns:
            The account if found, None otherwise
        """
        return self.accounts.get(account_id)
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get the current price for a cryptocurrency.
        
        Args:
            symbol: Symbol of the cryptocurrency (e.g., 'BTC', 'ETH')
            
        Returns:
            Current price in USD
        """
        # Check cache first
        now = datetime.now()
        if symbol in self.price_cache:
            price, timestamp = self.price_cache[symbol]
            if (now - timestamp).total_seconds() < self.price_cache_timeout:
                return price
        
        # Get fresh price
        try:
            # Fetch from CoinGecko
            symbol_lower = symbol.lower()
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol_lower}&vs_currencies=usd"
            response = requests.get(url)
            data = response.json()
            
            if symbol_lower in data and 'usd' in data[symbol_lower]:
                price = float(data[symbol_lower]['usd'])
                self.price_cache[symbol] = (price, now)
                return price
            
            # Fallback to simulated price if API fails
            logger.warning(f"Could not get price for {symbol} from API, using simulated price")
            if symbol in self.price_cache:
                last_price, _ = self.price_cache[symbol]
                # Add some random fluctuation
                price = last_price * (1 + (np.random.random() - 0.5) * 0.01)
            else:
                # Default prices if no data available
                default_prices = {"BTC": 50000.0, "ETH": 3000.0, "USDT": 1.0}
                price = default_prices.get(symbol.upper(), 100.0)
            
            self.price_cache[symbol] = (price, now)
            return price
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            # Fallback to default price
            default_prices = {"BTC": 50000.0, "ETH": 3000.0, "USDT": 1.0}
            return default_prices.get(symbol.upper(), 100.0)
    
    def execute_trade(self, account_id: str, symbol: str, action: str, 
                      amount: float, position_type: str = "long") -> Dict[str, Any]:
        """
        Execute a trade on the sandbox exchange.
        
        Args:
            account_id: ID of the account to execute the trade for
            symbol: Symbol of the cryptocurrency (e.g., 'BTC', 'ETH')
            action: Trade action ('buy' or 'sell')
            amount: Amount to buy or sell in USD
            position_type: Type of position ('long' or 'short')
            
        Returns:
            Trade result with details
        """
        account = self.get_account(account_id)
        if not account:
            return {"error": f"Account {account_id} not found"}
        
        symbol = symbol.upper()
        action = action.lower()
        position_type = position_type.lower()
        
        # Get current price
        current_price = self.get_current_price(symbol)
        
        # Calculate cryptocurrency amount
        crypto_amount = amount / current_price
        
        result = {
            "account_id": account_id,
            "symbol": symbol,
            "action": action,
            "amount_usd": amount,
            "price": current_price,
            "crypto_amount": crypto_amount,
            "position_type": position_type,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        try:
            if action == "buy":
                # Check if enough balance
                if account.balance_usd < amount:
                    result["status"] = "failed"
                    result["error"] = "Insufficient funds"
                    return result
                
                # Update account
                account.balance_usd -= amount
                
                # Update position
                position_key = f"{symbol}_{position_type}"
                if position_key not in account.positions:
                    account.positions[position_key] = {
                        "symbol": symbol,
                        "position_type": position_type,
                        "total_amount_usd": 0,
                        "crypto_amount": 0,
                        "average_price": 0
                    }
                
                position = account.positions[position_key]
                
                # Calculate new average price
                total_crypto = position["crypto_amount"] + crypto_amount
                position["average_price"] = ((position["average_price"] * position["crypto_amount"]) + 
                                             (current_price * crypto_amount)) / total_crypto if total_crypto > 0 else current_price
                
                position["crypto_amount"] = total_crypto
                position["total_amount_usd"] = total_crypto * current_price
                
            elif action == "sell":
                # Check if position exists
                position_key = f"{symbol}_{position_type}"
                if position_key not in account.positions or account.positions[position_key]["crypto_amount"] < crypto_amount:
                    result["status"] = "failed"
                    result["error"] = "Insufficient cryptocurrency amount"
                    return result
                
                # Update account
                account.balance_usd += amount
                
                # Update position
                position = account.positions[position_key]
                position["crypto_amount"] -= crypto_amount
                position["total_amount_usd"] = position["crypto_amount"] * current_price
                
                # Remove position if fully sold
                if position["crypto_amount"] <= 0:
                    del account.positions[position_key]
            
            # Add trade to history
            account.add_trade(result)
            
            # Save accounts
            self._save_accounts()
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result
    
    def get_account_summary(self, account_id: str) -> Dict[str, Any]:
        """
        Get a summary of account status including balance and positions.
        
        Args:
            account_id: ID of the account to get summary for
            
        Returns:
            Summary with balance, positions and portfolio value
        """
        account = self.get_account(account_id)
        if not account:
            return {"error": f"Account {account_id} not found"}
        
        # Update position values with current prices
        portfolio_value = account.balance_usd
        positions = []
        
        for position_key, position in account.positions.items():
            symbol = position["symbol"]
            current_price = self.get_current_price(symbol)
            current_value = position["crypto_amount"] * current_price
            
            # Calculate profit/loss
            investment = position["crypto_amount"] * position["average_price"]
            pnl = current_value - investment
            pnl_percentage = (pnl / investment) * 100 if investment > 0 else 0
            
            position_summary = {
                "symbol": symbol,
                "position_type": position["position_type"],
                "crypto_amount": position["crypto_amount"],
                "average_price": position["average_price"],
                "current_price": current_price,
                "current_value_usd": current_value,
                "pnl_usd": pnl,
                "pnl_percentage": pnl_percentage
            }
            
            positions.append(position_summary)
            portfolio_value += current_value
        
        return {
            "account_id": account_id,
            "balance_usd": account.balance_usd,
            "positions": positions,
            "portfolio_value_usd": portfolio_value,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_trade_history(self, account_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get trade history for an account.
        
        Args:
            account_id: ID of the account to get history for
            limit: Maximum number of trades to return
            
        Returns:
            List of trades ordered by most recent first
        """
        account = self.get_account(account_id)
        if not account:
            return {"error": f"Account {account_id} not found"}
        
        # Return most recent trades first
        return sorted(
            account.trade_history, 
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )[:limit]


class SandboxTradingTool:
    """Tool for interacting with the sandbox trading environment."""
    
    def __init__(self):
        """Initialize the sandbox trading tool."""
        self.exchange = SandboxExchange()
    
    def create_sandbox_account(self, initial_balance_usd: float = 10000.0) -> Dict[str, Any]:
        """
        Create a new sandbox trading account.
        
        Args:
            initial_balance_usd: Initial USD balance for the account
            
        Returns:
            Details of the created account
        """
        account = self.exchange.create_account(initial_balance_usd)
        return {
            "account_id": account.account_id,
            "balance_usd": account.balance_usd,
            "created_at": account.created_at.isoformat(),
            "message": f"Sandbox account created with ${initial_balance_usd} USD balance"
        }
    
    def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get the current price for a cryptocurrency.
        
        Args:
            symbol: Symbol of the cryptocurrency (e.g., 'BTC', 'ETH')
            
        Returns:
            Current price information
        """
        price = self.exchange.get_current_price(symbol)
        return {
            "symbol": symbol,
            "price_usd": price,
            "timestamp": datetime.now().isoformat()
        }
    
    def execute_trade(self, account_id: str, symbol: str, action: str, 
                     amount: float, position_type: str = "long") -> Dict[str, Any]:
        """
        Execute a trade on the sandbox exchange.
        
        Args:
            account_id: ID of the account to execute the trade for
            symbol: Symbol of the cryptocurrency (e.g., 'BTC', 'ETH')
            action: Trade action ('buy' or 'sell')
            amount: Amount to buy or sell in USD
            position_type: Type of position ('long' or 'short')
            
        Returns:
            Trade result with details
        """
        return self.exchange.execute_trade(
            account_id, symbol, action, amount, position_type
        )
    
    def get_account_summary(self, account_id: str) -> Dict[str, Any]:
        """
        Get a summary of account status including balance and positions.
        
        Args:
            account_id: ID of the account to get summary for
            
        Returns:
            Summary with balance, positions and portfolio value
        """
        return self.exchange.get_account_summary(account_id)
    
    def get_trade_history(self, account_id: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get trade history for an account.
        
        Args:
            account_id: ID of the account to get history for
            limit: Maximum number of trades to return
            
        Returns:
            List of trades ordered by most recent first
        """
        trades = self.exchange.get_trade_history(account_id, limit)
        return {
            "account_id": account_id,
            "trades": trades,
            "count": len(trades)
        }
    
    def _run(self, action: str, **kwargs) -> Dict[str, Any]:
        """
        Run a sandbox trading action.
        
        Args:
            action: The action to perform (create_account, execute_trade, etc.)
            **kwargs: Arguments for the specific action
            
        Returns:
            Result of the action
        """
        # Map action to method
        action_mapping = {
            "create_account": self.create_sandbox_account,
            "get_price": self.get_current_price,
            "execute_trade": self.execute_trade,
            "account_summary": self.get_account_summary,
            "trade_history": self.get_trade_history
        }
        
        if action not in action_mapping:
            return {
                "error": f"Unknown action: {action}",
                "available_actions": list(action_mapping.keys())
            }
        
        # Call the appropriate method
        return action_mapping[action](**kwargs) 