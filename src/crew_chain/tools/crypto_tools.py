import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type

class CryptoPriceCheckInput(BaseModel):
    """Input for the CryptoPriceCheck tool."""
    crypto_symbol: str = Field(..., description="The symbol of the cryptocurrency to check, e.g., BTC, ETH, SOL")

class CryptoPriceCheckTool(BaseTool):
    """Tool for checking cryptocurrency prices and basic market data."""
    name = "cryptocurrency_price_check"
    description = "Useful for getting the current price and basic market data of a cryptocurrency."
    args_schema: Type[BaseModel] = CryptoPriceCheckInput

    def _run(self, crypto_symbol: str) -> str:
        """Check cryptocurrency price and market data using CoinGecko API."""
        try:
            # CoinGecko free API endpoint for crypto data
            url = f"https://api.coingecko.com/api/v3/coins/{self._get_coin_id(crypto_symbol)}"
            
            headers = {
                "accept": "application/json"
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            current_price = data['market_data']['current_price']['usd']
            market_cap = data['market_data']['market_cap']['usd']
            volume = data['market_data']['total_volume']['usd']
            price_change_24h = data['market_data']['price_change_percentage_24h']
            price_change_7d = data['market_data']['price_change_percentage_7d']
            ath = data['market_data']['ath']['usd']
            ath_change_percentage = data['market_data']['ath_change_percentage']['usd']
            
            result = f"""
Price data for {data['name']} ({data['symbol'].upper()}):
Current Price: ${current_price:,.2f}
24h Change: {price_change_24h:.2f}%
7d Change: {price_change_7d:.2f}%
Market Cap: ${market_cap:,.0f}
24h Trading Volume: ${volume:,.0f}
All Time High: ${ath:,.2f} ({ath_change_percentage:.2f}% from ATH)
"""
            return result
        
        except requests.exceptions.RequestException as e:
            return f"Error fetching cryptocurrency data: {str(e)}"
        except (KeyError, ValueError) as e:
            return f"Error processing cryptocurrency data: {str(e)}"
    
    def _get_coin_id(self, symbol: str) -> str:
        """Convert common cryptocurrency symbols to CoinGecko IDs."""
        symbol = symbol.lower()
        # Common mappings (expand as needed)
        mappings = {
            "btc": "bitcoin",
            "eth": "ethereum",
            "sol": "solana",
            "xrp": "ripple",
            "doge": "dogecoin",
            "ada": "cardano",
            "dot": "polkadot",
            "bnb": "binancecoin",
            "usdt": "tether",
            "usdc": "usd-coin",
            "link": "chainlink",
            "avax": "avalanche-2"
        }
        return mappings.get(symbol, symbol)

class CryptoHistoricalDataInput(BaseModel):
    """Input for the CryptoHistoricalData tool."""
    crypto_symbol: str = Field(..., description="The symbol of the cryptocurrency to check, e.g., BTC, ETH, SOL")
    days: int = Field(default=30, description="Number of days of historical data to retrieve (max 90)")

class CryptoHistoricalDataTool(BaseTool):
    """Tool for retrieving historical price data for cryptocurrencies."""
    name = "cryptocurrency_historical_data"
    description = "Useful for getting historical price data of a cryptocurrency for technical analysis."
    args_schema: Type[BaseModel] = CryptoHistoricalDataInput

    def _run(self, crypto_symbol: str, days: int = 30) -> str:
        """Get historical price data for a cryptocurrency."""
        try:
            # Limit days to 90 to prevent abuse of free API
            days = min(days, 90)
            
            # CoinGecko free API endpoint for historical data
            coin_id = self._get_coin_id(crypto_symbol)
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "daily"
            }
            
            headers = {
                "accept": "application/json"
            }
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Process price data
            prices = data.get('prices', [])
            
            if not prices:
                return f"No historical price data available for {crypto_symbol.upper()}"
            
            # Calculate some basic statistics
            latest_price = prices[-1][1]
            oldest_price = prices[0][1]
            highest_price = max(price[1] for price in prices)
            lowest_price = min(price[1] for price in prices)
            price_change = ((latest_price - oldest_price) / oldest_price) * 100
            
            # Find support and resistance levels (simplified approach)
            # Using the lowest and highest prices in different segments of the data
            segment_size = len(prices) // 3
            support_level = min(price[1] for price in prices[-segment_size:])
            resistance_level = max(price[1] for price in prices[-segment_size:])
            
            result = f"""
Historical price analysis for {crypto_symbol.upper()} over the past {days} days:

Current Price: ${latest_price:.2f}
Price {days} days ago: ${oldest_price:.2f}
Price Change: {price_change:.2f}%
Highest Price: ${highest_price:.2f}
Lowest Price: ${lowest_price:.2f}

Recent Support Level: ${support_level:.2f}
Recent Resistance Level: ${resistance_level:.2f}

Trading Recommendation:
- If current price is near ${support_level:.2f}: Consider phased buying
- If current price is near ${resistance_level:.2f}: Consider phased selling
- Current Price Location: {self._get_price_location(latest_price, support_level, resistance_level)}
"""
            return result
        
        except requests.exceptions.RequestException as e:
            return f"Error fetching historical data: {str(e)}"
        except (KeyError, ValueError, IndexError) as e:
            return f"Error processing historical data: {str(e)}"
    
    def _get_coin_id(self, symbol: str) -> str:
        """Convert common cryptocurrency symbols to CoinGecko IDs."""
        symbol = symbol.lower()
        # Common mappings (expand as needed)
        mappings = {
            "btc": "bitcoin",
            "eth": "ethereum",
            "sol": "solana",
            "xrp": "ripple",
            "doge": "dogecoin",
            "ada": "cardano",
            "dot": "polkadot",
            "bnb": "binancecoin",
            "usdt": "tether",
            "usdc": "usd-coin",
            "link": "chainlink",
            "avax": "avalanche-2"
        }
        return mappings.get(symbol, symbol)
    
    def _get_price_location(self, current_price, support_level, resistance_level):
        """Determine where the current price is located relative to support and resistance."""
        range_size = resistance_level - support_level
        if range_size <= 0:
            return "Uncertain (support and resistance levels are too close)"
        
        position = (current_price - support_level) / range_size
        
        if position <= 0.25:
            return "Near support level (potential buying zone)"
        elif position >= 0.75:
            return "Near resistance level (potential selling zone)"
        else:
            return "In the middle of the range (neutral zone)" 