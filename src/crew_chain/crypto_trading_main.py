#!/usr/bin/env python
import sys
import os
import warnings
import argparse
from datetime import datetime

from src.crew_chain.crypto_trading_crew import CryptoTradingCrew

# Suppress warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run(target_crypto="BTC"):
    """
    Run the crypto trading crew with the specified cryptocurrency.
    
    Args:
        target_crypto (str): The cryptocurrency symbol to analyze and trade (default: "BTC")
    """
    print(f"Starting automated cryptocurrency trading analysis for {target_crypto}...")
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"trading_report_{target_crypto}_{timestamp}.md"
    
    inputs = {
        'target_crypto': target_crypto
    }
    
    # Initialize and run the crew
    crew = CryptoTradingCrew().crew()
    result = crew.kickoff(inputs=inputs)
    
    print(f"Trading analysis completed. Results saved to {output_file}")
    return result

def main():
    """Main function to parse arguments and run the appropriate command."""
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading AI Crew')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the crypto trading crew')
    run_parser.add_argument('--crypto', type=str, default='BTC', help='Cryptocurrency symbol to analyze and trade')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'run' or args.command is None:
        # Default to 'run' if no command is specified
        run(args.crypto if hasattr(args, 'crypto') else 'BTC')
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 