#!/bin/bash
# Script to run the Crew Chain trading system using uv

# Check if the virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Running installation first..."
    ./install_with_uv.sh
fi

# Run the application using uv
echo "Starting Crew Chain trading system..."
uv run src/crew_chain/crypto_trading_main.py "$@"

# Alternative approach using module name:
# uv run -m crew_chain.crypto_trading_main "$@" 