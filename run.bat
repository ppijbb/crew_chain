@echo off
REM Script to run the Crew Chain trading system using uv on Windows

REM Check if the virtual environment exists
if not exist ".venv" (
    echo Virtual environment not found. Running installation first...
    powershell -ExecutionPolicy Bypass -Command "iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex"
    uv venv .venv
    call .venv\Scripts\activate
    uv pip install -r requirements.txt
    uv lock
)

REM Activate virtual environment if not already activated
if "%VIRTUAL_ENV%"=="" (
    call .venv\Scripts\activate
)

REM Run the application using uv
echo Starting Crew Chain trading system...
uv run src/crew_chain/crypto_trading_main.py %*

REM Alternative approach using module name:
REM uv run -m crew_chain.crypto_trading_main %* 