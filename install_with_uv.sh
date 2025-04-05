#!/bin/bash
# Script to install dependencies using uv

set -e  # Exit on error

echo "=== Crew Chain Installation Script ==="
echo "This script will install dependencies using uv for faster package resolution"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing now..."
    pip install uv
else
    echo "uv already installed"
fi

# Create virtual environment
echo "Creating virtual environment..."
uv venv .venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Activating virtual environment (Windows)..."
    source .venv/Scripts/activate
else
    echo "Activating virtual environment (Unix)..."
    source .venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies using uv..."
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
    
    # Generate lock file for reproducibility
    echo "Generating lock file for reproducible builds..."
    uv lock
    echo "Lock file created/updated: uv.lock"
elif [ -f "pyproject.toml" ]; then
    echo "Found pyproject.toml, installing using Poetry configuration..."
    uv pip install -e .
    
    # Generate lock file for reproducibility
    echo "Generating lock file for reproducible builds..."
    uv lock
    echo "Lock file created/updated: uv.lock"
else
    echo "No requirements.txt or pyproject.toml found. Please create one of these files first."
    exit 1
fi

echo ""
echo "=== Installation Complete ==="
echo "Virtual environment activated and dependencies installed using uv."
echo ""
echo "To activate the virtual environment in the future, run:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  source .venv/Scripts/activate"
else
    echo "  source .venv/bin/activate"
fi

echo ""
echo "For reproducible installations on other machines, use:"
echo "  uv sync  # This will install exact dependency versions from uv.lock" 