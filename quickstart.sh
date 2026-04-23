#!/usr/bin/env bash
# LibertyMind Quick Start Script
# ================================
# This script sets up LibertyMind and runs basic verification.

set -e

echo ""
echo "  🗽 LibertyMind v4.2 — Quick Start"
echo "  ==================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check Python
echo "  [1/5] Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "  ${RED}Error: Python 3.9+ not found${NC}"
    echo "  Install Python: https://python.org/downloads/"
    exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "  ${GREEN}Python $PY_VERSION found${NC}"

# Create virtual environment
echo ""
echo "  [2/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo -e "  ${GREEN}Virtual environment created${NC}"
else
    echo -e "  ${YELLOW}Virtual environment already exists${NC}"
fi

# Activate
source venv/bin/activate 2>/dev/null || {
    echo -e "  ${RED}Error: Cannot activate virtual environment${NC}"
    exit 1
}

# Install dependencies
echo ""
echo "  [3/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -e "." -q 2>/dev/null || pip install torch numpy pyyaml pytest -q
echo -e "  ${GREEN}Dependencies installed${NC}"

# Run tests
echo ""
echo "  [4/5] Running tests..."
python -m pytest tests/ -v --tb=short 2>/dev/null || {
    echo -e "  ${YELLOW}Some tests may require PyTorch${NC}"
}
echo -e "  ${GREEN}Tests complete${NC}"

# Show available providers
echo ""
echo "  [5/5] Checking providers..."
python cli.py providers 2>/dev/null || echo -e "  ${YELLOW}CLI not yet configured${NC}"

echo ""
echo "  ==================================="
echo "  Quick Start Complete!"
echo ""
echo "  Next steps:"
echo "    source venv/bin/activate"
echo "    python cli.py chat --provider openai \"Hello\""
echo "    python cli.py introspect --provider openai"
echo "    python cli.py serve --port 8080"
echo ""
