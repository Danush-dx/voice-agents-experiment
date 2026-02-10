#!/bin/bash
# Startup script for VoiceStreamAI Telephony Server
# Configuration is loaded from .env file

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}VoiceStreamAI Telephony Server${NC}"
echo "================================"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}ERROR: .env file not found${NC}"
    echo ""
    echo "Please create a .env file from the template:"
    echo "  cp .env.example .env"
    echo ""
    echo "Then edit .env and add your credentials:"
    echo "  - BASE_URL (ngrok URL)"
    echo "  - GROQ_API_KEY"
    echo "  - CARTESIA_API_KEY"
    echo ""
    exit 1
fi

# Source .env file to validate
set -a
source .env
set +a

# Validate required variables
MISSING=""

if [ -z "$BASE_URL" ] || [ "$BASE_URL" = "https://<<URL>>.app" ]; then
    MISSING="${MISSING}\n  - BASE_URL (ngrok or public URL)"
fi

if [ ! -z "$MISSING" ]; then
    echo -e "${RED}ERROR: Missing required configuration in .env file:${NC}"
    echo -e "$MISSING"
    echo ""
    echo "Please edit .env and set these values."
    exit 1
fi

echo -e "${GREEN}Configuration validated!${NC}"
echo ""
echo "Starting server..."
echo ""

# Start the telephony server
# Configuration is automatically loaded from .env by src/config.py
python -m src.telephony_main

# If you want to override any config, you can use command-line args:
# python -m src.telephony_main --port 8000
