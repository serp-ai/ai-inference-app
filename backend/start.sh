#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
export PYTHONUNBUFFERED=true
export HF_HOME="/"

# Set default API configuration
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"

echo "AI Inference APP - Starting backend"
echo "API will be available at http://${API_HOST}:${API_PORT}"

# Check if API key is set
if [ -n "$API_KEY" ]; then
    echo "🔒 API Key authentication enabled"
else
    echo "🔓 API Key authentication disabled (set API_KEY env var to enable)"
fi

# Start the server
cd /src
exec uvicorn api_handler:app \
    --host ${API_HOST} \
    --port ${API_PORT} \
    --workers 1 \
    --log-level info \
    --access-log \
    --loop asyncio