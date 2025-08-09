#!/usr/bin/env bash

# Combined startup script for frontend + backend

echo "🚀 Starting AI Inference App - Combined Deployment"
echo "Frontend will be available at http://0.0.0.0:3000"
echo "Backend API will be available at http://0.0.0.0:8000"

# Check if API key is set
if [ -n "$API_KEY" ]; then
    echo "🔒 API Key authentication enabled"
else
    echo "🔓 API Key authentication disabled (set API_KEY env var to enable)"
fi

# Create log directory for supervisor
mkdir -p /var/log/supervisor

# Start supervisor to manage both processes
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf