#!/bin/bash

# Build commands for frontend deployment
echo "Building frontend Docker images..."

# Standard build (works on x86_64/amd64)
echo "Building standard frontend image..."
docker buildx build -t serpcompany/ai-inference-app-frontend:1.0.0 . --push

# ARM64 build (for Apple Silicon/ARM processors)
echo "Building ARM64 frontend image..."
docker buildx build --platform=linux/arm64 -t serpcompany/ai-inference-app-frontend:1.0.0arm64 . --push

echo "Frontend build complete!"
echo ""
echo "Available images:"
echo "  serpcompany/ai-inference-app-frontend:latest - Standard x86_64 build"
echo "  serpcompany/ai-inference-app-frontend:arm64  - ARM64 build (Apple Silicon)"