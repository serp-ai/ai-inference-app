#!/bin/bash

# Build script for combined frontend + backend deployment

echo "Building combined deployment..."

# First build the backend images (if not already built)
echo "Building backend images..."
cd backend
./build-examples.sh
cd ..

# Build combined images using the backend as base
echo "Building combined images..."

# CUDA combined
echo "Building CUDA combined image..."
docker buildx build --build-arg BACKEND_IMAGE="serpcompany/ai-inference-app:cuda" -t serpcompany/ai-inference-app-combined:cuda . --push

# CPU combined
echo "Building CPU combined image..."
docker buildx build --build-arg BACKEND_IMAGE="serpcompany/ai-inference-app:cpu" -t serpcompany/ai-inference-app-combined:cpu . --push

# AMD combined
echo "Building AMD combined image..."
docker buildx build --build-arg BACKEND_IMAGE="serpcompany/ai-inference-app:amd" -t serpcompany/ai-inference-app-combined:amd . --push

# OSX combined
echo "Building OSX combined image..."
docker buildx build --platform=linux/arm64 --build-arg BACKEND_IMAGE="serpcompany/ai-inference-app:osx" -t serpcompany/ai-inference-app-combined:osx . --push

echo "Combined build complete!"
echo ""
echo "Available combined images:"
echo "  serpcompany/ai-inference-app:cuda - NVIDIA GPU + Frontend"
echo "  serpcompany/ai-inference-app:cpu  - CPU-only + Frontend"
echo "  serpcompany/ai-inference-app:amd  - AMD GPU + Frontend"
echo "  serpcompany/ai-inference-app:osx  - Apple Silicon/ARM + Frontend"
echo ""
echo "Usage:"
echo "  docker run -p 80:80 -p 8000:8000 serpcompany/ai-inference-app:cuda"
echo "  Frontend: http://localhost:80"
echo "  API: http://localhost:8000"