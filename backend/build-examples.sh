#!/bin/bash

# Build commands for different hardware configurations

echo "Building Docker images for different hardware configurations..."

# CUDA build (default)
echo "Building CUDA image..."
docker buildx build --build-arg INCLUDE_FLUX_KONTEXT="true" --build-arg REQUIREMENTS_FILE="requirements-cuda.txt" -t serpcompany/ai-inference-app:cuda . --push

# CUDA optimized build
echo "Building CUDA optimized image..."
docker buildx build --build-arg BASE_IMAGE="nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04" --build-arg PYTHON_VERSION="3.12" --build-arg REQUIREMENTS_FILE="requirements-cuda-optimized.txt" -t serpcompany/ai-inference-app:cuda128 . --push

docker buildx build  --build-arg BASE_IMAGE="nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04" --build-arg INSTALL_SAGEATTENTION="true" --build-arg PYTHON_VERSION="3.12" --build-arg REQUIREMENTS_FILE="requirements-cuda-optimized.txt" -t serpcompany/ai-inference-app:cuda128_120 . --push

# AMD ROCm build
echo "Building AMD ROCm image..."
docker buildx build --build-arg INCLUDE_FLUX_KONTEXT="true" --build-arg REQUIREMENTS_FILE="requirements-amd.txt" --build-arg BASE_IMAGE="rocm/pytorch:rocm6.2.3_ubuntu22.04_py3.10_pytorch_release_2.3.0" -t serpcompany/ai-inference-app:amd . --push

# OSX build
echo "Building OSX image..."
docker buildx build --platform=linux/arm64 --build-arg INCLUDE_FLUX_KONTEXT="true" --build-arg REQUIREMENTS_FILE="requirements-osx.txt" --build-arg BASE_IMAGE="ubuntu:22.04" -t serpcompany/ai-inference-app:osx . --push

# CPU build
echo "Building CPU image..."
docker buildx build --build-arg INCLUDE_FLUX_KONTEXT="true" --build-arg REQUIREMENTS_FILE="requirements-cpu.txt" --build-arg BASE_IMAGE="ubuntu:22.04" -t serpcompany/ai-inference-app:cpu . --push

echo "Build complete!"
echo ""
echo "Available images:"
echo "  serpcompany/ai-inference-app:cuda      - NVIDIA GPU support"
echo "  serpcompany/ai-inference-app:amd       - AMD GPU support (ROCm)"
echo "  serpcompany/ai-inference-app:osx       - Apple Silicon/ARM support"
echo "  serpcompany/ai-inference-app:cpu       - CPU-only (x86_64)"