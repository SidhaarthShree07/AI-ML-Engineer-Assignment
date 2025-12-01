#!/bin/bash

# Build script for HybridAutoMLE Docker image

set -e

echo "=========================================="
echo "Building HybridAutoMLE Docker Image"
echo "=========================================="

# Configuration
IMAGE_NAME="hybrid-auto-mle-agent"
IMAGE_TAG="${1:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo "Image: ${FULL_IMAGE_NAME}"
echo ""

# Build the image
echo "Building Docker image..."
docker build \
  --tag "${FULL_IMAGE_NAME}" \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  .

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "Image: ${FULL_IMAGE_NAME}"
echo ""
echo "To run the agent:"
echo "  docker run --gpus all -e GEMINI_API_KEY=\$GEMINI_API_KEY ${FULL_IMAGE_NAME} --help"
echo ""
echo "To run with docker-compose:"
echo "  export GEMINI_API_KEY=your-key"
echo "  docker-compose up"
echo ""
