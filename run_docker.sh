#!/bin/bash

# Run script for HybridAutoMLE Docker container

set -e

# Configuration
IMAGE_NAME="hybrid-auto-mle-agent:latest"
CONTAINER_NAME="hybrid-agent-$(date +%Y%m%d_%H%M%S)"

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY environment variable is not set"
    echo "Please set it with: export GEMINI_API_KEY=your-api-key"
    exit 1
fi

# Default paths
DATASET_PATH="${DATASET_PATH:-./datasets/example}"
COMPETITION_ID="${COMPETITION_ID:-example-competition}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
MAX_RUNTIME_HOURS="${MAX_RUNTIME_HOURS:-24}"

echo "=========================================="
echo "Running HybridAutoMLE Agent"
echo "=========================================="
echo "Container: ${CONTAINER_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Competition: ${COMPETITION_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "Max Runtime: ${MAX_RUNTIME_HOURS} hours"
echo "=========================================="
echo ""

# Create output directories if they don't exist
mkdir -p "${OUTPUT_DIR}"
mkdir -p ./logs

# Run the container
docker run \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --shm-size=16g \
  --memory=440g \
  --cpus=36 \
  --rm \
  -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$(pwd)/datasets:/workspace/datasets:ro" \
  -v "$(pwd)/outputs:/workspace/outputs" \
  -v "$(pwd)/logs:/workspace/logs" \
  "${IMAGE_NAME}" \
  --dataset_path "/workspace/datasets/$(basename ${DATASET_PATH})" \
  --competition_id "${COMPETITION_ID}" \
  --output_dir /workspace/outputs \
  --max_runtime_hours "${MAX_RUNTIME_HOURS}" \
  "$@"

echo ""
echo "=========================================="
echo "Execution Complete"
echo "=========================================="
echo "Check outputs in: ${OUTPUT_DIR}"
echo "Check logs in: ./logs"
echo ""
