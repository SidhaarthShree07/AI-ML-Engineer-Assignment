#!/bin/bash

set -e

echo "🚀 Starting ML training inside Docker..."
echo "=========================================="
echo ""

# Run the generated ML script
python /app/workspace/generated_code.py

echo ""
echo "=========================================="
echo "✅ Training completed successfully!"
