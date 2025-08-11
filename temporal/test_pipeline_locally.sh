#!/bin/bash

set -e  # stop on error

echo "🚀 Starting local Temporal pipeline test..."

# 1. Go into your Temporal project folder
cd temporal_5 || exit 1

# 2. Install Python requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# 3. Start Temporal stack
echo "🐳 Starting Temporal stack..."
docker-compose -f docker-compose.temporal.yaml up --build -d

echo "⏳ Waiting for Temporal to fully start..."
sleep 25

# 4. Build your Docker image
echo "🐳 Building processing image..."
docker build -t s2-product-pipeline:latest .

# 5. Run a test container
echo "🧪 Running test processing container..."
docker run -v "$(pwd):/app" s2-product-pipeline:latest \
  python run_one_product.py \
  --step download \
  --config config/config.yaml \
  --product-id S2A_MSIL2A_20220129T132231_N0510_R038_T23LLD_20240508T030029.SAFE

# 6. Start Temporal worker in background
echo "👷 Starting worker..."
python worker.py &

# 7. Trigger the workflow
echo "🚀 Triggering workflow..."
sleep 5
python start_workflow.py

echo "✅ All steps completed successfully!"
